# HY-BAS-ADV1 Run 6: RNAPro Template Integration
## Stanford RNA 3D Folding Part 2

**Status:** DESIGN PHASE
**Base:** Run 5 (IPA) + pre-computed RNAPro predictions
**Competition deadline:** March 25, 2026

---

## The Core Idea

RNAPro is the best public RNA 3D folding model (488M params, TM-score ~0.635 on Part 1 private leaderboard). It **cannot run on Kaggle T4** (needs A100/H100, 40+ GB VRAM). But we can run it **offline** on cloud hardware, pre-compute 3D structure predictions for Part 2 test sequences, and feed those predictions into our lightweight IPA notebook as high-quality templates.

This is exactly what the top Part 2 teams are doing. The template oracle baseline (0.554) still beats every ML submission (best: 0.499). Better templates = higher scores.

---

## Two-Phase Architecture

```
PHASE A: OFFLINE (Cloud A100, ~$3-5)
  Part 2 test_sequences.csv
    → RNAPro inference (multiple seeds)
    → CIF output files (all-atom 3D structures)
    → Extract C1' atom coordinates
    → Format as submission.csv
    → convert_templates_to_pt_files.py → template_features.pt
    → Upload to Kaggle as dataset

PHASE B: KAGGLE NOTEBOOK (T4, Run 6)
  Load pre-computed RNAPro templates from /kaggle/input/
    → Use as HIGH-QUALITY init_coords for IPA
    → Use in TemplateEncoder for pair features
    → Fall back to Run 5 behavior for unknown sequences
    → IPA refines RNAPro predictions (small deltas)
    → 5-slot submission with RNAPro-seeded diversity
```

---

## Phase A: Offline RNAPro Inference (Step-by-Step)

### Step A1: Rent Cloud GPU

| Provider | GPU | VRAM | Cost/hr | Est. total cost |
|----------|-----|------|---------|-----------------|
| **Vast.ai** | A100 80GB | 80 GB | ~$0.80-1.50 | ~$3-5 |
| RunPod | A100 40GB | 40 GB | ~$1.00-1.50 | ~$3-5 |
| Lambda Labs | A100 | 40 GB | ~$1.10 | ~$4-6 |
| Colab Pro | A100 | 40 GB | ~$10/mo | Risk of disconnect |

**Recommendation:** Vast.ai — cheapest, dedicated instance, no disconnect.

### Step A2: Set Up RNAPro Environment

```bash
# Option 1: Docker (recommended by NVIDIA)
docker pull nvcr.io/nvidia/pytorch:25.09-py3
docker run --gpus all -it nvcr.io/nvidia/pytorch:25.09-py3

# Inside container:
git clone https://github.com/NVIDIA-Digital-Bio/RNAPro
cd RNAPro
pip install -e .
```

### Step A3: Download Required Data (~100GB)

```bash
mkdir release_data && cd release_data

# CCD cache (required)
python3 scripts/gen_ccd_cache.py

# Protenix pretrained checkpoint
mkdir protenix_models && cd protenix_models
wget https://af3-dev.tos-cn-beijing.volces.com/release_model/protenix_base_default_v0.5.0.pt
cd ..

# RibonanzaNet2 checkpoint
mkdir ribonanzanet2_checkpoint && cd ribonanzanet2_checkpoint
curl -L -o ribonanzanet2.tar.gz \
  https://www.kaggle.com/api/v1/models/shujun717/ribonanzanet2/pyTorch/alpha/1/download
tar -xzvf ribonanzanet2.tar.gz && rm ribonanzanet2.tar.gz
cd ..
```

### Step A4: Download RNAPro Model Checkpoint

```bash
# From HuggingFace — use PUBLIC-Best for Part 2 public LB
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('nvidia/RNAPro-Public-Best-500M', local_dir='./rnapro_public_best')
"
```

### Step A5: Prepare Part 2 Test Sequences

Download `test_sequences.csv` from the Kaggle competition page. It has columns: `target_id`, `sequence`.

```bash
# RNAPro expects exactly this format — no conversion needed
cp test_sequences.csv release_data/part2_test_sequences.csv
```

### Step A6: Generate Templates for Test Sequences

RNAPro needs templates as input. Use the Kaggle public notebooks:

**Option 1 (stronger):** Kaggle 1st-place TBM notebook:
  - Source: https://www.kaggle.com/code/jaejohn/rna-3d-folds-tbm-only-approach
  - Produces `submission.csv` with template coords

**Option 2 (faster):** MMseqs2 template search:
  - Source: https://www.kaggle.com/code/rhijudas/mmseqs2-3d-rna-template-identification

Convert to RNAPro format:
```bash
python preprocess/convert_templates_to_pt_files.py \
  --input_csv submission.csv \
  --output_name release_data/part2_template_features.pt \
  --max_n 40
```

### Step A7: Generate MSAs (optional)

```bash
# Place MSA files in release_data/part2_msa/
# One .fasta per target_id
# RNAPro can run without MSAs (lower quality but still good)
```

### Step A8: Run RNAPro Inference

```bash
python -m rnapro.runner.inference \
  --model_name rnapro_base \
  --load_checkpoint_path ./rnapro_public_best/model.pt \
  --input_csv release_data/part2_test_sequences.csv \
  --dump_dir ./rnapro_output/ \
  --seeds 42,123,456,789,1024 \
  --dtype bf16 \
  --use_msa \
  --rna_msa_dir release_data/part2_msa/ \
  --use_template ca_precomputed \
  --template_data release_data/part2_template_features.pt
```

**NOTE:** RNAPro crops sequences to 512 nt. For longer sequences, fall back to Run 5 IPA.

**Expected output:** CIF files in `rnapro_output/` — one per (target, seed).

### Step A9: Extract C1' Coordinates from CIF

```python
"""extract_c1_from_cif.py — Convert RNAPro CIF output to submission format"""
import os, glob
import numpy as np
import pandas as pd

try:
    import gemmi
    USE_GEMMI = True
except ImportError:
    from Bio.PDB.MMCIFParser import MMCIFParser
    USE_GEMMI = False


def extract_c1_prime(cif_path):
    """Extract C1' atom coords from CIF."""
    if USE_GEMMI:
        structure = gemmi.read_structure(cif_path)
        coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        if atom.name == "C1'":
                            coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
        return np.array(coords, dtype=np.float32)
    else:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('rna', cif_path)
        coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        if atom.get_name() == "C1'":
                            pos = atom.get_vector()
                            coords.append([pos[0], pos[1], pos[2]])
        return np.array(coords, dtype=np.float32)


def process_all_outputs(output_dir, test_csv_path):
    """Process RNAPro CIF outputs into numpy dict."""
    test_df = pd.read_csv(test_csv_path)
    coords_dict = {}
    
    for _, row in test_df.iterrows():
        target_id = row['target_id']
        
        for seed_dir in sorted(glob.glob(f"{output_dir}/seed_*")):
            seed_idx = int(seed_dir.split('_')[-1])
            cif_files = glob.glob(f"{seed_dir}/*{target_id}*.cif")
            if cif_files:
                coords = extract_c1_prime(cif_files[0])
                coords_dict[f"{target_id}_seed{seed_idx}"] = coords
    
    np.savez_compressed('rnapro_part2_coords.npz', **coords_dict)
    print(f"Saved {len(coords_dict)} predictions to rnapro_part2_coords.npz")


if __name__ == '__main__':
    process_all_outputs('./rnapro_output/', 'release_data/part2_test_sequences.csv')
```

### Step A10: Upload to Kaggle

```bash
kaggle datasets init -p ./rnapro_upload/
cp rnapro_part2_coords.npz ./rnapro_upload/
# Edit dataset-metadata.json, then:
kaggle datasets create -p ./rnapro_upload/ --dir-mode zip
```

**Dataset name:** `rnapro-part2-templates-v1`

---

## Phase B: Kaggle Notebook Changes (Run 6 vs Run 5)

### What Changes

| Cell | Run 5 | Run 6 |
|------|-------|-------|
| 0 | Config params | ADD `RNAPRO_DATASET_PATH`, `RNAPRO_CONFIDENCE` |
| 3 | Find Run 4 checkpoint | ADD search for rnapro_part2_coords.npz |
| 9 | Template search | ADD RNAPro template loading for known targets |
| 13 | IPA training | UNCHANGED (or add distillation loss — Run 6b) |
| 14 | IPA inference | REPLACE init_coords with RNAPro when available |

### Cell 0 Additions

```python
RNAPRO_CONFIDENCE = 0.9  # RNAPro predictions = 90% confident templates
```

### Cell 3 Additions

```python
RNAPRO_COORDS_FILE = None
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f == "rnapro_part2_coords.npz":
            RNAPRO_COORDS_FILE = os.path.join(root, f)
print(f"RNAPro coords: {RNAPRO_COORDS_FILE}")
```

### Cell 9 Changes

```python
# Load RNAPro pre-computed templates
rnapro_coords = {}
if RNAPRO_COORDS_FILE:
    data = np.load(RNAPRO_COORDS_FILE)
    for key in data.files:
        parts = key.rsplit('_seed', 1)
        target_id, seed_idx = parts[0], int(parts[1])
        if target_id not in rnapro_coords:
            rnapro_coords[target_id] = {}
        rnapro_coords[target_id][seed_idx] = data[key]
    print(f"Loaded RNAPro predictions for {len(rnapro_coords)} targets")

# In template search loop:
for idx, row in test_seqs.iterrows():
    target_id = row['target_id']
    if target_id in rnapro_coords:
        template_coords_per_target[target_id] = rnapro_coords[target_id][0]
        template_confidence_per_target[target_id] = RNAPRO_CONFIDENCE
    else:
        # Fall back to existing template search
        ...
```

### Cell 14 Changes (Most Critical)

```python
if target_id in rnapro_coords:
    # RNAPro AVAILABLE — use as high-quality seeds
    seeds = rnapro_coords[target_id]
    
    # Slot 1: Raw RNAPro seed 0 (no modification)
    coords_list.append(seeds[0])
    
    # Slot 2: RNAPro seed 0 → IPA refinement
    coords_list.append(run_ipa(seeds[0], seed=0))
    
    # Slot 3: RNAPro seed 1 → IPA refinement (or perturbed seed 0)
    init_1 = seeds.get(1, seeds[0] + noise(0.5))
    coords_list.append(run_ipa(init_1, seed=1))
    
    # Slot 4: RNAPro seed 2 (raw, for diversity)
    coords_list.append(seeds.get(2, run_ipa(seeds[0] + noise(1.0), seed=2)))
    
    # Slot 5: IPA cold start (maximum diversity)
    coords_list.append(run_ipa(zeros_np, seed=4))

elif tmpl_conf > HYBRID_THRESHOLD:
    # ... existing Run 5 template+IPA behavior ...
else:
    # ... existing Run 5 IPA-only behavior ...
```

---

## Expected Impact

| Scenario | Est. Score |
|----------|-----------|
| Run 5 current (IPA, no RNAPro) | ~0.46 |
| Run 6 conservative (RNAPro templates) | ~0.48-0.49 |
| Run 6 optimistic (RNAPro + IPA refinement) | ~0.50-0.52 |
| Template oracle ceiling | 0.554 |

---

## Risks

1. **Unknown private test sequences** — RNAPro predictions won't exist. Falls back to Run 5.
2. **512 nt crop** — RNAPro truncates long sequences. Falls back to Run 5 for those.
3. **Setup complexity** — ~100GB data download + Docker. Budget 2-3 hours.
4. **Part 2 targets may be harder** — RNAPro was trained on Part 1. Template-free targets may get poor predictions.

---

## Optional: Knowledge Distillation (Run 6b)

1. Run RNAPro on ALL ~5135 training sequences (additional ~$3-5 cloud cost)
2. Add distillation loss: `distill_loss = FAPE(ipa_output, rnapro_prediction) * 0.3`
3. IPA learns to mimic RNAPro — helps on unknown test sequences where no pre-computed RNAPro exists.

---

## Action Steps

1. **Rent A100** on Vast.ai (~$1/hr)
2. **Set up RNAPro** via Docker (Steps A2-A4, ~1 hr)
3. **Download test_sequences.csv** from Kaggle
4. **Generate templates** using TBM notebook (Step A6, ~30 min)
5. **Run RNAPro inference** with 5 seeds (Step A8, ~1-2 hrs)
6. **Extract C1' coords** from CIF (Step A9, ~10 min)
7. **Upload to Kaggle** as dataset (Step A10, ~10 min)
8. **Modify Run 5 → Run 6** (Phase B, ~2-3 hrs coding)
9. **Submit Run 6**

**Total time:** ~6-8 hours | **Total cost:** ~$3-5
