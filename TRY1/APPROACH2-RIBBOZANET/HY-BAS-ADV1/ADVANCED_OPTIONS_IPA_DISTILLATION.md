# Advanced Options: IPA/FAPE and RNAPro Distillation
## Detailed Steps, Hardware Requirements, and Feasibility Analysis

---

## Table of Contents

1. All Runs — Past, Current, and Planned
2. Option A: IPA + FAPE (Replace MDS)
3. Option B: RNAPro Distillation (Teacher-Student)
4. Comparison: Which to Pursue?
5. Combined Roadmap

---

## 1. All Runs — Past, Current, and Planned

### Completed Runs

| Run | Score | pair_dim | Backbone | Inference | Epochs | Key Change | Lesson Learned |
|-----|-------|----------|----------|-----------|--------|-----------|----------------|
| BASIC | 0.092 | 64 | Frozen | MDS | 50 (local) | First submission | Pre-made CSV, NN never ran on Kaggle |
| Fork 2 | 0.172 | N/A | N/A | Template copy | N/A | Template matching | Bigger database = better matches |
| Fork 2+OptB | 0.287 | N/A | N/A | Template copy | N/A | 24K database + ID fix | Template copying beats NN. Current best. |
| Run 1 | 0.109 | 80 | Frozen | MDS | 15 | Add templates to NN | Broken pickle = trained on nothing |
| Run 2A | 0.109 | 80 | Frozen | MDS | 30 | Fix pickle, more epochs | MDS is the bottleneck, not training |
| Run 2B | 0.107 | 80 | Frozen | MDS | 30 | Duplicate of 2A | Confirms MDS problem is systematic |

### Current Run

| Run | Score | pair_dim | Backbone | Inference | Epochs | Key Change | Status |
|-----|-------|----------|----------|-----------|--------|-----------|--------|
| Run 3 OptB | pending | 80 | Layers 7-8 unfrozen | Template-seeded refinement | 30 | Bypass MDS + hybrid slots + unfreeze | Committed, waiting for score |

### Ready to Submit

| Run | Score | pair_dim | Backbone | Inference | Epochs | Key Change | Status |
|-----|-------|----------|----------|-----------|--------|-----------|--------|
| Run 4 (MSA) | — | 88 | Layers 7-8 unfrozen | Template-seeded refinement | 50 | +8 MSA evolutionary features | Coded, ready |

### Planned / Candidate Runs

| Run | pair_dim | Backbone | Inference | Key Change | Hardware Needed | Feasibility |
|-----|----------|----------|-----------|-----------|----------------|-------------|
| Run 5a: IPA | 88 | Layers 7-8 unfrozen | IPA (replaces MDS entirely) | IPA module + FAPE loss | Kaggle T4 (16 GB) | Medium — 8-12 hrs coding |
| Run 5b: Distillation | 88 | Layers 7-8 unfrozen | Template-seeded refinement | Train on RNAPro predictions | Lab GPU 24+ GB VRAM | **Blocked** — no hardware |
| Run 5c: IPA + Distillation | 88 | Layers 7-8 unfrozen | IPA | Both IPA and distilled targets | Lab GPU + coding time | **Blocked** — no hardware |
| Run 6: Full pipeline | 88+ | Partially unfrozen | IPA | MSA + IPA + distillation | Lab GPU | Post-competition |

---

## 2. Option A: IPA + FAPE (Replace MDS)

### What is IPA?

IPA (Invariant Point Attention) is the method AlphaFold2 uses to turn
pairwise features into 3D coordinates. Instead of predicting distances
and then reconstructing 3D (our current approach), IPA directly moves
atoms into their correct 3D positions through iterative refinement.

Think of it like this:
- **Current method (MDS/refinement):** Predict a distance table → do math
  to find 3D positions. If the distances are noisy, the math produces garbage.
- **IPA:** Start all atoms at a guess position. Each atom looks at all other
  atoms and the pairwise features, decides how to move. All atoms move
  simultaneously. Repeat 8 times. Each round, atoms self-correct.

### What is FAPE?

FAPE (Frame Aligned Point Error) is the loss function used to train IPA.
Instead of comparing predicted vs true distances (our current MSE loss),
FAPE compares predicted vs true coordinates in local reference frames.
This is rotationally invariant — the loss is the same regardless of how
you orient the molecule.

Why FAPE matters: With MSE on distances, the model can produce mirror-flipped
structures that have correct distances but wrong 3D shape. FAPE catches this
because mirrored structures have different local frame orientations.

### What Changes in Our Pipeline

```
CURRENT (Run 3 OptB / Run 4):
  Backbone(64) + Template(16) + MSA(8) = 88 channels
  → DistanceMatrixHead → predicted distances (N×N)
  → Template-seeded refinement → 3D coordinates

WITH IPA (Run 5a):
  Backbone(64) + Template(16) + MSA(8) = 88 channels
  → IPA module (8 iterations) → 3D coordinates DIRECTLY
  DistanceMatrixHead is REMOVED
  Template-seeded refinement is REMOVED
  MDS is REMOVED
```

### Step-by-Step Implementation Plan

#### Step 1: Choose IPA Library (30 min research)

| Library | Source | Size | Pros | Cons |
|---------|--------|------|------|------|
| lucidrains/invariant-point-attention | PyPI/GitHub | ~500 lines | Clean API, standalone, pip installable | May not handle RNA frames out of box |
| OpenFold IPA module | GitHub (open source) | ~1000 lines | Proven on proteins, well-tested | Tightly coupled to OpenFold, extraction needed |
| RhoFold+ structure module | GitHub (open source) | ~800 lines | Designed for RNA specifically | May require RhoFold+ dependencies |
| Protenix (from RNAPro) | GitHub (Apache 2.0) | ~2000 lines | State-of-the-art for RNA | Large, complex, NVIDIA-specific optimizations |

**Recommendation:** lucidrains/invariant-point-attention for simplicity,
or extract from RhoFold+ for RNA-specific frame definitions.

#### Step 2: Define RNA Nucleotide Frames (2-3 hrs)

IPA operates on "frames" — each residue has a position (translation)
and orientation (rotation). AlphaFold2 defines protein frames using
backbone atoms N, Cα, C. RNA doesn't have these atoms.

RNA frame convention (from RhoFold+):
- **Origin:** C1' atom (what we predict)
- **X-axis:** C1' → N1 (purines) or C1' → N9 (pyrimidines)
- **Y-axis:** cross product to complete right-handed frame
- **Z-axis:** cross product of X and Y

For our C1'-only prediction, a simplified frame:
- **Origin:** C1' coordinate of residue i
- **X-axis:** direction from residue i to residue i+1
- **Y-axis/Z-axis:** arbitrary orthogonal completion

This simplified frame loses some structural detail but is compatible
with our C1'-only coordinate prediction.

**Code needed:** ~50-100 lines
```python
def build_rna_frames(coords):
    """Convert (N, 3) C1' coordinates to (N, 4, 4) homogeneous frames.
    Each frame = 3x3 rotation + 3x1 translation."""
    N = coords.shape[0]
    # Translation = C1' position
    translations = coords  # (N, 3)
    # Rotation: x-axis from i to i+1, orthogonalize
    x_axes = coords[1:] - coords[:-1]  # (N-1, 3)
    x_axes = F.normalize(x_axes, dim=-1)
    # Pad last residue
    x_axes = torch.cat([x_axes, x_axes[-1:]], dim=0)
    # y-axis: arbitrary perpendicular
    arbitrary = torch.tensor([0, 0, 1.0], device=coords.device)
    y_axes = torch.cross(x_axes, arbitrary.expand_as(x_axes), dim=-1)
    y_axes = F.normalize(y_axes, dim=-1)
    z_axes = torch.cross(x_axes, y_axes, dim=-1)
    rotations = torch.stack([x_axes, y_axes, z_axes], dim=-1)  # (N, 3, 3)
    return translations, rotations
```

#### Step 3: Implement IPA Attention Module (3-4 hrs if using library, 8-12 hrs from scratch)

**Using lucidrains library:**
```python
# pip install invariant-point-attention (upload as wheel for Kaggle)
from invariant_point_attention import InvariantPointAttention

ipa_layer = InvariantPointAttention(
    dim=88,              # pair_dim (our combined features)
    heads=8,             # attention heads
    scalar_key_dim=16,
    scalar_value_dim=16,
    point_key_dim=4,
    point_value_dim=4,
)
```

**What you still need to build around the library:**
```python
class IPAStructureModule(nn.Module):
    """Replaces DistanceMatrixHead + MDS + refine_coords.
    Takes pairwise features (N, N, 88) and produces 3D coords (N, 3)."""

    def __init__(self, pair_dim=88, n_iterations=8):
        super().__init__()
        self.n_iterations = n_iterations
        # Single representation from pair (project pair to per-residue)
        self.pair_to_single = nn.Linear(pair_dim, 256)
        # IPA layers (one per iteration, or shared weights)
        self.ipa_layers = nn.ModuleList([
            InvariantPointAttention(dim=256, ...)
            for _ in range(n_iterations)
        ])
        # Frame update: predict rotation and translation updates
        self.frame_update = nn.Linear(256, 6)  # 3 rotation + 3 translation

    def forward(self, pair_features, initial_coords=None):
        # Initialize frames from template coords or zeros
        if initial_coords is not None:
            translations, rotations = build_rna_frames(initial_coords)
        else:
            translations = torch.zeros(N, 3)
            rotations = torch.eye(3).expand(N, 3, 3)

        # Project pair features to single representation
        single = self.pair_to_single(pair_features.mean(dim=1))  # (N, 256)

        # Iterative refinement
        for ipa_layer in self.ipa_layers:
            single = ipa_layer(single, pairwise=pair_features,
                              rotations=rotations, translations=translations)
            # Update frames
            update = self.frame_update(single)
            # Apply rotation and translation updates...

        return translations  # Final 3D coordinates (N, 3)
```

#### Step 4: Implement FAPE Loss (1-2 hrs)

```python
def fape_loss(pred_coords, true_coords, pred_frames, true_frames):
    """Frame Aligned Point Error.
    For each residue i, transform all coordinates into residue i's
    local frame, then compute the distance between predicted and true
    positions in that local frame. Average over all residues.

    This is rotationally invariant — the loss is the same regardless
    of global orientation. It also catches mirror-flips because local
    frames have handedness."""
    N = pred_coords.shape[0]
    total_loss = 0.0
    for i in range(N):
        # Transform all coords into frame i
        pred_local = apply_inverse_frame(pred_frames[i], pred_coords)
        true_local = apply_inverse_frame(true_frames[i], true_coords)
        # Clamped L2 distance
        dist = torch.clamp(torch.norm(pred_local - true_local, dim=-1), max=10.0)
        total_loss += dist.mean()
    return total_loss / N
```

#### Step 5: Integrate into Kaggle Notebook (2-3 hrs)

| Cell | Change |
|------|--------|
| Cell 0 | Add IPA config: `IPA_ITERATIONS = 8`, `IPA_DIM = 256` |
| Cell 10 | Add `IPAStructureModule` class, `build_rna_frames()`, `fape_loss()` |
| Cell 12 | Replace `DistanceMatrixHead` with `IPAStructureModule`. Warm-start: cannot reuse distance head weights (architecture change). Template encoder and backbone layers CAN be reused. |
| Cell 13 | Replace MSE distance loss with FAPE loss. Remove consecutive distance regularization (IPA handles this implicitly). Training loop produces coords directly, not distances. |
| Cell 14 | Replace distance prediction → refinement with IPA forward pass → coords. Template-seeded refinement becomes IPA initialization (start from template frames). Remove MDS entirely. |

#### Step 6: Test and Debug (2-4 hrs)

Most likely issues:
- Frame initialization produces NaN rotations (division by zero for parallel vectors)
- FAPE loss explodes in first epoch (initial coords far from true)
- IPA attention OOM on T4 for long sequences (N>300)
- Mirror-flip debugging (if frames are initialized wrong)

### Hardware Requirements for IPA

| Component | Requirement | Available? |
|-----------|-------------|-----------|
| GPU for training | Kaggle T4 (16 GB) | YES |
| GPU for inference | Kaggle T4 (16 GB) | YES |
| IPA library | pip wheel uploaded as dataset | Need to prepare |
| VRAM estimate | ~12-14 GB for N=256, 8 IPA iterations | Fits on T4 |
| Lab GPU needed? | **NO** | IPA runs entirely on Kaggle |
| Internet needed? | Only for initial pip install / wheel prep | Can use offline wheel |

### Time Estimate for IPA

| Step | Time | Cumulative |
|------|------|-----------|
| Research + choose library | 30 min | 30 min |
| Define RNA frames | 2-3 hrs | 3 hrs |
| IPA module integration | 3-4 hrs | 7 hrs |
| FAPE loss implementation | 1-2 hrs | 9 hrs |
| Notebook integration | 2-3 hrs | 11 hrs |
| Testing + debugging | 2-4 hrs | 14 hrs |
| Kaggle commit + score | 2.5 hrs | 16.5 hrs |
| **Total** | **~12-16 hrs** | |

### Risk Assessment for IPA

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Frame initialization bugs (NaN) | High (50%) | Blocks training | Test frame builder separately on known structures first |
| FAPE loss diverges | Medium (30%) | Training fails | Start with MSE loss, transition to FAPE after warm-up |
| OOM on T4 for long sequences | Medium (30%) | Can't train/infer | Reduce IPA iterations (8→4), reduce sequence length |
| IPA library incompatible with our features | Low (20%) | Need to rewrite | Use lucidrains which has clean API |
| IPA produces worse results than template-seeded refinement | Medium (30%) | Wasted effort | Keep template-seeded refinement as fallback in Slots 1-2 |
| Deadline exceeded | High (60%) | No submission | Only attempt if 24+ hrs remain after Run 4 |

---

## 3. Option B: RNAPro Distillation (Teacher-Student)

### What is RNAPro?

RNAPro is NVIDIA's post-competition RNA 3D structure prediction model.
It combines:
- RibonanzaNet2 (frozen RNA foundation model) for sequence features
- MSA module for evolutionary features
- Template module using the same Fork 1/Fork 2 templates we use
- Pairformer blocks for feature refinement
- Diffusion module for 3D structure generation

It's a 500M parameter model that retrospectively outperformed all
individual Kaggle competition entries. Weights are publicly available
on HuggingFace (Apache 2.0 license).

### What is Distillation?

Instead of training our small model on PDB ground-truth distances,
we train it on RNAPro's predicted distances. RNAPro's predictions
encode vast structural knowledge that our model can absorb.

```
NORMAL TRAINING (current):
  Training targets = distances from 734 PDB structures
  Model learns from: experimental data only

DISTILLATION:
  Training targets = distances predicted by RNAPro
  Model learns from: RNAPro's 500M-param knowledge
  + still uses PDB ground truth for validation
```

### Distillation Levels

| Level | What's Distilled | Difficulty | Quality |
|-------|-----------------|-----------|---------|
| Level 1: Pseudo-labels | RNAPro's predicted 3D coords → compute distances → use as training targets | Easy | Good |
| Level 2: Distance matrices | Extract RNAPro's internal distance predictions before diffusion | Medium | Better |
| Level 3: Feature distillation | Extract RNAPro's Pairformer pair representations → train our model to match | Hard | Best |

**Level 1 is the practical choice** — it requires only running RNAPro
inference and saving the output coordinates. No internal model access needed.

### Step-by-Step Implementation Plan

#### Step 1: Set Up RNAPro on Lab GPU (4-6 hrs)

```bash
# Clone RNAPro repo
git clone https://github.com/NVIDIA-Digital-Bio/RNAPro
cd RNAPro

# Install dependencies (Python 3.10+, PyTorch 2.0+, CUDA 11.7+)
pip install -r requirements.txt

# Download model weights from HuggingFace
# Option A: huggingface-cli
huggingface-cli download nvidia/RNAPro-Public-Best-500M --local-dir release_data/protenix_models/

# Option B: wget
wget https://huggingface.co/nvidia/RNAPro-Public-Best-500M/resolve/main/model.pt \
     -O release_data/protenix_models/model.pt

# Download RibonanzaNet2 weights (used by RNAPro's encoder)
# Follow instructions in RNAPro's README

# Prepare CCD cache (chemical component dictionary)
python preprocess/gen_ccd_cache.py

# Convert our Fork 2 templates to RNAPro format
python preprocess/convert_templates_to_pt_files.py \
    --input_csv path/to/submission-fork2.csv \
    --output_name release_data/kaggle/template_features.pt \
    --max_n 40
```

#### Step 2: Run RNAPro Inference on Training Sequences (2-4 hrs)

```bash
# Run on all 734 training sequences (or a subset)
python inference.py \
    --input_csv train_sequences.csv \
    --output_dir rnapro_predictions/ \
    --model.use_RibonanzaNet2 true \
    --model.ribonanza_net_path release_data/ribonanzanet2_checkpoint \
    --use_template ca_precomputed \
    --template_data release_data/kaggle/template_features.pt \
    --max_len 512

# Also run on the 28 test sequences
python inference.py \
    --input_csv test_sequences.csv \
    --output_dir rnapro_test_predictions/ \
    --max_len 512
```

#### Step 3: Extract Distance Matrices from RNAPro Output (1 hr)

```python
# RNAPro outputs CIF files with all-atom coordinates
# Extract C1' coordinates and compute distance matrices
import numpy as np
import pickle

distillation_data = {}
for target_id in all_targets:
    cif_path = f"rnapro_predictions/{target_id}.cif"
    coords = extract_c1_prime_from_cif(cif_path)  # (N, 3)
    distances = np.sqrt(((coords[:, None] - coords[None, :])**2).sum(-1))
    distillation_data[target_id] = {
        'coords': coords,
        'distances': distances,
        'sequence': sequences[target_id]
    }

# Save as pickle for Kaggle upload
with open('rnapro_distillation_data.pkl', 'wb') as f:
    pickle.dump(distillation_data, f)
```

#### Step 4: Upload to Kaggle as Dataset (5 min)

Upload `rnapro_distillation_data.pkl` as a new Kaggle dataset:
"adv1-rnapro-distillation"

#### Step 5: Modify Training Loop (Cell 13) (2-3 hrs)

```python
# Load RNAPro predictions
with open(DISTILLATION_PICKLE, 'rb') as f:
    rnapro_data = pickle.load(f)

# During training, mix PDB ground truth with RNAPro predictions:
# - 70% of batches: train on PDB ground truth (real distances)
# - 30% of batches: train on RNAPro predicted distances
# This prevents the student from inheriting RNAPro's errors while
# still absorbing its structural knowledge.

for item in batch:
    if random.random() < 0.3 and item['target_id'] in rnapro_data:
        # Use RNAPro's predicted distances as target
        true_dist = rnapro_data[item['target_id']]['distances']
    else:
        # Use PDB experimental distances as target
        true_dist = compute_distances(item['coords'])
```

#### Step 6: Optionally Use RNAPro Test Predictions for Refinement (1 hr)

```python
# In Cell 14, for template-seeded refinement:
# Instead of starting from our template search results,
# start from RNAPro's predicted coordinates for test targets.
# RNAPro's predictions are likely more accurate than our templates.

rnapro_test_coords = rnapro_data[target_id]['coords']
start_coords = torch.tensor(rnapro_test_coords, dtype=torch.float32, device=device)
# Then refine toward our NN's predicted distances
refined = refine_coords(start_coords, pred_dist, steps=100, lr=0.01)
```

### Hardware Requirements for Distillation

| Component | Requirement | Available? | Notes |
|-----------|-------------|-----------|-------|
| **Lab GPU for RNAPro inference** | **NVIDIA GPU with 24-40+ GB VRAM** | **NO** | **This is the blocker** |
| Suitable GPUs | RTX 3090 (24 GB), RTX 4090 (24 GB), A100 (40/80 GB), A6000 (48 GB) | None on team | |
| Cloud alternative | Google Colab Pro+ (A100), AWS p4d, Lambda Labs | Costs $1-5/hr | Possible if budget allows |
| CUDA version | 11.7+ | Depends on GPU machine | |
| Python | 3.10+ | Usually available | |
| Disk space | ~5 GB (model weights + dependencies) | Usually fine | |
| RAM | 32+ GB recommended | Depends on machine | |
| Time on GPU | 2-4 hours for inference on 734 + 28 sequences | — | |
| **Kaggle GPU for our model** | T4 (16 GB) | YES | Only runs OUR model, not RNAPro |
| **Kaggle integration** | Upload .pkl as dataset | YES | Trivial |

### Time Estimate for Distillation

| Step | Time | Where | Hardware |
|------|------|-------|---------|
| Set up RNAPro on lab GPU | 4-6 hrs | Lab machine | 24+ GB GPU |
| Run RNAPro on training seqs | 2-4 hrs | Lab machine | 24+ GB GPU |
| Run RNAPro on test seqs | 30 min | Lab machine | 24+ GB GPU |
| Extract distances + save pickle | 1 hr | Lab machine | CPU only |
| Upload to Kaggle | 5 min | Browser | None |
| Modify Cell 13 (training) | 2-3 hrs | Local editor | None |
| Modify Cell 14 (inference, optional) | 1 hr | Local editor | None |
| Kaggle commit + score | 2.5 hrs | Kaggle | T4 |
| **Total** | **~13-18 hrs** | | |

Of which **~7-11 hrs require the lab GPU** (Steps 1-4).

### Risk Assessment for Distillation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| No lab GPU available | **100% currently** | **Completely blocked** | Seek cloud GPU ($1-5/hr) or university cluster |
| RNAPro setup fails (dependency hell) | Medium (40%) | Delays by 4-8 hrs | Use Docker image if available |
| RNAPro predictions are wrong for some targets | Low (20%) | Student inherits errors | Mix with PDB ground truth (70/30 ratio) |
| RNAPro OOM on 24 GB GPU for long sequences | Medium (30%) | Some sequences skipped | Use --max_len 512 flag |
| Upload size too large for Kaggle | Low (10%) | Need to compress | Distance matrices for 734 structures ≈ 50-200 MB |
| Student model doesn't improve from distillation | Low (20%) | Wasted effort | The distance targets from RNAPro are provably better than our NN's current predictions |

---

## 4. Comparison: Which to Pursue?

| Factor | IPA + FAPE | RNAPro Distillation |
|--------|-----------|-------------------|
| **Hardware needed** | Kaggle T4 only | Lab GPU 24+ GB (**blocked**) |
| **Coding effort** | 8-12 hrs (complex — rotation math, frames, new loss) | 4-6 hrs (simple — change training targets) |
| **Architecture change** | Major — replaces distance head + MDS + refinement | None — only training targets change |
| **Risk of breaking what works** | High — entirely new output pipeline | Low — inference pipeline unchanged |
| **Potential score improvement** | High (eliminates MDS quality loss entirely) | Very high (absorbs 500M-param model's knowledge) |
| **Can build on Run 4?** | Partially — reuses backbone + template encoder, but distance head is replaced | Fully — identical architecture, just better training data |
| **MSA features still useful?** | Yes — IPA reads same 88-channel pairwise features | Yes — same features, better distance predictions |
| **Template-seeded refinement** | Replaced — IPA initializes from template frames directly | Preserved — can optionally start from RNAPro coords |
| **Debugging difficulty** | High — rotation math, frame alignment, 3D geometry | Low — standard training loop, just different targets |
| **Time to first result** | 14-18 hrs | 13-18 hrs (but ~8 hrs blocked on lab GPU) |
| **Feasible before March 25?** | Barely — only if started immediately after Run 4 | **Not feasible without lab GPU** |

### Decision Matrix

| If you have... | Best option |
|----------------|------------|
| No lab GPU, 24+ hrs left | IPA (risky but possible on Kaggle) |
| No lab GPU, <24 hrs left | Stay with Run 4 (MSA). Don't attempt IPA. |
| Lab GPU available, 24+ hrs left | **Distillation** (lower risk, higher ceiling) |
| Lab GPU + 36+ hrs left | Both: Distillation first (quick win), then IPA |
| Cloud GPU budget ($5-10) | Distillation via Colab Pro+ or Lambda Labs |

---

## 5. Combined Roadmap

### Scenario A: No Lab GPU (Current Reality)

```
NOW          Run 3 OptB score arrives
  |          Upload Run 3 checkpoint as dataset
  +1 hr      Run 4 (MSA) committed on Kaggle
  +3.5 hrs   Run 4 score arrives
  |
  |--- If 24+ hrs remain AND Run 4 works:
  |      +4 hrs    Code IPA module (using lucidrains library)
  |      +4 hrs    Integrate into notebook (Run 5a)
  |      +3 hrs    Debug + test
  |      +2.5 hrs  Kaggle commit + score
  |      = ~13.5 hrs total for IPA
  |
  |--- If <24 hrs remain:
  |      Stay with best of Run 3 OptB / Run 4 / Fork 2 (0.287)
  |      Use remaining submissions for parameter tuning
  |
  DEADLINE: March 25, 2026 11:59 PM UTC
```

### Scenario B: Lab GPU Becomes Available

```
NOW          Run 3 OptB score arrives
  |          Start RNAPro setup on lab GPU (parallel)
  +1 hr      Run 4 (MSA) committed on Kaggle
  +3.5 hrs   Run 4 score arrives
  |          RNAPro inference running on lab GPU
  +6 hrs     RNAPro predictions ready → upload as Kaggle dataset
  +2 hrs     Code Run 5b (distillation) notebook
  +2.5 hrs   Run 5b committed + scored on Kaggle
  = ~14 hrs total for distillation
  |
  |--- If time remains:
  |      Run 6: IPA + distillation combined
  |
  DEADLINE: March 25, 2026 11:59 PM UTC
```

### Scenario C: Cloud GPU Budget Available ($5-10)

```
Same as Scenario B, but use:
  - Google Colab Pro+ ($10/month, A100 GPU)
  - Lambda Labs ($1.10/hr, A100)
  - AWS p4d ($32/hr, 8×A100 — overkill, use spot instance)
  - Paperspace ($2.30/hr, A100)

Cheapest option: Google Colab Pro+ ($10 for the month)
  → 24 hrs of A100 access, enough for RNAPro inference
```

---

## 6. Quick Reference: What IPA Replaces in Our Code

| Current Component | With IPA | Status |
|-------------------|----------|--------|
| `DistanceMatrixHead` (Cell 10) | **Replaced** by `IPAStructureModule` | New class |
| `mds_reconstruct()` (Cell 10) | **Deleted** — IPA produces coords directly | Removed |
| `refine_coords()` (Cell 10) | **Deleted** — IPA iteratively refines internally | Removed |
| MSE distance loss (Cell 13) | **Replaced** by FAPE loss | New function |
| Consecutive distance regularization (Cell 13) | **Deleted** — IPA handles backbone implicitly | Removed |
| Template-seeded refinement (Cell 14) | **Replaced** by IPA frame initialization from template | Modified |
| MDS fallback for no-template targets (Cell 14) | **Replaced** by IPA zero initialization | Modified |
| `pred_dist` prediction (Cell 14) | **Replaced** by `pred_coords` from IPA | Modified |
| Diversity via distance noise (Cell 14) | **Replaced** by diversity via initial frame noise | Modified |

Components that **stay the same** with IPA:
- Backbone loading + unfreezing (Cell 11) ✓
- Template encoder (Cell 10) ✓
- MSA features (Cell 9, 9.5) ✓
- Template search (Cell 9) ✓
- Hybrid slot logic (Cell 14, Slots 1-2 template) ✓
- Post-processing (Cell 15) ✓
- All data loading (Cells 1-8) ✓

---

## 7. Quick Reference: What Distillation Changes in Our Code

| Current Component | With Distillation | Status |
|-------------------|-------------------|--------|
| Training targets in Cell 13 | **Mixed** PDB + RNAPro distances | Modified (~20 lines) |
| `TRAIN_PICKLE` loading | Also load `DISTILLATION_PICKLE` | Added (~10 lines) |
| Template-seeded refinement (Cell 14) | Optionally start from RNAPro coords | Modified (~5 lines) |
| Cell 3 auto-discovery | Also find `rnapro_distillation_data.pkl` | Added (~5 lines) |

Components that **stay the same** with distillation:
- **Everything else** ✓
- Architecture is identical
- Inference pipeline is identical
- Loss function is identical (MSE on distances)
- Hybrid slots, template search, MSA, post-processing — all unchanged

**This is why distillation is lower risk than IPA:** it's a training-data
change, not an architecture change. If it doesn't help, you've lost nothing.
If it helps, you've gained RNAPro's knowledge for free.
