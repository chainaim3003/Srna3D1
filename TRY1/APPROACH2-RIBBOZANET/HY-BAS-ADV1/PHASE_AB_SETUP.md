# LOCAL + KAGGLE SETUP GUIDE: Phase A/B Training Pipeline
## Runs 4, 5, and 6 — Stanford RNA 3D Folding Part 2

---

## 1. The Checkpoint Chain

Each run's Phase A produces a checkpoint that the next run consumes. Phase B files load their own Phase A's checkpoint and produce `submission.csv`.

```
Run 3 OptB (already done)
  └── adv1_best_run3optb_model.pt
        │
        ▼
Run 4 Phase A (trains distance head + MSA)
  └── adv1_best_model.pt
        │
        ├──────────────────────────────────────────┐
        ▼                                          ▼
Run 4 Phase B (inference → submission.csv)   Run 5 Phase A (trains IPA module)
                                               └── adv1_run5_best_model.pt
                                                     │
                                                     ├──────────────────────────┐
                                                     ▼                          ▼
                                               Run 5 Phase B              Run 6 Phase B
                                               (inference →               (IPA + RNAPro →
                                                submission.csv)            submission.csv)
                                                                                ▲
Run 6 Phase A (RNAPro on cloud A100)                                           │
  └── rnapro_part2_coords.npz ─────────────────────────────────────────────────┘
```

---

## 2. What Each Phase A Needs and Produces

| Run | Phase A Script | Warm-starts From | Produces | Where to Run |
|-----|----------------|-----------------|----------|-------------|
| **4** | `hy_bas_adv1_run4_commit_PhaseA_NB.py` | `adv1_best_run3optb_model.pt` | `adv1_best_model.pt` | Local RTX 3070 Ti or Kaggle |
| **5** | `hy_bas_adv1_run5_api_PhaseA_NB.py` | `adv1_best_model.pt` (from Run 4) | `adv1_run5_best_model.pt` | Local RTX 3070 Ti or Kaggle |
| **6** | `hy_bas_adv1_run6_PhaseA_NB.py` | None (standalone RNAPro) | `rnapro_part2_coords.npz` | Cloud A100 (Vast.ai/RunPod) |

| Run | Phase B Script | Loads Checkpoint | Also Needs | Produces |
|-----|----------------|-----------------|------------|----------|
| **4** | `hy_bas_adv1_run4_commit_PhaseB_NB.py` | `adv1_best_model.pt` | — | `submission.csv` |
| **5** | `hy_bas_adv1_run5_api_PhaseB_NB.py` | `adv1_run5_best_model.pt` | — | `submission.csv` |
| **6** | `hy_bas_adv1_run6_PhaseB_NB.py` | `adv1_run5_best_model.pt` | `rnapro_part2_coords.npz` | `submission.csv` |

---

## 3. Local Machine Setup (Your RTX 3070 Ti)

### 3.1 Prerequisites (already done)

You have already completed these steps:
- NVIDIA Driver 595.97 installed
- Armoury Crate set to Optimized (not Eco Mode)
- PyTorch 2.5.1+cu121 installed
- einops 0.8.2 and biopython 1.86 installed
- GPU verified with `gpu_check.py`

### 3.2 Python Dependencies (complete list)

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install einops biopython numpy pandas scipy tqdm pyyaml
```

All of these are already installed on your machine.

### 3.3 Data Files Needed Locally

You need to copy these files from Kaggle to your local machine (external hard drive recommended).

Set your `DATA_ROOT` to wherever you put them. Example: `E:/kaggle_data`

```
DATA_ROOT/
├── stanford-rna-3d-folding-2/          # Competition data folder
│   ├── train_sequences.csv             # ~1 MB
│   ├── test_sequences.csv              # ~10 KB
│   ├── train_labels.csv                # ~50 MB
│   └── sample_submission.csv           # ~200 KB
│
├── rna_sequences.csv                   # Extended sequences (~100 MB)
├── rna_coordinates.csv                 # Extended coordinates (~100 MB)
├── pdb_xyz_data.pkl                    # Training structures (~52 MB)
│
├── RibonanzaNet.pt                     # Backbone weights (~43 MB)
├── best_model.pt                       # BASIC distance head (~312 KB)
├── adv1_best_run3optb_model.pt         # Run 3 OptB checkpoint (~9 MB)
│
└── ribonanzanet_repo/                  # RibonanzaNet source code
    ├── Network.py
    ├── configs/
    │   └── pairwise.yaml
    └── ...
```

**Where to get these files:**

| File | Source |
|------|--------|
| `train_sequences.csv`, `test_sequences.csv`, etc. | Kaggle competition page → Data tab → Download All |
| `rna_sequences.csv`, `rna_coordinates.csv` | Your existing Kaggle dataset (rna-cif-to-csv) |
| `pdb_xyz_data.pkl` | Your existing Kaggle dataset |
| `RibonanzaNet.pt` | Your existing Kaggle dataset (ADV1 weights) |
| `best_model.pt` | Your existing Kaggle dataset (ADV1 weights) |
| `adv1_best_run3optb_model.pt` | Your local file from Run 3 OptB |
| `ribonanzanet_repo/` | `git clone https://github.com/DasLab/RibonanzaNet` |

### 3.4 Running Phase A Locally

**Step 1: Edit Cell 0 in the Phase A file**

Open the file and uncomment the LOCAL block. Comment out the KAGGLE block. Set `DATA_ROOT`:

```python
# --- OPTION A: LOCAL PATHS (uncomment and edit) ---------------
DATA_ROOT          = 'E:/kaggle_data'           # <-- YOUR PATH
COMP_BASE          = f'{DATA_ROOT}/stanford-rna-3d-folding-2'
EXTENDED_SEQ_CSV   = f'{DATA_ROOT}/rna_sequences.csv'
EXTENDED_COORD_CSV = f'{DATA_ROOT}/rna_coordinates.csv'
TRAIN_PICKLE       = f'{DATA_ROOT}/pdb_xyz_data.pkl'
BACKBONE_WEIGHTS   = f'{DATA_ROOT}/RibonanzaNet.pt'
REPO_PATH          = f'{DATA_ROOT}/ribonanzanet_repo'
RUN3_CHECKPOINT    = f'{DATA_ROOT}/adv1_best_run3optb_model.pt'
OUTPUT_DIR         = './run4_output'
PLATFORM           = 'LOCAL'
# -----------------------------------------------------------------

# --- OPTION B: KAGGLE PATHS (default — auto-discovers) -----------
# EXTENDED_SEQ_CSV   = None
# ...
# -----------------------------------------------------------------
```

**Step 2: Increase epochs (optional but recommended)**

Since you have no time limit locally, change:
```python
TRAIN_EPOCHS = 80  # or 100 (default is 50 for Kaggle)
```

**Step 3: Run it**

```bash
cd C:\SATHYA\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH2-RIBBOZANET\HY-BAS-ADV1\kaggle
python hy_bas_adv1_run4_commit_PhaseA_NB.py
```

**Step 4: Find the output**

The checkpoint is saved to `OUTPUT_DIR`:
- Run 4: `./run4_output/adv1_best_model.pt`
- Run 5: `./run5_output/adv1_run5_best_model.pt`

---

## 4. Execution Order

### Fastest path: Run 4 → Run 5 → submit both

```
1. Run 4 Phase A locally      (2-4 hours, overnight OK)
2. Upload adv1_best_model.pt to Kaggle as dataset
3. Run 4 Phase B on Kaggle    (30-45 min → submission.csv → submit)
4. Copy adv1_best_model.pt to DATA_ROOT for Run 5
5. Run 5 Phase A locally      (2-4 hours, overnight OK)
6. Upload adv1_run5_best_model.pt to Kaggle as dataset
7. Run 5 Phase B on Kaggle    (30-45 min → submission.csv → submit)
```

### Full path with Run 6 (adds RNAPro):

```
8. Rent A100 on Vast.ai       (~$2-5)
9. Run 6 Phase A on cloud     (1-2 hours)
10. Download rnapro_part2_coords.npz
11. Upload to Kaggle as dataset
12. Run 6 Phase B on Kaggle   (2-4 hours → submission.csv → submit)
```

---

## 5. Uploading Phase A Outputs to Kaggle

After each Phase A completes locally, you need to upload its output as a Kaggle dataset so Phase B can find it.

### For Run 4 Phase A output:

1. Go to https://www.kaggle.com → Your Work → Datasets → + New Dataset
2. Name it anything (e.g., `run4-checkpoint`)
3. Upload: `adv1_best_model.pt` (from `./run4_output/`)
4. Click Create
5. When creating Phase B notebook, attach this dataset

### For Run 5 Phase A output:

1. Same process, name it e.g., `run5-checkpoint`
2. Upload: `adv1_run5_best_model.pt` (from `./run5_output/`)
3. Attach to Run 5 Phase B notebook AND Run 6 Phase B notebook

### For Run 6 Phase A output:

1. Same process, name it e.g., `rnapro-templates`
2. Upload: `rnapro_part2_coords.npz` (from cloud output)
3. Attach to Run 6 Phase B notebook

**How Phase B finds the checkpoint:**
The code walks `/kaggle/input/` and searches by filename. Dataset names don't matter.

---

## 6. Kaggle Phase B Setup (all runs)

### Datasets to Attach for Phase B Notebooks

| Dataset | Run 4 Phase B | Run 5 Phase B | Run 6 Phase B |
|---------|:---:|:---:|:---:|
| Competition data (auto after joining) | ✅ | ✅ | ✅ |
| RibonanzaNet repo (`Network.py`) | ✅ | ✅ | ✅ |
| ADV1 weights (`RibonanzaNet.pt`) | ✅ | ✅ | ✅ |
| Extended RNA data (CSVs) | ✅ | ✅ | ✅ |
| Biopython wheel (`.whl`) | ✅ | ✅ | ✅ |
| **Run 4 checkpoint** (`adv1_best_model.pt`) | ✅ | — | — |
| **Run 5 checkpoint** (`adv1_run5_best_model.pt`) | — | ✅ | ✅ |
| **RNAPro templates** (`rnapro_part2_coords.npz`) | — | — | ✅ |
| Training data (`pdb_xyz_data.pkl`) | — | — | — |
| Run 3 OptB checkpoint | — | — | — |

Note: Phase B does NOT need `pdb_xyz_data.pkl` or Run 3 checkpoint (those are only for training).

### Kaggle Notebook Settings

| Setting | Value |
|---------|-------|
| Accelerator | GPU T4 x2 |
| Persistence | Files only |
| Internet | OFF |
| Language | Python |

### Running Phase B

1. Create new notebook
2. Attach all required datasets (see table above)
3. Paste the Phase B `.py` file into one cell
4. Save Version → Save & Run All (Commit)
5. Wait for completion → submit `submission.csv`

---

## 7. Run 6 Cloud Setup (A100)

Run 6 Phase A is different — it runs RNAPro (488M params, needs 25+ GB VRAM).

### Cloud options:

| Provider | GPU | Cost | Link |
|----------|-----|------|------|
| Vast.ai | A100 80GB | ~$0.80-1.50/hr | vast.ai |
| RunPod | A100 80GB | ~$1.00-1.50/hr | runpod.io |
| Lambda Labs | A100 80GB | ~$1.50/hr | lambdalabs.com |

### Setup on cloud:

```bash
# 1. Clone RNAPro
git clone https://github.com/NVIDIA-Digital-Bio/RNAPro
cd RNAPro && pip install -e .

# 2. Download model checkpoint from HuggingFace
# (follow instructions at https://huggingface.co/nvidia/RNAPro-Public-Best-500M)

# 3. Upload test_sequences.csv from Kaggle

# 4. Run Phase A
python hy_bas_adv1_run6_PhaseA_NB.py \
  --test-csv test_sequences.csv \
  --checkpoint ./rnapro_public_best/model.pt \
  --output-dir ./rnapro_output \
  --seeds 42,123,456,789,1024

# 5. Download outputs
# rnapro_output/rnapro_part2_coords.npz  (~5-50 MB)
```

---

## 8. File Inventory

All files live in:
```
HY-BAS-ADV1/kaggle/
```

| File | Type | Status |
|------|------|--------|
| `hy_bas_adv1_run4_commit_NB.py` | Original monolithic | Unchanged |
| `hy_bas_adv1_run4_commit_PhaseA_NB.py` | Run 4 training | NEW |
| `hy_bas_adv1_run4_commit_PhaseB_NB.py` | Run 4 inference | NEW |
| `hy_bas_adv1_run5_api_NB.py` | Original monolithic | Unchanged |
| `hy_bas_adv1_run5_api_PhaseA_NB.py` | Run 5 training | NEW |
| `hy_bas_adv1_run5_api_PhaseB_NB.py` | Run 5 inference | NEW |
| `hy_bas_adv1_run6_PhaseA_NB.py` | RNAPro cloud | NEW |
| `hy_bas_adv1_run6_PhaseB_NB.py` | Run 6 inference | NEW |

---

## 9. Quick Reference Card

### Run 4
```
LOCAL:   python hy_bas_adv1_run4_commit_PhaseA_NB.py
         → produces: ./run4_output/adv1_best_model.pt
UPLOAD:  adv1_best_model.pt → Kaggle dataset
KAGGLE:  hy_bas_adv1_run4_commit_PhaseB_NB.py → submission.csv
```

### Run 5
```
LOCAL:   python hy_bas_adv1_run5_api_PhaseA_NB.py
         → produces: ./run5_output/adv1_run5_best_model.pt
UPLOAD:  adv1_run5_best_model.pt → Kaggle dataset
KAGGLE:  hy_bas_adv1_run5_api_PhaseB_NB.py → submission.csv
```

### Run 6
```
CLOUD:   python hy_bas_adv1_run6_PhaseA_NB.py --test-csv ... --checkpoint ...
         → produces: rnapro_part2_coords.npz
UPLOAD:  rnapro_part2_coords.npz → Kaggle dataset
KAGGLE:  hy_bas_adv1_run6_PhaseB_NB.py → submission.csv
         (also needs Run 5's checkpoint attached)
```

---

## 10. Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| GPU not detected locally | Armoury Crate in Eco Mode | Switch to Optimized, restart |
| `Network.py` not found | `REPO_PATH` wrong | Set to folder containing `Network.py` |
| CUDA out of memory | Sequence too long or batch too large | Reduce `MAX_SEQ_LEN` or `BATCH_SIZE` |
| Phase B says "checkpoint not found" | Dataset not attached on Kaggle | Attach the dataset containing the `.pt` file |
| Run 5 Phase A can't find Run 4 checkpoint | Wrong path in Cell 0 | Set `RUN4_CHECKPOINT` to where you saved `adv1_best_model.pt` |
| Run 6 Phase B degrades to Run 5 | `rnapro_part2_coords.npz` not found | This is OK — Run 6 gracefully falls back |
