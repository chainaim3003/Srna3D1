# Approach 2 BASIC — Run Steps

## What BASIC Does

```
RNA sequence
  → RibonanzaNet (FROZEN, 100M params) → pairwise features (B, N, N, 64)
  → Distance Head (TRAINABLE, ~100K params, small MLP) → predicted distance matrix (N×N)
  → MDS + gradient refinement → 3D coordinates (N×3)
  → 5 diverse predictions → submission.csv
```

---

## Prerequisites

| Item | Location | How to get |
|------|----------|------------|
| RibonanzaNet repo (cloned) | `Srna3D1/RibonanzaNet/` | `git clone https://github.com/Shujun-He/RibonanzaNet` |
| Ranger optimizer (cloned) | `Srna3D1/Ranger-Deep-Learning-Optimizer/` | `git clone https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer` |
| Pretrained weights | `Srna3D1/ribonanza-weights/RibonanzaNet.pt` | Download from https://www.kaggle.com/datasets/shujun717/ribonanzanet-weights |
| Training data pickle | `Srna3D1/stanford3d-pickle/pdb_xyz_data.pkl` | Download from https://www.kaggle.com/datasets/shujun717/stanford3d-dataprocessing-pickle |
| Test sequences | `Srna3D1/TRY1/APPROACH1-TEMPLATE/test_sequences (1).csv` | From competition data |
| Python 3.10+ | System PATH | https://www.python.org/downloads/ — CHECK "Add to PATH" during install |
| NVIDIA GPU + CUDA | — | Run `nvidia-smi` to verify |

---

## Step 1: Open Project in VSCode

```
File → Open Folder → C:\sathya\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1
```

Open a terminal: `Ctrl+`` or `Terminal → New Terminal`

---

## Step 2: Install Python Packages

Run each command one at a time. Wait for each to finish before running the next.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

```bash
pip install pyyaml numpy scipy pandas biopython tqdm einops matplotlib polars
```

```bash
cd C:\sathya\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\Ranger-Deep-Learning-Optimizer
pip install -e .
```

---

## Step 3: Navigate to BASIC

```bash
cd C:\sathya\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH2-RIBBOZANET\BASIC
```

---

## Step 4: Run Smoke Tests

Run each test one at a time. All three must print SUCCESS before proceeding.

**Test 1 — Backbone loads:**

```bash
python -c "import yaml; config=yaml.safe_load(open('config.yaml')); from models.backbone import load_backbone; b=load_backbone(config); print(f'SUCCESS: ninp={b.ninp}, pair_dim={b.pairwise_dim}')"
```

Expected output:
```
Loaded official config from .../RibonanzaNet/configs/pairwise.yaml
constructing 9 ConvTransformerEncoderLayers
Loaded backbone weights from .../ribonanza-weights/RibonanzaNet.pt
Backbone frozen — parameters will NOT be updated during training.
SUCCESS: ninp=256, pair_dim=64
```

**Test 2 — Data loads:**

```bash
python -c "import yaml; config=yaml.safe_load(open('config.yaml')); from data.dataset import load_training_data; t,v=load_training_data(config); print(f'SUCCESS: {len(t)} train, {len(v)} val')"
```

Expected output:
```
Pickle format: dict with 844 structures
  Parsed: 789 structures
After filtering to max_len=256: 734 structures
Train: 661, Val: 73
SUCCESS: 661 train, 73 val
```

**Test 3 — Forward pass works:**

```bash
python -c "import torch; from models.distance_head import DistanceMatrixHead; h=DistanceMatrixHead(64,128,3); d=h(torch.randn(2,20,20,64)); print(f'Distance head: {d.shape}'); from models.reconstructor import reconstruct_3d; c=reconstruct_3d(d[0],method='mds_then_refine',refine_steps=20); print(f'Reconstructed: {c.shape}'); print('SUCCESS: All smoke tests passed!')"
```

Expected output:
```
Distance head: torch.Size([2, 20, 20])
Reconstructed: torch.Size([20, 3])
SUCCESS: All smoke tests passed!
```

---

## Step 5: Train

### Fresh start (first time):

```bash
python train.py --config config.yaml
```

### Resume from checkpoint (after stopping with Ctrl+C):

```bash
python train.py --config config.yaml --resume checkpoints/latest_model.pt
```

Or resume from best model:

```bash
python train.py --config config.yaml --resume checkpoints/best_model.pt
```

### Controlling epoch count:

Edit `config.yaml` → change `training.epochs` to desired number (e.g., 35).
When resuming, training runs from the saved epoch to the epoch count in config.

### What to expect:

```
Epoch 1/35  | Train: 800.xx | Val: 850.xx | LR: 0.000100 | Time: ~570s
Epoch 5/35  | Train: 700.xx | Val: 740.xx | LR: 0.000098 | Time: ~570s
Epoch 15/35 | Train: 600.xx | Val: 650.xx | LR: 0.000085 | Time: ~570s
Epoch 35/35 | Train: 400.xx | Val: 450.xx | LR: 0.000050 | Time: ~570s
```

- **Both train and val loss going down** = good (model is learning)
- **Val loss going UP while train goes down** = overfitting (stop training)
- **Loss is NaN** = something broke (stop and debug)
- **~9.5 min per epoch** on local GPU is normal
- **"New best model saved"** messages = good

### Checkpoints saved:

| File | What | When saved |
|------|------|------------|
| `checkpoints/best_model.pt` | Best val loss so far | Every time val loss improves |
| `checkpoints/latest_model.pt` | Most recent completed epoch | Every epoch (for resume) |
| `checkpoints/model_epochN.pt` | Periodic snapshot | Every 10 epochs |

---

## Step 6: Generate Predictions

Open a **second terminal** (training can keep running in the first):

```bash
cd C:\sathya\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH2-RIBBOZANET\BASIC

python predict.py --config config.yaml --checkpoint checkpoints/best_model.pt --test_csv "../../APPROACH1-TEMPLATE/test_sequences (1).csv" --output submission.csv
```

This generates `submission.csv` with 5 diverse 3D predictions for each of the 28 test targets.

### Verify the output:

```bash
python -c "import pandas as pd; df=pd.read_csv('submission.csv'); print(f'Shape: {df.shape}'); print(f'Targets: {df.ID.str.split(\"_\").str[0].nunique()}'); print(f'Resnames: {df.resname.unique()[:10]}'); print(df.head())"
```

Expected: resname column shows A, U, G, C (RNA bases), NOT >, 8, Z, etc.

---

## Step 7: Submit to Kaggle

### Option A: Upload CSV directly

1. Go to https://www.kaggle.com/competitions/stanford-rna-3d-folding-2
2. Click **"Submit Predictions"**
3. Upload `submission.csv`

### Option B: Run on Kaggle notebook (with GPU)

See the "Running on Kaggle" section below.

---

## Running on Kaggle (with T4 GPU)

### Setup: Upload code as Kaggle Dataset

1. Zip the `BASIC` folder and the `RibonanzaNet` folder
2. Go to https://www.kaggle.com/datasets → **"+ New Dataset"**
3. Upload each zip as a private dataset

### Create notebook

1. Go to competition page → Code → **New Notebook**
2. Settings: **GPU T4 x2**, **Internet ON**
3. Attach datasets:
   - Your code dataset (e.g., `rna-basic`)
   - `shujun717/ribonanzanet-weights`
   - `shujun717/stanford3d-dataprocessing-pickle`
   - `stanford-rna-3d-folding-2` (competition data)

### Notebook cells

**Cell 1 — Setup:**
```python
import os, shutil

# Check what's in your dataset (Kaggle lowercases the name)
print(os.listdir('/kaggle/input/'))

# Copy code to writable location
shutil.copytree('/kaggle/input/rna-basic', '/kaggle/working/BASIC')
os.chdir('/kaggle/working/BASIC')

!pip install einops polars biopython tqdm --quiet
```

**Cell 2 — Update config paths:**
```python
import yaml

config = yaml.safe_load(open('config.yaml'))
config['backbone']['repo_path'] = '/kaggle/input/my-ribonanzanet-repo/RibonanzaNet'
config['backbone']['weights_path'] = '/kaggle/input/ribonanzanet-weights/RibonanzaNet.pt'
config['data']['train_pickle_path'] = '/kaggle/input/stanford3d-dataprocessing-pickle/pdb_xyz_data.pkl'
config['data']['test_csv_path'] = '/kaggle/input/stanford-rna-3d-folding-2/test_sequences.csv'
config['training']['save_dir'] = '/kaggle/working/checkpoints'

with open('config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print("Config updated for Kaggle paths")
```

**Cell 3 — Train:**
```python
!python train.py --config config.yaml
```

**Cell 4 — Predict:**
```python
!python predict.py --config config.yaml \
    --checkpoint /kaggle/working/checkpoints/best_model.pt \
    --test_csv /kaggle/input/stanford-rna-3d-folding-2/test_sequences.csv \
    --output /kaggle/working/submission.csv
```

**Cell 5 — Verify:**
```python
import pandas as pd
df = pd.read_csv('/kaggle/working/submission.csv')
print(f'Shape: {df.shape}')
print(f'Resnames: {df.resname.unique()}')
print(df.head())
```

---

## Folder Structure

```
Srna3D1/
├── RibonanzaNet/                          # Cloned backbone code
│   ├── Network.py                         # Model architecture
│   ├── configs/pairwise.yaml              # Official model config
│   └── dropout.py, Functions.py           # Dependencies
├── Ranger-Deep-Learning-Optimizer/        # Optimizer (pip install -e .)
├── ribonanza-weights/
│   └── RibonanzaNet.pt                    # 43.3 MB pretrained weights
├── stanford3d-pickle/
│   └── pdb_xyz_data.pkl                   # 52.3 MB training data (844 structures)
└── TRY1/APPROACH2-RIBBOZANET/BASIC/
    ├── config.yaml                        # All settings in one place
    ├── train.py                           # Training loop (with --resume)
    ├── predict.py                         # Inference → submission.csv
    ├── models/
    │   ├── backbone.py                    # Frozen RibonanzaNet wrapper
    │   ├── distance_head.py               # Trainable MLP (3 layers)
    │   └── reconstructor.py               # MDS + gradient refinement → 3D
    ├── data/
    │   ├── dataset.py                     # Loads pickle, extracts C1' coords
    │   ├── collate.py                     # Batch padding
    │   └── augmentation.py                # Random rotation/translation
    ├── losses/
    │   ├── distance_loss.py               # MSE on distance matrices
    │   ├── constraint_loss.py             # Bond length + clash penalties
    │   └── tm_score_approx.py             # Differentiable TM-score
    ├── utils/
    │   ├── submission.py                  # Formats competition CSV
    │   └── pdb_parser.py                  # CIF file parser
    └── checkpoints/
        ├── best_model.pt                  # Best validation checkpoint
        ├── latest_model.pt                # Most recent epoch (for resume)
        └── model_epoch10.pt               # Periodic snapshots
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Python was not found` | Install Python 3.10+, CHECK "Add to PATH" |
| `No module named 'torch'` | `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| `No module named 'einops'` | `pip install einops matplotlib polars` |
| `'ConfigNamespace' has no attribute 'k'` | backbone.py needs update — loads `configs/pairwise.yaml` not `config.yaml` |
| `float() argument must be ... 'defaultdict'` | dataset.py needs update — C1' = `sugar_ring[0]` per nucleotide |
| resname shows `>`, `8`, `Z` instead of A/U/G/C | submission.py needs update — use `sequence` column, not `all_sequences` |
| `No training data loaded` | Check `train_pickle_path` in config.yaml points to actual file |
| Loss is NaN | Reduce learning_rate to 5e-5, check for missing coordinates in data |
| Training takes >10 min/epoch | Normal for 661 structures through 100M-param backbone on local GPU |
| Ctrl+C lost my progress | Resume with `--resume checkpoints/latest_model.pt` |

---

## Key Config Settings (config.yaml)

| Setting | Default | What it controls |
|---------|---------|-----------------|
| `training.epochs` | 35 | How many epochs to train |
| `training.batch_size` | 4 | Sequences per batch (limited by GPU memory) |
| `training.learning_rate` | 0.0001 | Starting learning rate (cosine decay) |
| `data.max_seq_len` | 256 | Maximum RNA sequence length |
| `data.val_fraction` | 0.1 | Fraction of data held out for validation |
| `backbone.freeze` | true | Keep backbone frozen (BASIC always true) |
| `prediction.num_predictions` | 5 | Competition requires exactly 5 |
| `training.save_every` | 10 | Save periodic checkpoint every N epochs |
