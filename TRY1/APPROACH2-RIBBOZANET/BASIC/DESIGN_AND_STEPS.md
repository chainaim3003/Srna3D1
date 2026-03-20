# Approach 2 BASIC — Design Document & Execution Steps

---
---

# PART 1: DESIGN — What BASIC Does and Why

---

## 1.1 Problem Statement

**Input:** RNA sequence string (e.g., "ACCGUGACGGGCCUUUUGGCUAUACGCGGU")
**Output:** 5 sets of (x, y, z) coordinates for the C1' atom of each nucleotide
**Scoring:** TM-score (0–1, higher = better), best-of-5, averaged across all test targets
**Constraint:** Must run on Kaggle T4 GPU (16 GB VRAM), no internet, ≤8 hours

## 1.2 Architecture Decision: Distance Matrix Approach

BASIC uses the **simplest viable architecture** that reasons about pairwise
relationships between nucleotides. It's deliberately NOT the state-of-the-art
(that would be IPA / diffusion — see ADV1). It's designed to be:
- Easy to understand
- Easy to debug
- Fast to train
- A working baseline to submit and get a leaderboard score

### Why Distance Matrix + MDS (not IPA, not direct coordinate MLP)

| Approach | Pros | Cons | BASIC? |
|---|---|---|---|
| **Direct MLP (Linear→coords)** | Simplest possible | Treats each nucleotide independently → physically impossible structures | No — too naive |
| **Distance Matrix + MDS** | Reasons about ALL pairs; MDS is proven math; debuggable | 2-step lossy pipeline; MDS is not differentiable through | **Yes — this is BASIC** |
| **IPA (Invariant Point Attention)** | State-of-the-art; end-to-end differentiable; geometry-aware | Complex to implement; higher VRAM; harder to debug | No — that's ADV1 |
| **Diffusion (Protenix/AlphaFold3)** | Best accuracy; handles uncertainty | Needs 40-80 GB VRAM; months to train | No — needs A100+ |

## 1.3 Two-Phase Design

BASIC has two completely separate phases:

### Phase A: Training (teach the distance head)

```
Known RNA structure (sequence + true coordinates)
    │
    ├── Compute TRUE distance matrix from known coordinates
    │   (just measure distance between every pair of C1' atoms)
    │
    └── RNA sequence
            │
            ▼
        RibonanzaNet (FROZEN) ← pretrained, never changes
            │
            ▼
        Pairwise features (B, N, N, 64)
            │
            ▼
        Distance Head (TRAINABLE) ← small MLP, this is what learns
            │
            ▼
        PREDICTED distance matrix (B, N, N)
            │
            ▼
        Loss = MSE(predicted, true) + bond_constraint + clash_penalty
            │
            ▼
        Backprop → update ONLY the distance head's weights
```

**What gets trained:** ONLY the distance head (~100K parameters)
**What stays frozen:** RibonanzaNet backbone (~100M parameters)

### Phase B: Prediction (generate submission)

```
Test RNA sequence (unknown structure)
    │
    ▼
RibonanzaNet (FROZEN) → Pairwise features
    │
    ▼
Trained Distance Head → Predicted distance matrix (N×N)
    │
    ├── Prediction 1: Clean distances → MDS → 50 refine steps → coords
    ├── Prediction 2: Clean distances → MDS → 100 refine steps → coords
    ├── Prediction 3: Noisy distances (σ=0.3) → MDS → 100 refine steps → coords
    ├── Prediction 4: Noisy distances (σ=0.5) → MDS → 100 refine steps → coords
    └── Prediction 5: Noisy distances (σ=0.7) → MDS → 150 refine steps → coords
    │
    ▼
submission.csv (5 coordinate sets per nucleotide per target)
```

**Why 5 different predictions:** Competition scores best-of-5. Adding noise to
distances and varying refinement steps creates structural diversity — like taking
5 shots at a target instead of 1.

## 1.4 Component Design

### Component 1: Backbone (`models/backbone.py`)

| Property | Value | Source |
|---|---|---|
| Model | RibonanzaNet (official) | github.com/Shujun-He/RibonanzaNet |
| Weights | Pretrained on 2M RNA chemical mapping sequences | kaggle.com/datasets/shujun717/ribonanzanet-weights |
| Parameters | ~100M (all frozen) | Official config |
| Embedding dim (ninp) | 256 | Official config.yaml |
| Pairwise dim | 64 | Official config.yaml: pairwise_dimension=64 |
| Transformer layers | 9 | Official config.yaml |
| Token encoding | A=0, C=1, G=2, U=3, pad=4 | Official Dataset.py |
| Output 1 | Single repr: (B, N, 256) per-nucleotide features | Network.py |
| Output 2 | Pairwise repr: (B, N, N, 64) per-pair features | Network.py |

**Two loading modes:**
- **Option A (recommended):** Clone official repo, import Network.py, hook into internal pairwise_features
- **Option B:** `pip install multimolecule`, use HuggingFace API (but may not expose pairwise repr directly)

**If pairwise repr not directly available:** `PairRepresentationBuilder` constructs it
from single repr via outer product + relative position encoding.

### Component 2: Distance Head (`models/distance_head.py`)

| Property | Value |
|---|---|
| Architecture | 3-layer MLP |
| Input | Pairwise repr: (B, N, N, 64) |
| Layer 1 | Linear(64, 128) → LayerNorm → ReLU → Dropout(0.1) |
| Layer 2 | Linear(128, 128) → LayerNorm → ReLU → Dropout(0.1) |
| Layer 3 | Linear(128, 1) |
| Activation | Softplus (ensures positive output) |
| Post-processing | Symmetrize: dist[i,j] = (dist[i,j] + dist[j,i]) / 2 |
| Post-processing | Zero diagonal: dist[i,i] = 0 |
| Output | (B, N, N) symmetric distance matrix in Ångströms |
| Trainable params | ~100K |

### Component 3: Reconstructor (`models/reconstructor.py`)

**Stage A — Classical MDS (scipy.linalg.eigh):**
1. Square the distance matrix: D²
2. Double-center: B = -½ H D² H (where H = I - 1/N · ones)
3. Eigendecompose B
4. Top 3 eigenvectors × √eigenvalues = (x, y, z) coordinates
5. NOT differentiable — used at inference only

**Stage B — Gradient refinement (torch.optim.Adam):**
1. Start from MDS coordinates
2. Optimize: minimize |current_dist[i,j] - predicted_dist[i,j]|²
3. Plus constraint: consecutive C1' distance ≈ 5.9 Å
4. 50–150 Adam steps at lr=0.01
5. Polishes MDS artifacts

### Component 4: Loss Functions (`losses/`)

| Loss | Weight | What it penalizes | File |
|---|---|---|---|
| **Distance MSE** | 1.0 | (predicted_dist - true_dist)² averaged over upper triangle | distance_loss.py |
| **Bond constraint** | 0.1 | Consecutive C1' distances far from 5.9 Å (with 1.0 Å tolerance) | constraint_loss.py |
| **Clash penalty** | 0.05 | Non-bonded C1' atoms closer than 3.0 Å | constraint_loss.py |

**Total loss = 1.0 × MSE + 0.1 × bond + 0.05 × clash**

### Component 5: Data Pipeline (`data/`)

| File | What it does |
|---|---|
| `dataset.py` | Loads training data from pickle or CIF directory; returns (tokens, distance_matrix, coords, mask) |
| `collate.py` | Pads variable-length sequences to same length within a batch; creates attention masks |
| `augmentation.py` | Random 3D rotation + translation of coordinates (teaches SE(3) invariance) |

### Component 6: Utilities (`utils/`)

| File | What it does |
|---|---|
| `pdb_parser.py` | Extracts C1' coordinates from CIF files using BioPython MMCIFParser |
| `submission.py` | Formats 5 predictions per target into competition CSV format |

## 1.5 Memory Budget

| Component | seq_len=128 | seq_len=256 | seq_len=512 |
|---|---|---|---|
| Backbone (frozen, no grads) | ~2 GB | ~3 GB | ~5 GB |
| Pairwise repr (N²×64) | 0.13 GB | 0.5 GB | 2.0 GB |
| Distance head (with grads) | 0.1 GB | 0.3 GB | 1.0 GB |
| Batch overhead (batch=4) | ~1 GB | ~2 GB | ~4 GB |
| **Total** | **~3.2 GB** | **~5.8 GB** | **~12 GB** |
| Fits RTX 3070 (8 GB)? | Yes | Tight | No |
| Fits Kaggle T4 (16 GB)? | Yes | Yes | Tight |

**Default: max_seq_len=256, batch_size=4** — fits both local and Kaggle.

---
---

# PART 2: FILE MAP — What Each File Does

---

```
BASIC/
│
├── README.md                  Overview, setup instructions, architecture diagram
├── BEGINNERS_GUIDE.md         High-school-level explanation of everything
├── DESIGN_AND_STEPS.md        THIS DOCUMENT
├── config.yaml                All hyperparameters (paths, dims, LR, loss weights)
├── requirements.txt           Python dependencies (torch, numpy, scipy, biopython, etc.)
│
├── models/
│   ├── __init__.py            Exports: load_backbone, tokenize_sequence, DistanceMatrixHead, etc.
│   ├── backbone.py            Load RibonanzaNet (official or multimolecule), extract features
│   │                          Key classes: OfficialBackboneWrapper, MultimoleculeBackboneWrapper
│   │                          Key function: load_backbone(config) → nn.Module
│   ├── distance_head.py       3-layer MLP: pairwise features → distance predictions
│   │                          Key class: DistanceMatrixHead
│   └── reconstructor.py       MDS + gradient refinement: distance matrix → 3D coordinates
│                              Key functions: mds_from_distances_numpy(), refine_coordinates(),
│                                             reconstruct_3d(), reconstruct_batch()
│
├── data/
│   ├── __init__.py            Exports: RNAStructureDataset, load_training_data, etc.
│   ├── dataset.py             Training data loading from pickle or CIF directory
│   │                          Key class: RNAStructureDataset
│   │                          Key functions: load_from_pickle(), load_training_data()
│   ├── collate.py             Batch padding for variable-length sequences
│   │                          Key function: collate_rna_structures()
│   └── augmentation.py        Random 3D rotation and translation
│                              Key functions: random_rotation(), random_translation()
│
├── losses/
│   ├── __init__.py            Exports: DistanceMatrixLoss, BondConstraintLoss, etc.
│   ├── distance_loss.py       MSE on predicted vs true distance matrices
│   │                          Key class: DistanceMatrixLoss
│   ├── constraint_loss.py     Bond length + steric clash penalties
│   │                          Key classes: BondConstraintLoss, ClashPenaltyLoss
│   └── tm_score_approx.py     TM-score computation (numpy, for eval) + differentiable approx (torch)
│                              Key functions: tm_score_numpy(), kabsch_align_numpy(), compute_d0()
│
├── train.py                   Full training loop
│                              - Loads backbone + creates distance head
│                              - Loads data + creates DataLoaders
│                              - Runs epoch loop: forward → loss → backward → update
│                              - Validates + saves best checkpoint
│                              Usage: python train.py --config config.yaml
│
├── predict.py                 Inference + submission generation
│                              - Loads backbone + trained distance head checkpoint
│                              - For each test sequence: predict distances → reconstruct ×5
│                              - Writes submission.csv
│                              Usage: python predict.py --config config.yaml
│                                     --checkpoint checkpoints/best_model.pt
│                                     --test_csv path/to/test_sequences.csv
│
└── utils/
    ├── __init__.py            Exports: extract_rna_structures_from_directory, etc.
    ├── pdb_parser.py          CIF file parsing via BioPython
    │                          Key function: extract_c1prime_from_structure()
    └── submission.py          Competition CSV formatting
                               Key functions: format_submission(), load_test_sequences()
```

---
---

# PART 3: EXECUTION STEPS — What to Do, In Order

---

## Step 0: Prerequisites (one-time setup)

### 0a. Python environment
```bash
# Create a conda environment (recommended)
conda create -n rna3d python=3.10
conda activate rna3d

# Install PyTorch with CUDA
# Check your CUDA version: nvidia-smi
# Then go to https://pytorch.org/get-started/locally/ for the right command
# Example for CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
cd C:\SATHYA\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH2-RIBBOZANET\BASIC
pip install -r requirements.txt
```

### 0b. Verify GPU
```python
import torch
print(torch.cuda.is_available())         # Should print: True
print(torch.cuda.get_device_name(0))     # Should print your GPU name
print(f"{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")  # VRAM
```

## Step 1: Download the 3 external resources

### 1a. Clone RibonanzaNet code (the "recipe")
```bash
cd C:\SATHYA\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1
git clone https://github.com/Shujun-He/RibonanzaNet.git
```
This creates `RibonanzaNet/` with `Network.py`, `config.yaml`, etc.

**Also install RibonanzaNet's dependency:**
```bash
git clone https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
cd Ranger-Deep-Learning-Optimizer
pip install -e .
cd ..
```

### 1b. Download pretrained weights (the "brain")
Go to: https://www.kaggle.com/datasets/shujun717/ribonanzanet-weights
Click "Download" → save to a known location, e.g.:
```
C:\SATHYA\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\ribonanzanet_weights\
```

### 1c. Download training data (the "answers to practice with")
Go to: https://www.kaggle.com/datasets/shujun717/stanford3d-dataprocessing-pickle
Click "Download" → save to a known location, e.g.:
```
C:\SATHYA\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\stanford3d_pickle\
```

## Step 2: Update config.yaml with your actual paths

Open `BASIC/config.yaml` and set:
```yaml
backbone:
  repo_path: "C:/SATHYA/CHAINAIM3003/mcp-servers/STANFORD-RNA/Srna3D1/RibonanzaNet"
  weights_path: "C:/SATHYA/CHAINAIM3003/mcp-servers/STANFORD-RNA/Srna3D1/ribonanzanet_weights/best_model.pt"

data:
  train_pickle_path: "C:/SATHYA/CHAINAIM3003/mcp-servers/STANFORD-RNA/Srna3D1/stanford3d_pickle/data.pkl"
```

**NOTE:** Use forward slashes `/` even on Windows (Python handles them fine).

## Step 3: Smoke test — verify backbone loads

```bash
cd BASIC
python -c "
import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)
from models.backbone import load_backbone
backbone = load_backbone(config)
print('Backbone loaded successfully!')
print(f'ninp={backbone.ninp}, pairwise_dim={backbone.pairwise_dim}')
"
```

**If this fails:** The most likely issue is that `Network.py` imports can't find
dependencies. Check that the Ranger optimizer is installed and that the repo_path
is correct.

## Step 4: Smoke test — verify data loads

```bash
python -c "
import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)
from data.dataset import load_training_data
train_data, val_data = load_training_data(config)
print(f'Train: {len(train_data)} structures')
print(f'Val: {len(val_data)} structures')
if train_data:
    d = train_data[0]
    print(f'First: seq={d[\"sequence\"][:30]}... coords={d[\"coords\"].shape}')
"
```

**If this fails:** The pickle format might not match our parser. The error message
will tell you what the actual format is. You may need to adapt `load_from_pickle()`
in `data/dataset.py`.

## Step 5: Smoke test — full forward pass on dummy data

```bash
python -c "
import torch
from models.distance_head import DistanceMatrixHead

# Simulate pairwise features (B=2, N=20, pair_dim=64)
fake_pair = torch.randn(2, 20, 20, 64)
head = DistanceMatrixHead(pair_dim=64, hidden_dim=128, num_layers=3)
dist = head(fake_pair)
print(f'Distance head output: {dist.shape}')  # Should be (2, 20, 20)
print(f'Symmetric: {torch.allclose(dist, dist.transpose(-1,-2))}')  # Should be True
print(f'Diagonal zeros: {(dist.diagonal(dim1=-2, dim2=-1) == 0).all()}')  # Should be True
print(f'All positive: {(dist >= 0).all()}')  # Should be True

from models.reconstructor import reconstruct_3d
coords = reconstruct_3d(dist[0], method='mds_then_refine', refine_steps=20)
print(f'Reconstructed coords: {coords.shape}')  # Should be (20, 3)
print('All smoke tests passed!')
"
```

## Step 6: Train

```bash
cd BASIC
python train.py --config config.yaml
```

**What to expect:**
- Epoch 1: Loss will be high (model hasn't learned anything yet)
- Epochs 5-20: Loss should decrease steadily
- Epochs 20-50: Loss decrease slows — model is learning finer details
- Epochs 50-100: Diminishing returns; watch for val loss diverging from train loss (overfitting)

**Training time estimates:**

| Hardware | seq_len=128 | seq_len=256 |
|---|---|---|
| RTX 3070 (8 GB) | ~1-2 hours for 100 epochs | ~3-5 hours |
| Kaggle T4 (16 GB) | ~2-3 hours | ~5-8 hours |

**Key things to watch in the console output:**
```
Epoch 1/100 | Train: 45.2341 | Val: 44.8901 | LR: 0.000100 | Time: 32.1s
Epoch 2/100 | Train: 38.1234 | Val: 37.9012 | LR: 0.000100 | Time: 31.8s
...
Epoch 50/100 | Train: 2.3456 | Val: 3.1234 | LR: 0.000050 | Time: 31.5s
  ✓ New best model saved (val_loss=3.1234)
```

**Red flags:**
- Train loss goes DOWN but val loss goes UP → overfitting (reduce epochs or add regularization)
- Loss is NaN → learning rate too high or gradient explosion (reduce LR or check data)
- Loss doesn't decrease at all → bug in data pipeline or backbone not loading correctly

## Step 7: Predict

```bash
python predict.py \
    --config config.yaml \
    --checkpoint checkpoints/best_model.pt \
    --test_csv "C:/path/to/test_sequences.csv" \
    --output submission.csv
```

**What it produces:**
A `submission.csv` file with columns:
```
ID, resname, resid, x_1, y_1, z_1, x_2, y_2, z_2, x_3, y_3, z_3, x_4, y_4, z_4, x_5, y_5, z_5
```

## Step 8: Validate locally (optional but recommended)

If you have the competition's `train_labels.csv` (known true coordinates):
```bash
python -c "
import numpy as np
from losses.tm_score_approx import tm_score_numpy

# Compare your prediction vs known truth for a training sequence
pred = np.load('some_prediction.npy')   # Your predicted coords
true = np.load('some_truth.npy')        # Known true coords
score = tm_score_numpy(pred, true)
print(f'TM-score: {score:.4f}')
# Good: > 0.5
# Great: > 0.7
# State-of-the-art: > 0.8
"
```

## Step 9: Convert to Kaggle notebook for submission

For the actual competition submission, you need to package everything into a
Kaggle Notebook that runs offline on a T4 GPU in ≤8 hours.

### 9a. Upload model weights as Kaggle Dataset
1. Go to kaggle.com → Datasets → New Dataset
2. Upload: `checkpoints/best_model.pt` (your trained distance head)
3. Also upload: RibonanzaNet pretrained weights
4. Name it something like "ribonanzanet-distance-head-v1"

### 9b. Create Kaggle Notebook
1. Go to the competition → Code → New Notebook
2. Attach your dataset + competition data
3. Set: GPU T4, Internet OFF
4. Copy the predict.py logic into notebook cells
5. Adjust paths to `/kaggle/input/your-dataset/...`
6. Run → verify it produces submission.csv within 8 hours
7. Submit

## Step 10: Iterate

After submitting, check your leaderboard score. Then:
- If score is low → check if predictions are reasonable (not all zeros, not straight lines)
- If score is moderate → proceed to ADV1 for improvements
- If score is competitive → ensemble with Approach 1 (TBM) for hybrid submission

---
---

# PART 4: KNOWN LIMITATIONS & UPGRADE PATH

---

| BASIC Limitation | Impact | Fixed in ADV1? |
|---|---|---|
| Backbone completely frozen | Features not optimized for 3D | Yes — Phase 1: unfreeze last layers |
| No template features | Misses strongest signal | Yes — Phase 2: add template input |
| Simple MLP distance head | Can't capture complex geometry | Yes — Phase 3: IPA structure module |
| No MSA features | Misses evolutionary covariation | Yes — Phase 4: MSA integration |
| No recycling | Single-pass prediction | Yes — Phase 3: add recycling |
| Distance → MDS → coords pipeline | Two-step lossy conversion | Yes — Phase 3: direct coord prediction via IPA |

**BASIC is the foundation.** Everything in ADV1 builds on top of it. The training
pipeline, data loading, loss functions, and submission formatting all carry forward.
The upgrades are additive — each phase adds capability without breaking what works.

---
---

# PART 5: OFFICIAL SOURCES REFERENCED

---

| What | URL | What we use from it |
|---|---|---|
| RibonanzaNet repo | https://github.com/Shujun-He/RibonanzaNet | Network.py architecture, config defaults |
| RibonanzaNet weights | https://www.kaggle.com/datasets/shujun717/ribonanzanet-weights | Pretrained .pt file |
| RibonanzaNet paper | https://biorxiv.org/content/10.1101/2024.02.24.581671v1 | Architecture details, fine-tuning strategy |
| multimolecule (HuggingFace) | https://huggingface.co/multimolecule/ribonanzanet | Alternative loading method |
| Training data pickle | https://www.kaggle.com/datasets/shujun717/stanford3d-dataprocessing-pickle | Pre-processed structures |
| Competition page | https://www.kaggle.com/competitions/stanford-rna-3d-folding-2 | Rules, data, submission format |
| Competition paper | https://pmc.ncbi.nlm.nih.gov/articles/PMC12776560/ | Results, winning strategies |
| RNAPro (NVIDIA) | https://github.com/NVIDIA-Digital-Bio/RNAPro | Reference for what state-of-the-art looks like |
| PyTorch docs | https://pytorch.org/docs/stable/ | API reference for nn, optim, amp |
| SciPy linalg | https://docs.scipy.org/doc/scipy/reference/linalg.html | eigh for MDS eigendecomposition |
| BioPython PDB | https://biopython.org/wiki/The_Biopython_Structural_Bioinformatics_FAQ | CIF/PDB file parsing |
