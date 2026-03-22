# HY-BAS-ADV1: Hybrid Design Document
## Holistic Design for Kaggle Submission

---

## CRITICAL CORRECTION: BASIC's Kaggle Score (0.092) Was NOT Neural Network Inference

What BASIC's Kaggle notebook (RNA-Basic, Version 6) actually did:

```
1. Read a pre-made submission_fixed.csv from an uploaded Kaggle dataset
2. Read sample_submission.csv from competition data
3. Map our pre-made predictions to sample IDs
4. Write submission.csv
```

It did NOT:
- Load RibonanzaNet.pt (the 43MB backbone)
- Load best_model.pt (the trained distance head)
- Run PyTorch inference on any sequences
- Import torch, or any model code

This means: **We have NEVER proven that our neural network can load and
run on Kaggle.** This is a risk that must be addressed FIRST.

---

## TRAINING vs INFERENCE — Explained Simply

Think of it like studying for an exam vs taking the exam.

### TRAINING (Already Done — Happened Locally on Our GPU)
- **What:** We showed the model 661 RNA structures from PDB.
  For each one, we said: "Here's the sequence ACGUU..., and here are
  the true distances between every pair of nucleotides."
  The model learned to predict distances from sequences.
- **Output:** best_model.pt (312KB file) — this is the "learned brain"
  of the distance prediction network
- **Where:** Ran on our local GPU. Took ~2-4 hours. 50 epochs.
- **Do we need to redo this?** YES, partially. For ADV1, we need to
  retrain with template features (80 channels instead of 64).
  But we warm-start from best_model.pt, so it converges faster (~30 epochs).

### INFERENCE (Needs to Happen on Kaggle)
- **What:** Given a NEW sequence we've never seen before, predict its
  3D structure. No learning happens. We just use the trained "brain"
  to make predictions.
- **Steps:**
  1. Take sequence "ACGUU..."
  2. Feed it through frozen RibonanzaNet backbone → pairwise features
  3. Feed pairwise features through trained distance head → distance matrix
  4. Convert distance matrix to 3D coordinates (MDS + refinement)
- **Where:** Must happen on Kaggle (Code Competition)
- **Time per target:** ~1-5 seconds
- **This is what we've NEVER tested on Kaggle**

### WHY THE DISTINCTION MATTERS

Training = expensive, one-time, local GPU, produces weights file
Inference = cheap, per-sequence, must work on Kaggle, uses weights file

For ADV1, we need:
- ~2-4 hours of LOCAL training (retrain with template features)
- ~5-30 minutes of KAGGLE inference (run the trained model on test data)

---

## "TEST LOCALLY" — What It Means Concretely

"Test locally if possible (requires RibonanzaNet repo + weights)" means:

1. You have the RibonanzaNet repo cloned locally:
   ```
   Srna3D1/RibonanzaNet/          (the cloned github repo)
   ```

2. You have the pretrained backbone weights:
   ```
   Srna3D1/ribonanza-weights/RibonanzaNet.pt   (43MB file)
   ```

3. You have the trained distance head:
   ```
   BASIC/checkpoints/best_model.pt              (312KB file)
   ```

4. You run predict.py on your local machine:
   ```
   cd BASIC
   python predict.py --config config.yaml
   ```
   This loads both files, runs inference on test_sequences.csv,
   and produces submission.csv locally.

5. If this works → the model code is correct
6. Then you package the same code + weights for Kaggle

If you DON'T have items 1 and 2 locally, you can skip local testing
and go straight to Kaggle — but debugging on Kaggle is slower.

---

## PRE-STEP 1: Verify Neural Network Runs on Kaggle

### What Needs to be Verified

We need to confirm that on Kaggle:
- PyTorch is available (it is — Kaggle has it pre-installed)
- RibonanzaNet.pt can be loaded from an uploaded dataset
- Our backbone.py code can import and use the RibonanzaNet repo
- best_model.pt can be loaded
- Inference produces coordinates (not errors)
- The whole thing runs within the 8-hour time limit

### How to Test (Simple Test Notebook)

Create a Kaggle notebook with ONE cell:

```python
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Test: Can we load the weights?
import sys
sys.path.append("/kaggle/input/ribonanzanet-repo/RibonanzaNet")

# Try importing
from Network import RibonanzaNet
print("RibonanzaNet imported successfully")

# Load backbone weights
weights = torch.load("/kaggle/input/adv1-weights/RibonanzaNet.pt",
                     map_location="cpu")
print("Backbone weights loaded:", len(weights), "tensors")

# Load distance head weights
head_weights = torch.load("/kaggle/input/adv1-weights/best_model.pt",
                          map_location="cpu")
print("Distance head loaded:", type(head_weights))

print("ALL CHECKS PASSED")
```

This requires uploading 3 Kaggle datasets:
- ribonanzanet-repo: The RibonanzaNet/ folder
- adv1-weights: RibonanzaNet.pt + best_model.pt
- adv1-code: All our Python files (backbone.py, distance_head.py, etc.)

### If This Fails

Common issues and fixes:
- "No module named Network" → RibonanzaNet repo path wrong
- "CUDA out of memory" → Reduce batch size or use CPU
- "KeyError in state_dict" → Weight file mismatch with code
- Timeout → Model too slow, need to optimize

---

## HOLISTIC HYBRID DESIGN

### What HY-BAS-ADV1 Does (The Full Pipeline)

```
┌─────────────────────────────────────────────────────────────┐
│                    KAGGLE NOTEBOOK                           │
│                                                              │
│  PHASE 1: Template Search (from Fork 1)                     │
│  ┌────────────────────────────────────────┐                  │
│  │ test_sequences.csv                      │                 │
│  │         │                               │                 │
│  │         ▼                               │                 │
│  │    MMseqs2 Search                       │                 │
│  │         │                               │                 │
│  │         ▼                               │                 │
│  │  Template coords + e-values             │                 │
│  │  (per target)                           │                 │
│  └────────────────────────────────────────┘                  │
│                      │                                       │
│                      ▼                                       │
│  PHASE 2: Neural Inference (from BASIC + ADV1)              │
│  ┌────────────────────────────────────────┐                  │
│  │ For each target:                        │                 │
│  │                                         │                 │
│  │   sequence ──► RibonanzaNet backbone    │                 │
│  │                     │                   │                 │
│  │              pairwise features (N,N,64) │                 │
│  │                     │                   │                 │
│  │   template coords ──► template encoder  │                 │
│  │                     │                   │                 │
│  │              template features (N,N,16) │                 │
│  │                     │                   │                 │
│  │              CONCATENATE (N,N,80)       │                 │
│  │                     │                   │                 │
│  │              Distance Head (ADV1)       │                 │
│  │                     │                   │                 │
│  │              distance matrix (N,N)      │                 │
│  │                     │                   │                 │
│  │              MDS + Refinement           │                 │
│  │                     │                   │                 │
│  │              3D coordinates (N,3) x 5   │                 │
│  └────────────────────────────────────────┘                  │
│                      │                                       │
│                      ▼                                       │
│  PHASE 3: Post-Processing (from BASIC + Option B)           │
│  ┌────────────────────────────────────────┐                  │
│  │  Read sample_submission.csv             │                 │
│  │  Map predictions to expected IDs        │                 │
│  │  Fill zeros for missing targets         │                 │
│  │  Write submission.csv                   │                 │
│  └────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

### Training Required (LOCAL, Before Kaggle)

ADV1 needs a RETRAINED distance head because:
- BASIC distance head: Linear(64 → 128) as first layer
- ADV1 distance head: Linear(80 → 128) as first layer (16 extra template channels)

Training plan:
```
Input:   BASIC/checkpoints/best_model.pt (existing trained weights)
Process: Retrain with template features for ~30 epochs on local GPU
Output:  HY-BAS-ADV1/checkpoints/adv1_best_model.pt (new trained weights)
```

Warm-start strategy:
```python
# Load BASIC weights (first layer is 64→128)
basic_state = torch.load("BASIC/checkpoints/best_model.pt")

# Create ADV1 distance head (first layer is 80→128)
adv1_head = DistanceMatrixHead(pair_dim=80, hidden_dim=128, ...)

# Copy what we can:
# First layer weight: (128, 64) → expand to (128, 80)
# - Columns 0-63: copy from BASIC
# - Columns 64-79: random initialization (template channels)
# All other layers: identical shape, copy directly
```

This means the model starts with everything BASIC already learned,
plus random initialization for the 16 new template channels. It only
needs to learn what to do with template information — everything else
is already trained.

Estimated training time: 1-3 hours on a local GPU.

---

## DIRECTORY STRUCTURE

```
HY-BAS-ADV1/
├── DESIGN.md                       (THIS FILE)
├── config_adv1.yaml                (modified config for ADV1)
├── models/
│   ├── backbone.py                 (COPY from BASIC — unchanged)
│   ├── distance_head.py            (COPY from BASIC — change pair_dim default)
│   ├── reconstructor.py            (COPY from BASIC — unchanged)
│   ├── template_encoder.py         (NEW — ~100 lines)
│   ├── template_loader.py          (NEW — ~100 lines)
│   └── __init__.py
├── data/                           (COPY from BASIC — unchanged)
│   ├── dataset.py
│   └── collate.py
├── losses/                         (COPY from BASIC — unchanged)
│   ├── distance_loss.py
│   └── constraint_loss.py
├── utils/                          (COPY from BASIC — unchanged)
│   └── submission.py
├── train_adv1.py                   (MODIFIED from BASIC train.py)
├── predict_adv1.py                 (MODIFIED from BASIC predict.py)
├── checkpoints/
│   └── (adv1_best_model.pt — created after training)
└── kaggle/
    ├── kaggle_hybrid_notebook.py   (THE FULL KAGGLE NOTEBOOK)
    └── KAGGLE_UPLOAD_GUIDE.md      (What to upload as datasets)
```

### What Gets Uploaded to Kaggle (3 Datasets)

| Kaggle Dataset Name | Contents | Size |
|---------------------|----------|------|
| adv1-weights | adv1_best_model.pt + RibonanzaNet.pt | ~44 MB |
| adv1-code | All .py files from HY-BAS-ADV1/ | ~50 KB |
| ribonanzanet-repo | Cloned RibonanzaNet/ folder (for imports) | ~10 MB |

---

## STEP-BY-STEP EXECUTION PLAN

### Phase 1: Setup + Pre-Step 1 Verification (2-3 hours)

**Step 1.1: Create HY-BAS-ADV1 directory**
- Copy unchanged files from BASIC
- Create new template_encoder.py and template_loader.py stubs

**Step 1.2: Upload test datasets to Kaggle**
- Upload RibonanzaNet.pt + best_model.pt as "adv1-weights-test"
- Upload RibonanzaNet/ repo folder as "ribonanzanet-repo"
- Upload BASIC/models/*.py as "adv1-code-test"

**Step 1.3: Create Pre-Step 1 test notebook on Kaggle**
- Simple notebook: import torch, load weights, run inference on 1 target
- If it works → proceed to Phase 2
- If it fails → debug (path issues, import issues, memory)

### Phase 2: Build ADV1 (4-6 hours)

**Step 2.1: Write template_encoder.py (LOCAL)**
- Input: template coordinates (N,3) from MMseqs2 search
- Output: template features (N,N,16)
- ~100 lines of code

**Step 2.2: Write template_loader.py (LOCAL)**
- Reads MMseqs2 Result.txt + template coordinate files
- Produces template features for each target
- ~100 lines of code

**Step 2.3: Modify distance_head.py (LOCAL)**
- Change pair_dim default from 64 to 80
- Everything else stays the same

**Step 2.4: Write train_adv1.py (LOCAL)**
- Based on BASIC's train.py
- Adds template feature loading and concatenation in forward pass
- Adds warm-start weight copying from BASIC

**Step 2.5: Write predict_adv1.py (LOCAL)**
- Based on BASIC's predict.py
- Same template feature additions

**Step 2.6: Test locally (if RibonanzaNet repo + weights available)**
```bash
cd HY-BAS-ADV1
python predict_adv1.py --config config_adv1.yaml \
    --checkpoint ../APPROACH2-RIBBOZANET/BASIC/checkpoints/best_model.pt \
    --test_csv ../APPROACH1-TEMPLATE/test_sequences\ \(1\).csv
```
If this runs without errors (even with garbage output), the code is correct.

### Phase 3: Train ADV1 (2-4 hours)

**Step 3.1: Run training locally**
```bash
python train_adv1.py --config config_adv1.yaml \
    --resume ../APPROACH2-RIBBOZANET/BASIC/checkpoints/best_model.pt
```
- 30-50 epochs, warm-started from BASIC weights
- Needs: local GPU, RibonanzaNet repo, training pickle data
- Output: checkpoints/adv1_best_model.pt

**Step 3.2: Verify predictions locally**
```bash
python predict_adv1.py --config config_adv1.yaml \
    --checkpoint checkpoints/adv1_best_model.pt
```
- Compare output with BASIC's submission.csv
- Template-rich targets should have different coordinates

### Phase 4: Build Kaggle Notebook (3-4 hours)

**Step 4.1: Write kaggle_hybrid_notebook.py**
This single file contains ALL cells for the Kaggle notebook:

```
Cell 1: Symlink fix (same as Fork 2)
Cell 2: Install dependencies
Cell 3-8: MMseqs2 pipeline (adapted from Fork 1 / rhijudas)
Cell 9: Load ADV1 model from uploaded datasets
Cell 10: Run ADV1 inference on all targets
Cell 11: Post-processing (Option B pattern — read sample, map IDs)
Cell 12: Verification
```

**Step 4.2: Upload final datasets to Kaggle**
- Upload adv1_best_model.pt + RibonanzaNet.pt
- Upload all .py code files
- Upload RibonanzaNet repo

**Step 4.3: Test on Kaggle in draft mode**
- Run All with Internet ON first (for pip installs)
- Then Run All with Internet OFF
- Verify submission.csv is correct

### Phase 5: Submit (30 min)

**Step 5.1: Commit and submit**
- Internet OFF
- Save & Run All (Commit)
- Submit to Competition

---

## RUNTIME BUDGET ON KAGGLE (8 hours max)

| Phase | Estimated Time |
|-------|---------------|
| Dependencies install | 2-3 min |
| MMseqs2 database build | 5-10 min |
| MMseqs2 search | 5-15 min |
| Template coordinate extraction | 2-5 min |
| Load ADV1 model | 1-2 min |
| Run ADV1 inference (28+ targets) | 5-20 min |
| Post-processing | < 1 min |
| **TOTAL** | **~25-55 min** |

Well within 8-hour limit. Room for longer hidden test sets.

---

## WHAT HAPPENS FOR EACH TARGET TYPE

| Target Type | Phase 1 (MMseqs2) | Phase 2 (ADV1) | Expected TM |
|-------------|-------------------|----------------|-------------|
| Strong template (6 targets) | Finds close homologs | Template features (N,N,16) + pairwise (N,N,64) = rich signal | 0.40-0.80 |
| Weak template (10 targets) | Finds distant matches | Partial template + pairwise features | 0.05-0.20 |
| No template (12+ targets) | No hits | Template channels = zeros, relies on pairwise only (like BASIC) | 0.03-0.10 |
| Hidden targets | Depends on PDB | Same as above based on MMseqs2 results | Varies |

**Why this is better than Fork 1/2 alone:**
- No-template targets get REAL neural predictions instead of zeros
- Every hidden target gets something instead of nothing

**Why this is better than BASIC alone:**
- Template features dramatically boost 6 strong-template targets
- From ~0.05 (BASIC) to potentially 0.40-0.80 for those targets

---

## RISK REGISTER

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| RibonanzaNet won't load on Kaggle | BLOCKS everything | Medium | Pre-Step 1 test |
| ADV1 training doesn't converge | Score = BASIC level | Low | Warm-started from BASIC |
| Kaggle notebook times out | No submission | Low | Budget is 55 min / 8 hr |
| MMseqs2 install fails on Kaggle | No templates | Low | Fork 1 already proved it works |
| Template encoder bugs | Wrong features | Medium | Test locally first |
| Weight file too large for Kaggle | Can't upload | Low | 44MB is within 100MB limit |
| Kaggle re-run on hidden data fails | Scoring error | Medium | Option B post-processing handles unknown targets |

---

## DEPENDENCIES ON OTHER WORK STREAMS

| HY-BAS-ADV1 Needs | From | Status |
|-------------------|------|--------|
| best_model.pt (BASIC trained weights) | BASIC training | DONE ✅ |
| RibonanzaNet.pt (backbone weights) | Kaggle download | DONE ✅ |
| RibonanzaNet/ repo code | GitHub clone | DONE ✅ (need to verify path) |
| MMseqs2 pipeline that works on Kaggle | Fork 1 (rhijudas) | DONE ✅ (proven) |
| Template coordinate format understanding | Result.txt analysis | DONE ✅ |
| pdb_xyz_data.pkl (training data) | Earlier data prep | DONE ✅ |
| Option B post-processing pattern | BASIC + Fork 2 work | DONE ✅ |
| ADV1 training (new) | Phase 3 | TODO |
| Template encoder code (new) | Phase 2 | TODO |
| Kaggle dataset uploads (new) | Phase 4 | TODO |
