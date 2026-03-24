# HY-BAS-ADV1 Run 2: Kaggle Notebook Setup Guide

## Changes from Run 1

| # | Change | Cell | What |
|---|--------|------|------|
| 1 | More training | Cell 13 | `TRAIN_EPOCHS = 30` (was 15) |
| 3 | Biopython fix | Cell 1 | Removed pip install, uses pre-installed version |
| 4 | Pickle parsing fix | Cell 13 | Correct dict-of-parallel-lists extraction with `sugar_ring[0]` |

Full rationale: see `RUN2_DESIGN.md` in this folder.

---

## The Notebook Source File

```
APPROACH2-RIBBOZANET\HY-BAS-ADV1\kaggle\hy_bas_adv1_run2_notebook.py
```

This is a single Python file with 15 cells. Each cell is marked:
```
# ============================================================
# CELL N: Description
# ============================================================
```

---

## Step 1: Create a NEW Kaggle Notebook

1. Go to: https://www.kaggle.com/competitions/stanford-rna-3d-folding-2
2. Click **"Code"** tab -> **"New Notebook"**
3. Name it: **HY-BAS-ADV1-Run2**

Do NOT edit the Run 1 notebook. Keep it intact for reference and rollback.

---

## Step 2: Attach Inputs (same as Run 1)

In the right sidebar -> **"Input"** -> **"+ Add Input"**:

| # | Type | Name | How to find |
|---|------|------|-------------|
| 1 | Competition | Stanford RNA 3D Folding Part 2 | Already attached |
| 2 | Dataset | rna-cif-to-csv | Search: "rna_cif_to_csv" by jaejohn |
| 3 | Dataset | adv1-weights | Your private dataset (RibonanzaNet.pt + best_model.pt) |
| 4 | Dataset | adv1-training-data | Your private dataset (pdb_xyz_data.pkl) |
| 5 | Dataset | ribonanzanet-repo | Your private dataset (RibonanzaNet/ folder) |
| 6 | Dataset | extended-rna | Search: "extended_rna" by jaejohn (optional) |
| 7 | Dataset | biopython-wheel-v2 | Your dataset: cp310 + cp311 .whl files (see below) |

Datasets 1-6 are the SAME as Run 1. **Dataset 7 is new** (or update the
existing `biopyhon-wheel` dataset with correct wheels — see Step 2b).

---

## Step 2b: Biopython Wheel Dataset (NEW for Run 2)

Biopython is NOT pre-installed in the Kaggle GPU image. The Run 1 approach
(`!pip install biopython`) fails with Internet OFF. We need offline wheels.

**Download from PyPI** (https://pypi.org/project/biopython/1.86/#files):
```
biopython-1.86-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl
biopython-1.86-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl
```

Do NOT download the cp312 or win_amd64 versions. Kaggle runs Linux with
Python 3.10 or 3.11 — you need both cp310 and cp311 manylinux x86_64.

**Option A — Update existing dataset:**
1. Go to your existing `biopyhon-wheel` dataset on Kaggle
2. Click Edit → delete the old cp312 wheel
3. Upload both new wheels → Save

**Option B — Create new dataset:**
1. Go to https://www.kaggle.com/datasets/new
2. Upload both .whl files
3. Name it `biopython-wheel-v2` (private)
4. Attach to notebook in Step 2

Cell 1 in the notebook auto-detects the Python version and picks the
correct wheel. If biopython happens to be pre-installed, it skips the
install entirely.

---

## Step 3: Settings

| Setting | Value |
|---------|-------|
| Accelerator | **GPU T4 x2** |
| Internet | **ON** (for development/draft run) |

---

## Step 4: Paste Cells

Open `hy_bas_adv1_run2_notebook.py` in **Notepad** (not from chat — to avoid
smart quote corruption).

For each `CELL N` section in the file:
1. Click **"+ Code"** in Kaggle to add a new cell
2. Copy the code between one `# ====` header and the next
3. Paste into the Kaggle cell

There are 15 cells total (Cell 1 through Cell 15).

**Cell 1** tries `import Bio` first. If that fails, it finds the correct
`.whl` file from the attached biopython-wheel dataset and installs it
offline. No Internet needed.

---

## Step 5: Draft Run (Internet ON)

Click **"Run All"** with Internet ON.

Expected timeline:
```
Cell 1-6:   Setup + data loading              ~3 min
Cell 7:     Process labels (train_coords_dict) ~15 min
Cell 8:     Template functions defined         ~1 sec
Cell 9:     Fork 2 template search             ~5 min
Cell 10-12: ADV1 model setup + warm-start      ~1 min
Cell 13:    ADV1 training (30 epochs)           ~32 min
Cell 14:    ADV1 inference (28 targets)         ~10-20 min
Cell 15:    Post-processing                    ~10 sec
TOTAL:                                         ~65 min
```

### What to Watch For During Draft Run

**Cell 1:** Should print either `Biopython 1.86 already installed` or
`Biopython 1.86 installed successfully` (from wheel). If it prints
`Found wheels: []` and then errors, the biopython-wheel dataset is not
attached — go to Step 2, attach it, and re-run.

**Cell 13 — training data loading:** Should print:
```
Pickle type: <class 'dict'>
Keys: ['sequence', 'xyz', 'publication_date', ...]
Total structures: 661
Parsed: ~600 structures, Skipped: ~61
```
If it prints `Pickle is a list` or `Parsed: 0 structures`, something is
wrong with the pickle parsing — stop and investigate.

**Cell 13 — training loop:** Should print epoch lines like:
```
Epoch 1/30: train=XXX.XXXX, val=XXX.XXXX, lr=0.000050, time=XXs
```
Loss should decrease over epochs. If loss is NaN, reduce LEARNING_RATE
to 1e-5 in Cell 13.

**Cell 15 — final output:** Should print:
```
Matched from ADV1: 9762
Filled with zeros: 0
```
If "Filled with zeros" is large (>100), some targets had mismatched IDs.

---

## Step 6: Verify Output

After all cells complete, add a verification cell:
```python
import os
f = '/kaggle/working/submission.csv'
print(f"Exists: {os.path.exists(f)}")
print(f"Size: {os.path.getsize(f):,} bytes")
import pandas as pd
df = pd.read_csv(f)
print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"Any NaN: {df.isnull().any().any()}")
```

Expected: 9762 rows, 18 columns, no NaN.

---

## Step 7: Commit (Internet OFF)

1. **Turn Internet OFF** in Settings (right sidebar)
2. Click **"Save Version"** (top right)
3. Select **"Save & Run All (Commit)"**
4. Click **Save**
5. Wait ~65 minutes for the committed run to finish

---

## Step 8: Submit

1. Go to the committed notebook page (not the editor)
2. Click **"Output"** tab — verify `submission.csv` exists
3. Click **"Submit to Competition"**
4. Add description: `HY-BAS-ADV1-Run2: epochs 30, biopython fix, pickle fix`
5. Confirm submission
6. Check score at: https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/submissions

---

## Step 9: Save Artifacts

After the score appears, download and save to this folder
(`HY-BAS-ADV1-RUN2-REMOTE/`):

- [ ] The Kaggle notebook (.ipynb) — download from notebook page
- [ ] submission.csv — download from Output tab
- [ ] Screenshot of the score — save as run2-score.png or .docx
- [ ] Copy the training log output (epoch losses) for analysis

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'Bio'` | Wheel dataset not attached. Attach biopython-wheel-v2 dataset and re-run |
| `ModuleNotFoundError: No module named 'Network'` | ribonanzanet-repo dataset not attached |
| `FileNotFoundError: RibonanzaNet.pt` | adv1-weights dataset not attached |
| `FileNotFoundError: pdb_xyz_data.pkl` | adv1-training-data dataset not attached |
| `Parsed: 0 structures` | Pickle format different than expected — check Cell 13 output |
| `CUDA out of memory` | Change `BATCH_SIZE = 4` to `BATCH_SIZE = 2` in Cell 13 |
| `Training loss is NaN` | Change `LEARNING_RATE = 5e-5` to `LEARNING_RATE = 1e-5` in Cell 13 |
| `Notebook times out (>9hrs)` | Reduce `TRAIN_EPOCHS = 30` to `TRAIN_EPOCHS = 25` in Cell 13 |
| Commit fails with Internet OFF | Cell 1 should not need internet — verify no pip install remains |

---

## Score Expectations

| Outcome | Score | What It Means |
|---------|-------|---------------|
| > 0.35 | Excellent | NN clearly helping, proceed to Run 3 (unfreeze layers) |
| 0.29 – 0.35 | Good | Marginal gain, Run 3 unfreezing needed for bigger jump |
| = 0.287 | Neutral | NN matching template-only, bugs may still exist |
| < 0.287 | Problem | NN producing worse predictions, investigate training logs |

---

## Files in This Folder

| File | Purpose |
|------|---------|
| `RUN2_DESIGN.md` | Why these 3 changes, why we expect > 0.287 |
| `RUN2_KAGGLE_SETUP.md` | This file — step-by-step Kaggle instructions |
| *(after run)* `hyb-bas-adv1-run2.ipynb` | Downloaded Kaggle notebook |
| *(after run)* `submission.csv` | Downloaded submission output |
| *(after run)* `run2-score.*` | Score screenshot/documentation |
