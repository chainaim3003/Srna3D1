# Kaggle Setup Guide: Running HY-BAS-ADV1 Notebooks

## Two Notebooks

| Notebook File | Purpose | Runtime |
|---|---|---|
| `hy_bas_adv1_run3_optb-fix1-3_commit_NB.py` | Fix validation (skips training, tests 3 inference fixes) | ~19 min |
| `hy_bas_adv1_run4_commit_NB.py` | Run 4 with MSA evolutionary features + all 3 fixes | ~80-90 min |

Both run on Kaggle with **Internet OFF** in **Save Version (Commit)** mode.
Both require the same 7 datasets attached.

---

## Step 1: Join the Competition

1. Go to: **https://www.kaggle.com/competitions/stanford-rna-3d-folding-2**
2. Click **Join Competition**
3. Accept the rules

This gives you Dataset 1 (competition data) automatically.

---

## Step 2: Create 6 Kaggle Datasets

You need to upload files into 6 separate Kaggle datasets.
Dataset names do NOT matter. The code searches by filename, not by dataset name.
You can name each dataset anything you want.

To create a dataset:
1. Go to **https://www.kaggle.com**
2. Click your profile icon (top right) → **Your Work** → **Datasets** → **+ New Dataset**
3. Give it any name
4. Upload the file(s) listed below
5. Click **Create**

---

### Dataset 1: Competition Data

**No action needed.** This is automatically available after joining the competition in Step 1.

Contains: `train_sequences.csv`, `test_sequences.csv`, `train_labels.csv`, `sample_submission.csv`

---

### Dataset 2: RibonanzaNet Repository

**What to upload:** The RibonanzaNet source code folder.

**Where to get it:**
1. Go to: **https://github.com/DasLab/RibonanzaNet**
2. Click the green **Code** button → **Download ZIP**
3. Extract the ZIP on your computer
4. Upload the extracted folder to a new Kaggle dataset

**The code needs these files from it:**
- `Network.py`
- `configs/pairwise.yaml`

---

### Dataset 3: ADV1 Weights

**What to upload:** Two weight files:

| File | Size | What it is |
|---|---|---|
| `RibonanzaNet.pt` | ~43 MB | Pretrained RibonanzaNet backbone weights |
| `best_model.pt` | ~312 KB | BASIC distance head weights |

**Where to get them:** These were generated in earlier runs of this project. If you are setting up from scratch, you need these from the project team.

---

### Dataset 4: ADV1 Training Data

**What to upload:** One file:

| File | Size | What it is |
|---|---|---|
| `pdb_xyz_data.pkl` | ~52 MB | 844 RNA structures with 3D coordinates for training |

**Where to get it:** Generated from PDB/CIF RNA structure files in earlier project steps.

---

### Dataset 5: Extended RNA Data

**What to upload:** Two CSV files:

| File | Size | What it is |
|---|---|---|
| `rna_sequences.csv` | ~100 MB | 24,000+ RNA sequences for template matching |
| `rna_coordinates.csv` | ~100 MB | Corresponding 3D coordinates |

**Where to get them:** Generated from PDB/CIF RNA structure files in earlier project steps.

---

### Dataset 6: Biopython Wheel

**What to upload:** One `.whl` file downloaded from PyPI.

**Step-by-step download instructions:**

1. Go to: **https://pypi.org/project/biopython/1.86/#files**

2. Scroll down to the file list. There are many files listed for different
   operating systems and Python versions. You need this EXACT one:

   ```
   biopython-1.86-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl
   ```

   Look for the row that says:
   - **CPython 3.12**
   - **manylinux: glibc 2.17+ x86-64**
   - **manylinux: glibc 2.28+ x86-64**
   - Size: **3.2 MB**

3. Click **Download** next to that file

4. The file saves to your computer (usually your Downloads folder)

5. Create a new Kaggle dataset, upload this single `.whl` file, click Create

**DO NOT download any of these (wrong platform):**
- ❌ `...aarch64...` — that is ARM architecture, Kaggle uses x86
- ❌ `...win_amd64...` or `...win32...` — that is Windows, Kaggle is Linux
- ❌ `...macosx...` — that is Mac, Kaggle is Linux
- ❌ `...cp310...` or `...cp311...` — those are older Python versions (they work but slower)

**Why cp312:** Kaggle runs Python 3.12.12. The cp312 wheel matches exactly,
so the compiled C extensions load natively and sequence alignments run at full speed.

**Why a wheel file at all:** Kaggle commit mode has Internet OFF. Biopython is not
pre-installed on Kaggle. The wheel file provides biopython offline. When you upload
a `.whl` file to Kaggle, Kaggle automatically extracts it. The code finds the
extracted `Bio/` package folder and imports it directly.

---

### Dataset 7: Run 3 OptB Checkpoint

**What to upload:** One model checkpoint file:

| File | Size | What it is |
|---|---|---|
| `adv1_best_run3optb_model.pt` | 8.89 MB | Trained neural network from Run 3 OptB |

**Where to get it:** This file is on your local machine at:
```
HY-BAS-ADV1-RUN3-REMOTE/HY-BAS-ADV1-RUN3-OPTB-REMOTE/adv1_best_run3optb_model .pt
```

Note: The filename on disk may have a space before `.pt` (a Kaggle artifact).
That is fine. The code handles it with `f.strip()`.

**What each notebook uses it for:**
- **Fix NB:** Loads this checkpoint directly, skips training entirely (saves ~19 min)
- **Run 4 NB:** Loads this as a warm-start, then trains 50 more epochs with MSA features

---

## Step 3: Create a New Kaggle Notebook

1. Go to **https://www.kaggle.com**
2. Click **+** (top left) → **New Notebook**
3. A new notebook opens with one empty cell

---

## Step 4: Configure Notebook Settings

Click the **Settings** sidebar (gear icon on the right side):

| Setting | Value |
|---|---|
| Accelerator | **GPU T4 x2** (or GPU P100) |
| Persistence | **Files only** |
| Internet | **OFF** ← CRITICAL |
| Language | **Python** |

---

## Step 5: Attach All 7 Datasets

In the Settings sidebar, scroll to **Data** and click **+ Add Input** for each:

**Under the Competitions tab:**
- [ ] Search for `stanford-rna-3d-folding-2` and add it

**Under the Datasets tab (search by whatever name you gave each):**
- [ ] RibonanzaNet repo dataset (contains `Network.py`)
- [ ] ADV1 weights dataset (contains `RibonanzaNet.pt` and `best_model.pt`)
- [ ] ADV1 training data dataset (contains `pdb_xyz_data.pkl`)
- [ ] Extended RNA data dataset (contains `rna_sequences.csv` and `rna_coordinates.csv`)
- [ ] Biopython wheel dataset (contains the `.whl` file you downloaded from PyPI)
- [ ] Run 3 OptB checkpoint dataset (contains `adv1_best_run3optb_model.pt`)

Verify all 7 appear in the Input section before proceeding.

---

## Step 6: Paste the Notebook Code

1. Delete the default empty cell in the notebook
2. Create **one** new cell
3. Open the `.py` file you want to run on your computer:
   - For fix validation: `hy_bas_adv1_run3_optb-fix1-3_commit_NB.py`
   - For Run 4 MSA: `hy_bas_adv1_run4_commit_NB.py`
4. Select All (Ctrl+A), Copy (Ctrl+C)
5. Click into the Kaggle notebook cell, Paste (Ctrl+V)

The entire `.py` file goes into **one single cell**. The `# CELL 1`, `# CELL 2` comments
inside are organizational markers, not separate Kaggle cells. Do not split them.

---

## Step 7: Save Version (Commit Mode)

1. Click **Save Version** (top right button)
2. Select **Save & Run All (Commit)** ← NOT Draft
3. Optionally set a version name (e.g. "Fix1-3 validation" or "Run 4 MSA")
4. Click **Save**

The notebook now runs in the background on Kaggle's servers.

**Do NOT use "Save & Run All (Draft)"** — Draft mode may not have GPU access and
times out after inactivity. Commit mode runs headless with full GPU for up to 9 hours.

---

## Step 8: Monitor Progress and Submit

1. Go to your notebook's page
2. Click the **Versions** tab
3. The latest version shows status: **Running** → **Complete** or **Error**
4. Click on the version to see log output

**Expected runtimes:**
- Fix NB: ~19 minutes
- Run 4: ~80-90 minutes

5. Once status is **Complete**, scroll to the **Output** section
6. You should see `submission.csv`
7. Click **Submit to Competition**
8. Wait for scoring (usually 10-60 minutes)
9. Check the leaderboard for your score

---

## How the Code Finds Datasets

The code does NOT care about dataset names. It scans every file under `/kaggle/input/`
by **filename**:

```python
# Cell 1 (biopython):
glob.glob('/kaggle/input/**/Bio/__init__.py', recursive=True)

# Cell 3 (everything else):
for root, dirs, files in os.walk('/kaggle/input'):
    for f in files:
        if f == 'Network.py': ...
        if f == 'RibonanzaNet.pt': ...
        if f == 'pdb_xyz_data.pkl': ...
        if f.strip() in ('adv1_best_run3optb_model.pt',): ...
```

This means:
- You can name your datasets anything
- The order you attach them does not matter
- Other attached datasets that don't contain these filenames are simply ignored

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| Cell 1 crashes with pip error | Biopython wheel dataset not attached | Attach the dataset containing the `.whl` file |
| "Could not find rna_sequences.csv" | Extended RNA data dataset not attached | Attach the dataset containing the CSV files |
| "No pre-trained checkpoint found" | Checkpoint dataset not attached | Attach the dataset containing `adv1_best_run3optb_model.pt` |
| Score lower than expected | Fixes may not be active | Check log for "Targets with 1 template slot" (should show 12) |

---

## Quick Reference Checklist

| # | What to Upload | Key File(s) | Size | Source |
|---|---|---|---|---|
| 1 | Competition data | (auto) | ~50 MB | Join competition |
| 2 | RibonanzaNet repo | `Network.py`, `configs/pairwise.yaml` | ~5 MB | github.com/DasLab/RibonanzaNet |
| 3 | ADV1 weights | `RibonanzaNet.pt`, `best_model.pt` | ~43 MB | Project files |
| 4 | ADV1 training data | `pdb_xyz_data.pkl` | ~52 MB | Project files |
| 5 | Extended RNA data | `rna_sequences.csv`, `rna_coordinates.csv` | ~200 MB | Project files |
| 6 | Biopython wheel | `biopython-1.86-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl` | 3.2 MB | pypi.org/project/biopython/1.86/#files |
| 7 | Run 3 OptB checkpoint | `adv1_best_run3optb_model.pt` | 8.89 MB | Project files |
