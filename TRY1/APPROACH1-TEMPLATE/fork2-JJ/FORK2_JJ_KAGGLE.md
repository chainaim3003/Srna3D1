# Fork 2: jaejohn 1st Place TBM — Run on Kaggle

## Source
- **URL:** https://www.kaggle.com/code/jaejohn/rna-3d-folds-tbm-only-approach
- **Author:** jaejohn ("john") — 1st place Part 1 (0.593 TM-score)
- **Built for:** Part 1 (needs adaptation for Part 2)
- **Runtime:** ~20 minutes

## CRITICAL CONTEXT
- This is a **Code Competition** — Kaggle re-runs your notebook on HIDDEN test sequences
- This notebook was built for Part 1 — it needs Part 2 competition data attached
- It has 3 specific input dependencies that MUST all be present
- The notebook must run end-to-end with Internet OFF for final submission

---

## The 3 Required Inputs

The original jaejohn notebook needs ALL of these:

| # | Type | Name | What it contains |
|---|------|------|-----------------|
| 1 | Competition | Stanford RNA 3D Folding Part 2 | test_sequences.csv, PDB_RNA/, sample_submission.csv |
| 2 | Dataset | extended_rna | train_labels_v2.csv, train_sequences_v2.csv |
| 3 | Dataset | rna_cif_to_csv | rna_coordinates.csv, rna_sequences.csv |
| 4 | Notebook Output | Dependency Installation Script | Pre-installed packages (by jensen01) |

If ANY are missing → ModuleNotFoundError or FileNotFoundError.

---

## Step-by-Step

### Step 1: Open and Fork

1. Go to: https://www.kaggle.com/code/jaejohn/rna-3d-folds-tbm-only-approach
2. Log in to Kaggle
3. Click **"Copy & Edit"** (top right)
4. Kaggle SHOULD carry over all inputs automatically — but verify in Step 2

### Step 2: Verify ALL Inputs Are Attached

In the right sidebar, you MUST see:

```
COMPETITIONS
  Stanford RNA 3D Folding Part 2

DATASETS
  extended_rna          (contains train_labels_v2.csv, train_sequences_v2.csv)
  rna_cif_to_csv        (contains rna_coordinates.csv, rna_sequences.csv)

DEPENDENCY INSTALLATION CODE
  (shows as notebook output, ~132KB)
```

**If competition is wrong (Part 1 instead of Part 2):**
1. Remove the Part 1 competition data
2. Click "+ Add Input" → search `stanford-rna-3d-folding-2` → add it

**If a dataset is missing:**
1. Click "+ Add Input" → "Datasets" tab
2. Search for `extended_rna` or `rna_cif_to_csv`
3. Look for datasets by user `jaejohn`
4. Add them

**If Dependency Installation Script is missing:**
1. Click "+ Add Input" → "Notebooks" or "Code" tab
2. Search for `Dependency Installation Script`
3. Look for the one by `jensen01`
4. Add it

### Step 3: Fix the Competition Data Path

The notebook code references `/kaggle/input/stanford-rna-3d-folding-2/` but Kaggle may mount it at `/kaggle/input/competitions/stanford-rna-3d-folding-2/`.

Add a **NEW CELL at the very top** of the notebook:

```python
import os
src = "/kaggle/input/competitions/stanford-rna-3d-folding-2"
dst = "/kaggle/input/stanford-rna-3d-folding-2"
if os.path.exists(src) and not os.path.exists(dst):
    os.symlink(src, dst)
    print("Symlink created: " + dst)
else:
    print("Path already correct or symlink exists")
```

Run this cell FIRST with Shift+Enter.

### Step 4: Install Missing Python Packages

Add another cell right after the symlink cell:

```python
!pip install biopython -q
```

Run with Shift+Enter. This fixes `ModuleNotFoundError: No module named 'Bio'`.

### Step 5: Configure Settings

In the right sidebar → **"Settings"**:

| Setting | Value | Why |
|---------|-------|-----|
| **Accelerator** | GPU T4 x2 | Some processing benefits from GPU |
| **Internet** | **ON** | Needed for pip install during development |

### Step 6: Run Cell by Cell (First Time)

Run cells one at a time with **Shift+Enter**. Stop at any error.

Common errors we encountered and their fixes:

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'Bio'` | Step 4 above |
| `FileNotFoundError: /kaggle/input/stanford-rna-3d-folding-2/train_sequences.csv` | Step 3 above (symlink) |
| `FileNotFoundError: /kaggle/input/extended_rna/...` | Dataset not attached — Step 2 |
| `FileNotFoundError: /kaggle/input/rna_cif_to_csv/...` | Dataset not attached — Step 2 |
| `ModuleNotFoundError: No module named 'XXX'` | Add `!pip install XXX -q` cell at top |
| Path references to `stanford-rna-3d-folding` (Part 1) | Change to `stanford-rna-3d-folding-2` in the cell |

### Step 7: Handle Part 1 vs Part 2 Path Issues

The notebook was built for Part 1. Some cells may reference Part 1 paths:
```python
/kaggle/input/stanford-rna-3d-folding/    # WRONG — Part 1
```

Change these to:
```python
/kaggle/input/stanford-rna-3d-folding-2/  # CORRECT — Part 2
```

Use Find & Replace (Ctrl+H in the Kaggle editor) to fix all occurrences at once.

### Step 8: Verify Output

After all cells complete:
- Check that `submission.csv` exists in `/kaggle/working/`
- Verify it contains Part 2 target IDs (8ZNQ, 9IWF, 9JGM, etc.)
- NOT Part 1 target IDs

### Step 9: Submit to Competition

1. Turn **Internet → OFF** in Settings
2. Click **"Save Version"** → **"Save & Run All (Commit)"**
3. Wait for completion (~20 min)
4. Go to committed notebook → **Output** tab → **"Submit to Competition"**

### Step 10: Save Results Locally

Download from the Output tab:
- `submission.csv`
- Any intermediate files

Save to:
```
C:\sathya\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH1-TEMPLATE\fork2-JJ\
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Submission Scoring Error" | Notebook must work on hidden data. Ensure it reads test_sequences.csv dynamically. |
| Wrong target IDs in output | Part 1 data attached instead of Part 2. Fix inputs. |
| "Submit" button not visible | Must commit first (Save Version → Save & Run All). Not available in draft mode. |
| Notebook runs >8 hours | Time limit for Code Competition. Check for stuck cells or infinite loops. |
| Multiple `ModuleNotFoundError` | The Dependency Installation Script may not have loaded. Re-add it as input. |

---

## Why This Notebook Scores Higher Than Baseline

jaejohn's notebook uses:
- Pre-processed data (extended_rna, rna_cif_to_csv) to skip slow CIF parsing
- Custom template ranking beyond simple e-value sorting
- Sophisticated gap-filling for partial template matches
- Multiple template diversity for the 5 predictions
- Scored 0.593 in Part 1 vs ~0.4 for DasLab baseline
