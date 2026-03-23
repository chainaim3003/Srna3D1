# Fork 2: jaejohn — CORRECTED Step-by-Step Run Guide

## IMPORTANT CONTEXT
- This notebook was built for **Part 1** of the competition
- It depends on 3 specific inputs that must ALL be attached
- It runs entirely on Kaggle — do NOT try to run it locally
- Runtime: ~20 minutes

---

## The 3 Required Inputs (from the original notebook)

The original jaejohn notebook uses these 3 inputs:

| # | Type | Name | What it is |
|---|------|------|-----------|
| 1 | Competition | Stanford RNA 3D Folding | Competition data (CIF files, test_sequences.csv, etc.) |
| 2 | Dataset | extended_rna | Pre-processed RNA data by jaejohn |
| 3 | Dataset | rna_cif_to_csv | Pre-processed CIF-to-CSV conversions by jaejohn |
| 4 | Notebook | Dependency Installation Script | Installs required Python packages (mmseqs2, biopython, etc.) |

If ANY of these are missing, you will get module not found errors or file not found errors.

---

## Step-by-Step Instructions

### Step 1: Open the Original Notebook

Go to: https://www.kaggle.com/code/jaejohn/rna-3d-folds-tbm-only-approach

Make sure you are logged in to Kaggle.

### Step 2: Click "Copy & Edit"

This creates your own fork. When you fork, Kaggle SHOULD automatically
carry over all the input datasets and notebook dependencies from the
original. However, sometimes they don't — that's why you need to verify.

### Step 3: Verify ALL Inputs Are Attached

In the right sidebar of your forked notebook, check the "Input" section.

You MUST see ALL of these:

```
COMPETITIONS
  Stanford RNA 3D Folding          (or stanford-rna-3d-folding-2)

DATASETS
  extended_rna
  rna_cif_to_csv

NOTEBOOKS
  Dependency Installation Script
```

If ANY are missing, you need to add them:

**To add a missing dataset:**
1. Click "+ Add Input"
2. Click "Datasets" tab
3. Search for "extended_rna" or "rna_cif_to_csv"
4. Look for datasets by user "jaejohn" or "jensen01"
5. Click "Add"

**To add the missing notebook dependency:**
1. Click "+ Add Input"
2. Click "Notebooks" tab (or "Code" tab)
3. Search for "Dependency Installation Script"
4. Look for the one by "jensen01"
5. Click "Add"

**To switch competition from Part 1 to Part 2:**
1. If you see "Stanford RNA 3D Folding" (Part 1), you need to
   also add "Stanford RNA 3D Folding Part 2"
2. Click "+ Add Input" → Competitions → search "stanford-rna-3d-folding-2"
3. The notebook code may need path adjustments (see Step 5)

### Step 4: Configure Settings

In the right sidebar → "Settings":

| Setting | Value | Why |
|---------|-------|-----|
| Accelerator | GPU T4 x2 | Some processing benefits from GPU |
| Internet | ON | Needed if notebook installs packages via pip/apt |

### Step 5: Check the First Few Cells Before Running

Before clicking "Run All", read the first few cells. Look for:

**Cell 1 or 2: Dependency installation**
This cell likely installs packages. It may look like:
```python
!pip install mmseqs2
```
or it may reference the Dependency Installation Script notebook output:
```python
import sys
sys.path.append('/kaggle/input/dependency-installation-script/')
```

**Path references:**
Look for paths like:
```python
/kaggle/input/stanford-rna-3d-folding/
```
If you're using Part 2, you may need to change these to:
```python
/kaggle/input/stanford-rna-3d-folding-2/
```

### Step 6: Run All Cells

Click "Run All" at the top.

Watch for errors in each cell output. Common issues:

| Error | Cause | Fix |
|-------|-------|-----|
| ModuleNotFoundError: No module named 'XXX' | Dependency Installation Script not attached | Add it as input (Step 3) |
| FileNotFoundError: /kaggle/input/extended_rna/... | extended_rna dataset not attached | Add it as input (Step 3) |
| FileNotFoundError: /kaggle/input/rna_cif_to_csv/... | rna_cif_to_csv dataset not attached | Add it as input (Step 3) |
| FileNotFoundError: /kaggle/input/stanford-rna-3d-folding/... | Competition data not attached or wrong competition version | Check competition input |
| No module named 'Bio' | biopython not installed | Add pip install biopython to first cell |
| No module named 'mmseqs' | mmseqs2 not installed | Dependency script should handle this |

### Step 7: If "Run All" Fails — Debug Cell by Cell

Instead of "Run All", run cells one at a time with Shift+Enter.
Stop at the first cell that errors. The error message will tell you
exactly what's missing.

### Step 8: Check Output

After all cells finish:
1. Click "Output" tab
2. You should see submission.csv
3. Verify it has the correct Part 2 target IDs

### Step 9: For Actual Competition Submission

To submit to the Part 2 competition:
1. Make sure "stanford-rna-3d-folding-2" is attached as input
2. The notebook must read test_sequences.csv from Part 2 data
3. Turn Internet OFF
4. Click "Save Version" → "Save & Run All (Commit)"
5. After completion → Output tab → "Submit to Competition"

---

## Why This Notebook Is Different from the DasLab Baseline

jaejohn's notebook uses:
- Pre-processed data (extended_rna, rna_cif_to_csv) — skips slow CIF parsing
- A dependency installation script as a separate notebook output
- Custom template ranking and gap-filling logic
- Multiple template diversity for the 5 predictions

This is why it scored 0.593 in Part 1 vs ~0.4 for baseline approaches.

---

## If You Cannot Get It Working

The rhijudas notebook (fork1-RJ) is simpler and self-contained.
If jaejohn's dependencies are too complex to resolve:

1. Go to: https://www.kaggle.com/code/rhijudas/mmseqs2-3d-rna-template-identification-part-2
2. Click "Copy & Edit"
3. This notebook was built for Part 2 and has fewer dependencies
4. Run All → Submit

The rhijudas notebook will score lower than jaejohn but is much easier to get running.
