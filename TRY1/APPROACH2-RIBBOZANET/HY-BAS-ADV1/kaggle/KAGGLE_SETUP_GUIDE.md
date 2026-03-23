# HY-BAS-ADV1: Kaggle Notebook Setup Guide

## The Notebook File
```
kaggle/hy_bas_adv1_run1_notebook.py
```

This is a single Python file containing ALL cells for the Kaggle notebook.
Copy each CELL section into a separate Kaggle notebook cell.

---

## Step 1: Prepare 3 Kaggle Datasets (Upload from Your Machine)

### Dataset 1: "adv1-weights"
Create a folder and copy these files into it:
```
From: C:\sathya\...\Srna3D1\ribonanza-weights\RibonanzaNet.pt    (43 MB)
From: C:\sathya\...\Srna3D1\TRY1\APPROACH2-RIBBOZANET\BASIC\checkpoints\best_model.pt   (312 KB)
```
Upload to Kaggle as private dataset named "adv1-weights"

### Dataset 2: "adv1-training-data"
```
From: C:\sathya\...\Srna3D1\stanford3d-pickle\pdb_xyz_data.pkl   (52 MB)
```
Upload to Kaggle as private dataset named "adv1-training-data"

### Dataset 3: "ribonanzanet-repo"
Copy the ENTIRE folder:
```
From: C:\sathya\...\Srna3D1\RibonanzaNet\   (the whole folder with Network.py)
```
Upload to Kaggle as private dataset named "ribonanzanet-repo"

---

## Step 2: Create Kaggle Notebook

1. Go to: https://www.kaggle.com/competitions/stanford-rna-3d-folding-2
2. Click "Code" tab -> "New Notebook"
3. Name it: "HY-BAS-ADV1-Hybrid"

---

## Step 3: Attach Inputs (6 total)

In right sidebar -> "Input" -> "+ Add Input":

| # | Type | Name | Source |
|---|------|------|--------|
| 1 | Competition | Stanford RNA 3D Folding Part 2 | Already attached |
| 2 | Dataset | rna-cif-to-csv | Search: "rna_cif_to_csv" by jaejohn |
| 3 | Dataset | adv1-weights | Your upload |
| 4 | Dataset | adv1-training-data | Your upload |
| 5 | Dataset | ribonanzanet-repo | Your upload |
| 6 | Dataset | extended-rna | Optional: search "extended_rna" by jaejohn |

---

## Step 4: Settings

| Setting | Value |
|---------|-------|
| Accelerator | GPU T4 x2 |
| Internet | ON (for development) |

---

## Step 5: Paste Cells

Open `kaggle/hy_bas_adv1_run1_notebook.py` in Notepad.

The file has 15 cells, each marked with:
```
# ============================================================
# CELL N: Description
# ============================================================
```

Create a new code cell for each CELL section and paste the code.

CELL 1 is special -- it's a pip install command. Create it as:
```
!pip install biopython -q
```

---

## Step 6: Run

Click "Run All" with Internet ON for the first test.

Expected timeline:
```
Cell 1-6:   Setup + data loading              ~3 min
Cell 7:     Process labels (train_coords_dict) ~15 min
Cell 8:     Template functions defined         ~1 sec
Cell 9:     Fork 2 template search             ~5 min
Cell 10-12: ADV1 model setup + warm-start      ~1 min
Cell 13:    ADV1 training (15 epochs)           ~1-2 hrs
Cell 14:    ADV1 inference (28 targets)         ~10-20 min
Cell 15:    Post-processing                    ~10 sec
TOTAL:                                         ~2-3 hrs
```

---

## Step 7: Verify Output

After all cells complete:
```python
import os
print(os.path.exists("/kaggle/working/submission.csv"))
print(os.path.getsize("/kaggle/working/submission.csv"))
```

---

## Step 8: Submit

1. Turn Internet OFF
2. Save Version -> Save & Run All (Commit)
3. Wait ~2-3 hours for the committed run
4. Go to Output tab -> Submit to Competition

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| Network.py not found | ribonanzanet-repo dataset not attached or has wrong structure |
| RibonanzaNet.pt not found | adv1-weights dataset not attached |
| pdb_xyz_data.pkl not found | adv1-training-data dataset not attached |
| CUDA out of memory | Change BATCH_SIZE from 4 to 2 in Cell 13 |
| biopython not found | Cell 1 pip install didn't run |
| Training loss is NaN | Reduce LEARNING_RATE to 1e-5 in Cell 13 |
| Notebook times out (>8hrs) | Reduce TRAIN_EPOCHS from 15 to 10 in Cell 13 |
