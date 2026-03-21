# Approach 1: Run `approach1_kaggle_submission.ipynb` on Kaggle

## Estimated Runtime: ~15–30 minutes
## Cost: Free (Kaggle free tier)

---

## Step 1: Open the Competition on Kaggle

Go to: `https://www.kaggle.com/competitions/stanford-rna-3d-folding-2`

Make sure you are logged in and have already joined the competition.

---

## Step 2: Create a New Notebook

On the competition page → click **"Code"** tab → click **"New Notebook"**.

This creates a blank notebook with competition data auto-attached.

---

## Step 3: Upload Your Notebook

Click **"File" → "Import Notebook"** → upload `approach1_kaggle_submission.ipynb` from your computer at:

```
C:\SATHYA\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH1-TEMPLATE\approach1_kaggle_submission.ipynb
```

---

## Step 4: Verify Competition Data is Attached

In the **right sidebar** → look under **"Input"** section.

You should see: `stanford-rna-3d-folding-2`

This gives the notebook access to:
```
/kaggle/input/stanford-rna-3d-folding-2/
├── PDB_RNA/
│   ├── *.cif (thousands of CIF files)
│   ├── pdb_seqres_NA.fasta (13 MB, 26,255 sequences)
│   └── pdb_release_dates_NA.csv
├── test_sequences.csv (28 targets)
├── train_sequences.csv
├── train_labels.csv
├── sample_submission.csv
└── MSA/
```

**If it's NOT there**: Click **"+ Add Input"** → search `stanford-rna-3d-folding-2` → add it.

---

## Step 5: Set Accelerator

Right sidebar → **"Settings"** → **"Accelerator"** → select **"GPU T4 x2"**

---

## Step 6: Turn Internet ON

Right sidebar → **"Settings"** → **"Internet"** → **ON**

This is needed for:
- Cell 3: `apt-get install mmseqs2`
- Cell 4: `git clone https://github.com/DasLab/create_templates`

(You'll turn it OFF later for final scored submission.)

---

## Step 7: Run All Cells

Click **"Run All"** at the top, or run each cell with **Shift+Enter**.

### What each cell does:

| Cell | Time | What happens |
|------|------|-------------|
| 0 | — | Markdown header (nothing to run) |
| 1 | ~5 sec | Verifies competition data paths exist |
| 2 | ~5 sec | Loads 28 test sequences from test_sequences.csv |
| 3 | ~1 min | Installs MMseqs2 and BioPython |
| 4 | ~30 sec | Clones DasLab create_templates from GitHub |
| 5 | ~5 sec | Converts test sequences to FASTA format |
| 6 | ~5–10 min | **MMseqs2 search** (28 queries vs 26,255 PDB sequences) |
| 7 | ~10–20 min | **DasLab script** reads matched CIFs, transfers coordinates |
| 8 | ~5 sec | Converts templates.csv → submission.csv |
| 9 | ~5 sec | Validates submission (coordinate ranges, coverage) |

---

## Step 8: Check the Output

After Cell 9, look for these signs of success:

**Good signs:**
- "Test targets with hits: X / 28" — X should be > 0 (ideally 20+)
- Coordinate range is wide (e.g., -50 to +50), not just 0 to 170
- "Non-zero coordinates" percentage is high (> 50%)

**Bad signs:**
- "Total MMseqs2 hits: 0" — something went wrong with the search
- All coordinates along a straight line — means fallback dummy coords

---

## Step 9: Download submission.csv

Click the **"Output"** tab at the top of the notebook.

You should see `submission.csv` → click to download.

---

## Step 10: Submit (Development — Internet ON)

For a quick test score:
1. Go to competition page → **"Submit Predictions"**
2. Select your notebook or upload submission.csv
3. Wait for scoring (~30 min to a few hours)

---

## Step 11: Final Submission (Internet OFF)

For the actual scored submission, internet must be OFF.

### Pre-requisites:
1. Upload MMseqs2 binary as a Kaggle Dataset
   (already exists at: `https://www.kaggle.com/datasets/rhijudas/mmseqs2-binary`)
2. Clone DasLab `create_templates` repo locally, upload as Kaggle Dataset
3. Add both as **Input** datasets to your notebook

### Modify the notebook:
- Cell 3: Replace `apt-get install mmseqs2` with path to uploaded binary
- Cell 4: Replace `git clone` with path to uploaded dataset

### Then:
1. Turn **Internet → OFF**
2. Click **"Save Version"** → **"Save & Run All (Commit)"**
3. Wait for it to complete
4. Go to **Output** tab → click **"Submit"** on submission.csv

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| Cell 1 says "MISSING" | Competition data not attached | Click "+ Add Input" in sidebar |
| Cell 3 fails to install mmseqs2 | Internet is OFF | Turn Internet ON in Settings |
| Cell 4 fails to clone repo | Internet is OFF | Turn Internet ON in Settings |
| Cell 6 returns 0 hits | Shouldn't happen with full DB | Check Cell 6 output for errors |
| Cell 7 takes > 1 hour | Many large CIF files matched | Normal for ribosomal targets — wait |
| Notebook times out at 8 hrs | Pipeline too slow | Consider forking jaejohn's optimized notebook |
| "Disk space exceeded" | Too much temp data | Delete mmseqs_tmp/ between steps |

---

## Alternative: Fork an Official Notebook Instead

If any issues arise, fork a battle-tested notebook:

**1st place TBM (strongest):**
```
https://www.kaggle.com/code/jaejohn/rna-3d-folds-tbm-only-approach
```

**DasLab baseline (simplest):**
```
https://www.kaggle.com/code/rhijudas/mmseqs2-3d-rna-template-identification-part-2
```

Steps: Open URL → **"Copy & Edit"** → verify data attached → **Run All**
