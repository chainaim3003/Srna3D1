# Fork 1: rhijudas Part 2 Baseline — Run Steps

## Source
- **URL:** https://www.kaggle.com/code/rhijudas/mmseqs2-3d-rna-template-identification-part-2
- **Author:** rhijudas (DasLab / Competition Organizer)
- **NOTE:** This is a KAGGLE NOTEBOOK, not a GitHub repo. You fork it directly on Kaggle.

## Why Run This
- Built specifically for Part 2 by the competition organizers
- Handles temporal cutoffs correctly (your notebook skips them)
- Compare results against your run at: `mine/kalai/run1/`

---

## Exact Steps

### Step 1: Open the Notebook on Kaggle

Open this URL in your browser:
```
https://www.kaggle.com/code/rhijudas/mmseqs2-3d-rna-template-identification-part-2
```

Make sure you are logged in to Kaggle.

### Step 2: Fork the Notebook

Click the **"Copy & Edit"** button (top right of the notebook page).

This creates YOUR OWN copy of the notebook under your Kaggle account.
You can now edit and run it freely. The original is untouched.

### Step 3: Verify Competition Data is Attached

In the **right sidebar** → look under **"Input"** section.

You should see: `stanford-rna-3d-folding-2`

This gives the notebook access to:
```
/kaggle/input/stanford-rna-3d-folding-2/
├── PDB_RNA/           (thousands of CIF files + pdb_seqres_NA.fasta)
├── test_sequences.csv (28 Part 2 targets)
├── train_sequences.csv
├── sample_submission.csv
└── MSA/
```

**If it's NOT there:** Click **"+ Add Input"** → search `stanford-rna-3d-folding-2` → add it.

### Step 4: Configure Notebook Settings

In the right sidebar → **"Settings"**:

| Setting | Value | Why |
|---------|-------|-----|
| **Accelerator** | GPU T4 x2 | MMseqs2 runs on CPU but GPU doesn't hurt |
| **Internet** | **ON** | Needed for `apt-get install mmseqs2` and `git clone` |

### Step 5: Run All Cells

Click **"Run All"** at the top of the notebook.

Wait 15-30 minutes. Watch the cell outputs for:
- MMseqs2 installation
- Database building
- Search progress
- Template coordinate transfer
- Submission generation

### Step 6: Check Output

After all cells finish, look for:
- "Test targets with hits: X / 28" — X should be > 0 (ideally 20+)
- A `submission.csv` in the Output tab
- Possibly a `Result.txt` or equivalent intermediate file

### Step 7: Download ALL Output Files

Click the **"Output"** tab at the top of the notebook.

Download **EVERY file** listed there. At minimum:
- `submission.csv` — the template coordinates

Also look for and download:
- Any `.txt` files (may contain MMseqs2 raw results like Result.txt)
- Any other `.csv` files (intermediate template data)
- Any log files

### Step 8: Save Downloaded Files

Save everything to:
```
C:\sathya\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH1-TEMPLATE\fork1-RJ\
```

Rename files with a timestamp for clarity:
```
fork1-RJ/
├── submission_rhijudas.csv
├── Result_rhijudas.txt          (if available)
└── [any other output files]
```

### Step 9: Also Save the Notebook Itself

In Kaggle, go to **File → Download Notebook** to save the `.ipynb` file.
Save as:
```
fork1-RJ/rhijudas_part2_forked.ipynb
```

This preserves exactly what code ran, in case you need to re-run or debug later.

---

## Comparing with Your Run

After downloading, compare:

| What to compare | Your run (`mine/kalai/run1/`) | rhijudas fork (`fork1-RJ/`) |
|----------------|------------------------------|---------------------------|
| submission.csv | `submission_20260321_1100_UTC_kalai.csv.csv` | `submission_rhijudas.csv` |
| Result.txt | `Result_20260321_1100_UTC_kalai.txt.txt` | `Result_rhijudas.txt` (if saved) |
| Temporal cutoff | Skipped (`--skip_temporal_cutoff`) | Properly handled |
| Self-matches | Included (8ZNQ matches 8ZNQ.cif) | May be excluded by cutoff |

### Key Differences to Look For

1. **Does rhijudas exclude self-matches?**
   Your Result.txt shows 8ZNQ matching itself (8ZNQ_A). If rhijudas handles
   temporal cutoffs, it may exclude structures released after the cutoff date.
   This means different (possibly worse-scoring but more honest) templates.

2. **Different template rankings?**
   Even with the same hits, template selection and coordinate transfer
   may differ. Compare the actual x,y,z values in submission.csv.

3. **Coverage — do both cover all 28 targets?**
   Check if both submissions have rows for all 28 test targets.

### Quick Comparison Commands (run in VSCode terminal)

```bash
cd C:\sathya\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH1-TEMPLATE

# Count rows in each
python -c "import pandas as pd; mine=pd.read_csv('mine/kalai/run1/submission_20260321_1100_UTC_kalai.csv.csv'); rj=pd.read_csv('fork1-RJ/submission_rhijudas.csv'); print(f'Mine: {mine.shape}, RJ: {rj.shape}')"

# Count targets in each
python -c "import pandas as pd; mine=pd.read_csv('mine/kalai/run1/submission_20260321_1100_UTC_kalai.csv.csv'); rj=pd.read_csv('fork1-RJ/submission_rhijudas.csv'); print(f'Mine targets: {mine.ID.str.rsplit(\"_\",n=1).str[0].nunique()}'); print(f'RJ targets: {rj.ID.str.rsplit(\"_\",n=1).str[0].nunique()}')"
```

---

## Submit to Kaggle for Scoring

1. Go to: `https://www.kaggle.com/competitions/stanford-rna-3d-folding-2`
2. Click **"Submit Predictions"**
3. Upload `fork1-RJ/submission_rhijudas.csv`
4. Compare TM-score against your notebook's submission
5. The better score indicates which template source to use for ADV1

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Copy & Edit" button not visible | Make sure you're logged in to Kaggle |
| Competition data not attached | Click "+ Add Input" → search "stanford-rna-3d-folding-2" |
| Cell fails to install mmseqs2 | Internet is OFF → turn ON in Settings |
| Notebook times out (>8 hrs) | Should finish in 15-30 min. If stuck, check for errors in cell outputs. |
| No submission.csv in Output | Check if the notebook errored midway. Look at cell outputs for errors. |
| "Failed to save draft" | Kaggle UI glitch. Refresh page or try incognito window. |
