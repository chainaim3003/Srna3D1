# Fork 2: jaejohn 1st Place TBM — Run Steps

## Source
- **URL:** https://www.kaggle.com/code/jaejohn/rna-3d-folds-tbm-only-approach
- **Author:** jaejohn ("john") — 1st place winner of Stanford RNA 3D Folding Part 1
- **NOTE:** This is a KAGGLE NOTEBOOK, not a GitHub repo. You fork it directly on Kaggle.

## Why Run This
- Scored **0.593 TM-score** in Part 1 — strongest template-based result in the entire competition
- Uses custom template ranking, gap-filling, and diversity strategies (better than basic DasLab pipeline)
- Compare results against your run at: `mine/kalai/run1/` and `fork1-RJ/`
- **CAUTION:** Built for Part 1. The notebook reads `test_sequences.csv` from competition data, so if Part 2 data is attached, it should automatically process the 28 Part 2 targets. But verify the output targets match Part 2.

---

## Exact Steps

### Step 1: Open the Notebook on Kaggle

Open this URL in your browser:
```
https://www.kaggle.com/code/jaejohn/rna-3d-folds-tbm-only-approach
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

**IMPORTANT — Check for Part 1 data:** This notebook was originally built for Part 1. If you see `stanford-rna-3d-folding` (WITHOUT the `-2`) attached as input, **remove it** and make sure only `stanford-rna-3d-folding-2` (WITH the `-2`) is attached. Having both could cause the notebook to process the wrong test targets.

### Step 4: Check for Additional Dataset Dependencies

jaejohn's notebook may require extra datasets beyond just the competition data. After clicking "Copy & Edit", look in the right sidebar under **"Input"** for any additional datasets already attached (e.g., pre-built MMseqs2 databases, binary tools, or pre-computed alignments).

Common extras this notebook may need:
- `mmseqs2-binary` — pre-compiled MMseqs2 (for offline mode)
- Any dataset the original author attached

**If you see errors about missing files when running:** Check the original notebook page for what datasets are listed under its "Input" section, then add those same datasets to your fork.

### Step 5: Configure Notebook Settings

In the right sidebar → **"Settings"**:

| Setting | Value | Why |
|---------|-------|-----|
| **Accelerator** | GPU T4 x2 | MMseqs2 runs on CPU but GPU doesn't hurt |
| **Internet** | **ON** | Needed for `apt-get install mmseqs2` and any `git clone` or `pip install` commands |

### Step 6: Scan the Code Before Running (Recommended)

Since this was built for Part 1, quickly scan the notebook cells for:

1. **Hardcoded paths to Part 1 data** — look for `stanford-rna-3d-folding` without the `-2`. If found, change to `stanford-rna-3d-folding-2`.
2. **Hardcoded test target lists** — the notebook should read from `test_sequences.csv` (which will be Part 2's file). If target IDs are hardcoded, they'll be wrong.
3. **Temporal cutoff handling** — jaejohn's approach may or may not apply cutoffs. Note what it does for your comparison.

### Step 7: Run All Cells

Click **"Run All"** at the top of the notebook.

Wait 15-45 minutes (may be longer than rhijudas since jaejohn does more sophisticated processing). Watch the cell outputs for:
- MMseqs2 installation / database building
- Search progress (28 queries vs PDB)
- Template ranking and selection
- Coordinate transfer and gap-filling
- Diversity generation for 5 predictions
- Submission file creation

### Step 8: Verify Output Targets are Part 2

After cells finish, **before downloading**, check that the output contains the correct Part 2 targets. Look for these target IDs in the output or submission.csv:

```
Part 2 targets (should see these): 8ZNQ, 9IWF, 9JGM, 9MME, 9J09, 9E9Q, 9CFN, 9OBM, 
    9G4P, 9G4Q, 9G4R, 9RVP, 9JFS, 9LEC, 9LEL, 9I9W, 9HRO, 9QZJ, 9JFO, 9OD4, 
    9WHV, 9E74, 9E75, 9G4J, 9KGG, 9EBP, 9LJN, 9ZCC
```

If you see different target IDs (Part 1 targets), the notebook picked up the wrong test_sequences.csv. Go back and check the data attachment (Step 3).

### Step 9: Download ALL Output Files

Click the **"Output"** tab at the top of the notebook.

Download **EVERY file** listed there. At minimum:
- `submission.csv` — the template coordinates (this is the key file)

Also look for and download:
- Any `.txt` files (may contain MMseqs2 raw results)
- Any other `.csv` files (intermediate template data, rankings)
- Any log files or alignment files

### Step 10: Save Downloaded Files

Save everything to:
```
C:\sathya\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH1-TEMPLATE\fork2-JJ\
```

Rename files with a timestamp for clarity:
```
fork2-JJ/
├── submission_jaejohn.csv
├── Result_jaejohn.txt           (if available)
└── [any other output files]
```

### Step 11: Also Save the Notebook Itself

In Kaggle, go to **File → Download Notebook** to save the `.ipynb` file.
Save as:
```
fork2-JJ/jaejohn_tbm_forked.ipynb
```

This preserves exactly what code ran, in case you need to re-run or debug later.

---

## Comparing with Your Run and Fork 1

After downloading, compare all three:

| What to compare | Your run (`mine/kalai/run1/`) | rhijudas (`fork1-RJ/`) | jaejohn (`fork2-JJ/`) |
|----------------|------------------------------|------------------------|-----------------------|
| submission.csv | `submission_..._kalai.csv.csv` | `submission_rhijudas.csv` | `submission_jaejohn.csv` |
| Result.txt | `Result_..._kalai.txt.txt` | `Result_rhijudas.txt` | `Result_jaejohn.txt` |
| Temporal cutoff | Skipped | Properly handled | Unknown — check code |
| Pipeline | DasLab baseline | DasLab (organizer) | Custom (1st place) |
| Template ranking | Basic (by e-value) | Basic (by e-value) | Advanced (custom scoring) |
| Gap-filling | Basic interpolation | Basic interpolation | Sophisticated (1st place tricks) |
| Diversity strategy | Random perturbation | Random perturbation | Likely better (multiple templates) |

### Key Differences to Look For

1. **Better template selection?**
   jaejohn scored 0.593 in Part 1 vs the baseline ~0.4. The key difference
   is HOW templates are selected and ranked, not just WHICH ones are found
   (MMseqs2 finds the same hits for all three). Look at whether jaejohn's
   submission.csv has different coordinates for the same targets.

2. **Better gap-filling for partial matches?**
   When a template only covers part of the test sequence, gap-filling
   determines the coordinates for unmatched residues. jaejohn's approach
   likely produces better coordinates in gap regions.

3. **Better diversity across 5 predictions?**
   The competition scores best-of-5. jaejohn's notebook probably uses
   multiple different templates for predictions 2-5 rather than just
   adding noise to the best template.

4. **Coverage — does it handle long sequences better?**
   Check 9MME (4184 nt) and 9ZCC (1460 nt). Does jaejohn produce
   better coordinates for these very long targets?

### Quick Comparison Commands (run in VSCode terminal)

```bash
cd C:\sathya\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH1-TEMPLATE

# Count rows in each
python -c "
import pandas as pd
mine = pd.read_csv('mine/kalai/run1/submission_20260321_1100_UTC_kalai.csv.csv')
rj   = pd.read_csv('fork1-RJ/submission_rhijudas.csv')
jj   = pd.read_csv('fork2-JJ/submission_jaejohn.csv')
print(f'Mine: {mine.shape}')
print(f'RJ:   {rj.shape}')
print(f'JJ:   {jj.shape}')
"

# Count targets in each
python -c "
import pandas as pd
mine = pd.read_csv('mine/kalai/run1/submission_20260321_1100_UTC_kalai.csv.csv')
rj   = pd.read_csv('fork1-RJ/submission_rhijudas.csv')
jj   = pd.read_csv('fork2-JJ/submission_jaejohn.csv')
for name, df in [('Mine', mine), ('RJ', rj), ('JJ', jj)]:
    targets = df['ID'].str.rsplit('_', n=1).str[0].unique()
    print(f'{name}: {len(targets)} targets — {sorted(targets)[:5]}...')
"

# Check for zeros in prediction 1 (Kalai had zeros, Harishmitha didn't)
python -c "
import pandas as pd
jj = pd.read_csv('fork2-JJ/submission_jaejohn.csv')
zeros = jj[(jj['x_1']==0) & (jj['y_1']==0) & (jj['z_1']==0)]
print(f'JJ pred1 all-zero rows: {len(zeros)} / {len(jj)} ({len(zeros)/len(jj)*100:.1f}%)')
"
```

---

## Submit to Kaggle for Scoring

1. Go to: `https://www.kaggle.com/competitions/stanford-rna-3d-folding-2`
2. Click **"Submit Predictions"**
3. Upload `fork2-JJ/submission_jaejohn.csv`
4. Compare TM-score against:
   - Your notebook's submission (Kalai / Harishmitha)
   - rhijudas fork submission
   - BASIC neural network submission
5. **The best-scoring Approach 1 submission becomes the template source for ADV1**

---

## After All 3 Approach 1 Runs

Once you have all three submissions scored:

```
APPROACH1-TEMPLATE/
├── mine/
│   ├── kalai/run1/       → submission + Result.txt  (your DasLab run)
│   └── Harishmitha/run1/ → submission + Result.txt  (your 2nd DasLab run)
├── fork1-RJ/             → submission (rhijudas organizer baseline)
└── fork2-JJ/             → submission (jaejohn 1st place)
```

Pick the highest TM-score submission as:
- **Direct competition entry** (slots 1-2 in hybrid submission)
- **Template source for ADV1** (template_loader.py reads this CSV)
- **Confidence source** (Result.txt e-values for template_encoder.py)

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Copy & Edit" button not visible | Make sure you're logged in to Kaggle |
| Competition data not attached | Click "+ Add Input" → search "stanford-rna-3d-folding-2" |
| Notebook has Part 1 data attached | Remove `stanford-rna-3d-folding` (no `-2`), keep only `stanford-rna-3d-folding-2` |
| Cell fails to install mmseqs2 | Internet is OFF → turn ON in Settings |
| Output has wrong target IDs | Part 1 data was attached instead of Part 2. Fix data and re-run. |
| Missing dataset error | Check original notebook's Input section for extra datasets needed |
| Notebook times out (>8 hrs) | jaejohn's notebook may be slower. Check if it's stuck on a specific cell. |
| No submission.csv in Output | Check if the notebook errored midway. Look at cell outputs for errors. |
| "Failed to save draft" | Kaggle UI glitch. Refresh page or try incognito window. |
| Coordinates look like straight lines | Gap-filling may have failed for long targets. Check the actual template hits. |
