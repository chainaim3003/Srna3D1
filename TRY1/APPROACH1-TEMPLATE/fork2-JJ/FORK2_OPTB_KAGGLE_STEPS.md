# Fork 2 + Option B: Exact Steps to Submit on Kaggle

---

## BACKGROUND: The Problem With Fork 2's Raw Output

jaejohn's notebook (1st place Part 1, 0.593 TM-score) was built for Part 1.
When adapted for Part 2, it produces predictions with **wrong row counts**
for multi-chain targets. For example:

| Target | Kaggle Expects | jaejohn Produces | Diff |
|--------|---------------|-----------------|------|
| 9MME   | 4184 rows     | 4640 rows       | +456 |
| 9LEL   | 449 rows      | 476 rows        | +27  |
| 9G4J   | 341 rows      | 334 rows        | -7   |

This happens because jaejohn's code computes residue counts from
stoichiometry/all_sequences (per-chain) instead of the `sequence` column
(concatenated) that Kaggle uses. The row IDs don't match
sample_submission.csv, so Kaggle rejects it with "Submission Scoring Error".

---

## TWO OPTIONS TO FIX THIS

### Option A: Surgical Fix (Fix the Core Code) — NOT CHOSEN

**What it does:** Download jaejohn's notebook, find the exact place in the
code where it determines how many residues each target has, and change it
to use `len(row['sequence'])` from test_sequences.csv instead of the
per-chain calculation.

**Pros:**
- Clean, correct fix at the source
- No wasted computation on extra residues
- No missing residues (e.g., the 7 missing rows for 9G4J would be predicted)

**Cons:**
- Requires deep understanding of jaejohn's codebase
- Risk of breaking his template coordinate transfer logic (which may depend on per-chain structure)
- Time-consuming — could take hours to find, fix, and verify
- NOT chosen due to time pressure (4 days to competition deadline)

### Option B: Post-Processing Fix (Add a Cell at the End) — CHOSEN

**What it does:** Leave jaejohn's code completely untouched. Let it run and
produce its "wrong" submission.csv. Then add ONE new cell at the very end
of the notebook that:

1. Reads `sample_submission.csv` from the competition data
   (this file defines the EXACT IDs and order Kaggle expects)
2. Reads jaejohn's raw `submission.csv`
3. For each expected ID in the sample:
   - If jaejohn produced that ID → use his predicted coordinates
   - If jaejohn did NOT produce that ID → use zeros from the sample
4. Writes a corrected `submission.csv` that exactly matches the sample

**Why this works for hidden test data too:**
Kaggle swaps `sample_submission.csv` with a hidden version during scoring.
Our post-processing cell reads it dynamically, so it automatically adapts
to whatever IDs the hidden test expects.

**Pros:**
- Zero risk of breaking jaejohn's core logic — we don't touch his code
- Fast to implement — one cell, ~60 lines of Python
- Works for hidden test data (reads sample_submission.csv dynamically)
- Easy to understand and debug

**Cons:**
- Wastes computation on extra rows that get discarded (e.g., 456 extra for 9MME)
- For targets where jaejohn produces FEWER rows than expected (e.g., 9G4J: 334 vs 341), the 7 missing residues get filled with zeros — small scoring loss
- Feels hacky — papers over the mismatch rather than fixing it

**Why Option B is good enough:**
- The vast majority of predictions (9200+ out of 9762 rows) map correctly
- Only ~500 rows are affected (filled with zeros or dropped)
- The time saved is better spent on ADV1 (which will replace this approach)

---

## What This Submission Achieves

This is a **fallback score** while ADV1 is being built. Expected outcome:
- Per-target scores: 0.56-0.92 for 6 strong-template targets, 0.02-0.10 for the rest
- Overall score: estimated 0.10-0.25 (much better than BASIC's 0.092)
- Provides immediate leaderboard presence

---

## Prerequisites

- You already have the fork at: shmitha/rna-3d-folds-tbm-only-approach
- All 4 inputs are already attached:
  - Stanford RNA 3D Folding Part 2 (competition)
  - extended_rna (dataset by jaejohn)
  - rna_cif_to_csv (dataset by jaejohn)
  - Dependency Installation Code (notebook output by jensen01)
- You already ran it successfully in draft mode

---

## STEP 1: Open Your Existing Fork

1. Go to: https://www.kaggle.com/code/shmitha/rna-3d-folds-tbm-only-approach/edit
2. You should see the notebook editor with all of jaejohn's cells

## STEP 2: Verify Settings

In the right sidebar, check:
- Internet: ON (needed for pip install during development)
- Accelerator: GPU T4 x2
- All 4 inputs attached (competition, extended_rna, rna_cif_to_csv, dependency script)

## STEP 3: Add Fix Cell 1 — Symlink (AT THE VERY TOP)

Click at the very top of the notebook, above all existing cells.
Click "+ Code" to add a new cell.

Download the file fork2_fix_cell_1_symlink.py (provided separately).
Open it in Notepad (NOT from chat), copy ALL contents, paste into this new cell.

Run this cell with Shift+Enter.

## STEP 4: Add Fix Cell 2 — Biopython (RIGHT AFTER Cell 1)

Add another new code cell right after the symlink cell.
Type directly: !pip install biopython -q

Run with Shift+Enter.

## STEP 5: Run ALL of Jaejohn's Original Cells

Run through jaejohn's cells. Takes ~20 minutes.

## STEP 6: Add Fix Cell 3 — Option B Post-Processing (AT THE VERY END)

After ALL jaejohn cells complete, add a new cell at the bottom.

Download fork2_option_b_cell.py. Open in Notepad. Copy. Paste.
Run with Shift+Enter.

## STEP 7: Verify the Output

Add verification cell. Check file exists and size is ~1-2 MB.

## STEP 8: Test With Internet OFF

Turn Internet OFF. Run All in draft. Verify biopython still works.

## STEP 9: Commit

Internet OFF. Save Version. Save & Run All (Commit). Wait ~20-25 min.

## STEP 10: Check Committed Run

Go to notebook page. Verify successful. Check Output tab.

## STEP 11: Submit to Competition

Click Submit to Competition. Add description.

## STEP 12: Wait for Score

Check submissions page for score.

---

## NOTEBOOK CELL ORDER (Final State)

[Cell 1]  Symlink fix                          <- NEW
[Cell 2]  !pip install biopython -q            <- NEW
[Cell 3]  jaejohn's original cell 1            <- ORIGINAL
...
[Cell N]  jaejohn's last original cell         <- ORIGINAL
[Cell N+1] Option B post-processing            <- NEW
[Cell N+2] Verification                        <- NEW (optional)

---

## FILES PROVIDED

fork2_fix_cell_1_symlink.py  — Fixes competition data path (first cell)
fork2_option_b_cell.py       — Option B post-processing (last cell)
FORK2_OPTB_KAGGLE_STEPS.md   — This guide
