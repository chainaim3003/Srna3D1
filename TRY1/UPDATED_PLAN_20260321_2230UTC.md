# UPDATED PLAN — Stanford RNA 3D Folding Part 2
## Timestamp: 2026-03-21 ~22:30 UTC

---

## KEY DISCOVERY: This is a Code Competition

Kaggle **re-runs your notebook on hidden test data** with different sequences.
- A pre-made submission.csv alone does NOT work
- The notebook must read test_sequences.csv dynamically
- The notebook must produce predictions for ALL sequences (public + hidden)
- Internet must be OFF for final commit/submission
- The "Submit" button only appears after committing (not in draft mode)

---

## WHAT WE HAVE ACCOMPLISHED

### BASIC (Approach 2) — SUBMITTED, Scored 0.092 TM-score

**What was done:**
- Trained 50 epochs on 661 PDB structures, frozen RibonanzaNet backbone + trainable distance head
- Generated submission.csv locally (predict.py)
- Discovered submission.csv had targets in test_sequences.csv order, but Kaggle sample expects alphabetical order
- Created fix_submission_order.py to reorder rows to match sample_submission.csv
- Verified row count: 9762 rows match sample exactly, all IDs match
- Uploaded submission_fixed.csv as Kaggle dataset
- First attempt: "Submission Scoring Error" — learned that direct CSV pre-loading only covers 28 public targets, not hidden ones
- Created kaggle_dynamic_v2.py — reads sample_submission.csv at runtime, fills our predictions where available, zeros for unknown targets
- Successfully submitted via Kaggle notebook "RNA-Basic" (Version 6, 25s runtime)
- **TM-score: 0.092** — ranks ~2199 of 2248, essentially the default/baseline score

**Why 0.092 is expected (not a bug):**
- Hidden test targets get zero predictions (only public targets have real coordinates)
- 6 of 28 public targets exceed max_seq_len=256, producing extrapolated garbage coordinates
- The frozen backbone + tiny distance head + lossy MDS pipeline is a minimal baseline

**Kaggle path discovery:** Competition data mounts at `/kaggle/input/competitions/stanford-rna-3d-folding-2/` not `/kaggle/input/stanford-rna-3d-folding-2/`. Symlink fix required.

---

### Approach 1 — MINE (Kalai / Harishmitha)

**What exists:**
- Two runs completed on Kaggle using DasLab baseline notebook
- Files saved locally:
  - `mine/kalai/run1/` — submission.csv, Result.txt, notebook
  - `mine/Harishmitha/run1/` — submission.csv, Result.txt, notebook

**What we found:**
- Both Result.txt files are byte-for-byte identical (same search, same `--skip_temporal_cutoff`)
- Both use the DasLab baseline notebook (approach1_kaggle_submission.ipynb)
- Submission CSVs differ slightly in predictions 2-5 (random perturbation noise)
- Harishmitha has better 9ZCC coverage (no zeros in prediction 1)
- Per-target TM-scores from MMSeqs2 search (shared across all Approach 1 runs):
  - Strong: 9G4J (0.924), 9LEC (0.856), 9LJN (0.786), 9LEL (0.683), 9E74 (0.569), 9E9Q (0.560)
  - Weak: remaining ~16 targets score 0.02-0.10

**STATUS: STUCK**
- These are pre-made CSVs from a Kaggle notebook run
- They CANNOT be submitted as-is because this is a Code Competition
- The original DasLab notebook was run with `--skip_temporal_cutoff` (includes self-matches)
- To submit: would need to fork the same notebook on Kaggle, ensure it runs end-to-end with Internet OFF, and commit
- Lower priority since fork1-RJ and fork2-JJ use the same or better approaches

**RECOMMENDED ACTION:** Deprioritize. Fork1 and Fork2 are better paths. The Kalai/Harishmitha results serve as reference scores for comparison.

---

### Approach 1 — Fork 1 (rhijudas Part 2 Baseline)

**Source:** https://www.kaggle.com/code/rhijudas/mmseqs2-3d-rna-template-identification-part-2

**What we know:**
- Built specifically for Part 2 by the competition organizer (DasLab)
- URL explicitly says "part-2"
- Handles temporal cutoffs properly (unlike our MINE runs)
- The original notebook scored well in Part 1 reference tests

**What was done:**
- Downloaded the notebook .ipynb file locally to `fork1-RJ/`
- Created run step markdowns: FORK1_RJ_KAGGLE.md, FORK1_RJ_LOCAL.md
- Analyzed testResult_fork1.txt vs Kalai's Result.txt:
  - Search results are identical (same hits, same e-values)
  - Critical difference: fork1 correctly maps multi-chain positions (9MME chains at different qstart offsets) while Kalai stacks all chains at position 1-580

**STATUS: RUNNING ON KAGGLE — Expected to complete in 2-3 hours**
- Running under shmitha's Kaggle account (or similar)
- Need to verify when it finishes:
  1. Does it produce submission.csv?
  2. Are row counts per target correct (match sample_submission.csv)?
  3. Are target IDs correct for Part 2?
  4. Did the symlink cell work for the `/kaggle/input/competitions/` path issue?

**WHAT CHANGES WERE NEEDED ON KAGGLE:**
- Added symlink cell at top:
  ```python
  import os
  src = "/kaggle/input/competitions/stanford-rna-3d-folding-2"
  dst = "/kaggle/input/stanford-rna-3d-folding-2"
  if os.path.exists(src) and not os.path.exists(dst):
      os.symlink(src, dst)
  ```
- Added `!pip install biopython -q` cell (for `from Bio import pairwise2`)
- Internet must be ON during development, OFF for final commit

**NEXT STEPS WHEN IT FINISHES:**
1. Check if submission.csv was generated
2. Download and verify row counts match sample
3. If row counts match → turn Internet OFF → Save & Run All (Commit) → Submit
4. If row counts DON'T match → apply Option B post-processing (same as BASIC fix)
5. This is our highest-confidence notebook for a valid Code Competition submission

---

### Approach 1 — Fork 2 (jaejohn 1st Place TBM)

**Source:** https://www.kaggle.com/code/jaejohn/rna-3d-folds-tbm-only-approach

**What we know:**
- 1st place Part 1 winner (0.593 TM-score)
- Built for Part 1 — required adaptation for Part 2
- Has 3 extra input dependencies: extended_rna dataset, rna_cif_to_csv dataset, Dependency Installation Script notebook
- Uses custom template ranking, gap-filling, and diversity strategies

**What was done:**
- Forked as shmitha/rna-3d-folds-tbm-only-approach on Kaggle
- Verified all 4 inputs attached (competition + 2 datasets + dependency script)
- Fixed symlink for competition data path
- Fixed `ModuleNotFoundError: No module named 'Bio'` with `!pip install biopython -q`
- Fixed `FileNotFoundError` for train_sequences.csv with symlink
- Ran successfully — produced submission.csv

**What we found (analyzing the submission.csv):**
- 28 targets present, 9762 total rows, no NaN, no zeros
- Excellent prediction diversity across 5 predictions (truly different structures)
- **CRITICAL PROBLEM: Row count mismatches for multi-chain targets**

| Target | Expected | Fork2 | Diff |
|--------|----------|-------|------|
| 9MME | 4184 | 4640 | +456 |
| 9LEL | 449 | 476 | +27 |
| 9E75 | 155 | 165 | +10 |
| 9LEC | 370 | 378 | +8 |
| 9G4J | 341 | 334 | -7 |
| 9E74 | 249 | 255 | +6 |
| Others | — | — | +1 to +4 |

- Target order is test_sequences.csv order (not alphabetical like sample)
- Root cause: jaejohn's code computes residue counts from stoichiometry/all_sequences instead of the `sequence` column

**STATUS: SUBMISSION.CSV GENERATED BUT CANNOT BE SUBMITTED AS-IS**
- Row count mismatches mean IDs won't match sample_submission.csv
- Even if we fix the CSV, the notebook will produce wrong row counts on hidden data too

**TWO OPTIONS DISCUSSED:**

**Option A: Surgical Fix (fix core code)**
- Download notebook, find where it determines per-target residue count
- Change to use `len(row['sequence'])` from test_sequences.csv
- Test locally, upload back to Kaggle
- Pros: Clean, correct, works on hidden data
- Cons: Requires deep understanding of jaejohn's code, risk of breaking things, time-consuming

**Option B: Post-Processing Fix (add cell at end) — RECOMMENDED**
- Leave jaejohn's code untouched
- Add one cell at the end that:
  1. Reads sample_submission.csv from competition data (gets swapped with hidden data by Kaggle)
  2. Reads jaejohn's raw submission.csv
  3. For each expected ID: if jaejohn produced it, use his coordinates; if not, use zeros
  4. Overwrites submission.csv with corrected version
- Pros: Zero risk of breaking core logic, fast to implement, works for hidden data
- Cons: Wastes computation on extra rows, some residues get zeros (e.g., 7 missing for 9G4J), hacky
- CRITICAL: The notebook is the submission — we commit it, and Kaggle re-runs it on hidden data

**NEXT STEPS:**
1. Add Option B post-processing cell to the Kaggle notebook (shmitha/rna-3d-folds-tbm-only-approach)
2. Test in draft mode — verify it produces corrected submission.csv
3. Turn Internet OFF → Save & Run All (Commit) → Submit
4. Compare TM-score against BASIC (0.092) and fork1

---

## COMPETITION CONTEXT

| Team/Approach | Public Score | Notes |
|---------------|-------------|-------|
| best_template_oracle | 0.554 | Benchmark reference |
| 1st place (AyPy) | 0.499 | Current leader |
| Top 50 range | 0.44-0.49 | Competitive |
| **Fork1 (rhijudas)** | **TBD** | **Running now** |
| **Fork2 (jaejohn)** | **TBD** | **Needs post-processing fix** |
| **BASIC (our neural net)** | **0.092** | **Submitted, baseline score** |
| Default/zeros | 0.079-0.092 | Bottom of leaderboard |

- Competition ends: **March 25, 2026** (~4 days remaining)
- Submissions per day: **5** (failed submissions count)
- Final submissions for judging: **2** (auto-selected from best if not manually chosen)

---

## PRIORITY ACTION ITEMS (Ordered)

### IMMEDIATE (Today)

1. **Fork 2 — Add Option B post-processing cell** to shmitha/rna-3d-folds-tbm-only-approach on Kaggle
   - Same pattern as kaggle_dynamic_v2.py used for BASIC
   - Commit with Internet OFF → Submit
   - Expected to score significantly higher than 0.092

2. **Fork 1 — Monitor and submit when complete**
   - Check if it finishes successfully
   - Verify row counts match sample
   - If good: commit with Internet OFF → Submit
   - If row count issues: apply same Option B fix

### SHORT TERM (Next 1-2 days)

3. **Compare all TM-scores** once Fork1 and Fork2 are submitted
   - Pick best 2 as "Final Submissions" for private leaderboard

4. **Fork 2 — Option A surgical fix** (if time permits)
   - Would eliminate the row-count mismatch at the source
   - Better for hidden test targets (no wasted computation, no missing residues)

5. **BASIC — Build proper inference notebook** (if time permits)
   - Upload full model code + weights as Kaggle datasets
   - Create notebook that runs predict.py on any test_sequences.csv
   - Would give a real BASIC score instead of 0.092

### STRETCH (If competition extended or for learning)

6. **ADV1** — Build on top of BASIC with template features from Approach 1
7. **Hybrid submission** — Combine best of template-based and neural approaches
8. **Git push** — Tag and commit all new files (markdowns, scripts, plans)

---

## FILE INVENTORY

```
Srna3D1/TRY1/
├── MASTER_PLAN.md                          (original plan from earlier session)
├── UPDATED_PLAN_20260321_2230UTC.md        (THIS FILE)
├── APPROACH1-TEMPLATE/
│   ├── sample_submission.csv               (competition reference — 9762 rows, alphabetical order)
│   ├── test_sequences (1).csv              (28 Part 2 targets)
│   ├── submission (3).csv                  (another submission reference)
│   ├── mine/
│   │   ├── kalai/run1/                     (submission.csv, Result.txt, notebook)
│   │   └── Harishmitha/run1/              (submission.csv, Result.txt, notebook)
│   ├── fork1-RJ/
│   │   ├── FORK1_RJ_RUN_STEPS.md
│   │   ├── FORK1_RJ_KAGGLE.md
│   │   ├── FORK1_RJ_LOCAL.md
│   │   ├── testResult_fork1.txt           (MMseqs2 search results)
│   │   └── mmseqs2-3d-rna-...-part-2.ipynb (downloaded notebook)
│   └── fork2-JJ/
│       ├── FORK2_JJ_RUN_STEPS.md
│       ├── FORK2_JJ_KAGGLE.md
│       ├── FORK2_JJ_LOCAL.md
│       └── submission (2).csv              (raw output — HAS ROW COUNT ISSUES)
└── APPROACH2-RIBBOZANET/
    └── BASIC/
        ├── train.py, predict.py, config.yaml, etc.
        ├── checkpoints/best_model.pt       (trained weights)
        ├── submission.csv                  (original — wrong target order)
        ├── submission_fixed.csv            (reordered to match sample)
        ├── fix_submission_order.py         (script that reordered)
        ├── verify_fixed.py                 (verification script)
        └── KAGGLE_SUBMISSION_GUIDE.md
```

## KAGGLE NOTEBOOKS STATUS

| Notebook | Account | Status | Score |
|----------|---------|--------|-------|
| RNA-Basic (Version 6) | tarunsathyab | Submitted | 0.092 |
| rna-3d-folds-tbm-only-approach | shmitha | Draft — needs post-processing cell | TBD |
| fork1 (rhijudas Part 2) | TBD | Running (~2-3 hrs remaining) | TBD |

---

## LESSONS LEARNED THIS SESSION

1. **Code Competition = notebook is the submission.** Pre-made CSVs don't work because Kaggle re-runs on hidden data.
2. **Kaggle path quirk:** Competition data mounts at `/kaggle/input/competitions/...` not `/kaggle/input/...`. Symlink fixes this.
3. **Smart quotes from chat break Python.** Always download .py files and copy from Notepad, never paste directly from chat.
4. **Row ordering matters.** sample_submission.csv defines the exact ID order Kaggle expects.
5. **Part 1 notebooks need adaptation for Part 2.** Different competition slug, different test targets, different stoichiometry handling.
6. **The "Submit" button only appears after committing.** Draft sessions don't have it.
7. **Failed submissions count toward the daily 5 limit.**
8. **Multi-chain targets are the hardest format issue.** Different notebooks interpret stoichiometry differently, leading to row count mismatches.
