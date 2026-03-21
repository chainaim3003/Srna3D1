# Approach 1: Template-Based Modeling — Source Notebooks

## 3 Notebooks to Run (all on Kaggle, all produce submission.csv)

### 1. Mine (DasLab baseline pipeline)
- **File:** `approach1_kaggle_submission.ipynb` (in this folder)
- **Based on:** https://github.com/DasLab/create_templates
- **What it does:** MMseqs2 search + DasLab create_templates_csv.py
- **Save results to:** `results/mine/`

### 2. rhijudas — Competition Organizer Part 2 Baseline
- **URL:** https://www.kaggle.com/code/rhijudas/mmseqs2-3d-rna-template-identification-part-2
- **Who:** DasLab (competition organizers), specifically for Part 2
- **How to run:** Fork (Copy & Edit) → attach competition data → Run All
- **Save results to:** `results/rhijudas/`

### 3. jaejohn — 1st Place Part 1 Winner
- **URL:** https://www.kaggle.com/code/jaejohn/rna-3d-folds-tbm-only-approach
- **Who:** "john" — 1st place winner of Part 1 (TM-score 0.593)
- **How to run:** Fork (Copy & Edit) → attach competition data → Run All
- **Note:** Built for Part 1. May auto-adapt to Part 2 if it reads test_sequences.csv from competition data. Check output targets match Part 2.
- **Save results to:** `results/jaejohn/`

## What to Download from Each

After each notebook finishes, go to the **Output** tab and download:

1. `submission.csv` — the final coordinates (REQUIRED)
2. `Result.txt` — raw MMseqs2 hits with e-values and alignments (IMPORTANT for ADV1)
3. `templates.csv` — intermediate template output if present (OPTIONAL)

## How Results Will Be Used

- **Direct submission:** Any submission.csv can be uploaded to the competition as-is
- **Hybrid with Approach 2:** Mix template coords (slots 1-2) with neural network predictions (slots 3-5)
- **ADV1 template features:** Result.txt + submission.csv feed into ADV1's template encoder
