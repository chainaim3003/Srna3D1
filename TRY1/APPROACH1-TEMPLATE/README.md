# Approach 1: Template-Based Modeling — Code & Execution

## What this directory contains

| File | What it is |
|------|-----------|
| `approach1_kaggle_submission.ipynb` | **THE notebook.** Upload this to Kaggle and run it. |
| `run_local.sh` | Shell script for local/Colab testing (needs full PDB_RNA) |
| `EXECUTION_GUIDE.md` | Step-by-step instructions |
| `README.md` | This file |

## CRITICAL FACT: The core code is NOT written from scratch

The template-based modeling pipeline has **two official, verified implementations**:

1. **DasLab `create_templates`** — https://github.com/DasLab/create_templates
   - Written by the competition organizers (Stanford Das Lab)
   - The `create_templates_csv.py` script does the heavy lifting:
     parses CIF files, aligns sequences, transfers C1' coordinates
   - Apache-2.0 license

2. **MMseqs2** — https://github.com/soedinglab/MMseqs2
   - Fast sequence search tool (like BLAST but faster)
   - Used to find similar RNA sequences in the PDB
   - GPL-3.0 license

Our notebook **clones and calls these official tools**. We do NOT rewrite them.
Rewriting would risk introducing bugs in coordinate transfer logic that has 
already been validated by the competition organizers and 1st-place winners.

## The pipeline

```
test_sequences.csv
       │
       ▼
  [MMseqs2 createdb]  ← convert to MMseqs2 database format
       │
       ▼
  [MMseqs2 search]    ← search against entire PDB RNA sequence database
       │                  (pdb_seqres_NA.fasta — provided by competition)
       ▼
  [MMseqs2 convertalis] ← convert results to readable TSV
       │
       ▼
  [DasLab create_templates_csv.py]  ← for each hit:
       │                                - open the CIF file
       │                                - align query to template sequence
       │                                - copy C1' coordinates
       │                                - handle gaps/insertions
       ▼
  submission.csv       ← competition-format output
```

## Official sources (verified, no hallucination)

| Source | URL | Verified |
|--------|-----|----------|
| DasLab create_templates repo | https://github.com/DasLab/create_templates | Yes — Apache-2.0 |
| DasLab create_templates_csv.py CLI | See repo README for full --help output | Yes |
| MMseqs2 official docs | https://github.com/soedinglab/MMseqs2/wiki | Yes |
| Official Kaggle baseline notebook | https://www.kaggle.com/code/rhijudas/mmseqs2-3d-rna-template-identification | Yes — by competition organizer |
| Official Part 2 notebook | https://www.kaggle.com/code/rhijudas/mmseqs2-3d-rna-template-identification-part-2 | Yes |
| 1st place TBM notebook | https://www.kaggle.com/code/jaejohn/rna-3d-folds-tbm-only-approach | Yes — by Part 1 winner |
| Pre-computed templates dataset | https://www.kaggle.com/datasets/rhijudas/rna-3d-folding-templates | Yes |
| Competition paper (PMC) | https://pmc.ncbi.nlm.nih.gov/articles/PMC12776560/ | Yes — peer reviewed |
| RNAPro template docs | https://github.com/NVIDIA-Digital-Bio/RNAPro | Yes — references same notebooks |
