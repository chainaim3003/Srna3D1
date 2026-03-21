# MASTER PLAN — All Runs, All Phases, All Submissions

## Competition: Stanford RNA 3D Folding Part 2
## Goal: Predict 3D coordinates of C1' atoms for 28 test RNA targets
## Scoring: TM-score (0-1), best-of-5 predictions, averaged across targets

---

## Quick Reference: What Feeds Into What

```
PHASE A: Approach 1 (3 Kaggle notebook runs)
  │
  ├── A1 your notebook ──► results/mine/
  │     ├── submission.csv     (coordinates for direct submission)
  │     ├── templates.csv      (raw template output, may differ from submission)
  │     ├── Result.txt         (ALL MMseqs2 hits with e-values + alignments)
  │     └── query.fasta        (test sequences in FASTA format)
  │
  ├── A2 rhijudas Part 2 ──► results/rhijudas/
  │     ├── submission.csv     (coordinates for direct submission)
  │     └── [intermediate files — download whatever is in Output tab]
  │
  └── A3 jaejohn 1st place ──► results/jaejohn/
        ├── submission.csv     (coordinates for direct submission)
        └── [intermediate files — download whatever is in Output tab]

PHASE B: BASIC (local training)
  │
  ├── checkpoints/best_model.pt   (trained distance head → warm-starts ADV1)
  └── submission.csv               (coordinates for direct submission)

PHASE C: ADV1-Run1 (local or Kaggle training)
  │
  │  INPUTS:
  │    ├── BASIC best_model.pt           (partial weight initialization)
  │    ├── Approach 1 submission.csv     (template coordinates → features)
  │    └── Approach 1 Result.txt         (e-values → confidence, alignments → masks)
  │
  │  OUTPUTS:
  │    ├── checkpoints/best_model.pt     (trained ADV1 model)
  │    └── submission.csv                (coordinates for direct submission)

PHASE D: Hybrid Submissions (mix-and-match script)
  │
  │  INPUTS:
  │    ├── A1/A2/A3 submission.csv files (template coordinates for slots 1-2)
  │    ├── BASIC submission.csv          (neural net coords for diversity)
  │    └── ADV1 submission.csv           (neural net + template coords for slots 3-5)
  │
  │  OUTPUT:
  │    └── hybrid_submission.csv         (best-of-5 diversity across methods)
```

---
---

# PHASE A: Approach 1 — Template-Based Modeling (3 Runs on Kaggle)

---

## Run A1: Your Notebook (DasLab Baseline Pipeline)

### What to run
Your uploaded notebook: `approach1_kaggle_submission.ipynb`

### Kaggle settings
- **Data:** `stanford-rna-3d-folding-2` attached
- **Internet:** ON
- **Accelerator:** GPU T4 x2

### What the notebook produces (9 cells)

| Cell | What it creates | File location on Kaggle |
|------|----------------|------------------------|
| Cell 5 | `query.fasta` — 28 test sequences in FASTA format | `/kaggle/working/query.fasta` |
| Cell 6 | `Result.txt` — ALL MMseqs2 search hits | `/kaggle/working/Result.txt` |
| Cell 7 | `templates.csv` — DasLab template coordinates | `/kaggle/working/templates.csv` |
| Cell 8 | `submission.csv` — reformatted for competition | `/kaggle/working/submission.csv` |

### What to download from Output tab

| File | Size (est.) | What's inside | Why you need it |
|------|-------------|---------------|-----------------|
| **`submission.csv`** | ~50-200 KB | 28 targets × N residues, 5 coordinate sets each. Columns: ID, resname, resid, x_1,y_1,z_1 ... x_5,y_5,z_5 | Direct submission to Kaggle. Template coords for slots 1-2 of hybrid. Template coords → distance features for ADV1 inference. |
| **`templates.csv`** | ~50-200 KB | Raw output from DasLab `create_templates_csv.py`. May have different column format than submission.csv. | Backup — contains the same coordinates before reformatting. |
| **`Result.txt`** | ~10-500 KB | Tab-separated: query, target, evalue, qstart, qend, tstart, tend, qaln, taln. One row per MMseqs2 hit. Multiple hits per target, ranked by e-value. | **CRITICAL for ADV1:** (1) E-values → template confidence scores for template encoder. (2) Alignment columns (qaln, taln) → which residues matched vs gap-filled → mask for template features. (3) All hits, not just top 5 → try different template combos. (4) Count hits per target → identify which targets have NO templates (pure neural net fallback). |
| **`query.fasta`** | ~5 KB | 28 test sequences in FASTA format | Minor — useful reference, can regenerate from test_sequences.csv |

### Save to
```
APPROACH1-TEMPLATE/results/mine/
├── submission.csv
├── templates.csv
├── Result.txt
└── query.fasta
```

---

## Run A2: rhijudas Part 2 Baseline (Competition Organizer)

### What to run
Fork: `https://www.kaggle.com/code/rhijudas/mmseqs2-3d-rna-template-identification-part-2`

### Kaggle settings
Same as A1: data attached, Internet ON, GPU T4 x2

### What the notebook produces
This is the organizer's notebook — we don't control what intermediate files it creates. It will produce at minimum a `submission.csv` in the Output tab. It may also produce `Result.txt` or equivalent intermediate files.

### What to download from Output tab

Download **EVERYTHING** in the Output tab. At minimum:

| File | Why you need it |
|------|-----------------|
| **`submission.csv`** | Direct submission. Template coords for hybrid. Potentially best template source for ADV1 (organizer handles temporal cutoffs correctly). |
| **Any `.txt` files** | May contain MMseqs2 raw results (equivalent to Result.txt) |
| **Any `.csv` besides submission** | May contain intermediate template data |

### Why this notebook matters
Built specifically for Part 2 by the competition organizers. Handles temporal cutoffs correctly — Part 2 test targets have specific date cutoffs meaning templates released after that date should be excluded. Your notebook uses `--skip_temporal_cutoff` which is less precise.

### Save to
```
APPROACH1-TEMPLATE/results/rhijudas/
├── submission.csv
└── [all other Output tab files]
```

---

## Run A3: jaejohn 1st Place (Part 1 Winner)

### What to run
Fork: `https://www.kaggle.com/code/jaejohn/rna-3d-folds-tbm-only-approach`

### Kaggle settings
Same as A1: data attached, Internet ON, GPU T4 x2

### What the notebook produces
This is the 1st place winner's notebook — we don't control its internals. It scored 0.593 TM-score in Part 1 through likely better template ranking, smarter gap-filling, and more diverse prediction strategies.

### What to download from Output tab

Download **EVERYTHING** in the Output tab. At minimum:

| File | Why you need it |
|------|-----------------|
| **`submission.csv`** | Direct submission. Potentially highest quality template coords. |
| **Any intermediate files** | May contain search results, template rankings, alignment data |

### Important caveat
This notebook was built for Part 1. Check that the output contains the 28 Part 2 test targets (8ZNQ, etc.), not Part 1 targets. If it reads `test_sequences.csv` from the attached competition data, it auto-adapts. If it has Part 1 targets hardcoded, the output is for the wrong targets.

### Save to
```
APPROACH1-TEMPLATE/results/jaejohn/
├── submission.csv
└── [all other Output tab files]
```

---

## After All 3 Approach 1 Runs: What You Have

```
APPROACH1-TEMPLATE/results/
├── mine/
│   ├── submission.csv        ← Coordinates (DasLab baseline)
│   ├── templates.csv         ← Raw template output
│   ├── Result.txt            ← ALL search hits (CRITICAL for ADV1)
│   └── query.fasta           ← Test sequences as FASTA
├── rhijudas/
│   ├── submission.csv        ← Coordinates (organizer, Part 2 specific)
│   └── [other files]         ← Download everything available
└── jaejohn/
    ├── submission.csv        ← Coordinates (1st place quality)
    └── [other files]         ← Download everything available
```

### Immediate action: Submit all 3 to Kaggle
Upload each `submission.csv` to the competition to get TM-scores.
This tells you which template approach works best for Part 2.

### What Result.txt contains (example)

```
8ZNQ    4ABC_A_1    1.2e-12    1    30    5    35    ACCGUGACGGG    ACCGUGACGGG
8ZNQ    5DEF_B_1    3.4e-08    1    28    3    31    ACCGUGA-GGG    ACCGUGAGGG-
8ZNQ    7GHI_C_1    1.1e-03    5    25    10   30    GACGGGCCU      GACGGGCCU
R1107   9JKL_A_1    2.5e-15    1    45    1    45    AUGCUUAG...    AUGCUUAG...
...
```

Each row = one hit. Columns:
- `query` = test target ID (e.g., 8ZNQ)
- `target` = PDB chain that matched
- `evalue` = match quality (lower = better match)
- `qstart/qend` = which residues in the test sequence aligned
- `tstart/tend` = which residues in the template aligned
- `qaln/taln` = the actual sequence alignment (with gaps shown as `-`)

**This is the raw material for ADV1's template encoder:**
- E-value → confidence score (how much the model should trust this template)
- Alignment → which residues have real template coords vs gap-filled interpolation
- All hits → not just top 5, so you can experiment with different template selections

---
---

# PHASE B: Approach 2 BASIC — Train + Predict (Local)

---

## Run B1: Complete BASIC Training

### Status
Currently running (epochs 14→35)

### Resume command (if needed)
```bash
cd C:\sathya\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH2-RIBBOZANET\BASIC
python train.py --config config.yaml --resume checkpoints/best_model.pt
```

### Predict command (after training completes)
```bash
python predict.py --config config.yaml --checkpoint checkpoints/best_model.pt --test_csv "../../APPROACH1-TEMPLATE/test_sequences (1).csv" --output submission.csv
```

### What BASIC produces

| File | What's inside | Used downstream? |
|------|---------------|-----------------|
| **`checkpoints/best_model.pt`** | Trained distance head weights + optimizer state + scheduler state + epoch number + val loss | **YES — ADV1 uses this for partial weight initialization.** The first 64 input channels of the distance head transfer directly. |
| **`checkpoints/latest_model.pt`** | Same as best_model.pt but from most recent epoch (for resume) | Resume training only |
| **`checkpoints/model_epoch10.pt`** | Periodic snapshot at epoch 10 | Checkpoint ensemble (Level 5 in Beyond ADV1) |
| **`submission.csv`** | 28 targets × N residues, 5 coordinate sets. Columns: ID, resname, resid, x_1..z_5 | **YES — direct submission + slot 5 in hybrid submission for diversity** |

### What's inside best_model.pt (checkpoint dict)

```python
{
    'epoch': 35,                        # Which epoch this was saved at
    'model_state_dict': {...},          # Distance head weights — THE KEY THING
    'optimizer_state_dict': {...},      # Adam optimizer state (momentum etc.)
    'scheduler_state_dict': {...},      # LR schedule position
    'scaler_state_dict': {...},         # Mixed precision scaler state
    'val_loss': 450.0,                  # Validation loss at this epoch
    'best_val_loss': 450.0,             # Best val loss seen so far
    'best_epoch': 35,                   # Which epoch had best val loss
    'config': {...},                    # Full config.yaml as dict
}
```

### What ADV1 extracts from best_model.pt

```python
# model_state_dict contains these keys:
'mlp.0.weight'  → shape (128, 64)    # First layer — PARTIALLY transferred
'mlp.0.bias'    → shape (128,)       # First layer bias — transferred
'mlp.1.weight'  → shape (128,)       # LayerNorm — transferred
'mlp.1.bias'    → shape (128,)       # LayerNorm — transferred
'mlp.4.weight'  → shape (128, 128)   # Second layer — transferred directly
'mlp.4.bias'    → shape (128,)       # transferred
'mlp.5.weight'  → shape (128,)       # LayerNorm — transferred
'mlp.5.bias'    → shape (128,)       # transferred
'mlp.8.weight'  → shape (1, 128)     # Final layer — transferred directly
'mlp.8.bias'    → shape (1,)         # transferred

# ADV1's first layer is (128, 80) instead of (128, 64)
# Columns 0-63: copied from BASIC (knows pairwise features)
# Columns 64-79: random init (template feature channels, learns during ADV1 training)
```

### Save to (already in place)
```
APPROACH2-RIBBOZANET/BASIC/
├── checkpoints/
│   ├── best_model.pt         ← Carries forward to ADV1
│   ├── latest_model.pt       ← For resume
│   └── model_epoch10.pt      ← For checkpoint ensemble
└── submission.csv             ← Direct submission + hybrid slot
```

---
---

# PHASE C: Approach 2 ADV1-Run1 — Train + Predict

---

## Prerequisites (must be complete before starting)

| Prerequisite | From | Specifically needed |
|-------------|------|---------------------|
| BASIC training done | Phase B | `BASIC/checkpoints/best_model.pt` — distance head weights for warm-start |
| Best Approach 1 result chosen | Phase A | `results/[best]/submission.csv` — template coordinates for encoding |
| Result.txt from your notebook | Phase A1 | `results/mine/Result.txt` — e-values and alignments for template confidence |
| ADV1 code written | Step C1 below | Modified backbone.py, new template_encoder.py, etc. |

## Step C0: Pick Best Template Source

After submitting all 3 Approach 1 results to Kaggle, compare TM-scores:

| If best score is from | Use as primary template source |
|----------------------|-------------------------------|
| A1 (your notebook) | `results/mine/submission.csv` + `results/mine/Result.txt` |
| A2 (rhijudas) | `results/rhijudas/submission.csv` + `results/mine/Result.txt` (use your Result.txt for e-values since rhijudas may not save it) |
| A3 (jaejohn) | `results/jaejohn/submission.csv` + `results/mine/Result.txt` |

**Note:** You always use YOUR `Result.txt` for e-values and alignment data, regardless of which submission.csv you pick. The other notebooks may not save their intermediate search results. The MMseqs2 search is the same across all 3 (same query, same database) — only the template selection and coordinate transfer differ.

## Step C1: Build ADV1 Code

Copy BASIC to ADV1 and create/modify files per ADV1_RUN1_STEPS.md:

```bash
cd C:\sathya\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH2-RIBBOZANET
xcopy BASIC\*.py ADV1\ /S /Y
xcopy BASIC\models\*.py ADV1\models\ /S /Y
xcopy BASIC\data\*.py ADV1\data\ /S /Y
xcopy BASIC\losses\*.py ADV1\losses\ /S /Y
xcopy BASIC\utils\*.py ADV1\utils\ /S /Y
copy BASIC\config.yaml ADV1\config.yaml
```

New files to write:
- `models/template_encoder.py` — encode template coords → distance features (N, N, 16)
- `data/template_loader.py` — parse Approach 1 CSV → per-target coordinate arrays

Modified files:
- `models/backbone.py` — selective unfreezing (layers 7-8 trainable)
- `models/distance_head.py` — input dim 64 → 80 (64 pairwise + 16 template)
- `data/dataset.py` — load template features alongside structures
- `data/collate.py` — batch template tensors
- `train.py` — warmup, discriminative LR, partial weight loading from BASIC
- `predict.py` — load templates at inference time
- `config.yaml` — new settings

## Step C2: Train ADV1

```bash
cd C:\sathya\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH2-RIBBOZANET\ADV1
python train.py --config config.yaml
```

## Step C3: Predict

```bash
python predict.py --config config.yaml --checkpoint checkpoints/best_model.pt --test_csv "../../APPROACH1-TEMPLATE/test_sequences (1).csv" --output submission.csv
```

### What ADV1 produces

| File | What's inside | Used downstream? |
|------|---------------|-----------------|
| **`checkpoints/best_model.pt`** | ADV1 distance head (80-dim input) + unfrozen backbone layers | Future phases (IPA, etc.) |
| **`submission.csv`** | 28 targets, 5 predictions each. Better than BASIC because of template features + adapted backbone. | Direct submission + hybrid slots 3-4 |

---
---

# PHASE D: Hybrid Submissions — Mix and Match

---

## Why Hybrid Matters

The competition scores **best-of-5** predictions per target. If ANY one of the 5 slots is close to the true structure, you get a high TM-score for that target. Different methods produce different predictions — combining them maximizes the chance that at least one is close.

## Hybrid Submission Strategies

### Strategy D1: Conservative (Template-Heavy)

Best when Approach 1 templates are strong.

| Slot | Source | What |
|------|--------|------|
| 1 | Best Approach 1 | Template #1 coordinates (best MMseqs2 hit) |
| 2 | Best Approach 1 | Template #2 coordinates (2nd best hit) |
| 3 | ADV1 | Neural network prediction (clean, no noise) |
| 4 | ADV1 | Neural network prediction (with noise for diversity) |
| 5 | BASIC | Neural network prediction (different model = more diversity) |

### Strategy D2: Maximum Diversity (Cross-Method)

Best when you're unsure which approach works best.

| Slot | Source | What |
|------|--------|------|
| 1 | A1 (your notebook) | Template coords from DasLab pipeline |
| 2 | A2 (rhijudas) | Template coords from organizer pipeline |
| 3 | A3 (jaejohn) | Template coords from 1st place pipeline |
| 4 | ADV1 | Neural network + template features |
| 5 | BASIC | Pure neural network (no templates) |

### Strategy D3: Neural-Net Heavy (for targets with weak templates)

For targets where Approach 1 found few or poor hits.

| Slot | Source | What |
|------|--------|------|
| 1 | ADV1 | Clean prediction |
| 2 | ADV1 | Noisy prediction (noise_scale=0.3) |
| 3 | ADV1 | Noisy prediction (noise_scale=0.5) |
| 4 | ADV1 | Different refinement steps |
| 5 | BASIC | Different model for diversity |

### Strategy D4: Per-Target Adaptive

Use Result.txt to decide PER TARGET which strategy to use.

```
For each of the 28 targets:
  Count hits in Result.txt for this target
  Look at best e-value

  If best_evalue < 1e-10 (strong template match):
    → Use Strategy D1 (template-heavy)

  Elif best_evalue < 1e-3 (moderate match):
    → Use Strategy D1 but with ADV1 in slots 1-2
      (ADV1 combines templates + neural net)

  Elif hits == 0 (no template at all):
    → Use Strategy D3 (neural-net only)
```

**This is the smartest strategy** — it adapts to each target's template availability.

## Building the Hybrid (Python script)

A simple script reads the individual submission.csv files and assembles slots:

```python
# hybrid_builder.py (to be written)
# Reads:
#   results/mine/submission.csv      (or best Approach 1)
#   results/mine/Result.txt          (for per-target hit counts)
#   BASIC/submission.csv
#   ADV1/submission.csv
# Writes:
#   hybrid_submission.csv

# For each target:
#   Check Result.txt for hit quality
#   Pick strategy (D1/D2/D3) based on hit quality
#   Assemble 5 slots from the appropriate source CSVs
#   Write combined rows to hybrid_submission.csv
```

---
---

# COMPLETE FILE INVENTORY

## What Gets Saved Where

```
Srna3D1/TRY1/
│
├── APPROACH1-TEMPLATE/
│   ├── approach1_kaggle_submission.ipynb    ← Your notebook
│   ├── APPROACH1_RUN_STEPS.md
│   ├── SOURCES.md                          ← URLs for all 3 notebooks
│   ├── results/
│   │   ├── mine/
│   │   │   ├── submission.csv              ← Template coords (DasLab)
│   │   │   ├── templates.csv              ← Raw template output
│   │   │   ├── Result.txt                 ← ALL MMseqs2 hits ← CRITICAL
│   │   │   └── query.fasta                ← Test seqs as FASTA
│   │   ├── rhijudas/
│   │   │   ├── submission.csv              ← Template coords (organizer)
│   │   │   └── [other output files]
│   │   └── jaejohn/
│   │       ├── submission.csv              ← Template coords (1st place)
│   │       └── [other output files]
│   └── test_sequences (1).csv              ← 28 Part 2 test targets
│
├── APPROACH2-RIBBOZANET/
│   ├── BASIC/
│   │   ├── checkpoints/
│   │   │   ├── best_model.pt              ← Distance head weights ← CARRIES TO ADV1
│   │   │   ├── latest_model.pt            ← For resume
│   │   │   └── model_epoch10.pt           ← Periodic snapshot
│   │   ├── submission.csv                  ← BASIC predictions
│   │   ├── config.yaml                     ← freeze: true, epochs: 35
│   │   ├── train.py                        ← With --resume feature
│   │   ├── predict.py                      ← Fixed resname parsing
│   │   ├── models/backbone.py              ← Frozen backbone wrapper
│   │   ├── models/distance_head.py         ← MLP (64 → 128 → 128 → 1)
│   │   ├── models/reconstructor.py         ← MDS + gradient refinement
│   │   ├── data/dataset.py                 ← Pickle loader (C1' from sugar_ring[0])
│   │   ├── data/collate.py                 ← Batch padding
│   │   ├── data/augmentation.py            ← Random rotation/translation
│   │   ├── losses/distance_loss.py         ← MSE loss
│   │   ├── losses/constraint_loss.py       ← Bond + clash penalties
│   │   ├── losses/tm_score_approx.py       ← Differentiable TM-score
│   │   ├── utils/submission.py             ← Fixed: uses 'sequence' not 'all_sequences'
│   │   └── utils/pdb_parser.py             ← CIF parser
│   │
│   └── ADV1/
│       ├── DESIGN.md                       ← Full design (Options A-F)
│       ├── ADV1_RUN1_STEPS.md              ← Run steps for Phase 1+2
│       ├── BEYOND_ADV1_RUN1.md             ← Future phases
│       ├── checkpoints/
│       │   └── best_model.pt              ← ADV1 trained model (after Phase C)
│       ├── submission.csv                  ← ADV1 predictions (after Phase C)
│       ├── config.yaml                     ← freeze: false, freeze_first_n: 7, template settings
│       ├── models/
│       │   ├── backbone.py                ← Modified: selective unfreezing
│       │   ├── distance_head.py           ← Modified: input dim 80 (64+16)
│       │   ├── template_encoder.py        ← NEW: coords → distance bins → features
│       │   └── reconstructor.py           ← Same as BASIC
│       ├── data/
│       │   ├── dataset.py                 ← Extended: loads templates too
│       │   ├── collate.py                 ← Extended: batches template tensors
│       │   ├── augmentation.py            ← Same as BASIC
│       │   └── template_loader.py         ← NEW: parses Approach 1 CSV → tensors
│       ├── losses/                        ← Same as BASIC (all 3 files)
│       ├── train.py                       ← Modified: warmup, discriminative LR, partial weight load
│       ├── predict.py                     ← Modified: loads templates at inference
│       └── utils/                         ← Same as BASIC (both files)
```

---
---

# EXECUTION ORDER

```
PARALLEL TRACK A (Kaggle, ~30 min each):
  A1: Run your notebook ──────────► Download results/mine/
  A2: Fork+run rhijudas Part 2 ───► Download results/rhijudas/
  A3: Fork+run jaejohn 1st place ─► Download results/jaejohn/
  ► Submit all 3 submission.csv files to get TM-scores
  ► Pick the best one as primary template source

PARALLEL TRACK B (Local, ~3.5 hours):
  B1: Finish BASIC training (epochs 14→35)
  B2: Generate BASIC submission.csv
  ► Submit to get BASIC TM-score

SEQUENTIAL (after A + B complete):
  C1: Build ADV1 code (copy BASIC + write new files)
  C2: Smoke test ADV1 components
  C3: Train ADV1 (~5-8 hours)
  C4: Generate ADV1 submission.csv
  ► Submit to get ADV1 TM-score

FINAL:
  D1: Build hybrid submission (mix Approach 1 + BASIC + ADV1)
  D2: Submit hybrid to Kaggle
  ► Compare all scores, iterate if time permits
```

---
---

# STATUS TRACKER

| Step | Phase | Status | Notes |
|------|-------|--------|-------|
| A1: Run your notebook | Approach 1 | NOT STARTED | ~15-30 min on Kaggle |
| A2: Fork+run rhijudas | Approach 1 | NOT STARTED | ~15-30 min on Kaggle |
| A3: Fork+run jaejohn | Approach 1 | NOT STARTED | ~15-30 min on Kaggle |
| A-submit: Submit all 3 | Approach 1 | NOT STARTED | Compare TM-scores |
| B1: BASIC training | BASIC | IN PROGRESS | Epochs 14→35 running |
| B2: BASIC prediction | BASIC | NOT STARTED | After B1 completes |
| B-submit: Submit BASIC | BASIC | NOT STARTED | Get baseline TM-score |
| C1: Build ADV1 code | ADV1 | NOT STARTED | Copy BASIC + new files |
| C2: ADV1 smoke tests | ADV1 | NOT STARTED | Test each component |
| C3: ADV1 training | ADV1 | NOT STARTED | After C1+C2 pass |
| C4: ADV1 prediction | ADV1 | NOT STARTED | After C3 completes |
| C-submit: Submit ADV1 | ADV1 | NOT STARTED | Compare vs BASIC |
| D1: Build hybrid | Hybrid | NOT STARTED | After A + B + C done |
| D2: Submit hybrid | Hybrid | NOT STARTED | Final best submission |

---
---

# REFERENCES (Official Documentation Only)

| Source | URL | Used in |
|--------|-----|---------|
| Competition | https://www.kaggle.com/competitions/stanford-rna-3d-folding-2 | Everything |
| Competition paper | https://pmc.ncbi.nlm.nih.gov/articles/PMC12776560/ | Scoring methodology |
| DasLab create_templates | https://github.com/DasLab/create_templates | Your notebook (A1) |
| rhijudas Part 2 baseline | https://www.kaggle.com/code/rhijudas/mmseqs2-3d-rna-template-identification-part-2 | Run A2 |
| jaejohn 1st place TBM | https://www.kaggle.com/code/jaejohn/rna-3d-folds-tbm-only-approach | Run A3 |
| RibonanzaNet | https://github.com/Shujun-He/RibonanzaNet | Backbone for BASIC + ADV1 |
| RibonanzaNet weights | https://www.kaggle.com/datasets/shujun717/ribonanzanet-weights | Pretrained weights |
| Training pickle | https://www.kaggle.com/datasets/shujun717/stanford3d-dataprocessing-pickle | Training data |
| RNAPro | https://github.com/NVIDIA-Digital-Bio/RNAPro | Architecture reference for ADV1 |
| MMseqs2 | https://github.com/soedinglab/MMseqs2 | Sequence search in Approach 1 |
| ADV1 DESIGN.md | Local: `ADV1/DESIGN.md` | Options A-F, decision matrix |
