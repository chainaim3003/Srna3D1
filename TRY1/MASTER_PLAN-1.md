# MASTER PLAN — All Runs, All Phases, All Submissions

## Competition: Stanford RNA 3D Folding Part 2
## Goal: Predict 3D coordinates of C1' atoms for 28 test RNA targets
## Scoring: TM-score (0-1), best-of-5 predictions, averaged across targets

---

## Overview: The Complete Pipeline

```
PHASE A: Approach 1 (3 flavors on Kaggle)
  └── 3 notebook runs → template coordinates + search results

PHASE B: BASIC (1 flavor, local training)
  └── Frozen RibonanzaNet + distance head → trained checkpoint + predictions

PHASE C: ADV1-Run1 (local or Kaggle)
  └── Combines BASIC checkpoint + Approach 1 templates → better predictions

PHASE D: Hybrid Submissions (mix-and-match)
  └── Combine all approaches into best-of-5 diversity → highest TM-score

PHASE E: Beyond ADV1 (future)
  └── IPA → Recycling → MSA → Distillation → Ensemble
```

---
---

# PHASE A: APPROACH 1 — TEMPLATE-BASED MODELING (3 FLAVORS)

## What It Does

Searches PDB for RNA structures with similar sequences to our 28 test targets.
Copies C1' coordinates from matching structures (templates) to the test targets.
This is the simplest, most proven approach — 1st place in Part 1 scored 0.593 TM-score with templates alone.

## 3 Flavors — Same Input, Different Processing, Different Quality

All 3 use the exact same input data from the competition:
- `test_sequences.csv` — 28 test RNA sequences
- `PDB_RNA/*.cif` — ~15,000+ CIF structure files
- `PDB_RNA/pdb_seqres_NA.fasta` — all RNA sequences in PDB

### Flavor A1: Your Notebook (DasLab Baseline)

**Notebook:** `approach1_kaggle_submission.ipynb` (uploaded to Kaggle)
**Pipeline:** MMseqs2 search → DasLab `create_templates_csv.py` → coordinates
**Temporal cutoff:** Skipped (`--skip_temporal_cutoff`)

**Files produced on Kaggle:**

| File | What's inside | Size (est.) | Why save it |
|------|---------------|-------------|-------------|
| `submission.csv` | 28 targets × N residues, 5 coordinate sets. Columns: ID, resname, resid, x_1,y_1,z_1 ... x_5,y_5,z_5 | ~50-200 KB | Direct submission. Template coords for hybrid slots 1-2. Template coords → distance features for ADV1 template encoder. |
| `templates.csv` | Raw output from DasLab `create_templates_csv.py`. May differ from submission.csv formatting. | ~50-200 KB | Backup before reformatting. |
| `Result.txt` | Tab-separated: query, target, evalue, qstart, qend, tstart, tend, qaln, taln. One row per MMseqs2 hit. Multiple hits per target ranked by e-value. | ~10-500 KB | **CRITICAL for ADV1.** See "How Approach 1 feeds into ADV1" section below. |
| `query.fasta` | 28 test sequences in FASTA format. | ~5 KB | Minor reference, can regenerate. |

**Save to:** `APPROACH1-TEMPLATE/results/mine/`

### Flavor A2: rhijudas Part 2 Baseline (Competition Organizer)

**URL:** `https://www.kaggle.com/code/rhijudas/mmseqs2-3d-rna-template-identification-part-2`
**Pipeline:** MMseqs2 search → DasLab tools → coordinates (handles temporal cutoffs correctly)
**Why this one:** Built specifically for Part 2 by competition organizers. Safest choice.

**Files to download:** EVERYTHING in the Output tab. At minimum `submission.csv`. May also contain intermediate search results.

**Save to:** `APPROACH1-TEMPLATE/results/rhijudas/`

### Flavor A3: jaejohn 1st Place (Part 1 Winner)

**URL:** `https://www.kaggle.com/code/jaejohn/rna-3d-folds-tbm-only-approach`
**Pipeline:** MMseqs2 search → custom code (better template ranking, gap-filling, diversity)
**Why this one:** Scored 0.593 TM-score in Part 1. Likely strongest results.
**Caution:** Built for Part 1. Check that output targets match 28 Part 2 test targets.

**Files to download:** EVERYTHING in the Output tab. At minimum `submission.csv`.

**Save to:** `APPROACH1-TEMPLATE/results/jaejohn/`

### After All 3 Runs

```
APPROACH1-TEMPLATE/results/
├── mine/
│   ├── submission.csv        ← Coordinates (DasLab baseline)
│   ├── templates.csv         ← Raw template output
│   ├── Result.txt            ← ALL search hits ← CRITICAL for ADV1
│   └── query.fasta           ← Test sequences as FASTA
├── rhijudas/
│   ├── submission.csv        ← Coordinates (organizer, Part 2 specific)
│   └── [all other Output files]
└── jaejohn/
    ├── submission.csv        ← Coordinates (1st place quality)
    └── [all other Output files]
```

**Immediate action:** Submit all 3 submission.csv files to Kaggle. Compare TM-scores. Pick the best one as primary template source for ADV1.

---

### What's Inside Result.txt (Example)

```
8ZNQ    4ABC_A_1    1.2e-12    1    30    5    35    ACCGUGACGGG    ACCGUGACGGG
8ZNQ    5DEF_B_1    3.4e-08    1    28    3    31    ACCGUGA-GGG    ACCGUGAGGG-
8ZNQ    7GHI_C_1    1.1e-03    5    25    10   30    GACGGGCCU      GACGGGCCU
R1107   9JKL_A_1    2.5e-15    1    45    1    45    AUGCUUAG...    AUGCUUAG...
```

| Column | What | Used in ADV1 for |
|--------|------|-----------------|
| query | Test target ID (e.g., 8ZNQ) | Group hits per target |
| target | PDB chain that matched | Identify which structure was used |
| evalue | Match quality (lower = better) | **Template confidence score** — how much model should trust this template |
| qstart/qend | Which test residues aligned | **Alignment mask** — which residues have real template coords |
| tstart/tend | Which template residues aligned | Map test → template positions |
| qaln/taln | Actual alignment (with gaps as `-`) | **Gap mask** — which residues were gap-filled (interpolated, less reliable) |

---
---

# PHASE B: BASIC — FROZEN RIBONANZANET + DISTANCE HEAD (1 FLAVOR)

## What It Does

```
RNA sequence → RibonanzaNet (FROZEN, 100M params) → pairwise features (B, N, N, 64)
  → Distance Head (TRAINABLE, ~100K params, 3-layer MLP) → predicted distance matrix (N×N)
  → MDS + gradient refinement → 3D coordinates (N×3) × 5 diverse predictions
```

**BASIC is 1 flavor. No variations.** One codebase, one config (`freeze: true`), one training run (epochs 1→50), one `best_model.pt`, one `submission.csv`.

## What BASIC Produces

| File | What's inside | Used downstream? |
|------|---------------|-----------------|
| `checkpoints/best_model.pt` | Distance head weights + optimizer + scheduler + epoch + val_loss | **YES — ADV1 loads this for partial weight initialization** |
| `checkpoints/latest_model.pt` | Same structure, from most recent completed epoch | Resume training only |
| `checkpoints/model_epoch10.pt` | Periodic snapshot | Checkpoint ensemble (Phase D) |
| `checkpoints/model_epoch20.pt` | Periodic snapshot | Checkpoint ensemble |
| `checkpoints/model_epoch30.pt` | Periodic snapshot | Checkpoint ensemble |
| `checkpoints/model_epoch40.pt` | Periodic snapshot | Checkpoint ensemble |
| `checkpoints/model_epoch50.pt` | Periodic snapshot | Checkpoint ensemble |
| `submission.csv` | 28 targets × N residues, 5 coordinate sets | Direct submission + hybrid slot 5 |

## What's Inside best_model.pt

```python
{
    'epoch': 42,                        # Whichever epoch had lowest val loss
    'model_state_dict': {               # Distance head weights — THE KEY THING
        'mlp.0.weight':  (128, 64),     # First layer ← PARTIALLY transferred to ADV1
        'mlp.0.bias':    (128,),        # First layer bias ← transferred
        'mlp.1.weight':  (128,),        # LayerNorm ← transferred
        'mlp.1.bias':    (128,),        # LayerNorm ← transferred
        'mlp.4.weight':  (128, 128),    # Second layer ← transferred directly
        'mlp.4.bias':    (128,),        # ← transferred
        'mlp.5.weight':  (128,),        # LayerNorm ← transferred
        'mlp.5.bias':    (128,),        # ← transferred
        'mlp.8.weight':  (1, 128),      # Final layer ← transferred directly
        'mlp.8.bias':    (1,),          # ← transferred
    },
    'optimizer_state_dict': {...},      # Adam state (momentum etc.)
    'scheduler_state_dict': {...},      # LR schedule position
    'scaler_state_dict': {...},         # Mixed precision state
    'val_loss': 420.0,                  # Val loss at best epoch
    'best_val_loss': 420.0,
    'best_epoch': 42,
    'config': {...},                    # Full config as dict
}
```

---
---

# HOW APPROACH 1 FEEDS INTO ADV1 — DETAILED DATA FLOW

## Step-by-Step: From Kaggle Output to Model Input

### Step 1: template_loader.py parses submission.csv

```
Input:  results/[best]/submission.csv

  8ZNQ_1, A, 1, 12.3, -5.4, 8.1, ...
  8ZNQ_2, C, 2, 15.1, -3.2, 9.8, ...
  8ZNQ_3, C, 3, 18.7, -1.1, 11.2, ...

Processing:
  Group rows by target_id (extract "8ZNQ" from "8ZNQ_1")
  For each target, collect (x_1, y_1, z_1) into numpy array

Output: dict mapping target_id → coordinates
  {"8ZNQ": array([[12.3, -5.4, 8.1],
                  [15.1, -3.2, 9.8],
                  [18.7, -1.1, 11.2],
                  ...]), shape (30, 3)}
```

### Step 2: template_encoder.py converts coordinates to features

```
Input:  coordinates array (30, 3) for target 8ZNQ

Step 2a: Compute pairwise distance matrix from template coords

  residue 1 to residue 1:  0.0 Å
  residue 1 to residue 2:  5.9 Å
  residue 1 to residue 3: 11.2 Å
  residue 1 to residue 4: 15.8 Å
  ...

  Result: distance matrix (30, 30) — real numbers in Ångströms

Step 2b: Bin distances into discrete categories

  0.0 Å  → bin 0  (0-2 Å)
  5.9 Å  → bin 2  (4-6 Å)
  11.2 Å → bin 5  (10-12 Å)
  15.8 Å → bin 7  (14-16 Å)

  Result: binned matrix (30, 30) — integers 0 to 21

Step 2c: One-hot encode the bins

  bin 2 → [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

  Result: (30, 30, 22)

Step 2d: Project through a linear layer to compress

  Linear(22, 16)

  Result: (30, 30, 16) — template features

Output: template feature tensor (30, 30, 16)
```

### Step 3: Confidence extraction from Result.txt

```
Input:  Result.txt line "8ZNQ  4ABC_A  1.2e-12  1  30  5  35  ACCGUGAC...  ACCGUGAC..."

  e-value 1.2e-12 → very confident match
  Convert: confidence = -log10(1.2e-12) = 11.9 → normalize to ~1.0

  For residues that ALIGNED (qaln/taln match): confidence = 1.0
  For residues that were GAP-FILLED (gaps in alignment): confidence = 0.3
  For targets with NO HITS in Result.txt: confidence = 0.0 everywhere

Output: confidence mask (30, 30, 1) — added as extra channel
```

### Step 4: Concatenation in model forward pass

```
RibonanzaNet backbone:     pairwise features  (30, 30, 64)
Template encoder:          template features   (30, 30, 16)

Concatenate:               combined features   (30, 30, 80)

Distance head:             Linear(80, 128) → ... → distances (30, 30)
```

### Visual Diagram: Complete ADV1 Data Flow

```
           Approach 2 path                    Approach 1 path
           ─────────────                      ─────────────

RNA sequence "ACCGUGACGGG..."        submission.csv    Result.txt
        │                              from Kaggle      from Kaggle
        ▼                                  │                │
   RibonanzaNet                       template_loader.py    │
   (unfrozen last 2 layers)           (parse CSV → coords)  │
        │                                  │                │
        ▼                                  ▼                ▼
   pairwise features               template_encoder.py  confidence
   (30, 30, 64)                    (coords → dist       extraction
        │                           → bins → features)   (e-values →
        │                                  │             mask)
        │                                  ▼                │
        │                           template features       │
        │                           (30, 30, 16)   ◄───────┘
        │                                  │
        └──────────── CONCATENATE ─────────┘
                          │
                          ▼
                   combined features
                   (30, 30, 80)
                          │
                          ▼
                    distance head
                    (MLP: 80 → 128 → 128 → 1)
                    [first 64 channels warm-started from BASIC]
                    [last 16 channels learned fresh in ADV1]
                          │
                          ▼
                   predicted distances
                   (30, 30)
                          │
                          ▼
                   MDS + refinement
                          │
                          ▼
                   3D coordinates
                   (30, 3)
```

### For Targets with NO Template Hits

When Result.txt has zero hits for a target:
- Template features are all zeros: `(N, N, 16)` filled with 0.0
- Confidence mask is all zeros
- Distance head sees 80 channels where the last 16 are blank
- It relies purely on the pairwise features (first 64 channels)
- This is effectively the same as BASIC

This is why BASIC's trained weights for the first 64 channels carry forward — they handle the "no template" case. ADV1 only needs to LEARN the template channels during training.

---
---

# HOW BASIC FEEDS INTO ADV1 — PARTIAL WEIGHT LOADING

## The Problem

BASIC distance head first layer: `Linear(64, 128)` — weight shape (128, 64)
ADV1 distance head first layer: `Linear(80, 128)` — weight shape (128, 80)

Shapes don't match — you can't `load_state_dict()` directly.

## The Solution

```python
# In ADV1's train.py:

# 1. Create ADV1 distance head with new input dim
adv1_head = DistanceMatrixHead(pair_dim=80, hidden_dim=128, num_layers=3)

# 2. Load BASIC checkpoint
basic_ckpt = torch.load('BASIC/checkpoints/best_model.pt')
basic_weights = basic_ckpt['model_state_dict']

# 3. Partial transfer
adv1_state = adv1_head.state_dict()

for key, basic_param in basic_weights.items():
    if key == 'mlp.0.weight':
        # First layer: BASIC is (128, 64), ADV1 is (128, 80)
        # Copy BASIC weights into the first 64 columns
        # Leave columns 65-80 as random initialization
        adv1_state[key][:, :64] = basic_param
    elif key in adv1_state and adv1_state[key].shape == basic_param.shape:
        # All other layers match — copy directly
        adv1_state[key] = basic_param

adv1_head.load_state_dict(adv1_state)
```

## What the Weight Matrix Looks Like

```
BASIC first layer: (128 rows × 64 cols)
┌──────────────────────────┐
│                          │
│   TRAINED WEIGHTS        │  ← learned over 50 epochs
│   (knows pairwise        │
│    features → distances) │
│                          │
└──────────────────────────┘

ADV1 first layer: (128 rows × 80 cols)
┌──────────────────────────┬───────────┐
│                          │           │
│   COPIED FROM BASIC      │  RANDOM   │  ← template feature channels
│   (columns 0-63)         │  (64-79)  │    start random, learn during
│                          │           │    ADV1 training
│                          │           │
└──────────────────────────┴───────────┘

All other layers (128→128, 128→1): transferred directly, identical shapes.
```

## The Result

On ADV1 epoch 1, the model already predicts reasonable distances from the pairwise portion (columns 0-63 have BASIC's trained weights). It doesn't yet know how to use template features (columns 64-79 are random). ADV1 training teaches it to use templates. Much faster than training from scratch.

---
---

# PHASE C: ADV1-RUN1 — UNFREEZE + TEMPLATES

## What Changes from BASIC

| Aspect | BASIC | ADV1-Run1 |
|--------|-------|-----------|
| Backbone | Fully frozen (100M params locked) | Last 2 layers unfrozen (adapt to 3D) |
| Template features | None | Template coords from Approach 1 → encoded features |
| Distance head input | 64 (pairwise only) | 80 (64 pairwise + 16 template) |
| Distance head init | Random | Warm-started from BASIC checkpoint |
| Learning rate | 1e-4 (head only) | 1e-5 backbone, 1e-4 head (discriminative) |
| LR warmup | None | 5% of epochs |
| Effective batch size | 4 | 16 (batch 4 × grad accum 4) |

## Prerequisites

| Item | Source | Must be complete |
|------|--------|-----------------|
| `best_model.pt` | BASIC Phase B | YES — for weight warm-start |
| Best `submission.csv` | Approach 1 Phase A | YES — for template coordinates |
| `Result.txt` | Your notebook (A1) | YES — for confidence scores |
| ADV1 code written | New code | YES — before training |

## Picking the Template Source

After submitting all 3 Approach 1 flavors to Kaggle, compare TM-scores:

| Best score from | Use for ADV1 |
|----------------|--------------|
| A1 (yours) | `results/mine/submission.csv` + `results/mine/Result.txt` |
| A2 (rhijudas) | `results/rhijudas/submission.csv` + `results/mine/Result.txt` |
| A3 (jaejohn) | `results/jaejohn/submission.csv` + `results/mine/Result.txt` |

**Note:** Always use YOUR `Result.txt` for e-values and alignments. The other notebooks may not save their intermediate search results. The MMseqs2 search is the same across all 3 (same query, same database) — only template selection and coordinate transfer differ.

## Training

```bash
cd C:\sathya\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH2-RIBBOZANET\ADV1
python train.py --config config.yaml
```

## Prediction

```bash
python predict.py --config config.yaml --checkpoint checkpoints/best_model.pt \
  --test_csv "../../APPROACH1-TEMPLATE/test_sequences (1).csv" --output submission.csv
```

At inference, predict.py:
1. Loads test sequence
2. Loads template coords from Approach 1 submission.csv for that target
3. Extracts confidence from Result.txt for that target
4. Runs backbone (unfrozen layers in eval mode) → pairwise features (N, N, 64)
5. Runs template encoder → template features (N, N, 16)
6. Concatenates → (N, N, 80)
7. Runs distance head → distances (N, N)
8. MDS + refinement → 3D coordinates (N, 3)

---
---

# PHASE D: HYBRID SUBMISSIONS — MIX AND MATCH

## Why Hybrid Matters

Competition scores **best-of-5** per target. Different methods produce different predictions. Combining them maximizes the chance that at least one slot is close to the true structure.

## Strategy D1: Conservative (Template-Heavy)

Best when Approach 1 templates are strong.

| Slot | Source | What |
|------|--------|------|
| 1 | Best Approach 1 | Template #1 (best MMseqs2 hit) — direct copy |
| 2 | Best Approach 1 | Template #2 (2nd best hit) — direct copy |
| 3 | ADV1 | Neural network prediction (clean) |
| 4 | ADV1 | Neural network prediction (with noise) |
| 5 | BASIC | Neural network prediction (different model = diversity) |

## Strategy D2: Maximum Diversity (Cross-Method)

Best when unsure which approach works best.

| Slot | Source | What |
|------|--------|------|
| 1 | A1 (your notebook) | Template coords from DasLab pipeline |
| 2 | A2 (rhijudas) | Template coords from organizer pipeline |
| 3 | A3 (jaejohn) | Template coords from 1st place pipeline |
| 4 | ADV1 | Neural network + template features |
| 5 | BASIC | Pure neural network |

## Strategy D3: Neural-Net Heavy (weak templates)

For targets where Approach 1 found few or poor hits.

| Slot | Source | What |
|------|--------|------|
| 1 | ADV1 | Clean prediction |
| 2 | ADV1 | Noisy (noise_scale=0.3) |
| 3 | ADV1 | Noisy (noise_scale=0.5) |
| 4 | ADV1 | Different refinement steps |
| 5 | BASIC | Different model |

## Strategy D4: Per-Target Adaptive (SMARTEST)

Use Result.txt to decide PER TARGET:

```
For each of the 28 targets:
  Count hits in Result.txt
  Check best e-value

  If best_evalue < 1e-10 (strong match):
    → Strategy D1 (template-heavy)

  Elif best_evalue < 1e-3 (moderate match):
    → Slots 1-2: ADV1 (uses templates internally)
      Slots 3-4: Approach 1 templates
      Slot 5: BASIC

  Elif hits == 0 (no template):
    → Strategy D3 (neural-net only)
```

---
---

# PHASE E: BEYOND ADV1 — FUTURE IMPROVEMENT LEVELS

## Roadmap

```
BASIC (done)
  └── ADV1-Run1 (unfreeze + templates)
        └── Level 1: IPA head (replace lossy MDS pipeline)
              └── Level 2: Recycling (3 iterations)
              └── Level 3: MSA features (evolutionary signal)
                    └── Level 4: Knowledge distillation from RNAPro
                          └── Level 5: Ensemble strategies
                          └── Level 6: Data enhancement
```

## Level 1: Replace Distance Head with IPA

**What:** Invariant Point Attention predicts 3D coordinates directly from features. Removes the lossy features→distances→MDS→coords pipeline.
**How:** `pip install invariant-point-attention` (lucidrains). 4 IPA blocks. FAPE loss.
**VRAM:** ~8-10 GB (fits T4).
**Effort:** 1-2 weeks. Complex rotation math.
**Impact:** Large — state-of-the-art architecture.
**Source:** DESIGN.md Option C. AlphaFold2, RhoFold+, RNAPro all use this.

## Level 2: Recycling

**What:** Run IPA 3 times, feeding each round's output back as input.
**How:** For-loop around IPA forward pass. Stop-gradient on earlier iterations.
**Effort:** Low once IPA exists.
**Impact:** Medium — progressive refinement.
**Source:** AlphaFold2 uses 3 recycling iterations.

## Level 3: MSA Features

**What:** Evolutionary covariation from Multiple Sequence Alignments.
**How:** Competition provides MSAs in `MSA/` folder. Encode as covariance matrix + conservation scores.
**VRAM:** +4-6 GB.
**Effort:** 1 week. Column-wise attention.
**Impact:** Medium-High — orthogonal signal.
**Source:** DESIGN.md Option E. RNAPro + RhoFold+ both use MSAs.

## Level 4: Knowledge Distillation from RNAPro

**What:** Rent cloud A100, run RNAPro on all sequences, train small model to mimic its outputs.
**How:** Student loss = α × ground_truth_loss + β × RNAPro_prediction_loss.
**Cost:** ~$50-200 cloud GPU rental.
**Impact:** Medium-Large.
**Source:** DESIGN.md Option F. RNAPro: Apache-2.0, weights on HuggingFace.

## Level 5: Ensemble Strategies

**What:** Multiple seeds, checkpoint ensemble, MC Dropout, template variation.
**Impact:** Small-Medium (1-5% improvement).
**Source:** DESIGN.md Section 5.

## Level 6: Data Enhancement

**What:** Noisy student (pseudo-labels), homology augmentation, crop augmentation.
**Impact:** Small-Medium.
**Source:** DESIGN.md Section 4. Ribonanza paper confirms noisy student works.

---
---

# COMPLETE FILE INVENTORY

```
Srna3D1/
├── .gitignore                                    ← Excludes RibonanzaNet/, weights, pickle
├── RibonanzaNet/                                 ← NOT in git (clone separately)
├── Ranger-Deep-Learning-Optimizer/               ← NOT in git (clone separately)
├── ribonanza-weights/                            ← NOT in git (download from Kaggle)
│   └── RibonanzaNet.pt                           (43.3 MB)
├── stanford3d-pickle/                            ← NOT in git (download from Kaggle)
│   └── pdb_xyz_data.pkl                          (52.3 MB)
│
└── TRY1/
    ├── MASTER_PLAN.md                            ← THIS DOCUMENT
    │
    ├── APPROACH1-TEMPLATE/
    │   ├── approach1_kaggle_submission.ipynb      ← Your notebook
    │   ├── APPROACH1_RUN_STEPS.md
    │   ├── EXECUTION_GUIDE.md
    │   ├── SOURCES.md                            ← URLs for all 3 notebooks
    │   ├── test_sequences (1).csv                ← 28 Part 2 test targets
    │   └── results/
    │       ├── mine/
    │       │   ├── submission.csv                ← Template coords (DasLab)
    │       │   ├── templates.csv                 ← Raw template output
    │       │   ├── Result.txt                    ← ALL MMseqs2 hits ← CRITICAL
    │       │   └── query.fasta                   ← Test seqs as FASTA
    │       ├── rhijudas/
    │       │   ├── submission.csv                ← Template coords (organizer)
    │       │   └── [other output files]
    │       └── jaejohn/
    │           ├── submission.csv                ← Template coords (1st place)
    │           └── [other output files]
    │
    └── APPROACH2-RIBBOZANET/
        ├── BASIC/
        │   ├── BASIC_RUN_STEPS.md
        │   ├── config.yaml                       ← freeze: true, epochs: 50
        │   ├── train.py                          ← With --resume feature
        │   ├── predict.py                        ← Fixed resname parsing
        │   ├── models/
        │   │   ├── backbone.py                   ← Frozen backbone wrapper
        │   │   ├── distance_head.py              ← MLP: Linear(64→128→128→1)
        │   │   └── reconstructor.py              ← MDS + gradient refinement
        │   ├── data/
        │   │   ├── dataset.py                    ← Pickle parser (C1' = sugar_ring[0])
        │   │   ├── collate.py
        │   │   └── augmentation.py
        │   ├── losses/
        │   │   ├── distance_loss.py
        │   │   ├── constraint_loss.py
        │   │   └── tm_score_approx.py
        │   ├── utils/
        │   │   ├── submission.py                 ← Fixed: uses 'sequence' column
        │   │   └── pdb_parser.py
        │   ├── checkpoints/
        │   │   ├── best_model.pt                 ← CARRIES TO ADV1
        │   │   ├── latest_model.pt               ← For resume
        │   │   └── model_epoch{10,20,30,40,50}.pt
        │   └── submission.csv                    ← BASIC predictions
        │
        └── ADV1/
            ├── DESIGN.md                         ← Full design (Options A-F)
            ├── ADV1_RUN1_STEPS.md                ← Run steps for Phase 1+2
            ├── BEYOND_ADV1_RUN1.md               ← IPA, MSA, distillation
            ├── config.yaml                       ← freeze: false, freeze_first_n: 7
            ├── train.py                          ← Warmup + discriminative LR
            ├── predict.py                        ← Loads templates at inference
            ├── models/
            │   ├── backbone.py                   ← Selective unfreezing
            │   ├── distance_head.py              ← MLP: Linear(80→128→128→1)
            │   ├── template_encoder.py           ← NEW: coords→dist→bins→features
            │   └── reconstructor.py              ← Same as BASIC
            ├── data/
            │   ├── dataset.py                    ← Extended: loads templates too
            │   ├── collate.py                    ← Extended: batches templates
            │   ├── augmentation.py               ← Same as BASIC
            │   └── template_loader.py            ← NEW: parses Approach 1 CSV
            ├── losses/                           ← Same as BASIC (all 3 files)
            ├── utils/                            ← Same as BASIC (both files)
            ├── checkpoints/
            │   └── best_model.pt                 ← ADV1 trained model
            └── submission.csv                    ← ADV1 predictions
```

---
---

# EXECUTION ORDER

```
PARALLEL TRACK A (Kaggle, ~30 min each):
  A1: Run your notebook ──────────────► Download results/mine/ (4 files)
  A2: Fork+run rhijudas Part 2 ───────► Download results/rhijudas/
  A3: Fork+run jaejohn 1st place ─────► Download results/jaejohn/
  ► Submit all 3 submission.csv to Kaggle → get TM-scores
  ► Pick best as primary template source for ADV1

PARALLEL TRACK B (Local, ~2.5 hrs remaining):
  B1: BASIC training epochs 36→50 (currently running)
  B2: Generate BASIC submission.csv
  B3: git commit + push
  ► Submit BASIC submission.csv → get baseline TM-score

SEQUENTIAL (after A + B complete):
  C1: Copy BASIC code to ADV1
  C2: Write new files (template_encoder.py, template_loader.py)
  C3: Modify files (backbone.py, distance_head.py, etc.)
  C4: Smoke test ADV1 components
  C5: Train ADV1 (~5-8 hours)
  C6: Generate ADV1 submission.csv
  ► Submit ADV1 → compare vs BASIC

FINAL:
  D1: Build hybrid submission (mix Approach 1 + BASIC + ADV1)
  D2: Submit hybrid → compare vs individual submissions
  ► Best score wins
```

---
---

# STATUS TRACKER

| Step | Phase | Status | Notes |
|------|-------|--------|-------|
| A1: Run your notebook | Approach 1 | IN PROGRESS | Uploaded to Kaggle, checking setup |
| A2: Fork+run rhijudas | Approach 1 | NOT STARTED | ~15-30 min on Kaggle |
| A3: Fork+run jaejohn | Approach 1 | NOT STARTED | ~15-30 min on Kaggle |
| A-submit: Submit all 3 | Approach 1 | NOT STARTED | Compare TM-scores |
| B1: BASIC training 36→50 | BASIC | IN PROGRESS | Running on other computer |
| B2: BASIC prediction | BASIC | NOT STARTED | After B1 completes |
| B3: git commit + push | BASIC | NOT STARTED | Save everything |
| B-submit: Submit BASIC | BASIC | NOT STARTED | Get baseline TM-score |
| C1: Copy BASIC → ADV1 | ADV1 | NOT STARTED | Preserve BASIC as-is |
| C2: Write new ADV1 files | ADV1 | NOT STARTED | template_encoder.py, template_loader.py |
| C3: Modify ADV1 files | ADV1 | NOT STARTED | backbone.py, distance_head.py, etc. |
| C4: ADV1 smoke tests | ADV1 | NOT STARTED | Test each component |
| C5: ADV1 training | ADV1 | NOT STARTED | ~5-8 hours |
| C6: ADV1 prediction | ADV1 | NOT STARTED | After C5 |
| C-submit: Submit ADV1 | ADV1 | NOT STARTED | Compare vs BASIC |
| D1: Build hybrid | Hybrid | NOT STARTED | After A + B + C |
| D2: Submit hybrid | Hybrid | NOT STARTED | Final best submission |

---
---

# REFERENCES (Official Documentation Only)

| Source | URL | Used in |
|--------|-----|---------|
| Competition | https://www.kaggle.com/competitions/stanford-rna-3d-folding-2 | Everything |
| Competition paper | https://pmc.ncbi.nlm.nih.gov/articles/PMC12776560/ | Scoring |
| DasLab create_templates | https://github.com/DasLab/create_templates | Your notebook (A1) |
| rhijudas Part 2 baseline | https://www.kaggle.com/code/rhijudas/mmseqs2-3d-rna-template-identification-part-2 | A2 |
| jaejohn 1st place | https://www.kaggle.com/code/jaejohn/rna-3d-folds-tbm-only-approach | A3 |
| RibonanzaNet | https://github.com/Shujun-He/RibonanzaNet | Backbone |
| RibonanzaNet weights | https://www.kaggle.com/datasets/shujun717/ribonanzanet-weights | Pretrained |
| Training pickle | https://www.kaggle.com/datasets/shujun717/stanford3d-dataprocessing-pickle | Training data |
| RNAPro | https://github.com/NVIDIA-Digital-Bio/RNAPro | Architecture reference |
| RhoFold+ | https://www.nature.com/articles/s41592-024-02487-0 | IPA + MSA reference |
| AlphaFold2 | https://www.nature.com/articles/s41586-021-03819-2 | IPA, FAPE, recycling |
| MMseqs2 | https://github.com/soedinglab/MMseqs2 | Sequence search |
| lucidrains IPA | pip install invariant-point-attention | IPA implementation |
| ADV1 DESIGN.md | Local: ADV1/DESIGN.md | Options A-F, decision matrix |
