# HYBRID NOTEBOOK — Detailed Design & Run Steps
## Fork 2 (0.287) + BASIC Neural Inference
## Timestamp: 2026-03-22

---

## SECTION 1: PRE-PROCESSING ANALYSIS RESULTS

We compared Fork 2, BASIC, and Fork 1 submissions target-by-target.
This analysis was run BEFORE designing the hybrid. Key findings:

### Finding 1: Fork 1 Adds NOTHING to the Hybrid

Fork 1 has ALL-ZERO predictions for 21 of 28 targets. It only has
predictions for 7 targets — and those same targets are already covered
better by Fork 2. **Fork 1 is excluded from the hybrid design.**

### Finding 2: Fork 2 Has Non-Zero Predictions for ALL 28 Targets

Fork 2's gap-filling logic produces predictions even for targets with
no strong template. The spread ranges from 27-6493 Angstroms.
No target is all-zero in Fork 2.

### Finding 3: Fork 2 Has EXCELLENT Prediction Diversity, BASIC Has ALMOST NONE

| Target | Fork2 slot1↔slot4 distance | BASIC slot1↔slot4 distance |
|--------|---------------------------|---------------------------|
| 9EBP   | 495 A                     | <1 A                      |
| 9IWF   | 475 A                     | <1 A                      |
| 9JFS   | 425 A                     | <1 A                      |
| 9J09   | 371 A                     | <2 A                      |
| 9CFN   | 388 A                     | <1 A                      |
| 9OBM   | 125 A                     | 24 A (flipped sign)       |

Fork 2's 5 slots are truly diverse structures. BASIC's 5 slots are
essentially the same structure with <3 Angstrom noise. This means:
- BASIC effectively contributes ONE prediction (slots 1-5 are identical)
- Replacing Fork 2 slots 4+5 with BASIC slots 1+2 = replacing with
  one BASIC prediction duplicated

### Finding 4: All 9762 IDs Match Between Fork 2 and BASIC

Perfect overlap. The combiner can do a clean per-ID merge without
any missing ID issues.

### Finding 5: BASIC Could Help on 19 of 28 Targets

19 targets have weak/no template in Fork 2 AND are within BASIC's
max_seq_len of 256. For these, BASIC's neural prediction is a
genuinely DIFFERENT structure than Fork 2's template-based prediction.

### Finding 6: The Hybrid Is RISK-FREE

Best-of-5 scoring means Kaggle takes the MAX TM-score across 5 slots
per target. If BASIC's prediction (in slot 4 or 5) is better than
all of Fork 2's 3 slots for ANY target, the score improves. If BASIC
is worse for all targets, the score stays at 0.287. It cannot go DOWN.

---

## SECTION 2: WHAT THE EXISTING ADV1 CODE CONTRIBUTES

### ADV1 Code That Is NOT Needed for the Hybrid

| File | Why Not Needed |
|------|---------------|
| `ADV1/train.py` | No training needed — we use existing BASIC weights |
| `ADV1/models/template_encoder.py` | We use Fork 2's raw coordinates directly, not encoded through a neural network |
| `ADV1/data/template_loader.py` | Not needed — we read Fork 2's CSV directly |
| `ADV1/data/dataset.py` | No training data loading needed |
| `ADV1/data/collate.py` | No batch collation needed |
| `ADV1/data/augmentation.py` | No augmentation needed |
| `ADV1/losses/*.py` | No loss computation needed |
| `ADV1/config.yaml` | ADV1-specific config — we use BASIC's config for inference |

### ADV1/BASIC Code That IS Needed for the Hybrid

| File | What We Need From It | Source |
|------|---------------------|--------|
| `BASIC/models/backbone.py` | Load frozen RibonanzaNet → pairwise features | BASIC directory |
| `BASIC/models/distance_head.py` | Load trained MLP (pair_dim=64) | BASIC directory |
| `BASIC/models/reconstructor.py` | MDS + gradient refinement → 3D coords | BASIC directory |
| `BASIC/predict.py` | Per-target inference logic | BASIC directory |
| `BASIC/utils/submission.py` | Format output CSV | BASIC directory |
| `BASIC/config.yaml` | Backbone and head config (NOT ADV1's config) | BASIC directory |
| `BASIC/checkpoints/best_model.pt` | Trained distance head weights (312KB) | BASIC directory |
| `RibonanzaNet.pt` | Pretrained backbone weights (43MB) | ribonanza-weights/ |
| `RibonanzaNet/` | Network.py class definitions | RibonanzaNet/ repo |

### Key Insight: The Hybrid Uses BASIC's Code, Not ADV1's Code

The hybrid does NOT use template features through a neural network.
It uses Fork 2's predictions as-is in slots 1-3 and BASIC's neural
predictions in slots 4-5. No template encoding, no concatenation.

The ADV1 code remains valuable for the FUTURE (if we want to train
a model that integrates templates through a neural network), but for
the immediate hybrid submission, we only need BASIC's inference code.

---

## SECTION 3: THE HYBRID NOTEBOOK DESIGN

### Slot Strategy

| Slot | Source | What It Contains |
|------|--------|-----------------|
| 1 | Fork 2's slot 1 | Best template prediction |
| 2 | Fork 2's slot 2 | 2nd template prediction |
| 3 | Fork 2's slot 3 | 3rd template prediction |
| 4 | **BASIC slot 1** | Neural network prediction (NO template) |
| 5 | Fork 2's slot 4 | 4th template prediction (keeps diversity) |

Why not replace slot 5 too? Fork 2's slot 4 diversity is very high
(57-1321 A from slot 1). BASIC's 5 slots are nearly identical (<3 A).
One BASIC prediction is enough — the second would be a duplicate.

### The Kaggle Notebook Cell Structure

```
=== Part A: Fork 2's Full Pipeline (jaejohn's code, unchanged) ===

[Cell 1]  Symlink fix
[Cell 2]  !pip install biopython -q
[Cell 3-N] jaejohn's original cells
          → produces /kaggle/working/submission.csv (raw Fork 2 output)

=== Part B: Rename Fork 2 Output ===

[Cell N+1] Rename Fork 2 output
           import os
           os.rename('/kaggle/working/submission.csv', '/kaggle/working/fork2_raw.csv')

=== Part C: BASIC Neural Inference ===

[Cell N+2] Load BASIC code from uploaded dataset
           import sys
           sys.path.insert(0, '/kaggle/input/basic-rna3d-code/')

[Cell N+3] Load BASIC model
           Load RibonanzaNet.pt + best_model.pt
           Initialize backbone + distance_head (pair_dim=64)

[Cell N+4] Run BASIC inference on ALL test sequences
           Read test_sequences.csv
           For each target: tokenize → backbone → distance_head → MDS → coords
           → produces /kaggle/working/basic_raw.csv

=== Part D: Combine Fork 2 + BASIC ===

[Cell N+5] The combiner script
           Read fork2_raw.csv (all 5 slots)
           Read basic_raw.csv (only slot 1 needed)
           For each ID:
             - Slots 1-3: Fork 2's x_1..z_3
             - Slot 4: BASIC's x_1, y_1, z_1
             - Slot 5: Fork 2's x_4, y_4, z_4
           Write /kaggle/working/hybrid_raw.csv

=== Part E: Option B Post-Processing (safety net) ===

[Cell N+6] Read sample_submission.csv
           Map hybrid predictions to expected IDs
           Fill zeros for any missing hidden-target IDs
           Write final /kaggle/working/submission.csv
```

### What to Upload as Kaggle Datasets

| Dataset Name | Contents | Approx Size |
|-------------|----------|-------------|
| `basic-rna3d-code` | All .py files: models/backbone.py, models/distance_head.py, models/reconstructor.py, predict.py (modified for Kaggle), utils/submission.py, config.yaml | ~50 KB |
| `basic-rna3d-weights` | best_model.pt | 312 KB |
| `ribonanzanet-weights` | RibonanzaNet.pt | 43 MB |
| `ribonanzanet-repo` | Cloned RibonanzaNet/ folder (Network.py, etc.) | ~10 MB |

Note: Some of these may already be uploaded from our earlier BASIC
submission work. Check existing Kaggle datasets first.

---

## SECTION 4: NEW CODE NEEDED

### Code File 1: `hybrid_combiner.py` (NEW — does not exist in ADV1)

Purpose: Reads Fork 2 and BASIC CSVs, merges slots, writes hybrid CSV.

```
Input:  fork2_raw.csv (9762 rows, 18 columns — all 5 prediction slots from Fork 2)
        basic_raw.csv (9762 rows, 18 columns — all 5 prediction slots from BASIC)
Output: hybrid_raw.csv (9762 rows, 18 columns — slots 1-3 from Fork2, slot 4 from BASIC, slot 5 from Fork2)
```

This is a simple CSV merge — no neural network, no training, no GPU.
~40 lines of Python.

### Code File 2: `basic_inference_kaggle.py` (NEW — does not exist)

Purpose: Runs BASIC model inference in the Kaggle environment.
This is a simplified version of BASIC/predict.py adapted for Kaggle paths.

Differences from BASIC/predict.py:
- Paths point to /kaggle/input/ datasets instead of local paths
- No argparse — all config is inline
- Loads BASIC checkpoint (pair_dim=64, frozen backbone)
- Does NOT use template_encoder or template features
- Outputs CSV matching competition format

~80 lines of Python.

### Code File 3: Option B post-processing (ALREADY EXISTS)

We already have fork2_option_b_cell.py from the previous session.
It works unchanged — reads sample_submission.csv, maps IDs, fills zeros.

---

## SECTION 5: WHAT TO DO BEFORE CREATING THE HYBRID

### Pre-Step 1: Verify BASIC Model Can Run on Kaggle (CRITICAL)

Before building the hybrid, test that BASIC inference works on Kaggle.
Create a simple test notebook that:
1. Loads RibonanzaNet.pt from uploaded dataset
2. Loads best_model.pt from uploaded dataset
3. Runs inference on ONE target (e.g., 8ZNQ, 30 nt)
4. Prints coordinates

If this fails (import errors, path issues, GPU problems), debug it
BEFORE adding it to Fork 2's notebook. This avoids wasting a submission
attempt on a broken notebook.

### Pre-Step 2: Upload BASIC Code/Weights to Kaggle

Package these into Kaggle datasets:
1. Create dataset `basic-rna3d-code` with all .py files
2. Create dataset `basic-rna3d-weights` with best_model.pt
3. Check if `ribonanzanet-weights` already exists from earlier

### Pre-Step 3: Local Dry Run of the Combiner

Run hybrid_combiner.py locally against the actual Fork 2 and BASIC CSVs:
```bash
python hybrid_combiner.py \
    --fork2 fork2-JJ/REMOTE/run1/submission-fork2.csv \
    --basic BASIC/submission.csv \
    --output hybrid_test.csv
```
Verify output has 9762 rows, 18 columns, correct ID order.

---

## SECTION 6: STEP-BY-STEP RUN PLAN

### Step 1: Write hybrid_combiner.py locally
Location: ADV1/hybrid_combiner.py (or a new HYBRID/ directory)
Test locally against existing CSVs.

### Step 2: Write basic_inference_kaggle.py locally
Location: ADV1/basic_inference_kaggle.py
Test locally if possible (requires RibonanzaNet repo + weights).

### Step 3: Upload to Kaggle
Create 2-3 new Kaggle datasets with BASIC code + weights.

### Step 4: Open Fork 2's Kaggle notebook
URL: https://www.kaggle.com/code/shmitha/rna-3d-folds-tbm-only-approach/edit

### Step 5: Add cells AFTER Fork 2's code + Option B cell
The order becomes:
```
[Existing] Cell 1: Symlink
[Existing] Cell 2: pip install biopython
[Existing] Cells 3-N: jaejohn's code → submission.csv
[NEW]      Cell N+1: Rename to fork2_raw.csv
[NEW]      Cell N+2: Load BASIC code + model
[NEW]      Cell N+3: Run BASIC inference → basic_raw.csv
[NEW]      Cell N+4: Combine Fork2 + BASIC → hybrid_raw.csv
[UPDATE]   Cell N+5: Option B post-processing on hybrid_raw.csv → submission.csv
```

### Step 6: Test in draft mode with Internet ON
Run all cells. Verify each step produces expected output.

### Step 7: Test with Internet OFF
Ensure all pip installs and model loads work offline.

### Step 8: Commit and submit
Save & Run All (Commit) with Internet OFF → Submit to Competition.

### Step 9: Compare scores
- Fork 2 alone: 0.287
- Hybrid (Fork 2 + BASIC): ???
- If hybrid > 0.287 → BASIC neural predictions are adding value
- If hybrid = 0.287 → BASIC predictions aren't helping but aren't hurting

---

## SECTION 7: WHY THIS MIGHT (AND MIGHT NOT) BEAT 0.287

### Why It Might

For 19 weak-template targets (TM < 0.1 in Fork 2), BASIC provides a
COMPLETELY DIFFERENT structure. Even if BASIC's structure is only
slightly closer to truth, best-of-5 picks it. Example:
- Fork 2 slot 1 for 9OBM: (158, 135, 123) — template-based guess
- BASIC slot 1 for 9OBM: (-9, 12, -10) — neural network guess
- If true structure is near (-5, 8, -12), BASIC wins this target

Over 19 targets, even a few wins at +0.02-0.05 each could push
the overall score from 0.287 to 0.30-0.32.

### Why It Might Not

1. BASIC scored 0.092 overall — its neural predictions may genuinely
   be worse than Fork 2's template-based guesses for every single target
2. BASIC was trained with frozen backbone on only 661 structures — the
   learned features may not be useful for these specific test targets
3. Best-of-5 only helps if at least ONE of BASIC's predictions beats
   ALL of Fork 2's 4 remaining predictions for some target

### The Bottom Line

This is a LOW-RISK experiment. It costs one Kaggle submission (~15 min
to test). If it works: great, we beat 0.287. If it doesn't: we still
have Fork 2 at 0.287 as our fallback. The time investment is writing
the combiner (~40 lines) + Kaggle inference adapter (~80 lines) +
uploading datasets + testing.

---

## SECTION 8: FILES TO CREATE (DO NOT CREATE YET)

| File | Location | Lines | Purpose |
|------|----------|-------|---------|
| `hybrid_combiner.py` | ADV1/ or HYBRID/ | ~40 | Merge Fork2 + BASIC CSVs by replacing slots |
| `basic_inference_kaggle.py` | ADV1/ or HYBRID/ | ~80 | Run BASIC model inference adapted for Kaggle paths |
| `HYBRID_KAGGLE_STEPS.md` | ADV1/ or HYBRID/ | ~100 | Step-by-step Kaggle submission guide |
| Config path updates | BASIC/config.yaml copy | minimal | Kaggle-specific paths |

Total new code: ~120 lines. Everything else reuses existing BASIC code.
