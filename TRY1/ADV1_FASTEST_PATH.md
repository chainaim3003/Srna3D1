# ADV1 — Fastest Path from Current State to Kaggle Submission

## What ADV1 Is (Reminder)

ADV1 = BASIC's neural network + template features from Approach 1.

```
RibonanzaNet backbone → pairwise features (N,N,64)
Approach 1 templates  → template features  (N,N,16)
                                              ↓
                        CONCATENATE → (N,N,80)
                                              ↓
                        Distance Head (modified: 80→128→128→1)
                                              ↓
                        Distance Matrix → MDS → 3D coordinates
```

For targets WITH good templates: template features guide the prediction.
For targets WITHOUT templates: template channels are zeros, model relies on pairwise features (same as BASIC).

---

## What We Already Have (Prerequisites — All Done)

| Component | Status | Location |
|-----------|--------|----------|
| BASIC trained weights | ✅ Done | `BASIC/checkpoints/best_model.pt` (312KB) |
| RibonanzaNet pretrained | ✅ Done | `ribonanza-weights/RibonanzaNet.pt` (43MB) |
| All BASIC code | ✅ Done | `BASIC/models/`, `BASIC/data/`, `BASIC/utils/` |
| Training data pickle | ✅ Done | `stanford3d-pickle/pdb_xyz_data.pkl` (52MB) |
| Approach 1 Result.txt | ✅ Done | `mine/kalai/run1/Result_...txt` |
| Approach 1 submission.csv | ✅ Done | `mine/kalai/run1/submission_...csv` (for template coords) |
| MMseqs2 search working on Kaggle | ✅ Done (via Fork 1/Fork 2 notebooks) |

---

## Phase 1: Build ADV1 Code Locally (Day 1 — ~4-6 hours)

### New Files to Write (2 files, ~200 lines total)

**File 1: `BASIC/models/template_encoder.py` (~100 lines)**

What it does: Takes template coordinates (N,3) → computes distance matrix (N,N) → bins distances into 22 bins → Linear(22,16) → template features (N,N,16).

```
Input: template_coords (N, 3) from Approach 1 submission.csv
       confidence (scalar) from Result.txt e-value

Step 1: coords → pairwise distance matrix (N, N)
Step 2: distances → distance bins (N, N, 22) using fixed bin edges [0, 2, 4, 6, ..., 40, inf]
Step 3: bins → Linear(22, 16) → template features (N, N, 16)
Step 4: multiply by confidence mask (N, N, 1)

Output: template_features (N, N, 16)
```

Trainable parameters: just the Linear(22, 16) layer = 22×16 + 16 = 368 params.

**File 2: `BASIC/models/template_loader.py` (~100 lines)**

What it does: Reads Approach 1 outputs and provides template features per target.

For training: reads from locally saved Approach 1 submission.csv + Result.txt
For Kaggle inference: reads from MMseqs2 output generated in the same notebook run

### Files to Modify (3 files, ~20 lines each)

**Modify 1: `BASIC/models/distance_head.py`**
- Change: `pair_dim: int = 64` → `pair_dim: int = 80`
- That's it. The MLP architecture stays the same, just wider input.

**Modify 2: `BASIC/predict.py`**
- Add: load template_encoder and template_loader
- Add: before distance_head forward, concatenate template features to pairwise repr
- Change: `pairwise_repr (B,N,N,64)` → `concat(pairwise_repr, template_feat) (B,N,N,80)`

**Modify 3: `BASIC/train.py`**
- Same changes as predict.py for the forward pass
- Add: template_encoder parameters to optimizer
- Add: template data loading in the training loop

**Modify 4: `BASIC/config.yaml`**
- Add template section with paths and hyperparameters

### Weight Loading (Partial Transfer from BASIC)

```python
# BASIC distance_head first layer: Linear(64, 128)
# ADV1 distance_head first layer:  Linear(80, 128)
#
# Copy strategy:
#   ADV1_weight[:, 0:64]  = BASIC_weight[:, 0:64]   # copy existing
#   ADV1_weight[:, 64:80] = random_init               # new template channels
#   ADV1_bias             = BASIC_bias                 # copy directly
#
# All other layers: identical shapes, copy directly
```

### Training ADV1 Locally

```bash
cd BASIC
python train.py --config config_adv1.yaml --resume checkpoints/best_model.pt
```

- Epochs: 30-50 (warm-started, converges faster)
- GPU: Your local GPU
- Time: 2-4 hours depending on GPU
- Output: `checkpoints/adv1_best_model.pt`

### Verify Locally

```bash
python predict.py --config config_adv1.yaml --checkpoint checkpoints/adv1_best_model.pt \
    --test_csv ../../APPROACH1-TEMPLATE/test_sequences\ \(1\).csv \
    --template_csv ../../APPROACH1-TEMPLATE/mine/kalai/run1/submission_...csv \
    --output submission_adv1.csv
```

Compare submission_adv1.csv against BASIC's submission.csv — coordinates should be different, especially for targets where templates exist.

---

## Phase 2: Build Kaggle Notebook (Day 2-3 — ~4-6 hours)

The Kaggle notebook must do EVERYTHING in one run:

```
[Cell 1-5: Fork 1's MMseqs2 pipeline]
   test_sequences.csv → MMseqs2 search → template coordinates + e-values

[Cell 6: Load ADV1 model]
   Load RibonanzaNet.pt + adv1_best_model.pt from uploaded datasets

[Cell 7: Run ADV1 inference]
   For each test sequence:
     - Get pairwise features from RibonanzaNet
     - Get template features from MMseqs2 output (or zeros if no template)
     - Concatenate → distance head → distance matrix → MDS → 3D coords

[Cell 8: Post-processing]
   Read sample_submission.csv, map predictions, output corrected submission.csv
```

### What to Upload as Kaggle Datasets

| Dataset Name | Contents | Size |
|-------------|----------|------|
| `adv1-weights` | `adv1_best_model.pt` + `RibonanzaNet.pt` | ~44 MB |
| `adv1-code` | All .py files from BASIC/models/, BASIC/utils/, BASIC/data/ | ~50 KB |
| `ribonanzanet-repo` | The cloned RibonanzaNet/ folder (needed for backbone loading) | ~10 MB |

### The Base Notebook

Start from **Fork 1 (rhijudas)** — it already:
- Installs MMseqs2
- Runs the template search on test_sequences.csv
- Transfers coordinates from templates
- Handles Part 2 data correctly

Then APPEND cells that:
- Install PyTorch dependencies (`!pip install einops -q`)
- Load the ADV1 model from uploaded datasets
- Run neural inference using both pairwise features AND template features from the MMseqs2 search
- Write submission.csv

### Notebook Runtime Budget (8 hours max)

| Step | Estimated Time |
|------|---------------|
| MMseqs2 search + coordinate transfer | 15-30 min |
| ADV1 model loading | 1-2 min |
| ADV1 inference on all targets | 5-15 min (GPU) |
| Post-processing + write CSV | <1 min |
| **Total** | **~30-50 min** |

Well within the 8-hour limit.

---

## Phase 3: Submit and Iterate (Day 3-4)

1. Commit notebook with Internet OFF
2. Submit to competition
3. Get TM-score
4. If time remains: adjust noise/diversity settings, retrain with more epochs, resubmit

---

## Why This Should Score Better

| Target Type | BASIC Score | Fork 1 Score | ADV1 Expected |
|-------------|-------------|--------------|---------------|
| Strong template (6 targets) | ~0.03-0.05 | 0.56-0.92 | 0.50-0.90 (template features help) |
| Weak template (10 targets) | ~0.03-0.05 | 0.02-0.10 | 0.05-0.15 (some template signal + pairwise) |
| No template (12 targets) | ~0.03-0.05 | 0.00 (zeros) | 0.03-0.08 (pairwise only, same as BASIC) |
| **Hidden targets** | **0.00 (zeros)** | **depends** | **0.03-0.10 (model actually runs)** |

The key advantage over pure Fork 1/Fork 2: **ADV1 produces real predictions for ALL targets** (including hidden ones with no template), instead of zeros. Even mediocre predictions (TM 0.03-0.08) are better than zero.

The key advantage over pure BASIC: **template features dramatically improve the 6 strong-template targets** from ~0.05 to potentially 0.5+.

Estimated overall score: **0.15 - 0.25** (vs 0.092 BASIC, vs ~0.15-0.20 Fork 1/Fork 2)

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| ADV1 training doesn't converge | Low — warm-started from BASIC | Fall back to BASIC weights with template features as additive signal |
| Kaggle notebook exceeds 8 hours | Low — budget is ~50 min | MMseqs2 is the bottleneck, and Fork 1 runs in 15-30 min |
| Template features don't improve scores | Medium | No-template targets still get BASIC-level predictions |
| Upload/path issues on Kaggle | Medium — we've debugged these before | Apply same symlink and path fixes we've already learned |
| ADV1 code has bugs | Medium | Test thoroughly locally before uploading to Kaggle |

---

## Concrete Task Assignment

| Who | Task | Timeline |
|-----|------|----------|
| Claude + Team Lead | Write template_encoder.py, template_loader.py | Day 1, 2-3 hours |
| Claude + Team Lead | Modify distance_head.py, predict.py, train.py, config.yaml | Day 1, 1-2 hours |
| Team (local GPU) | Train ADV1 | Day 1 evening, 2-4 hours |
| Team | Verify predictions locally | Day 2 morning, 1 hour |
| Claude + Team Lead | Build Kaggle notebook (Fork 1 + ADV1 inference) | Day 2, 3-4 hours |
| Team | Upload datasets to Kaggle, test, debug | Day 2-3, 2-3 hours |
| Team | Commit and submit | Day 3 |
| Team | Iterate if time permits | Day 3-4 |

---

## PARALLEL TRACK (Do Not Block On This)

While ADV1 is being built:
1. Get Fork 2 submitted with Option B post-processing (1-2 hours)
2. Get Fork 1 status confirmed from Kalai and submitted if ready
3. These provide fallback scores while ADV1 is in development
