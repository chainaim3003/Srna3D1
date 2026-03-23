# HY-BAS-ADV1 Run 2 — Design Document

## Goal

Beat the current best score of **0.287** (Run 1, template-only TBM approach).

## What Run 1 Actually Was

Run 1 (the notebook `hyb-bas-adv1-template-with-neural-net`) combined:
- Fork 2's MMseqs2 template search pipeline
- A frozen RibonanzaNet backbone (100M params, no gradients)
- A distance head MLP (~25K trainable params) warm-started from BASIC
- A template encoder (368 trainable params)
- Training for only **15 epochs** on Kaggle GPU

Run 1's Kaggle score: **NOT YET CONFIRMED** — but we know the 0.287 came
from the pure template-matching Fork 2 submission, not from ADV1's neural
network. Run 1 may have scored similarly or worse due to the issues below.

## What Went Wrong in Run 1

Three concrete problems were identified:

### Problem 1: Undertrained (15 epochs)
BASIC trained for 50 epochs locally. Run 1 used only 15 epochs on Kaggle
because the epoch count was set conservatively to avoid timeout. The model
almost certainly hadn't converged — it needed at least 30 epochs for the
distance head to learn meaningful patterns.

**Evidence:** BASIC's local training showed loss still improving at epoch 30.
At epoch 13, loss was ~661 (roughly 25 Angstrom average error). By epoch 50,
it was significantly lower.

### Problem 2: Biopython install failed with Internet OFF
Cell 1 had `!pip install biopython -q`. During the committed run (Internet
OFF), this line either:
- Failed silently and biopython was already pre-installed (lucky), or
- Failed and caused downstream `from Bio import pairwise2` to crash

This is unnecessary because biopython is **pre-installed in Kaggle's Docker
image** (confirmed: `Bio.__version__` prints `1.86` without any pip install).

**Evidence:** The Kaggle Docker image Dockerfile historically includes
`pip install biopython`. When tested interactively, `import Bio` works
without any install step.

### Problem 3: Pickle data parsing may have been broken
Cell 13's training data loader initially had a bug: it treated the pickle
as a list of dicts when it's actually a dict of parallel lists. The fix
(using `raw_data['sequence']` and `raw_data['xyz']` with `sugar_ring[0]`
for C1' coordinates) was developed mid-session, but it's unclear whether
the corrected version made it into the committed notebook.

If the old broken code ran, the model trained on zero or garbage structures
and learned nothing.

---

## Run 2 Changes (1+3+4 only)

Run 2 applies exactly **three targeted fixes** with no architectural changes.
The backbone stays frozen. The model structure is identical to Run 1. This
is a low-risk, fast-turnaround run to establish a new baseline.

### Change 1: TRAIN_EPOCHS = 30 (was 15)

**What:** In Cell 13, change `TRAIN_EPOCHS = 15` to `TRAIN_EPOCHS = 30`.

**Why it helps:** Doubles the training time, giving the distance head and
template encoder 2x more gradient updates to learn. BASIC needed 50 epochs
to converge; 15 was clearly insufficient.

**Time impact:** Training goes from ~16 min to ~32 min. Total notebook
runtime goes from ~55 min to ~65 min. Well within Kaggle's 9-hour limit.

**Risk:** Near zero. The cosine LR scheduler adapts automatically to the
new epoch count. Best model is saved whenever validation loss improves, so
even if epoch 30 is slightly overfit, the saved checkpoint is the best one.

### Change 3: Remove biopython pip install

**What:** Replace Cell 1's `!pip install biopython -q` with:
```python
import Bio
print(f"Biopython {Bio.__version__} pre-installed")
```

**Why it helps:** Eliminates the only cell that requires Internet access.
The current Cell 1 fails or is wasted effort because biopython is already
pre-installed in Kaggle's Docker image (confirmed version 1.86).

**Time impact:** Saves ~10 seconds. More importantly, removes a failure
point during Internet-OFF commits.

**Risk:** Zero. This is removing unnecessary code, not adding anything.

### Change 4: Verify pickle parsing is correct

**What:** Ensure Cell 13's training data loading uses the corrected logic:
```python
sequences = raw_data['sequence']
xyz_list = raw_data['xyz']
# For each structure:
#   sugar_ring = residue_atoms['sugar_ring']
#   c1_prime = sugar_ring[0]  # C1' coordinate
```

**Why it helps:** If Run 1 had the broken parser, the model trained on
nothing. With the correct parser, it trains on ~600 real RNA structures.

**Time impact:** None.

**Risk:** Zero. This is a bugfix, not a new feature.

---

## Why We Expect Run 2 to Beat 0.287

### Reasoning

The 0.287 score came from **pure template matching** (Fork 2, jaejohn's
TBM-only approach). That approach has no neural network — it simply finds
the closest known structure in PDB and copies its coordinates.

Run 2 adds a **trained neural network** on top of template matching:
- The frozen RibonanzaNet backbone extracts 64-dimensional pairwise
  features from the RNA sequence. These features encode evolutionary and
  structural information learned from millions of RNA sequences.
- The distance head MLP converts these features into predicted inter-
  residue distances.
- The template encoder adds 16 channels of template information when
  a good template exists, and the model falls back to pairwise-only
  features when no template is found.

For the **~6 targets with strong templates**, the neural network should
perform at least as well as pure template matching (it has the same
template information plus learned sequence features).

For the **~12 targets with no or weak templates**, the neural network
provides predictions where pure template matching returns zeros. This is
where the score improvement comes from — the 0.287 is dragged down by
targets where template matching found nothing useful.

### Conservative estimate

| Category | # Targets | Run 1 (0.287 basis) | Run 2 expected |
|----------|-----------|--------------------:|---------------:|
| Strong template | ~6 | ~0.50 TM avg | ~0.50 (same) |
| Weak template | ~10 | ~0.15 TM avg | ~0.20 (NN helps) |
| No template | ~12 | ~0.05 TM avg | ~0.10 (NN from scratch) |
| **Weighted avg** | **28** | **~0.20** | **~0.25** |

This is conservative because:
- We assume the NN only marginally helps. In practice, even a mediocre
  distance prediction + MDS reconstruction beats zero-coordinate fills.
- The actual 0.287 already performs well on some targets. If the NN
  doesn't regress on those AND improves on the bad ones, the overall
  score rises.

### The key insight

0.287 uses template matching for SOME targets and fills zeros for others.
Run 2 uses template matching + neural network for ALL targets. Even if the
neural network predictions are poor (TM ~0.05-0.10), they're better than
the zeros that dragged down the 0.287 score.

**Expected score: 0.30 – 0.40**

If the pickle parsing was broken in Run 1 (Problem 3), then Run 2 is the
first time the neural network actually trains on real data — in which case
the improvement could be much larger.

---

## What Run 2 Does NOT Do

- Does NOT unfreeze backbone layers (that's Change 2, saved for Run 3)
- Does NOT change model architecture
- Does NOT change template search pipeline
- Does NOT change post-processing
- Does NOT change datasets or inputs

This is purely: fix bugs + train longer. Maximum confidence, minimum risk.

---

## File Locations

| Item | Path |
|------|------|
| Run 2 notebook source | `APPROACH2-RIBBOZANET/HY-BAS-ADV1/kaggle/hy_bas_adv1_run2_notebook.py` |
| Run 1 notebook source | `APPROACH2-RIBBOZANET/HY-BAS-ADV1/kaggle/hy_bas_adv1_run1_notebook.py` |
| Run 2 Kaggle artifacts | `TRY1/HY-BAS-ADV1-RUN2-REMOTE/` |
| Run 1 Kaggle artifacts | `TRY1/HY-BAS-ADV1-RUN1-REMOTE/` (rename pending) |
| Shared model code | `APPROACH2-RIBBOZANET/HY-BAS-ADV1/models/` |
| Shared config | `APPROACH2-RIBBOZANET/HY-BAS-ADV1/config_adv1.yaml` |

---

## Kaggle Notebook Name

Create a NEW notebook on Kaggle: **"HY-BAS-ADV1-Run2"**

Do NOT edit the Run 1 notebook. Keep it as-is for reference and rollback.

## Kaggle Inputs (same as Run 1)

1. Stanford RNA 3D Folding Part 2 (competition)
2. rna-cif-to-csv (jaejohn)
3. adv1-weights (your dataset: RibonanzaNet.pt + best_model.pt)
4. adv1-training-data (your dataset: pdb_xyz_data.pkl)
5. ribonanzanet-repo (your dataset: RibonanzaNet/ folder)
6. biopython-wheel (your dataset — kept as safety net, but Cell 1 won't use it)
7. extended-rna (jaejohn, optional)

## Estimated Timeline

| Phase | Time |
|-------|------|
| Create notebook + paste cells | 15 min |
| Draft run (Internet ON) | ~65 min |
| Verify output | 5 min |
| Commit (Internet OFF) | ~65 min |
| Score appears | 10-60 min |
| **Total wall clock** | **~2.5 hours** |

---

## Success Criteria

| Outcome | Score | Verdict |
|---------|-------|---------|
| > 0.35 | Excellent | NN clearly helping, proceed to Run 3 with unfreezing |
| 0.29 – 0.35 | Good | Marginal improvement, Run 3 unfreezing needed |
| 0.287 | Neutral | NN not helping, but at least bugs are fixed |
| < 0.287 | Problem | Investigate — NN may be producing worse predictions |

---

## Next Step After Run 2

If Run 2 beats 0.287: create Run 3 notebook adding **Change 2** (unfreeze
last 2 backbone layers with discriminative learning rate 1e-5 for backbone,
5e-5 for head). This is the high-impact architectural change.

If Run 2 does NOT beat 0.287: investigate the training logs from the draft
run to understand why the neural network isn't helping, before attempting
unfreezing.
