# HY-BAS-ADV1 Run 3 — Design Document

## Goal

Beat 0.287 (Fork 2's template-only score) and Run 2's score.

---

## The Core Problem We're Solving

**Fork 2 scored 0.287** by finding similar known RNA structures in PDB and
directly copying their experimental 3D coordinates. Simple, no neural network.

**ADV1 Run 1 scored 0.109** by running those same templates through a neural
network pipeline: template coords → encode to features → mix with backbone
features → predict distances → MDS reconstruct → 3D coords. Every step lost
information. The neural network produced worse coordinates than just copying
the template directly.

**Run 2 fixed bugs** (broken pickle parser, only 15 epochs) but the
fundamental architecture is the same: the NN takes good template coordinates,
destroys them through a lossy pipeline, and produces worse coordinates.

**The question for Run 3:** How do we make the neural network help instead
of hurt?

---

## Run 3's Two-Part Answer

### Part 1: Better Neural Network (Change 2 — Backbone Unfreezing)

The frozen backbone was trained for chemical mapping (DMS/2A3 reactivity) —
a 2D property. Its pairwise features are correlated with 3D proximity but
not optimized for it. Unfreezing the last 2 layers lets them adapt their
attention patterns to produce features specifically useful for distance
prediction — our actual objective.

This is DESIGN.md Option A. The RibonanzaNet paper showed fine-tuning
outperformed training from scratch (F1 0.89 vs 0.70).

With ~600 training structures and only 2 unfrozen layers (~2M params),
overfitting risk is manageable. Discriminative LR (1e-5 backbone vs 5e-5
head) prevents the pretrained layers from diverging.

### Part 2: Don't Throw Away Good Templates (Change 5 — Hybrid Inference)

**This is the most important insight in the entire project.**

Run 1 and Run 2 both made the same mistake: they send every target through
the neural network pipeline, even when the template coordinates are already
excellent. Fork 2 proved that directly using template coordinates scores
0.287. The NN pipeline can only match that if its distance predictions are
accurate enough that MDS reconstruction produces coordinates as good as the
original template — and Run 1 (0.109) proved it can't.

The hybrid approach: **if the template is good, use it directly. If the
template is weak, let the NN try.**

---

## Why Not Just Run Run 3 Without Hybrid and Add It Later?

We considered running Run 3 as pure NN (like Run 2 but with unfreezing) to
establish a baseline, then adding hybrid in Run 4. We rejected this because:

1. **Submissions are scarce.** 3 left today, 5 tomorrow (deadline). Every
   submission must maximize score, not gather diagnostics.

2. **We already know pure NN underperforms templates.** Run 1 scored 0.109.
   Run 2's result (pending) will confirm. Running another pure-NN experiment
   wastes a submission on something we already know.

3. **The hybrid can only help, never hurt.** For high-confidence targets, it
   guarantees at least Fork 2's quality. For low-confidence targets, it uses
   NN predictions (same as without hybrid). There is no scenario where adding
   hybrid makes the score worse.

4. **We can still measure unfreezing's effect** from the training logs. If
   Run 3's val_loss is lower than Run 2's (170.98), unfreezing helped —
   regardless of the final TM score.

---

## What Changed from Run 2 (5 modifications in 3 cells)

### Cell 1 — Bug fix
- Added `import os` to prevent potential NameError in wheel cleanup code.

### Cell 11 — Backbone Loading (2 changes for Change 2)

1. **Selective unfreezing**: Layers 0-6 frozen, layers 7-8 unfrozen.
   ~2M new trainable backbone params.

2. **Two feature extraction functions**:
   - `get_pairwise_features()` — inference mode, no gradients (Cell 14)
   - `get_pairwise_features_train()` — gradients through layers 7-8 only

### Cell 13 — Training Loop (2 changes for Change 2)

3. **BATCH_SIZE = 2** (was 4) — VRAM headroom for backbone gradients.

4. **Discriminative LR** via two optimizer param groups:
   - Head + template encoder: 5e-5 (same as Run 2)
   - Unfrozen backbone layers: 1e-5 (5x lower)

### Cell 14 — Inference (1 change for Change 5)

5. **Hybrid template/NN inference**:
   - `HYBRID_THRESHOLD = 0.3`
   - If `tmpl_conf > 0.3`: Slots 1-2 use template coordinates directly
     (Slot 1 = clean template with constraint refinement, Slot 2 = template
     with small Gaussian noise for diversity). Slots 3-5 use NN predictions.
   - If `tmpl_conf <= 0.3`: All 5 slots use NN predictions.
   - Template coordinates go through `adaptive_rna_constraints()` (same
     refinement Fork 2 uses) before being placed in the submission.

---

## What is Preserved from Run 2

Everything from Cells 1-10, 12, 13 (pickle parsing), 15 is identical:
- Change 1: TRAIN_EPOCHS = 30 ✅
- Change 3: Biopython wheel install ✅
- Change 4: Pickle parsing fix ✅
- Cell 5: Hardcoded tarunsathyab path ✅
- Template search, template encoder, distance head, MDS, Option B ✅

---

## How the 5 Diversity Slots Work

The competition requires 5 predicted structures per target. TM-score is
computed as the best of 5. This means even one good prediction in 5 slots
gives a high score.

### For high-confidence template targets (tmpl_conf > 0.3):
```
Slot 1: Template coords (clean, refined) — guaranteed Fork 2 quality
Slot 2: Template coords + small noise — slight diversity from Slot 1
Slot 3: NN prediction (no noise) — NN's best attempt
Slot 4: NN prediction (noise=0.3) — diverse NN attempt
Slot 5: NN prediction (noise=0.5) — more diverse NN attempt
```
**Scoring:** TM-score takes the best of 5. If NN is worse than template,
Slots 1-2 dominate. If NN is better, Slots 3-5 dominate. Either way, we
get the best available quality.

### For low-confidence template targets (tmpl_conf <= 0.3):
```
Slots 1-5: All NN predictions with varying noise levels
```
**Rationale:** Weak templates may be misleading. The NN, which was trained
on 661 real structures, might produce better coordinates than a distant,
poorly-aligned template.

---

## Template Confidence Scores (from Run 2 log)

| Target | Best Template | Score | Hybrid Decision |
|--------|--------------|-------|----------------|
| 8ZNQ | 1PBR | 0.307 | → Template slots 1-2 |
| 9CFN | 4FRN | 0.357 | → Template slots 1-2 |
| 9E73 | 9E73 | 0.281 | → NN all 5 |
| 9E74 | 3J6X | 0.262 | → NN all 5 |
| 9E9N | 7PKQ_2 | 0.288 | → NN all 5 |
| 9E9Q | 8UYS | 1.175 | → Template slots 1-2 |
| 9G4J | 7VNV_A | 0.256 | → NN all 5 |
| 9G4K | 7S36 | 0.289 | → NN all 5 |
| 9G4L | 7Q4K_D2 | 0.284 | → NN all 5 |
| 9G4M | 7OHX_2 | 0.151 | → NN all 5 |
| 9G4R | 5HR6_D | 0.359 | → Template slots 1-2 |
| 9LEC | 2AHT | 0.487 | → Template slots 1-2 |
| 9LEL | 9C0I_R | 0.263 | → NN all 5 |
| 9LMF | 9LMF_A | 1.372 | → Template slots 1-2 |
| 9MME | 6CHR_A | 0.274 | → NN all 5 |
| 9I9W | 3CGR | 0.650 | → Template slots 1-2 |
| 9IPK | 1F6Z | 0.446 | → Template slots 1-2 |
| 9NYB | 1NYB | 0.826 | → Template slots 1-2 |
| 9QDM | 7XHT | 0.327 | → Template slots 1-2 |
| 9RJG | 5NRL_I | 0.330 | → Template slots 1-2 |
| 9WHV | 8EYU | 0.324 | → Template slots 1-2 |
| 9WRD | 7UO2_B | 0.289 | → NN all 5 |
| 9ELY | 9ELY | 0.742 | → Template slots 1-2 |
| 9G4I | 9G4I | 1.402 | → Template slots 1-2 |
| 9LEK | 1Y0Q_A | 0.549 | → Template slots 1-2 |
| 9EBP | 4V8P_B3 | 0.301 | → Template slots 1-2 |
| 9EBQ | 8CBK | 0.254 | → NN all 5 |
| 9MMD | 1JGO | 0.258 | → NN all 5 |

**~16 targets use template directly, ~12 use NN.** The 16 template targets
are the ones that drove Fork 2's 0.287 score. By preserving their template
coords in Slots 1-2, we guarantee those targets don't regress.

---

## Score Expectations

| Scenario | Expected Score | Reasoning |
|----------|---------------|-----------|
| Floor (hybrid guarantee) | ~0.287 | Template slots match Fork 2 for strong targets |
| NN helps on weak targets | 0.30-0.40 | NN adds value where templates are weak |
| NN + unfreezing helps significantly | 0.35-0.45 | Adapted backbone features produce good distances |
| NN predictions beat templates on some targets | 0.40+ | Best case — NN slots 3-5 outscore template slots |

The hybrid guarantees the floor can't go below ~0.25-0.287. The ceiling
depends on how well the unfrozen backbone learned distance prediction.

---

## Runtime Estimate

```
Cells 1-9:   Setup + template search        ~25 min (same as Run 2)
Cell 13:     Training (30 epochs × ~3 min)   ~90 min
Cell 14:     Inference (hybrid)              ~15 min (same — NN runs either way)
Cell 15:     Post-processing                 ~10 sec
TOTAL:                                       ~2.5 hours
```

---

## Datasets (same as Run 2, no new uploads)

Same 6 datasets. No changes.

---

## File Locations

| Item | Path |
|------|------|
| Run 3 notebook source | `APPROACH2-RIBBOZANET/HY-BAS-ADV1/kaggle/hy_bas_adv1_run3_notebook.py` |
| Run 3 Kaggle artifacts | `TRY1/HY-BAS-ADV1-RUN3-REMOTE/` |
| Run 3 design doc | `TRY1/HY-BAS-ADV1-RUN3-REMOTE/RUN3_DESIGN.md` (this file) |

---

## Decision Log

| Decision | Chosen | Rejected | Why |
|----------|--------|----------|-----|
| Add hybrid to Run 3 vs Run 4 | Run 3 | Separate Run 4 | Submissions scarce; hybrid only helps; Run 3 baseline is scientifically interesting but competitively useless |
| Hybrid threshold | 0.3 | 0.5 or 0.1 | 0.3 splits the 28 targets roughly 16/12; scores below 0.3 are distant matches unlikely to have good coords |
| Slots 1-2 template, 3-5 NN | 2+3 split | All 5 template or all 5 NN | Best-of-5 scoring means even one good slot wins; this gives both template and NN a chance |
| Run pure NN Run 3 in parallel | No | Yes | Burns extra submission; we already know NN < template from Run 1 |
