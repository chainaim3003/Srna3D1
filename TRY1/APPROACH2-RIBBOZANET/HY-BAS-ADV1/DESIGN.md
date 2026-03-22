# HY-BAS-ADV1: Hybrid Design Document
## Holistic Design for Kaggle Submission

## Location: TRY1/APPROACH2-RIBBOZANET/HY-BAS-ADV1/

---

## CRITICAL CORRECTION: BASIC's Kaggle Score (0.092) Was NOT Neural Network Inference

What BASIC's Kaggle notebook (RNA-Basic, Version 6) actually did:

```
1. Read a pre-made submission_fixed.csv from an uploaded Kaggle dataset
2. Read sample_submission.csv from competition data
3. Map our pre-made predictions to sample IDs
4. Write submission.csv
```

It did NOT:
- Load RibonanzaNet.pt (the 43MB backbone)
- Load best_model.pt (the trained distance head)
- Run PyTorch inference on any sequences
- Import torch, or any model code

This means: **We have NEVER proven that our neural network can load and
run on Kaggle.** This is a risk that must be addressed FIRST.

---

## TRAINING vs INFERENCE — Explained Simply

Think of it like studying for an exam vs taking the exam.

### TRAINING (Already Done — Happened Locally on Our GPU)
- **What:** We showed the model 661 RNA structures from PDB.
  For each one, we said: "Here's the sequence ACGUU..., and here are
  the true distances between every pair of nucleotides."
  The model learned to predict distances from sequences.
- **Output:** best_model.pt (312KB file) — this is the "learned brain"
  of the distance prediction network
- **Where:** Ran on our local GPU. Took ~2-4 hours. 50 epochs.
- **Do we need to redo this?** YES, partially. For ADV1, we need to
  retrain with template features (80 channels instead of 64).
  But we warm-start from best_model.pt, so it converges faster (~30 epochs).

### INFERENCE (Needs to Happen on Kaggle)
- **What:** Given a NEW sequence we've never seen before, predict its
  3D structure. No learning happens. We just use the trained "brain"
  to make predictions.
- **Steps:**
  1. Take sequence "ACGUU..."
  2. Feed it through frozen RibonanzaNet backbone -> pairwise features
  3. Feed pairwise features through trained distance head -> distance matrix
  4. Convert distance matrix to 3D coordinates (MDS + refinement)
- **Where:** Must happen on Kaggle (Code Competition)
- **Time per target:** ~1-5 seconds
- **This is what we've NEVER tested on Kaggle**

### WHY THE DISTINCTION MATTERS

Training = expensive, one-time, local GPU, produces weights file
Inference = cheap, per-sequence, must work on Kaggle, uses weights file

For ADV1, we need:
- ~2-4 hours of LOCAL training (retrain with template features)
- ~5-30 minutes of KAGGLE inference (run the trained model on test data)

---

## HOLISTIC HYBRID DESIGN

### What HY-BAS-ADV1 Does (The Full Pipeline on Kaggle)

```
PHASE 1: Template Search (from Fork 1 / rhijudas)
  test_sequences.csv -> MMseqs2 Search -> template coords + e-values

PHASE 2: Neural Inference (from BASIC + ADV1)
  For each target:
    sequence -> RibonanzaNet backbone -> pairwise features (N,N,64)
    template coords -> template encoder -> template features (N,N,16)
    CONCATENATE -> (N,N,80)
    -> Distance Head (ADV1) -> distance matrix (N,N)
    -> MDS + Refinement -> 3D coordinates (N,3) x 5

PHASE 3: Post-Processing (Option B pattern)
  Read sample_submission.csv -> map predictions -> fill zeros -> submission.csv
```

### Training Required (LOCAL, Before Kaggle)

ADV1 needs a RETRAINED distance head because:
- BASIC distance head first layer: Linear(64, 128)
- ADV1 distance head first layer:  Linear(80, 128) — 16 extra template channels

Warm-start: Copy BASIC weights for columns 0-63, random-init columns 64-79.
Estimated training: 30 epochs, 1-3 hours on local GPU.

---

## WHAT HAPPENS FOR EACH TARGET TYPE

| Target Type | Phase 1 (MMseqs2) | Phase 2 (ADV1) | Expected TM |
|-------------|-------------------|----------------|-------------|
| Strong template (6 targets) | Finds close homologs | Template (N,N,16) + pairwise (N,N,64) | 0.40-0.80 |
| Weak template (10 targets) | Finds distant matches | Partial template + pairwise | 0.05-0.20 |
| No template (12+ targets) | No hits | Template = zeros, pairwise only | 0.03-0.10 |
| Hidden targets | Depends on PDB | Same as above | Varies |

---

## RISK REGISTER

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| RibonanzaNet won't load on Kaggle | BLOCKS everything | Medium | Pre-Step 1 test |
| ADV1 training doesn't converge | Score = BASIC level | Low | Warm-started |
| Kaggle notebook times out | No submission | Low | Budget 55min/8hr |
| MMseqs2 install fails on Kaggle | No templates | Low | Fork 1 proved it works |
| Template encoder bugs | Wrong features | Medium | Test locally first |
