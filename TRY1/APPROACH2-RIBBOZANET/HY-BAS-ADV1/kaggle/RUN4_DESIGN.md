# HY-BAS-ADV1 Run 4 — Design Document

## Goal

Improve on Run 3 OptB's score by adding MSA (evolutionary) features.

---

## What is MSA? (Simple Explanation)

MSA = Multiple Sequence Alignment. Imagine you have an RNA sequence from
a human. You find the "same" RNA in a mouse, a frog, a fish, and a fly.
Their sequences are similar but not identical — they've each mutated
slightly over millions of years of evolution.

If position 15 and position 42 always mutate TOGETHER across species,
they're probably physically touching in 3D. Why? Because if only one
changed, the structure would break and the organism would die. But if
both change in a compatible way, the structure stays intact. This
co-mutation pattern is called "covariation" and it's one of the strongest
signals for predicting which parts of an RNA are close in 3D space.

From the alignment of cousin sequences, we extract three signals:
1. **Covariation** (pairwise): Do positions i and j mutate together?
2. **Conservation** (per-position): Does this position never change?
3. **Gap frequency** (per-position): Is this position often missing?

These become 8 feature channels fed to the neural network alongside the
existing 80 channels (64 backbone + 16 template).

---

## The Core Insight

Run 3 OptB gives the neural network two sources of information:
- **Backbone features (64 channels):** What the RibonanzaNet model learned
  about this specific RNA sequence from its chemistry.
- **Template features (16 channels):** What one similar known structure
  looks like in 3D.

Both are valuable, but neither directly answers the question: "Which pairs
of positions in this RNA are physically close to each other?"

MSA covariation answers this question directly, using evidence from
millions of years of evolution. AlphaFold2, RhoFold+, and RNAPro all use
MSA features because they provide information that sequence features alone
cannot capture. Specifically:

- **Backbone features** know about local chemistry (base stacking, sugar
  pucker) but can't detect long-range 3D contacts from sequence alone.
- **Template features** know about one similar structure, but if the
  template is distant (weak match), the 3D information is noisy.
- **MSA covariation** detects long-range contacts even when no close
  template exists, because co-evolution reveals 3D proximity regardless
  of how different the sequences are.

This is why MSA is most valuable on weak-template targets (~12 of 28).

---

## What Changed from Run 3 Option B

### Change 7: MSA Features (the ONLY change)

Everything from Run 3 OptB is preserved:
- Change 1: TRAIN_EPOCHS = 30 ✓
- Change 2: Unfreeze backbone layers 7,8 + discriminative LR ✓
- Change 3: Biopython offline wheel install ✓
- Change 4: Fixed pickle parsing ✓
- Change 5: Hybrid inference (template slots + NN slots) ✓
- Change 6: Template-seeded refinement (NN starts from template) ✓

Run 4 adds Change 7 across 6 cells:

| Cell | What Changed | Lines Added/Modified |
|------|-------------|---------------------|
| Cell 0 | Added `MSA_TOP_N = 20`, `MSA_DIM = 8` config | 3 lines |
| Cell 3 | Auto-discovers `adv1_best_model.pt` as `RUN3_CHECKPOINT` | 5 lines |
| Cell 9 | Template search collects top-20 (was top-1) | 5 lines |
| Cell 9.5 | **NEW** — `compute_msa_features()` + pre-compute for all targets | ~150 lines |
| Cell 10 | `DistanceMatrixHead` default `pair_dim=88` (was 80) | 1 line |
| Cell 12 | `PAIR_DIM=88`, warm-start from Run 3 (80→88 expansion) | 20 lines |
| Cell 13 | MSA masking + `make_training_msa_proxy()` + concat 88-dim | 40 lines |
| Cell 14 | Load MSA features + concat 88-dim during inference | 15 lines |

Cells 1, 2, 4, 5, 6, 7, 8, 11, 15 are **UNCHANGED**.

---

## The 8 MSA Feature Channels

| # | Channel | Type | What It Captures |
|---|---------|------|-----------------|
| 1 | Covariation(i,j) | Pairwise (N×N) | Mutual information — do positions i,j co-mutate? |
| 2 | APC-corrected covariation | Pairwise (N×N) | Same but with background noise removed |
| 3 | Conservation(i) | Per-position → broadcast | Does position i never change? (low entropy) |
| 4 | Conservation(j) | Per-position → broadcast | Same for position j |
| 5 | Conservation(i) × Conservation(j) | Product → pairwise | Are both positions conserved? |
| 6 | Gap frequency(i) | Per-position → broadcast | Is position i often missing? (flexible loop) |
| 7 | Gap frequency(j) | Per-position → broadcast | Same for position j |
| 8 | Neff | Scalar → broadcast | Number of effective sequences (MSA quality) |

All 8 channels are (N, N) matrices stored as a (N, N, 8) tensor,
concatenated with backbone (N,N,64) and template (N,N,16) to produce
the combined (N,N,88) input to the distance head.

---

## How MSA Features Are Built (Step by Step)

### During Inference (Cell 9 + Cell 9.5)

1. **Cell 9:** For each test target, `find_similar_sequences()` now returns
   top-20 similar sequences (was top-1). Best hit is still used as template.
   All 20 are stored in `msa_hits_per_target`.

2. **Cell 9.5:** For each test target, `compute_msa_features()`:
   a. Aligns all 20 cousin sequences to the query using pairwise alignment
   b. Builds an alignment matrix: rows=sequences, columns=positions
   c. Computes per-position frequencies (A/C/G/U/gap at each column)
   d. Computes conservation (Shannon entropy) and gap frequency per position
   e. Computes pairwise mutual information (covariation) for all (i,j) pairs
   f. Applies APC correction to remove background noise from covariation
   g. Assembles all 8 channels into (N, N, 8) feature tensor

### During Training (Cell 13)

We don't have real MSA for the 734 training structures (that would require
running the template search on every training sequence — too slow). Instead:

- **50% of the time:** MSA channels are set to zero (masked). This teaches
  the model to work without MSA, just like template masking teaches it to
  work without templates.
- **50% of the time:** A "training MSA proxy" is generated using Watson-Crick
  base-pair potential (A-U=1, G-C=1, G-U=0.5). This is a rough approximation
  of covariation — not as good as real MSA, but it teaches the model that
  the MSA channels contain pairwise structural information.

This proxy approach works because:
- The model learns that channels 1-2 (covariation) correlate with 3D proximity
- The model learns to use MSA when available but not crash when it's zero
- During test-time inference, the REAL MSA features (from 20 aligned sequences)
  provide much stronger signal than the training proxy, so the model gets
  better input than it was trained on — which is fine (it generalizes up)

---

## Warm-Start Strategy (Cell 12)

Run 4's distance head has pair_dim=88 (was 80 in Run 3).

### Preferred: Warm-start from Run 3 OptB checkpoint

```
Run 3 checkpoint contains:
  distance_head:       mlp.0.weight shape (128, 80)
  template_encoder:    projection.weight shape (16, 22)
  backbone layers 7-8: ~2M params

Run 4 creates:
  distance_head:       mlp.0.weight shape (128, 88)

Warm-start:
  Columns 0-79:  copied from Run 3's (128, 80) weights
  Columns 80-87: initialized to zero (8 new MSA channels)
  All other layers: copied directly (shapes unchanged)
  Template encoder: copied directly
  Backbone layers 7-8: copied directly
```

This means Run 4 starts exactly where Run 3 left off. The model already
knows how to use backbone features (cols 0-63) and template features
(cols 64-79). It only needs to learn what to do with the 8 new MSA
channels (cols 80-87). This should converge fast — the existing 80
channels are already trained, and the new 8 channels start at zero
(contributing nothing initially) and gradually learn.

### Fallback: Warm-start from BASIC checkpoint

If Run 3's checkpoint is not available:
  BASIC has mlp.0.weight shape (128, 64)
  Expanded to (128, 88): cols 0-63 copied, cols 64-87 zeroed.
  No template encoder or backbone layers to load.

This works but loses all of Run 3's training progress.

### Required Kaggle Dataset

Upload Run 3 OptB's `adv1_best_model.pt` (8.89 MB) as a new Kaggle dataset
(e.g., "adv1-run3-checkpoint"). The notebook auto-discovers it by filename.

---

## Adjustable Parameter (Based on Run 3 Results)

If Run 3 OptB's training log shows val_loss didn't improve below 171:

```python
# In Cell 13, change:
BACKBONE_LR = 1e-5    # current value
# To:
BACKBONE_LR = 5e-6    # halved — more conservative unfreezing
```

This is the ONLY adjustment needed based on Run 3 results. Everything else
is independent.

---

## Additional Improvement: NN Slot Diversity (Cell 14)

Run 3 OptB's draft submission showed slots 3-5 were too similar
(0.01-0.03 Å apart). The noise settings weren't creating meaningfully
different structures. Run 4 increases the noise scale:

```
Run 3 OptB:                     Run 4:
  noise: 0.0                      noise: 0.0  (unchanged)
  noise: 0.3                      noise: 0.5  (increased)
  noise: 0.5                      noise: 1.0  (increased)
  noise: 0.7                      noise: 1.5  (increased)
  noise: 0.0                      noise: 0.0  (unchanged)
```

More noise = more diverse structures = higher chance that at least one
of the 5 slots is close to the real answer.

---

## Does MSA Help Without IPA?

Yes. MSA improves the INPUT features, not the OUTPUT method:
- MSA → better distance predictions from the distance head
- Better distance predictions → template-seeded refinement starts from
  good template coords AND refines toward better target distances
- The output method (template-seeded refinement vs MDS vs IPA) is
  independent of the input features

If IPA is added later (Run 5+), MSA features carry over — IPA would
operate on the same 88-channel pairwise representation. AlphaFold2
uses both MSA and IPA together by design.

---

## Score Expectations

| Scenario | Score | Reasoning |
|----------|-------|-----------|
| MSA provides strong covariation signal | 0.35-0.45 | NN distance predictions improve significantly on weak-template targets |
| MSA provides modest signal | 0.30-0.40 | Some improvement on weak targets, strong targets unchanged |
| MSA proxy doesn't train well | 0.28-0.35 | Model doesn't learn to use MSA channels effectively from proxy data |
| Run 3 baseline (for comparison) | 0.28-0.40 | Template + NN without MSA |

MSA helps most on weak-template targets (~12 of 28) where the NN is
doing all the work. For strong-template targets (~16 of 28), template
slots 1-2 already carry the score — MSA provides marginal improvement
to NN slots 3-5.

---

## Runtime Estimate

```
Cells 1-8:    Setup + data loading + template functions  ~20 min
Cell 9:       Template search (top-20 per target)        ~8-10 min (was ~5 min)
Cell 9.5:     MSA feature computation                    ~2-5 min (NEW)
Cell 10-12:   Model setup + warm-start                   ~1 min
Cell 13:      Training (30 epochs, pair_dim=88)           ~50-65 min
Cell 14:      Inference (88-dim features)                 ~5-10 min
Cell 15:      Post-processing                            ~10 sec
TOTAL:                                                   ~90-110 min
```

Well within Kaggle's 9-hour GPU limit.

---

## Datasets Required

Same 6 datasets as Run 3, plus one new upload:

| # | Dataset | Contents | New? |
|---|---------|----------|------|
| 1 | Stanford RNA 3D Folding Part 2 | Competition data | No |
| 2 | rna-cif-to-csv | rna_sequences.csv, rna_coordinates.csv | No |
| 3 | adv1-weights | RibonanzaNet.pt + best_model.pt | No |
| 4 | adv1-training-data | pdb_xyz_data.pkl | No |
| 5 | ribonanzanet-repo | RibonanzaNet/ folder | No |
| 6 | biopython-wheel | .whl files for offline install | No |
| **7** | **adv1-run3-checkpoint** | **adv1_best_model.pt (8.89 MB from Run 3)** | **YES** |

If dataset 7 is not attached, the notebook falls back to warm-starting
from BASIC's best_model.pt (dataset 3). This works but loses Run 3's
training progress.

---

## File Locations

| Item | Path |
|------|------|
| Run 4 notebook source | `HY-BAS-ADV1/kaggle/hy_bas_adv1_run4_notebook.py` |
| Run 4 design doc | `HY-BAS-ADV1-RUN4-REMOTE/RUN4_DESIGN.md` (this file) |
| Run 3 OptB notebook source | `HY-BAS-ADV1/kaggle/hy_bas_adv1_run3_optb_notebook.py` |
| Run 3 OptB checkpoint | `HY-BAS-ADV1-RUN3-REMOTE/adv1_best_model.pt` |

---

## Decision Log

| Decision | Chosen | Rejected | Why |
|----------|--------|----------|-----|
| MSA source | Build from template search hits (top-20) | Use competition MSA files (.a3m) | Our template search already runs; competition MSA files may not exist for Part 2; self-contained pipeline |
| MSA training | Proxy (base-pair potential) + 50% masking | Skip MSA during training entirely | Model needs to see non-zero MSA channels during training to learn to use them; proxy is cheap to compute |
| pair_dim | 88 (64+16+8) | 96 (add more channels) or 80 (no MSA) | 8 channels capture the key signals (covariation, conservation, gaps); more channels = more VRAM + slower training with diminishing returns |
| Warm-start source | Run 3 OptB checkpoint (preferred) | Always from BASIC | Run 3's checkpoint has 30 epochs of trained backbone + distance head. Starting from BASIC loses that. |
| NN slot noise | Increased (0.5, 1.0, 1.5) | Same as Run 3 (0.3, 0.5, 0.7) | Run 3 draft showed slots 3-5 were nearly identical. More noise = more diverse structures. |

---

## Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| MSA computation is too slow (>30 min) | Delays training start | Low | Only 28 targets × 20 sequences each; MI computation is O(N² × K) but N≤512, K≤20 |
| Training MSA proxy doesn't help | MSA channels ignored by model | Medium | 50% masking means model doesn't depend on MSA; real MSA at test time is stronger than proxy anyway |
| pair_dim=88 causes OOM on T4 | Training crash | Low | Only 8 extra channels; BATCH_SIZE already 2; marginal VRAM increase (~5-10%) |
| Run 3 checkpoint not uploaded | Falls back to BASIC warm-start | Medium | Notebook handles gracefully with fallback logic; just loses Run 3's training progress |
| Top-20 search takes too long | Template search phase doubles | Low | kmer prefilter + top 100 alignment is the bottleneck, not changing top_n from 1 to 20 (already computed) |

---

## What Run 4 Does NOT Do

- Does NOT add IPA (that's Run 5+ — requires replacing the entire
  distance→coordinate pipeline)
- Does NOT change model architecture beyond pair_dim expansion
- Does NOT use competition-provided MSA files (builds MSA from
  template search hits instead — self-contained)
- Does NOT use distillation from RNAPro (requires 24+ GB GPU we don't have)
- Does NOT change the hybrid slot assignment logic
- Does NOT change the post-processing pipeline
