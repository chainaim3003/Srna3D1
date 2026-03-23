# Stanford RNA 3D Folding Part 2 — Project Summary
## Team Ribbozle3D | Competition Deadline: March 25, 2026

---

## 1. Competition Overview

- **Competition:** Stanford RNA 3D Folding Part 2 (Kaggle Code Competition)
- **Task:** Predict 3D coordinates (C1' atom) for 28 RNA sequences
- **Submission:** 5 predicted structures per target. Scored as best-of-5 (TM-score)
- **Constraint:** Notebook runs with Internet OFF. All code + data must be self-contained.
- **Leaderboard reference:** #1 AyPy (0.499), #2 Brad & Jack (0.485), Benchmark oracle (0.554)

---

## 2. Score History — All Runs

| Run | Score | Method | One-liner |
|-----|-------|--------|-----------|
| BASIC | 0.092 | NN only (pre-made CSV) | Uploaded pre-computed submission. Never ran NN on Kaggle. |
| Fork 1 (RJ) | not scored | Template only | MMseqs2 search. Never submitted separately. |
| Fork 2 (JJ) | 0.172 | Template only | jaejohn's TBM notebook, 5.7K structure database. |
| Fork 2 + OptB | 0.287 | Template only | Same + 24K extended database + Option B ID post-processing. **Current best.** |
| Run 1 (ADV1) | 0.109 | NN + template | Broken pickle parser + 15 epochs + MDS. Trained on nothing. |
| Run 2A | 0.109 | NN + template | Fixed pickle (734 structs), 30 epochs, frozen backbone. MDS destroyed quality. |
| Run 2B | 0.107 | NN + template | Duplicate of 2A. Confirms MDS bottleneck is systematic. |
| Run 3 OptB | pending | Hybrid + NN | Unfreeze backbone + hybrid slots + template-seeded refinement. Committed. |
| Run 4 | coded | Hybrid + NN + MSA | Run 3 + 8 MSA features. pair_dim 80→88. 50 epochs. Ready to submit. |

---

## 3. What Each Run Does (Plain English)

**BASIC:** Trained a tiny distance-prediction MLP locally on 844 structures. Predicted distances → MDS → 3D coords. No templates.

**Fork 2:** For each test RNA, searches 24K known structures for the most similar sequence. Copies that structure's real experimental 3D coordinates. Simple template copying — no neural network.

**Run 1 (ADV1):** First attempt at combining templates with neural network. RibonanzaNet backbone (frozen) → pairwise features + template features → distance head → MDS → 3D coords. Failed: pickle parser was broken, so model trained on zero real structures.

**Run 2:** Same architecture, fixed pickle parser. Training worked perfectly (val_loss dropped 215→171 over 30 epochs). But score stayed at 0.109 because MDS reconstruction turns noisy distance predictions into garbage 3D coordinates. **Key lesson: MDS is the bottleneck, not training quality.**

**Run 3 Option B:** Three fixes: (1) Unfreeze last 2 backbone layers so features adapt to distance prediction. (2) Hybrid inference — strong templates go directly into Slots 1-2, NN predictions into Slots 3-5. (3) Template-seeded refinement — NN slots start from template coordinates instead of MDS output, then refine toward NN distances via gradient descent.

**Run 4:** Everything from Run 3 OptB, plus 8 MSA (evolutionary) feature channels. Covariation, conservation, gap frequency computed from top-20 similar sequences per target. pair_dim expands 80→88. Warm-starts from Run 3 checkpoint. 50 epochs.

---

## 4. Architecture Evolution

| Component | BASIC | Run 1-2 | Run 3 OptB | Run 4 |
|-----------|-------|---------|------------|-------|
| Backbone | Frozen (100M) | Frozen | Layers 7-8 unfrozen (~2M trainable) | Same |
| Template encoder | None | 368 params | 368 params | Same |
| Distance head input | 64 channels | 80 (64+16) | 80 (64+16) | **88 (64+16+8)** |
| MSA features | None | None | None | **8 channels** |
| Inference | MDS only | MDS only | **Template-seeded refinement** | Same |
| Hybrid slots | No | No | **Yes** (1-2 template, 3-5 NN) | Same |
| Epochs | 50 (local) | 15→30 | 30 | **50** |
| Batch size | 4 | 4 | 2 | 2 |

---

## 5. Key Concepts Explained (High School Level)

### What is MDS and Why It Failed

MDS (Multidimensional Scaling) takes a table of distances between points and figures out where to place them in 3D space. It's like having distances between cities and drawing a map. Works perfectly with exact distances. But our neural network predicts distances with ~13 Angstrom average error. MDS amplifies these errors — small noise becomes large coordinate errors, structures get flattened, mirror-flipped, or collapsed. That's why Run 2 scored 0.109 despite the NN learning well.

### What is Template-Seeded Refinement (Option B)

Instead of MDS (build 3D from scratch using noisy distances), start from real template coordinates (a known similar structure) and gently nudge them toward the NN's predicted distances using gradient descent. Like the difference between drawing a portrait from measurements vs starting with a photo of someone similar and making adjustments.

### What is MSA (Multiple Sequence Alignment)

Find the "same" RNA in mouse, frog, fish, fly. Line up their sequences. If positions 15 and 42 always mutate together across species, they're physically touching in 3D (covariation). Evolution has run a billion-year experiment telling us which parts of the RNA touch each other. This signal is independent of templates and backbone features — it's new information the model has never seen.

### What is IPA (Invariant Point Attention)

Instead of predicting distances then converting to 3D (lossy), IPA directly moves atoms into place through iterative refinement. Each atom "looks at" all other atoms, decides how to move, then all atoms update simultaneously. Repeat 8 times. Used by AlphaFold2. No mirror-flip problem, no MDS quality loss. But complex to implement.

### What is Distillation

A small "student" model learns to imitate a large "teacher" model (RNAPro, 500M params) instead of learning from raw data. The teacher's predictions encode vast knowledge. Student inherits that knowledge through training targets. Requires running the teacher on a powerful GPU (24+ GB VRAM) — which our team doesn't have.

---

## 6. Critical Problems Discovered and Fixed

| Problem | Found In | Fixed In | Root Cause |
|---------|----------|----------|-----------|
| Pickle parser assumed list of dicts | Run 1 | Run 2 | Actual format: dict of parallel lists with sugar_ring[0] = C1' |
| Biopython not pre-installed (Python 3.12 GPU image) | Run 1 commit | Run 2 | Offline wheel install from uploaded dataset |
| MDS destroys coordinate quality | Run 2 (score 0.109) | Run 3 OptB | Template-seeded refinement bypasses MDS |
| 15 epochs insufficient | Run 1 | Run 2 (30 epochs) | Cosine LR still high at epoch 15 |
| Frozen backbone features for chemistry, not 3D | Design analysis | Run 3 OptB | Unfreeze layers 7-8 with discriminative LR |
| NN worse than template copying on strong targets | Run 1/2 vs Fork 2 | Run 3 OptB | Hybrid slots preserve template quality |
| NN slots too similar (0.01Å diversity) | Run 3 OptB draft | Run 4 | Increased noise scale (0.5, 1.0, 1.5) |
| No evolutionary features | Architecture review | Run 4 | 8 MSA channels (covariation, conservation, gaps) |

---

## 7. The 8 MSA Feature Channels (Run 4)

| # | Channel | Type | What It Captures |
|---|---------|------|-----------------|
| 1 | Covariation(i,j) | Pairwise N×N | Mutual information — do positions i,j co-mutate? |
| 2 | APC-corrected covariation | Pairwise N×N | Same with background noise removed |
| 3 | Conservation(i) | Per-position broadcast | Does position i never change? |
| 4 | Conservation(j) | Per-position broadcast | Same for position j |
| 5 | Conservation(i) × Conservation(j) | Product | Both positions conserved? |
| 6 | Gap frequency(i) | Per-position broadcast | Position i often missing? |
| 7 | Gap frequency(j) | Per-position broadcast | Same for position j |
| 8 | Neff | Scalar broadcast | Number of effective sequences |

---

## 8. Template Confidence Breakdown (28 Test Targets)

| Category | Count | Hybrid Decision | Run 3-4 Behavior |
|----------|-------|----------------|-------------------|
| Strong (conf > 0.3) | ~16 | Slots 1-2 template, 3-5 NN | Template slots guarantee Fork 2 quality |
| Weak (0.01 < conf ≤ 0.3) | ~10 | All 5 slots NN | Template-seeded refinement from weak template |
| None (conf = 0) | ~2 | All 5 slots MDS fallback | Only case where MDS is used |

---

## 9. Why Run 3 OptB Should Beat 0.287

**Mechanism 1 — Template slots guarantee Fork 2 quality:** ~16 targets with strong templates use template coordinates directly in Slots 1-2. Best-of-5 scoring means template slots carry even if NN slots are bad.

**Mechanism 2 — Template-seeded refinement bypasses MDS:** NN slots start from real template coordinates and refine toward NN distances. No MDS reconstruction. Even weak templates provide better starting points than MDS output.

**Mechanism 3 — Unfrozen backbone adapts to distance prediction:** Last 2 layers of RibonanzaNet fine-tune for 3D distance features instead of chemical mapping features.

**Risk:** Not guaranteed. Our template search may find slightly different templates than Fork 2's original code. Backbone unfreezing could destabilize if LR is too high.

---

## 10. Why Run 4 (MSA) Should Beat Run 3

MSA provides NEW information neither backbone nor template features capture:

- **Backbone features** know chemistry but can't detect long-range 3D contacts from sequence alone
- **Template features** know one similar structure, but weak templates are noisy
- **MSA covariation** detects long-range contacts from evolutionary evidence, even without close templates

MSA helps most on **weak-template targets** (~12 of 28) where the NN is doing all the work. For strong-template targets, template slots already carry the score.

---

## 11. Future Options (After Run 4)

| Option | Feasibility | Impact | Requirement |
|--------|------------|--------|-------------|
| Run 5: IPA (replace MDS entirely) | Medium — 8-12 hrs with library | High — eliminates MDS quality loss, no mirror-flip | Coding time before deadline |
| Distillation from RNAPro | Blocked | Very high — teacher model is state-of-the-art | NVIDIA GPU 24+ GB (not available) |
| More epochs (50→100) | Easy | Low — diminishing returns past 50 | Just change one number |
| Pre-compute backbone features | Medium — ~30 lines code | Saves ~30 min per run | Code change risk |

---

## 12. Score Expectations

| Run | Floor | Expected | Ceiling | Key Variable |
|-----|-------|----------|---------|-------------|
| Run 3 OptB | ~0.22 | 0.28-0.40 | ~0.45 | Template search quality + backbone unfreezing |
| Run 4 | ~0.25 | 0.30-0.45 | ~0.50 | MSA signal strength on weak-template targets |
| Run 5 (IPA) | ~0.25 | 0.35-0.50 | ~0.55+ | IPA library integration quality |
| Fork 2 safety net | 0.287 | 0.287 | 0.287 | Guaranteed fallback |

---

## 13. Run Plan Going Forward

| Priority | Run | Status | Action | Time | Dependencies |
|----------|-----|--------|--------|------|-------------|
| 1 | Run 3 OptB | Committed | Wait for score. Debug from Logs if fails. | ~30 min | None |
| 2 | Run 4 (MSA) | Coded | Upload Run 3 checkpoint → Create notebook → Skip draft → Commit | ~3 hrs | Run 3 checkpoint (preferred) |
| 3 | Run 5 (IPA) | Not started | Only if 24+ hrs remain after Run 4 | 8-12 hrs | Run 4 checkpoint |
| 4 | Distillation | Blocked | No NVIDIA GPU available | N/A | Lab GPU (24+ GB) |

---

## 14. Time-Saving Workflow Rules

| Rule | Saves | Rationale |
|------|-------|-----------|
| Skip draft runs — go straight to commit | 2-3 hrs/run | Logic is proven from prior runs. Env issues are 1-line fixes. |
| Debug from Logs tab, never re-run draft | 2-3 hrs/debug | Draft success tells you nothing new. Commit logs show the exact failure. |
| Code next run while current is on Kaggle | 4-5 hrs wall clock | Never wait idle. |
| Upload checkpoint datasets in advance | 15 min | Avoids blocking the next run. |
| Never resubmit same architecture with more epochs | 1 slot + 2.5 hrs | MSA channels provide more value than 20 extra epochs on same features. |

---

## 15. Kaggle Datasets Required

| # | Dataset | Contents | Size | New for Run 4? |
|---|---------|----------|------|---------------|
| 1 | Competition data | Stanford RNA 3D Folding Part 2 | N/A | No |
| 2 | rna-cif-to-csv | rna_sequences.csv + rna_coordinates.csv | ~100 MB | No |
| 3 | adv1-weights | RibonanzaNet.pt + best_model.pt | ~43 MB | No |
| 4 | adv1-training-data | pdb_xyz_data.pkl | ~52 MB | No |
| 5 | ribonanzanet-repo | RibonanzaNet/ folder with Network.py | ~10 MB | No |
| 6 | biopython-wheel | .whl files (cp310+cp311+cp312) | ~5 MB | No |
| **7** | **adv1-run3-checkpoint** | **adv1_best_model.pt** | **8.89 MB** | **YES** |

---

## 16. Codebase File Inventory

```
APPROACH2-RIBBOZANET/HY-BAS-ADV1/kaggle/
├── hy_bas_adv1_run1_notebook.py         31.95 KB  (historical)
├── hy_bas_adv1_run2_notebook.py         36.44 KB  (historical)
├── hy_bas_adv1_run3_notebook.py         41.93 KB  (Option A, never submitted)
├── hy_bas_adv1_run3_optb_notebook.py    55.57 KB  (submitted, pending score)
├── hy_bas_adv1_run4_notebook.py         60.25 KB  (ready to submit)
├── RUN4_DESIGN.md                       14.49 KB
└── KAGGLE_SETUP_GUIDE.md                 3.73 KB

HY-BAS-ADV1-RUN2-REMOTE/
├── adv1_best_model (1).pt               114 KB  (Run 2 checkpoint)
├── download (1).txt                     604 KB  (Run 2 full training log)
├── submission (5).csv                   1.58 MB
└── submission_raw (1).csv               1.57 MB

HY-BAS-ADV1-RUN3-REMOTE/
├── adv1_best_model.pt                   8.89 MB  (Run 3 checkpoint — upload for Run 4)
├── submission.csv                       1.98 MB  (draft run output)
├── submission_raw.csv                   1.98 MB
├── __notebook_source__.ipynb            56.68 KB
└── RUN3_DESIGN.md                       9.63 KB
```

---

## 17. Hardware Constraints

| Machine | GPU | Can Train? | Can Run RNAPro? |
|---------|-----|-----------|----------------|
| Team desktop (label says GeForce RTX) | Intel Iris Xe only (no NVIDIA detected) | No | No |
| Kaggle (free) | T4 x2 (16 GB each) | Yes — all training here | No (too small for 500M model) |

**Implication:** All compute happens on Kaggle. No local training. No RNAPro distillation.

---

## 18. MDS vs IPA vs Template-Seeded Refinement

| Method | How It Works | Quality | Mirror-Flip? | In Our Pipeline? |
|--------|-------------|---------|-------------|-----------------|
| MDS | Distance matrix → eigendecomposition → 3D coords | Poor with noisy distances (0.109 score) | Yes — random sign of eigenvectors | Fallback only (when no template) |
| Template-seeded refinement | Start from template coords → gradient descent toward NN distances | Good — starts from real structure | No — starts from real coords | Primary method (Run 3 OptB+) |
| IPA | Iteratively move atoms using attention over pairwise features | Best — direct 3D, equivariant | No — rotations preserve handedness | Not yet (Run 5 candidate) |

---

## 19. Decision Log

| Decision | Chosen | Rejected | Why |
|----------|--------|----------|-----|
| Skip Run 3 Option A | Yes | Run both A and B | Option A uses MDS for NN slots. Option B is strictly better. |
| MSA before IPA | MSA first (Run 4) | IPA first (Run 5) | MSA is additive, low-risk, 4-5 hrs. IPA is architectural change, 8-12 hrs, high-risk. |
| 50 epochs for Run 4 | Yes | Keep 30 | 8 new MSA channels need integration time. Warm-start means epochs 1-30 re-learn existing, 31-50 jointly optimize. |
| Skip draft for Run 4 | Yes | Run draft first | Saves 2.5 hrs. Pattern proven from 4 prior runs. Debug from Logs tab if commit fails. |
| Distillation from RNAPro | Cannot do | Would be high-impact | No NVIDIA GPU (24+ GB) available on the team. |
| MSA source | Build from template search hits (top-20) | Use competition MSA files | Self-contained; template search already runs; no dependency on external files. |
| Warm-start Run 4 from Run 3 | Yes (80→88 expansion) | From BASIC (64→88) | Run 3's checkpoint has 30 epochs of backbone + distance head training. |
