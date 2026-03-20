# ADV1 — Detailed Design: Fine-Tuning & Optimization for Approach 2

## DESIGN ONLY — No code changes until design is approved.

---
---

# SECTION 1: WHERE WE ARE (BASIC Recap)

## What BASIC does

```
RNA sequence
    → RibonanzaNet (FROZEN backbone) → pairwise features (B, N, N, 64)
    → Distance Head (small MLP, TRAINABLE) → predicted distance matrix (N×N)
    → MDS + gradient refinement → 3D coordinates (N×3)
    → 5 diverse predictions → submission.csv
```

## BASIC limitations

| Limitation | Why it matters |
|---|---|
| Backbone is **completely frozen** | We're only training a tiny MLP on top. The backbone's features may not be optimal for 3D prediction — it was trained for chemical mapping (2D), not 3D. |
| Distance matrix approach | Loses information. Going from features → distances → coordinates is a 2-step lossy pipeline. Direct coordinate prediction (IPA) would be better. |
| No MSA features | Multiple Sequence Alignments contain evolutionary covariation signals that are critical for structure prediction. Competition provides MSA/ folder. |
| No template features | When templates exist, they're the single strongest signal (Approach 1 scored 0.593 with TBM alone). BASIC ignores them entirely. |
| Simple MLP head | 3 linear layers can't capture the complex geometry of 3D folding. |
| No recycling | AlphaFold2/RhoFold+ use 3+ recycling iterations to progressively refine structures. |

---
---

# SECTION 2: IS RAG AN OPTION? IS AGENTIC RAG AN OPTION?

## Short answer: NOT in the traditional LLM sense. But YES in spirit.

### What RAG is (in the LLM world)
Retrieval-Augmented Generation: an LLM retrieves relevant documents from a database
at inference time, then uses those documents as context to generate a better answer.

### Why traditional RAG does NOT apply here

This is NOT a text generation problem. We're predicting 3D coordinates (numbers),
not generating text. There's no "prompt" to augment with retrieved documents.
The model is a neural network (RibonanzaNet + head), not an LLM.

### But the CONCEPT of retrieval-augmented prediction DOES apply

The "retrieval" idea maps directly to **template-based modeling (TBM)**:

| LLM RAG concept | RNA structure equivalent |
|---|---|
| Query document | Test RNA sequence |
| Vector database | PDB RNA sequence database |
| Retrieved documents | Template structures (similar known RNAs) |
| Augmented context | Template coordinates fed as additional features |
| Generation | 3D coordinate prediction |

**This is exactly what RNAPro does.** It searches PDB for similar RNA structures
(retrieval), then feeds those template coordinates into the neural network
alongside the sequence features (augmentation). The model then predicts the
final structure using both learned features AND retrieved templates (generation).

### What "Agentic RAG" would mean in this context

An agentic approach would be a multi-step pipeline where the model:
1. Decides WHICH databases to search (PDB, Rfam, RNACentral)
2. Evaluates template quality and decides how many to use
3. Chooses whether to trust the template or rely on learned features
4. Iteratively refines by searching for more templates based on initial predictions

**Verdict:** This is conceptually interesting but overkill for this competition.
The competition runs in a fixed notebook with no internet. The "agentic" decisions
(which templates to use, how to weight them) can be hardcoded as heuristics.
RNAPro already does this with its template selection strategy.

### RECOMMENDATION on RAG/Agentic RAG

**Don't pursue RAG as a separate architecture.** Instead, incorporate the same
principle by adding template features to the model (Option D below). This is
the proven approach — RNAPro scored highest by combining RibonanzaNet2 features
+ template coordinates + MSA features in a single end-to-end model.

---
---

# SECTION 3: FINE-TUNING OPTIONS (from simplest to most complex)

---

## Option A: Unfreeze Last N Backbone Layers (Simplest Upgrade)

### What changes from BASIC
In BASIC: `backbone.freeze = true` (all parameters frozen)
In ADV1:  Unfreeze the last 2-4 transformer layers of RibonanzaNet

### How it works
- Keep the first 5-7 layers frozen (they learn general RNA features)
- Unfreeze the last 2-4 layers so they can adapt to 3D distance prediction
- Use a much lower learning rate for unfrozen layers (1e-5) vs the distance head (1e-4)
  This is called "discriminative learning rates"

### Pros
- Minimal code change (just change config: `freeze: false`, add `freeze_first_n: 5`)
- Backbone features become optimized for our specific task
- The Ribonanza paper confirms: fine-tuning with pre-training outperformed training from scratch (F1 0.89 vs 0.7)

### Cons
- Risk of overfitting if training data is small
- Slower training (more parameters to update)
- Higher VRAM usage (gradients stored for unfrozen layers)

### VRAM estimate
- Frozen BASIC: ~3-4 GB for batch_size=4, seq_len=128
- Unfrozen last 4 layers: ~6-8 GB (same config)
- Fits on T4 (16 GB) but with reduced batch size

### Critical question
**Q: How many layers to unfreeze?**
The Ribonanza paper fine-tuned the ENTIRE model for downstream tasks (secondary structure,
degradation). But they had 2M+ training sequences. We have far fewer 3D structures (~1000s).
Start with 2 layers, validate, then try 4 layers.

---

## Option B: LoRA / Adapter Fine-Tuning (Parameter-Efficient)

### What is LoRA?
Low-Rank Adaptation: instead of updating ALL parameters in a layer, inject small
"adapter" matrices that modify the layer's behavior with very few extra parameters.

For a linear layer W (256×256 = 65,536 parameters):
- Full fine-tuning: update all 65,536 parameters
- LoRA with rank 8: add A (256×8) and B (8×256) → only 4,096 parameters
  The effective weight becomes W + A·B

### How it works
- Keep ALL backbone parameters frozen
- Insert LoRA adapters into the attention layers (Q, K, V projections)
- Train only the LoRA parameters + the distance head
- Total trainable params: ~50K (LoRA) + ~100K (head) = ~150K vs 100M (full backbone)

### Pros
- Very few trainable parameters → much less overfitting risk
- Minimal VRAM overhead (almost same as frozen BASIC)
- Can experiment with different LoRA configurations quickly
- Well-established technique (used in LLM fine-tuning extensively)

### Cons
- Less expressive than full fine-tuning (limited by low rank)
- LoRA for transformer attention is well-studied for text models but less proven for structural biology models
- Need to integrate LoRA library (peft or custom implementation)

### Implementation
- Use HuggingFace `peft` library: `pip install peft`
- Apply LoRA to the self-attention Q/K/V projections in RibonanzaNet's transformer layers
- Typical config: rank=8, alpha=16, dropout=0.05

### Critical question
**Q: Does LoRA work for pairwise representations?**
Standard LoRA targets attention Q/K/V matrices. RibonanzaNet also has triangle
multiplicative updates and outer product mean for pairwise features. Those may
need their own adapters. This is untested territory.

---

## Option C: Upgrade Distance Head → IPA Structure Module (Architecture Change)

### What changes from BASIC
Replace the simple 3-layer MLP distance head with the Invariant Point Attention (IPA)
structure module described in your `week2_3_ipa_head_steps_explained.md`.

### How it works
- Instead of: features → distance matrix → MDS → coordinates
- Do: features → IPA blocks (iterative 3D refinement) → coordinates directly
- IPA operates in 3D space with frames (rotation + translation per nucleotide)
- Each IPA block updates positions using geometry-aware attention
- This is what AlphaFold2, RhoFold+, and RNAPro all use

### Pros
- End-to-end differentiable (no MDS reconstruction step)
- Geometry-aware (enforces SE(3) invariance — physics-compatible)
- State-of-the-art approach used by all top methods
- Can use FAPE loss (AlphaFold2's loss) which is stronger than distance MSE

### Cons
- Significantly more complex to implement correctly
- Higher VRAM (IPA blocks + frame computation)
- Longer training time
- Debugging IPA is hard — subtle bugs in rotation math cause silent failures

### VRAM estimate
- 4 IPA blocks, seq_len=128: ~8-10 GB
- 8 IPA blocks, seq_len=128: ~12-14 GB (tight on T4)

### Critical questions
**Q: Use lucidrains' library or implement from scratch?**
`pip install invariant-point-attention` gives you a working IPA block.
Recommended to start with this, then customize if needed.

**Q: How many IPA blocks?**
AlphaFold2 uses 8. RhoFold+ uses 8. Start with 4 for VRAM reasons.

---

## Option D: Add Template Features (Retrieval-Augmented — the "RAG" equivalent)

### What changes from BASIC
Feed template coordinates (from Approach 1's TBM) as additional input features.

### How it works
1. Run Approach 1 (MMseqs2 TBM) to get template coordinates for each test target
2. Encode template coordinates as a "template embedding":
   - Distance matrix from template coords
   - Backbone torsion angles from template
   - Template confidence scores
3. Concatenate template embedding with RibonanzaNet's pairwise features
4. Feed the combined features into the distance head (or IPA head)

### This is exactly what RNAPro does
From the RNAPro GitHub (verified):
```
--use_template ca_precomputed
--template_data path/to/template_features.pt
--num_templates 4
```

### Pros
- Leverages the single strongest signal in the competition (templates won Part 1)
- Combines learned features + retrieved knowledge (the "RAG" idea)
- RNAPro proved this combination beats either approach alone

### Cons
- Requires pre-computing templates (Approach 1 must run first)
- Template quality varies — some targets have great templates, some have none
- Model must learn WHEN to trust templates vs learned features
- More complex input pipeline

### Critical question
**Q: What about targets with NO template?**
For ~30% of targets, MMseqs2 finds no useful template. The model must gracefully
fall back to pure learned prediction. This means the architecture needs a
"template present" / "template absent" mode.

---

## Option E: Add MSA Features (Evolutionary Signal)

### What changes from BASIC
Incorporate Multiple Sequence Alignments (MSAs) provided by the competition.

### How it works
- Competition provides pre-computed MSAs at `/kaggle/input/.../MSA/`
- MSAs contain sequences of related RNAs from other organisms
- Coevolution patterns in MSAs reveal which nucleotide pairs are likely in contact
  (if position i and position j change together across species, they probably interact)
- Encode MSA features (covariance matrix, conservation scores) as additional input

### This is what RNAPro and RhoFold+ both use
From the RNAPro GitHub:
```
release_data/kaggle/MSA_v2/  — MSA files for each target
```

### Pros
- Evolutionary signal is orthogonal to sequence features (adds new information)
- Covariation is one of the strongest predictors of base pairing
- Competition specifically provides MSAs because they're important

### Cons
- MSA quality varies (some targets have deep MSAs, others shallow)
- Processing MSAs requires column-wise attention (Evoformer-like), which is complex
- Significant VRAM increase (MSA depth × sequence length × features)

### Critical question
**Q: How deep are the competition MSAs?**
Need to inspect the actual files. If shallow (few sequences), the covariation
signal may be weak. If deep (hundreds of sequences), this is a major advantage.

---

## Option F: Knowledge Distillation from RNAPro (Teacher-Student)

### What changes from BASIC
Use RNAPro (which needs A100 80GB GPU) as a "teacher" to train our small "student" model.

### How it works
1. Rent cloud GPU (A100, ~$1-2/hour) and run RNAPro on ALL training sequences
2. Save RNAPro's predicted coordinates (these are very accurate)
3. Train our small student model to mimic RNAPro's predictions
4. Student loss = distance_to_RNAPro_prediction + distance_to_ground_truth

### Pros
- Student can learn from a much more accurate teacher
- Student model is small enough for Kaggle T4
- No need to replicate RNAPro's complex architecture — just learn its outputs
- Can augment limited training data with RNAPro predictions on unlabeled sequences

### Cons
- Requires access to A100 GPU (cloud rental cost ~$50-200)
- Student is bounded by its architecture — can't fully replicate teacher
- RNAPro may not be available or may have licensing restrictions
- Time-consuming to run inference on all training sequences

### Critical question
**Q: Is RNAPro publicly available to run?**
Yes — Apache-2.0 license, weights on HuggingFace. But requires NVIDIA GPU with 40-80GB VRAM.
Cloud instances: AWS p4d.24xlarge ($32/hr) or Lambda Labs A100 ($1.10/hr).

---
---

# SECTION 4: DATA SOURCES & ENHANCEMENT

---

## Available data sources

| Source | What | Size | URL |
|---|---|---|---|
| Competition training data | RNA structures with C1' coordinates | ~1000s of structures | Competition Data tab |
| Pre-processed pickle | Same, ready to load | ~1-2 GB | kaggle.com/datasets/shujun717/stanford3d-dataprocessing-pickle |
| PDB_RNA (competition) | All RNA CIF files in PDB | ~15,000 files, 310 GB | Competition Data tab |
| RNACentral | Comprehensive RNA sequence database | Millions of sequences | rnacentral.org |
| Rfam | RNA families with known secondary structures | ~4,000 families | rfam.org |
| Eterna (citizen science) | RNA sequences with chemical mapping data | 2M+ sequences | eternagame.org |
| RNAPro predictions | Teacher model outputs (if distilling) | Must compute | Run RNAPro yourself |

## Data enhancement techniques

### 1. Structure augmentation (already in BASIC)
- Random 3D rotation and translation of coordinates
- Does NOT change the structure — just changes the viewing angle
- Teaches the model SE(3) invariance

### 2. Crop augmentation
- For long sequences (>256 nt), randomly crop to shorter windows
- Different crops in each epoch → model sees different parts of long RNAs
- This is standard in AlphaFold2/RNAPro training

### 3. Noisy student (self-training)
- Train initial model → predict on unlabeled RNA sequences → use predictions as pseudo-labels → retrain
- The Ribonanza paper used this exact technique (pseudo-labels from top Kaggle submissions)
- Confirmed beneficial: "improvements of RibonanzaNet test accuracy as more pseudo-labels were included"

### 4. Multi-task training
- Don't just predict distances. Also predict:
  - Secondary structure (base-pair probability matrix)
  - Solvent accessibility
  - Chemical mapping profiles (DMS, 2A3)
- These auxiliary tasks improve the learned representations
- The Ribonanza paper showed RibonanzaNet fine-tuned on secondary structure improved 3D prediction when used with trRosettaRNA

### 5. Homology-based augmentation
- For each training structure, find similar sequences via MMseqs2
- Align them and transfer coordinates (like TBM but for training data)
- Creates "approximate" structures that augment the training set

---
---

# SECTION 5: ML OPTIMIZATION TECHNIQUES

---

## Training optimizations

| Technique | What | When to use | Expected impact |
|---|---|---|---|
| **Cosine annealing LR** | Learning rate decays following cosine curve | Always (already in BASIC) | Baseline |
| **Warmup** | Low LR for first 5% of training, then ramp up | When fine-tuning backbone | Prevents early catastrophic updates |
| **Gradient accumulation** | Simulate larger batch by accumulating gradients over N steps | When batch size limited by VRAM | Stabilizes training |
| **Exponential moving average (EMA)** | Maintain a running average of model weights | Always for final evaluation | Smoother, more robust predictions |
| **Stochastic weight averaging (SWA)** | Average weights from last K epochs | Final submission | Can improve generalization |
| **Label smoothing** | Soft targets instead of hard (for classification heads) | If adding secondary structure prediction | Reduces overfitting |
| **Mixed precision (FP16)** | Already in BASIC | Always | ~2x speedup, ~40% less VRAM |
| **Gradient checkpointing** | Trade compute for VRAM by recomputing activations during backward | When VRAM is tight (>256 nt sequences) | Enables longer sequences |

## Loss function improvements (over BASIC)

| Loss | BASIC | ADV1 option | Why better |
|---|---|---|---|
| **Distance MSE** | Yes (primary) | Keep | Still useful as auxiliary |
| **FAPE** | No | Add if using IPA | Frame-aligned, invariant to global rotation |
| **Differentiable TM-score** | Optional in BASIC | Add as secondary | Directly optimizes competition metric |
| **Contact map loss** | No | Add | Binary classification: are residues i,j within 8Å? |
| **Dihedral angle loss** | No | Add | Penalizes unphysical backbone angles |
| **Auxiliary 2D loss** | No | Add | Predict secondary structure as multi-task |

## Ensemble strategies (for 5 predictions)

| Strategy | What | BASIC | ADV1 |
|---|---|---|---|
| **Noise injection** | Add noise to predicted distances | Yes (only method) | Keep as one option |
| **Multiple seeds** | Train N models with different random seeds, use all | No | Yes — train 3-5 models |
| **Dropout variation** | Enable dropout at inference, run multiple times | No | Yes — MC Dropout |
| **Template variation** | Different templates for different predictions | No | Yes — if using templates |
| **Checkpoint ensemble** | Use models from different training epochs | No | Yes — epochs 80, 85, 90, 95, 100 |

---
---

# SECTION 6: RECOMMENDED ADV1 STRATEGY (Phased)

---

## Phase 1 (quickest win — 2-3 days)
**Unfreeze last 2 backbone layers + better training**
- Copy BASIC code to ADV1
- Change `freeze: true` → selective unfreezing of last 2 layers
- Add warmup (5% of epochs) + discriminative LR (backbone: 1e-5, head: 1e-4)
- Add gradient accumulation (effective batch size = 16)
- Add EMA for evaluation
- Expected improvement: moderate (features better adapted to 3D task)

## Phase 2 (medium effort — 1 week)
**Add template features**
- Run Approach 1 notebook to get template coordinates for all training + test targets
- Encode templates as distance matrices
- Concatenate with RibonanzaNet pairwise features
- Retrain distance head on the larger feature set
- Expected improvement: significant (templates are the strongest signal)

## Phase 3 (high effort — 2 weeks)
**Replace distance head with IPA structure module**
- Implement IPA using lucidrains library
- Add FAPE loss
- Add recycling (3 iterations)
- Train end-to-end with unfrozen backbone
- Expected improvement: large (state-of-the-art architecture)

## Phase 4 (if time permits)
**Knowledge distillation + MSA features**
- Run RNAPro on cloud GPU for all training sequences
- Use RNAPro predictions as soft targets
- Add MSA features if VRAM permits
- Expected improvement: incremental on top of Phase 3

---
---

# SECTION 7: CRITICAL QUESTIONS FOR REVIEW

---

1. **VRAM budget**: Your local machine has RTX 3070 (8GB). Kaggle T4 has 16GB.
   Options C and E (IPA + MSA) may not fit on your local machine for testing.
   Should we target Kaggle T4 only, or optimize for local development too?

2. **Time budget**: Entry deadline was March 18, 2026 (past). Final submissions
   March 25, 2026 (5 days from now). Which phases are realistic to complete?

3. **Phase 1 vs Phase 2 first**: Unfreezing backbone (Phase 1) helps the model
   learn better features. Adding templates (Phase 2) adds the strongest signal.
   If you can only do ONE, Phase 2 is likely higher impact. Your call.

4. **LoRA vs full unfreeze**: LoRA is safer (less overfitting) but less expressive.
   Full unfreeze of 2-4 layers is riskier but has higher ceiling. Given the small
   training set, LoRA might be the pragmatic choice.

5. **Pre-computed templates**: For Phase 2, you need to run Approach 1 first on
   all training sequences (not just test). Have you done that?

6. **Cloud GPU access**: For Phase 4 (knowledge distillation), do you have access
   to A100 instances (Lambda Labs, RunPod, AWS)?

7. **Evaluation**: Are you tracking TM-score on a held-out validation set?
   Without proper evaluation, you won't know if ADV1 is actually better than BASIC.

---
---

# SECTION 8: DECISION MATRIX

---

| Option | Effort | Expected Impact | VRAM | Risk | Recommendation |
|---|---|---|---|---|---|
| A: Unfreeze layers | Low | Medium | +3-4 GB | Medium (overfitting) | **DO — Phase 1** |
| B: LoRA adapters | Low-Medium | Medium | +0.5 GB | Low | ALTERNATIVE to A |
| C: IPA head | High | High | +6-8 GB | High (complex impl) | DO if time permits — Phase 3 |
| D: Template features | Medium | **Very High** | +1-2 GB | Low (proven approach) | **DO — Phase 2 (highest priority)** |
| E: MSA features | Medium-High | Medium-High | +4-6 GB | Medium | DO after C |
| F: Distillation | Medium | Medium-High | Same | Low | DO if cloud GPU available |

**Bottom line**: If you can only do ONE thing, **add template features (Option D)**.
Templates won Part 1. Even a simple concatenation of template distance features
with RibonanzaNet features would be a major upgrade over BASIC.

---
---

# SECTION 9: FILE STRUCTURE FOR ADV1

---

```
ADV1/
├── README.md
├── DESIGN.md                    ← THIS DOCUMENT
├── config.yaml                  ← Extended config (from BASIC + new options)
│
├── models/
│   ├── backbone.py              ← Modified: selective layer unfreezing, LoRA option
│   ├── distance_head.py         ← Same as BASIC (kept for compatibility)
│   ├── ipa_head.py              ← NEW: IPA structure module (Phase 3)
│   ├── template_encoder.py      ← NEW: encode template coords as features (Phase 2)
│   ├── reconstructor.py         ← Same as BASIC
│   └── ensemble.py              ← NEW: multi-model ensemble logic
│
├── data/
│   ├── dataset.py               ← Extended: loads templates + MSA features
│   ├── collate.py               ← Extended: handles template tensors
│   ├── augmentation.py          ← Extended: crop augmentation, noisy student
│   └── template_loader.py       ← NEW: loads Approach 1 template outputs
│
├── losses/
│   ├── distance_loss.py         ← Same as BASIC
│   ├── constraint_loss.py       ← Same as BASIC
│   ├── tm_score_approx.py       ← Same as BASIC
│   ├── fape_loss.py             ← NEW: Frame Aligned Point Error (Phase 3)
│   └── multitask_loss.py        ← NEW: combined loss with weighting
│
├── train.py                     ← Extended: EMA, warmup, gradient accum, multi-phase
├── predict.py                   ← Extended: ensemble, template integration
└── utils/
    ├── pdb_parser.py            ← Same as BASIC
    ├── submission.py            ← Same as BASIC
    └── evaluation.py            ← NEW: TM-score evaluation on validation set
```
