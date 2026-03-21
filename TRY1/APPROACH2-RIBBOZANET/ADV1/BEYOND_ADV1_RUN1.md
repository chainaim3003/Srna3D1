# Beyond ADV1-Run1 — Future Improvement Phases

## Where You Are After ADV1-Run1

```
Unfrozen backbone (last 2 layers) + template features → distance matrix → MDS → 3D coords
```

The remaining bottleneck is the **lossy pipeline** — going from features → distances → MDS → coordinates throws away information at each step. Every improvement beyond Run1 targets this.

---

## Level 1: Replace Distance Head with IPA (DESIGN.md Phase 3)

**What:** Replace the MLP distance head with Invariant Point Attention (IPA) structure module.

**Why:** Instead of predicting distances then reconstructing 3D coords via MDS, IPA predicts coordinates **directly** from features. This removes the lossy MDS step entirely. This is what AlphaFold2, RhoFold+, and RNAPro all use.

**How:**
- Install: `pip install invariant-point-attention` (lucidrains library)
- Replace `distance_head.py` with `ipa_head.py`
- Add FAPE loss (Frame Aligned Point Error — AlphaFold2's loss function)
- Start with 4 IPA blocks (AlphaFold2 and RhoFold+ use 8)
- Train end-to-end with unfrozen backbone

**VRAM:** ~8-10 GB for 4 IPA blocks, seq_len=128. Fits on T4 (16 GB).

**Effort:** High — 1-2 weeks. IPA involves rotation matrices and SE(3) equivariance. Subtle bugs in rotation math cause silent failures. Debugging is difficult.

**Expected impact:** Large. This is the single biggest architectural upgrade available.

**Source:** DESIGN.md Option C, Section 3

---

## Level 2: Add Recycling (DESIGN.md Phase 3 addition)

**What:** Run the IPA module 3 times, feeding each round's output coordinates back as input to the next round.

**Why:** Each pass refines the structure. The first pass gives a rough shape, the second corrects errors, the third polishes details. AlphaFold2 uses 3 recycling iterations.

**How:**
- Wrap the IPA forward pass in a loop (3 iterations)
- Each iteration receives the previous iteration's coordinates as additional input
- Gradients only flow through the final iteration (stop-gradient on earlier ones)
- Config setting: `recycling_iterations: 3`

**VRAM:** Minimal additional cost if using stop-gradient on earlier iterations.

**Effort:** Low once IPA exists — it's a config change plus a for-loop wrapper.

**Expected impact:** Medium. Improves accuracy of IPA predictions.

**Source:** DESIGN.md Option C, Section 3

---

## Level 3: Add MSA Features (DESIGN.md Phase 3 addition)

**What:** Incorporate Multiple Sequence Alignments (MSAs) provided by the competition in the `MSA/` folder.

**Why:** Coevolution patterns in MSAs reveal which nucleotide pairs are in contact. If positions i and j mutate together across species, they probably interact physically. This signal is orthogonal to sequence features — it's genuinely new information that the model cannot learn from sequence alone.

**How:**
- Competition provides pre-computed MSAs at `/kaggle/input/.../MSA/`
- Encode MSA features: covariance matrix, conservation scores, gap frequencies
- Process with column-wise attention (simplified Evoformer)
- Concatenate MSA features with pairwise features before the structure module

**VRAM:** +4-6 GB depending on MSA depth and sequence length.

**Effort:** Medium-High — 1 week. Processing MSAs requires column-wise attention (Evoformer-like), which is complex but well-documented.

**Expected impact:** Medium-High. Evolutionary signal is one of the strongest predictors of base pairing. Competition specifically provides MSAs because they're important.

**Critical question from DESIGN.md:** How deep are the competition MSAs? Deep MSAs (hundreds of sequences) give strong covariation signal. Shallow MSAs (few sequences) give weak signal. Must inspect actual files before committing.

**Sources:**
- DESIGN.md Option E, Section 3
- RNAPro GitHub: `release_data/kaggle/MSA_v2/`
- RhoFold+ paper: https://www.nature.com/articles/s41592-024-02487-0

---

## Level 4: Knowledge Distillation from RNAPro (DESIGN.md Phase 4)

**What:** Use RNAPro as a "teacher" model to train your smaller "student" model.

**Why:** RNAPro combines IPA + templates + MSA + recycling in one model (~500M params, needs A100 80GB). It's too large for Kaggle T4, but its predictions are very accurate. Train your small model to mimic RNAPro's outputs — student stays small enough for T4 but learns from a much more powerful teacher.

**How:**
1. Rent cloud GPU: Lambda Labs A100 at ~$1.10/hour, or AWS p4d.24xlarge at ~$32/hour
2. Install RNAPro (Apache-2.0 license, weights on HuggingFace)
3. Run RNAPro inference on ALL training sequences + test sequences
4. Save RNAPro's predicted coordinates
5. Train your model with combined loss:
   `loss = α * distance_to_ground_truth + β * distance_to_RNAPro_prediction`

**VRAM:** Same as your model (no change to inference). Cloud GPU only needed for the teacher inference step.

**Effort:** Medium — days of work plus $50-200 cloud cost.

**Expected impact:** Medium-Large. Student can learn patterns that would take much more training data to discover on its own.

**Source:**
- DESIGN.md Option F, Section 3
- RNAPro GitHub: https://github.com/NVIDIA-Digital-Bio/RNAPro

---

## Level 5: Ensemble Strategies (DESIGN.md Section 5)

**What:** Combine multiple models/predictions for better accuracy.

**Strategies from DESIGN.md:**

| Strategy | What | How |
|----------|------|-----|
| Multiple seeds | Train 3-5 models with different random seeds | Average their distance predictions before MDS |
| Checkpoint ensemble | Use models from different training epochs | Epochs 80, 85, 90, 95, 100 give diverse snapshots |
| MC Dropout | Enable dropout at inference, run multiple times | Each run gives slightly different predictions |
| Template variation | Different templates for different prediction slots | Prediction 1 uses template #1, prediction 2 uses template #3, etc. |
| Cross-approach ensemble | Mix Approach 1 + Approach 2 predictions | Best-of-5 from different methods |

**Best-of-5 slot allocation (recommended):**
```
For targets WITH template hits:
  Slot 1: Approach 1 template #1 (best MMseqs2 hit)
  Slot 2: Approach 1 template #2 (2nd best hit)
  Slot 3: ADV1 model (clean prediction)
  Slot 4: ADV1 model (different seed or MC Dropout)
  Slot 5: ADV1 model (noisy distance + different refinement)

For targets WITHOUT template hits:
  Slots 1-5: All ADV1 model with diversity (noise, seeds, dropout)
```

**Effort:** Low-Medium per strategy. Training multiple seeds takes N× the training time.

**Expected impact:** Small-Medium. Ensembles typically improve by 1-5% over single models.

**Source:** DESIGN.md Section 5, Ensemble strategies table

---

## Level 6: Data Enhancement (DESIGN.md Section 4)

### 6a: Noisy Student / Self-Training

**What:** Train model → predict on unlabeled RNA sequences → use predictions as pseudo-labels → retrain on larger dataset.

**Why:** The Ribonanza paper confirmed this works: "improvements of RibonanzaNet test accuracy as more pseudo-labels were included."

**How:**
1. Train your model on the 661 labeled structures (already done)
2. Predict 3D structures for unlabeled RNA sequences (from RNACentral or Rfam)
3. Use those predictions as pseudo-labels (approximate ground truth)
4. Retrain on labeled + pseudo-labeled data (much larger dataset)
5. Repeat 2-3 times

**Source:** DESIGN.md Section 4, technique #3. Ribonanza paper.

### 6b: Homology-Based Augmentation

**What:** For each training structure, use MMseqs2 to find similar sequences and transfer coordinates to create approximate new training structures.

**How:** Same as Approach 1 TBM but applied to training data — creating more (approximate) training examples from existing ones.

**Source:** DESIGN.md Section 4, technique #5

### 6c: Crop Augmentation

**What:** For long sequences (>256 nt), randomly crop to shorter windows. Different crops each epoch means the model sees different parts of long RNAs.

**Source:** DESIGN.md Section 4, technique #2. Standard in AlphaFold2/RNAPro training.

---

## Level 7: Training Optimizations (DESIGN.md Section 5)

| Technique | What | When to add |
|-----------|------|-------------|
| EMA (Exponential Moving Average) | Running average of model weights for smoother predictions | Any phase — improves all models |
| SWA (Stochastic Weight Averaging) | Average weights from last K epochs | Final submission |
| Gradient checkpointing | Trade compute for VRAM by recomputing activations during backward | When VRAM is tight (IPA + MSA) |
| Contact map loss | Binary classification: are residues i,j within 8Å? | After IPA is working |
| Dihedral angle loss | Penalizes unphysical backbone angles | After IPA is working |
| Multi-task training | Predict secondary structure + distances simultaneously | After base model is stable |

**Source:** DESIGN.md Section 5, tables on training optimizations and loss functions

---

## Summary: Full Improvement Roadmap

```
BASIC (done)
  └── ADV1-Run1 (unfreeze + templates) ← YOU ARE HERE
        └── Level 1: IPA head (replace MDS pipeline)
              └── Level 2: Recycling (3 iterations)
              └── Level 3: MSA features
                    └── Level 4: Knowledge distillation from RNAPro
                          └── Level 5: Ensemble strategies
                          └── Level 6: Data enhancement
                          └── Level 7: Training optimizations
```

Each level builds on the previous. The biggest jumps are:
- **BASIC → ADV1-Run1:** Templates (proven 0.593 TM-score signal)
- **ADV1-Run1 → IPA:** Removes lossy MDS pipeline (state-of-the-art architecture)
- **IPA → MSA:** Adds evolutionary signal (orthogonal information)

The ceiling is roughly what RNAPro achieves — it combines ALL of these in one model.

---

## References (Official Documentation Only)

| Source | What |
|--------|------|
| DESIGN.md (local) | `ADV1/DESIGN.md` — Full design with options A-F, all referenced sections |
| RNAPro | https://github.com/NVIDIA-Digital-Bio/RNAPro — SOTA RNA structure prediction |
| RhoFold+ | https://www.nature.com/articles/s41592-024-02487-0 — IPA + MSA for RNA |
| AlphaFold2 | https://www.nature.com/articles/s41586-021-03819-2 — IPA, FAPE loss, recycling |
| lucidrains IPA | `pip install invariant-point-attention` — IPA implementation |
| Ribonanza paper | https://arxiv.org/abs/2311.08710 — Noisy student, fine-tuning results |
| Competition | https://www.kaggle.com/competitions/stanford-rna-3d-folding-2 |
| Competition paper | https://pmc.ncbi.nlm.nih.gov/articles/PMC12776560/ |
