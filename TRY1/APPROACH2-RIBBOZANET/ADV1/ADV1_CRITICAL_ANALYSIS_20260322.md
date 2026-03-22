# ADV1 Critical Analysis & Detailed Design Plan
## Based on Actual Scores: BASIC 0.092, Fork1 0.17, Fork2 0.287
## Timestamp: 2026-03-22 ~01:00 UTC

---

## SECTION 1: WHAT THE SCORES TELL US

### Score Comparison

| Approach | TM-Score | Rank | What It Does |
|----------|----------|------|-------------|
| BASIC | 0.092 | ~2199 | Neural only, no templates, pre-made CSV (hidden targets = zeros) |
| Fork 1 (rhijudas) | 0.17 | ~1500? | Template-only, Part 2 baseline, runs end-to-end on Kaggle |
| Fork 2 (jaejohn) | **0.287** | **1756** | Template-only, 1st place Part 1 logic, Option B post-processing |
| best_template_oracle | 0.554 | benchmark | Perfect template for every target |
| 1st place (AyPy) | 0.499 | 1 | Unknown approach (likely neural + templates) |

### Key Insights

1. **Fork 2 (0.287) >> Fork 1 (0.17)**: jaejohn's custom template ranking,
   gap-filling, and diversity strategy adds 0.117 points over the DasLab
   baseline. This is a HUGE difference from the same search results.

2. **Fork 2 (0.287) vs oracle (0.554)**: There's 0.267 points of headroom
   from better template usage alone. Some of this is from hidden targets
   where no template exists — neural predictions would fill that gap.

3. **BASIC (0.092) is essentially zero contribution**: The frozen backbone
   + pre-made CSV approach adds almost nothing. But the BASIC checkpoint
   weights DO have value as a warm-start for ADV1.

4. **The gap to competitive (0.44)** is 0.153 from Fork 2. This gap comes from:
   - No-template targets getting zeros instead of neural predictions
   - Hidden test targets having no predictions at all from pre-made CSV
   - Potentially better neural models that improve on template predictions

### What This Means for ADV1

ADV1's value proposition is clearest for:
- **Hidden targets**: Fork 2 fills with zeros. ADV1 would produce real
  (even if mediocre) predictions. Even TM=0.05 per hidden target beats 0.0.
- **No-template public targets**: ~16 of 28 score <0.1 with templates alone.
  ADV1's neural features could push these to 0.05-0.15.
- **Strong-template targets**: Template features fed through ADV1 should
  PRESERVE the 0.56-0.92 scores Fork 2 achieves, not degrade them.

---

## SECTION 2: ADV1 CODEBASE STATUS — ALREADY BUILT

### Critical Finding: The Code Already Exists

The ADV1 directory contains a COMPLETE codebase:

| File | Status | What It Does |
|------|--------|-------------|
| `models/template_encoder.py` | ✅ Written | Distance binning → Linear(22,16) → template features |
| `models/distance_head.py` | ✅ Written | Same as BASIC but pair_dim is configurable (64 or 80) |
| `models/backbone.py` | ✅ Written | Same as BASIC (selective unfreezing via config) |
| `models/reconstructor.py` | ✅ Written | MDS + gradient refinement |
| `data/template_loader.py` | ✅ Written | Parses submission.csv + Result.txt → per-target template data |
| `data/dataset.py` | ✅ Written | Returns template_coords + template_confidence per item |
| `data/collate.py` | ✅ Written | Pads template data in batches |
| `data/augmentation.py` | ✅ Written | Random rotation/translation |
| `train.py` | ✅ Written | Discriminative LR, warmup, gradient accumulation, partial weight loading |
| `predict.py` | ✅ Written | Loads templates, concatenates with pairwise, outputs submission.csv |
| `config.yaml` | ✅ Written | All hyperparameters, paths need updating |
| `losses/*.py` | ✅ Written | Distance MSE + bond constraint + clash penalty + TM-score approx |

### What's NOT Done

1. **Config paths need updating** to point to actual Fork 2 result files
2. **Code has NOT been tested** — may have bugs
3. **Training has NOT been run** — no checkpoints exist yet
4. **Kaggle notebook NOT built** — need to combine MMseqs2 + ADV1 inference
5. **No verification** that partial weight loading from BASIC actually works

---

## SECTION 3: CRITICAL DESIGN DECISIONS BASED ON NEW SCORES

### Decision 1: Which Template Source for ADV1 Training?

**Answer: Use Fork 2's (jaejohn's) submission.csv, NOT Fork 1's.**

Rationale:
- Fork 2 scored 0.287 vs Fork 1's 0.17 — jaejohn's coordinates are
  objectively better quality
- The template_loader.py reads submission.csv prediction slot 1 (x_1, y_1, z_1)
- For training data targets that also appear in Fork 2's output, we get
  high-quality template features
- For training data targets NOT in Fork 2's output, template features
  are zeros — the model learns to rely on pairwise features alone

**Config change needed:**
```yaml
template:
  test_template_csv: "../../APPROACH1-TEMPLATE/fork2-JJ/REMOTE/run1/submission-fork2.csv"
  result_txt: "../../APPROACH1-TEMPLATE/fork1-RJ/REMOTE/run1/fork1-Result.txt"
```

Note: Result.txt from Fork 1 is used for confidence scores because Fork 2's
result file is just a log, not actual MMseqs2 output. Fork 1's Result.txt
has the actual e-values we need.

### Decision 2: Should Backbone Be Unfrozen?

**Answer: Yes, but CAUTIOUSLY — keep freeze_first_n=7.**

Rationale:
- BASIC's frozen backbone scored 0.092. The features are NOT optimized
  for 3D prediction.
- Unfreezing last 2 layers (7-8 of 0-8) allows the backbone to adapt
  its pairwise features for distance prediction
- With lr_backbone=1e-5 (10x lower than head), catastrophic forgetting
  is unlikely
- The warmup_fraction=0.05 further protects early training

The current config already has this right:
```yaml
backbone:
  freeze: false
  freeze_first_n: 7
```

### Decision 3: What About max_seq_len=256?

**Answer: Keep at 256 for ADV1 Run 1. Increase later if time permits.**

Rationale:
- 22 of 28 public targets are ≤256 nt — these get full predictions
- 6 targets >256 get truncated + linear extrapolation (same as BASIC)
- Increasing max_seq_len would require retraining and more GPU memory
- The biggest score improvement comes from template features, not seq_len
- Fork 2 handles long sequences through templates anyway

### Decision 4: Training Data Templates Problem

**IMPORTANT ISSUE:** During training, the model learns from PDB structures
(pdb_xyz_data.pkl — 661 structures). These training targets do NOT appear
in Fork 2's submission.csv (which only has the 28 test targets).

This means during training:
- template_coords = zeros for ALL training samples
- template_confidence = zeros for ALL training samples
- The model only learns to use pairwise features during training
- Template features are only non-zero at inference time

**This is actually the original design in the codebase.** From dataset.py:
```python
# During training (Run1): templates are zeros (no training templates yet)
```

**Is this a problem?**

Partially. The template encoder learns its Linear(22,16) projection during
training, but since all inputs are zero during training, the gradients
through template_encoder are always zero. The template encoder weights
stay at their random initialization.

**Mitigation options (ranked by effort):**

**(a) Accept it as-is (lowest effort, what current code does)**
The template features at inference time are just randomly projected
distance bins. Since they're concatenated with pairwise features (which
ARE trained), the distance head can still learn to use both. The template
features provide a "hint" even with random projection weights. The head's
first layer effectively learns to weight the 64 pairwise dimensions
heavily and the 16 template dimensions as noise.

**(b) Pre-train template encoder on training data (medium effort)**
For each training PDB structure, we can compute "self-templates" by using
the true coordinates as if they were template coordinates. This teaches
the template encoder to project real distance patterns into useful features.
Add this as a pre-training step before the main training loop.

**(c) Use training structures as templates for other structures (high effort)**
Run MMseqs2 on training sequences to find homologs within the training set.
Use those as templates during training. This is the most realistic but
requires running MMseqs2 locally on 661 sequences.

**Recommendation: Start with (a), upgrade to (b) if time permits.**
Option (a) is what the code already does. The key value of ADV1 isn't
the template encoder — it's that the Kaggle notebook runs both MMseqs2
AND neural inference end-to-end, producing real predictions for ALL targets.

---

## SECTION 4: DETAILED DESIGN FOR KAGGLE SUBMISSION

### The Final Kaggle Notebook Structure

This is the most critical design — everything converges here.

```
[Phase 1: MMseqs2 Template Search — from Fork 1's pipeline]
  Cell 1: Symlink fix + pip installs
  Cell 2: Install MMseqs2
  Cell 3: Build sequence database from test_sequences.csv
  Cell 4: Run MMseqs2 search against PDB
  Cell 5: Parse results → template coordinates + Result.txt equivalent
  Cell 6: Write intermediate template files
  
  Output: template_coords per target, e-values per target
  Time: ~15-30 minutes

[Phase 2: ADV1 Neural Inference]
  Cell 7: pip install torch, einops (if not available)
  Cell 8: Load RibonanzaNet.pt + adv1_best_model.pt from uploaded datasets
  Cell 9: Load ADV1 code (backbone, distance_head, template_encoder, reconstructor)
  Cell 10: For each test sequence:
    a. Get pairwise features from RibonanzaNet
    b. Get template features from Phase 1 output (or zeros if no template)
    c. Concatenate → distance head → distance matrix → MDS → 3D coords
    d. Generate 5 diverse predictions
  Cell 11: Write raw submission.csv
  
  Output: submission.csv with predictions for all targets
  Time: ~5-15 minutes (GPU)

[Phase 3: Post-Processing — Option B safety net]
  Cell 12: Read sample_submission.csv
  Cell 13: Map ADV1 predictions to expected IDs, fill zeros for any missing
  Cell 14: Write final submission.csv
  
  Output: corrected submission.csv matching sample format exactly
  Time: <1 minute

Total estimated runtime: 20-45 minutes (well within 8-hour limit)
```

### What to Upload as Kaggle Datasets

| Dataset Name | Contents | Size |
|-------------|----------|------|
| `adv1-model-weights` | `adv1_best_model.pt` (ADV1 checkpoint) | ~400 KB |
| `ribonanzanet-weights` | `RibonanzaNet.pt` (pretrained backbone) | 43 MB |
| `adv1-code` | All .py files from ADV1/models/, ADV1/data/, ADV1/utils/, ADV1/losses/ | ~50 KB |
| `ribonanzanet-repo` | Cloned RibonanzaNet/ folder (for backbone loading) | ~10 MB |

### Base Notebook Selection

**Use Fork 1 (rhijudas) as the base**, NOT Fork 2 (jaejohn).

Rationale:
- Fork 1 is built for Part 2 — correct paths, correct competition
- Fork 1's MMseqs2 pipeline is simpler and produces proper Result.txt
- Fork 1 already runs on Kaggle with our symlink + biopython fixes
- Fork 2's pipeline has the multi-chain row count issue we'd have to work around
- We only need the MMseqs2 SEARCH from Fork 1 — template coordinate transfer
  is replaced by ADV1's neural prediction

### Hybrid Prediction Strategy

For the 5 prediction slots in the final submission:

| Slot | Source | Rationale |
|------|--------|-----------|
| Pred 1 | ADV1 (clean, no noise) | Best neural prediction |
| Pred 2 | ADV1 (noise=0.3) | Diverse neural prediction |
| Pred 3 | ADV1 (noise=0.5) | Diverse neural prediction |
| Pred 4 | Fork 2 template (if available) | Best template prediction for strong-template targets |
| Pred 5 | ADV1 (noise=0.7) | Maximum diversity neural prediction |

**Why slot 4 = Fork 2 template:** For the 6 strong-template targets
(0.56-0.92), Fork 2's raw template coordinates are likely BETTER than
ADV1's neural predictions. By putting them in slot 4, best-of-5 scoring
will pick the better one. For no-template targets, slot 4 can be another
ADV1 variant.

---

## SECTION 5: WHAT TO DO — ORDERED TASK LIST

### Phase 1: Verify and Fix ADV1 Code Locally (4-6 hours)

1. **Update config.yaml paths** to point to actual files:
   ```yaml
   template:
     test_template_csv: "../../APPROACH1-TEMPLATE/fork2-JJ/REMOTE/run1/submission-fork2.csv"
     result_txt: "../../APPROACH1-TEMPLATE/fork1-RJ/REMOTE/run1/fork1-Result.txt"
   training:
     basic_checkpoint: "../BASIC/checkpoints/best_model.pt"
   ```

2. **Test template_loader.py** standalone:
   ```bash
   cd ADV1
   python -c "from data.template_loader import load_test_templates; t = load_test_templates('../../APPROACH1-TEMPLATE/fork2-JJ/REMOTE/run1/submission-fork2.csv', '../../APPROACH1-TEMPLATE/fork1-RJ/REMOTE/run1/fork1-Result.txt'); print(len(t), 'targets loaded')"
   ```

3. **Test train.py can start** (even 1 epoch):
   ```bash
   python train.py --config config.yaml
   ```
   Fix any import errors, path errors, shape mismatches.

4. **Run full training** (~2-4 hours on GPU):
   ```bash
   python train.py --config config.yaml
   ```
   Monitor validation loss. It should decrease from BASIC's starting point.

5. **Test predict.py**:
   ```bash
   python predict.py --config config.yaml \
       --checkpoint checkpoints/best_model.pt \
       --test_csv "../../APPROACH1-TEMPLATE/test_sequences (1).csv" \
       --output submission_adv1.csv
   ```

6. **Verify output** matches sample_submission format.

### Phase 2: Build Kaggle Notebook (4-6 hours)

7. **Fork the rhijudas notebook** on Kaggle (or reuse existing fork)

8. **Upload datasets** (RibonanzaNet.pt, adv1_best_model.pt, code files)

9. **Append ADV1 inference cells** after the MMseqs2 search cells

10. **Add Option B post-processing** as safety net at the end

11. **Test in draft mode** with Internet ON

12. **Commit with Internet OFF** and submit

### Phase 3: Iterate (if time permits)

13. Compare ADV1 score vs Fork 2 (0.287)

14. If ADV1 < Fork 2: the template features aren't helping enough.
    Consider:
    - More training epochs
    - Pre-training template encoder (Decision 4, option b)
    - Using Fork 2 predictions directly for strong-template targets

15. If ADV1 > Fork 2: the neural component is adding value.
    Consider:
    - Better diversity settings
    - Hybrid predictions (mix ADV1 + Fork 2 in 5 slots)

---

## SECTION 6: EXPECTED SCORE ESTIMATES

### Conservative Estimate (template encoder untrained, neural adds little)

The model essentially passes through template features with random projection.
For strong-template targets, the template signal dominates → similar to Fork 2.
For no-template targets, the model is similar to BASIC → near zero.
But hidden targets now get SOMETHING instead of zeros.

**Estimated score: 0.20 - 0.30** (similar to or slightly above Fork 2)

### Optimistic Estimate (backbone adaptation helps, templates integrate well)

The unfrozen backbone layers adapt pairwise features for 3D prediction.
Template features provide strong signal for 6 targets.
Neural features provide modest signal for remaining targets.
Hidden targets get real predictions.

**Estimated score: 0.30 - 0.40** (meaningful improvement over Fork 2)

### Reality Check

The top teams (0.44-0.49) likely use much more sophisticated approaches:
- Full RibonanzaNet v2 or custom architectures
- MSA features (we don't use these)
- IPA heads instead of MDS reconstruction
- Recycling iterations
- Ensemble methods
- Much more training data / compute

ADV1 is a solid mid-range approach. Getting above 0.30 would be a
meaningful achievement. Getting above 0.40 would require the beyond-ADV1
improvements listed in BEYOND_ADV1_RUN1.md.

---

## SECTION 7: RISK ASSESSMENT

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| ADV1 code has bugs | HIGH | Blocks everything | Test locally first, fix iteratively |
| Template encoder learns nothing (Decision 4a) | MEDIUM | ADV1 ≈ BASIC for no-template targets | Template features still provide signal via distance bins |
| Backbone unfreezing causes instability | LOW | Training diverges | lr=1e-5 is very cautious; warmup protects early epochs |
| Kaggle notebook exceeds 8 hours | LOW | Submission fails | Budget is ~45 min; well within limit |
| adv1_best_model.pt is too large for Kaggle | LOW | Upload rejected | Checkpoint is ~400KB; well under limits |
| Path/import issues on Kaggle | MEDIUM | Need debugging | We've solved these before (symlink, pip install) |
| ADV1 scores WORSE than Fork 2 | MEDIUM | Wasted effort | Keep Fork 2 as fallback; use hybrid strategy |

---

## SECTION 8: WHAT NOT TO CHANGE IN ADV1 CODE

Per the instruction "DO NOT change anything in the codebase for now":

The following need code changes but should be planned carefully:

1. **config.yaml** — paths need updating (minimal, safe change)
2. **data/dataset.py** — may need fix for how templates are provided during training (Decision 4)
3. **predict.py** — may need hybrid prediction logic (Section 4 slot strategy)
4. **No changes** to template_encoder.py, distance_head.py, backbone.py, train.py logic, losses/

The architecture and training logic are sound as designed. The changes
needed are configuration (paths) and integration (Kaggle notebook), not
fundamental design changes.
