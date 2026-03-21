# ADV1-Run1 — Run Steps

## What This Document Covers

ADV1-Run1 combines the two highest-impact improvements from the ADV1 DESIGN.md:

- **Phase 1 (Option A):** Unfreeze last 2 backbone layers — adapts RibonanzaNet features for 3D prediction
- **Phase 2 (Option D):** Add template features — feeds Approach 1 TBM results into the model

Per the DESIGN.md Decision Matrix (Section 8):
- Option D (template features) has **"Very High"** expected impact and is the **#1 priority**
- Option A (unfreeze layers) has **"Medium"** expected impact and is the **quickest win**
- Combined, they address the two biggest BASIC limitations: frozen features and no template signal

---

## What ADV1-Run1 Does vs BASIC

```
BASIC (what we have now):
  RNA sequence
    → RibonanzaNet (FROZEN backbone) → pairwise features (B, N, N, 64)
    → Distance Head (small MLP) → predicted distance matrix (N×N)
    → MDS + refine → 3D coords

ADV1-Run1 (what we're building):
  RNA sequence
    → RibonanzaNet (LAST 2 LAYERS UNFROZEN) → pairwise features (B, N, N, 64)
    +
  Template coords from Approach 1
    → Template Encoder → template distance features (B, N, N, T)
    +
  [Concatenate pairwise + template features] → (B, N, N, 64 + T)
    → Distance Head (MLP, slightly larger input) → predicted distance matrix (N×N)
    → MDS + refine → 3D coords
```

Key differences from BASIC:
1. Backbone last 2 layers are trainable (adapt features for 3D)
2. Template coordinates are encoded and concatenated with pairwise features
3. Distance head input dimension is larger (64 + template features)
4. Discriminative learning rates: backbone 1e-5, head 1e-4
5. LR warmup for first 5% of epochs (prevents catastrophic updates to unfrozen layers)

---

## What ADV1-Run1 Does NOT Do

These are NOT in Run1 — they are future phases per the DESIGN.md:

| Feature | Phase | Why not in Run1 |
|---------|-------|-----------------|
| IPA structure module | Phase 3 | High complexity, 2-week effort |
| MSA features | Phase 3+ | Requires Evoformer-like attention, high VRAM |
| LoRA adapters | Alternative to Phase 1 | Unfreezing is simpler and more direct |
| Knowledge distillation | Phase 4 | Requires A100 cloud GPU |
| Recycling | Phase 3 | Tied to IPA architecture |

---

## Prerequisites

### From BASIC (must be complete before starting ADV1):

| Item | What | Where |
|------|------|-------|
| Trained distance head | `best_model.pt` from BASIC training | `BASIC/checkpoints/best_model.pt` |
| Working BASIC code | All code files running and tested | `BASIC/` folder |
| Python packages | torch, einops, scipy, etc. all installed | pip environment |
| RibonanzaNet repo | Cloned and working | `Srna3D1/RibonanzaNet/` |
| Pretrained weights | `RibonanzaNet.pt` | `Srna3D1/ribonanza-weights/` |
| Training data pickle | `pdb_xyz_data.pkl` | `Srna3D1/stanford3d-pickle/` |

### From Approach 1 (must run on Kaggle before ADV1 training):

| Item | What | Where to get |
|------|------|--------------|
| Template coordinates for test targets | `submission.csv` or `templates.csv` from MMseqs2 notebook | Run Approach 1 on Kaggle (Step 1 below) |
| Template coordinates for training targets | Same pipeline on training sequences | Run modified Approach 1 on training data |

---

## Step 1: Run Approach 1 on Kaggle to Get Template Coordinates

**This must happen BEFORE building ADV1 code.** You need template results as input.

### 1a: Get templates for TEST targets (required)

1. Open: `https://www.kaggle.com/code/jaejohn/rna-3d-folds-tbm-only-approach`
   (1st place winner — strongest results)
2. Click **"Copy & Edit"**
3. Verify `stanford-rna-3d-folding-2` is attached in the sidebar
4. Settings: **GPU T4 x2**, **Internet ON**
5. Click **"Run All"**
6. Wait 15-30 minutes
7. Go to **Output** tab → download `submission.csv`
8. Rename to `approach1_test_templates.csv`
9. Save to: `Srna3D1/TRY1/APPROACH1-TEMPLATE/approach1_test_templates.csv`

### 1b: Get templates for TRAINING targets (recommended for stronger ADV1)

The training data in `pdb_xyz_data.pkl` has 844 structures. To use templates
during training, you need to run MMseqs2 on the training sequences too.

**Option A (simpler):** Skip training templates for Run1. Only use templates at
inference time. The distance head trains on BASIC-style features only, but at
prediction time, templates are concatenated. This is simpler but weaker.

**Option B (stronger):** Modify the Approach 1 notebook to also process training
sequences. This requires extracting sequences from `pdb_xyz_data.pkl`, converting
to FASTA, and running MMseqs2 against PDB. The training-time template features
teach the model HOW to use templates.

**Recommendation for Run1:** Start with Option A (test-only templates). This
lets you build and test the pipeline quickly. Add training templates later
if time permits.

### 1c: Alternative — use pre-computed templates

The competition organizers provide pre-computed Part 1 templates at:
`https://www.kaggle.com/datasets/rhijudas/rna-3d-folding-templates`

**WARNING:** This dataset was built for Part 1, not Part 2. The test targets
are different. You MUST run the Part 2 notebook (Step 1a) for Part 2 targets.

---

## Step 2: Understand the Template Output Format

After Step 1a, you have `approach1_test_templates.csv` with columns:

```
ID, resname, resid, x_1, y_1, z_1, x_2, y_2, z_2, x_3, y_3, z_3, x_4, y_4, z_4, x_5, y_5, z_5
```

Each row is one nucleotide of one test target. The x/y/z columns are
C1' coordinates from the top-5 template hits. From these coordinates,
you can compute a **template distance matrix** for each target.

For ADV1-Run1, the template encoder needs:
1. Parse the CSV to extract per-target coordinate arrays
2. Compute pairwise distance matrix from template coords: (N, N)
3. Stack top-K templates: (N, N, K) where K = number of templates used
4. Optionally add a "template confidence" channel (e-value from MMseqs2)

---

## Step 3: Create ADV1 Folder Structure

The ADV1 code is built by copying BASIC and adding new files.
**No BASIC files are modified — ADV1 is a separate copy.**

```bash
cd C:\sathya\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH2-RIBBOZANET

# Copy BASIC to ADV1 (preserve BASIC as-is)
xcopy BASIC\*.py ADV1\ /S /Y
xcopy BASIC\models\*.py ADV1\models\ /S /Y
xcopy BASIC\data\*.py ADV1\data\ /S /Y
xcopy BASIC\losses\*.py ADV1\losses\ /S /Y
xcopy BASIC\utils\*.py ADV1\utils\ /S /Y
copy BASIC\config.yaml ADV1\config.yaml
copy BASIC\requirements.txt ADV1\requirements.txt
```

Then create new files per DESIGN.md Section 9:

```
ADV1/
├── DESIGN.md                    ← Already exists
├── ADV1_RUN_STEPS.md            ← THIS document
├── config.yaml                  ← Modified from BASIC (new settings)
│
├── models/
│   ├── backbone.py              ← Modified: selective layer unfreezing
│   ├── distance_head.py         ← Modified: larger input dim for template features
│   ├── template_encoder.py      ← NEW: encode template coords as features
│   └── reconstructor.py         ← Same as BASIC
│
├── data/
│   ├── dataset.py               ← Extended: loads templates alongside structures
│   ├── collate.py               ← Extended: handles template tensors in batch
│   ├── augmentation.py          ← Same as BASIC
│   └── template_loader.py       ← NEW: parses Approach 1 CSV into tensors
│
├── losses/
│   ├── distance_loss.py         ← Same as BASIC
│   ├── constraint_loss.py       ← Same as BASIC
│   └── tm_score_approx.py       ← Same as BASIC
│
├── train.py                     ← Modified: warmup, discriminative LR, resume
├── predict.py                   ← Modified: loads templates at inference
└── utils/
    ├── submission.py            ← Same as BASIC
    └── pdb_parser.py            ← Same as BASIC
```

---

## Step 4: Code Changes Required (What Needs to Be Written)

### 4a: backbone.py — Selective Layer Unfreezing

Per DESIGN.md Option A:
- Freeze layers 0–6 of `model.transformer_encoder` (general RNA features)
- Unfreeze layers 7–8 (last 2 layers) so they adapt to 3D prediction
- Discriminative LR: unfrozen layers get 1e-5, distance head gets 1e-4
- Add config options: `freeze: false`, `freeze_first_n: 7`

Key change: Instead of freezing ALL parameters, iterate through
`model.transformer_encoder` layers and only freeze the first N.

### 4b: template_encoder.py — NEW file

Encodes template coordinates from Approach 1 into features:

```
Input:  Template coords (N, 3) from Approach 1
Output: Template features (N, N, T) where T = template feature dim

Steps:
1. Compute distance matrix from template coords: (N, N)
2. Bin distances into discrete bins (0-2Å, 2-4Å, ..., 20+Å)
3. One-hot encode bins → (N, N, num_bins)
4. Add "template present" mask (1 where template exists, 0 where no template)
5. Project through linear layer → (N, N, T) where T ≈ 16-32
```

This is a simplified version of what RNAPro does (verified from the RNAPro
GitHub: `--use_template ca_precomputed`).

### 4c: template_loader.py — NEW file

Parses Approach 1's CSV output into per-target template tensors:

```
Input:  approach1_test_templates.csv
Output: Dict mapping target_id → template coords array (N, 3)

Steps:
1. Read CSV with pandas
2. Group by target_id (extract from ID column: "8ZNQ_1" → "8ZNQ")
3. For each target, extract (x_1, y_1, z_1) coords → numpy array (N, 3)
4. Return dict: {"8ZNQ": array(N,3), "R1107": array(N,3), ...}
```

### 4d: distance_head.py — Modify input dimension

Per DESIGN.md: concatenate template features with pairwise features.

```
BASIC:  input_dim = pairwise_dim (64)
ADV1:   input_dim = pairwise_dim + template_dim (64 + T)
```

The first linear layer changes from `Linear(64, 128)` to `Linear(64+T, 128)`.
Everything else stays the same.

### 4e: train.py — Warmup + Discriminative LR

Per DESIGN.md Section 5 and Section 6 Phase 1:

```python
# Two parameter groups with different learning rates
optimizer = AdamW([
    {'params': unfrozen_backbone_params, 'lr': 1e-5},    # Backbone (cautious)
    {'params': distance_head.parameters(), 'lr': 1e-4},  # Head (normal)
], weight_decay=0.01)

# Warmup: start at 0, ramp to target LR over first 5% of epochs
scheduler = LinearWarmup + CosineAnnealing
```

### 4f: config.yaml — New settings

```yaml
backbone:
  freeze: false           # Changed from true
  freeze_first_n: 7       # Freeze layers 0-6, unfreeze 7-8

template:
  enabled: true
  test_template_csv: "../../APPROACH1-TEMPLATE/approach1_test_templates.csv"
  train_template_csv: null  # Set this if you have training templates
  feature_dim: 16           # Template feature output dimension
  distance_bins: 22         # Number of distance bins (0-2, 2-4, ..., 40+)

training:
  learning_rate_backbone: 0.00001   # 1e-5 for unfrozen layers
  learning_rate_head: 0.0001        # 1e-4 for distance head
  warmup_fraction: 0.05             # 5% of epochs for LR warmup
  epochs: 50
  gradient_accumulation_steps: 4    # Effective batch size = 4 * 4 = 16
```

---

## Step 5: Initialize Distance Head from BASIC Checkpoint

Per our earlier discussion, BASIC's trained distance head carries forward.

**However:** The distance head input dimension changes from 64 to (64 + T).
So you cannot directly load the BASIC checkpoint into ADV1's distance head.

**Solution — partial weight loading:**
1. Create ADV1 distance head with new input dim (64 + T)
2. Load BASIC checkpoint
3. Copy weights for the first 64 input channels from BASIC
4. Initialize the remaining T input channels randomly
5. This gives the head a warm start while allowing it to learn template features

```python
# Pseudocode for partial weight loading
basic_state = torch.load('BASIC/checkpoints/best_model.pt')
basic_weights = basic_state['model_state_dict']

# First layer weight shape: BASIC = (128, 64), ADV1 = (128, 64+T)
adv1_head.layers[0].weight.data[:, :64] = basic_weights['layers.0.weight']
adv1_head.layers[0].bias.data = basic_weights['layers.0.bias']
# Remaining layers transfer directly (same dimensions)
```

---

## Step 6: Train ADV1-Run1

Once ADV1 code is written and tested:

```bash
cd C:\sathya\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH2-RIBBOZANET\ADV1

python train.py --config config.yaml
```

### Expected differences from BASIC training:

| Metric | BASIC | ADV1-Run1 |
|--------|-------|-----------|
| Time per epoch | ~9.5 min | ~12-15 min (unfrozen layers need gradient computation) |
| VRAM usage | ~3-4 GB | ~6-8 GB (gradients for unfrozen layers) |
| Starting loss | ~800 (from scratch) | ~600 (warm-started from BASIC) |
| Expected final loss | ~350-400 | ~250-350 (better features + templates) |
| batch_size | 4 | 2-4 (higher VRAM → may need to reduce) |

### Resume works the same as BASIC:

```bash
python train.py --config config.yaml --resume checkpoints/latest_model.pt
```

---

## Step 7: Generate Predictions

```bash
python predict.py --config config.yaml \
    --checkpoint checkpoints/best_model.pt \
    --test_csv "../../APPROACH1-TEMPLATE/test_sequences (1).csv" \
    --output submission.csv
```

At inference time, predict.py:
1. Loads test sequence
2. Loads template coords from `approach1_test_templates.csv` for that target
3. Runs backbone (with unfrozen layers in eval mode) → pairwise features
4. Runs template encoder → template features
5. Concatenates pairwise + template features
6. Runs distance head → predicted distances
7. Reconstructs 3D coords via MDS + refinement

---

## Step 8: Hybrid Submission Strategy

Per the DESIGN.md and Approach 1 EXECUTION_GUIDE, the best strategy
combines Approach 1 + ADV1 outputs.

The competition scores **best-of-5** predictions. Use this to your advantage:

```
For each test target:
  Prediction 1: Approach 1 template #1 (best MMseqs2 hit) — direct copy
  Prediction 2: Approach 1 template #2 (2nd best hit) — direct copy
  Prediction 3: ADV1-Run1 prediction (clean, no noise)
  Prediction 4: ADV1-Run1 prediction (with noise for diversity)
  Prediction 5: ADV1-Run1 prediction (different refinement steps)
```

For targets with NO template hits from Approach 1:
```
  Predictions 1-5: All from ADV1-Run1 (with noise diversity, same as BASIC)
```

This gives you the best of both worlds:
- Templates are very accurate when they exist (scored 0.593 alone in Part 1)
- Neural network predictions fill in where templates don't exist
- Best-of-5 scoring means at least one prediction may be close

---

## Step 9: Submit to Kaggle

Same as BASIC:

1. Go to `https://www.kaggle.com/competitions/stanford-rna-3d-folding-2`
2. Click **"Submit Predictions"**
3. Upload `submission.csv`

Or run as a Kaggle notebook (see BASIC_RUN_STEPS.md Kaggle section for details).

---

## VRAM Budget (from DESIGN.md Section 7)

| Component | VRAM |
|-----------|------|
| Frozen backbone forward pass | ~2 GB |
| Unfrozen last 2 layers + gradients | ~3-4 GB |
| Distance head + template encoder | ~0.5 GB |
| Batch data (batch_size=4, seq_len=256) | ~1-2 GB |
| **Total estimate** | **~6-8 GB** |
| Kaggle T4 available | 16 GB ✓ |
| Local GPU (if applicable) | Check with `nvidia-smi` |

If VRAM is tight, reduce `batch_size` to 2 and use `gradient_accumulation_steps: 8`
to maintain effective batch size of 16.

---

## What Carries Forward from BASIC (Reused As-Is)

| BASIC File | Used in ADV1? | How |
|------------|--------------|-----|
| `losses/distance_loss.py` | Yes | Same loss function |
| `losses/constraint_loss.py` | Yes | Same constraints |
| `losses/tm_score_approx.py` | Yes | Same TM-score |
| `models/reconstructor.py` | Yes | Same MDS + refinement |
| `utils/submission.py` | Yes | Same CSV formatting |
| `utils/pdb_parser.py` | Yes | Same CIF parsing |
| `data/augmentation.py` | Yes | Same rotation/translation |
| `checkpoints/best_model.pt` | Yes | Partial weight initialization |

---

## What's NEW in ADV1 (Must Be Written)

| New File | Purpose | Effort |
|----------|---------|--------|
| `models/template_encoder.py` | Encode template coords → features | Medium |
| `data/template_loader.py` | Parse Approach 1 CSV → tensors | Low |
| `models/backbone.py` (modified) | Selective layer unfreezing | Low |
| `models/distance_head.py` (modified) | Larger input dim for templates | Low |
| `data/dataset.py` (modified) | Load template features alongside structures | Medium |
| `data/collate.py` (modified) | Batch template tensors | Low |
| `train.py` (modified) | Warmup, discriminative LR, gradient accum | Medium |
| `predict.py` (modified) | Load templates at inference | Low |
| `config.yaml` (modified) | New settings for templates + unfreezing | Low |

---

## Summary: Complete Run Order

```
1. PREREQUISITE: BASIC training complete (best_model.pt exists)
2. PREREQUISITE: Approach 1 run on Kaggle → approach1_test_templates.csv saved

3. Create ADV1 folder by copying BASIC code
4. Write new files: template_encoder.py, template_loader.py
5. Modify files: backbone.py, distance_head.py, dataset.py, collate.py,
                 train.py, predict.py, config.yaml
6. Smoke test each component
7. Train: python train.py --config config.yaml
8. Predict: python predict.py --config config.yaml --checkpoint checkpoints/best_model.pt ...
9. Build hybrid submission (Approach 1 templates + ADV1 predictions)
10. Submit to Kaggle
```

---

## Status Tracker

| Step | Status | Notes |
|------|--------|-------|
| BASIC training complete | IN PROGRESS | Currently training epoch 14→35 |
| Approach 1 run on Kaggle | NOT STARTED | ~15-30 min, can run in parallel |
| ADV1 folder created | NOT STARTED | Copy from BASIC |
| template_encoder.py written | NOT STARTED | New file |
| template_loader.py written | NOT STARTED | New file |
| backbone.py modified | NOT STARTED | Selective unfreezing |
| distance_head.py modified | NOT STARTED | Larger input dim |
| dataset.py modified | NOT STARTED | Template loading |
| train.py modified | NOT STARTED | Warmup + discriminative LR |
| predict.py modified | NOT STARTED | Template loading at inference |
| config.yaml modified | NOT STARTED | New settings |
| ADV1 smoke tests pass | NOT STARTED | Test each component |
| ADV1 training | NOT STARTED | After all code is ready |
| Hybrid submission built | NOT STARTED | After ADV1 + Approach 1 both done |
| Submitted to Kaggle | NOT STARTED | Final step |

---

## References (Official Documentation Only)

| Source | URL | What it provides |
|--------|-----|-----------------|
| ADV1 DESIGN.md | Local: `ADV1/DESIGN.md` | Full design with options A-F, decision matrix |
| Approach 1 EXECUTION_GUIDE.md | Local: `APPROACH1-TEMPLATE/EXECUTION_GUIDE.md` | How to run TBM on Kaggle |
| RibonanzaNet official repo | `https://github.com/Shujun-He/RibonanzaNet` | Backbone architecture |
| RNAPro (template usage) | `https://github.com/NVIDIA-Digital-Bio/RNAPro` | How templates are used in SOTA |
| DasLab create_templates | `https://github.com/DasLab/create_templates` | Template generation pipeline |
| 1st place TBM notebook | `https://www.kaggle.com/code/jaejohn/rna-3d-folds-tbm-only-approach` | Best template notebook |
| MMseqs2 baseline (Part 2) | `https://www.kaggle.com/code/rhijudas/mmseqs2-3d-rna-template-identification-part-2` | Organizer baseline |
| Competition | `https://www.kaggle.com/competitions/stanford-rna-3d-folding-2` | Rules, data, submission |
| Competition paper | `https://pmc.ncbi.nlm.nih.gov/articles/PMC12776560/` | Scoring methodology |
