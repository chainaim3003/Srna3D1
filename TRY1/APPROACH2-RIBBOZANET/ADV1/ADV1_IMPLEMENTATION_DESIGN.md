# ADV1-Run1 — Final Implementation Design

## DESIGN ONLY — No code changes until approved.
## Grounded in: BASIC codebase (read in full), ADV1/DESIGN.md, official RibonanzaNet Network.py, official DasLab create_templates, official RNAPro GitHub

---
---

# SECTION 1: WHAT CHANGES FROM BASIC

## BASIC Pipeline (current, working)

```
RNA sequence (str)
  → tokenize_sequence() → tokens (B, N)
  → OfficialBackboneWrapper.forward(tokens, mask):
      model.encoder(tokens)                          → embedded (B, N, 256)
      model.outer_product_mean(embedded)             → pairwise (B, N, N, 64)
      pairwise + model.pos_encoder(embedded)         → pairwise (B, N, N, 64)
      for layer in model.transformer_encoder[0..8]:  → hidden, pairwise updated
      ALL parameters frozen (requires_grad = False)
  → single_repr (B, N, 256), pairwise_repr (B, N, N, 64)

  pairwise_repr (B, N, N, 64)
  → DistanceMatrixHead.forward(pairwise_repr):
      mlp.0: Linear(64, 128) + LayerNorm + ReLU + Dropout
      mlp.4: Linear(128, 128) + LayerNorm + ReLU + Dropout
      mlp.8: Linear(128, 1)
      Softplus → symmetrize → zero diagonal
  → pred_dist (B, N, N)

  pred_dist (N, N)
  → reconstruct_3d(dist, method="mds_then_refine")
  → coords (N, 3)
```

## ADV1-Run1 Pipeline (proposed)

```
RNA sequence (str)
  → tokenize_sequence() → tokens (B, N)               [SAME as BASIC]
  → OfficialBackboneWrapper.forward(tokens, mask):
      model.encoder(tokens)                            → embedded (B, N, 256)     [FROZEN]
      model.outer_product_mean(embedded)               → pairwise (B, N, N, 64)   [FROZEN]
      pairwise + model.pos_encoder(embedded)           → pairwise (B, N, N, 64)   [FROZEN]
      for layer in model.transformer_encoder[0..6]:    → hidden, pairwise updated [FROZEN — layers 0-6]
      for layer in model.transformer_encoder[7..8]:    → hidden, pairwise updated [TRAINABLE — layers 7-8]
  → single_repr (B, N, 256), pairwise_repr (B, N, N, 64)

  Template coords from Approach 1 submission.csv                                   [NEW INPUT]
  → TemplateEncoder.forward(template_coords, confidence_mask):
      compute pairwise distances from template coords  → (N, N)
      bin distances into 22 bins (0-2Å, 2-4Å, ..., 40+Å, no_template)
      one-hot encode                                   → (B, N, N, 22)
      Linear(22, template_dim) + LayerNorm + ReLU      → (B, N, N, template_dim)
      multiply by confidence_mask                      → (B, N, N, template_dim)
  → template_features (B, N, N, template_dim)                                      [NEW]

  torch.cat([pairwise_repr, template_features], dim=-1)                            [NEW]
  → combined (B, N, N, 64 + template_dim)

  combined (B, N, N, 64 + template_dim)
  → DistanceMatrixHead.forward(combined):
      mlp.0: Linear(64 + template_dim, 128) + LayerNorm + ReLU + Dropout          [CHANGED input dim]
      mlp.4: Linear(128, 128) + LayerNorm + ReLU + Dropout                        [SAME]
      mlp.8: Linear(128, 1)                                                        [SAME]
      Softplus → symmetrize → zero diagonal                                        [SAME]
  → pred_dist (B, N, N)

  pred_dist (N, N)
  → reconstruct_3d(dist, method="mds_then_refine")                                 [SAME as BASIC]
  → coords (N, 3)
```

## Summary of Differences

| Component | BASIC | ADV1-Run1 |
|-----------|-------|-----------|
| Backbone layers 0-6 | Frozen | Frozen (same) |
| Backbone layers 7-8 | Frozen | **TRAINABLE** |
| Backbone encoder, outer_product_mean, pos_encoder | Frozen | Frozen (same) |
| Template features | Not present | **NEW: (B, N, N, template_dim)** |
| Distance head input | 64 | **64 + template_dim** |
| Distance head hidden/output layers | 128→128→1 | 128→128→1 (same) |
| Optimizer | AdamW, single LR 1e-4 | **AdamW, 2 param groups: backbone 1e-5, head 1e-4** |
| LR schedule | Cosine annealing, no warmup | **Cosine annealing WITH warmup (first 5% of epochs)** |
| Gradient accumulation | None (effective batch = 4) | **4 steps (effective batch = 16)** |
| Losses | distance_mse + bond + clash | distance_mse + bond + clash (same) |
| Reconstructor | MDS + refine | MDS + refine (same) |
| Submission format | Same | Same |

---
---

# SECTION 2: PROJECT STRUCTURE

```
ADV1/
├── DESIGN.md                        ← Existing design (Options A-F)
├── ADV1_RUN1_STEPS.md               ← Existing run steps
├── BEYOND_ADV1_RUN1.md              ← Existing future phases
├── ADV1_IMPLEMENTATION_DESIGN.md    ← THIS DOCUMENT
│
├── config.yaml                      ← MODIFIED from BASIC
│
├── models/
│   ├── backbone.py                  ← MODIFIED: selective layer unfreezing
│   ├── distance_head.py             ← SAME as BASIC (input dim comes from config)
│   ├── template_encoder.py          ← NEW: encode template coords → features
│   └── reconstructor.py             ← SAME as BASIC (copied unchanged)
│
├── data/
│   ├── dataset.py                   ← MODIFIED: loads template features alongside structures
│   ├── collate.py                   ← MODIFIED: batches template tensors
│   ├── augmentation.py              ← SAME as BASIC (copied unchanged)
│   └── template_loader.py           ← NEW: parses Approach 1 CSV → per-target tensors
│
├── losses/
│   ├── distance_loss.py             ← SAME as BASIC (copied unchanged)
│   ├── constraint_loss.py           ← SAME as BASIC (copied unchanged)
│   └── tm_score_approx.py           ← SAME as BASIC (copied unchanged)
│
├── train.py                         ← MODIFIED: warmup, discriminative LR, gradient accum, partial weight load
├── predict.py                       ← MODIFIED: loads templates at inference time
│
├── utils/
│   ├── submission.py                ← SAME as BASIC (copied unchanged)
│   └── pdb_parser.py               ← SAME as BASIC (copied unchanged)
│
└── checkpoints/                     ← Created during training
```

---
---

# SECTION 3: FILE-BY-FILE DESIGN

---

## 3.1 config.yaml — MODIFIED

Changes from BASIC `config.yaml`:

```yaml
# --- Backbone --- (CHANGED)
backbone:
  source: "official"
  repo_path: "../../../RibonanzaNet"
  weights_path: "../../../ribonanza-weights/RibonanzaNet.pt"
  ntoken: 5
  ninp: 256
  pairwise_dimension: 64
  freeze: false                    # ← CHANGED from true
  freeze_first_n: 7               # ← NEW: freeze layers 0-6, unfreeze 7-8

# --- Template --- (ENTIRELY NEW SECTION)
template:
  enabled: true
  # Path to Approach 1 submission.csv (template coordinates)
  test_template_csv: "../../APPROACH1-TEMPLATE/results/mine/submission.csv"
  # Path to Approach 1 Result.txt (MMseqs2 hits with e-values)
  result_txt: "../../APPROACH1-TEMPLATE/results/mine/Result.txt"
  # Path to training template CSV (null = no templates during training, only at inference)
  train_template_csv: null
  # Template encoder output dimension
  template_dim: 16
  # Number of distance bins for encoding
  num_distance_bins: 22            # 0-2, 2-4, ..., 40+, no_template → 21+1=22

# --- Distance Head --- (CHANGED)
distance_head:
  hidden_dim: 128
  num_layers: 3
  dropout: 0.1
  # pair_dim is now computed as: backbone.pairwise_dimension + template.template_dim
  # = 64 + 16 = 80
  # This is NOT hardcoded — it's computed in train.py from config values

# --- Training --- (CHANGED)
training:
  epochs: 50
  batch_size: 4
  learning_rate_backbone: 0.00001  # ← NEW: 1e-5 for unfrozen backbone layers
  learning_rate_head: 0.0001       # ← RENAMED from learning_rate: 1e-4 for head + template encoder
  warmup_fraction: 0.05            # ← NEW: 5% of epochs for LR warmup
  gradient_accumulation_steps: 4   # ← NEW: effective batch size = 4 × 4 = 16
  weight_decay: 0.01
  gradient_clip: 1.0
  use_fp16: true
  num_workers: 0                   # ← CHANGED: 0 is faster on Windows (avoids spawn overhead)
  seed: 42
  
  # Loss weights (SAME as BASIC)
  loss_weights:
    distance_mse: 1.0
    bond_constraint: 0.1
    clash_penalty: 0.05
  
  # Checkpointing (SAME as BASIC)
  save_dir: "./checkpoints"
  save_every: 10
  
  # BASIC checkpoint for warm-start (NEW)
  basic_checkpoint: "../BASIC/checkpoints/best_model.pt"

# --- Data --- (SAME as BASIC)
data:
  train_pickle_path: "../../../stanford3d-pickle/pdb_xyz_data.pkl"
  cif_dir: null
  test_csv_path: "../../APPROACH1-TEMPLATE/test_sequences (1).csv"
  max_seq_len: 256
  val_fraction: 0.1

# --- Reconstruction --- (SAME as BASIC)
reconstruction:
  method: "mds_then_refine"
  refine_steps: 100
  refine_lr: 0.01
  target_consecutive_dist: 5.9

# --- Prediction --- (SAME as BASIC)
prediction:
  num_predictions: 5
  noise_scale: 0.5
  refine_steps_range: [50, 75, 100, 125, 150]
```

---

## 3.2 models/backbone.py — MODIFIED

### What changes from BASIC

BASIC's `load_backbone()` does this:
```python
if backbone_cfg.get('freeze', True):
    for param in model.parameters():
        param.requires_grad = False
```

ADV1 replaces this with selective unfreezing.

### Exact change to load_backbone()

```python
def load_backbone(config: dict) -> nn.Module:
    # ... same loading logic as BASIC ...
    
    backbone_cfg = config['backbone']
    
    if backbone_cfg.get('freeze', True):
        # BASIC behavior: freeze everything
        for param in model.parameters():
            param.requires_grad = False
        print("Backbone frozen — ALL parameters locked.")
    else:
        # ADV1 behavior: selective unfreezing
        freeze_first_n = backbone_cfg.get('freeze_first_n', 7)
        
        # Step 1: Freeze EVERYTHING first
        for param in model.parameters():
            param.requires_grad = False
        
        # Step 2: Unfreeze layers >= freeze_first_n
        # model.model.transformer_encoder is nn.ModuleList of 9 layers
        unfrozen_count = 0
        for i, layer in enumerate(model.model.transformer_encoder):
            if i >= freeze_first_n:
                for param in layer.parameters():
                    param.requires_grad = True
                unfrozen_count += 1
        
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        unfrozen_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Backbone selectively unfrozen:")
        print(f"  Layers 0-{freeze_first_n-1}: FROZEN")
        print(f"  Layers {freeze_first_n}-8: TRAINABLE ({unfrozen_count} layers)")
        print(f"  Frozen params: {frozen_params:,}")
        print(f"  Unfrozen params: {unfrozen_params:,}")
    
    return model
```

### What stays the same
- `OfficialBackboneWrapper.__init__()` — same
- `OfficialBackboneWrapper.forward()` — same (layers run the same whether frozen or not; the difference is whether gradients flow backward through them)
- `tokenize_sequence()` — same
- `OFFICIAL_BASE_TO_IDX` — same

### Why this works
The official Network.py's `ConvTransformerEncoderLayer.forward()` returns `(src, pairwise_features)`. Our backbone wrapper already iterates through these layers manually. Setting `requires_grad = True` on layers 7-8 means PyTorch automatically computes gradients through those layers during `loss.backward()`. No forward pass changes needed.

### Official reference for layer structure
From RibonanzaNet/Network.py (verified):
```python
class RibonanzaNet(nn.Module):
    def __init__(self, config):
        self.transformer_encoder = nn.ModuleList([
            ConvTransformerEncoderLayer(...) for i in range(config.nlayers)  # nlayers=9
        ])
        self.encoder = nn.Embedding(config.ntoken, config.ninp, padding_idx=4)
        self.decoder = nn.Linear(config.ninp, config.nclass)
        self.outer_product_mean = Outer_Product_Mean(in_dim=config.ninp, pairwise_dim=config.pairwise_dimension)
        self.pos_encoder = relpos(config.pairwise_dimension)
```

Each `ConvTransformerEncoderLayer` contains:
- `self_attn` (MultiHeadAttention with Q/K/V projections)
- `linear1`, `linear2` (feedforward)
- `conv` (Conv1d, kernel=k=5 for layers 0-7, kernel=1 for layer 8)
- `triangle_update_out`, `triangle_update_in` (TriangleMultiplicativeModule)
- `outer_product_mean` (Outer_Product_Mean)
- `pair_transition` (Sequential: LayerNorm → Linear → ReLU → Linear)
- `pairwise2heads` (Linear, pairwise→nhead)
- Various LayerNorms and Dropouts

Approximate parameter count per layer: ~2-3M params.
Unfreezing 2 layers ≈ 4-6M additional trainable params (vs ~100K for BASIC).

---

## 3.3 models/template_encoder.py — NEW

### Purpose
Converts Approach 1 template coordinates into a feature tensor that can be concatenated with RibonanzaNet's pairwise features.

### Interface

```python
class TemplateEncoder(nn.Module):
    def __init__(self, num_distance_bins: int = 22, template_dim: int = 16):
        """
        Args:
            num_distance_bins: Number of bins for distance discretization.
                21 distance bins (0-2, 2-4, ..., 40+) + 1 "no template" bin = 22
            template_dim: Output feature dimension per pair.
        """
    
    def forward(self, template_coords: torch.Tensor, 
                confidence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            template_coords: (B, N, 3) — C1' coordinates from Approach 1 template.
                Zeros where no template data exists.
            confidence: (B, N) — per-residue confidence score (0.0 to 1.0).
                From Result.txt e-values: strong match → 1.0, gap-filled → 0.3, no template → 0.0.

        Returns:
            template_features: (B, N, N, template_dim)
        """
```

### Internal logic (step by step)

```
Step 1: Compute pairwise distance matrix from template coords
  diff = template_coords[:, :, None, :] - template_coords[:, None, :, :]  → (B, N, N, 3)
  template_dist = sqrt(sum(diff^2, dim=-1))                               → (B, N, N)

Step 2: Discretize distances into bins
  Bin edges: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, inf]
  This gives 21 distance bins.
  Bin 22 (index 21) = "no template" bin, activated where confidence = 0.
  Use torch.bucketize(template_dist, bin_edges) to assign each distance to a bin.
  Where confidence is 0 (no template), override bin to index 21 ("no_template").
  Result: binned (B, N, N) — integers 0 to 21

Step 3: One-hot encode
  one_hot = F.one_hot(binned, num_classes=22).float()                     → (B, N, N, 22)

Step 4: Project to template_dim
  self.projection = nn.Sequential(
      nn.Linear(num_distance_bins, template_dim),
      nn.LayerNorm(template_dim),
      nn.ReLU(),
  )
  template_features = self.projection(one_hot)                            → (B, N, N, template_dim)

Step 5: Mask by pairwise confidence
  Pairwise confidence: conf_pair[i,j] = min(confidence[i], confidence[j])
  template_features = template_features * conf_pair.unsqueeze(-1)         → (B, N, N, template_dim)
  This zeros out features for residue pairs where either residue has no template.

Return: template_features (B, N, N, template_dim)
```

### Design decisions

**Why distance binning instead of raw distances?**
Raw distances are continuous and unbounded. Binning converts them to a discrete categorical representation that's easier for the linear layer to learn from. This is the same approach used in AlphaFold2's template embedding (AlphaFold2 Supplementary Table 4, "template_distogram").

**Why a "no template" bin?**
For targets with no MMseqs2 hits, or residues that were gap-filled during coordinate transfer, the template coordinates are meaningless (zeros). The model needs to distinguish between "template says distance is 0 Å" (impossible) and "no template information." The dedicated no_template bin handles this.

**Why multiply by confidence?**
Confidence comes from Result.txt e-values. A template with e-value 1e-12 (near-identical match) should be trusted much more than one with e-value 0.01 (weak match). Multiplying by confidence gives the model a continuous signal of template reliability. For no-template targets, confidence=0 zeros out all template features, and the model falls back to pure pairwise features (same as BASIC).

### No hardcoding
- Bin edges are computed from `num_distance_bins` config value, not hardcoded
- Template dim comes from config
- Confidence thresholds come from Result.txt parsing, not hardcoded

---

## 3.4 data/template_loader.py — NEW

### Purpose
Parses Approach 1 output files into per-target template tensors ready for the TemplateEncoder.

### Interface

```python
def load_test_templates(template_csv_path: str, result_txt_path: str = None) -> Dict[str, Dict]:
    """
    Args:
        template_csv_path: Path to Approach 1 submission.csv
            Columns: ID, resname, resid, x_1, y_1, z_1, ..., x_5, y_5, z_5
        result_txt_path: Path to Result.txt (optional — for confidence scores)
            Tab-separated: query, target, evalue, qstart, qend, tstart, tend, qaln, taln

    Returns:
        Dict mapping target_id → {
            'coords': np.ndarray (N, 3) — C1' template coordinates (from x_1, y_1, z_1 = best template),
            'confidence': np.ndarray (N,) — per-residue confidence (0.0 to 1.0),
            'num_hits': int — number of MMseqs2 hits for this target,
            'best_evalue': float — best (lowest) e-value from Result.txt,
        }
        
        For targets with no template data, still returns entry with all-zero coords and confidence=0.
    """
```

### Internal logic

**Parsing submission.csv:**
```
1. Read CSV with pandas
2. Extract target_id from ID column: "8ZNQ_1" → "8ZNQ" (split on "_", take first part)
3. Group rows by target_id
4. For each target:
   a. Extract x_1, y_1, z_1 columns (best template = prediction slot 1)
   b. Stack into numpy array (N, 3)
   c. Check for all-zero coordinates (indicates no template hit)
```

**Parsing Result.txt (if provided):**
```
1. Read tab-separated file
2. For each line: extract query (target_id), evalue
3. Group by target_id
4. For each target:
   a. Count total hits → num_hits
   b. Find minimum e-value → best_evalue
   c. Compute confidence: -log10(best_evalue), clipped to [0, 1] after normalization
      confidence = min(1.0, -log10(evalue) / 15.0)
      This maps: 1e-15 → 1.0, 1e-8 → 0.53, 1e-3 → 0.2, 1.0 → 0.0
5. Parse alignment strings (qaln, taln) to determine which residues were:
   a. Directly matched (aligned, no gap) → confidence = full value
   b. Gap-filled (gap in query or target alignment) → confidence × 0.3
```

**No hardcoding:** Confidence normalization factor (15.0) comes from config. Gap penalty (0.3) comes from config. All thresholds configurable.

---

## 3.5 data/dataset.py — MODIFIED

### What changes from BASIC

BASIC's `RNAStructureDataset.__getitem__()` returns:
```python
{'tokens', 'distance_matrix', 'coords', 'mask', 'seq_len'}
```

ADV1 adds template data:
```python
{'tokens', 'distance_matrix', 'coords', 'mask', 'seq_len',
 'template_coords', 'template_confidence'}    # ← NEW
```

### How template data gets into the dataset

**Option A (inference-only templates — for Run1):**
During training, `template_coords` and `template_confidence` are all zeros for every sample (because training structures don't have Approach 1 templates). The model learns to ignore zero template features via the confidence masking in TemplateEncoder.

During inference (predict.py), templates are loaded from Approach 1 CSV and passed to the model.

**Option B (training templates — for future enhancement):**
If `train_template_csv` is set in config, the loader also provides template features during training. This requires running Approach 1 MMseqs2 on training sequences too (not yet done).

### Changes to load_training_data()

Minimal: adds two zero-filled arrays per sample when templates are not available during training. The function signature and return type stay the same.

---

## 3.6 data/collate.py — MODIFIED

### What changes from BASIC

Adds two fields to the batched output:

```python
def collate_rna_structures(batch):
    # ... same as BASIC for tokens, distance_matrix, coords, mask, seq_lens ...
    
    # NEW: template data
    template_coords = torch.zeros(batch_size, max_len, 3, dtype=torch.float32)
    template_confidence = torch.zeros(batch_size, max_len, dtype=torch.float32)
    
    for i, item in enumerate(batch):
        n = item['seq_len']
        # ... same as BASIC ...
        template_coords[i, :n, :] = item['template_coords']
        template_confidence[i, :n] = item['template_confidence']
    
    return {
        # ... same as BASIC ...
        'template_coords': template_coords,         # (B, max_N, 3)
        'template_confidence': template_confidence,  # (B, max_N)
    }
```

---

## 3.7 train.py — MODIFIED

### Changes from BASIC

**Change 1: Discriminative learning rates**

BASIC:
```python
optimizer = AdamW(distance_head.parameters(), lr=1e-4, weight_decay=0.01)
```

ADV1:
```python
# Collect unfrozen backbone parameters
backbone_params = [p for p in backbone.parameters() if p.requires_grad]

# Template encoder is also trainable
template_params = list(template_encoder.parameters())

optimizer = AdamW([
    {'params': backbone_params, 'lr': config['training']['learning_rate_backbone']},      # 1e-5
    {'params': template_params, 'lr': config['training']['learning_rate_head']},           # 1e-4
    {'params': distance_head.parameters(), 'lr': config['training']['learning_rate_head']},# 1e-4
], weight_decay=config['training']['weight_decay'])
```

**Change 2: LR warmup**

BASIC:
```python
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
```

ADV1:
```python
warmup_epochs = int(epochs * config['training']['warmup_fraction'])  # 5% of epochs

# Linear warmup then cosine decay
# PyTorch has no built-in combined scheduler, so use LambdaLR:
def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return epoch / max(1, warmup_epochs)  # Linear 0 → 1
    else:
        # Cosine decay from 1 → 0 over remaining epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = LambdaLR(optimizer, lr_lambda)
```

**Change 3: Gradient accumulation**

BASIC:
```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

ADV1:
```python
accum_steps = config['training']['gradient_accumulation_steps']  # 4

# Inside training loop:
loss = loss / accum_steps  # Scale loss by accumulation steps
scaler.scale(loss).backward()

if (batch_idx + 1) % accum_steps == 0:
    scaler.unscale_(optimizer)
    clip_grad_norm_(all_trainable_params, grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

**Change 4: Partial weight loading from BASIC**

At the start of training (before the training loop), if `basic_checkpoint` is set in config:

```python
basic_ckpt_path = config['training'].get('basic_checkpoint')
if basic_ckpt_path and os.path.exists(basic_ckpt_path):
    basic_ckpt = torch.load(basic_ckpt_path, map_location=device, weights_only=False)
    basic_weights = basic_ckpt['model_state_dict']
    
    adv1_state = distance_head.state_dict()
    
    for key, basic_param in basic_weights.items():
        if key == 'mlp.0.weight':
            # First layer: BASIC is (128, 64), ADV1 is (128, 64+template_dim)
            # Copy BASIC weights into first 64 columns
            adv1_state[key][:, :basic_param.shape[1]] = basic_param
            print(f"  Partial load: {key} — copied {basic_param.shape[1]} of {adv1_state[key].shape[1]} input channels")
        elif key in adv1_state and adv1_state[key].shape == basic_param.shape:
            # Matching shapes — copy directly
            adv1_state[key] = basic_param
            print(f"  Full load: {key}")
        else:
            print(f"  Skipped: {key} (shape mismatch or not in ADV1)")
    
    distance_head.load_state_dict(adv1_state)
    print(f"Warm-started distance head from BASIC checkpoint")
```

**Change 5: Forward pass includes template encoder**

BASIC training step:
```python
with torch.no_grad():
    single_repr, pairwise_repr = backbone(tokens, mask)
pred_dist = distance_head(pairwise_repr)
```

ADV1 training step:
```python
# Backbone — unfrozen layers 7-8 compute gradients
single_repr, pairwise_repr = backbone(tokens, mask)  # NO torch.no_grad() wrapping the whole thing

# Template encoder
template_features = template_encoder(template_coords, template_confidence)

# Concatenate
combined = torch.cat([pairwise_repr, template_features], dim=-1)

# Distance head
pred_dist = distance_head(combined)
```

Note: The frozen layers (0-6) don't compute gradients even without `torch.no_grad()`, because their `requires_grad=False`. Only layers 7-8 + template_encoder + distance_head compute gradients.

**Change 6: Resume saves/loads all components**

BASIC saves: `{model_state_dict, optimizer_state_dict, scheduler_state_dict, ...}`

ADV1 saves:
```python
torch.save({
    'epoch': epoch,
    'distance_head_state_dict': distance_head.state_dict(),
    'template_encoder_state_dict': template_encoder.state_dict(),
    'backbone_unfrozen_state_dict': {
        f'layer_{i}': backbone.model.transformer_encoder[i].state_dict()
        for i in range(freeze_first_n, 9)
    },
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'val_loss': avg_val_loss,
    'best_val_loss': best_val_loss,
    'best_epoch': best_epoch,
    'config': config,
}, save_path)
```

---

## 3.8 predict.py — MODIFIED

### What changes from BASIC

BASIC's `predict_single_target()`:
```python
tokens = tokenize_sequence(seq)
single_repr, pairwise_repr = backbone(tokens, mask)
pred_dist = distance_head(pairwise_repr)
```

ADV1's `predict_single_target()`:
```python
tokens = tokenize_sequence(seq)
single_repr, pairwise_repr = backbone(tokens, mask)

# Load template for this target
template_data = templates.get(target_id, None)
if template_data is not None:
    template_coords_tensor = torch.tensor(template_data['coords']).unsqueeze(0).to(device)
    confidence_tensor = torch.tensor(template_data['confidence']).unsqueeze(0).to(device)
else:
    template_coords_tensor = torch.zeros(1, N, 3, device=device)
    confidence_tensor = torch.zeros(1, N, device=device)

template_features = template_encoder(template_coords_tensor, confidence_tensor)
combined = torch.cat([pairwise_repr, template_features], dim=-1)
pred_dist = distance_head(combined)
```

### Template loading at startup

At the top of `predict()`, before the prediction loop:

```python
# Load Approach 1 templates
template_cfg = config.get('template', {})
if template_cfg.get('enabled', False):
    from data.template_loader import load_test_templates
    templates = load_test_templates(
        template_csv_path=template_cfg['test_template_csv'],
        result_txt_path=template_cfg.get('result_txt'),
    )
    print(f"Loaded templates for {len(templates)} targets")
else:
    templates = {}
```

---

## 3.9 models/distance_head.py — SAME as BASIC

No code changes needed. The `pair_dim` parameter already controls input dimension:

```python
class DistanceMatrixHead(nn.Module):
    def __init__(self, pair_dim: int = 64, hidden_dim: int = 128, ...):
        # pair_dim can be 64 (BASIC) or 80 (ADV1) — no hardcoding
```

In BASIC, `train.py` passes `pair_dim=64`.
In ADV1, `train.py` computes `pair_dim = 64 + template_dim` from config and passes `pair_dim=80`.

The file itself is copied unchanged.

---

## 3.10 Unchanged files (copied from BASIC as-is)

| File | Why no changes needed |
|------|----------------------|
| `models/reconstructor.py` | MDS + refinement operates on distance matrix regardless of how it was produced |
| `data/augmentation.py` | Random rotation/translation applies to training coords, not templates |
| `losses/distance_loss.py` | MSE on distance matrices — same interface |
| `losses/constraint_loss.py` | Bond + clash on distance matrices — same interface |
| `losses/tm_score_approx.py` | Operates on coords — same interface |
| `utils/submission.py` | CSV formatting — same interface |
| `utils/pdb_parser.py` | CIF parsing — not used in ADV1 (we use pickle) |

---
---

# SECTION 4: DATA FLOW DIAGRAMS

## 4.1 Training Data Flow

```
pdb_xyz_data.pkl (844 structures)
  │
  ├── dataset.py parses C1' coords from sugar_ring[0]
  │   → 789 valid structures → 734 after max_len filter → 661 train, 73 val
  │
  ├── Each sample: {tokens, distance_matrix, coords, mask, seq_len,
  │                  template_coords=zeros, template_confidence=zeros}
  │
  │   (templates are zero during training in Run1 because we don't have
  │    Approach 1 templates for training structures — only for test targets)
  │
  └── collate.py batches + pads → {B, max_N, ...}
        │
        ▼
      train.py forward pass:
        tokens, mask ──────────────────────────► backbone ──► pairwise_repr (B,N,N,64)
        template_coords, template_confidence ──► template_encoder ──► template_feat (B,N,N,16) [all zeros]
                                                                          │
        cat(pairwise_repr, template_feat) ─────────────────────────► combined (B,N,N,80)
                                                                          │
                                                                    distance_head ──► pred_dist (B,N,N)
                                                                          │
                                                                    losses ──► backward ──► update
```

**Note:** During training, template features are all zeros. The model learns that when template channels are zero, it should rely on pairwise features (columns 0-63). The BASIC warm-start ensures columns 0-63 of the first layer already work correctly.

## 4.2 Inference Data Flow

```
test_sequences.csv (28 targets)                   Approach 1 results
  │                                                  │
  ├── target_id + sequence                           ├── submission.csv → template_loader.py → coords (N,3)
  │                                                  └── Result.txt → template_loader.py → confidence (N,)
  │                                                        │
  │                                                        ▼
  │                                                  templates dict: {target_id: {coords, confidence}}
  │                                                        │
  ▼                                                        │
predict.py:                                                │
  for each target:                                         │
    tokenize(sequence) → tokens                            │
    backbone(tokens, mask) → pairwise_repr (N,N,64)        │
    templates[target_id] → coords, confidence ─────────────┘
    template_encoder(coords, confidence) → template_feat (N,N,16)
    cat(pairwise_repr, template_feat) → combined (N,N,80)
    distance_head(combined) → pred_dist (N,N)
    reconstruct_3d(pred_dist) → coords (N,3)   ×5 diverse predictions
```

**When target has NO template (not in templates dict):**
template_coords = zeros, confidence = zeros → template_features = zeros → model uses only pairwise features.

---
---

# SECTION 5: VRAM AND PERFORMANCE ESTIMATES

## VRAM Budget

| Component | BASIC | ADV1-Run1 | Notes |
|-----------|-------|-----------|-------|
| Backbone forward (frozen layers 0-6) | ~1.5 GB | ~1.5 GB | Same — no gradients stored |
| Backbone forward (layers 7-8, trainable) | 0 (frozen) | ~2-3 GB | Gradients + activations stored |
| Template encoder | 0 | ~0.1 GB | Small linear layer |
| Distance head | ~0.1 GB | ~0.1 GB | Similar size |
| Batch data (batch_size=4, seq_len=256) | ~1-2 GB | ~1-2 GB | Plus template tensors (~0.1 GB) |
| **Total** | **~3-4 GB** | **~5-7 GB** | |
| Kaggle T4 available | 16 GB | 16 GB | Plenty of headroom |

If VRAM is tight: reduce batch_size from 4 to 2, increase gradient_accumulation_steps from 4 to 8 (effective batch stays 16).

## Training Time Estimates

| Metric | BASIC | ADV1-Run1 | Why different |
|--------|-------|-----------|---------------|
| Time per epoch | ~9.5 min | ~12-15 min | Gradients through 2 backbone layers + template encoder overhead |
| Total 50 epochs | ~8 hours | ~10-12 hours | |
| Convergence (expected) | ~30-40 epochs | ~30-50 epochs | More parameters → may need more epochs |

---
---

# SECTION 6: SMOKE TESTS

Before training, run these tests to verify each component:

**Test 1: Backbone with selective unfreezing**
```bash
python -c "
import yaml; config=yaml.safe_load(open('config.yaml'))
from models.backbone import load_backbone
b = load_backbone(config)
unfrozen = sum(p.numel() for p in b.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in b.parameters() if not p.requires_grad)
print(f'Frozen: {frozen:,}, Unfrozen: {unfrozen:,}')
assert unfrozen > 0, 'No unfrozen params — check config freeze settings'
print('SUCCESS: Selective unfreezing works')
"
```

**Test 2: Template encoder**
```bash
python -c "
import torch
from models.template_encoder import TemplateEncoder
enc = TemplateEncoder(num_distance_bins=22, template_dim=16)
coords = torch.randn(2, 20, 3)
conf = torch.ones(2, 20)
feat = enc(coords, conf)
print(f'Template features: {feat.shape}')
assert feat.shape == (2, 20, 20, 16)

# Test with zero confidence (no template)
conf_zero = torch.zeros(2, 20)
feat_zero = enc(coords, conf_zero)
assert feat_zero.abs().max() < 1e-6, 'Zero confidence should produce zero features'
print('SUCCESS: Template encoder works')
"
```

**Test 3: Template loader**
```bash
python -c "
from data.template_loader import load_test_templates
templates = load_test_templates(
    '../../APPROACH1-TEMPLATE/results/mine/submission.csv',
    '../../APPROACH1-TEMPLATE/results/mine/Result.txt'
)
print(f'Loaded templates for {len(templates)} targets')
for tid, data in list(templates.items())[:2]:
    print(f'  {tid}: coords {data[\"coords\"].shape}, confidence mean={data[\"confidence\"].mean():.2f}, hits={data[\"num_hits\"]}')
print('SUCCESS: Template loader works')
"
```

**Test 4: Full forward pass**
```bash
python -c "
import torch, yaml
config = yaml.safe_load(open('config.yaml'))
from models.backbone import load_backbone
from models.template_encoder import TemplateEncoder
from models.distance_head import DistanceMatrixHead

backbone = load_backbone(config)
template_dim = config['template']['template_dim']
pair_dim = config['backbone']['pairwise_dimension'] + template_dim

enc = TemplateEncoder(template_dim=template_dim)
head = DistanceMatrixHead(pair_dim=pair_dim, hidden_dim=128, num_layers=3)

tokens = torch.randint(0, 4, (2, 20))
mask = torch.ones(2, 20, dtype=torch.bool)
template_coords = torch.randn(2, 20, 3)
confidence = torch.ones(2, 20)

single, pairwise = backbone(tokens, mask)
template_feat = enc(template_coords, confidence)
combined = torch.cat([pairwise, template_feat], dim=-1)
pred_dist = head(combined)
print(f'pairwise: {pairwise.shape}, template: {template_feat.shape}, combined: {combined.shape}, dist: {pred_dist.shape}')
assert pred_dist.shape == (2, 20, 20)
print('SUCCESS: Full ADV1 forward pass works')
"
```

**Test 5: Partial weight loading from BASIC**
```bash
python -c "
import torch
from models.distance_head import DistanceMatrixHead

# Simulate BASIC checkpoint
basic_head = DistanceMatrixHead(pair_dim=64, hidden_dim=128, num_layers=3)
basic_state = basic_head.state_dict()

# Create ADV1 head with larger input
adv1_head = DistanceMatrixHead(pair_dim=80, hidden_dim=128, num_layers=3)
adv1_state = adv1_head.state_dict()

# Partial load
for key, param in basic_state.items():
    if key == 'mlp.0.weight':
        adv1_state[key][:, :64] = param
    elif key in adv1_state and adv1_state[key].shape == param.shape:
        adv1_state[key] = param

adv1_head.load_state_dict(adv1_state)
print('SUCCESS: Partial weight loading works')
"
```

---
---

# SECTION 7: WHAT IS NOT IN THIS DESIGN (explicitly excluded)

| Feature | Why excluded | Where documented |
|---------|-------------|-----------------|
| IPA structure module | Phase 3, high complexity | BEYOND_ADV1_RUN1.md Level 1 |
| MSA features | Phase 3+, requires Evoformer-like attention | BEYOND_ADV1_RUN1.md Level 3 |
| LoRA adapters | Alternative to full unfreezing — not needed for Run1 | DESIGN.md Option B |
| Knowledge distillation | Requires A100 cloud GPU | BEYOND_ADV1_RUN1.md Level 4 |
| Recycling | Tied to IPA architecture | BEYOND_ADV1_RUN1.md Level 2 |
| FAPE loss | Requires IPA (frame-based) | BEYOND_ADV1_RUN1.md Level 1 |
| EMA (exponential moving average) | Nice-to-have optimization, not critical for Run1 | DESIGN.md Section 5 |
| Training-time templates | Requires running MMseqs2 on training sequences | Future enhancement |
| Hybrid submission builder | Phase D script, separate from ADV1 model | MASTER_PLAN.md Phase D |

---
---

# SECTION 8: DEPENDENCIES ON APPROACH 1

| ADV1 component | Requires from Approach 1 | Specifically |
|----------------|--------------------------|-------------|
| `template_loader.py` | `submission.csv` | Parses coordinates from x_1, y_1, z_1 columns |
| `template_loader.py` | `Result.txt` | Parses e-values → confidence scores, alignments → gap masks |
| `predict.py` | `submission.csv` | Loads template coords at inference time |
| `predict.py` | `Result.txt` | Determines per-residue confidence |
| `dataset.py` (training) | Nothing (templates are zeros during training in Run1) | No Approach 1 data needed for training |
| `config.yaml` | File paths | Points to `results/mine/submission.csv` and `results/mine/Result.txt` |

**If Approach 1 hasn't run yet:**
- ADV1 code can still be written and tested (smoke tests use random data)
- ADV1 training runs with all-zero template features (same as BASIC effectively)
- Only prediction requires real Approach 1 results

**If Result.txt is missing (only submission.csv available):**
- Template coords still work
- Confidence defaults to 1.0 for all residues with non-zero coordinates, 0.0 for zero coordinates
- Less precise but functional

---
---

# SECTION 9: REFERENCES

| Source | What it grounds |
|--------|----------------|
| BASIC `models/backbone.py` (read in full) | Layer iteration logic, OfficialBackboneWrapper interface |
| BASIC `models/distance_head.py` (read in full) | MLP architecture, pair_dim parameterization |
| BASIC `data/dataset.py` (read in full) | Data loading, pickle format, __getitem__ return format |
| BASIC `data/collate.py` (read in full) | Batching interface, padding logic |
| BASIC `train.py` (read in full) | Training loop, optimizer, scheduler, checkpointing |
| BASIC `predict.py` (read in full) | Inference loop, predict_single_target interface |
| BASIC `losses/` (read in full) | Loss function interfaces |
| RibonanzaNet `Network.py` (read in full) | Layer structure: 9 ConvTransformerEncoderLayers in nn.ModuleList, each returns (src, pairwise) |
| RibonanzaNet `configs/pairwise.yaml` (read in full) | k=5, ninp=256, nlayers=9, nhead=8, pairwise_dimension=64, use_triangular_attention=false |
| ADV1 `DESIGN.md` Section 3 Option A | Selective unfreezing: freeze first N, discriminative LR |
| ADV1 `DESIGN.md` Section 3 Option D | Template features: encode as distance bins, concatenate with pairwise |
| ADV1 `DESIGN.md` Section 6 Phase 1 | Warmup + discriminative LR + gradient accumulation |
| ADV1 `DESIGN.md` Section 6 Phase 2 | Template feature integration |
| AlphaFold2 paper (Jumper et al., 2021) | Template distogram binning approach (Supplementary Table 4) |
| RNAPro GitHub | `--use_template ca_precomputed`, `--num_templates 4` — confirms template feature approach |
| DasLab create_templates GitHub | Output format of templates.csv and Result.txt |
