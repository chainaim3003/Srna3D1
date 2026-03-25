# HY-BAS-ADV1 Run 5-1 IPA — Design Document

## Status: DESIGN PHASE (not yet coded)

---

## Locked Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| IPA source code | **Inline B** — copy-paste lucidrains' proven library source into Cell 10 | Zero wheel dependency. No pip install. No Kaggle dataset needed. Same proven code. No risk of rewriting rotation math from scratch. |
| n_iterations | **4 (shared weights)** | Conservative VRAM. Shared weights = change to 6 or 8 in later runs with zero checkpoint friction. |
| Weight sharing | **Shared** — same IPA layer reused each iteration | 661 training structures too few for unshared (would be 600K+ params). Shared = 150K params. Free iteration changes between runs. |
| FAPE loss | **Yes** + 0.1× auxiliary distance MSE | FAPE is rotationally invariant, catches mirror-flips. Aux MSE guides early training. |
| BATCH_SIZE | **1** | VRAM safety for IPA activations across 4 iterations. |
| Teacher forcing | **50%** | Half the time start from true coords + noise, half from template/zeros. |
| Inference fallback | **Option A — deleted.** `refine_coords()` and `DistanceMatrixHead` are fully removed. | Template slots 1-2 are the safety net if IPA is bad. Keeping two parallel inference paths adds complexity and a confusing threshold decision. IPA either works or the templates carry the score. |

---

## ⚠️ CRITICAL OPEN ISSUE: TRAIN_EPOCHS vs Kaggle Time Limit

**Status: OPEN — must be resolved before coding Cell 13.**

### The Problem

Run 4 used `TRAIN_EPOCHS = 70` with a simple MLP distance head.
IPA is fundamentally heavier per epoch. On Kaggle T4 (9 hr limit),
70 epochs of IPA training is very likely unreachable.

### Why IPA Is Slower Per Epoch Than Run 4

| Step | Run 4 | Run 5 IPA | Slowdown |
|------|-------|-----------|---------|
| Backbone forward | same | same | 1× |
| Head computation | Tiny MLP over pair features | 4× IPABlock (full 3D point attention) | ~5-8× |
| Loss computation | MSE scalar — trivial | FAPE: O(N²) frame transforms per residue | ~3-5× |
| Backward pass | Through small MLP only | Through 4 IPA iterations + frame updates | ~4-6× |

**Estimated epoch time on T4:**
- Run 4: ~10-20 min/epoch (MLP head, simple loss)
- Run 5 IPA: ~30-70 min/epoch (IPA head, FAPE loss)
- Most likely: ~45 min/epoch (midpoint estimate)

**Estimated epochs achievable on Kaggle T4 (9 hr total, ~35 min overhead):**
- At 30 min/epoch → ~16 epochs max
- At 45 min/epoch → ~10-11 epochs max
- At 70 min/epoch → ~7 epochs max

**The proposal of TRAIN_EPOCHS = 70 is almost certainly unreachable on Kaggle T4.**

### Is 10-16 Epochs Enough for IPA to Learn?

This is the critical unknown. Arguments on both sides:

**Reasons it might be enough:**
- Backbone is warm-started from Run 4 — already produces good pairwise
  features that encode structural information. IPA only needs to learn
  to read coordinates out of features that already encode structure.
- 175K new params is tiny. Small models converge in fewer epochs.
- Training set is only 661 structures. Small datasets converge faster.
- Val loss plateau may be reached at epoch 15-20, not epoch 50-70.

**Reasons it might NOT be enough:**
- IPA starts from random weights and must learn 3D rotation math from scratch.
- FAPE loss is complex — may need many epochs to escape a bad local minimum.
- RNA structures are geometrically complex — may need more signal than proteins.

**Bottom line: We do not know until we run it and watch the val_loss curve.**
The first 10 epochs of training will tell us everything.

### Three Strategies to Solve This (ranked by effort)

---

#### Strategy 1: Freeze backbone entirely during IPA training (RECOMMENDED FIRST TRY)

Keep backbone 100% frozen. Only the 175K new IPA params update.
This eliminates backbone gradient computation entirely — roughly
**2-3× speedup** per epoch.

Estimated epoch time with frozen backbone: ~15-25 min
Estimated epochs achievable: ~20-30 epochs

**Trade-off:** Backbone does not adapt to IPA's needs. IPA must learn
to work with whatever features Run 4's backbone already produces.
This may actually be fine — the backbone is already good.

**Implementation:** One-line change in Cell 13:
```python
FREEZE_BACKBONE_DURING_IPA = True  # New flag
# In training loop: skip backbone.unfreeze() step entirely
```

This is the lowest-risk, fastest-to-implement solution.
Try this first. If 20-30 epochs converges, we're done.

---

#### Strategy 2: Train outside Kaggle, upload checkpoint (BEST QUALITY)

Train the full IPA module to convergence (50-100 epochs) on external
hardware, save checkpoint, upload to Kaggle as a dataset, then the
Kaggle notebook loads checkpoint and runs inference only.

**Workflow:**
```
External machine:
  python train_run5_ipa.py  # runs 50-100 epochs, saves best checkpoint

Upload to Kaggle:
  Create dataset "hy-bas-adv1-run5-checkpoint"
  Upload adv1_run5_ipa_best.pt

Kaggle notebook (Run 5):
  Cell 3:  Load from /kaggle/input/hy-bas-adv1-run5-checkpoint/
  Cell 13: SKIP entirely (training already done)
  Cell 14: Inference only → submit
  Total Kaggle time: < 30 min
```

This is the approach top Kaggle teams use for heavy models.
Training time moves off Kaggle entirely. Kaggle is inference-only.

**Hardware options for external training:**

| Option | GPU | VRAM | Est. min/epoch | 70 epochs | Cost | Reliability |
|--------|-----|------|---------------|-----------|------|-------------|
| Google Colab Free | T4 | 16 GB | ~45 min | ~52 hrs | Free | Poor — disconnects at ~8-12 hrs |
| Google Colab Pro | A100 | 40 GB | ~12 min | ~14 hrs | ~$10-12 | Medium — A100 not guaranteed |
| Google Colab Pro+ | A100 | 40 GB | ~12 min | ~14 hrs | ~$50/mo | Good — background execution |
| Vast.ai rental | A100 80GB | 80 GB | ~8 min | ~9 hrs | ~$3-5 total | Very good — dedicated GPU |
| RunPod | A100 | 40-80 GB | ~10 min | ~12 hrs | ~$3-5 total | Very good — dedicated GPU |
| Lambda Labs | A100 | 40 GB | ~10 min | ~12 hrs | ~$3-5 total | Very good |
| Local RTX 3090 | consumer | 24 GB | ~20 min | ~23 hrs | hardware owned | Excellent |
| Local RTX 4090 | consumer | 24 GB | ~15 min | ~18 hrs | hardware owned | Excellent |

**Why A100 specifically:**
- 40-80 GB VRAM vs T4's 16 GB — no OOM risk even at larger batch sizes
- ~3-4× faster than T4 on attention-heavy workloads (IPA is attention)
- Can train at batch size > 1 — further speedup

**Why Vast.ai / RunPod over Colab for a long run:**
- Colab disconnects after ~8-12 hrs even on Pro
- 70 epochs at 12 min/epoch = 14 hrs — likely to disconnect mid-run
- Vast.ai / RunPod give a dedicated instance that does not disconnect
- Total cost for one 10-hour training run: under $5

**Colab Pro is borderline viable** only if we implement checkpoint
saving every 5 epochs with resume logic. Then a disconnect only
costs 5 epochs of work, not the whole run.

---

#### Strategy 3: Stage training across multiple Kaggle sessions (FREE BUT PAINFUL)

Train 10 epochs → Kaggle saves output checkpoint → next notebook loads
that checkpoint and trains 10 more → repeat 7 times for 70 epochs.

```
Session 1:  epochs 1-10  → output: run5_ckpt_epoch10.pt
Session 2:  epochs 11-20 → output: run5_ckpt_epoch20.pt
...
Session 7:  epochs 61-70 → output: run5_ckpt_epoch70.pt  ← FINAL
```

**Trade-off:** Very tedious. Requires manual intervention 7 times.
Each session needs ~10 min setup. But it is completely free and
requires no external hardware.

This is the fallback if Strategies 1 and 2 are not viable.

---

### Decision Tree: Which Strategy to Use

```
Start here:
  ↓
Does frozen backbone (Strategy 1) achieve
val_loss convergence in 20-30 epochs?
  ├─ YES → Use Strategy 1. Done.
  └─ NO  → val_loss still dropping at epoch 20-30, needs more
       ↓
  Is external hardware (A100 cloud) available for ~$5?
  ├─ YES → Use Strategy 2 (external train + upload checkpoint). Best quality.
  └─ NO  → Use Strategy 3 (staged Kaggle sessions). Free but manual.
```

**Recommended order of attempts:**
1. **First:** Try Strategy 1 (frozen backbone). Lowest effort, no cost.
   Watch val_loss at epochs 5, 10, 15, 20. If converging → done.
2. **If plateau not reached by epoch 20:** Switch to Strategy 2.
   Rent A100 on Vast.ai (~$3-5), train to convergence, upload checkpoint.
3. **If no external hardware available:** Fall back to Strategy 3.

---

### Early Stopping as a Safety Valve

Regardless of which strategy is used, implement early stopping
in Cell 13:

```python
EARLY_STOP_PATIENCE = 10  # Stop if val_loss doesn't improve for 10 epochs
```

This means if val_loss converges at epoch 18, training stops at
epoch 28 automatically — no wasted compute, no manual intervention.
This is critical for both Kaggle time management and cloud cost control.

---

### Key Monitoring Point: First 10 Epochs

**Watch these metrics after the first 10 epochs of any training run:**

| Metric | Good sign | Bad sign |
|--------|-----------|----------|
| val_loss trend | Steadily decreasing | Flat or increasing |
| val_loss value | < 5.0 Å FAPE | > 20.0 Å FAPE |
| NaN/Inf in loss | Never appears | Appears → FAPE bug |
| Epoch time | < 30 min on T4 | > 60 min → OOM risk |
| GPU memory | < 14 GB | > 15 GB → reduce n_iterations |

If val_loss is still > 20 Å after 10 epochs and not clearly
decreasing, there is likely a bug in frame initialization or
FAPE loss — stop and debug before continuing.

---

## IPA Library Reference

### Primary: lucidrains/invariant-point-attention (USED — Inline B)

- **Repository:** https://github.com/lucidrains/invariant-point-attention
- **PyPI:** https://pypi.org/project/invariant-point-attention/
- **Version:** 0.2.2 (latest, released Nov 2, 2022)
- **License:** MIT
- **Author:** Phil Wang (lucidrains)
- **Dependencies:** `einops>=0.3` (pre-installed on Kaggle), `torch>=1.7` (pre-installed)
- **Core code:** ~300 lines of PyTorch in one file
- **What it provides:**
  - `InvariantPointAttention` — the core IPA attention layer
  - `IPABlock` — IPA + feedforward + layernorm (complete transformer block)
  - `IPATransformer` — multi-layer IPA with frame updates (full structure module)

### Inline B Approach

Copy source of `InvariantPointAttention` and `IPABlock` directly
into Cell 10. Marked clearly:

```python
# ============================================================
# BEGIN: Copied from lucidrains/invariant-point-attention v0.2.2
# Source: https://github.com/lucidrains/invariant-point-attention
# License: MIT
# DO NOT MODIFY this section — it is proven library code.
# ============================================================
```

### Alternative Libraries (NOT used, documented for reference)

| Library | Why Not Used |
|---------|-------------|
| `Invariant-Attention` (Rishit Dagli) | TensorFlow, not PyTorch |
| OpenFold IPA module | Tightly coupled to OpenFold, hard to extract |
| RhoFold+ structure module | RNA-specific but requires RhoFold dependencies |
| Protenix (RNAPro) | Too heavy, NVIDIA ecosystem |

### Alternative Delivery: pip wheel (NOT used, reference only)

```bash
pip download invariant-point-attention --no-deps -d ./wheels/
# Upload .whl as Kaggle dataset, install in Cell 1
```
Not chosen. Inline B achieves same result with zero overhead.

---

## Quick Library Validation Test

```python
!pip install invariant-point-attention -q
import torch
from einops import repeat
from invariant_point_attention import InvariantPointAttention, IPABlock

attn = InvariantPointAttention(dim=64, heads=4,
    scalar_key_dim=16, scalar_value_dim=16,
    point_key_dim=4, point_value_dim=4)
seq = torch.randn(1, 50, 64)
pair = torch.randn(1, 50, 50, 64)
rot = repeat(torch.eye(3), '... -> b n ...', b=1, n=50)
trans = torch.zeros(1, 50, 3)
mask = torch.ones(1, 50).bool()
out = attn(seq, pair, rotations=rot, translations=trans, mask=mask)
assert out.shape == (1, 50, 64)

if torch.cuda.is_available():
    out_gpu = attn.cuda()(seq.cuda(), pair.cuda(),
        rotations=rot.cuda(), translations=trans.cuda(), mask=mask.cuda())
    print(f"GPU test passed: {out_gpu.shape}")

print("ALL IPA TESTS PASSED")
```

To extract source for inline copy:
```python
import inspect, invariant_point_attention
print(inspect.getsource(invariant_point_attention.invariant_point_attention))
```

---

## Architecture: What IPA Replaces

```
RUN 4 (CURRENT):
  backbone(64) + template(16) + msa(8) = pair_repr (1, N, N, 88)
  → DistanceMatrixHead MLP → distances (N, N)
  → template-seeded refinement (gradient descent) → coords (N, 3)

RUN 5 (IPA):
  backbone(64) + template(16) + msa(8) = pair_repr (1, N, N, 88)
  backbone hidden repr = single_repr (1, N, 256)
  → IPAStructureModule (4 iterations of IPABlock + frame updates)
  → coords (N, 3) DIRECTLY

  REMOVED: DistanceMatrixHead, mds_reconstruct, refine_coords
  REMOVED: MSE distance loss
  ADDED:   IPABlock (inlined from lucidrains), FAPE loss, RNA frame builder
```

---

## Inference Fallback: Why It Was Deleted (Option A)

Deleted to keep the notebook clean. Template slots 1-2 are the
safety net if IPA produces poor coordinates for slots 3-5.

**Can be restored later** if IPA underperforms: copy-paste
`DistanceMatrixHead`, `mds_reconstruct`, `refine_coords()` from
Run 4 notebook into Run 5, add threshold check in Cell 14.
Estimated restoration effort: ~30 minutes.

Reason for deleting now:
1. Two parallel inference paths doubles debugging surface area.
2. The threshold value ("IPA is bad enough to fall back") is
   unknowable before training — we'd be guessing.
3. Template slots 1-2 already bound the downside. Score cannot
   fall below a 2-template submission even if IPA fails entirely.

---

## Iteration Design: Shared Weights

```
Iteration 1:  single → IPABlock_shared → update_frames → new coords
Iteration 2:  single → IPABlock_shared → update_frames → new coords  ← SAME weights
Iteration 3:  single → IPABlock_shared → update_frames → new coords  ← SAME weights
Iteration 4:  single → IPABlock_shared → update_frames → new coords  ← SAME weights
```

Config:
```python
IPA_ITERATIONS = 4    # Adjustable between runs. Shared weights = no checkpoint change.
```

---

## VRAM Budget (4 iterations, N=256, T4 16 GB)

| Component | Estimate |
|-----------|----------|
| Backbone frozen | ~400 MB |
| Backbone unfrozen + gradients | ~400 MB |
| Pair repr (256×256×88) | ~23 MB |
| Single repr | ~0.25 MB |
| IPA weights (150K, shared) | ~0.6 MB |
| IPA activations (4 iter) | ~250 MB |
| IPA gradients (4 iter) | ~250 MB |
| FAPE loss | ~100 MB |
| PyTorch overhead | ~1-2 GB |
| **TOTAL** | **~5-6 GB** |
| **T4 available** | **16 GB** |
| **Headroom** | **~10 GB** |

---

## New Parameters (IPA module, shared weights)

| Component | Params | Source |
|-----------|--------|--------|
| IPABlock (shared, 1 copy) | ~150K | Random init (NEW) |
| pair_to_single projection | ~23K | Random init (NEW) |
| Frame update linear | ~1.5K | Random init (NEW) |
| Coord output linear | ~0.8K | Random init (NEW) |
| **Total new** | **~175K** | |
| Backbone layers 7-8 | ~2M | Warm-start from Run 4 |
| Template encoder | 368 | Warm-start from Run 4 |
| **Total trainable** | **~2.2M** | |

---

## Components Written by Us vs Library

| Component | Author | Lines | Bug Risk |
|-----------|--------|-------|----------|
| InvariantPointAttention | lucidrains (copy-paste) | ~150 | Very low |
| IPABlock | lucidrains (copy-paste) | ~80 | Very low |
| IPAStructureModule | Us (wrapper) | ~60 | Low |
| build_rna_frames | Us (custom) | ~40 | Medium |
| fape_loss | Us (custom) | ~50 | Medium |
| apply_rotation_update | Us (custom) | ~20 | Medium |
| **Total** | | **~400** | |

---

## Open Decisions (not yet locked)

| Decision | Proposal | Notes |
|----------|----------|-------|
| TRAIN_EPOCHS | See critical section above | Depends on strategy chosen |
| Freeze backbone during IPA training | Yes (Strategy 1 first) | Speeds up epoch ~2-3× |
| n_heads | 4 | Reduced from AF2's 12. Not confirmed. |
| Auxiliary loss weight | 0.1 × MSE distance | Alongside FAPE. Not confirmed. |
| Teacher forcing ratio | 50% | Not confirmed. |
| Single repr hook | Backbone hidden repr (N,256) | Depends on actual backbone API. |
| Diversity strategy (slots 3-5) | Frame perturbation σ=0.5/1.0/1.5 Å | Not confirmed. |
| Frame init at inference | Iteratively from predicted coords | Not confirmed. |
| FAPE clamp value | 10.0 Å | Not confirmed. |
| Gradient checkpointing | Off (VRAM headroom ~10 GB) | Revisit if OOM. |
| Early stopping patience | 10 epochs | Not confirmed. |

---

## Future Options (Deferred)

| Option | Trigger | Effort |
|--------|---------|--------|
| Restore inference fallback | IPA slots 3-5 score poorly | ~30 min |
| Increase IPA iterations 4→6→8 | IPA converges well, want more quality | Config change only |
| Unshared weights | Dataset grows significantly (>2000 structs) | Architecture change |
| External training (Strategy 2) | Strategy 1 epochs insufficient | ~$3-5, ~1 day setup |

---

## Dependencies on Run 3 OptB Score Investigation

**IMPORTANT:** Run 3 OptB scored 0.251 (below Fork 2's 0.287).

Before coding Run 5, diagnose why 0.251 happened.

Pending:
- [ ] Run 3 OptB training logs (val_loss trajectory)
- [ ] Template search results
- [ ] Hybrid counters (template vs NN slots)
- [ ] Coordinate corruption check in NN slots

---

## File Locations

| Item | Path |
|------|------|
| Run 5 design doc | `HY-BAS-ADV1/RUN5_IPA_DESIGN.md` (this file) |
| Run 5 notebook (not yet created) | `HY-BAS-ADV1/kaggle/hy_bas_adv1_run5-1_ipa_notebook.py` |
| lucidrains source (reference) | https://github.com/lucidrains/invariant-point-attention |
| Run 4 notebook (base) | `HY-BAS-ADV1/kaggle/hy_bas_adv1_run4_notebook.py` |
