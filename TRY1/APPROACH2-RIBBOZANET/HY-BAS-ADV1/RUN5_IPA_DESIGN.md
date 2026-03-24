# HY-BAS-ADV1 Run 5-1 IPA — Design Document

## Status: DESIGN PHASE (not yet coded)

---

## Locked Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| IPA source code | **Inline B** — copy-paste lucidrains' proven library source into notebook | Zero wheel dependency. Same proven code. No risk of rewriting rotation math from scratch. |
| n_iterations | **4 (shared weights)** | Conservative VRAM. Shared weights = change to 6 or 8 in later runs with zero checkpoint friction. |
| Weight sharing | **Shared** — same IPA layer reused each iteration | 661 training structures too few for unshared (would be 600K+ params). Shared = 150K params. Free iteration changes between runs. |
| FAPE loss | **Yes** + 0.1× auxiliary distance MSE | FAPE is rotationally invariant, catches mirror-flips. Aux MSE guides early training. |
| BATCH_SIZE | **1** | VRAM safety for IPA activations across 4 iterations. |
| TRAIN_EPOCHS | **70** | IPA module trains from scratch (~150K new params). Needs more epochs than Run 4. |
| Teacher forcing | **50%** | Half the time start from true coords + noise, half from template/zeros. |

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
- **Stars:** 148 on GitHub
- **What it provides:**
  - `InvariantPointAttention` — the core IPA attention layer
  - `IPABlock` — IPA + feedforward + layernorm (complete transformer block)
  - `IPATransformer` — multi-layer IPA with frame updates (full structure module)
- **API:**
  ```python
  from invariant_point_attention import InvariantPointAttention, IPABlock

  # Core attention layer
  attn = InvariantPointAttention(
      dim=64,              # single repr dimension
      heads=8,             # attention heads
      scalar_key_dim=16,   # scalar Q/K dimension per head
      scalar_value_dim=16, # scalar V dimension per head
      point_key_dim=4,     # 3D point Q/K dimension per head
      point_value_dim=4,   # 3D point V dimension per head
      require_pairwise_repr=True  # use pairwise features
  )

  # Inputs
  single_repr = torch.randn(1, N, 64)       # (batch, seq, dim)
  pairwise_repr = torch.randn(1, N, N, 64)  # (batch, seq, seq, dim)
  rotations = torch.eye(3).expand(1, N, 3, 3)  # (batch, seq, 3, 3)
  translations = torch.zeros(1, N, 3)       # (batch, seq, 3)
  mask = torch.ones(1, N).bool()             # (batch, seq)

  # Forward
  out = attn(single_repr, pairwise_repr,
             rotations=rotations, translations=translations, mask=mask)
  # out shape: (1, N, 64)

  # IPABlock = IPA + feedforward + layernorm
  block = IPABlock(dim=64, heads=8, ...)
  block_out = block(single_repr, pairwise_repr=pairwise_repr,
                    rotations=rotations, translations=translations, mask=mask)
  ```

### Inline B Approach

We copy the source code of `InvariantPointAttention` and `IPABlock`
directly into Cell 10 of the notebook. This is the SAME code — not a
rewrite. We are NOT reimplementing the rotation math, attention weights,
or point transformations. We are copy-pasting proven code to avoid a
wheel dependency.

The copied code will be clearly marked:
```python
# ============================================================
# BEGIN: Copied from lucidrains/invariant-point-attention v0.2.2
# Source: https://github.com/lucidrains/invariant-point-attention
# License: MIT
# DO NOT MODIFY this section — it is proven library code.
# ============================================================
... (library code) ...
# ============================================================
# END: lucidrains library code
# ============================================================
```

### Alternative Libraries (NOT used, documented for reference)

| Library | Why Not Used |
|---------|-------------|
| `Invariant-Attention` (Rishit Dagli) | TensorFlow, not PyTorch |
| OpenFold IPA module | Tightly coupled to OpenFold, hard to extract |
| RhoFold+ structure module | RNA-specific but requires RhoFold dependencies |
| Protenix (RNAPro) | Too heavy, NVIDIA ecosystem |

### Alternative Delivery: Wheel (NOT used, documented for reference)

If we wanted to import the library instead of inlining:
```bash
# Download wheel (one time, on any machine with internet)
pip download invariant-point-attention --no-deps -d ./wheels/

# Upload the .whl file as Kaggle dataset "ipa-wheel"
# Attach to notebook, install in Cell 1:
pip install /kaggle/input/ipa-wheel/invariant_point_attention-0.2.2-py3-none-any.whl
```
This was rejected because inline B achieves the same result with zero
dataset management overhead, and the code is small enough (~300 lines)
to paste directly.

---

## Quick Library Validation Test

Before writing the full notebook, run this 30-second test in any
Kaggle notebook with Internet ON to confirm the library works:

```python
!pip install invariant-point-attention -q
import torch
from einops import repeat
from invariant_point_attention import InvariantPointAttention, IPABlock

# Test 1: Basic IPA attention
attn = InvariantPointAttention(dim=64, heads=4,
    scalar_key_dim=16, scalar_value_dim=16,
    point_key_dim=4, point_value_dim=4)
seq = torch.randn(1, 50, 64)
pair = torch.randn(1, 50, 50, 64)
rot = repeat(torch.eye(3), '... -> b n ...', b=1, n=50)
trans = torch.zeros(1, 50, 3)
mask = torch.ones(1, 50).bool()
out = attn(seq, pair, rotations=rot, translations=trans, mask=mask)
print(f"IPA attention: input {seq.shape} -> output {out.shape}")
assert out.shape == (1, 50, 64), "Shape mismatch!"

# Test 2: IPABlock (IPA + feedforward)
block = IPABlock(dim=64, heads=4, scalar_key_dim=16, scalar_value_dim=16,
                 point_key_dim=4, point_value_dim=4)
block_out = block(seq, pairwise_repr=pair, rotations=rot,
                  translations=trans, mask=mask)
print(f"IPABlock: input {seq.shape} -> output {block_out.shape}")
assert block_out.shape == (1, 50, 64), "Shape mismatch!"

# Test 3: GPU if available
if torch.cuda.is_available():
    attn_gpu = attn.cuda()
    out_gpu = attn_gpu(seq.cuda(), pair.cuda(), rotations=rot.cuda(),
                       translations=trans.cuda(), mask=mask.cuda())
    print(f"GPU test passed: {out_gpu.shape}")

print("ALL IPA TESTS PASSED")
```

If this passes, copy the library source from the installed package:
```python
import inspect, invariant_point_attention
print(inspect.getsource(invariant_point_attention.invariant_point_attention))
```
This prints the exact source code to paste into our notebook.

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

## Iteration Design: Shared Weights

```
Iteration 1:  single → IPABlock_shared → update_frames → new coords
Iteration 2:  single → IPABlock_shared → update_frames → new coords  ← SAME weights
Iteration 3:  single → IPABlock_shared → update_frames → new coords  ← SAME weights
Iteration 4:  single → IPABlock_shared → update_frames → new coords  ← SAME weights
```

**Why shared weights:**
- 150K params (shared) vs 600K (unshared) — less overfitting on 661 structures
- Changing iterations between runs is FREE — same checkpoint, different config
- Run 5-1 at 4 iterations → Run 5-2 at 6 → Run 5-3 at 8, all same checkpoint
- This is exactly analogous to changing TRAIN_EPOCHS between runs

**Config:**
```python
IPA_ITERATIONS = 4    # Adjustable between runs. Shared weights = no checkpoint change.
```

---

## VRAM Budget (4 iterations, N=256, T4 16 GB)

| Component | Estimate |
|-----------|----------|
| Backbone (frozen layers 0-6) | ~400 MB |
| Backbone (unfrozen layers 7-8 + gradients) | ~400 MB |
| Pair repr (256×256×88) | ~23 MB |
| Single repr (256×256) | ~0.25 MB |
| IPA weights (150K params, shared) | ~0.6 MB |
| IPA activations (4 iterations) | ~250 MB |
| IPA gradients (4 iterations) | ~250 MB |
| FAPE loss computation | ~100 MB |
| Template encoder | ~1 MB |
| PyTorch overhead | ~1-2 GB |
| **TOTAL** | **~5-6 GB** |
| **T4 available** | **16 GB** |
| **Headroom** | **~10 GB** |

Comfortable at 4 iterations. Could increase to 6 (add ~125 MB) or 8
(add ~250 MB) with plenty of room.

---

## New Parameters (IPA module, shared weights)

| Component | Params | Trainable | Source |
|-----------|--------|-----------|--------|
| IPABlock (shared, 1 copy) | ~150K | Yes | Random init (NEW) |
| pair_to_single projection | ~23K | Yes | Random init (NEW) |
| Frame update linear | ~1.5K | Yes | Random init (NEW) |
| Coord output linear | ~0.8K | Yes | Random init (NEW) |
| **Total new** | **~175K** | | |
| Backbone layers 7-8 | ~2M | Yes | Warm-start from Run 4 |
| Template encoder | 368 | Yes | Warm-start from Run 4 |
| **Total trainable** | **~2.2M** | | |

---

## Components Written by Us vs Library

| Component | Author | Lines | Bug Risk |
|-----------|--------|-------|----------|
| InvariantPointAttention | lucidrains (copy-paste) | ~150 | Very low — proven |
| IPABlock | lucidrains (copy-paste) | ~80 | Very low — proven |
| IPAStructureModule | Us (wrapper) | ~60 | Low — simple loop |
| build_rna_frames | Us (custom) | ~40 | Medium — vector math, testable |
| fape_loss | Us (custom) | ~50 | Medium — local frame transforms |
| apply_rotation_update | Us (custom) | ~20 | Medium — rotation composition |
| **Total** | | **~400** | |
| Of which proven library code | | **~230** | Very low |
| Of which our custom code | | **~170** | Medium |

---

## Dependencies on Run 3 OptB Score Investigation

**IMPORTANT:** Run 3 OptB scored 0.251 (below Fork 2's 0.287).

Before coding Run 5, we need to diagnose why 0.251 happened. If the
root cause is in template search or constraint functions (Cells 8-9),
those same bugs carry into Run 5. If the root cause is in the NN
pipeline (Cells 13-14), IPA may fix it by replacing the problematic
distance→refinement approach entirely.

**Do NOT code Run 5 until the 0.251 root cause is identified.**

Pending investigation items:
- [ ] Run 3 OptB training logs (val_loss trajectory)
- [ ] Template search results (which templates were found)
- [ ] Hybrid counters (how many targets got template vs NN slots)
- [ ] Compare our template coordinates vs Fork 2's original output
- [ ] Check for coordinate corruption in NN slots

---

## File Locations

| Item | Path |
|------|------|
| Run 5 design doc | `HY-BAS-ADV1/RUN5_IPA_DESIGN.md` (this file) |
| Run 5 notebook (not yet created) | `HY-BAS-ADV1/kaggle/hy_bas_adv1_run5-1_ipa_notebook.py` |
| lucidrains source (reference) | https://github.com/lucidrains/invariant-point-attention |
| Run 4 notebook (base) | `HY-BAS-ADV1/kaggle/hy_bas_adv1_run4_notebook.py` |
