"""
tm_score_approx.py — Differentiable TM-score approximation.
IDENTICAL to BASIC/losses/tm_score_approx.py — no changes needed for ADV1.
"""

import torch
import numpy as np
from typing import Optional


def compute_d0(L: int) -> float:
    d0 = 1.24 * max(L - 15, 1) ** (1.0 / 3.0) - 1.8
    return max(d0, 0.5)


def kabsch_align_numpy(pred, true):
    pred_center = pred.mean(axis=0)
    true_center = true.mean(axis=0)
    pred_centered = pred - pred_center
    true_centered = true - true_center
    H = pred_centered.T @ true_centered
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T
    aligned = pred_centered @ R.T + true_center
    return aligned


def tm_score_numpy(pred, true):
    N = pred.shape[0]
    if N < 4:
        return 0.0
    aligned = kabsch_align_numpy(pred, true)
    dist = np.sqrt(((aligned - true) ** 2).sum(axis=-1))
    d0 = compute_d0(N)
    tm_per_residue = 1.0 / (1.0 + (dist / d0) ** 2)
    return float(tm_per_residue.sum() / N)


def tm_score_loss_torch(pred_coords, true_coords, mask=None):
    B, N, _ = pred_coords.shape
    if mask is not None:
        mask_f = mask.float().unsqueeze(-1)
        pred_center = (pred_coords * mask_f).sum(dim=1, keepdim=True) / (mask_f.sum(dim=1, keepdim=True) + 1e-8)
        true_center = (true_coords * mask_f).sum(dim=1, keepdim=True) / (mask_f.sum(dim=1, keepdim=True) + 1e-8)
    else:
        pred_center = pred_coords.mean(dim=1, keepdim=True)
        true_center = true_coords.mean(dim=1, keepdim=True)
    pred_centered = pred_coords - pred_center
    true_centered = true_coords - true_center
    dist = torch.sqrt(((pred_centered - true_centered) ** 2).sum(dim=-1) + 1e-8)
    if mask is not None:
        L = mask.float().sum(dim=1)
    else:
        L = torch.tensor([N] * B, dtype=torch.float, device=pred_coords.device)
    d0 = 1.24 * torch.clamp(L - 15, min=1.0) ** (1.0 / 3.0) - 1.8
    d0 = torch.clamp(d0, min=0.5)
    tm_per_residue = 1.0 / (1.0 + (dist / d0.unsqueeze(1)) ** 2)
    if mask is not None:
        tm_score = (tm_per_residue * mask.float()).sum(dim=1) / (mask.float().sum(dim=1) + 1e-8)
    else:
        tm_score = tm_per_residue.mean(dim=1)
    return (1.0 - tm_score).mean()
