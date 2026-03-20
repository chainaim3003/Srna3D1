"""
tm_score_approx.py — Differentiable TM-score approximation.

TM-score is the competition metric (0 to 1, higher = better).
This module provides a non-differentiable TM-score for evaluation
and a differentiable approximation for optional use as a training loss.

TM-score formula:
    TM = (1/L) * sum_i [ 1 / (1 + (d_i / d0)^2) ]

Where:
    L = sequence length
    d_i = distance between predicted and true position of residue i (after alignment)
    d0 = 1.24 * (L - 15)^(1/3) - 1.8  (length-dependent normalization)

Reference:
    Zhang & Skolnick, "Scoring function for automated assessment of
    protein structure template quality", Proteins, 2004.
"""

import torch
import numpy as np
from typing import Optional


def compute_d0(L: int) -> float:
    """Compute the TM-score normalization factor d0.

    Args:
        L: Sequence length (number of residues).

    Returns:
        d0 value in Angstroms.
    """
    d0 = 1.24 * max(L - 15, 1) ** (1.0 / 3.0) - 1.8
    return max(d0, 0.5)


def kabsch_align_numpy(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Align predicted coordinates to true coordinates using Kabsch algorithm.

    Finds the optimal rotation and translation to minimize RMSD.
    Uses SVD decomposition — a standard approach.

    Args:
        pred: (N, 3) predicted coordinates.
        true: (N, 3) true coordinates.

    Returns:
        aligned: (N, 3) predicted coordinates after alignment.
    """
    # Center both structures
    pred_center = pred.mean(axis=0)
    true_center = true.mean(axis=0)

    pred_centered = pred - pred_center
    true_centered = true - true_center

    # Compute cross-covariance matrix
    H = pred_centered.T @ true_centered  # (3, 3)

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Ensure proper rotation (not reflection)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])

    # Optimal rotation
    R = Vt.T @ sign_matrix @ U.T

    # Apply rotation and translation
    aligned = pred_centered @ R.T + true_center

    return aligned


def tm_score_numpy(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute TM-score between predicted and true structures.

    Non-differentiable. Used for evaluation only.

    Args:
        pred: (N, 3) predicted C1' coordinates.
        true: (N, 3) true C1' coordinates.

    Returns:
        TM-score between 0 and 1.
    """
    N = pred.shape[0]
    if N < 4:
        return 0.0

    # Align
    aligned = kabsch_align_numpy(pred, true)

    # Per-residue distance after alignment
    dist = np.sqrt(((aligned - true) ** 2).sum(axis=-1))  # (N,)

    # d0
    d0 = compute_d0(N)

    # TM-score
    tm_per_residue = 1.0 / (1.0 + (dist / d0) ** 2)
    tm_score = tm_per_residue.sum() / N

    return float(tm_score)


def tm_score_loss_torch(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Differentiable approximation of (1 - TM-score).

    Minimizing this loss = maximizing TM-score.

    NOTE: This requires aligned coordinates. We use a simple centering
    alignment (not full Kabsch) to keep it differentiable. For accurate
    TM-score evaluation, use tm_score_numpy instead.

    Args:
        pred_coords: (B, N, 3) predicted coordinates.
        true_coords: (B, N, 3) true coordinates.
        mask: (B, N) boolean mask.

    Returns:
        Scalar loss = 1 - approximate_TM-score.
    """
    B, N, _ = pred_coords.shape

    # Simple alignment: center both structures
    if mask is not None:
        # Compute masked mean
        mask_f = mask.float().unsqueeze(-1)  # (B, N, 1)
        pred_center = (pred_coords * mask_f).sum(dim=1, keepdim=True) / (mask_f.sum(dim=1, keepdim=True) + 1e-8)
        true_center = (true_coords * mask_f).sum(dim=1, keepdim=True) / (mask_f.sum(dim=1, keepdim=True) + 1e-8)
    else:
        pred_center = pred_coords.mean(dim=1, keepdim=True)
        true_center = true_coords.mean(dim=1, keepdim=True)

    pred_centered = pred_coords - pred_center
    true_centered = true_coords - true_center

    # Per-residue distance
    dist = torch.sqrt(((pred_centered - true_centered) ** 2).sum(dim=-1) + 1e-8)  # (B, N)

    # d0 (using max sequence length in batch)
    if mask is not None:
        L = mask.float().sum(dim=1)  # (B,)
    else:
        L = torch.tensor([N] * B, dtype=torch.float, device=pred_coords.device)

    d0 = 1.24 * torch.clamp(L - 15, min=1.0) ** (1.0 / 3.0) - 1.8
    d0 = torch.clamp(d0, min=0.5)  # (B,)

    # TM-score per residue
    tm_per_residue = 1.0 / (1.0 + (dist / d0.unsqueeze(1)) ** 2)  # (B, N)

    if mask is not None:
        tm_score = (tm_per_residue * mask.float()).sum(dim=1) / (mask.float().sum(dim=1) + 1e-8)
    else:
        tm_score = tm_per_residue.mean(dim=1)

    # Loss = 1 - TM-score (averaged over batch)
    return (1.0 - tm_score).mean()
