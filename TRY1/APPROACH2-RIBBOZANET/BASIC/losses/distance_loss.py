"""
distance_loss.py — MSE loss on predicted vs true distance matrices.

This is the PRIMARY training loss for the BASIC approach.
We train the distance head to predict accurate pairwise distances,
then reconstruct 3D coordinates at inference time.
"""

import torch
import torch.nn as nn


class DistanceMatrixLoss(nn.Module):
    """Mean Squared Error on pairwise distance matrices.

    Only computes loss on valid (non-padded) pairs.
    Uses the upper triangle to avoid double-counting symmetric pairs.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_dist: torch.Tensor,
        true_dist: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute MSE loss between predicted and true distance matrices.

        Args:
            pred_dist: (B, N, N) predicted distance matrix.
            true_dist: (B, N, N) ground truth distance matrix.
            mask: (B, N) boolean mask. True = valid, False = padding.

        Returns:
            Scalar loss value.
        """
        B, N, _ = pred_dist.shape

        # Upper triangle mask (avoid counting each pair twice)
        triu = torch.triu(torch.ones(N, N, device=pred_dist.device, dtype=torch.bool), diagonal=1)
        triu = triu.unsqueeze(0).expand(B, -1, -1)  # (B, N, N)

        # If mask provided, only count pairs where both positions are valid
        if mask is not None:
            valid_pairs = mask.unsqueeze(1) & mask.unsqueeze(2)  # (B, N, N)
            triu = triu & valid_pairs

        # Squared error on valid upper-triangle pairs
        sq_error = (pred_dist - true_dist) ** 2
        loss = (sq_error * triu.float()).sum() / (triu.float().sum() + 1e-8)

        return loss
