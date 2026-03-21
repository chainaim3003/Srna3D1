"""
distance_loss.py — MSE loss on predicted vs true distance matrices.
IDENTICAL to BASIC/losses/distance_loss.py — no changes needed for ADV1.
"""

import torch
import torch.nn as nn


class DistanceMatrixLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_dist, true_dist, mask=None):
        B, N, _ = pred_dist.shape
        triu = torch.triu(torch.ones(N, N, device=pred_dist.device, dtype=torch.bool), diagonal=1)
        triu = triu.unsqueeze(0).expand(B, -1, -1)
        if mask is not None:
            valid_pairs = mask.unsqueeze(1) & mask.unsqueeze(2)
            triu = triu & valid_pairs
        sq_error = (pred_dist - true_dist) ** 2
        loss = (sq_error * triu.float()).sum() / (triu.float().sum() + 1e-8)
        return loss
