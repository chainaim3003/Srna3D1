"""
constraint_loss.py — Physical constraint losses for RNA structures.
IDENTICAL to BASIC/losses/constraint_loss.py — no changes needed for ADV1.
"""

import torch
import torch.nn as nn


class BondConstraintLoss(nn.Module):
    def __init__(self, target_dist: float = 5.9, tolerance: float = 1.0):
        super().__init__()
        self.target_dist = target_dist
        self.tolerance = tolerance

    def forward(self, pred_dist, mask=None):
        B, N, _ = pred_dist.shape
        if N < 2:
            return torch.tensor(0.0, device=pred_dist.device)
        idx = torch.arange(N - 1, device=pred_dist.device)
        consecutive_dist = pred_dist[:, idx, idx + 1]
        deviation = torch.abs(consecutive_dist - self.target_dist)
        penalty = torch.relu(deviation - self.tolerance) ** 2
        if mask is not None:
            consec_mask = mask[:, :-1] & mask[:, 1:]
            penalty = penalty * consec_mask.float()
            loss = penalty.sum() / (consec_mask.float().sum() + 1e-8)
        else:
            loss = penalty.mean()
        return loss


class ClashPenaltyLoss(nn.Module):
    def __init__(self, min_dist: float = 3.0):
        super().__init__()
        self.min_dist = min_dist

    def forward(self, pred_dist, mask=None):
        B, N, _ = pred_dist.shape
        non_bonded = torch.ones(N, N, device=pred_dist.device, dtype=torch.bool)
        for offset in [-1, 0, 1]:
            diag_idx = torch.arange(max(0, -offset), min(N, N - offset), device=pred_dist.device)
            non_bonded[diag_idx, diag_idx + offset] = False
        triu = torch.triu(non_bonded, diagonal=2)
        triu = triu.unsqueeze(0).expand(B, -1, -1)
        if mask is not None:
            valid = mask.unsqueeze(1) & mask.unsqueeze(2)
            triu = triu & valid
        violations = torch.relu(self.min_dist - pred_dist)
        penalty = (violations ** 2) * triu.float()
        loss = penalty.sum() / (triu.float().sum() + 1e-8)
        return loss
