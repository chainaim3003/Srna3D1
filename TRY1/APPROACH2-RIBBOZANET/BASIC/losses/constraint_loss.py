"""
constraint_loss.py — Physical constraint losses for RNA structures.

These losses enforce known physical properties of RNA molecules:
  1. Consecutive C1' atoms are ~5.9 Angstroms apart
  2. Non-bonded atoms shouldn't be closer than ~3.0 Angstroms (steric clash)

Applied to the PREDICTED distance matrix (not reconstructed coords),
so they are fully differentiable and can be used during training.
"""

import torch
import torch.nn as nn


class BondConstraintLoss(nn.Module):
    """Penalize predicted consecutive C1'-C1' distances far from 5.9 Angstroms.

    In real RNA structures, consecutive nucleotides have their C1' atoms
    approximately 5.9 Angstroms apart. This varies somewhat (3.5-8.5 Å
    depending on geometry), but 5.9 Å is the typical value.

    This loss penalizes the diagonal+1 entries of the predicted distance
    matrix (i.e., dist[i, i+1] for all i) if they deviate from the target.
    """

    def __init__(self, target_dist: float = 5.9, tolerance: float = 1.0):
        """
        Args:
            target_dist: Target consecutive C1'-C1' distance in Å.
            tolerance: Allowed deviation before penalty kicks in.
                       No penalty if |predicted - target| < tolerance.
        """
        super().__init__()
        self.target_dist = target_dist
        self.tolerance = tolerance

    def forward(
        self,
        pred_dist: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute bond constraint loss.

        Args:
            pred_dist: (B, N, N) predicted distance matrix.
            mask: (B, N) boolean mask.

        Returns:
            Scalar loss value.
        """
        B, N, _ = pred_dist.shape

        if N < 2:
            return torch.tensor(0.0, device=pred_dist.device)

        # Extract consecutive distances: dist[b, i, i+1] for all i
        # These are the diagonal+1 entries
        idx = torch.arange(N - 1, device=pred_dist.device)
        consecutive_dist = pred_dist[:, idx, idx + 1]  # (B, N-1)

        # Deviation from target
        deviation = torch.abs(consecutive_dist - self.target_dist)

        # Only penalize deviations beyond tolerance (hinge loss)
        penalty = torch.relu(deviation - self.tolerance) ** 2

        # Mask: only count valid consecutive pairs
        if mask is not None:
            # Both position i and i+1 must be valid
            consec_mask = mask[:, :-1] & mask[:, 1:]  # (B, N-1)
            penalty = penalty * consec_mask.float()
            loss = penalty.sum() / (consec_mask.float().sum() + 1e-8)
        else:
            loss = penalty.mean()

        return loss


class ClashPenaltyLoss(nn.Module):
    """Penalize non-bonded atoms predicted to be too close together.

    In real molecules, non-bonded atoms have a minimum distance due to
    van der Waals repulsion. For RNA C1' atoms, a reasonable minimum
    is about 3.0 Angstroms for non-consecutive nucleotides.
    """

    def __init__(self, min_dist: float = 3.0):
        """
        Args:
            min_dist: Minimum allowed distance for non-bonded pairs in Å.
        """
        super().__init__()
        self.min_dist = min_dist

    def forward(
        self,
        pred_dist: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute steric clash penalty loss.

        Args:
            pred_dist: (B, N, N) predicted distance matrix.
            mask: (B, N) boolean mask.

        Returns:
            Scalar loss value.
        """
        B, N, _ = pred_dist.shape

        # Create mask for non-bonded pairs:
        # Exclude diagonal (self) and immediate neighbors (i±1)
        non_bonded = torch.ones(N, N, device=pred_dist.device, dtype=torch.bool)
        for offset in [-1, 0, 1]:
            diag_idx = torch.arange(max(0, -offset), min(N, N - offset), device=pred_dist.device)
            non_bonded[diag_idx, diag_idx + offset] = False

        # Upper triangle only (avoid double counting)
        triu = torch.triu(non_bonded, diagonal=2)
        triu = triu.unsqueeze(0).expand(B, -1, -1)

        # Valid pairs mask
        if mask is not None:
            valid = mask.unsqueeze(1) & mask.unsqueeze(2)
            triu = triu & valid

        # Penalty: how much the predicted distance is BELOW the minimum
        violations = torch.relu(self.min_dist - pred_dist)  # positive if too close
        penalty = (violations ** 2) * triu.float()

        loss = penalty.sum() / (triu.float().sum() + 1e-8)
        return loss
