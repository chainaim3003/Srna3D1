"""
reconstructor.py — Convert distance matrix to 3D coordinates.

Two-stage approach:
  1. Classical MDS (Multidimensional Scaling) for initial coordinates
  2. Gradient-based refinement to minimize distance violations

Classical MDS math:
  Given distance matrix D (N×N), we want coordinates X (N×3) such that
  ||X_i - X_j|| ≈ D_ij for all i, j.

  Steps:
  1. Compute squared distance matrix: D² (element-wise squaring)
  2. Double-centering: B = -0.5 * H * D² * H where H = I - 1/N * 11^T
  3. Eigendecompose B: B = V * Λ * V^T
  4. Take top 3 eigenvalues/vectors: X = V[:, :3] * sqrt(Λ[:3])

This is a standard linear algebra technique — see scipy.linalg.eigh docs.
"""

import torch
import numpy as np
from scipy.spatial.distance import squareform
from scipy.linalg import eigh


def mds_from_distances_numpy(dist_matrix: np.ndarray) -> np.ndarray:
    """Classical MDS: convert distance matrix to 3D coordinates.

    Uses scipy.linalg.eigh for stable eigendecomposition.

    Args:
        dist_matrix: (N, N) symmetric distance matrix in Angstroms.

    Returns:
        coords: (N, 3) 3D coordinates.
    """
    N = dist_matrix.shape[0]

    if N < 4:
        # Too few points for meaningful MDS — return simple line
        coords = np.zeros((N, 3))
        for i in range(N):
            coords[i, 0] = i * 5.9  # Space along x-axis at ~5.9Å intervals
        return coords

    # Step 1: Squared distance matrix
    D_sq = dist_matrix ** 2

    # Step 2: Double centering
    # H = I - (1/N) * ones_matrix
    # B = -0.5 * H @ D_sq @ H
    H = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * H @ D_sq @ H

    # Make B symmetric (numerical stability)
    B = (B + B.T) / 2.0

    # Step 3: Eigendecomposition
    # eigh returns eigenvalues in ascending order
    eigenvalues, eigenvectors = eigh(B)

    # Step 4: Take top 3 eigenvalues (largest, at the end of the array)
    # Negative eigenvalues indicate the distance matrix is not perfectly embeddable
    # in Euclidean space — we clamp them to a small positive value.
    top3_idx = np.argsort(eigenvalues)[-3:][::-1]  # Indices of 3 largest
    top3_vals = eigenvalues[top3_idx]
    top3_vecs = eigenvectors[:, top3_idx]

    # Clamp negative eigenvalues to small positive
    top3_vals = np.maximum(top3_vals, 1e-6)

    # Coordinates = eigenvectors * sqrt(eigenvalues)
    coords = top3_vecs * np.sqrt(top3_vals)[np.newaxis, :]

    return coords  # (N, 3)


def mds_from_distances_torch(dist_matrix: torch.Tensor) -> torch.Tensor:
    """Classical MDS using PyTorch (for GPU acceleration).

    NOT differentiable — eigendecomposition gradients can be unstable.
    Use this for inference, not training.

    Args:
        dist_matrix: (N, N) symmetric distance matrix tensor.

    Returns:
        coords: (N, 3) coordinate tensor.
    """
    N = dist_matrix.shape[0]
    device = dist_matrix.device

    if N < 4:
        coords = torch.zeros(N, 3, device=device)
        for i in range(N):
            coords[i, 0] = i * 5.9
        return coords

    D_sq = dist_matrix ** 2
    H = torch.eye(N, device=device) - torch.ones(N, N, device=device) / N
    B = -0.5 * H @ D_sq @ H
    B = (B + B.T) / 2.0

    # torch.linalg.eigh returns (eigenvalues, eigenvectors) in ascending order
    eigenvalues, eigenvectors = torch.linalg.eigh(B)

    # Take top 3 (last 3, since ascending order)
    top3_vals = eigenvalues[-3:].flip(0)
    top3_vecs = eigenvectors[:, -3:].flip(1)

    top3_vals = torch.clamp(top3_vals, min=1e-6)

    coords = top3_vecs * torch.sqrt(top3_vals).unsqueeze(0)

    return coords  # (N, 3)


def refine_coordinates(
    coords: torch.Tensor,
    target_distances: torch.Tensor,
    mask: torch.Tensor = None,
    num_steps: int = 100,
    lr: float = 0.01,
    consecutive_target: float = 5.9,
    consecutive_weight: float = 1.0,
) -> torch.Tensor:
    """Refine MDS coordinates via gradient descent on distance violations.

    Minimizes the stress function:
        stress = sum_{i<j} (||x_i - x_j|| - d_ij)^2

    Plus a constraint on consecutive C1' distances (~5.9Å).

    Args:
        coords: (N, 3) initial coordinates from MDS.
        target_distances: (N, N) target distance matrix.
        mask: (N,) boolean mask for valid positions.
        num_steps: Number of gradient descent iterations.
        lr: Learning rate for refinement.
        consecutive_target: Expected consecutive C1'-C1' distance in Å.
        consecutive_weight: Weight for consecutive distance constraint.

    Returns:
        refined_coords: (N, 3) refined coordinates.
    """
    # Clone so we don't modify the input
    refined = coords.clone().detach().requires_grad_(True)
    device = coords.device

    optimizer = torch.optim.Adam([refined], lr=lr)

    N = coords.shape[0]

    for step in range(num_steps):
        optimizer.zero_grad()

        # Compute current pairwise distances
        # torch.cdist computes all pairwise Euclidean distances
        current_dist = torch.cdist(refined.unsqueeze(0), refined.unsqueeze(0)).squeeze(0)
        # current_dist: (N, N)

        # Stress loss: squared difference between current and target distances
        # Only count upper triangle (avoid double-counting)
        triu_mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
        if mask is not None:
            # Only count pairs where both positions are valid
            valid_pairs = mask.unsqueeze(0) & mask.unsqueeze(1)
            triu_mask = triu_mask & valid_pairs

        diff = (current_dist - target_distances) * triu_mask.float()
        stress = (diff ** 2).sum() / (triu_mask.float().sum() + 1e-8)

        # Consecutive distance constraint
        if N > 1:
            consec_dist = torch.norm(refined[1:] - refined[:-1], dim=-1)
            consec_loss = ((consec_dist - consecutive_target) ** 2).mean()
        else:
            consec_loss = torch.tensor(0.0, device=device)

        total_loss = stress + consecutive_weight * consec_loss
        total_loss.backward()
        optimizer.step()

    return refined.detach()


def reconstruct_3d(
    dist_matrix: torch.Tensor,
    mask: torch.Tensor = None,
    method: str = "mds_then_refine",
    refine_steps: int = 100,
    refine_lr: float = 0.01,
    consecutive_target: float = 5.9,
) -> torch.Tensor:
    """Full 3D reconstruction pipeline.

    Args:
        dist_matrix: (N, N) predicted distance matrix tensor.
        mask: (N,) boolean mask for valid positions.
        method: "mds_only" or "mds_then_refine"
        refine_steps: Number of refinement steps (only for mds_then_refine).
        refine_lr: Learning rate for refinement.
        consecutive_target: Expected consecutive C1' distance.

    Returns:
        coords: (N, 3) reconstructed 3D coordinates.
    """
    # Step 1: MDS for initial coordinates
    # Use numpy version for numerical stability
    dist_np = dist_matrix.detach().cpu().numpy()
    coords_np = mds_from_distances_numpy(dist_np)
    coords = torch.tensor(coords_np, dtype=torch.float32, device=dist_matrix.device)

    if method == "mds_only":
        return coords

    # Step 2: Gradient-based refinement
    coords = refine_coordinates(
        coords=coords,
        target_distances=dist_matrix.detach(),
        mask=mask,
        num_steps=refine_steps,
        lr=refine_lr,
        consecutive_target=consecutive_target,
    )

    return coords


def reconstruct_batch(
    dist_matrices: torch.Tensor,
    masks: torch.Tensor = None,
    method: str = "mds_then_refine",
    refine_steps: int = 100,
    refine_lr: float = 0.01,
    consecutive_target: float = 5.9,
) -> torch.Tensor:
    """Reconstruct 3D coordinates for a batch of distance matrices.

    Args:
        dist_matrices: (B, N, N) batch of distance matrices.
        masks: (B, N) batch of boolean masks. None = all valid.
        method: Reconstruction method.
        refine_steps: Number of refinement steps.
        refine_lr: Learning rate for refinement.
        consecutive_target: Expected consecutive C1' distance.

    Returns:
        coords: (B, N, 3) batch of 3D coordinates.
    """
    B, N, _ = dist_matrices.shape
    all_coords = []

    for b in range(B):
        mask_b = masks[b] if masks is not None else None
        coords_b = reconstruct_3d(
            dist_matrix=dist_matrices[b],
            mask=mask_b,
            method=method,
            refine_steps=refine_steps,
            refine_lr=refine_lr,
            consecutive_target=consecutive_target,
        )
        all_coords.append(coords_b)

    return torch.stack(all_coords, dim=0)  # (B, N, 3)
