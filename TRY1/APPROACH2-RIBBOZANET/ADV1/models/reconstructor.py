"""
reconstructor.py — Convert distance matrix to 3D coordinates.
IDENTICAL to BASIC/models/reconstructor.py — no changes needed for ADV1.
"""

import torch
import numpy as np
from scipy.spatial.distance import squareform
from scipy.linalg import eigh


def mds_from_distances_numpy(dist_matrix: np.ndarray) -> np.ndarray:
    N = dist_matrix.shape[0]
    if N < 4:
        coords = np.zeros((N, 3))
        for i in range(N):
            coords[i, 0] = i * 5.9
        return coords
    D_sq = dist_matrix ** 2
    H = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * H @ D_sq @ H
    B = (B + B.T) / 2.0
    eigenvalues, eigenvectors = eigh(B)
    top3_idx = np.argsort(eigenvalues)[-3:][::-1]
    top3_vals = eigenvalues[top3_idx]
    top3_vecs = eigenvectors[:, top3_idx]
    top3_vals = np.maximum(top3_vals, 1e-6)
    coords = top3_vecs * np.sqrt(top3_vals)[np.newaxis, :]
    return coords


def mds_from_distances_torch(dist_matrix: torch.Tensor) -> torch.Tensor:
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
    eigenvalues, eigenvectors = torch.linalg.eigh(B)
    top3_vals = eigenvalues[-3:].flip(0)
    top3_vecs = eigenvectors[:, -3:].flip(1)
    top3_vals = torch.clamp(top3_vals, min=1e-6)
    coords = top3_vecs * torch.sqrt(top3_vals).unsqueeze(0)
    return coords


def refine_coordinates(coords, target_distances, mask=None, num_steps=100,
                       lr=0.01, consecutive_target=5.9, consecutive_weight=1.0):
    refined = coords.clone().detach().requires_grad_(True)
    device = coords.device
    optimizer = torch.optim.Adam([refined], lr=lr)
    N = coords.shape[0]
    for step in range(num_steps):
        optimizer.zero_grad()
        current_dist = torch.cdist(refined.unsqueeze(0), refined.unsqueeze(0)).squeeze(0)
        triu_mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
        if mask is not None:
            valid_pairs = mask.unsqueeze(0) & mask.unsqueeze(1)
            triu_mask = triu_mask & valid_pairs
        diff = (current_dist - target_distances) * triu_mask.float()
        stress = (diff ** 2).sum() / (triu_mask.float().sum() + 1e-8)
        if N > 1:
            consec_dist = torch.norm(refined[1:] - refined[:-1], dim=-1)
            consec_loss = ((consec_dist - consecutive_target) ** 2).mean()
        else:
            consec_loss = torch.tensor(0.0, device=device)
        total_loss = stress + consecutive_weight * consec_loss
        total_loss.backward()
        optimizer.step()
    return refined.detach()


def reconstruct_3d(dist_matrix, mask=None, method="mds_then_refine",
                   refine_steps=100, refine_lr=0.01, consecutive_target=5.9):
    dist_np = dist_matrix.detach().cpu().numpy()
    coords_np = mds_from_distances_numpy(dist_np)
    coords = torch.tensor(coords_np, dtype=torch.float32, device=dist_matrix.device)
    if method == "mds_only":
        return coords
    coords = refine_coordinates(
        coords=coords, target_distances=dist_matrix.detach(), mask=mask,
        num_steps=refine_steps, lr=refine_lr, consecutive_target=consecutive_target,
    )
    return coords


def reconstruct_batch(dist_matrices, masks=None, method="mds_then_refine",
                      refine_steps=100, refine_lr=0.01, consecutive_target=5.9):
    B, N, _ = dist_matrices.shape
    all_coords = []
    for b in range(B):
        mask_b = masks[b] if masks is not None else None
        coords_b = reconstruct_3d(
            dist_matrix=dist_matrices[b], mask=mask_b, method=method,
            refine_steps=refine_steps, refine_lr=refine_lr,
            consecutive_target=consecutive_target,
        )
        all_coords.append(coords_b)
    return torch.stack(all_coords, dim=0)
