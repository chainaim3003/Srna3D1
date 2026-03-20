"""
distance_head.py — Predict pairwise distances from pairwise representation.

Takes the (B, N, N, pairwise_dim) representation from RibonanzaNet
and predicts a symmetric distance matrix (B, N, N) in Angstroms.

The distance matrix represents predicted C1'-C1' distances between
every pair of nucleotides in the RNA.
"""

import torch
import torch.nn as nn


class DistanceMatrixHead(nn.Module):
    """MLP that maps pairwise features to distance predictions.

    Architecture:
        pairwise_repr (B, N, N, pair_dim)
        → Linear → ReLU → Dropout
        → Linear → ReLU → Dropout
        → Linear → Softplus
        → distance (B, N, N)

    Softplus ensures predicted distances are always positive.
    The output is symmetrized: dist[i,j] = (dist[i,j] + dist[j,i]) / 2
    """

    def __init__(self, pair_dim: int = 64, hidden_dim: int = 128,
                 num_layers: int = 3, dropout: float = 0.1):
        """
        Args:
            pair_dim: Dimension of input pairwise features.
                      Must match RibonanzaNet's pairwise_dimension (default 64).
            hidden_dim: Hidden layer width.
            num_layers: Number of linear layers in the MLP.
            dropout: Dropout probability.
        """
        super().__init__()

        layers = []
        in_dim = pair_dim

        for i in range(num_layers - 1):
            out_dim = hidden_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim

        # Final layer: predict a single scalar (distance) per pair
        layers.append(nn.Linear(in_dim, 1))

        self.mlp = nn.Sequential(*layers)

        # Softplus: smooth approximation of ReLU, ensures positive output
        # Softplus(x) = log(1 + exp(x))
        self.activation = nn.Softplus()

    def forward(self, pairwise_repr: torch.Tensor) -> torch.Tensor:
        """Predict distance matrix from pairwise features.

        Args:
            pairwise_repr: (B, N, N, pair_dim) — pairwise features from backbone

        Returns:
            dist_matrix: (B, N, N) — predicted C1'-C1' distances in Angstroms.
                         Symmetric, non-negative, diagonal = 0.
        """
        # Run MLP on each pair independently
        raw_dist = self.mlp(pairwise_repr)  # (B, N, N, 1)
        raw_dist = raw_dist.squeeze(-1)      # (B, N, N)

        # Ensure positive distances
        dist = self.activation(raw_dist)     # (B, N, N)

        # Symmetrize: distance i→j should equal distance j→i
        dist = (dist + dist.transpose(-1, -2)) / 2.0

        # Zero diagonal: distance from a nucleotide to itself is 0
        B, N, _ = dist.shape
        mask = torch.eye(N, device=dist.device, dtype=torch.bool).unsqueeze(0)
        dist = dist.masked_fill(mask, 0.0)

        return dist
