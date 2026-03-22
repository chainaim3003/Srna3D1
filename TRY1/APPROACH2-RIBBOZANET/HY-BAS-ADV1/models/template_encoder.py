"""
template_encoder.py — Convert template 3D coordinates into pairwise features.

Pipeline:
    template_coords (N, 3)
        -> pairwise distances (N, N)
        -> distance bins (N, N, num_bins)
        -> Linear(num_bins, template_dim) -> template features (N, N, template_dim)
        -> multiply by confidence mask

The template features get concatenated with the backbone pairwise features
before being fed into the distance head:
    [pairwise_repr (N,N,64), template_feat (N,N,16)] -> (N,N,80) -> distance_head

For targets with NO template (no MMseqs2 hit):
    template features are all zeros (N,N,16), so the model falls back to
    pairwise-only prediction (same behavior as BASIC).
"""

import torch
import torch.nn as nn
import numpy as np


class TemplateEncoder(nn.Module):
    """Encode template 3D coordinates as pairwise features.

    Architecture:
        coords (N, 3) -> distances (N, N) -> bins (N, N, num_bins)
        -> Linear(num_bins, template_dim) -> features (N, N, template_dim)

    Trainable parameters: Linear(num_bins, template_dim) = 22*16 + 16 = 368 params
    """

    def __init__(self, template_dim: int = 16, num_bins: int = 22,
                 max_dist: float = 40.0):
        """
        Args:
            template_dim: Output feature dimension per pair. Default 16.
            num_bins: Number of distance bins. Default 22.
                      Bins: [0,2), [2,4), [4,6), ..., [38,40), [40, inf)
            max_dist: Maximum distance before overflow bin (Angstroms).
        """
        super().__init__()
        self.template_dim = template_dim
        self.num_bins = num_bins
        self.max_dist = max_dist

        # Fixed bin edges: [0, 2, 4, 6, ..., 38, 40]
        # This creates num_bins-1 = 21 edges, giving 22 bins
        # Last bin catches everything > max_dist
        bin_width = max_dist / (num_bins - 1)
        edges = torch.arange(0, max_dist + bin_width, bin_width)[:num_bins]
        self.register_buffer('bin_edges', edges)

        # Learnable projection from one-hot bins to template features
        self.projection = nn.Linear(num_bins, template_dim, bias=True)

        # Initialize with small weights so template features start near zero
        # This means at the start of training, ADV1 behaves like BASIC
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.projection.bias)

    def coords_to_dist(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distance matrix from 3D coordinates.

        Args:
            coords: (N, 3) C1' atom coordinates in Angstroms.

        Returns:
            dist: (N, N) pairwise Euclidean distances.
        """
        # dist[i,j] = ||coords[i] - coords[j]||
        diff = coords.unsqueeze(0) - coords.unsqueeze(1)  # (N, N, 3)
        dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)  # (N, N)
        return dist

    def dist_to_bins(self, dist: torch.Tensor) -> torch.Tensor:
        """Convert distances to one-hot bin representation.

        Args:
            dist: (N, N) distance matrix in Angstroms.

        Returns:
            bins: (N, N, num_bins) one-hot encoded distance bins.
        """
        N = dist.shape[0]
        # Assign each distance to a bin index
        # torch.bucketize: finds the bin index for each distance
        bin_idx = torch.bucketize(dist, self.bin_edges)
        # Clamp to valid range [0, num_bins-1]
        bin_idx = bin_idx.clamp(0, self.num_bins - 1)
        # One-hot encode
        bins = torch.zeros(N, N, self.num_bins, device=dist.device)
        bins.scatter_(2, bin_idx.unsqueeze(-1), 1.0)
        return bins

    def forward(self, coords: torch.Tensor, confidence: float = 1.0,
                has_template: bool = True) -> torch.Tensor:
        """Encode template coordinates as pairwise features.

        Args:
            coords: (N, 3) template C1' coordinates. Can be zeros if no template.
            confidence: Scalar 0-1 indicating template quality.
                        Derived from MMseqs2 e-value:
                        confidence = min(1.0, -log10(evalue) / 50.0)
            has_template: If False, returns zeros (no template available).

        Returns:
            features: (N, N, template_dim) template pairwise features.
        """
        N = coords.shape[0]

        if not has_template:
            return torch.zeros(N, N, self.template_dim, device=coords.device)

        # Step 1: Coordinates -> distances
        dist = self.coords_to_dist(coords)  # (N, N)

        # Step 2: Distances -> bins
        bins = self.dist_to_bins(dist)  # (N, N, num_bins)

        # Step 3: Bins -> features via learned projection
        features = self.projection(bins)  # (N, N, template_dim)

        # Step 4: Scale by confidence
        features = features * confidence

        return features
