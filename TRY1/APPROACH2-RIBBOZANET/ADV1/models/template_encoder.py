"""
template_encoder.py — Encode Approach 1 template coordinates into features.

Converts C1' coordinates from template-based modeling (Approach 1) into a
feature tensor that can be concatenated with RibonanzaNet's pairwise features.

Pipeline:
  template coords (B, N, 3)
    → pairwise distance matrix (B, N, N)
    → bin distances into discrete categories (B, N, N) integers
    → one-hot encode (B, N, N, num_bins)
    → linear projection (B, N, N, template_dim)
    → multiply by pairwise confidence mask
    → template features (B, N, N, template_dim)

Design decisions:
  - Distance binning (not raw distances): Categorical representation is easier
    for the linear layer to learn from. Same approach as AlphaFold2's template
    distogram (AlphaFold2 Supplementary Table 4).
  - "No template" bin: Distinguishes "template says distance is 0" (impossible)
    from "no template information available."
  - Confidence masking: Scales features by template reliability (from MMseqs2
    e-values). Strong matches → full features, no template → zero features.

Sources:
  - AlphaFold2 (Jumper et al., 2021): template distogram binning
  - RNAPro GitHub: --use_template ca_precomputed confirms template feature approach
  - ADV1_IMPLEMENTATION_DESIGN.md Section 3.3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemplateEncoder(nn.Module):
    """Encode template C1' coordinates into pairwise feature tensor.

    Input: template coordinates (B, N, 3) + confidence scores (B, N)
    Output: template features (B, N, N, template_dim)
    """

    def __init__(
        self,
        num_distance_bins: int = 22,
        template_dim: int = 16,
        max_distance: float = 40.0,
    ):
        """
        Args:
            num_distance_bins: Total bins including "no template" bin.
                Default 22 = 21 distance bins (0-2, 2-4, ..., 40+) + 1 no_template bin.
            template_dim: Output feature dimension per pair.
            max_distance: Maximum distance for binning (Angstroms). Distances
                beyond this go into the last distance bin.
        """
        super().__init__()

        self.num_distance_bins = num_distance_bins
        self.template_dim = template_dim
        self.max_distance = max_distance

        # Number of actual distance bins (excluding no_template bin)
        num_dist_bins = num_distance_bins - 1  # 21

        # Compute bin edges: [0, 2, 4, ..., 40] → 21 bins
        # Bin i covers [edge[i], edge[i+1])
        # Last bin covers [max_distance, inf)
        bin_width = max_distance / num_dist_bins
        bin_edges = torch.arange(0, max_distance, bin_width)
        self.register_buffer('bin_edges', bin_edges)

        # Index of the "no template" bin (last bin)
        self.no_template_bin = num_distance_bins - 1

        # Linear projection: one-hot bins → template features
        self.projection = nn.Sequential(
            nn.Linear(num_distance_bins, template_dim),
            nn.LayerNorm(template_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        template_coords: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        """Encode template coordinates into features.

        Args:
            template_coords: (B, N, 3) — C1' coordinates from Approach 1.
                Zeros where no template data exists.
            confidence: (B, N) — per-residue confidence score (0.0 to 1.0).
                From Result.txt e-values. 0.0 means no template.

        Returns:
            template_features: (B, N, N, template_dim)
        """
        B, N, _ = template_coords.shape
        device = template_coords.device

        # Step 1: Compute pairwise distance matrix from template coords
        # diff[b, i, j, :] = coords[b, i, :] - coords[b, j, :]
        diff = template_coords.unsqueeze(2) - template_coords.unsqueeze(1)  # (B, N, N, 3)
        template_dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)  # (B, N, N)

        # Step 2: Bin distances into discrete categories
        # torch.bucketize assigns each distance to the appropriate bin
        binned = torch.bucketize(template_dist, self.bin_edges)  # (B, N, N), values 0 to num_dist_bins
        # Clamp to valid range [0, num_dist_bins - 1] for distance bins
        binned = torch.clamp(binned, 0, self.num_distance_bins - 2)

        # Step 3: Override bin to "no_template" where confidence is zero
        # Build a pairwise "has template" mask: both residues must have template data
        has_template_i = (confidence > 0).unsqueeze(2)   # (B, N, 1)
        has_template_j = (confidence > 0).unsqueeze(1)   # (B, 1, N)
        has_template_pair = has_template_i & has_template_j  # (B, N, N)

        # Where either residue has no template, set bin to no_template_bin
        binned = torch.where(
            has_template_pair,
            binned,
            torch.full_like(binned, self.no_template_bin),
        )

        # Step 4: One-hot encode the bins
        one_hot = F.one_hot(binned.long(), num_classes=self.num_distance_bins)  # (B, N, N, num_bins)
        one_hot = one_hot.float()

        # Step 5: Project to template feature dimension
        template_features = self.projection(one_hot)  # (B, N, N, template_dim)

        # Step 6: Mask by pairwise confidence
        # Pairwise confidence = min(confidence[i], confidence[j])
        conf_i = confidence.unsqueeze(2)  # (B, N, 1)
        conf_j = confidence.unsqueeze(1)  # (B, 1, N)
        conf_pair = torch.minimum(conf_i, conf_j)  # (B, N, N)

        # Scale features by confidence — zeros out features for no-template pairs
        template_features = template_features * conf_pair.unsqueeze(-1)  # (B, N, N, template_dim)

        return template_features
