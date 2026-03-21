"""
backbone.py — Load RibonanzaNet with SELECTIVE LAYER UNFREEZING.

MODIFIED from BASIC/models/backbone.py:
  - BASIC: freeze=true → ALL parameters frozen
  - ADV1:  freeze=false + freeze_first_n=7 → layers 0-6 frozen, layers 7-8 trainable

Official RibonanzaNet architecture (from Network.py, verified):
  - model.encoder: nn.Embedding(ntoken=5, ninp=256, padding_idx=4)
  - model.outer_product_mean: Outer_Product_Mean(in_dim=256, pairwise_dim=64)
  - model.pos_encoder: relpos(dim=64)
  - model.transformer_encoder: nn.ModuleList of 9 ConvTransformerEncoderLayers
  - model.decoder: nn.Linear(256, 2)

Each ConvTransformerEncoderLayer contains (~2-3M params):
  - self_attn (MultiHeadAttention with Q/K/V)
  - linear1, linear2 (feedforward)
  - conv (Conv1d, kernel=5 for layers 0-7, kernel=1 for layer 8)
  - triangle_update_out, triangle_update_in (TriangleMultiplicativeModule)
  - outer_product_mean, pair_transition, pairwise2heads
  - Various LayerNorms and Dropouts

Sources:
  - RibonanzaNet/Network.py (read in full)
  - RibonanzaNet/configs/pairwise.yaml: k=5, ninp=256, nlayers=9, nhead=8, pairwise_dimension=64
  - ADV1_IMPLEMENTATION_DESIGN.md Section 3.2
"""

import sys
import os
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any


# ============================================================
# Token mapping — must match official RibonanzaNet encoding
# From official Dataset.py: {'A': 0, 'C': 1, 'G': 2, 'U': 3}
# padding_idx=4 in nn.Embedding
# ============================================================
OFFICIAL_BASE_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
OFFICIAL_PAD_IDX = 4


# ============================================================
# Default config values from official configs/pairwise.yaml
# ALL fields that RibonanzaNet.__init__ and its layers access.
# ============================================================
OFFICIAL_DEFAULT_CONFIG = {
    'ntoken': 5,
    'ninp': 256,
    'nhead': 8,
    'nlayers': 9,
    'nclass': 2,
    'k': 5,
    'dropout': 0.05,
    'pairwise_dimension': 64,
    'use_triangular_attention': False,
}


def tokenize_sequence(sequence: str, base_to_idx: Dict[str, int] = None) -> torch.Tensor:
    """Convert RNA sequence string to integer tensor."""
    if base_to_idx is None:
        base_to_idx = OFFICIAL_BASE_TO_IDX
    tokens = []
    for base in sequence.upper():
        if base in base_to_idx:
            tokens.append(base_to_idx[base])
        else:
            tokens.append(OFFICIAL_PAD_IDX)
    return torch.tensor(tokens, dtype=torch.long)


class OfficialBackboneWrapper(nn.Module):
    """Wraps the official RibonanzaNet to expose single + pairwise representations.

    Supports both fully frozen mode (BASIC) and selective unfreezing (ADV1).
    """

    def __init__(self, repo_path: str, weights_path: str, config_overrides: Dict[str, Any] = None):
        super().__init__()

        repo_path = os.path.abspath(repo_path)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        try:
            from Network import RibonanzaNet as OfficialRibonanzaNet
        except ImportError as e:
            raise ImportError(
                f"Could not import RibonanzaNet from {repo_path}/Network.py. "
                f"Make sure you cloned https://github.com/Shujun-He/RibonanzaNet. Error: {e}"
            )

        # Load the official config from configs/pairwise.yaml
        cfg_dict = dict(OFFICIAL_DEFAULT_CONFIG)
        try:
            import yaml
            config_paths_to_try = [
                os.path.join(repo_path, "configs", "pairwise.yaml"),
                os.path.join(repo_path, "config.yaml"),
            ]
            for config_path in config_paths_to_try:
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        loaded = yaml.safe_load(f)
                    if loaded:
                        cfg_dict.update(loaded)
                    print(f"Loaded official config from {config_path}")
                    break
            else:
                print(f"No official config found, using defaults")
        except Exception as e:
            print(f"Could not load official config: {e}, using defaults")

        if config_overrides:
            cfg_dict.update(config_overrides)

        class ConfigNamespace:
            pass
        config_obj = ConfigNamespace()
        for k, v in cfg_dict.items():
            setattr(config_obj, k, v)

        self.model = OfficialRibonanzaNet(config_obj)

        weights_path = os.path.abspath(weights_path)
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded backbone weights from {weights_path}")
        else:
            print(f"WARNING: Weights not found at {weights_path}. Using random init.")

        self.ninp = cfg_dict.get('ninp', 256)
        self.pairwise_dim = cfg_dict.get('pairwise_dimension', 64)

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run RibonanzaNet and return single + pairwise representations."""
        B, N = tokens.shape

        if mask is not None:
            src_mask = mask.long()
        else:
            src_mask = torch.ones(B, N, dtype=torch.long, device=tokens.device)

        embedded = self.model.encoder(tokens)
        pairwise = self.model.outer_product_mean(embedded)
        pairwise = pairwise + self.model.pos_encoder(embedded)

        hidden = embedded
        for layer in self.model.transformer_encoder:
            result = layer(hidden, pairwise, src_mask=src_mask)
            if isinstance(result, tuple):
                hidden, pairwise = result
            else:
                hidden = result

        return hidden, pairwise


# ============================================================
# Factory function — MODIFIED for selective unfreezing
# ============================================================

def load_backbone(config: dict) -> nn.Module:
    """Load the RibonanzaNet backbone with optional selective unfreezing.

    Config options:
        backbone.freeze: true  → freeze ALL parameters (BASIC mode)
        backbone.freeze: false → selective unfreezing based on freeze_first_n
        backbone.freeze_first_n: int → freeze layers 0..(N-1), unfreeze layers N..8

    Args:
        config: Dictionary with 'backbone' section from config.yaml

    Returns:
        nn.Module with forward(tokens, mask) → (single_repr, pairwise_repr)
    """
    backbone_cfg = config['backbone']
    source = backbone_cfg.get('source', 'official')

    if source == 'official':
        model = OfficialBackboneWrapper(
            repo_path=backbone_cfg['repo_path'],
            weights_path=backbone_cfg['weights_path'],
            config_overrides={
                'ntoken': backbone_cfg.get('ntoken', 5),
                'ninp': backbone_cfg.get('ninp', 256),
                'pairwise_dimension': backbone_cfg.get('pairwise_dimension', 64),
            }
        )
    else:
        raise ValueError(f"Unknown backbone source: {source}. Use 'official'.")

    freeze = backbone_cfg.get('freeze', True)

    if freeze:
        # BASIC mode: freeze everything
        for param in model.parameters():
            param.requires_grad = False
        print("Backbone frozen — ALL parameters locked.")
    else:
        # ADV1 mode: selective unfreezing
        freeze_first_n = backbone_cfg.get('freeze_first_n', 7)
        nlayers = len(model.model.transformer_encoder)

        # Step 1: Freeze EVERYTHING first
        for param in model.parameters():
            param.requires_grad = False

        # Step 2: Unfreeze layers >= freeze_first_n
        unfrozen_layers = 0
        unfrozen_params = 0
        for i, layer in enumerate(model.model.transformer_encoder):
            if i >= freeze_first_n:
                for param in layer.parameters():
                    param.requires_grad = True
                    unfrozen_params += param.numel()
                unfrozen_layers += 1

        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

        print(f"Backbone selectively unfrozen:")
        print(f"  Total layers: {nlayers}")
        print(f"  Layers 0-{freeze_first_n - 1}: FROZEN")
        print(f"  Layers {freeze_first_n}-{nlayers - 1}: TRAINABLE ({unfrozen_layers} layers)")
        print(f"  Frozen params:   {frozen_params:,}")
        print(f"  Unfrozen params: {unfrozen_params:,}")

    return model
