"""
backbone.py — Load frozen RibonanzaNet and extract representations.

Supports two loading modes:
  - "official": Clone github.com/Shujun-He/RibonanzaNet, import Network.py directly
  - "multimolecule": Use `pip install multimolecule` HuggingFace wrapper

Official RibonanzaNet architecture (from Network.py):
  - Embedding: nn.Embedding(ntoken=5, ninp=256, padding_idx=4)
    Tokens: A=0, C=1, G=2, U=3, pad=4
  - Transformer encoder layers with internal pairwise_features (dim=64)
  - Outer product mean to build pairwise representation
  - Triangle multiplicative updates on pairwise features
  - Decoder: nn.Linear(ninp, nclass) for chemical mapping prediction

We need TWO outputs:
  1. single_repr: (B, N, ninp) — per-nucleotide features
  2. pairwise_repr: (B, N, N, pairwise_dimension) — per-pair features

Sources:
  - Official repo: https://github.com/Shujun-He/RibonanzaNet
  - Network.py: defines RibonanzaNet class with transformer_encoder, pairwise_features
  - Config: configs/pairwise.yaml — pairwise_dimension=64, ninp=256, k=5
  - Weights: https://www.kaggle.com/datasets/shujun717/ribonanzanet-weights
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
# This is the single source of truth if the YAML file can't be loaded.
# ============================================================
OFFICIAL_DEFAULT_CONFIG = {
    'ntoken': 5,                    # A=0, C=1, G=2, U=3, pad=4
    'ninp': 256,                    # Embedding / hidden dimension
    'nhead': 8,                     # Number of attention heads
    'nlayers': 9,                   # Number of transformer layers
    'nclass': 2,                    # Output classes (chemical mapping: 2A3 + DMS)
    'k': 5,                         # Convolution kernel size
    'dropout': 0.05,                # Dropout rate
    'pairwise_dimension': 64,       # Pairwise feature dimension
    'use_triangular_attention': False,  # Whether to use triangle attention
}


def tokenize_sequence(sequence: str, base_to_idx: Dict[str, int] = None) -> torch.Tensor:
    """Convert RNA sequence string to integer tensor.

    Args:
        sequence: RNA sequence like "AUGCUUAGCG"
        base_to_idx: Mapping from nucleotide to integer.
                     Defaults to official RibonanzaNet encoding.

    Returns:
        Tensor of shape (seq_len,) with integer token IDs.
    """
    if base_to_idx is None:
        base_to_idx = OFFICIAL_BASE_TO_IDX

    tokens = []
    for base in sequence.upper():
        if base in base_to_idx:
            tokens.append(base_to_idx[base])
        else:
            tokens.append(OFFICIAL_PAD_IDX)

    return torch.tensor(tokens, dtype=torch.long)


# ============================================================
# Option A: Load from official RibonanzaNet repo
# ============================================================

class OfficialBackboneWrapper(nn.Module):
    """Wraps the official RibonanzaNet to expose single + pairwise representations.

    The official forward() only returns the decoder output (chemical mapping predictions).
    We manually run through the layers to capture the internal pairwise_features tensor
    and the pre-decoder hidden state as the single representation.

    Requirements:
        - Clone https://github.com/Shujun-He/RibonanzaNet
        - Set repo_path in config to point to the cloned directory
        - Download weights from kaggle.com/datasets/shujun717/ribonanzanet-weights
    """

    def __init__(self, repo_path: str, weights_path: str, config_overrides: Dict[str, Any] = None):
        super().__init__()

        # Add the repo to Python path so we can import Network.py
        repo_path = os.path.abspath(repo_path)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        # Import the official model class
        try:
            from Network import RibonanzaNet as OfficialRibonanzaNet
        except ImportError as e:
            raise ImportError(
                f"Could not import RibonanzaNet from {repo_path}/Network.py. "
                f"Make sure you cloned https://github.com/Shujun-He/RibonanzaNet "
                f"and set backbone.repo_path correctly in config.yaml. Error: {e}"
            )

        # Load the official config from configs/pairwise.yaml
        # This is the ACTUAL config file location in the official repo.
        cfg_dict = dict(OFFICIAL_DEFAULT_CONFIG)  # Start with safe defaults

        try:
            import yaml
            # Try the actual config path first: configs/pairwise.yaml
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

        # Apply any overrides from our config.yaml
        if config_overrides:
            cfg_dict.update(config_overrides)

        # Convert dict to namespace object (official code accesses config.ninp, etc.)
        class ConfigNamespace:
            pass
        config_obj = ConfigNamespace()
        for k, v in cfg_dict.items():
            setattr(config_obj, k, v)

        # Instantiate the official model
        self.model = OfficialRibonanzaNet(config_obj)

        # Load pretrained weights
        weights_path = os.path.abspath(weights_path)
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
            # Handle potential key mismatches
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded backbone weights from {weights_path}")
        else:
            print(f"WARNING: Weights not found at {weights_path}. Using random init.")

        # Store dimensions for downstream use
        self.ninp = cfg_dict.get('ninp', 256)
        self.pairwise_dim = cfg_dict.get('pairwise_dimension', 64)

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run RibonanzaNet and return single + pairwise representations.

        Args:
            tokens: (B, N) integer token tensor
            mask: (B, N) boolean mask (True=valid, False=padding).
                  If None, all positions are treated as valid.

        Returns:
            single_repr: (B, N, ninp) — per-nucleotide features
            pairwise_repr: (B, N, N, pairwise_dim) — per-pair features
        """
        B, N = tokens.shape

        # Build src_mask for padding positions
        if mask is not None:
            src_mask = mask.long()
        else:
            src_mask = torch.ones(B, N, dtype=torch.long, device=tokens.device)

        # Run through the embedding layer
        # Official code: self.encoder(src) → (B, N, ninp)
        embedded = self.model.encoder(tokens)

        # Build initial pairwise features via outer product mean + relpos
        # This mimics what the official forward() does before the transformer layers
        pairwise = self.model.outer_product_mean(embedded)
        pairwise = pairwise + self.model.pos_encoder(embedded)

        # Run through transformer encoder layers
        # Each ConvTransformerEncoderLayer.forward() returns (src, pairwise_features)
        hidden = embedded
        for layer in self.model.transformer_encoder:
            result = layer(hidden, pairwise, src_mask=src_mask)
            if isinstance(result, tuple):
                hidden, pairwise = result
            else:
                hidden = result

        single_repr = hidden         # (B, N, ninp)
        pairwise_repr = pairwise     # (B, N, N, pairwise_dim)

        return single_repr, pairwise_repr


# ============================================================
# Option B: Load from multimolecule (HuggingFace wrapper)
# ============================================================

class MultimoleculeBackboneWrapper(nn.Module):
    """Wraps the multimolecule/ribonanzanet HuggingFace model.

    The multimolecule package provides a clean API but may not expose
    internal pairwise representations. In that case, we construct pair
    representations from the single representation using outer product.

    Install: pip install multimolecule
    Model: multimolecule/ribonanzanet on HuggingFace Hub

    IMPORTANT NOTES from HuggingFace model card:
    - This is an UNOFFICIAL implementation
    - The official repo is at Shujun-He/RibonanzaNet
    - There are known differences in attention mask handling
    """

    def __init__(self, model_name: str = "multimolecule/ribonanzanet"):
        super().__init__()

        try:
            from multimolecule import RnaTokenizer, RibonanzaNetModel
        except ImportError:
            raise ImportError(
                "multimolecule package not installed. "
                "Install with: pip install multimolecule"
            )

        self.tokenizer = RnaTokenizer.from_pretrained(model_name)
        self.model = RibonanzaNetModel.from_pretrained(model_name)

        self.ninp = self.model.config.hidden_size
        self.pairwise_dim = 64

        self.pair_builder = PairRepresentationBuilder(
            single_dim=self.ninp,
            pair_dim=self.pairwise_dim,
        )

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_mask = mask.long() if mask is not None else None
        outputs = self.model(input_ids=tokens, attention_mask=attention_mask)
        single_repr = outputs.last_hidden_state
        pairwise_repr = self.pair_builder(single_repr)
        return single_repr, pairwise_repr


# ============================================================
# Pair Representation Builder (used when pairwise not available)
# ============================================================

class PairRepresentationBuilder(nn.Module):
    """Constructs pairwise features from single-residue features."""

    def __init__(self, single_dim: int = 256, pair_dim: int = 64, max_rel_pos: int = 32):
        super().__init__()
        self.max_rel_pos = max_rel_pos
        self.rel_pos_embed = nn.Embedding(2 * max_rel_pos + 1, pair_dim)
        self.projection = nn.Sequential(
            nn.Linear(single_dim * 2, pair_dim),
            nn.LayerNorm(pair_dim),
            nn.ReLU(),
        )

    def forward(self, single_repr: torch.Tensor) -> torch.Tensor:
        B, N, d = single_repr.shape
        device = single_repr.device

        s_i = single_repr.unsqueeze(2).expand(B, N, N, d)
        s_j = single_repr.unsqueeze(1).expand(B, N, N, d)

        pair_concat = torch.cat([s_i, s_j], dim=-1)
        pair_features = self.projection(pair_concat)

        pos = torch.arange(N, device=device)
        rel_pos = (pos.unsqueeze(0) - pos.unsqueeze(1))
        rel_pos = rel_pos.clamp(-self.max_rel_pos, self.max_rel_pos) + self.max_rel_pos
        rel_pos_features = self.rel_pos_embed(rel_pos)

        pair_features = pair_features + rel_pos_features.unsqueeze(0)
        return pair_features


# ============================================================
# Factory function
# ============================================================

def load_backbone(config: dict) -> nn.Module:
    """Load the RibonanzaNet backbone based on config.

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
    elif source == 'multimolecule':
        model = MultimoleculeBackboneWrapper()
    else:
        raise ValueError(f"Unknown backbone source: {source}. Use 'official' or 'multimolecule'.")

    # Freeze if configured
    if backbone_cfg.get('freeze', True):
        for param in model.parameters():
            param.requires_grad = False
        print("Backbone frozen — parameters will NOT be updated during training.")

    return model
