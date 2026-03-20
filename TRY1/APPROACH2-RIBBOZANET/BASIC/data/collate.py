"""
collate.py — Custom collation for variable-length RNA sequences.

RNA sequences have different lengths, but PyTorch requires tensors in a batch
to have the same shape. This collate function pads shorter sequences to the
length of the longest sequence in the batch.
"""

import torch
from typing import Dict, List


def collate_rna_structures(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad and batch variable-length RNA structure data.

    Args:
        batch: List of dicts from RNAStructureDataset.__getitem__().
               Each dict has: tokens, distance_matrix, coords, mask, seq_len

    Returns:
        Batched dict with tensors padded to the max sequence length in the batch.
        - tokens: (B, max_N) — padded with OFFICIAL_PAD_IDX=4
        - distance_matrix: (B, max_N, max_N) — padded with 0
        - coords: (B, max_N, 3) — padded with 0
        - mask: (B, max_N) — False for padded positions
        - seq_lens: (B,) — actual lengths
    """
    batch_size = len(batch)
    max_len = max(item['seq_len'] for item in batch)

    # Initialize padded tensors
    tokens = torch.full((batch_size, max_len), fill_value=4, dtype=torch.long)  # pad=4
    distance_matrix = torch.zeros(batch_size, max_len, max_len, dtype=torch.float32)
    coords = torch.zeros(batch_size, max_len, 3, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    seq_lens = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        n = item['seq_len']
        tokens[i, :n] = item['tokens']
        distance_matrix[i, :n, :n] = item['distance_matrix']
        coords[i, :n, :] = item['coords']
        mask[i, :n] = item['mask']
        seq_lens[i] = n

    return {
        'tokens': tokens,
        'distance_matrix': distance_matrix,
        'coords': coords,
        'mask': mask,
        'seq_lens': seq_lens,
    }
