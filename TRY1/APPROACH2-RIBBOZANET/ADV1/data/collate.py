"""
collate.py — Custom collation for variable-length RNA sequences WITH template support.

MODIFIED from BASIC/data/collate.py:
  - Adds template_coords (B, max_N, 3) and template_confidence (B, max_N) to batch.
"""

import torch
from typing import Dict, List


def collate_rna_structures(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad and batch variable-length RNA structure data including templates.

    Returns:
        - tokens: (B, max_N) — padded with 4
        - distance_matrix: (B, max_N, max_N) — padded with 0
        - coords: (B, max_N, 3) — padded with 0
        - mask: (B, max_N) — False for padded positions
        - seq_lens: (B,)
        - template_coords: (B, max_N, 3) — padded with 0
        - template_confidence: (B, max_N) — padded with 0
    """
    batch_size = len(batch)
    max_len = max(item['seq_len'] for item in batch)

    tokens = torch.full((batch_size, max_len), fill_value=4, dtype=torch.long)
    distance_matrix = torch.zeros(batch_size, max_len, max_len, dtype=torch.float32)
    coords = torch.zeros(batch_size, max_len, 3, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    seq_lens = torch.zeros(batch_size, dtype=torch.long)
    template_coords = torch.zeros(batch_size, max_len, 3, dtype=torch.float32)
    template_confidence = torch.zeros(batch_size, max_len, dtype=torch.float32)

    for i, item in enumerate(batch):
        n = item['seq_len']
        tokens[i, :n] = item['tokens']
        distance_matrix[i, :n, :n] = item['distance_matrix']
        coords[i, :n, :] = item['coords']
        mask[i, :n] = item['mask']
        seq_lens[i] = n
        template_coords[i, :n, :] = item['template_coords']
        template_confidence[i, :n] = item['template_confidence']

    return {
        'tokens': tokens,
        'distance_matrix': distance_matrix,
        'coords': coords,
        'mask': mask,
        'seq_lens': seq_lens,
        'template_coords': template_coords,
        'template_confidence': template_confidence,
    }
