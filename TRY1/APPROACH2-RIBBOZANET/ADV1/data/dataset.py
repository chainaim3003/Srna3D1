"""
dataset.py — Load RNA structure training data WITH template support.

MODIFIED from BASIC/data/dataset.py:
  - __getitem__ now returns template_coords and template_confidence
  - During training (Run1): templates are zeros (no training templates yet)
  - During inference: templates come from Approach 1 via template_loader.py

VERIFIED FORMAT of pdb_xyz_data.pkl:
  Dict with keys: 'sequence', 'xyz', 'publication_date', 'filtered_cif_files', 'cluster'
  xyz[i] = list of defaultdicts per nucleotide
  C1' atom = sugar_ring[0] (first atom of sugar ring)
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple

from models.backbone import OFFICIAL_BASE_TO_IDX, tokenize_sequence
from data.augmentation import random_rotation, random_translation


class RNAStructureDataset(Dataset):
    """Dataset of RNA structures for training the distance prediction head.

    Each item returns:
        tokens: (N,) integer tensor
        distance_matrix: (N, N) float tensor
        coords: (N, 3) float tensor
        mask: (N,) boolean tensor
        seq_len: int
        template_coords: (N, 3) float tensor — template C1' coords (zeros if no template)
        template_confidence: (N,) float tensor — per-residue confidence (zeros if no template)
    """

    def __init__(
        self,
        data: List[Dict],
        max_seq_len: int = 256,
        augment: bool = True,
        templates: Dict[str, Dict] = None,
    ):
        """
        Args:
            data: List of dicts with keys 'sequence' and 'coords' (np.ndarray (N,3))
            max_seq_len: Maximum sequence length.
            augment: Whether to apply random rotation/translation.
            templates: Optional dict from template_loader.load_test_templates().
                       Maps target_id → {coords, confidence}. Used during inference.
                       During training, this is None (templates are zeros).
        """
        self.data = data
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.templates = templates or {}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        sequence = item['sequence']
        coords = item['coords'].copy()

        seq_len = min(len(sequence), self.max_seq_len, coords.shape[0])
        sequence = sequence[:seq_len]
        coords = coords[:seq_len]

        if self.augment:
            coords = random_rotation(coords)
            coords = random_translation(coords, scale=10.0)

        # Distance matrix
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        distance_matrix = np.sqrt((diff ** 2).sum(axis=-1))

        # Tokenize
        tokens = tokenize_sequence(sequence)

        # Template data
        target_id = item.get('target_id', None)
        if target_id and target_id in self.templates:
            tmpl = self.templates[target_id]
            tmpl_coords = tmpl['coords'][:seq_len].copy()
            tmpl_confidence = tmpl['confidence'][:seq_len].copy()
            # Pad if template is shorter than sequence
            if tmpl_coords.shape[0] < seq_len:
                pad_len = seq_len - tmpl_coords.shape[0]
                tmpl_coords = np.pad(tmpl_coords, ((0, pad_len), (0, 0)), mode='constant')
                tmpl_confidence = np.pad(tmpl_confidence, (0, pad_len), mode='constant')
        else:
            # No template — zeros
            tmpl_coords = np.zeros((seq_len, 3), dtype=np.float32)
            tmpl_confidence = np.zeros(seq_len, dtype=np.float32)

        return {
            'tokens': tokens,
            'distance_matrix': torch.tensor(distance_matrix, dtype=torch.float32),
            'coords': torch.tensor(coords, dtype=torch.float32),
            'mask': torch.ones(seq_len, dtype=torch.bool),
            'seq_len': seq_len,
            'template_coords': torch.tensor(tmpl_coords, dtype=torch.float32),
            'template_confidence': torch.tensor(tmpl_confidence, dtype=torch.float32),
        }


# ============================================================
# Data loading functions (same as BASIC with verified pickle format)
# ============================================================

def load_from_pickle(pickle_path: str) -> List[Dict]:
    """Load training data from pdb_xyz_data.pkl.

    VERIFIED FORMAT: dict of parallel lists, xyz entries are lists of defaultdicts,
    C1' = sugar_ring[0].
    """
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(
            f"Pickle not found at {pickle_path}. "
            f"Download from: https://www.kaggle.com/datasets/shujun717/stanford3d-dataprocessing-pickle"
        )

    with open(pickle_path, 'rb') as f:
        raw_data = pickle.load(f)

    processed = []

    if isinstance(raw_data, dict) and 'sequence' in raw_data and 'xyz' in raw_data:
        sequences = raw_data['sequence']
        xyz_list = raw_data['xyz']
        n = len(sequences)
        print(f"Pickle format: dict with {n} structures")

        skipped_short = 0
        skipped_error = 0

        for i in range(n):
            try:
                seq = sequences[i]
                residue_list = xyz_list[i]

                if seq is None or residue_list is None:
                    skipped_error += 1
                    continue

                c1_coords = []
                valid_bases = []

                for j, residue_atoms in enumerate(residue_list):
                    if not hasattr(residue_atoms, 'keys') or 'sugar_ring' not in residue_atoms:
                        continue
                    sugar_ring = residue_atoms['sugar_ring']
                    if sugar_ring is None or len(sugar_ring) == 0:
                        continue
                    c1_prime = sugar_ring[0]
                    if np.isnan(c1_prime).any():
                        continue
                    c1_coords.append(c1_prime)
                    if j < len(seq):
                        valid_bases.append(seq[j])

                if len(c1_coords) < 10:
                    skipped_short += 1
                    continue

                coords = np.array(c1_coords, dtype=np.float32)
                clean_seq = ''.join(valid_bases[:len(c1_coords)])
                min_len = min(len(clean_seq), coords.shape[0])
                clean_seq = clean_seq[:min_len]
                coords = coords[:min_len]

                if min_len < 10:
                    skipped_short += 1
                    continue

                processed.append({'sequence': clean_seq, 'coords': coords})

            except Exception as e:
                skipped_error += 1
                if skipped_error <= 3:
                    print(f"  Warning: Error on structure {i}: {e}")

        print(f"  Parsed: {len(processed)} structures")
        print(f"  Skipped: {skipped_short} too short, {skipped_error} errors")

    elif isinstance(raw_data, list):
        print(f"Pickle format: list of {len(raw_data)} items")
        for entry in raw_data:
            if isinstance(entry, dict):
                seq = entry.get('sequence', entry.get('seq', ''))
                coords = entry.get('coords', entry.get('xyz', None))
                if seq and coords is not None:
                    coords = np.array(coords, dtype=np.float32)
                    if coords.ndim == 2 and coords.shape[1] == 3 and len(seq) >= 10:
                        processed.append({'sequence': seq, 'coords': coords})

    print(f"Loaded {len(processed)} structures from pickle")
    return processed


def load_from_cif_directory(cif_dir: str) -> List[Dict]:
    from utils.pdb_parser import extract_rna_structures_from_directory
    return extract_rna_structures_from_directory(cif_dir)


def load_training_data(config: dict) -> Tuple[List[Dict], List[Dict]]:
    """Load and split training data."""
    data_cfg = config['data']
    all_data = []

    if data_cfg.get('train_pickle_path'):
        all_data = load_from_pickle(data_cfg['train_pickle_path'])

    if len(all_data) == 0 and data_cfg.get('cif_dir'):
        all_data = load_from_cif_directory(data_cfg['cif_dir'])

    if len(all_data) == 0:
        raise RuntimeError(
            "No training data loaded. Set data.train_pickle_path or data.cif_dir in config.yaml."
        )

    max_len = data_cfg.get('max_seq_len', 256)
    all_data = [d for d in all_data if len(d['sequence']) <= max_len]
    print(f"After filtering to max_len={max_len}: {len(all_data)} structures")

    val_frac = data_cfg.get('val_fraction', 0.1)
    np.random.seed(config['training'].get('seed', 42))
    np.random.shuffle(all_data)

    val_size = max(1, int(len(all_data) * val_frac))
    val_data = all_data[:val_size]
    train_data = all_data[val_size:]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    return train_data, val_data
