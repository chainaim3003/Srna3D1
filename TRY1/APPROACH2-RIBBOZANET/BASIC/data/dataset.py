"""
dataset.py — Load RNA structure training data.

Supports two data sources:
  1. Pre-processed pickle from:
     https://www.kaggle.com/datasets/shujun717/stanford3d-dataprocessing-pickle
  2. CIF files parsed via BioPython (from PDB or competition data)

Each training example is:
  - sequence: RNA string (e.g., "AUGCUUAGCG")
  - coords: numpy array of shape (N, 3) — C1' atom coordinates in Angstroms
  - distance_matrix: numpy array of shape (N, N) — pairwise C1' distances
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
        tokens: (N,) integer tensor — tokenized RNA sequence
        distance_matrix: (N, N) float tensor — true C1'-C1' pairwise distances
        coords: (N, 3) float tensor — true C1' coordinates (for evaluation)
        mask: (N,) boolean tensor — True for valid positions
        seq_len: int — actual sequence length before padding
    """

    def __init__(
        self,
        data: List[Dict],
        max_seq_len: int = 256,
        augment: bool = True,
    ):
        """
        Args:
            data: List of dicts with keys 'sequence' (str) and 'coords' (np.ndarray (N,3))
            max_seq_len: Maximum sequence length. Longer sequences are truncated.
            augment: Whether to apply random rotation/translation to coordinates.
        """
        self.data = data
        self.max_seq_len = max_seq_len
        self.augment = augment

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        sequence = item['sequence']
        coords = item['coords'].copy()  # (N, 3) numpy array

        # Truncate to max_seq_len
        seq_len = min(len(sequence), self.max_seq_len, coords.shape[0])
        sequence = sequence[:seq_len]
        coords = coords[:seq_len]

        # Data augmentation: random rotation and translation
        # This teaches the model that structure doesn't depend on orientation
        if self.augment:
            coords = random_rotation(coords)
            coords = random_translation(coords, scale=10.0)

        # Compute ground truth distance matrix from coordinates
        # scipy.spatial.distance.cdist would also work
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (N, N, 3)
        distance_matrix = np.sqrt((diff ** 2).sum(axis=-1))  # (N, N)

        # Tokenize sequence
        tokens = tokenize_sequence(sequence)

        # Convert to tensors
        coords_tensor = torch.tensor(coords, dtype=torch.float32)
        dist_tensor = torch.tensor(distance_matrix, dtype=torch.float32)
        mask = torch.ones(seq_len, dtype=torch.bool)

        return {
            'tokens': tokens,
            'distance_matrix': dist_tensor,
            'coords': coords_tensor,
            'mask': mask,
            'seq_len': seq_len,
        }


# ============================================================
# Data loading functions
# ============================================================

def load_from_pickle(pickle_path: str) -> List[Dict]:
    """Load pre-processed training data from pickle.

    The pickle from https://www.kaggle.com/datasets/shujun717/stanford3d-dataprocessing-pickle
    contains processed RNA structures with sequences and coordinates.

    Args:
        pickle_path: Path to the pickle file.

    Returns:
        List of dicts with 'sequence' and 'coords' keys.
    """
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(
            f"Pickle not found at {pickle_path}. "
            f"Download from: https://www.kaggle.com/datasets/shujun717/stanford3d-dataprocessing-pickle"
        )

    with open(pickle_path, 'rb') as f:
        raw_data = pickle.load(f)

    # The pickle format may vary — adapt this parsing based on actual content.
    # Expected: list of dicts or similar structure with sequence + coordinate data.
    # We handle common formats:

    processed = []

    if isinstance(raw_data, list):
        for entry in raw_data:
            if isinstance(entry, dict):
                seq = entry.get('sequence', entry.get('seq', ''))
                coords = entry.get('coords', entry.get('xyz', None))
                if seq and coords is not None:
                    coords = np.array(coords, dtype=np.float32)
                    if len(seq) == coords.shape[0] and coords.shape[1] == 3:
                        processed.append({'sequence': seq, 'coords': coords})

    elif isinstance(raw_data, dict):
        # Could be a dict mapping IDs to structures
        for key, entry in raw_data.items():
            if isinstance(entry, dict):
                seq = entry.get('sequence', entry.get('seq', ''))
                coords = entry.get('coords', entry.get('xyz', None))
                if seq and coords is not None:
                    coords = np.array(coords, dtype=np.float32)
                    if len(seq) == coords.shape[0] and coords.shape[1] == 3:
                        processed.append({'sequence': seq, 'coords': coords})

    print(f"Loaded {len(processed)} structures from pickle")

    if len(processed) == 0:
        print(
            "WARNING: No structures parsed from pickle. "
            "The pickle format may differ from expected. "
            "Inspect the pickle contents manually: "
            "  import pickle; data = pickle.load(open(path, 'rb')); print(type(data), len(data))"
        )

    return processed


def load_from_cif_directory(cif_dir: str) -> List[Dict]:
    """Load training data by parsing CIF files with BioPython.

    Extracts C1' atom coordinates from each RNA chain.

    Args:
        cif_dir: Path to directory containing .cif files.

    Returns:
        List of dicts with 'sequence' and 'coords' keys.
    """
    from utils.pdb_parser import extract_rna_structures_from_directory
    return extract_rna_structures_from_directory(cif_dir)


def load_training_data(config: dict) -> Tuple[List[Dict], List[Dict]]:
    """Load and split training data based on config.

    Args:
        config: Full config dictionary.

    Returns:
        (train_data, val_data) — lists of dicts with 'sequence' and 'coords'
    """
    data_cfg = config['data']

    # Try pickle first, then CIF directory
    all_data = []

    if data_cfg.get('train_pickle_path'):
        all_data = load_from_pickle(data_cfg['train_pickle_path'])

    if len(all_data) == 0 and data_cfg.get('cif_dir'):
        all_data = load_from_cif_directory(data_cfg['cif_dir'])

    if len(all_data) == 0:
        raise RuntimeError(
            "No training data loaded. Set data.train_pickle_path or data.cif_dir in config.yaml. "
            "Pickle: https://www.kaggle.com/datasets/shujun717/stanford3d-dataprocessing-pickle"
        )

    # Filter by max sequence length
    max_len = data_cfg.get('max_seq_len', 256)
    all_data = [d for d in all_data if len(d['sequence']) <= max_len]
    print(f"After filtering to max_len={max_len}: {len(all_data)} structures")

    # Split into train/val
    val_frac = data_cfg.get('val_fraction', 0.1)
    np.random.seed(config['training'].get('seed', 42))
    np.random.shuffle(all_data)

    val_size = max(1, int(len(all_data) * val_frac))
    val_data = all_data[:val_size]
    train_data = all_data[val_size:]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    return train_data, val_data
