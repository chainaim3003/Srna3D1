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
        if self.augment:
            coords = random_rotation(coords)
            coords = random_translation(coords, scale=10.0)

        # Compute ground truth distance matrix from coordinates
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

    VERIFIED FORMAT of pdb_xyz_data.pkl (from stanford3d-dataprocessing-pickle):
        {
            'sequence': list of 844 RNA sequence strings,
            'xyz': list of 844 items, where each item is a LIST of defaultdicts,
                   one defaultdict per nucleotide, with keys:
                     'phosphate':  np.array of shape (5, 3) — phosphate group atoms
                     'sugar_ring': np.array of shape (6, 3) — sugar ring atoms
                     'base':       np.array of shape (N, 3) — base atoms (varies)
                   C1' atom = sugar_ring[0] (first atom of sugar ring)
            'publication_date': list of 844 date strings,
            'filtered_cif_files': list of 844 CIF filenames,
            'cluster': pandas Series of 844 cluster IDs
        }
    """
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(
            f"Pickle not found at {pickle_path}. "
            f"Download from: https://www.kaggle.com/datasets/shujun717/stanford3d-dataprocessing-pickle"
        )

    with open(pickle_path, 'rb') as f:
        raw_data = pickle.load(f)

    processed = []

    # ============================================================
    # FORMAT: Dict with 'sequence' and 'xyz' as parallel lists
    # xyz[i] = list of defaultdicts (one per nucleotide)
    # Each defaultdict has 'phosphate', 'sugar_ring', 'base' arrays
    # C1' = sugar_ring[0] (first atom of sugar ring)
    # ============================================================
    if isinstance(raw_data, dict) and 'sequence' in raw_data and 'xyz' in raw_data:
        sequences = raw_data['sequence']
        xyz_list = raw_data['xyz']
        n = len(sequences)
        print(f"Pickle format: dict with {n} structures")
        print(f"  Keys: {list(raw_data.keys())}")

        skipped_nan = 0
        skipped_short = 0
        skipped_error = 0

        for i in range(n):
            try:
                seq = sequences[i]
                residue_list = xyz_list[i]  # List of defaultdicts, one per nucleotide

                if seq is None or residue_list is None:
                    skipped_error += 1
                    continue

                # Extract C1' coordinate from each nucleotide
                # C1' is the first atom in the sugar_ring group
                c1_coords = []
                valid_bases = []

                for j, residue_atoms in enumerate(residue_list):
                    if not hasattr(residue_atoms, 'keys') or 'sugar_ring' not in residue_atoms:
                        # Skip residues without sugar_ring data
                        continue

                    sugar_ring = residue_atoms['sugar_ring']
                    if sugar_ring is None or len(sugar_ring) == 0:
                        continue

                    # C1' = first atom of sugar_ring
                    c1_prime = sugar_ring[0]  # shape (3,)

                    # Check for NaN
                    if np.isnan(c1_prime).any():
                        continue

                    c1_coords.append(c1_prime)
                    if j < len(seq):
                        valid_bases.append(seq[j])

                if len(c1_coords) < 10:
                    skipped_short += 1
                    continue

                coords = np.array(c1_coords, dtype=np.float32)  # (N, 3)
                clean_seq = ''.join(valid_bases[:len(c1_coords)])

                # Final length check
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
        print(f"  Skipped: {skipped_short} too short, {skipped_nan} NaN, {skipped_error} errors")

    # ============================================================
    # FALLBACK: List of dicts (alternative format)
    # ============================================================
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

    else:
        print(f"WARNING: Unrecognized pickle format: {type(raw_data)}")
        if isinstance(raw_data, dict):
            print(f"  Keys: {list(raw_data.keys())}")

    print(f"Loaded {len(processed)} structures from pickle")

    if len(processed) == 0:
        print(
            "WARNING: No structures parsed from pickle. "
            "Inspect the pickle contents manually:\n"
            "  import pickle\n"
            "  data = pickle.load(open(path, 'rb'))\n"
            "  print(type(data))\n"
            "  if isinstance(data, dict): print(list(data.keys()))"
        )

    return processed


def load_from_cif_directory(cif_dir: str) -> List[Dict]:
    """Load training data by parsing CIF files with BioPython."""
    from utils.pdb_parser import extract_rna_structures_from_directory
    return extract_rna_structures_from_directory(cif_dir)


def load_training_data(config: dict) -> Tuple[List[Dict], List[Dict]]:
    """Load and split training data based on config."""
    data_cfg = config['data']

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
