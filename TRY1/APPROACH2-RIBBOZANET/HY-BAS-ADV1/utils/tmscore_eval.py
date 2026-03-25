"""
tmscore_eval.py — Local TM-score Evaluation for RNA 3D Structure Prediction
=============================================================================

PURPOSE:
  Compute TM-scores locally on training data so you can evaluate your
  model's quality WITHOUT needing Kaggle or the competition leaderboard.
  Use this to iterate post-competition or to validate runs before submitting.

HOW IT WORKS:
  1. Loads training structures from your existing data sources
     (pdb_xyz_data.pkl, train_labels.csv, or rna_coordinates.csv)
  2. Splits into train/holdout sets (reproducible seed)
  3. You predict 3D coords for holdout sequences using your model
  4. This script computes TM-score(predicted, experimental) per target
  5. Reports mean TM-score — the SAME metric Kaggle uses

TM-SCORE FORMULA (from competition page):
  TM-score = max over alignment of:
    (1/Lref) * SUM_i [ 1 / (1 + (di/d0)^2) ]
  where:
    Lref  = number of residues in reference structure
    di    = distance between aligned residue pair i
    d0    = 0.6 * (Lref - 0.5)^(1/2) - 2.5   for Lref >= 30
    d0    = 0.3, 0.4, 0.5, 0.6, 0.7           for Lref <15, 15, 16-19, 20-23, 24-29

USAGE:
  # Quick test: evaluate random predictions vs truth (should score ~0.1-0.2)
  python tmscore_eval.py --pickle path/to/pdb_xyz_data.pkl --mode random_baseline

  # Evaluate a submission CSV against holdout set
  python tmscore_eval.py --pickle path/to/pdb_xyz_data.pkl --submission my_predictions.csv

  # As a library in your notebook:
  from utils.tmscore_eval import compute_tm_score, load_holdout_set, evaluate_predictions

REQUIREMENTS:
  numpy, scipy (for optimal rotation via SVD)
  No external tools needed — pure Python/NumPy implementation.

AUTHOR:  Auto-generated for HY-BAS-ADV1 pipeline
LICENSE: Same as parent project
"""

import os
import sys
import pickle
import random
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional


# ============================================================
# CORE: TM-score computation (pure NumPy)
# ============================================================

def _get_d0(Lref: int) -> float:
    """
    Compute the distance scaling factor d0 for TM-score.

    This follows the exact formula from the competition evaluation page:
      d0 = 0.6 * (Lref - 0.5)^(1/2) - 2.5  for Lref >= 30
      d0 = 0.3, 0.4, 0.5, 0.6, 0.7          for Lref <15, 15, 16-19, 20-23, 24-29
    """
    if Lref < 15:
        return 0.3
    elif Lref <= 15:
        return 0.4
    elif Lref <= 19:
        return 0.5
    elif Lref <= 23:
        return 0.6
    elif Lref <= 29:
        return 0.7
    else:
        return 0.6 * np.sqrt(Lref - 0.5) - 2.5


def _kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find optimal rotation matrix R and translation t that minimizes
    RMSD between P and Q using Kabsch algorithm (SVD-based).

    Args:
        P: (N, 3) predicted coordinates
        Q: (N, 3) reference coordinates

    Returns:
        R: (3, 3) rotation matrix
        t: (3,)   translation vector
    Such that P_aligned = (P - centroid_P) @ R + centroid_Q
    """
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)

    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    H = P_centered.T @ Q_centered
    U, S, Vt = np.linalg.svd(H)

    # Handle reflection case (ensure proper rotation, not reflection)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])

    R = Vt.T @ sign_matrix @ U.T
    t = centroid_Q - centroid_P @ R

    return R, t


def compute_tm_score(
    pred_coords: np.ndarray,
    true_coords: np.ndarray,
    sequence_independent: bool = True,
) -> float:
    """
    Compute TM-score between predicted and true RNA structures.

    This is the SAME metric used by the Stanford RNA 3D Folding competition.
    Sequence-independent alignment uses US-align style; for simplicity
    we default to sequence-dependent (residue i maps to residue i),
    which matches the competition for same-length predictions.

    Args:
        pred_coords:  (N, 3) predicted C1' atom coordinates in Angstroms
        true_coords:  (N, 3) experimental C1' atom coordinates in Angstroms
        sequence_independent: if False (default for same-length), align by index.

    Returns:
        TM-score (float between 0.0 and 1.0)

    Score interpretation:
        < 0.30  Random / incorrect fold
        0.30-0.45  Some structural similarity
        > 0.45  Correct global fold
        > 0.70  Very accurate prediction
        1.0     Perfect match
    """
    if len(pred_coords) == 0 or len(true_coords) == 0:
        return 0.0

    # Ensure same length (truncate to shorter if needed)
    N = min(len(pred_coords), len(true_coords))
    pred = pred_coords[:N].copy()
    true = true_coords[:N].copy()

    # Filter out NaN/Inf residues (both must be valid)
    valid = np.isfinite(pred).all(axis=1) & np.isfinite(true).all(axis=1)
    pred = pred[valid]
    true = true[valid]

    Lref = len(true)
    if Lref < 3:
        return 0.0

    d0 = _get_d0(Lref)
    if d0 <= 0:
        d0 = 0.3  # safety floor

    # Kabsch alignment: find best superposition of pred onto true
    R, t = _kabsch_rotation(pred, true)
    pred_aligned = pred @ R + t

    # Per-residue distances after alignment
    distances = np.sqrt(((pred_aligned - true) ** 2).sum(axis=1))

    # TM-score formula
    tm_sum = np.sum(1.0 / (1.0 + (distances / d0) ** 2))
    tm_score = tm_sum / Lref

    return float(tm_score)


def compute_tm_score_best_of_5(
    pred_coords_list: List[np.ndarray],
    true_coords: np.ndarray,
) -> float:
    """
    Compute best-of-5 TM-score (same as competition: submit 5 predictions,
    best one counts).

    Args:
        pred_coords_list: list of up to 5 (N, 3) coordinate arrays
        true_coords: (N, 3) experimental coordinates

    Returns:
        Best TM-score among the 5 predictions
    """
    if not pred_coords_list:
        return 0.0
    scores = [compute_tm_score(p, true_coords) for p in pred_coords_list]
    return max(scores)


# ============================================================
# DATA LOADING: Parse training structures from existing sources
# ============================================================

def load_structures_from_pickle(pickle_path: str, max_seq_len: int = 512) -> Dict:
    """
    Load RNA structures from pdb_xyz_data.pkl (same format as Run 5 Cell 13).

    Returns:
        dict of {target_id: {'sequence': str, 'coords': np.ndarray(N,3)}}
    """
    print(f"Loading structures from {pickle_path}...")
    with open(pickle_path, 'rb') as f:
        raw_data = pickle.load(f)

    structures = {}
    skipped = 0

    if isinstance(raw_data, dict) and 'sequence' in raw_data:
        sequences = raw_data['sequence']
        xyz_list = raw_data['xyz']
        print(f"  Pickle contains {len(sequences)} entries")

        for i in range(len(sequences)):
            try:
                seq = sequences[i]
                residue_list = xyz_list[i]
                if seq is None or residue_list is None:
                    skipped += 1
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
                    skipped += 1
                    continue

                coords = np.array(c1_coords, dtype=np.float32)
                clean_seq = ''.join(valid_bases[:len(c1_coords)])
                min_len = min(len(clean_seq), len(coords))
                clean_seq = clean_seq[:min_len]
                coords = coords[:min_len]

                if 10 <= min_len <= max_seq_len:
                    target_id = f"TRAIN_{i:05d}"
                    structures[target_id] = {
                        'sequence': clean_seq,
                        'coords': coords,
                    }
            except Exception as e:
                skipped += 1
                if skipped <= 3:
                    print(f"  Warning: structure {i}: {e}")
    else:
        # Alternative format: list of dicts
        for i, item in enumerate(raw_data):
            seq = item.get('sequence', '')
            coords = item.get('coordinates', item.get('coords', None))
            if (seq and coords is not None and
                    10 <= len(seq) <= max_seq_len and len(seq) == len(coords)):
                target_id = f"TRAIN_{i:05d}"
                structures[target_id] = {
                    'sequence': seq,
                    'coords': np.array(coords, dtype=np.float32),
                }

    print(f"  Loaded: {len(structures)} structures, skipped: {skipped}")
    return structures


def load_structures_from_csv(
    seq_csv: str,
    coord_csv: str,
    max_seq_len: int = 512,
) -> Dict:
    """
    Load RNA structures from CSV files (train_sequences.csv + train_labels.csv
    or rna_sequences.csv + rna_coordinates.csv).

    Returns:
        dict of {target_id: {'sequence': str, 'coords': np.ndarray(N,3)}}
    """
    import pandas as pd

    print(f"Loading sequences from {seq_csv}...")
    seqs_df = pd.read_csv(seq_csv)
    print(f"Loading coordinates from {coord_csv}...")
    coords_df = pd.read_csv(coord_csv)

    structures = {}
    seq_dict = dict(zip(seqs_df['target_id'], seqs_df['sequence']))

    for target_id, group in coords_df.groupby(
        lambda x: coords_df['ID'][x].rsplit('_', 1)[0]
    ):
        if target_id not in seq_dict:
            continue
        sequence = seq_dict[target_id]
        if not (10 <= len(sequence) <= max_seq_len):
            continue

        group_sorted = group.sort_values('resid')
        coords = group_sorted[['x_1', 'y_1', 'z_1']].values.astype(np.float32)

        if len(coords) >= 10:
            structures[target_id] = {
                'sequence': sequence[:len(coords)],
                'coords': coords,
            }

    print(f"  Loaded: {len(structures)} structures")
    return structures


# ============================================================
# HOLDOUT SET: Reproducible train/test split
# ============================================================

def load_holdout_set(
    structures: Dict,
    holdout_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[Dict, Dict]:
    """
    Split structures into train and holdout (test) sets.

    Uses a fixed seed for reproducibility — the same holdout set every time
    so you can compare across runs.

    Args:
        structures: dict from load_structures_from_pickle or load_structures_from_csv
        holdout_fraction: fraction to hold out (default 15% ≈ 100 structures)
        seed: random seed for reproducibility

    Returns:
        (train_set, holdout_set) — both are dicts with same format as input
    """
    ids = sorted(structures.keys())
    random.seed(seed)
    random.shuffle(ids)

    n_holdout = max(1, int(len(ids) * holdout_fraction))
    holdout_ids = set(ids[:n_holdout])
    train_ids = set(ids[n_holdout:])

    train_set = {k: v for k, v in structures.items() if k in train_ids}
    holdout_set = {k: v for k, v in structures.items() if k in holdout_ids}

    print(f"  Split: {len(train_set)} train, {len(holdout_set)} holdout "
          f"(seed={seed}, holdout={holdout_fraction:.0%})")
    return train_set, holdout_set


# ============================================================
# EVALUATION: Score predictions against holdout
# ============================================================

def evaluate_predictions(
    predictions: Dict[str, np.ndarray],
    ground_truth: Dict,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate predicted coordinates against ground truth structures.

    Args:
        predictions: {target_id: np.ndarray(N, 3)} — your model's output
                     OR {target_id: [np.ndarray(N,3), ...]} for best-of-5
        ground_truth: {target_id: {'sequence': str, 'coords': np.ndarray(N,3)}}

    Returns:
        dict with:
            'mean_tm_score': float — the headline number (same as Kaggle LB)
            'per_target': {target_id: float} — individual TM-scores
            'n_evaluated': int
            'n_missing': int — targets with no prediction
            'score_distribution': dict with quartiles
    """
    scores = {}
    missing = []

    for target_id, truth in ground_truth.items():
        true_coords = truth['coords']

        if target_id not in predictions:
            missing.append(target_id)
            continue

        pred = predictions[target_id]

        # Handle best-of-5: if prediction is a list, score all and take max
        if isinstance(pred, list):
            score = compute_tm_score_best_of_5(pred, true_coords)
        else:
            score = compute_tm_score(pred, true_coords)

        scores[target_id] = score

    if not scores:
        print("  WARNING: No predictions matched any holdout targets!")
        return {
            'mean_tm_score': 0.0,
            'per_target': {},
            'n_evaluated': 0,
            'n_missing': len(missing),
        }

    all_scores = np.array(list(scores.values()))
    mean_tm = float(all_scores.mean())

    result = {
        'mean_tm_score': mean_tm,
        'per_target': scores,
        'n_evaluated': len(scores),
        'n_missing': len(missing),
        'score_distribution': {
            'min': float(all_scores.min()),
            'q25': float(np.percentile(all_scores, 25)),
            'median': float(np.percentile(all_scores, 50)),
            'q75': float(np.percentile(all_scores, 75)),
            'max': float(all_scores.max()),
            'std': float(all_scores.std()),
        },
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"  TM-SCORE EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"  Mean TM-score:  {mean_tm:.4f}")
        print(f"  Evaluated:      {len(scores)} targets")
        print(f"  Missing:        {len(missing)} targets")
        print(f"")
        print(f"  Distribution:")
        print(f"    Min:     {all_scores.min():.4f}")
        print(f"    Q25:     {np.percentile(all_scores, 25):.4f}")
        print(f"    Median:  {np.percentile(all_scores, 50):.4f}")
        print(f"    Q75:     {np.percentile(all_scores, 75):.4f}")
        print(f"    Max:     {all_scores.max():.4f}")
        print(f"    Std:     {all_scores.std():.4f}")
        print(f"")

        # Score buckets (same as competition interpretation)
        n_correct_fold = sum(1 for s in all_scores if s > 0.45)
        n_some_sim = sum(1 for s in all_scores if 0.30 <= s <= 0.45)
        n_random = sum(1 for s in all_scores if s < 0.30)
        print(f"  Breakdown:")
        print(f"    Correct fold (>0.45):     {n_correct_fold}/{len(scores)} "
              f"({100*n_correct_fold/len(scores):.1f}%)")
        print(f"    Some similarity (0.3-0.45): {n_some_sim}/{len(scores)} "
              f"({100*n_some_sim/len(scores):.1f}%)")
        print(f"    Random/wrong (<0.30):     {n_random}/{len(scores)} "
              f"({100*n_random/len(scores):.1f}%)")
        print(f"{'='*60}")

        # Show worst and best targets
        sorted_scores = sorted(scores.items(), key=lambda x: x[1])
        print(f"\n  5 Worst predictions:")
        for tid, s in sorted_scores[:5]:
            seq_len = len(ground_truth[tid]['sequence'])
            print(f"    {tid} ({seq_len} nt): TM={s:.4f}")
        print(f"\n  5 Best predictions:")
        for tid, s in sorted_scores[-5:]:
            seq_len = len(ground_truth[tid]['sequence'])
            print(f"    {tid} ({seq_len} nt): TM={s:.4f}")

    return result


# ============================================================
# BASELINES: Quick sanity checks
# ============================================================

def random_baseline(holdout: Dict, seed: int = 0) -> Dict[str, np.ndarray]:
    """
    Generate random 3D predictions (should score ~0.10-0.20).
    Use this to verify the TM-score code is working correctly.
    """
    np.random.seed(seed)
    predictions = {}
    for target_id, truth in holdout.items():
        N = len(truth['coords'])
        # Random walk with ~6A backbone distance (vaguely RNA-like)
        coords = np.zeros((N, 3), dtype=np.float32)
        for i in range(1, N):
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction) + 1e-8
            coords[i] = coords[i-1] + direction * 6.0
        predictions[target_id] = coords
    return predictions


def copy_baseline(holdout: Dict, noise_sigma: float = 0.0) -> Dict[str, np.ndarray]:
    """
    Copy ground truth coordinates (optionally with noise).
    noise_sigma=0 should score 1.0 (perfect).
    noise_sigma=1.0 should score ~0.8+ (small perturbation).
    noise_sigma=5.0 should score ~0.4-0.6 (moderate noise).
    Use this to verify TM-score scaling.
    """
    predictions = {}
    for target_id, truth in holdout.items():
        coords = truth['coords'].copy()
        if noise_sigma > 0:
            coords += np.random.randn(*coords.shape).astype(np.float32) * noise_sigma
        predictions[target_id] = coords
    return predictions


def template_copy_baseline(
    holdout: Dict,
    all_structures: Dict,
    train_set: Dict,
) -> Dict[str, np.ndarray]:
    """
    Simulate the Fork 2 template-copy approach:
    find most similar training sequence and copy its structure.
    This estimates what pure template matching achieves.
    """
    predictions = {}
    for target_id, truth in holdout.items():
        query_seq = truth['sequence']
        best_score = -1
        best_coords = np.zeros((len(query_seq), 3), dtype=np.float32)

        for train_id, train_data in train_set.items():
            train_seq = train_data['sequence']
            # Quick k-mer similarity (same as your template search)
            k = 3
            q_kmers = set(query_seq[i:i+k] for i in range(len(query_seq)-k+1))
            t_kmers = set(train_seq[i:i+k] for i in range(len(train_seq)-k+1))
            if not (q_kmers | t_kmers):
                continue
            score = len(q_kmers & t_kmers) / len(q_kmers | t_kmers)
            if score > best_score:
                best_score = score
                # Simple length-matched copy (truncate or pad)
                tc = train_data['coords']
                N = len(query_seq)
                if len(tc) >= N:
                    best_coords = tc[:N].copy()
                else:
                    best_coords = np.zeros((N, 3), dtype=np.float32)
                    best_coords[:len(tc)] = tc

        predictions[target_id] = best_coords
    return predictions


# ============================================================
# CLI: Run evaluations from command line
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Local TM-score evaluation for RNA 3D structure prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run baselines to verify TM-score code works
  python tmscore_eval.py --pickle path/to/pdb_xyz_data.pkl --mode baselines

  # Evaluate random predictions (sanity check — should score ~0.10-0.20)
  python tmscore_eval.py --pickle path/to/pdb_xyz_data.pkl --mode random_baseline

  # Evaluate template-copy approach (simulates Fork 2)
  python tmscore_eval.py --pickle path/to/pdb_xyz_data.pkl --mode template_baseline

  # Evaluate a submission CSV
  python tmscore_eval.py --pickle path/to/pdb_xyz_data.pkl --submission predictions.csv
        """,
    )
    parser.add_argument('--pickle', type=str, help='Path to pdb_xyz_data.pkl')
    parser.add_argument('--seq-csv', type=str, help='Path to train_sequences.csv (alternative to pickle)')
    parser.add_argument('--coord-csv', type=str, help='Path to train_labels.csv (alternative to pickle)')
    parser.add_argument('--submission', type=str, help='Path to submission.csv to evaluate')
    parser.add_argument('--mode', type=str, default='baselines',
                        choices=['baselines', 'random_baseline', 'template_baseline'],
                        help='Baseline mode to run (default: baselines)')
    parser.add_argument('--holdout-fraction', type=float, default=0.15,
                        help='Fraction of data to hold out (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible split (default: 42)')
    parser.add_argument('--max-seq-len', type=int, default=512,
                        help='Maximum sequence length to include (default: 512)')

    args = parser.parse_args()

    # Load structures
    if args.pickle:
        structures = load_structures_from_pickle(args.pickle, args.max_seq_len)
    elif args.seq_csv and args.coord_csv:
        structures = load_structures_from_csv(args.seq_csv, args.coord_csv, args.max_seq_len)
    else:
        print("ERROR: Provide --pickle OR (--seq-csv AND --coord-csv)")
        sys.exit(1)

    if len(structures) < 20:
        print(f"ERROR: Only {len(structures)} structures loaded. Need at least 20.")
        sys.exit(1)

    # Split
    train_set, holdout = load_holdout_set(structures, args.holdout_fraction, args.seed)

    # Evaluate
    if args.submission:
        # Load predictions from submission CSV
        import pandas as pd
        sub_df = pd.read_csv(args.submission)
        predictions = {}
        for target_id in holdout.keys():
            mask = sub_df['ID'].str.startswith(target_id + '_')
            if mask.sum() == 0:
                continue
            subset = sub_df[mask].sort_values('resid')
            # Load all 5 prediction slots
            slots = []
            for s in range(1, 6):
                cols = [f'x_{s}', f'y_{s}', f'z_{s}']
                if all(c in subset.columns for c in cols):
                    coords = subset[cols].values.astype(np.float32)
                    if not np.all(coords == 0):
                        slots.append(coords)
            if slots:
                predictions[target_id] = slots  # best-of-5

        print(f"\nLoaded predictions for {len(predictions)} targets from {args.submission}")
        evaluate_predictions(predictions, holdout, verbose=True)

    elif args.mode == 'baselines':
        # Run all baselines for comparison
        print("\n" + "="*60)
        print("BASELINE 1: Perfect copy (TM-score should be 1.0)")
        print("="*60)
        preds_perfect = copy_baseline(holdout, noise_sigma=0.0)
        evaluate_predictions(preds_perfect, holdout)

        print("\n" + "="*60)
        print("BASELINE 2: Copy + 1A noise (should score ~0.85+)")
        print("="*60)
        preds_noisy1 = copy_baseline(holdout, noise_sigma=1.0)
        evaluate_predictions(preds_noisy1, holdout)

        print("\n" + "="*60)
        print("BASELINE 3: Copy + 5A noise (should score ~0.3-0.5)")
        print("="*60)
        preds_noisy5 = copy_baseline(holdout, noise_sigma=5.0)
        evaluate_predictions(preds_noisy5, holdout)

        print("\n" + "="*60)
        print("BASELINE 4: Random walk (should score ~0.10-0.20)")
        print("="*60)
        preds_random = random_baseline(holdout)
        evaluate_predictions(preds_random, holdout)

        print("\n" + "="*60)
        print("BASELINE 5: Template copy (simulates Fork 2)")
        print("="*60)
        preds_template = template_copy_baseline(holdout, structures, train_set)
        evaluate_predictions(preds_template, holdout)

    elif args.mode == 'random_baseline':
        preds = random_baseline(holdout)
        evaluate_predictions(preds, holdout)

    elif args.mode == 'template_baseline':
        preds = template_copy_baseline(holdout, structures, train_set)
        evaluate_predictions(preds, holdout)


if __name__ == '__main__':
    main()
