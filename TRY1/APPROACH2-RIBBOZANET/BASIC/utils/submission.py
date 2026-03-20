"""
submission.py — Format predictions into the competition submission CSV.

Competition format (from https://www.kaggle.com/competitions/stanford-rna-3d-folding-2):
  Columns: ID, resname, resid, x_1, y_1, z_1, x_2, y_2, z_2, x_3, y_3, z_3, x_4, y_4, z_4, x_5, y_5, z_5
  
  - ID: "{target_id}_{resid}" (e.g., "R1107_1")
  - resname: nucleotide name (A, U, G, C)
  - resid: 1-based residue index
  - x_1..z_5: 5 sets of (x, y, z) coordinates for C1' atom
  
  All coordinates clipped to [-999.999, 9999.999].
  5 predictions per target — best-of-5 is scored by TM-score.
"""

import pandas as pd
import numpy as np
from typing import List, Dict


def format_submission(
    predictions: List[Dict],
    output_path: str = "submission.csv",
) -> pd.DataFrame:
    """Format predictions into competition submission CSV.

    Args:
        predictions: List of dicts, each with:
            - 'target_id': str (e.g., "R1107")
            - 'sequence': str (e.g., "AUGCUUAGCG")
            - 'coords_list': list of 5 np.ndarray, each (N, 3)
              These are the 5 diverse predictions for this target.
        output_path: Path to write the submission CSV.

    Returns:
        DataFrame of the submission.
    """
    rows = []

    for pred in predictions:
        target_id = pred['target_id']
        sequence = pred['sequence']
        coords_list = pred['coords_list']  # List of 5 arrays, each (N, 3)

        assert len(coords_list) == 5, (
            f"Need exactly 5 predictions per target, got {len(coords_list)} for {target_id}"
        )

        seq_len = len(sequence)

        for resid_0based in range(seq_len):
            resname = sequence[resid_0based]
            resid = resid_0based + 1  # 1-based

            row = {
                'ID': f"{target_id}_{resid}",
                'resname': resname,
                'resid': resid,
            }

            # Add 5 sets of coordinates
            for pred_idx in range(5):
                coords = coords_list[pred_idx]

                if resid_0based < coords.shape[0]:
                    x, y, z = coords[resid_0based]
                else:
                    # Padding — use zeros (shouldn't happen if lengths match)
                    x, y, z = 0.0, 0.0, 0.0

                # Clip coordinates to competition range
                x = np.clip(x, -999.999, 9999.999)
                y = np.clip(y, -999.999, 9999.999)
                z = np.clip(z, -999.999, 9999.999)

                suffix = pred_idx + 1
                row[f'x_{suffix}'] = round(float(x), 3)
                row[f'y_{suffix}'] = round(float(y), 3)
                row[f'z_{suffix}'] = round(float(z), 3)

            rows.append(row)

    # Create DataFrame with correct column order
    columns = ['ID', 'resname', 'resid']
    for i in range(1, 6):
        columns.extend([f'x_{i}', f'y_{i}', f'z_{i}'])

    df = pd.DataFrame(rows, columns=columns)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} — {len(df)} rows, {len(predictions)} targets")

    return df


def load_test_sequences(csv_path: str) -> List[Dict]:
    """Load test sequences from competition CSV.

    The test_sequences.csv has columns: target_id, sequence
    (and possibly others like sequence_length, temporal_cutoff, etc.)

    Args:
        csv_path: Path to test_sequences.csv

    Returns:
        List of dicts with 'target_id' and 'sequence' keys.
    """
    df = pd.read_csv(csv_path)

    # The column names may vary slightly
    id_col = None
    seq_col = None

    for col in df.columns:
        col_lower = col.lower()
        if 'target' in col_lower and 'id' in col_lower:
            id_col = col
        elif col_lower == 'id' and id_col is None:
            id_col = col
        elif 'seq' in col_lower and 'len' not in col_lower:
            seq_col = col

    if id_col is None:
        # Fallback: first column is ID
        id_col = df.columns[0]
    if seq_col is None:
        # Fallback: second column is sequence
        seq_col = df.columns[1]

    sequences = []
    for _, row in df.iterrows():
        sequences.append({
            'target_id': str(row[id_col]),
            'sequence': str(row[seq_col]),
        })

    print(f"Loaded {len(sequences)} test sequences from {csv_path}")
    return sequences
