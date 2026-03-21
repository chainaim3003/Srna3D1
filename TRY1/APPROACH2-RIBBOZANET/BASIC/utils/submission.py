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

    VERIFIED FORMAT of test_sequences.csv:
      Columns: target_id, sequence, temporal_cutoff, description,
               stoichiometry, all_sequences, ligand_ids, ligand_SMILES

      - target_id: e.g. "8ZNQ"
      - sequence: the actual RNA bases, e.g. "ACCGUGACGGGCCUUUUGGCUAUACGCGGU"
      - all_sequences: FASTA-formatted with HEADER, e.g.
            ">8ZNQ_1|Chain A[auth A]|RNA (30-MER)\nACCGUGACGGGCCUUUUGGCUAUACGCGGU"

    IMPORTANT: We must use the 'sequence' column, NOT 'all_sequences'.
    The all_sequences column includes FASTA headers like ">8ZNQ_1|Chain A..."
    which would be read character-by-character as residue names.
    """
    df = pd.read_csv(csv_path)

    print(f"Test CSV columns: {list(df.columns)}")
    print(f"Test CSV shape: {df.shape}")

    # ============================================================
    # Find the target_id column
    # ============================================================
    id_col = None
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower == 'target_id':
            id_col = col
            break
        elif col_lower == 'id':
            id_col = col
            # Don't break — keep looking for exact 'target_id' match

    if id_col is None:
        id_col = df.columns[0]
        print(f"  Warning: No target_id column found, using first column: '{id_col}'")

    # ============================================================
    # Find the sequence column — MUST be exact 'sequence', NOT 'all_sequences'
    # The all_sequences column has FASTA headers that break everything.
    # ============================================================
    seq_col = None
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower == 'sequence':
            seq_col = col
            break  # Exact match — stop immediately

    if seq_col is None:
        # Fallback: look for column with 'seq' but not 'all_seq'
        for col in df.columns:
            col_lower = col.lower().strip()
            if 'seq' in col_lower and 'all' not in col_lower and 'len' not in col_lower:
                seq_col = col
                break

    if seq_col is None:
        seq_col = df.columns[1]
        print(f"  Warning: No sequence column found, using second column: '{seq_col}'")

    print(f"  Using ID column: '{id_col}', Sequence column: '{seq_col}'")

    sequences = []
    for _, row in df.iterrows():
        target_id = str(row[id_col]).strip()
        sequence = str(row[seq_col]).strip()

        # Validate: sequence should only contain RNA bases
        valid_bases = set('AUGCaugcNn')
        if not all(c in valid_bases for c in sequence):
            # Filter to only valid RNA characters
            filtered = ''.join(c for c in sequence if c.upper() in 'AUGCN')
            if len(filtered) > 0:
                print(f"  Warning: {target_id} sequence had non-RNA chars, filtered {len(sequence)} → {len(filtered)}")
                sequence = filtered
            else:
                print(f"  Warning: {target_id} has no valid RNA bases in sequence column!")
                continue

        sequences.append({
            'target_id': target_id,
            'sequence': sequence.upper(),
        })

    print(f"Loaded {len(sequences)} test sequences")

    # Show first sequence as sanity check
    if sequences:
        s = sequences[0]
        print(f"  First: {s['target_id']} — {s['sequence'][:40]}... (len={len(s['sequence'])})")

    return sequences
