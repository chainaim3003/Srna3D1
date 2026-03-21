"""
template_loader.py — Parse Approach 1 output files into per-target template tensors.

Reads:
  1. submission.csv from Approach 1 (template C1' coordinates)
  2. Result.txt from MMseqs2 (e-values, alignments for confidence scores)

Returns a dict mapping target_id → {coords, confidence, num_hits, best_evalue}
ready for use by TemplateEncoder.

Sources:
  - DasLab create_templates: https://github.com/DasLab/create_templates
  - ADV1_IMPLEMENTATION_DESIGN.md Section 3.4
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Optional


def parse_result_txt(result_txt_path: str, confidence_scale: float = 15.0,
                     gap_penalty: float = 0.3) -> Dict[str, Dict]:
    """Parse MMseqs2 Result.txt to extract per-target search statistics.

    Args:
        result_txt_path: Path to Result.txt (tab-separated).
            Columns: query, target, evalue, qstart, qend, tstart, tend, qaln, taln
        confidence_scale: Normalization for e-value → confidence conversion.
            confidence = min(1.0, -log10(evalue) / confidence_scale)
            With scale=15: 1e-15 → 1.0, 1e-8 → 0.53, 1e-3 → 0.2
        gap_penalty: Confidence multiplier for gap-filled residues.

    Returns:
        Dict mapping target_id → {
            'num_hits': int,
            'best_evalue': float,
            'confidence_score': float (0.0 to 1.0),
            'best_qaln': str (query alignment string, for gap detection),
            'best_taln': str (target alignment string),
            'qstart': int,
            'qend': int,
        }
    """
    if not os.path.exists(result_txt_path):
        print(f"Warning: Result.txt not found at {result_txt_path}")
        return {}

    results = {}

    with open(result_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue

            query = parts[0]
            evalue = float(parts[2]) if len(parts) > 2 else 1.0

            if query not in results:
                results[query] = {
                    'num_hits': 0,
                    'best_evalue': float('inf'),
                    'confidence_score': 0.0,
                    'best_qaln': '',
                    'best_taln': '',
                    'qstart': 0,
                    'qend': 0,
                }

            results[query]['num_hits'] += 1

            if evalue < results[query]['best_evalue']:
                results[query]['best_evalue'] = evalue

                # Compute confidence from e-value
                if evalue > 0:
                    raw_conf = -np.log10(max(evalue, 1e-30)) / confidence_scale
                else:
                    raw_conf = 1.0
                results[query]['confidence_score'] = min(1.0, max(0.0, raw_conf))

                # Save alignment info from best hit
                if len(parts) >= 9:
                    results[query]['best_qaln'] = parts[7]
                    results[query]['best_taln'] = parts[8]
                if len(parts) >= 5:
                    results[query]['qstart'] = int(parts[3]) if parts[3].isdigit() else 0
                    results[query]['qend'] = int(parts[4]) if parts[4].isdigit() else 0

    print(f"Parsed Result.txt: {len(results)} targets with hits")
    for tid, info in list(results.items())[:3]:
        print(f"  {tid}: {info['num_hits']} hits, best_evalue={info['best_evalue']:.2e}, "
              f"confidence={info['confidence_score']:.2f}")

    return results


def build_per_residue_confidence(
    seq_len: int,
    search_info: Optional[Dict],
    gap_penalty: float = 0.3,
) -> np.ndarray:
    """Build per-residue confidence array from search results.

    Args:
        seq_len: Length of the target sequence.
        search_info: Dict from parse_result_txt for this target.
            None if no Result.txt or no hits.
        gap_penalty: Confidence multiplier for gap-filled residues.

    Returns:
        confidence: np.ndarray of shape (seq_len,), values 0.0 to 1.0.
    """
    confidence = np.zeros(seq_len, dtype=np.float32)

    if search_info is None:
        return confidence

    base_conf = search_info['confidence_score']
    if base_conf <= 0:
        return confidence

    qaln = search_info.get('best_qaln', '')
    qstart = search_info.get('qstart', 0)

    if qaln:
        # Use alignment to determine which residues were directly matched vs gap-filled
        residue_idx = max(0, qstart - 1)  # Convert 1-based to 0-based
        for aln_char in qaln:
            if residue_idx >= seq_len:
                break
            if aln_char == '-':
                # Gap in query → this position was not in the template
                # Don't advance residue_idx for gaps in query
                continue
            else:
                # Matched position → full confidence
                confidence[residue_idx] = base_conf
                residue_idx += 1

        # Positions not covered by alignment get reduced confidence
        # (they were gap-filled/interpolated by create_templates_csv.py)
        for i in range(seq_len):
            if confidence[i] == 0 and base_conf > 0:
                # Position not covered by alignment → gap-filled
                confidence[i] = base_conf * gap_penalty
    else:
        # No alignment info available — uniform confidence for all residues
        confidence[:] = base_conf

    return confidence


def load_test_templates(
    template_csv_path: str,
    result_txt_path: str = None,
    confidence_scale: float = 15.0,
    gap_penalty: float = 0.3,
) -> Dict[str, Dict]:
    """Load Approach 1 template data for all test targets.

    Args:
        template_csv_path: Path to Approach 1 submission.csv.
            Columns: ID, resname, resid, x_1, y_1, z_1, ..., x_5, y_5, z_5
        result_txt_path: Path to Result.txt (optional, for confidence scores).
        confidence_scale: E-value → confidence normalization (see parse_result_txt).
        gap_penalty: Confidence reduction for gap-filled residues.

    Returns:
        Dict mapping target_id → {
            'coords': np.ndarray (N, 3) — C1' coordinates from best template,
            'confidence': np.ndarray (N,) — per-residue confidence (0.0 to 1.0),
            'num_hits': int — number of MMseqs2 hits,
            'best_evalue': float — best e-value,
        }
    """
    if not os.path.exists(template_csv_path):
        print(f"Warning: Template CSV not found at {template_csv_path}")
        return {}

    # Parse Result.txt for confidence info
    search_results = {}
    if result_txt_path and os.path.exists(result_txt_path):
        search_results = parse_result_txt(
            result_txt_path,
            confidence_scale=confidence_scale,
            gap_penalty=gap_penalty,
        )

    # Read template coordinates from submission.csv
    df = pd.read_csv(template_csv_path)
    print(f"Template CSV: {df.shape[0]} rows, columns: {list(df.columns)[:8]}...")

    # Extract target_id from ID column: "8ZNQ_1" → "8ZNQ"
    # Handle various ID formats
    if 'ID' in df.columns:
        id_col = 'ID'
    elif 'id' in df.columns:
        id_col = 'id'
    else:
        id_col = df.columns[0]

    df['_target_id'] = df[id_col].astype(str).apply(
        lambda x: '_'.join(x.split('_')[:-1]) if '_' in x else x
    )

    # Identify coordinate columns (use prediction slot 1 = best template)
    coord_cols = ['x_1', 'y_1', 'z_1']
    for col in coord_cols:
        if col not in df.columns:
            print(f"Warning: Column {col} not found in template CSV")
            return {}

    templates = {}

    for target_id, group in df.groupby('_target_id'):
        group = group.sort_values('resid' if 'resid' in group.columns else id_col)

        # Extract coordinates from prediction slot 1 (best template)
        x = group['x_1'].values.astype(np.float32)
        y = group['y_1'].values.astype(np.float32)
        z = group['z_1'].values.astype(np.float32)
        coords = np.stack([x, y, z], axis=-1)  # (N, 3)

        seq_len = coords.shape[0]

        # Check for all-zero coordinates (no template hit)
        is_all_zero = np.all(np.abs(coords) < 1e-6)

        # Build confidence array
        search_info = search_results.get(target_id, None)

        if is_all_zero:
            # No template hit — zero confidence
            confidence = np.zeros(seq_len, dtype=np.float32)
            num_hits = 0
            best_evalue = float('inf')
        else:
            confidence = build_per_residue_confidence(
                seq_len=seq_len,
                search_info=search_info,
                gap_penalty=gap_penalty,
            )

            # If no Result.txt info but coords exist, assign default confidence
            if search_info is None and not is_all_zero:
                confidence[:] = 0.5  # Default moderate confidence

            num_hits = search_info['num_hits'] if search_info else 0
            best_evalue = search_info['best_evalue'] if search_info else float('inf')

        # Handle NaN coordinates
        nan_mask = np.isnan(coords).any(axis=1)
        if nan_mask.any():
            coords[nan_mask] = 0.0
            confidence[nan_mask] = 0.0

        templates[target_id] = {
            'coords': coords,
            'confidence': confidence,
            'num_hits': num_hits,
            'best_evalue': best_evalue,
        }

    # Summary
    targets_with_template = sum(1 for t in templates.values()
                                 if t['confidence'].max() > 0)
    print(f"Loaded templates for {len(templates)} targets "
          f"({targets_with_template} with template data, "
          f"{len(templates) - targets_with_template} with no template)")

    return templates
