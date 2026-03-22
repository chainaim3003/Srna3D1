"""
template_loader.py — Load template coordinates from Approach 1 outputs.

Two modes:
  1. LOCAL (training + local testing):
     Reads from saved Approach 1 submission.csv + Result.txt files.

  2. KAGGLE (inference during Kaggle notebook run):
     Reads from MMseqs2 output generated in the same notebook run.
     The MMseqs2 pipeline (from Fork 1/rhijudas) writes:
       - Result.txt: search hits with e-values
       - Template coordinate files (CIF or extracted CSV)

This module provides a simple interface:
    loader = TemplateLoader(mode="local", ...)
    coords, confidence, has_template = loader.get_template("9G4J", seq_len=341)
"""

import os
import csv
import math
import numpy as np
from typing import Tuple, Optional, Dict


class TemplateLoader:
    """Load and serve template coordinates for each target.

    Attributes:
        templates: Dict mapping target_id -> {
            'coords': np.ndarray (N, 3),
            'confidence': float (0-1),
            'evalue': float
        }
    """

    def __init__(self, mode: str = "local",
                 submission_csv: str = None,
                 result_txt: str = None,
                 kaggle_template_dir: str = None):
        """
        Args:
            mode: "local" or "kaggle"
            submission_csv: Path to Approach 1 submission.csv (local mode)
            result_txt: Path to MMseqs2 Result.txt (local mode)
            kaggle_template_dir: Directory with Kaggle MMseqs2 outputs (kaggle mode)
        """
        self.mode = mode
        self.templates: Dict[str, dict] = {}

        if mode == "local" and submission_csv:
            self._load_from_submission_csv(submission_csv, result_txt)
        elif mode == "kaggle" and kaggle_template_dir:
            self._load_from_kaggle(kaggle_template_dir)

    def _load_from_submission_csv(self, csv_path: str,
                                   result_txt: str = None):
        """Load template coordinates from an Approach 1 submission.csv.

        The submission.csv has columns:
            ID, resname, resid, x_1, y_1, z_1, ..., x_5, y_5, z_5

        We use prediction 1 (x_1, y_1, z_1) as the template coordinates.
        """
        if not os.path.exists(csv_path):
            print(f"WARNING: Template CSV not found: {csv_path}")
            return

        # Read e-values from Result.txt if available
        evalues = {}
        if result_txt and os.path.exists(result_txt):
            evalues = self._parse_result_txt(result_txt)

        # Read coordinates from submission.csv
        target_coords = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # ID format: "9G4J_123" -> target_id = "9G4J"
                full_id = row['ID']
                parts = full_id.rsplit('_', 1)
                target_id = parts[0]
                resid = int(parts[1]) if len(parts) > 1 else 0

                if target_id not in target_coords:
                    target_coords[target_id] = []

                x = float(row.get('x_1', 0.0))
                y = float(row.get('y_1', 0.0))
                z = float(row.get('z_1', 0.0))
                target_coords[target_id].append((resid, x, y, z))

        # Build template dict
        for target_id, coord_list in target_coords.items():
            # Sort by resid
            coord_list.sort(key=lambda x: x[0])
            coords = np.array([[c[1], c[2], c[3]] for c in coord_list],
                              dtype=np.float32)

            # Check if coordinates are all zeros (no real prediction)
            is_all_zeros = np.allclose(coords, 0.0, atol=1e-6)

            # Get e-value and compute confidence
            evalue = evalues.get(target_id, 1.0)
            confidence = self._evalue_to_confidence(evalue)

            if is_all_zeros:
                confidence = 0.0

            self.templates[target_id] = {
                'coords': coords,
                'confidence': confidence,
                'evalue': evalue,
                'has_template': not is_all_zeros and confidence > 0.01
            }

        print(f"Loaded templates for {len(self.templates)} targets")
        n_with = sum(1 for t in self.templates.values() if t['has_template'])
        print(f"  With real templates: {n_with}")
        print(f"  Without templates: {len(self.templates) - n_with}")

    def _parse_result_txt(self, path: str) -> Dict[str, float]:
        """Parse MMseqs2 Result.txt to get best e-value per target.

        Result.txt format (tab-separated):
            query_id  target_id  evalue  qstart  qend  tstart  tend  qseq  tseq

        Returns dict: target_id -> best (lowest) e-value
        """
        evalues = {}
        try:
            with open(path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        query_id = parts[0]
                        evalue = float(parts[2])
                        # Keep the best (lowest) e-value per query
                        if query_id not in evalues or evalue < evalues[query_id]:
                            evalues[query_id] = evalue
        except Exception as e:
            print(f"WARNING: Could not parse Result.txt: {e}")
        return evalues

    def _evalue_to_confidence(self, evalue: float) -> float:
        """Convert MMseqs2 e-value to a confidence score (0-1).

        Lower e-value = better match = higher confidence.
        Scale: -log10(evalue) / 50, clamped to [0, 1].

        Examples:
            evalue = 1e-300 -> confidence = 1.0 (perfect match)
            evalue = 1e-50  -> confidence = 1.0
            evalue = 1e-10  -> confidence = 0.2
            evalue = 1e-5   -> confidence = 0.1
            evalue = 1.0    -> confidence = 0.0
        """
        if evalue <= 0:
            return 1.0
        try:
            neg_log = -math.log10(evalue)
        except (ValueError, OverflowError):
            return 1.0
        confidence = min(1.0, max(0.0, neg_log / 50.0))
        return confidence

    def _load_from_kaggle(self, template_dir: str):
        """Load templates from MMseqs2 output during a Kaggle notebook run.

        This is called when the MMseqs2 pipeline has just run as part of
        the hybrid notebook. The template_dir contains the outputs from
        the search + coordinate extraction steps.

        Implementation depends on how Fork 1's pipeline saves its outputs.
        This is a placeholder that will be filled in during Phase 4
        (Kaggle notebook integration).
        """
        # TODO: Implement Kaggle-mode loading
        # This will read from the same format that Fork 1's notebook produces
        print(f"Kaggle mode: looking for templates in {template_dir}")
        print("WARNING: Kaggle mode not yet implemented - using empty templates")

    def get_template(self, target_id: str,
                     seq_len: int) -> Tuple[np.ndarray, float, bool]:
        """Get template data for a specific target.

        Args:
            target_id: e.g., "9G4J"
            seq_len: Expected sequence length (for padding/trimming)

        Returns:
            coords: np.ndarray (seq_len, 3) — template C1' coordinates
            confidence: float (0-1) — template quality score
            has_template: bool — whether a real template exists
        """
        if target_id not in self.templates:
            # No template found — return zeros
            return np.zeros((seq_len, 3), dtype=np.float32), 0.0, False

        tmpl = self.templates[target_id]

        if not tmpl['has_template']:
            return np.zeros((seq_len, 3), dtype=np.float32), 0.0, False

        coords = tmpl['coords']

        # Handle length mismatch between template and sequence
        if len(coords) == seq_len:
            # Perfect match
            return coords, tmpl['confidence'], True
        elif len(coords) > seq_len:
            # Template has more residues — truncate
            return coords[:seq_len], tmpl['confidence'], True
        else:
            # Template has fewer residues — pad with zeros
            padded = np.zeros((seq_len, 3), dtype=np.float32)
            padded[:len(coords)] = coords
            # Reduce confidence proportionally to coverage
            coverage = len(coords) / seq_len
            return padded, tmpl['confidence'] * coverage, True
