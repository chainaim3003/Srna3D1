"""
pdb_parser.py — Extract RNA C1' coordinates from CIF/PDB files using BioPython.

BioPython's Bio.PDB module is the standard tool for parsing structural biology files.
Documentation: https://biopython.org/wiki/The_Biopython_Structural_Bioinformatics_FAQ

CIF (Crystallographic Information File) is the modern replacement for PDB format.
The competition provides structures in CIF format in the PDB_RNA/ directory.

We extract:
  - RNA sequence (A, U, G, C residues only)
  - C1' atom coordinates for each nucleotide
"""

import os
import numpy as np
from typing import List, Dict, Optional

# Standard RNA residue names in PDB/CIF files
# These are the 3-letter codes used in structural files
RNA_RESIDUES_3LETTER = {'A', 'U', 'G', 'C', 'ADE', 'URA', 'GUA', 'CYT',
                         'DA', 'DU', 'DG', 'DC',  # Some files use D-prefix
                         'RA', 'RU', 'RG', 'RC'}  # Some use R-prefix

# Map 3-letter codes to 1-letter
RESNAME_TO_BASE = {
    'A': 'A', 'ADE': 'A', 'DA': 'A', 'RA': 'A',
    'U': 'U', 'URA': 'U', 'DU': 'U', 'RU': 'U',
    'G': 'G', 'GUA': 'G', 'DG': 'G', 'RG': 'G',
    'C': 'C', 'CYT': 'C', 'DC': 'C', 'RC': 'C',
}


def extract_c1prime_from_structure(filepath: str, min_length: int = 10
                                    ) -> List[Dict]:
    """Extract RNA chains with C1' coordinates from a CIF or PDB file.

    Args:
        filepath: Path to .cif or .pdb file.
        min_length: Minimum chain length to include.

    Returns:
        List of dicts with keys:
            - 'id': str — identifier (filename_chainID)
            - 'sequence': str — RNA sequence (e.g., "AUGCUUAGCG")
            - 'coords': np.ndarray of shape (N, 3) — C1' coordinates in Angstroms
    """
    try:
        from Bio.PDB import PDBParser, MMCIFParser
    except ImportError:
        raise ImportError(
            "BioPython is required for CIF/PDB parsing. "
            "Install with: pip install biopython"
        )

    # Choose parser based on file extension
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.cif':
        parser = MMCIFParser(QUIET=True)
    elif ext in ('.pdb', '.ent'):
        parser = PDBParser(QUIET=True)
    else:
        print(f"Skipping unsupported file format: {filepath}")
        return []

    filename = os.path.basename(filepath)

    try:
        structure = parser.get_structure(filename, filepath)
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return []

    results = []

    for model in structure:
        for chain in model:
            sequence = []
            coords = []

            for residue in chain:
                # Skip water and hetero atoms
                hetflag = residue.get_id()[0]
                if hetflag.strip() and hetflag != ' ':
                    # Check if it's a modified nucleotide (HETATM)
                    # Some RNA residues are marked as HETATM
                    pass  # Still try to process it

                resname = residue.get_resname().strip()

                # Check if this is an RNA residue
                base = RESNAME_TO_BASE.get(resname)
                if base is None:
                    continue

                # Look for C1' atom
                if "C1'" not in residue:
                    # Some files use C1* instead of C1'
                    if "C1*" in residue:
                        atom = residue["C1*"]
                    else:
                        continue
                else:
                    atom = residue["C1'"]

                coord = atom.get_vector().get_array()
                sequence.append(base)
                coords.append(coord)

            # Only keep chains with enough nucleotides
            if len(sequence) >= min_length:
                chain_id = chain.get_id()
                results.append({
                    'id': f"{filename}_{chain_id}",
                    'sequence': ''.join(sequence),
                    'coords': np.array(coords, dtype=np.float32),
                })

    return results


def extract_rna_structures_from_directory(
    directory: str,
    min_length: int = 10,
    max_files: Optional[int] = None,
) -> List[Dict]:
    """Extract RNA structures from all CIF/PDB files in a directory.

    Args:
        directory: Path to directory containing .cif or .pdb files.
        min_length: Minimum chain length to include.
        max_files: Maximum number of files to process (None = all).

    Returns:
        List of dicts with 'id', 'sequence', 'coords' keys.
    """
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Not a directory: {directory}")

    # Find all structure files
    extensions = {'.cif', '.pdb', '.ent'}
    files = [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if os.path.splitext(f)[1].lower() in extensions
    ]

    if max_files is not None:
        files = files[:max_files]

    print(f"Found {len(files)} structure files in {directory}")

    all_structures = []
    for i, filepath in enumerate(files):
        if (i + 1) % 50 == 0:
            print(f"  Processing file {i + 1}/{len(files)}...")

        structures = extract_c1prime_from_structure(filepath, min_length)
        all_structures.extend(structures)

    print(f"Extracted {len(all_structures)} RNA chains (min length {min_length})")
    return all_structures
