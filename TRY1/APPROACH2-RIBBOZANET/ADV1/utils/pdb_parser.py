"""
pdb_parser.py — Extract RNA C1' coordinates from CIF/PDB files using BioPython.
IDENTICAL to BASIC/utils/pdb_parser.py — no changes needed for ADV1.
"""

import os
import numpy as np
from typing import List, Dict, Optional

RNA_RESIDUES_3LETTER = {'A', 'U', 'G', 'C', 'ADE', 'URA', 'GUA', 'CYT',
                         'DA', 'DU', 'DG', 'DC', 'RA', 'RU', 'RG', 'RC'}

RESNAME_TO_BASE = {
    'A': 'A', 'ADE': 'A', 'DA': 'A', 'RA': 'A',
    'U': 'U', 'URA': 'U', 'DU': 'U', 'RU': 'U',
    'G': 'G', 'GUA': 'G', 'DG': 'G', 'RG': 'G',
    'C': 'C', 'CYT': 'C', 'DC': 'C', 'RC': 'C',
}


def extract_c1prime_from_structure(filepath, min_length=10):
    try:
        from Bio.PDB import PDBParser, MMCIFParser
    except ImportError:
        raise ImportError("BioPython required. Install with: pip install biopython")

    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.cif':
        parser = MMCIFParser(QUIET=True)
    elif ext in ('.pdb', '.ent'):
        parser = PDBParser(QUIET=True)
    else:
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
                resname = residue.get_resname().strip()
                base = RESNAME_TO_BASE.get(resname)
                if base is None:
                    continue
                if "C1'" not in residue:
                    if "C1*" in residue:
                        atom = residue["C1*"]
                    else:
                        continue
                else:
                    atom = residue["C1'"]
                coord = atom.get_vector().get_array()
                sequence.append(base)
                coords.append(coord)
            if len(sequence) >= min_length:
                chain_id = chain.get_id()
                results.append({
                    'id': f"{filename}_{chain_id}",
                    'sequence': ''.join(sequence),
                    'coords': np.array(coords, dtype=np.float32),
                })
    return results


def extract_rna_structures_from_directory(directory, min_length=10, max_files=None):
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Not a directory: {directory}")
    extensions = {'.cif', '.pdb', '.ent'}
    files = [os.path.join(directory, f) for f in sorted(os.listdir(directory))
             if os.path.splitext(f)[1].lower() in extensions]
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
