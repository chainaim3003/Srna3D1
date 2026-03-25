# ============================================================
# HY-BAS-ADV1 RUN 6 — PHASE A: Offline RNAPro Inference
# ============================================================
#
# WHAT THIS IS:
#   A standalone script that runs on a CLOUD GPU (A100/H100)
#   to generate RNAPro 3D structure predictions for Part 2
#   test sequences. The outputs are saved as a compact .npz
#   file that Phase B (Kaggle notebook) consumes as templates.
#
# THIS SCRIPT DOES NOT RUN ON KAGGLE.
#   It runs on: Vast.ai, RunPod, Lambda Labs, Colab Pro,
#   or any machine with 24+ GB VRAM and CUDA.
#
# PREREQUISITES:
#   1. RNAPro installed:
#        git clone https://github.com/NVIDIA-Digital-Bio/RNAPro
#        cd RNAPro && pip install -e .
#   2. Model checkpoint downloaded:
#        From HuggingFace: nvidia/RNAPro-Public-Best-500M
#        OR from NGC: nvidia/clara/rnapro
#   3. Supporting data prepared:
#        release_data/protenix_models/protenix_base_default_v0.5.0.pt
#        release_data/ribonanzanet2_checkpoint/
#        release_data/ccd_cache/
#   4. test_sequences.csv from Kaggle Part 2 competition
#
# OUTPUTS:
#   rnapro_part2_coords.npz   — C1' coords per (target, seed)
#   rnapro_part2_all_atom.npz — Full CIF paths for reference
#   rnapro_vram_usage.txt     — Actual peak VRAM (for GPU shopping)
#
# USAGE:
#   # Full pipeline (RNAPro installed, checkpoint ready):
#   python hy_bas_adv1_run6_PhaseA_NB.py \
#     --test-csv test_sequences.csv \
#     --checkpoint ./rnapro_public_best/model.pt \
#     --output-dir ./rnapro_output \
#     --seeds 42,123,456,789,1024
#
#   # Extract-only mode (CIF files already generated):
#   python hy_bas_adv1_run6_PhaseA_NB.py \
#     --test-csv test_sequences.csv \
#     --cif-dir ./rnapro_output \
#     --extract-only
#
#   # Measure VRAM only (run 1 sequence, report peak memory):
#   python hy_bas_adv1_run6_PhaseA_NB.py \
#     --test-csv test_sequences.csv \
#     --checkpoint ./rnapro_public_best/model.pt \
#     --measure-vram
#
# ESTIMATED RUNTIME:
#   A100 80GB:  ~1-2 hours for full test set (5 seeds)
#   A100 40GB:  ~1-2 hours (may need shorter max_len)
#   RTX 4090:   ~2-4 hours (VRAM may limit — measure first)
#
# ESTIMATED COST:
#   Vast.ai A100: ~$0.80-1.50/hr → $2-5 total
#   RunPod A100:  ~$1.00-1.50/hr → $2-5 total
# ============================================================

import os
import sys
import glob
import time
import argparse
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# SECTION 1: Configuration
# ============================================================

DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
MAX_SEQ_LEN   = 512  # RNAPro crops to 512 nt

# RNAPro inference command template
# This gets filled in by run_rnapro_inference()
RNAPRO_CMD_TEMPLATE = """
python -m rnapro.runner.inference \
  --model_name rnapro_base \
  --load_checkpoint_path {checkpoint} \
  --input_csv {input_csv} \
  --dump_dir {dump_dir} \
  --seeds {seeds} \
  --dtype bf16 \
  {msa_flag} \
  {template_flag}
""".strip()


# ============================================================
# SECTION 2: VRAM Measurement
# ============================================================

def measure_vram_usage():
    """
    Report current GPU VRAM usage via nvidia-smi.
    Call before and after inference to measure peak usage.
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                used, total = line.split(',')
                print(f"  GPU {i}: {used.strip()} MB / {total.strip()} MB")
            return lines
    except Exception as e:
        print(f"  nvidia-smi failed: {e}")
    return None


def measure_peak_vram(checkpoint, test_csv, rnapro_dir=None):
    """
    Run RNAPro on a SINGLE short sequence to measure actual VRAM usage.
    This tells you the minimum GPU you need.
    """
    import torch

    print("\n" + "="*60)
    print("VRAM MEASUREMENT MODE")
    print("="*60)

    # Read first test sequence
    df = pd.read_csv(test_csv)
    shortest = df.loc[df['sequence'].str.len().idxmin()]
    print(f"  Test sequence: {shortest['target_id']} ({len(shortest['sequence'])} nt)")

    # Create temp CSV with just this sequence
    temp_csv = '/tmp/vram_test_seq.csv'
    pd.DataFrame([shortest]).to_csv(temp_csv, index=False)

    print(f"\n  VRAM before inference:")
    measure_vram_usage()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Try to run inference
    dump_dir = '/tmp/vram_test_output'
    os.makedirs(dump_dir, exist_ok=True)

    print(f"\n  Running RNAPro on 1 sequence...")
    try:
        run_rnapro_inference(
            checkpoint=checkpoint,
            input_csv=temp_csv,
            dump_dir=dump_dir,
            seeds=[42],
            rnapro_dir=rnapro_dir,
        )
        success = True
    except Exception as e:
        print(f"  Inference failed: {e}")
        success = False

    print(f"\n  VRAM after inference:")
    measure_vram_usage()

    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        peak_gb = peak_mb / 1000
        print(f"\n  PyTorch peak VRAM: {peak_mb:.0f} MB ({peak_gb:.1f} GB)")
        print(f"\n  GPU RECOMMENDATIONS based on {peak_gb:.1f} GB peak:")
        print(f"    RTX 3060 (12 GB):  {'YES' if peak_gb < 10 else 'NO'}")
        print(f"    RTX 4070 (12 GB):  {'YES' if peak_gb < 10 else 'NO'}")
        print(f"    RTX 4080 (16 GB):  {'YES' if peak_gb < 14 else 'NO'}")
        print(f"    RTX 4090 (24 GB):  {'YES' if peak_gb < 22 else 'NO'}")
        print(f"    RTX 5070 (12 GB):  {'YES' if peak_gb < 10 else 'NO'}")
        print(f"    RTX 5080 (16 GB):  {'YES' if peak_gb < 14 else 'NO'}")
        print(f"    RTX 5090 (32 GB):  {'YES' if peak_gb < 30 else 'NO'}")
        print(f"    A100    (40 GB):   {'YES' if peak_gb < 38 else 'NO'}")
        print(f"    A100    (80 GB):   {'YES' if peak_gb < 78 else 'NO'}")

        # Save to file
        with open('rnapro_vram_usage.txt', 'w') as f:
            f.write(f"Peak VRAM: {peak_mb:.0f} MB ({peak_gb:.1f} GB)\n")
            f.write(f"Test sequence length: {len(shortest['sequence'])} nt\n")
            f.write(f"Success: {success}\n")
        print(f"\n  Saved to rnapro_vram_usage.txt")

    return success


# ============================================================
# SECTION 3: RNAPro Inference
# ============================================================

def run_rnapro_inference(
    checkpoint: str,
    input_csv: str,
    dump_dir: str,
    seeds: list,
    rnapro_dir: str = None,
    msa_dir: str = None,
    template_pt: str = None,
):
    """
    Run RNAPro inference via subprocess.

    This calls RNAPro's own inference runner, which handles:
      - Loading the model checkpoint
      - Processing input sequences
      - Running diffusion sampling with multiple seeds
      - Saving CIF output files

    Args:
        checkpoint:  Path to RNAPro .pt checkpoint
        input_csv:   Path to CSV with target_id, sequence columns
        dump_dir:    Output directory for CIF files
        seeds:       List of random seeds for diversity
        rnapro_dir:  Path to RNAPro repo root (if not in PYTHONPATH)
        msa_dir:     Path to MSA directory (optional)
        template_pt: Path to precomputed template .pt file (optional)
    """
    os.makedirs(dump_dir, exist_ok=True)

    seeds_str = ','.join(str(s) for s in seeds)

    msa_flag = ''
    if msa_dir and os.path.isdir(msa_dir):
        msa_flag = f'--use_msa --rna_msa_dir {msa_dir}'

    template_flag = ''
    if template_pt and os.path.isfile(template_pt):
        template_flag = f'--use_template ca_precomputed --template_data {template_pt}'

    cmd = RNAPRO_CMD_TEMPLATE.format(
        checkpoint=checkpoint,
        input_csv=input_csv,
        dump_dir=dump_dir,
        seeds=seeds_str,
        msa_flag=msa_flag,
        template_flag=template_flag,
    )

    print(f"\n  Running RNAPro inference...")
    print(f"  Command: {cmd}")
    print(f"  Seeds: {seeds}")
    print(f"  Output: {dump_dir}")

    env = os.environ.copy()
    if rnapro_dir:
        env['PYTHONPATH'] = rnapro_dir + ':' + env.get('PYTHONPATH', '')

    start = time.time()
    result = subprocess.run(
        cmd, shell=True, env=env,
        capture_output=False,  # let output stream to terminal
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  WARNING: RNAPro exited with code {result.returncode}")
        print(f"  Check the output above for errors.")
        print(f"  Common issues:")
        print(f"    - CUDA OOM: reduce max sequence length or use larger GPU")
        print(f"    - Missing files: check release_data/ directory structure")
        print(f"    - Import errors: ensure RNAPro is installed (pip install -e .)")
    else:
        print(f"\n  RNAPro inference complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return result.returncode


# ============================================================
# SECTION 4: CIF → C1' Coordinate Extraction
# ============================================================

def extract_c1_prime_gemmi(cif_path: str) -> np.ndarray:
    """Extract C1' atom coordinates from CIF using gemmi (fast, preferred)."""
    import gemmi
    structure = gemmi.read_structure(cif_path)
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.name == "C1'":
                        coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                        break  # one C1' per residue
    return np.array(coords, dtype=np.float32) if coords else np.zeros((0, 3), dtype=np.float32)


def extract_c1_prime_biopython(cif_path: str) -> np.ndarray:
    """Extract C1' atom coordinates from CIF using BioPython (fallback)."""
    from Bio.PDB.MMCIFParser import MMCIFParser
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('rna', cif_path)
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.get_name() == "C1'":
                        pos = atom.get_vector()
                        coords.append([float(pos[0]), float(pos[1]), float(pos[2])])
                        break
    return np.array(coords, dtype=np.float32) if coords else np.zeros((0, 3), dtype=np.float32)


def extract_c1_prime(cif_path: str) -> np.ndarray:
    """Extract C1' coordinates, trying gemmi first, then BioPython."""
    try:
        import gemmi
        return extract_c1_prime_gemmi(cif_path)
    except ImportError:
        pass
    try:
        return extract_c1_prime_biopython(cif_path)
    except ImportError:
        pass
    raise ImportError(
        "Neither gemmi nor BioPython is installed. Install one:\n"
        "  pip install gemmi\n"
        "  pip install biopython"
    )


def find_cif_files(dump_dir: str) -> dict:
    """
    Discover CIF files in RNAPro output directory.

    RNAPro output structure varies by version. We search for any .cif
    files and try to extract target_id and seed from the path/filename.

    Returns:
        {target_id: {seed_idx: cif_path, ...}, ...}
    """
    cif_files = {}

    # Search patterns (RNAPro may organize differently)
    patterns = [
        f"{dump_dir}/**/*.cif",
        f"{dump_dir}/*.cif",
    ]

    all_cifs = []
    for pattern in patterns:
        all_cifs.extend(glob.glob(pattern, recursive=True))
    all_cifs = sorted(set(all_cifs))

    print(f"  Found {len(all_cifs)} CIF files in {dump_dir}")

    for cif_path in all_cifs:
        # Try to extract target_id from filename
        basename = os.path.splitext(os.path.basename(cif_path))[0]
        parent_dir = os.path.basename(os.path.dirname(cif_path))

        # Try to detect seed from parent directory or filename
        seed_idx = 0
        for part in [parent_dir, basename]:
            for prefix in ['seed_', 'seed', 's']:
                if prefix in part.lower():
                    try:
                        seed_str = part.lower().split(prefix)[-1].split('_')[0].split('/')[0]
                        seed_idx = int(seed_str)
                        break
                    except (ValueError, IndexError):
                        pass

        # Target ID: try to match known patterns
        target_id = basename
        # Strip common prefixes/suffixes
        for strip in ['_pred', '_predicted', '_output', '_result']:
            if target_id.endswith(strip):
                target_id = target_id[:-len(strip)]

        if target_id not in cif_files:
            cif_files[target_id] = {}
        cif_files[target_id][seed_idx] = cif_path

    return cif_files


def extract_all_coordinates(
    dump_dir: str,
    test_csv: str,
    output_path: str = 'rnapro_part2_coords.npz',
) -> dict:
    """
    Extract C1' coordinates from all RNAPro CIF outputs.

    Args:
        dump_dir:    Directory containing RNAPro CIF output
        test_csv:    Path to test_sequences.csv (for target matching)
        output_path: Where to save the .npz file

    Returns:
        dict of {f"{target_id}_seed{i}": np.ndarray(N,3)}
    """
    print("\n" + "="*60)
    print("EXTRACTING C1' COORDINATES FROM CIF FILES")
    print("="*60)

    test_df = pd.read_csv(test_csv)
    target_ids = set(test_df['target_id'].values)
    target_seqs = dict(zip(test_df['target_id'], test_df['sequence']))
    print(f"  Test sequences: {len(target_ids)}")

    # Find CIF files
    cif_map = find_cif_files(dump_dir)
    print(f"  Targets with CIF files: {len(cif_map)}")

    # Match CIF targets to test targets
    # (CIF filenames may not exactly match target_ids — try fuzzy matching)
    matched = {}
    for target_id in target_ids:
        if target_id in cif_map:
            matched[target_id] = cif_map[target_id]
        else:
            # Try partial match
            for cif_target in cif_map:
                if target_id in cif_target or cif_target in target_id:
                    matched[target_id] = cif_map[cif_target]
                    break

    print(f"  Matched to test targets: {len(matched)}/{len(target_ids)}")
    if len(matched) < len(target_ids):
        unmatched = target_ids - set(matched.keys())
        print(f"  Unmatched targets (will use Run 5 fallback): {len(unmatched)}")
        for tid in sorted(unmatched)[:5]:
            print(f"    {tid}")
        if len(unmatched) > 5:
            print(f"    ... and {len(unmatched)-5} more")

    # Extract coordinates
    coords_dict = {}
    n_extracted = 0
    n_failed = 0

    for target_id, seed_cifs in matched.items():
        seq_len = len(target_seqs.get(target_id, ''))

        for seed_idx, cif_path in sorted(seed_cifs.items()):
            try:
                coords = extract_c1_prime(cif_path)

                if len(coords) == 0:
                    print(f"    WARNING: {target_id} seed {seed_idx}: no C1' atoms found")
                    n_failed += 1
                    continue

                # Validate: coords should roughly match sequence length
                if seq_len > 0 and abs(len(coords) - seq_len) > seq_len * 0.2:
                    print(f"    WARNING: {target_id} seed {seed_idx}: "
                          f"{len(coords)} coords vs {seq_len} residues "
                          f"(>20% mismatch)")

                # Truncate or pad to match sequence length
                if seq_len > 0:
                    if len(coords) > seq_len:
                        coords = coords[:seq_len]
                    elif len(coords) < seq_len:
                        pad = np.zeros((seq_len - len(coords), 3), dtype=np.float32)
                        # Extrapolate last direction for padding
                        if len(coords) >= 2:
                            direction = coords[-1] - coords[-2]
                            direction = direction / (np.linalg.norm(direction) + 1e-8)
                            for i in range(len(pad)):
                                pad[i] = coords[-1] + direction * 5.9 * (i + 1)
                        coords = np.concatenate([coords, pad])

                key = f"{target_id}_seed{seed_idx}"
                coords_dict[key] = coords.astype(np.float32)
                n_extracted += 1

            except Exception as e:
                print(f"    ERROR: {target_id} seed {seed_idx}: {e}")
                n_failed += 1

    print(f"\n  Extracted: {n_extracted} coordinate sets")
    print(f"  Failed:    {n_failed}")

    # Count targets with at least one seed
    targets_with_coords = set()
    for key in coords_dict:
        tid = key.rsplit('_seed', 1)[0]
        targets_with_coords.add(tid)
    print(f"  Targets with coords: {len(targets_with_coords)}/{len(target_ids)}")

    # Save
    np.savez_compressed(output_path, **coords_dict)
    file_size = os.path.getsize(output_path) / 1e6
    print(f"\n  Saved: {output_path} ({file_size:.1f} MB)")
    print(f"  Keys:  {len(coords_dict)}")

    # Verify by loading back
    verify = np.load(output_path)
    print(f"  Verify: {len(verify.files)} arrays loaded successfully")

    return coords_dict


# ============================================================
# SECTION 5: Generate Kaggle Upload Package
# ============================================================

def create_kaggle_dataset(
    npz_path: str,
    output_dir: str = './rnapro_kaggle_upload',
    dataset_name: str = 'rnapro-part2-templates-v1',
):
    """
    Create a directory ready for kaggle datasets create.

    Args:
        npz_path:     Path to rnapro_part2_coords.npz
        output_dir:   Directory to create the upload package in
        dataset_name: Kaggle dataset slug
    """
    import shutil
    import json

    os.makedirs(output_dir, exist_ok=True)
    shutil.copy2(npz_path, os.path.join(output_dir, os.path.basename(npz_path)))

    # Create dataset-metadata.json
    metadata = {
        "title": "RNAPro Part 2 Pre-computed Templates",
        "id": f"INSERT_YOUR_KAGGLE_USERNAME/{dataset_name}",
        "licenses": [{"name": "apache-2.0"}],
    }
    meta_path = os.path.join(output_dir, 'dataset-metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Kaggle upload package created: {output_dir}/")
    print(f"  Files:")
    for fname in os.listdir(output_dir):
        fsize = os.path.getsize(os.path.join(output_dir, fname)) / 1e6
        print(f"    {fname} ({fsize:.1f} MB)")
    print(f"\n  To upload:")
    print(f"    1. Edit {meta_path} — replace INSERT_YOUR_KAGGLE_USERNAME")
    print(f"    2. Run: kaggle datasets create -p {output_dir} --dir-mode zip")


# ============================================================
# SECTION 6: Setup Verification
# ============================================================

def verify_setup(args):
    """Check that all prerequisites are in place before running."""
    print("\n" + "="*60)
    print("SETUP VERIFICATION")
    print("="*60)

    ok = True

    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"  [OK] GPU: {name} ({vram:.1f} GB)")
        else:
            print(f"  [!!] No CUDA GPU detected")
            ok = False
    except ImportError:
        print(f"  [!!] PyTorch not installed")
        ok = False

    # Check test CSV
    if args.test_csv and os.path.isfile(args.test_csv):
        df = pd.read_csv(args.test_csv)
        print(f"  [OK] Test CSV: {len(df)} sequences")
        lens = df['sequence'].str.len()
        print(f"       Lengths: {lens.min()}-{lens.max()} nt "
              f"(mean {lens.mean():.0f}, >{MAX_SEQ_LEN}: {(lens > MAX_SEQ_LEN).sum()})")
    else:
        print(f"  [!!] Test CSV not found: {args.test_csv}")
        ok = False

    # Check checkpoint
    if hasattr(args, 'checkpoint') and args.checkpoint:
        if os.path.isfile(args.checkpoint):
            size = os.path.getsize(args.checkpoint) / 1e9
            print(f"  [OK] Checkpoint: {args.checkpoint} ({size:.1f} GB)")
        else:
            print(f"  [!!] Checkpoint not found: {args.checkpoint}")
            ok = False

    # Check CIF extraction libraries
    gemmi_ok = False
    bio_ok = False
    try:
        import gemmi
        gemmi_ok = True
        print(f"  [OK] gemmi installed")
    except ImportError:
        pass
    try:
        from Bio.PDB.MMCIFParser import MMCIFParser
        bio_ok = True
        print(f"  [OK] BioPython installed")
    except ImportError:
        pass
    if not gemmi_ok and not bio_ok:
        print(f"  [!!] Neither gemmi nor BioPython installed (needed for CIF extraction)")
        print(f"       pip install gemmi   OR   pip install biopython")
        ok = False

    # Check RNAPro
    if not args.extract_only:
        try:
            import rnapro
            print(f"  [OK] RNAPro importable")
        except ImportError:
            print(f"  [??] RNAPro not importable — may work via subprocess if in PATH")

    print(f"\n  Overall: {'READY' if ok else 'ISSUES FOUND — see above'}")
    return ok


# ============================================================
# SECTION 7: Main Pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run 6 Phase A: Offline RNAPro inference for Stanford RNA 3D Folding Part 2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline:
  python hy_bas_adv1_run6_PhaseA_NB.py \\
    --test-csv test_sequences.csv \\
    --checkpoint ./rnapro_public_best/model.pt \\
    --output-dir ./rnapro_output

  # Extract coordinates from existing CIF files:
  python hy_bas_adv1_run6_PhaseA_NB.py \\
    --test-csv test_sequences.csv \\
    --cif-dir ./rnapro_output \\
    --extract-only

  # Measure VRAM usage (run 1 sequence):
  python hy_bas_adv1_run6_PhaseA_NB.py \\
    --test-csv test_sequences.csv \\
    --checkpoint ./rnapro_public_best/model.pt \\
    --measure-vram
        """,
    )
    parser.add_argument('--test-csv', required=True,
                        help='Path to test_sequences.csv from Kaggle Part 2')
    parser.add_argument('--checkpoint', default=None,
                        help='Path to RNAPro .pt checkpoint')
    parser.add_argument('--output-dir', default='./rnapro_output',
                        help='Directory for RNAPro CIF output (default: ./rnapro_output)')
    parser.add_argument('--cif-dir', default=None,
                        help='Directory with existing CIF files (for --extract-only)')
    parser.add_argument('--seeds', default='42,123,456,789,1024',
                        help='Comma-separated random seeds (default: 42,123,456,789,1024)')
    parser.add_argument('--rnapro-dir', default=None,
                        help='Path to RNAPro repo root')
    parser.add_argument('--msa-dir', default=None,
                        help='Path to MSA directory (optional)')
    parser.add_argument('--template-pt', default=None,
                        help='Path to precomputed template .pt file (optional)')
    parser.add_argument('--npz-output', default='rnapro_part2_coords.npz',
                        help='Output .npz filename (default: rnapro_part2_coords.npz)')
    parser.add_argument('--extract-only', action='store_true',
                        help='Skip inference, only extract coords from existing CIF files')
    parser.add_argument('--measure-vram', action='store_true',
                        help='Run 1 sequence and report peak VRAM usage')
    parser.add_argument('--create-kaggle-dataset', action='store_true',
                        help='Create Kaggle upload package after extraction')

    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]

    print("="*60)
    print("HY-BAS-ADV1 RUN 6 — PHASE A: RNAPro Offline Inference")
    print("="*60)
    print(f"  Test CSV:    {args.test_csv}")
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"  Seeds:       {seeds}")
    print(f"  NPZ output:  {args.npz_output}")

    # Verify setup
    verify_setup(args)

    # Mode: Measure VRAM only
    if args.measure_vram:
        if not args.checkpoint:
            print("ERROR: --checkpoint required for --measure-vram")
            sys.exit(1)
        measure_peak_vram(args.checkpoint, args.test_csv, args.rnapro_dir)
        return

    # Mode: Extract only (CIF files already exist)
    if args.extract_only:
        cif_dir = args.cif_dir or args.output_dir
        coords_dict = extract_all_coordinates(cif_dir, args.test_csv, args.npz_output)

        if args.create_kaggle_dataset:
            create_kaggle_dataset(args.npz_output)
        return

    # Mode: Full pipeline (inference + extraction)
    if not args.checkpoint:
        print("ERROR: --checkpoint required for inference mode")
        print("  Use --extract-only if CIF files already exist")
        sys.exit(1)

    # Step 1: Run RNAPro inference
    print("\n" + "="*60)
    print("STEP 1: RNAPro INFERENCE")
    print("="*60)

    measure_vram_usage()

    retcode = run_rnapro_inference(
        checkpoint=args.checkpoint,
        input_csv=args.test_csv,
        dump_dir=args.output_dir,
        seeds=seeds,
        rnapro_dir=args.rnapro_dir,
        msa_dir=args.msa_dir,
        template_pt=args.template_pt,
    )

    if retcode != 0:
        print(f"\n  RNAPro inference failed (exit code {retcode})")
        print(f"  You can re-run extraction later with --extract-only --cif-dir {args.output_dir}")
        sys.exit(1)

    # Step 2: Extract C1' coordinates
    print("\n" + "="*60)
    print("STEP 2: EXTRACT C1' COORDINATES")
    print("="*60)

    coords_dict = extract_all_coordinates(args.output_dir, args.test_csv, args.npz_output)

    # Step 3: Create Kaggle upload package
    if args.create_kaggle_dataset:
        print("\n" + "="*60)
        print("STEP 3: CREATE KAGGLE DATASET PACKAGE")
        print("="*60)
        create_kaggle_dataset(args.npz_output)

    # Summary
    print("\n" + "="*60)
    print("PHASE A COMPLETE")
    print("="*60)
    print(f"  NPZ file: {args.npz_output}")
    print(f"  Targets:  {len(set(k.rsplit('_seed',1)[0] for k in coords_dict))}")
    print(f"  Total coordinate sets: {len(coords_dict)}")
    print(f"\n  NEXT STEPS:")
    print(f"    1. Upload {args.npz_output} to Kaggle as a dataset")
    print(f"    2. Run Phase B notebook (hy_bas_adv1_run6_PhaseB_NB.py) on Kaggle")
    print(f"    3. Phase B loads this file and uses RNAPro coords as IPA templates")


if __name__ == '__main__':
    main()
