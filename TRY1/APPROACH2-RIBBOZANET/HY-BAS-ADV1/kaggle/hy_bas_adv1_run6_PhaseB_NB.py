# ============================================================
# HY-BAS-ADV1 RUN 6 — PHASE B: IPA + RNAPro Templates (Kaggle)
# ============================================================
#
# WHAT THIS IS:
#   Run 6 = Run 5 (IPA) + pre-computed RNAPro 3D predictions
#   used as high-quality template init_coords.
#
# SIMPLE EXPLANATION (for a high schooler):
#   Run 5 said: "start from zeros or a rough template, then
#   refine using IPA attention." The problem: starting from
#   zeros means IPA has to do ALL the work from scratch.
#
#   Run 6 says: "start from RNAPro's 3D prediction (which is
#   already pretty good), then refine using IPA." IPA only
#   needs to make small corrections — much easier.
#
# WHAT CHANGES vs RUN 5:
#   Cell 0:   ADD RNAPRO_CONFIDENCE config param
#   Cell 3:   ADD search for rnapro_part2_coords.npz
#   Cell 9:   ADD load RNAPro pre-computed templates for test targets
#             RNAPro templates OVERRIDE template search when available
#   Cell 14:  NEW slot assembly when RNAPro available:
#             Slot 1: raw RNAPro seed 0 (no IPA)
#             Slot 2: RNAPro seed 0 → IPA refinement
#             Slot 3: RNAPro seed 1 → IPA refinement
#             Slot 4: RNAPro seed 2 (raw, diversity)
#             Slot 5: IPA cold start (maximum diversity)
#             Falls back to Run 5 behavior for unknown targets
#
# WHAT IS UNCHANGED vs RUN 5:
#   Cells 1,2,4,5,6,7,8,9.5,10,11,12,13,15: entirely unchanged
#   IPA module architecture: unchanged
#   FAPE loss: unchanged
#   Training: unchanged (trains on same data, same way)
#   TemplateEncoder, backbone, MSA features: unchanged
#
# PHASE A PREREQUISITE:
#   Run hy_bas_adv1_run6_PhaseA_NB.py on cloud GPU first.
#   Upload rnapro_part2_coords.npz to Kaggle as a dataset.
#   Attach that dataset to this notebook.
#
# IF PHASE A WAS NOT RUN:
#   This notebook degrades gracefully to Run 5 behavior.
#   All RNAPro-specific code is gated behind:
#     if RNAPRO_COORDS_FILE is not None:
#   If the .npz file is not found, everything works as Run 5.
#
# ESTIMATED RUNTIME: ~2.5-4 hours on T4 GPU (same as Run 5)
# ============================================================


# ============================================================
# CELL 0: USER-CONFIGURABLE PATHS AND IPA CONFIG
# ============================================================
# CHANGE from Run 5: added RNAPRO_CONFIDENCE
# ============================================================
EXTENDED_SEQ_CSV   = None
EXTENDED_COORD_CSV = None
OUTPUT_DIR         = '/kaggle/working'
COMP_BASE_PRIMARY  = '/kaggle/input/competitions/stanford-rna-3d-folding-2/'
COMP_BASE_FALLBACK = '/kaggle/input/stanford-rna-3d-folding-2/'

# MSA configuration (unchanged from Run 5)
MSA_TOP_N = 20
MSA_DIM   = 8

# IPA configuration (unchanged from Run 5)
IPA_DIM          = 256
IPA_ITERATIONS   = 4
IPA_HEADS        = 4
FAPE_CLAMP       = 10.0
AUX_DIST_WEIGHT  = 0.1

# Training strategy (unchanged from Run 5)
FREEZE_BACKBONE_IPA = True
TRAIN_EPOCHS        = 40
EARLY_STOP_PATIENCE = 10

# NEW in Run 6: RNAPro template integration
RNAPRO_CONFIDENCE = 0.90   # Treat RNAPro predictions as 90% confident
                            # Higher = more trust in RNAPro, less IPA correction
                            # Lower = IPA will adjust more aggressively


# ============================================================
# CELL 1: Install biopython (direct sys.path injection)
# ============================================================
# UNCHANGED from Run 5.
# ============================================================
import sys, os, glob

py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"
print(f"Python {sys.version_info.major}.{sys.version_info.minor} (tag: {py_ver})")

try:
    import Bio
    print(f"Biopython {Bio.__version__} already installed")
except ImportError:
    bio_inits = glob.glob('/kaggle/input/**/Bio/__init__.py', recursive=True)
    print(f"Found Bio dirs: {bio_inits}")
    matching = [p for p in bio_inits if py_ver in p]
    chosen   = matching[0] if matching else (bio_inits[0] if bio_inits else None)
    if chosen:
        bio_parent = os.path.dirname(os.path.dirname(chosen))
        print(f"Adding to sys.path: {bio_parent}")
        sys.path.insert(0, bio_parent)
    else:
        print("Bio/ not found, trying pip...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'biopython', '-q'])
    import Bio
    print(f"Biopython {Bio.__version__} available")


# ============================================================
# CELL 2: Imports and setup
# ============================================================
# UNCHANGED from Run 5.
# ============================================================
import os, sys, time, csv, math, random, pickle, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.linalg import eigh
from scipy.spatial import distance_matrix as scipy_dist_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from einops import rearrange, repeat

from Bio import pairwise2
from Bio.Seq import Seq
from scipy.spatial.transform import Rotation as R

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ============================================================
# CELL 3: Find all data paths (auto-discovery)
# ============================================================
# CHANGE from Run 5: added RNAPro coords discovery
# ============================================================
BASE = COMP_BASE_PRIMARY
if not os.path.exists(BASE):
    BASE = COMP_BASE_FALLBACK
print(f"Competition data: {BASE}")

REPO_PATH = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "Network.py" in files:
        REPO_PATH = root
        break
print(f"RibonanzaNet repo: {REPO_PATH}")

BACKBONE_WEIGHTS = None
ADV1_WEIGHTS     = None
RUN4_CHECKPOINT  = None
RUN5_CHECKPOINT  = None  # NEW: prefer Run 5 checkpoint if available

for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f == "RibonanzaNet.pt":
            BACKBONE_WEIGHTS = os.path.join(root, f)
        if f == "best_model.pt":
            ADV1_WEIGHTS = os.path.join(root, f)
        if f.strip() == "adv1_best_model.pt":
            RUN4_CHECKPOINT = os.path.join(root, f)
        if f.strip() == "adv1_run5_best_model.pt":
            RUN5_CHECKPOINT = os.path.join(root, f)

print(f"Backbone weights: {BACKBONE_WEIGHTS}")
print(f"BASIC weights:    {ADV1_WEIGHTS}")
print(f"Run 4 checkpoint: {RUN4_CHECKPOINT}")
print(f"Run 5 checkpoint: {RUN5_CHECKPOINT}")

# --- NEW in Run 6: Find RNAPro pre-computed coordinates ---
RNAPRO_COORDS_FILE = None
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f == "rnapro_part2_coords.npz":
            RNAPRO_COORDS_FILE = os.path.join(root, f)
            break
    if RNAPRO_COORDS_FILE:
        break

if RNAPRO_COORDS_FILE:
    print(f"RNAPro coords:    {RNAPRO_COORDS_FILE}  *** RUN 6 ACTIVE ***")
else:
    print(f"RNAPro coords:    NOT FOUND (will fall back to Run 5 behavior)")

TRAIN_PICKLE = None
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f == "pdb_xyz_data.pkl":
            TRAIN_PICKLE = os.path.join(root, f)
print(f"Training pickle: {TRAIN_PICKLE}")

SAMPLE_CSV = None
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f == "sample_submission.csv":
            SAMPLE_CSV = os.path.join(root, f)
print(f"Sample submission: {SAMPLE_CSV}")

if EXTENDED_SEQ_CSV is None or EXTENDED_COORD_CSV is None:
    for root, dirs, files in os.walk("/kaggle/input"):
        for f in files:
            if f == "rna_sequences.csv" and EXTENDED_SEQ_CSV is None:
                EXTENDED_SEQ_CSV = os.path.join(root, f)
            if f == "rna_coordinates.csv" and EXTENDED_COORD_CSV is None:
                EXTENDED_COORD_CSV = os.path.join(root, f)
print(f"Extended seqs CSV:   {EXTENDED_SEQ_CSV}")
print(f"Extended coords CSV: {EXTENDED_COORD_CSV}")

CHECKPOINT_PATH       = f'{OUTPUT_DIR}/adv1_run6_best_model.pt'
RAW_SUBMISSION_PATH   = f'{OUTPUT_DIR}/submission_raw.csv'
FINAL_SUBMISSION_PATH = f'{OUTPUT_DIR}/submission.csv'


# ============================================================
# CELLS 4-8: Load data, extend, process, template functions
# ============================================================
# ALL UNCHANGED from Run 5. Copied verbatim.
# See Run 5 source for full code. Below is a compact reference.
# ============================================================

# --- Cell 4: Load competition data ---
print("\nLoading competition data...")
train_seqs   = pd.read_csv(BASE + 'train_sequences.csv')
test_seqs    = pd.read_csv(BASE + 'test_sequences.csv')
train_labels = pd.read_csv(BASE + 'train_labels.csv')
print(f"Train: {len(train_seqs)}, Test: {len(test_seqs)}")

# --- Cell 5: Load extended data ---
if EXTENDED_SEQ_CSV is None or EXTENDED_COORD_CSV is None:
    raise FileNotFoundError(
        "Could not find rna_sequences.csv or rna_coordinates.csv. "
        "Attach the rna-cif-to-csv dataset or set paths in Cell 0."
    )
print("Loading extended data...")
train_seqs_v2   = pd.read_csv(EXTENDED_SEQ_CSV)
train_labels_v2 = pd.read_csv(EXTENDED_COORD_CSV)
print(f"Extended: {len(train_seqs_v2)} seqs, {len(train_labels_v2)} labels")

# --- Cell 6: Extend datasets ---
def extend_dataset(original_df, v2_df, key_col):
    orig_keys   = set(original_df[key_col])
    new_mask    = ~v2_df[key_col].isin(orig_keys)
    new_records = v2_df[new_mask].copy()
    extended    = pd.concat([original_df, new_records], ignore_index=True)
    print(f"  Original: {len(original_df)} -> Extended: {len(extended)} (+{len(new_records)})")
    return extended

print("Extending sequences...")
train_seqs_extended = extend_dataset(train_seqs, train_seqs_v2, 'target_id')

print("Extending labels...")
train_labels['_key']    = train_labels['ID']    + '_' + train_labels['resid'].astype(str)
train_labels_v2['_key'] = train_labels_v2['ID'] + '_' + train_labels_v2['resid'].astype(str)
orig_keys  = set(train_labels['_key'])
new_mask   = ~train_labels_v2['_key'].isin(orig_keys)
new_labels = train_labels_v2[new_mask].copy()
train_labels_extended = pd.concat([train_labels, new_labels], ignore_index=True)
for df in [train_labels_extended, train_labels, train_labels_v2]:
    df.drop('_key', axis=1, inplace=True, errors='ignore')
print(f"  Labels: {len(train_labels)} -> {len(train_labels_extended)}")

# --- Cell 7: Process labels into coordinate dictionary ---
print("\nProcessing labels into coordinate dict...")
train_coords_dict = {}
id_groups = train_labels_extended.groupby(
    lambda x: train_labels_extended['ID'][x].rsplit('_', 1)[0]
)
for id_prefix, group in tqdm(id_groups, desc="Processing structures"):
    coords = []
    for _, row in group.sort_values('resid').iterrows():
        coords.append([row['x_1'], row['y_1'], row['z_1']])
    train_coords_dict[id_prefix] = np.array(coords)
print(f"Loaded {len(train_coords_dict)} structures")

# --- Cell 8: Fork 2 template functions (ALL UNCHANGED) ---
def adaptive_rna_constraints(coordinates, sequence, confidence=1.0):
    refined_coords      = coordinates.copy()
    n_residues          = len(sequence)
    constraint_strength = 0.8 * (1.0 - min(confidence, 0.8))
    seq_min_dist, seq_max_dist = 5.5, 6.5
    for i in range(n_residues - 1):
        current_dist = np.linalg.norm(refined_coords[i+1] - refined_coords[i])
        if current_dist < seq_min_dist or current_dist > seq_max_dist:
            target_dist = (seq_min_dist + seq_max_dist) / 2
            direction   = refined_coords[i+1] - refined_coords[i]
            direction   = direction / (np.linalg.norm(direction) + 1e-10)
            adjustment  = (target_dist - current_dist) * constraint_strength
            refined_coords[i+1] = refined_coords[i] + direction * (current_dist + adjustment)
    dist_mat    = scipy_dist_matrix(refined_coords, refined_coords)
    min_allowed = 3.8
    clashes     = np.where((dist_mat < min_allowed) & (dist_mat > 0))
    for idx in range(len(clashes[0])):
        i, j = clashes[0][idx], clashes[1][idx]
        if abs(i - j) <= 1 or i >= j:
            continue
        direction = refined_coords[j] - refined_coords[i]
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        adj = (min_allowed - dist_mat[i, j]) * constraint_strength
        refined_coords[i] -= direction * (adj / 2)
        refined_coords[j] += direction * (adj / 2)
    return refined_coords

def adapt_template_to_query(query_seq, template_seq, template_coords):
    query_seq_obj    = Seq(query_seq)
    template_seq_obj = Seq(template_seq)
    alignments = pairwise2.align.globalms(
        query_seq_obj, template_seq_obj,
        2.9, -1, -10, -0.5, one_alignment_only=True
    )
    if not alignments:
        return generate_rna_structure(query_seq)
    alignment        = alignments[0]
    aligned_query    = alignment.seqA
    aligned_template = alignment.seqB
    query_coords     = np.full((len(query_seq), 3), np.nan)
    qi, ti = 0, 0
    for i in range(len(aligned_query)):
        qc, tc = aligned_query[i], aligned_template[i]
        if qc != '-' and tc != '-':
            if ti < len(template_coords):
                query_coords[qi] = template_coords[ti]
            ti += 1; qi += 1
        elif qc != '-':
            qi += 1
        elif tc != '-':
            ti += 1
    backbone_distance = 5.9
    for i in range(len(query_coords)):
        if np.isnan(query_coords[i, 0]):
            prev_valid = next_valid = None
            for j in range(i-1, -1, -1):
                if not np.isnan(query_coords[j, 0]):
                    prev_valid = j; break
            for j in range(i+1, len(query_coords)):
                if not np.isnan(query_coords[j, 0]):
                    next_valid = j; break
            if prev_valid is not None and next_valid is not None:
                gap = next_valid - prev_valid
                for k, idx2 in enumerate(range(prev_valid+1, next_valid)):
                    w = (k+1) / gap
                    query_coords[idx2] = (1-w)*query_coords[prev_valid] + w*query_coords[next_valid]
            elif prev_valid is not None:
                d = (np.array([1,0,0]) if prev_valid == 0
                     else query_coords[prev_valid] - query_coords[prev_valid-1])
                d = d / (np.linalg.norm(d) + 1e-10)
                query_coords[i] = query_coords[prev_valid] + d * backbone_distance * (i - prev_valid)
            elif next_valid is not None:
                query_coords[i] = query_coords[next_valid] - \
                    np.array([backbone_distance * (next_valid - i), 0, 0])
    return np.nan_to_num(query_coords)

def generate_rna_structure(sequence, seed=None):
    if seed is not None:
        np.random.seed(seed); random.seed(seed)
    n      = len(sequence)
    coords = np.zeros((n, 3))
    for i in range(min(3, n)):
        angle     = i * 0.6
        coords[i] = [10*np.cos(angle), 10*np.sin(angle), i*2.5]
    direction = np.array([0, 0, 1.0])
    for i in range(3, n):
        if random.random() < 0.3:
            axis      = np.random.normal(0, 1, 3)
            axis      = axis / (np.linalg.norm(axis) + 1e-10)
            rot       = R.from_rotvec(random.uniform(0.2, 0.6) * axis)
            direction = rot.apply(direction)
        else:
            direction += np.random.normal(0, 0.15, 3)
            direction  = direction / (np.linalg.norm(direction) + 1e-10)
        step      = random.uniform(3.5, 4.5)
        coords[i] = coords[i-1] + step * direction
    return coords

def find_similar_sequences(query_seq, train_seqs_df, train_coords_dict, top_n=5):
    query_seq_obj = Seq(query_seq)
    candidates    = []
    k             = 3
    q_kmers       = set(query_seq[i:i+k] for i in range(len(query_seq)-k+1))
    for _, row in train_seqs_df.iterrows():
        tid  = row['target_id']
        tseq = row['sequence']
        if tid not in train_coords_dict:
            continue
        lr = abs(len(tseq) - len(query_seq)) / max(len(tseq), len(query_seq))
        if lr > 0.4:
            continue
        t_kmers = set(tseq[i:i+k] for i in range(len(tseq)-k+1))
        score   = len(q_kmers & t_kmers) / len(q_kmers | t_kmers) if q_kmers | t_kmers else 0
        candidates.append((tid, tseq, score, train_coords_dict[tid]))
    candidates.sort(key=lambda x: x[2], reverse=True)
    candidates = candidates[:100]
    similar = []
    for tid, tseq, _, coords in candidates:
        alns = pairwise2.align.globalms(
            query_seq_obj, tseq, 2.9, -1, -10, -0.5, one_alignment_only=True
        )
        if alns:
            s = alns[0].score / (2 * min(len(query_seq), len(tseq)))
            if s > 0:
                similar.append((tid, tseq, s, coords))
    similar.sort(key=lambda x: x[2], reverse=True)
    return similar[:top_n]


# ============================================================
# CELL 9: Template search + collect MSA + LOAD RNAPRO
# ============================================================
# CHANGE from Run 5: Load RNAPro pre-computed coordinates
#   and use them as PRIMARY templates for matching targets.
#   Falls back to template search for unmatched targets.
# ============================================================
print("\n" + "="*60)
print("PHASE 1: Template Search + MSA + RNAPro Template Loading")
print("="*60)

# --- NEW in Run 6: Load RNAPro pre-computed templates ---
rnapro_coords = {}  # {target_id: {seed_idx: np.ndarray(N,3)}}
n_rnapro_targets = 0

if RNAPRO_COORDS_FILE:
    print(f"\nLoading RNAPro coordinates from {RNAPRO_COORDS_FILE}...")
    data = np.load(RNAPRO_COORDS_FILE)
    for key in data.files:
        # Key format: "R1234_seed0", "R1234_seed1", etc.
        parts = key.rsplit('_seed', 1)
        if len(parts) == 2:
            target_id = parts[0]
            seed_idx = int(parts[1])
        else:
            target_id = key
            seed_idx = 0
        if target_id not in rnapro_coords:
            rnapro_coords[target_id] = {}
        rnapro_coords[target_id][seed_idx] = data[key]
    n_rnapro_targets = len(rnapro_coords)
    n_rnapro_total = len(data.files)
    print(f"  Loaded {n_rnapro_total} coordinate sets for {n_rnapro_targets} targets")
else:
    print("\n  No RNAPro coordinates found. Using Run 5 template search only.")

# --- Template search (with RNAPro override) ---
template_coords_per_target     = {}
template_confidence_per_target = {}
msa_hits_per_target            = {}
rnapro_used_per_target         = {}  # NEW: track which targets use RNAPro

start_time = time.time()
n_rnapro_used = 0
n_search_used = 0

for idx, row in test_seqs.iterrows():
    target_id = row['target_id']
    sequence  = row['sequence']
    if idx % 5 == 0:
        elapsed = time.time() - start_time
        print(f"  {idx+1}/{len(test_seqs)}: {target_id} ({len(sequence)} nt) [{elapsed:.0f}s]")

    # Check if RNAPro has predictions for this target
    if target_id in rnapro_coords and 0 in rnapro_coords[target_id]:
        # USE RNAPRO TEMPLATE (highest quality)
        rnapro_seed0 = rnapro_coords[target_id][0]
        N = len(sequence)

        # Length match: truncate or pad to sequence length
        if len(rnapro_seed0) > N:
            rnapro_seed0 = rnapro_seed0[:N]
        elif len(rnapro_seed0) < N:
            pad = np.zeros((N, 3), dtype=np.float32)
            pad[:len(rnapro_seed0)] = rnapro_seed0
            if len(rnapro_seed0) >= 2:
                direction = rnapro_seed0[-1] - rnapro_seed0[-2]
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                for i in range(len(rnapro_seed0), N):
                    pad[i] = pad[i-1] + direction * 5.9
            rnapro_seed0 = pad

        template_coords_per_target[target_id]     = rnapro_seed0
        template_confidence_per_target[target_id] = RNAPRO_CONFIDENCE
        rnapro_used_per_target[target_id]         = True
        n_rnapro_used += 1

        if idx % 5 == 0:
            n_seeds = len(rnapro_coords[target_id])
            print(f"    -> RNAPRO template ({n_seeds} seeds, conf={RNAPRO_CONFIDENCE})")

        # Still run MSA search for MSA features (but template comes from RNAPro)
        similar = find_similar_sequences(
            sequence, train_seqs_extended, train_coords_dict, top_n=MSA_TOP_N
        )
        msa_hits_per_target[target_id] = similar
    else:
        # FALL BACK TO RUN 5 TEMPLATE SEARCH
        similar = find_similar_sequences(
            sequence, train_seqs_extended, train_coords_dict, top_n=MSA_TOP_N
        )
        msa_hits_per_target[target_id] = similar
        rnapro_used_per_target[target_id] = False
        n_search_used += 1

        if similar:
            best_tid, best_seq, best_score, best_coords = similar[0]
            adapted  = adapt_template_to_query(sequence, best_seq, best_coords)
            refined  = adaptive_rna_constraints(adapted, sequence, confidence=best_score)
            template_coords_per_target[target_id]     = refined
            template_confidence_per_target[target_id] = best_score
            if idx % 5 == 0:
                print(f"    -> Search template: {best_tid} (score={best_score:.3f})")
        else:
            template_coords_per_target[target_id]     = np.zeros((len(sequence), 3))
            template_confidence_per_target[target_id] = 0.0
            msa_hits_per_target[target_id]            = []
            if idx % 5 == 0:
                print(f"    -> No template found")

n_with_tmpl = sum(1 for v in template_confidence_per_target.values() if v > 0)
n_with_msa  = sum(1 for v in msa_hits_per_target.values() if len(v) >= 3)
print(f"\nTemplate search complete:")
print(f"  RNAPro templates:  {n_rnapro_used}/{len(test_seqs)}")
print(f"  Search templates:  {n_search_used}/{len(test_seqs)}")
print(f"  Total with tmpl:   {n_with_tmpl}/{len(test_seqs)}")
print(f"  MSA hits >= 3:     {n_with_msa}/{len(test_seqs)}")


# ============================================================
# CELL 9.5-13: MSA features, NN modules, backbone, training
# ============================================================
# ALL UNCHANGED from Run 5. These cells are identical.
# For brevity, the full code is copied from Run 5 verbatim.
# See hy_bas_adv1_run5_api_NB.py Cells 9.5 through 13.
#
# KEY POINT: Training uses the SAME data and SAME process as Run 5.
# RNAPro templates only affect INFERENCE (Cell 14), not training.
# The IPA module learns from the same pickle training data.
#
# If you have a Run 5 checkpoint, you can skip training entirely
# by loading it in Cell 12.
# ============================================================

# --- [PASTE CELLS 9.5, 10, 11, 12, 13 FROM RUN 5 VERBATIM HERE] ---
# --- They are ~600 lines, identical to Run 5                     ---
# --- See hy_bas_adv1_run5_api_NB.py for the full source          ---

# PLACEHOLDER: In the actual notebook, copy Cells 9.5-13 from Run 5.
# The code is 100% identical. No changes needed.
#
# For this file, we import the key objects that Cell 14 needs:
#   backbone          — RibonanzaNet backbone (loaded in Cell 11)
#   template_encoder  — TemplateEncoder (loaded in Cell 12)
#   ipa_module        — IPAStructureModule (loaded in Cell 12/13)
#   get_pairwise_and_single_features()  — from Cell 11
#   compute_msa_features()              — from Cell 9.5
#   msa_features_per_target             — from Cell 9.5
#
# IMPORTANT: When creating the actual Kaggle notebook, copy-paste
# the FULL code from Cells 9.5 through 13 of Run 5 here.

print("\n" + "="*60)
print("CELLS 9.5-13: COPY FROM RUN 5 (UNCHANGED)")
print("="*60)
print("In the actual Kaggle notebook, paste Run 5 Cells 9.5-13 here.")
print("They define: MSA features, IPA modules, backbone, training.")
print("No changes needed — they are 100% identical to Run 5.")


# ============================================================
# CELL 14: IPA Inference + RNAPro-Seeded Slot Assembly
# ============================================================
#
# RUN 6 CHANGES vs RUN 5:
#   NEW 4-branch slot assembly:
#     Branch 1: RNAPro available (RNAPRO_CONFIDENCE > threshold)
#       Slot 1: Raw RNAPro seed 0 (no IPA, no modification)
#       Slot 2: RNAPro seed 0 → IPA refinement
#       Slot 3: RNAPro seed 1 → IPA refinement (or noisy seed 0)
#       Slot 4: RNAPro seed 2 raw (or IPA from noisy seed 0)
#       Slot 5: IPA cold start (maximum diversity)
#
#     Branch 2-4: No RNAPro → same as Run 5 (unchanged)
#
# WHY Slot 1 is RAW RNAPro (no IPA):
#   RNAPro is a 488M param model trained on the same competition data.
#   Its raw prediction is likely better than anything our small IPA
#   can produce. By including the raw prediction as Slot 1, we
#   guarantee that at minimum, one slot has RNAPro-quality output.
#   If IPA improves it, great — Slots 2-3 will score higher.
#   If IPA hurts it, the raw Slot 1 is the safety net.
#
# PRESERVED FROM RUN 5:
#   Fix 1: raw template ALWAYS in Slot 1 (for non-RNAPro targets)
#   HYBRID_THRESHOLD = 0.20
#   adaptive_rna_constraints() on template slots
#   Long-sequence extrapolation
# ============================================================
print("\n" + "="*60)
print("PHASE 4: IPA Inference + RNAPro Templates (Run 6)")
print("="*60)

MAX_INFER_LEN    = 512
HYBRID_THRESHOLD = 0.20

all_predictions = []
infer_start     = time.time()

n_rnapro_slots = 0   # targets using RNAPro-seeded slots
n_hybrid_2slot = 0
n_hybrid_1slot = 0
n_no_template  = 0
n_ipa_slots    = 0

for idx, row in test_seqs.iterrows():
    target_id = row['target_id']
    sequence  = row['sequence']
    seq       = sequence[:MAX_INFER_LEN]
    N         = len(seq)

    if idx % 5 == 0:
        print(f"  Predicting {idx+1}/{len(test_seqs)}: "
              f"{target_id} ({len(sequence)} nt)")

    with torch.no_grad():
        # Backbone features (pairwise + single)
        pairwise, single = get_pairwise_and_single_features(seq)

        # Template features
        tmpl_coords = template_coords_per_target.get(target_id, np.zeros((N, 3)))
        tmpl_conf   = template_confidence_per_target.get(target_id, 0.0)
        uses_rnapro = rnapro_used_per_target.get(target_id, False)

        if len(tmpl_coords) > N:
            tmpl_coords = tmpl_coords[:N]
        elif len(tmpl_coords) < N:
            pad = np.zeros((N, 3))
            pad[:len(tmpl_coords)] = tmpl_coords
            tmpl_coords = pad

        coords_t  = torch.tensor(tmpl_coords, dtype=torch.float32, device=device)
        has_tmpl  = tmpl_conf > 0.01
        tmpl_feat = template_encoder(
            coords_t, confidence=tmpl_conf, has_template=has_tmpl
        ).unsqueeze(0)

        # MSA features
        msa_np = msa_features_per_target.get(target_id, np.zeros((N, N, MSA_DIM)))
        if msa_np.shape[0] > N:
            msa_np = msa_np[:N, :N, :]
        elif msa_np.shape[0] < N:
            pad = np.zeros((N, N, MSA_DIM), dtype=np.float32)
            m   = msa_np.shape[0]
            pad[:m, :m, :] = msa_np
            msa_np = pad
        msa_feat = torch.tensor(msa_np, dtype=torch.float32, device=device).unsqueeze(0)

        combined = torch.cat([pairwise, tmpl_feat, msa_feat], dim=-1)

        # IPA prediction helper
        def run_ipa(init_np, seed=0):
            torch.manual_seed(seed)
            init_t = torch.tensor(init_np, dtype=torch.float32, device=device)
            return ipa_module(single, combined, init_t).cpu().numpy()

        def add_noise(arr, sigma, seed=0):
            np.random.seed(seed)
            return (arr + np.random.normal(0, sigma, arr.shape)).astype(np.float32)

        zeros_np = np.zeros((N, 3), dtype=np.float32)

    # -------------------------------------------------------
    # NEW BRANCH: RNAPro available for this target
    # -------------------------------------------------------
    if uses_rnapro and target_id in rnapro_coords:
        n_rnapro_slots += 1
        seeds = rnapro_coords[target_id]

        # Get RNAPro seeds, length-matched to N
        def get_rnapro_seed(seed_idx):
            if seed_idx not in seeds:
                return None
            s = seeds[seed_idx].copy()
            if len(s) > N:
                s = s[:N]
            elif len(s) < N:
                pad = np.zeros((N, 3), dtype=np.float32)
                pad[:len(s)] = s
                if len(s) >= 2:
                    d = s[-1] - s[-2]
                    d = d / (np.linalg.norm(d) + 1e-8)
                    for i in range(len(s), N):
                        pad[i] = pad[i-1] + d * 5.9
                s = pad
            return s.astype(np.float32)

        seed0 = get_rnapro_seed(0)
        seed1 = get_rnapro_seed(1)
        seed2 = get_rnapro_seed(2)

        with torch.no_grad():
            # Slot 1: Raw RNAPro seed 0 (safety net — no IPA modification)
            slot1 = seed0.copy()

            # Slot 2: RNAPro seed 0 → IPA refinement (small delta on top)
            slot2 = run_ipa(seed0, seed=0)

            # Slot 3: RNAPro seed 1 → IPA refinement (or noisy seed 0)
            if seed1 is not None:
                slot3 = run_ipa(seed1, seed=1)
            else:
                slot3 = run_ipa(add_noise(seed0, 0.5, 1), seed=1)

            # Slot 4: RNAPro seed 2 raw (diversity) or IPA noisy
            if seed2 is not None:
                slot4 = seed2.copy()
            else:
                slot4 = run_ipa(add_noise(seed0, 1.0, 2), seed=2)

            # Slot 5: IPA cold start (maximum diversity — no RNAPro init)
            slot5 = run_ipa(zeros_np, seed=4)

        coords_list = [slot1, slot2, slot3, slot4, slot5]
        n_ipa_slots += 3  # slots 2,3,5 use IPA

        if idx % 5 == 0:
            n_seeds = len(seeds)
            print(f"    RNAPRO: {n_seeds} seeds -> "
                  f"slot1=raw, slots2-3=IPA refined, slot4=seed2, slot5=cold")

    # -------------------------------------------------------
    # Run 5 behavior: high-confidence template
    # -------------------------------------------------------
    elif tmpl_conf > HYBRID_THRESHOLD:
        n_hybrid_2slot += 1
        n_ipa_slots    += 3
        t_np = tmpl_coords.astype(np.float32)

        with torch.no_grad():
            ipa_0 = run_ipa(t_np,                    seed=0)
            ipa_1 = run_ipa(add_noise(t_np, 0.5, 1), seed=1)
            ipa_2 = run_ipa(add_noise(t_np, 1.0, 2), seed=2)

        full_tmpl = template_coords_per_target.get(target_id, np.zeros((len(sequence), 3)))
        if len(full_tmpl) > len(sequence):
            full_tmpl = full_tmpl[:len(sequence)]
        elif len(full_tmpl) < len(sequence):
            pad = np.zeros((len(sequence), 3))
            pad[:len(full_tmpl)] = full_tmpl
            full_tmpl = pad

        tmpl_r = adaptive_rna_constraints(full_tmpl.copy(), sequence, confidence=tmpl_conf)
        np.random.seed(42)
        noisy  = full_tmpl + np.random.normal(0, 0.5, full_tmpl.shape)
        tmpl_n = adaptive_rna_constraints(noisy, sequence, confidence=tmpl_conf)

        coords_list = [tmpl_r, tmpl_n, ipa_0, ipa_1, ipa_2]

        if idx % 5 == 0:
            print(f"    HYBRID: conf={tmpl_conf:.3f} > {HYBRID_THRESHOLD} "
                  f"-> slots 1-2 template, 3-5 IPA")

    # -------------------------------------------------------
    # Run 5 behavior: weak template (Fix 1)
    # -------------------------------------------------------
    elif tmpl_conf > 0.01:
        n_hybrid_1slot += 1
        n_ipa_slots    += 4
        t_np = tmpl_coords.astype(np.float32)

        with torch.no_grad():
            ipa_0 = run_ipa(t_np,                    seed=0)
            ipa_1 = run_ipa(add_noise(t_np, 0.5, 1), seed=1)
            ipa_2 = run_ipa(add_noise(t_np, 1.0, 2), seed=2)
            ipa_3 = run_ipa(add_noise(t_np, 1.5, 3), seed=3)

        full_tmpl = template_coords_per_target.get(target_id, np.zeros((len(sequence), 3)))
        if len(full_tmpl) > len(sequence):
            full_tmpl = full_tmpl[:len(sequence)]
        elif len(full_tmpl) < len(sequence):
            pad = np.zeros((len(sequence), 3))
            pad[:len(full_tmpl)] = full_tmpl
            full_tmpl = pad

        tmpl_r = adaptive_rna_constraints(full_tmpl.copy(), sequence, confidence=tmpl_conf)
        coords_list = [tmpl_r, ipa_0, ipa_1, ipa_2, ipa_3]

        if idx % 5 == 0:
            print(f"    FIX1: conf={tmpl_conf:.3f} <= {HYBRID_THRESHOLD} "
                  f"-> slot 1 raw template, 2-5 IPA")

    # -------------------------------------------------------
    # Run 5 behavior: no template
    # -------------------------------------------------------
    else:
        n_no_template += 1
        n_ipa_slots   += 5

        with torch.no_grad():
            ipa_0 = run_ipa(zeros_np,                    seed=0)
            ipa_1 = run_ipa(add_noise(zeros_np, 0.5, 1), seed=1)
            ipa_2 = run_ipa(add_noise(zeros_np, 1.0, 2), seed=2)
            ipa_3 = run_ipa(add_noise(zeros_np, 1.5, 3), seed=3)
            ipa_4 = run_ipa(zeros_np,                    seed=5)

        coords_list = [ipa_0, ipa_1, ipa_2, ipa_3, ipa_4]

        if idx % 5 == 0:
            print(f"    NO TEMPLATE: conf={tmpl_conf:.3f} -> all 5 IPA")

    # Extend to full sequence length if truncated
    def extend_to_full(arr):
        if len(sequence) <= MAX_INFER_LEN:
            return arr
        remaining = len(sequence) - MAX_INFER_LEN
        last_dir  = (arr[-1] - arr[-2]) if arr.shape[0] >= 2 else np.array([5.9, 0., 0.])
        last_dir  = last_dir / (np.linalg.norm(last_dir) + 1e-8) * 5.9
        extra     = np.array([arr[-1] + last_dir*(i+1) for i in range(remaining)])
        return np.concatenate([arr, extra])

    coords_list = [extend_to_full(c) for c in coords_list]

    # Write prediction rows
    for j in range(len(sequence)):
        pred_row = {
            'ID':      f"{target_id}_{j+1}",
            'resname':  sequence[j],
            'resid':    j + 1,
        }
        for i in range(5):
            pred_row[f'x_{i+1}'] = float(coords_list[i][j][0])
            pred_row[f'y_{i+1}'] = float(coords_list[i][j][1])
            pred_row[f'z_{i+1}'] = float(coords_list[i][j][2])
        all_predictions.append(pred_row)

print(f"\nInference complete: {len(all_predictions)} rows in {time.time()-infer_start:.0f}s")
print(f"  Targets with RNAPro templates:                     {n_rnapro_slots}")
print(f"  Targets 2-slot template (conf > {HYBRID_THRESHOLD}): {n_hybrid_2slot}")
print(f"  Targets 1-slot template (Fix 1):                    {n_hybrid_1slot}")
print(f"  Targets no template (all IPA):                      {n_no_template}")
print(f"  Total IPA-predicted slots:                          {n_ipa_slots}")

submission_df = pd.DataFrame(all_predictions)
col_order     = ['ID', 'resname', 'resid']
for i in range(1, 6):
    for c in ['x', 'y', 'z']:
        col_order.append(f'{c}_{i}')
submission_df = submission_df[col_order]
submission_df.to_csv(RAW_SUBMISSION_PATH, index=False)
print(f"Raw submission saved: {len(submission_df)} rows -> {RAW_SUBMISSION_PATH}")


# ============================================================
# CELL 15: Post-Processing (UNCHANGED from Run 5)
# ============================================================
print("\n" + "="*60)
print("PHASE 5: Post-Processing (Option B)")
print("="*60)

sample_rows  = {}
sample_order = []
cols         = None
with open(SAMPLE_CSV, "r") as f:
    reader = csv.DictReader(f)
    cols   = reader.fieldnames
    for row in reader:
        sample_rows[row["ID"]] = row
        sample_order.append(row["ID"])
print(f"Sample expects {len(sample_order)} rows")

raw_rows = {}
with open(RAW_SUBMISSION_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        raw_rows[row["ID"]] = row
print(f"Run 6 produced {len(raw_rows)} rows")

matched = 0
filled  = 0
with open(FINAL_SUBMISSION_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=cols)
    writer.writeheader()
    for sid in sample_order:
        if sid in raw_rows:
            writer.writerow(raw_rows[sid]); matched += 1
        else:
            writer.writerow(sample_rows[sid]); filled += 1

print(f"Final submission.csv:")
print(f"  Matched from Run 6: {matched}")
print(f"  Filled with zeros:  {filled}")
print(f"  Total rows:         {matched + filled}")
print(f"  File size:          {os.path.getsize(FINAL_SUBMISSION_PATH)} bytes")

print("\n" + "="*60)
print("DONE. submission.csv ready. Run 6 (IPA + RNAPro) complete.")
print("="*60)
print(f"\nRun 6 summary:")
print(f"  RNAPro templates used:  {n_rnapro_slots} targets")
print(f"  Fallback to Run 5:      {n_hybrid_2slot + n_hybrid_1slot + n_no_template} targets")
if n_rnapro_slots > 0:
    print(f"  RNAPro coverage:        {100*n_rnapro_slots/len(test_seqs):.1f}%")
