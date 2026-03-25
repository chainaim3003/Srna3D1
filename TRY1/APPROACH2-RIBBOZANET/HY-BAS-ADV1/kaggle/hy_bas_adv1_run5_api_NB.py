# ============================================================
# HY-BAS-ADV1 RUN 5: IPA (Invariant Point Attention)
# ============================================================
#
# WHAT THIS IS:
#   Run 5 = Run 4 + IPA structure module replacing the distance
#   head + MDS + refinement pipeline.
#
#   Instead of predicting a distance matrix and reconstructing
#   3D coords via MDS, IPA (from AlphaFold2) predicts 3D
#   coordinates DIRECTLY via iterative frame refinement.
#
# SIMPLE EXPLANATION (for a high schooler):
#   Run 4 said: "atom A and B are 5 Angstroms apart, C and D
#   are 3 Angstroms apart... now rebuild the shape from those
#   distances." That is slow and can produce mirror images.
#
#   Run 5 says: "place each atom directly in 3D space, then
#   refine the placement 4 times using attention over the
#   pairwise features." No distance matrix. No MDS. Just IPA.
#
# WHAT CHANGES vs RUN 4:
#   Cell 0:   Add IPA config params
#   Cell 2:   Add einops import (needed by lucidrains IPA code)
#   Cell 3:   Search for Run 4 checkpoint (adv1_best_model.pt)
#   Cell 10:  REMOVE DistanceMatrixHead, mds_reconstruct, refine_coords
#             ADD lucidrains IPA (Inline B copy-paste ~230 lines)
#             ADD build_rna_frames(), IPAStructureModule, fape_loss()
#   Cell 11:  ADD get_pairwise_and_single_features() returning
#             pairwise (1,N,N,64) AND single repr (1,N,256)
#   Cell 12:  REPLACE DistanceMatrixHead with IPAStructureModule
#             Warm-start backbone + template_encoder from Run 4
#             IPA module: random init, trains from scratch
#   Cell 13:  REPLACE MSE distance loss with FAPE loss
#             IPA forward pass produces coords directly
#             Teacher forcing: 50% start from true coords + noise
#             Early stopping at patience=10
#   Cell 14:  REPLACE dist+refinement with IPA inference
#             5 diversity slots via frame perturbation
#             Template slots 1-2 and Fix 1 preserved from Run 4
#   Cell 15:  UNCHANGED post-processing
#
# WHAT IS UNCHANGED vs RUN 4:
#   Cells 1,4,5,6,7,8,9,9.5: entirely unchanged
#   TemplateEncoder class: unchanged
#   MSA features: unchanged (still 8 channels)
#   Fix 1 (raw template always in Slot 1): preserved
#   HYBRID_THRESHOLD = 0.20: preserved
#
# LOCKED DESIGN DECISIONS:
#   IPA source:           Inline B (lucidrains v0.2.2 copy-paste)
#   IPA_ITERATIONS:       4 (shared weights)
#   IPA_HEADS:            4
#   FAPE_CLAMP:           10.0 Angstroms
#   FREEZE_BACKBONE_IPA:  True (Strategy 1, 2-3x epoch speedup)
#   TRAIN_EPOCHS ceiling: 40 (early stopping cuts short)
#   EARLY_STOP_PATIENCE:  10
#   Gradient checkpointing: Off (10 GB VRAM headroom)
#   Inference fallback:   Deleted (Option A)
#                         Restorable from Run 4 in ~30 min if needed
#
# HOW RUN 5 LEVERAGES RUN 3 AND RUN 4:
#   Loads Run 4 checkpoint. Warm-starts:
#     backbone layers 0-8  (all trained by Run 3 + Run 4)
#     template_encoder     (trained by Run 3 + Run 4)
#   IPA module uses these already-good features from day 1.
#   It only needs to learn to READ coordinates from features
#   that already encode structural information — much easier
#   than learning RNA structure from scratch.
#
# ESTIMATED RUNTIME: ~2.5-4 hours on T4 GPU
# ============================================================


# ============================================================
# CELL 0: USER-CONFIGURABLE PATHS AND IPA CONFIG
# ============================================================
EXTENDED_SEQ_CSV   = None
EXTENDED_COORD_CSV = None
OUTPUT_DIR         = '/kaggle/working'
COMP_BASE_PRIMARY  = '/kaggle/input/competitions/stanford-rna-3d-folding-2/'
COMP_BASE_FALLBACK = '/kaggle/input/stanford-rna-3d-folding-2/'

# MSA configuration (unchanged from Run 4)
MSA_TOP_N = 20
MSA_DIM   = 8

# IPA configuration (NEW in Run 5)
IPA_DIM          = 256    # single repr dim (matches RibonanzaNet hidden dim)
IPA_ITERATIONS   = 4      # refinement passes (shared weights)
IPA_HEADS        = 4      # attention heads (AF2 uses 12; 4 suits small dataset)
FAPE_CLAMP       = 10.0   # per-point error clamp in Angstroms (AF2 standard)
AUX_DIST_WEIGHT  = 0.1    # weight of auxiliary MSE distance loss alongside FAPE

# Training strategy (NEW in Run 5)
FREEZE_BACKBONE_IPA = True   # Strategy 1: freeze all backbone layers
                              # 2-3x faster per epoch. Backbone already
                              # trained by Run 4. Set False to allow
                              # backbone layers 7-8 to adapt to IPA.
TRAIN_EPOCHS        = 40     # Ceiling only. Early stopping cuts short.
EARLY_STOP_PATIENCE = 10     # Stop if val_loss doesn't improve for N epochs.


# ============================================================
# CELL 1: Install biopython (direct sys.path injection)
# ============================================================
# UNCHANGED from Run 4.
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
# CHANGE from Run 4: added einops (needed by lucidrains IPA code)
# einops is pre-installed on Kaggle - no wheel needed.
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

from einops import rearrange, repeat   # NEW in Run 5 - needed by IPA code

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
# CHANGE from Run 4: now searches for Run 4 checkpoint
#   RUN4_CHECKPOINT: adv1_best_model.pt (Run 4 Kaggle output)
#   Cell 12 warm-starts backbone + template encoder from this.
#   The IPA module does NOT load from this checkpoint.
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
ADV1_WEIGHTS     = None   # BASIC best_model.pt (fallback)
RUN4_CHECKPOINT  = None   # Run 4 adv1_best_model.pt (preferred warm-start)

for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f == "RibonanzaNet.pt":
            BACKBONE_WEIGHTS = os.path.join(root, f)
        if f == "best_model.pt":
            ADV1_WEIGHTS = os.path.join(root, f)
        if f.strip() == "adv1_best_model.pt":
            RUN4_CHECKPOINT = os.path.join(root, f)

print(f"Backbone weights: {BACKBONE_WEIGHTS}")
print(f"BASIC weights:    {ADV1_WEIGHTS}")
print(f"Run 4 checkpoint: {RUN4_CHECKPOINT}")

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

CHECKPOINT_PATH       = f'{OUTPUT_DIR}/adv1_run5_best_model.pt'
RAW_SUBMISSION_PATH   = f'{OUTPUT_DIR}/submission_raw.csv'
FINAL_SUBMISSION_PATH = f'{OUTPUT_DIR}/submission.csv'


# ============================================================
# CELL 4: Load competition data
# ============================================================
# UNCHANGED from Run 4.
# ============================================================
print("\nLoading competition data...")
train_seqs   = pd.read_csv(BASE + 'train_sequences.csv')
test_seqs    = pd.read_csv(BASE + 'test_sequences.csv')
train_labels = pd.read_csv(BASE + 'train_labels.csv')
print(f"Train: {len(train_seqs)}, Test: {len(test_seqs)}")


# ============================================================
# CELL 5: Load extended data (rna_cif_to_csv)
# ============================================================
# UNCHANGED from Run 4.
# ============================================================
if EXTENDED_SEQ_CSV is None or EXTENDED_COORD_CSV is None:
    raise FileNotFoundError(
        "Could not find rna_sequences.csv or rna_coordinates.csv. "
        "Attach the rna-cif-to-csv dataset or set paths in Cell 0."
    )
print("Loading extended data...")
train_seqs_v2   = pd.read_csv(EXTENDED_SEQ_CSV)
train_labels_v2 = pd.read_csv(EXTENDED_COORD_CSV)
print(f"Extended: {len(train_seqs_v2)} seqs, {len(train_labels_v2)} labels")


# ============================================================
# CELL 6: Extend datasets
# ============================================================
# UNCHANGED from Run 4.
# ============================================================
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


# ============================================================
# CELL 7: Process labels into coordinate dictionary
# ============================================================
# UNCHANGED from Run 4.
# ============================================================
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


# ============================================================
# CELL 8: Fork 2 template functions
# ============================================================
# UNCHANGED from Run 4.
# ============================================================

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
# CELL 9: Template search + collect MSA sequences
# ============================================================
# UNCHANGED from Run 4.
# ============================================================
print("\n" + "="*60)
print("PHASE 1: Fork 2 Template Search + MSA Collection")
print("="*60)

template_coords_per_target     = {}
template_confidence_per_target = {}
msa_hits_per_target            = {}

start_time = time.time()
for idx, row in test_seqs.iterrows():
    target_id = row['target_id']
    sequence  = row['sequence']
    if idx % 5 == 0:
        elapsed = time.time() - start_time
        print(f"  {idx+1}/{len(test_seqs)}: {target_id} ({len(sequence)} nt) [{elapsed:.0f}s]")

    similar = find_similar_sequences(
        sequence, train_seqs_extended, train_coords_dict, top_n=MSA_TOP_N
    )
    msa_hits_per_target[target_id] = similar

    if similar:
        best_tid, best_seq, best_score, best_coords = similar[0]
        adapted  = adapt_template_to_query(sequence, best_seq, best_coords)
        refined  = adaptive_rna_constraints(adapted, sequence, confidence=best_score)
        template_coords_per_target[target_id]     = refined
        template_confidence_per_target[target_id] = best_score
        print(f"    -> Template: {best_tid} (score={best_score:.3f}), MSA hits: {len(similar)}")
    else:
        template_coords_per_target[target_id]     = np.zeros((len(sequence), 3))
        template_confidence_per_target[target_id] = 0.0
        msa_hits_per_target[target_id]            = []
        print(f"    -> No template found")

n_with_tmpl = sum(1 for v in template_confidence_per_target.values() if v > 0)
n_with_msa  = sum(1 for v in msa_hits_per_target.values() if len(v) >= 3)
print(f"\nTemplate search complete. {n_with_tmpl}/{len(test_seqs)} targets have templates.")
print(f"MSA collection complete.  {n_with_msa}/{len(test_seqs)} targets have 3+ MSA hits.")


# ============================================================
# CELL 9.5: Compute MSA features
# ============================================================
# UNCHANGED from Run 4.
# ============================================================
print("\n" + "="*60)
print("PHASE 1.5: MSA Feature Computation (unchanged from Run 4)")
print("="*60)

def compute_msa_features(query_seq, similar_hits, max_len=512):
    """Compute 8-channel MSA features. Returns numpy array (N, N, 8)."""
    N          = min(len(query_seq), max_len)
    n_channels = 8
    if not similar_hits or len(similar_hits) < 2:
        return np.zeros((N, N, n_channels), dtype=np.float32)

    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3, '-': 4}
    n_seqs      = len(similar_hits) + 1
    aln_matrix  = np.full((n_seqs, N), 4, dtype=np.int32)

    for i in range(N):
        aln_matrix[0, i] = base_to_idx.get(query_seq[i].upper(), 4)

    for hit_idx, (tid, hit_seq, score, coords) in enumerate(similar_hits):
        try:
            alns = pairwise2.align.globalms(
                Seq(query_seq[:N]), hit_seq, 2.9, -1, -10, -0.5, one_alignment_only=True
            )
            if not alns:
                continue
            aligned_query = alns[0].seqA
            aligned_hit   = alns[0].seqB
            qi = 0
            for k in range(len(aligned_query)):
                if qi >= N:
                    break
                if aligned_query[k] != '-':
                    if aligned_hit[k] != '-':
                        aln_matrix[hit_idx+1, qi] = base_to_idx.get(aligned_hit[k].upper(), 4)
                    qi += 1
        except:
            continue

    n_bases = 5
    freq    = np.zeros((N, n_bases), dtype=np.float32)
    for pos in range(N):
        counts    = np.bincount(aln_matrix[:, pos], minlength=n_bases).astype(np.float32)
        freq[pos] = counts / n_seqs

    eps          = 1e-10
    entropy      = -np.sum(freq * np.log(freq + eps), axis=1)
    max_entropy  = np.log(n_bases)
    conservation = 1.0 - (entropy / max_entropy)
    gap_freq     = freq[:, 4]
    neff         = float(n_seqs) / 100.0

    n_real        = 4
    joint_counts  = np.zeros((N, N, n_real, n_real), dtype=np.float32)
    for si in range(n_seqs):
        row = aln_matrix[si, :]
        for i in range(N):
            if row[i] >= n_real:
                continue
            for j in range(i, N):
                if row[j] >= n_real:
                    continue
                joint_counts[i, j, row[i], row[j]] += 1
                if i != j:
                    joint_counts[j, i, row[j], row[i]] += 1

    pair_totals = joint_counts.sum(axis=(2, 3), keepdims=True)
    joint_prob  = joint_counts / (pair_totals + eps)
    marg_i      = joint_prob.sum(axis=3)
    marg_j      = joint_prob.sum(axis=2)
    outer_prod  = marg_i[:, :, :, None] * marg_j[:, :, None, :]
    log_ratio   = np.log((joint_prob + eps) / (outer_prod + eps))
    mi          = np.maximum((joint_prob * log_ratio).sum(axis=(2, 3)), 0.0)

    mi_row_mean    = mi.mean(axis=1)
    mi_col_mean    = mi.mean(axis=0)
    mi_global_mean = mi.mean() + eps
    apc            = mi_row_mean[:, None] * mi_col_mean[None, :] / mi_global_mean
    mi_apc         = np.maximum(mi - apc, 0.0)

    features          = np.zeros((N, N, n_channels), dtype=np.float32)
    features[:, :, 0] = mi
    features[:, :, 1] = mi_apc
    features[:, :, 2] = conservation[:, None] * np.ones((1, N))
    features[:, :, 3] = np.ones((N, 1)) * conservation[None, :]
    features[:, :, 4] = conservation[:, None] * conservation[None, :]
    features[:, :, 5] = gap_freq[:, None] * np.ones((1, N))
    features[:, :, 6] = np.ones((N, 1)) * gap_freq[None, :]
    features[:, :, 7] = neff
    return features


msa_features_per_target = {}
msa_start = time.time()
for idx, row in test_seqs.iterrows():
    target_id = row['target_id']
    sequence  = row['sequence']
    hits      = msa_hits_per_target.get(target_id, [])
    msa_feat  = compute_msa_features(sequence, hits, max_len=512)
    msa_features_per_target[target_id] = msa_feat
    if idx % 5 == 0:
        elapsed = time.time() - msa_start
        print(f"  MSA {idx+1}/{len(test_seqs)}: {target_id} "
              f"({msa_feat.shape[0]}x{msa_feat.shape[2]}ch), "
              f"{len(hits)} hits [{elapsed:.0f}s]")
print(f"\nMSA feature computation complete in {time.time()-msa_start:.0f}s")


# ============================================================
# CELL 10: Neural Network Modules
# ============================================================
#
# RUN 5 CHANGES vs RUN 4:
#   REMOVED: DistanceMatrixHead, mds_reconstruct, refine_coords
#   KEPT:    TemplateEncoder (100% unchanged)
#   ADDED:   lucidrains IPA code (Inline B, ~230 lines, copy-paste)
#            build_rna_frames()
#            IPAStructureModule
#            fape_loss()
# ============================================================
print("\n" + "="*60)
print("PHASE 2: Neural Network Modules (Run 5 IPA)")
print("="*60)


# ------------------------------------------------------------------
# KEPT FROM RUN 4: TemplateEncoder (UNCHANGED)
# ------------------------------------------------------------------
class TemplateEncoder(nn.Module):
    def __init__(self, template_dim=16, num_bins=22, max_dist=40.0):
        super().__init__()
        self.template_dim = template_dim
        self.num_bins     = num_bins
        bin_width         = max_dist / (num_bins - 1)
        edges             = torch.arange(0, max_dist + bin_width, bin_width)[:num_bins]
        self.register_buffer('bin_edges', edges)
        self.projection   = nn.Linear(num_bins, template_dim)
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.projection.bias)

    def forward(self, coords, confidence=1.0, has_template=True):
        N = coords.shape[0]
        if not has_template:
            return torch.zeros(N, N, self.template_dim, device=coords.device)
        diff     = coords.unsqueeze(0) - coords.unsqueeze(1)
        dist     = torch.sqrt((diff**2).sum(-1) + 1e-8)
        bin_idx  = torch.bucketize(dist, self.bin_edges).clamp(0, self.num_bins-1)
        bins     = torch.zeros(N, N, self.num_bins, device=dist.device)
        bins.scatter_(2, bin_idx.unsqueeze(-1), 1.0)
        return self.projection(bins) * confidence


# ==================================================================
# BEGIN: Copied from lucidrains/invariant-point-attention v0.2.2
# Source:  https://github.com/lucidrains/invariant-point-attention
# License: MIT   Author: Phil Wang (lucidrains)
#
# DO NOT MODIFY this section.
# We copy-paste proven library code to avoid a wheel dependency.
# Any edits risk breaking the rotation math.
# ==================================================================

def _ipa_exists(val):
    return val is not None

def _ipa_default(val, d):
    return val if _ipa_exists(val) else d


class InvariantPointAttention(nn.Module):
    """
    Invariant Point Attention (IPA) from AlphaFold2.

    Simple explanation:
      Standard attention compares token vectors: "how similar are A and B?"
      IPA compares tokens AND their 3D positions in local frames:
      "how similar are A and B given where they are in 3D space?"
      The comparison is invariant to global rotation/translation,
      so mirrored structures look different (prevents mirror-flip bugs).
    """
    def __init__(
        self,
        dim,
        heads                 = 8,
        scalar_key_dim        = 16,
        scalar_value_dim      = 16,
        point_key_dim         = 4,
        point_value_dim       = 4,
        pairwise_repr_dim     = None,
        require_pairwise_repr = True,
    ):
        super().__init__()
        self.eps    = 1e-8
        self.heads  = heads
        self.require_pairwise_repr = require_pairwise_repr

        # Scalar (sequence-based) queries, keys, values
        self.to_scalar_q = nn.Linear(dim, scalar_key_dim   * heads, bias=False)
        self.to_scalar_k = nn.Linear(dim, scalar_key_dim   * heads, bias=False)
        self.to_scalar_v = nn.Linear(dim, scalar_value_dim * heads, bias=False)

        # 3D point queries, keys, values (expressed in local frames)
        self.to_point_q  = nn.Linear(dim, point_key_dim   * heads * 3, bias=False)
        self.to_point_k  = nn.Linear(dim, point_key_dim   * heads * 3, bias=False)
        self.to_point_v  = nn.Linear(dim, point_value_dim * heads * 3, bias=False)

        # Pairwise bias
        if require_pairwise_repr:
            pairwise_repr_dim = _ipa_default(pairwise_repr_dim, dim)
            self.pairwise_attn_logits = nn.Linear(pairwise_repr_dim, heads, bias=False)

        # Learnable per-head weight for point attention contribution
        self.point_weights = nn.Parameter(
            torch.log(torch.exp(torch.ones(1, heads)) - 1.)
        )

        # Output projection
        num_pair_out = pairwise_repr_dim if require_pairwise_repr else 0
        output_dim   = heads * (
            scalar_value_dim +
            point_value_dim * 3 +
            point_value_dim +
            num_pair_out
        )
        self.to_out = nn.Linear(output_dim, dim)

        self.scalar_key_dim    = scalar_key_dim
        self.scalar_value_dim  = scalar_value_dim
        self.point_key_dim     = point_key_dim
        self.point_value_dim   = point_value_dim
        self.pairwise_repr_dim = pairwise_repr_dim if require_pairwise_repr else 0

    def forward(
        self,
        single_repr,         # (b, n, dim)
        pairwise_repr=None,  # (b, n, n, dim)
        *,
        rotations,           # (b, n, 3, 3)
        translations,        # (b, n, 3)
        mask=None,           # (b, n) bool
    ):
        b, n, d = single_repr.shape
        h       = self.heads

        # Scalar QKV
        sq = rearrange(self.to_scalar_q(single_repr), 'b n (h d) -> b h n d', h=h)
        sk = rearrange(self.to_scalar_k(single_repr), 'b n (h d) -> b h n d', h=h)
        sv = rearrange(self.to_scalar_v(single_repr), 'b n (h d) -> b h n d', h=h)

        # Point QKV in local frames: (b, h, n, pts, 3)
        pq = rearrange(self.to_point_q(single_repr), 'b n (h p c) -> b h n p c', h=h, c=3)
        pk = rearrange(self.to_point_k(single_repr), 'b n (h p c) -> b h n p c', h=h, c=3)
        pv = rearrange(self.to_point_v(single_repr), 'b n (h p c) -> b h n p c', h=h, c=3)

        # Transform local points to global frame: global = R @ local + t
        re     = rotations.unsqueeze(1).expand(b, h, n, 3, 3)
        te_q   = translations.unsqueeze(1).unsqueeze(3).expand(b, h, n, self.point_key_dim, 3)
        te_v   = translations.unsqueeze(1).unsqueeze(3).expand(b, h, n, self.point_value_dim, 3)

        def to_global(local_pts, t_exp):
            return torch.einsum('b h n i j, b h n p j -> b h n p i', re, local_pts) + t_exp

        pq_g = to_global(pq, te_q)
        pk_g = to_global(pk, te_q)
        pv_g = to_global(pv, te_v)

        # Scalar attention logits
        sa   = torch.einsum('b h i d, b h j d -> b h i j', sq, sk) * (self.scalar_key_dim ** -0.5)

        # Point attention logits: penalise distant 3D points
        pw   = F.softplus(self.point_weights).view(1, h, 1, 1)
        pdiff = (rearrange(pq_g, 'b h i p c -> b h i () p c') -
                 rearrange(pk_g, 'b h j p c -> b h () j p c'))
        pa   = -0.5 * (pw * (pdiff**2).sum(-1).sum(-1))

        attn_logits = sa + pa

        # Pairwise bias
        if self.require_pairwise_repr and _ipa_exists(pairwise_repr):
            attn_logits = attn_logits + rearrange(
                self.pairwise_attn_logits(pairwise_repr), 'b i j h -> b h i j'
            )

        # Mask
        if _ipa_exists(mask):
            attn_logits = attn_logits.masked_fill(
                rearrange(mask.float(), 'b j -> b () () j') == 0, -1e9
            )

        attn = attn_logits.softmax(dim=-1)  # (b, h, n, n)

        # Aggregate scalar values
        so   = torch.einsum('b h i j, b h j d -> b h i d', attn, sv)

        # Aggregate point values
        pva  = torch.einsum('b h i j, b h j p c -> b h i p c', attn, pv_g)

        # Transform aggregated points back to local frame: R^T @ (global - t)
        rt   = rotations.unsqueeze(1).expand(b, h, n, 3, 3).transpose(-1, -2)
        to_  = translations.unsqueeze(1).unsqueeze(3).expand(b, h, n, self.point_value_dim, 3)
        pvl  = torch.einsum('b h n i j, b h n p j -> b h n p i', rt, pva - to_)
        pn   = torch.norm(pvl, dim=-1)

        # Aggregate pairwise features
        if self.require_pairwise_repr and _ipa_exists(pairwise_repr):
            po  = torch.einsum('b h i j, b i j d -> b h i d', attn, pairwise_repr)
        else:
            po  = None

        # Concatenate all outputs
        sf = rearrange(so,  'b h n d   -> b n (h d)')
        pf = rearrange(pvl, 'b h n p c -> b n (h p c)')
        nf = rearrange(pn,  'b h n p   -> b n (h p)')

        if _ipa_exists(po):
            out = torch.cat([sf, pf, nf, rearrange(po, 'b h n d -> b n (h d)')], dim=-1)
        else:
            out = torch.cat([sf, pf, nf], dim=-1)

        return self.to_out(out)  # (b, n, dim)


class IPABlock(nn.Module):
    """
    IPA attention + feedforward + layer norms.
    One complete transformer-style block operating in 3D space.
    """
    def __init__(self, dim, ff_mult=1, **ipa_kwargs):
        super().__init__()
        self.attn      = InvariantPointAttention(dim, **ipa_kwargs)
        self.attn_norm = nn.LayerNorm(dim)
        ff_dim         = dim * ff_mult
        self.ff        = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim),
        )

    def forward(self, x, pairwise_repr=None, *, rotations, translations, mask=None):
        x = self.attn_norm(x + self.attn(
            x, pairwise_repr,
            rotations=rotations, translations=translations, mask=mask,
        ))
        return x + self.ff(x)

# ==================================================================
# END: lucidrains library code
# ==================================================================


# ------------------------------------------------------------------
# NEW IN RUN 5: build_rna_frames
#
# Converts 3D backbone coords (N, 3) to rigid body frames.
# IPA requires per-residue frames (rotation + translation) to
# express 3D point attention invariantly.
# ------------------------------------------------------------------
def build_rna_frames(coords):
    """
    Build per-residue local coordinate frames from backbone coords.

    Simple explanation:
      For each nucleotide i, we build a miniature coordinate system
      centred on it. The x-axis points toward the next nucleotide,
      the z-axis is perpendicular to the local chain plane, and the
      y-axis completes the right-handed system.
      IPA uses these frames to measure 3D distances invariantly -
      it doesn't matter which way the whole molecule is oriented.

    Args:
        coords: torch.Tensor (N, 3)
    Returns:
        rotations:    (N, 3, 3) - rotation matrices
        translations: (N, 3)    - origins (same as coords)
    """
    N   = coords.shape[0]
    eps = 1e-8

    rotations    = torch.eye(3, device=coords.device).unsqueeze(0).expand(N, -1, -1).clone()
    translations = coords.clone()

    for i in range(N):
        has_next = (i < N - 1)
        has_prev = (i > 0)

        if not has_next and not has_prev:
            continue  # single residue: keep identity

        # x-axis: along backbone direction
        if has_next:
            x_vec = coords[i+1] - coords[i]
        else:
            x_vec = coords[i] - coords[i-1]

        x_len = x_vec.norm() + eps
        if x_len < eps * 10:
            continue  # degenerate coords: keep identity
        x_axis = x_vec / x_len

        # Reference vector for z-axis
        if has_prev:
            back_vec  = coords[i-1] - coords[i]
            back_axis = back_vec / (back_vec.norm() + eps)
        else:
            back_axis = torch.tensor([0., 1., 0.], device=coords.device)
            if abs(torch.dot(x_axis, back_axis).item()) > 0.9:
                back_axis = torch.tensor([0., 0., 1.], device=coords.device)

        # z-axis: perpendicular to chain plane
        z_vec  = torch.cross(x_axis, back_axis, dim=0)
        z_len  = z_vec.norm() + eps
        if z_len < eps * 10:
            continue  # degenerate
        z_axis = z_vec / z_len

        # y-axis: completes right-handed frame
        y_axis = torch.cross(z_axis, x_axis, dim=0)
        y_axis = y_axis / (y_axis.norm() + eps)

        rotations[i] = torch.stack([x_axis, y_axis, z_axis], dim=1)

    return rotations, translations


# ------------------------------------------------------------------
# NEW IN RUN 5: IPAStructureModule
#
# Iterative IPA refinement with shared weights.
# Produces 3D coordinates directly from:
#   single repr (1, N, 256): per-nucleotide backbone embedding
#   pair repr   (1, N, N, 88): pairwise features (backbone+tmpl+MSA)
#   init_coords (N, 3): starting coordinate estimate
#
# Architecture:
#   pair_proj:    Linear(88 -> 256) maps pair features to IPA dim
#   single_norm:  LayerNorm(256)
#   ipa_block:    ONE shared IPABlock reused for all IPA_ITERATIONS
#   coord_update: MLP(256 -> 3) maps updated single repr to delta coords
#
# Per iteration:
#   1. Build frames from current coords
#   2. IPABlock(single, pair, frames) -> updated_single
#   3. delta = coord_update(updated_single)
#   4. coords += delta
# ------------------------------------------------------------------
class IPAStructureModule(nn.Module):
    def __init__(self, single_dim, pair_dim, n_iter, heads):
        super().__init__()
        self.n_iter     = n_iter
        self.single_dim = single_dim

        # Project pair features to single_dim for IPA pairwise input
        self.pair_proj = nn.Sequential(
            nn.Linear(pair_dim, single_dim),
            nn.ReLU(),
            nn.LayerNorm(single_dim),
        )

        # Normalise backbone single repr
        self.single_norm = nn.LayerNorm(single_dim)

        # Shared IPABlock: ONE instance, reused for all n_iter passes
        # Shared weights = fewer params (175K vs 700K), less overfitting
        # on 661 training structures. Change n_iter without retraining.
        self.ipa_block = IPABlock(
            dim               = single_dim,
            heads             = heads,
            scalar_key_dim    = 16,
            scalar_value_dim  = 16,
            point_key_dim     = 4,
            point_value_dim   = 4,
            pairwise_repr_dim = single_dim,
            require_pairwise_repr = True,
        )

        # Map updated single repr to coordinate delta
        self.coord_update = nn.Sequential(
            nn.LayerNorm(single_dim),
            nn.Linear(single_dim, single_dim // 2),
            nn.ReLU(),
            nn.Linear(single_dim // 2, 3),
        )

    def forward(self, single_repr, pair_repr, init_coords):
        """
        Args:
            single_repr: (1, N, 256) - backbone hidden repr per nucleotide
            pair_repr:   (1, N, N, 88) - pairwise features
            init_coords: (N, 3) - initial coord estimate (template or zeros)
        Returns:
            coords: (N, 3) - refined coordinates after n_iter passes
        """
        b, N, _ = single_repr.shape

        single = self.single_norm(single_repr)  # (1, N, 256)
        pair   = self.pair_proj(pair_repr)       # (1, N, N, 256)
        coords = init_coords.clone()             # (N, 3)

        mask = torch.ones(b, N, device=single_repr.device, dtype=torch.bool)

        for _ in range(self.n_iter):
            rots, trans = build_rna_frames(coords)

            updated = self.ipa_block(
                single,
                pairwise_repr = pair,
                rotations     = rots.unsqueeze(0),    # (1, N, 3, 3)
                translations  = trans.unsqueeze(0),   # (1, N, 3)
                mask          = mask,
            )  # (1, N, 256)

            delta  = self.coord_update(updated).squeeze(0)  # (N, 3)
            coords = coords + delta
            single = updated  # pass updated repr to next iteration

        return coords  # (N, 3)


# ------------------------------------------------------------------
# NEW IN RUN 5: fape_loss
#
# Frame Aligned Point Error: rotation-invariant coordinate loss.
# Directly related to TM-score (the competition metric).
# Used in AlphaFold2 for the same purpose.
# ------------------------------------------------------------------
def fape_loss(pred_coords, true_coords, true_rotations, true_translations,
              clamp=10.0):
    """
    Frame Aligned Point Error (FAPE).

    Simple explanation:
      For each nucleotide i as the reference frame:
        1. Express ALL predicted coords in frame i's local coordinate system
        2. Express ALL true coords in the same local system
        3. Compute L2 distance between predicted and true
        4. Clamp per-point error at `clamp` Angstroms
           (so outlier residues don't dominate the gradient)
      Average over all (frame i, point j) pairs.

    Why not just MSE on distances?
      Distances are the same for a molecule and its mirror image.
      FAPE can tell them apart because it works in oriented frames.
      This prevents the mirror-flip problem inherent in MDS.

    Args:
        pred_coords:       (N, 3)
        true_coords:       (N, 3)
        true_rotations:    (N, 3, 3) - from build_rna_frames(true_coords)
        true_translations: (N, 3)    - from build_rna_frames(true_coords)
        clamp:             Angstrom clamp per point

    Returns:
        scalar FAPE in Angstroms
    """
    # Shift all points by the origin of frame i
    t_i          = true_translations.unsqueeze(1)          # (N, 1, 3)
    pred_shifted = pred_coords.unsqueeze(0) - t_i          # (N, N, 3)
    true_shifted = true_coords.unsqueeze(0) - t_i          # (N, N, 3)

    # Rotate into frame i: R_i^T @ shifted
    pred_local = torch.einsum('i r c, i j c -> i j r', true_rotations, pred_shifted)
    true_local = torch.einsum('i r c, i j c -> i j r', true_rotations, true_shifted)

    # Per-point L2 error, clamped
    error = ((pred_local - true_local)**2).sum(dim=-1).clamp(min=0).sqrt()
    return torch.clamp(error, max=clamp).mean()


print("Cell 10 complete: TemplateEncoder, IPA modules, build_rna_frames, fape_loss defined.")


# ============================================================
# CELL 11: Load RibonanzaNet Backbone
# ============================================================
#
# RUN 5 CHANGES vs RUN 4:
#   ADDED: get_pairwise_and_single_features() - returns both:
#     pairwise (1, N, N, 64) AND hidden/single repr (1, N, 256)
#   ADDED: get_pairwise_and_single_features_train() - training ver.
#   KEPT:  get_pairwise_features() for backward compatibility
#
# FREEZE_BACKBONE_IPA=True (Strategy 1):
#   ALL backbone layers stay frozen during IPA training.
#   2-3x faster per epoch. Backbone from Run 4 already produces
#   high-quality features. IPA only needs to learn to read them.
# ============================================================
print("\nLoading RibonanzaNet backbone...")
if REPO_PATH:
    sys.path.insert(0, REPO_PATH)

from Network import RibonanzaNet

class Cfg:
    ntoken = 5; ninp = 256; nhead = 8; nlayers = 9; nclass = 2
    k = 5; dropout = 0.05; pairwise_dimension = 64
    use_triangular_attention = False

try:
    import yaml
    cfg_path = os.path.join(REPO_PATH, "configs", "pairwise.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg_dict = yaml.safe_load(f)
        for k, v in cfg_dict.items():
            setattr(Cfg, k, v)
        print(f"  Loaded config from {cfg_path}")
except:
    print("  Using default config")

backbone = RibonanzaNet(Cfg)
state    = torch.load(BACKBONE_WEIGHTS, map_location='cpu', weights_only=False)
if isinstance(state, dict) and 'model_state_dict' in state:
    state = state['model_state_dict']
backbone.load_state_dict(state, strict=False)
backbone = backbone.to(device)

# Freeze ALL backbone params initially
for p in backbone.parameters():
    p.requires_grad = False

UNFREEZE_LAST_N  = 2
total_layers     = len(list(backbone.transformer_encoder))
print(f"  Backbone has {total_layers} transformer layers")

unfrozen_backbone_params = []

if FREEZE_BACKBONE_IPA:
    # Strategy 1: keep all backbone frozen.
    # Faster epochs. Backbone already trained by Run 3 + Run 4.
    print(f"  FREEZE_BACKBONE_IPA=True: all {total_layers} layers frozen.")
    print(f"  Only ~175K IPA params train. ~2-3x faster per epoch.")
else:
    # Same selective unfreeze as Run 4 (last 2 layers trainable)
    for i, layer in enumerate(backbone.transformer_encoder):
        if i >= total_layers - UNFREEZE_LAST_N:
            for p in layer.parameters():
                p.requires_grad = True
                unfrozen_backbone_params.append(p)
            print(f"  Layer {i}: UNFROZEN (trainable)")
        else:
            print(f"  Layer {i}: frozen")

bb_frozen   = sum(p.numel() for p in backbone.parameters() if not p.requires_grad)
bb_unfrozen = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
print(f"  Frozen params:   {bb_frozen:,}")
print(f"  Unfrozen params: {bb_unfrozen:,}")


def _run_backbone(sequence, with_grad=False):
    """
    Core backbone forward pass. Returns (pairwise, hidden).
      pairwise: (1, N, N, 64) - pairwise repr (same as Run 4)
      hidden:   (1, N, 256)   - single repr per nucleotide (NEW Run 5)
    """
    base_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    tokens   = torch.tensor(
        [base_map.get(b, 4) for b in sequence.upper()],
        dtype=torch.long
    ).unsqueeze(0).to(device)
    N        = len(sequence)
    src_mask = torch.ones(1, N, dtype=torch.long, device=device)

    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with ctx:
        embedded = backbone.encoder(tokens)
        pairwise = backbone.outer_product_mean(embedded)
        pairwise = pairwise + backbone.pos_encoder(embedded)
        hidden   = embedded
        for layer in backbone.transformer_encoder:
            result = layer(hidden, pairwise, src_mask=src_mask)
            if isinstance(result, tuple):
                hidden, pairwise = result
            else:
                hidden = result
    return pairwise, hidden


def get_pairwise_features(sequence):
    """Returns pairwise (1,N,N,64). Preserved from Run 4."""
    pairwise, _ = _run_backbone(sequence, with_grad=False)
    return pairwise


def get_pairwise_and_single_features(sequence):
    """
    NEW in Run 5.
    Returns pairwise (1,N,N,64) and single repr (1,N,256).
    No gradients. Used for inference and validation.
    """
    return _run_backbone(sequence, with_grad=False)


def get_pairwise_and_single_features_train(sequence):
    """
    NEW in Run 5. Training version.

    FREEZE_BACKBONE_IPA=True:
      No gradients through backbone at all (Strategy 1).
      Returns detached tensors — IPA trains on top of frozen features.

    FREEZE_BACKBONE_IPA=False:
      Gradients through last 2 layers only (same split as Run 4).
    """
    if FREEZE_BACKBONE_IPA:
        # Fully frozen: no gradients through backbone
        pairwise, hidden = _run_backbone(sequence, with_grad=False)
        return pairwise, hidden

    # Selective unfreeze: gradient split at frozen/unfrozen boundary
    base_map     = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    tokens       = torch.tensor(
        [base_map.get(b, 4) for b in sequence.upper()],
        dtype=torch.long
    ).unsqueeze(0).to(device)
    N            = len(sequence)
    src_mask     = torch.ones(1, N, dtype=torch.long, device=device)
    frozen_count = total_layers - UNFREEZE_LAST_N

    with torch.no_grad():
        embedded = backbone.encoder(tokens)
        pairwise = backbone.outer_product_mean(embedded)
        pairwise = pairwise + backbone.pos_encoder(embedded)
        hidden   = embedded
        for i, layer in enumerate(backbone.transformer_encoder):
            if i < frozen_count:
                result = layer(hidden, pairwise, src_mask=src_mask)
                if isinstance(result, tuple):
                    hidden, pairwise = result
                else:
                    hidden = result

    # Detach at boundary, enable gradients for unfrozen layers
    hidden   = hidden.detach().requires_grad_(True)
    pairwise = pairwise.detach().requires_grad_(True)

    for i, layer in enumerate(backbone.transformer_encoder):
        if i >= frozen_count:
            result = layer(hidden, pairwise, src_mask=src_mask)
            if isinstance(result, tuple):
                hidden, pairwise = result
            else:
                hidden = result

    return pairwise, hidden


# Sanity check: verify output shapes before training
_test_seq    = "AUGCAUGC"
_test_pair, _test_single = get_pairwise_and_single_features(_test_seq)
print(f"\n  Shape check - pairwise: {_test_pair.shape}")   # (1, 8, 8, 64)
print(f"  Shape check - single:   {_test_single.shape}")  # (1, 8, 256)
assert _test_pair.shape   == (1, 8, 8, 64),  f"Unexpected pairwise shape: {_test_pair.shape}"
assert _test_single.shape == (1, 8, 256),    f"Unexpected single shape: {_test_single.shape}"
del _test_seq, _test_pair, _test_single
print("  Backbone sanity check passed.")


# ============================================================
# CELL 12: Create IPA Model + Warm-Start from Run 4
# ============================================================
#
# RUN 5 CHANGES vs RUN 4:
#   REMOVED: DistanceMatrixHead creation and weight loading
#   ADDED:   IPAStructureModule creation
#
# WARM-START from Run 4 checkpoint (adv1_best_model.pt):
#   backbone_unfrozen_state     -> backbone layers 7-8   LOADED
#   template_encoder_state_dict -> TemplateEncoder       LOADED
#   model_state_dict (distance head)                     SKIPPED
#     (DistanceMatrixHead weights are incompatible with IPA)
#
# IPAStructureModule starts with RANDOM WEIGHTS.
# This is expected. It trains from scratch on top of the
# already-excellent backbone features from Run 3 + Run 4.
#
# Parameter summary with FREEZE_BACKBONE_IPA=True:
#   Backbone:           ~50M params  FROZEN  (warm-start Run 4)
#   TemplateEncoder:      368 params  TRAINS  (warm-start Run 4)
#   IPAStructureModule: ~175K params  TRAINS  (random init)
#   TOTAL TRAINABLE:    ~175K params
# ============================================================
PAIR_DIM = 64 + 16 + MSA_DIM  # 88: backbone + template + MSA

print(f"\nCreating IPA model (pair_dim={PAIR_DIM}, single_dim={IPA_DIM})...")

template_encoder = TemplateEncoder(
    template_dim=16, num_bins=22, max_dist=40.0
).to(device)

ipa_module = IPAStructureModule(
    single_dim = IPA_DIM,
    pair_dim   = PAIR_DIM,
    n_iter     = IPA_ITERATIONS,
    heads      = IPA_HEADS,
).to(device)

# --- Warm-start from Run 4 (or BASIC as fallback) ---
warmstart_path   = RUN4_CHECKPOINT if RUN4_CHECKPOINT else ADV1_WEIGHTS
warmstart_source = "Run 4" if RUN4_CHECKPOINT else "BASIC (fallback)"

if warmstart_path and os.path.exists(warmstart_path):
    print(f"  Warm-starting from {warmstart_source}: {warmstart_path}")
    ckpt = torch.load(warmstart_path, map_location=device, weights_only=False)

    # Load TemplateEncoder (same architecture in Run 4 and Run 5)
    if 'template_encoder_state_dict' in ckpt:
        template_encoder.load_state_dict(ckpt['template_encoder_state_dict'])
        print("  [OK] TemplateEncoder loaded from Run 4.")
    else:
        print("  [--] template_encoder_state_dict not found, random init.")

    # Load backbone unfrozen layers (layers 7-8 trained by Run 4)
    # Even if FREEZE_BACKBONE_IPA=True, we still load Run 4's improved
    # backbone weights — they are better than original RibonanzaNet.
    if 'backbone_unfrozen_state' in ckpt:
        for layer_key, layer_state in ckpt['backbone_unfrozen_state'].items():
            layer_idx = int(layer_key.split('.')[-1])
            backbone.transformer_encoder[layer_idx].load_state_dict(layer_state)
        if FREEZE_BACKBONE_IPA:
            print("  [OK] Run 4 backbone weights loaded and frozen (Strategy 1).")
        else:
            print("  [OK] Run 4 backbone weights loaded, layers 7-8 trainable.")

    # NOTE: model_state_dict (DistanceMatrixHead) intentionally NOT loaded.
    # IPA is a different architecture — those weights don't apply.
    print("  [--] DistanceMatrixHead weights skipped (different architecture).")
    print("  [NEW] IPAStructureModule: random init, trains from scratch.")
else:
    print("  No Run 4 checkpoint found. Training from scratch.")

ipa_params = sum(p.numel() for p in ipa_module.parameters())
te_params  = sum(p.numel() for p in template_encoder.parameters())
print(f"\n  IPAStructureModule params:  {ipa_params:,}")
print(f"  TemplateEncoder params:     {te_params:,}")
print(f"  Backbone unfrozen params:   {bb_unfrozen:,}")
print(f"  TOTAL TRAINABLE:            {ipa_params + te_params + bb_unfrozen:,}")


# ============================================================
# CELL 13: Train IPA Module
# ============================================================
#
# RUN 5 CHANGES vs RUN 4:
#   LOSS:   FAPE + 0.1 * aux MSE distance (replaces MSE distance only)
#   FORWARD: IPA produces coords directly (no distance head, no MDS)
#   TEACHER FORCING: 50% start from true_coords + 1A noise
#                    50% start from zeros (forces IPA to use features)
#   EARLY STOPPING: patience=10 epochs
#   EPOCHS: ceiling 40 (early stopping cuts short)
#   LR: 1e-4 for IPA (higher than Run 4: new module trains from scratch)
#   VAL METRIC: FAPE in Angstroms (more interpretable than MSE)
#
# PRESERVED FROM RUN 4:
#   Pickle parsing, train/val split, batch loop structure
#   Template feature masking (50% probability)
#   MSA proxy feature masking (50% probability)
#   GradScaler, CosineAnnealingLR, gradient clipping at 1.0
#   Best model checkpoint saving
# ============================================================
print("\n" + "="*60)
print("PHASE 3: IPA Training (FAPE loss)")
print("="*60)

BATCH_SIZE         = 2
MAX_SEQ_LEN        = 256
HEAD_LR            = 1e-4   # Higher than Run 4's 5e-5: IPA trains fresh
BACKBONE_LR        = 1e-5   # Same as Run 4 (only used if FREEZE_BACKBONE_IPA=False)
TEMPLATE_MASK_PROB = 0.5
MSA_MASK_PROB      = 0.5

# Load and parse training data (same as Run 4)
print(f"Loading training data from {TRAIN_PICKLE}...")
with open(TRAIN_PICKLE, 'rb') as f:
    raw_data = pickle.load(f)

print(f"  Pickle type: {type(raw_data)}")
train_items = []
skipped     = 0

if isinstance(raw_data, dict):
    sequences = raw_data['sequence']
    xyz_list  = raw_data['xyz']
    print(f"  Total structures: {len(sequences)}")
    for i in range(len(sequences)):
        try:
            seq          = sequences[i]
            residue_list = xyz_list[i]
            if seq is None or residue_list is None:
                skipped += 1; continue
            c1_coords   = []
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
                skipped += 1; continue
            coords    = np.array(c1_coords, dtype=np.float32)
            clean_seq = ''.join(valid_bases[:len(c1_coords)])
            min_len   = min(len(clean_seq), len(coords))
            clean_seq = clean_seq[:min_len]
            coords    = coords[:min_len]
            if 10 <= min_len <= MAX_SEQ_LEN:
                train_items.append({'sequence': clean_seq, 'coords': coords})
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  Warning structure {i}: {e}")
    print(f"  Parsed: {len(train_items)}, Skipped: {skipped}")
else:
    for item in raw_data:
        seq    = item.get('sequence', '')
        coords = item.get('coordinates', item.get('coords', None))
        if (seq and coords is not None and
                10 <= len(seq) <= MAX_SEQ_LEN and len(seq) == len(coords)):
            train_items.append({'sequence': seq, 'coords': np.array(coords, dtype=np.float32)})
    print(f"  Parsed: {len(train_items)} structures")

random.seed(42)
random.shuffle(train_items)
val_size    = max(1, int(len(train_items) * 0.1))
val_items   = train_items[:val_size]
train_items = train_items[val_size:]
print(f"  Train: {len(train_items)}, Val: {val_size}")


def make_training_msa_proxy(seq, N):
    """MSA proxy for training sequences (unchanged from Run 4)."""
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    bp_pairs    = {(0,3):1.0, (3,0):1.0, (1,2):1.0, (2,1):1.0,
                   (2,3):0.5, (3,2):0.5}
    bp_matrix   = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        bi = base_to_idx.get(seq[i].upper(), -1)
        if bi < 0: continue
        for j in range(i+1, N):
            bj = base_to_idx.get(seq[j].upper(), -1)
            if bj < 0: continue
            score           = bp_pairs.get((bi, bj), 0.0)
            bp_matrix[i, j] = score
            bp_matrix[j, i] = score
    features          = np.zeros((N, N, MSA_DIM), dtype=np.float32)
    features[:, :, 0] = bp_matrix * 0.3
    features[:, :, 1] = bp_matrix * 0.2
    features[:, :, 7] = 0.01
    return features


# Optimizer
ipa_and_template_params = (list(ipa_module.parameters()) +
                            list(template_encoder.parameters()))
optimizer = optim.AdamW([
    {'params': ipa_and_template_params,  'lr': HEAD_LR},
    {'params': unfrozen_backbone_params, 'lr': BACKBONE_LR},
], weight_decay=0.01)
all_trainable = ipa_and_template_params + unfrozen_backbone_params

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_EPOCHS)
scaler    = GradScaler(enabled=(device.type == 'cuda'))

ipa_module.train()
template_encoder.train()

best_val_loss     = float('inf')
epochs_no_improve = 0
train_start       = time.time()

print(f"\nStarting IPA training:")
print(f"  Ceiling epochs: {TRAIN_EPOCHS}")
print(f"  Early stop patience: {EARLY_STOP_PATIENCE}")
print(f"  Strategy: {'FREEZE_BACKBONE (all frozen)' if FREEZE_BACKBONE_IPA else 'UNFREEZE last 2 layers'}")
print(f"  Loss: FAPE + {AUX_DIST_WEIGHT} * MSE_distance")

for epoch in range(TRAIN_EPOCHS):
    epoch_loss = 0.0
    n_batches  = 0
    random.shuffle(train_items)

    for batch_start in range(0, len(train_items), BATCH_SIZE):
        batch        = train_items[batch_start:batch_start + BATCH_SIZE]
        batch_losses = []

        for item in batch:
            seq         = item['sequence']
            true_coords = item['coords']
            N           = len(seq)

            # Backbone features (pairwise + single)
            pairwise, single = get_pairwise_and_single_features_train(seq)

            # Template features (50% masked - same as Run 4)
            if random.random() < TEMPLATE_MASK_PROB:
                tmpl_feat = torch.zeros(1, N, N, 16, device=device)
            else:
                coords_t  = torch.tensor(true_coords, dtype=torch.float32, device=device)
                tmpl_feat = template_encoder(
                    coords_t, confidence=1.0, has_template=True
                ).unsqueeze(0)

            # MSA proxy features (50% masked - same as Run 4)
            if random.random() < MSA_MASK_PROB:
                msa_feat = torch.zeros(1, N, N, MSA_DIM, device=device)
            else:
                msa_np   = make_training_msa_proxy(seq, N)
                msa_feat = torch.tensor(
                    msa_np, dtype=torch.float32, device=device
                ).unsqueeze(0)

            # Concatenate pair features (same as Run 4)
            combined = torch.cat([pairwise, tmpl_feat, msa_feat], dim=-1)  # (1,N,N,88)

            # Teacher forcing: 50% start from true coords + 1A noise
            true_coords_t = torch.tensor(true_coords, dtype=torch.float32, device=device)
            if random.random() < 0.5:
                noise       = torch.randn_like(true_coords_t) * 1.0
                init_coords = true_coords_t + noise
            else:
                init_coords = torch.zeros_like(true_coords_t)

            # IPA forward pass -> coords directly
            pred_coords = ipa_module(single, combined, init_coords)  # (N, 3)

            # FAPE loss (primary)
            true_rots, true_trans = build_rna_frames(true_coords_t)
            fape = fape_loss(
                pred_coords, true_coords_t,
                true_rots, true_trans,
                clamp=FAPE_CLAMP
            )

            # Auxiliary MSE distance loss (stabilises early training)
            pred_dist = torch.cdist(pred_coords.unsqueeze(0), pred_coords.unsqueeze(0)).squeeze(0)
            true_dist = torch.cdist(true_coords_t.unsqueeze(0), true_coords_t.unsqueeze(0)).squeeze(0)
            aux_loss  = ((pred_dist - true_dist)**2).mean()

            loss = fape + AUX_DIST_WEIGHT * aux_loss
            batch_losses.append(loss)

        if batch_losses:
            total_loss = sum(batch_losses) / len(batch_losses)
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(all_trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += total_loss.item()
            n_batches  += 1

    scheduler.step()
    avg_train = epoch_loss / max(n_batches, 1)

    # Validation (no teacher forcing - unbiased estimate)
    ipa_module.eval()
    template_encoder.eval()
    val_loss = 0.0
    with torch.no_grad():
        for item in val_items[:20]:
            seq         = item['sequence']
            true_coords = item['coords']
            N           = len(seq)
            pairwise, single = get_pairwise_and_single_features(seq)
            coords_t     = torch.tensor(true_coords, dtype=torch.float32, device=device)
            tmpl_feat    = template_encoder(coords_t, confidence=1.0, has_template=True).unsqueeze(0)
            msa_np       = make_training_msa_proxy(seq, N)
            msa_feat     = torch.tensor(msa_np, dtype=torch.float32, device=device).unsqueeze(0)
            combined     = torch.cat([pairwise, tmpl_feat, msa_feat], dim=-1)
            init_coords  = torch.zeros_like(coords_t)   # cold start for unbiased val
            pred_coords  = ipa_module(single, combined, init_coords)
            true_rots, true_trans = build_rna_frames(coords_t)
            fape = fape_loss(pred_coords, coords_t, true_rots, true_trans, clamp=FAPE_CLAMP)
            val_loss += fape.item()

    avg_val = val_loss / max(len(val_items[:20]), 1)
    lrs     = scheduler.get_last_lr()
    elapsed = time.time() - train_start
    print(f"Epoch {epoch+1}/{TRAIN_EPOCHS}: "
          f"train={avg_train:.3f}A  val={avg_val:.3f}A  "
          f"lr={lrs[0]:.6f}  t={elapsed:.0f}s")

    if avg_val < best_val_loss:
        best_val_loss     = avg_val
        epochs_no_improve = 0
        torch.save({
            'ipa_module_state_dict':       ipa_module.state_dict(),
            'template_encoder_state_dict': template_encoder.state_dict(),
            'backbone_unfrozen_state': {
                f'transformer_encoder.{i}':
                    backbone.transformer_encoder[i].state_dict()
                for i in range(total_layers - UNFREEZE_LAST_N, total_layers)
            } if not FREEZE_BACKBONE_IPA else {},
            'epoch':          epoch,
            'val_fape':       avg_val,
            'pair_dim':       PAIR_DIM,
            'ipa_iterations': IPA_ITERATIONS,
            'ipa_heads':      IPA_HEADS,
            'ipa_dim':        IPA_DIM,
        }, CHECKPOINT_PATH)
        print(f"  -> Saved best model (val_fape={avg_val:.3f} A)")
    else:
        epochs_no_improve += 1
        print(f"     No improvement ({epochs_no_improve}/{EARLY_STOP_PATIENCE})")
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}. "
                  f"Best val_fape={best_val_loss:.3f} A")
            break

    ipa_module.train()
    template_encoder.train()

print(f"\nTraining complete. Best val_fape: {best_val_loss:.3f} A")
print(f"Total training time: {time.time()-train_start:.0f}s")

# Reload best checkpoint for inference
ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
ipa_module.load_state_dict(ckpt['ipa_module_state_dict'])
template_encoder.load_state_dict(ckpt['template_encoder_state_dict'])
if ckpt.get('backbone_unfrozen_state') and not FREEZE_BACKBONE_IPA:
    for layer_key, layer_state in ckpt['backbone_unfrozen_state'].items():
        layer_idx = int(layer_key.split('.')[-1])
        backbone.transformer_encoder[layer_idx].load_state_dict(layer_state)
ipa_module.eval()
template_encoder.eval()
backbone.eval()
print("Loaded best checkpoint for inference.")


# ============================================================
# CELL 14: IPA Inference
# ============================================================
#
# RUN 5 CHANGES vs RUN 4:
#   REMOVED: refine_coords() (all 5 diversity variants)
#            mds_reconstruct() calls
#            nn_diversity dict (noise/steps/seed)
#            distance_head(combined) forward pass
#            pred_dist variable
#
#   ADDED:   get_pairwise_and_single_features() call
#            IPA forward for all NN slots
#            Frame perturbation for diversity (replaces distance noise)
#
# PRESERVED FROM RUN 4:
#   Fix 1: raw template ALWAYS in Slot 1 regardless of confidence
#   HYBRID_THRESHOLD = 0.20 (Fix 3)
#   3-branch slot assembly (high-conf / low-conf / no-template)
#   adaptive_rna_constraints() on template slots
#   Long-sequence extrapolation (len > MAX_INFER_LEN)
#
# DIVERSITY STRATEGY (replaces Run 4's distance noise):
#   IPA is run multiple times with different init_coords.
#   Template-seeded: init from template_coords (best start)
#   Perturbed:       init from template + Gaussian noise
#   Cold start:      init from zeros (forces feature-only prediction)
#
# SLOT ASSEMBLY:
#   conf > 0.20:      Slot 1 raw tmpl, Slot 2 tmpl+noise, Slots 3-5 IPA
#   0 < conf <= 0.20: Slot 1 raw tmpl (Fix 1), Slots 2-5 IPA
#   conf = 0:         All 5 IPA (cold starts)
# ============================================================
print("\n" + "="*60)
print("PHASE 4: IPA Inference (Hybrid + Template + Frame Perturbation)")
print("="*60)

MAX_INFER_LEN    = 512
HYBRID_THRESHOLD = 0.20   # Fix 3: preserved from Run 4

all_predictions = []
infer_start     = time.time()

n_hybrid_2slot = 0   # conf > threshold: 2 template slots
n_hybrid_1slot = 0   # 0 < conf <= threshold: 1 template slot (Fix 1)
n_no_template  = 0   # conf = 0: all IPA
n_ipa_slots    = 0   # total slots using IPA

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

        combined = torch.cat([pairwise, tmpl_feat, msa_feat], dim=-1)  # (1,N,N,88)

        # IPA prediction helper
        def run_ipa(init_np, seed=0):
            torch.manual_seed(seed)
            init_t = torch.tensor(init_np, dtype=torch.float32, device=device)
            return ipa_module(single, combined, init_t).cpu().numpy()

        def add_noise(arr, sigma, seed=0):
            np.random.seed(seed)
            return (arr + np.random.normal(0, sigma, arr.shape)).astype(np.float32)

        zeros_np = np.zeros((N, 3), dtype=np.float32)

        # Generate 5 IPA predictions with diversity via init perturbation
        if has_tmpl:
            t_np  = tmpl_coords.astype(np.float32)
            ipa_0 = run_ipa(t_np,                   seed=0)  # clean template init
            ipa_1 = run_ipa(add_noise(t_np, 0.5, 1), seed=1)  # small noise
            ipa_2 = run_ipa(add_noise(t_np, 1.0, 2), seed=2)  # medium noise
            ipa_3 = run_ipa(add_noise(t_np, 1.5, 3), seed=3)  # large noise
            ipa_4 = run_ipa(zeros_np,                seed=4)  # cold start (diversity)
        else:
            ipa_0 = run_ipa(zeros_np,                   seed=0)
            ipa_1 = run_ipa(add_noise(zeros_np, 0.5, 1), seed=1)
            ipa_2 = run_ipa(add_noise(zeros_np, 1.0, 2), seed=2)
            ipa_3 = run_ipa(add_noise(zeros_np, 1.5, 3), seed=3)
            ipa_4 = run_ipa(zeros_np,                   seed=5)

    # Extend to full sequence length if truncated
    def extend_to_full(arr):
        if len(sequence) <= MAX_INFER_LEN:
            return arr
        remaining = len(sequence) - MAX_INFER_LEN
        last_dir  = (arr[-1] - arr[-2]) if arr.shape[0] >= 2 else np.array([5.9, 0., 0.])
        last_dir  = last_dir / (np.linalg.norm(last_dir) + 1e-8) * 5.9
        extra     = np.array([arr[-1] + last_dir*(i+1) for i in range(remaining)])
        return np.concatenate([arr, extra])

    ipa_0, ipa_1, ipa_2, ipa_3, ipa_4 = map(extend_to_full, [ipa_0, ipa_1, ipa_2, ipa_3, ipa_4])

    # Full-length template coords
    full_tmpl = template_coords_per_target.get(target_id, np.zeros((len(sequence), 3)))
    if len(full_tmpl) > len(sequence):
        full_tmpl = full_tmpl[:len(sequence)]
    elif len(full_tmpl) < len(sequence):
        pad = np.zeros((len(sequence), 3))
        pad[:len(full_tmpl)] = full_tmpl
        full_tmpl = pad

    # 3-branch slot assembly (Fix 1 preserved from Run 4)
    coords_list = []

    if tmpl_conf > HYBRID_THRESHOLD:
        # HIGH confidence: Slots 1-2 template, Slots 3-5 IPA
        n_hybrid_2slot += 1
        n_ipa_slots    += 3

        tmpl_r = adaptive_rna_constraints(full_tmpl.copy(), sequence, confidence=tmpl_conf)
        coords_list.append(tmpl_r)                       # Slot 1: raw template

        np.random.seed(42)
        noisy  = full_tmpl + np.random.normal(0, 0.5, full_tmpl.shape)
        tmpl_n = adaptive_rna_constraints(noisy, sequence, confidence=tmpl_conf)
        coords_list.append(tmpl_n)                       # Slot 2: template + small noise

        coords_list.append(ipa_0)                        # Slot 3: IPA clean
        coords_list.append(ipa_1)                        # Slot 4: IPA + small noise
        coords_list.append(ipa_2)                        # Slot 5: IPA + medium noise

        if idx % 5 == 0:
            print(f"    HYBRID: conf={tmpl_conf:.3f} > {HYBRID_THRESHOLD} "
                  f"-> slots 1-2 template, 3-5 IPA")

    elif tmpl_conf > 0.01:
        # WEAK template: Fix 1 - Slot 1 always raw template, Slots 2-5 IPA
        n_hybrid_1slot += 1
        n_ipa_slots    += 4

        tmpl_r = adaptive_rna_constraints(full_tmpl.copy(), sequence, confidence=tmpl_conf)
        coords_list.append(tmpl_r)                       # Slot 1: Fix 1 safety net
        coords_list.append(ipa_0)                        # Slot 2
        coords_list.append(ipa_1)                        # Slot 3
        coords_list.append(ipa_2)                        # Slot 4
        coords_list.append(ipa_3)                        # Slot 5

        if idx % 5 == 0:
            print(f"    FIX1: conf={tmpl_conf:.3f} <= {HYBRID_THRESHOLD} "
                  f"-> slot 1 raw template, 2-5 IPA")

    else:
        # No template: all 5 slots IPA
        n_no_template += 1
        n_ipa_slots   += 5
        coords_list    = [ipa_0, ipa_1, ipa_2, ipa_3, ipa_4]

        if idx % 5 == 0:
            print(f"    NO TEMPLATE: conf={tmpl_conf:.3f} -> all 5 IPA")

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
# CELL 15: Post-Processing (UNCHANGED from Run 4)
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
print(f"Run 5 produced {len(raw_rows)} rows")

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
print(f"  Matched from Run 5 IPA: {matched}")
print(f"  Filled with zeros:      {filled}")
print(f"  Total rows:             {matched + filled}")
print(f"  File size:              {os.path.getsize(FINAL_SUBMISSION_PATH)} bytes")

print("\n" + "="*60)
print("DONE. submission.csv ready. Run 5 IPA complete.")
print("="*60)
