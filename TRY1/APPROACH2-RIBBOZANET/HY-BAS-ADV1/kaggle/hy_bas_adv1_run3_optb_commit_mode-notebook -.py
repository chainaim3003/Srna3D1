# ============================================================
# HY-BAS-ADV1 RUN 3 Option B: All-in-One Kaggle Notebook
# ============================================================
#
# WHAT THIS IS:
#   Run 3 Option B = Run 3 + template-seeded refinement.
#   Instead of using MDS (which destroys coordinate quality),
#   the NN slots start from template coordinates and refine
#   toward the NN's predicted distances using gradient descent.
#
# WHY OPTION B:
#   Run 2 scored 0.109 despite correct training (val_loss 171).
#   The bottleneck is MDS reconstruction, not training quality.
#   MDS takes noisy predicted distances and produces garbage
#   coordinates. Template-seeded refinement avoids MDS entirely
#   by starting from real experimental coordinates (templates)
#   and nudging them toward the NN's distance predictions.
#
# CHANGES FROM RUN 3 (Option A):
#   Change 6: Template-seeded refinement (Cell 14)
#             - NN slots start from template coords, NOT MDS output
#             - MDS is kept ONLY as fallback when no template exists
#             - refine_coords() adjusts template toward NN distances
#             - This is ~10 lines changed in Cell 14's NN branch
#
# CHANGES FROM RUN 2 (carried forward):
#   Change 2: Unfreeze last 2 backbone layers (Cell 11)
#             - Selective unfreezing of layers 7,8 (of 9)
#             - get_pairwise_features_train() with gradients
#             - BATCH_SIZE = 2 (was 4) for VRAM
#             - Discriminative LR: backbone 1e-5, head 5e-5
#   Change 5: Hybrid inference (Cell 14)
#             - High-confidence templates (>0.3): slots 1-2 template,
#               slots 3-5 NN (template-seeded refinement in Option B)
#             - Low-confidence templates: all 5 slots NN
#
# PRESERVED FROM RUN 2:
#   Change 1: TRAIN_EPOCHS = 30
#   Change 3: Biopython offline wheel install
#   Change 4: Fixed pickle parsing (dict of parallel lists)
#
# ESTIMATED RUNTIME: ~2.5 hours on T4 GPU (same as Run 3 Option A)
# ============================================================


# ============================================================
# CELL 0: USER-CONFIGURABLE PATHS (OPTIONAL OVERRIDES)
# ============================================================
# Each variable below has a sensible default or is auto-discovered.
# You only need to change something if auto-discovery fails or you
# want to point to a specific location.
#
# PATTERN: Set to None = auto-discover. Set to a string = use that path.
# Each user runs in their own Kaggle notebook with their own
# /kaggle/working, so there are no cross-user conflicts.
#
# HOW TO FIND YOUR PATHS (if needed): Run this in a Kaggle cell:
#   import os
#   for root, dirs, files in os.walk("/kaggle/input"):
#       for f in files:
#           if f in ('Network.py', 'RibonanzaNet.pt', 'best_model.pt',
#                    'pdb_xyz_data.pkl', 'sample_submission.csv',
#                    'rna_sequences.csv', 'rna_coordinates.csv'):
#               print(f"{f}: {os.path.join(root, f)}")
#       for f in files:
#           if f.endswith('.whl'):
#               print(f"WHEEL: {os.path.join(root, f)}")
# ============================================================

# --- Extended RNA data (rna_sequences.csv + rna_coordinates.csv) ---
# Set to None to auto-discover, or set explicit paths if needed.
# Example override: '/kaggle/input/datasets/yourusername/your-dataset-slug'
EXTENDED_SEQ_CSV = None       # Auto-finds rna_sequences.csv
EXTENDED_COORD_CSV = None     # Auto-finds rna_coordinates.csv

# --- Output / working directory ---
# Default: /kaggle/working (standard Kaggle output dir)
# Each user's notebook has its own /kaggle/working — no conflicts.
OUTPUT_DIR = '/kaggle/working'

# --- Competition data base path ---
# Auto-detected. Override only if your competition input is mounted differently.
COMP_BASE_PRIMARY = '/kaggle/input/competitions/stanford-rna-3d-folding-2/'
COMP_BASE_FALLBACK = '/kaggle/input/stanford-rna-3d-folding-2/'


# ============================================================
# CELL 1: Install biopython (direct sys.path injection)
# ============================================================
import sys, os, glob

py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"
print(f"Python {sys.version_info.major}.{sys.version_info.minor} (tag: {py_ver})")

try:
    import Bio
    print(f"Biopython {Bio.__version__} already installed")
except ImportError:
    # Kaggle fully extracts .whl files — Bio/ folder already exists
    # Find Bio/__init__.py inside the extracted wheel directories
    bio_inits = glob.glob('/kaggle/input/**/Bio/__init__.py', recursive=True)
    print(f"Found Bio dirs: {bio_inits}")

    # Prefer the one matching our Python version (cp312)
    matching = [p for p in bio_inits if py_ver in p]
    chosen = matching[0] if matching else (bio_inits[0] if bio_inits else None)

    if chosen:
        # Add the folder containing Bio/ to sys.path
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
import os, sys, time, csv, math, random, pickle, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.linalg import eigh
from scipy.spatial import distance_matrix as scipy_dist_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from Bio import pairwise2
from Bio.Seq import Seq
from scipy.spatial.transform import Rotation as R

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    #print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ============================================================
# CELL 3: Find all data paths (auto-discovery)
# ============================================================
# Competition data: tries primary path, falls back to alternative.
# All other files (weights, pickle, repo, extended CSVs) are found
# by scanning /kaggle/input for specific filenames. This means the
# notebook works regardless of dataset slugs — zero configuration
# needed if you attach the right datasets. Override in Cell 0 if needed.
# ============================================================
import os

BASE = COMP_BASE_PRIMARY
if not os.path.exists(BASE):
    BASE = COMP_BASE_FALLBACK
print(f"Competition data: {BASE}")

# Auto-discover RibonanzaNet repo (folder containing Network.py)
REPO_PATH = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "Network.py" in files:
        REPO_PATH = root
        break
print(f"RibonanzaNet repo: {REPO_PATH}")

# Auto-discover weight files
BACKBONE_WEIGHTS = None   # RibonanzaNet.pt (~43MB backbone)
ADV1_WEIGHTS = None       # best_model.pt (~312KB BASIC distance head)
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f == "RibonanzaNet.pt":
            BACKBONE_WEIGHTS = os.path.join(root, f)
        if f == "best_model.pt":
            ADV1_WEIGHTS = os.path.join(root, f)
print(f"Backbone weights: {BACKBONE_WEIGHTS}")
print(f"BASIC distance head: {ADV1_WEIGHTS}")

# Auto-discover training pickle
TRAIN_PICKLE = None       # pdb_xyz_data.pkl (~52MB, 844 structures)
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f == "pdb_xyz_data.pkl":
            TRAIN_PICKLE = os.path.join(root, f)
print(f"Training pickle: {TRAIN_PICKLE}")

# Auto-discover sample submission (from competition data)
SAMPLE_CSV = None
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f == "sample_submission.csv":
            SAMPLE_CSV = os.path.join(root, f)
print(f"Sample submission: {SAMPLE_CSV}")

# Auto-discover extended RNA data (rna_sequences.csv, rna_coordinates.csv)
# Only scan if not explicitly set in Cell 0
if EXTENDED_SEQ_CSV is None or EXTENDED_COORD_CSV is None:
    for root, dirs, files in os.walk("/kaggle/input"):
        for f in files:
            if f == "rna_sequences.csv" and EXTENDED_SEQ_CSV is None:
                EXTENDED_SEQ_CSV = os.path.join(root, f)
            if f == "rna_coordinates.csv" and EXTENDED_COORD_CSV is None:
                EXTENDED_COORD_CSV = os.path.join(root, f)
print(f"Extended sequences CSV: {EXTENDED_SEQ_CSV}")
print(f"Extended coordinates CSV: {EXTENDED_COORD_CSV}")

# Derived output paths (from Cell 0's OUTPUT_DIR)
CHECKPOINT_PATH = f'{OUTPUT_DIR}/adv1_best_model.pt'
RAW_SUBMISSION_PATH = f'{OUTPUT_DIR}/submission_raw.csv'
FINAL_SUBMISSION_PATH = f'{OUTPUT_DIR}/submission.csv'


# ============================================================
# CELL 4: Load competition data
# ============================================================
print("\nLoading competition data...")
train_seqs = pd.read_csv(BASE + 'train_sequences.csv')
test_seqs = pd.read_csv(BASE + 'test_sequences.csv')
train_labels = pd.read_csv(BASE + 'train_labels.csv')
print(f"Train: {len(train_seqs)}, Test: {len(test_seqs)}")


# ============================================================
# CELL 5: Load extended data (rna_cif_to_csv)
# ============================================================
# Paths come from Cell 0 (explicit override) or Cell 3 (auto-discovery).
# This dataset contains ~18K additional RNA structures beyond
# the competition's 5.7K, giving us 24K+ structures for the
# template search in Cell 9.
#
# If auto-discovery failed (both are None), the notebook will
# error here with a clear message.
# ============================================================
if EXTENDED_SEQ_CSV is None or EXTENDED_COORD_CSV is None:
    raise FileNotFoundError(
        "Could not find rna_sequences.csv or rna_coordinates.csv. "
        "Either attach the rna-cif-to-csv dataset to your notebook, "
        "or set EXTENDED_SEQ_CSV and EXTENDED_COORD_CSV in Cell 0."
    )

print(f"Loading extended data...")
print(f"  Sequences: {EXTENDED_SEQ_CSV}")
print(f"  Coordinates: {EXTENDED_COORD_CSV}")
train_seqs_v2 = pd.read_csv(EXTENDED_SEQ_CSV)
train_labels_v2 = pd.read_csv(EXTENDED_COORD_CSV)
print(f"Extended seqs: {len(train_seqs_v2)}, Extended labels: {len(train_labels_v2)}")


# ============================================================
# CELL 6: Extend datasets (merge original + v2)
# ============================================================
# Combines competition data with extended data, deduplicating
# by target_id (for sequences) and ID+resid (for labels).
# ============================================================
def extend_dataset(original_df, v2_df, key_col):
    orig_keys = set(original_df[key_col])
    new_mask = ~v2_df[key_col].isin(orig_keys)
    new_records = v2_df[new_mask].copy()
    extended = pd.concat([original_df, new_records], ignore_index=True)
    print(f"  Original: {len(original_df)} -> Extended: {len(extended)} (+{len(new_records)})")
    return extended

print("Extending sequences...")
train_seqs_extended = extend_dataset(train_seqs, train_seqs_v2, 'target_id')

print("Extending labels...")
train_labels['_key'] = train_labels['ID'] + '_' + train_labels['resid'].astype(str)
train_labels_v2['_key'] = train_labels_v2['ID'] + '_' + train_labels_v2['resid'].astype(str)
orig_keys = set(train_labels['_key'])
new_mask = ~train_labels_v2['_key'].isin(orig_keys)
new_labels = train_labels_v2[new_mask].copy()
train_labels_extended = pd.concat([train_labels, new_labels], ignore_index=True)
train_labels_extended.drop('_key', axis=1, inplace=True, errors='ignore')
train_labels.drop('_key', axis=1, inplace=True, errors='ignore')
train_labels_v2.drop('_key', axis=1, inplace=True, errors='ignore')
print(f"  Labels: {len(train_labels)} -> {len(train_labels_extended)}")


# ============================================================
# CELL 7: Process labels into coordinate dictionary (~12-15 min)
# ============================================================
# Builds a dict: target_id -> numpy array of (N, 3) coordinates.
# Uses x_1, y_1, z_1 columns (first model's coordinates).
# This dict is used by the template search in Cell 9.
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
# CELL 8: Fork 2's template functions
# ============================================================
# These 4 functions implement the template-based structure prediction
# pipeline from Fork 2 (scored 0.287). They:
#   1. find_similar_sequences(): k-mer + alignment search for templates
#   2. adapt_template_to_query(): align template coords to query seq
#   3. adaptive_rna_constraints(): fix bond lengths and steric clashes
#   4. generate_rna_structure(): random structure fallback (no template)
# ============================================================

def adaptive_rna_constraints(coordinates, sequence, confidence=1.0):
    """Refine coordinates to satisfy RNA backbone constraints.
    Fixes consecutive distances to ~6.0A and resolves steric clashes.
    Lower confidence = stronger constraints (more correction applied)."""
    refined_coords = coordinates.copy()
    n_residues = len(sequence)
    constraint_strength = 0.8 * (1.0 - min(confidence, 0.8))
    seq_min_dist, seq_max_dist = 5.5, 6.5

    # Fix consecutive backbone distances
    for i in range(n_residues - 1):
        current_dist = np.linalg.norm(refined_coords[i+1] - refined_coords[i])
        if current_dist < seq_min_dist or current_dist > seq_max_dist:
            target_dist = (seq_min_dist + seq_max_dist) / 2
            direction = refined_coords[i+1] - refined_coords[i]
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            adjustment = (target_dist - current_dist) * constraint_strength
            refined_coords[i+1] = refined_coords[i] + direction * (current_dist + adjustment)

    # Resolve steric clashes (atoms too close)
    dist_mat = scipy_dist_matrix(refined_coords, refined_coords)
    min_allowed = 3.8
    clashes = np.where((dist_mat < min_allowed) & (dist_mat > 0))
    for idx in range(len(clashes[0])):
        i, j = clashes[0][idx], clashes[1][idx]
        if abs(i-j) <= 1 or i >= j:
            continue
        direction = refined_coords[j] - refined_coords[i]
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        adj = (min_allowed - dist_mat[i,j]) * constraint_strength
        refined_coords[i] -= direction * (adj/2)
        refined_coords[j] += direction * (adj/2)
    return refined_coords


def adapt_template_to_query(query_seq, template_seq, template_coords):
    """Map template 3D coordinates onto query sequence via alignment.
    Aligned positions copy coordinates. Gaps are interpolated.
    Returns (N, 3) array matching query sequence length."""
    query_seq_obj = Seq(query_seq)
    template_seq_obj = Seq(template_seq)
    alignments = pairwise2.align.globalms(query_seq_obj, template_seq_obj,
                                          2.9, -1, -10, -0.5, one_alignment_only=True)
    if not alignments:
        return generate_rna_structure(query_seq)

    alignment = alignments[0]
    aligned_query = alignment.seqA
    aligned_template = alignment.seqB

    query_coords = np.full((len(query_seq), 3), np.nan)
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

    # Interpolate gaps between aligned positions
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
                for k, idx in enumerate(range(prev_valid+1, next_valid)):
                    w = (k+1) / gap
                    query_coords[idx] = (1-w)*query_coords[prev_valid] + w*query_coords[next_valid]
            elif prev_valid is not None:
                d = np.array([1,0,0]) if prev_valid == 0 else \
                    (query_coords[prev_valid]-query_coords[prev_valid-1])
                d = d / (np.linalg.norm(d)+1e-10)
                query_coords[i] = query_coords[prev_valid] + d * backbone_distance * (i-prev_valid)
            elif next_valid is not None:
                query_coords[i] = query_coords[next_valid] - np.array([backbone_distance*(next_valid-i),0,0])
    return np.nan_to_num(query_coords)


def generate_rna_structure(sequence, seed=None):
    """Generate a random RNA-like 3D structure. Used as fallback
    when no template is found. Produces a chain with realistic
    backbone distances (~3.5-4.5A) and random turns."""
    if seed is not None:
        np.random.seed(seed); random.seed(seed)
    n = len(sequence)
    coords = np.zeros((n, 3))
    for i in range(min(3, n)):
        angle = i * 0.6
        coords[i] = [10*np.cos(angle), 10*np.sin(angle), i*2.5]
    direction = np.array([0,0,1.0])
    for i in range(3, n):
        if random.random() < 0.3:
            axis = np.random.normal(0,1,3)
            axis = axis/(np.linalg.norm(axis)+1e-10)
            rot = R.from_rotvec(random.uniform(0.2,0.6)*axis)
            direction = rot.apply(direction)
        else:
            direction += np.random.normal(0,0.15,3)
            direction = direction/(np.linalg.norm(direction)+1e-10)
        step = random.uniform(3.5, 4.5)
        coords[i] = coords[i-1] + step*direction
    return coords


def find_similar_sequences(query_seq, train_seqs_df, train_coords_dict, top_n=5):
    """Find the most similar known RNA structures to the query.
    Stage 1: k-mer (k=3) Jaccard similarity for fast filtering.
    Stage 2: Global pairwise alignment for accurate scoring.
    Returns list of (target_id, sequence, score, coords) tuples."""
    query_seq_obj = Seq(query_seq)
    candidates = []
    k = 3
    q_kmers = set(query_seq[i:i+k] for i in range(len(query_seq)-k+1))
    for _, row in train_seqs_df.iterrows():
        tid = row['target_id']
        tseq = row['sequence']
        if tid not in train_coords_dict: continue
        # Skip if length ratio > 40% different
        lr = abs(len(tseq)-len(query_seq)) / max(len(tseq),len(query_seq))
        if lr > 0.4: continue
        t_kmers = set(tseq[i:i+k] for i in range(len(tseq)-k+1))
        score = len(q_kmers & t_kmers) / len(q_kmers | t_kmers) if q_kmers | t_kmers else 0
        candidates.append((tid, tseq, score, train_coords_dict[tid]))
    candidates.sort(key=lambda x: x[2], reverse=True)
    candidates = candidates[:100]  # Top 100 by k-mer score
    # Stage 2: precise alignment scoring
    similar = []
    for tid, tseq, _, coords in candidates:
        alns = pairwise2.align.globalms(query_seq_obj, tseq, 2.9,-1,-10,-0.5, one_alignment_only=True)
        if alns:
            s = alns[0].score / (2*min(len(query_seq),len(tseq)))
            if s > 0:
                similar.append((tid, tseq, s, coords))
    similar.sort(key=lambda x: x[2], reverse=True)
    return similar[:top_n]


# ============================================================
# CELL 9: Run Fork 2 template search for ALL test targets (~5 min)
# ============================================================
# For each of the 28 test targets, find the best matching known
# RNA structure from our 24K+ extended dataset. Store the adapted
# template coordinates and alignment confidence score. These are
# used both for hybrid inference (Slots 1-2) and as starting
# points for template-seeded NN refinement (Slots 3-5, Option B).
# ============================================================
print("\n" + "="*60)
print("PHASE 1: Fork 2 Template Search")
print("="*60)

template_coords_per_target = {}       # target_id -> (N, 3) numpy array
template_confidence_per_target = {}   # target_id -> float (alignment score)

start_time = time.time()
for idx, row in test_seqs.iterrows():
    target_id = row['target_id']
    sequence = row['sequence']
    if idx % 5 == 0:
        elapsed = time.time() - start_time
        print(f"  Template search {idx+1}/{len(test_seqs)}: {target_id} ({len(sequence)} nt) [{elapsed:.0f}s]")

    similar = find_similar_sequences(sequence, train_seqs_extended, train_coords_dict, top_n=1)

    if similar:
        best_tid, best_seq, best_score, best_coords = similar[0]
        adapted = adapt_template_to_query(sequence, best_seq, best_coords)
        refined = adaptive_rna_constraints(adapted, sequence, confidence=best_score)
        template_coords_per_target[target_id] = refined
        template_confidence_per_target[target_id] = best_score
        print(f"    -> Template: {best_tid} (score={best_score:.3f})")
    else:
        template_coords_per_target[target_id] = np.zeros((len(sequence), 3))
        template_confidence_per_target[target_id] = 0.0
        print(f"    -> No template found")

print(f"\nTemplate search complete. {sum(1 for v in template_confidence_per_target.values() if v > 0)}/{len(test_seqs)} targets have templates.")


# ============================================================
# CELL 10: Define ADV1 Neural Network Components (INLINED)
# ============================================================
# All model components are defined inline (not imported from files)
# so the notebook is self-contained on Kaggle.
#
# Architecture:
#   RibonanzaNet backbone -> pairwise features (N,N,64)
#   TemplateEncoder -> template features (N,N,16)
#   Concatenate -> (N,N,80)
#   DistanceMatrixHead MLP -> predicted distance matrix (N,N)
#   Template-seeded refinement -> 3D coordinates (N,3)
# ============================================================
print("\n" + "="*60)
print("PHASE 2: ADV1 Neural Network Setup")
print("="*60)

# --- Template Encoder ---
# Converts 3D template coordinates into pairwise distance bins,
# then projects to template_dim features. Scaled by confidence.
class TemplateEncoder(nn.Module):
    def __init__(self, template_dim=16, num_bins=22, max_dist=40.0):
        super().__init__()
        self.template_dim = template_dim
        self.num_bins = num_bins
        bin_width = max_dist / (num_bins - 1)
        edges = torch.arange(0, max_dist + bin_width, bin_width)[:num_bins]
        self.register_buffer('bin_edges', edges)
        self.projection = nn.Linear(num_bins, template_dim)
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.projection.bias)

    def forward(self, coords, confidence=1.0, has_template=True):
        N = coords.shape[0]
        if not has_template:
            return torch.zeros(N, N, self.template_dim, device=coords.device)
        diff = coords.unsqueeze(0) - coords.unsqueeze(1)
        dist = torch.sqrt((diff**2).sum(-1) + 1e-8)
        bin_idx = torch.bucketize(dist, self.bin_edges).clamp(0, self.num_bins-1)
        bins = torch.zeros(N, N, self.num_bins, device=dist.device)
        bins.scatter_(2, bin_idx.unsqueeze(-1), 1.0)
        features = self.projection(bins) * confidence
        return features

# --- Distance Matrix Head ---
# 3-layer MLP: (pair_dim) -> 128 -> 128 -> 1, with Softplus activation.
# Output is symmetric and zero-diagonal (valid distance matrix).
class DistanceMatrixHead(nn.Module):
    def __init__(self, pair_dim=80, hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        layers = []
        in_dim = pair_dim
        for i in range(num_layers - 1):
            out_dim = hidden_dim
            layers.extend([nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim),
                           nn.ReLU(), nn.Dropout(dropout)])
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)
        self.activation = nn.Softplus()

    def forward(self, pairwise_repr):
        raw = self.mlp(pairwise_repr).squeeze(-1)
        dist = self.activation(raw)
        dist = (dist + dist.transpose(-1, -2)) / 2.0  # Symmetrize
        B, N, _ = dist.shape
        mask = torch.eye(N, device=dist.device, dtype=torch.bool).unsqueeze(0)
        dist = dist.masked_fill(mask, 0.0)  # Zero diagonal
        return dist

# --- MDS Reconstruction (FALLBACK ONLY in Option B) ---
# Classical Multidimensional Scaling via eigendecomposition.
# Used ONLY when no template exists. Template-seeded refinement
# is preferred (see Cell 14) because MDS produces poor coordinates
# from noisy distance predictions (Run 2 proved this: 0.109 score).
def mds_reconstruct(dist_matrix_np):
    N = dist_matrix_np.shape[0]
    if N < 4:
        coords = np.zeros((N, 3))
        for i in range(N): coords[i, 0] = i * 5.9
        return coords
    D_sq = dist_matrix_np ** 2
    H = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * H @ D_sq @ H
    B = (B + B.T) / 2.0
    eigenvalues, eigenvectors = eigh(B)
    top3_idx = np.argsort(eigenvalues)[-3:][::-1]
    top3_vals = np.maximum(eigenvalues[top3_idx], 1e-6)
    top3_vecs = eigenvectors[:, top3_idx]
    return top3_vecs * np.sqrt(top3_vals)[np.newaxis, :]

# --- Coordinate Refinement via Gradient Descent ---
# Takes initial coordinates (from template or MDS) and adjusts them
# to match the target distance matrix using Adam optimizer.
# Also enforces consecutive backbone distance of ~5.9A.
def refine_coords(coords_t, target_dist_t, steps=100, lr=0.01):
    refined = coords_t.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([refined], lr=lr)
    N = coords_t.shape[0]
    triu = torch.triu(torch.ones(N, N, device=coords_t.device, dtype=torch.bool), diagonal=1)
    for _ in range(steps):
        opt.zero_grad()
        cur_dist = torch.cdist(refined.unsqueeze(0), refined.unsqueeze(0)).squeeze(0)
        diff = (cur_dist - target_dist_t) * triu.float()
        stress = (diff**2).sum() / (triu.float().sum() + 1e-8)
        if N > 1:
            consec = torch.norm(refined[1:] - refined[:-1], dim=-1)
            stress = stress + ((consec - 5.9)**2).mean()
        stress.backward()
        opt.step()
    return refined.detach()


# ============================================================
# CELL 11: Load RibonanzaNet Backbone (SELECTIVE UNFREEZE)
# ============================================================
# Change 2: Unfreeze last 2 of 9 transformer layers.
# The frozen layers (0-6) were trained for chemical mapping (DMS/2A3).
# The unfrozen layers (7-8) adapt to distance prediction during
# training in Cell 13. This is DESIGN.md Option A.
#
# Two feature extraction functions:
#   get_pairwise_features()       — inference: no gradients (Cell 14)
#   get_pairwise_features_train() — training: gradients through 7,8 (Cell 13)
# ============================================================
print("\nLoading RibonanzaNet backbone...")
if REPO_PATH:
    sys.path.insert(0, REPO_PATH)

from Network import RibonanzaNet

# Default config — overridden by pairwise.yaml if found in repo
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

# Load pretrained weights (trained on chemical mapping task)
state = torch.load(BACKBONE_WEIGHTS, map_location='cpu', weights_only=False)
if isinstance(state, dict) and 'model_state_dict' in state:
    state = state['model_state_dict']
backbone.load_state_dict(state, strict=False)
print(f"  Backbone loaded from {BACKBONE_WEIGHTS}")

backbone = backbone.to(device)

# --- Selective unfreezing ---
# Step 1: Freeze ALL parameters
for p in backbone.parameters():
    p.requires_grad = False

# Step 2: Unfreeze last N transformer layers
UNFREEZE_LAST_N = 2
total_layers = len(list(backbone.transformer_encoder))
print(f"  Backbone has {total_layers} transformer layers")

unfrozen_backbone_params = []
for i, layer in enumerate(backbone.transformer_encoder):
    if i >= total_layers - UNFREEZE_LAST_N:
        for p in layer.parameters():
            p.requires_grad = True
            unfrozen_backbone_params.append(p)
        print(f"  Layer {i}: UNFROZEN (trainable)")
    else:
        print(f"  Layer {i}: frozen")

bb_frozen = sum(p.numel() for p in backbone.parameters() if not p.requires_grad)
bb_unfrozen = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
print(f"  Backbone frozen params: {bb_frozen:,}")
print(f"  Backbone UNFROZEN params: {bb_unfrozen:,}")

# --- Feature extraction: INFERENCE mode (no gradients) ---
def get_pairwise_features(sequence):
    """Extract (1, N, N, 64) pairwise features. Used in Cell 14 inference."""
    base_map = {'A':0, 'C':1, 'G':2, 'U':3}
    tokens = torch.tensor([base_map.get(b, 4) for b in sequence.upper()],
                          dtype=torch.long).unsqueeze(0).to(device)
    N = len(sequence)
    src_mask = torch.ones(1, N, dtype=torch.long, device=device)
    with torch.no_grad():
        embedded = backbone.encoder(tokens)
        pairwise = backbone.outer_product_mean(embedded)
        pairwise = pairwise + backbone.pos_encoder(embedded)
        hidden = embedded
        for layer in backbone.transformer_encoder:
            result = layer(hidden, pairwise, src_mask=src_mask)
            if isinstance(result, tuple):
                hidden, pairwise = result
            else:
                hidden = result
    return pairwise

# --- Feature extraction: TRAINING mode (gradients through unfrozen layers) ---
def get_pairwise_features_train(sequence):
    """Extract pairwise features with gradient flow through layers 7,8.
    Frozen layers (0-6) run under no_grad, then detach.
    Unfrozen layers (7,8) run with gradients for backprop."""
    base_map = {'A':0, 'C':1, 'G':2, 'U':3}
    tokens = torch.tensor([base_map.get(b, 4) for b in sequence.upper()],
                          dtype=torch.long).unsqueeze(0).to(device)
    N = len(sequence)
    src_mask = torch.ones(1, N, dtype=torch.long, device=device)

    # Frozen portion: no gradients
    with torch.no_grad():
        embedded = backbone.encoder(tokens)
        pairwise = backbone.outer_product_mean(embedded)
        pairwise = pairwise + backbone.pos_encoder(embedded)
        hidden = embedded
        frozen_layer_count = total_layers - UNFREEZE_LAST_N
        for i, layer in enumerate(backbone.transformer_encoder):
            if i < frozen_layer_count:
                result = layer(hidden, pairwise, src_mask=src_mask)
                if isinstance(result, tuple):
                    hidden, pairwise = result
                else:
                    hidden = result

    # Detach: cuts gradient flow from frozen layers
    hidden = hidden.detach().requires_grad_(True)
    pairwise = pairwise.detach().requires_grad_(True)

    # Unfrozen portion: gradients flow for backprop
    for i, layer in enumerate(backbone.transformer_encoder):
        if i >= frozen_layer_count:
            result = layer(hidden, pairwise, src_mask=src_mask)
            if isinstance(result, tuple):
                hidden, pairwise = result
            else:
                hidden = result

    return pairwise  # (1, N, N, 64) with gradient graph


# ============================================================
# CELL 12: Create ADV1 model + warm-start from BASIC
# ============================================================
# Creates distance head (pair_dim=80 -> distances) and template
# encoder (3D coords -> 16 pairwise features).
# Warm-starts distance head from BASIC's best_model.pt (pair_dim=64),
# expanding first layer weights from 64 to 80 columns (16 new
# template channels initialized to zero).
# ============================================================
print("\nCreating ADV1 model...")
template_encoder = TemplateEncoder(template_dim=16, num_bins=22, max_dist=40.0).to(device)
distance_head = DistanceMatrixHead(pair_dim=80, hidden_dim=128, num_layers=3, dropout=0.1).to(device)

# Warm-start from BASIC checkpoint
if ADV1_WEIGHTS and os.path.exists(ADV1_WEIGHTS):
    print(f"  Warm-starting from {ADV1_WEIGHTS}")
    basic_ckpt = torch.load(ADV1_WEIGHTS, map_location=device, weights_only=False)
    if 'model_state_dict' in basic_ckpt:
        basic_state = basic_ckpt['model_state_dict']
    else:
        basic_state = basic_ckpt

    adv1_state = distance_head.state_dict()
    for key in adv1_state:
        if key in basic_state:
            bp = basic_state[key]
            ap = adv1_state[key]
            if bp.shape == ap.shape:
                # Exact match: copy directly
                adv1_state[key] = bp
            elif len(bp.shape) == 2 and len(ap.shape) == 2 and \
                 bp.shape[0] == ap.shape[0] and bp.shape[1] < ap.shape[1]:
                # Width expansion: copy old columns, leave new columns at zero
                print(f"    Expanding {key}: {bp.shape} -> {ap.shape}")
                adv1_state[key][:, :bp.shape[1]] = bp
    distance_head.load_state_dict(adv1_state)
    print("  Warm-start complete")

dh_params = sum(p.numel() for p in distance_head.parameters())
te_params = sum(p.numel() for p in template_encoder.parameters())
print(f"  Distance head params: {dh_params:,}")
print(f"  Template encoder params: {te_params:,}")
print(f"  Unfrozen backbone params: {bb_unfrozen:,}")
print(f"  TOTAL trainable params: {dh_params + te_params + bb_unfrozen:,}")


# ============================================================
# CELL 13: Train ADV1 on pdb_xyz_data.pkl
# ============================================================
# Training configuration:
#   - 30 epochs (Change 1, was 15 in Run 1)
#   - BATCH_SIZE=2 (Change 2, was 4 — reduced for backbone VRAM)
#   - Discriminative LR: head 5e-5, backbone 1e-5 (Change 2)
#   - 50% template masking (teaches head to work without templates)
#   - Saves checkpoint with distance head + template encoder +
#     unfrozen backbone layer weights
#
# Pickle format (Change 4 fix):
#   Dict with keys: 'sequence', 'xyz', 'publication_date', ...
#   xyz[i] = list of defaultdicts per nucleotide
#   Each dict has 'sugar_ring' where sugar_ring[0] = C1' coords (3,)
# ============================================================
print("\n" + "="*60)
print("PHASE 3: ADV1 Training (with backbone unfreezing)")
print("="*60)

TRAIN_EPOCHS = 30
BATCH_SIZE = 2                     # Reduced for VRAM (backbone gradients)
MAX_SEQ_LEN = 256
HEAD_LR = 5e-5                     # Distance head + template encoder
BACKBONE_LR = 1e-5                 # Unfrozen backbone layers (5x lower)
TEMPLATE_MASK_PROB = 0.5           # 50% chance of masking template during training

# --- Load and parse training data ---
print(f"Loading training data from {TRAIN_PICKLE}...")
with open(TRAIN_PICKLE, 'rb') as f:
    raw_data = pickle.load(f)

print(f"  Pickle type: {type(raw_data)}")
if isinstance(raw_data, dict):
    print(f"  Keys: {list(raw_data.keys())}")
    sequences = raw_data['sequence']
    xyz_list = raw_data['xyz']
    print(f"  Total structures: {len(sequences)}")

    train_items = []
    skipped = 0

    for i in range(len(sequences)):
        try:
            seq = sequences[i]
            residue_list = xyz_list[i]

            if seq is None or residue_list is None:
                skipped += 1
                continue

            # Extract C1' coordinate from each nucleotide's sugar ring
            c1_coords = []
            valid_bases = []

            for j, residue_atoms in enumerate(residue_list):
                if not hasattr(residue_atoms, 'keys') or 'sugar_ring' not in residue_atoms:
                    continue
                sugar_ring = residue_atoms['sugar_ring']
                if sugar_ring is None or len(sugar_ring) == 0:
                    continue
                c1_prime = sugar_ring[0]  # C1' = first atom of sugar ring
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

            if min_len <= MAX_SEQ_LEN and min_len >= 10:
                train_items.append({'sequence': clean_seq, 'coords': coords})

        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  Warning: Error on structure {i}: {e}")

    print(f"  Parsed: {len(train_items)} structures, Skipped: {skipped}")

else:
    # Fallback: if pickle is a list (shouldn't happen with our data)
    print(f"  Pickle is a list with {len(raw_data)} items")
    train_items = []
    for item in raw_data:
        seq = item.get('sequence', '')
        coords = item.get('coordinates', item.get('coords', None))
        if seq and coords is not None and len(seq) <= MAX_SEQ_LEN and len(seq) == len(coords):
            train_items.append({'sequence': seq, 'coords': np.array(coords, dtype=np.float32)})
    print(f"  After filtering: {len(train_items)} structures")

print(f"  Final training set: {len(train_items)} structures (<= {MAX_SEQ_LEN} nt, >= 10 nt)")

# --- Train/val split (90/10, seeded for reproducibility) ---
random.seed(42)
random.shuffle(train_items)
val_size = max(1, int(len(train_items) * 0.1))
val_items = train_items[:val_size]
train_items = train_items[val_size:]
print(f"  Train: {len(train_items)}, Val: {val_size}")

# --- Optimizer: discriminative learning rate ---
# Head+template learn at 5x the rate of unfrozen backbone layers.
# This prevents the pretrained backbone from diverging too fast.
head_and_template_params = list(distance_head.parameters()) + list(template_encoder.parameters())
optimizer = optim.AdamW([
    {'params': head_and_template_params, 'lr': HEAD_LR},
    {'params': unfrozen_backbone_params, 'lr': BACKBONE_LR},
], weight_decay=0.01)

all_trainable_params = head_and_template_params + unfrozen_backbone_params
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_EPOCHS)
scaler = GradScaler(enabled=(device.type == 'cuda'))

distance_head.train()
template_encoder.train()

best_val_loss = float('inf')
train_start = time.time()

for epoch in range(TRAIN_EPOCHS):
    epoch_loss = 0.0
    n_batches = 0
    random.shuffle(train_items)

    for batch_start in range(0, len(train_items), BATCH_SIZE):
        batch = train_items[batch_start:batch_start+BATCH_SIZE]
        batch_losses = []

        for item in batch:
            seq = item['sequence']
            true_coords = item['coords']
            N = len(seq)

            # Compute ground truth pairwise distance matrix
            true_dist = np.sqrt(((true_coords[:, None] - true_coords[None, :])**2).sum(-1))
            true_dist_t = torch.tensor(true_dist, dtype=torch.float32, device=device).unsqueeze(0)

            # Get pairwise features (gradients through unfrozen layers)
            pairwise = get_pairwise_features_train(seq)  # (1, N, N, 64)

            # Template features: 50% masked to teach head to work without
            if random.random() < TEMPLATE_MASK_PROB:
                tmpl_feat = torch.zeros(1, N, N, 16, device=device)
            else:
                coords_t = torch.tensor(true_coords, dtype=torch.float32, device=device)
                tmpl_feat = template_encoder(coords_t, confidence=1.0, has_template=True)
                tmpl_feat = tmpl_feat.unsqueeze(0)

            # Predict distances from concatenated features
            combined = torch.cat([pairwise, tmpl_feat], dim=-1)  # (1, N, N, 80)
            pred_dist = distance_head(combined)

            # Loss: MSE on all pairwise distances
            loss = ((pred_dist - true_dist_t)**2).mean()

            # Consecutive distance regularization (~5.9A between neighbors)
            if N > 1:
                consec_pred = pred_dist[0, torch.arange(N-1), torch.arange(1, N)]
                consec_loss = ((consec_pred - 5.9)**2).mean()
                loss = loss + 0.1 * consec_loss

            batch_losses.append(loss)

        if batch_losses:
            total_loss = sum(batch_losses) / len(batch_losses)
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(all_trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += total_loss.item()
            n_batches += 1

    scheduler.step()
    avg_train = epoch_loss / max(n_batches, 1)

    # --- Validation (inference mode, no gradients) ---
    distance_head.eval()
    template_encoder.eval()
    val_loss = 0.0
    with torch.no_grad():
        for item in val_items[:20]:
            seq = item['sequence']
            true_coords = item['coords']
            N = len(seq)
            true_dist = np.sqrt(((true_coords[:,None]-true_coords[None,:])**2).sum(-1))
            true_dist_t = torch.tensor(true_dist, dtype=torch.float32, device=device).unsqueeze(0)
            pairwise = get_pairwise_features(seq)
            coords_t = torch.tensor(true_coords, dtype=torch.float32, device=device)
            tmpl_feat = template_encoder(coords_t, confidence=1.0, has_template=True).unsqueeze(0)
            combined = torch.cat([pairwise, tmpl_feat], dim=-1)
            pred_dist = distance_head(combined)
            val_loss += ((pred_dist - true_dist_t)**2).mean().item()
    avg_val = val_loss / max(len(val_items[:20]), 1)

    lrs = scheduler.get_last_lr()
    elapsed = time.time() - train_start
    print(f"Epoch {epoch+1}/{TRAIN_EPOCHS}: train={avg_train:.4f}, val={avg_val:.4f}, "
          f"lr_head={lrs[0]:.6f}, lr_bb={lrs[1]:.7f}, time={elapsed:.0f}s")

    # Save best checkpoint (distance head + template encoder + backbone layers)
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save({
            'model_state_dict': distance_head.state_dict(),
            'template_encoder_state_dict': template_encoder.state_dict(),
            'backbone_unfrozen_state': {
                f'transformer_encoder.{i}': backbone.transformer_encoder[i].state_dict()
                for i in range(total_layers - UNFREEZE_LAST_N, total_layers)
            },
            'epoch': epoch, 'val_loss': avg_val
        }, CHECKPOINT_PATH)
        print(f"  -> Saved best model (val={avg_val:.4f})")

    distance_head.train()
    template_encoder.train()

print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
print(f"Total training time: {time.time()-train_start:.0f}s")

# --- Reload best checkpoint for inference ---
ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
distance_head.load_state_dict(ckpt['model_state_dict'])
template_encoder.load_state_dict(ckpt['template_encoder_state_dict'])
if 'backbone_unfrozen_state' in ckpt:
    for layer_key, layer_state in ckpt['backbone_unfrozen_state'].items():
        layer_idx = int(layer_key.split('.')[-1])
        backbone.transformer_encoder[layer_idx].load_state_dict(layer_state)
    print("Loaded best backbone unfrozen layers")
distance_head.eval()
template_encoder.eval()
backbone.eval()
print("Loaded best checkpoint for inference")


# ============================================================
# CELL 14: ADV1 Inference — Hybrid + Template-Seeded Refinement
# ============================================================
# This is the key inference cell. For each test target:
#
# STEP 1: Run NN to get predicted distance matrix
# STEP 2: Decide hybrid strategy based on template confidence
#
# HIGH confidence (> HYBRID_THRESHOLD):
#   Slots 1-2: Template coords directly (like Fork 2, proven 0.287)
#   Slots 3-5: Template-seeded NN refinement (OPTION B — NEW)
#     -> Start from template coords (real structure)
#     -> Refine toward NN's predicted distances via gradient descent
#     -> Much better than MDS reconstruction (Run 2 proved MDS = 0.109)
#
# LOW confidence (<= HYBRID_THRESHOLD):
#   All 5 slots: Template-seeded NN refinement
#     -> Even weak templates provide better starting coords than MDS
#     -> Fallback to MDS only if template confidence = 0 (no template)
#
# WHY NOT MDS (the Run 2 lesson):
#   MDS takes an NxN distance matrix and reconstructs 3D coords via
#   eigendecomposition. With perfect distances, MDS works. But our NN
#   predicts distances with ~13A average error (val_loss=171, sqrt=13).
#   MDS amplifies these errors: D^2 computation, triangle inequality
#   violations, dimensionality collapse. Result: 0.109 score.
#   Template-seeded refinement avoids all this by starting from real
#   coordinates and making small adjustments.
# ============================================================
print("\n" + "="*60)
print("PHASE 4: ADV1 Inference (Hybrid + Template-Seeded Refinement)")
print("="*60)

MAX_INFER_LEN = 512       # Truncate sequences longer than this for NN
HYBRID_THRESHOLD = 0.3    # Template confidence threshold for hybrid decision

all_predictions = []
infer_start = time.time()

# Counters for logging
n_hybrid_template = 0   # Targets using template slots 1-2
n_hybrid_nn_only = 0    # Targets using all-NN slots
n_template_seeded = 0   # NN slots that used template-seeded refinement
n_mds_fallback = 0      # NN slots that fell back to MDS (no template)

for idx, row in test_seqs.iterrows():
    target_id = row['target_id']
    sequence = row['sequence']
    seq = sequence[:MAX_INFER_LEN]    # Truncated for NN (backbone VRAM limit)
    N = len(seq)

    if idx % 5 == 0:
        print(f"  Predicting {idx+1}/{len(test_seqs)}: {target_id} ({len(sequence)} nt)")

    # --- STEP 1: Get NN predicted distance matrix ---
    with torch.no_grad():
        pairwise = get_pairwise_features(seq)   # (1, N, N, 64) backbone features

        # Get template coords (truncated to N for NN input)
        tmpl_coords = template_coords_per_target.get(target_id, np.zeros((N, 3)))
        tmpl_conf = template_confidence_per_target.get(target_id, 0.0)

        # Pad/truncate template to match NN sequence length N
        if len(tmpl_coords) > N:
            tmpl_coords = tmpl_coords[:N]
        elif len(tmpl_coords) < N:
            padded = np.zeros((N, 3))
            padded[:len(tmpl_coords)] = tmpl_coords
            tmpl_coords = padded

        # Encode template as features for the distance head
        coords_t = torch.tensor(tmpl_coords, dtype=torch.float32, device=device)
        has_tmpl = tmpl_conf > 0.01
        tmpl_feat = template_encoder(coords_t, confidence=tmpl_conf, has_template=has_tmpl)
        tmpl_feat = tmpl_feat.unsqueeze(0)   # (1, N, N, 16)

        # Predict distance matrix from combined features
        combined = torch.cat([pairwise, tmpl_feat], dim=-1)  # (1, N, N, 80)
        pred_dist = distance_head(combined).squeeze(0)        # (N, N)

    # --- STEP 2: Generate 5 diverse coordinate predictions ---

    # Get full-length template coords for the submission (not truncated)
    full_tmpl = template_coords_per_target.get(target_id, np.zeros((len(sequence), 3)))
    if len(full_tmpl) > len(sequence):
        full_tmpl = full_tmpl[:len(sequence)]
    elif len(full_tmpl) < len(sequence):
        padded = np.zeros((len(sequence), 3))
        padded[:len(full_tmpl)] = full_tmpl
        full_tmpl = padded

    # NN diversity settings: different noise levels and refinement steps
    nn_diversity = [
        {'noise': 0.0, 'steps': 100, 'seed': 0},   # Clean NN prediction
        {'noise': 0.3, 'steps': 100, 'seed': 1},   # Light noise for diversity
        {'noise': 0.5, 'steps': 100, 'seed': 2},   # Medium noise
        {'noise': 0.7, 'steps': 150, 'seed': 3},   # Heavy noise, more refinement
        {'noise': 0.0, 'steps': 50, 'seed': 4},    # Clean, less refinement
    ]

    # Generate NN-based predictions using TEMPLATE-SEEDED REFINEMENT (Option B)
    nn_coords_list = []
    for div in nn_diversity:
        torch.manual_seed(div['seed'])

        # Add noise to predicted distances for diversity
        noisy = pred_dist.clone()
        if div['noise'] > 0:
            noise = torch.randn_like(pred_dist) * div['noise']
            noise = (noise + noise.T) / 2.0       # Keep symmetric
            noise.fill_diagonal_(0.0)               # Zero diagonal
            noisy = torch.clamp(noisy + noise, min=0.0)  # Non-negative

        # --- OPTION B: Template-seeded refinement ---
        # Instead of MDS (which produces garbage from noisy distances),
        # start from template coordinates and refine toward NN distances.
        # Even a weak template (conf=0.15) is better than MDS output.
        if tmpl_conf > 0.01:
            # Use template coords (truncated to N) as starting point
            start_coords = torch.tensor(
                tmpl_coords[:N], dtype=torch.float32, device=device
            )
            n_template_seeded += 1
        else:
            # No template at all: fall back to MDS (only option)
            dist_np = noisy.detach().cpu().numpy()
            start_coords = torch.tensor(
                mds_reconstruct(dist_np), dtype=torch.float32, device=device
            )
            n_mds_fallback += 1

        # Refine starting coords toward NN's predicted distances
        refined = refine_coords(start_coords, noisy.detach(),
                                steps=div['steps'], lr=0.01)
        coords_np = refined.cpu().numpy()

        # Handle sequences longer than MAX_INFER_LEN: extrapolate
        if len(sequence) > MAX_INFER_LEN:
            remaining = len(sequence) - MAX_INFER_LEN
            last_dir = coords_np[-1] - coords_np[-2] if N >= 2 else np.array([5.9,0,0])
            last_dir = last_dir / (np.linalg.norm(last_dir)+1e-8) * 5.9
            extra = np.array([coords_np[-1] + last_dir*(i+1) for i in range(remaining)])
            coords_np = np.concatenate([coords_np, extra])

        nn_coords_list.append(coords_np)

    # --- STEP 3: Assemble 5 slots based on hybrid decision ---
    coords_list = []

    if tmpl_conf > HYBRID_THRESHOLD:
        # HIGH confidence: Slots 1-2 = template (Fork 2 quality), Slots 3-5 = NN
        n_hybrid_template += 1

        # Slot 1: Clean template coords with constraint refinement
        tmpl_refined = adaptive_rna_constraints(full_tmpl.copy(), sequence, confidence=tmpl_conf)
        coords_list.append(tmpl_refined)

        # Slot 2: Template with small perturbation for diversity
        np.random.seed(42)
        small_noise = np.random.normal(0, 0.5, full_tmpl.shape)
        tmpl_noisy = adaptive_rna_constraints(full_tmpl + small_noise, sequence, confidence=tmpl_conf)
        coords_list.append(tmpl_noisy)

        # Slots 3-5: NN predictions (template-seeded refinement)
        coords_list.append(nn_coords_list[0])
        coords_list.append(nn_coords_list[1])
        coords_list.append(nn_coords_list[2])

        if idx % 5 == 0:
            print(f"    HYBRID: conf={tmpl_conf:.3f} > {HYBRID_THRESHOLD} -> slots 1-2 template, 3-5 NN (template-seeded)")
    else:
        # LOW confidence: all 5 slots = NN (template-seeded or MDS fallback)
        n_hybrid_nn_only += 1
        coords_list = nn_coords_list[:5]

        if idx % 5 == 0:
            seed_type = "template-seeded" if tmpl_conf > 0.01 else "MDS fallback"
            print(f"    HYBRID: conf={tmpl_conf:.3f} <= {HYBRID_THRESHOLD} -> all 5 slots NN ({seed_type})")

    # --- STEP 4: Write submission rows ---
    for j in range(len(sequence)):
        pred_row = {
            'ID': f"{target_id}_{j+1}",
            'resname': sequence[j],
            'resid': j + 1
        }
        for i in range(5):
            pred_row[f'x_{i+1}'] = coords_list[i][j][0]
            pred_row[f'y_{i+1}'] = coords_list[i][j][1]
            pred_row[f'z_{i+1}'] = coords_list[i][j][2]
        all_predictions.append(pred_row)

# --- Summary ---
print(f"\nInference complete. {len(all_predictions)} rows in {time.time()-infer_start:.0f}s")
print(f"  Targets with template slots (1-2): {n_hybrid_template}")
print(f"  Targets with all-NN slots: {n_hybrid_nn_only}")
print(f"  NN slots using template-seeded refinement: {n_template_seeded}")
print(f"  NN slots using MDS fallback: {n_mds_fallback}")

# Save raw submission
submission_df = pd.DataFrame(all_predictions)
col_order = ['ID', 'resname', 'resid']
for i in range(1, 6):
    for c in ['x', 'y', 'z']:
        col_order.append(f'{c}_{i}')
submission_df = submission_df[col_order]
submission_df.to_csv(RAW_SUBMISSION_PATH, index=False)
print(f"Raw submission: {len(submission_df)} rows")


# ============================================================
# CELL 15: Option B Post-Processing (match sample_submission.csv IDs)
# ============================================================
# Ensures every ID in sample_submission.csv has a row in our output.
# If ADV1 produced the ID, use our prediction. Otherwise, use the
# sample's zero-filled row. In practice, ADV1 covers all 28 targets
# so filled=0.
# ============================================================
print("\n" + "="*60)
print("PHASE 5: Post-Processing (Option B)")
print("="*60)

sample_rows = {}
sample_order = []
cols = None
with open(SAMPLE_CSV, "r") as f:
    reader = csv.DictReader(f)
    cols = reader.fieldnames
    for row in reader:
        sample_rows[row["ID"]] = row
        sample_order.append(row["ID"])
print(f"Sample expects {len(sample_order)} rows")

raw_rows = {}
with open(RAW_SUBMISSION_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        raw_rows[row["ID"]] = row
print(f"ADV1 produced {len(raw_rows)} rows")

matched = 0
filled = 0
with open(FINAL_SUBMISSION_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=cols)
    writer.writeheader()
    for sid in sample_order:
        if sid in raw_rows:
            writer.writerow(raw_rows[sid])
            matched += 1
        else:
            writer.writerow(sample_rows[sid])
            filled += 1

print(f"Final submission.csv:")
print(f"  Matched from ADV1: {matched}")
print(f"  Filled with zeros: {filled}")
print(f"  Total rows: {matched + filled}")
print(f"  File size: {os.path.getsize(FINAL_SUBMISSION_PATH)} bytes")

print("\n" + "="*60)
print("DONE! submission.csv is ready for scoring")
print("="*60)
