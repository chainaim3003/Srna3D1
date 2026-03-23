# ============================================================
# HY-BAS-ADV1 RUN 4: All-in-One Kaggle Notebook
# ============================================================
#
# WHAT THIS IS:
#   Run 4 = Run 3 Option B + MSA (evolutionary) features.
#   Everything from Run 3 OptB is preserved. The ONLY addition
#   is 8 new MSA feature channels that tell the neural network
#   which RNA positions are likely close together in 3D, based
#   on evolutionary evidence (co-mutation patterns).
#
# SIMPLE EXPLANATION (for a high schooler):
#   Imagine you have an RNA sequence from a human. You find the
#   "same" RNA in a mouse, frog, fish, and fly. Their sequences
#   are similar but not identical — they've mutated over millions
#   of years. If position 15 and position 42 always mutate
#   TOGETHER across species, they're probably physically touching
#   in 3D. That co-mutation pattern is called "covariation."
#   We compute these patterns and feed them to the neural network
#   as 8 extra input channels, alongside the existing 80 channels.
#
# WHAT'S NEW IN RUN 4 (Change 7):
#   Change 7: MSA features (Cells 9, 9.5, 10, 12, 13, 14)
#     - Cell 9:   Template search now collects top-20 similar
#                 sequences per target (was top-1) for MSA
#     - Cell 9.5: NEW — computes 8 MSA features per target from
#                 the aligned similar sequences
#     - Cell 10:  NEW — compute_msa_features() function added
#     - Cell 12:  pair_dim expanded 80 -> 88 (8 new MSA channels)
#                 Warm-starts from Run 3 OptB checkpoint
#     - Cell 13:  Training concatenates MSA features (64+16+8=88)
#                 Pre-computes MSA for training sequences
#     - Cell 14:  Inference concatenates MSA features (64+16+8=88)
#
# PRESERVED FROM RUN 3 Option B:
#   Change 1: TRAIN_EPOCHS = 30
#   Change 2: Unfreeze backbone layers 7,8 + discriminative LR
#   Change 3: Biopython offline wheel install
#   Change 4: Fixed pickle parsing
#   Change 5: Hybrid inference (template slots + NN slots)
#   Change 6: Template-seeded refinement (NN starts from template)
#
# THE 8 MSA CHANNELS (what the neural network sees):
#   1. Covariation (i,j): Do positions i and j mutate together?
#      High value = likely physically close in 3D.
#   2. APC-corrected covariation: Same but with background noise
#      removed. More reliable than raw covariation.
#   3. Conservation at i: Does position i never change across
#      species? High = structurally critical, rigid.
#   4. Conservation at j: Same for the other position.
#   5. Conservation product (i×j): Both positions are conserved?
#      Then they're both in the rigid core of the structure.
#   6. Gap frequency at i: Is position i missing in many species?
#      High = flexible loop, not rigid core.
#   7. Gap frequency at j: Same for the other position.
#   8. Neff: How many similar sequences did we find?
#      More sequences = more reliable MSA signal.
#
# ESTIMATED RUNTIME: ~2.5-3 hours on T4 GPU
# ============================================================


# ============================================================
# CELL 0: USER-CONFIGURABLE PATHS (OPTIONAL OVERRIDES)
# ============================================================
EXTENDED_SEQ_CSV = None
EXTENDED_COORD_CSV = None
OUTPUT_DIR = '/kaggle/working'
COMP_BASE_PRIMARY = '/kaggle/input/competitions/stanford-rna-3d-folding-2/'
COMP_BASE_FALLBACK = '/kaggle/input/stanford-rna-3d-folding-2/'

# --- RUN 4 NEW: MSA configuration ---
MSA_TOP_N = 20    # Number of similar sequences to collect for MSA
                  # More = better MSA signal but slower search (~15 sec vs ~5 sec per target)
MSA_DIM = 8       # Number of MSA feature channels (do not change)


# ============================================================
# CELL 1: Install biopython (offline wheel install)
# ============================================================
import subprocess, sys, glob, os
import shutil

py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"
print(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} (wheel tag: {py_ver})")

try:
    import Bio
    print(f"Biopython {Bio.__version__} already installed")
except ImportError:
    wheels = sorted(glob.glob('/kaggle/input/**/biopython*.whl', recursive=True))
    print(f"Found wheels: {wheels}")
    clean_wheels = []
    for whl in wheels:
        basename = os.path.basename(whl)
        clean_name = basename.replace(' (1)', '').replace(' (2)', '').replace(' (3)', '')
        dest = f'{OUTPUT_DIR}/{clean_name}'
        if not os.path.exists(dest):
            shutil.copy2(whl, dest)
        clean_wheels.append(dest)

    if clean_wheels:
        matching = [w for w in clean_wheels if py_ver in w]
        chosen = matching[0] if matching else clean_wheels[-1]
        print(f"Installing: {chosen}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', chosen, '-q'])
    else:
        print("No wheel found, trying pip install...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'biopython', '-q'])
    import Bio
    print(f"Biopython {Bio.__version__} installed successfully")


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
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")


# ============================================================
# CELL 3: Find all data paths (auto-discovery)
# ============================================================
import os

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
ADV1_WEIGHTS = None        # BASIC's best_model.pt (fallback warm-start)
RUN3_CHECKPOINT = None     # Run 3 OptB's adv1_best_model.pt (preferred warm-start)
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f == "RibonanzaNet.pt":
            BACKBONE_WEIGHTS = os.path.join(root, f)
        if f == "best_model.pt":
            ADV1_WEIGHTS = os.path.join(root, f)
        # --- RUN 4 NEW: look for Run 3 checkpoint ---
        if f == "adv1_best_model.pt":
            RUN3_CHECKPOINT = os.path.join(root, f)
print(f"Backbone weights: {BACKBONE_WEIGHTS}")
print(f"BASIC distance head: {ADV1_WEIGHTS}")
print(f"Run 3 checkpoint: {RUN3_CHECKPOINT}")

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
print(f"Extended sequences CSV: {EXTENDED_SEQ_CSV}")
print(f"Extended coordinates CSV: {EXTENDED_COORD_CSV}")

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
# Same as Run 3 OptB. No changes needed.
# find_similar_sequences() already accepts top_n parameter.
# ============================================================

def adaptive_rna_constraints(coordinates, sequence, confidence=1.0):
    refined_coords = coordinates.copy()
    n_residues = len(sequence)
    constraint_strength = 0.8 * (1.0 - min(confidence, 0.8))
    seq_min_dist, seq_max_dist = 5.5, 6.5
    for i in range(n_residues - 1):
        current_dist = np.linalg.norm(refined_coords[i+1] - refined_coords[i])
        if current_dist < seq_min_dist or current_dist > seq_max_dist:
            target_dist = (seq_min_dist + seq_max_dist) / 2
            direction = refined_coords[i+1] - refined_coords[i]
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            adjustment = (target_dist - current_dist) * constraint_strength
            refined_coords[i+1] = refined_coords[i] + direction * (current_dist + adjustment)
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
    query_seq_obj = Seq(query_seq)
    candidates = []
    k = 3
    q_kmers = set(query_seq[i:i+k] for i in range(len(query_seq)-k+1))
    for _, row in train_seqs_df.iterrows():
        tid = row['target_id']
        tseq = row['sequence']
        if tid not in train_coords_dict: continue
        lr = abs(len(tseq)-len(query_seq)) / max(len(tseq),len(query_seq))
        if lr > 0.4: continue
        t_kmers = set(tseq[i:i+k] for i in range(len(tseq)-k+1))
        score = len(q_kmers & t_kmers) / len(q_kmers | t_kmers) if q_kmers | t_kmers else 0
        candidates.append((tid, tseq, score, train_coords_dict[tid]))
    candidates.sort(key=lambda x: x[2], reverse=True)
    candidates = candidates[:100]
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
# CELL 9: Template search + collect MSA sequences (CHANGED)
# ============================================================
# RUN 4 CHANGE: Now collects top-20 similar sequences per target
# (was top-1 in Run 3). The best match is still used for template
# coordinates. All 20 matches are stored for MSA computation
# in Cell 9.5.
#
# SIMPLE EXPLANATION:
#   Before: "Find the ONE most similar RNA for each test target."
#   Now:    "Find the 20 most similar RNAs. Use the best one as a
#            template (same as before). Use all 20 to figure out
#            which positions co-evolve (MSA features)."
# ============================================================
print("\n" + "="*60)
print("PHASE 1: Fork 2 Template Search + MSA Collection")
print("="*60)

template_coords_per_target = {}
template_confidence_per_target = {}
msa_hits_per_target = {}              # RUN 4 NEW: store all hits for MSA

start_time = time.time()
for idx, row in test_seqs.iterrows():
    target_id = row['target_id']
    sequence = row['sequence']
    if idx % 5 == 0:
        elapsed = time.time() - start_time
        print(f"  Template search {idx+1}/{len(test_seqs)}: {target_id} ({len(sequence)} nt) [{elapsed:.0f}s]")

    # RUN 4 CHANGE: collect MSA_TOP_N hits (was top_n=1)
    similar = find_similar_sequences(sequence, train_seqs_extended, train_coords_dict, top_n=MSA_TOP_N)

    # Store ALL hits for MSA computation in Cell 9.5
    msa_hits_per_target[target_id] = similar   # RUN 4 NEW

    if similar:
        best_tid, best_seq, best_score, best_coords = similar[0]
        adapted = adapt_template_to_query(sequence, best_seq, best_coords)
        refined = adaptive_rna_constraints(adapted, sequence, confidence=best_score)
        template_coords_per_target[target_id] = refined
        template_confidence_per_target[target_id] = best_score
        print(f"    -> Template: {best_tid} (score={best_score:.3f}), MSA hits: {len(similar)}")
    else:
        template_coords_per_target[target_id] = np.zeros((len(sequence), 3))
        template_confidence_per_target[target_id] = 0.0
        msa_hits_per_target[target_id] = []
        print(f"    -> No template found, no MSA hits")

n_with_tmpl = sum(1 for v in template_confidence_per_target.values() if v > 0)
n_with_msa = sum(1 for v in msa_hits_per_target.values() if len(v) >= 3)
print(f"\nTemplate search complete. {n_with_tmpl}/{len(test_seqs)} targets have templates.")
print(f"MSA collection complete. {n_with_msa}/{len(test_seqs)} targets have 3+ MSA hits.")


# ============================================================
# CELL 9.5: Compute MSA features from collected sequences (NEW)
# ============================================================
# This is entirely new in Run 4. It takes the similar sequences
# collected in Cell 9 and computes 8 evolutionary feature channels
# for each test target.
#
# SIMPLE EXPLANATION:
#   We have 20 "cousin" sequences for each test RNA. We line them
#   all up (align them). Then we ask two questions:
#     1. At each position: how much does it change across cousins?
#        (conservation and gap frequency)
#     2. For each pair of positions: do they always change together?
#        (covariation = evolutionary evidence of 3D proximity)
#
# THE 8 CHANNELS:
#   1. Covariation(i,j): mutual information between positions i,j
#   2. APC-corrected covariation: removes background noise
#   3. Conservation(i): Shannon entropy at position i (broadcast)
#   4. Conservation(j): Shannon entropy at position j (broadcast)
#   5. Conservation(i) × Conservation(j): both conserved?
#   6. Gap frequency(i): how often is position i missing? (broadcast)
#   7. Gap frequency(j): how often is position j missing? (broadcast)
#   8. Neff: number of effective sequences (scalar, broadcast)
# ============================================================
print("\n" + "="*60)
print("PHASE 1.5: MSA Feature Computation (RUN 4 NEW)")
print("="*60)

def compute_msa_features(query_seq, similar_hits, max_len=512):
    """Compute 8-channel MSA features from aligned similar sequences.

    Args:
        query_seq: the target RNA sequence (string)
        similar_hits: list of (tid, seq, score, coords) from find_similar_sequences
        max_len: truncate to this length (for VRAM)

    Returns:
        numpy array of shape (N, N, 8) where N = min(len(query_seq), max_len)

    STEP-BY-STEP (high school level):
      1. Align each cousin sequence to the query
      2. Build a matrix: rows = cousins, columns = positions
      3. At each position, count A/C/G/U/gap frequencies
      4. Conservation = how uniform the frequencies are (low entropy = conserved)
      5. Gap frequency = fraction of cousins with a gap at that position
      6. Covariation = for each pair (i,j), do the bases co-vary?
         If knowing position i tells you about position j, they co-evolve.
      7. APC correction = subtract background to remove noise
    """
    N = min(len(query_seq), max_len)
    n_channels = 8

    # If no hits, return zeros
    if not similar_hits or len(similar_hits) < 2:
        return np.zeros((N, N, n_channels), dtype=np.float32)

    # --- Step 1: Build alignment matrix ---
    # Each row is a sequence aligned to the query. Columns = query positions.
    # Values: 0=A, 1=C, 2=G, 3=U, 4=gap
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3, '-': 4}
    n_seqs = len(similar_hits) + 1  # hits + query itself
    aln_matrix = np.full((n_seqs, N), 4, dtype=np.int32)  # default = gap

    # Row 0 = query sequence
    for i in range(N):
        aln_matrix[0, i] = base_to_idx.get(query_seq[i].upper(), 4)

    # Rows 1..K = aligned hit sequences
    for hit_idx, (tid, hit_seq, score, coords) in enumerate(similar_hits):
        # Align hit to query
        try:
            alns = pairwise2.align.globalms(
                Seq(query_seq[:N]), hit_seq, 2.9, -1, -10, -0.5,
                one_alignment_only=True
            )
            if not alns:
                continue
            aligned_query = alns[0].seqA
            aligned_hit = alns[0].seqB

            # Map aligned positions back to query positions
            qi = 0
            for k in range(len(aligned_query)):
                if qi >= N:
                    break
                if aligned_query[k] != '-':
                    if aligned_hit[k] != '-':
                        aln_matrix[hit_idx + 1, qi] = base_to_idx.get(aligned_hit[k].upper(), 4)
                    # else: gap in hit at this query position (stays as 4)
                    qi += 1
        except:
            continue

    # --- Step 2: Per-position features ---
    n_bases = 5  # A, C, G, U, gap

    # Frequency matrix: (N, 5) — fraction of each base at each position
    freq = np.zeros((N, n_bases), dtype=np.float32)
    for pos in range(N):
        counts = np.bincount(aln_matrix[:, pos], minlength=n_bases).astype(np.float32)
        freq[pos] = counts / n_seqs

    # Conservation: 1 - normalized Shannon entropy
    # High conservation = position rarely changes = structurally important
    eps = 1e-10
    entropy = -np.sum(freq * np.log(freq + eps), axis=1)  # (N,)
    max_entropy = np.log(n_bases)
    conservation = 1.0 - (entropy / max_entropy)  # 0=variable, 1=conserved

    # Gap frequency per position
    gap_freq = freq[:, 4]  # (N,)

    # Neff: effective number of sequences (weighted by similarity)
    neff = float(n_seqs) / 100.0  # Normalize to small range

    # --- Step 3: Pairwise covariation (mutual information) ---
    # For each pair of positions (i,j), compute how much knowing
    # the base at i tells you about the base at j.
    # MI(i,j) = sum over all base pairs: P(a,b) * log(P(a,b) / P(a)*P(b))

    # Joint frequency: (N, N, 5, 5) — but that's too big for large N.
    # Use a simplified approach: compute MI only for 4 real bases (not gaps)
    n_real = 4  # A, C, G, U (exclude gaps for MI)
    freq_real = freq[:, :4]  # (N, 4)
    freq_real_sum = freq_real.sum(axis=1, keepdims=True)
    freq_real_norm = freq_real / (freq_real_sum + eps)  # Normalize excluding gaps

    # Compute MI using vectorized approach
    mi = np.zeros((N, N), dtype=np.float32)

    # Joint frequency table for each pair (i,j)
    for si in range(n_seqs):
        row = aln_matrix[si, :]  # (N,) — bases for this sequence
        for sj in range(si, n_seqs):
            # Count co-occurrences (skip gaps)
            pass  # Too slow for N>100, use batch approach below

    # Faster MI computation: build co-occurrence counts
    # For each pair of positions (i,j), count how often base_a appears at i
    # AND base_b appears at j across all sequences in the alignment.
    joint_counts = np.zeros((N, N, n_real, n_real), dtype=np.float32)
    for si in range(n_seqs):
        row = aln_matrix[si, :]
        for i in range(N):
            if row[i] >= n_real:
                continue  # Skip gaps at position i
            for j in range(i, N):
                if row[j] >= n_real:
                    continue  # Skip gaps at position j
                joint_counts[i, j, row[i], row[j]] += 1
                if i != j:
                    joint_counts[j, i, row[j], row[i]] += 1

    # Normalize to joint probabilities
    pair_totals = joint_counts.sum(axis=(2, 3), keepdims=True)
    joint_prob = joint_counts / (pair_totals + eps)  # (N, N, 4, 4)

    # Marginals from joint (more accurate than from freq_real_norm)
    marg_i = joint_prob.sum(axis=3)  # (N, N, 4) — P(base_a at i | pair i,j)
    marg_j = joint_prob.sum(axis=2)  # (N, N, 4) — P(base_b at j | pair i,j)

    # MI(i,j) = sum_{a,b} P(a,b) * log(P(a,b) / (P(a) * P(b)))
    outer_prod = marg_i[:, :, :, None] * marg_j[:, :, None, :]  # (N,N,4,4)
    log_ratio = np.log((joint_prob + eps) / (outer_prod + eps))
    mi = (joint_prob * log_ratio).sum(axis=(2, 3))  # (N, N)
    mi = np.maximum(mi, 0.0)  # MI is non-negative

    # --- Step 4: APC correction ---
    # Average Product Correction removes background covariation noise.
    # APC(i,j) = MI(i,j) - (MI(i,.) * MI(.,j)) / MI(.,.)
    # This makes the signal cleaner and more specific to true contacts.
    mi_row_mean = mi.mean(axis=1)          # (N,)
    mi_col_mean = mi.mean(axis=0)          # (N,)
    mi_global_mean = mi.mean() + eps       # scalar
    apc = mi_row_mean[:, None] * mi_col_mean[None, :] / mi_global_mean
    mi_apc = np.maximum(mi - apc, 0.0)    # (N, N)

    # --- Step 5: Assemble 8-channel feature tensor ---
    features = np.zeros((N, N, n_channels), dtype=np.float32)
    features[:, :, 0] = mi                                          # Raw covariation
    features[:, :, 1] = mi_apc                                      # APC-corrected covariation
    features[:, :, 2] = conservation[:, None] * np.ones((1, N))      # Conservation at i (broadcast)
    features[:, :, 3] = np.ones((N, 1)) * conservation[None, :]      # Conservation at j (broadcast)
    features[:, :, 4] = conservation[:, None] * conservation[None, :] # Conservation product
    features[:, :, 5] = gap_freq[:, None] * np.ones((1, N))          # Gap freq at i (broadcast)
    features[:, :, 6] = np.ones((N, 1)) * gap_freq[None, :]          # Gap freq at j (broadcast)
    features[:, :, 7] = neff                                         # Neff (scalar broadcast)

    return features


# --- Compute MSA features for all test targets ---
msa_features_per_target = {}    # target_id -> (N, N, 8) numpy array
msa_start = time.time()

for idx, row in test_seqs.iterrows():
    target_id = row['target_id']
    sequence = row['sequence']
    hits = msa_hits_per_target.get(target_id, [])

    msa_feat = compute_msa_features(sequence, hits, max_len=512)
    msa_features_per_target[target_id] = msa_feat

    if idx % 5 == 0:
        n_hits = len(hits)
        elapsed = time.time() - msa_start
        print(f"  MSA features {idx+1}/{len(test_seqs)}: {target_id} "
              f"({msa_feat.shape[0]}x{msa_feat.shape[0]}x{msa_feat.shape[2]}), "
              f"{n_hits} hits, [{elapsed:.0f}s]")

print(f"\nMSA feature computation complete in {time.time()-msa_start:.0f}s")
n_nonzero = sum(1 for v in msa_features_per_target.values() if v.max() > 0)
print(f"  {n_nonzero}/{len(test_seqs)} targets have non-zero MSA features")


# ============================================================
# CELL 10: Define ADV1 Neural Network Components (INLINED)
# ============================================================
# Same as Run 3 OptB. No changes to TemplateEncoder,
# DistanceMatrixHead, mds_reconstruct, or refine_coords.
# pair_dim is set in Cell 12 when creating the model.
# ============================================================
print("\n" + "="*60)
print("PHASE 2: ADV1 Neural Network Setup")
print("="*60)

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

class DistanceMatrixHead(nn.Module):
    def __init__(self, pair_dim=88, hidden_dim=128, num_layers=3, dropout=0.1):
        """RUN 4 CHANGE: pair_dim default is now 88 (was 80).
        64 backbone + 16 template + 8 MSA = 88."""
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
        dist = (dist + dist.transpose(-1, -2)) / 2.0
        B, N, _ = dist.shape
        mask = torch.eye(N, device=dist.device, dtype=torch.bool).unsqueeze(0)
        dist = dist.masked_fill(mask, 0.0)
        return dist

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
# Identical to Run 3 OptB. No changes.
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

state = torch.load(BACKBONE_WEIGHTS, map_location='cpu', weights_only=False)
if isinstance(state, dict) and 'model_state_dict' in state:
    state = state['model_state_dict']
backbone.load_state_dict(state, strict=False)
print(f"  Backbone loaded from {BACKBONE_WEIGHTS}")

backbone = backbone.to(device)

for p in backbone.parameters():
    p.requires_grad = False

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

def get_pairwise_features(sequence):
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

def get_pairwise_features_train(sequence):
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
        frozen_layer_count = total_layers - UNFREEZE_LAST_N
        for i, layer in enumerate(backbone.transformer_encoder):
            if i < frozen_layer_count:
                result = layer(hidden, pairwise, src_mask=src_mask)
                if isinstance(result, tuple):
                    hidden, pairwise = result
                else:
                    hidden = result
    hidden = hidden.detach().requires_grad_(True)
    pairwise = pairwise.detach().requires_grad_(True)
    for i, layer in enumerate(backbone.transformer_encoder):
        if i >= frozen_layer_count:
            result = layer(hidden, pairwise, src_mask=src_mask)
            if isinstance(result, tuple):
                hidden, pairwise = result
            else:
                hidden = result
    return pairwise


# ============================================================
# CELL 12: Create ADV1 model + warm-start from Run 3 OptB (CHANGED)
# ============================================================
# RUN 4 CHANGE: pair_dim is now 88 (was 80 in Run 3).
#   64 (backbone) + 16 (template) + 8 (MSA) = 88
#
# Warm-start priority:
#   1. Run 3 OptB checkpoint (adv1_best_model.pt) — preferred
#      Expands first layer from 80 -> 88 columns (8 new MSA channels)
#   2. BASIC checkpoint (best_model.pt) — fallback
#      Expands first layer from 64 -> 88 columns
#
# SIMPLE EXPLANATION:
#   The distance head is like a calculator that takes 80 numbers
#   as input and predicts one distance. Now we're giving it 88
#   numbers instead (80 old + 8 new MSA). We copy the old weights
#   for the first 80 inputs and start the 8 new inputs at zero,
#   so the calculator starts exactly where Run 3 left off and
#   gradually learns to use the new MSA information.
# ============================================================
PAIR_DIM = 64 + 16 + MSA_DIM  # 88 total

print(f"\nCreating ADV1 model (pair_dim={PAIR_DIM})...")
template_encoder = TemplateEncoder(template_dim=16, num_bins=22, max_dist=40.0).to(device)
distance_head = DistanceMatrixHead(pair_dim=PAIR_DIM, hidden_dim=128, num_layers=3, dropout=0.1).to(device)

# Warm-start: prefer Run 3 checkpoint, fall back to BASIC
warmstart_path = RUN3_CHECKPOINT if RUN3_CHECKPOINT else ADV1_WEIGHTS
warmstart_source = "Run 3 OptB" if RUN3_CHECKPOINT else "BASIC"

if warmstart_path and os.path.exists(warmstart_path):
    print(f"  Warm-starting from {warmstart_source}: {warmstart_path}")
    ckpt = torch.load(warmstart_path, map_location=device, weights_only=False)

    # Load distance head weights (with expansion)
    if 'model_state_dict' in ckpt:
        source_state = ckpt['model_state_dict']
    else:
        source_state = ckpt

    adv1_state = distance_head.state_dict()
    for key in adv1_state:
        if key in source_state:
            bp = source_state[key]
            ap = adv1_state[key]
            if bp.shape == ap.shape:
                adv1_state[key] = bp
            elif len(bp.shape) == 2 and len(ap.shape) == 2 and \
                 bp.shape[0] == ap.shape[0] and bp.shape[1] < ap.shape[1]:
                print(f"    Expanding {key}: {bp.shape} -> {ap.shape}")
                adv1_state[key][:, :bp.shape[1]] = bp
    distance_head.load_state_dict(adv1_state)

    # Load template encoder weights (if present in checkpoint)
    if 'template_encoder_state_dict' in ckpt:
        template_encoder.load_state_dict(ckpt['template_encoder_state_dict'])
        print("  Loaded template encoder from checkpoint")

    # Load unfrozen backbone layers (if present in checkpoint)
    if 'backbone_unfrozen_state' in ckpt:
        for layer_key, layer_state in ckpt['backbone_unfrozen_state'].items():
            layer_idx = int(layer_key.split('.')[-1])
            backbone.transformer_encoder[layer_idx].load_state_dict(layer_state)
        print("  Loaded unfrozen backbone layers from checkpoint")

    print(f"  Warm-start complete (source pair_dim -> {PAIR_DIM})")
else:
    print("  No checkpoint found — training from scratch")

dh_params = sum(p.numel() for p in distance_head.parameters())
te_params = sum(p.numel() for p in template_encoder.parameters())
print(f"  Distance head params: {dh_params:,}")
print(f"  Template encoder params: {te_params:,}")
print(f"  Unfrozen backbone params: {bb_unfrozen:,}")
print(f"  TOTAL trainable params: {dh_params + te_params + bb_unfrozen:,}")


# ============================================================
# CELL 13: Train ADV1 on pdb_xyz_data.pkl (CHANGED)
# ============================================================
# RUN 4 CHANGE: Training now includes MSA features.
#   - MSA features for training sequences are MASKED (set to zero)
#     with MSA_MASK_PROB probability during training
#   - When not masked, a quick "self-MSA" is computed from the
#     training sequence's base-pair potential as a proxy
#   - This teaches the model to use MSA features when available
#     but not depend on them (same pattern as template masking)
#
# SIMPLE EXPLANATION:
#   During training, we sometimes give the model the MSA channels
#   (filled with base-pair potential information) and sometimes
#   set them to zero. This way the model learns to use MSA when
#   it's available but doesn't break when it's missing.
#
# The concatenation is now: backbone(64) + template(16) + msa(8) = 88
# ============================================================
print("\n" + "="*60)
print("PHASE 3: ADV1 Training (with MSA features)")
print("="*60)

TRAIN_EPOCHS = 30
BATCH_SIZE = 2
MAX_SEQ_LEN = 256
HEAD_LR = 5e-5
BACKBONE_LR = 1e-5           # Adjust this if Run 3 val_loss didn't improve
TEMPLATE_MASK_PROB = 0.5
MSA_MASK_PROB = 0.5           # RUN 4 NEW: 50% chance of masking MSA during training

# --- Load and parse training data (same as Run 3 OptB) ---
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
            c1_coords = []
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
    print(f"  Pickle is a list with {len(raw_data)} items")
    train_items = []
    for item in raw_data:
        seq = item.get('sequence', '')
        coords = item.get('coordinates', item.get('coords', None))
        if seq and coords is not None and len(seq) <= MAX_SEQ_LEN and len(seq) == len(coords):
            train_items.append({'sequence': seq, 'coords': np.array(coords, dtype=np.float32)})
    print(f"  After filtering: {len(train_items)} structures")

print(f"  Final training set: {len(train_items)} structures")

random.seed(42)
random.shuffle(train_items)
val_size = max(1, int(len(train_items) * 0.1))
val_items = train_items[:val_size]
train_items = train_items[val_size:]
print(f"  Train: {len(train_items)}, Val: {val_size}")


# --- RUN 4 NEW: Helper to generate training MSA proxy ---
def make_training_msa_proxy(seq, N):
    """Generate a simple MSA proxy for training sequences.
    Uses base-pair potential (Watson-Crick complementarity) as a
    cheap approximation of covariation. Not as good as real MSA,
    but teaches the model to use the MSA channels.

    Returns: (N, N, 8) numpy array."""
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    # Base-pair potential: A-U=1, G-C=1, G-U=0.5
    bp_pairs = {(0,3): 1.0, (3,0): 1.0,  # A-U
                (1,2): 1.0, (2,1): 1.0,  # C-G
                (2,3): 0.5, (3,2): 0.5}  # G-U wobble

    bp_matrix = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        bi = base_to_idx.get(seq[i].upper(), -1)
        if bi < 0: continue
        for j in range(i+1, N):
            bj = base_to_idx.get(seq[j].upper(), -1)
            if bj < 0: continue
            score = bp_pairs.get((bi, bj), 0.0)
            bp_matrix[i, j] = score
            bp_matrix[j, i] = score

    # Build 8 channels (approximations)
    features = np.zeros((N, N, MSA_DIM), dtype=np.float32)
    features[:, :, 0] = bp_matrix * 0.3       # Pseudo-covariation
    features[:, :, 1] = bp_matrix * 0.2       # Pseudo-APC
    # Channels 2-7: leave at zero (no real MSA available for training seqs)
    # Channel 7 (neff): 0.01 to indicate "weak MSA signal"
    features[:, :, 7] = 0.01
    return features


# --- Optimizer: discriminative learning rate ---
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

            true_dist = np.sqrt(((true_coords[:, None] - true_coords[None, :])**2).sum(-1))
            true_dist_t = torch.tensor(true_dist, dtype=torch.float32, device=device).unsqueeze(0)

            pairwise = get_pairwise_features_train(seq)  # (1, N, N, 64)

            # Template features (50% masked)
            if random.random() < TEMPLATE_MASK_PROB:
                tmpl_feat = torch.zeros(1, N, N, 16, device=device)
            else:
                coords_t = torch.tensor(true_coords, dtype=torch.float32, device=device)
                tmpl_feat = template_encoder(coords_t, confidence=1.0, has_template=True)
                tmpl_feat = tmpl_feat.unsqueeze(0)

            # --- RUN 4 NEW: MSA features (50% masked) ---
            if random.random() < MSA_MASK_PROB:
                msa_feat = torch.zeros(1, N, N, MSA_DIM, device=device)
            else:
                msa_np = make_training_msa_proxy(seq, N)
                msa_feat = torch.tensor(msa_np, dtype=torch.float32, device=device).unsqueeze(0)

            # RUN 4 CHANGE: concatenate all three feature sources
            # backbone(64) + template(16) + msa(8) = 88
            combined = torch.cat([pairwise, tmpl_feat, msa_feat], dim=-1)  # (1, N, N, 88)
            pred_dist = distance_head(combined)

            loss = ((pred_dist - true_dist_t)**2).mean()

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

    # --- Validation ---
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
            # RUN 4 CHANGE: include MSA proxy in validation too
            msa_np = make_training_msa_proxy(seq, N)
            msa_feat_t = torch.tensor(msa_np, dtype=torch.float32, device=device).unsqueeze(0)
            combined = torch.cat([pairwise, tmpl_feat, msa_feat_t], dim=-1)  # (1, N, N, 88)
            pred_dist = distance_head(combined)
            val_loss += ((pred_dist - true_dist_t)**2).mean().item()
    avg_val = val_loss / max(len(val_items[:20]), 1)

    lrs = scheduler.get_last_lr()
    elapsed = time.time() - train_start
    print(f"Epoch {epoch+1}/{TRAIN_EPOCHS}: train={avg_train:.4f}, val={avg_val:.4f}, "
          f"lr_head={lrs[0]:.6f}, lr_bb={lrs[1]:.7f}, time={elapsed:.0f}s")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save({
            'model_state_dict': distance_head.state_dict(),
            'template_encoder_state_dict': template_encoder.state_dict(),
            'backbone_unfrozen_state': {
                f'transformer_encoder.{i}': backbone.transformer_encoder[i].state_dict()
                for i in range(total_layers - UNFREEZE_LAST_N, total_layers)
            },
            'epoch': epoch, 'val_loss': avg_val,
            'pair_dim': PAIR_DIM     # RUN 4 NEW: save pair_dim for future warm-starts
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
# CELL 14: ADV1 Inference — Hybrid + Template-Seeded + MSA (CHANGED)
# ============================================================
# RUN 4 CHANGE: MSA features are now included in the distance
# prediction. The concatenation is backbone(64) + template(16)
# + msa(8) = 88, matching the training pair_dim.
#
# MSA features come from Cell 9.5 (pre-computed for each test target
# from the 20 similar sequences found in Cell 9).
#
# Everything else (hybrid slots, template-seeded refinement,
# diversity settings) is identical to Run 3 OptB.
# ============================================================
print("\n" + "="*60)
print("PHASE 4: ADV1 Inference (Hybrid + Template-Seeded + MSA)")
print("="*60)

MAX_INFER_LEN = 512
HYBRID_THRESHOLD = 0.3

all_predictions = []
infer_start = time.time()

n_hybrid_template = 0
n_hybrid_nn_only = 0
n_template_seeded = 0
n_mds_fallback = 0

for idx, row in test_seqs.iterrows():
    target_id = row['target_id']
    sequence = row['sequence']
    seq = sequence[:MAX_INFER_LEN]
    N = len(seq)

    if idx % 5 == 0:
        print(f"  Predicting {idx+1}/{len(test_seqs)}: {target_id} ({len(sequence)} nt)")

    with torch.no_grad():
        pairwise = get_pairwise_features(seq)   # (1, N, N, 64)

        tmpl_coords = template_coords_per_target.get(target_id, np.zeros((N, 3)))
        tmpl_conf = template_confidence_per_target.get(target_id, 0.0)

        if len(tmpl_coords) > N:
            tmpl_coords = tmpl_coords[:N]
        elif len(tmpl_coords) < N:
            padded = np.zeros((N, 3))
            padded[:len(tmpl_coords)] = tmpl_coords
            tmpl_coords = padded

        coords_t = torch.tensor(tmpl_coords, dtype=torch.float32, device=device)
        has_tmpl = tmpl_conf > 0.01
        tmpl_feat = template_encoder(coords_t, confidence=tmpl_conf, has_template=has_tmpl)
        tmpl_feat = tmpl_feat.unsqueeze(0)   # (1, N, N, 16)

        # --- RUN 4 NEW: Get pre-computed MSA features for this target ---
        msa_np = msa_features_per_target.get(target_id, np.zeros((N, N, MSA_DIM)))
        # Pad/truncate MSA features to match N
        if msa_np.shape[0] > N:
            msa_np = msa_np[:N, :N, :]
        elif msa_np.shape[0] < N:
            padded = np.zeros((N, N, MSA_DIM), dtype=np.float32)
            m = msa_np.shape[0]
            padded[:m, :m, :] = msa_np
            msa_np = padded
        msa_feat = torch.tensor(msa_np, dtype=torch.float32, device=device).unsqueeze(0)  # (1, N, N, 8)

        # RUN 4 CHANGE: concatenate all three feature sources
        combined = torch.cat([pairwise, tmpl_feat, msa_feat], dim=-1)  # (1, N, N, 88)
        pred_dist = distance_head(combined).squeeze(0)        # (N, N)

    # --- STEP 2: Generate 5 diverse coordinate predictions ---
    # (identical to Run 3 OptB from here down)

    full_tmpl = template_coords_per_target.get(target_id, np.zeros((len(sequence), 3)))
    if len(full_tmpl) > len(sequence):
        full_tmpl = full_tmpl[:len(sequence)]
    elif len(full_tmpl) < len(sequence):
        padded = np.zeros((len(sequence), 3))
        padded[:len(full_tmpl)] = full_tmpl
        full_tmpl = padded

    nn_diversity = [
        {'noise': 0.0, 'steps': 100, 'seed': 0},
        {'noise': 0.5, 'steps': 100, 'seed': 1},   # RUN 4 TWEAK: increased from 0.3
        {'noise': 1.0, 'steps': 100, 'seed': 2},   # RUN 4 TWEAK: increased from 0.5
        {'noise': 1.5, 'steps': 150, 'seed': 3},   # RUN 4 TWEAK: increased from 0.7
        {'noise': 0.0, 'steps': 50, 'seed': 4},
    ]

    nn_coords_list = []
    for div in nn_diversity:
        torch.manual_seed(div['seed'])

        noisy = pred_dist.clone()
        if div['noise'] > 0:
            noise = torch.randn_like(pred_dist) * div['noise']
            noise = (noise + noise.T) / 2.0
            noise.fill_diagonal_(0.0)
            noisy = torch.clamp(noisy + noise, min=0.0)

        if tmpl_conf > 0.01:
            start_coords = torch.tensor(
                tmpl_coords[:N], dtype=torch.float32, device=device
            )
            n_template_seeded += 1
        else:
            dist_np = noisy.detach().cpu().numpy()
            start_coords = torch.tensor(
                mds_reconstruct(dist_np), dtype=torch.float32, device=device
            )
            n_mds_fallback += 1

        refined = refine_coords(start_coords, noisy.detach(),
                                steps=div['steps'], lr=0.01)
        coords_np = refined.cpu().numpy()

        if len(sequence) > MAX_INFER_LEN:
            remaining = len(sequence) - MAX_INFER_LEN
            last_dir = coords_np[-1] - coords_np[-2] if N >= 2 else np.array([5.9,0,0])
            last_dir = last_dir / (np.linalg.norm(last_dir)+1e-8) * 5.9
            extra = np.array([coords_np[-1] + last_dir*(i+1) for i in range(remaining)])
            coords_np = np.concatenate([coords_np, extra])

        nn_coords_list.append(coords_np)

    coords_list = []

    if tmpl_conf > HYBRID_THRESHOLD:
        n_hybrid_template += 1
        tmpl_refined = adaptive_rna_constraints(full_tmpl.copy(), sequence, confidence=tmpl_conf)
        coords_list.append(tmpl_refined)
        np.random.seed(42)
        small_noise = np.random.normal(0, 0.5, full_tmpl.shape)
        tmpl_noisy = adaptive_rna_constraints(full_tmpl + small_noise, sequence, confidence=tmpl_conf)
        coords_list.append(tmpl_noisy)
        coords_list.append(nn_coords_list[0])
        coords_list.append(nn_coords_list[1])
        coords_list.append(nn_coords_list[2])
        if idx % 5 == 0:
            print(f"    HYBRID: conf={tmpl_conf:.3f} > {HYBRID_THRESHOLD} -> slots 1-2 template, 3-5 NN+MSA")
    else:
        n_hybrid_nn_only += 1
        coords_list = nn_coords_list[:5]
        if idx % 5 == 0:
            seed_type = "template-seeded" if tmpl_conf > 0.01 else "MDS fallback"
            print(f"    HYBRID: conf={tmpl_conf:.3f} <= {HYBRID_THRESHOLD} -> all 5 slots NN+MSA ({seed_type})")

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

print(f"\nInference complete. {len(all_predictions)} rows in {time.time()-infer_start:.0f}s")
print(f"  Targets with template slots (1-2): {n_hybrid_template}")
print(f"  Targets with all-NN slots: {n_hybrid_nn_only}")
print(f"  NN slots using template-seeded refinement: {n_template_seeded}")
print(f"  NN slots using MDS fallback: {n_mds_fallback}")

submission_df = pd.DataFrame(all_predictions)
col_order = ['ID', 'resname', 'resid']
for i in range(1, 6):
    for c in ['x', 'y', 'z']:
        col_order.append(f'{c}_{i}')
submission_df = submission_df[col_order]
submission_df.to_csv(RAW_SUBMISSION_PATH, index=False)
print(f"Raw submission: {len(submission_df)} rows")


# ============================================================
# CELL 15: Option B Post-Processing (UNCHANGED)
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
