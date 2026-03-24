# ============================================================
# HY-BAS-ADV1 RUN 3 Option B v2: Fix 1-3 Validation Run
# ============================================================
#
# WHAT THIS IS:
#   Run 3 Option B v2 = Run 3 OptB + three inference fixes.
#   Training is SKIPPED — uses pre-trained checkpoint from Run 3 OptB
#   (val_loss=129.2, uploaded as Kaggle dataset).
#   Only the inference cell (Cell 14) is changed.
#
# WHY v2:
#   Run 3 OptB scored 0.251 (below Fork 2's 0.287).
#   ROOT CAUSE: For the 12 weak-template targets (conf <= 0.3),
#   ALL 5 slots were NN-refined — no raw template in any slot.
#   NN refinement (100 steps toward distances with ~11A error)
#   pushed good template starting coords AWAY from correct positions.
#   Fork 2 scored 0.287 by using raw templates for ALL targets.
#
# THREE FIXES (all in Cell 14, inference only):
#
#   FIX 1 (CRITICAL): Always put raw template in Slot 1
#     - For ALL targets regardless of confidence
#     - Guarantees at least Fork 2 quality in one slot
#     - Best-of-5 scoring means raw template carries if NN is worse
#     - This is the main fix — addresses the root cause directly
#
#   FIX 2: Reduce NN refinement steps from 100 to 30
#     - Less gradient descent = less drift from good template coords
#     - NN slots stay closer to template starting point
#     - Operates on different slots than Fix 1's Slot 1
#
#   FIX 3: Lower HYBRID_THRESHOLD from 0.3 to 0.20
#     - More targets get template protection in Slots 1-2
#     - Borderline targets (conf 0.20-0.30) now get 2 template slots
#     - Mostly redundant with Fix 1 but provides extra safety
#
# TRAINING SKIP:
#   Cell 13 detects pre-trained checkpoint (adv1_best_model.pt)
#   uploaded as Kaggle dataset and skips the 30-epoch training loop.
#   This cuts runtime from ~39 min to ~19 min.
#
# PRESERVED FROM RUN 3 Option B (everything except Cell 14 fixes):
#   Change 1: TRAIN_EPOCHS = 30 (skipped via checkpoint)
#   Change 2: Unfreeze backbone layers 7,8
#   Change 3: Biopython offline wheel install
#   Change 4: Fixed pickle parsing
#   Change 5: Hybrid inference (modified by fixes)
#   Change 6: Template-seeded refinement (modified by Fix 2)
#
# ESTIMATED RUNTIME: ~19 min on T4 GPU (training skipped)
# NEW DATASET NEEDED: Upload adv1_best_model.pt as "adv1-run3-checkpoint"
# ============================================================


# ============================================================
# CELL 0: USER-CONFIGURABLE PATHS (OPTIONAL OVERRIDES)
# ============================================================
EXTENDED_SEQ_CSV = None
EXTENDED_COORD_CSV = None
OUTPUT_DIR = '/kaggle/working'
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
    bio_inits = glob.glob('/kaggle/input/**/Bio/__init__.py', recursive=True)
    print(f"Found Bio dirs: {bio_inits}")
    matching = [p for p in bio_inits if py_ver in p]
    chosen = matching[0] if matching else (bio_inits[0] if bio_inits else None)
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
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ============================================================
# CELL 3: Find all data paths (auto-discovery)
# ============================================================
# CHANGED IN v2: Also searches for Run 3 OptB checkpoint
# (adv1_best_model.pt) uploaded as a separate Kaggle dataset.
# If found, Cell 13 will skip training and use this checkpoint.
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
ADV1_WEIGHTS = None
# --- v2 NEW: Search for Run 3 OptB pre-trained checkpoint ---
# This is the trained model from Run 3 OptB (val_loss=129.2).
# If found, Cell 13 skips the 30-epoch training loop entirely.
# Handles filenames with/without space before .pt (Kaggle quirk).
RUN3_CHECKPOINT = None
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f == "RibonanzaNet.pt":
            BACKBONE_WEIGHTS = os.path.join(root, f)
        if f == "best_model.pt":
            ADV1_WEIGHTS = os.path.join(root, f)
        # Look for the Run 3 OptB checkpoint (uniquely named to avoid confusion)
        # Handles filename with/without space before .pt (Kaggle download quirk)
        if f.strip() in ("adv1_best_run3optb_model.pt",):
            RUN3_CHECKPOINT = os.path.join(root, f)
print(f"Backbone weights: {BACKBONE_WEIGHTS}")
print(f"BASIC distance head: {ADV1_WEIGHTS}")
print(f"Run 3 OptB checkpoint: {RUN3_CHECKPOINT}")   # v2 NEW

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
# CELL 8: Fork 2's template functions (UNCHANGED)
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
# CELL 9: Run Fork 2 template search (UNCHANGED)
# ============================================================
print("\n" + "="*60)
print("PHASE 1: Fork 2 Template Search")
print("="*60)

template_coords_per_target = {}
template_confidence_per_target = {}

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
# CELL 10: Define ADV1 Neural Network Components (UNCHANGED)
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
# CELL 11: Load RibonanzaNet Backbone (UNCHANGED)
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
# CELL 12: Create ADV1 model + warm-start (UNCHANGED)
# ============================================================
print("\nCreating ADV1 model...")
template_encoder = TemplateEncoder(template_dim=16, num_bins=22, max_dist=40.0).to(device)
distance_head = DistanceMatrixHead(pair_dim=80, hidden_dim=128, num_layers=3, dropout=0.1).to(device)

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
                adv1_state[key] = bp
            elif len(bp.shape) == 2 and len(ap.shape) == 2 and \
                 bp.shape[0] == ap.shape[0] and bp.shape[1] < ap.shape[1]:
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
# CELL 13: Train ADV1 — WITH SKIP LOGIC (CHANGED IN v2)
# ============================================================
# v2 CHANGE: If a pre-trained Run 3 OptB checkpoint is found
# (uploaded as Kaggle dataset), skip the entire 30-epoch training
# loop and load the checkpoint directly. This saves ~19.5 minutes.
#
# The checkpoint contains:
#   - distance_head weights (val_loss=129.2)
#   - template_encoder weights
#   - backbone unfrozen layers 7-8 weights
#
# If no checkpoint is found, training proceeds normally (same as Run 3 OptB).
# ============================================================
print("\n" + "="*60)
print("PHASE 3: ADV1 Training (with backbone unfreezing)")
print("="*60)

# --- v2 NEW: Check for pre-trained checkpoint and skip training ---
if RUN3_CHECKPOINT is not None:
    print(f"  Found pre-trained Run 3 OptB checkpoint: {RUN3_CHECKPOINT}")
    print(f"  SKIPPING TRAINING — loading pre-trained weights directly")
    print(f"  (This saves ~19.5 minutes of training time)")
    ckpt = torch.load(RUN3_CHECKPOINT, map_location=device, weights_only=False)
    distance_head.load_state_dict(ckpt['model_state_dict'])
    if 'template_encoder_state_dict' in ckpt:
        template_encoder.load_state_dict(ckpt['template_encoder_state_dict'])
        print(f"  Loaded template encoder from checkpoint")
    if 'backbone_unfrozen_state' in ckpt:
        for layer_key, layer_state in ckpt['backbone_unfrozen_state'].items():
            layer_idx = int(layer_key.split('.')[-1])
            backbone.transformer_encoder[layer_idx].load_state_dict(layer_state)
        print(f"  Loaded unfrozen backbone layers from checkpoint")
    print(f"  Checkpoint val_loss: {ckpt.get('val_loss', 'unknown')}")
    print(f"  Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
    distance_head.eval()
    template_encoder.eval()
    backbone.eval()
    print(f"  Model ready for inference (training skipped)")

else:
    # --- No checkpoint found: run full training (same as Run 3 OptB) ---
    print(f"  No pre-trained checkpoint found. Running full training...")

    TRAIN_EPOCHS = 30
    BATCH_SIZE = 2
    MAX_SEQ_LEN = 256
    HEAD_LR = 5e-5
    BACKBONE_LR = 1e-5
    TEMPLATE_MASK_PROB = 0.5

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
                    skipped += 1; continue
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
                    skipped += 1; continue
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
                pairwise = get_pairwise_features_train(seq)
                if random.random() < TEMPLATE_MASK_PROB:
                    tmpl_feat = torch.zeros(1, N, N, 16, device=device)
                else:
                    coords_t = torch.tensor(true_coords, dtype=torch.float32, device=device)
                    tmpl_feat = template_encoder(coords_t, confidence=1.0, has_template=True)
                    tmpl_feat = tmpl_feat.unsqueeze(0)
                combined = torch.cat([pairwise, tmpl_feat], dim=-1)
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
# CELL 14: ADV1 Inference — FIXED (3 fixes applied)
# ============================================================
#
# === FIX 1 (CRITICAL): Raw template ALWAYS in Slot 1 ===
#   Run 3 OptB scored 0.251 because weak-template targets had
#   ALL 5 slots NN-refined — no raw template anywhere.
#   Fork 2 scored 0.287 by using raw templates for ALL targets.
#   FIX: Slot 1 is ALWAYS the raw template (constraint-refined),
#   regardless of confidence. This guarantees at least Fork 2
#   quality in one slot. Best-of-5 picks the best.
#
# === FIX 2: Reduce refinement steps 100 -> 30 ===
#   100 steps of gradient descent toward noisy NN distances
#   (~11A avg error) pushes template coords too far from truth.
#   30 steps = gentler adjustment, coords stay closer to template.
#   Operates on NN slots (2-5 or 3-5), NOT on Slot 1 (raw template).
#
# === FIX 3: Lower HYBRID_THRESHOLD 0.3 -> 0.20 ===
#   More targets get template protection in Slots 1-2.
#   Borderline targets (conf 0.20-0.30) now get 2 template slots
#   instead of 1. Mostly redundant with Fix 1 but extra safety.
#
# Slot assignment after fixes:
#   HIGH confidence (conf > 0.20):
#     Slot 1: raw template (Fix 1)
#     Slot 2: template + noise
#     Slots 3-5: NN (30 steps, Fix 2)
#
#   LOW confidence (conf <= 0.20):
#     Slot 1: raw template (Fix 1 — THIS IS THE KEY CHANGE)
#     Slots 2-5: NN (30 steps, Fix 2)
#
#   NO template (conf = 0):
#     All 5 slots: MDS fallback (unchanged — but 0/28 targets hit this)
# ============================================================
print("\n" + "="*60)
print("PHASE 4: ADV1 Inference (FIXED: raw template in Slot 1 + less refinement)")
print("="*60)

MAX_INFER_LEN = 512
# === FIX 3: Lowered from 0.3 to 0.20 ===
# More targets get 2 template slots (1-2) instead of just 1
HYBRID_THRESHOLD = 0.20    # was 0.3 in Run 3 OptB

all_predictions = []
infer_start = time.time()

n_hybrid_template_2slot = 0   # Targets with conf > threshold (2 template slots)
n_hybrid_template_1slot = 0   # Targets with conf <= threshold but > 0 (1 template slot via Fix 1)
n_no_template = 0             # Targets with conf = 0 (no template at all)
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
        pairwise = get_pairwise_features(seq)
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
        tmpl_feat = tmpl_feat.unsqueeze(0)
        combined = torch.cat([pairwise, tmpl_feat], dim=-1)
        pred_dist = distance_head(combined).squeeze(0)

    full_tmpl = template_coords_per_target.get(target_id, np.zeros((len(sequence), 3)))
    if len(full_tmpl) > len(sequence):
        full_tmpl = full_tmpl[:len(sequence)]
    elif len(full_tmpl) < len(sequence):
        padded = np.zeros((len(sequence), 3))
        padded[:len(full_tmpl)] = full_tmpl
        full_tmpl = padded

    # === FIX 2: Reduced steps from 100 to 30 (less drift from template) ===
    nn_diversity = [
        {'noise': 0.0, 'steps': 30, 'seed': 0},   # was 100 — FIX 2
        {'noise': 0.3, 'steps': 30, 'seed': 1},   # was 100 — FIX 2
        {'noise': 0.5, 'steps': 30, 'seed': 2},   # was 100 — FIX 2
        {'noise': 0.7, 'steps': 50, 'seed': 3},   # was 150 — FIX 2
        {'noise': 0.0, 'steps': 15, 'seed': 4},   # was 50  — FIX 2
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

    # --- STEP 3: Assemble 5 slots (FIXED) ---
    coords_list = []

    if tmpl_conf > HYBRID_THRESHOLD:
        # HIGH confidence (> 0.20): Slots 1-2 template, Slots 3-5 NN
        n_hybrid_template_2slot += 1

        # Slot 1: Raw template (constraint-refined) — guaranteed Fork 2 quality
        tmpl_refined = adaptive_rna_constraints(full_tmpl.copy(), sequence, confidence=tmpl_conf)
        coords_list.append(tmpl_refined)

        # Slot 2: Template + small noise for diversity
        np.random.seed(42)
        small_noise = np.random.normal(0, 0.5, full_tmpl.shape)
        tmpl_noisy = adaptive_rna_constraints(full_tmpl + small_noise, sequence, confidence=tmpl_conf)
        coords_list.append(tmpl_noisy)

        # Slots 3-5: NN predictions (template-seeded, 30 steps)
        coords_list.append(nn_coords_list[0])
        coords_list.append(nn_coords_list[1])
        coords_list.append(nn_coords_list[2])

        if idx % 5 == 0:
            print(f"    HYBRID: conf={tmpl_conf:.3f} > {HYBRID_THRESHOLD} -> slots 1-2 template, 3-5 NN")

    elif tmpl_conf > 0.01:
        # === FIX 1 (CRITICAL): Weak template — but Slot 1 is STILL raw template ===
        # In Run 3 OptB, this branch put ALL 5 slots as NN-refined.
        # That was the root cause of 0.251 < 0.287.
        # NOW: Slot 1 = raw template (safety net), Slots 2-5 = NN.
        # Best-of-5 means the raw template carries if NN is worse.
        n_hybrid_template_1slot += 1

        # Slot 1: Raw template — ALWAYS present regardless of confidence
        # This is exactly what Fork 2 does. Guaranteed baseline quality.
        tmpl_refined = adaptive_rna_constraints(full_tmpl.copy(), sequence, confidence=tmpl_conf)
        coords_list.append(tmpl_refined)

        # Slots 2-5: NN predictions (template-seeded, 30 steps via Fix 2)
        coords_list.append(nn_coords_list[0])
        coords_list.append(nn_coords_list[1])
        coords_list.append(nn_coords_list[2])
        coords_list.append(nn_coords_list[3])

        if idx % 5 == 0:
            print(f"    FIX1: conf={tmpl_conf:.3f} <= {HYBRID_THRESHOLD} -> slot 1 RAW TEMPLATE, 2-5 NN")

    else:
        # No template at all (conf = 0): all 5 slots MDS fallback
        # In practice, 0/28 test targets hit this branch.
        n_no_template += 1
        coords_list = nn_coords_list[:5]

        if idx % 5 == 0:
            print(f"    NO TEMPLATE: conf={tmpl_conf:.3f} -> all 5 slots MDS fallback")

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
print(f"  Targets with 2 template slots (conf > {HYBRID_THRESHOLD}): {n_hybrid_template_2slot}")
print(f"  Targets with 1 template slot (FIX 1, conf <= {HYBRID_THRESHOLD}): {n_hybrid_template_1slot}")
print(f"  Targets with no template (MDS fallback): {n_no_template}")
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
# CELL 15: Post-Processing (UNCHANGED)
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
