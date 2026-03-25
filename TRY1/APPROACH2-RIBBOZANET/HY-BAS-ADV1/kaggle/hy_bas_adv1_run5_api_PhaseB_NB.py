# ============================================================
# HY-BAS-ADV1 RUN 5 — PHASE B: IPA Inference Only (Kaggle)
# ============================================================
#
# WHAT THIS IS:
#   The INFERENCE half of Run 5. Loads a checkpoint trained by
#   Phase A (locally or on Kaggle) and runs inference only.
#   The original hy_bas_adv1_run5_api_NB.py is UNCHANGED.
#
# PREREQUISITES:
#   Phase A checkpoint (adv1_run5_best_model.pt) must be
#   uploaded as a Kaggle dataset and attached to this notebook.
#
# WHAT IS SKIPPED vs MONOLITHIC RUN 5:
#   Cell 13 (training) — ENTIRELY SKIPPED.
#   All training was done in Phase A.
#
# ESTIMATED RUNTIME: ~30-45 min on Kaggle T4
#   (vs ~3-4 hours for the monolithic version)
# ============================================================


# ============================================================
# CELL 0: PATHS AND CONFIG
# ============================================================
#
# *** SET YOUR PATHS HERE ***
# This file is designed for KAGGLE but can run locally too.
#
# ============================================================

# --- OPTION A: LOCAL PATHS (uncomment and edit) ---------------
# DATA_ROOT          = 'E:/kaggle_data'
# COMP_BASE          = f'{DATA_ROOT}/stanford-rna-3d-folding-2'
# EXTENDED_SEQ_CSV   = f'{DATA_ROOT}/rna_sequences.csv'
# EXTENDED_COORD_CSV = f'{DATA_ROOT}/rna_coordinates.csv'
# BACKBONE_WEIGHTS   = f'{DATA_ROOT}/RibonanzaNet.pt'
# REPO_PATH          = f'{DATA_ROOT}/ribonanzanet_repo'
# PHASE_A_CHECKPOINT = './run5_output/adv1_run5_best_model.pt'
# OUTPUT_DIR         = './run5_output'
# SAMPLE_CSV         = f'{COMP_BASE}/sample_submission.csv'
# PLATFORM           = 'LOCAL'
# -----------------------------------------------------------------

# --- OPTION B: KAGGLE PATHS (default — auto-discovers) -----------
EXTENDED_SEQ_CSV   = None
EXTENDED_COORD_CSV = None
BACKBONE_WEIGHTS   = None
REPO_PATH          = None
PHASE_A_CHECKPOINT = None    # auto-discovered from /kaggle/input/
OUTPUT_DIR         = '/kaggle/working'
SAMPLE_CSV         = None
PLATFORM           = 'KAGGLE'
COMP_BASE_PRIMARY  = '/kaggle/input/competitions/stanford-rna-3d-folding-2/'
COMP_BASE_FALLBACK = '/kaggle/input/stanford-rna-3d-folding-2/'
# -----------------------------------------------------------------

# IPA configuration (MUST match Phase A — loaded from checkpoint)
MSA_TOP_N = 20
MSA_DIM   = 8
IPA_DIM          = 256
IPA_ITERATIONS   = 4
IPA_HEADS        = 4
FAPE_CLAMP       = 10.0

# These are NOT used in Phase B (no training) but kept for model init
FREEZE_BACKBONE_IPA = True
AUX_DIST_WEIGHT     = 0.1

RAW_SUBMISSION_PATH   = f'{OUTPUT_DIR}/submission_raw.csv'
FINAL_SUBMISSION_PATH = f'{OUTPUT_DIR}/submission.csv'


# ============================================================
# CELL 1: Install biopython
# ============================================================
import sys, os, glob

py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"
print(f"Python {sys.version_info.major}.{sys.version_info.minor} (tag: {py_ver})")

try:
    import Bio
    print(f"Biopython {Bio.__version__} already installed")
except ImportError:
    if PLATFORM == 'KAGGLE':
        bio_inits = glob.glob('/kaggle/input/**/Bio/__init__.py', recursive=True)
        matching = [p for p in bio_inits if py_ver in p]
        chosen = matching[0] if matching else (bio_inits[0] if bio_inits else None)
        if chosen:
            sys.path.insert(0, os.path.dirname(os.path.dirname(chosen)))
        else:
            import subprocess
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'biopython', '-q'])
    else:
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
import torch.nn.functional as F
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
print(f"Platform: {PLATFORM}")


# ============================================================
# CELL 3: Find all data paths + PHASE A CHECKPOINT
# ============================================================
if PLATFORM == 'KAGGLE':
    BASE = COMP_BASE_PRIMARY
    if not os.path.exists(BASE):
        BASE = COMP_BASE_FALLBACK

    for root, dirs, files in os.walk("/kaggle/input"):
        if "Network.py" in files and REPO_PATH is None:
            REPO_PATH = root
        for f in files:
            if f == "RibonanzaNet.pt" and BACKBONE_WEIGHTS is None:
                BACKBONE_WEIGHTS = os.path.join(root, f)
            if f == "adv1_run5_best_model.pt" and PHASE_A_CHECKPOINT is None:
                PHASE_A_CHECKPOINT = os.path.join(root, f)
            if f == "sample_submission.csv" and SAMPLE_CSV is None:
                SAMPLE_CSV = os.path.join(root, f)
    if EXTENDED_SEQ_CSV is None or EXTENDED_COORD_CSV is None:
        for root, dirs, files in os.walk("/kaggle/input"):
            for f in files:
                if f == "rna_sequences.csv" and EXTENDED_SEQ_CSV is None:
                    EXTENDED_SEQ_CSV = os.path.join(root, f)
                if f == "rna_coordinates.csv" and EXTENDED_COORD_CSV is None:
                    EXTENDED_COORD_CSV = os.path.join(root, f)
else:
    BASE = COMP_BASE

print(f"Competition data:     {BASE}")
print(f"RibonanzaNet repo:    {REPO_PATH}")
print(f"Backbone weights:     {BACKBONE_WEIGHTS}")
print(f"Phase A checkpoint:   {PHASE_A_CHECKPOINT}")
print(f"Sample submission:    {SAMPLE_CSV}")

if PHASE_A_CHECKPOINT is None or not os.path.exists(PHASE_A_CHECKPOINT):
    raise FileNotFoundError(
        "Phase A checkpoint (adv1_run5_best_model.pt) not found!\n"
        "Run Phase A first, then upload its output as a Kaggle dataset."
    )


# ============================================================
# CELL 4: Load competition data
# ============================================================
print("\nLoading competition data...")
train_seqs   = pd.read_csv(BASE + ('/' if PLATFORM == 'LOCAL' else '') + 'train_sequences.csv')
test_seqs    = pd.read_csv(BASE + ('/' if PLATFORM == 'LOCAL' else '') + 'test_sequences.csv')
train_labels = pd.read_csv(BASE + ('/' if PLATFORM == 'LOCAL' else '') + 'train_labels.csv')
print(f"Train: {len(train_seqs)}, Test: {len(test_seqs)}")


# ============================================================
# CELLS 5-7: Load extended data, process coords
# ============================================================
if EXTENDED_SEQ_CSV is None or EXTENDED_COORD_CSV is None:
    raise FileNotFoundError("Extended CSV files not found. Set paths in Cell 0.")

print("Loading extended data...")
train_seqs_v2   = pd.read_csv(EXTENDED_SEQ_CSV)
train_labels_v2 = pd.read_csv(EXTENDED_COORD_CSV)

def extend_dataset(original_df, v2_df, key_col):
    orig_keys = set(original_df[key_col])
    new_records = v2_df[~v2_df[key_col].isin(orig_keys)].copy()
    return pd.concat([original_df, new_records], ignore_index=True)

train_seqs_extended = extend_dataset(train_seqs, train_seqs_v2, 'target_id')

train_labels['_key'] = train_labels['ID'] + '_' + train_labels['resid'].astype(str)
train_labels_v2['_key'] = train_labels_v2['ID'] + '_' + train_labels_v2['resid'].astype(str)
new_labels = train_labels_v2[~train_labels_v2['_key'].isin(set(train_labels['_key']))].copy()
train_labels_extended = pd.concat([train_labels, new_labels], ignore_index=True)
for df in [train_labels_extended, train_labels, train_labels_v2]:
    df.drop('_key', axis=1, inplace=True, errors='ignore')

print("\nProcessing labels into coordinate dict...")
train_coords_dict = {}
for id_prefix, group in train_labels_extended.groupby(
    lambda x: train_labels_extended['ID'][x].rsplit('_', 1)[0]
):
    coords = []
    for _, row in group.sort_values('resid').iterrows():
        coords.append([row['x_1'], row['y_1'], row['z_1']])
    train_coords_dict[id_prefix] = np.array(coords)
print(f"Loaded {len(train_coords_dict)} structures")


# ============================================================
# CELL 8: Template functions (unchanged from Run 5)
# ============================================================
def adaptive_rna_constraints(coordinates, sequence, confidence=1.0):
    refined_coords = coordinates.copy(); n_residues = len(sequence)
    constraint_strength = 0.8 * (1.0 - min(confidence, 0.8))
    for i in range(n_residues - 1):
        current_dist = np.linalg.norm(refined_coords[i+1] - refined_coords[i])
        if current_dist < 5.5 or current_dist > 6.5:
            target_dist = 6.0; direction = refined_coords[i+1] - refined_coords[i]
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            adjustment = (target_dist - current_dist) * constraint_strength
            refined_coords[i+1] = refined_coords[i] + direction * (current_dist + adjustment)
    dist_mat = scipy_dist_matrix(refined_coords, refined_coords)
    clashes = np.where((dist_mat < 3.8) & (dist_mat > 0))
    for idx in range(len(clashes[0])):
        i, j = clashes[0][idx], clashes[1][idx]
        if abs(i - j) <= 1 or i >= j: continue
        direction = refined_coords[j] - refined_coords[i]
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        adj = (3.8 - dist_mat[i, j]) * constraint_strength
        refined_coords[i] -= direction * (adj / 2); refined_coords[j] += direction * (adj / 2)
    return refined_coords

def adapt_template_to_query(query_seq, template_seq, template_coords):
    alns = pairwise2.align.globalms(Seq(query_seq), Seq(template_seq), 2.9, -1, -10, -0.5, one_alignment_only=True)
    if not alns: return generate_rna_structure(query_seq)
    aln = alns[0]; query_coords = np.full((len(query_seq), 3), np.nan); qi = ti = 0
    for i in range(len(aln.seqA)):
        if aln.seqA[i] != '-' and aln.seqB[i] != '-':
            if ti < len(template_coords): query_coords[qi] = template_coords[ti]
            ti += 1; qi += 1
        elif aln.seqA[i] != '-': qi += 1
        elif aln.seqB[i] != '-': ti += 1
    for i in range(len(query_coords)):
        if np.isnan(query_coords[i, 0]):
            prev_valid = next_valid = None
            for j in range(i-1, -1, -1):
                if not np.isnan(query_coords[j, 0]): prev_valid = j; break
            for j in range(i+1, len(query_coords)):
                if not np.isnan(query_coords[j, 0]): next_valid = j; break
            if prev_valid is not None and next_valid is not None:
                gap = next_valid - prev_valid
                for k, idx2 in enumerate(range(prev_valid+1, next_valid)):
                    w = (k+1) / gap
                    query_coords[idx2] = (1-w)*query_coords[prev_valid] + w*query_coords[next_valid]
            elif prev_valid is not None:
                d = (np.array([1,0,0]) if prev_valid == 0 else query_coords[prev_valid] - query_coords[prev_valid-1])
                d = d / (np.linalg.norm(d) + 1e-10)
                query_coords[i] = query_coords[prev_valid] + d * 5.9 * (i - prev_valid)
            elif next_valid is not None:
                query_coords[i] = query_coords[next_valid] - np.array([5.9 * (next_valid - i), 0, 0])
    return np.nan_to_num(query_coords)

def generate_rna_structure(sequence, seed=None):
    if seed is not None: np.random.seed(seed); random.seed(seed)
    n = len(sequence); coords = np.zeros((n, 3))
    for i in range(min(3, n)): coords[i] = [10*np.cos(i*0.6), 10*np.sin(i*0.6), i*2.5]
    direction = np.array([0, 0, 1.0])
    for i in range(3, n):
        if random.random() < 0.3:
            axis = np.random.normal(0, 1, 3); axis /= (np.linalg.norm(axis) + 1e-10)
            direction = R.from_rotvec(random.uniform(0.2, 0.6) * axis).apply(direction)
        else:
            direction += np.random.normal(0, 0.15, 3); direction /= (np.linalg.norm(direction) + 1e-10)
        coords[i] = coords[i-1] + random.uniform(3.5, 4.5) * direction
    return coords

def find_similar_sequences(query_seq, train_seqs_df, train_coords_dict, top_n=5):
    candidates = []; k = 3; q_kmers = set(query_seq[i:i+k] for i in range(len(query_seq)-k+1))
    for _, row in train_seqs_df.iterrows():
        tid = row['target_id']; tseq = row['sequence']
        if tid not in train_coords_dict: continue
        if abs(len(tseq) - len(query_seq)) / max(len(tseq), len(query_seq)) > 0.4: continue
        t_kmers = set(tseq[i:i+k] for i in range(len(tseq)-k+1))
        score = len(q_kmers & t_kmers) / len(q_kmers | t_kmers) if q_kmers | t_kmers else 0
        candidates.append((tid, tseq, score, train_coords_dict[tid]))
    candidates.sort(key=lambda x: x[2], reverse=True); candidates = candidates[:100]
    similar = []
    for tid, tseq, _, coords in candidates:
        alns = pairwise2.align.globalms(Seq(query_seq), tseq, 2.9, -1, -10, -0.5, one_alignment_only=True)
        if alns:
            s = alns[0].score / (2 * min(len(query_seq), len(tseq)))
            if s > 0: similar.append((tid, tseq, s, coords))
    similar.sort(key=lambda x: x[2], reverse=True)
    return similar[:top_n]


# ============================================================
# CELL 9: Template search + MSA collection
# ============================================================
print("\n" + "="*60)
print("PHASE 1: Template Search + MSA Collection")
print("="*60)

template_coords_per_target = {}; template_confidence_per_target = {}; msa_hits_per_target = {}
start_time = time.time()
for idx, row in test_seqs.iterrows():
    target_id = row['target_id']; sequence = row['sequence']
    if idx % 5 == 0: print(f"  {idx+1}/{len(test_seqs)}: {target_id} ({len(sequence)} nt)")
    similar = find_similar_sequences(sequence, train_seqs_extended, train_coords_dict, top_n=MSA_TOP_N)
    msa_hits_per_target[target_id] = similar
    if similar:
        best_tid, best_seq, best_score, best_coords = similar[0]
        adapted = adapt_template_to_query(sequence, best_seq, best_coords)
        refined = adaptive_rna_constraints(adapted, sequence, confidence=best_score)
        template_coords_per_target[target_id] = refined
        template_confidence_per_target[target_id] = best_score
    else:
        template_coords_per_target[target_id] = np.zeros((len(sequence), 3))
        template_confidence_per_target[target_id] = 0.0

print(f"Template search complete in {time.time()-start_time:.0f}s")


# ============================================================
# CELL 9.5: Compute MSA features
# ============================================================
print("\nComputing MSA features...")

def compute_msa_features(query_seq, similar_hits, max_len=512):
    N = min(len(query_seq), max_len); n_channels = 8
    if not similar_hits or len(similar_hits) < 2:
        return np.zeros((N, N, n_channels), dtype=np.float32)
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3, '-': 4}
    n_seqs = len(similar_hits) + 1; aln_matrix = np.full((n_seqs, N), 4, dtype=np.int32)
    for i in range(N): aln_matrix[0, i] = base_to_idx.get(query_seq[i].upper(), 4)
    for hit_idx, (tid, hit_seq, score, coords) in enumerate(similar_hits):
        try:
            alns = pairwise2.align.globalms(Seq(query_seq[:N]), hit_seq, 2.9, -1, -10, -0.5, one_alignment_only=True)
            if not alns: continue
            qi = 0
            for k in range(len(alns[0].seqA)):
                if qi >= N: break
                if alns[0].seqA[k] != '-':
                    if alns[0].seqB[k] != '-': aln_matrix[hit_idx+1, qi] = base_to_idx.get(alns[0].seqB[k].upper(), 4)
                    qi += 1
        except: continue
    eps = 1e-10; n_bases = 5; freq = np.zeros((N, n_bases), dtype=np.float32)
    for pos in range(N): freq[pos] = np.bincount(aln_matrix[:, pos], minlength=n_bases).astype(np.float32) / n_seqs
    entropy = -np.sum(freq * np.log(freq + eps), axis=1); conservation = 1.0 - (entropy / np.log(n_bases))
    gap_freq = freq[:, 4]; neff = float(n_seqs) / 100.0; n_real = 4
    joint_counts = np.zeros((N, N, n_real, n_real), dtype=np.float32)
    for si in range(n_seqs):
        row = aln_matrix[si, :]
        for i in range(N):
            if row[i] >= n_real: continue
            for j in range(i, N):
                if row[j] >= n_real: continue
                joint_counts[i, j, row[i], row[j]] += 1
                if i != j: joint_counts[j, i, row[j], row[i]] += 1
    pair_totals = joint_counts.sum(axis=(2, 3), keepdims=True)
    joint_prob = joint_counts / (pair_totals + eps)
    marg_i = joint_prob.sum(axis=3); marg_j = joint_prob.sum(axis=2)
    mi = np.maximum((joint_prob * np.log((joint_prob + eps) / (marg_i[:,:,:,None] * marg_j[:,:,None,:] + eps))).sum(axis=(2,3)), 0.0)
    mi_apc = np.maximum(mi - mi.mean(axis=1)[:,None] * mi.mean(axis=0)[None,:] / (mi.mean() + eps), 0.0)
    features = np.zeros((N, N, n_channels), dtype=np.float32)
    features[:,:,0] = mi; features[:,:,1] = mi_apc
    features[:,:,2] = conservation[:,None]; features[:,:,3] = conservation[None,:]
    features[:,:,4] = conservation[:,None] * conservation[None,:]
    features[:,:,5] = gap_freq[:,None]; features[:,:,6] = gap_freq[None,:]; features[:,:,7] = neff
    return features

msa_features_per_target = {}
for idx, row in test_seqs.iterrows():
    target_id = row['target_id']; sequence = row['sequence']
    msa_features_per_target[target_id] = compute_msa_features(sequence, msa_hits_per_target.get(target_id, []))
print(f"MSA features computed for {len(msa_features_per_target)} targets")


# ============================================================
# CELL 10: Neural Network Modules (model definitions only)
# ============================================================
print("\n" + "="*60)
print("PHASE 2: Load Models + Phase A Checkpoint")
print("="*60)

class TemplateEncoder(nn.Module):
    def __init__(self, template_dim=16, num_bins=22, max_dist=40.0):
        super().__init__()
        self.template_dim = template_dim; self.num_bins = num_bins
        edges = torch.arange(0, max_dist + max_dist/(num_bins-1), max_dist/(num_bins-1))[:num_bins]
        self.register_buffer('bin_edges', edges)
        self.projection = nn.Linear(num_bins, template_dim)
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.01); nn.init.zeros_(self.projection.bias)
    def forward(self, coords, confidence=1.0, has_template=True):
        N = coords.shape[0]
        if not has_template: return torch.zeros(N, N, self.template_dim, device=coords.device)
        diff = coords.unsqueeze(0) - coords.unsqueeze(1)
        dist = torch.sqrt((diff**2).sum(-1) + 1e-8)
        bin_idx = torch.bucketize(dist, self.bin_edges).clamp(0, self.num_bins-1)
        bins = torch.zeros(N, N, self.num_bins, device=dist.device)
        bins.scatter_(2, bin_idx.unsqueeze(-1), 1.0)
        return self.projection(bins) * confidence

def _ipa_exists(val): return val is not None
def _ipa_default(val, d): return val if _ipa_exists(val) else d

class InvariantPointAttention(nn.Module):
    def __init__(self, dim, heads=8, scalar_key_dim=16, scalar_value_dim=16,
                 point_key_dim=4, point_value_dim=4, pairwise_repr_dim=None, require_pairwise_repr=True):
        super().__init__()
        self.eps = 1e-8; self.heads = heads; self.require_pairwise_repr = require_pairwise_repr
        self.to_scalar_q = nn.Linear(dim, scalar_key_dim * heads, bias=False)
        self.to_scalar_k = nn.Linear(dim, scalar_key_dim * heads, bias=False)
        self.to_scalar_v = nn.Linear(dim, scalar_value_dim * heads, bias=False)
        self.to_point_q = nn.Linear(dim, point_key_dim * heads * 3, bias=False)
        self.to_point_k = nn.Linear(dim, point_key_dim * heads * 3, bias=False)
        self.to_point_v = nn.Linear(dim, point_value_dim * heads * 3, bias=False)
        if require_pairwise_repr:
            pairwise_repr_dim = _ipa_default(pairwise_repr_dim, dim)
            self.pairwise_attn_logits = nn.Linear(pairwise_repr_dim, heads, bias=False)
        self.point_weights = nn.Parameter(torch.log(torch.exp(torch.ones(1, heads)) - 1.))
        num_pair_out = pairwise_repr_dim if require_pairwise_repr else 0
        self.to_out = nn.Linear(heads * (scalar_value_dim + point_value_dim * 3 + point_value_dim + num_pair_out), dim)
        self.scalar_key_dim = scalar_key_dim; self.scalar_value_dim = scalar_value_dim
        self.point_key_dim = point_key_dim; self.point_value_dim = point_value_dim
        self.pairwise_repr_dim = pairwise_repr_dim if require_pairwise_repr else 0
    def forward(self, single_repr, pairwise_repr=None, *, rotations, translations, mask=None):
        b, n, d = single_repr.shape; h = self.heads
        sq = rearrange(self.to_scalar_q(single_repr), 'b n (h d) -> b h n d', h=h)
        sk = rearrange(self.to_scalar_k(single_repr), 'b n (h d) -> b h n d', h=h)
        sv = rearrange(self.to_scalar_v(single_repr), 'b n (h d) -> b h n d', h=h)
        pq = rearrange(self.to_point_q(single_repr), 'b n (h p c) -> b h n p c', h=h, c=3)
        pk = rearrange(self.to_point_k(single_repr), 'b n (h p c) -> b h n p c', h=h, c=3)
        pv = rearrange(self.to_point_v(single_repr), 'b n (h p c) -> b h n p c', h=h, c=3)
        re = rotations.unsqueeze(1).expand(b, h, n, 3, 3)
        te_q = translations.unsqueeze(1).unsqueeze(3).expand(b, h, n, self.point_key_dim, 3)
        te_v = translations.unsqueeze(1).unsqueeze(3).expand(b, h, n, self.point_value_dim, 3)
        def to_global(lp, te): return torch.einsum('b h n i j, b h n p j -> b h n p i', re, lp) + te
        pq_g = to_global(pq, te_q); pk_g = to_global(pk, te_q); pv_g = to_global(pv, te_v)
        sa = torch.einsum('b h i d, b h j d -> b h i j', sq, sk) * (self.scalar_key_dim ** -0.5)
        pw = F.softplus(self.point_weights).view(1, h, 1, 1)
        pdiff = rearrange(pq_g, 'b h i p c -> b h i () p c') - rearrange(pk_g, 'b h j p c -> b h () j p c')
        attn_logits = sa + (-0.5 * (pw * (pdiff**2).sum(-1).sum(-1)))
        if self.require_pairwise_repr and _ipa_exists(pairwise_repr):
            attn_logits = attn_logits + rearrange(self.pairwise_attn_logits(pairwise_repr), 'b i j h -> b h i j')
        if _ipa_exists(mask): attn_logits = attn_logits.masked_fill(rearrange(mask.float(), 'b j -> b () () j') == 0, -1e9)
        attn = attn_logits.softmax(dim=-1)
        so = torch.einsum('b h i j, b h j d -> b h i d', attn, sv)
        pva = torch.einsum('b h i j, b h j p c -> b h i p c', attn, pv_g)
        rt = rotations.unsqueeze(1).expand(b, h, n, 3, 3).transpose(-1, -2)
        to_ = translations.unsqueeze(1).unsqueeze(3).expand(b, h, n, self.point_value_dim, 3)
        pvl = torch.einsum('b h n i j, b h n p j -> b h n p i', rt, pva - to_); pn = torch.norm(pvl, dim=-1)
        sf = rearrange(so, 'b h n d -> b n (h d)'); pf = rearrange(pvl, 'b h n p c -> b n (h p c)'); nf = rearrange(pn, 'b h n p -> b n (h p)')
        if self.require_pairwise_repr and _ipa_exists(pairwise_repr):
            po = torch.einsum('b h i j, b i j d -> b h i d', attn, pairwise_repr)
            out = torch.cat([sf, pf, nf, rearrange(po, 'b h n d -> b n (h d)')], dim=-1)
        else: out = torch.cat([sf, pf, nf], dim=-1)
        return self.to_out(out)

class IPABlock(nn.Module):
    def __init__(self, dim, ff_mult=1, **kw):
        super().__init__()
        self.attn = InvariantPointAttention(dim, **kw); self.attn_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim*ff_mult), nn.ReLU(), nn.Linear(dim*ff_mult, dim))
    def forward(self, x, pairwise_repr=None, *, rotations, translations, mask=None):
        x = self.attn_norm(x + self.attn(x, pairwise_repr, rotations=rotations, translations=translations, mask=mask))
        return x + self.ff(x)

def build_rna_frames(coords):
    N = coords.shape[0]; eps = 1e-8
    rotations = torch.eye(3, device=coords.device).unsqueeze(0).expand(N, -1, -1).clone()
    translations = coords.clone()
    for i in range(N):
        has_next = i < N-1; has_prev = i > 0
        if not has_next and not has_prev: continue
        x_vec = (coords[i+1] - coords[i]) if has_next else (coords[i] - coords[i-1])
        x_len = x_vec.norm() + eps
        if x_len < eps*10: continue
        x_axis = x_vec / x_len
        if has_prev: back_axis = (coords[i-1] - coords[i]) / ((coords[i-1] - coords[i]).norm() + eps)
        else:
            back_axis = torch.tensor([0.,1.,0.], device=coords.device)
            if abs(torch.dot(x_axis, back_axis).item()) > 0.9: back_axis = torch.tensor([0.,0.,1.], device=coords.device)
        z_vec = torch.cross(x_axis, back_axis, dim=0); z_len = z_vec.norm() + eps
        if z_len < eps*10: continue
        z_axis = z_vec / z_len; y_axis = torch.cross(z_axis, x_axis, dim=0); y_axis = y_axis / (y_axis.norm() + eps)
        rotations[i] = torch.stack([x_axis, y_axis, z_axis], dim=1)
    return rotations, translations

class IPAStructureModule(nn.Module):
    def __init__(self, single_dim, pair_dim, n_iter, heads):
        super().__init__()
        self.n_iter = n_iter
        self.pair_proj = nn.Sequential(nn.Linear(pair_dim, single_dim), nn.ReLU(), nn.LayerNorm(single_dim))
        self.single_norm = nn.LayerNorm(single_dim)
        self.ipa_block = IPABlock(dim=single_dim, heads=heads, scalar_key_dim=16, scalar_value_dim=16,
            point_key_dim=4, point_value_dim=4, pairwise_repr_dim=single_dim, require_pairwise_repr=True)
        self.coord_update = nn.Sequential(nn.LayerNorm(single_dim), nn.Linear(single_dim, single_dim//2), nn.ReLU(), nn.Linear(single_dim//2, 3))
    def forward(self, single_repr, pair_repr, init_coords):
        b, N, _ = single_repr.shape
        single = self.single_norm(single_repr); pair = self.pair_proj(pair_repr); coords = init_coords.clone()
        mask = torch.ones(b, N, device=single_repr.device, dtype=torch.bool)
        for _ in range(self.n_iter):
            rots, trans = build_rna_frames(coords)
            updated = self.ipa_block(single, pairwise_repr=pair, rotations=rots.unsqueeze(0), translations=trans.unsqueeze(0), mask=mask)
            coords = coords + self.coord_update(updated).squeeze(0); single = updated
        return coords


# ============================================================
# CELL 11: Load backbone
# ============================================================
print("\nLoading RibonanzaNet backbone...")
if REPO_PATH: sys.path.insert(0, REPO_PATH)
from Network import RibonanzaNet

class Cfg:
    ntoken=5; ninp=256; nhead=8; nlayers=9; nclass=2; k=5; dropout=0.05; pairwise_dimension=64; use_triangular_attention=False
try:
    import yaml
    cfg_path = os.path.join(REPO_PATH, "configs", "pairwise.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            for k2, v in yaml.safe_load(f).items(): setattr(Cfg, k2, v)
except: pass

backbone = RibonanzaNet(Cfg)
state = torch.load(BACKBONE_WEIGHTS, map_location='cpu', weights_only=False)
if isinstance(state, dict) and 'model_state_dict' in state: state = state['model_state_dict']
backbone.load_state_dict(state, strict=False); backbone = backbone.to(device)
for p in backbone.parameters(): p.requires_grad = False
total_layers = len(list(backbone.transformer_encoder))

def get_pairwise_and_single_features(sequence):
    base_map = {'A':0,'C':1,'G':2,'U':3}
    tokens = torch.tensor([base_map.get(b,4) for b in sequence.upper()], dtype=torch.long).unsqueeze(0).to(device)
    N = len(sequence); src_mask = torch.ones(1, N, dtype=torch.long, device=device)
    with torch.no_grad():
        embedded = backbone.encoder(tokens)
        pairwise = backbone.outer_product_mean(embedded) + backbone.pos_encoder(embedded)
        hidden = embedded
        for layer in backbone.transformer_encoder:
            result = layer(hidden, pairwise, src_mask=src_mask)
            if isinstance(result, tuple): hidden, pairwise = result
            else: hidden = result
    return pairwise, hidden
print("  Backbone loaded (all layers frozen for inference).")


# ============================================================
# CELL 12: Create IPA Model + LOAD PHASE A CHECKPOINT
# ============================================================
PAIR_DIM = 64 + 16 + MSA_DIM

template_encoder = TemplateEncoder(template_dim=16, num_bins=22, max_dist=40.0).to(device)
ipa_module = IPAStructureModule(single_dim=IPA_DIM, pair_dim=PAIR_DIM, n_iter=IPA_ITERATIONS, heads=IPA_HEADS).to(device)

# --- LOAD PHASE A CHECKPOINT ---
print(f"\nLoading Phase A checkpoint: {PHASE_A_CHECKPOINT}")
ckpt = torch.load(PHASE_A_CHECKPOINT, map_location=device, weights_only=False)

ipa_module.load_state_dict(ckpt['ipa_module_state_dict'])
print(f"  [OK] IPA module loaded")

template_encoder.load_state_dict(ckpt['template_encoder_state_dict'])
print(f"  [OK] TemplateEncoder loaded")

if ckpt.get('backbone_unfrozen_state'):
    for layer_key, layer_state in ckpt['backbone_unfrozen_state'].items():
        layer_idx = int(layer_key.split('.')[-1])
        backbone.transformer_encoder[layer_idx].load_state_dict(layer_state)
    print(f"  [OK] Backbone unfrozen layers loaded")

print(f"  Trained epoch: {ckpt.get('epoch', '?')}")
print(f"  Val FAPE: {ckpt.get('val_fape', '?')}")

ipa_module.eval(); template_encoder.eval(); backbone.eval()
print("  All models set to eval mode. NO TRAINING IN PHASE B.")


# ============================================================
# CELL 13: SKIPPED — training was done in Phase A
# ============================================================
print("\n  Cell 13 (training): SKIPPED — checkpoint loaded from Phase A.")


# ============================================================
# CELL 14: IPA Inference (unchanged from Run 5)
# ============================================================
print("\n" + "="*60)
print("PHASE 4: IPA Inference")
print("="*60)

MAX_INFER_LEN = 512; HYBRID_THRESHOLD = 0.20
all_predictions = []; infer_start = time.time()
n_hybrid_2slot = 0; n_hybrid_1slot = 0; n_no_template = 0; n_ipa_slots = 0

for idx, row in test_seqs.iterrows():
    target_id = row['target_id']; sequence = row['sequence']
    seq = sequence[:MAX_INFER_LEN]; N = len(seq)
    if idx % 5 == 0: print(f"  {idx+1}/{len(test_seqs)}: {target_id} ({len(sequence)} nt)")

    with torch.no_grad():
        pairwise, single = get_pairwise_and_single_features(seq)
        tmpl_coords = template_coords_per_target.get(target_id, np.zeros((N, 3)))
        tmpl_conf = template_confidence_per_target.get(target_id, 0.0)
        if len(tmpl_coords) > N: tmpl_coords = tmpl_coords[:N]
        elif len(tmpl_coords) < N:
            pad = np.zeros((N, 3)); pad[:len(tmpl_coords)] = tmpl_coords; tmpl_coords = pad
        coords_t = torch.tensor(tmpl_coords, dtype=torch.float32, device=device)
        has_tmpl = tmpl_conf > 0.01
        tmpl_feat = template_encoder(coords_t, confidence=tmpl_conf, has_template=has_tmpl).unsqueeze(0)
        msa_np = msa_features_per_target.get(target_id, np.zeros((N, N, MSA_DIM)))
        if msa_np.shape[0] > N: msa_np = msa_np[:N, :N, :]
        elif msa_np.shape[0] < N:
            pad = np.zeros((N, N, MSA_DIM), dtype=np.float32); m = msa_np.shape[0]; pad[:m,:m,:] = msa_np; msa_np = pad
        msa_feat = torch.tensor(msa_np, dtype=torch.float32, device=device).unsqueeze(0)
        combined = torch.cat([pairwise, tmpl_feat, msa_feat], dim=-1)

        def run_ipa(init_np, seed=0):
            torch.manual_seed(seed)
            return ipa_module(single, combined, torch.tensor(init_np, dtype=torch.float32, device=device)).cpu().numpy()
        def add_noise(arr, sigma, seed=0):
            np.random.seed(seed); return (arr + np.random.normal(0, sigma, arr.shape)).astype(np.float32)
        zeros_np = np.zeros((N, 3), dtype=np.float32)

        if has_tmpl:
            t_np = tmpl_coords.astype(np.float32)
            ipa_0 = run_ipa(t_np, seed=0); ipa_1 = run_ipa(add_noise(t_np, 0.5, 1), seed=1)
            ipa_2 = run_ipa(add_noise(t_np, 1.0, 2), seed=2); ipa_3 = run_ipa(add_noise(t_np, 1.5, 3), seed=3)
            ipa_4 = run_ipa(zeros_np, seed=4)
        else:
            ipa_0 = run_ipa(zeros_np, seed=0); ipa_1 = run_ipa(add_noise(zeros_np, 0.5, 1), seed=1)
            ipa_2 = run_ipa(add_noise(zeros_np, 1.0, 2), seed=2); ipa_3 = run_ipa(add_noise(zeros_np, 1.5, 3), seed=3)
            ipa_4 = run_ipa(zeros_np, seed=5)

    def extend_to_full(arr):
        if len(sequence) <= MAX_INFER_LEN: return arr
        remaining = len(sequence) - MAX_INFER_LEN
        last_dir = (arr[-1] - arr[-2]) if arr.shape[0] >= 2 else np.array([5.9, 0., 0.])
        last_dir = last_dir / (np.linalg.norm(last_dir) + 1e-8) * 5.9
        return np.concatenate([arr, np.array([arr[-1] + last_dir*(i+1) for i in range(remaining)])])
    ipa_0, ipa_1, ipa_2, ipa_3, ipa_4 = map(extend_to_full, [ipa_0, ipa_1, ipa_2, ipa_3, ipa_4])

    full_tmpl = template_coords_per_target.get(target_id, np.zeros((len(sequence), 3)))
    if len(full_tmpl) > len(sequence): full_tmpl = full_tmpl[:len(sequence)]
    elif len(full_tmpl) < len(sequence):
        pad = np.zeros((len(sequence), 3)); pad[:len(full_tmpl)] = full_tmpl; full_tmpl = pad

    coords_list = []
    if tmpl_conf > HYBRID_THRESHOLD:
        n_hybrid_2slot += 1; n_ipa_slots += 3
        tmpl_r = adaptive_rna_constraints(full_tmpl.copy(), sequence, confidence=tmpl_conf)
        np.random.seed(42); tmpl_n = adaptive_rna_constraints(full_tmpl + np.random.normal(0, 0.5, full_tmpl.shape), sequence, confidence=tmpl_conf)
        coords_list = [tmpl_r, tmpl_n, ipa_0, ipa_1, ipa_2]
    elif tmpl_conf > 0.01:
        n_hybrid_1slot += 1; n_ipa_slots += 4
        tmpl_r = adaptive_rna_constraints(full_tmpl.copy(), sequence, confidence=tmpl_conf)
        coords_list = [tmpl_r, ipa_0, ipa_1, ipa_2, ipa_3]
    else:
        n_no_template += 1; n_ipa_slots += 5
        coords_list = [ipa_0, ipa_1, ipa_2, ipa_3, ipa_4]

    for j in range(len(sequence)):
        pred_row = {'ID': f"{target_id}_{j+1}", 'resname': sequence[j], 'resid': j + 1}
        for i in range(5):
            pred_row[f'x_{i+1}'] = float(coords_list[i][j][0])
            pred_row[f'y_{i+1}'] = float(coords_list[i][j][1])
            pred_row[f'z_{i+1}'] = float(coords_list[i][j][2])
        all_predictions.append(pred_row)

print(f"\nInference complete: {len(all_predictions)} rows in {time.time()-infer_start:.0f}s")

submission_df = pd.DataFrame(all_predictions)
col_order = ['ID', 'resname', 'resid']
for i in range(1, 6):
    for c in ['x', 'y', 'z']: col_order.append(f'{c}_{i}')
submission_df = submission_df[col_order]
submission_df.to_csv(RAW_SUBMISSION_PATH, index=False)


# ============================================================
# CELL 15: Post-Processing
# ============================================================
print("\n" + "="*60)
print("PHASE 5: Post-Processing")
print("="*60)

sample_rows = {}; sample_order = []; cols = None
with open(SAMPLE_CSV, "r") as f:
    reader = csv.DictReader(f); cols = reader.fieldnames
    for row in reader: sample_rows[row["ID"]] = row; sample_order.append(row["ID"])

raw_rows = {}
with open(RAW_SUBMISSION_PATH, "r") as f:
    for row in csv.DictReader(f): raw_rows[row["ID"]] = row

matched = filled = 0
with open(FINAL_SUBMISSION_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=cols); writer.writeheader()
    for sid in sample_order:
        if sid in raw_rows: writer.writerow(raw_rows[sid]); matched += 1
        else: writer.writerow(sample_rows[sid]); filled += 1

print(f"  Matched: {matched}  Filled: {filled}  Total: {matched + filled}")
print(f"  File: {FINAL_SUBMISSION_PATH}")

print("\n" + "="*60)
print("DONE. Run 5 Phase B (inference only) complete.")
print("="*60)
