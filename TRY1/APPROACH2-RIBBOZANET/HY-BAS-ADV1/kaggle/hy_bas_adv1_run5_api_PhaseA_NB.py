# ============================================================
# HY-BAS-ADV1 RUN 5 — PHASE A: IPA Training (Local or Kaggle)
# ============================================================
#
# WHAT THIS IS:
#   The TRAINING half of Run 5, split from the monolithic file.
#   The original hy_bas_adv1_run5_api_NB.py is UNCHANGED.
#
#   Phase A = Cells 0 through 13 (data loading + model + training)
#   Phase B = Cells 0 through 12 + 14-15 (load checkpoint + inference)
#
# WHERE TO RUN:
#   LOCAL:  python hy_bas_adv1_run5_api_PhaseA_NB.py
#           RTX 3050 (8 GB) or RTX 3060 (12 GB) — plenty of VRAM
#           No time limit. Train 100+ epochs overnight.
#           Use tmscore_eval.py for instant TM-score feedback.
#
#   KAGGLE: Paste into notebook, run, save output as dataset.
#           Output checkpoint becomes input for Phase B notebook.
#
# OUTPUTS:
#   adv1_run5_best_model.pt — contains:
#     ipa_module_state_dict, template_encoder_state_dict,
#     backbone_unfrozen_state, epoch, val_fape, config params
#
# NEXT STEP:
#   Upload adv1_run5_best_model.pt as Kaggle dataset.
#   Then run Phase B on Kaggle (inference only, ~30 min).
# ============================================================


# ============================================================
# CELL 0: PATHS AND CONFIG
# ============================================================
#
# *** SET YOUR PATHS HERE ***
# Uncomment ONE block below (LOCAL or KAGGLE), edit paths.
#
# ============================================================

# --- OPTION A: LOCAL PATHS (uncomment and edit) ---------------
# DATA_ROOT          = 'E:/kaggle_data'
# COMP_BASE          = f'{DATA_ROOT}/stanford-rna-3d-folding-2'
# EXTENDED_SEQ_CSV   = f'{DATA_ROOT}/rna_sequences.csv'
# EXTENDED_COORD_CSV = f'{DATA_ROOT}/rna_coordinates.csv'
# TRAIN_PICKLE       = f'{DATA_ROOT}/pdb_xyz_data.pkl'
# BACKBONE_WEIGHTS   = f'{DATA_ROOT}/RibonanzaNet.pt'
# REPO_PATH          = f'{DATA_ROOT}/ribonanzanet_repo'
# RUN4_CHECKPOINT    = f'{DATA_ROOT}/adv1_best_model.pt'
# OUTPUT_DIR         = './run5_output'
# PLATFORM           = 'LOCAL'
# -----------------------------------------------------------------

# --- OPTION B: KAGGLE PATHS (default — auto-discovers) -----------
EXTENDED_SEQ_CSV   = None
EXTENDED_COORD_CSV = None
TRAIN_PICKLE       = None
BACKBONE_WEIGHTS   = None
REPO_PATH          = None
RUN4_CHECKPOINT    = None
OUTPUT_DIR         = '/kaggle/working'
PLATFORM           = 'KAGGLE'
COMP_BASE_PRIMARY  = '/kaggle/input/competitions/stanford-rna-3d-folding-2/'
COMP_BASE_FALLBACK = '/kaggle/input/stanford-rna-3d-folding-2/'
# -----------------------------------------------------------------

# MSA configuration
MSA_TOP_N = 20
MSA_DIM   = 8

# IPA configuration
IPA_DIM          = 256
IPA_ITERATIONS   = 4
IPA_HEADS        = 4
FAPE_CLAMP       = 10.0
AUX_DIST_WEIGHT  = 0.1

# Training strategy
# LOCAL TIP: Set FREEZE_BACKBONE_IPA = False to unfreeze backbone
#            layers 7-8. You have time — no 9hr Kaggle limit.
# KAGGLE TIP: Keep True for 2-3x faster epochs.
FREEZE_BACKBONE_IPA = True
TRAIN_EPOCHS        = 40     # LOCAL: increase to 80-100 if desired
EARLY_STOP_PATIENCE = 10

# Phase A output checkpoint name
CHECKPOINT_PATH = f'{OUTPUT_DIR}/adv1_run5_best_model.pt'

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
        chosen   = matching[0] if matching else (bio_inits[0] if bio_inits else None)
        if chosen:
            bio_parent = os.path.dirname(os.path.dirname(chosen))
            sys.path.insert(0, bio_parent)
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
print(f"Platform: {PLATFORM}")


# ============================================================
# CELL 3: Find all data paths
# ============================================================
if PLATFORM == 'KAGGLE':
    BASE = COMP_BASE_PRIMARY
    if not os.path.exists(BASE):
        BASE = COMP_BASE_FALLBACK
    print(f"Competition data: {BASE}")

    for root, dirs, files in os.walk("/kaggle/input"):
        if "Network.py" in files and REPO_PATH is None:
            REPO_PATH = root
        for f in files:
            if f == "RibonanzaNet.pt" and BACKBONE_WEIGHTS is None:
                BACKBONE_WEIGHTS = os.path.join(root, f)
            if f == "best_model.pt":
                pass  # BASIC weights — not needed for Phase A
            if f.strip() == "adv1_best_model.pt" and RUN4_CHECKPOINT is None:
                RUN4_CHECKPOINT = os.path.join(root, f)
            if f == "pdb_xyz_data.pkl" and TRAIN_PICKLE is None:
                TRAIN_PICKLE = os.path.join(root, f)
    if EXTENDED_SEQ_CSV is None or EXTENDED_COORD_CSV is None:
        for root, dirs, files in os.walk("/kaggle/input"):
            for f in files:
                if f == "rna_sequences.csv" and EXTENDED_SEQ_CSV is None:
                    EXTENDED_SEQ_CSV = os.path.join(root, f)
                if f == "rna_coordinates.csv" and EXTENDED_COORD_CSV is None:
                    EXTENDED_COORD_CSV = os.path.join(root, f)
else:
    BASE = COMP_BASE

print(f"RibonanzaNet repo:  {REPO_PATH}")
print(f"Backbone weights:   {BACKBONE_WEIGHTS}")
print(f"Run 4 checkpoint:   {RUN4_CHECKPOINT}")
print(f"Training pickle:    {TRAIN_PICKLE}")
print(f"Extended seqs CSV:  {EXTENDED_SEQ_CSV}")
print(f"Extended coords CSV:{EXTENDED_COORD_CSV}")
print(f"Output dir:         {OUTPUT_DIR}")
print(f"Checkpoint path:    {CHECKPOINT_PATH}")


# ============================================================
# CELL 4: Load competition data (for training label processing)
# ============================================================
print("\nLoading competition data...")
train_seqs   = pd.read_csv(BASE + '/train_sequences.csv' if PLATFORM == 'LOCAL' else BASE + 'train_sequences.csv')
train_labels = pd.read_csv(BASE + '/train_labels.csv' if PLATFORM == 'LOCAL' else BASE + 'train_labels.csv')
print(f"Train: {len(train_seqs)} sequences")


# ============================================================
# CELL 5: Load extended data
# ============================================================
if EXTENDED_SEQ_CSV is None or EXTENDED_COORD_CSV is None:
    raise FileNotFoundError(
        "Could not find rna_sequences.csv or rna_coordinates.csv. "
        "Set paths in Cell 0."
    )
print("Loading extended data...")
train_seqs_v2   = pd.read_csv(EXTENDED_SEQ_CSV)
train_labels_v2 = pd.read_csv(EXTENDED_COORD_CSV)
print(f"Extended: {len(train_seqs_v2)} seqs, {len(train_labels_v2)} labels")


# ============================================================
# CELL 6: Extend datasets
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
# CELL 8-9: Template functions + MSA collection
# ============================================================
# NOTE: Phase A does NOT need template search or MSA for test seqs.
# But Cell 13 training uses MSA proxy features, so we keep Cell 9.5.
# Template functions are included for the training MSA proxy.
# ============================================================

# --- CELL 9.5 FUNCTIONS (MSA proxy for training) ---
def compute_msa_features(query_seq, similar_hits, max_len=512):
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
            if not alns: continue
            aligned_query = alns[0].seqA
            aligned_hit   = alns[0].seqB
            qi = 0
            for k in range(len(aligned_query)):
                if qi >= N: break
                if aligned_query[k] != '-':
                    if aligned_hit[k] != '-':
                        aln_matrix[hit_idx+1, qi] = base_to_idx.get(aligned_hit[k].upper(), 4)
                    qi += 1
        except: continue
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
            if row[i] >= n_real: continue
            for j in range(i, N):
                if row[j] >= n_real: continue
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


# ============================================================
# CELL 10: Neural Network Modules
# ============================================================
print("\n" + "="*60)
print("PHASE 2: Neural Network Modules (Run 5 IPA)")
print("="*60)

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

# --- lucidrains IPA (Inline B, copy-paste from v0.2.2) ---
def _ipa_exists(val): return val is not None
def _ipa_default(val, d): return val if _ipa_exists(val) else d

class InvariantPointAttention(nn.Module):
    def __init__(self, dim, heads=8, scalar_key_dim=16, scalar_value_dim=16,
                 point_key_dim=4, point_value_dim=4, pairwise_repr_dim=None,
                 require_pairwise_repr=True):
        super().__init__()
        self.eps = 1e-8; self.heads = heads
        self.require_pairwise_repr = require_pairwise_repr
        self.to_scalar_q = nn.Linear(dim, scalar_key_dim * heads, bias=False)
        self.to_scalar_k = nn.Linear(dim, scalar_key_dim * heads, bias=False)
        self.to_scalar_v = nn.Linear(dim, scalar_value_dim * heads, bias=False)
        self.to_point_q  = nn.Linear(dim, point_key_dim * heads * 3, bias=False)
        self.to_point_k  = nn.Linear(dim, point_key_dim * heads * 3, bias=False)
        self.to_point_v  = nn.Linear(dim, point_value_dim * heads * 3, bias=False)
        if require_pairwise_repr:
            pairwise_repr_dim = _ipa_default(pairwise_repr_dim, dim)
            self.pairwise_attn_logits = nn.Linear(pairwise_repr_dim, heads, bias=False)
        self.point_weights = nn.Parameter(torch.log(torch.exp(torch.ones(1, heads)) - 1.))
        num_pair_out = pairwise_repr_dim if require_pairwise_repr else 0
        output_dim = heads * (scalar_value_dim + point_value_dim * 3 + point_value_dim + num_pair_out)
        self.to_out = nn.Linear(output_dim, dim)
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
        def to_global(local_pts, t_exp):
            return torch.einsum('b h n i j, b h n p j -> b h n p i', re, local_pts) + t_exp
        pq_g = to_global(pq, te_q); pk_g = to_global(pk, te_q); pv_g = to_global(pv, te_v)
        sa = torch.einsum('b h i d, b h j d -> b h i j', sq, sk) * (self.scalar_key_dim ** -0.5)
        pw = F.softplus(self.point_weights).view(1, h, 1, 1)
        pdiff = (rearrange(pq_g, 'b h i p c -> b h i () p c') - rearrange(pk_g, 'b h j p c -> b h () j p c'))
        pa = -0.5 * (pw * (pdiff**2).sum(-1).sum(-1))
        attn_logits = sa + pa
        if self.require_pairwise_repr and _ipa_exists(pairwise_repr):
            attn_logits = attn_logits + rearrange(self.pairwise_attn_logits(pairwise_repr), 'b i j h -> b h i j')
        if _ipa_exists(mask):
            attn_logits = attn_logits.masked_fill(rearrange(mask.float(), 'b j -> b () () j') == 0, -1e9)
        attn = attn_logits.softmax(dim=-1)
        so = torch.einsum('b h i j, b h j d -> b h i d', attn, sv)
        pva = torch.einsum('b h i j, b h j p c -> b h i p c', attn, pv_g)
        rt = rotations.unsqueeze(1).expand(b, h, n, 3, 3).transpose(-1, -2)
        to_ = translations.unsqueeze(1).unsqueeze(3).expand(b, h, n, self.point_value_dim, 3)
        pvl = torch.einsum('b h n i j, b h n p j -> b h n p i', rt, pva - to_)
        pn = torch.norm(pvl, dim=-1)
        if self.require_pairwise_repr and _ipa_exists(pairwise_repr):
            po = torch.einsum('b h i j, b i j d -> b h i d', attn, pairwise_repr)
        else: po = None
        sf = rearrange(so, 'b h n d -> b n (h d)')
        pf = rearrange(pvl, 'b h n p c -> b n (h p c)')
        nf = rearrange(pn, 'b h n p -> b n (h p)')
        if _ipa_exists(po):
            out = torch.cat([sf, pf, nf, rearrange(po, 'b h n d -> b n (h d)')], dim=-1)
        else:
            out = torch.cat([sf, pf, nf], dim=-1)
        return self.to_out(out)

class IPABlock(nn.Module):
    def __init__(self, dim, ff_mult=1, **ipa_kwargs):
        super().__init__()
        self.attn = InvariantPointAttention(dim, **ipa_kwargs)
        self.attn_norm = nn.LayerNorm(dim)
        ff_dim = dim * ff_mult
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, dim))
    def forward(self, x, pairwise_repr=None, *, rotations, translations, mask=None):
        x = self.attn_norm(x + self.attn(x, pairwise_repr, rotations=rotations, translations=translations, mask=mask))
        return x + self.ff(x)

def build_rna_frames(coords):
    N = coords.shape[0]; eps = 1e-8
    rotations = torch.eye(3, device=coords.device).unsqueeze(0).expand(N, -1, -1).clone()
    translations = coords.clone()
    for i in range(N):
        has_next = (i < N - 1); has_prev = (i > 0)
        if not has_next and not has_prev: continue
        if has_next: x_vec = coords[i+1] - coords[i]
        else: x_vec = coords[i] - coords[i-1]
        x_len = x_vec.norm() + eps
        if x_len < eps * 10: continue
        x_axis = x_vec / x_len
        if has_prev:
            back_vec = coords[i-1] - coords[i]; back_axis = back_vec / (back_vec.norm() + eps)
        else:
            back_axis = torch.tensor([0., 1., 0.], device=coords.device)
            if abs(torch.dot(x_axis, back_axis).item()) > 0.9:
                back_axis = torch.tensor([0., 0., 1.], device=coords.device)
        z_vec = torch.cross(x_axis, back_axis, dim=0); z_len = z_vec.norm() + eps
        if z_len < eps * 10: continue
        z_axis = z_vec / z_len
        y_axis = torch.cross(z_axis, x_axis, dim=0); y_axis = y_axis / (y_axis.norm() + eps)
        rotations[i] = torch.stack([x_axis, y_axis, z_axis], dim=1)
    return rotations, translations

class IPAStructureModule(nn.Module):
    def __init__(self, single_dim, pair_dim, n_iter, heads):
        super().__init__()
        self.n_iter = n_iter; self.single_dim = single_dim
        self.pair_proj = nn.Sequential(nn.Linear(pair_dim, single_dim), nn.ReLU(), nn.LayerNorm(single_dim))
        self.single_norm = nn.LayerNorm(single_dim)
        self.ipa_block = IPABlock(dim=single_dim, heads=heads, scalar_key_dim=16, scalar_value_dim=16,
            point_key_dim=4, point_value_dim=4, pairwise_repr_dim=single_dim, require_pairwise_repr=True)
        self.coord_update = nn.Sequential(nn.LayerNorm(single_dim), nn.Linear(single_dim, single_dim // 2),
            nn.ReLU(), nn.Linear(single_dim // 2, 3))
    def forward(self, single_repr, pair_repr, init_coords):
        b, N, _ = single_repr.shape
        single = self.single_norm(single_repr); pair = self.pair_proj(pair_repr); coords = init_coords.clone()
        mask = torch.ones(b, N, device=single_repr.device, dtype=torch.bool)
        for _ in range(self.n_iter):
            rots, trans = build_rna_frames(coords)
            updated = self.ipa_block(single, pairwise_repr=pair, rotations=rots.unsqueeze(0),
                translations=trans.unsqueeze(0), mask=mask)
            delta = self.coord_update(updated).squeeze(0); coords = coords + delta; single = updated
        return coords

def fape_loss(pred_coords, true_coords, true_rotations, true_translations, clamp=10.0):
    t_i = true_translations.unsqueeze(1)
    pred_shifted = pred_coords.unsqueeze(0) - t_i; true_shifted = true_coords.unsqueeze(0) - t_i
    pred_local = torch.einsum('i r c, i j c -> i j r', true_rotations, pred_shifted)
    true_local = torch.einsum('i r c, i j c -> i j r', true_rotations, true_shifted)
    error = ((pred_local - true_local)**2).sum(dim=-1).clamp(min=0).sqrt()
    return torch.clamp(error, max=clamp).mean()

print("Cell 10 complete: TemplateEncoder, IPA modules, build_rna_frames, fape_loss defined.")


# ============================================================
# CELL 11: Load RibonanzaNet Backbone
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
backbone = backbone.to(device)

for p in backbone.parameters():
    p.requires_grad = False

UNFREEZE_LAST_N = 2
total_layers = len(list(backbone.transformer_encoder))
print(f"  Backbone has {total_layers} transformer layers")

unfrozen_backbone_params = []

if FREEZE_BACKBONE_IPA:
    print(f"  FREEZE_BACKBONE_IPA=True: all {total_layers} layers frozen.")
else:
    for i, layer in enumerate(backbone.transformer_encoder):
        if i >= total_layers - UNFREEZE_LAST_N:
            for p in layer.parameters():
                p.requires_grad = True
                unfrozen_backbone_params.append(p)
            print(f"  Layer {i}: UNFROZEN")
        else:
            print(f"  Layer {i}: frozen")

bb_frozen = sum(p.numel() for p in backbone.parameters() if not p.requires_grad)
bb_unfrozen = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
print(f"  Frozen: {bb_frozen:,}  Unfrozen: {bb_unfrozen:,}")

def _run_backbone(sequence, with_grad=False):
    base_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    tokens = torch.tensor([base_map.get(b, 4) for b in sequence.upper()], dtype=torch.long).unsqueeze(0).to(device)
    N = len(sequence); src_mask = torch.ones(1, N, dtype=torch.long, device=device)
    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with ctx:
        embedded = backbone.encoder(tokens)
        pairwise = backbone.outer_product_mean(embedded)
        pairwise = pairwise + backbone.pos_encoder(embedded)
        hidden = embedded
        for layer in backbone.transformer_encoder:
            result = layer(hidden, pairwise, src_mask=src_mask)
            if isinstance(result, tuple): hidden, pairwise = result
            else: hidden = result
    return pairwise, hidden

def get_pairwise_and_single_features(sequence):
    return _run_backbone(sequence, with_grad=False)

def get_pairwise_and_single_features_train(sequence):
    if FREEZE_BACKBONE_IPA:
        return _run_backbone(sequence, with_grad=False)
    base_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    tokens = torch.tensor([base_map.get(b, 4) for b in sequence.upper()], dtype=torch.long).unsqueeze(0).to(device)
    N = len(sequence); src_mask = torch.ones(1, N, dtype=torch.long, device=device)
    frozen_count = total_layers - UNFREEZE_LAST_N
    with torch.no_grad():
        embedded = backbone.encoder(tokens)
        pairwise = backbone.outer_product_mean(embedded)
        pairwise = pairwise + backbone.pos_encoder(embedded)
        hidden = embedded
        for i, layer in enumerate(backbone.transformer_encoder):
            if i < frozen_count:
                result = layer(hidden, pairwise, src_mask=src_mask)
                if isinstance(result, tuple): hidden, pairwise = result
                else: hidden = result
    hidden = hidden.detach().requires_grad_(True)
    pairwise = pairwise.detach().requires_grad_(True)
    for i, layer in enumerate(backbone.transformer_encoder):
        if i >= frozen_count:
            result = layer(hidden, pairwise, src_mask=src_mask)
            if isinstance(result, tuple): hidden, pairwise = result
            else: hidden = result
    return pairwise, hidden

_test_pair, _test_single = get_pairwise_and_single_features("AUGCAUGC")
assert _test_pair.shape == (1, 8, 8, 64)
assert _test_single.shape == (1, 8, 256)
del _test_pair, _test_single
print("  Backbone sanity check passed.")


# ============================================================
# CELL 12: Create IPA Model + Warm-Start
# ============================================================
PAIR_DIM = 64 + 16 + MSA_DIM  # 88
print(f"\nCreating IPA model (pair_dim={PAIR_DIM})...")

template_encoder = TemplateEncoder(template_dim=16, num_bins=22, max_dist=40.0).to(device)
ipa_module = IPAStructureModule(single_dim=IPA_DIM, pair_dim=PAIR_DIM, n_iter=IPA_ITERATIONS, heads=IPA_HEADS).to(device)

warmstart_path = RUN4_CHECKPOINT
if warmstart_path and os.path.exists(warmstart_path):
    print(f"  Warm-starting from: {warmstart_path}")
    ckpt = torch.load(warmstart_path, map_location=device, weights_only=False)
    if 'template_encoder_state_dict' in ckpt:
        template_encoder.load_state_dict(ckpt['template_encoder_state_dict'])
        print("  [OK] TemplateEncoder loaded.")
    if 'backbone_unfrozen_state' in ckpt:
        for layer_key, layer_state in ckpt['backbone_unfrozen_state'].items():
            layer_idx = int(layer_key.split('.')[-1])
            backbone.transformer_encoder[layer_idx].load_state_dict(layer_state)
        print("  [OK] Backbone warm-started.")
    print("  [NEW] IPAStructureModule: random init.")
else:
    print("  No warm-start checkpoint found. Training from scratch.")

ipa_params = sum(p.numel() for p in ipa_module.parameters())
te_params = sum(p.numel() for p in template_encoder.parameters())
print(f"  IPA params: {ipa_params:,}  TE params: {te_params:,}  Backbone unfrozen: {bb_unfrozen:,}")
print(f"  TOTAL TRAINABLE: {ipa_params + te_params + bb_unfrozen:,}")


# ============================================================
# CELL 13: Train IPA Module (THE CORE OF PHASE A)
# ============================================================
print("\n" + "="*60)
print("PHASE 3: IPA Training (FAPE loss)")
print("="*60)

BATCH_SIZE = 2; MAX_SEQ_LEN = 256; HEAD_LR = 1e-4; BACKBONE_LR = 1e-5
TEMPLATE_MASK_PROB = 0.5; MSA_MASK_PROB = 0.5

print(f"Loading training data from {TRAIN_PICKLE}...")
with open(TRAIN_PICKLE, 'rb') as f:
    raw_data = pickle.load(f)

train_items = []; skipped = 0
if isinstance(raw_data, dict):
    sequences = raw_data['sequence']; xyz_list = raw_data['xyz']
    for i in range(len(sequences)):
        try:
            seq = sequences[i]; residue_list = xyz_list[i]
            if seq is None or residue_list is None: skipped += 1; continue
            c1_coords = []; valid_bases = []
            for j, residue_atoms in enumerate(residue_list):
                if not hasattr(residue_atoms, 'keys') or 'sugar_ring' not in residue_atoms: continue
                sugar_ring = residue_atoms['sugar_ring']
                if sugar_ring is None or len(sugar_ring) == 0: continue
                c1_prime = sugar_ring[0]
                if np.isnan(c1_prime).any(): continue
                c1_coords.append(c1_prime)
                if j < len(seq): valid_bases.append(seq[j])
            if len(c1_coords) < 10: skipped += 1; continue
            coords = np.array(c1_coords, dtype=np.float32)
            clean_seq = ''.join(valid_bases[:len(c1_coords)])
            min_len = min(len(clean_seq), len(coords))
            if 10 <= min_len <= MAX_SEQ_LEN:
                train_items.append({'sequence': clean_seq[:min_len], 'coords': coords[:min_len]})
        except Exception as e:
            skipped += 1
            if skipped <= 3: print(f"  Warning {i}: {e}")
    print(f"  Parsed: {len(train_items)}, Skipped: {skipped}")

random.seed(42); random.shuffle(train_items)
val_size = max(1, int(len(train_items) * 0.1))
val_items = train_items[:val_size]; train_items = train_items[val_size:]
print(f"  Train: {len(train_items)}, Val: {val_size}")

def make_training_msa_proxy(seq, N):
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    bp_pairs = {(0,3):1.0, (3,0):1.0, (1,2):1.0, (2,1):1.0, (2,3):0.5, (3,2):0.5}
    bp_matrix = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        bi = base_to_idx.get(seq[i].upper(), -1)
        if bi < 0: continue
        for j in range(i+1, N):
            bj = base_to_idx.get(seq[j].upper(), -1)
            if bj < 0: continue
            score = bp_pairs.get((bi, bj), 0.0)
            bp_matrix[i, j] = score; bp_matrix[j, i] = score
    features = np.zeros((N, N, MSA_DIM), dtype=np.float32)
    features[:, :, 0] = bp_matrix * 0.3; features[:, :, 1] = bp_matrix * 0.2; features[:, :, 7] = 0.01
    return features

ipa_and_template_params = list(ipa_module.parameters()) + list(template_encoder.parameters())
optimizer = optim.AdamW([
    {'params': ipa_and_template_params, 'lr': HEAD_LR},
    {'params': unfrozen_backbone_params, 'lr': BACKBONE_LR},
], weight_decay=0.01)
all_trainable = ipa_and_template_params + unfrozen_backbone_params

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_EPOCHS)
scaler = GradScaler(enabled=(device.type == 'cuda'))

ipa_module.train(); template_encoder.train()
best_val_loss = float('inf'); epochs_no_improve = 0; train_start = time.time()

print(f"\nStarting training: {TRAIN_EPOCHS} epoch ceiling, patience={EARLY_STOP_PATIENCE}")
print(f"Strategy: {'FROZEN backbone' if FREEZE_BACKBONE_IPA else 'UNFROZEN last 2 layers'}")

for epoch in range(TRAIN_EPOCHS):
    epoch_loss = 0.0; n_batches = 0; random.shuffle(train_items)
    for batch_start in range(0, len(train_items), BATCH_SIZE):
        batch = train_items[batch_start:batch_start + BATCH_SIZE]; batch_losses = []
        for item in batch:
            seq = item['sequence']; true_coords = item['coords']; N = len(seq)
            pairwise, single = get_pairwise_and_single_features_train(seq)
            if random.random() < TEMPLATE_MASK_PROB:
                tmpl_feat = torch.zeros(1, N, N, 16, device=device)
            else:
                coords_t = torch.tensor(true_coords, dtype=torch.float32, device=device)
                tmpl_feat = template_encoder(coords_t, confidence=1.0, has_template=True).unsqueeze(0)
            if random.random() < MSA_MASK_PROB:
                msa_feat = torch.zeros(1, N, N, MSA_DIM, device=device)
            else:
                msa_np = make_training_msa_proxy(seq, N)
                msa_feat = torch.tensor(msa_np, dtype=torch.float32, device=device).unsqueeze(0)
            combined = torch.cat([pairwise, tmpl_feat, msa_feat], dim=-1)
            true_coords_t = torch.tensor(true_coords, dtype=torch.float32, device=device)
            if random.random() < 0.5:
                init_coords = true_coords_t + torch.randn_like(true_coords_t) * 1.0
            else:
                init_coords = torch.zeros_like(true_coords_t)
            pred_coords = ipa_module(single, combined, init_coords)
            true_rots, true_trans = build_rna_frames(true_coords_t)
            fape = fape_loss(pred_coords, true_coords_t, true_rots, true_trans, clamp=FAPE_CLAMP)
            pred_dist = torch.cdist(pred_coords.unsqueeze(0), pred_coords.unsqueeze(0)).squeeze(0)
            true_dist = torch.cdist(true_coords_t.unsqueeze(0), true_coords_t.unsqueeze(0)).squeeze(0)
            aux_loss = ((pred_dist - true_dist)**2).mean()
            loss = fape + AUX_DIST_WEIGHT * aux_loss
            batch_losses.append(loss)
        if batch_losses:
            total_loss = sum(batch_losses) / len(batch_losses)
            optimizer.zero_grad(); scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(all_trainable, 1.0)
            scaler.step(optimizer); scaler.update()
            epoch_loss += total_loss.item(); n_batches += 1

    scheduler.step(); avg_train = epoch_loss / max(n_batches, 1)

    ipa_module.eval(); template_encoder.eval()
    val_loss = 0.0
    with torch.no_grad():
        for item in val_items[:20]:
            seq = item['sequence']; true_coords = item['coords']; N = len(seq)
            pairwise, single = get_pairwise_and_single_features(seq)
            coords_t = torch.tensor(true_coords, dtype=torch.float32, device=device)
            tmpl_feat = template_encoder(coords_t, confidence=1.0, has_template=True).unsqueeze(0)
            msa_np = make_training_msa_proxy(seq, N)
            msa_feat = torch.tensor(msa_np, dtype=torch.float32, device=device).unsqueeze(0)
            combined = torch.cat([pairwise, tmpl_feat, msa_feat], dim=-1)
            init_coords = torch.zeros_like(coords_t)
            pred_coords = ipa_module(single, combined, init_coords)
            true_rots, true_trans = build_rna_frames(coords_t)
            fape = fape_loss(pred_coords, coords_t, true_rots, true_trans, clamp=FAPE_CLAMP)
            val_loss += fape.item()
    avg_val = val_loss / max(len(val_items[:20]), 1)
    elapsed = time.time() - train_start
    print(f"Epoch {epoch+1}/{TRAIN_EPOCHS}: train={avg_train:.3f}A  val={avg_val:.3f}A  t={elapsed:.0f}s")

    if avg_val < best_val_loss:
        best_val_loss = avg_val; epochs_no_improve = 0
        torch.save({
            'ipa_module_state_dict': ipa_module.state_dict(),
            'template_encoder_state_dict': template_encoder.state_dict(),
            'backbone_unfrozen_state': {
                f'transformer_encoder.{i}': backbone.transformer_encoder[i].state_dict()
                for i in range(total_layers - UNFREEZE_LAST_N, total_layers)
            } if not FREEZE_BACKBONE_IPA else {},
            'epoch': epoch, 'val_fape': avg_val, 'pair_dim': PAIR_DIM,
            'ipa_iterations': IPA_ITERATIONS, 'ipa_heads': IPA_HEADS, 'ipa_dim': IPA_DIM,
        }, CHECKPOINT_PATH)
        print(f"  -> Saved best model (val_fape={avg_val:.3f} A)")
    else:
        epochs_no_improve += 1
        print(f"     No improvement ({epochs_no_improve}/{EARLY_STOP_PATIENCE})")
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}. Best val_fape={best_val_loss:.3f} A")
            break
    ipa_module.train(); template_encoder.train()

print(f"\nTraining complete. Best val_fape: {best_val_loss:.3f} A")
print(f"Total training time: {time.time()-train_start:.0f}s")
print(f"Checkpoint saved: {CHECKPOINT_PATH}")

# ============================================================
# PHASE A COMPLETE
# ============================================================
print("\n" + "="*60)
print("PHASE A COMPLETE")
print("="*60)
print(f"  Checkpoint: {CHECKPOINT_PATH}")
print(f"  Best val FAPE: {best_val_loss:.3f} A")
print(f"  Platform: {PLATFORM}")
if PLATFORM == 'LOCAL':
    print(f"\n  NEXT STEPS:")
    print(f"    1. Upload {CHECKPOINT_PATH} to Kaggle as a dataset")
    print(f"    2. Run Phase B on Kaggle (inference only, ~30 min)")
elif PLATFORM == 'KAGGLE':
    print(f"\n  NEXT STEPS:")
    print(f"    1. Save this notebook's output as a dataset")
    print(f"    2. Attach that dataset to Phase B notebook")
    print(f"    3. Run Phase B (inference only, ~30 min)")
