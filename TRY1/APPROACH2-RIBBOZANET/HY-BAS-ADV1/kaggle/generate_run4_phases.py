"""
generate_run4_phases.py — Generates Phase A and Phase B from the original Run 4 file.

Usage:
    cd to this directory, then:
    python generate_run4_phases.py

Creates:
    hy_bas_adv1_run4_commit_PhaseA_NB.py  (training only, Cells 0-13)
    hy_bas_adv1_run4_commit_PhaseB_NB.py  (inference only, Cells 0-12 + 14-15)
"""
import os, re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGINAL = os.path.join(SCRIPT_DIR, 'hy_bas_adv1_run4_commit_NB.py')

if not os.path.exists(ORIGINAL):
    print(f"ERROR: Cannot find {ORIGINAL}")
    exit(1)

with open(ORIGINAL, 'r') as f:
    lines = f.readlines()

print(f"Read {len(lines)} lines from original")

# Find cell boundaries
cell_line = {}
for i, line in enumerate(lines):
    if '# CELL 0:' in line: cell_line[0] = i
    if '# CELL 1:' in line and 'biopython' in line.lower(): cell_line[1] = i
    if '# CELL 13:' in line and 'Train' in line: cell_line[13] = i
    if '# CELL 14:' in line: cell_line[14] = i
    if '# CELL 15:' in line: cell_line[15] = i
    if 'Loaded best checkpoint for inference' in line:
        cell_line['reload_end'] = i

for k, v in sorted(cell_line.items(), key=lambda x: str(x[0])):
    print(f"  Cell {k}: line {v+1}")

# ============================================================
# COMMON: New Cell 0 header with LOCAL/KAGGLE support
# ============================================================
CELL0_PHASEA = '''# ============================================================
# HY-BAS-ADV1 RUN 4 — PHASE A: Distance Head Training (Local or Kaggle)
# ============================================================
#
# WHAT THIS IS:
#   The TRAINING half of Run 4, split from the monolithic file.
#   The original hy_bas_adv1_run4_commit_NB.py is UNCHANGED.
#
#   Phase A = Cells 0-13 (data loading, model, training)
#   Phase B = Cells 0-12 + 14-15 (load checkpoint, inference)
#
# WHERE TO RUN:
#   LOCAL:  python hy_bas_adv1_run4_commit_PhaseA_NB.py
#           RTX 3070 Ti (8 GB) — plenty of VRAM for distance head.
#           No time limit. Train 80-100 epochs overnight.
#
#   KAGGLE: Paste into notebook, run, save output as dataset.
#           Output checkpoint becomes input for Phase B notebook.
#
# OUTPUTS:
#   adv1_best_model.pt
#
# NEXT STEP:
#   Upload adv1_best_model.pt as Kaggle dataset.
#   Then run Phase B on Kaggle (inference only, ~30-45 min).
# ============================================================


# ============================================================
# CELL 0: PATHS AND CONFIG
# ============================================================

# --- OPTION A: LOCAL PATHS (uncomment and edit) ---------------
# DATA_ROOT          = 'E:/kaggle_data'
# COMP_BASE          = f'{DATA_ROOT}/stanford-rna-3d-folding-2'
# EXTENDED_SEQ_CSV   = f'{DATA_ROOT}/rna_sequences.csv'
# EXTENDED_COORD_CSV = f'{DATA_ROOT}/rna_coordinates.csv'
# TRAIN_PICKLE       = f'{DATA_ROOT}/pdb_xyz_data.pkl'
# BACKBONE_WEIGHTS   = f'{DATA_ROOT}/RibonanzaNet.pt'
# REPO_PATH          = f'{DATA_ROOT}/ribonanzanet_repo'
# RUN3_CHECKPOINT    = f'{DATA_ROOT}/adv1_best_run3optb_model.pt'
# OUTPUT_DIR         = './run4_output'
# PLATFORM           = 'LOCAL'
# -----------------------------------------------------------------

# --- OPTION B: KAGGLE PATHS (default — auto-discovers) -----------
EXTENDED_SEQ_CSV   = None
EXTENDED_COORD_CSV = None
TRAIN_PICKLE       = None
BACKBONE_WEIGHTS   = None
REPO_PATH          = None
RUN3_CHECKPOINT    = None
OUTPUT_DIR         = '/kaggle/working'
PLATFORM           = 'KAGGLE'
COMP_BASE_PRIMARY  = '/kaggle/input/competitions/stanford-rna-3d-folding-2/'
COMP_BASE_FALLBACK = '/kaggle/input/stanford-rna-3d-folding-2/'
# -----------------------------------------------------------------

MSA_TOP_N = 20
MSA_DIM   = 8
TRAIN_EPOCHS = 50  # LOCAL: increase to 80-100

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHECKPOINT_PATH = f'{OUTPUT_DIR}/adv1_best_model.pt'
RAW_SUBMISSION_PATH = f'{OUTPUT_DIR}/submission_raw.csv'
FINAL_SUBMISSION_PATH = f'{OUTPUT_DIR}/submission.csv'

'''

CELL0_PHASEB = '''# ============================================================
# HY-BAS-ADV1 RUN 4 — PHASE B: Inference Only (Kaggle)
# ============================================================
#
# WHAT THIS IS:
#   The INFERENCE half of Run 4. Loads a checkpoint trained by
#   Phase A (locally or on Kaggle) and runs inference only.
#   The original hy_bas_adv1_run4_commit_NB.py is UNCHANGED.
#
# PREREQUISITES:
#   Phase A checkpoint (adv1_best_model.pt) uploaded as Kaggle dataset.
#
# WHAT IS SKIPPED:
#   Cell 13 (training) — ENTIRELY SKIPPED.
#
# ESTIMATED RUNTIME: ~30-45 min on Kaggle T4
# ============================================================


# ============================================================
# CELL 0: PATHS AND CONFIG
# ============================================================

# --- OPTION A: LOCAL PATHS (uncomment and edit) ---------------
# DATA_ROOT          = 'E:/kaggle_data'
# COMP_BASE          = f'{DATA_ROOT}/stanford-rna-3d-folding-2'
# EXTENDED_SEQ_CSV   = f'{DATA_ROOT}/rna_sequences.csv'
# EXTENDED_COORD_CSV = f'{DATA_ROOT}/rna_coordinates.csv'
# BACKBONE_WEIGHTS   = f'{DATA_ROOT}/RibonanzaNet.pt'
# REPO_PATH          = f'{DATA_ROOT}/ribonanzanet_repo'
# PHASE_A_CHECKPOINT = './run4_output/adv1_best_model.pt'
# OUTPUT_DIR         = './run4_output'
# SAMPLE_CSV         = f'{COMP_BASE}/sample_submission.csv'
# PLATFORM           = 'LOCAL'
# -----------------------------------------------------------------

# --- OPTION B: KAGGLE PATHS (default) ---------------------------
EXTENDED_SEQ_CSV   = None
EXTENDED_COORD_CSV = None
BACKBONE_WEIGHTS   = None
REPO_PATH          = None
PHASE_A_CHECKPOINT = None
OUTPUT_DIR         = '/kaggle/working'
SAMPLE_CSV         = None
PLATFORM           = 'KAGGLE'
COMP_BASE_PRIMARY  = '/kaggle/input/competitions/stanford-rna-3d-folding-2/'
COMP_BASE_FALLBACK = '/kaggle/input/stanford-rna-3d-folding-2/'
# -----------------------------------------------------------------

MSA_TOP_N = 20
MSA_DIM   = 8

RAW_SUBMISSION_PATH   = f'{OUTPUT_DIR}/submission_raw.csv'
FINAL_SUBMISSION_PATH = f'{OUTPUT_DIR}/submission.csv'

'''

# ============================================================
# Cell 3 replacement: supports LOCAL + KAGGLE path discovery
# ============================================================
CELL3_PHASEA = '''
# ============================================================
# CELL 3: Find all data paths
# ============================================================
import os
if PLATFORM == 'KAGGLE':
    BASE = COMP_BASE_PRIMARY
    if not os.path.exists(BASE):
        BASE = COMP_BASE_FALLBACK
    ADV1_WEIGHTS = None
    for root, dirs, files in os.walk("/kaggle/input"):
        if "Network.py" in files and REPO_PATH is None:
            REPO_PATH = root
        for f in files:
            if f == "RibonanzaNet.pt" and BACKBONE_WEIGHTS is None:
                BACKBONE_WEIGHTS = os.path.join(root, f)
            if f == "best_model.pt":
                ADV1_WEIGHTS = os.path.join(root, f)
            if f.strip() == "adv1_best_run3optb_model.pt" and RUN3_CHECKPOINT is None:
                RUN3_CHECKPOINT = os.path.join(root, f)
            if f == "pdb_xyz_data.pkl" and TRAIN_PICKLE is None:
                TRAIN_PICKLE = os.path.join(root, f)
            if f == "sample_submission.csv":
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
    ADV1_WEIGHTS = None

print(f"Platform:           {PLATFORM}")
print(f"Competition data:   {BASE}")
print(f"RibonanzaNet repo:  {REPO_PATH}")
print(f"Backbone weights:   {BACKBONE_WEIGHTS}")
print(f"Run 3 checkpoint:   {RUN3_CHECKPOINT}")
print(f"Training pickle:    {TRAIN_PICKLE}")
print(f"Extended seqs CSV:  {EXTENDED_SEQ_CSV}")
print(f"Extended coords CSV:{EXTENDED_COORD_CSV}")
print(f"Output dir:         {OUTPUT_DIR}")
print(f"Checkpoint path:    {CHECKPOINT_PATH}")

'''

CELL3_PHASEB = '''
# ============================================================
# CELL 3: Find all data paths + PHASE A CHECKPOINT
# ============================================================
import os
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
            if f == "adv1_best_model.pt" and PHASE_A_CHECKPOINT is None:
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

print(f"Platform:           {PLATFORM}")
print(f"Competition data:   {BASE}")
print(f"Phase A checkpoint: {PHASE_A_CHECKPOINT}")
print(f"Sample submission:  {SAMPLE_CSV}")

if PHASE_A_CHECKPOINT is None or not os.path.exists(PHASE_A_CHECKPOINT):
    raise FileNotFoundError(
        "Phase A checkpoint (adv1_best_model.pt) not found!\\n"
        "Run Phase A first, then upload its output as a Kaggle dataset."
    )

'''

# ============================================================
# Checkpoint load block for Phase B (replaces Cell 13)
# ============================================================
CHECKPOINT_LOAD = '''
# ============================================================
# CELL 13: SKIPPED — Training was done in Phase A
# ============================================================
print("\\n  Cell 13 (training): SKIPPED — loading Phase A checkpoint.")

print(f"\\nLoading Phase A checkpoint: {PHASE_A_CHECKPOINT}")
ckpt = torch.load(PHASE_A_CHECKPOINT, map_location=device, weights_only=False)
distance_head.load_state_dict(ckpt['model_state_dict'])
print(f"  [OK] Distance head loaded")
if 'template_encoder_state_dict' in ckpt:
    template_encoder.load_state_dict(ckpt['template_encoder_state_dict'])
    print(f"  [OK] TemplateEncoder loaded")
if 'backbone_unfrozen_state' in ckpt:
    for layer_key, layer_state in ckpt['backbone_unfrozen_state'].items():
        layer_idx = int(layer_key.split('.')[-1])
        backbone.transformer_encoder[layer_idx].load_state_dict(layer_state)
    print(f"  [OK] Backbone unfrozen layers loaded")
print(f"  Trained epoch: {ckpt.get('epoch', '?')}, Val loss: {ckpt.get('val_loss', '?')}")
distance_head.eval()
template_encoder.eval()
backbone.eval()
print("  All models set to eval mode. NO TRAINING IN PHASE B.")

'''

PHASEA_FOOTER = '''
# ============================================================
# PHASE A COMPLETE
# ============================================================
print("\\n" + "="*60)
print("PHASE A COMPLETE")
print("="*60)
print(f"  Checkpoint: {CHECKPOINT_PATH}")
print(f"  Best val_loss: {best_val_loss:.4f}")
print(f"  Platform: {PLATFORM}")
if PLATFORM == 'LOCAL':
    print(f"\\n  NEXT STEPS:")
    print(f"    1. Upload {CHECKPOINT_PATH} to Kaggle as a dataset")
    print(f"    2. Run Phase B on Kaggle (inference only, ~30-45 min)")
elif PLATFORM == 'KAGGLE':
    print(f"\\n  NEXT STEPS:")
    print(f"    1. Save this notebook output as a dataset")
    print(f"    2. Attach that dataset to Phase B notebook")
    print(f"    3. Run Phase B (inference only, ~30-45 min)")
'''

# ============================================================
# Find exact line numbers for Cell boundaries
# ============================================================
# Cell 1 starts 2 lines before "# CELL 1:"
cell1_start = cell_line[1] - 2
# Cell 3 ends where Cell 4 starts (look for "# CELL 4:")
cell4_start = None
for i, line in enumerate(lines):
    if '# CELL 4:' in line:
        cell4_start = i - 2
        break

# Cell 13 starts 2 lines before marker
cell13_start = cell_line[13] - 2
# Cell 14 starts 2 lines before marker
cell14_start = cell_line[14] - 2
# End of checkpoint reload
reload_end = cell_line['reload_end'] + 1

print(f"  Cell 1 starts: {cell1_start+1}")
print(f"  Cell 4 starts: {cell4_start+1}")
print(f"  Cell 13 starts: {cell13_start+1}")
print(f"  Cell 14 starts: {cell14_start+1}")
print(f"  Reload ends: {reload_end+1}")

# ============================================================
# GENERATE PHASE A
# ============================================================

# Cells 1-2 (biopython install + imports)
cells_1_2 = ''.join(lines[cell1_start:cell4_start])
# Fix biopython for LOCAL
cells_1_2 = cells_1_2.replace(
    "    bio_inits = glob.glob('/kaggle/input/**/Bio/__init__.py', recursive=True)",
    "    if PLATFORM == 'KAGGLE':\n        bio_inits = glob.glob('/kaggle/input/**/Bio/__init__.py', recursive=True)\n    else:\n        bio_inits = []"
)

# Cell 4 onwards through end of training + reload
cells_4_to_reload = ''.join(lines[cell4_start:reload_end + 1])
# Fix path joins
cells_4_to_reload = cells_4_to_reload.replace(
    "train_seqs = pd.read_csv(BASE + 'train_sequences.csv')",
    "train_seqs = pd.read_csv(os.path.join(BASE, 'train_sequences.csv'))"
).replace(
    "test_seqs = pd.read_csv(BASE + 'test_sequences.csv')",
    "test_seqs = pd.read_csv(os.path.join(BASE, 'test_sequences.csv'))"
).replace(
    "train_labels = pd.read_csv(BASE + 'train_labels.csv')",
    "train_labels = pd.read_csv(os.path.join(BASE, 'train_labels.csv'))"
)

phaseA = CELL0_PHASEA + cells_1_2 + CELL3_PHASEA + cells_4_to_reload + PHASEA_FOOTER

phaseA_path = os.path.join(SCRIPT_DIR, 'hy_bas_adv1_run4_commit_PhaseA_NB.py')
with open(phaseA_path, 'w', newline='\n') as f:
    f.write(phaseA)
print(f"\n[OK] Phase A: {len(phaseA.splitlines())} lines -> {phaseA_path}")

# ============================================================
# GENERATE PHASE B
# ============================================================

# Cells 1-2 (same biopython fix)
# Cell 4 through end of Cell 12
cells_4_to_12 = ''.join(lines[cell4_start:cell13_start])
cells_4_to_12 = cells_4_to_12.replace(
    "train_seqs = pd.read_csv(BASE + 'train_sequences.csv')",
    "train_seqs = pd.read_csv(os.path.join(BASE, 'train_sequences.csv'))"
).replace(
    "test_seqs = pd.read_csv(BASE + 'test_sequences.csv')",
    "test_seqs = pd.read_csv(os.path.join(BASE, 'test_sequences.csv'))"
).replace(
    "train_labels = pd.read_csv(BASE + 'train_labels.csv')",
    "train_labels = pd.read_csv(os.path.join(BASE, 'train_labels.csv'))"
)

# Cells 14-15 (inference + post-processing)
cells_14_15 = ''.join(lines[cell14_start:])
cells_14_15 = cells_14_15.replace(
    'print("DONE! submission.csv is ready for scoring")',
    'print("DONE! Run 4 Phase B (inference only) complete.")'
)

phaseB = CELL0_PHASEB + cells_1_2 + CELL3_PHASEB + cells_4_to_12 + CHECKPOINT_LOAD + cells_14_15

phaseB_path = os.path.join(SCRIPT_DIR, 'hy_bas_adv1_run4_commit_PhaseB_NB.py')
with open(phaseB_path, 'w', newline='\n') as f:
    f.write(phaseB)
print(f"[OK] Phase B: {len(phaseB.splitlines())} lines -> {phaseB_path}")

# Verify
print(f"\nVerification:")
print(f"  Original unchanged: {os.path.exists(ORIGINAL)} ({len(lines)} lines)")
print(f"  Phase A created:    {os.path.exists(phaseA_path)}")
print(f"  Phase B created:    {os.path.exists(phaseB_path)}")
print(f"\nDone! You can delete this script (generate_run4_phases.py).")
