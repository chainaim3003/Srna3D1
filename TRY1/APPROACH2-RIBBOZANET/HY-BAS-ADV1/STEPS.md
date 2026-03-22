# HY-BAS-ADV1: Step-by-Step Execution Guide

## Final Directory
```
C:\SATHYA\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH2-RIBBOZANET\HY-BAS-ADV1\
```

---

## STEP 1: Copy BASIC Files — DONE ✅

All files created via re-import wrappers (each .py imports from ../BASIC/):

```
HY-BAS-ADV1/
├── config_adv1.yaml           ✅ Created + paths verified
├── DESIGN.md                  ✅ Created
├── STEPS.md                   ✅ This file
├── predict_adv1.py            ✅ Inference with templates + warm-start
├── train_adv1.py              ✅ Training with self-templates + 50% masking
├── setup_from_basic.bat       ✅ Optional flat copy script
├── models/
│   ├── __init__.py            ✅ Updated imports
│   ├── backbone.py            ✅ Re-imports from BASIC
│   ├── distance_head.py       ✅ Re-imports from BASIC
│   ├── reconstructor.py       ✅ Re-imports from BASIC
│   ├── template_encoder.py    ✅ NEW: coords->distances->bins->features
│   └── template_loader.py     ✅ NEW: reads Approach 1 outputs
├── data/
│   ├── __init__.py            ✅ Re-imports from BASIC
│   ├── dataset.py             ✅ Re-imports from BASIC
│   └── collate.py             ✅ Re-imports from BASIC
├── losses/
│   ├── __init__.py            ✅ Created
│   ├── distance_loss.py       ✅ Re-imports from BASIC
│   └── constraint_loss.py     ✅ Re-imports from BASIC
├── utils/
│   ├── __init__.py            ✅ Created
│   └── submission.py          ✅ Re-imports from BASIC
└── checkpoints/               ✅ Empty (fills during training)
```

---

## STEP 2: Verify Config Paths — DONE ✅ (BLOCKERS FOUND + ROOT CAUSE IDENTIFIED)

### Paths That EXIST:

| Config Key | File | Status |
|-----------|------|--------|
| `warm_start.basic_checkpoint` | `../BASIC/checkpoints/best_model.pt` (312KB) | ✅ |
| `template.local_result_txt` | `../../APPROACH1-TEMPLATE/mine/kalai/run1/Result_20260321_1100_UTC_kalai.txt.txt` | ✅ |
| `template.local_submission_csv` | `../../APPROACH1-TEMPLATE/mine/kalai/run1/submission_20260321_1100_UTC_kalai.csv.csv` | ✅ |
| `data.test_csv_path` | `../../APPROACH1-TEMPLATE/test_sequences (1).csv` | ✅ |

### Paths That DO NOT EXIST — ROOT CAUSE FOUND:

**Srna3D1/.gitignore confirms these directories EXISTED during BASIC training:**
```
# Cloned repos (download separately)
RibonanzaNet/
Ranger-Deep-Learning-Optimizer/

# Large data files (download from Kaggle)
ribonanza-weights/
stanford3d-pickle/
```

In a previous session, `Srna3D1/` contained:
```
[DIR] Ranger-Deep-Learning-Optimizer    (git clone)
[DIR] ribonanza-weights                 (Kaggle download, ~43MB)
[DIR] RibonanzaNet                      (git clone)
[DIR] stanford3d-pickle                 (training data, ~52MB)
[DIR] TRY1                             (our work)
```

BASIC training COMPLETED using these (proof: best_model.pt exists at 312KB).
They are now missing — either deleted after training or on a different machine.

### RECOVERY COMMANDS (Run Once):

```cmd
cd C:\SATHYA\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1

REM 1. Clone RibonanzaNet repo (~10MB)
git clone https://github.com/Shujun-He/RibonanzaNet

REM 2. Clone Ranger optimizer (needed by RibonanzaNet)
git clone https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer

REM 3. Download pretrained backbone weights (~43MB)
REM Option A: Kaggle CLI
mkdir ribonanza-weights
kaggle datasets download shujun717/ribonanzanet-weights -p ribonanza-weights --unzip

REM Option B: Manual download from browser
REM Go to: https://www.kaggle.com/datasets/shujun717/ribonanzanet-weights
REM Download and extract to Srna3D1/ribonanza-weights/

REM 4. Training data pickle (~52MB)
REM This was generated during BASIC data preparation.
REM Check if it was saved elsewhere or needs regeneration.
REM The pickle was created from CIF files in an earlier session.
mkdir stanford3d-pickle
REM If pdb_xyz_data.pkl exists elsewhere, copy it here.
REM If not, it needs to be regenerated from:
REM   material/PDB files-20260302T033106Z-1-001/PDB_Files/
```

### After Recovery, Verify:
```cmd
dir Srna3D1\RibonanzaNet\Network.py
dir Srna3D1\ribonanza-weights\RibonanzaNet.pt
dir Srna3D1\stanford3d-pickle\pdb_xyz_data.pkl
dir Srna3D1\Ranger-Deep-Learning-Optimizer\ranger\ranger.py
```

All 4 should exist. If any is missing, ADV1 cannot proceed.

---

## STEP 3: Pre-Step 1 — Verify Neural Network Runs on Kaggle (2-3 hours)

**BLOCKED UNTIL:** Recovery commands from Step 2 complete.
Specifically need: RibonanzaNet.pt to upload to Kaggle.

### 3.1: Prepare 3 Kaggle Datasets

**Dataset 1: "adv1-weights-test"**
```
Contents: RibonanzaNet.pt + best_model.pt
Source:   Srna3D1/ribonanza-weights/RibonanzaNet.pt
          BASIC/checkpoints/best_model.pt
```

**Dataset 2: "ribonanzanet-repo"**
```
Contents: Entire RibonanzaNet/ folder
Source:   Srna3D1/RibonanzaNet/
```

**Dataset 3: "adv1-code-test"**
```
Contents: FULL .py files (not re-import wrappers)
Source:   BASIC/models/backbone.py      (the FULL 400-line file)
          BASIC/models/distance_head.py  (the FULL file)
          BASIC/models/reconstructor.py  (the FULL file)
          HY-BAS-ADV1/models/template_encoder.py
          HY-BAS-ADV1/models/template_loader.py
          BASIC/utils/submission.py       (the FULL file)
```

IMPORTANT: For Kaggle uploads, use the FULL .py files from BASIC/,
not the re-import wrappers in HY-BAS-ADV1/. The wrappers use
sys.path hacks that only work in the local directory structure.

### 3.2: Upload to Kaggle
1. kaggle.com/datasets -> New Dataset
2. Upload each as private dataset
3. Note exact slugs

### 3.3: Create Test Notebook
```python
import torch, sys, os

print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Find RibonanzaNet repo
for root, dirs, files in os.walk("/kaggle/input"):
    if "Network.py" in files:
        repo_path = root
        print("Found RibonanzaNet at:", repo_path)
        sys.path.insert(0, repo_path)
        break

from Network import RibonanzaNet
print("RibonanzaNet class imported OK")

# Load weights
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f == "RibonanzaNet.pt":
            w = torch.load(os.path.join(root, f), map_location="cpu")
            print("Backbone weights loaded:", len(w), "tensors")
            break

for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f == "best_model.pt":
            h = torch.load(os.path.join(root, f), map_location="cpu")
            print("Distance head loaded:", type(h))
            break

print("\nALL CHECKS PASSED")
```

If ALL CHECKS PASSED -> proceed to Step 4.

---

## STEP 4: Train ADV1 Locally (2-4 hours)

**BLOCKED UNTIL:** All recovery commands from Step 2 complete.

```cmd
cd C:\SATHYA\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH2-RIBBOZANET\HY-BAS-ADV1

python train_adv1.py --config config_adv1.yaml
```

What happens:
1. Loads frozen RibonanzaNet backbone from Srna3D1/RibonanzaNet/
2. Creates ADV1 distance head (pair_dim=80)
3. Creates template encoder (22 bins -> 16 features)
4. Warm-starts from BASIC's best_model.pt
5. Trains 30 epochs, 50% template masking
6. Saves checkpoints/adv1_best_model.pt

Common errors:
- "No module named Network" -> RibonanzaNet not cloned (Step 2 item 1)
- "FileNotFoundError: RibonanzaNet.pt" -> weights not downloaded (Step 2 item 3)
- "CUDA out of memory" -> reduce batch_size to 2 in config
- "FileNotFoundError: pdb_xyz_data.pkl" -> training data missing (Step 2 item 4)

---

## STEP 5: Test Predictions Locally (15 minutes)

**BLOCKED UNTIL:** Step 4 produces adv1_best_model.pt

```cmd
python predict_adv1.py --config config_adv1.yaml --checkpoint checkpoints/adv1_best_model.pt
```

Verify: 9762 rows, 18 columns in submission_adv1.csv.

---

## STEP 6: Build Kaggle Notebook (3-4 hours)

**BLOCKED UNTIL:** Steps 3, 4, 5 pass.

Notebook cells:
```
Cell 1:   Symlink fix
Cell 2:   pip install dependencies
Cell 3-8: MMseqs2 pipeline (from Fork 1 rhijudas)
Cell 9:   Load ADV1 model + template encoder
Cell 10:  Run inference with templates
Cell 11:  Option B post-processing
Cell 12:  Verification
```

---

## STEP 7: Submit (30 minutes)

Internet OFF -> Save & Run All (Commit) -> Submit.

---

## CURRENT STATUS SUMMARY

| Step | Status | Blocker |
|------|--------|---------|
| Step 1: Create files | ✅ DONE | — |
| Step 2: Verify paths | ✅ DONE | 3 missing dirs need recovery |
| Step 2b: Recovery | ❌ TODO | Team must run git clone + downloads |
| Step 3: Kaggle pre-test | ❌ BLOCKED | Needs RibonanzaNet.pt |
| Step 4: Train locally | ❌ BLOCKED | Needs all 3 recovered dirs |
| Step 5: Test locally | ❌ BLOCKED | Needs Step 4 |
| Step 6: Kaggle notebook | ❌ BLOCKED | Needs Steps 3-5 |
| Step 7: Submit | ❌ BLOCKED | Needs Step 6 |

## IMMEDIATE ACTION NEEDED

**The team must run the 4 recovery commands from Step 2** to restore:
1. `Srna3D1/RibonanzaNet/` (git clone)
2. `Srna3D1/Ranger-Deep-Learning-Optimizer/` (git clone)
3. `Srna3D1/ribonanza-weights/RibonanzaNet.pt` (Kaggle download)
4. `Srna3D1/stanford3d-pickle/pdb_xyz_data.pkl` (from previous session or regenerate)

Steps 3 and 4 can run in PARALLEL once recovery is done.

## PARALLEL TRACK REMINDER

While resolving blockers, submit Fork 2 + Option B NOW:
- See: fork2-JJ/FORK2_OPTB_KAGGLE_STEPS.md
- Does NOT depend on any of these missing files
- Gets a fallback score on the leaderboard immediately

## OLD DIRECTORY CLEANUP

Delete stale directory (has old DESIGN.md):
```cmd
rmdir /s C:\SATHYA\...\Srna3D1\TRY1\HY-BAS-ADV1
```
The canonical location is TRY1/APPROACH2-RIBBOZANET/HY-BAS-ADV1/
