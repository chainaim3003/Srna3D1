# HY-BAS-ADV1: Step-by-Step Execution Guide

## Final Directory
```
C:\SATHYA\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH2-RIBBOZANET\HY-BAS-ADV1\
```

## Sibling Reference Directories
```
..\BASIC\                          (BASIC model code + trained weights)
..\..\APPROACH1-TEMPLATE\          (Approach 1 outputs: Result.txt, submission.csv)
..\..\..\..\RibonanzaNet\          (Cloned RibonanzaNet GitHub repo)
..\..\..\..\ribonanza-weights\     (RibonanzaNet.pt pretrained backbone)
..\..\..\..\stanford3d-pickle\     (pdb_xyz_data.pkl training data)
```

---

## WHAT EXISTS RIGHT NOW (Already Created)

| File | Status | Description |
|------|--------|-------------|
| `DESIGN.md` | DONE | Full design document with architecture, training vs inference explanation |
| `config_adv1.yaml` | DONE | Configuration file (pair_dim=80, template_dim=16, paths) |
| `predict_adv1.py` | DONE | Inference script with template features + warm-start logic |
| `train_adv1.py` | DONE | Training script with self-template training + 50% masking |
| `setup_from_basic.bat` | DONE | Batch script to copy unchanged files from BASIC |
| `models/__init__.py` | DONE | Updated imports including TemplateEncoder, TemplateLoader |
| `models/template_encoder.py` | DONE | NEW: coords -> distances -> bins -> Linear(22,16) -> features |
| `models/template_loader.py` | DONE | NEW: reads Approach 1 outputs, serves templates per target |
| `checkpoints/` | DONE | Empty directory (fills during training) |

## WHAT STILL NEEDS TO HAPPEN

| File/Action | Status | How |
|-------------|--------|-----|
| `models/backbone.py` | NEEDS COPY | Run setup_from_basic.bat |
| `models/distance_head.py` | NEEDS COPY | Run setup_from_basic.bat |
| `models/reconstructor.py` | NEEDS COPY | Run setup_from_basic.bat |
| `data/` directory | NEEDS COPY | Run setup_from_basic.bat |
| `losses/` directory | NEEDS COPY | Run setup_from_basic.bat |
| `utils/` directory | NEEDS COPY | Run setup_from_basic.bat |
| Verify config paths | NEEDS CHECK | Step 2 below |
| Local training | NEEDS RUN | Step 4 below (2-4 hours on GPU) |
| Pre-Step 1 Kaggle test | NEEDS RUN | Step 3 below |
| Kaggle notebook | NEEDS WRITE | Step 6 below (Phase 4) |

---

## STEP 1: Copy BASIC Files (5 minutes)

Open a command prompt and run:

```cmd
cd C:\SATHYA\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH2-RIBBOZANET\HY-BAS-ADV1

setup_from_basic.bat
```

This copies from `..\BASIC\`:
- `models/backbone.py` (unchanged)
- `models/reconstructor.py` (unchanged)
- `data/dataset.py`, `data/collate.py`
- `losses/distance_loss.py`, `losses/constraint_loss.py`
- `utils/submission.py`

**NOTE:** `models/distance_head.py` is also copied from BASIC unchanged.
The pair_dim=80 is set via config_adv1.yaml, not in the code itself.
The DistanceMatrixHead class takes pair_dim as a constructor argument,
so the same code works for both BASIC (64) and ADV1 (80).

After running, verify:

```cmd
dir models\
dir data\
dir losses\
dir utils\
```

All should have files.

---

## STEP 2: Verify Config Paths (10 minutes)

Open `config_adv1.yaml` and verify each path exists on your machine:

```yaml
backbone:
  repo_path: "../../../../RibonanzaNet"
  #  = C:\SATHYA\...\Srna3D1\RibonanzaNet\
  #  CHECK: Does this directory exist? Does it have Network.py?

  weights_path: "../../../../ribonanza-weights/RibonanzaNet.pt"
  #  = C:\SATHYA\...\Srna3D1\ribonanza-weights\RibonanzaNet.pt
  #  CHECK: Does this file exist? Is it ~43MB?

data:
  train_pickle_path: "../../../../stanford3d-pickle/pdb_xyz_data.pkl"
  #  = C:\SATHYA\...\Srna3D1\stanford3d-pickle\pdb_xyz_data.pkl
  #  CHECK: Does this file exist? Is it ~52MB?

  test_csv_path: "../../APPROACH1-TEMPLATE/test_sequences (1).csv"
  #  = C:\SATHYA\...\TRY1\APPROACH1-TEMPLATE\test_sequences (1).csv
  #  CHECK: Does this file exist? Has 28 targets?

template:
  local_submission_csv: "../../APPROACH1-TEMPLATE/mine/kalai/run1/submission_kalai.csv"
  #  CHECK: Does this file exist? Verify exact filename.

  local_result_txt: "../../APPROACH1-TEMPLATE/mine/kalai/run1/Result_20260321_1100_UTC_kalai.txt.txt"
  #  CHECK: Does this file exist? Verify exact filename.

warm_start:
  basic_checkpoint: "../BASIC/checkpoints/best_model.pt"
  #  CHECK: Does this file exist? Is it ~312KB?
```

**If any path is wrong:** Edit config_adv1.yaml to fix it.
The most common issue: filenames with spaces or special characters.

---

## STEP 3: Pre-Step 1 — Verify BASIC Neural Network Runs on Kaggle (2-3 hours)

**WHY THIS MATTERS:** We have NEVER loaded RibonanzaNet or run PyTorch
inference on Kaggle. The 0.092 score was from a pre-made CSV, not from
the neural network. If this fails on Kaggle, ADV1 cannot work.

### Step 3.1: Prepare 3 Kaggle Datasets for Upload

**Dataset 1: "adv1-weights-test"**
Create a folder on your machine containing:
```
adv1-weights-test/
  RibonanzaNet.pt     (copy from Srna3D1/ribonanza-weights/)
  best_model.pt       (copy from BASIC/checkpoints/)
```

**Dataset 2: "ribonanzanet-repo"**
The entire cloned RibonanzaNet folder:
```
ribonanzanet-repo/
  RibonanzaNet/       (the full repo with Network.py, etc.)
```

**Dataset 3: "adv1-code-test"**
```
adv1-code-test/
  backbone.py         (from HY-BAS-ADV1/models/)
  distance_head.py    (from HY-BAS-ADV1/models/)
  reconstructor.py    (from HY-BAS-ADV1/models/)
  template_encoder.py (from HY-BAS-ADV1/models/)
  template_loader.py  (from HY-BAS-ADV1/models/)
  submission.py       (from HY-BAS-ADV1/utils/)
```

### Step 3.2: Upload to Kaggle

1. Go to https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Upload each folder as a separate dataset (private)
4. Note the exact dataset slugs (e.g., "tarunsathyab/adv1-weights-test")

### Step 3.3: Create Test Notebook

1. Go to https://www.kaggle.com/ -> "New Notebook"
2. Add all 3 datasets as inputs (+ Add Input)
3. Add the competition data as input
4. Paste this single cell:

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

# Try importing
from Network import RibonanzaNet
print("RibonanzaNet class imported OK")

# Load backbone weights
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f == "RibonanzaNet.pt":
            w_path = os.path.join(root, f)
            weights = torch.load(w_path, map_location="cpu")
            print("Backbone weights loaded from:", w_path)
            print("  Keys:", len(weights), "tensors")
            break

# Load distance head weights
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f == "best_model.pt":
            h_path = os.path.join(root, f)
            head = torch.load(h_path, map_location="cpu")
            print("Distance head loaded from:", h_path)
            print("  Type:", type(head))
            break

print("\nALL CHECKS PASSED - ready for ADV1")
```

5. Run with Shift+Enter
6. If ALL CHECKS PASSED -> proceed to Step 4
7. If errors -> debug (see DESIGN.md troubleshooting section)

---

## STEP 4: Train ADV1 Locally (2-4 hours)

**Prerequisites:** Step 1 and Step 2 complete. Local GPU available.

```cmd
cd C:\SATHYA\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH2-RIBBOZANET\HY-BAS-ADV1

python train_adv1.py --config config_adv1.yaml
```

**What happens:**
1. Loads frozen RibonanzaNet backbone
2. Creates ADV1 distance head (pair_dim=80)
3. Creates template encoder (22 bins -> 16 features)
4. Warm-starts distance head from BASIC's best_model.pt:
   - Copies columns 0-63 from BASIC
   - Random-inits columns 64-79 (template channels)
5. Trains for 30 epochs on 661 PDB structures
6. During training: 50% of the time, templates are masked (zeros)
   to teach the model to work both WITH and WITHOUT templates
7. Saves checkpoints/adv1_best_model.pt

**Expected output:**
```
Device: cuda
GPU: NVIDIA GeForce ...
Warm-starting from BASIC: ../BASIC/checkpoints/best_model.pt
  Expanding mlp.0.weight: torch.Size([128, 64]) -> torch.Size([128, 80])
Warm-start complete.
Distance head trainable params: ~25,000
Template encoder trainable params: 368
Total trainable params: ~25,368
Epoch 1: train_loss=X.XXXX, val_loss=X.XXXX
...
Epoch 30: train_loss=X.XXXX, val_loss=X.XXXX
Training complete. Best val_loss: X.XXXX
```

**If training fails:**
- "No module named models.backbone" -> Step 1 not done (run setup_from_basic.bat)
- "FileNotFoundError: RibonanzaNet" -> config paths wrong (Step 2)
- "CUDA out of memory" -> reduce batch_size in config_adv1.yaml to 2
- "No module named losses" -> setup_from_basic.bat didn't copy losses/

---

## STEP 5: Test Predictions Locally (15 minutes)

```cmd
python predict_adv1.py --config config_adv1.yaml --checkpoint checkpoints/adv1_best_model.pt
```

**This produces:** `submission_adv1.csv` in the current directory.

**Verify:**
```cmd
python -c "import csv; r=list(csv.reader(open('submission_adv1.csv'))); print('Rows:', len(r)-1); print('Cols:', len(r[0])); print('First:', r[1][:3])"
```

Should show 9762 rows, 18 columns.

**Compare with BASIC:**
```cmd
python -c "
import csv
basic = {r['ID']: float(r['x_1']) for r in csv.DictReader(open('../BASIC/submission.csv'))}
adv1 = {r['ID']: float(r['x_1']) for r in csv.DictReader(open('submission_adv1.csv'))}
same = sum(1 for k in basic if k in adv1 and abs(basic[k] - adv1[k]) < 0.01)
diff = sum(1 for k in basic if k in adv1 and abs(basic[k] - adv1[k]) >= 0.01)
print(f'Same coords: {same}, Different coords: {diff}')
print('Template targets should show DIFFERENT coordinates')
"
```

If all coordinates are the same -> template features aren't working.
If template targets (9G4J, 9LEC, etc.) differ -> ADV1 is learning from templates.

---

## STEP 6: Build Kaggle Notebook (3-4 hours)

### Step 6.1: Update Kaggle Datasets

Replace "adv1-weights-test" with final weights:
```
adv1-weights/
  adv1_best_model.pt    (from HY-BAS-ADV1/checkpoints/ — NEW trained weights)
  RibonanzaNet.pt       (same as before)
```

Update "adv1-code-test" with all final code:
```
adv1-code/
  backbone.py
  distance_head.py
  reconstructor.py
  template_encoder.py
  template_loader.py
  submission.py
  predict_adv1.py       (add this too — for reference)
```

### Step 6.2: Create Kaggle Notebook

Create a NEW Kaggle notebook: "HY-BAS-ADV1-Hybrid"

Add inputs:
- Stanford RNA 3D Folding Part 2 (competition)
- adv1-weights (dataset)
- adv1-code (dataset)
- ribonanzanet-repo (dataset)

Settings:
- GPU T4 x2
- Internet ON (for development)

### Step 6.3: Notebook Cells

The notebook needs these cells in order:

```
CELL 1: Symlink fix
CELL 2: pip install dependencies
CELL 3-8: MMseqs2 pipeline (copy from Fork 1 rhijudas notebook)
  - Install mmseqs2
  - Build database from PDB sequences
  - Search test sequences against database
  - Extract template coordinates from hits
CELL 9: Set up model paths and imports
CELL 10: Load ADV1 model + template encoder
CELL 11: Run inference on all test sequences with templates
CELL 12: Option B post-processing (read sample_submission.csv, map IDs)
CELL 13: Verification
```

NOTE: Cell 3-8 is the part that requires the most work.
It needs to be adapted from Fork 1's notebook.
This is the Kaggle integration step that will be written in Phase 4.

### Step 6.4: Test on Kaggle

1. Run All with Internet ON
2. Check submission.csv is generated
3. Turn Internet OFF -> Run All
4. If it works -> proceed to Step 7

---

## STEP 7: Submit (30 minutes)

1. Internet OFF
2. Save Version -> Save & Run All (Commit)
3. Wait for completion (~30-55 min)
4. Go to committed notebook -> Output -> Submit to Competition
5. Check submissions page for score

---

## SUMMARY: Total Estimated Timeline

| Step | What | Time | Depends On |
|------|------|------|-----------|
| Step 1 | Copy BASIC files | 5 min | Nothing |
| Step 2 | Verify config paths | 10 min | Step 1 |
| Step 3 | Pre-Step 1 Kaggle test | 2-3 hrs | Steps 1-2 + Kaggle uploads |
| Step 4 | Train ADV1 locally | 2-4 hrs | Steps 1-2 |
| Step 5 | Test predictions locally | 15 min | Step 4 |
| Step 6 | Build Kaggle notebook | 3-4 hrs | Steps 3-5 |
| Step 7 | Submit | 30 min | Step 6 |
| **TOTAL** | | **~8-12 hrs** | |

Steps 3 and 4 can run in PARALLEL (Kaggle test + local training).
This brings the wall-clock time down to ~6-8 hours.

---

## PARALLEL TRACK REMINDER

While HY-BAS-ADV1 is being built:
- Fork 2 + Option B should already be submitted (see FORK2_OPTB_KAGGLE_STEPS.md)
- Fork 1 status needs confirmation from Kalai
- These provide fallback scores regardless of HY-BAS-ADV1 outcome
