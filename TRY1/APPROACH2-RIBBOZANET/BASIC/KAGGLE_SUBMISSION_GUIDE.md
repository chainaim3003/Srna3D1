# Submitting BASIC submission.csv to Kaggle

## Your File
```
C:\sathya\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH2-RIBBOZANET\BASIC\submission.csv
```

---

## Method 1: Direct CSV Upload (Quick Development Score)

This is the fastest way to get a TM-score. Takes ~2 minutes.

### Steps

1. Go to: `https://www.kaggle.com/competitions/stanford-rna-3d-folding-2`
2. Click the **"Submit Predictions"** tab (or **"Late Submission"** if past deadline)
3. Click **"Upload Submission File"**
4. Browse to and select:
   ```
   C:\sathya\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH2-RIBBOZANET\BASIC\submission.csv
   ```
5. Add a description (optional): `BASIC epoch50 frozen-RibonanzaNet distance-head`
6. Click **"Make Submission"**
7. Wait for scoring (~10 min to a few hours)
8. Your TM-score will appear on the Leaderboard tab

### Limitations
- This is a "development" submission — it uses the public leaderboard
- Some competitions limit the number of direct uploads per day (usually 5)
- If the competition requires notebook-only submissions for final scoring, this won't count for the private leaderboard

---

## Method 2: Kaggle Notebook Submission (Required for Final Scoring)

Many competitions require the submission.csv to be produced by a Kaggle notebook with **Internet OFF**. This method wraps your local file into a notebook.

### Step 1: Upload submission.csv as a Kaggle Dataset

You need to make your file accessible inside a Kaggle notebook.

1. Go to: `https://www.kaggle.com/datasets`
2. Click **"+ New Dataset"**
3. Enter a name: `basic-rna3d-submission` (or anything you like)
4. Click **"Select Files to Upload"** → browse to:
   ```
   C:\sathya\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH2-RIBBOZANET\BASIC\submission.csv
   ```
5. Set visibility: **Private**
6. Click **"Create"**
7. Wait for upload to complete

Your file is now at:
```
/kaggle/input/basic-rna3d-submission/submission.csv
```

### Step 2: Create a Kaggle Notebook

1. Go to: `https://www.kaggle.com/competitions/stanford-rna-3d-folding-2`
2. Click **"Code"** tab → **"New Notebook"**

### Step 3: Attach Your Dataset

In the right sidebar → **"Input"** section:

1. You should see `stanford-rna-3d-folding-2` (competition data) already attached
2. Click **"+ Add Input"** → **"Your Datasets"** tab → select `basic-rna3d-submission`

### Step 4: Write the Notebook Code

Delete any existing cells and create ONE cell with this code:

```python
import shutil
import os

# Path to your uploaded submission.csv
src = '/kaggle/input/basic-rna3d-submission/submission.csv'

# Kaggle looks for output in /kaggle/working/
dst = '/kaggle/working/submission.csv'

# Simply copy the file
shutil.copy(src, dst)

# Verify
print(f"File size: {os.path.getsize(dst):,} bytes")

# Quick sanity check
import pandas as pd
df = pd.read_csv(dst)
print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"Targets: {df.ID.str.rsplit('_', n=1).str[0].nunique()}")
print(f"\nFirst 3 rows:")
print(df.head(3))
print(f"\nLast 3 rows:")
print(df.tail(3))
print(f"\nDone — submission.csv ready for scoring")
```

### Step 5: Configure Settings

In the right sidebar → **"Settings"**:

| Setting | Value | Why |
|---------|-------|-----|
| **Accelerator** | None | No GPU needed, we're just copying a file |
| **Internet** | **OFF** | Required for final scored submissions |

### Step 6: Save & Run

1. Click **"Save Version"** (top right)
2. Select **"Save & Run All (Commit)"**
3. Wait for the notebook to run (~1 minute)

### Step 7: Submit

1. After the notebook finishes, go to its page
2. Click the **"Output"** tab
3. You should see `submission.csv`
4. Click **"Submit"** next to it

### Step 8: Check Score

1. Go to the competition page → **"Leaderboard"** tab
2. Find your submission
3. Note the TM-score

---

## Method 3: Kaggle Notebook That Runs Your Model (Most Robust)

If you want to run predict.py directly on Kaggle (so the notebook actually generates predictions from scratch rather than copying a pre-made file), you would need to:

1. Upload your entire BASIC codebase + checkpoints + RibonanzaNet weights as Kaggle datasets
2. Create a notebook that installs dependencies, runs predict.py, outputs submission.csv

This is more complex but proves the code is reproducible. Here's the outline:

### Datasets to Upload (as separate Kaggle Datasets)

| Dataset Name | What to Upload | Size |
|-------------|---------------|------|
| `ribonanzanet-weights` | `RibonanzaNet.pt` | 43 MB |
| `basic-checkpoint` | `checkpoints/best_model.pt` | 312 KB |
| `basic-code` | All `.py` files + `config.yaml` from BASIC/ | ~50 KB |
| `ribonanzanet-repo` | The entire `RibonanzaNet/` folder | ~10 MB |
| `ranger-optimizer` | The `Ranger-Deep-Learning-Optimizer/` folder | ~1 MB |

### Notebook Code (outline)

```python
import subprocess, sys, shutil, os

# Install packages
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'einops', 'pyyaml', 'scipy'])

# Set up paths
os.makedirs('/kaggle/working/models', exist_ok=True)
os.makedirs('/kaggle/working/data', exist_ok=True)
os.makedirs('/kaggle/working/utils', exist_ok=True)
os.makedirs('/kaggle/working/losses', exist_ok=True)

# Copy code from uploaded datasets
# ... copy .py files from /kaggle/input/basic-code/ to /kaggle/working/
# ... copy RibonanzaNet from /kaggle/input/ribonanzanet-repo/

# Run prediction
subprocess.check_call([
    sys.executable, 'predict.py',
    '--config', 'config.yaml',
    '--checkpoint', '/kaggle/input/basic-checkpoint/best_model.pt',
    '--test_csv', '/kaggle/input/stanford-rna-3d-folding-2/test_sequences.csv',
    '--output', '/kaggle/working/submission.csv'
])
```

This requires more setup but is the "proper" way for a final competition entry.

---

## Recommendation

| Goal | Use Method |
|------|-----------|
| Quick score to compare approaches | **Method 1** (direct upload) — 2 minutes |
| Official scored submission | **Method 2** (notebook with pre-made CSV) — 10 minutes |
| Prove reproducibility / final competition entry | **Method 3** (notebook runs model) — 1-2 hours setup |

**Start with Method 1** to get your baseline TM-score immediately. You can always do Method 2 or 3 later.
