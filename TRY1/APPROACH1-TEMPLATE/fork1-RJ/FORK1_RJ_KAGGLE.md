# Fork 1: rhijudas — Run on Kaggle

## Source
- **URL:** https://www.kaggle.com/code/rhijudas/mmseqs2-3d-rna-template-identification-part-2
- **Author:** rhijudas (DasLab / Competition Organizer)
- **Built for:** Part 2 (correct competition)
- **Runtime:** ~15-30 minutes

## CRITICAL CONTEXT
- This is a **Code Competition** — Kaggle re-runs your notebook on HIDDEN test sequences
- The notebook must read test_sequences.csv dynamically and produce predictions for ALL sequences
- A pre-made submission.csv will NOT work — we learned this the hard way (scored 0.092)
- The notebook must run end-to-end with Internet OFF for final submission

---

## Step-by-Step

### Step 1: Open and Fork

1. Go to: https://www.kaggle.com/code/rhijudas/mmseqs2-3d-rna-template-identification-part-2
2. Log in to Kaggle
3. Click **"Copy & Edit"** (top right)

### Step 2: Verify Inputs

In the right sidebar under **"Input"**, you must see:

```
COMPETITIONS
  Stanford RNA 3D Folding Part 2
```

If missing: Click **"+ Add Input"** → search `stanford-rna-3d-folding-2` → add it.

### Step 3: Fix the Competition Data Path

Kaggle mounts competition data at an unexpected path. Add a **NEW CELL at the very top** of the notebook with this code:

```python
import os
src = "/kaggle/input/competitions/stanford-rna-3d-folding-2"
dst = "/kaggle/input/stanford-rna-3d-folding-2"
if os.path.exists(src) and not os.path.exists(dst):
    os.symlink(src, dst)
    print("Symlink created: " + dst)
else:
    print("Path already correct or symlink exists")
```

Run this cell FIRST with Shift+Enter.

### Step 4: Install Missing Dependencies

Add another cell right after the symlink cell:

```python
!pip install biopython -q
```

Run with Shift+Enter. This prevents `ModuleNotFoundError: No module named 'Bio'`.

### Step 5: Configure Settings

In the right sidebar → **"Settings"**:

| Setting | Value | Why |
|---------|-------|-----|
| **Accelerator** | GPU T4 x2 | MMseqs2 uses CPU but GPU doesn't hurt |
| **Internet** | **ON** | Needed for pip install during development |

### Step 6: Run Cell by Cell (First Time)

Do NOT click "Run All" the first time. Run cells one at a time with **Shift+Enter**.
Stop at any error and fix before continuing.

| Error | Fix |
|-------|-----|
| `No module named 'Bio'` | Step 4 above |
| `FileNotFoundError: /kaggle/input/stanford-rna-3d-folding-2/...` | Step 3 above |
| `apt-get install mmseqs2 fails` | Internet must be ON |
| `No module named 'XXX'` | Add `!pip install XXX -q` cell |

### Step 7: Verify Output

After all cells complete:
- Check that `submission.csv` exists in `/kaggle/working/`
- Verify it contains the 28 Part 2 target IDs

### Step 8: Submit to Competition

1. Turn **Internet → OFF** in Settings
2. Click **"Save Version"** → **"Save & Run All (Commit)"**
3. Wait for completion (~15-30 min)
4. Go to committed notebook → **Output** tab → **"Submit to Competition"**

### Step 9: Save Results Locally

Download from the Output tab:
- `submission.csv`
- Any `Result.txt` or intermediate files

Save to:
```
C:\sathya\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH1-TEMPLATE\fork1-RJ\
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Submission Scoring Error" after submit | The notebook must produce valid output when re-run on hidden data. Check that it reads test_sequences.csv dynamically, not hardcoded targets. |
| Notebook times out (>8 hrs) | Should finish in 15-30 min. Check for infinite loops or stuck cells. |
| "Submit" button not visible | You must commit first (Save Version → Save & Run All). Submit only appears on committed versions, not draft sessions. |
