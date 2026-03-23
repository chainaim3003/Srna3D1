# Fork 1: rhijudas — Run Locally

## Source
- **URL:** https://www.kaggle.com/code/rhijudas/mmseqs2-3d-rna-template-identification-part-2
- **Author:** rhijudas (DasLab / Competition Organizer)

## IMPORTANT LIMITATIONS OF LOCAL RUNS
- Local runs do NOT count as competition submissions
- The competition is a Code Competition — final scoring requires a Kaggle notebook
- Local runs only produce predictions for the 28 public targets
- Hidden test targets are only available inside Kaggle's scoring system
- Use local runs for development/debugging only

---

## Prerequisites

### Software

| Software | How to Install |
|----------|---------------|
| Python 3.10+ | Already installed |
| MMseqs2 | See Step 1 below |
| BioPython | `pip install biopython` |
| pandas | `pip install pandas` |
| numpy | `pip install numpy` |
| Jupyter | `pip install jupyter` |

### Competition Data

Download from: https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/data

You need:
```
PDB_RNA/                  (directory with thousands of CIF files)
pdb_seqres_NA.fasta       (RNA sequences in PDB)
test_sequences.csv        (28 Part 2 test targets)
sample_submission.csv
train_sequences.csv
validation_sequences.csv
```

Save to a local directory, e.g.:
```
C:\sathya\CHAINAIM3003\mcp-servers\STANFORD-RNA\competition-data\
```

---

## Step-by-Step

### Step 1: Install MMseqs2

**Option A — Conda (recommended):**
```bash
conda install -c conda-forge -c bioconda mmseqs2
```

**Option B — Windows manual:**
1. Go to: https://github.com/soedinglab/MMseqs2/releases
2. Download the Windows release ZIP
3. Extract to `C:\tools\mmseqs\`
4. Add to PATH in PowerShell:
   ```powershell
   $env:PATH += ";C:\tools\mmseqs\bin"
   ```
5. Verify:
   ```bash
   mmseqs version
   ```

**Option C — WSL (Windows Subsystem for Linux):**
```bash
sudo apt-get update
sudo apt-get install mmseqs2
```

### Step 2: Install Python Dependencies

```bash
pip install biopython pandas numpy jupyter
```

### Step 3: Download the Notebook

1. Go to: https://www.kaggle.com/code/rhijudas/mmseqs2-3d-rna-template-identification-part-2
2. Click the download icon (top right)
3. Save the .ipynb to:
   ```
   C:\sathya\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH1-TEMPLATE\fork1-RJ\rhijudas_part2.ipynb
   ```

### Step 4: Fix Data Paths

The notebook uses Kaggle paths like `/kaggle/input/stanford-rna-3d-folding-2/`.
You must change these to your local paths.

**Option A — Find and replace in the .ipynb:**
Open the notebook in VS Code or a text editor. Replace all occurrences:
```
/kaggle/input/stanford-rna-3d-folding-2/
```
with:
```
C:/sathya/CHAINAIM3003/mcp-servers/STANFORD-RNA/competition-data/
```

Also replace output path:
```
/kaggle/working/
```
with:
```
C:/sathya/CHAINAIM3003/mcp-servers/STANFORD-RNA/Srna3D1/TRY1/APPROACH1-TEMPLATE/fork1-RJ/
```

**Option B — Create symlinks to mimic Kaggle paths:**
In PowerShell as Administrator:
```powershell
mkdir C:\kaggle\input -Force
New-Item -ItemType Junction -Path "C:\kaggle\input\stanford-rna-3d-folding-2" -Target "C:\sathya\CHAINAIM3003\mcp-servers\STANFORD-RNA\competition-data"
mkdir C:\kaggle\working -Force
```

### Step 5: Run the Notebook

```bash
cd C:\sathya\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH1-TEMPLATE\fork1-RJ
jupyter notebook rhijudas_part2.ipynb
```

Run cells one at a time. Fix any path errors as they appear.

### Step 6: Collect Output

After completion, copy results to:
```
fork1-RJ/
├── submission_rhijudas.csv
├── Result_rhijudas.txt          (if generated)
└── rhijudas_part2.ipynb
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `mmseqs: command not found` | MMseqs2 not in PATH. Reinstall or add to PATH. |
| `No module named 'Bio'` | `pip install biopython` |
| `FileNotFoundError` | Path mismatch — update Kaggle paths to local paths (Step 4) |
| MMseqs2 very slow | Normal on first run (builds database). 8+ GB RAM recommended. |
| Permission errors on Windows | Run terminal as Administrator, or use WSL instead |
| `!pip install` or `!apt-get` in cells | These are shell commands for Kaggle/Linux. Replace `!pip` with running `pip` in your terminal. Remove `!apt-get` lines entirely. |
