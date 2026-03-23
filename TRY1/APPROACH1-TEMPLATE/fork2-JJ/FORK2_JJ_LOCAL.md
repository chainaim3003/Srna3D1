# Fork 2: jaejohn 1st Place TBM — Run Locally

## Source
- **URL:** https://www.kaggle.com/code/jaejohn/rna-3d-folds-tbm-only-approach
- **Author:** jaejohn ("john") — 1st place Part 1 (0.593 TM-score)

## IMPORTANT LIMITATIONS OF LOCAL RUNS
- Local runs do NOT count as competition submissions
- The competition is a Code Competition — final scoring requires a Kaggle notebook
- Local runs only produce predictions for the 28 public targets
- Hidden test targets are only available inside Kaggle's scoring system
- Use local runs for development/debugging only
- This notebook is HARDER to run locally than fork1-RJ because it has 3 extra data dependencies

---

## Prerequisites

### Software

| Software | How to Install |
|----------|---------------|
| Python 3.10+ | Already installed |
| MMseqs2 | See below |
| BioPython | `pip install biopython` |
| pandas | `pip install pandas` |
| numpy | `pip install numpy` |
| Jupyter | `pip install jupyter` |

**Install MMseqs2:**

Windows (conda):
```bash
conda install -c conda-forge -c bioconda mmseqs2
```

Windows (manual): Download from https://github.com/soedinglab/MMseqs2/releases

WSL/Linux:
```bash
sudo apt-get install mmseqs2
```

### Data — 4 Sources Needed

This notebook requires data from 4 separate sources:

#### Source 1: Competition Data
Download from: https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/data
```
Save to: C:\sathya\data\stanford-rna-3d-folding-2\
Contains: PDB_RNA/, test_sequences.csv, train_sequences.csv, sample_submission.csv, etc.
```

#### Source 2: extended_rna Dataset
Download from: https://www.kaggle.com/datasets/jaejohn/extended-rna
(Search Kaggle datasets for "extended_rna" by jaejohn)
```
Save to: C:\sathya\data\extended-rna\
Contains: train_labels_v2.csv, train_sequences_v2.csv
```

#### Source 3: rna_cif_to_csv Dataset
Download from: https://www.kaggle.com/datasets/jaejohn/rna-cif-to-csv
(Search Kaggle datasets for "rna_cif_to_csv" by jaejohn)
```
Save to: C:\sathya\data\rna-cif-to-csv\
Contains: rna_coordinates.csv, rna_sequences.csv
```

#### Source 4: Dependency Installation Script Output
This is a Kaggle notebook that pre-installs packages. For local use,
you install the packages yourself (Step 2 below), so you can skip
downloading this. But if the notebook imports from its output path,
you'll need to check what files it provides and replicate them locally.

---

## Step-by-Step

### Step 1: Install Python Dependencies

```bash
pip install biopython pandas numpy jupyter scipy scikit-learn
```

If the notebook uses additional packages, install them as you encounter
ModuleNotFoundError during Step 5.

### Step 2: Download the Notebook

1. Go to: https://www.kaggle.com/code/jaejohn/rna-3d-folds-tbm-only-approach
2. Click the download icon (top right)
3. Save to:
   ```
   C:\sathya\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH1-TEMPLATE\fork2-JJ\jaejohn_tbm.ipynb
   ```

### Step 3: Download All Required Datasets

Use the Kaggle CLI (faster) or download manually from the browser.

**Install Kaggle CLI:**
```bash
pip install kaggle
```

**Configure Kaggle API key:**
1. Go to https://www.kaggle.com/settings → API → "Create New Token"
2. Save `kaggle.json` to `C:\Users\<your_user>\.kaggle\kaggle.json`

**Download datasets:**
```bash
# Competition data
kaggle competitions download -c stanford-rna-3d-folding-2 -p C:\sathya\data\stanford-rna-3d-folding-2
cd C:\sathya\data\stanford-rna-3d-folding-2
# Unzip if needed

# extended_rna (find exact slug on Kaggle)
kaggle datasets download jaejohn/extended-rna -p C:\sathya\data\extended-rna --unzip

# rna_cif_to_csv (find exact slug on Kaggle)
kaggle datasets download jaejohn/rna-cif-to-csv -p C:\sathya\data\rna-cif-to-csv --unzip
```

**Or download manually** from the Kaggle website for each dataset.

### Step 4: Fix Data Paths in the Notebook

The notebook uses Kaggle paths. You must replace them with local paths.

Open `jaejohn_tbm.ipynb` in a text editor. Find and replace ALL of these:

| Kaggle Path | Local Path |
|-------------|-----------|
| `/kaggle/input/stanford-rna-3d-folding-2/` | `C:/sathya/data/stanford-rna-3d-folding-2/` |
| `/kaggle/input/stanford-rna-3d-folding/` | `C:/sathya/data/stanford-rna-3d-folding-2/` |
| `/kaggle/input/extended-rna/` | `C:/sathya/data/extended-rna/` |
| `/kaggle/input/extended_rna/` | `C:/sathya/data/extended-rna/` |
| `/kaggle/input/rna-cif-to-csv/` | `C:/sathya/data/rna-cif-to-csv/` |
| `/kaggle/input/rna_cif_to_csv/` | `C:/sathya/data/rna-cif-to-csv/` |
| `/kaggle/working/` | `C:/sathya/CHAINAIM3003/mcp-servers/STANFORD-RNA/Srna3D1/TRY1/APPROACH1-TEMPLATE/fork2-JJ/` |

NOTE: The Kaggle dataset slug uses hyphens (extended-rna) but the notebook
code may reference with underscores (extended_rna). Check both variations.

**Alternative — Create symlinks (PowerShell as Administrator):**
```powershell
mkdir C:\kaggle\input -Force
New-Item -ItemType Junction -Path "C:\kaggle\input\stanford-rna-3d-folding-2" -Target "C:\sathya\data\stanford-rna-3d-folding-2"
New-Item -ItemType Junction -Path "C:\kaggle\input\extended-rna" -Target "C:\sathya\data\extended-rna"
New-Item -ItemType Junction -Path "C:\kaggle\input\rna-cif-to-csv" -Target "C:\sathya\data\rna-cif-to-csv"
mkdir C:\kaggle\working -Force
```

### Step 5: Handle the Dependency Installation Script

The notebook may import from the Dependency Installation Script output path:
```python
sys.path.append('/kaggle/input/dependency-installation-script/')
```

For local use, comment out these lines. You've already installed the
packages in Step 1. If the script provides custom .py files (not just
pip packages), you'll need to download the notebook output and place
the files at the expected path.

### Step 6: Remove Shell Commands That Won't Work on Windows

The notebook may contain Linux-specific commands:
```python
!apt-get install mmseqs2       # Remove — you installed mmseqs2 in Prerequisites
!pip install biopython         # Keep — or run in your terminal instead
```

For any `!command` lines:
- `!pip install X` → Keep or run manually
- `!apt-get install X` → Remove (install the tool via conda/manual)
- `!chmod +x script.sh` → Remove (Windows doesn't need this)
- `!./script.sh` → Convert to Windows equivalent or run in WSL

### Step 7: Run the Notebook

```bash
cd C:\sathya\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH1-TEMPLATE\fork2-JJ
jupyter notebook jaejohn_tbm.ipynb
```

Run cells one at a time with Shift+Enter. Fix errors as they appear.

### Step 8: Collect Output

After completion:
```
fork2-JJ/
├── submission_jaejohn.csv
├── jaejohn_tbm.ipynb
└── [any intermediate files]
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No module named 'Bio'` | `pip install biopython` |
| `No module named 'XXX'` | `pip install XXX` |
| `FileNotFoundError` | Path mismatch — check Step 4 replacements |
| `mmseqs: command not found` | MMseqs2 not installed or not in PATH |
| `!apt-get` fails on Windows | Remove the line — install the tool via conda or manual download |
| Paths with forward slashes fail | Use `C:/path/to/file` (forward slashes work in Python on Windows) |
| Notebook imports from dependency script | Comment out the sys.path.append line, install packages directly |
| Permission errors | Run terminal as Administrator or use WSL |

---

## Recommendation

**Running this notebook locally is significantly harder than on Kaggle** because of the 3 extra data dependencies and the Dependency Installation Script. If you encounter too many issues locally:

1. Run it on Kaggle instead (see FORK2_JJ_KAGGLE.md)
2. Or fall back to fork1-RJ (rhijudas) which is simpler and self-contained
