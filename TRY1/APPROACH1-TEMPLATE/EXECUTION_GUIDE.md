# Approach 1: Template-Based Modeling (TBM) — EXECUTION GUIDE

---

## WHY YOU DON'T NEED TO WRITE THIS CODE FROM SCRATCH

The template-based approach already has TWO official, battle-tested Kaggle notebooks
that run on Kaggle with the FULL PDB RNA database (~15,000+ CIF files).
Writing your own version would be reinventing the wheel — and worse, because
these notebooks are written by the competition organizers and the 1st-place winner.

**Your TRY0 attempt failed because:** You ran on Google Colab with only 130 CIF files.
The MMseqs2 search found 0 hits because your template database was too small.
On Kaggle, the competition provides the ENTIRE PDB RNA database.

---

## THE TWO OFFICIAL NOTEBOOKS (pick one or both)

### Option A: MMseqs2 Baseline (by rhijudas / DasLab — the competition organizers)

| | |
|---|---|
| **URL (Part 1)** | https://www.kaggle.com/code/rhijudas/mmseqs2-3d-rna-template-identification |
| **URL (Part 2)** | https://www.kaggle.com/code/rhijudas/mmseqs2-3d-rna-template-identification-part-2 |
| **Who made it** | The competition organizers (DasLab at Stanford) |
| **What it does** | MMseqs2 sequence search against PDB_RNA → coordinate transfer |
| **Difficulty** | Simpler baseline |
| **GitHub source** | https://github.com/DasLab/create_templates |

### Option B: 1st Place TBM-Only (by jaejohn — the Part 1 WINNER)

| | |
|---|---|
| **URL** | https://www.kaggle.com/code/jaejohn/rna-3d-folds-tbm-only-approach |
| **Who made it** | "john" — 1st place winner of Part 1 (beat ALL deep learning approaches!) |
| **What it does** | Advanced template search + coordinate transfer + diversity generation |
| **Difficulty** | More sophisticated, stronger results |
| **Score (Part 1)** | 0.593 TM-score (beat deep learning entries!) |

### Pre-computed template output (if you don't want to run the notebooks yourself)

| | |
|---|---|
| **URL** | https://www.kaggle.com/datasets/rhijudas/rna-3d-folding-templates |
| **What it is** | The OUTPUT of the MMseqs2 notebook — already computed template submission.csv |
| **When to use** | If you just want the template results without running the notebook |

---

## WHY THE FULL PDB DATABASE MATTERS

| | Your TRY0 (Colab) | On Kaggle |
|---|---|---|
| **CIF files** | 130 (manually downloaded) | ~15,319 (competition-provided) |
| **Sequence database** | 140 RNA chains | Entire PDB RNA (pdb_seqres_NA.fasta) |
| **MMseqs2 hits** | 0 (too few templates) | Many (nearly all test targets get hits) |
| **Result** | Dummy straight-line coords | Real 3D structures from templates |

The competition provides the ENTIRE PDB RNA database at:
```
/kaggle/input/stanford-rna-3d-folding-2/PDB_RNA/
```
This includes:
- {PDB_id}.cif files for EVERY RNA-containing PDB entry
- pdb_seqres_NA.fasta — sequences of ALL nucleic acid chains in PDB
- pdb_release_dates_NA.csv — release dates (for temporal cutoff filtering)

---

## STEP-BY-STEP EXECUTION (Do This Now)

---

### Step 1: Go to the Kaggle notebook

Open ONE of these URLs in your browser:

**Recommended (stronger):**
```
https://www.kaggle.com/code/jaejohn/rna-3d-folds-tbm-only-approach
```

**Simpler baseline:**
```
https://www.kaggle.com/code/rhijudas/mmseqs2-3d-rna-template-identification-part-2
```

### Step 2: Fork (Copy) the notebook

1. Click the **"Copy & Edit"** button (top right of the notebook page)
2. This creates YOUR OWN copy of the notebook under your Kaggle account
3. You can now edit and run it freely

### Step 3: Verify the data source is attached

In the right sidebar of your forked notebook, check **"Input"** section:
- It should show `stanford-rna-3d-folding-2` as an attached dataset
- This gives you access to `/kaggle/input/stanford-rna-3d-folding-2/`
- Inside that: `PDB_RNA/`, `test_sequences.csv`, `MSA/`, etc.

If it's not attached:
1. Click **"+ Add Input"** in the sidebar
2. Search for "stanford-rna-3d-folding-2"
3. Select the competition data

### Step 4: Configure the notebook

Make sure:
- **Accelerator**: GPU T4 x2 (or GPU P100 — check what's available)
- **Internet**: ON (for installing MMseqs2 during the notebook run)
  
  IMPORTANT: For final SUBMISSION, internet must be OFF. But for 
  DEVELOPMENT runs, keep it ON so MMseqs2 can be installed. The 
  competition notebooks handle this by including MMseqs2 as a Kaggle 
  Dataset for offline use.

### Step 5: Run All Cells

Click **"Run All"** or run cells one by one.

The notebook will:
1. Install MMseqs2 (sequence search tool)
2. Build a sequence database from PDB_RNA/pdb_seqres_NA.fasta
3. Convert test sequences to FASTA format
4. Run MMseqs2 search: test sequences vs. the entire PDB sequence DB
5. For each hit: parse the matching CIF file, align sequences, copy C1' coordinates
6. Generate 5 diverse predictions per target (using top-5 template hits)
7. Write submission.csv

### Step 6: Download the output

After the notebook finishes:
1. Go to the **"Output"** tab of the notebook
2. Download `submission.csv`
3. This is your Approach 1 submission!

### Step 7: Submit

1. Go to https://www.kaggle.com/competitions/stanford-rna-3d-folding-2
2. Click **"Submit Predictions"**
3. Select your notebook (if code competition) or upload submission.csv

---

## WHAT THE PIPELINE DOES (Conceptual Overview)

```
For each test RNA sequence:
    1. SEARCH: MMseqs2 finds similar sequences in the PDB database
       "Is there a known RNA structure with a similar sequence?"
       
    2. ALIGN: Align the test sequence to the template sequence
       "Which nucleotides in the test match which in the template?"
       
    3. COPY: For matched positions, copy C1' coordinates from the template
       "Nucleotide #5 in test matches nucleotide #12 in template → 
        copy template #12's (x,y,z) to test #5"
       
    4. GAP-FILL: For unmatched positions (insertions/deletions),
       interpolate or extrapolate coordinates
       
    5. DIVERSIFY: Use top-5 different template hits for the 5 predictions
       "Template hit #1 → prediction 1, hit #2 → prediction 2, etc."
```

This is essentially: "Find a known RNA structure that looks like the target,
and copy its shape." It's the same principle as homology modeling in proteins.

---

## FOR THE ADVANCED USER: Running Locally (Not Recommended for Submission)

If you want to understand the code locally (NOT for actual submission):

1. Clone the DasLab create_templates repo:
   ```bash
   git clone https://github.com/DasLab/create_templates.git
   ```

2. Install MMseqs2 locally:
   ```bash
   # On Ubuntu/WSL:
   apt-get install mmseqs2
   # On Mac:
   brew install mmseqs2
   # On Windows: download from https://github.com/soedinglab/MMseqs2/releases
   ```

3. You need the FULL PDB_RNA database (~310 GB) or a subset.
   The competition provides this on Kaggle at /kaggle/input/.

4. Run the create_templates_csv.py script:
   ```bash
   cd create_templates/example/
   
   # Build MMseqs2 database from PDB sequences
   mmseqs createdb pdb_seqres_NA.fasta targetDB
   
   # Search test sequences against PDB
   mmseqs createdb test_query.fasta queryDB
   mmseqs search queryDB targetDB resultDB tmp -s 7.5
   mmseqs convertalis queryDB targetDB resultDB Result.txt \
     --format-output query,target,evalue,qstart,qend,tstart,tend,qaln,taln
   
   # Generate template coordinates
   python3 ../create_templates_csv.py \
     -s test_sequences.csv \
     --mmseqs_results_file Result.txt \
     --outfile templates.csv
   ```

   But again — this requires the full PDB. It's much easier to just fork 
   the Kaggle notebook.

---

## RELATIONSHIP TO APPROACH 2 (RibonanzaNet)

| | Approach 1 (TBM) | Approach 2 (RibonanzaNet + Distance) |
|---|---|---|
| **Works when** | Similar structure exists in PDB | Always (learned prediction) |
| **Fails when** | No template available (novel RNA) | Always produces something |
| **Accuracy** | Very high if template exists | Lower but universal |
| **Complexity** | Just search + copy | Neural network training |

**Best strategy = HYBRID:**
- Use Approach 1 (TBM) for targets where templates exist
- Use Approach 2 (RibonanzaNet) for targets with no templates
- Combine both into one submission.csv

This is exactly what RNAPro (NVIDIA's model) does — and it beat all 
individual approaches.

---

## QUICK REFERENCE: Key URLs

| Resource | URL |
|----------|-----|
| Competition | https://www.kaggle.com/competitions/stanford-rna-3d-folding-2 |
| MMseqs2 baseline notebook | https://www.kaggle.com/code/rhijudas/mmseqs2-3d-rna-template-identification |
| MMseqs2 baseline (Part 2) | https://www.kaggle.com/code/rhijudas/mmseqs2-3d-rna-template-identification-part-2 |
| 1st place TBM notebook | https://www.kaggle.com/code/jaejohn/rna-3d-folds-tbm-only-approach |
| Pre-computed templates | https://www.kaggle.com/datasets/rhijudas/rna-3d-folding-templates |
| DasLab create_templates | https://github.com/DasLab/create_templates |
| RNAPro (uses templates) | https://github.com/NVIDIA-Digital-Bio/RNAPro |
| Competition paper | https://pmc.ncbi.nlm.nih.gov/articles/PMC12776560/ |
