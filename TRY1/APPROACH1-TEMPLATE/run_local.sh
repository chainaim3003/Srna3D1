#!/bin/bash
# ============================================================
# Approach 1: Template-Based Modeling — Local Run Script
# ============================================================
# 
# PREREQUISITES:
#   1. MMseqs2 installed (apt-get install mmseqs2 OR brew install mmseqs2)
#   2. Python 3.8+ with biopython (pip install biopython)
#   3. Git (to clone DasLab repo)
#   4. The PDB RNA database:
#      - pdb_seqres_NA.fasta (RNA sequences from PDB)
#      - CIF files directory
#      - pdb_release_dates_NA.csv
#
# WHERE TO GET THE PDB RNA DATA:
#   Option A (recommended): Download competition data from Kaggle
#     https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/data
#     The PDB_RNA/ folder contains everything needed.
#
#   Option B: Download directly from RCSB PDB
#     Sequences: https://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz
#     CIF files: RCSB batch download (search for "polymer entity type: RNA")
#
# USAGE:
#   ./run_local.sh /path/to/PDB_RNA /path/to/test_sequences.csv
#
# Official sources:
#   DasLab create_templates: https://github.com/DasLab/create_templates
#   MMseqs2 wiki: https://github.com/soedinglab/MMseqs2/wiki
# ============================================================

set -e  # Exit on any error

# --- Arguments ---
PDB_RNA_DIR="${1:?Usage: $0 /path/to/PDB_RNA /path/to/test_sequences.csv}"
TEST_CSV="${2:?Usage: $0 /path/to/PDB_RNA /path/to/test_sequences.csv}"

FASTA_DB="${PDB_RNA_DIR}/pdb_seqres_NA.fasta"
WORK_DIR="./tbm_work"
RESULT_TXT="${WORK_DIR}/Result.txt"
OUTPUT_CSV="${WORK_DIR}/templates.csv"

# --- Validate inputs ---
echo "=== Approach 1: Template-Based Modeling ==="
echo ""

if [ ! -d "$PDB_RNA_DIR" ]; then
    echo "ERROR: PDB_RNA directory not found: $PDB_RNA_DIR"
    exit 1
fi

if [ ! -f "$TEST_CSV" ]; then
    echo "ERROR: test_sequences.csv not found: $TEST_CSV"
    exit 1
fi

if [ ! -f "$FASTA_DB" ]; then
    echo "ERROR: pdb_seqres_NA.fasta not found in $PDB_RNA_DIR"
    echo "This file should be inside the PDB_RNA/ directory."
    exit 1
fi

CIF_COUNT=$(find "$PDB_RNA_DIR" -name "*.cif" -o -name "*.cif.gz" | wc -l)
echo "PDB_RNA directory: $PDB_RNA_DIR"
echo "CIF files found:   $CIF_COUNT"
echo "FASTA database:    $FASTA_DB"
echo "Test sequences:    $TEST_CSV"
echo ""

if [ "$CIF_COUNT" -lt 100 ]; then
    echo "WARNING: Only $CIF_COUNT CIF files. For good results, you need"
    echo "the FULL PDB RNA database (~15,000+ files)."
    echo "Download from: https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/data"
    echo ""
fi

# --- Check dependencies ---
if ! command -v mmseqs &> /dev/null; then
    echo "ERROR: MMseqs2 not installed."
    echo "Install: apt-get install mmseqs2 (Ubuntu) or brew install mmseqs2 (Mac)"
    echo "Or download: https://github.com/soedinglab/MMseqs2/releases"
    exit 1
fi

if ! python3 -c "import Bio" 2>/dev/null; then
    echo "Installing BioPython..."
    pip install biopython --quiet
fi

echo "MMseqs2 version: $(mmseqs version)"
echo ""

# --- Clone DasLab create_templates if needed ---
if [ ! -d "create_templates" ]; then
    echo "Cloning DasLab/create_templates..."
    git clone https://github.com/DasLab/create_templates.git
fi

# --- Setup work directory ---
mkdir -p "$WORK_DIR"
mkdir -p "${WORK_DIR}/tmp"

# --- Step 1: Convert test sequences to FASTA ---
echo "Step 1/5: Converting test sequences to FASTA..."
QUERY_FASTA="${WORK_DIR}/query.fasta"
python3 -c "
import pandas as pd
df = pd.read_csv('$TEST_CSV')
with open('$QUERY_FASTA', 'w') as f:
    for _, row in df.iterrows():
        f.write(f'>{row[\"target_id\"]}\n{row[\"sequence\"]}\n')
print(f'  Written {len(df)} sequences to $QUERY_FASTA')
"

# --- Step 2: Build MMseqs2 databases ---
# Documentation: https://github.com/soedinglab/MMseqs2/wiki#database-creation
echo ""
echo "Step 2/5: Building MMseqs2 databases..."
echo "  Building query DB..."
mmseqs createdb "$QUERY_FASTA" "${WORK_DIR}/queryDB" --dbtype 2

echo "  Building target DB from PDB sequences..."
mmseqs createdb "$FASTA_DB" "${WORK_DIR}/targetDB" --dbtype 2

# --- Step 3: Run MMseqs2 search ---
# Documentation: https://github.com/soedinglab/MMseqs2/wiki#searching
# -s 7.5 = sensitivity (from DasLab documentation)
# --search-type 3 = nucleotide mode (from MMseqs2 docs)
echo ""
echo "Step 3/5: Running MMseqs2 search..."
mmseqs search \
    "${WORK_DIR}/queryDB" \
    "${WORK_DIR}/targetDB" \
    "${WORK_DIR}/resultDB" \
    "${WORK_DIR}/tmp" \
    -s 7.5 \
    --search-type 3 \
    -e 10

# --- Step 4: Convert to readable format ---
# Format string from DasLab create_templates README
echo ""
echo "Step 4/5: Converting results..."
mmseqs convertalis \
    "${WORK_DIR}/queryDB" \
    "${WORK_DIR}/targetDB" \
    "${WORK_DIR}/resultDB" \
    "$RESULT_TXT" \
    --format-output query,target,evalue,qstart,qend,tstart,tend,qaln,taln

HIT_COUNT=$(wc -l < "$RESULT_TXT" 2>/dev/null || echo "0")
echo "  Total hits: $HIT_COUNT"

if [ "$HIT_COUNT" -eq "0" ]; then
    echo ""
    echo "WARNING: 0 hits found. This likely means your PDB_RNA database is too small."
    echo "For the competition, you need the FULL database (~15,000+ CIF files)."
    echo "On Kaggle, this is provided at /kaggle/input/stanford-rna-3d-folding-2/PDB_RNA/"
fi

# --- Step 5: Generate template coordinates ---
# Using official DasLab create_templates_csv.py
# Arguments documented at: https://github.com/DasLab/create_templates
echo ""
echo "Step 5/5: Generating template coordinates..."
cd create_templates
python3 create_templates_csv.py \
    -s "$TEST_CSV" \
    --mmseqs_results_file "$RESULT_TXT" \
    --cif_dir "$PDB_RNA_DIR" \
    --outfile "$OUTPUT_CSV" \
    --max_templates 5 \
    --skip_temporal_cutoff
cd ..

# --- Done ---
echo ""
echo "=== COMPLETE ==="
if [ -f "$OUTPUT_CSV" ]; then
    ROWS=$(wc -l < "$OUTPUT_CSV")
    echo "Output: $OUTPUT_CSV ($ROWS rows)"
else
    echo "WARNING: Output file not generated. Check errors above."
fi
