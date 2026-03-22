import os, csv

# ============================================================
# OPTION B POST-PROCESSING
# This cell runs AFTER all of jaejohn's code has finished.
# It fixes the submission.csv to match the exact IDs and order
# that Kaggle expects (from sample_submission.csv).
#
# Why this is needed:
# - jaejohn's code produces different row counts for multi-chain
#   targets (e.g., 4640 rows for 9MME instead of 4184)
# - Kaggle requires EXACT ID match with sample_submission.csv
# - This cell reads sample_submission.csv (which Kaggle swaps
#   with hidden data during scoring) so it works for hidden
#   targets too
# ============================================================

# Step 1: Find sample_submission.csv from competition data
sample_csv = None
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f == "sample_submission.csv":
            sample_csv = os.path.join(root, f)
            break
    if sample_csv:
        break

print("Sample submission: " + str(sample_csv))

# Step 2: Find jaejohn's raw submission.csv
raw_csv = "/kaggle/working/submission.csv"
if not os.path.exists(raw_csv):
    for f in os.listdir("/kaggle/working"):
        if f.endswith(".csv") and "submission" in f.lower():
            raw_csv = "/kaggle/working/" + f
            break

print("Raw submission: " + str(raw_csv))

# Step 3: Read sample to get ALL expected IDs and order
sample_rows = {}
sample_order = []
cols = None
with open(sample_csv, "r") as f:
    reader = csv.DictReader(f)
    cols = reader.fieldnames
    for row in reader:
        sample_rows[row["ID"]] = row
        sample_order.append(row["ID"])

print("Sample expects " + str(len(sample_order)) + " rows")

# Step 4: Read jaejohn's raw predictions
raw_rows = {}
with open(raw_csv, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        raw_rows[row["ID"]] = row

print("Raw predictions: " + str(len(raw_rows)) + " rows")

# Step 5: Build corrected submission
# - Use jaejohn's predictions where IDs match
# - Use sample zeros for missing IDs (hidden targets or mismatched rows)
dst = "/kaggle/working/submission.csv"
matched = 0
filled = 0

with open(dst, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=cols)
    writer.writeheader()
    for sid in sample_order:
        if sid in raw_rows:
            writer.writerow(raw_rows[sid])
            matched += 1
        else:
            writer.writerow(sample_rows[sid])
            filled += 1

print("\nCorrected submission written to: " + dst)
print("Total rows: " + str(matched + filled))
print("Matched from jaejohn: " + str(matched))
print("Filled with zeros: " + str(filled))
print("File size: " + str(os.path.getsize(dst)) + " bytes")
print("\nDone - submission.csv is ready for scoring")
