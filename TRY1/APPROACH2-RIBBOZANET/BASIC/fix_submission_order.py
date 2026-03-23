"""
Fix submission.csv row ordering to match sample_submission.csv.
Kaggle requires rows in the exact same order as the sample.
"""
import pandas as pd
import os

# Paths
sample_path = r"C:\sathya\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH1-TEMPLATE\sample_submission.csv"
basic_path = r"C:\sathya\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH2-RIBBOZANET\BASIC\submission.csv"
output_path = r"C:\sathya\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH2-RIBBOZANET\BASIC\submission_fixed.csv"

print("Reading sample submission...")
sample = pd.read_csv(sample_path)
print("  Sample rows: " + str(len(sample)))

print("Reading BASIC submission...")
basic = pd.read_csv(basic_path)
print("  BASIC rows: " + str(len(basic)))

# Compare
sample_ids = list(sample["ID"])
basic_ids = set(basic["ID"])
sample_id_set = set(sample_ids)

print("\n--- DIAGNOSIS ---")
print("Sample row count: " + str(len(sample)))
print("BASIC row count: " + str(len(basic)))

only_in_sample = sample_id_set - basic_ids
only_in_basic = basic_ids - sample_id_set

if only_in_sample:
    print("IDs in sample but NOT in BASIC: " + str(len(only_in_sample)))
if only_in_basic:
    print("IDs in BASIC but NOT in sample: " + str(len(only_in_basic)))
if not only_in_sample and not only_in_basic:
    print("All IDs match! Just need reordering.")

# Per-target comparison
sample_tc = sample["ID"].str.rsplit("_", n=1).str[0].value_counts().sort_index()
basic_tc = basic["ID"].str.rsplit("_", n=1).str[0].value_counts().sort_index()
all_targets = sorted(set(sample_tc.index) | set(basic_tc.index))
for t in all_targets:
    s = sample_tc.get(t, 0)
    b = basic_tc.get(t, 0)
    if s != b:
        print("  MISMATCH " + t + ": sample=" + str(s) + " basic=" + str(b))

# --- FIX ---
print("\n--- FIXING ---")

# Build lookup from BASIC
basic_lookup = {}
for _, row in basic.iterrows():
    basic_lookup[row["ID"]] = row

# Build fixed submission in sample order
fixed_rows = []
missing = 0
for sid in sample_ids:
    if sid in basic_lookup:
        fixed_rows.append(basic_lookup[sid])
    else:
        # Use sample row (all zeros) for missing IDs
        fixed_rows.append(sample[sample["ID"] == sid].iloc[0])
        missing += 1

fixed = pd.DataFrame(fixed_rows)

# Ensure column order matches sample exactly
fixed = fixed[sample.columns]

# Save
fixed.to_csv(output_path, index=False)

print("Saved: " + output_path)
print("Rows: " + str(len(fixed)))
print("Missing IDs filled with zeros: " + str(missing))
print("File size: " + str(os.path.getsize(output_path)) + " bytes")

# Verify
print("\n--- VERIFICATION ---")
print("First ID: " + str(fixed["ID"].iloc[0]))
print("Last ID: " + str(fixed["ID"].iloc[-1]))
print("Sample first: " + str(sample["ID"].iloc[0]))
print("Sample last: " + str(sample["ID"].iloc[-1]))
match = list(fixed["ID"]) == list(sample["ID"])
print("ID order matches sample: " + str(match))
print("\nDone! Upload submission_fixed.csv to Kaggle.")
