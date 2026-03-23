import pandas as pd

sample_path = r"C:\sathya\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH1-TEMPLATE\sample_submission.csv"
fixed_path = r"C:\sathya\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH2-RIBBOZANET\BASIC\submission_fixed.csv"

sample = pd.read_csv(sample_path)
fixed = pd.read_csv(fixed_path)

print("Row count - sample: " + str(len(sample)) + ", fixed: " + str(len(fixed)))
print("Match: " + str(len(sample) == len(fixed)))
print("Column check: " + str(list(sample.columns) == list(fixed.columns)))

if len(sample) == len(fixed):
    mismatches = 0
    for i in range(len(sample)):
        if sample["ID"].iloc[i] != fixed["ID"].iloc[i]:
            mismatches += 1
            if mismatches <= 5:
                print("  Row " + str(i) + ": sample=" + str(sample["ID"].iloc[i]) + " fixed=" + str(fixed["ID"].iloc[i]))
    print("ID mismatches: " + str(mismatches))
    if mismatches == 0:
        print("ALL IDs MATCH - submission_fixed.csv is READY")
else:
    print("*** ROW COUNT MISMATCH ***")
    sample_ids = set(sample["ID"])
    fixed_ids = set(fixed["ID"])
    only_sample = sample_ids - fixed_ids
    only_fixed = fixed_ids - sample_ids
    print("IDs only in sample: " + str(len(only_sample)))
    print("IDs only in fixed: " + str(len(only_fixed)))
