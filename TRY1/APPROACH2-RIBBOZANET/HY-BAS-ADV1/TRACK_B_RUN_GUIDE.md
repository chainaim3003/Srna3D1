# Track B: Running HY-BAS-ADV1

## Directory
```
C:\sathya\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH2-RIBBOZANET\HY-BAS-ADV1\
```

---

## Template Source: Fork 2 (jaejohn, 0.287 TM-score)

We use Fork 2's outputs because:
- Scored 0.287 on Kaggle (confirmed) vs Kalai's estimated ~0.15-0.20
- jaejohn's superior template ranking and gap-filling produces better coordinates
- The Kaggle hybrid notebook will also use Fork 2's pipeline

During TRAINING: templates come from self (true training coordinates, not CSVs).
During LOCAL INFERENCE: templates come from Fork 2's submission-fork2.csv.
During KAGGLE INFERENCE: templates come from Fork 2's live MMseqs2 pipeline.

---

## All Dependencies Verified

| Dependency | Path (from HY-BAS-ADV1/) | Size | Status |
|-----------|--------------------------|------|--------|
| RibonanzaNet repo | `../../../RibonanzaNet/Network.py` | ~10 MB | ✅ |
| Backbone weights | `../../../ribonanza-weights/RibonanzaNet.pt` | 43.3 MB | ✅ |
| Training data | `../../../stanford3d-pickle/pdb_xyz_data.pkl` | 52.3 MB | ✅ |
| BASIC trained head | `../BASIC/checkpoints/best_model.pt` | 312 KB | ✅ |
| Fork 2 submission | `../../APPROACH1-TEMPLATE/fork2-JJ/REMOTE/run1/submission-fork2.csv` | | ✅ |
| E-values (Kalai) | `../../APPROACH1-TEMPLATE/mine/kalai/run1/Result_20260321_1100_UTC_kalai.txt.txt` | | ✅ |
| Ranger optimizer | `../../../Ranger-Deep-Learning-Optimizer/` | | ✅ |
| Test sequences | `../../APPROACH1-TEMPLATE/test_sequences (1).csv` | | ✅ |

---

## STEP 1: Open a Terminal

```cmd
cd C:\sathya\CHAINAIM3003\mcp-servers\STANFORD-RNA\Srna3D1\TRY1\APPROACH2-RIBBOZANET\HY-BAS-ADV1
```

---

## STEP 2: Train ADV1

```cmd
python train_adv1.py --config config_adv1.yaml
```

### What This Does
1. Loads frozen RibonanzaNet backbone (43 MB)
2. Creates ADV1 distance head with pair_dim=80 (64 pairwise + 16 template)
3. Creates template encoder (22 distance bins -> 16 learned features)
4. Warm-starts distance head from BASIC's `best_model.pt`:
   - First layer expands from (128,64) to (128,80)
   - Columns 0-63: copied from BASIC
   - Columns 64-79: random initialization (new template channels)
   - All other layers: copied directly
5. Loads 661 PDB training structures from `pdb_xyz_data.pkl`
6. Trains 30 epochs with:
   - Self-template training: uses each structure's own TRUE coordinates
     as template input (NOT from any submission CSV)
   - 50% template masking: half the time, templates are zeroed out
     (teaches model to work both WITH and WITHOUT templates)
   - Learning rate: 5e-5 (lower than BASIC since warm-started)
7. Saves `checkpoints/adv1_best_model.pt` (best validation loss)

### Expected Console Output
```
Device: cuda
GPU: NVIDIA GeForce ...
Loading backbone...
Loaded official config from .../configs/pairwise.yaml
Loaded backbone weights from .../RibonanzaNet.pt
Backbone frozen
Warm-starting from BASIC: ../BASIC/checkpoints/best_model.pt
  Expanding mlp.0.weight: torch.Size([128, 64]) -> torch.Size([128, 80])
Warm-start complete.
Distance head trainable params: ~25,000
Template encoder trainable params: 368
Total trainable params: ~25,368
Epoch 1: train_loss=X.XXXX, val_loss=X.XXXX, lr=0.000050
...
Epoch 30: train_loss=X.XXXX, val_loss=X.XXXX, lr=0.000002
  Saved BEST: ./checkpoints/adv1_best_model.pt
Training complete. Best val_loss: X.XXXX
```

### Expected Duration
- With GPU: 1-3 hours
- Without GPU (CPU only): 6-12 hours (not recommended)

### If It Fails

| Error | Cause | Fix |
|-------|-------|-----|
| `No module named 'Network'` | RibonanzaNet repo path wrong | Verify `Srna3D1/RibonanzaNet/Network.py` exists |
| `No module named 'ranger'` | Ranger optimizer not found | `pip install ranger-fm` or verify `Srna3D1/Ranger-Deep-Learning-Optimizer/` |
| `FileNotFoundError: RibonanzaNet.pt` | Backbone weights missing | Verify `Srna3D1/ribonanza-weights/RibonanzaNet.pt` |
| `FileNotFoundError: pdb_xyz_data.pkl` | Training data missing | Verify `Srna3D1/stanford3d-pickle/pdb_xyz_data.pkl` |
| `CUDA out of memory` | GPU VRAM insufficient | Edit `config_adv1.yaml`: change `batch_size: 4` to `batch_size: 2` |
| `No module named 'models.backbone'` | Wrong working directory | Make sure you `cd` to HY-BAS-ADV1 before running |
| `ModuleNotFoundError: yaml` | Missing package | `pip install pyyaml` |
| `ModuleNotFoundError: scipy` | Missing package | `pip install scipy` |
| `ModuleNotFoundError: tqdm` | Missing package | `pip install tqdm` |
| `ModuleNotFoundError: einops` | Missing package | `pip install einops` |

---

## STEP 3: Verify Training Output

After training completes:

```cmd
dir checkpoints\
```

Expected:
```
adv1_best_model.pt        (~300-400 KB)
adv1_epoch5.pt
adv1_epoch10.pt
...
adv1_epoch30.pt
```

The key file is `adv1_best_model.pt`.

---

## STEP 4: Run Local Inference (Uses Fork 2 Templates)

```cmd
python predict_adv1.py --config config_adv1.yaml --checkpoint checkpoints/adv1_best_model.pt
```

### What This Does
1. Loads frozen RibonanzaNet backbone
2. Loads ADV1 distance head from `adv1_best_model.pt`
3. Loads template encoder
4. Loads template coordinates from **Fork 2's submission-fork2.csv** (0.287 TM)
5. Loads e-values from **Kalai's Result.txt** (same search results as Fork 2)
6. For each of 28 test targets:
   - Pairwise features (N,N,64) from backbone
   - Template features (N,N,16) from Fork 2 templates
   - Concatenate -> (N,N,80) -> distance head -> MDS -> 5 structures
7. Writes `submission_adv1.csv`

### Expected Duration
5-15 minutes

### Verify Output

```cmd
python -c "import csv; rows=list(csv.DictReader(open('submission_adv1.csv'))); print('Rows:', len(rows)); print('Targets:', len(set(r['ID'].rsplit('_',1)[0] for r in rows)))"
```

Expected: 9762 rows, 28 targets.

---

## STEP 5: Compare ADV1 vs BASIC vs Fork 2

```cmd
python -c "
import csv

basic = {}
for r in csv.DictReader(open('../BASIC/submission.csv')):
    basic[r['ID']] = float(r['x_1'])

adv1 = {}
for r in csv.DictReader(open('submission_adv1.csv')):
    adv1[r['ID']] = float(r['x_1'])

fork2 = {}
for r in csv.DictReader(open('../../APPROACH1-TEMPLATE/fork2-JJ/REMOTE/run1/submission-fork2.csv')):
    fork2[r['ID']] = float(r['x_1'])

# Compare first residue of template-rich targets
for t in ['9G4J_1', '9LEC_1', '9LJN_1', '9E74_1', '9E9Q_1']:
    b = basic.get(t, 0)
    a = adv1.get(t, 0)
    f = fork2.get(t, 0)
    print(f'{t}: BASIC={b:.1f}, ADV1={a:.1f}, Fork2={f:.1f}')

print()
same_as_basic = sum(1 for k in basic if k in adv1 and abs(basic[k]-adv1[k]) < 0.01)
diff_from_basic = sum(1 for k in basic if k in adv1 and abs(basic[k]-adv1[k]) >= 0.01)
print(f'Same as BASIC: {same_as_basic}, Different: {diff_from_basic}')
"
```

**What to look for:**
- ADV1 coordinates should differ from BASIC (template features working)
- ADV1 coordinates may differ from Fork 2 too (neural network adds its own signal)
- If ADV1 = BASIC everywhere, template features aren't being used (bug)

---

## STEP 6: Build Kaggle Notebook

After local testing passes, build the Kaggle hybrid notebook:

1. Start from **Fork 2's working notebook** (`shmitha/rna-3d-folds-tbm-only-approach`)
   — it already runs, scores 0.287, all dependencies attached

2. Keep jaejohn's MMseqs2 + template pipeline cells (they produce template coordinates)

3. **Add new cells at the end** that:
   - Load ADV1 model from uploaded Kaggle datasets
   - Run neural inference using jaejohn's template output as input
   - Write final submission.csv with Option B post-processing

4. Upload to Kaggle:
   - `adv1_best_model.pt` + `RibonanzaNet.pt` (as "adv1-weights")
   - All FULL .py files from BASIC/models/ + HY-BAS-ADV1/models/ (as "adv1-code")
   - RibonanzaNet repo folder (as "ribonanzanet-repo")

5. Test -> Commit with Internet OFF -> Submit

**Expected score: higher than 0.287** because ADV1 produces real neural
predictions for no-template targets (where Fork 2 alone scores near zero).

---

## SUMMARY

```
Step 1: cd to HY-BAS-ADV1                              (30 seconds)
Step 2: python train_adv1.py --config config_adv1.yaml  (1-3 hours, GPU)
Step 3: Verify checkpoints/adv1_best_model.pt exists    (30 seconds)
Step 4: python predict_adv1.py ... --checkpoint ...     (5-15 minutes)
Step 5: Compare ADV1 vs BASIC vs Fork 2                 (1 minute)
Step 6: Build Kaggle notebook on Fork 2 base            (next session)
```

## KEY DESIGN DECISION LOG

| Decision | Choice | Reason |
|----------|--------|--------|
| Template source for inference | Fork 2 (jaejohn) | Scored 0.287 vs Kalai's ~0.15-0.20 |
| E-value source | Kalai Result.txt | Same MMseqs2 search across all runs |
| Training template source | Self-templates (true coords) | Training doesn't use submission CSVs |
| Kaggle notebook base | Fork 2 (shmitha fork) | Already runs, all deps attached, proven 0.287 |
| Kaggle MMseqs2 pipeline | Fork 2's jaejohn pipeline | Better template ranking than Fork 1 |
