# Approach 2 BASIC — RibonanzaNet + Distance Matrix → MDS 3D Reconstruction

## What This Is

A baseline pipeline for the Stanford RNA 3D Folding Part 2 competition that:
1. Uses **RibonanzaNet** (frozen, pretrained) as a feature backbone
2. Predicts **pairwise distances** between nucleotides from those features
3. Reconstructs **3D coordinates** via MDS + gradient refinement
4. Generates **5 diverse predictions** per target for best-of-5 scoring

## Prerequisites

### Option A: Official RibonanzaNet Repo (Recommended)
```bash
git clone https://github.com/Shujun-He/RibonanzaNet.git
# Download weights from: https://www.kaggle.com/datasets/shujun717/ribonanzanet-weights
```

### Option B: multimolecule Package
```bash
pip install multimolecule
# Note: May not expose internal pairwise representations directly
```

### Training Data
- Pre-processed pickle: https://www.kaggle.com/datasets/shujun717/stanford3d-dataprocessing-pickle
- Or use competition train_labels.csv + CIF files

## Setup
```bash
pip install -r requirements.txt
# Edit config.yaml to set paths to your data and weights
```

## Usage

### Train
```bash
python train.py --config config.yaml
```

### Predict (generate submission.csv)
```bash
python predict.py --config config.yaml --checkpoint best_model.pt --test_csv path/to/test_sequences.csv
```

## Architecture
```
RNA Sequence → RibonanzaNet (frozen) → Single + Pairwise Repr
                                            ↓
                                    Distance Head (trainable)
                                            ↓
                                    Predicted Distance Matrix (N×N)
                                            ↓
                                    MDS + Gradient Refinement
                                            ↓
                                    3D Coordinates (N×3)
```

## Key Dimensions (from official RibonanzaNet config)
- Single representation: (B, N, ninp) where ninp is configurable (default 256)
- Pairwise representation: (B, N, N, 64) — pairwise_dimension=64 in official config
- Distance matrix: (B, N, N) — symmetric, predicted in Ångströms
- Output coordinates: (B, N, 3) — C1' atom positions

## Official Sources
- RibonanzaNet repo: https://github.com/Shujun-He/RibonanzaNet
- RibonanzaNet weights: https://www.kaggle.com/datasets/shujun717/ribonanzanet-weights
- Competition: https://www.kaggle.com/competitions/stanford-rna-3d-folding-2
- Data processing: https://github.com/Shujun-He/Stanford3Dfolding_dataprocessing
- Training pickle: https://www.kaggle.com/datasets/shujun717/stanford3d-dataprocessing-pickle
