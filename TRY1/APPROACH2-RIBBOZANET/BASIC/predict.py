"""
predict.py — Inference: generate submission.csv from test sequences.

Pipeline:
    1. Load frozen RibonanzaNet backbone + trained distance head
    2. For each test sequence:
       a. Tokenize → backbone → pairwise repr → distance head → distance matrix
       b. Generate 5 diverse 3D reconstructions from the distance matrix
       c. Each reconstruction uses different noise / refinement settings
    3. Format all predictions into competition submission CSV

Diversity strategy for 5 predictions:
    - Prediction 1: Clean MDS + 50 refinement steps
    - Prediction 2: Clean MDS + 100 refinement steps
    - Prediction 3: Noisy distances (scale=0.3) + 100 refinement steps
    - Prediction 4: Noisy distances (scale=0.5) + 100 refinement steps
    - Prediction 5: Noisy distances (scale=0.7) + 150 refinement steps

The competition scores best-of-5 by TM-score, so diverse predictions
improve the chance that at least one is close to the true structure.
"""

import os
import argparse
import yaml
import numpy as np
import torch
from tqdm import tqdm

from models.backbone import load_backbone, tokenize_sequence
from models.distance_head import DistanceMatrixHead
from models.reconstructor import reconstruct_3d
from utils.submission import format_submission, load_test_sequences


def predict_single_target(
    sequence: str,
    backbone: torch.nn.Module,
    distance_head: torch.nn.Module,
    device: torch.device,
    config: dict,
) -> list:
    """Generate 5 diverse 3D predictions for a single RNA target.

    Args:
        sequence: RNA sequence string.
        backbone: Frozen RibonanzaNet backbone.
        distance_head: Trained distance prediction head.
        device: torch device.
        config: Configuration dict.

    Returns:
        List of 5 np.ndarray, each of shape (N, 3) — C1' coordinates.
    """
    pred_cfg = config.get('prediction', {})
    recon_cfg = config.get('reconstruction', {})
    max_seq_len = config['data'].get('max_seq_len', 256)

    # Truncate if needed
    seq = sequence[:max_seq_len]
    N = len(seq)

    # Tokenize
    tokens = tokenize_sequence(seq).unsqueeze(0).to(device)  # (1, N)
    mask = torch.ones(1, N, dtype=torch.bool, device=device)

    # Forward through backbone + distance head
    with torch.no_grad():
        single_repr, pairwise_repr = backbone(tokens, mask)
        pred_dist = distance_head(pairwise_repr)  # (1, N, N)

    pred_dist = pred_dist.squeeze(0)  # (N, N)

    # Generate 5 diverse predictions
    noise_scale = pred_cfg.get('noise_scale', 0.5)
    refine_steps_range = pred_cfg.get('refine_steps_range', [50, 75, 100, 125, 150])
    consecutive_target = recon_cfg.get('target_consecutive_dist', 5.9)
    refine_lr = recon_cfg.get('refine_lr', 0.01)

    coords_list = []

    # Diversity configurations for each of the 5 predictions
    diversity_configs = [
        {'noise': 0.0, 'steps': refine_steps_range[0], 'seed': 0},
        {'noise': 0.0, 'steps': refine_steps_range[2], 'seed': 1},
        {'noise': noise_scale * 0.6, 'steps': refine_steps_range[2], 'seed': 2},
        {'noise': noise_scale, 'steps': refine_steps_range[2], 'seed': 3},
        {'noise': noise_scale * 1.4, 'steps': refine_steps_range[4], 'seed': 4},
    ]

    for div_cfg in diversity_configs:
        torch.manual_seed(div_cfg['seed'])
        np.random.seed(div_cfg['seed'])

        # Add noise to distance matrix for diversity
        noisy_dist = pred_dist.clone()
        if div_cfg['noise'] > 0:
            noise = torch.randn_like(pred_dist) * div_cfg['noise']
            # Keep noise symmetric and keep diagonal at 0
            noise = (noise + noise.T) / 2.0
            noise.fill_diagonal_(0.0)
            noisy_dist = torch.clamp(noisy_dist + noise, min=0.0)

        # Reconstruct 3D coordinates
        coords = reconstruct_3d(
            dist_matrix=noisy_dist,
            mask=None,
            method="mds_then_refine",
            refine_steps=div_cfg['steps'],
            refine_lr=refine_lr,
            consecutive_target=consecutive_target,
        )

        coords_np = coords.cpu().numpy()

        # Handle case where sequence was truncated
        if len(sequence) > max_seq_len:
            # Pad remaining positions with extrapolated coordinates
            remaining = len(sequence) - max_seq_len
            last_direction = coords_np[-1] - coords_np[-2] if N >= 2 else np.array([5.9, 0, 0])
            last_direction = last_direction / (np.linalg.norm(last_direction) + 1e-8) * 5.9
            extra = np.array([
                coords_np[-1] + last_direction * (i + 1)
                for i in range(remaining)
            ])
            coords_np = np.concatenate([coords_np, extra], axis=0)

        coords_list.append(coords_np)

    return coords_list


def predict(config: dict, checkpoint_path: str, test_csv_path: str, output_path: str):
    """Generate predictions for all test sequences.

    Args:
        config: Configuration dictionary.
        checkpoint_path: Path to trained distance head checkpoint.
        test_csv_path: Path to test_sequences.csv.
        output_path: Path to write submission.csv.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- Load backbone ----
    print("Loading backbone...")
    backbone = load_backbone(config)
    backbone = backbone.to(device)
    backbone.eval()

    # ---- Load trained distance head ----
    print(f"Loading distance head from {checkpoint_path}...")
    head_cfg = config['distance_head']
    pairwise_dim = config['backbone'].get('pairwise_dimension', 64)

    distance_head = DistanceMatrixHead(
        pair_dim=pairwise_dim,
        hidden_dim=head_cfg.get('hidden_dim', 128),
        num_layers=head_cfg.get('num_layers', 3),
        dropout=head_cfg.get('dropout', 0.1),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        distance_head.load_state_dict(checkpoint['model_state_dict'])
    else:
        distance_head.load_state_dict(checkpoint)

    distance_head = distance_head.to(device)
    distance_head.eval()

    # ---- Load test sequences ----
    print(f"Loading test sequences from {test_csv_path}...")
    test_sequences = load_test_sequences(test_csv_path)

    # ---- Generate predictions ----
    print(f"\nGenerating predictions for {len(test_sequences)} targets...")
    all_predictions = []

    for test_item in tqdm(test_sequences, desc="Predicting"):
        target_id = test_item['target_id']
        sequence = test_item['sequence']

        coords_list = predict_single_target(
            sequence=sequence,
            backbone=backbone,
            distance_head=distance_head,
            device=device,
            config=config,
        )

        all_predictions.append({
            'target_id': target_id,
            'sequence': sequence,
            'coords_list': coords_list,
        })

    # ---- Format and save submission ----
    print(f"\nFormatting submission...")
    df = format_submission(all_predictions, output_path)

    print(f"\nDone! Submission saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Targets: {len(all_predictions)}")
    print(f"Coordinate range: [{df.filter(like='x_').min().min():.1f}, {df.filter(like='x_').max().max():.1f}]")


def main():
    parser = argparse.ArgumentParser(description="Generate competition submission")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained distance head checkpoint (.pt)")
    parser.add_argument("--test_csv", type=str, default=None,
                        help="Path to test_sequences.csv (overrides config)")
    parser.add_argument("--output", type=str, default="submission.csv",
                        help="Path for output submission CSV")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override test CSV path if provided
    test_csv = args.test_csv or config['data'].get('test_csv_path')
    if test_csv is None:
        raise ValueError(
            "Test CSV path not specified. Use --test_csv or set data.test_csv_path in config."
        )

    predict(config, args.checkpoint, test_csv, args.output)


if __name__ == "__main__":
    main()
