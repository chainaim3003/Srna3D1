"""
predict_adv1.py — Inference for HY-BAS-ADV1 (Hybrid model).

Same as BASIC predict.py but with template features:
    1. Load frozen RibonanzaNet backbone + trained ADV1 distance head + template encoder
    2. Load templates from Approach 1 outputs (or MMseqs2 run)
    3. For each test sequence:
       a. Tokenize -> backbone -> pairwise repr (N,N,64)
       b. Get template coords -> template encoder -> template features (N,N,16)
       c. Concatenate -> (N,N,80)
       d. Distance head -> distance matrix -> MDS -> 3D coordinates
    4. Format into competition submission CSV
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
from models.template_encoder import TemplateEncoder
from models.template_loader import TemplateLoader
from utils.submission import format_submission, load_test_sequences


def predict_single_target(
    sequence: str,
    target_id: str,
    backbone: torch.nn.Module,
    distance_head: torch.nn.Module,
    template_encoder: torch.nn.Module,
    template_loader: TemplateLoader,
    device: torch.device,
    config: dict,
) -> list:
    """Generate 5 diverse 3D predictions for a single RNA target.

    This is the ADV1 version — same as BASIC but with template features
    concatenated to the pairwise representation before the distance head.
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

    # Get template data for this target
    tmpl_coords, tmpl_confidence, has_template = template_loader.get_template(
        target_id, seq_len=N
    )

    # Forward through backbone
    with torch.no_grad():
        single_repr, pairwise_repr = backbone(tokens, mask)
        # pairwise_repr: (1, N, N, 64)

        # Encode template features
        tmpl_coords_tensor = torch.from_numpy(tmpl_coords).float().to(device)
        template_feat = template_encoder(
            tmpl_coords_tensor, confidence=tmpl_confidence,
            has_template=has_template
        )  # (N, N, 16)
        template_feat = template_feat.unsqueeze(0)  # (1, N, N, 16)

        # Concatenate: (1, N, N, 64) + (1, N, N, 16) = (1, N, N, 80)
        combined = torch.cat([pairwise_repr, template_feat], dim=-1)

        # Distance head on combined features
        pred_dist = distance_head(combined)  # (1, N, N)

    pred_dist = pred_dist.squeeze(0)  # (N, N)

    # Generate 5 diverse predictions (same as BASIC)
    noise_scale = pred_cfg.get('noise_scale', 0.5)
    refine_steps_range = pred_cfg.get('refine_steps_range', [50, 75, 100, 125, 150])
    consecutive_target = recon_cfg.get('target_consecutive_dist', 5.9)
    refine_lr = recon_cfg.get('refine_lr', 0.01)

    coords_list = []

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

        noisy_dist = pred_dist.clone()
        if div_cfg['noise'] > 0:
            noise = torch.randn_like(pred_dist) * div_cfg['noise']
            noise = (noise + noise.T) / 2.0
            noise.fill_diagonal_(0.0)
            noisy_dist = torch.clamp(noisy_dist + noise, min=0.0)

        coords = reconstruct_3d(
            dist_matrix=noisy_dist,
            mask=None,
            method="mds_then_refine",
            refine_steps=div_cfg['steps'],
            refine_lr=refine_lr,
            consecutive_target=consecutive_target,
        )

        coords_np = coords.cpu().numpy()

        # Handle truncated sequences
        if len(sequence) > max_seq_len:
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


def warm_start_from_basic(adv1_head: DistanceMatrixHead,
                          basic_checkpoint_path: str,
                          device: torch.device):
    """Load BASIC weights into ADV1 distance head with expanded first layer.

    BASIC first layer: Linear(64, 128) -> weight shape (128, 64)
    ADV1 first layer:  Linear(80, 128) -> weight shape (128, 80)

    Strategy: copy columns 0-63, random-init columns 64-79.
    """
    print(f"Warm-starting from BASIC: {basic_checkpoint_path}")
    basic_ckpt = torch.load(basic_checkpoint_path, map_location=device)

    if 'model_state_dict' in basic_ckpt:
        basic_state = basic_ckpt['model_state_dict']
    else:
        basic_state = basic_ckpt

    adv1_state = adv1_head.state_dict()

    for key in adv1_state:
        if key in basic_state:
            basic_param = basic_state[key]
            adv1_param = adv1_state[key]

            if basic_param.shape == adv1_param.shape:
                # Same shape: copy directly
                adv1_state[key] = basic_param
            elif len(basic_param.shape) == 2 and len(adv1_param.shape) == 2:
                # Weight matrix with different input dim (first layer)
                # basic: (128, 64), adv1: (128, 80)
                if basic_param.shape[0] == adv1_param.shape[0] and \
                   basic_param.shape[1] < adv1_param.shape[1]:
                    print(f"  Expanding {key}: {basic_param.shape} -> {adv1_param.shape}")
                    adv1_state[key][:, :basic_param.shape[1]] = basic_param
                    # Columns 64-79 stay random-initialized
                else:
                    print(f"  Skipping {key}: shapes incompatible "
                          f"{basic_param.shape} vs {adv1_param.shape}")
            else:
                print(f"  Skipping {key}: shapes incompatible "
                      f"{basic_param.shape} vs {adv1_param.shape}")
        else:
            print(f"  New param (not in BASIC): {key}")

    adv1_head.load_state_dict(adv1_state)
    print("Warm-start complete.")


def predict(config: dict, checkpoint_path: str, test_csv_path: str,
            output_path: str):
    """Generate predictions for all test sequences using ADV1."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- Load backbone ----
    print("Loading backbone...")
    backbone = load_backbone(config)
    backbone = backbone.to(device)
    backbone.eval()

    # ---- Load ADV1 distance head ----
    head_cfg = config['distance_head']
    pair_dim = head_cfg.get('pair_dim', 80)  # 64 pairwise + 16 template

    distance_head = DistanceMatrixHead(
        pair_dim=pair_dim,
        hidden_dim=head_cfg.get('hidden_dim', 128),
        num_layers=head_cfg.get('num_layers', 3),
        dropout=head_cfg.get('dropout', 0.1),
    )

    # Load ADV1 checkpoint (or warm-start from BASIC)
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading ADV1 checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            distance_head.load_state_dict(checkpoint['model_state_dict'])
        else:
            distance_head.load_state_dict(checkpoint)
    else:
        # No ADV1 checkpoint — warm-start from BASIC
        basic_ckpt = config.get('warm_start', {}).get('basic_checkpoint')
        if basic_ckpt and os.path.exists(basic_ckpt):
            warm_start_from_basic(distance_head, basic_ckpt, device)
        else:
            print("WARNING: No checkpoint found. Using random weights.")

    distance_head = distance_head.to(device)
    distance_head.eval()

    # ---- Load template encoder ----
    tmpl_cfg = config.get('template', {})
    template_encoder = TemplateEncoder(
        template_dim=tmpl_cfg.get('template_dim', 16),
        num_bins=tmpl_cfg.get('num_bins', 22),
        max_dist=tmpl_cfg.get('max_dist', 40.0),
    )
    template_encoder = template_encoder.to(device)
    template_encoder.eval()

    # ---- Load template data ----
    print("Loading templates...")
    template_loader = TemplateLoader(
        mode="local",
        submission_csv=tmpl_cfg.get('local_submission_csv'),
        result_txt=tmpl_cfg.get('local_result_txt'),
    )

    # ---- Load test sequences ----
    print(f"Loading test sequences from {test_csv_path}...")
    test_sequences = load_test_sequences(test_csv_path)

    # ---- Generate predictions ----
    print(f"\nGenerating ADV1 predictions for {len(test_sequences)} targets...")
    all_predictions = []

    for test_item in tqdm(test_sequences, desc="Predicting"):
        target_id = test_item['target_id']
        sequence = test_item['sequence']

        coords_list = predict_single_target(
            sequence=sequence,
            target_id=target_id,
            backbone=backbone,
            distance_head=distance_head,
            template_encoder=template_encoder,
            template_loader=template_loader,
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


def main():
    parser = argparse.ArgumentParser(description="ADV1 Hybrid Prediction")
    parser.add_argument("--config", type=str, default="config_adv1.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="ADV1 checkpoint. If not provided, warm-starts from BASIC.")
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--output", type=str, default="submission_adv1.csv")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    test_csv = args.test_csv or config['data'].get('test_csv_path')
    predict(config, args.checkpoint, test_csv, args.output)


if __name__ == "__main__":
    main()
