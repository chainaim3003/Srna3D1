"""
predict.py — ADV1 inference: generate submission.csv with template features.

MODIFIED from BASIC/predict.py:
  - Loads Approach 1 templates at startup
  - Passes template coords + confidence through TemplateEncoder
  - Concatenates template features with pairwise features before distance head
  - Loads unfrozen backbone layer weights from ADV1 checkpoint
"""

import os
import argparse
import yaml
import numpy as np
import torch
from tqdm import tqdm

from models.backbone import load_backbone, tokenize_sequence
from models.distance_head import DistanceMatrixHead
from models.template_encoder import TemplateEncoder
from models.reconstructor import reconstruct_3d
from utils.submission import format_submission, load_test_sequences
from data.template_loader import load_test_templates


def predict_single_target(
    sequence: str,
    target_id: str,
    backbone: torch.nn.Module,
    distance_head: torch.nn.Module,
    template_encoder: torch.nn.Module,
    templates: dict,
    device: torch.device,
    config: dict,
) -> list:
    """Generate 5 diverse 3D predictions for a single RNA target."""
    pred_cfg = config.get('prediction', {})
    recon_cfg = config.get('reconstruction', {})
    max_seq_len = config['data'].get('max_seq_len', 256)

    seq = sequence[:max_seq_len]
    N = len(seq)

    # Tokenize
    tokens = tokenize_sequence(seq).unsqueeze(0).to(device)
    mask = torch.ones(1, N, dtype=torch.bool, device=device)

    # Template data for this target
    template_data = templates.get(target_id, None)
    if template_data is not None:
        tmpl_coords = template_data['coords'][:N]
        tmpl_conf = template_data['confidence'][:N]
        # Pad if template shorter than sequence
        if tmpl_coords.shape[0] < N:
            pad_len = N - tmpl_coords.shape[0]
            tmpl_coords = np.pad(tmpl_coords, ((0, pad_len), (0, 0)), mode='constant')
            tmpl_conf = np.pad(tmpl_conf, (0, pad_len), mode='constant')
        tmpl_coords_t = torch.tensor(tmpl_coords, dtype=torch.float32).unsqueeze(0).to(device)
        tmpl_conf_t = torch.tensor(tmpl_conf, dtype=torch.float32).unsqueeze(0).to(device)
    else:
        tmpl_coords_t = torch.zeros(1, N, 3, device=device)
        tmpl_conf_t = torch.zeros(1, N, device=device)

    # Forward through backbone + template encoder + distance head
    with torch.no_grad():
        single_repr, pairwise_repr = backbone(tokens, mask)

        if template_encoder is not None:
            template_features = template_encoder(tmpl_coords_t, tmpl_conf_t)
            combined = torch.cat([pairwise_repr, template_features], dim=-1)
        else:
            combined = pairwise_repr

        pred_dist = distance_head(combined)

    pred_dist = pred_dist.squeeze(0)  # (N, N)

    # Generate 5 diverse predictions
    noise_scale = pred_cfg.get('noise_scale', 0.5)
    refine_steps_range = pred_cfg.get('refine_steps_range', [50, 75, 100, 125, 150])
    consecutive_target = recon_cfg.get('target_consecutive_dist', 5.9)
    refine_lr = recon_cfg.get('refine_lr', 0.01)

    diversity_configs = [
        {'noise': 0.0, 'steps': refine_steps_range[0], 'seed': 0},
        {'noise': 0.0, 'steps': refine_steps_range[2], 'seed': 1},
        {'noise': noise_scale * 0.6, 'steps': refine_steps_range[2], 'seed': 2},
        {'noise': noise_scale, 'steps': refine_steps_range[2], 'seed': 3},
        {'noise': noise_scale * 1.4, 'steps': refine_steps_range[4], 'seed': 4},
    ]

    coords_list = []
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
            dist_matrix=noisy_dist, mask=None,
            method="mds_then_refine",
            refine_steps=div_cfg['steps'],
            refine_lr=refine_lr,
            consecutive_target=consecutive_target,
        )
        coords_np = coords.cpu().numpy()

        # Pad if sequence was truncated
        if len(sequence) > max_seq_len:
            remaining = len(sequence) - max_seq_len
            last_direction = coords_np[-1] - coords_np[-2] if N >= 2 else np.array([5.9, 0, 0])
            last_direction = last_direction / (np.linalg.norm(last_direction) + 1e-8) * 5.9
            extra = np.array([coords_np[-1] + last_direction * (i + 1) for i in range(remaining)])
            coords_np = np.concatenate([coords_np, extra], axis=0)

        coords_list.append(coords_np)

    return coords_list


def predict(config: dict, checkpoint_path: str, test_csv_path: str, output_path: str):
    """Generate predictions for all test sequences."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- Load backbone ----
    print("Loading backbone...")
    backbone = load_backbone(config)
    backbone = backbone.to(device)

    # ---- Load template encoder ----
    tmpl_cfg = config.get('template', {})
    template_enabled = tmpl_cfg.get('enabled', False)
    template_dim = tmpl_cfg.get('template_dim', 16)

    if template_enabled:
        template_encoder = TemplateEncoder(
            num_distance_bins=tmpl_cfg.get('num_distance_bins', 22),
            template_dim=template_dim,
        )
        template_encoder = template_encoder.to(device)
    else:
        template_encoder = None
        template_dim = 0

    # ---- Load distance head ----
    head_cfg = config['distance_head']
    pairwise_dim = config['backbone'].get('pairwise_dimension', 64)
    total_pair_dim = pairwise_dim + template_dim

    distance_head = DistanceMatrixHead(
        pair_dim=total_pair_dim,
        hidden_dim=head_cfg.get('hidden_dim', 128),
        num_layers=head_cfg.get('num_layers', 3),
        dropout=head_cfg.get('dropout', 0.1),
    )

    # ---- Load checkpoint ----
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load distance head
    if 'distance_head_state_dict' in checkpoint:
        distance_head.load_state_dict(checkpoint['distance_head_state_dict'])
    elif 'model_state_dict' in checkpoint:
        distance_head.load_state_dict(checkpoint['model_state_dict'])

    # Load template encoder
    if 'template_encoder_state_dict' in checkpoint and template_encoder is not None:
        template_encoder.load_state_dict(checkpoint['template_encoder_state_dict'])

    # Load unfrozen backbone layers
    if 'backbone_unfrozen_state_dict' in checkpoint:
        freeze_first_n = config['backbone'].get('freeze_first_n', 7)
        for layer_key, layer_state in checkpoint['backbone_unfrozen_state_dict'].items():
            layer_idx = int(layer_key.split('_')[1])
            backbone.model.transformer_encoder[layer_idx].load_state_dict(layer_state)
        print(f"Loaded unfrozen backbone layers from checkpoint")

    distance_head = distance_head.to(device)
    backbone.eval()
    distance_head.eval()
    if template_encoder is not None:
        template_encoder.eval()

    # ---- Load Approach 1 templates ----
    templates = {}
    if template_enabled:
        template_csv = tmpl_cfg.get('test_template_csv')
        result_txt = tmpl_cfg.get('result_txt')
        if template_csv:
            templates = load_test_templates(
                template_csv_path=template_csv,
                result_txt_path=result_txt,
                confidence_scale=tmpl_cfg.get('confidence_scale', 15.0),
                gap_penalty=tmpl_cfg.get('gap_penalty', 0.3),
            )
        else:
            print("WARNING: template.enabled=true but no test_template_csv set")

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
            target_id=target_id,
            backbone=backbone,
            distance_head=distance_head,
            template_encoder=template_encoder,
            templates=templates,
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

    # Template usage summary
    targets_with_template = sum(1 for p in all_predictions if p['target_id'] in templates
                                and templates[p['target_id']]['confidence'].max() > 0)
    print(f"Targets with template features: {targets_with_template}/{len(all_predictions)}")


def main():
    parser = argparse.ArgumentParser(description="Generate ADV1 competition submission")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--output", type=str, default="submission.csv")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    test_csv = args.test_csv or config['data'].get('test_csv_path')
    if test_csv is None:
        raise ValueError("Test CSV path not specified.")

    predict(config, args.checkpoint, test_csv, args.output)


if __name__ == "__main__":
    main()
