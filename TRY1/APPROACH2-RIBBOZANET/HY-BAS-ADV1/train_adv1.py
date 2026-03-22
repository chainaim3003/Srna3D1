"""
train_adv1.py — Training loop for ADV1 hybrid model.

Same as BASIC train.py but:
  1. Creates TemplateEncoder alongside DistanceMatrixHead
  2. Warm-starts distance head from BASIC checkpoint
  3. Uses self-templates during training: the TRUE coordinates of each
     training example serve as template input (simulating what happens
     when MMseqs2 finds a close homolog)
  4. Random masking: 50% of the time, zeroes out template features
     (simulating targets with no template available)

Usage:
    First time (warm-start from BASIC):
      python train_adv1.py --config config_adv1.yaml

    Resume ADV1 training:
      python train_adv1.py --config config_adv1.yaml --resume checkpoints/adv1_epoch10.pt
"""

import os
import argparse
import yaml
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from models.backbone import load_backbone
from models.distance_head import DistanceMatrixHead
from models.template_encoder import TemplateEncoder
from data.dataset import RNAStructureDataset, load_training_data
from data.collate import collate_rna_structures
from losses.distance_loss import DistanceMatrixLoss
from losses.constraint_loss import BondConstraintLoss, ClashPenaltyLoss
from predict_adv1 import warm_start_from_basic


def train(config: dict, resume_path: str = None):
    """Main ADV1 training function."""

    train_cfg = config['training']
    seed = train_cfg.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- Load backbone (frozen) ----
    print("\n--- Loading backbone ---")
    backbone = load_backbone(config)
    backbone = backbone.to(device)
    backbone.eval()

    # ---- Create distance head (ADV1: pair_dim=80) ----
    head_cfg = config['distance_head']
    pair_dim = head_cfg.get('pair_dim', 80)

    distance_head = DistanceMatrixHead(
        pair_dim=pair_dim,
        hidden_dim=head_cfg.get('hidden_dim', 128),
        num_layers=head_cfg.get('num_layers', 3),
        dropout=head_cfg.get('dropout', 0.1),
    )

    # ---- Create template encoder (trainable) ----
    tmpl_cfg = config.get('template', {})
    template_encoder = TemplateEncoder(
        template_dim=tmpl_cfg.get('template_dim', 16),
        num_bins=tmpl_cfg.get('num_bins', 22),
        max_dist=tmpl_cfg.get('max_dist', 40.0),
    )

    # ---- Warm-start or resume ----
    start_epoch = 0

    if resume_path and os.path.exists(resume_path):
        # Resume ADV1 training
        print(f"\nResuming from ADV1 checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        distance_head.load_state_dict(checkpoint['model_state_dict'])
        if 'template_encoder_state_dict' in checkpoint:
            template_encoder.load_state_dict(checkpoint['template_encoder_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resuming from epoch {start_epoch}")
    else:
        # First time: warm-start from BASIC
        basic_ckpt = config.get('warm_start', {}).get('basic_checkpoint')
        if basic_ckpt and os.path.exists(basic_ckpt):
            warm_start_from_basic(distance_head, basic_ckpt, device)
        else:
            print("WARNING: No BASIC checkpoint found. Training from scratch.")

    distance_head = distance_head.to(device)
    template_encoder = template_encoder.to(device)

    dh_params = sum(p.numel() for p in distance_head.parameters() if p.requires_grad)
    te_params = sum(p.numel() for p in template_encoder.parameters() if p.requires_grad)
    print(f"\nDistance head trainable params: {dh_params:,}")
    print(f"Template encoder trainable params: {te_params:,}")
    print(f"Total trainable params: {dh_params + te_params:,}")

    # ---- Load data ----
    print("\n--- Loading training data ---")
    train_data, val_data = load_training_data(config)
    max_seq_len = config['data'].get('max_seq_len', 256)

    train_dataset = RNAStructureDataset(train_data, max_seq_len=max_seq_len, augment=True)
    val_dataset = RNAStructureDataset(val_data, max_seq_len=max_seq_len, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get('batch_size', 4),
        shuffle=True,
        num_workers=train_cfg.get('num_workers', 2),
        collate_fn=collate_rna_structures,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.get('batch_size', 4),
        shuffle=False,
        num_workers=train_cfg.get('num_workers', 2),
        collate_fn=collate_rna_structures,
        pin_memory=True,
    )

    # ---- Loss functions ----
    loss_weights = train_cfg.get('loss_weights', {})
    dist_loss_fn = DistanceMatrixLoss()
    bond_loss_fn = BondConstraintLoss(
        target_dist=config['reconstruction'].get('target_consecutive_dist', 5.9),
    )
    clash_loss_fn = ClashPenaltyLoss(min_dist=3.0)

    w_dist = loss_weights.get('distance_mse', 1.0)
    w_bond = loss_weights.get('bond_constraint', 0.1)
    w_clash = loss_weights.get('clash_penalty', 0.05)

    # ---- Optimizer (trains BOTH distance_head AND template_encoder) ----
    epochs = train_cfg.get('epochs', 30)
    all_params = list(distance_head.parameters()) + list(template_encoder.parameters())

    optimizer = optim.AdamW(
        all_params,
        lr=train_cfg.get('learning_rate', 5e-5),
        weight_decay=train_cfg.get('weight_decay', 0.01),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    use_fp16 = train_cfg.get('use_fp16', True) and device.type == 'cuda'
    scaler = GradScaler(enabled=use_fp16)

    # ---- Checkpointing ----
    save_dir = train_cfg.get('save_dir', './checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    save_every = train_cfg.get('save_every', 5)

    best_val_loss = float('inf')
    template_mask_prob = 0.5  # 50% chance of hiding template during training

    # ---- Training loop ----
    print(f"\n--- Training ADV1 for {epochs} epochs (starting at {start_epoch}) ---")
    print(f"Template masking probability: {template_mask_prob}")

    for epoch in range(start_epoch, start_epoch + epochs):
        distance_head.train()
        template_encoder.train()

        epoch_loss = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            tokens = batch['tokens'].to(device)        # (B, N)
            masks = batch['masks'].to(device)           # (B, N)
            true_dists = batch['distances'].to(device)  # (B, N, N)
            true_coords = batch['coords'].to(device)    # (B, N, 3)

            B, N = tokens.shape

            optimizer.zero_grad()

            with autocast(enabled=use_fp16):
                # Backbone (frozen)
                with torch.no_grad():
                    single_repr, pairwise_repr = backbone(tokens, masks)
                    # pairwise_repr: (B, N, N, 64)

                # Template features from TRUE coordinates (self-template)
                # This simulates what happens when MMseqs2 finds a close hit
                template_feats = []
                for b in range(B):
                    # Random masking: 50% of the time, provide no template
                    if torch.rand(1).item() < template_mask_prob:
                        # No template — zeros
                        tf = torch.zeros(N, N, tmpl_cfg.get('template_dim', 16),
                                         device=device)
                    else:
                        # Use true coordinates as self-template
                        tf = template_encoder(
                            true_coords[b],  # (N, 3)
                            confidence=1.0,
                            has_template=True
                        )  # (N, N, 16)
                    template_feats.append(tf)

                template_feats = torch.stack(template_feats)  # (B, N, N, 16)

                # Concatenate: (B, N, N, 64) + (B, N, N, 16) = (B, N, N, 80)
                combined = torch.cat([pairwise_repr, template_feats], dim=-1)

                # Distance head
                pred_dists = distance_head(combined)  # (B, N, N)

                # Compute losses
                dist_loss = dist_loss_fn(pred_dists, true_dists, masks)
                bond_loss = bond_loss_fn(pred_dists, masks)
                clash_loss = clash_loss_fn(pred_dists, masks)

                total_loss = (w_dist * dist_loss +
                              w_bond * bond_loss +
                              w_clash * clash_loss)

            scaler.scale(total_loss).backward()
            if train_cfg.get('gradient_clip', 0) > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(all_params,
                                         train_cfg['gradient_clip'])
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = epoch_loss / max(n_batches, 1)

        # ---- Validation ----
        distance_head.eval()
        template_encoder.eval()
        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                tokens = batch['tokens'].to(device)
                masks = batch['masks'].to(device)
                true_dists = batch['distances'].to(device)
                true_coords = batch['coords'].to(device)
                B, N = tokens.shape

                single_repr, pairwise_repr = backbone(tokens, masks)

                # Validation: always use template (test best-case)
                template_feats = []
                for b in range(B):
                    tf = template_encoder(true_coords[b], confidence=1.0,
                                          has_template=True)
                    template_feats.append(tf)
                template_feats = torch.stack(template_feats)

                combined = torch.cat([pairwise_repr, template_feats], dim=-1)
                pred_dists = distance_head(combined)

                loss = dist_loss_fn(pred_dists, true_dists, masks)
                val_loss += loss.item()
                n_val += 1

        avg_val_loss = val_loss / max(n_val, 1)

        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
              f"val_loss={avg_val_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")

        # ---- Save checkpoint ----
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss

        if (epoch + 1) % save_every == 0 or is_best:
            ckpt = {
                'epoch': epoch,
                'model_state_dict': distance_head.state_dict(),
                'template_encoder_state_dict': template_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config,
            }

            if (epoch + 1) % save_every == 0:
                path = os.path.join(save_dir, f"adv1_epoch{epoch+1}.pt")
                torch.save(ckpt, path)
                print(f"  Saved: {path}")

            if is_best:
                path = os.path.join(save_dir, "adv1_best_model.pt")
                torch.save(ckpt, path)
                print(f"  Saved BEST: {path} (val_loss={avg_val_loss:.4f})")

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train ADV1 Hybrid Model")
    parser.add_argument("--config", type=str, default="config_adv1.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="ADV1 checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    train(config, args.resume)


if __name__ == "__main__":
    main()
