"""
train.py — Training loop for the Distance Matrix prediction head.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --resume checkpoints/best_model.pt
    python train.py --config config.yaml --resume checkpoints/model_epoch10.pt

What this trains:
    ONLY the DistanceMatrixHead (small MLP on pairwise features).
    The RibonanzaNet backbone is FROZEN — its parameters do not change.

Training pipeline:
    1. Load frozen RibonanzaNet backbone
    2. Load training data (from pickle or CIF files)
    3. For each batch:
       a. Tokenize sequences → run through frozen backbone → get pairwise repr
       b. Run pairwise repr through distance head → predicted distances
       c. Compute loss: MSE(predicted_distances, true_distances) + constraints
       d. Backprop through distance head only → update distance head parameters
    4. Save best checkpoint based on validation loss

The 3D reconstruction (MDS + refinement) is NOT used during training.
It only runs at inference time (see predict.py).
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
from data.dataset import RNAStructureDataset, load_training_data
from data.collate import collate_rna_structures
from losses.distance_loss import DistanceMatrixLoss
from losses.constraint_loss import BondConstraintLoss, ClashPenaltyLoss


def train(config: dict, resume_path: str = None):
    """Main training function.

    Args:
        config: Configuration dictionary loaded from config.yaml.
        resume_path: Path to checkpoint to resume from. If provided,
                     loads model weights, optimizer state, scheduler state,
                     and continues from the saved epoch.
    """
    # ---- Setup ----
    train_cfg = config['training']
    seed = train_cfg.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ---- Load backbone (frozen) ----
    print("\n--- Loading backbone ---")
    backbone = load_backbone(config)
    backbone = backbone.to(device)
    backbone.eval()  # Always in eval mode since it's frozen

    # ---- Create distance head (trainable) ----
    head_cfg = config['distance_head']
    pairwise_dim = config['backbone'].get('pairwise_dimension', 64)

    distance_head = DistanceMatrixHead(
        pair_dim=pairwise_dim,
        hidden_dim=head_cfg.get('hidden_dim', 128),
        num_layers=head_cfg.get('num_layers', 3),
        dropout=head_cfg.get('dropout', 0.1),
    )
    distance_head = distance_head.to(device)

    trainable_params = sum(p.numel() for p in distance_head.parameters() if p.requires_grad)
    print(f"Distance head trainable parameters: {trainable_params:,}")

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

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

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

    # ---- Optimizer & scheduler ----
    epochs = train_cfg.get('epochs', 100)

    optimizer = optim.AdamW(
        distance_head.parameters(),
        lr=train_cfg.get('learning_rate', 1e-4),
        weight_decay=train_cfg.get('weight_decay', 0.01),
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    # Mixed precision
    use_fp16 = train_cfg.get('use_fp16', True) and device.type == 'cuda'
    scaler = GradScaler(enabled=use_fp16)

    # ---- Checkpointing ----
    save_dir = train_cfg.get('save_dir', './checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    save_every = train_cfg.get('save_every', 10)

    best_val_loss = float('inf')
    best_epoch = -1
    start_epoch = 1

    # ---- Resume from checkpoint ----
    if resume_path and os.path.exists(resume_path):
        print(f"\n--- Resuming from {resume_path} ---")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)

        # Load model weights
        if 'model_state_dict' in checkpoint:
            distance_head.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Loaded model weights")
        else:
            distance_head.load_state_dict(checkpoint)
            print(f"  Loaded model weights (raw state dict)")

        # Load optimizer state (for correct momentum/adam state)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"  Loaded optimizer state")

        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"  Loaded scheduler state")
        elif 'epoch' in checkpoint:
            # If no scheduler state saved, step scheduler to catch up
            for _ in range(checkpoint['epoch']):
                scheduler.step()
            print(f"  Fast-forwarded scheduler to epoch {checkpoint['epoch']}")

        # Load scaler state (for mixed precision)
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print(f"  Loaded scaler state")

        # Resume from next epoch
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print(f"  Resuming from epoch {start_epoch}")

        # Restore best val loss tracking
        if 'val_loss' in checkpoint:
            best_val_loss = checkpoint['val_loss']
            best_epoch = checkpoint.get('epoch', -1)
            print(f"  Best val loss so far: {best_val_loss:.4f} (epoch {best_epoch})")

        # If best_val_loss was saved separately (from best_model.pt)
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
            best_epoch = checkpoint.get('best_epoch', best_epoch)

        print(f"  Will train epochs {start_epoch} → {epochs}")

    elif resume_path:
        print(f"\n  WARNING: Resume path not found: {resume_path}")
        print(f"  Starting training from scratch.")

    # ---- Training loop ----
    grad_clip = train_cfg.get('gradient_clip', 1.0)

    print(f"\n--- Training epochs {start_epoch} → {epochs} ---")
    print(f"Loss weights: dist={w_dist}, bond={w_bond}, clash={w_clash}")
    print(f"FP16: {use_fp16}, Grad clip: {grad_clip}")

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        # ---- Train phase ----
        distance_head.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False):
            tokens = batch['tokens'].to(device)
            true_dist = batch['distance_matrix'].to(device)
            mask = batch['mask'].to(device)

            optimizer.zero_grad()

            with autocast(enabled=use_fp16):
                # Forward through frozen backbone
                with torch.no_grad():
                    single_repr, pairwise_repr = backbone(tokens, mask)

                # Forward through trainable distance head
                pred_dist = distance_head(pairwise_repr)

                # Compute losses
                loss_dist = dist_loss_fn(pred_dist, true_dist, mask)
                loss_bond = bond_loss_fn(pred_dist, mask)
                loss_clash = clash_loss_fn(pred_dist, mask)

                total_loss = w_dist * loss_dist + w_bond * loss_bond + w_clash * loss_clash

            # Backward + update
            scaler.scale(total_loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(distance_head.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            train_losses.append(total_loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0

        # ---- Validation phase ----
        distance_head.eval()
        val_losses = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]", leave=False):
                tokens = batch['tokens'].to(device)
                true_dist = batch['distance_matrix'].to(device)
                mask = batch['mask'].to(device)

                with autocast(enabled=use_fp16):
                    single_repr, pairwise_repr = backbone(tokens, mask)
                    pred_dist = distance_head(pairwise_repr)

                    loss_dist = dist_loss_fn(pred_dist, true_dist, mask)
                    loss_bond = bond_loss_fn(pred_dist, mask)
                    loss_clash = clash_loss_fn(pred_dist, mask)

                    total_loss = w_dist * loss_dist + w_bond * loss_bond + w_clash * loss_clash

                val_losses.append(total_loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0

        # Step scheduler
        scheduler.step()

        elapsed = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train: {avg_train_loss:.4f} | "
            f"Val: {avg_val_loss:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

        # ---- Save best model ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            save_path = os.path.join(save_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': distance_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'config': config,
            }, save_path)
            print(f"  New best model saved (val_loss={avg_val_loss:.4f})")

        # ---- Periodic save (always save latest for resume) ----
        if epoch % save_every == 0:
            save_path = os.path.join(save_dir, f"model_epoch{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': distance_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'config': config,
            }, save_path)

        # ---- Always save latest (for resume after Ctrl+C) ----
        save_path = os.path.join(save_dir, "latest_model.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': distance_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'val_loss': avg_val_loss,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'config': config,
        }, save_path)

    print(f"\n--- Training complete ---")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Best model saved at: {os.path.join(save_dir, 'best_model.pt')}")


def main():
    parser = argparse.ArgumentParser(description="Train Distance Matrix Head")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from. "
                             "Loads model, optimizer, scheduler, and epoch number. "
                             "Example: --resume checkpoints/latest_model.pt")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    train(config, resume_path=args.resume)


if __name__ == "__main__":
    main()
