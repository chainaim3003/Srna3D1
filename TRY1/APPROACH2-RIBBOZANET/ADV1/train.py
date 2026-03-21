"""
train.py — ADV1 training loop with selective unfreezing + template features.

MODIFIED from BASIC/train.py:
  1. Discriminative learning rates: backbone 1e-5, head+encoder 1e-4
  2. LR warmup for first 5% of epochs
  3. Gradient accumulation (effective batch size = batch_size × accum_steps)
  4. Partial weight loading from BASIC checkpoint
  5. Template encoder in forward pass
  6. Resume support (same as BASIC)

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --resume checkpoints/latest_model.pt
"""

import os
import math
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


def load_basic_checkpoint_partial(distance_head, basic_ckpt_path, device):
    """Load BASIC checkpoint weights into ADV1 distance head (partial transfer).

    BASIC head: first layer is Linear(64, 128)
    ADV1 head:  first layer is Linear(64+template_dim, 128)

    Columns 0-63 are copied from BASIC. Columns 64+ stay random.
    All other layers transfer directly (same shapes).
    """
    if not os.path.exists(basic_ckpt_path):
        print(f"  BASIC checkpoint not found at {basic_ckpt_path} — skipping warm-start")
        return

    basic_ckpt = torch.load(basic_ckpt_path, map_location=device, weights_only=False)
    basic_weights = basic_ckpt.get('model_state_dict', basic_ckpt)

    adv1_state = distance_head.state_dict()
    transferred = 0
    partial = 0

    for key, basic_param in basic_weights.items():
        if key not in adv1_state:
            continue

        adv1_param = adv1_state[key]

        if basic_param.shape == adv1_param.shape:
            # Matching shapes — copy directly
            adv1_state[key] = basic_param
            transferred += 1
        elif key.endswith('.weight') and basic_param.dim() == 2 and adv1_param.dim() == 2:
            # Shape mismatch on a weight matrix — partial transfer
            # Copy as many rows/cols as match
            min_rows = min(basic_param.shape[0], adv1_param.shape[0])
            min_cols = min(basic_param.shape[1], adv1_param.shape[1])
            adv1_state[key][:min_rows, :min_cols] = basic_param[:min_rows, :min_cols]
            partial += 1
            print(f"  Partial load: {key} — copied ({min_rows},{min_cols}) of {tuple(adv1_param.shape)}")
        else:
            print(f"  Skipped: {key} (shape {tuple(basic_param.shape)} vs {tuple(adv1_param.shape)})")

    distance_head.load_state_dict(adv1_state)
    print(f"  Warm-started distance head: {transferred} full transfers, {partial} partial transfers")


def make_lr_lambda(warmup_epochs, total_epochs):
    """Create LR schedule: linear warmup then cosine decay.

    Args:
        warmup_epochs: Number of warmup epochs (LR ramps from 0 to 1).
        total_epochs: Total training epochs.

    Returns:
        Lambda function for torch.optim.lr_scheduler.LambdaLR.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup: 0 → 1
            return max(1e-3, epoch / max(1, warmup_epochs))
        else:
            # Cosine decay: 1 → 0
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return max(1e-6, 0.5 * (1 + math.cos(math.pi * progress)))
    return lr_lambda


def train(config: dict, resume_path: str = None):
    """Main training function."""
    train_cfg = config['training']
    seed = train_cfg.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ---- Load backbone (with selective unfreezing) ----
    print("\n--- Loading backbone ---")
    backbone = load_backbone(config)
    backbone = backbone.to(device)

    # In ADV1: backbone is NOT always eval. Unfrozen layers need train mode for dropout.
    freeze = config['backbone'].get('freeze', True)
    if freeze:
        backbone.eval()
    # If not fully frozen, backbone stays in train mode (default)

    # ---- Create template encoder (trainable) ----
    print("\n--- Creating template encoder ---")
    tmpl_cfg = config.get('template', {})
    template_enabled = tmpl_cfg.get('enabled', False)
    template_dim = tmpl_cfg.get('template_dim', 16)
    num_distance_bins = tmpl_cfg.get('num_distance_bins', 22)

    if template_enabled:
        template_encoder = TemplateEncoder(
            num_distance_bins=num_distance_bins,
            template_dim=template_dim,
        )
        template_encoder = template_encoder.to(device)
        print(f"Template encoder: {sum(p.numel() for p in template_encoder.parameters()):,} params")
    else:
        template_encoder = None
        template_dim = 0
        print("Template encoder: DISABLED")

    # ---- Create distance head (trainable) ----
    head_cfg = config['distance_head']
    pairwise_dim = config['backbone'].get('pairwise_dimension', 64)
    total_pair_dim = pairwise_dim + template_dim  # 64 + 16 = 80

    distance_head = DistanceMatrixHead(
        pair_dim=total_pair_dim,
        hidden_dim=head_cfg.get('hidden_dim', 128),
        num_layers=head_cfg.get('num_layers', 3),
        dropout=head_cfg.get('dropout', 0.1),
    )
    distance_head = distance_head.to(device)

    head_params = sum(p.numel() for p in distance_head.parameters() if p.requires_grad)
    backbone_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"Distance head params: {head_params:,}")
    print(f"Backbone trainable params: {backbone_trainable:,}")
    print(f"Total trainable: {head_params + backbone_trainable + (sum(p.numel() for p in template_encoder.parameters()) if template_encoder else 0):,}")

    # ---- Partial weight loading from BASIC ----
    basic_ckpt_path = train_cfg.get('basic_checkpoint', None)
    if basic_ckpt_path and resume_path is None:
        # Only warm-start from BASIC if NOT resuming an ADV1 checkpoint
        print(f"\n--- Warm-starting from BASIC checkpoint ---")
        load_basic_checkpoint_partial(distance_head, basic_ckpt_path, device)

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
        num_workers=train_cfg.get('num_workers', 0),
        collate_fn=collate_rna_structures,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.get('batch_size', 4),
        shuffle=False,
        num_workers=train_cfg.get('num_workers', 0),
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

    # ---- Optimizer with discriminative learning rates ----
    epochs = train_cfg.get('epochs', 50)
    lr_backbone = train_cfg.get('learning_rate_backbone', 1e-5)
    lr_head = train_cfg.get('learning_rate_head', 1e-4)

    param_groups = []

    # Group 1: Unfrozen backbone parameters
    backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': lr_backbone})
        print(f"Optimizer group 1: {len(backbone_params)} backbone tensors, lr={lr_backbone}")

    # Group 2: Template encoder parameters
    if template_encoder is not None:
        tmpl_params = list(template_encoder.parameters())
        param_groups.append({'params': tmpl_params, 'lr': lr_head})
        print(f"Optimizer group 2: {len(tmpl_params)} template encoder tensors, lr={lr_head}")

    # Group 3: Distance head parameters
    param_groups.append({'params': distance_head.parameters(), 'lr': lr_head})
    print(f"Optimizer group 3: distance head, lr={lr_head}")

    optimizer = optim.AdamW(
        param_groups,
        weight_decay=train_cfg.get('weight_decay', 0.01),
    )

    # ---- LR scheduler with warmup ----
    warmup_fraction = train_cfg.get('warmup_fraction', 0.05)
    warmup_epochs = max(1, int(epochs * warmup_fraction))
    lr_lambda = make_lr_lambda(warmup_epochs, epochs)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"LR schedule: {warmup_epochs} warmup epochs, then cosine decay to epoch {epochs}")

    # ---- Gradient accumulation ----
    accum_steps = train_cfg.get('gradient_accumulation_steps', 4)
    effective_batch = train_cfg.get('batch_size', 4) * accum_steps
    print(f"Gradient accumulation: {accum_steps} steps, effective batch size: {effective_batch}")

    # ---- Mixed precision ----
    use_fp16 = train_cfg.get('use_fp16', True) and device.type == 'cuda'
    scaler = GradScaler(enabled=use_fp16)

    # ---- Checkpointing ----
    save_dir = train_cfg.get('save_dir', './checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    save_every = train_cfg.get('save_every', 10)

    best_val_loss = float('inf')
    best_epoch = -1
    start_epoch = 1

    # ---- Resume from ADV1 checkpoint ----
    if resume_path and os.path.exists(resume_path):
        print(f"\n--- Resuming from {resume_path} ---")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)

        if 'distance_head_state_dict' in checkpoint:
            distance_head.load_state_dict(checkpoint['distance_head_state_dict'])
            print(f"  Loaded distance head weights")
        elif 'model_state_dict' in checkpoint:
            # Backward compatibility with BASIC checkpoints
            try:
                distance_head.load_state_dict(checkpoint['model_state_dict'])
                print(f"  Loaded distance head from model_state_dict")
            except RuntimeError:
                print(f"  Could not load model_state_dict directly — attempting partial load")
                load_basic_checkpoint_partial(distance_head, resume_path, device)

        if 'template_encoder_state_dict' in checkpoint and template_encoder is not None:
            template_encoder.load_state_dict(checkpoint['template_encoder_state_dict'])
            print(f"  Loaded template encoder weights")

        if 'backbone_unfrozen_state_dict' in checkpoint:
            freeze_first_n = config['backbone'].get('freeze_first_n', 7)
            for layer_key, layer_state in checkpoint['backbone_unfrozen_state_dict'].items():
                layer_idx = int(layer_key.split('_')[1])
                backbone.model.transformer_encoder[layer_idx].load_state_dict(layer_state)
            print(f"  Loaded unfrozen backbone layer weights")

        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"  Loaded optimizer state")
            except (ValueError, KeyError) as e:
                print(f"  Could not load optimizer state ({e}) — using fresh optimizer")

        if 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"  Loaded scheduler state")
            except Exception:
                if 'epoch' in checkpoint:
                    for _ in range(checkpoint['epoch']):
                        scheduler.step()
                    print(f"  Fast-forwarded scheduler to epoch {checkpoint['epoch']}")

        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print(f"  Resuming from epoch {start_epoch}")

        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
            best_epoch = checkpoint.get('best_epoch', -1)
            print(f"  Best val loss so far: {best_val_loss:.4f} (epoch {best_epoch})")

        print(f"  Will train epochs {start_epoch} → {epochs}")

    elif resume_path:
        print(f"\n  WARNING: Resume path not found: {resume_path}")

    # ---- Training loop ----
    grad_clip = train_cfg.get('gradient_clip', 1.0)

    print(f"\n--- Training epochs {start_epoch} → {epochs} ---")
    print(f"Loss weights: dist={w_dist}, bond={w_bond}, clash={w_clash}")
    print(f"FP16: {use_fp16}, Grad clip: {grad_clip}")

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        # ---- Train phase ----
        distance_head.train()
        if template_encoder is not None:
            template_encoder.train()
        if not freeze:
            backbone.train()
            # Keep frozen layers in eval mode for stable BatchNorm/Dropout
            freeze_first_n = config['backbone'].get('freeze_first_n', 7)
            for i in range(freeze_first_n):
                backbone.model.transformer_encoder[i].eval()

        train_losses = []
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)):
            tokens = batch['tokens'].to(device)
            true_dist = batch['distance_matrix'].to(device)
            mask = batch['mask'].to(device)
            tmpl_coords = batch['template_coords'].to(device)
            tmpl_conf = batch['template_confidence'].to(device)

            with autocast(enabled=use_fp16):
                # Forward through backbone
                # Unfrozen layers compute gradients automatically (requires_grad=True)
                # Frozen layers don't (requires_grad=False) — no torch.no_grad() needed
                single_repr, pairwise_repr = backbone(tokens, mask)

                # Template encoder
                if template_encoder is not None:
                    template_features = template_encoder(tmpl_coords, tmpl_conf)
                    combined = torch.cat([pairwise_repr, template_features], dim=-1)
                else:
                    combined = pairwise_repr

                # Distance head
                pred_dist = distance_head(combined)

                # Losses
                loss_dist = dist_loss_fn(pred_dist, true_dist, mask)
                loss_bond = bond_loss_fn(pred_dist, mask)
                loss_clash = clash_loss_fn(pred_dist, mask)
                total_loss = w_dist * loss_dist + w_bond * loss_bond + w_clash * loss_clash

                # Scale by accumulation steps
                total_loss = total_loss / accum_steps

            scaler.scale(total_loss).backward()

            # Step optimizer every accum_steps batches
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)

                # Clip gradients for all trainable parameters
                all_params = list(distance_head.parameters())
                if template_encoder is not None:
                    all_params += list(template_encoder.parameters())
                all_params += backbone_params
                torch.nn.utils.clip_grad_norm_(all_params, grad_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_losses.append(total_loss.item() * accum_steps)  # Undo scaling for logging

        avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0

        # ---- Validation phase ----
        distance_head.eval()
        if template_encoder is not None:
            template_encoder.eval()
        backbone.eval()
        val_losses = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]", leave=False):
                tokens = batch['tokens'].to(device)
                true_dist = batch['distance_matrix'].to(device)
                mask = batch['mask'].to(device)
                tmpl_coords = batch['template_coords'].to(device)
                tmpl_conf = batch['template_confidence'].to(device)

                with autocast(enabled=use_fp16):
                    single_repr, pairwise_repr = backbone(tokens, mask)

                    if template_encoder is not None:
                        template_features = template_encoder(tmpl_coords, tmpl_conf)
                        combined = torch.cat([pairwise_repr, template_features], dim=-1)
                    else:
                        combined = pairwise_repr

                    pred_dist = distance_head(combined)

                    loss_dist = dist_loss_fn(pred_dist, true_dist, mask)
                    loss_bond = bond_loss_fn(pred_dist, mask)
                    loss_clash = clash_loss_fn(pred_dist, mask)
                    total_loss = w_dist * loss_dist + w_bond * loss_bond + w_clash * loss_clash

                val_losses.append(total_loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0

        scheduler.step()
        elapsed = time.time() - epoch_start
        current_lrs = [group['lr'] * scheduler.get_last_lr()[0] / scheduler.get_last_lr()[0]
                       for group in optimizer.param_groups]
        lr_str = ', '.join(f'{lr:.6f}' for lr in scheduler.get_last_lr())

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train: {avg_train_loss:.4f} | "
            f"Val: {avg_val_loss:.4f} | "
            f"LR: {lr_str} | "
            f"Time: {elapsed:.1f}s"
        )

        # ---- Save checkpoint helper ----
        def save_checkpoint(path, is_best=False):
            freeze_first_n = config['backbone'].get('freeze_first_n', 7)
            ckpt = {
                'epoch': epoch,
                'distance_head_state_dict': distance_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'config': config,
            }
            if template_encoder is not None:
                ckpt['template_encoder_state_dict'] = template_encoder.state_dict()
            if not freeze:
                ckpt['backbone_unfrozen_state_dict'] = {
                    f'layer_{i}': backbone.model.transformer_encoder[i].state_dict()
                    for i in range(freeze_first_n, len(backbone.model.transformer_encoder))
                }
            torch.save(ckpt, path)

        # ---- Save best model ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            save_checkpoint(os.path.join(save_dir, "best_model.pt"), is_best=True)
            print(f"  New best model saved (val_loss={avg_val_loss:.4f})")

        # ---- Periodic save ----
        if epoch % save_every == 0:
            save_checkpoint(os.path.join(save_dir, f"model_epoch{epoch}.pt"))

        # ---- Always save latest (for resume) ----
        save_checkpoint(os.path.join(save_dir, "latest_model.pt"))

    print(f"\n--- Training complete ---")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Best model saved at: {os.path.join(save_dir, 'best_model.pt')}")


def main():
    parser = argparse.ArgumentParser(description="Train ADV1 Distance Matrix Head")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to ADV1 checkpoint to resume from.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    train(config, resume_path=args.resume)


if __name__ == "__main__":
    main()
