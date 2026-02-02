"""
Training script for Stage 1: Audio-only Control Learning

Phase 1-A: Synthetic warmup
Phase 1-B: Remix fine-tuning
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import wandb
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import (
    SyntheticPairDataset,
    RemixPairDataset,
    get_audio_transform,
    create_dataloader,
)
from src.models import SEncoder, AudioGenerator
from src.losses import (
    MultiResolutionSTFTLoss,
    StructurePreservationLoss,
    PairwiseRankingLoss,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage 1")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage1_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["1a", "1b"],
        default="1a",
        help="Training phase (1a: synthetic, 1b: remix)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    return parser.parse_args()


def train_epoch(
    s_encoder,
    generator,
    dataloader,
    optimizer_s,
    optimizer_g,
    device,
    config,
    phase="1a",
    loss_mrstft=None,
    loss_struct=None,
    loss_rank=None,
):
    """Single training epoch"""
    s_encoder.train()
    generator.train()

    total_loss = 0
    total_mrstft = 0
    total_struct = 0
    total_rank = 0

    pbar = tqdm(dataloader, desc=f"Phase {phase}")

    for batch in pbar:
        # Move to device
        audio_init = batch["audio_init_mel"].to(device)
        audio_edit = batch["audio_edit_mel"].to(device)

        # Forward: S_encoder
        S_pred = s_encoder(audio_init, audio_edit)

        # Forward: Generator (reconstruction)
        audio_recon = generator(audio_init, S_pred)

        # Compute losses based on Spec v2
        loss = 0
        loss_dict = {}

        # 1. Multi-Resolution STFT Loss (reconstruction quality)
        if loss_mrstft is not None:
            mrstft_loss = loss_mrstft(audio_edit, audio_recon)
            loss_dict['mrstft'] = mrstft_loss.item()
            loss += config.losses.mrstft.weight * mrstft_loss
            total_mrstft += mrstft_loss.item()

        # 2. Structure Preservation Loss (head consistency)
        if loss_struct is not None:
            struct_loss = loss_struct(
                S_pred,
                head_target=batch.get('head_target'),
            )
            loss_dict['struct'] = struct_loss.item()
            loss += config.losses.structure.weight * struct_loss
            total_struct += struct_loss.item()

        # 3. Pairwise Ranking Loss (relative ordering)
        if loss_rank is not None and phase == "1b":
            # For remix pairs, we can use ranking loss
            rank_loss = loss_rank(audio_init, audio_edit, audio_recon, S_pred)
            loss_dict['rank'] = rank_loss.item()
            loss += config.losses.rank.weight * rank_loss
            total_rank += rank_loss.item()

        # Backward
        optimizer_s.zero_grad()
        optimizer_g.zero_grad()
        loss.backward()

        # Gradient clipping
        if config.training.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                s_encoder.parameters(),
                config.training.gradient_clip,
            )
            torch.nn.utils.clip_grad_norm_(
                generator.parameters(),
                config.training.gradient_clip,
            )

        optimizer_s.step()
        optimizer_g.step()

        total_loss += loss.item()
        pbar.set_postfix(loss_dict)

    metrics = {
        'total': total_loss / len(dataloader),
        'mrstft': total_mrstft / len(dataloader),
        'struct': total_struct / len(dataloader),
    }

    if phase == "1b":
        metrics['rank'] = total_rank / len(dataloader)

    return metrics


def main():
    args = parse_args()

    # Load config
    config = OmegaConf.load(args.config)
    print(f"Loaded config from {args.config}")
    print(OmegaConf.to_yaml(config))

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Initialize wandb (optional)
    if config.logging.use_wandb:
        wandb.init(
            project=config.logging.wandb_project,
            name=f"stage1_{args.phase}",
            config=OmegaConf.to_container(config),
        )

    # Create dataset and dataloader
    audio_transform = get_audio_transform(config.data.audio)

    if args.phase == "1a":
        # Synthetic dataset
        dataset = SyntheticPairDataset(
            data_dir=config.data.synthetic.dataset_path,
            audio_transform=audio_transform,
            split="train",
        )
        batch_size = config.data.synthetic.batch_size
        epochs = config.training.phase_1a.epochs
        lr = config.training.phase_1a.learning_rate
    else:
        # Remix dataset
        dataset = RemixPairDataset(
            data_dir=config.data.remix.dataset_path,
            audio_transform=audio_transform,
            split="train",
        )
        batch_size = config.data.remix.batch_size
        epochs = config.training.phase_1b.epochs
        lr = config.training.phase_1b.learning_rate

    dataloader = create_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=config.training.num_workers,
        shuffle=True,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")

    # Create models
    s_encoder = SEncoder(
        backbone=config.model.s_encoder.backbone,
        hidden_dim=config.model.s_encoder.hidden_dim,
        num_heads=config.model.num_heads,
        head_dim=config.model.head_dim,
    ).to(device)

    generator = AudioGenerator(
        num_heads=config.model.num_heads,
        head_dim=config.model.head_dim,
        use_lora=config.model.audio_generator.use_lora,
        lora_rank=config.model.audio_generator.lora_rank,
    ).to(device)

    # Optimizers
    optimizer_s = torch.optim.AdamW(
        s_encoder.parameters(),
        lr=lr,
        weight_decay=config.optimizer.weight_decay,
    )
    optimizer_g = torch.optim.AdamW(
        generator.parameters(),
        lr=lr,
        weight_decay=config.optimizer.weight_decay,
    )

    # Loss functions (based on Spec v2)
    loss_mrstft = MultiResolutionSTFTLoss(
        scales=config.losses.mrstft.scales,
    ).to(device)

    loss_struct = StructurePreservationLoss(
        num_heads=config.model.num_heads,
        struct_heads=config.losses.structure.struct_heads,
        style_heads=config.losses.structure.style_heads,
    ).to(device)

    loss_rank = PairwiseRankingLoss(
        margin=config.losses.rank.margin,
    ).to(device) if args.phase == "1b" else None

    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        s_encoder.load_state_dict(checkpoint["s_encoder"])
        generator.load_state_dict(checkpoint["generator"])
        optimizer_s.load_state_dict(checkpoint["optimizer_s"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # Train
        metrics = train_epoch(
            s_encoder,
            generator,
            dataloader,
            optimizer_s,
            optimizer_g,
            device,
            config,
            phase=args.phase,
            loss_mrstft=loss_mrstft,
            loss_struct=loss_struct,
            loss_rank=loss_rank,
        )

        print(f"Train loss: {metrics['total']:.4f}")
        print(f"  MRSTFT: {metrics['mrstft']:.4f}")
        print(f"  Struct: {metrics['struct']:.4f}")
        if 'rank' in metrics:
            print(f"  Rank: {metrics['rank']:.4f}")

        # Log to wandb
        if config.logging.use_wandb:
            log_dict = {"epoch": epoch}
            log_dict.update({f"train/{k}": v for k, v in metrics.items()})
            wandb.log(log_dict)

        # Save checkpoint
        if (epoch + 1) % config.logging.save_every_n_epochs == 0:
            save_dir = Path(config.output.output_path)
            save_dir.mkdir(parents=True, exist_ok=True)

            checkpoint = {
                "epoch": epoch,
                "s_encoder": s_encoder.state_dict(),
                "generator": generator.state_dict(),
                "optimizer_s": optimizer_s.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "config": config,
            }

            save_path = save_dir / f"checkpoint_phase{args.phase}_epoch{epoch+1}.pt"
            torch.save(checkpoint, save_path)
            print(f"Saved checkpoint to {save_path}")

    # Save final models
    save_dir = Path(config.output.output_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(s_encoder.state_dict(), save_dir / "s_encoder_final.pt")
    torch.save(generator.state_dict(), save_dir / "generator_final.pt")
    print(f"Saved final models to {save_dir}")

    # Compute and save S_proxy statistics (for Stage 2)
    print("\nComputing S_proxy statistics...")
    s_encoder.eval()

    all_t_values = [[] for _ in range(config.model.num_heads)]
    all_g_values = [[] for _ in range(config.model.num_heads)]

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing statistics"):
            audio_init = batch["audio_init_mel"].to(device)
            audio_edit = batch["audio_edit_mel"].to(device)

            S_pred = s_encoder(audio_init, audio_edit)

            # Collect statistics per head
            for h in range(config.model.num_heads):
                t_h, g_h = S_pred[h]
                all_t_values[h].append(t_h.cpu())
                all_g_values[h].append(g_h.cpu())

    # Compute mean and std for each head
    statistics = {}
    for h in range(config.model.num_heads):
        t_all = torch.cat(all_t_values[h], dim=0)  # (N, head_dim)
        g_all = torch.cat(all_g_values[h], dim=0)  # (N, 1)

        statistics[f'head_{h}'] = {
            't_mean': t_all.mean(dim=0),  # (head_dim,)
            't_std': t_all.std(dim=0),    # (head_dim,)
            'g_mean': g_all.mean(),       # scalar
            'g_std': g_all.std(),         # scalar
        }

    # Save statistics
    torch.save(statistics, save_dir / "S_proxy_statistics.pt")
    print(f"Saved S_proxy statistics to {save_dir / 'S_proxy_statistics.pt'}")

    if config.logging.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
