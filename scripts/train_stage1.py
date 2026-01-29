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
    phase="1a",
):
    """Single training epoch"""
    s_encoder.train()
    generator.train()

    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Phase {phase}")

    for batch in pbar:
        # Move to device
        audio_init = batch["audio_init_mel"].to(device)
        audio_edit = batch["audio_edit_mel"].to(device)

        # Forward: S_encoder
        S_pred = s_encoder(audio_init, audio_edit)

        # Forward: Generator (reconstruction)
        audio_recon = generator(audio_init, S_pred)

        # TODO: Compute losses
        # loss_recon = ...
        # loss_struct = ...
        # loss_rank = ...

        # Placeholder loss
        loss = nn.functional.mse_loss(audio_recon, audio_edit)

        # Backward
        optimizer_s.zero_grad()
        optimizer_g.zero_grad()
        loss.backward()
        optimizer_s.step()
        optimizer_g.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    return total_loss / len(dataloader)


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
        train_loss = train_epoch(
            s_encoder,
            generator,
            dataloader,
            optimizer_s,
            optimizer_g,
            device,
            phase=args.phase,
        )

        print(f"Train loss: {train_loss:.4f}")

        # Log to wandb
        if config.logging.use_wandb:
            wandb.log({"epoch": epoch, "train_loss": train_loss})

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

    # TODO: Save S_proxy statistics
    # Collect all S_pred outputs and compute statistics

    if config.logging.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
