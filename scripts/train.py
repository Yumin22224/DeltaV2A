"""
Training Script for Delta Mapper

Core experiment: Can we learn ΔV → ΔA mapping with context modulation?
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from datetime import datetime

os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.models.embedder import ImageBindEmbedder
from src.models.delta_mapper import DeltaMapper, DeltaLoss
from src.data.dataset import DeltaPairDataset, collate_delta_pairs
from src.data.transforms import AudioTransform, ImageTransform


def parse_args():
    parser = argparse.ArgumentParser(description="Train Delta Mapper")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/mps/cpu)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    return parser.parse_args()


def get_device(config, args):
    """Determine device to use"""
    if args.device:
        return torch.device(args.device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train_epoch(
    model: DeltaMapper,
    embedder: ImageBindEmbedder,
    dataloader: DataLoader,
    criterion: DeltaLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config,
) -> dict:
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    total_l2 = 0.0
    total_cosine = 0.0
    total_cosine_sim = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        # Move to device
        i_init = batch['i_init'].to(device)
        i_edit = batch['i_edit'].to(device)
        a_init = batch['a_init_waveform'].to(device)
        a_edit = batch['a_edit_waveform'].to(device)

        # Get embeddings (frozen ImageBind)
        with torch.no_grad():
            z_i_init = embedder.embed_image(i_init)
            z_i_edit = embedder.embed_image(i_edit)
            z_a_init = embedder.embed_audio(a_init)
            z_a_edit = embedder.embed_audio(a_edit)

        # Compute deltas
        delta_v = z_i_edit - z_i_init  # Visual delta
        delta_a_true = z_a_edit - z_a_init  # Audio delta (target)

        # Forward pass
        delta_a_pred = model(delta_v, z_a_init, z_i_init)

        # Compute loss
        loss, loss_dict = criterion(delta_a_pred, delta_a_true)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss += loss_dict['total']
        total_l2 += loss_dict['l2']
        total_cosine += loss_dict['cosine']
        total_cosine_sim += loss_dict['cosine_sim_mean']
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'cos_sim': f"{loss_dict['cosine_sim_mean']:.4f}",
        })

    return {
        'loss': total_loss / num_batches,
        'l2': total_l2 / num_batches,
        'cosine': total_cosine / num_batches,
        'cosine_sim': total_cosine_sim / num_batches,
    }


@torch.no_grad()
def validate(
    model: DeltaMapper,
    embedder: ImageBindEmbedder,
    dataloader: DataLoader,
    criterion: DeltaLoss,
    device: torch.device,
) -> dict:
    """Validation pass"""
    model.eval()

    total_loss = 0.0
    total_l2 = 0.0
    total_cosine = 0.0
    total_cosine_sim = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Validating"):
        i_init = batch['i_init'].to(device)
        i_edit = batch['i_edit'].to(device)
        a_init = batch['a_init_waveform'].to(device)
        a_edit = batch['a_edit_waveform'].to(device)

        # Get embeddings
        z_i_init = embedder.embed_image(i_init)
        z_i_edit = embedder.embed_image(i_edit)
        z_a_init = embedder.embed_audio(a_init)
        z_a_edit = embedder.embed_audio(a_edit)

        # Compute deltas
        delta_v = z_i_edit - z_i_init
        delta_a_true = z_a_edit - z_a_init

        # Forward pass
        delta_a_pred = model(delta_v, z_a_init, z_i_init)

        # Compute loss
        loss, loss_dict = criterion(delta_a_pred, delta_a_true)

        total_loss += loss_dict['total']
        total_l2 += loss_dict['l2']
        total_cosine += loss_dict['cosine']
        total_cosine_sim += loss_dict['cosine_sim_mean']
        num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'l2': total_l2 / num_batches,
        'cosine': total_cosine / num_batches,
        'cosine_sim': total_cosine_sim / num_batches,
    }


def main():
    args = parse_args()

    # Load config
    config = OmegaConf.load(args.config)
    OmegaConf.resolve(config)

    print(f"Loaded config from {args.config}")

    # Setup device
    device = get_device(config, args)
    print(f"Using device: {device}")

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(config.output.checkpoint_dir) / timestamp
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")

    # Save config
    OmegaConf.save(config, checkpoint_dir / "config.yaml")

    # Build embedder (frozen)
    print("\nLoading ImageBind embedder...")
    embedder = ImageBindEmbedder(device=str(device), freeze=True)

    # Build model
    print("Building Delta Mapper...")
    model = DeltaMapper(
        embed_dim=config.model.embed_dim,
        hidden_dim=config.model.delta_mapper.hidden_dim,
        num_layers=config.model.delta_mapper.num_layers,
        dropout=config.model.delta_mapper.dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Delta Mapper parameters: {num_params:,}")

    # Build loss
    criterion = DeltaLoss(
        l2_weight=config.training.loss.l2_weight,
        cosine_weight=config.training.loss.cosine_weight,
    )

    # Build optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Build dataset
    print("\nLoading dataset...")
    image_transform = ImageTransform(
        resolution=config.data.image.resolution,
        mean=tuple(config.data.image.mean),
        std=tuple(config.data.image.std),
    )
    audio_transform = AudioTransform(
        sample_rate=config.data.audio.sample_rate,
        n_mels=config.data.audio.n_mels,
        target_lufs=config.data.audio.target_lufs,
    )

    dataset = DeltaPairDataset(
        data_dir=config.data.dataset_path,
        image_transform=image_transform,
        audio_transform=audio_transform,
    )

    # Split dataset
    val_size = int(len(dataset) * config.training.val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.training.seed),
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Build dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=collate_delta_pairs,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=collate_delta_pairs,
        pin_memory=True,
    )

    # Resume if specified
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    # Training loop
    print(f"\n{'='*60}")
    print("Starting training")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, config.training.num_epochs):
        # Train
        train_metrics = train_epoch(
            model, embedder, train_loader, criterion, optimizer, device, epoch, config
        )

        # Validate
        val_metrics = validate(model, embedder, val_loader, criterion, device)

        # Log
        print(f"\nEpoch {epoch}:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
              f"L2: {train_metrics['l2']:.4f}, "
              f"Cosine: {train_metrics['cosine']:.4f}, "
              f"CosSim: {train_metrics['cosine_sim']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
              f"L2: {val_metrics['l2']:.4f}, "
              f"Cosine: {val_metrics['cosine']:.4f}, "
              f"CosSim: {val_metrics['cosine_sim']:.4f}")

        # Save checkpoint
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']

        if (epoch + 1) % config.training.save_interval == 0 or is_best:
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'best_val_loss': best_val_loss,
                'config': OmegaConf.to_container(config),
            }

            # Save latest
            torch.save(checkpoint, checkpoint_dir / "latest.pt")

            # Save best
            if is_best:
                torch.save(checkpoint, checkpoint_dir / "best.pt")
                print(f"  ✓ New best model saved (loss: {best_val_loss:.4f})")

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
