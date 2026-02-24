"""
Phase 3 Training Loop

Trains the DSP parameter decoder using audio + text conditioning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, List
from tqdm import tqdm

from src.models.decoder import DSPParameterDecoder
from experiment.src.phase3_dataset import Phase3Dataset


class Phase3Trainer:
    """Trainer for Phase 3 decoder."""

    def __init__(
        self,
        model: DSPParameterDecoder,
        embedder,  # CLAPEmbedder or MultimodalEmbedder
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cpu",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        """
        Args:
            model: DSPParameterDecoder instance
            embedder: CLAP embedder for audio and text embeddings
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay for AdamW
        """
        self.model = model.to(device)
        self.embedder = embedder
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Loss function
        self.criterion = nn.MSELoss()

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'best_val_loss': float('inf'),
        }

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.train_loader, desc="Training"):
            # Get batch data
            audio_raw = batch['audio_raw'].to(self.device)
            text_condition = batch['text_condition']
            effect_names = batch['effect_name']
            params_gt = batch['parameters'].to(self.device)

            # Extract embeddings (no grad for embedder)
            with torch.no_grad():
                # Audio embeddings
                audio_embed = self.embedder.embed_audio(
                    audio_raw,
                    sample_rate=self.embedder.sample_rate,
                )

                # Text embeddings
                text_embed = self.embedder.embed_text(text_condition)

            # Forward pass (predict parameters for each sample's effect)
            batch_loss = 0.0
            for i in range(len(audio_embed)):
                # Predict parameters
                params_pred = self.model(
                    audio_embed[i:i+1],
                    text_embed[i:i+1],
                    effect_name=effect_names[i],
                )

                # Compute loss
                loss = self.criterion(params_pred, params_gt[i:i+1])
                batch_loss += loss

            # Average loss over batch
            batch_loss = batch_loss / len(audio_embed)

            # Backward pass
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            total_loss += batch_loss.item()
            num_batches += 1

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # Per-effect losses
        effect_losses = {}

        for batch in tqdm(self.val_loader, desc="Validation"):
            # Get batch data
            audio_raw = batch['audio_raw'].to(self.device)
            text_condition = batch['text_condition']
            effect_names = batch['effect_name']
            params_gt = batch['parameters'].to(self.device)

            # Extract embeddings
            audio_embed = self.embedder.embed_audio(
                audio_raw,
                sample_rate=self.embedder.sample_rate,
            )
            text_embed = self.embedder.embed_text(text_condition)

            # Forward pass
            batch_loss = 0.0
            for i in range(len(audio_embed)):
                effect_name = effect_names[i]

                # Predict parameters
                params_pred = self.model(
                    audio_embed[i:i+1],
                    text_embed[i:i+1],
                    effect_name=effect_name,
                )

                # Compute loss
                loss = self.criterion(params_pred, params_gt[i:i+1])
                batch_loss += loss

                # Track per-effect loss
                if effect_name not in effect_losses:
                    effect_losses[effect_name] = []
                effect_losses[effect_name].append(loss.item())

            # Average loss over batch
            batch_loss = batch_loss / len(audio_embed)
            total_loss += batch_loss.item()
            num_batches += 1

        # Average per-effect losses
        for effect_name in effect_losses:
            effect_losses[effect_name] = np.mean(effect_losses[effect_name])

        return {
            'val_loss': total_loss / num_batches,
            'effect_losses': effect_losses,
        }

    def train(
        self,
        num_epochs: int,
        save_dir: str,
        save_every: int = 10,
    ):
        """
        Train for multiple epochs.

        Args:
            num_epochs: Number of epochs
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")

            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            print(f"Train Loss: {train_loss:.6f}")

            # Validate
            val_results = self.validate()
            val_loss = val_results['val_loss']
            self.history['val_loss'].append(val_loss)
            print(f"Val Loss: {val_loss:.6f}")

            # Print per-effect losses
            print("\nPer-effect validation losses:")
            for effect_name, loss in val_results['effect_losses'].items():
                print(f"  {effect_name}: {loss:.6f}")

            # Save best model
            if val_loss < self.history['best_val_loss']:
                self.history['best_val_loss'] = val_loss
                best_path = save_dir / 'decoder_best.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'history': self.history,
                }, best_path)
                print(f"✓ Saved best model (val_loss: {val_loss:.6f})")

            # Save checkpoint
            if epoch % save_every == 0:
                checkpoint_path = save_dir / f'decoder_epoch_{epoch}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'history': self.history,
                }, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")

        # Save final model
        final_path = save_dir / 'decoder_final.pt'
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history,
        }, final_path)

        # Save training history
        history_path = save_dir / 'training_log.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"Best val loss: {self.history['best_val_loss']:.6f}")
        print(f"Final model saved to: {final_path}")
        print(f"{'='*60}\n")


def run_phase3_training(
    dataset: Phase3Dataset,
    embedder,
    output_dir: str,
    batch_size: int = 32,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    val_split: float = 0.2,
    device: str = "cpu",
):
    """
    Run Phase 3 training.

    Args:
        dataset: Phase3Dataset instance
        embedder: CLAP embedder for audio and text embeddings
        output_dir: Directory to save results
        batch_size: Batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        val_split: Validation split ratio
        device: Device to train on
    """
    print("\n" + "="*60)
    print("PHASE 3: TRAINING (Learning - The Decoder)")
    print("="*60)

    # Split dataset
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"\nDataset split:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device != "cpu" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device != "cpu" else False,
    )

    # Initialize model
    print("\nInitializing decoder model...")
    model = DSPParameterDecoder(
        audio_embed_dim=embedder.embed_dim,
        text_embed_dim=embedder.embed_dim,
        hidden_dims=[512, 256, 128],
        num_heads=8,
        dropout=0.1,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize trainer
    trainer = Phase3Trainer(
        model=model,
        embedder=embedder,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
    )

    # Train
    trainer.train(
        num_epochs=num_epochs,
        save_dir=output_dir,
        save_every=10,
    )

    print("\n" + "="*60)
    print("PHASE 3 COMPLETE")
    print("="*60)
