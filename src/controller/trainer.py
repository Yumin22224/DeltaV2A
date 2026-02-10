"""
Controller Training Loop (Phase B)

Trains AudioController using the inverse mapping database.
Loss: MSE between predicted and ground-truth normalized parameters.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import json
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from .model import AudioController
from ..database.inverse_mapping import InverseMappingDB, InverseMappingDataset


class ControllerTrainer:
    """Trainer for AudioController."""

    def __init__(
        self,
        model: AudioController,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cpu",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.history = {'train_loss': [], 'val_loss': [], 'best_val_loss': float('inf')}

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n = 0
        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            clap_emb = batch['clap_embedding'].to(self.device)
            style = batch['style_label'].to(self.device)
            params_gt = batch['normalized_params'].to(self.device)

            params_pred = self.model(clap_emb, style)
            loss = self.criterion(params_pred, params_gt)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n += 1
        return total_loss / max(n, 1)

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n = 0
        for batch in self.val_loader:
            clap_emb = batch['clap_embedding'].to(self.device)
            style = batch['style_label'].to(self.device)
            params_gt = batch['normalized_params'].to(self.device)

            loss = self.criterion(self.model(clap_emb, style), params_gt)
            total_loss += loss.item()
            n += 1
        return total_loss / max(n, 1)

    def train(self, num_epochs: int, save_dir: str, save_every: int = 10):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nTraining AudioController for {num_epochs} epochs...")
        print(f"  Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        print(f"  Params: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            print(f"Epoch {epoch}/{num_epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")

            if val_loss < self.history['best_val_loss']:
                self.history['best_val_loss'] = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, save_dir / 'controller_best.pt')
                print(f"  -> Best model saved (val_loss: {val_loss:.6f})")

            if epoch % save_every == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                }, save_dir / f'controller_epoch_{epoch}.pt')

        torch.save({
            'epoch': num_epochs,
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
        }, save_dir / 'controller_final.pt')

        with open(save_dir / 'training_log.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"\nTraining complete. Best val loss: {self.history['best_val_loss']:.6f}")


def train_controller(
    db_path: str,
    output_dir: str,
    style_vocab_size: int,
    total_params: int,
    batch_size: int = 64,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    val_split: float = 0.2,
    hidden_dims: List[int] = None,
    dropout: float = 0.1,
    device: str = "cpu",
):
    """High-level function to train the AudioController."""
    db = InverseMappingDB(db_path)
    dataset = InverseMappingDataset(db)
    print(f"Loaded {len(dataset)} records from {db_path}")

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = AudioController(
        audio_embed_dim=512,
        style_vocab_size=style_vocab_size,
        total_params=total_params,
        hidden_dims=hidden_dims or [512, 256, 128],
        dropout=dropout,
    )

    trainer = ControllerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
    )
    trainer.train(num_epochs=num_epochs, save_dir=output_dir)
