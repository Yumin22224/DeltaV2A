"""
Controller Training Loop (Phase B)

Trains AudioController using the inverse mapping database.
Loss:
  - weighted parameter regression loss (MSE or Huber)
  - optional effect activity BCE loss (multi-label)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import json
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

from .model import AudioController
from ..database.inverse_mapping import InverseMappingDB, InverseMappingDataset
from ..effects.pedalboard_effects import EFFECT_CATALOG


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
        effect_names: Optional[List[str]] = None,
        inactive_param_weight: float = 1.0,
        activity_loss_weight: float = 0.0,
        param_loss_type: str = "mse",
        huber_delta: float = 0.05,
        effect_loss_weights: Optional[Dict[str, float]] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.activity_criterion = nn.BCEWithLogitsLoss()
        self.effect_names = effect_names or []
        self.inactive_param_weight = float(inactive_param_weight)
        self.activity_loss_weight = float(activity_loss_weight)
        self.param_loss_type = str(param_loss_type).lower()
        if self.param_loss_type not in {"mse", "huber"}:
            raise ValueError(f"Unsupported param_loss_type: {param_loss_type}")
        self.huber_delta = float(huber_delta)
        self.effect_loss_weights = effect_loss_weights or {}
        self.effect_param_slices = self._build_effect_param_slices(self.effect_names)
        self.param_weight_vector = self._build_param_weight_vector(
            self.effect_names, self.effect_loss_weights
        ).to(device)
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_param_loss': [],
            'val_param_loss': [],
            'train_activity_loss': [],
            'val_activity_loss': [],
            'best_val_loss': float('inf'),
            'best_val_param_loss': float('inf'),
            'best_val_activity_loss': float('inf'),
        }

    @staticmethod
    def _build_effect_param_slices(effect_names: List[str]) -> List[slice]:
        slices: List[slice] = []
        start = 0
        for effect_name in effect_names:
            width = EFFECT_CATALOG[effect_name].num_params
            slices.append(slice(start, start + width))
            start += width
        return slices

    @staticmethod
    def _build_param_weight_vector(
        effect_names: List[str],
        effect_loss_weights: Dict[str, float],
    ) -> torch.Tensor:
        weights: List[float] = []
        for effect_name in effect_names:
            effect_weight = float(effect_loss_weights.get(effect_name, 1.0))
            width = EFFECT_CATALOG[effect_name].num_params
            weights.extend([effect_weight] * width)
        if not weights:
            return torch.ones(0, dtype=torch.float32)
        return torch.tensor(weights, dtype=torch.float32)

    def _expand_effect_mask(self, effect_mask: torch.Tensor) -> torch.Tensor:
        if not self.effect_param_slices:
            raise ValueError("effect_param_slices is empty; cannot expand effect mask.")
        chunks = []
        for i, sl in enumerate(self.effect_param_slices):
            width = sl.stop - sl.start
            chunks.append(effect_mask[:, i:i + 1].expand(-1, width))
        return torch.cat(chunks, dim=1)

    def _compute_losses(
        self,
        params_pred: torch.Tensor,
        params_gt: torch.Tensor,
        effect_active_mask: Optional[torch.Tensor],
        activity_logits: Optional[torch.Tensor],
    ):
        if self.param_loss_type == "huber":
            param_err = F.smooth_l1_loss(
                params_pred,
                params_gt,
                reduction='none',
                beta=self.huber_delta,
            )
        else:
            param_err = torch.square(params_pred - params_gt)

        weights = torch.ones_like(param_err)
        if self.param_weight_vector.numel() == param_err.shape[1]:
            weights = weights * self.param_weight_vector.unsqueeze(0)

        if effect_active_mask is not None and self.inactive_param_weight < 1.0:
            param_active_mask = self._expand_effect_mask(effect_active_mask)
            active_weights = torch.where(
                param_active_mask > 0.5,
                torch.ones_like(param_active_mask),
                torch.full_like(param_active_mask, self.inactive_param_weight),
            )
            weights = weights * active_weights

        weight_denom = torch.clamp(weights.sum(), min=1e-8)
        param_loss = (param_err * weights).sum() / weight_denom

        activity_loss = torch.zeros((), device=params_gt.device)
        if activity_logits is not None and effect_active_mask is not None:
            activity_loss = self.activity_criterion(activity_logits, effect_active_mask)

        total_loss = param_loss + (self.activity_loss_weight * activity_loss)
        return total_loss, param_loss, activity_loss

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_param_loss = 0.0
        total_activity_loss = 0.0
        n = 0
        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            clap_emb = batch['clap_embedding'].to(self.device)
            style = batch['style_label'].to(self.device)
            params_gt = batch['normalized_params'].to(self.device)
            effect_active_mask = None
            if 'effect_active_mask' in batch:
                effect_active_mask = batch['effect_active_mask'].to(self.device)

            params_pred, activity_logits = self.model.forward_with_activity(clap_emb, style)
            loss, param_loss, activity_loss = self._compute_losses(
                params_pred=params_pred,
                params_gt=params_gt,
                effect_active_mask=effect_active_mask,
                activity_logits=activity_logits,
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_param_loss += param_loss.item()
            total_activity_loss += activity_loss.item()
            n += 1
        denom = max(n, 1)
        return (
            total_loss / denom,
            total_param_loss / denom,
            total_activity_loss / denom,
        )

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        total_param_loss = 0.0
        total_activity_loss = 0.0
        n = 0
        for batch in self.val_loader:
            clap_emb = batch['clap_embedding'].to(self.device)
            style = batch['style_label'].to(self.device)
            params_gt = batch['normalized_params'].to(self.device)
            effect_active_mask = None
            if 'effect_active_mask' in batch:
                effect_active_mask = batch['effect_active_mask'].to(self.device)

            params_pred, activity_logits = self.model.forward_with_activity(clap_emb, style)
            loss, param_loss, activity_loss = self._compute_losses(
                params_pred=params_pred,
                params_gt=params_gt,
                effect_active_mask=effect_active_mask,
                activity_logits=activity_logits,
            )
            total_loss += loss.item()
            total_param_loss += param_loss.item()
            total_activity_loss += activity_loss.item()
            n += 1
        denom = max(n, 1)
        return (
            total_loss / denom,
            total_param_loss / denom,
            total_activity_loss / denom,
        )

    def train(self, num_epochs: int, save_dir: str, save_every: int = 10):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nTraining AudioController for {num_epochs} epochs...")
        print(f"  Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        print(f"  Params: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, num_epochs + 1):
            train_loss, train_param_loss, train_activity_loss = self.train_epoch()
            val_loss, val_param_loss, val_activity_loss = self.validate()
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_param_loss'].append(train_param_loss)
            self.history['val_param_loss'].append(val_param_loss)
            self.history['train_activity_loss'].append(train_activity_loss)
            self.history['val_activity_loss'].append(val_activity_loss)

            print(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train(total/param/act): {train_loss:.6f}/{train_param_loss:.6f}/{train_activity_loss:.6f}, "
                f"Val(total/param/act): {val_loss:.6f}/{val_param_loss:.6f}/{val_activity_loss:.6f}"
            )

            if val_loss < self.history['best_val_loss']:
                self.history['best_val_loss'] = val_loss
                self.history['best_val_param_loss'] = val_param_loss
                self.history['best_val_activity_loss'] = val_activity_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_param_loss': val_param_loss,
                    'val_activity_loss': val_activity_loss,
                    'model_config': self.model.get_model_config(),
                }, save_dir / 'controller_best.pt')
                print(f"  -> Best model saved (val_loss: {val_loss:.6f})")

            if epoch % save_every == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'model_config': self.model.get_model_config(),
                }, save_dir / f'controller_epoch_{epoch}.pt')

        torch.save({
            'epoch': num_epochs,
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'model_config': self.model.get_model_config(),
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
    use_activity_head: bool = False,
    activity_loss_weight: float = 0.0,
    inactive_param_weight: float = 1.0,
    param_loss_type: str = "mse",
    huber_delta: float = 0.05,
    effect_loss_weights: Optional[Dict[str, float]] = None,
    device: str = "cpu",
):
    """High-level function to train the AudioController."""
    db = InverseMappingDB(db_path)
    dataset = InverseMappingDataset(db)
    print(f"Loaded {len(dataset)} records from {db_path}")
    metadata = db.get_metadata()
    effect_names_raw = str(metadata.get('effect_names', '')).strip()
    effect_names = [x for x in effect_names_raw.split(',') if x]
    if effect_names:
        print(f"Effect names from DB: {effect_names}")

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
        use_activity_head=use_activity_head,
        num_effects=len(effect_names) if use_activity_head else 0,
    )

    trainer = ControllerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        effect_names=effect_names,
        inactive_param_weight=inactive_param_weight,
        activity_loss_weight=activity_loss_weight,
        param_loss_type=param_loss_type,
        huber_delta=huber_delta,
        effect_loss_weights=effect_loss_weights,
    )
    trainer.train(num_epochs=num_epochs, save_dir=output_dir)
