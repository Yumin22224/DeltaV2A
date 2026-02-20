"""
AR Hybrid Controller Trainer (Phase B)

Trains ARController using the inverse mapping database.

Loss at each AR step t:
  - effect_loss: CrossEntropy over (num_effects + 1) classes
  - param_loss:  Huber on predicted params vs ground-truth (active steps only)

total_loss = effect_loss_weight * effect_loss + param_loss_weight * param_loss
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .ar_model import ARController
from ..database.inverse_mapping import InverseMappingDB, InverseMappingDataset
from ..effects.pedalboard_effects import EFFECT_CATALOG


# ─────────────────────────────────────────────────────────────────────────────
# Dataset wrapper that exposes effect_order
# ─────────────────────────────────────────────────────────────────────────────

class ARInverseMappingDataset(torch.utils.data.Dataset):
    """
    Wraps InverseMappingDataset and exposes effect_order.
    Falls back to a dummy order (first active effect) if effect_order is absent.
    """

    def __init__(self, base: InverseMappingDataset, max_steps: int):
        self.base = base
        self.max_steps = max_steps
        self.has_order = (base.effect_order is not None)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.base[idx]
        if self.has_order:
            order = torch.from_numpy(
                self.base.effect_order[idx].astype(np.int64)
            )  # (max_steps,) with -1 for padding
        else:
            # Fallback: derive order from active mask (unordered)
            mask = item['effect_active_mask'].numpy()
            active_idxs = np.where(mask > 0.5)[0].tolist()
            order_arr = np.full(self.max_steps, -1, dtype=np.int64)
            for i, eff_i in enumerate(active_idxs[:self.max_steps]):
                order_arr[i] = eff_i
            order = torch.from_numpy(order_arr)
        item['effect_order'] = order
        return item


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class ARControllerTrainer:
    """Trainer for ARController."""

    def __init__(
        self,
        model: ARController,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cpu",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        effect_loss_weight: float = 1.0,
        param_loss_weight: float = 1.0,
        huber_delta: float = 0.02,
        lr_scheduler_type: str = "none",
        lr_min: float = 1e-6,
        output_dir: str = "outputs/pipeline/ar_controller",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.effect_loss_weight = effect_loss_weight
        self.param_loss_weight = param_loss_weight
        self.huber_delta = huber_delta
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_min = lr_min

    # ──────────────────────────────────────────────────────────────────────
    # Loss computation
    # ──────────────────────────────────────────────────────────────────────

    def _compute_loss(
        self,
        effect_logits_list: List[torch.Tensor],
        param_preds_list: List[Optional[torch.Tensor]],
        effect_order: torch.Tensor,   # (B, max_steps) int64
        normalized_params: torch.Tensor,  # (B, total_params)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined effect + param loss across all AR steps.

        Effect targets:
          step t < max_steps: effect_order[:, t] if >= 0, else stop_idx
          step max_steps:     always stop_idx

        Param targets (only for active steps where gt effect >= 0):
          ground-truth slice of normalized_params for the predicted effect.
        """
        B = effect_order.shape[0]
        device = effect_order.device
        max_steps = self.model.max_steps
        stop_idx = self.model.stop_idx

        total_effect_loss = torch.tensor(0.0, device=device)
        total_param_loss = torch.tensor(0.0, device=device)
        n_effect_steps = 0
        n_param_steps = 0

        for step, effect_logit in enumerate(effect_logits_list):
            # ── Effect target ──────────────────────────────────────────────
            if step < max_steps:
                gt_eff = effect_order[:, step].clone()   # -1 = STOP/pad
                # Samples with -1 should predict STOP
                gt_eff[gt_eff < 0] = stop_idx
            else:
                # Final step: everyone predicts STOP
                gt_eff = torch.full((B,), stop_idx, dtype=torch.long, device=device)

            effect_loss = F.cross_entropy(effect_logit, gt_eff)
            total_effect_loss = total_effect_loss + effect_loss
            n_effect_steps += 1

            # ── Param target (only for non-final steps with active effects) ─
            if step < max_steps and step < len(param_preds_list):
                param_pred = param_preds_list[step]   # (B, max_params)
                if param_pred is None:
                    continue

                raw_gt_eff = effect_order[:, step]    # original, may have -1
                active = (raw_gt_eff >= 0)            # samples with a real effect
                if not active.any():
                    continue

                # Gather ground-truth params for each active sample
                gt_param = torch.zeros_like(param_pred)
                for eff_i, sl in enumerate(self.model._effect_slices):
                    mask = (raw_gt_eff == eff_i)
                    if not mask.any():
                        continue
                    n = self.model.effect_param_counts[eff_i]
                    gt_param[mask, :n] = normalized_params[mask, sl]

                # Huber loss only on active samples and their valid param slots
                param_loss = torch.tensor(0.0, device=device)
                active_count = 0
                for eff_i, sl in enumerate(self.model._effect_slices):
                    mask = (raw_gt_eff == eff_i)
                    if not mask.any():
                        continue
                    n = self.model.effect_param_counts[eff_i]
                    param_loss = param_loss + F.huber_loss(
                        param_pred[mask, :n],
                        gt_param[mask, :n],
                        delta=self.huber_delta,
                        reduction='mean',
                    )
                    active_count += 1

                if active_count > 0:
                    total_param_loss = total_param_loss + param_loss / active_count
                    n_param_steps += 1

        effect_loss_mean = total_effect_loss / max(n_effect_steps, 1)
        param_loss_mean = total_param_loss / max(n_param_steps, 1)

        loss = (
            self.effect_loss_weight * effect_loss_mean
            + self.param_loss_weight * param_loss_mean
        )

        return loss, {
            "effect_loss": float(effect_loss_mean.item()),
            "param_loss": float(param_loss_mean.item()),
            "total_loss": float(loss.item()),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Epoch helpers
    # ──────────────────────────────────────────────────────────────────────

    def _run_epoch(
        self, loader: DataLoader, train: bool
    ) -> Dict[str, float]:
        self.model.train(train)
        totals: Dict[str, float] = {
            "total_loss": 0.0, "effect_loss": 0.0, "param_loss": 0.0
        }
        n_batches = 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                style_label = batch['style_label'].to(self.device)
                effect_order = batch['effect_order'].to(self.device)
                normalized_params = batch['normalized_params'].to(self.device)

                effect_logits, param_preds = self.model.forward_train(
                    style_label, effect_order, normalized_params
                )
                loss, metrics = self._compute_loss(
                    effect_logits, param_preds, effect_order, normalized_params
                )

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                for k, v in metrics.items():
                    totals[k] = totals.get(k, 0.0) + v
                n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    # ──────────────────────────────────────────────────────────────────────
    # Main training loop
    # ──────────────────────────────────────────────────────────────────────

    def train(self, num_epochs: int = 150) -> Dict:
        scheduler = None
        if self.lr_scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=num_epochs, eta_min=self.lr_min
            )

        best_val_loss = float("inf")
        best_epoch = 0
        training_log = []

        print(f"\n{'='*60}")
        print("AR CONTROLLER TRAINING")
        print(f"{'='*60}")
        print(f"  Epochs:   {num_epochs}")
        print(f"  LR:       {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Scheduler:{self.lr_scheduler_type}")
        print(f"  Max steps:{self.model.max_steps}")
        print(f"  Device:   {self.device}")
        print(f"  Output:   {self.output_dir}")

        for epoch in range(1, num_epochs + 1):
            train_metrics = self._run_epoch(self.train_loader, train=True)
            val_metrics = self._run_epoch(self.val_loader, train=False)

            if scheduler is not None:
                scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            val_loss = val_metrics["total_loss"]
            is_best = val_loss < best_val_loss

            if is_best:
                best_val_loss = val_loss
                best_epoch = epoch
                self._save_checkpoint("ar_controller_best.pt")

            log_entry = {
                "epoch": epoch,
                "lr": current_lr,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
            }
            training_log.append(log_entry)

            if epoch % 10 == 0 or epoch == 1 or is_best:
                marker = " ✓" if is_best else ""
                print(
                    f"Epoch {epoch:3d}/{num_epochs} | "
                    f"train_loss={train_metrics['total_loss']:.4f} "
                    f"(eff={train_metrics['effect_loss']:.4f} "
                    f"par={train_metrics['param_loss']:.4f}) | "
                    f"val_loss={val_loss:.4f} "
                    f"(eff={val_metrics['effect_loss']:.4f} "
                    f"par={val_metrics['param_loss']:.4f}){marker}"
                )

        # Save final checkpoint and log
        self._save_checkpoint("ar_controller_last.pt")
        log_path = self.output_dir / "ar_training_log.json"
        log_path.write_text(json.dumps(training_log, indent=2))

        print(f"\nTraining done. Best val_loss={best_val_loss:.4f} at epoch {best_epoch}.")
        return {
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "training_log": training_log,
        }

    def _save_checkpoint(self, filename: str):
        path = self.output_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_config": self.model.get_model_config(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def train_ar_controller(
    db_path: str,
    effect_names: List[str],
    output_dir: str = "outputs/pipeline/ar_controller",
    # Model
    style_vocab_size: int = 24,
    condition_dim: int = 128,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    max_steps: int = 2,
    # Training
    num_epochs: int = 150,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-3,
    val_split: float = 0.2,
    seed: int = 42,
    device: str = "cpu",
    # Loss
    effect_loss_weight: float = 1.0,
    param_loss_weight: float = 1.0,
    huber_delta: float = 0.02,
    # LR schedule
    lr_scheduler_type: str = "cosine",
    lr_min: float = 1e-6,
) -> Dict:
    """Build model, dataloaders, and run AR controller training."""

    torch.manual_seed(seed)

    # ── Load dataset ───────────────────────────────────────────────────────
    db = InverseMappingDB(db_path)
    base_ds = InverseMappingDataset(db)
    ar_ds = ARInverseMappingDataset(base_ds, max_steps=max_steps)

    n_val = max(1, int(len(ar_ds) * val_split))
    n_train = len(ar_ds) - n_val
    train_ds, val_ds = random_split(
        ar_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)

    print(f"AR Dataset: {len(ar_ds)} records  "
          f"(train={n_train}, val={n_val})")
    print(f"effect_order available: {ar_ds.has_order}")

    # ── Model ──────────────────────────────────────────────────────────────
    model = ARController(
        effect_names=effect_names,
        style_vocab_size=style_vocab_size,
        condition_dim=condition_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        max_steps=max_steps,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ARController parameters: {total_params:,}")

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = ARControllerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        effect_loss_weight=effect_loss_weight,
        param_loss_weight=param_loss_weight,
        huber_delta=huber_delta,
        lr_scheduler_type=lr_scheduler_type,
        lr_min=lr_min,
        output_dir=output_dir,
    )

    return trainer.train(num_epochs=num_epochs)
