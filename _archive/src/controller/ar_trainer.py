"""
AR Hybrid Controller Trainer (Phase B)

Trains ARController using the inverse mapping database.

Loss at each AR step t:
  - effect_loss: CrossEntropy over (num_effects + 1) classes
  - param_loss:  Huber on predicted params vs ground-truth (active steps only)

total_loss = effect_loss_weight * effect_loss + param_loss_weight * param_loss
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from .ar_model import ARController
from ..database.inverse_mapping import InverseMappingDB, InverseMappingDataset


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
            mask = item["effect_active_mask"].numpy()
            active_idxs = np.where(mask > 0.5)[0].tolist()
            order_arr = np.full(self.max_steps, -1, dtype=np.int64)
            for i, eff_i in enumerate(active_idxs[: self.max_steps]):
                order_arr[i] = eff_i
            order = torch.from_numpy(order_arr)
        item["effect_order"] = order
        return item


def _compute_effect_class_weights_from_subset(
    ar_ds: ARInverseMappingDataset,
    subset_indices: List[int],
    stop_idx: int,
    max_steps: int,
    num_classes: int,
    power: float = 0.5,
    min_weight: float = 0.5,
    max_weight: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-class CE weights from train subset frequency.
    """
    idx = np.asarray(subset_indices, dtype=np.int64)
    if idx.size == 0:
        return np.ones((num_classes,), dtype=np.float32), np.zeros((num_classes,), dtype=np.int64)

    if ar_ds.has_order and ar_ds.base.effect_order is not None:
        order = ar_ds.base.effect_order[idx].astype(np.int64)
    else:
        masks = ar_ds.base.effect_active_mask[idx]
        order = np.full((len(idx), max_steps), -1, dtype=np.int64)
        for r, mask in enumerate(masks):
            active = np.where(mask > 0.5)[0][:max_steps]
            if active.size > 0:
                order[r, : active.size] = active

    order = order[:, :max_steps]
    gt = order.copy()
    gt[gt < 0] = stop_idx
    counts = np.bincount(gt.reshape(-1), minlength=num_classes).astype(np.float64)
    counts[stop_idx] += float(len(idx))  # explicit final STOP step

    p = float(max(power, 0.0))
    if p <= 1e-8:
        weights = np.ones_like(counts, dtype=np.float64)
    else:
        inv = np.power(np.maximum(counts, 1.0), -p)
        weights = inv / max(float(inv.mean()), 1e-12)
        weights = np.clip(weights, float(min_weight), float(max_weight))
        weights = weights / max(float(weights.mean()), 1e-12)

    return weights.astype(np.float32), counts.astype(np.int64)


def _build_param_effect_weights(
    effect_names: List[str],
    param_effect_weights: Optional[Dict[str, float]],
) -> np.ndarray:
    arr = np.ones((len(effect_names),), dtype=np.float32)
    if not isinstance(param_effect_weights, dict):
        return arr
    for i, name in enumerate(effect_names):
        try:
            arr[i] = float(param_effect_weights.get(name, 1.0))
        except (TypeError, ValueError):
            arr[i] = 1.0
    return np.maximum(arr, 1e-6).astype(np.float32)


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
        effect_ce_class_weights: Optional[np.ndarray] = None,
        effect_ce_label_smoothing: float = 0.0,
        param_effect_weights: Optional[np.ndarray] = None,
        lr_scheduler_type: str = "none",
        lr_min: float = 1e-6,
        confidence_weighting_enabled: bool = False,
        confidence_weight_power: float = 1.0,
        confidence_min_weight: float = 0.2,
        confidence_use_delta_norm: bool = False,
        confidence_style_alpha: float = 1.0,
        confidence_delta_scale: float = 1.0,
        output_dir: str = "outputs/pipeline/ar_controller",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.effect_loss_weight = effect_loss_weight
        self.param_loss_weight = param_loss_weight
        self.huber_delta = huber_delta
        self.effect_ce_label_smoothing = float(max(effect_ce_label_smoothing, 0.0))
        self.effect_ce_class_weights = (
            torch.as_tensor(effect_ce_class_weights, dtype=torch.float32, device=device)
            if effect_ce_class_weights is not None
            else None
        )
        self.param_effect_weights = (
            torch.as_tensor(param_effect_weights, dtype=torch.float32, device=device)
            if param_effect_weights is not None
            else None
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_min = lr_min
        self.confidence_weighting_enabled = bool(confidence_weighting_enabled)
        self.confidence_weight_power = float(max(confidence_weight_power, 0.0))
        self.confidence_min_weight = float(np.clip(confidence_min_weight, 0.0, 1.0))
        self.confidence_use_delta_norm = bool(confidence_use_delta_norm)
        self.confidence_style_alpha = float(np.clip(confidence_style_alpha, 0.0, 1.0))
        self.confidence_delta_scale = float(max(confidence_delta_scale, 1e-6))

    def _compute_confidence_weights(
        self,
        style_label: torch.Tensor,
        clap_delta_norm: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Compute per-sample confidence weights from style-label entropy.
        weight_i in [confidence_min_weight, 1.0].
        """
        if not self.confidence_weighting_enabled:
            return None
        if style_label.ndim != 2 or style_label.shape[1] <= 1:
            return None
        eps = 1e-8
        p = torch.clamp(style_label, min=eps)
        entropy = -torch.sum(p * torch.log(p), dim=1)
        max_entropy = max(math.log(float(style_label.shape[1])), eps)
        norm_entropy = torch.clamp(entropy / max_entropy, min=0.0, max=1.0)
        style_confidence = 1.0 - norm_entropy
        confidence = style_confidence
        if self.confidence_use_delta_norm and clap_delta_norm is not None:
            delta = torch.clamp(clap_delta_norm.view(-1), min=0.0)
            delta_confidence = delta / (delta + self.confidence_delta_scale)
            a = self.confidence_style_alpha
            confidence = (a * style_confidence) + ((1.0 - a) * delta_confidence)
        if self.confidence_weight_power > 1e-8 and self.confidence_weight_power != 1.0:
            confidence = torch.pow(confidence, self.confidence_weight_power)
        min_w = self.confidence_min_weight
        weights = min_w + (1.0 - min_w) * confidence
        return torch.clamp(weights, min=min_w, max=1.0)

    def _compute_loss(
        self,
        effect_logits_list: List[torch.Tensor],
        param_preds_list: List[Optional[torch.Tensor]],
        effect_order: torch.Tensor,  # (B, max_steps)
        normalized_params: torch.Tensor,  # (B, total_params)
        sample_weights: Optional[torch.Tensor] = None,  # (B,)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined effect + param loss across all AR steps.
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
            if step < max_steps:
                gt_eff = effect_order[:, step].clone()
                gt_eff[gt_eff < 0] = stop_idx
            else:
                gt_eff = torch.full((B,), stop_idx, dtype=torch.long, device=device)

            effect_loss_vec = F.cross_entropy(
                effect_logit,
                gt_eff,
                weight=self.effect_ce_class_weights,
                label_smoothing=self.effect_ce_label_smoothing,
                reduction="none",
            )
            if sample_weights is not None:
                effect_loss = torch.sum(effect_loss_vec * sample_weights) / torch.clamp(sample_weights.sum(), min=1e-8)
            else:
                effect_loss = effect_loss_vec.mean()
            total_effect_loss = total_effect_loss + effect_loss
            n_effect_steps += 1

            if step < max_steps and step < len(param_preds_list):
                param_pred = param_preds_list[step]
                if param_pred is None:
                    continue

                raw_gt_eff = effect_order[:, step]
                active = (raw_gt_eff >= 0)
                if not active.any():
                    continue

                gt_param = torch.zeros_like(param_pred)
                for eff_i, sl in enumerate(self.model._effect_slices):
                    mask = (raw_gt_eff == eff_i)
                    if not mask.any():
                        continue
                    n = self.model.effect_param_counts[eff_i]
                    gt_param[mask, :n] = normalized_params[mask, sl]

                param_loss = torch.tensor(0.0, device=device)
                active_weight_sum = 0.0
                for eff_i, sl in enumerate(self.model._effect_slices):
                    mask = (raw_gt_eff == eff_i)
                    if not mask.any():
                        continue
                    n = self.model.effect_param_counts[eff_i]
                    eff_weight = (
                        float(self.param_effect_weights[eff_i].item())
                        if self.param_effect_weights is not None
                        else 1.0
                    )
                    huber_elem = F.huber_loss(
                        param_pred[mask, :n],
                        gt_param[mask, :n],
                        delta=self.huber_delta,
                        reduction="none",
                    )
                    per_sample_loss = huber_elem.mean(dim=1)
                    if sample_weights is not None:
                        sw_eff = sample_weights[mask]
                        eff_loss = torch.sum(per_sample_loss * sw_eff) / torch.clamp(sw_eff.sum(), min=1e-8)
                    else:
                        eff_loss = per_sample_loss.mean()
                    param_loss = param_loss + eff_loss * eff_weight
                    active_weight_sum += eff_weight

                if active_weight_sum > 0.0:
                    total_param_loss = total_param_loss + param_loss / active_weight_sum
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

    def _run_epoch(self, loader: DataLoader, train: bool) -> Dict[str, float]:
        self.model.train(train)
        totals: Dict[str, float] = {
            "total_loss": 0.0,
            "effect_loss": 0.0,
            "param_loss": 0.0,
        }
        n_batches = 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                style_label = batch["style_label"].to(self.device)
                effect_order = batch["effect_order"].to(self.device)
                normalized_params = batch["normalized_params"].to(self.device)
                clap_delta_norm = batch.get("clap_delta_norm")
                if clap_delta_norm is not None:
                    clap_delta_norm = clap_delta_norm.to(self.device)
                clap_embedding = batch.get("clap_embedding")
                if clap_embedding is not None and self.model.clap_embed_dim > 0:
                    clap_embedding = clap_embedding.to(self.device)
                else:
                    clap_embedding = None

                effect_logits, param_preds = self.model.forward_train(
                    style_label, effect_order, normalized_params, clap_embedding
                )
                sample_weights = self._compute_confidence_weights(
                    style_label,
                    clap_delta_norm=clap_delta_norm,
                )
                loss, metrics = self._compute_loss(
                    effect_logits, param_preds, effect_order, normalized_params, sample_weights=sample_weights
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
        print(f"  CE label smoothing: {self.effect_ce_label_smoothing:.4f}")
        print(
            f"  Confidence weighting: enabled={self.confidence_weighting_enabled} "
            f"(power={self.confidence_weight_power:.3f}, min={self.confidence_min_weight:.3f})"
        )
        if self.confidence_weighting_enabled:
            print(
                f"    use_delta_norm={self.confidence_use_delta_norm} "
                f"(style_alpha={self.confidence_style_alpha:.3f}, delta_scale={self.confidence_delta_scale:.3f})"
            )
        if self.effect_ce_class_weights is not None:
            print(f"  CE class weights: {self.effect_ce_class_weights.detach().cpu().numpy().tolist()}")
        if self.param_effect_weights is not None:
            print(f"  Param effect weights: {self.param_effect_weights.detach().cpu().numpy().tolist()}")

        for epoch in range(1, num_epochs + 1):
            train_metrics = self._run_epoch(self.train_loader, train=True)
            val_metrics = self._run_epoch(self.val_loader, train=False)

            if scheduler is not None:
                scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
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
                marker = " *" if is_best else ""
                print(
                    f"Epoch {epoch:3d}/{num_epochs} | "
                    f"train_loss={train_metrics['total_loss']:.4f} "
                    f"(eff={train_metrics['effect_loss']:.4f} "
                    f"par={train_metrics['param_loss']:.4f}) | "
                    f"val_loss={val_loss:.4f} "
                    f"(eff={val_metrics['effect_loss']:.4f} "
                    f"par={val_metrics['param_loss']:.4f}){marker}"
                )

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
    clap_embed_dim: int = 0,
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
    auto_effect_class_weights: bool = False,
    effect_class_weight_power: float = 0.5,
    effect_class_weight_min: float = 0.5,
    effect_class_weight_max: float = 3.0,
    effect_ce_label_smoothing: float = 0.0,
    param_effect_weights: Optional[Dict[str, float]] = None,
    confidence_weighting_enabled: bool = False,
    confidence_weight_power: float = 1.0,
    confidence_min_weight: float = 0.2,
    confidence_use_delta_norm: bool = False,
    confidence_style_alpha: float = 1.0,
    confidence_delta_scale: float = 1.0,
    # LR schedule
    lr_scheduler_type: str = "cosine",
    lr_min: float = 1e-6,
) -> Dict:
    """Build model, dataloaders, and run AR controller training."""

    torch.manual_seed(seed)

    db = InverseMappingDB(db_path)
    base_ds = InverseMappingDataset(db)
    ar_ds = ARInverseMappingDataset(base_ds, max_steps=max_steps)

    n_val = max(1, int(len(ar_ds) * val_split))
    n_train = len(ar_ds) - n_val
    train_ds, val_ds = random_split(
        ar_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    print(f"AR Dataset: {len(ar_ds)} records  (train={n_train}, val={n_val})")
    print(f"effect_order available: {ar_ds.has_order}")

    effect_ce_class_weights = None
    if bool(auto_effect_class_weights):
        effect_ce_class_weights, effect_class_counts = _compute_effect_class_weights_from_subset(
            ar_ds=ar_ds,
            subset_indices=list(train_ds.indices),
            stop_idx=len(effect_names),
            max_steps=max_steps,
            num_classes=len(effect_names) + 1,
            power=float(effect_class_weight_power),
            min_weight=float(effect_class_weight_min),
            max_weight=float(effect_class_weight_max),
        )
        print(f"AR effect class counts (train, incl. final STOP): {effect_class_counts.tolist()}")
        print(f"AR effect CE class weights: {effect_ce_class_weights.tolist()}")

    param_effect_weight_arr = _build_param_effect_weights(effect_names, param_effect_weights)
    print(f"AR param effect weights: {param_effect_weight_arr.tolist()}")

    model = ARController(
        effect_names=effect_names,
        style_vocab_size=style_vocab_size,
        condition_dim=condition_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        max_steps=max_steps,
        clap_embed_dim=clap_embed_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ARController parameters: {total_params:,}")

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
        effect_ce_class_weights=effect_ce_class_weights,
        effect_ce_label_smoothing=effect_ce_label_smoothing,
        param_effect_weights=param_effect_weight_arr,
        lr_scheduler_type=lr_scheduler_type,
        lr_min=lr_min,
        confidence_weighting_enabled=confidence_weighting_enabled,
        confidence_weight_power=confidence_weight_power,
        confidence_min_weight=confidence_min_weight,
        confidence_use_delta_norm=confidence_use_delta_norm,
        confidence_style_alpha=confidence_style_alpha,
        confidence_delta_scale=confidence_delta_scale,
        output_dir=output_dir,
    )

    return trainer.train(num_epochs=num_epochs)
