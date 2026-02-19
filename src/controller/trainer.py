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
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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
        activity_mismatch_weight: float = 0.0,
        activity_mismatch_gamma: float = 2.0,
        activity_loss_type: str = "bce",
        focal_gamma: float = 2.0,
        asl_gamma_pos: float = 0.0,
        asl_gamma_neg: float = 4.0,
        asl_clip: float = 0.05,
        param_loss_weight: float = 1.0,
        param_loss_type: str = "mse",
        huber_delta: float = 0.05,
        effect_loss_weights: Optional[Dict[str, float]] = None,
        selection_metric: str = "val_param_loss",
        stage_name: Optional[str] = None,
        train_backbone: bool = True,
        train_param_head: bool = True,
        train_activity_head: bool = True,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.effect_names = effect_names or []
        self.inactive_param_weight = float(inactive_param_weight)
        self.activity_loss_weight = float(activity_loss_weight)
        self.activity_mismatch_weight = float(activity_mismatch_weight)
        self.activity_mismatch_gamma = float(activity_mismatch_gamma)
        self.activity_loss_type = str(activity_loss_type).lower()
        self.focal_gamma = float(focal_gamma)
        self.asl_gamma_pos = float(asl_gamma_pos)
        self.asl_gamma_neg = float(asl_gamma_neg)
        self.asl_clip = float(asl_clip)
        self.param_loss_weight = float(param_loss_weight)
        if self.activity_mismatch_weight < 0.0:
            raise ValueError("activity_mismatch_weight must be >= 0.0")
        if self.activity_mismatch_gamma < 0.0:
            raise ValueError("activity_mismatch_gamma must be >= 0.0")
        self.param_loss_type = str(param_loss_type).lower()
        if self.param_loss_type not in {"mse", "huber"}:
            raise ValueError(f"Unsupported param_loss_type: {param_loss_type}")
        if self.activity_loss_type not in {"bce", "focal", "asl"}:
            raise ValueError(
                f"Unsupported activity_loss_type: {activity_loss_type}. "
                "Expected one of ['bce', 'focal', 'asl']"
            )
        self.huber_delta = float(huber_delta)
        self.effect_loss_weights = effect_loss_weights or {}
        self.selection_metric = str(selection_metric)
        self.stage_name = stage_name or "stage"
        valid_selection_metrics = {
            "val_loss",
            "val_param_loss",
            "val_active_param_rmse",
            "val_activity_macro_f1",
            "val_activity_micro_f1",
        }
        if self.selection_metric not in valid_selection_metrics:
            raise ValueError(
                f"Unsupported selection_metric: {self.selection_metric}. "
                f"Expected one of {sorted(valid_selection_metrics)}"
            )
        self.effect_param_slices = self._build_effect_param_slices(self.effect_names)
        self.param_weight_vector = self._build_param_weight_vector(
            self.effect_names, self.effect_loss_weights
        ).to(device)
        self.train_backbone = bool(train_backbone)
        self.train_param_head = bool(train_param_head)
        self.train_activity_head = bool(train_activity_head)
        self._configure_trainable_parts()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError(
                "No trainable parameters after applying train_* flags. "
                "Set at least one of train_backbone/train_param_head/train_activity_head to true."
            )
        self.optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_param_loss': [],
            'val_param_loss': [],
            'train_activity_loss': [],
            'val_activity_loss': [],
            'train_activity_macro_f1': [],
            'val_activity_macro_f1': [],
            'train_activity_micro_f1': [],
            'val_activity_micro_f1': [],
            'train_active_param_rmse': [],
            'val_active_param_rmse': [],
            'best_val_loss': float('inf'),
            'best_val_param_loss': float('inf'),
            'best_val_activity_loss': float('inf'),
            'best_val_activity_macro_f1': 0.0,
            'best_val_activity_micro_f1': 0.0,
            'best_val_active_param_rmse': float('inf'),
            'selection_metric': self.selection_metric,
            'best_selection_metric': float('-inf') if self._selection_mode_maximize() else float('inf'),
        }

    def _selection_mode_maximize(self) -> bool:
        return self.selection_metric in {"val_activity_macro_f1", "val_activity_micro_f1"}

    def _set_module_trainable(self, module: Optional[nn.Module], trainable: bool):
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = bool(trainable)

    def _configure_trainable_parts(self):
        self._set_module_trainable(self.model.backbone, self.train_backbone)
        self._set_module_trainable(self.model.param_head, self.train_param_head)
        self._set_module_trainable(self.model.activity_head, self.train_activity_head)

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

    def _compute_activity_loss(
        self,
        activity_logits: torch.Tensor,
        effect_active_mask: torch.Tensor,
    ) -> torch.Tensor:
        eps = 1e-8
        if self.activity_loss_type == "focal":
            bce = F.binary_cross_entropy_with_logits(
                activity_logits,
                effect_active_mask,
                reduction='none',
            )
            probs = torch.sigmoid(activity_logits)
            pt = (effect_active_mask * probs) + ((1.0 - effect_active_mask) * (1.0 - probs))
            loss = bce * torch.pow(torch.clamp(1.0 - pt, min=0.0), self.focal_gamma)
        elif self.activity_loss_type == "asl":
            probs = torch.sigmoid(activity_logits)
            prob_pos = torch.clamp(probs, min=eps, max=1.0 - eps)
            prob_neg = 1.0 - probs
            if self.asl_clip > 0.0:
                prob_neg = torch.clamp(prob_neg + self.asl_clip, max=1.0)

            pos_loss = effect_active_mask * torch.log(prob_pos) * torch.pow(
                torch.clamp(1.0 - prob_pos, min=0.0), self.asl_gamma_pos
            )
            neg_loss = (1.0 - effect_active_mask) * torch.log(torch.clamp(prob_neg, min=eps)) * torch.pow(
                torch.clamp(1.0 - prob_neg, min=0.0), self.asl_gamma_neg
            )
            loss = -(pos_loss + neg_loss)
        else:
            loss = F.binary_cross_entropy_with_logits(
                activity_logits,
                effect_active_mask,
                reduction='none',
            )

        if self.activity_mismatch_weight > 0.0:
            probs = torch.sigmoid(activity_logits)
            pt = (effect_active_mask * probs) + ((1.0 - effect_active_mask) * (1.0 - probs))
            mismatch = torch.pow(torch.clamp(1.0 - pt, min=0.0), self.activity_mismatch_gamma)
            loss = loss * (1.0 + self.activity_mismatch_weight * mismatch)

        return loss.mean()

    @staticmethod
    def _accumulate_activity_counts(
        activity_logits: torch.Tensor,
        effect_active_mask: torch.Tensor,
        tp: torch.Tensor,
        fp: torch.Tensor,
        fn: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred = (torch.sigmoid(activity_logits) >= threshold).float()
        tgt = (effect_active_mask >= 0.5).float()
        tp = tp + (pred * tgt).sum(dim=0)
        fp = fp + (pred * (1.0 - tgt)).sum(dim=0)
        fn = fn + ((1.0 - pred) * tgt).sum(dim=0)
        return tp, fp, fn

    @staticmethod
    def _activity_f1_from_counts(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> Tuple[float, float]:
        eps = 1e-8
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1_per_effect = (2.0 * precision * recall) / (precision + recall + eps)
        macro_f1 = float(f1_per_effect.mean().item())

        tp_sum = float(tp.sum().item())
        fp_sum = float(fp.sum().item())
        fn_sum = float(fn.sum().item())
        micro_precision = tp_sum / (tp_sum + fp_sum + eps)
        micro_recall = tp_sum / (tp_sum + fn_sum + eps)
        micro_f1 = float((2.0 * micro_precision * micro_recall) / (micro_precision + micro_recall + eps))
        return macro_f1, micro_f1

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
            activity_loss = self._compute_activity_loss(activity_logits, effect_active_mask)

        total_loss = (self.param_loss_weight * param_loss) + (self.activity_loss_weight * activity_loss)
        return total_loss, param_loss, activity_loss

    def _compute_active_param_rmse(
        self,
        params_pred: torch.Tensor,
        params_gt: torch.Tensor,
        effect_active_mask: Optional[torch.Tensor],
    ):
        """RMSE computed only on parameters whose effects are active."""
        if effect_active_mask is None or effect_active_mask.numel() == 0:
            return torch.zeros((), device=params_gt.device), False
        if not self.effect_param_slices:
            return torch.zeros((), device=params_gt.device), False

        param_active_mask = self._expand_effect_mask(effect_active_mask).float()
        denom = torch.clamp(param_active_mask.sum(), min=1e-8)
        if float(denom.item()) <= 1e-7:
            return torch.zeros((), device=params_gt.device), False

        mse = torch.square(params_pred - params_gt) * param_active_mask
        rmse = torch.sqrt(torch.clamp(mse.sum() / denom, min=0.0))
        return rmse, True

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_param_loss = 0.0
        total_activity_loss = 0.0
        total_active_param_rmse = 0.0
        has_activity_stats = False
        tp = None
        fp = None
        fn = None
        n = 0
        n_active = 0
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
            active_param_rmse, has_active = self._compute_active_param_rmse(
                params_pred=params_pred,
                params_gt=params_gt,
                effect_active_mask=effect_active_mask,
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_param_loss += param_loss.item()
            total_activity_loss += activity_loss.item()
            if has_active:
                total_active_param_rmse += active_param_rmse.item()
                n_active += 1
            if activity_logits is not None and effect_active_mask is not None:
                if tp is None:
                    tp = torch.zeros(effect_active_mask.shape[1], device=self.device)
                    fp = torch.zeros(effect_active_mask.shape[1], device=self.device)
                    fn = torch.zeros(effect_active_mask.shape[1], device=self.device)
                tp, fp, fn = self._accumulate_activity_counts(
                    activity_logits=activity_logits,
                    effect_active_mask=effect_active_mask,
                    tp=tp,
                    fp=fp,
                    fn=fn,
                    threshold=0.5,
                )
                has_activity_stats = True
            n += 1
        denom = max(n, 1)
        active_denom = max(n_active, 1)
        train_activity_macro_f1 = 0.0
        train_activity_micro_f1 = 0.0
        if has_activity_stats and tp is not None and fp is not None and fn is not None:
            train_activity_macro_f1, train_activity_micro_f1 = self._activity_f1_from_counts(tp, fp, fn)
        return (
            total_loss / denom,
            total_param_loss / denom,
            total_activity_loss / denom,
            total_active_param_rmse / active_denom,
            train_activity_macro_f1,
            train_activity_micro_f1,
        )

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        total_param_loss = 0.0
        total_activity_loss = 0.0
        total_active_param_rmse = 0.0
        has_activity_stats = False
        tp = None
        fp = None
        fn = None
        n = 0
        n_active = 0
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
            active_param_rmse, has_active = self._compute_active_param_rmse(
                params_pred=params_pred,
                params_gt=params_gt,
                effect_active_mask=effect_active_mask,
            )
            total_loss += loss.item()
            total_param_loss += param_loss.item()
            total_activity_loss += activity_loss.item()
            if has_active:
                total_active_param_rmse += active_param_rmse.item()
                n_active += 1
            if activity_logits is not None and effect_active_mask is not None:
                if tp is None:
                    tp = torch.zeros(effect_active_mask.shape[1], device=self.device)
                    fp = torch.zeros(effect_active_mask.shape[1], device=self.device)
                    fn = torch.zeros(effect_active_mask.shape[1], device=self.device)
                tp, fp, fn = self._accumulate_activity_counts(
                    activity_logits=activity_logits,
                    effect_active_mask=effect_active_mask,
                    tp=tp,
                    fp=fp,
                    fn=fn,
                    threshold=0.5,
                )
                has_activity_stats = True
            n += 1
        denom = max(n, 1)
        active_denom = max(n_active, 1)
        val_activity_macro_f1 = 0.0
        val_activity_micro_f1 = 0.0
        if has_activity_stats and tp is not None and fp is not None and fn is not None:
            val_activity_macro_f1, val_activity_micro_f1 = self._activity_f1_from_counts(tp, fp, fn)
        return (
            total_loss / denom,
            total_param_loss / denom,
            total_activity_loss / denom,
            total_active_param_rmse / active_denom,
            val_activity_macro_f1,
            val_activity_micro_f1,
        )

    @torch.no_grad()
    def tune_activity_thresholds(
        self,
        num_thresholds: int = 37,
        threshold_min: float = 0.05,
        threshold_max: float = 0.95,
    ) -> Optional[Dict[str, Any]]:
        if self.model.activity_head is None or not self.effect_names:
            return None

        logits_list: List[np.ndarray] = []
        target_list: List[np.ndarray] = []
        self.model.eval()
        for batch in self.val_loader:
            clap_emb = batch['clap_embedding'].to(self.device)
            style = batch['style_label'].to(self.device)
            effect_active_mask = batch.get('effect_active_mask')
            if effect_active_mask is None:
                continue
            effect_active_mask = effect_active_mask.to(self.device)
            _, activity_logits = self.model.forward_with_activity(clap_emb, style)
            if activity_logits is None:
                continue
            logits_list.append(activity_logits.detach().cpu().numpy())
            target_list.append(effect_active_mask.detach().cpu().numpy())

        if not logits_list:
            return None

        logits = np.concatenate(logits_list, axis=0)
        targets = np.concatenate(target_list, axis=0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        thresholds_grid = np.linspace(threshold_min, threshold_max, num=max(int(num_thresholds), 2))

        def _f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            tp = float(np.sum((y_true == 1) & (y_pred == 1)))
            fp = float(np.sum((y_true == 0) & (y_pred == 1)))
            fn = float(np.sum((y_true == 1) & (y_pred == 0)))
            precision = tp / max(tp + fp, 1e-8)
            recall = tp / max(tp + fn, 1e-8)
            return float((2.0 * precision * recall) / max(precision + recall, 1e-8))

        tuned_thresholds: List[float] = []
        per_effect: List[Dict[str, Any]] = []
        for i, effect_name in enumerate(self.effect_names):
            y_true = (targets[:, i] >= 0.5).astype(np.int32)
            best_thr = 0.5
            best_f1 = -1.0
            for thr in thresholds_grid:
                y_pred = (probs[:, i] >= float(thr)).astype(np.int32)
                f1 = _f1_score(y_true, y_pred)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thr = float(thr)
            tuned_thresholds.append(best_thr)
            f1_default = _f1_score(y_true, (probs[:, i] >= 0.5).astype(np.int32))
            per_effect.append(
                {
                    "effect": effect_name,
                    "threshold": best_thr,
                    "f1_at_best_threshold": float(best_f1),
                    "f1_at_0_5": float(f1_default),
                }
            )

        tuned_pred = (probs >= np.array(tuned_thresholds, dtype=np.float32)[None, :]).astype(np.int32)
        default_pred = (probs >= 0.5).astype(np.int32)
        y_true = (targets >= 0.5).astype(np.int32)

        macro_tuned = float(np.mean([_f1_score(y_true[:, i], tuned_pred[:, i]) for i in range(y_true.shape[1])]))
        macro_default = float(np.mean([_f1_score(y_true[:, i], default_pred[:, i]) for i in range(y_true.shape[1])]))
        micro_tuned = _f1_score(y_true.reshape(-1), tuned_pred.reshape(-1))
        micro_default = _f1_score(y_true.reshape(-1), default_pred.reshape(-1))

        return {
            "num_samples": int(y_true.shape[0]),
            "threshold_search": {
                "min": float(threshold_min),
                "max": float(threshold_max),
                "count": int(max(int(num_thresholds), 2)),
            },
            "macro_f1_default_0_5": macro_default,
            "macro_f1_tuned": macro_tuned,
            "micro_f1_default_0_5": micro_default,
            "micro_f1_tuned": micro_tuned,
            "per_effect": per_effect,
            "thresholds": {name: float(thr) for name, thr in zip(self.effect_names, tuned_thresholds)},
        }

    def train(self, num_epochs: int, save_dir: str, save_every: int = 10):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nTraining AudioController for {num_epochs} epochs... [{self.stage_name}]")
        print(f"  Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        print(f"  Params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Selection metric: {self.selection_metric}")
        print(
            f"  Loss weights: param={self.param_loss_weight}, activity={self.activity_loss_weight} | "
            f"Activity loss type={self.activity_loss_type} | "
            f"Activity mismatch weight={self.activity_mismatch_weight}, gamma={self.activity_mismatch_gamma} | "
            f"Trainable: backbone={self.train_backbone}, "
            f"param_head={self.train_param_head}, activity_head={self.train_activity_head}"
        )

        def _select_metric(
            val_loss: float,
            val_param_loss: float,
            val_active_param_rmse: float,
            val_activity_macro_f1: float,
            val_activity_micro_f1: float,
        ) -> float:
            if self.selection_metric == "val_loss":
                return float(val_loss)
            if self.selection_metric == "val_active_param_rmse":
                return float(val_active_param_rmse)
            if self.selection_metric == "val_activity_macro_f1":
                return float(val_activity_macro_f1)
            if self.selection_metric == "val_activity_micro_f1":
                return float(val_activity_micro_f1)
            return float(val_param_loss)

        def _is_better(current: float, best: float) -> bool:
            if self._selection_mode_maximize():
                return current > best
            return current < best

        best_activity_ckpt_macro_f1 = float("-inf")

        for epoch in range(1, num_epochs + 1):
            (
                train_loss,
                train_param_loss,
                train_activity_loss,
                train_active_param_rmse,
                train_activity_macro_f1,
                train_activity_micro_f1,
            ) = self.train_epoch()
            (
                val_loss,
                val_param_loss,
                val_activity_loss,
                val_active_param_rmse,
                val_activity_macro_f1,
                val_activity_micro_f1,
            ) = self.validate()
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_param_loss'].append(train_param_loss)
            self.history['val_param_loss'].append(val_param_loss)
            self.history['train_activity_loss'].append(train_activity_loss)
            self.history['val_activity_loss'].append(val_activity_loss)
            self.history['train_activity_macro_f1'].append(train_activity_macro_f1)
            self.history['val_activity_macro_f1'].append(val_activity_macro_f1)
            self.history['train_activity_micro_f1'].append(train_activity_micro_f1)
            self.history['val_activity_micro_f1'].append(val_activity_micro_f1)
            self.history['train_active_param_rmse'].append(train_active_param_rmse)
            self.history['val_active_param_rmse'].append(val_active_param_rmse)

            print(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train(total/param/act/active_rmse): "
                f"{train_loss:.6f}/{train_param_loss:.6f}/{train_activity_loss:.6f}/{train_active_param_rmse:.6f}, "
                f"Val(total/param/act/active_rmse): "
                f"{val_loss:.6f}/{val_param_loss:.6f}/{val_activity_loss:.6f}/{val_active_param_rmse:.6f}, "
                f"Val(act_macro_f1/act_micro_f1): {val_activity_macro_f1:.4f}/{val_activity_micro_f1:.4f}"
            )

            select_val = _select_metric(
                val_loss=val_loss,
                val_param_loss=val_param_loss,
                val_active_param_rmse=val_active_param_rmse,
                val_activity_macro_f1=val_activity_macro_f1,
                val_activity_micro_f1=val_activity_micro_f1,
            )
            if _is_better(select_val, self.history['best_selection_metric']):
                self.history['best_val_loss'] = val_loss
                self.history['best_val_param_loss'] = val_param_loss
                self.history['best_val_activity_loss'] = val_activity_loss
                self.history['best_val_activity_macro_f1'] = val_activity_macro_f1
                self.history['best_val_activity_micro_f1'] = val_activity_micro_f1
                self.history['best_val_active_param_rmse'] = val_active_param_rmse
                self.history['best_selection_metric'] = select_val
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_param_loss': val_param_loss,
                    'val_activity_loss': val_activity_loss,
                    'val_activity_macro_f1': val_activity_macro_f1,
                    'val_activity_micro_f1': val_activity_micro_f1,
                    'val_active_param_rmse': val_active_param_rmse,
                    'selection_metric': self.selection_metric,
                    'selection_metric_value': select_val,
                    'model_config': self.model.get_model_config(),
                }, save_dir / 'controller_best.pt')
                print(
                    f"  -> Best model saved ({self.selection_metric}: {select_val:.6f}, "
                    f"val_loss: {val_loss:.6f})"
                )

            # Additional activity-focused checkpoint.
            if self.model.activity_head is not None and val_activity_macro_f1 > best_activity_ckpt_macro_f1:
                best_activity_ckpt_macro_f1 = val_activity_macro_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_param_loss': val_param_loss,
                    'val_activity_loss': val_activity_loss,
                    'val_activity_macro_f1': val_activity_macro_f1,
                    'val_activity_micro_f1': val_activity_micro_f1,
                    'val_active_param_rmse': val_active_param_rmse,
                    'selection_metric': 'val_activity_macro_f1',
                    'selection_metric_value': val_activity_macro_f1,
                    'model_config': self.model.get_model_config(),
                }, save_dir / 'controller_best_activity.pt')

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

        print(
            f"\nTraining complete [{self.stage_name}]. "
            f"Best {self.selection_metric}: {self.history['best_selection_metric']:.6f}"
        )
        return self.history


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
    activity_mismatch_weight: float = 0.0,
    activity_mismatch_gamma: float = 2.0,
    activity_loss_type: str = "bce",
    focal_gamma: float = 2.0,
    asl_gamma_pos: float = 0.0,
    asl_gamma_neg: float = 4.0,
    asl_clip: float = 0.05,
    param_loss_weight: float = 1.0,
    inactive_param_weight: float = 1.0,
    param_loss_type: str = "mse",
    huber_delta: float = 0.05,
    effect_loss_weights: Optional[Dict[str, float]] = None,
    selection_metric: str = "val_param_loss",
    training_stages: Optional[List[Dict[str, Any]]] = None,
    balanced_sampler: Optional[Dict[str, Any]] = None,
    train_backbone: bool = True,
    train_param_head: bool = True,
    train_activity_head: bool = True,
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

    train_sampler = None
    if balanced_sampler and bool(balanced_sampler.get("enabled", False)):
        if not effect_names:
            print("WARNING: balanced_sampler enabled but effect_names missing in DB. Falling back to shuffle=True.")
        else:
            target_effects = [
                str(x) for x in balanced_sampler.get("effects", ["playback_rate", "delay"])
                if str(x) in effect_names
            ]
            boost = float(balanced_sampler.get("boost", 2.0))
            base_weight = float(balanced_sampler.get("base_weight", 1.0))
            mode = str(balanced_sampler.get("mode", "count_boost")).strip().lower()
            if target_effects and boost > 0:
                train_indices = np.array(train_ds.indices, dtype=np.int64)
                active_mask = dataset.effect_active_mask[train_indices]
                sample_weights = np.full(train_indices.shape[0], base_weight, dtype=np.float64)
                target_idxs = [effect_names.index(effect_name) for effect_name in target_effects]
                target_active = active_mask[:, target_idxs].astype(np.float64)
                if mode == "inverse_frequency":
                    # Reweight samples by inverse effect frequency so rare effects are sampled more.
                    freq = np.clip(target_active.mean(axis=0), 1e-8, 1.0)
                    inv_freq = (1.0 / freq)
                    inv_freq = inv_freq / np.mean(inv_freq)
                    sample_scores = (target_active * inv_freq[None, :]).mean(axis=1)
                    sample_weights += boost * sample_scores
                else:
                    for effect_name in target_effects:
                        effect_idx = effect_names.index(effect_name)
                        sample_weights += boost * active_mask[:, effect_idx].astype(np.float64)
                train_sampler = WeightedRandomSampler(
                    weights=torch.from_numpy(sample_weights),
                    num_samples=int(sample_weights.shape[0]),
                    replacement=True,
                )
                print(
                    f"Using balanced sampler: mode={mode}, effects={target_effects}, "
                    f"boost={boost}, base_weight={base_weight}"
                )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
    )
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

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    stage_cfgs: List[Dict[str, Any]] = []
    if training_stages:
        for i, raw in enumerate(training_stages):
            epochs_i = int(raw.get("epochs", 0))
            if epochs_i <= 0:
                continue
            stage_cfgs.append({
                "name": str(raw.get("name", f"stage_{i + 1}")),
                "epochs": epochs_i,
                "learning_rate": float(raw.get("learning_rate", learning_rate)),
                "activity_loss_weight": float(raw.get("activity_loss_weight", activity_loss_weight)),
                "activity_mismatch_weight": float(
                    raw.get("activity_mismatch_weight", activity_mismatch_weight)
                ),
                "activity_mismatch_gamma": float(
                    raw.get("activity_mismatch_gamma", activity_mismatch_gamma)
                ),
                "activity_loss_type": str(raw.get("activity_loss_type", activity_loss_type)),
                "focal_gamma": float(raw.get("focal_gamma", focal_gamma)),
                "asl_gamma_pos": float(raw.get("asl_gamma_pos", asl_gamma_pos)),
                "asl_gamma_neg": float(raw.get("asl_gamma_neg", asl_gamma_neg)),
                "asl_clip": float(raw.get("asl_clip", asl_clip)),
                "param_loss_weight": float(raw.get("param_loss_weight", param_loss_weight)),
                "effect_loss_weights": raw.get("effect_loss_weights", effect_loss_weights),
                "selection_metric": str(raw.get("selection_metric", selection_metric)),
                "train_backbone": bool(raw.get("train_backbone", train_backbone)),
                "train_param_head": bool(raw.get("train_param_head", train_param_head)),
                "train_activity_head": bool(raw.get("train_activity_head", train_activity_head)),
            })
    if not stage_cfgs:
        stage_cfgs = [{
            "name": "single_stage",
            "epochs": int(num_epochs),
            "learning_rate": float(learning_rate),
            "activity_loss_weight": float(activity_loss_weight),
            "activity_mismatch_weight": float(activity_mismatch_weight),
            "activity_mismatch_gamma": float(activity_mismatch_gamma),
            "activity_loss_type": str(activity_loss_type),
            "focal_gamma": float(focal_gamma),
            "asl_gamma_pos": float(asl_gamma_pos),
            "asl_gamma_neg": float(asl_gamma_neg),
            "asl_clip": float(asl_clip),
            "param_loss_weight": float(param_loss_weight),
            "effect_loss_weights": effect_loss_weights,
            "selection_metric": str(selection_metric),
            "train_backbone": bool(train_backbone),
            "train_param_head": bool(train_param_head),
            "train_activity_head": bool(train_activity_head),
        }]

    aggregate_keys = [
        "train_loss",
        "val_loss",
        "train_param_loss",
        "val_param_loss",
        "train_activity_loss",
        "val_activity_loss",
        "train_activity_macro_f1",
        "val_activity_macro_f1",
        "train_activity_micro_f1",
        "val_activity_micro_f1",
        "train_active_param_rmse",
        "val_active_param_rmse",
    ]
    aggregate_history: Dict[str, List[float]] = {k: [] for k in aggregate_keys}
    stages_summary: List[Dict[str, Any]] = []

    global_best_metric = float("inf")
    global_best_ckpt: Optional[Path] = None
    global_best_activity_ckpt: Optional[Path] = None
    global_best_thresholds: Optional[Path] = None
    final_stage_dir: Optional[Path] = None

    def _select_from_history(history: Dict[str, Any], metric_name: str) -> float:
        if metric_name == "val_loss":
            return float(history.get("best_val_loss", float("inf")))
        if metric_name == "val_active_param_rmse":
            return float(history.get("best_val_active_param_rmse", float("inf")))
        if metric_name == "val_activity_macro_f1":
            return float(history.get("best_val_activity_macro_f1", 0.0))
        if metric_name == "val_activity_micro_f1":
            return float(history.get("best_val_activity_micro_f1", 0.0))
        return float(history.get("best_val_param_loss", float("inf")))

    def _is_global_better(current: float, best: float, metric_name: str) -> bool:
        if metric_name in {"val_activity_macro_f1", "val_activity_micro_f1"}:
            return current > best
        return current < best

    if selection_metric in {"val_activity_macro_f1", "val_activity_micro_f1"}:
        global_best_metric = float("-inf")

    for i, stage in enumerate(stage_cfgs, start=1):
        stage_name = stage["name"]
        stage_dir = out_root / stage_name
        final_stage_dir = stage_dir
        print("\n" + "-" * 60)
        print(
            f"Controller stage {i}/{len(stage_cfgs)}: {stage_name} | "
            f"epochs={stage['epochs']} lr={stage['learning_rate']} "
            f"activity_w={stage['activity_loss_weight']} "
            f"activity_mismatch_w={stage['activity_mismatch_weight']} "
            f"activity_mismatch_gamma={stage['activity_mismatch_gamma']} "
            f"activity_loss_type={stage['activity_loss_type']} "
            f"param_w={stage['param_loss_weight']} "
            f"selection={stage['selection_metric']}"
        )
        print("-" * 60)

        trainer = ControllerTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=stage["learning_rate"],
            effect_names=effect_names,
            inactive_param_weight=inactive_param_weight,
            activity_loss_weight=stage["activity_loss_weight"],
            activity_mismatch_weight=stage["activity_mismatch_weight"],
            activity_mismatch_gamma=stage["activity_mismatch_gamma"],
            activity_loss_type=stage["activity_loss_type"],
            focal_gamma=stage["focal_gamma"],
            asl_gamma_pos=stage["asl_gamma_pos"],
            asl_gamma_neg=stage["asl_gamma_neg"],
            asl_clip=stage["asl_clip"],
            param_loss_weight=stage["param_loss_weight"],
            param_loss_type=param_loss_type,
            huber_delta=huber_delta,
            effect_loss_weights=stage["effect_loss_weights"],
            selection_metric=stage["selection_metric"],
            stage_name=stage_name,
            train_backbone=stage["train_backbone"],
            train_param_head=stage["train_param_head"],
            train_activity_head=stage["train_activity_head"],
        )
        stage_history = trainer.train(num_epochs=stage["epochs"], save_dir=str(stage_dir))

        for k in aggregate_keys:
            aggregate_history[k].extend(stage_history.get(k, []))

        stage_best_ckpt = stage_dir / "controller_best.pt"
        stage_best_activity_ckpt = stage_dir / "controller_best_activity.pt"
        stage_best_metric = _select_from_history(stage_history, selection_metric)
        stage_local_selection_metric = float(stage_history.get("best_selection_metric", float("inf")))

        threshold_tune_report: Optional[Dict[str, Any]] = None
        threshold_file: Optional[Path] = None
        if stage_best_ckpt.exists() and model.activity_head is not None:
            ckpt = torch.load(stage_best_ckpt, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            threshold_tune_report = trainer.tune_activity_thresholds()
            if threshold_tune_report is not None:
                threshold_file = stage_dir / "activity_thresholds.json"
                with open(threshold_file, "w") as f:
                    json.dump(threshold_tune_report, f, indent=2)

        stages_summary.append(
            {
                "name": stage_name,
                "epochs": stage["epochs"],
                "learning_rate": stage["learning_rate"],
                "activity_loss_weight": stage["activity_loss_weight"],
                "activity_mismatch_weight": stage["activity_mismatch_weight"],
                "activity_mismatch_gamma": stage["activity_mismatch_gamma"],
                "activity_loss_type": stage["activity_loss_type"],
                "focal_gamma": stage["focal_gamma"],
                "asl_gamma_pos": stage["asl_gamma_pos"],
                "asl_gamma_neg": stage["asl_gamma_neg"],
                "asl_clip": stage["asl_clip"],
                "param_loss_weight": stage["param_loss_weight"],
                "selection_metric": stage["selection_metric"],
                "train_backbone": stage["train_backbone"],
                "train_param_head": stage["train_param_head"],
                "train_activity_head": stage["train_activity_head"],
                "best_selection_metric": stage_local_selection_metric,
                "best_val_loss": float(stage_history.get("best_val_loss", float("inf"))),
                "best_val_param_loss": float(stage_history.get("best_val_param_loss", float("inf"))),
                "best_val_activity_loss": float(stage_history.get("best_val_activity_loss", float("inf"))),
                "best_val_activity_macro_f1": float(
                    stage_history.get("best_val_activity_macro_f1", 0.0)
                ),
                "best_val_activity_micro_f1": float(
                    stage_history.get("best_val_activity_micro_f1", 0.0)
                ),
                "best_val_active_param_rmse": float(
                    stage_history.get("best_val_active_param_rmse", float("inf"))
                ),
                "activity_thresholds_file": str(threshold_file) if threshold_file else None,
                "stage_dir": str(stage_dir),
            }
        )

        # Continue from the stage-best checkpoint.
        if stage_best_ckpt.exists():
            ckpt = torch.load(stage_best_ckpt, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            if _is_global_better(stage_best_metric, global_best_metric, selection_metric):
                global_best_metric = stage_best_metric
                global_best_ckpt = stage_best_ckpt
                global_best_thresholds = threshold_file

        if stage_best_activity_ckpt.exists():
            if global_best_activity_ckpt is None:
                global_best_activity_ckpt = stage_best_activity_ckpt
            else:
                prev = torch.load(global_best_activity_ckpt, map_location="cpu")
                cur = torch.load(stage_best_activity_ckpt, map_location="cpu")
                if float(cur.get("val_activity_macro_f1", 0.0)) > float(prev.get("val_activity_macro_f1", 0.0)):
                    global_best_activity_ckpt = stage_best_activity_ckpt

    if global_best_ckpt is not None:
        shutil.copy2(global_best_ckpt, out_root / "controller_best.pt")
        print(f"Global best checkpoint: {global_best_ckpt}")
    elif final_stage_dir is not None and (final_stage_dir / "controller_best.pt").exists():
        shutil.copy2(final_stage_dir / "controller_best.pt", out_root / "controller_best.pt")

    if final_stage_dir is not None and (final_stage_dir / "controller_final.pt").exists():
        shutil.copy2(final_stage_dir / "controller_final.pt", out_root / "controller_final.pt")

    if global_best_activity_ckpt is not None:
        shutil.copy2(global_best_activity_ckpt, out_root / "controller_best_activity.pt")
        print(f"Global best activity checkpoint: {global_best_activity_ckpt}")

    if global_best_thresholds is not None and global_best_thresholds.exists():
        shutil.copy2(global_best_thresholds, out_root / "activity_thresholds.json")
        print(f"Saved tuned activity thresholds: {out_root / 'activity_thresholds.json'}")

    if aggregate_history["val_loss"]:
        best_idx = int(np.argmin(np.array(aggregate_history["val_loss"], dtype=np.float64)))
        aggregate_history["best_val_loss"] = float(aggregate_history["val_loss"][best_idx])
        aggregate_history["best_val_param_loss"] = float(aggregate_history["val_param_loss"][best_idx])
        aggregate_history["best_val_activity_loss"] = float(
            aggregate_history["val_activity_loss"][best_idx]
        )
        aggregate_history["best_val_activity_macro_f1"] = float(
            np.max(np.array(aggregate_history["val_activity_macro_f1"], dtype=np.float64))
            if aggregate_history.get("val_activity_macro_f1")
            else 0.0
        )
        aggregate_history["best_val_activity_micro_f1"] = float(
            np.max(np.array(aggregate_history["val_activity_micro_f1"], dtype=np.float64))
            if aggregate_history.get("val_activity_micro_f1")
            else 0.0
        )
        aggregate_history["best_val_active_param_rmse"] = float(
            aggregate_history["val_active_param_rmse"][best_idx]
        )
        aggregate_history["selection_metric"] = selection_metric
        aggregate_history["best_selection_metric"] = float(global_best_metric)

    with open(out_root / "training_log.json", "w") as f:
        json.dump(aggregate_history, f, indent=2)

    with open(out_root / "training_stages_summary.json", "w") as f:
        json.dump(
            {
                "selection_metric": selection_metric,
                "global_best_selection_metric": float(global_best_metric),
                "global_best_checkpoint": str(global_best_ckpt) if global_best_ckpt else None,
                "global_best_activity_checkpoint": (
                    str(global_best_activity_ckpt) if global_best_activity_ckpt else None
                ),
                "global_best_activity_thresholds": (
                    str(global_best_thresholds) if global_best_thresholds else None
                ),
                "stages": stages_summary,
            },
            f,
            indent=2,
        )
