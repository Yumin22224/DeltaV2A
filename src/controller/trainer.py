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
import math
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

from .model import AudioController
from ..database.inverse_mapping import InverseMappingDB, InverseMappingDataset
from ..effects.pedalboard_effects import EFFECT_CATALOG, normalize_params


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
        fp_param_weight: float = 0.0,
        param_loss_type: str = "mse",
        huber_delta: float = 0.05,
        effect_loss_weights: Optional[Dict[str, float]] = None,
        selection_metric: str = "val_param_loss",
        stage_name: Optional[str] = None,
        train_backbone: bool = True,
        train_param_head: bool = True,
        train_activity_head: bool = True,
        lr_scheduler_type: str = "none",
        lr_min: float = 1e-6,
        confidence_weighting_enabled: bool = False,
        confidence_weight_power: float = 1.0,
        confidence_min_weight: float = 0.2,
        confidence_use_delta_norm: bool = False,
        confidence_style_alpha: float = 1.0,
        confidence_delta_scale: float = 1.0,
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
        self.fp_param_weight = float(fp_param_weight)
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
            "val_active_param_rmse_gated",
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
        if self.effect_names:
            bypass_np = normalize_params({}, self.effect_names).astype(np.float32)
            self.bypass_norm = torch.from_numpy(bypass_np).to(device)
        else:
            self.bypass_norm = torch.zeros(0, dtype=torch.float32, device=device)
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
        self.lr_scheduler_type = str(lr_scheduler_type).lower()
        self.lr_min = float(lr_min)
        self.confidence_weighting_enabled = bool(confidence_weighting_enabled)
        self.confidence_weight_power = float(max(confidence_weight_power, 0.0))
        self.confidence_min_weight = float(np.clip(confidence_min_weight, 0.0, 1.0))
        self.confidence_use_delta_norm = bool(confidence_use_delta_norm)
        self.confidence_style_alpha = float(np.clip(confidence_style_alpha, 0.0, 1.0))
        self.confidence_delta_scale = float(max(confidence_delta_scale, 1e-6))
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
            'train_active_param_rmse_gated': [],
            'val_active_param_rmse_gated': [],
            'best_val_loss': float('inf'),
            'best_val_param_loss': float('inf'),
            'best_val_activity_loss': float('inf'),
            'best_val_activity_macro_f1': 0.0,
            'best_val_activity_micro_f1': 0.0,
            'best_val_active_param_rmse': float('inf'),
            'best_val_active_param_rmse_gated': float('inf'),
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

    def _compute_activity_loss(
        self,
        activity_logits: torch.Tensor,
        effect_active_mask: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
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

        if sample_weights is None:
            return loss.mean()
        sw = sample_weights.view(-1, 1)
        denom = torch.clamp(sw.sum() * loss.shape[1], min=1e-8)
        return (loss * sw).sum() / denom

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
        sample_weights: Optional[torch.Tensor] = None,
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

        if sample_weights is not None:
            weights = weights * sample_weights.view(-1, 1)

        weight_denom = torch.clamp(weights.sum(), min=1e-8)
        param_loss = (param_err * weights).sum() / weight_denom

        activity_loss = torch.zeros((), device=params_gt.device)
        if activity_logits is not None and effect_active_mask is not None:
            activity_loss = self._compute_activity_loss(
                activity_logits,
                effect_active_mask,
                sample_weights=sample_weights,
            )

        # False-positive param penalty: penalize GT-inactive effects that the
        # activity head predicts as active, proportional to predicted probability.
        # Gradient flows through both the param head (toward bypass) and the activity
        # head (toward lower logit for GT-inactive effects). More principled than
        # inactive_param_weight because it only fires when the model actually mispredicts.
        fp_loss = torch.zeros((), device=params_gt.device)
        if (
            self.fp_param_weight > 0.0
            and activity_logits is not None
            and effect_active_mask is not None
            and self.effect_param_slices
            and self.bypass_norm.numel() > 0
        ):
            activity_probs = torch.sigmoid(activity_logits)           # (B, num_effects)
            inactive_mask = 1.0 - effect_active_mask                  # (B, num_effects)
            for i, (effect_name, sl) in enumerate(zip(self.effect_names, self.effect_param_slices)):
                if i >= activity_probs.shape[1]:
                    break
                soft_w = inactive_mask[:, i] * activity_probs[:, i]   # (B,) soft penalty weight
                w_sum = float(soft_w.sum().item())
                if w_sum < 1e-8:
                    continue
                bypass_i = self.bypass_norm[sl].unsqueeze(0)           # (1, n_params)
                pred_i = params_pred[:, sl]                            # (B, n_params)
                if self.param_loss_type == "huber":
                    err = F.smooth_l1_loss(pred_i, bypass_i.expand_as(pred_i),
                                           reduction='none', beta=self.huber_delta)
                else:
                    err = torch.square(pred_i - bypass_i)
                err_per_sample = err.mean(dim=1)                       # (B,)
                eff_w = float(self.effect_loss_weights.get(effect_name, 1.0))
                fp_loss = fp_loss + eff_w * (soft_w * err_per_sample).sum() / (w_sum + 1e-8)

        total_loss = (
            self.param_loss_weight * param_loss
            + self.activity_loss_weight * activity_loss
            + self.fp_param_weight * fp_loss
        )
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

    def _apply_activity_gating(
        self,
        params_pred: torch.Tensor,
        activity_logits: Optional[torch.Tensor],
        activity_thresholds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Gate predicted params by predicted activity.
        Inactive predicted effects are forced to bypass-normalized params.
        """
        if (
            activity_logits is None
            or activity_logits.numel() == 0
            or not self.effect_param_slices
        ):
            return params_pred

        probs = torch.sigmoid(activity_logits)
        if activity_thresholds is None:
            thresholds = torch.full(
                (1, probs.shape[1]), 0.5, device=probs.device, dtype=probs.dtype
            )
        else:
            thresholds = activity_thresholds
            if thresholds.ndim == 1:
                thresholds = thresholds.unsqueeze(0)
            thresholds = thresholds.to(device=probs.device, dtype=probs.dtype)
            if thresholds.shape[1] != probs.shape[1]:
                raise ValueError(
                    "activity_thresholds width mismatch: "
                    f"{thresholds.shape[1]} vs {probs.shape[1]}"
                )
        pred_on = probs >= thresholds

        pred_gated = params_pred.clone()
        for i, sl in enumerate(self.effect_param_slices):
            inactive = ~pred_on[:, i]
            if torch.any(inactive):
                pred_gated[inactive, sl] = self.bypass_norm[sl]
        return pred_gated

    def _compute_active_param_rmse_gated(
        self,
        params_pred: torch.Tensor,
        params_gt: torch.Tensor,
        effect_active_mask: Optional[torch.Tensor],
        activity_logits: Optional[torch.Tensor],
        activity_thresholds: Optional[torch.Tensor] = None,
    ):
        """
        Active-param RMSE after activity-based gating (selection-aligned proxy).
        """
        if (
            effect_active_mask is None
            or effect_active_mask.numel() == 0
            or activity_logits is None
            or activity_logits.numel() == 0
            or not self.effect_param_slices
        ):
            return torch.zeros((), device=params_gt.device), False

        param_active_mask = self._expand_effect_mask(effect_active_mask).float()
        denom = torch.clamp(param_active_mask.sum(), min=1e-8)
        if float(denom.item()) <= 1e-7:
            return torch.zeros((), device=params_gt.device), False

        pred_gated = self._apply_activity_gating(
            params_pred=params_pred,
            activity_logits=activity_logits,
            activity_thresholds=activity_thresholds,
        )
        mse = torch.square(pred_gated - params_gt) * param_active_mask
        rmse = torch.sqrt(torch.clamp(mse.sum() / denom, min=0.0))
        return rmse, True

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_param_loss = 0.0
        total_activity_loss = 0.0
        total_active_param_rmse = 0.0
        total_active_param_rmse_gated = 0.0
        has_activity_stats = False
        tp = None
        fp = None
        fn = None
        n = 0
        n_active = 0
        n_active_gated = 0
        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            clap_emb = batch['clap_embedding'].to(self.device)
            style = batch['style_label'].to(self.device)
            params_gt = batch['normalized_params'].to(self.device)
            clap_delta_norm = batch.get('clap_delta_norm')
            if clap_delta_norm is not None:
                clap_delta_norm = clap_delta_norm.to(self.device)
            sample_weights = self._compute_confidence_weights(style, clap_delta_norm=clap_delta_norm)
            effect_active_mask = None
            if 'effect_active_mask' in batch:
                effect_active_mask = batch['effect_active_mask'].to(self.device)

            params_pred, activity_logits = self.model.forward_with_activity(clap_emb, style)
            loss, param_loss, activity_loss = self._compute_losses(
                params_pred=params_pred,
                params_gt=params_gt,
                effect_active_mask=effect_active_mask,
                activity_logits=activity_logits,
                sample_weights=sample_weights,
            )
            active_param_rmse, has_active = self._compute_active_param_rmse(
                params_pred=params_pred,
                params_gt=params_gt,
                effect_active_mask=effect_active_mask,
            )
            active_param_rmse_gated, has_active_gated = self._compute_active_param_rmse_gated(
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
            if has_active:
                total_active_param_rmse += active_param_rmse.item()
                n_active += 1
            if has_active_gated:
                total_active_param_rmse_gated += active_param_rmse_gated.item()
                n_active_gated += 1
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
        active_gated_denom = max(n_active_gated, 1)
        train_activity_macro_f1 = 0.0
        train_activity_micro_f1 = 0.0
        if has_activity_stats and tp is not None and fp is not None and fn is not None:
            train_activity_macro_f1, train_activity_micro_f1 = self._activity_f1_from_counts(tp, fp, fn)
        return (
            total_loss / denom,
            total_param_loss / denom,
            total_activity_loss / denom,
            total_active_param_rmse / active_denom,
            total_active_param_rmse_gated / active_gated_denom,
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
        total_active_param_rmse_gated = 0.0
        has_activity_stats = False
        tp = None
        fp = None
        fn = None
        n = 0
        n_active = 0
        n_active_gated = 0
        for batch in self.val_loader:
            clap_emb = batch['clap_embedding'].to(self.device)
            style = batch['style_label'].to(self.device)
            params_gt = batch['normalized_params'].to(self.device)
            clap_delta_norm = batch.get('clap_delta_norm')
            if clap_delta_norm is not None:
                clap_delta_norm = clap_delta_norm.to(self.device)
            sample_weights = self._compute_confidence_weights(style, clap_delta_norm=clap_delta_norm)
            effect_active_mask = None
            if 'effect_active_mask' in batch:
                effect_active_mask = batch['effect_active_mask'].to(self.device)

            params_pred, activity_logits = self.model.forward_with_activity(clap_emb, style)
            loss, param_loss, activity_loss = self._compute_losses(
                params_pred=params_pred,
                params_gt=params_gt,
                effect_active_mask=effect_active_mask,
                activity_logits=activity_logits,
                sample_weights=sample_weights,
            )
            active_param_rmse, has_active = self._compute_active_param_rmse(
                params_pred=params_pred,
                params_gt=params_gt,
                effect_active_mask=effect_active_mask,
            )
            active_param_rmse_gated, has_active_gated = self._compute_active_param_rmse_gated(
                params_pred=params_pred,
                params_gt=params_gt,
                effect_active_mask=effect_active_mask,
                activity_logits=activity_logits,
            )
            total_loss += loss.item()
            total_param_loss += param_loss.item()
            total_activity_loss += activity_loss.item()
            if has_active:
                total_active_param_rmse += active_param_rmse.item()
                n_active += 1
            if has_active_gated:
                total_active_param_rmse_gated += active_param_rmse_gated.item()
                n_active_gated += 1
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
        active_gated_denom = max(n_active_gated, 1)
        val_activity_macro_f1 = 0.0
        val_activity_micro_f1 = 0.0
        if has_activity_stats and tp is not None and fp is not None and fn is not None:
            val_activity_macro_f1, val_activity_micro_f1 = self._activity_f1_from_counts(tp, fp, fn)
        return (
            total_loss / denom,
            total_param_loss / denom,
            total_activity_loss / denom,
            total_active_param_rmse / active_denom,
            total_active_param_rmse_gated / active_gated_denom,
            val_activity_macro_f1,
            val_activity_micro_f1,
        )

    @torch.no_grad()
    def tune_activity_thresholds(
        self,
        num_thresholds: int = 37,
        threshold_min: float = 0.05,
        threshold_max: float = 0.95,
        objective: str = "active_param_rmse_gated",
        coord_passes: int = 2,
        min_macro_f1_ratio: float = 0.95,
        min_micro_f1_ratio: float = 0.95,
        f1_penalty_weight: float = 1.0,
    ) -> Optional[Dict[str, Any]]:
        if self.model.activity_head is None or not self.effect_names:
            return None

        logits_list: List[np.ndarray] = []
        target_list: List[np.ndarray] = []
        params_pred_list: List[np.ndarray] = []
        params_gt_list: List[np.ndarray] = []
        self.model.eval()
        for batch in self.val_loader:
            clap_emb = batch['clap_embedding'].to(self.device)
            style = batch['style_label'].to(self.device)
            params_gt = batch.get('normalized_params')
            effect_active_mask = batch.get('effect_active_mask')
            if effect_active_mask is None or params_gt is None:
                continue
            effect_active_mask = effect_active_mask.to(self.device)
            params_gt = params_gt.to(self.device)
            params_pred, activity_logits = self.model.forward_with_activity(clap_emb, style)
            if activity_logits is None:
                continue
            logits_list.append(activity_logits.detach().cpu().numpy())
            target_list.append(effect_active_mask.detach().cpu().numpy())
            params_pred_list.append(params_pred.detach().cpu().numpy())
            params_gt_list.append(params_gt.detach().cpu().numpy())

        if not logits_list:
            return None

        logits = np.concatenate(logits_list, axis=0)
        targets = np.concatenate(target_list, axis=0)
        params_pred = np.concatenate(params_pred_list, axis=0)
        params_gt = np.concatenate(params_gt_list, axis=0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        thresholds_grid = np.linspace(threshold_min, threshold_max, num=max(int(num_thresholds), 2))
        objective_mode = str(objective).lower().strip()
        default_pred = (probs >= 0.5).astype(np.int32)

        def _f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            tp = float(np.sum((y_true == 1) & (y_pred == 1)))
            fp = float(np.sum((y_true == 0) & (y_pred == 1)))
            fn = float(np.sum((y_true == 1) & (y_pred == 0)))
            precision = tp / max(tp + fp, 1e-8)
            recall = tp / max(tp + fn, 1e-8)
            return float((2.0 * precision * recall) / max(precision + recall, 1e-8))

        y_true = (targets >= 0.5).astype(np.int32)
        tuned_thresholds_arr = np.full((len(self.effect_names),), 0.5, dtype=np.float32)
        objective_trace: List[float] = []

        def _expand_effect_mask_np(effect_mask: np.ndarray) -> np.ndarray:
            chunks = []
            for i, sl in enumerate(self.effect_param_slices):
                width = sl.stop - sl.start
                chunks.append(np.repeat(effect_mask[:, i:i + 1], width, axis=1))
            return np.concatenate(chunks, axis=1).astype(np.float32)

        def _active_param_rmse_gated(thresholds: np.ndarray) -> float:
            pred_on = probs >= thresholds[None, :]
            pred_gated = params_pred.copy()
            bypass = self.bypass_norm.detach().cpu().numpy().astype(np.float32)
            for i, sl in enumerate(self.effect_param_slices):
                inactive = ~pred_on[:, i]
                if np.any(inactive):
                    pred_gated[inactive, sl] = bypass[sl]
            active_param_mask = _expand_effect_mask_np(y_true)
            denom = float(max(active_param_mask.sum(), 1e-8))
            mse = float((((pred_gated - params_gt) ** 2) * active_param_mask).sum() / denom)
            return float(np.sqrt(max(mse, 0.0)))

        if objective_mode in {"active_param_rmse_gated", "active_param_rmse_gated_balanced"} and self.effect_param_slices:
            tuned_thresholds_arr = np.full((len(self.effect_names),), 0.5, dtype=np.float32)
            objective_trace.append(_active_param_rmse_gated(tuned_thresholds_arr))
            macro_floor = 0.0
            micro_floor = 0.0
            if objective_mode == "active_param_rmse_gated_balanced":
                floor_macro_ratio = float(np.clip(min_macro_f1_ratio, 0.0, 1.0))
                floor_micro_ratio = float(np.clip(min_micro_f1_ratio, 0.0, 1.0))
                macro_floor = floor_macro_ratio * float(
                    np.mean([_f1_score(y_true[:, i], default_pred[:, i]) for i in range(y_true.shape[1])])
                )
                micro_floor = floor_micro_ratio * float(_f1_score(y_true.reshape(-1), default_pred.reshape(-1)))
                penalty_w = float(max(f1_penalty_weight, 0.0))
            else:
                penalty_w = 0.0

            def _objective_value(thresholds: np.ndarray) -> float:
                rmse_val = _active_param_rmse_gated(thresholds)
                if objective_mode != "active_param_rmse_gated_balanced":
                    return float(rmse_val)
                pred_bin = (probs >= thresholds[None, :]).astype(np.int32)
                macro_val = float(
                    np.mean([_f1_score(y_true[:, i], pred_bin[:, i]) for i in range(y_true.shape[1])])
                )
                micro_val = float(_f1_score(y_true.reshape(-1), pred_bin.reshape(-1)))
                penalty = max(0.0, macro_floor - macro_val) + max(0.0, micro_floor - micro_val)
                return float(rmse_val + (penalty_w * penalty))

            n_passes = max(int(coord_passes), 1)
            for _ in range(n_passes):
                for i in range(len(self.effect_names)):
                    best_thr = float(tuned_thresholds_arr[i])
                    best_obj = float("inf")
                    for thr in thresholds_grid:
                        cand = tuned_thresholds_arr.copy()
                        cand[i] = float(thr)
                        obj_val = _objective_value(cand)
                        if obj_val < best_obj:
                            best_obj = obj_val
                            best_thr = float(thr)
                    tuned_thresholds_arr[i] = best_thr
                objective_trace.append(_objective_value(tuned_thresholds_arr))
        else:
            # Fallback: optimize each effect threshold independently by F1.
            objective_mode = "f1_macro"
            for i in range(len(self.effect_names)):
                y_i = y_true[:, i]
                best_thr = 0.5
                best_f1 = -1.0
                for thr in thresholds_grid:
                    y_pred_i = (probs[:, i] >= float(thr)).astype(np.int32)
                    f1 = _f1_score(y_i, y_pred_i)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thr = float(thr)
                tuned_thresholds_arr[i] = float(best_thr)
            objective_trace.append(
                float(
                    np.mean(
                        [
                            _f1_score(y_true[:, i], (probs[:, i] >= tuned_thresholds_arr[i]).astype(np.int32))
                            for i in range(y_true.shape[1])
                        ]
                    )
                )
            )

        tuned_pred = (probs >= tuned_thresholds_arr[None, :]).astype(np.int32)
        macro_tuned = float(np.mean([_f1_score(y_true[:, i], tuned_pred[:, i]) for i in range(y_true.shape[1])]))
        macro_default = float(np.mean([_f1_score(y_true[:, i], default_pred[:, i]) for i in range(y_true.shape[1])]))
        micro_tuned = _f1_score(y_true.reshape(-1), tuned_pred.reshape(-1))
        micro_default = _f1_score(y_true.reshape(-1), default_pred.reshape(-1))

        active_rmse_default = None
        active_rmse_tuned = None
        if self.effect_param_slices:
            active_rmse_default = _active_param_rmse_gated(np.full_like(tuned_thresholds_arr, 0.5))
            active_rmse_tuned = _active_param_rmse_gated(tuned_thresholds_arr)

        per_effect: List[Dict[str, Any]] = []
        for i, effect_name in enumerate(self.effect_names):
            y_i = y_true[:, i]
            per_effect.append(
                {
                    "effect": effect_name,
                    "threshold": float(tuned_thresholds_arr[i]),
                    "f1_at_tuned_threshold": float(_f1_score(y_i, tuned_pred[:, i])),
                    "f1_at_0_5": float(_f1_score(y_i, default_pred[:, i])),
                }
            )

        return {
            "num_samples": int(y_true.shape[0]),
            "objective": objective_mode,
            "coord_passes": int(max(int(coord_passes), 1)),
            "min_macro_f1_ratio": float(min_macro_f1_ratio),
            "min_micro_f1_ratio": float(min_micro_f1_ratio),
            "f1_penalty_weight": float(f1_penalty_weight),
            "threshold_search": {
                "min": float(threshold_min),
                "max": float(threshold_max),
                "count": int(max(int(num_thresholds), 2)),
            },
            "macro_f1_default_0_5": macro_default,
            "macro_f1_tuned": macro_tuned,
            "micro_f1_default_0_5": micro_default,
            "micro_f1_tuned": micro_tuned,
            "active_param_rmse_gated_default_0_5": active_rmse_default,
            "active_param_rmse_gated_tuned": active_rmse_tuned,
            "objective_trace": objective_trace,
            "per_effect": per_effect,
            "thresholds": {name: float(thr) for name, thr in zip(self.effect_names, tuned_thresholds_arr.tolist())},
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

        scheduler = None
        if self.lr_scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=num_epochs, eta_min=self.lr_min
            )
            print(f"  LR scheduler: CosineAnnealingLR (T_max={num_epochs}, eta_min={self.lr_min})")

        def _select_metric(
            val_loss: float,
            val_param_loss: float,
            val_active_param_rmse: float,
            val_active_param_rmse_gated: float,
            val_activity_macro_f1: float,
            val_activity_micro_f1: float,
        ) -> float:
            if self.selection_metric == "val_loss":
                return float(val_loss)
            if self.selection_metric == "val_active_param_rmse":
                return float(val_active_param_rmse)
            if self.selection_metric == "val_active_param_rmse_gated":
                return float(val_active_param_rmse_gated)
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
                train_active_param_rmse_gated,
                train_activity_macro_f1,
                train_activity_micro_f1,
            ) = self.train_epoch()
            (
                val_loss,
                val_param_loss,
                val_activity_loss,
                val_active_param_rmse,
                val_active_param_rmse_gated,
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
            self.history['train_active_param_rmse_gated'].append(train_active_param_rmse_gated)
            self.history['val_active_param_rmse_gated'].append(val_active_param_rmse_gated)

            print(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train(total/param/act/active_rmse/active_rmse_g): "
                f"{train_loss:.6f}/{train_param_loss:.6f}/{train_activity_loss:.6f}/"
                f"{train_active_param_rmse:.6f}/{train_active_param_rmse_gated:.6f}, "
                f"Val(total/param/act/active_rmse/active_rmse_g): "
                f"{val_loss:.6f}/{val_param_loss:.6f}/{val_activity_loss:.6f}/"
                f"{val_active_param_rmse:.6f}/{val_active_param_rmse_gated:.6f}, "
                f"Val(act_macro_f1/act_micro_f1): {val_activity_macro_f1:.4f}/{val_activity_micro_f1:.4f}"
            )

            select_val = _select_metric(
                val_loss=val_loss,
                val_param_loss=val_param_loss,
                val_active_param_rmse=val_active_param_rmse,
                val_active_param_rmse_gated=val_active_param_rmse_gated,
                val_activity_macro_f1=val_activity_macro_f1,
                val_activity_micro_f1=val_activity_micro_f1,
            )
            if scheduler is not None:
                scheduler.step()

            if _is_better(select_val, self.history['best_selection_metric']):
                self.history['best_val_loss'] = val_loss
                self.history['best_val_param_loss'] = val_param_loss
                self.history['best_val_activity_loss'] = val_activity_loss
                self.history['best_val_activity_macro_f1'] = val_activity_macro_f1
                self.history['best_val_activity_micro_f1'] = val_activity_micro_f1
                self.history['best_val_active_param_rmse'] = val_active_param_rmse
                self.history['best_val_active_param_rmse_gated'] = val_active_param_rmse_gated
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
                    'val_active_param_rmse_gated': val_active_param_rmse_gated,
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
                    'val_active_param_rmse_gated': val_active_param_rmse_gated,
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
    weight_decay: float = 1e-5,
    val_split: float = 0.2,
    audio_embed_dim: int = 512,
    hidden_dims: List[int] = None,
    dropout: float = 0.1,
    fusion_mode: str = "concat",
    audio_gate_bias: float = -2.0,
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
    fp_param_weight: float = 0.0,
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
    lr_scheduler_type: str = "none",
    lr_min: float = 1e-6,
    confidence_weighting_enabled: bool = False,
    confidence_weight_power: float = 1.0,
    confidence_min_weight: float = 0.2,
    confidence_use_delta_norm: bool = False,
    confidence_style_alpha: float = 1.0,
    confidence_delta_scale: float = 1.0,
    threshold_tuning: Optional[Dict[str, Any]] = None,
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
        audio_embed_dim=audio_embed_dim,
        style_vocab_size=style_vocab_size,
        total_params=total_params,
        hidden_dims=hidden_dims or [512, 256, 128],
        dropout=dropout,
        fusion_mode=fusion_mode,
        audio_gate_bias=audio_gate_bias,
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
                "lr_scheduler_type": str(raw.get("lr_scheduler_type", lr_scheduler_type)),
                "lr_min": float(raw.get("lr_min", lr_min)),
                "confidence_weighting_enabled": bool(
                    raw.get("confidence_weighting_enabled", confidence_weighting_enabled)
                ),
                "confidence_weight_power": float(
                    raw.get("confidence_weight_power", confidence_weight_power)
                ),
                "confidence_min_weight": float(
                    raw.get("confidence_min_weight", confidence_min_weight)
                ),
                "confidence_use_delta_norm": bool(
                    raw.get("confidence_use_delta_norm", confidence_use_delta_norm)
                ),
                "confidence_style_alpha": float(
                    raw.get("confidence_style_alpha", confidence_style_alpha)
                ),
                "confidence_delta_scale": float(
                    raw.get("confidence_delta_scale", confidence_delta_scale)
                ),
                "threshold_tuning": raw.get("threshold_tuning", threshold_tuning),
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
            "lr_scheduler_type": str(lr_scheduler_type),
            "lr_min": float(lr_min),
            "confidence_weighting_enabled": bool(confidence_weighting_enabled),
            "confidence_weight_power": float(confidence_weight_power),
            "confidence_min_weight": float(confidence_min_weight),
            "confidence_use_delta_norm": bool(confidence_use_delta_norm),
            "confidence_style_alpha": float(confidence_style_alpha),
            "confidence_delta_scale": float(confidence_delta_scale),
            "threshold_tuning": threshold_tuning,
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
        "train_active_param_rmse_gated",
        "val_active_param_rmse_gated",
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
        if metric_name == "val_active_param_rmse_gated":
            return float(history.get("best_val_active_param_rmse_gated", float("inf")))
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
            weight_decay=weight_decay,
            effect_names=effect_names,
            inactive_param_weight=inactive_param_weight,
            fp_param_weight=fp_param_weight,
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
            lr_scheduler_type=stage["lr_scheduler_type"],
            lr_min=stage["lr_min"],
            confidence_weighting_enabled=stage["confidence_weighting_enabled"],
            confidence_weight_power=stage["confidence_weight_power"],
            confidence_min_weight=stage["confidence_min_weight"],
            confidence_use_delta_norm=stage["confidence_use_delta_norm"],
            confidence_style_alpha=stage["confidence_style_alpha"],
            confidence_delta_scale=stage["confidence_delta_scale"],
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
            tt_cfg = stage.get("threshold_tuning") or {}
            threshold_tune_report = trainer.tune_activity_thresholds(
                num_thresholds=int(tt_cfg.get("num_thresholds", 37)),
                threshold_min=float(tt_cfg.get("threshold_min", 0.05)),
                threshold_max=float(tt_cfg.get("threshold_max", 0.95)),
                objective=str(tt_cfg.get("objective", "active_param_rmse_gated")),
                coord_passes=int(tt_cfg.get("coord_passes", 2)),
                min_macro_f1_ratio=float(tt_cfg.get("min_macro_f1_ratio", 0.95)),
                min_micro_f1_ratio=float(tt_cfg.get("min_micro_f1_ratio", 0.95)),
                f1_penalty_weight=float(tt_cfg.get("f1_penalty_weight", 1.0)),
            )
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
                "best_val_active_param_rmse_gated": float(
                    stage_history.get("best_val_active_param_rmse_gated", float("inf"))
                ),
                "threshold_tuning": stage.get("threshold_tuning"),
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
        if selection_metric == "val_loss":
            best_idx = int(np.argmin(np.array(aggregate_history["val_loss"], dtype=np.float64)))
        elif selection_metric == "val_active_param_rmse":
            best_idx = int(np.argmin(np.array(aggregate_history["val_active_param_rmse"], dtype=np.float64)))
        elif selection_metric == "val_active_param_rmse_gated":
            best_idx = int(
                np.argmin(np.array(aggregate_history["val_active_param_rmse_gated"], dtype=np.float64))
            )
        elif selection_metric == "val_activity_macro_f1":
            best_idx = int(np.argmax(np.array(aggregate_history["val_activity_macro_f1"], dtype=np.float64)))
        elif selection_metric == "val_activity_micro_f1":
            best_idx = int(np.argmax(np.array(aggregate_history["val_activity_micro_f1"], dtype=np.float64)))
        else:
            best_idx = int(np.argmin(np.array(aggregate_history["val_param_loss"], dtype=np.float64)))
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
        aggregate_history["best_val_active_param_rmse_gated"] = float(
            aggregate_history["val_active_param_rmse_gated"][best_idx]
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
