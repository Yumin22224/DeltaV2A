"""
AR Hybrid Controller (Phase B)

Autoregressive effect chain predictor:
  style_label (24d) → GRU → [effect_token, params] × steps → STOP

At each step:
  - Effect head: discrete selection over 7 effects + STOP token (8 classes)
  - Param head:  per-effect MLP → continuous params in [0, 1] via sigmoid

Training uses teacher forcing with effect_order from the inverse mapping DB.
Inference runs autoregressively until STOP token or max_steps reached.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from ..effects.pedalboard_effects import EFFECT_CATALOG, denormalize_params


class ARController(nn.Module):
    """
    Autoregressive controller that predicts a chain of DSP effects.

    Args:
        effect_names: Ordered list of effect names (must match DB).
        style_vocab_size: Dimension of input style label.
        condition_dim: Hidden size of condition encoder.
        hidden_dim: GRU hidden size.
        dropout: Dropout probability.
        max_steps: Maximum AR steps per inference call (= max_active_effects).
    """

    # Special token indices
    BOS_OFFSET = 0   # BOS uses index = num_effects + 1
    STOP_OFFSET = 0  # STOP uses index = num_effects

    def __init__(
        self,
        effect_names: List[str],
        style_vocab_size: int = 24,
        condition_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        max_steps: int = 2,
    ):
        super().__init__()
        self.effect_names = list(effect_names)
        self.num_effects = len(effect_names)
        self.style_vocab_size = style_vocab_size
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps

        # Token indices
        self.stop_idx = self.num_effects          # 7 for 7 effects
        self.bos_idx = self.num_effects + 1       # 8 for 7 effects
        self.num_tokens = self.num_effects + 2    # effects + STOP + BOS

        # Per-effect param counts and flat slices into normalized_params
        self.effect_param_counts: List[int] = []
        self._effect_slices: List[slice] = []
        offset = 0
        for name in self.effect_names:
            count = EFFECT_CATALOG[name].num_params
            self.effect_param_counts.append(count)
            self._effect_slices.append(slice(offset, offset + count))
            offset += count
        self.total_params = offset
        self.max_params_per_step = max(self.effect_param_counts)

        # ── Condition encoder ──────────────────────────────────────────────
        # Maps style_label → initial GRU hidden state
        self.condition_encoder = nn.Sequential(
            nn.Linear(style_vocab_size, condition_dim),
            nn.LayerNorm(condition_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(condition_dim, hidden_dim),
        )

        # ── GRU input projection ───────────────────────────────────────────
        # GRU input = concat(token_embed, param_embed) → hidden_dim
        self.token_embed = nn.Embedding(self.num_tokens, hidden_dim // 2)
        self.param_proj = nn.Sequential(
            nn.Linear(self.max_params_per_step, hidden_dim // 2),
            nn.ReLU(),
        )

        # ── AR Decoder ─────────────────────────────────────────────────────
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # ── Heads ──────────────────────────────────────────────────────────
        # Effect head: predicts which effect (or STOP) at each step
        self.effect_head = nn.Linear(hidden_dim, self.num_effects + 1)  # excl. BOS

        # Per-effect param heads: each predicts its own continuous params
        self.param_heads = nn.ModuleDict({
            name: nn.Linear(hidden_dim, count)
            for name, count in zip(self.effect_names, self.effect_param_counts)
        })

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _encode_condition(self, style_label: torch.Tensor) -> torch.Tensor:
        """(B, V) → (B, hidden_dim)  initial GRU hidden state."""
        return self.condition_encoder(style_label)

    def _gru_input(
        self,
        token_idx: torch.Tensor,   # (B,) long
        params: torch.Tensor,      # (B, max_params_per_step)
    ) -> torch.Tensor:
        """Build GRU input vector from previous token + previous params."""
        tok = self.token_embed(token_idx)   # (B, hidden//2)
        par = self.param_proj(params)       # (B, hidden//2)
        return torch.cat([tok, par], dim=-1)  # (B, hidden)

    def _predict_params(
        self,
        feat: torch.Tensor,       # (B, hidden)
        effect_idx: torch.Tensor, # (B,) long — which effect for each sample
    ) -> torch.Tensor:
        """
        Predict params for a mixed batch (different effects per sample).
        Returns (B, max_params_per_step) with sigmoid, zeros for unused slots.
        """
        B = feat.shape[0]
        out = torch.zeros(B, self.max_params_per_step, device=feat.device)
        for eff_i, eff_name in enumerate(self.effect_names):
            mask = (effect_idx == eff_i)
            if not mask.any():
                continue
            n = self.effect_param_counts[eff_i]
            out[mask, :n] = torch.sigmoid(self.param_heads[eff_name](feat[mask]))
        return out

    # ──────────────────────────────────────────────────────────────────────
    # Training forward (teacher forcing)
    # ──────────────────────────────────────────────────────────────────────

    def forward_train(
        self,
        style_label: torch.Tensor,      # (B, V)
        effect_order: torch.Tensor,     # (B, max_steps) int, -1 = pad/STOP
        normalized_params: torch.Tensor,  # (B, total_params)
    ) -> Tuple[List[torch.Tensor], List[Optional[torch.Tensor]]]:
        """
        Teacher-forced forward pass.

        At step t the model receives the ground-truth (effect_t-1, params_t-1)
        as input and predicts effect_t (and params_t if not STOP).

        Returns:
            effect_logits_list: len = max_steps + 1
                logits over (num_effects + 1) classes at each step.
            param_preds_list: len = max_steps
                (B, max_params_per_step) sigmoid outputs; None if all STOP.
        """
        B = style_label.shape[0]
        device = style_label.device

        h = self._encode_condition(style_label)  # (B, hidden)

        # Step 0 input: BOS token + zero params
        token = torch.full((B,), self.bos_idx, dtype=torch.long, device=device)
        params_in = torch.zeros(B, self.max_params_per_step, device=device)

        effect_logits_list: List[torch.Tensor] = []
        param_preds_list: List[Optional[torch.Tensor]] = []

        for step in range(self.max_steps + 1):
            x = self._gru_input(token, params_in)
            h = self.gru(x, h)
            feat = self.dropout(h)

            # Effect prediction at this step
            effect_logits_list.append(self.effect_head(feat))  # (B, num_effects+1)

            if step == self.max_steps:
                break  # No param prediction needed after last STOP step

            # Ground-truth effect index at this step (-1 → treat as STOP)
            gt_eff = effect_order[:, step].long()     # (B,)
            active = (gt_eff >= 0)                    # samples still in chain

            # Param prediction (only for active/non-STOP samples)
            param_pred = torch.zeros(B, self.max_params_per_step, device=device)
            if active.any():
                param_pred[active] = self._predict_params(
                    feat[active], gt_eff[active]
                )
            param_preds_list.append(param_pred)

            # ── Teacher forcing: build next-step input from ground truth ──
            # Convert -1 (pad) → STOP index for embedding lookup
            tf_token = gt_eff.clone()
            tf_token[~active] = self.stop_idx

            # Ground-truth params for next input
            gt_params_in = torch.zeros(B, self.max_params_per_step, device=device)
            if active.any():
                for eff_i, sl in enumerate(self._effect_slices):
                    mask = (gt_eff == eff_i)
                    if not mask.any():
                        continue
                    n = self.effect_param_counts[eff_i]
                    gt_params_in[mask, :n] = normalized_params[mask, sl]

            token = tf_token
            params_in = gt_params_in

        return effect_logits_list, param_preds_list

    # ──────────────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def infer(
        self,
        style_label: torch.Tensor,  # (V,) or (1, V)
    ) -> List[Tuple[str, Dict[str, float]]]:
        """
        Autoregressive inference: returns ordered list of (effect_name, params_dict).
        Stops at STOP token or max_steps.
        """
        if style_label.ndim == 1:
            style_label = style_label.unsqueeze(0)
        device = style_label.device

        h = self._encode_condition(style_label)  # (1, hidden)
        token = torch.tensor([self.bos_idx], dtype=torch.long, device=device)
        params_in = torch.zeros(1, self.max_params_per_step, device=device)

        results: List[Tuple[str, Dict[str, float]]] = []

        for _ in range(self.max_steps):
            x = self._gru_input(token, params_in)
            h = self.gru(x, h)

            eff_idx = int(self.effect_head(h).argmax(dim=-1).item())
            if eff_idx == self.stop_idx:
                break

            eff_name = self.effect_names[eff_idx]
            n = self.effect_param_counts[eff_idx]
            param_vals = torch.sigmoid(self.param_heads[eff_name](h))[0, :n]

            # Build param dict using EFFECT_CATALOG param names
            param_keys = list(EFFECT_CATALOG[eff_name].params.keys())
            params_dict_normalized = {k: float(v) for k, v in zip(param_keys, param_vals)}

            results.append((eff_name, params_dict_normalized))

            # Next-step input: predicted token + predicted params
            next_params = torch.zeros(1, self.max_params_per_step, device=device)
            next_params[0, :n] = param_vals
            token = torch.tensor([eff_idx], dtype=torch.long, device=device)
            params_in = next_params

        return results

    def infer_to_params_dict(
        self,
        style_label: torch.Tensor,
    ) -> Dict[str, Dict[str, float]]:
        """
        Infer and return a params_dict compatible with PedalboardRenderer.render().
        Params are in normalized [0, 1] space — caller must denormalize if needed.
        """
        chain = self.infer(style_label)
        return {eff: params for eff, params in chain}

    def get_model_config(self) -> Dict:
        return {
            "model_type": "ar_controller",
            "effect_names": self.effect_names,
            "style_vocab_size": self.style_vocab_size,
            "condition_dim": self.condition_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": next(
                (float(m.p) for m in self.modules() if isinstance(m, nn.Dropout)),
                0.0,
            ),
            "max_steps": self.max_steps,
        }
