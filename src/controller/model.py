"""
Audio Controller Network (Phase B)

Predicts DSP parameters from audio embedding + target style label.

Fusion modes:
  - concat: legacy [CLAP, style] concatenation
  - gated_residual: style backbone + gated CLAP residual correction
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class AudioController(nn.Module):
    """
    Predicts DSP parameters from audio embedding + style label.

    Input:
        audio_embed: (B, 512) CLAP audio embedding
        style_label: (B, |V_aud|) soft label over AUD_VOCAB
    """

    def __init__(
        self,
        audio_embed_dim: int = 512,
        style_vocab_size: int = 20,
        total_params: int = 15,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        use_activity_head: bool = False,
        num_effects: int = 0,
        fusion_mode: str = "concat",
        audio_gate_bias: float = -2.0,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.audio_embed_dim = int(audio_embed_dim)
        self.style_vocab_size = int(style_vocab_size)
        self.total_params = int(total_params)
        self.use_activity_head = bool(use_activity_head)
        self.num_effects = int(num_effects) if num_effects is not None else 0
        self.fusion_mode = str(fusion_mode).strip().lower()
        self.audio_gate_bias = float(audio_gate_bias)
        if self.fusion_mode not in {"concat", "gated_residual"}:
            raise ValueError(
                f"Unsupported fusion_mode: {fusion_mode}. "
                "Expected one of ['concat', 'gated_residual']"
            )

        self._gated_enabled = bool(self.fusion_mode == "gated_residual" and self.audio_embed_dim > 0)
        input_dim = self.style_vocab_size if self._gated_enabled else (max(self.audio_embed_dim, 0) + self.style_vocab_size)

        layers: List[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        feat_dim = in_dim

        self.audio_residual_proj: Optional[nn.Module] = None
        self.audio_gate_proj: Optional[nn.Linear] = None
        if self._gated_enabled:
            self.audio_residual_proj = nn.Sequential(
                nn.Linear(self.audio_embed_dim, feat_dim),
                nn.LayerNorm(feat_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.audio_gate_proj = nn.Linear(feat_dim, feat_dim)

        self.param_head = nn.Linear(feat_dim, self.total_params)
        if self.use_activity_head:
            if self.num_effects <= 0:
                raise ValueError("num_effects must be > 0 when use_activity_head=True")
            self.activity_head = nn.Linear(feat_dim, self.num_effects)
        else:
            self.activity_head = None

    def _encode_features(
        self,
        audio_embed: torch.Tensor,
        style_label: torch.Tensor,
    ) -> torch.Tensor:
        if self._gated_enabled:
            # Style is the base signal; CLAP is a gated residual correction.
            style_feat = self.backbone(style_label)
            audio_feat = self.audio_residual_proj(audio_embed)
            gate = torch.sigmoid(self.audio_gate_proj(style_feat) + self.audio_gate_bias)
            return style_feat + gate * audio_feat

        if self.audio_embed_dim > 0:
            x = torch.cat([audio_embed, style_label], dim=-1)
        else:
            x = style_label
        return self.backbone(x)

    def forward(
        self,
        audio_embed: torch.Tensor,
        style_label: torch.Tensor,
    ) -> torch.Tensor:
        feat = self._encode_features(audio_embed, style_label)
        return torch.sigmoid(self.param_head(feat))

    def forward_with_activity(
        self,
        audio_embed: torch.Tensor,
        style_label: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        feat = self._encode_features(audio_embed, style_label)
        params = torch.sigmoid(self.param_head(feat))
        activity_logits = self.activity_head(feat) if self.activity_head is not None else None
        return params, activity_logits

    def get_model_config(self) -> Dict:
        return {
            "audio_embed_dim": self.audio_embed_dim,
            "style_vocab_size": self.style_vocab_size,
            "total_params": self.total_params,
            "hidden_dims": [m.out_features for m in self.backbone if isinstance(m, nn.Linear)],
            "dropout": next((float(m.p) for m in self.backbone if isinstance(m, nn.Dropout)), 0.0),
            "use_activity_head": self.use_activity_head,
            "num_effects": self.num_effects,
            "fusion_mode": self.fusion_mode,
            "audio_gate_bias": self.audio_gate_bias,
        }

