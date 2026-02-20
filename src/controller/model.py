"""
Audio Controller Network (Phase B)

Predicts DSP parameters from audio embedding + target style label.

Architecture:
    Input: [CLAP(A) (512d), style_label (|AUD_VOCAB|d)] -> concat
    Hidden: MLP with LayerNorm + ReLU + Dropout
    Output: (total_params,) in [0, 1] via sigmoid 
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


class AudioController(nn.Module):
    """
    Predicts DSP parameters from audio embedding + style label.

    Input:
        audio_embed: (B, 512) CLAP audio embedding
        style_label: (B, |V_aud|) soft label over AUD_VOCAB

    Output:
        params: (B, total_params) predicted DSP params in [0, 1]
    """

    def __init__(
        self,
        audio_embed_dim: int = 512,
        style_vocab_size: int = 20,
        total_params: int = 15,
        hidden_dims: List[int] = None,
        dropout: float = 0.1,
        use_activity_head: bool = False,
        num_effects: int = 0,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.audio_embed_dim = audio_embed_dim
        self.style_vocab_size = style_vocab_size
        self.total_params = total_params
        self.use_activity_head = bool(use_activity_head)
        self.num_effects = int(num_effects) if num_effects is not None else 0

        # When audio_embed_dim=0, the model operates in style-only mode:
        # only the 24-dim style label is used as input (no CLAP concatenation).
        input_dim = max(audio_embed_dim, 0) + style_vocab_size

        layers: List[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.param_head = nn.Linear(in_dim, total_params)
        if self.use_activity_head:
            if self.num_effects <= 0:
                raise ValueError("num_effects must be > 0 when use_activity_head=True")
            self.activity_head = nn.Linear(in_dim, self.num_effects)
        else:
            self.activity_head = None

    def forward(
        self,
        audio_embed: torch.Tensor,
        style_label: torch.Tensor,
    ) -> torch.Tensor:
        if self.audio_embed_dim > 0:
            x = torch.cat([audio_embed, style_label], dim=-1)
        else:
            x = style_label
        feat = self.backbone(x)
        params = torch.sigmoid(self.param_head(feat))
        return params

    def forward_with_activity(
        self,
        audio_embed: torch.Tensor,
        style_label: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.audio_embed_dim > 0:
            x = torch.cat([audio_embed, style_label], dim=-1)
        else:
            x = style_label
        feat = self.backbone(x)
        params = torch.sigmoid(self.param_head(feat))
        activity_logits = None
        if self.activity_head is not None:
            activity_logits = self.activity_head(feat)
        return params, activity_logits

    def get_model_config(self) -> Dict:
        return {
            "audio_embed_dim": self.audio_embed_dim,
            "style_vocab_size": self.style_vocab_size,
            "total_params": self.total_params,
            "hidden_dims": [
                m.out_features
                for m in self.backbone
                if isinstance(m, nn.Linear)
            ],
            "dropout": next(
                (float(m.p) for m in self.backbone if isinstance(m, nn.Dropout)),
                0.0,
            ),
            "use_activity_head": self.use_activity_head,
            "num_effects": self.num_effects,
        }
