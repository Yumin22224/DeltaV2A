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
from typing import List, Optional


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
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.audio_embed_dim = audio_embed_dim
        self.style_vocab_size = style_vocab_size
        self.total_params = total_params

        input_dim = audio_embed_dim + style_vocab_size

        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, total_params))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        audio_embed: torch.Tensor,
        style_label: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([audio_embed, style_label], dim=-1)
        x = self.mlp(x)
        return torch.sigmoid(x)
