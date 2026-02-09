"""
DSP Parameter Decoder

Cross-modal decoder that predicts DSP parameters from audio embeddings
conditioned on text descriptions (image effect correspondences).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class CrossAttention(nn.Module):
    """Cross-attention layer for conditioning audio on text."""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, audio_embed: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_embed: (B, embed_dim) audio embeddings (query)
            text_embed: (B, embed_dim) text embeddings (key, value)

        Returns:
            attended: (B, embed_dim) audio conditioned on text
        """
        # Add sequence dimension if not present
        if audio_embed.dim() == 2:
            audio_embed = audio_embed.unsqueeze(1)  # (B, 1, embed_dim)
        if text_embed.dim() == 2:
            text_embed = text_embed.unsqueeze(1)  # (B, 1, embed_dim)

        # Cross-attention: audio queries text
        attended, _ = self.attention(
            query=audio_embed,
            key=text_embed,
            value=text_embed,
        )

        # Residual connection + layer norm
        output = self.norm(audio_embed + self.dropout(attended))

        # Remove sequence dimension
        if output.size(1) == 1:
            output = output.squeeze(1)

        return output


class DSPParameterDecoder(nn.Module):
    """
    Decoder that predicts DSP parameters from audio + text condition.

    Architecture:
        1. Cross-attention: Condition audio embedding on text embedding
        2. MLP: Regress DSP parameters
    """

    def __init__(
        self,
        audio_embed_dim: int = 512,  # CLAP audio embedding
        text_embed_dim: int = 512,   # CLAP text embedding
        hidden_dims: List[int] = [512, 256, 128],
        num_heads: int = 8,
        dropout: float = 0.1,
        dsp_param_specs: Optional[Dict] = None,
    ):
        """
        Args:
            audio_embed_dim: Dimension of audio embeddings (CLAP)
            text_embed_dim: Dimension of text embeddings (CLAP text)
            hidden_dims: List of hidden layer dimensions for MLP
            num_heads: Number of attention heads
            dropout: Dropout rate
            dsp_param_specs: Dictionary mapping effect names to parameter specifications
                Example:
                {
                    'lpf': {'cutoff_freq': (20, 20000, 'log')},
                    'highshelf': {'gain': (-20, 20, 'linear'), 'freq': (1000, 20000, 'log')},
                    ...
                }
        """
        super().__init__()

        self.audio_embed_dim = audio_embed_dim
        self.text_embed_dim = text_embed_dim
        self.dsp_param_specs = dsp_param_specs or self._default_param_specs()

        # Project text embedding to audio embedding dimension if needed
        if text_embed_dim != audio_embed_dim:
            self.text_projection = nn.Linear(text_embed_dim, audio_embed_dim)
        else:
            self.text_projection = nn.Identity()

        # Cross-attention layer
        self.cross_attention = CrossAttention(
            embed_dim=audio_embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # MLP for parameter regression
        mlp_layers = []
        in_dim = audio_embed_dim

        for hidden_dim in hidden_dims:
            mlp_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*mlp_layers)

        # Parameter prediction heads (one per effect type)
        self.param_heads = nn.ModuleDict()
        for effect_name, param_dict in self.dsp_param_specs.items():
            num_params = len(param_dict)
            self.param_heads[effect_name] = nn.Linear(in_dim, num_params)

    def _default_param_specs(self) -> Dict:
        """Default DSP parameter specifications."""
        return {
            'lpf': {
                'cutoff_freq': (20.0, 20000.0, 'log'),  # Hz
            },
            'highshelf': {
                'gain': (-20.0, 20.0, 'linear'),  # dB
                'freq': (1000.0, 20000.0, 'log'),  # Hz
            },
            'saturation': {
                'drive': (0.0, 40.0, 'linear'),  # dB
            },
            'reverb': {
                'room_size': (0.0, 1.0, 'linear'),
                'damping': (0.0, 1.0, 'linear'),
                'wet_level': (0.0, 1.0, 'linear'),
            },
        }

    def forward(
        self,
        audio_embed: torch.Tensor,
        text_embed: torch.Tensor,
        effect_name: str,
    ) -> torch.Tensor:
        """
        Predict DSP parameters.

        Args:
            audio_embed: (B, audio_embed_dim) audio embeddings
            text_embed: (B, text_embed_dim) text condition embeddings
            effect_name: Name of the DSP effect (e.g., 'lpf', 'highshelf')

        Returns:
            params: (B, num_params) predicted parameters in [0, 1] range
        """
        # Project text embedding if needed
        text_embed = self.text_projection(text_embed)

        # Cross-attention: condition audio on text
        conditioned = self.cross_attention(audio_embed, text_embed)

        # MLP feature extraction
        features = self.mlp(conditioned)

        # Predict parameters for the specified effect
        if effect_name not in self.param_heads:
            raise ValueError(f"Unknown effect: {effect_name}. Available: {list(self.param_heads.keys())}")

        params = self.param_heads[effect_name](features)

        # Apply sigmoid to get [0, 1] range
        params = torch.sigmoid(params)

        return params

    def denormalize_params(
        self,
        params: torch.Tensor,
        effect_name: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Denormalize predicted parameters from [0, 1] to actual ranges.

        Args:
            params: (B, num_params) normalized parameters in [0, 1]
            effect_name: Name of the DSP effect

        Returns:
            Dictionary mapping parameter names to denormalized values
        """
        if effect_name not in self.dsp_param_specs:
            raise ValueError(f"Unknown effect: {effect_name}")

        param_dict = self.dsp_param_specs[effect_name]
        param_names = list(param_dict.keys())

        denormalized = {}
        for i, param_name in enumerate(param_names):
            min_val, max_val, scale_type = param_dict[param_name]

            # Get normalized value
            norm_val = params[:, i]

            # Denormalize based on scale type
            if scale_type == 'linear':
                denorm_val = norm_val * (max_val - min_val) + min_val
            elif scale_type == 'log':
                # Log scale: interpolate in log space
                log_min = torch.log(torch.tensor(min_val))
                log_max = torch.log(torch.tensor(max_val))
                log_val = norm_val * (log_max - log_min) + log_min
                denorm_val = torch.exp(log_val)
            else:
                raise ValueError(f"Unknown scale type: {scale_type}")

            denormalized[param_name] = denorm_val

        return denormalized

    def normalize_params(
        self,
        param_dict: Dict[str, torch.Tensor],
        effect_name: str,
    ) -> torch.Tensor:
        """
        Normalize ground-truth parameters from actual ranges to [0, 1].

        Args:
            param_dict: Dictionary mapping parameter names to actual values
            effect_name: Name of the DSP effect

        Returns:
            params: (B, num_params) normalized parameters in [0, 1]
        """
        if effect_name not in self.dsp_param_specs:
            raise ValueError(f"Unknown effect: {effect_name}")

        spec_dict = self.dsp_param_specs[effect_name]
        param_names = list(spec_dict.keys())

        normalized = []
        for param_name in param_names:
            if param_name not in param_dict:
                raise ValueError(f"Missing parameter: {param_name}")

            min_val, max_val, scale_type = spec_dict[param_name]
            actual_val = param_dict[param_name]

            # Normalize based on scale type
            if scale_type == 'linear':
                norm_val = (actual_val - min_val) / (max_val - min_val)
            elif scale_type == 'log':
                # Log scale: interpolate in log space
                log_min = torch.log(torch.tensor(min_val))
                log_max = torch.log(torch.tensor(max_val))
                log_val = torch.log(actual_val)
                norm_val = (log_val - log_min) / (log_max - log_min)
            else:
                raise ValueError(f"Unknown scale type: {scale_type}")

            # Clamp to [0, 1]
            norm_val = torch.clamp(norm_val, 0.0, 1.0)
            normalized.append(norm_val)

        # Stack along last dimension
        return torch.stack(normalized, dim=-1)
