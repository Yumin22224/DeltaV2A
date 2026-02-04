"""
Delta Mapper: FiLM-based ΔV → ΔA Mapping

Core hypothesis: Visual delta can be mapped to audio delta
with context modulation from initial states.

Architecture:
    h_delta = MLP_enc(ΔV)
    h_context = MLP_ctx(z_A_init ⊕ z_I_init)
    γ, β = MLP_mod(h_context)
    Δ̂A = γ ⊙ h_delta + β
"""

import torch
import torch.nn as nn
from typing import Tuple


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer"""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply FiLM modulation: γ ⊙ x + β

        Args:
            x: (B, D) features to modulate
            gamma: (B, D) scaling factor
            beta: (B, D) shift factor

        Returns:
            (B, D) modulated features
        """
        return gamma * x + beta


class DeltaMapper(nn.Module):
    """
    Maps visual delta to audio delta using context modulation.

    The key insight is that the same visual change should produce
    different audio changes depending on the initial context
    (where we start in the cross-modal manifold).
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        """
        Args:
            embed_dim: ImageBind embedding dimension (1024)
            hidden_dim: Hidden layer dimension
            num_layers: Number of layers in each MLP
            dropout: Dropout rate
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Delta encoder: ΔV → h_delta
        # Encodes the visual change direction
        self.delta_encoder = self._build_mlp(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Context encoder: [z_A_init, z_I_init] → h_context
        # Encodes the initial cross-modal alignment state
        self.context_encoder = self._build_mlp(
            input_dim=embed_dim * 2,  # Concatenated audio + image
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Modulation network: h_context → (γ, β)
        # Determines how to scale/shift the delta based on context
        self.modulation_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),  # Output both γ and β
        )

        # FiLM layer
        self.film = FiLMLayer(hidden_dim)

        # Output projection: h_delta (modulated) → ΔA
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Initialize modulation to identity (γ=1, β=0) at start
        self._init_modulation()

    def _build_mlp(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
    ) -> nn.Module:
        """Build MLP with residual connections"""
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]

        for _ in range(num_layers - 2):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])

        layers.append(nn.Linear(hidden_dim, output_dim))

        return nn.Sequential(*layers)

    def _init_modulation(self):
        """Initialize modulation to identity mapping"""
        # Initialize last layer of modulation_net
        # γ should start at 1, β should start at 0
        last_layer = self.modulation_net[-1]
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)
        # Set γ part of bias to 1
        last_layer.bias.data[:self.hidden_dim] = 1.0

    def forward(
        self,
        delta_v: torch.Tensor,
        z_a_init: torch.Tensor,
        z_i_init: torch.Tensor,
    ) -> torch.Tensor:
        """
        Map visual delta to predicted audio delta.

        Args:
            delta_v: (B, embed_dim) visual embedding delta (z_I_edit - z_I_init)
            z_a_init: (B, embed_dim) initial audio embedding
            z_i_init: (B, embed_dim) initial image embedding

        Returns:
            delta_a_pred: (B, embed_dim) predicted audio embedding delta
        """
        # 1. Encode delta
        h_delta = self.delta_encoder(delta_v)  # (B, hidden_dim)

        # 2. Encode context (concatenate initial states)
        context = torch.cat([z_a_init, z_i_init], dim=-1)  # (B, 2*embed_dim)
        h_context = self.context_encoder(context)  # (B, hidden_dim)

        # 3. Compute modulation parameters
        modulation = self.modulation_net(h_context)  # (B, 2*hidden_dim)
        gamma = modulation[:, :self.hidden_dim]  # (B, hidden_dim)
        beta = modulation[:, self.hidden_dim:]  # (B, hidden_dim)

        # 4. Apply FiLM modulation
        h_modulated = self.film(h_delta, gamma, beta)  # (B, hidden_dim)

        # 5. Project to output
        delta_a_pred = self.output_proj(h_modulated)  # (B, embed_dim)

        return delta_a_pred

    def get_modulation_params(
        self,
        z_a_init: torch.Tensor,
        z_i_init: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get modulation parameters for analysis/visualization.

        Returns:
            (gamma, beta): Both (B, hidden_dim)
        """
        context = torch.cat([z_a_init, z_i_init], dim=-1)
        h_context = self.context_encoder(context)
        modulation = self.modulation_net(h_context)

        gamma = modulation[:, :self.hidden_dim]
        beta = modulation[:, self.hidden_dim:]

        return gamma, beta


class DeltaLoss(nn.Module):
    """
    Combined loss for delta mapping.

    L = λ_l2 * L2_loss + λ_cos * Cosine_loss
    """

    def __init__(
        self,
        l2_weight: float = 1.0,
        cosine_weight: float = 1.0,
    ):
        super().__init__()
        self.l2_weight = l2_weight
        self.cosine_weight = cosine_weight
        self.cosine_sim = nn.CosineSimilarity(dim=-1)

    def forward(
        self,
        delta_a_pred: torch.Tensor,
        delta_a_true: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.

        Args:
            delta_a_pred: (B, D) predicted audio delta
            delta_a_true: (B, D) ground truth audio delta

        Returns:
            total_loss: scalar
            loss_dict: breakdown of losses
        """
        # L2 loss (MSE)
        l2_loss = torch.mean((delta_a_pred - delta_a_true) ** 2)

        # Cosine loss (1 - cosine_similarity)
        cosine_sim = self.cosine_sim(delta_a_pred, delta_a_true)
        cosine_loss = 1.0 - cosine_sim.mean()

        # Combined
        total_loss = self.l2_weight * l2_loss + self.cosine_weight * cosine_loss

        loss_dict = {
            'total': total_loss.item(),
            'l2': l2_loss.item(),
            'cosine': cosine_loss.item(),
            'cosine_sim_mean': cosine_sim.mean().item(),
        }

        return total_loss, loss_dict
