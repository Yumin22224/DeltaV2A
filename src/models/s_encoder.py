"""
S Encoder for Stage 1

Encodes audio pair (A_init, A_edit) into control signals S_pred
Used during Stage 1 to learn S_proxy space
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Tuple


class SEncoder(nn.Module):
    """
    S Encoder: (A_init, A_edit) → S_pred

    Architecture:
    - Input: Concatenated mel spectrograms (2 channels)
    - Backbone: ResNet18 CNN
    - Output: 6 heads × (t_h, g_h)
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        in_channels: int = 2,
        hidden_dim: int = 512,
        num_heads: int = 6,
        head_dim: int = 64,
    ):
        """
        Args:
            backbone: CNN backbone ('resnet18', 'resnet34')
            in_channels: Input channels (2 for concat mels)
            hidden_dim: Hidden feature dimension
            num_heads: Number of control heads
            head_dim: Dimension of each t_h (64)
        """
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim

        # Load ResNet backbone
        if backbone == "resnet18":
            resnet = models.resnet18(pretrained=True)
            feature_dim = 512
        elif backbone == "resnet34":
            resnet = models.resnet34(pretrained=True)
            feature_dim = 512
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Modify first conv layer for 2-channel input
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.cnn[0] = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Projection to hidden dim
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Per-head output layers
        self.head_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Linear(128, head_dim + 1),  # t_h (64) + g_h (1)
            )
            for _ in range(num_heads)
        ])

    def forward(
        self,
        A_init_mel: torch.Tensor,
        A_edit_mel: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode audio pair into control signals

        Args:
            A_init_mel: (B, 1, T, F) initial mel spectrogram
            A_edit_mel: (B, 1, T, F) edited mel spectrogram

        Returns:
            S_pred: List of 6 tuples (t_h, g_h)
                t_h: (B, head_dim)
                g_h: (B, 1)
        """
        # Concatenate along channel dimension
        x = torch.cat([A_init_mel, A_edit_mel], dim=1)  # (B, 2, T, F)

        # Extract features
        features = self.cnn(x)  # (B, feature_dim, 1, 1)
        features = features.flatten(1)  # (B, feature_dim)

        # Project to hidden dim
        hidden = self.projection(features)  # (B, hidden_dim)

        # Predict per-head outputs
        S_pred = []
        for h in range(self.num_heads):
            out = self.head_projectors[h](hidden)  # (B, head_dim + 1)

            t_h = out[:, :self.head_dim]  # (B, head_dim)
            g_h = torch.sigmoid(out[:, self.head_dim:self.head_dim+1])  # (B, 1)

            S_pred.append((t_h, g_h))

        return S_pred


# For testing
if __name__ == "__main__":
    # Test S encoder
    encoder = SEncoder(num_heads=6, head_dim=64)

    # Dummy inputs
    B = 4
    T, F = 800, 64  # Time, Frequency
    A_init = torch.randn(B, 1, T, F)
    A_edit = torch.randn(B, 1, T, F)

    # Forward
    S_pred = encoder(A_init, A_edit)

    print(f"Input shapes: {A_init.shape}, {A_edit.shape}")
    print(f"Output S_pred (6 heads):")
    for h, (t_h, g_h) in enumerate(S_pred):
        print(f"  Head {h}: t_h {t_h.shape}, g_h {g_h.shape}")
