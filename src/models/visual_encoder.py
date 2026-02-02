"""
Visual Delta Encoder

Encodes the difference between I_init and I_edit into a latent delta vector ΔV
Combines low-level features (ResNet) and high-level semantics (CLIP)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple


class LowLevelEncoder(nn.Module):
    """
    Encodes low-level visual features using ResNet
    Captures blur, contrast, color changes, etc.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        output_dim: int = 256,
        pretrained: bool = True,
    ):
        """
        Args:
            backbone: ResNet variant ('resnet18', 'resnet34', 'resnet50')
            output_dim: Dimension of output features
            pretrained: Whether to use ImageNet pretrained weights
        """
        super().__init__()

        # Load ResNet backbone
        if backbone == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == "resnet34":
            resnet = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif backbone == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Remove final FC layer
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W)

        Returns:
            features: (B, output_dim)
        """
        # Extract features
        features = self.encoder(images)  # (B, feature_dim, 1, 1)
        features = features.flatten(1)  # (B, feature_dim)

        # Project
        features = self.projection(features)  # (B, output_dim)

        return features


class HighLevelEncoder(nn.Module):
    """
    Encodes high-level semantic features using CLIP
    Captures object identity, scene context, mood, etc.
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        output_dim: int = 512,
        freeze: bool = True,
    ):
        """
        Args:
            model_name: CLIP model variant
            output_dim: Output dimension (typically CLIP's native dim)
            freeze: Whether to freeze CLIP weights
        """
        super().__init__()

        self.output_dim = output_dim
        self.freeze = freeze

        # Load CLIP
        try:
            import open_clip

            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='openai'
            )

            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False

        except ImportError:
            print("Warning: open_clip not available, using dummy encoder")
            self.model = None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W) - already normalized

        Returns:
            features: (B, output_dim)
        """
        if self.model is None:
            # Dummy implementation
            B = images.shape[0]
            return torch.randn(B, self.output_dim, device=images.device)

        # Extract CLIP features
        if self.freeze:
            with torch.no_grad():
                features = self.model.encode_image(images)
        else:
            features = self.model.encode_image(images)

        # Normalize
        features = F.normalize(features, dim=-1)

        return features


class VisualDeltaEncoder(nn.Module):
    """
    Main Visual Delta Encoder
    Combines low-level and high-level encoders to produce ΔV
    """

    def __init__(
        self,
        low_level_config: dict = None,
        high_level_config: dict = None,
        delta_dim: int = 512,
    ):
        """
        Args:
            low_level_config: Config for LowLevelEncoder
            high_level_config: Config for HighLevelEncoder
            delta_dim: Dimension of final delta vector
        """
        super().__init__()

        # Default configs
        low_level_config = low_level_config or {
            'backbone': 'resnet18',
            'output_dim': 256,
            'pretrained': True,
        }
        high_level_config = high_level_config or {
            'model_name': 'ViT-B/32',
            'output_dim': 512,
            'freeze': True,
        }

        # Encoders
        self.low_level_encoder = LowLevelEncoder(**low_level_config)
        self.high_level_encoder = HighLevelEncoder(**high_level_config)

        # Dimensions
        self.low_dim = low_level_config['output_dim']
        self.high_dim = high_level_config['output_dim']
        self.delta_dim = delta_dim

        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(self.low_dim + self.high_dim, delta_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(delta_dim, delta_dim),
        )

    def encode_single(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a single image into low and high level features

        Args:
            image: (B, 3, H, W)

        Returns:
            low_features: (B, low_dim)
            high_features: (B, high_dim)
        """
        low_features = self.low_level_encoder(image)
        high_features = self.high_level_encoder(image)

        return low_features, high_features

    def compute_delta(
        self,
        image_init: torch.Tensor,
        image_edit: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute delta vector from image pair

        Args:
            image_init: (B, 3, H, W)
            image_edit: (B, 3, H, W)

        Returns:
            delta_V: (B, delta_dim)
        """
        # Encode both images
        low_init, high_init = self.encode_single(image_init)
        low_edit, high_edit = self.encode_single(image_edit)

        # Compute deltas
        delta_low = low_edit - low_init  # (B, low_dim)
        delta_high = high_edit - high_init  # (B, high_dim)

        # Concatenate
        delta_concat = torch.cat([delta_low, delta_high], dim=-1)

        # Fuse
        delta_V = self.fusion(delta_concat)

        return delta_V

    def forward(
        self,
        image_init: torch.Tensor,
        image_edit: torch.Tensor,
    ) -> torch.Tensor:
        """
        Main forward pass

        Args:
            image_init: (B, 3, H, W)
            image_edit: (B, 3, H, W)

        Returns:
            delta_V: (B, delta_dim)
        """
        return self.compute_delta(image_init, image_edit)


# For testing
if __name__ == "__main__":
    # Test visual delta encoder
    encoder = VisualDeltaEncoder(delta_dim=512)

    # Dummy inputs
    B = 4
    image_init = torch.randn(B, 3, 512, 512)
    image_edit = torch.randn(B, 3, 512, 512)

    # Forward pass
    delta_V = encoder(image_init, image_edit)

    print(f"Input shapes: {image_init.shape}, {image_edit.shape}")
    print(f"Output delta_V shape: {delta_V.shape}")
    print(f"Expected: ({B}, 512)")
