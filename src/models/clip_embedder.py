"""
CLIP Embedder for Image Embeddings

Wrapper for OpenAI CLIP ViT-L/14 model (768-dimensional embeddings).
"""

import torch
import torch.nn as nn
from typing import List, Union
from pathlib import Path
from PIL import Image


class CLIPEmbedder(nn.Module):
    """
    CLIP (ViT-L/14) embedder for images.

    Uses OpenAI's pretrained CLIP model to extract 768-dimensional
    image embeddings.
    """

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: str = "cpu",
    ):
        """
        Args:
            model_name: CLIP model architecture (default: ViT-L-14)
            pretrained: Pretrained weights source (default: openai)
            device: Device to load model on
        """
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained
        self.embed_dim = 768  # ViT-L/14 embedding dimension

        self._load_clip()

    def _load_clip(self):
        """Load CLIP model and preprocessing pipeline."""
        try:
            import open_clip

            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self.device,
            )
            self.model.eval()

            print(f"Loaded CLIP {self.model_name} ({self.pretrained}) successfully")

        except ImportError as e:
            raise ImportError(
                f"open_clip not found. Install with: pip install open-clip-torch\n"
                f"Error: {e}"
            )

    @torch.no_grad()
    def embed_image_paths(self, image_paths: List[str]) -> torch.Tensor:
        """
        Extract image embeddings from file paths.

        Args:
            image_paths: List of image file paths

        Returns:
            embeddings: (B, 768) tensor
        """
        images = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            img_tensor = self.preprocess(img)
            images.append(img_tensor)

        # Stack and move to device
        images = torch.stack(images).to(self.device)

        # Extract embeddings
        embeddings = self.model.encode_image(images)

        return embeddings

    @torch.no_grad()
    def embed_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract image embeddings from preprocessed tensors.

        Args:
            images: (B, 3, H, W) tensor (any size, will be resized to 224x224)

        Returns:
            embeddings: (B, 768) tensor
        """
        # Resize to 224x224 if needed
        if images.shape[-1] != 224 or images.shape[-2] != 224:
            images = torch.nn.functional.interpolate(
                images, size=(224, 224), mode='bicubic', align_corners=False
            )

        # Normalize using CLIP normalization
        images = self._normalize_images(images)

        images = images.to(self.device)

        # Extract embeddings
        embeddings = self.model.encode_image(images)

        return embeddings

    def _normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply CLIP normalization.

        Assumes input is in [0, 1] range.
        """
        # CLIP normalization constants
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

        mean = mean.to(images.device)
        std = std.to(images.device)

        return (images - mean) / std

    def preprocess_pil_images(self, pil_images: List[Image.Image]) -> torch.Tensor:
        """
        Preprocess PIL images for CLIP.

        Args:
            pil_images: List of PIL Image objects

        Returns:
            preprocessed: (B, 3, 224, 224) tensor
        """
        images = [self.preprocess(img) for img in pil_images]
        return torch.stack(images)

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self.embed_dim
