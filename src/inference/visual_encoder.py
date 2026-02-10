"""
Visual Encoder (Phase C - Step 1)

Siamese architecture for visual delta encoding:
    1. CLIP(I)    -> (768,)
    2. CLIP(I')   -> (768,)
    3. CLIP(I'-I) -> (768,)  [difference image through CLIP]
    4. Concatenate -> (2304,)
    5. Projection layer -> (projection_dim,)
    6. L2 normalize -> z_visual

Self-supervised contrastive learning is a placeholder for now.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from PIL import Image


class VisualEncoder(nn.Module):
    """
    Encodes visual deltas (I, I') into a style-descriptive vector z_visual.

    Uses frozen CLIP for feature extraction + a trainable projection layer.
    """

    def __init__(
        self,
        clip_embedder,
        projection_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.clip = clip_embedder  # frozen, not registered as parameter
        self.clip_dim = clip_embedder.embedding_dim  # 768
        self.projection_dim = projection_dim

        concat_dim = self.clip_dim * 3  # [CLIP(I), CLIP(I'), CLIP(I'-I)]

        self.projection = nn.Sequential(
            nn.Linear(concat_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, projection_dim),
            nn.LayerNorm(projection_dim),
        )

        # Placeholder: contrastive learning head
        self.contrastive_head = nn.Linear(projection_dim, 128)

    @torch.no_grad()
    def _encode_images(
        self,
        original: torch.Tensor,
        edited: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract CLIP embeddings for original, edited, and difference images.

        Args:
            original: (B, 3, H, W) in [0, 1]
            edited: (B, 3, H, W) in [0, 1]

        Returns:
            clip_orig, clip_edit, clip_diff: each (B, 768)
        """
        diff = edited - original
        # Normalize diff to [0, 1] for CLIP: (diff + 1) / 2
        diff_normalized = torch.clamp((diff + 1.0) / 2.0, 0.0, 1.0)

        clip_orig = self.clip.embed_images(original)
        clip_edit = self.clip.embed_images(edited)
        clip_diff = self.clip.embed_images(diff_normalized)

        return clip_orig, clip_edit, clip_diff

    def forward(
        self,
        original: torch.Tensor,
        edited: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode visual delta into z_visual.

        Args:
            original: (B, 3, H, W) in [0, 1]
            edited: (B, 3, H, W) in [0, 1]

        Returns:
            z_visual: (B, projection_dim) L2-normalized
        """
        clip_orig, clip_edit, clip_diff = self._encode_images(original, edited)
        concat = torch.cat([clip_orig, clip_edit, clip_diff], dim=-1)
        z_visual = self.projection(concat)
        return nn.functional.normalize(z_visual, p=2, dim=-1)

    def encode_from_paths(
        self,
        original_path: str,
        edited_path: str,
    ) -> np.ndarray:
        """Convenience: encode single image pair from file paths."""
        orig_img = Image.open(original_path).convert("RGB")
        edit_img = Image.open(edited_path).convert("RGB")

        orig_tensor = self.clip.preprocess(orig_img).unsqueeze(0)
        edit_tensor = self.clip.preprocess(edit_img).unsqueeze(0)

        device = next(self.projection.parameters()).device
        orig_tensor = orig_tensor.to(device)
        edit_tensor = edit_tensor.to(device)

        with torch.no_grad():
            z_visual = self.forward(orig_tensor, edit_tensor)

        return z_visual[0].cpu().numpy()

    # === Placeholder: contrastive learning ===

    def contrastive_forward(
        self,
        original: torch.Tensor,
        edited: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for contrastive learning (placeholder).

        Positive pairs: same effect type, different images.
        Negative pairs: different effect types.
        Loss: InfoNCE.
        """
        z_visual = self.forward(original, edited)
        z_contrast = self.contrastive_head(z_visual)
        return nn.functional.normalize(z_contrast, p=2, dim=-1)
