"""
Multimodal Embedder

Unified interface for CLIP (image) and CLAP (audio) embeddings.
"""

import torch
from typing import List
from .clip_embedder import CLIPEmbedder
from .clap_embedder import CLAPEmbedder


class MultimodalEmbedder:
    """
    Unified embedder managing both CLIP and CLAP.

    Does NOT align embeddings internally - alignment is handled separately.
    """

    def __init__(
        self,
        clip_embedder: CLIPEmbedder,
        clap_embedder: CLAPEmbedder,
    ):
        """
        Args:
            clip_embedder: CLIP embedder for images
            clap_embedder: CLAP embedder for audio
        """
        self.clip = clip_embedder
        self.clap = clap_embedder

    def embed_image_paths(self, paths: List[str]) -> torch.Tensor:
        """
        Extract image embeddings.

        Args:
            paths: List of image file paths

        Returns:
            embeddings: (B, 768) tensor
        """
        return self.clip.embed_image_paths(paths)

    def embed_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract image embeddings from tensors.

        Args:
            images: (B, 3, H, W) tensor

        Returns:
            embeddings: (B, 768) tensor
        """
        return self.clip.embed_images(images)

    def embed_audio_paths(self, paths: List[str]) -> torch.Tensor:
        """
        Extract audio embeddings.

        Args:
            paths: List of audio file paths

        Returns:
            embeddings: (B, 512) tensor
        """
        return self.clap.embed_audio_paths(paths)

    def embed_audio(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 48000,
    ) -> torch.Tensor:
        """
        Extract audio embeddings from waveforms.

        Args:
            waveform: (B, T) or (T,) tensor
            sample_rate: Sample rate

        Returns:
            embeddings: (B, 512) tensor
        """
        return self.clap.embed_audio(waveform, sample_rate)

    def embed_image_text(self, texts: List[str]) -> torch.Tensor:
        """
        Extract image-domain text embeddings using CLIP.

        Args:
            texts: List of text strings

        Returns:
            embeddings: (B, 768) tensor
        """
        return self.clip.embed_text(texts)

    def embed_audio_text(self, texts: List[str]) -> torch.Tensor:
        """
        Extract audio-domain text embeddings using CLAP.

        Args:
            texts: List of text strings

        Returns:
            embeddings: (B, 512) tensor
        """
        return self.clap.embed_text(texts)

    @property
    def image_dim(self) -> int:
        """Image embedding dimension (768)."""
        return self.clip.embedding_dim

    @property
    def audio_dim(self) -> int:
        """Audio embedding dimension (512)."""
        return self.clap.embedding_dim

    @property
    def audio_sample_rate(self) -> int:
        """Audio sample rate (48000)."""
        return self.clap.sample_rate

    @property
    def audio_max_duration(self) -> float:
        """Audio max duration for truncate/pad."""
        return self.clap.max_duration
