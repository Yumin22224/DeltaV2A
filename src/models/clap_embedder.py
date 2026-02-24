"""
CLAP Embedder for Audio Embeddings

Wrapper for LAION-CLAP Music model (512-dimensional embeddings).
Handles variable-length audio via truncation/padding for batch processing.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import librosa
import soundfile as sf


class CLAPEmbedder(nn.Module):
    """
    LAION-CLAP embedder for audio.

    Uses LAION's music-specific CLAP model to extract 512-dimensional
    audio embeddings at 48kHz sample rate.
    """

    def __init__(
        self,
        model_id: int = 1,
        enable_fusion: bool = False,
        device: str = "cpu",
        max_duration: float = 20.0,
    ):
        """
        Args:
            model_id: CLAP checkpoint ID
                0: 630k non-fusion
                1: 630k+audioset non-fusion (recommended for music)
                2: 630k fusion
                3: 630k+audioset fusion
            enable_fusion: Use fusion model (slower but better)
            device: Device to load model on
            max_duration: Maximum audio duration in seconds (for truncate/pad)
        """
        super().__init__()
        self.device = device if device != "mps" else "cpu"  # CLAP doesn't support MPS
        self.model_id = model_id
        self.enable_fusion = enable_fusion
        self.max_duration = max_duration
        self.sample_rate = 48000  # CLAP uses 48kHz
        self.embed_dim = 512  # CLAP embedding dimension

        self._load_clap()

    def _load_clap(self):
        """Load CLAP model."""
        try:
            from laion_clap import CLAP_Module

            self.model = CLAP_Module(
                enable_fusion=self.enable_fusion,
                device=self.device if self.device != "mps" else None,
                amodel='HTSAT-tiny',
                tmodel='roberta',
            )

            # Load pretrained checkpoint
            self.model.load_ckpt(model_id=self.model_id)

            print(f"Loaded LAION-CLAP (model_id={self.model_id}, "
                  f"fusion={self.enable_fusion}) successfully")

        except ImportError as e:
            raise ImportError(
                f"laion_clap not found. Install with: pip install laion-clap\n"
                f"Error: {e}"
            )

    def preprocess_audio(
        self,
        audio_path: str,
    ) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess audio file.

        Critical: Truncates or pads to max_duration for consistent batch processing.

        Args:
            audio_path: Path to audio file

        Returns:
            waveform: (T,) numpy array at 48kHz
            sample_rate: 48000
        """
        # Load audio at 48kHz
        waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        # Truncate or pad to max_duration
        target_length = int(self.max_duration * self.sample_rate)

        if len(waveform) > target_length:
            # Truncate
            waveform = waveform[:target_length]
        elif len(waveform) < target_length:
            # Pad with zeros
            pad_length = target_length - len(waveform)
            waveform = np.pad(waveform, (0, pad_length), mode='constant')

        return waveform, self.sample_rate

    @torch.no_grad()
    def embed_audio_paths(self, audio_paths: List[str]) -> torch.Tensor:
        """
        Extract audio embeddings from file paths.

        Args:
            audio_paths: List of audio file paths

        Returns:
            embeddings: (B, 512) tensor
        """
        # Preprocess all audio files
        waveforms = []
        for path in audio_paths:
            waveform, _ = self.preprocess_audio(path)
            waveforms.append(waveform)

        # Stack into batch: (B, T)
        waveforms = np.stack(waveforms, axis=0)

        # Extract embeddings (returns numpy array)
        embeddings_np = self.model.get_audio_embedding_from_data(
            x=waveforms
        )

        # Convert to torch tensor
        embeddings = torch.from_numpy(embeddings_np).float()

        return embeddings

    @torch.no_grad()
    def embed_audio(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 48000,
    ) -> torch.Tensor:
        """
        Extract audio embeddings from waveform tensors.

        Args:
            waveform: (B, T) or (T,) tensor
            sample_rate: Sample rate of input (will resample if different)

        Returns:
            embeddings: (B, 512) tensor
        """
        # Handle single waveform
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Convert to numpy
        waveform_np = waveform.cpu().numpy()

        # Resample if needed
        if sample_rate != self.sample_rate:
            import librosa
            waveform_list = []
            for i in range(waveform_np.shape[0]):
                resampled = librosa.resample(
                    waveform_np[i],
                    orig_sr=sample_rate,
                    target_sr=self.sample_rate,
                )
                waveform_list.append(resampled)
            waveform_np = np.stack(waveform_list, axis=0)

        # Truncate/pad to max_duration
        target_length = int(self.max_duration * self.sample_rate)
        if waveform_np.shape[1] > target_length:
            waveform_np = waveform_np[:, :target_length]
        elif waveform_np.shape[1] < target_length:
            pad_length = target_length - waveform_np.shape[1]
            waveform_np = np.pad(waveform_np, ((0, 0), (0, pad_length)), mode='constant')

        # Extract embeddings
        embeddings_np = self.model.get_audio_embedding_from_data(
            x=waveform_np
        )

        embeddings = torch.from_numpy(embeddings_np).float()

        return embeddings

    @torch.no_grad()
    def embed_text(self, texts: List[str]) -> torch.Tensor:
        """
        Extract text embeddings.

        Args:
            texts: List of text strings

        Returns:
            embeddings: (B, 512) tensor
        """
        # Extract embeddings (returns numpy array)
        embeddings_np = self.model.get_text_embedding(texts)

        # Convert to torch tensor
        embeddings = torch.from_numpy(embeddings_np).float()

        return embeddings

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self.embed_dim
