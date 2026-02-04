"""
ImageBind Embedder Wrapper

Provides a clean interface for extracting embeddings from images and audio
using the pretrained ImageBind model.

Uses ImageBind's official preprocessing for consistency.
"""

import torch
import torch.nn as nn
import torchaudio
from torchvision import transforms
from typing import Dict, Optional, Tuple, List, Union
from pathlib import Path
import tempfile
import soundfile as sf


class ImageBindEmbedder(nn.Module):
    """
    Wrapper for ImageBind model to extract embeddings.

    ImageBind projects images and audio into a shared embedding space,
    which we use to compute delta vectors for cross-modal mapping.

    Uses ImageBind's official preprocessing functions for consistency.
    """

    def __init__(
        self,
        device: str = "cpu",
        freeze: bool = True,
    ):
        """
        Args:
            device: Device to load model on
            freeze: Whether to freeze ImageBind weights (recommended)
        """
        super().__init__()
        self.device = device
        self.freeze = freeze
        self.embed_dim = 1024  # ImageBind huge embedding dimension

        # Audio preprocessing params (from ImageBind)
        self.audio_sample_rate = 16000
        self.audio_num_mel_bins = 128
        self.audio_target_length = 204
        self.audio_mean = -4.268
        self.audio_std = 9.138

        # Load ImageBind
        self._load_imagebind()

        if freeze:
            self._freeze_model()

    def _load_imagebind(self):
        """Load pretrained ImageBind model"""
        try:
            import imagebind.models.imagebind_model as ib_model
            from imagebind.models.imagebind_model import ModalityType
            import imagebind.data as ib_data

            self.model = ib_model.imagebind_huge(pretrained=True)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.ModalityType = ModalityType
            self.ib_data = ib_data

            print("Loaded ImageBind successfully")

        except ImportError as e:
            raise ImportError(
                f"ImageBind not found. Install with: "
                f"pip install git+https://github.com/facebookresearch/ImageBind.git\n"
                f"Error: {e}"
            )

    def _freeze_model(self):
        """Freeze all ImageBind parameters"""
        for param in self.model.parameters():
            param.requires_grad = False

    def _waveform_to_melspec(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to mel spectrogram using ImageBind's method.

        Uses Kaldi-compatible fbank features (same as ImageBind).

        Args:
            waveform: (1, T) or (T,) tensor at 16kHz

        Returns:
            mel: (1, num_mel_bins, target_length) tensor
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Zero-mean normalization (ImageBind does this)
        waveform = waveform - waveform.mean()

        # Use Kaldi fbank (same as ImageBind)
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=self.audio_sample_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=self.audio_num_mel_bins,
            dither=0.0,
            frame_length=25,
            frame_shift=10,  # DEFAULT_AUDIO_FRAME_SHIFT_MS in ImageBind
        )

        # Convert to [mel_bins, num_frames] shape
        fbank = fbank.transpose(0, 1)

        # Pad/crop to target_length
        n_frames = fbank.size(1)
        p = self.audio_target_length - n_frames

        if p > 0:
            fbank = torch.nn.functional.pad(fbank, (0, p), mode="constant", value=0)
        elif p < 0:
            fbank = fbank[:, :self.audio_target_length]

        # Add channel dimension: (1, mel_bins, num_frames)
        fbank = fbank.unsqueeze(0)

        return fbank

    def preprocess_audio_waveforms(
        self,
        waveforms: torch.Tensor,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """
        Preprocess audio waveforms for ImageBind.

        Args:
            waveforms: (B, T) tensor of audio waveforms
            sample_rate: Sample rate of input (will resample to 16kHz if different)

        Returns:
            audio_input: (B, 1, 1, num_mel_bins, target_length) tensor
                        Ready for ImageBind model
        """
        if waveforms.dim() == 3:
            waveforms = waveforms.squeeze(1)

        B = waveforms.shape[0]

        # Resample if needed
        if sample_rate != self.audio_sample_rate:
            waveforms = torchaudio.functional.resample(
                waveforms, orig_freq=sample_rate, new_freq=self.audio_sample_rate
            )

        # Convert each waveform to mel spectrogram
        mel_specs = []
        for i in range(B):
            mel = self._waveform_to_melspec(waveforms[i].cpu())
            mel_specs.append(mel)

        # Stack: (B, 1, num_mel_bins, target_length)
        mel_batch = torch.stack(mel_specs, dim=0)

        # Normalize (ImageBind's normalization)
        normalize = transforms.Normalize(mean=self.audio_mean, std=self.audio_std)
        mel_batch = normalize(mel_batch)

        # ImageBind expects (B, clips, 1, mel_bins, frames)
        # For single clip: (B, 1, 1, mel_bins, frames)
        mel_batch = mel_batch.unsqueeze(1)

        return mel_batch.to(self.device)

    def preprocess_images(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Preprocess images for ImageBind.

        Applies CLIP-style normalization (used by ImageBind).

        Args:
            images: (B, 3, H, W) tensor (any normalization)

        Returns:
            images: (B, 3, 224, 224) tensor with CLIP normalization
        """
        # Resize and crop to 224x224
        if images.shape[-1] != 224 or images.shape[-2] != 224:
            images = torch.nn.functional.interpolate(
                images, size=(224, 224), mode='bicubic', align_corners=False
            )

        # Check if already normalized (values outside [0,1])
        # If so, denormalize from ImageNet first
        if images.min() < -0.5 or images.max() > 1.5:
            # Assume ImageNet normalization, denormalize
            imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
            imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
            images = images * imagenet_std + imagenet_mean

        # Apply CLIP normalization (used by ImageBind)
        clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(images.device)
        clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(images.device)
        images = (images - clip_mean) / clip_std

        return images.to(self.device)

    @torch.no_grad()
    def embed_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract image embeddings.

        Args:
            images: (B, 3, H, W) tensor

        Returns:
            embeddings: (B, embed_dim) tensor
        """
        images = self.preprocess_images(images)
        inputs = {self.ModalityType.VISION: images}
        embeddings = self.model(inputs)
        return embeddings[self.ModalityType.VISION]

    @torch.no_grad()
    def embed_audio(
        self,
        waveforms: torch.Tensor,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """
        Extract audio embeddings.

        Args:
            waveforms: (B, T) or (B, 1, T) tensor
            sample_rate: Sample rate of input waveforms

        Returns:
            embeddings: (B, embed_dim) tensor
        """
        audio_input = self.preprocess_audio_waveforms(waveforms, sample_rate)
        inputs = {self.ModalityType.AUDIO: audio_input}
        embeddings = self.model(inputs)
        return embeddings[self.ModalityType.AUDIO]

    @torch.no_grad()
    def embed_image_paths(self, image_paths: List[str]) -> torch.Tensor:
        """
        Extract image embeddings directly from file paths.

        Uses ImageBind's official preprocessing.

        Args:
            image_paths: List of image file paths

        Returns:
            embeddings: (B, embed_dim) tensor
        """
        images = self.ib_data.load_and_transform_vision_data(image_paths, self.device)
        inputs = {self.ModalityType.VISION: images}
        embeddings = self.model(inputs)
        return embeddings[self.ModalityType.VISION]

    @torch.no_grad()
    def embed_audio_paths(self, audio_paths: List[str]) -> torch.Tensor:
        """
        Extract audio embeddings directly from file paths.

        Uses ImageBind's official preprocessing.

        Args:
            audio_paths: List of audio file paths

        Returns:
            embeddings: (B, embed_dim) tensor
        """
        audio = self.ib_data.load_and_transform_audio_data(
            audio_paths,
            self.device,
            num_mel_bins=self.audio_num_mel_bins,
            target_length=self.audio_target_length,
            sample_rate=self.audio_sample_rate,
            clip_duration=2,
            clips_per_video=1,  # Single clip for our use case
            mean=self.audio_mean,
            std=self.audio_std,
        )
        inputs = {self.ModalityType.AUDIO: audio}
        embeddings = self.model(inputs)
        return embeddings[self.ModalityType.AUDIO]

    @torch.no_grad()
    def embed_pair(
        self,
        images: torch.Tensor,
        waveforms: torch.Tensor,
        sample_rate: int = 16000,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract embeddings for both modalities at once.

        Args:
            images: (B, 3, H, W) tensor
            waveforms: (B, T) or (B, 1, T) tensor
            sample_rate: Audio sample rate

        Returns:
            (image_embeddings, audio_embeddings): Both (B, embed_dim)
        """
        images = self.preprocess_images(images)
        audio_input = self.preprocess_audio_waveforms(waveforms, sample_rate)

        inputs = {
            self.ModalityType.VISION: images,
            self.ModalityType.AUDIO: audio_input,
        }
        embeddings = self.model(inputs)

        return (
            embeddings[self.ModalityType.VISION],
            embeddings[self.ModalityType.AUDIO],
        )

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        waveforms: Optional[torch.Tensor] = None,
        sample_rate: int = 16000,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for embedding extraction.

        Args:
            images: Optional (B, 3, H, W) tensor
            waveforms: Optional (B, T) tensor
            sample_rate: Audio sample rate

        Returns:
            Dict with 'image' and/or 'audio' embeddings
        """
        result = {}

        if images is not None:
            result['image'] = self.embed_image(images)

        if waveforms is not None:
            result['audio'] = self.embed_audio(waveforms, sample_rate)

        return result
