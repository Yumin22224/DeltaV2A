"""
Centralized model loading utilities for pretrained models (ImageBind, CLIP)

Avoids duplicated loading logic across prior.py, delta_c_predictor.py, etc.
Handles SSL certificate issues on macOS and CWD-relative weight paths.

Also provides ImageBind-compatible preprocessing for vision and audio:
- Vision: extracts 256 spatial patch tokens from ViT trunk (before CLS head)
- Audio: converts raw waveforms to Kaldi fbank spectrograms (128 mel bins, 204 frames)
"""

import os
import ssl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, Optional


# Module-level cache to avoid loading models multiple times
_imagebind_model = None
_clip_model = None

# Project root (two levels up from src/utils/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CHECKPOINT_DIR = _PROJECT_ROOT / ".checkpoints"

# ImageBind weight URL and filename
_IMAGEBIND_URL = "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth"
_IMAGEBIND_FILENAME = "imagebind_huge.pth"


def _get_ssl_context():
    """
    Create SSL context that works on macOS where system certs may not be found.
    Falls back to certifi bundle.
    """
    try:
        ctx = ssl.create_default_context()
        # Quick test - if this doesn't raise, system certs work
        ctx.load_default_certs()
        return ctx
    except Exception:
        pass

    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass

    return None


def _ensure_imagebind_weights() -> Path:
    """
    Ensure ImageBind weights exist at a known absolute path.

    ImageBind's source code uses a CWD-relative '.checkpoints/' path.
    We pre-download to project root so it works regardless of CWD,
    then symlink/copy to where ImageBind expects it.

    Returns:
        Path to the weight file
    """
    weight_path = _CHECKPOINT_DIR / _IMAGEBIND_FILENAME

    if weight_path.exists():
        return weight_path

    # Also check CWD-relative path (ImageBind's default)
    cwd_path = Path(".checkpoints") / _IMAGEBIND_FILENAME
    if cwd_path.exists():
        return cwd_path

    # Need to download
    print(f"Downloading ImageBind weights to {weight_path} ...")
    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    ssl_ctx = _get_ssl_context()
    if ssl_ctx is not None:
        import urllib.request
        req = urllib.request.Request(_IMAGEBIND_URL)
        with urllib.request.urlopen(req, context=ssl_ctx) as response:
            with open(weight_path, 'wb') as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
    else:
        # Last resort: use torch.hub which may fail on macOS
        torch.hub.download_url_to_file(_IMAGEBIND_URL, str(weight_path), progress=True)

    print(f"Downloaded ImageBind weights ({weight_path.stat().st_size / (1024**2):.0f} MB)")
    return weight_path


def _setup_imagebind_checkpoint_path():
    """
    Ensure ImageBind can find weights regardless of CWD.

    ImageBind hardcodes '.checkpoints/imagebind_huge.pth' relative to CWD.
    We create a symlink from CWD/.checkpoints -> project .checkpoints if needed.
    """
    project_weight = _CHECKPOINT_DIR / _IMAGEBIND_FILENAME
    cwd_checkpoints = Path(".checkpoints")
    cwd_weight = cwd_checkpoints / _IMAGEBIND_FILENAME

    if cwd_weight.exists():
        return

    if project_weight.exists():
        cwd_checkpoints.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(str(project_weight), str(cwd_weight))
        except OSError:
            # Symlink failed (e.g. cross-device), copy instead
            import shutil
            shutil.copy2(str(project_weight), str(cwd_weight))


def load_imagebind(freeze: bool = True, device: Optional[str] = None):
    """
    Load ImageBind model with caching.
    Handles SSL issues and CWD-relative weight paths.

    Args:
        freeze: Whether to freeze model parameters
        device: Target device (optional)

    Returns:
        model: ImageBind model instance

    Raises:
        ImportError: If ImageBind is not installed
        Exception: If weights cannot be loaded
    """
    global _imagebind_model

    if _imagebind_model is not None:
        return _imagebind_model

    from imagebind.models import imagebind_model

    # Ensure weights are available before ImageBind tries to download
    weight_path = _ensure_imagebind_weights()

    # Make sure CWD/.checkpoints/ points to our weights
    _setup_imagebind_checkpoint_path()

    model = imagebind_model.imagebind_huge(pretrained=True)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

    if device:
        model = model.to(device)

    _imagebind_model = model
    print("Loaded ImageBind successfully")
    return model


def load_clip(model_name: str = 'ViT-L-14', pretrained: str = 'openai', freeze: bool = True):
    """
    Load CLIP model via open_clip.

    Args:
        model_name: CLIP model variant
        pretrained: Pretrained weights source
        freeze: Whether to freeze model parameters

    Returns:
        model: CLIP model instance

    Raises:
        ImportError: If open_clip is not installed
    """
    global _clip_model

    if _clip_model is not None:
        return _clip_model

    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

    _clip_model = model
    print(f"Loaded CLIP ({model_name}) successfully")
    return model


def load_imagebind_or_clip(freeze: bool = True) -> Tuple[nn.Module, bool]:
    """
    Try loading ImageBind; fall back to CLIP if unavailable.

    Args:
        freeze: Whether to freeze model parameters

    Returns:
        (model, is_imagebind): The loaded model and whether it's ImageBind
    """
    try:
        model = load_imagebind(freeze=freeze)
        return model, True
    except (ImportError, Exception) as e:
        print(f"ImageBind not available ({e}), using CLIP as fallback")
        try:
            model = load_clip(freeze=freeze)
            return model, False
        except (ImportError, Exception) as e2:
            print(f"Warning: Neither ImageBind nor CLIP available ({e2})")
            return nn.Identity(), False


def clear_cache():
    """Clear cached models (useful for testing)"""
    global _imagebind_model, _clip_model
    _imagebind_model = None
    _clip_model = None


# ---------------------------------------------------------------------------
# ImageBind-compatible preprocessing utilities
# ---------------------------------------------------------------------------

# ImageBind normalization constants (different from ImageNet)
_IMAGEBIND_VISION_MEAN = (0.48145466, 0.4578275, 0.40821073)
_IMAGEBIND_VISION_STD = (0.26862954, 0.26130258, 0.27577711)

# ImageBind audio constants (Kaldi fbank)
_IMAGEBIND_AUDIO_MEAN = -4.268
_IMAGEBIND_AUDIO_STD = 9.138
_IMAGEBIND_AUDIO_NUM_MEL = 128
_IMAGEBIND_AUDIO_TARGET_LEN = 204
_IMAGEBIND_AUDIO_SAMPLE_RATE = 16000


def imagebind_preprocess_vision(images: torch.Tensor) -> torch.Tensor:
    """
    Preprocess images for ImageBind's vision encoder.

    ImageBind expects 224x224 images with its own normalization
    (not ImageNet). This function handles resize + renormalize
    from our pipeline's format.

    Args:
        images: (B, 3, H, W) images, assumed to be ImageNet-normalized

    Returns:
        (B, 3, 224, 224) images normalized for ImageBind
    """
    # Undo ImageNet normalization to get [0, 1] range
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    images_01 = images * imagenet_std + imagenet_mean
    images_01 = images_01.clamp(0, 1)

    # Resize to 224x224 (ImageBind's expected resolution)
    if images_01.shape[-2:] != (224, 224):
        images_01 = F.interpolate(images_01, size=(224, 224), mode='bicubic', align_corners=False)
        images_01 = images_01.clamp(0, 1)

    # Apply ImageBind normalization
    ib_mean = torch.tensor(_IMAGEBIND_VISION_MEAN, device=images.device).view(1, 3, 1, 1)
    ib_std = torch.tensor(_IMAGEBIND_VISION_STD, device=images.device).view(1, 3, 1, 1)
    images_ib = (images_01 - ib_mean) / ib_std

    return images_ib


def imagebind_extract_patch_tokens(
    model: nn.Module,
    images: torch.Tensor,
    target_n_tokens: int = 256,
) -> torch.Tensor:
    """
    Extract spatial patch tokens from ImageBind's vision transformer trunk,
    bypassing the CLS-selection head.

    ImageBind ViT-H with kernel_size=(2,14,14) on 224x224 produces:
      16x16 = 256 patch tokens + 1 CLS token = 257 total
    The standard forward() selects only CLS (index=0) via the head.
    Here we run preprocessor + trunk only, then take tokens[1:] to get
    the 256 spatial patch tokens.

    Args:
        model: ImageBind model instance
        images: (B, 3, 224, 224) ImageBind-normalized images
        target_n_tokens: desired number of output tokens (default 256)

    Returns:
        patch_tokens: (B, target_n_tokens, embed_dim) spatial patch tokens
    """
    from imagebind.models.imagebind_model import ModalityType

    with torch.no_grad():
        # Step 1: Run vision preprocessor
        preprocessed = model.modality_preprocessors[ModalityType.VISION](
            **{ModalityType.VISION: images}
        )
        trunk_inputs = preprocessed["trunk"]

        # Step 2: Run vision trunk (transformer)
        trunk_output = model.modality_trunks[ModalityType.VISION](**trunk_inputs)
        # trunk_output: (B, 1 + num_patches, embed_dim)
        # index 0 = CLS token, indices 1: = patch tokens

        # Step 3: Extract patch tokens (skip CLS at index 0)
        patch_tokens = trunk_output[:, 1:, :]  # (B, num_patches, embed_dim)

        # Interpolate if patch count doesn't match target
        num_patches = patch_tokens.shape[1]
        if num_patches != target_n_tokens:
            # (B, num_patches, D) -> (B, D, num_patches) -> interpolate -> (B, D, target) -> transpose
            patch_tokens = F.interpolate(
                patch_tokens.transpose(1, 2),
                size=target_n_tokens,
                mode='linear',
                align_corners=False,
            ).transpose(1, 2)

    return patch_tokens


def imagebind_extract_audio_embedding(
    model: nn.Module,
    audio_input: torch.Tensor,
) -> torch.Tensor:
    """
    Extract audio global embedding from ImageBind.

    Args:
        model: ImageBind model instance
        audio_input: (B, num_clips, 1, 128, 204) ImageBind-formatted audio

    Returns:
        audio_embedding: (B, 1024) global audio embedding
    """
    from imagebind.models.imagebind_model import ModalityType

    with torch.no_grad():
        outputs = model({ModalityType.AUDIO: audio_input})
        return outputs[ModalityType.AUDIO]  # (B, 1024)


def waveform_to_imagebind_audio(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    num_clips: int = 3,
    clip_duration: float = 2.0,
) -> torch.Tensor:
    """
    Convert raw waveform to ImageBind-compatible audio tensor.

    Uses torchaudio Kaldi fbank (same as ImageBind's waveform2melspec)
    to produce mel spectrograms with 128 bins and 204 time frames,
    normalized with ImageBind's audio statistics.

    Args:
        waveform: (1, T) or (T,) mono waveform at sample_rate
        sample_rate: waveform sample rate (will resample to 16kHz if different)
        num_clips: number of temporal clips to extract
        clip_duration: duration of each clip in seconds

    Returns:
        audio_tensor: (num_clips, 1, 128, 204) ready for ImageBind
    """
    import torchaudio

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Resample to 16kHz if needed
    if sample_rate != _IMAGEBIND_AUDIO_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, _IMAGEBIND_AUDIO_SAMPLE_RATE
        )

    clip_samples = int(clip_duration * _IMAGEBIND_AUDIO_SAMPLE_RATE)
    total_samples = waveform.shape[-1]

    # Ensure waveform is long enough for at least one clip
    if total_samples < clip_samples:
        # Pad by repeating
        repeats = (clip_samples + total_samples - 1) // total_samples
        waveform = waveform.repeat(1, repeats)[:, :clip_samples]
        total_samples = clip_samples

    clips = []
    for i in range(num_clips):
        # Uniformly spaced clips
        if num_clips > 1:
            start = int(i * (total_samples - clip_samples) / (num_clips - 1))
        else:
            start = (total_samples - clip_samples) // 2
        start = max(0, min(start, total_samples - clip_samples))

        clip_waveform = waveform[:, start:start + clip_samples]

        # Convert to Kaldi fbank mel spectrogram (same as ImageBind's waveform2melspec)
        clip_waveform = clip_waveform - clip_waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(
            clip_waveform,
            htk_compat=True,
            sample_frequency=_IMAGEBIND_AUDIO_SAMPLE_RATE,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=_IMAGEBIND_AUDIO_NUM_MEL,
            dither=0.0,
            frame_length=25,
            frame_shift=10,
        )  # (num_frames, 128)

        # Transpose to (128, num_frames) and pad/crop to target_len=204
        fbank = fbank.transpose(0, 1)  # (128, num_frames)
        num_frames = fbank.shape[1]

        if num_frames < _IMAGEBIND_AUDIO_TARGET_LEN:
            padding = _IMAGEBIND_AUDIO_TARGET_LEN - num_frames
            fbank = F.pad(fbank, (0, padding))
        else:
            fbank = fbank[:, :_IMAGEBIND_AUDIO_TARGET_LEN]

        fbank = fbank.unsqueeze(0)  # (1, 128, 204)
        clips.append(fbank)

    audio_tensor = torch.stack(clips, dim=0)  # (num_clips, 1, 128, 204)

    # Normalize with ImageBind's audio statistics
    audio_tensor = (audio_tensor - _IMAGEBIND_AUDIO_MEAN) / (_IMAGEBIND_AUDIO_STD * 2)

    return audio_tensor
