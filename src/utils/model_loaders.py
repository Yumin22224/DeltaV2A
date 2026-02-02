"""
Centralized model loading utilities for pretrained models (ImageBind, CLIP)

Avoids duplicated loading logic across prior.py, delta_c_predictor.py, etc.
Handles SSL certificate issues on macOS and CWD-relative weight paths.
"""

import os
import ssl
import torch
import torch.nn as nn
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
