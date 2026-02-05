"""
Image Effect Functions for Delta Correspondence Experiment

Effects:
    - brightness: Adjust image brightness (gain)
    - contrast: Adjust image contrast
    - saturation: Adjust color saturation
    - blur: Apply Gaussian blur

Each effect has intensity levels: low, mid, high
"""

from typing import Literal, Union
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


EffectType = Literal["brightness", "contrast", "saturation", "blur"]
IntensityLevel = Literal["low", "mid", "high"]


# Effect intensity mappings
BRIGHTNESS_LEVELS = {
    "low": 1.10,    # +10%
    "mid": 1.25,    # +25%
    "high": 1.40,   # +40%
}

CONTRAST_LEVELS = {
    "low": 1.10,    # +10%
    "mid": 1.25,    # +25%
    "high": 1.40,   # +40%
}

SATURATION_LEVELS = {
    "low": 1.10,    # +10%
    "mid": 1.25,    # +25%
    "high": 1.40,   # +40%
}

BLUR_LEVELS = {
    "low": 1.0,     # sigma=1.0
    "mid": 2.0,     # sigma=2.0
    "high": 3.0,    # sigma=3.0
}


def apply_brightness(image: Image.Image, level: IntensityLevel) -> Image.Image:
    """Apply brightness adjustment."""
    factor = BRIGHTNESS_LEVELS[level]
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def apply_contrast(image: Image.Image, level: IntensityLevel) -> Image.Image:
    """Apply contrast adjustment."""
    factor = CONTRAST_LEVELS[level]
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def apply_saturation(image: Image.Image, level: IntensityLevel) -> Image.Image:
    """Apply saturation adjustment."""
    factor = SATURATION_LEVELS[level]
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)


def apply_blur(image: Image.Image, level: IntensityLevel) -> Image.Image:
    """Apply Gaussian blur."""
    sigma = BLUR_LEVELS[level]
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


# Effect registry
IMAGE_EFFECTS = {
    "brightness": apply_brightness,
    "contrast": apply_contrast,
    "saturation": apply_saturation,
    "blur": apply_blur,
}


def apply_effect(
    image: Union[Image.Image, np.ndarray, str],
    effect_type: EffectType,
    intensity: IntensityLevel,
) -> Image.Image:
    """
    Apply an image effect with specified intensity.

    Args:
        image: Input image (PIL Image, numpy array, or path)
        effect_type: Type of effect ("brightness", "contrast", "saturation", "blur")
        intensity: Intensity level ("low", "mid", "high")

    Returns:
        Processed PIL Image
    """
    # Convert to PIL Image if needed
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if effect_type not in IMAGE_EFFECTS:
        raise ValueError(f"Unknown effect type: {effect_type}")

    return IMAGE_EFFECTS[effect_type](image, intensity)


def get_effect_types() -> list[EffectType]:
    """Return list of available effect types."""
    return list(IMAGE_EFFECTS.keys())


def get_intensity_levels() -> list[IntensityLevel]:
    """Return list of available intensity levels."""
    return ["low", "mid", "high"]
