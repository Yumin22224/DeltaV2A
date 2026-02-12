"""
Wand-based image effect functions.

Effects:
    - adaptive_blur
    - motion_blur
    - adaptive_sharpen
    - add_noise
    - spread
    - sepia_tone
    - solarize
"""

from io import BytesIO
from typing import Literal, Union

import numpy as np
from PIL import Image


EffectType = Literal[
    "adaptive_blur",
    "motion_blur",
    "adaptive_sharpen",
    "add_noise",
    "spread",
    "sepia_tone",
    "solarize",
]
IntensityLevel = Literal["low", "mid", "high"]


ADAPTIVE_BLUR_SIGMA = {"low": 1.0, "mid": 2.0, "high": 3.0}
MOTION_BLUR_LEVELS = {
    "low": (4.0, 1.0, 15.0),
    "mid": (8.0, 2.0, 30.0),
    "high": (12.0, 3.0, 45.0),
}
ADAPTIVE_SHARPEN_SIGMA = {"low": 0.6, "mid": 1.0, "high": 1.6}
NOISE_PASSES = {"low": 1, "mid": 2, "high": 3}
SPREAD_RADIUS = {"low": 0.8, "mid": 1.6, "high": 2.4}
SEPIA_THRESHOLD = {"low": "20%", "mid": "40%", "high": "65%"}
SOLARIZE_THRESHOLD = {"low": "85%", "mid": "70%", "high": "55%"}


def _require_wand():
    try:
        from wand.image import Image as WandImage  # type: ignore
    except Exception as e:
        raise ImportError(
            "Wand is required for image effects. Install 'Wand' and ImageMagick."
        ) from e
    return WandImage


def _to_pil(image: Union[Image.Image, np.ndarray, str]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return Image.fromarray(image).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def _apply_wand(pil_image: Image.Image, op) -> Image.Image:
    WandImage = _require_wand()
    buf = BytesIO()
    pil_image.save(buf, format="PNG")
    with WandImage(blob=buf.getvalue()) as wimg:
        op(wimg)
        out_blob = wimg.make_blob(format="png")
    return Image.open(BytesIO(out_blob)).convert("RGB")


def apply_adaptive_blur(image: Image.Image, level: IntensityLevel) -> Image.Image:
    sigma = ADAPTIVE_BLUR_SIGMA[level]

    def _op(wimg):
        try:
            wimg.adaptive_blur(radius=0.0, sigma=sigma)
        except TypeError:
            wimg.adaptive_blur(0.0, sigma)

    return _apply_wand(image, _op)


def apply_motion_blur(image: Image.Image, level: IntensityLevel) -> Image.Image:
    radius, sigma, angle = MOTION_BLUR_LEVELS[level]

    def _op(wimg):
        try:
            wimg.motion_blur(radius=radius, sigma=sigma, angle=angle)
        except TypeError:
            wimg.motion_blur(radius, sigma, angle)

    return _apply_wand(image, _op)


def apply_adaptive_sharpen(image: Image.Image, level: IntensityLevel) -> Image.Image:
    sigma = ADAPTIVE_SHARPEN_SIGMA[level]

    def _op(wimg):
        try:
            wimg.adaptive_sharpen(radius=0.0, sigma=sigma)
        except TypeError:
            wimg.adaptive_sharpen(0.0, sigma)

    return _apply_wand(image, _op)


def apply_add_noise(image: Image.Image, level: IntensityLevel) -> Image.Image:
    passes = NOISE_PASSES[level]

    def _op(wimg):
        for _ in range(passes):
            try:
                wimg.noise("gaussian")
            except Exception:
                wimg.add_noise("gaussian")

    return _apply_wand(image, _op)


def apply_spread(image: Image.Image, level: IntensityLevel) -> Image.Image:
    radius = SPREAD_RADIUS[level]

    def _op(wimg):
        try:
            wimg.spread(radius=radius)
        except TypeError:
            wimg.spread(radius)

    return _apply_wand(image, _op)


def _apply_threshold_op(wimg, method_name: str, threshold: str):
    method = getattr(wimg, method_name)
    try:
        method(threshold=threshold)
        return
    except TypeError:
        pass

    try:
        method(threshold)
        return
    except TypeError:
        pass

    value = float(threshold.rstrip("%"))
    method(value)


def apply_sepia_tone(image: Image.Image, level: IntensityLevel) -> Image.Image:
    threshold = SEPIA_THRESHOLD[level]
    return _apply_wand(image, lambda wimg: _apply_threshold_op(wimg, "sepia_tone", threshold))


def apply_solarize(image: Image.Image, level: IntensityLevel) -> Image.Image:
    threshold = SOLARIZE_THRESHOLD[level]
    return _apply_wand(image, lambda wimg: _apply_threshold_op(wimg, "solarize", threshold))


IMAGE_EFFECTS = {
    "adaptive_blur": apply_adaptive_blur,
    "motion_blur": apply_motion_blur,
    "adaptive_sharpen": apply_adaptive_sharpen,
    "add_noise": apply_add_noise,
    "spread": apply_spread,
    "sepia_tone": apply_sepia_tone,
    "solarize": apply_solarize,
}


def apply_effect(
    image: Union[Image.Image, np.ndarray, str],
    effect_type: EffectType,
    intensity: IntensityLevel,
) -> Image.Image:
    pil = _to_pil(image)
    if effect_type not in IMAGE_EFFECTS:
        raise ValueError(f"Unknown effect type: {effect_type}")
    return IMAGE_EFFECTS[effect_type](pil, intensity)


def get_effect_types() -> list[EffectType]:
    return list(IMAGE_EFFECTS.keys())


def get_intensity_levels() -> list[IntensityLevel]:
    return ["low", "mid", "high"]
