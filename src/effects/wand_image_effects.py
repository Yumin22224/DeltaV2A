"""Wand-based image effects with continuous intensity in [0, 1]."""

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

_BLUR_SIGMA_MIN, _BLUR_SIGMA_MAX = 0.5, 4.0
_MOTION_RADIUS_MIN, _MOTION_RADIUS_MAX = 2.0, 14.0
_MOTION_SIGMA_MIN, _MOTION_SIGMA_MAX = 0.6, 4.0
_MOTION_ANGLE_MIN, _MOTION_ANGLE_MAX = 0.0, 90.0
_SHARPEN_SIGMA_MIN, _SHARPEN_SIGMA_MAX = 0.2, 2.2
_SPREAD_RADIUS_MIN, _SPREAD_RADIUS_MAX = 0.2, 3.0
_SEPIA_THRESHOLD_MIN, _SEPIA_THRESHOLD_MAX = 10.0, 85.0
_SOLARIZE_THRESHOLD_MIN, _SOLARIZE_THRESHOLD_MAX = 20.0, 95.0


def _clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


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


def apply_adaptive_blur(image: Image.Image, intensity: float) -> Image.Image:
    t = _clamp01(intensity)
    sigma = _lerp(_BLUR_SIGMA_MIN, _BLUR_SIGMA_MAX, t)

    def _op(wimg):
        try:
            wimg.adaptive_blur(radius=0.0, sigma=sigma)
        except TypeError:
            wimg.adaptive_blur(0.0, sigma)

    return _apply_wand(image, _op)


def apply_motion_blur(image: Image.Image, intensity: float) -> Image.Image:
    t = _clamp01(intensity)
    radius = _lerp(_MOTION_RADIUS_MIN, _MOTION_RADIUS_MAX, t)
    sigma = _lerp(_MOTION_SIGMA_MIN, _MOTION_SIGMA_MAX, t)
    angle = _lerp(_MOTION_ANGLE_MIN, _MOTION_ANGLE_MAX, t)

    def _op(wimg):
        try:
            wimg.motion_blur(radius=radius, sigma=sigma, angle=angle)
        except TypeError:
            wimg.motion_blur(radius, sigma, angle)

    return _apply_wand(image, _op)


def apply_adaptive_sharpen(image: Image.Image, intensity: float) -> Image.Image:
    t = _clamp01(intensity)
    sigma = _lerp(_SHARPEN_SIGMA_MIN, _SHARPEN_SIGMA_MAX, t)

    def _op(wimg):
        try:
            wimg.adaptive_sharpen(radius=0.0, sigma=sigma)
        except TypeError:
            wimg.adaptive_sharpen(0.0, sigma)

    return _apply_wand(image, _op)


def apply_add_noise(image: Image.Image, intensity: float) -> Image.Image:
    t = _clamp01(intensity)

    def _op(wimg):
        try:
            wimg.noise("gaussian")
        except Exception:
            wimg.add_noise("gaussian")

    noisy = _apply_wand(image, _op)
    # Blend keeps noise amount continuous even when primitive API is discrete.
    return Image.blend(image, noisy, alpha=t)


def apply_spread(image: Image.Image, intensity: float) -> Image.Image:
    t = _clamp01(intensity)
    radius = _lerp(_SPREAD_RADIUS_MIN, _SPREAD_RADIUS_MAX, t)

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


def apply_sepia_tone(image: Image.Image, intensity: float) -> Image.Image:
    t = _clamp01(intensity)
    threshold = f"{_lerp(_SEPIA_THRESHOLD_MIN, _SEPIA_THRESHOLD_MAX, t):.2f}%"
    return _apply_wand(image, lambda wimg: _apply_threshold_op(wimg, "sepia_tone", threshold))


def apply_solarize(image: Image.Image, intensity: float) -> Image.Image:
    t = _clamp01(intensity)
    # Higher intensity => lower threshold (stronger solarization)
    threshold_val = _lerp(_SOLARIZE_THRESHOLD_MAX, _SOLARIZE_THRESHOLD_MIN, t)
    threshold = f"{threshold_val:.2f}%"
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
    intensity: float,
) -> Image.Image:
    pil = _to_pil(image)
    if effect_type not in IMAGE_EFFECTS:
        raise ValueError(f"Unknown effect type: {effect_type}")
    return IMAGE_EFFECTS[effect_type](pil, float(intensity))


def get_effect_types() -> list[EffectType]:
    return list(IMAGE_EFFECTS.keys())
