"""
Audio Effect Functions for Delta Correspondence Experiment

Effects:
    - lpf: Low-pass filter (dullness/muffled)
    - highshelf: High-shelf EQ (brightness/air)
    - saturation: Soft clipping (harmonic richness)
    - reverb: Convolution reverb (spatialness)

Each effect has intensity levels: low, mid, high
"""

from typing import Literal, Union, Optional
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
from scipy import signal


EffectType = Literal["lpf", "highshelf", "saturation", "reverb"]
IntensityLevel = Literal["low", "mid", "high"]


# Effect intensity mappings
LPF_CUTOFFS = {
    "low": 10000,   # 10kHz
    "mid": 6000,    # 6kHz
    "high": 3000,   # 3kHz (more aggressive)
}

HIGHSHELF_GAINS = {
    "low": 2.0,     # +2 dB at 8kHz
    "mid": 5.0,     # +5 dB
    "high": 8.0,    # +8 dB
}

SATURATION_DRIVES = {
    "low": 3.0,     # +3 dB drive
    "mid": 6.0,     # +6 dB drive
    "high": 9.0,    # +9 dB drive
}

REVERB_WETS = {
    "low": 0.10,    # 10% wet
    "mid": 0.25,    # 25% wet
    "high": 0.40,   # 40% wet
}


def apply_lpf(
    waveform: torch.Tensor,
    sample_rate: int,
    level: IntensityLevel
) -> torch.Tensor:
    """Apply low-pass filter."""
    cutoff = LPF_CUTOFFS[level]
    return F.lowpass_biquad(waveform, sample_rate, cutoff, Q=0.707)


def apply_highshelf(
    waveform: torch.Tensor,
    sample_rate: int,
    level: IntensityLevel
) -> torch.Tensor:
    """Apply high-shelf EQ boost."""
    gain_db = HIGHSHELF_GAINS[level]
    # High-shelf at 8kHz
    return F.highpass_biquad(waveform, sample_rate, 8000, Q=0.707) * (10 ** (gain_db / 20)) + \
           F.lowpass_biquad(waveform, sample_rate, 8000, Q=0.707)


def apply_saturation(
    waveform: torch.Tensor,
    sample_rate: int,
    level: IntensityLevel
) -> torch.Tensor:
    """Apply soft saturation/clipping."""
    drive_db = SATURATION_DRIVES[level]
    drive = 10 ** (drive_db / 20)

    # Apply drive and soft clip with tanh
    driven = waveform * drive
    saturated = torch.tanh(driven)

    # Normalize to prevent clipping
    max_val = saturated.abs().max()
    if max_val > 0:
        saturated = saturated / max_val * 0.95

    return saturated


def apply_reverb(
    waveform: torch.Tensor,
    sample_rate: int,
    level: IntensityLevel
) -> torch.Tensor:
    """
    Apply simple algorithmic reverb (comb filter approximation).

    Note: For production, consider using convolution reverb with real IRs.
    """
    wet = REVERB_WETS[level]
    dry = 1.0 - wet

    # Simple delay-based reverb approximation
    delay_samples = int(0.03 * sample_rate)  # 30ms delay
    decay = 0.5

    # Create delayed copies
    reverb = torch.zeros_like(waveform)
    for i, d in enumerate([1, 2, 3, 4]):
        delay = delay_samples * d
        decayed = waveform * (decay ** i)
        if delay < waveform.shape[-1]:
            reverb[..., delay:] += decayed[..., :-delay]

    # Mix dry and wet
    output = dry * waveform + wet * reverb

    # Normalize
    max_val = output.abs().max()
    if max_val > 0:
        output = output / max_val * 0.95

    return output


# Effect registry
AUDIO_EFFECTS = {
    "lpf": apply_lpf,
    "highshelf": apply_highshelf,
    "saturation": apply_saturation,
    "reverb": apply_reverb,
}


def apply_effect(
    waveform: Union[torch.Tensor, np.ndarray],
    sample_rate: int,
    effect_type: EffectType,
    intensity: IntensityLevel,
) -> torch.Tensor:
    """
    Apply an audio effect with specified intensity.

    Args:
        waveform: Input audio (torch Tensor or numpy array), shape (..., samples)
        sample_rate: Audio sample rate
        effect_type: Type of effect ("lpf", "highshelf", "saturation", "reverb")
        intensity: Intensity level ("low", "mid", "high")

    Returns:
        Processed audio as torch Tensor
    """
    # Convert to torch if needed
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).float()

    # Ensure 2D (channels, samples)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    if effect_type not in AUDIO_EFFECTS:
        raise ValueError(f"Unknown effect type: {effect_type}")

    return AUDIO_EFFECTS[effect_type](waveform, sample_rate, intensity)


def load_audio(path: str, target_sr: int = 16000) -> tuple[torch.Tensor, int]:
    """Load audio file and resample if needed."""
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    return waveform, target_sr


def get_effect_types() -> list[EffectType]:
    """Return list of available effect types."""
    return list(AUDIO_EFFECTS.keys())


def get_intensity_levels() -> list[IntensityLevel]:
    """Return list of available intensity levels."""
    return ["low", "mid", "high"]
