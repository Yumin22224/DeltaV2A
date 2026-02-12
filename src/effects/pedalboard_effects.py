"""
Pedalboard Audio Effects with Continuous Parameters

Wraps Spotify's pedalboard library for continuous-parameter DSP.
Used in the new pipeline (Phase A-C). Legacy torchaudio effects
for the experiment module are in experiment/src/effects/audio_effects.py.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class EffectParamSpec:
    """Specification for a single effect parameter."""
    name: str
    min_val: float
    max_val: float
    scale: str  # 'linear' or 'log'


@dataclass
class EffectSpec:
    """Specification for a pedalboard effect."""
    name: str
    params: List[EffectParamSpec] = field(default_factory=list)

    @property
    def num_params(self) -> int:
        return len(self.params)

    @property
    def param_names(self) -> List[str]:
        return [p.name for p in self.params]


# === Effect Catalog ===

EFFECT_CATALOG: Dict[str, EffectSpec] = {
    'lowpass': EffectSpec('lowpass', [
        EffectParamSpec('cutoff_hz', 200.0, 20000.0, 'log'),
    ]),
    'bitcrush': EffectSpec('bitcrush', [
        EffectParamSpec('bit_depth', 4.0, 16.0, 'linear'),
    ]),
    'reverb': EffectSpec('reverb', [
        EffectParamSpec('room_size', 0.0, 1.0, 'linear'),
        EffectParamSpec('damping', 0.0, 1.0, 'linear'),
        EffectParamSpec('wet_level', 0.0, 0.8, 'linear'),
        EffectParamSpec('dry_level', 0.2, 1.0, 'linear'),
    ]),
    'highpass': EffectSpec('highpass', [
        EffectParamSpec('cutoff_hz', 20.0, 6000.0, 'log'),
    ]),
    'distortion': EffectSpec('distortion', [
        EffectParamSpec('drive_db', 0.0, 40.0, 'linear'),
    ]),
    'phaser': EffectSpec('phaser', [
        EffectParamSpec('rate_hz', 0.1, 8.0, 'linear'),
        EffectParamSpec('depth', 0.0, 1.0, 'linear'),
        EffectParamSpec('centre_frequency_hz', 300.0, 4000.0, 'log'),
        EffectParamSpec('feedback', 0.0, 0.9, 'linear'),
        EffectParamSpec('mix', 0.0, 1.0, 'linear'),
    ]),
    'delay': EffectSpec('delay', [
        EffectParamSpec('delay_seconds', 0.05, 1.0, 'log'),
        EffectParamSpec('feedback', 0.0, 0.8, 'linear'),
        EffectParamSpec('mix', 0.0, 0.7, 'linear'),
    ]),
}

# Default "bypass-like" values used when an effect is inactive.
# These are used to encode missing effect params in normalized vectors.
BYPASS_PARAMS: Dict[str, Dict[str, float]] = {
    "lowpass": {"cutoff_hz": 20000.0},
    "bitcrush": {"bit_depth": 16.0},
    "reverb": {
        "room_size": 0.0,
        "damping": 0.0,
        "wet_level": 0.0,
        "dry_level": 1.0,
    },
    "highpass": {"cutoff_hz": 20.0},
    "distortion": {"drive_db": 0.0},
    "phaser": {
        "rate_hz": 0.1,
        "depth": 0.0,
        "centre_frequency_hz": 300.0,
        "feedback": 0.0,
        "mix": 0.0,
    },
    "delay": {
        "delay_seconds": 0.05,
        "feedback": 0.0,
        "mix": 0.0,
    },
}


def _default_param_value(effect_name: str, param_name: str, min_val: float) -> float:
    """Return bypass default for missing params, fallback to min value."""
    return BYPASS_PARAMS.get(effect_name, {}).get(param_name, min_val)


def get_total_param_count(effect_names: List[str] = None) -> int:
    """Get total parameter count across selected effects."""
    if effect_names is None:
        effect_names = list(EFFECT_CATALOG.keys())
    return sum(EFFECT_CATALOG[name].num_params for name in effect_names)


def normalize_params(
    params_dict: Dict[str, Dict[str, float]],
    effect_names: List[str],
) -> np.ndarray:
    """
    Normalize actual parameter values to [0, 1].

    Args:
        params_dict: {effect_name: {param_name: actual_value}}
        effect_names: Ordered list of effect names

    Returns:
        (total_params,) array of normalized values in [0, 1]
    """
    normalized = []
    for effect_name in effect_names:
        spec = EFFECT_CATALOG[effect_name]
        effect_params = params_dict.get(effect_name, {})
        for ps in spec.params:
            # Inactive/missing effect params are encoded at bypass defaults,
            # not at range minima.
            default_val = _default_param_value(effect_name, ps.name, ps.min_val)
            actual = effect_params.get(ps.name, default_val)
            if ps.scale == 'log':
                log_min = np.log(max(ps.min_val, 1e-10))
                log_max = np.log(ps.max_val)
                norm = (np.log(max(actual, 1e-10)) - log_min) / (log_max - log_min)
            else:
                norm = (actual - ps.min_val) / (ps.max_val - ps.min_val)
            normalized.append(np.clip(norm, 0.0, 1.0))
    return np.array(normalized, dtype=np.float32)


def denormalize_params(
    normalized: np.ndarray,
    effect_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Denormalize [0, 1] parameters to actual values.

    Args:
        normalized: (total_params,) array
        effect_names: Ordered list of effect names

    Returns:
        {effect_name: {param_name: actual_value}}
    """
    params_dict = {}
    idx = 0
    for effect_name in effect_names:
        spec = EFFECT_CATALOG[effect_name]
        params_dict[effect_name] = {}
        for ps in spec.params:
            val = float(normalized[idx])
            if ps.scale == 'log':
                log_min = np.log(max(ps.min_val, 1e-10))
                log_max = np.log(ps.max_val)
                actual = np.exp(val * (log_max - log_min) + log_min)
            else:
                actual = val * (ps.max_val - ps.min_val) + ps.min_val
            params_dict[effect_name][ps.name] = actual
            idx += 1
    return params_dict


def sample_random_params(
    effect_names: List[str],
    rng: np.random.Generator = None,
) -> Dict[str, Dict[str, float]]:
    """Sample random parameters uniformly in normalized space."""
    if rng is None:
        rng = np.random.default_rng()
    total = get_total_param_count(effect_names)
    normalized = rng.uniform(0.0, 1.0, size=total).astype(np.float32)
    return denormalize_params(normalized, effect_names)


class PedalboardRenderer:
    """
    Applies pedalboard effects to audio waveforms.

    Uses Spotify's pedalboard library for high-quality, real-time
    audio processing with continuous parameters.
    """

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def _create_plugin(self, effect_name: str, params: Dict[str, float]):
        """Create a single pedalboard plugin from params."""
        import pedalboard as pb

        if effect_name == 'lowpass':
            return pb.LowpassFilter(cutoff_frequency_hz=params['cutoff_hz'])
        elif effect_name == 'bitcrush':
            bit_depth = int(round(params['bit_depth']))
            bit_depth = max(1, min(32, bit_depth))
            return pb.Bitcrush(bit_depth=bit_depth)
        elif effect_name == 'reverb':
            return pb.Reverb(
                room_size=params['room_size'],
                damping=params['damping'],
                wet_level=params['wet_level'],
                dry_level=params['dry_level'],
            )
        elif effect_name == 'highpass':
            return pb.HighpassFilter(cutoff_frequency_hz=params['cutoff_hz'])
        elif effect_name == 'distortion':
            return pb.Distortion(drive_db=params['drive_db'])
        elif effect_name == 'phaser':
            try:
                return pb.Phaser(
                    rate_hz=params['rate_hz'],
                    depth=params['depth'],
                    centre_frequency_hz=params['centre_frequency_hz'],
                    feedback=params['feedback'],
                    mix=params['mix'],
                )
            except TypeError:
                return pb.Phaser(
                    rate_hz=params['rate_hz'],
                    depth=params['depth'],
                    center_frequency_hz=params['centre_frequency_hz'],
                    feedback=params['feedback'],
                    mix=params['mix'],
                )
        elif effect_name == 'delay':
            return pb.Delay(
                delay_seconds=params['delay_seconds'],
                feedback=params['feedback'],
                mix=params['mix'],
            )
        else:
            print(f"Warning: Unknown effect '{effect_name}', skipping")
            return None

    def build_board(self, params_dict: Dict[str, Dict[str, float]]):
        """Build a pedalboard.Pedalboard from parameter dict."""
        import pedalboard as pb

        plugins = []
        for effect_name, params in params_dict.items():
            plugin = self._create_plugin(effect_name, params)
            if plugin is not None:
                plugins.append(plugin)
        return pb.Pedalboard(plugins)

    def render(
        self,
        waveform: np.ndarray,
        params_dict: Dict[str, Dict[str, float]],
    ) -> np.ndarray:
        """
        Apply effects to audio.

        Args:
            waveform: (samples,) or (channels, samples) float32 array
            params_dict: {effect_name: {param_name: value}}

        Returns:
            processed: same shape as input, float32
        """
        squeeze = False
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]
            squeeze = True

        waveform = waveform.astype(np.float32)
        board = self.build_board(params_dict)
        processed = board(waveform, self.sample_rate)

        # Peak normalize to prevent clipping
        peak = np.abs(processed).max()
        if peak > 1.0:
            processed = processed / peak * 0.95

        if squeeze:
            processed = processed.squeeze(0)

        return processed

    def render_from_normalized(
        self,
        waveform: np.ndarray,
        normalized_params: np.ndarray,
        effect_names: List[str],
    ) -> np.ndarray:
        """Apply effects from normalized [0,1] parameter vector."""
        params_dict = denormalize_params(normalized_params, effect_names)
        return self.render(waveform, params_dict)
