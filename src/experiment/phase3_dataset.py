"""
Phase 3 Training Dataset

Generates training data by applying random DSP effects to audio
and creating text conditions based on discovered correspondences.
"""

import torch
import numpy as np
import json
import librosa
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset

from src.effects.audio_effects import (
    apply_lpf,
    apply_highshelf,
    apply_saturation,
    apply_reverb,
)


class Phase3Dataset(Dataset):
    """
    Training dataset for Phase 3 (Learning - The Decoder).

    Generates (audio, text_condition, parameters) tuples by:
    1. Loading raw audio
    2. Selecting DSP effect based on discovery matrix
    3. Generating random parameters
    4. Applying DSP to create augmented audio
    5. Creating text condition from corresponding image effect
    """

    def __init__(
        self,
        audio_paths: List[str],
        discovery_matrix: np.ndarray,
        image_effects: List[str],
        audio_effects: List[str],
        sample_rate: int = 48000,
        duration: float = 10.0,
        top_k: int = 3,
        text_template: str = "A {effect} image",
    ):
        """
        Args:
            audio_paths: List of paths to audio files
            discovery_matrix: (n_image_effects, n_audio_effects) discovery scores
            image_effects: List of image effect names (with intensity, e.g., "blur (high)")
            audio_effects: List of audio effect names (with intensity)
            sample_rate: Target sample rate
            duration: Duration to load (seconds)
            top_k: Sample from top-k correspondences per audio effect
            text_template: Template for generating text conditions
        """
        self.audio_paths = audio_paths
        self.discovery_matrix = discovery_matrix
        self.image_effects = image_effects
        self.audio_effects = audio_effects
        self.sample_rate = sample_rate
        self.duration = duration
        self.top_k = top_k
        self.text_template = text_template

        # Build effect correspondence lookup
        self._build_correspondence_lookup()

        # DSP parameter ranges
        self.param_specs = {
            'lpf': {
                'cutoff_freq': (20.0, 20000.0, 'log'),
            },
            'highshelf': {
                'gain': (-20.0, 20.0, 'linear'),
                'freq': (1000.0, 20000.0, 'log'),
            },
            'saturation': {
                'drive': (0.0, 40.0, 'linear'),
            },
            'reverb': {
                'room_size': (0.0, 1.0, 'linear'),
                'damping': (0.0, 1.0, 'linear'),
                'wet_level': (0.0, 1.0, 'linear'),
            },
        }

    def _build_correspondence_lookup(self):
        """Build lookup from audio effect to corresponding image effects."""
        self.correspondence = {}

        for j, audio_effect in enumerate(self.audio_effects):
            # Extract base effect name (remove intensity)
            audio_base = audio_effect.split(' (')[0]

            # Get scores for this audio effect
            scores = self.discovery_matrix[:, j]

            # Get top-k corresponding image effects
            top_indices = np.argsort(scores)[-self.top_k:][::-1]
            top_image_effects = [self.image_effects[i] for i in top_indices]
            top_scores = scores[top_indices]

            # Store with probabilities for weighted sampling
            probs = top_scores / top_scores.sum()
            self.correspondence[audio_base] = (top_image_effects, probs)

    def __len__(self) -> int:
        return len(self.audio_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Generate a training sample.

        Returns:
            {
                'audio_raw': (T,) raw audio waveform
                'audio_aug': (T,) augmented audio waveform
                'text_condition': str, text description
                'effect_name': str, base effect name (e.g., 'lpf')
                'parameters': (num_params,) normalized ground-truth parameters
            }
        """
        # Load audio
        audio_path = self.audio_paths[idx]
        audio_raw = self._load_audio(audio_path)

        # Sample a random audio effect
        effect_name = self._sample_effect()

        # Generate random parameters
        params_dict = self._generate_random_params(effect_name)

        # Apply DSP effect
        audio_aug = self._apply_effect(audio_raw, effect_name, params_dict)

        # Get corresponding text condition
        text_condition = self._get_text_condition(effect_name)

        # Normalize parameters to [0, 1]
        params_normalized = self._normalize_params(params_dict, effect_name)

        return {
            'audio_raw': torch.from_numpy(audio_raw).float(),
            'audio_aug': torch.from_numpy(audio_aug).float(),
            'text_condition': text_condition,
            'effect_name': effect_name,
            'parameters': params_normalized,
        }

    def _load_audio(self, path: str) -> np.ndarray:
        """Load and preprocess audio."""
        audio, sr = librosa.load(path, sr=self.sample_rate, mono=True)

        # Truncate or pad to target duration
        target_length = int(self.duration * self.sample_rate)

        if len(audio) > target_length:
            # Random crop
            start = random.randint(0, len(audio) - target_length)
            audio = audio[start:start + target_length]
        elif len(audio) < target_length:
            # Pad with zeros
            audio = np.pad(audio, (0, target_length - len(audio)))

        return audio

    def _sample_effect(self) -> str:
        """Sample a random effect from available effects."""
        # Get base effect names (without intensity)
        base_effects = list(self.param_specs.keys())
        return random.choice(base_effects)

    def _generate_random_params(self, effect_name: str) -> Dict[str, float]:
        """Generate random parameters for the given effect."""
        if effect_name not in self.param_specs:
            raise ValueError(f"Unknown effect: {effect_name}")

        params = {}
        for param_name, (min_val, max_val, scale_type) in self.param_specs[effect_name].items():
            if scale_type == 'linear':
                value = random.uniform(min_val, max_val)
            elif scale_type == 'log':
                # Sample in log space
                log_min = np.log(min_val)
                log_max = np.log(max_val)
                log_value = random.uniform(log_min, log_max)
                value = np.exp(log_value)
            else:
                raise ValueError(f"Unknown scale type: {scale_type}")

            params[param_name] = value

        return params

    def _apply_effect(
        self,
        audio: np.ndarray,
        effect_name: str,
        params: Dict[str, float],
    ) -> np.ndarray:
        """Apply DSP effect with given parameters."""
        if effect_name == 'lpf':
            return apply_lpf(audio, self.sample_rate, params['cutoff_freq'])
        elif effect_name == 'highshelf':
            return apply_highshelf(
                audio,
                self.sample_rate,
                params['gain'],
                params['freq'],
            )
        elif effect_name == 'saturation':
            return apply_saturation(audio, params['drive'])
        elif effect_name == 'reverb':
            return apply_reverb(
                audio,
                self.sample_rate,
                params['room_size'],
                params['damping'],
                params['wet_level'],
            )
        else:
            raise ValueError(f"Unknown effect: {effect_name}")

    def _get_text_condition(self, effect_name: str) -> str:
        """Get text condition based on corresponding image effect."""
        if effect_name not in self.correspondence:
            # Fallback to generic description
            return self.text_template.format(effect=effect_name)

        # Sample from top-k correspondences
        image_effects, probs = self.correspondence[effect_name]
        selected_image_effect = np.random.choice(image_effects, p=probs)

        # Extract base effect name (remove intensity if present)
        image_base = selected_image_effect.split(' (')[0]

        # Generate text condition
        return self.text_template.format(effect=image_base)

    def _normalize_params(
        self,
        params: Dict[str, float],
        effect_name: str,
    ) -> torch.Tensor:
        """Normalize parameters to [0, 1] range."""
        if effect_name not in self.param_specs:
            raise ValueError(f"Unknown effect: {effect_name}")

        normalized = []
        for param_name, (min_val, max_val, scale_type) in self.param_specs[effect_name].items():
            value = params[param_name]

            if scale_type == 'linear':
                norm_val = (value - min_val) / (max_val - min_val)
            elif scale_type == 'log':
                log_min = np.log(min_val)
                log_max = np.log(max_val)
                log_val = np.log(value)
                norm_val = (log_val - log_min) / (log_max - log_min)
            else:
                raise ValueError(f"Unknown scale type: {scale_type}")

            # Clamp to [0, 1]
            norm_val = np.clip(norm_val, 0.0, 1.0)
            normalized.append(norm_val)

        return torch.tensor(normalized, dtype=torch.float32)


def load_phase3_dataset(
    audio_dir: str,
    discovery_matrix_path: str,
    discovery_labels_path: str,
    sample_rate: int = 48000,
    duration: float = 10.0,
    max_files: Optional[int] = None,
) -> Phase3Dataset:
    """
    Load Phase 3 dataset from discovery results.

    Args:
        audio_dir: Directory containing audio files
        discovery_matrix_path: Path to discovery_matrix.npy
        discovery_labels_path: Path to discovery_labels.json
        sample_rate: Target sample rate
        duration: Audio duration in seconds
        max_files: Maximum number of audio files to use

    Returns:
        Phase3Dataset instance
    """
    # Load discovery results
    discovery_matrix = np.load(discovery_matrix_path)
    with open(discovery_labels_path, 'r') as f:
        labels = json.load(f)
    image_effects = labels['image_labels']
    audio_effects = labels['audio_labels']

    # Get audio file paths
    audio_dir = Path(audio_dir)
    audio_paths = []
    for ext in ['*.wav', '*.mp3', '*.flac']:
        audio_paths.extend(list(audio_dir.rglob(ext)))

    audio_paths = [str(p) for p in audio_paths]

    if max_files is not None:
        audio_paths = audio_paths[:max_files]

    print(f"Loaded {len(audio_paths)} audio files for Phase 3 training")

    return Phase3Dataset(
        audio_paths=audio_paths,
        discovery_matrix=discovery_matrix,
        image_effects=image_effects,
        audio_effects=audio_effects,
        sample_rate=sample_rate,
        duration=duration,
    )
