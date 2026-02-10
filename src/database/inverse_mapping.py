"""
Inverse Mapping Database (Phase A - Step 3)

Pre-computes training data for the AudioController:
1. Sample random pedalboard parameters
2. Apply to source audio files
3. Embed processed audio with CLAP -> CLAP(A')
4. Label with nearest AUD_VOCAB terms (soft label via softmax)
5. Store as HDF5 for efficient random access

Each record:
  - clap_embedding: (512,) float32
  - style_label: (|AUD_VOCAB|,) float32
  - normalized_params: (total_params,) float32
"""

import numpy as np
import torch
import h5py
from typing import Dict, List, Optional
from pathlib import Path
from tqdm import tqdm

from ..effects.pedalboard_effects import (
    EFFECT_CATALOG,
    PedalboardRenderer,
    sample_random_params,
    normalize_params,
    get_total_param_count,
)


class InverseMappingDB:
    """HDF5-backed database of (CLAP_embedding, style_label, params) records."""

    def __init__(self, path: str):
        self.path = path

    @property
    def exists(self) -> bool:
        return Path(self.path).exists()

    def get_size(self) -> int:
        with h5py.File(self.path, 'r') as f:
            return int(f.attrs.get('actual_records', f['clap_embeddings'].shape[0]))

    def get_metadata(self) -> Dict:
        with h5py.File(self.path, 'r') as f:
            return dict(f.attrs)


class InverseMappingDataset:
    """PyTorch-compatible dataset wrapper for InverseMappingDB."""

    def __init__(self, db: InverseMappingDB):
        self.db = db
        with h5py.File(db.path, 'r') as f:
            actual = int(f.attrs.get('actual_records', f['clap_embeddings'].shape[0]))
            self.clap_embeddings = f['clap_embeddings'][:actual]
            self.style_labels = f['style_labels'][:actual]
            self.normalized_params = f['normalized_params'][:actual]
        self._size = len(self.clap_embeddings)

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'clap_embedding': torch.from_numpy(self.clap_embeddings[idx]),
            'style_label': torch.from_numpy(self.style_labels[idx]),
            'normalized_params': torch.from_numpy(self.normalized_params[idx]),
        }


def _compute_style_label(
    clap_embedding: np.ndarray,
    aud_vocab_embeddings: np.ndarray,
    temperature: float = 0.1,
) -> np.ndarray:
    """
    Compute soft style label via similarity to AUD_VOCAB.

    Uses softmax over cosine similarities with temperature scaling.
    """
    sims = aud_vocab_embeddings @ clap_embedding
    logits = sims / temperature
    logits = logits - logits.max()
    exp_logits = np.exp(logits)
    return (exp_logits / exp_logits.sum()).astype(np.float32)


def build_inverse_mapping_db(
    audio_paths: List[str],
    clap_embedder,
    aud_vocab_embeddings: np.ndarray,
    effect_names: List[str],
    output_path: str,
    num_augmentations_per_audio: int = 10,
    sample_rate: int = 48000,
    max_duration: float = 20.0,
    temperature: float = 0.1,
    seed: int = 42,
) -> InverseMappingDB:
    """
    Build the inverse mapping database.

    For each audio file, generates multiple augmented versions with
    random pedalboard parameters, embeds them with CLAP, and labels
    with AUD_VOCAB soft labels.
    """
    import librosa

    rng = np.random.default_rng(seed)
    renderer = PedalboardRenderer(sample_rate=sample_rate)
    total_params = get_total_param_count(effect_names)
    vocab_size = aud_vocab_embeddings.shape[0]
    total_records = len(audio_paths) * num_augmentations_per_audio

    print(f"Building inverse mapping database...")
    print(f"  Audio files: {len(audio_paths)}")
    print(f"  Augmentations/file: {num_augmentations_per_audio}")
    print(f"  Total records: {total_records}")
    print(f"  Effects: {effect_names}")
    print(f"  Total params: {total_params}, Vocab size: {vocab_size}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        clap_ds = f.create_dataset('clap_embeddings', shape=(total_records, 512), dtype='float32')
        style_ds = f.create_dataset('style_labels', shape=(total_records, vocab_size), dtype='float32')
        params_ds = f.create_dataset('normalized_params', shape=(total_records, total_params), dtype='float32')

        f.attrs['effect_names'] = ','.join(effect_names)
        f.attrs['total_params'] = total_params
        f.attrs['vocab_size'] = vocab_size
        f.attrs['sample_rate'] = sample_rate
        f.attrs['temperature'] = temperature

        record_idx = 0

        for audio_path in tqdm(audio_paths, desc="Building inverse mapping DB"):
            try:
                waveform, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
                max_samples = int(max_duration * sample_rate)
                if len(waveform) > max_samples:
                    waveform = waveform[:max_samples]

                for _ in range(num_augmentations_per_audio):
                    # Random params
                    params_dict = sample_random_params(effect_names, rng)
                    norm_params = normalize_params(params_dict, effect_names)

                    # Apply effects
                    processed = renderer.render(waveform, params_dict)

                    # CLAP embed processed audio
                    processed_tensor = torch.from_numpy(processed).float()
                    if processed_tensor.ndim == 1:
                        processed_tensor = processed_tensor.unsqueeze(0)
                    clap_emb = clap_embedder.embed_audio(processed_tensor, sample_rate)[0].cpu().numpy()

                    # L2 normalize
                    norm = np.linalg.norm(clap_emb)
                    if norm > 0:
                        clap_emb = clap_emb / norm

                    # Style label
                    style_label = _compute_style_label(clap_emb, aud_vocab_embeddings, temperature)

                    clap_ds[record_idx] = clap_emb
                    style_ds[record_idx] = style_label
                    params_ds[record_idx] = norm_params
                    record_idx += 1

            except Exception as e:
                print(f"Warning: Failed {audio_path}: {e}")
                for _ in range(num_augmentations_per_audio):
                    if record_idx < total_records:
                        record_idx += 1
                continue

        f.attrs['actual_records'] = record_idx

    print(f"Saved {record_idx} records to {output_path}")
    return InverseMappingDB(output_path)
