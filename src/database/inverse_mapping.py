"""
Inverse Mapping Database (Phase A - Step 3)

Pre-computes training data for the AudioController:
1. Sample random pedalboard parameters
2. Apply to source audio files
3. Embed source audio with CLAP -> CLAP(A) as input context
4. Label with nearest AUD_VOCAB terms (soft label via softmax)
5. Store as HDF5 for efficient random access

Each record:
  - clap_embedding: (512,) float32   # CLAP(A), raw input context
  - style_label: (|AUD_VOCAB|,) float32
  - normalized_params: (total_params,) float32
"""

import numpy as np
import torch
import h5py
import json
from contextlib import nullcontext
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
    save_augmented_audio: bool = False,
    augmented_audio_dir: Optional[str] = None,
    min_active_effects: int = 1,
    max_active_effects: Optional[int] = None,
    effect_sampling_weights: Optional[Dict[str, float]] = None,
) -> InverseMappingDB:
    """
    Build the inverse mapping database.

    For each audio file, generates multiple augmented versions with random
    pedalboard parameters. Stores CLAP(A) as input context and computes
    style labels from CLAP(A') vs AUD_VOCAB similarity.
    """
    import librosa
    import soundfile as sf

    rng = np.random.default_rng(seed)
    renderer = PedalboardRenderer(sample_rate=sample_rate)
    total_params = get_total_param_count(effect_names)
    vocab_size = aud_vocab_embeddings.shape[0]
    total_records = len(audio_paths) * num_augmentations_per_audio
    if max_active_effects is None:
        max_active_effects = len(effect_names)
    min_active_effects = max(1, min(min_active_effects, len(effect_names)))
    max_active_effects = max(min_active_effects, min(max_active_effects, len(effect_names)))
    # Optional weighted sampling across effects (without replacement).
    # If unspecified, all effects are sampled uniformly.
    if effect_sampling_weights is None:
        effect_sampling_weights = {}
    raw_weights = np.array(
        [float(effect_sampling_weights.get(name, 1.0)) for name in effect_names],
        dtype=np.float64,
    )
    if np.any(raw_weights < 0):
        raise ValueError("effect_sampling_weights must be non-negative")
    if raw_weights.sum() <= 0:
        raise ValueError("effect_sampling_weights must have at least one positive value")
    sampling_probs = (raw_weights / raw_weights.sum()).astype(np.float64)

    print(f"Building inverse mapping database...")
    print(f"  Audio files: {len(audio_paths)}")
    print(f"  Augmentations/file: {num_augmentations_per_audio}")
    print(f"  Total records: {total_records}")
    print(f"  Effects: {effect_names}")
    print(f"  Active effects per sample: {min_active_effects}..{max_active_effects}")
    print("  Effect sampling probabilities:")
    for name, p in zip(effect_names, sampling_probs):
        print(f"    {name}: {p:.4f}")
    print(f"  Total params: {total_params}, Vocab size: {vocab_size}")
    if save_augmented_audio:
        if not augmented_audio_dir:
            raise ValueError("augmented_audio_dir must be set when save_augmented_audio=True")
        print(f"  Augmented audio dir: {augmented_audio_dir}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    aug_root = None
    manifest_path = None
    if save_augmented_audio:
        aug_root = Path(augmented_audio_dir)
        aug_root.mkdir(parents=True, exist_ok=True)
        manifest_path = aug_root / "manifest.jsonl"

    with h5py.File(output_path, 'w') as f:
        clap_ds = f.create_dataset('clap_embeddings', shape=(total_records, 512), dtype='float32')
        style_ds = f.create_dataset('style_labels', shape=(total_records, vocab_size), dtype='float32')
        params_ds = f.create_dataset('normalized_params', shape=(total_records, total_params), dtype='float32')

        f.attrs['effect_names'] = ','.join(effect_names)
        f.attrs['total_params'] = total_params
        f.attrs['vocab_size'] = vocab_size
        f.attrs['sample_rate'] = sample_rate
        f.attrs['temperature'] = temperature
        f.attrs['input_context'] = 'clap_raw_audio'
        f.attrs['min_active_effects'] = min_active_effects
        f.attrs['max_active_effects'] = max_active_effects
        f.attrs['effect_sampling_weights_json'] = json.dumps(
            {name: float(w) for name, w in zip(effect_names, raw_weights)}
        )

        record_idx = 0
        active_effect_counts = {name: 0 for name in effect_names}
        with open(manifest_path, 'w') if manifest_path else nullcontext() as manifest_fp:
            for audio_path in tqdm(audio_paths, desc="Building inverse mapping DB"):
                try:
                    waveform, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
                    max_samples = int(max_duration * sample_rate)
                    if len(waveform) > max_samples:
                        waveform = waveform[:max_samples]

                    # Input context uses raw audio A (not augmented A')
                    raw_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
                    clap_raw = clap_embedder.embed_audio(raw_tensor, sample_rate)[0].cpu().numpy()
                    raw_norm = np.linalg.norm(clap_raw)
                    if raw_norm > 0:
                        clap_raw = clap_raw / raw_norm

                    source = Path(audio_path)
                    category = source.parent.name
                    source_stem = source.stem

                    for aug_idx in range(num_augmentations_per_audio):
                        # Randomly select subset of effects, then sample params.
                        n_active = int(rng.integers(min_active_effects, max_active_effects + 1))
                        active_effects = list(
                            rng.choice(
                                effect_names,
                                size=n_active,
                                replace=False,
                                p=sampling_probs,
                            )
                        )
                        for effect in active_effects:
                            active_effect_counts[effect] += 1
                        params_dict = sample_random_params(active_effects, rng)
                        norm_params = normalize_params(params_dict, effect_names)

                        # Apply effects
                        processed = renderer.render(waveform, params_dict)

                        aug_relpath = None
                        if aug_root is not None:
                            aug_subdir = aug_root / category
                            aug_subdir.mkdir(parents=True, exist_ok=True)
                            aug_name = f"{source_stem}__aug_{aug_idx:04d}.wav"
                            aug_path = aug_subdir / aug_name
                            sf.write(str(aug_path), processed, sample_rate)
                            aug_relpath = str(aug_path)

                        # Style label is post-hoc from processed audio A'
                        processed_tensor = torch.from_numpy(processed).float().unsqueeze(0)
                        clap_proc = clap_embedder.embed_audio(processed_tensor, sample_rate)[0].cpu().numpy()
                        proc_norm = np.linalg.norm(clap_proc)
                        if proc_norm > 0:
                            clap_proc = clap_proc / proc_norm
                        style_label = _compute_style_label(clap_proc, aud_vocab_embeddings, temperature)

                        clap_ds[record_idx] = clap_raw
                        style_ds[record_idx] = style_label
                        params_ds[record_idx] = norm_params

                        if manifest_fp is not None:
                            # Keep a lightweight trace so generated audio can be audited later.
                            manifest_fp.write(json.dumps({
                                'record_index': record_idx,
                                'source_audio_path': audio_path,
                                'augmented_audio_path': aug_relpath,
                                'active_effects': active_effects,
                                'sample_rate': sample_rate,
                                'params': params_dict,
                            }) + "\n")

                        record_idx += 1

                except Exception as e:
                    print(f"Warning: Failed {audio_path}: {e}")
                    continue

        f.attrs['actual_records'] = record_idx

    print(f"Saved {record_idx} records to {output_path}")
    if record_idx > 0:
        print("Active effect selection counts:")
        total_selected = sum(active_effect_counts.values())
        for name in effect_names:
            count = active_effect_counts[name]
            frac = (count / total_selected) if total_selected > 0 else 0.0
            print(f"  {name}: {count} ({frac:.4f})")
    return InverseMappingDB(output_path)
