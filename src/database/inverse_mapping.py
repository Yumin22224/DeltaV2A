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
  - effect_active_mask: (|effects|,) float32
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
            self.clap_delta_norm = f['clap_delta_norm'][:actual] if 'clap_delta_norm' in f else None
            effect_names_raw = str(f.attrs.get('effect_names', '')).strip()
            self.effect_names = [x for x in effect_names_raw.split(',') if x]
            if 'effect_active_mask' in f:
                self.effect_active_mask = f['effect_active_mask'][:actual]
            else:
                self.effect_active_mask = _infer_effect_active_masks_from_params(
                    self.normalized_params, self.effect_names
                )
            _order_obj = f.get('effect_order')
            if isinstance(_order_obj, h5py.Dataset):
                self.effect_order = _order_obj[:actual]
            else:
                self.effect_order = None
        self._size = len(self.clap_embeddings)

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            'clap_embedding': torch.from_numpy(self.clap_embeddings[idx]),
            'style_label': torch.from_numpy(self.style_labels[idx]),
            'normalized_params': torch.from_numpy(self.normalized_params[idx]),
            'effect_active_mask': torch.from_numpy(self.effect_active_mask[idx]),
        }
        if self.clap_delta_norm is not None:
            item['clap_delta_norm'] = torch.tensor(float(self.clap_delta_norm[idx]), dtype=torch.float32)
        return item


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


def _compute_style_label_quality(style_label: np.ndarray) -> Dict[str, float]:
    """Compute simple quality diagnostics for a soft style label."""
    eps = 1e-12
    entropy_bits = float(-np.sum(style_label * np.log2(style_label + eps)))
    if style_label.size >= 2:
        top2 = np.partition(style_label, -2)[-2:]
        top1_mass = float(top2[-1])
        top2_mass = float(top2[-2])
    elif style_label.size == 1:
        top1_mass = float(style_label[0])
        top2_mass = 0.0
    else:
        top1_mass = 0.0
        top2_mass = 0.0
    return {
        'entropy_bits': entropy_bits,
        'top1_mass': top1_mass,
        'margin_top1_top2': float(top1_mass - top2_mass),
    }


def _effect_param_slices(effect_names: List[str]) -> List[slice]:
    slices: List[slice] = []
    start = 0
    for effect_name in effect_names:
        width = EFFECT_CATALOG[effect_name].num_params
        slices.append(slice(start, start + width))
        start += width
    return slices


def _infer_effect_active_masks_from_params(
    normalized_params: np.ndarray,
    effect_names: List[str],
    eps: float = 1e-4,
) -> np.ndarray:
    """
    Infer effect on/off masks from normalized params.

    This is used as a fallback for old DB files that were built before
    `effect_active_mask` was added.
    """
    if normalized_params.ndim != 2:
        raise ValueError(f"normalized_params must be 2D, got {normalized_params.shape}")
    if not effect_names:
        return np.zeros((normalized_params.shape[0], 0), dtype=np.float32)

    bypass_norm = normalize_params({}, effect_names)
    slices = _effect_param_slices(effect_names)
    masks = np.zeros((normalized_params.shape[0], len(effect_names)), dtype=np.float32)
    for i, sl in enumerate(slices):
        diff = np.abs(normalized_params[:, sl] - bypass_norm[sl])
        masks[:, i] = (np.max(diff, axis=1) > eps).astype(np.float32)
    return masks


def _save_checkpoint(
    checkpoint_path: Path,
    completed_audio_files: int,
    record_idx: int,
    rng: np.random.Generator,
    active_effect_counts: Dict[str, int],
) -> None:
    """Save build progress so it can be resumed later."""
    state = rng.bit_generator.state
    ckpt = {
        'completed_audio_files': completed_audio_files,
        'record_idx': record_idx,
        'active_effect_counts': active_effect_counts,
        'rng_state': {
            'bit_generator': state['bit_generator'],
            'state': str(state['state']['state']),
            'inc': str(state['state']['inc']),
            'has_uint32': int(state['has_uint32']),
            'uinteger': int(state['uinteger']),
        },
    }
    tmp = checkpoint_path.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(ckpt, f)
    tmp.replace(checkpoint_path)


def _load_checkpoint(checkpoint_path: Path) -> Dict:
    """Load a previously saved checkpoint."""
    with open(checkpoint_path, 'r') as f:
        ckpt = json.load(f)
    # Restore RNG state from JSON-safe format
    rs = ckpt['rng_state']
    ckpt['_rng_bit_generator_state'] = {
        'bit_generator': rs['bit_generator'],
        'state': {'state': int(rs['state']), 'inc': int(rs['inc'])},
        'has_uint32': rs['has_uint32'],
        'uinteger': rs['uinteger'],
    }
    return ckpt


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
    use_delta_clap: bool = True,
    resume: bool = False,
    param_min_intensity: float = 0.0,
    delta_min_norm: float = 0.0,
    delta_resample_attempts: int = 1,
    param_mid_bypass_exclusion: float = 0.0,
    single_effect_ratio: float = 0.0,
    label_entropy_max_bits: Optional[float] = None,
    label_top1_min_mass: float = 0.0,
    label_min_margin: float = 0.0,
    label_resample_attempts: int = 1,
) -> InverseMappingDB:
    """
    Build the inverse mapping database.

    For each audio file, generates multiple augmented versions with random
    pedalboard parameters. Stores CLAP(A) as input context and computes
    style labels from AUD_VOCAB similarity.

    When use_delta_clap=True (default), style labels are computed from
    the delta embedding CLAP(A') - CLAP(A), which isolates the effect-induced
    change and cancels the genre/content signal that dominates absolute
    embeddings. This aligns with inference where image deltas
    CLIP(I') - CLIP(I) are used for style retrieval.

    When resume=True, attempts to continue from a previous interrupted build
    by reading the checkpoint file (output_path.checkpoint.json). The RNG
    state, record index, and audio file index are restored so results are
    identical to a fresh build.
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
    single_effect_ratio = float(np.clip(single_effect_ratio, 0.0, 1.0))
    label_top1_min_mass = float(np.clip(label_top1_min_mass, 0.0, 1.0))
    label_min_margin = float(max(0.0, label_min_margin))
    label_resample_attempts = max(int(label_resample_attempts), 1)
    if label_entropy_max_bits is not None:
        label_entropy_max_bits = float(label_entropy_max_bits)
        if label_entropy_max_bits <= 0.0:
            label_entropy_max_bits = None

    # --- Resume support ---
    checkpoint_path = Path(output_path).with_suffix('.checkpoint.json')
    start_audio_idx = 0
    record_idx = 0
    active_effect_counts = {name: 0 for name in effect_names}
    resuming = False

    if resume and Path(output_path).exists() and checkpoint_path.exists():
        ckpt = _load_checkpoint(checkpoint_path)
        start_audio_idx = ckpt['completed_audio_files']
        record_idx = ckpt['record_idx']
        active_effect_counts = ckpt.get('active_effect_counts', active_effect_counts)
        rng.bit_generator.state = ckpt['_rng_bit_generator_state']
        resuming = True
        print(f"Resuming inverse mapping build from checkpoint:")
        print(f"  Completed audio files: {start_audio_idx}/{len(audio_paths)}")
        print(f"  Records written: {record_idx}/{total_records}")

    print(f"Building inverse mapping database...")
    print(f"  Audio files: {len(audio_paths)}")
    print(f"  Augmentations/file: {num_augmentations_per_audio}")
    print(f"  Total records: {total_records}")
    print(f"  Effects: {effect_names}")
    print(f"  Active effects per sample: {min_active_effects}..{max_active_effects}")
    if min_active_effects <= 1 <= max_active_effects:
        print(f"  Single-effect ratio target: {single_effect_ratio:.2f}")
    style_mode = "delta CLAP(A')-CLAP(A)" if use_delta_clap else "absolute CLAP(A')"
    print(f"  Style label mode: {style_mode}")
    if use_delta_clap:
        print(f"  Delta min norm: {float(delta_min_norm):.4f}")
        print(f"  Delta resample attempts: {int(max(delta_resample_attempts, 1))}")
    print(
        "  Label quality filter: "
        f"entropy<= {label_entropy_max_bits if label_entropy_max_bits is not None else 'off'}, "
        f"top1>= {label_top1_min_mass:.4f}, margin>= {label_min_margin:.4f}, "
        f"resample_attempts={label_resample_attempts}"
    )
    print("  Effect sampling probabilities:")
    for name, p in zip(effect_names, sampling_probs):
        print(f"    {name}: {p:.4f}")
    print(f"  Total params: {total_params}, Vocab size: {vocab_size}")
    if save_augmented_audio:
        if not augmented_audio_dir:
            raise ValueError("augmented_audio_dir must be set when save_augmented_audio=True")
        print(f"  Augmented audio dir: {augmented_audio_dir}")

    print(f"  Mid-bypass exclusion: {float(param_mid_bypass_exclusion):.4f}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    aug_root = None
    manifest_path = None
    if save_augmented_audio:
        aug_root = Path(augmented_audio_dir)
        aug_root.mkdir(parents=True, exist_ok=True)
        manifest_path = aug_root / "manifest.jsonl"

    h5_mode = 'r+' if resuming else 'w'
    manifest_mode = 'a' if resuming else 'w'

    with h5py.File(output_path, h5_mode) as f:
        delta_norm_ds = None
        if resuming:
            clap_ds = f['clap_embeddings']
            style_ds = f['style_labels']
            params_ds = f['normalized_params']
            active_ds = f['effect_active_mask']
            order_ds = f['effect_order'] if 'effect_order' in f else f.create_dataset(
                'effect_order', shape=(total_records, max_active_effects), dtype='int8',
                fillvalue=-1,
            )
            if use_delta_clap:
                if 'clap_delta_norm' in f:
                    delta_norm_ds = f['clap_delta_norm']
                else:
                    delta_norm_ds = f.create_dataset('clap_delta_norm', shape=(total_records,), dtype='float32')
        else:
            clap_ds = f.create_dataset('clap_embeddings', shape=(total_records, 512), dtype='float32')
            style_ds = f.create_dataset('style_labels', shape=(total_records, vocab_size), dtype='float32')
            params_ds = f.create_dataset('normalized_params', shape=(total_records, total_params), dtype='float32')
            active_ds = f.create_dataset('effect_active_mask', shape=(total_records, len(effect_names)), dtype='float32')
            order_ds = f.create_dataset('effect_order', shape=(total_records, max_active_effects), dtype='int8',
                                        fillvalue=-1)
            if use_delta_clap:
                delta_norm_ds = f.create_dataset('clap_delta_norm', shape=(total_records,), dtype='float32')

            f.attrs['effect_names'] = ','.join(effect_names)
            f.attrs['total_params'] = total_params
            f.attrs['vocab_size'] = vocab_size
            f.attrs['sample_rate'] = sample_rate
            f.attrs['temperature'] = temperature
            f.attrs['input_context'] = 'clap_raw_audio'
            f.attrs['use_delta_clap'] = bool(use_delta_clap)
            f.attrs['num_effects'] = len(effect_names)
            f.attrs['min_active_effects'] = min_active_effects
            f.attrs['max_active_effects'] = max_active_effects
            f.attrs['effect_sampling_weights_json'] = json.dumps(
                {name: float(w) for name, w in zip(effect_names, raw_weights)}
            )

        # Keep metadata consistent for both fresh and resumed runs.
        f.attrs['delta_min_norm'] = float(delta_min_norm)
        f.attrs['delta_resample_attempts'] = int(max(delta_resample_attempts, 1))
        f.attrs['param_min_intensity'] = float(param_min_intensity)
        f.attrs['param_mid_bypass_exclusion'] = float(param_mid_bypass_exclusion)
        f.attrs['single_effect_ratio'] = float(single_effect_ratio)
        f.attrs['label_entropy_max_bits'] = (
            float(label_entropy_max_bits) if label_entropy_max_bits is not None else -1.0
        )
        f.attrs['label_top1_min_mass'] = float(label_top1_min_mass)
        f.attrs['label_min_margin'] = float(label_min_margin)
        f.attrs['label_resample_attempts'] = int(label_resample_attempts)

        delta_norm_sum = 0.0
        delta_norm_count = 0
        delta_low_norm_rejected = 0
        delta_low_norm_fallback = 0
        label_entropy_sum = 0.0
        label_top1_sum = 0.0
        label_margin_sum = 0.0
        label_count = 0
        label_low_quality_rejected = 0
        label_low_quality_fallback = 0

        with open(manifest_path, manifest_mode) if manifest_path else nullcontext() as manifest_fp:
            pbar = tqdm(
                enumerate(audio_paths),
                total=len(audio_paths),
                initial=start_audio_idx,
                desc="Building inverse mapping DB",
            )
            for audio_idx, audio_path in pbar:
                if audio_idx < start_audio_idx:
                    continue

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

                    # Pre-compute original RMS for volume normalization.
                    rms_original = float(np.sqrt(np.mean(waveform ** 2)))

                    for aug_idx in range(num_augmentations_per_audio):
                        # Retry low-SNR deltas so style labels are less dominated by noise.
                        # If all attempts are low-norm, keep the best candidate to preserve record count.
                        attempts = max(
                            max(int(delta_resample_attempts), 1) if use_delta_clap else 1,
                            label_resample_attempts,
                        )
                        best_candidate = None
                        best_score = float("-inf")

                        for _attempt in range(attempts):
                            # Randomly select subset of effects, then sample params.
                            if min_active_effects == max_active_effects:
                                n_active = int(min_active_effects)
                            elif min_active_effects <= 1 <= max_active_effects and single_effect_ratio > 0.0:
                                if float(rng.random()) < single_effect_ratio:
                                    n_active = 1
                                else:
                                    low = max(2, min_active_effects)
                                    n_active = int(rng.integers(low, max_active_effects + 1))
                            else:
                                n_active = int(rng.integers(min_active_effects, max_active_effects + 1))
                            active_effects = list(
                                rng.choice(
                                    effect_names,
                                    size=n_active,
                                    replace=False,
                                    p=sampling_probs,
                                )
                            )
                            # Explicitly shuffle application order so the AR controller
                            # can learn diverse effect chain orderings.
                            rng.shuffle(active_effects)

                            params_dict = sample_random_params(
                                active_effects,
                                rng,
                                param_min_intensity=param_min_intensity,
                                mid_bypass_exclusion=param_mid_bypass_exclusion,
                            )
                            norm_params = normalize_params(params_dict, effect_names)
                            active_mask = np.array(
                                [1.0 if name in active_effects else 0.0 for name in effect_names],
                                dtype=np.float32,
                            )
                            # effect_order: indices of active effects in application order, -1 padded.
                            name_to_idx = {name: i for i, name in enumerate(effect_names)}
                            effect_order = np.full(max_active_effects, -1, dtype=np.int8)
                            for step, eff in enumerate(active_effects):
                                effect_order[step] = name_to_idx[eff]

                            # Apply effects in the shuffled order.
                            processed = renderer.render(waveform, params_dict)

                            # Volume normalization: match processed RMS to original so that
                            # CLAP delta captures timbral/spectral change only, not loudness shift.
                            rms_processed = float(np.sqrt(np.mean(processed ** 2)))
                            if rms_processed > 1e-8 and rms_original > 1e-8:
                                processed = processed * (rms_original / rms_processed)
                                # Re-clamp to prevent clipping after rescaling.
                                peak = float(np.abs(processed).max())
                                if peak > 1.0:
                                    processed = processed / peak * 0.95

                            # Style label is post-hoc from processed audio A'
                            processed_tensor = torch.from_numpy(processed).float().unsqueeze(0)
                            clap_proc = clap_embedder.embed_audio(processed_tensor, sample_rate)[0].cpu().numpy()
                            proc_norm = np.linalg.norm(clap_proc)
                            if proc_norm > 0:
                                clap_proc = clap_proc / proc_norm

                            delta_norm = 0.0
                            if use_delta_clap:
                                # Keep raw delta magnitude (no per-sample re-normalization):
                                # small norms are lower-confidence signals by construction.
                                clap_delta = clap_proc - clap_raw
                                delta_norm = float(np.linalg.norm(clap_delta))
                                style_label = _compute_style_label(clap_delta, aud_vocab_embeddings, temperature)
                            else:
                                style_label = _compute_style_label(clap_proc, aud_vocab_embeddings, temperature)

                            label_quality = _compute_style_label_quality(style_label)
                            label_entropy = float(label_quality['entropy_bits'])
                            label_top1 = float(label_quality['top1_mass'])
                            label_margin = float(label_quality['margin_top1_top2'])
                            max_entropy = float(np.log2(max(vocab_size, 2)))

                            delta_ok = (not use_delta_clap) or (delta_norm >= float(delta_min_norm))
                            label_ok = True
                            if label_entropy_max_bits is not None and label_entropy > label_entropy_max_bits:
                                label_ok = False
                            if label_top1 < label_top1_min_mass:
                                label_ok = False
                            if label_margin < label_min_margin:
                                label_ok = False

                            # Prefer candidates with stronger delta signal and sharper labels.
                            candidate_score = (
                                delta_norm
                                + 0.25 * label_top1
                                + 0.25 * label_margin
                                - 0.10 * (label_entropy / max(max_entropy, 1e-6))
                            )
                            if not delta_ok:
                                candidate_score -= 0.5
                            if not label_ok:
                                candidate_score -= 1.0

                            candidate = {
                                'active_effects': active_effects,
                                'params_dict': params_dict,
                                'norm_params': norm_params,
                                'active_mask': active_mask,
                                'effect_order': effect_order,
                                'processed': processed,
                                'style_label': style_label,
                                'delta_norm': delta_norm,
                                'label_entropy': label_entropy,
                                'label_top1': label_top1,
                                'label_margin': label_margin,
                                'delta_ok': bool(delta_ok),
                                'label_ok': bool(label_ok),
                                'candidate_score': float(candidate_score),
                            }

                            if candidate_score > best_score:
                                best_score = candidate_score
                                best_candidate = candidate

                            if delta_ok and label_ok:
                                break
                            if not delta_ok:
                                delta_low_norm_rejected += 1
                            if not label_ok:
                                label_low_quality_rejected += 1

                        if best_candidate is None:
                            continue

                        if use_delta_clap and not best_candidate['delta_ok']:
                            delta_low_norm_fallback += 1
                        if not best_candidate['label_ok']:
                            label_low_quality_fallback += 1

                        active_effects = best_candidate['active_effects']
                        params_dict = best_candidate['params_dict']
                        norm_params = best_candidate['norm_params']
                        active_mask = best_candidate['active_mask']
                        effect_order = best_candidate['effect_order']
                        processed = best_candidate['processed']
                        style_label = best_candidate['style_label']
                        delta_norm = float(best_candidate['delta_norm'])
                        label_entropy = float(best_candidate['label_entropy'])
                        label_top1 = float(best_candidate['label_top1'])
                        label_margin = float(best_candidate['label_margin'])

                        for effect in active_effects:
                            active_effect_counts[effect] += 1

                        label_entropy_sum += label_entropy
                        label_top1_sum += label_top1
                        label_margin_sum += label_margin
                        label_count += 1

                        if use_delta_clap:
                            delta_norm_sum += delta_norm
                            delta_norm_count += 1

                        aug_relpath = None
                        if aug_root is not None:
                            aug_subdir = aug_root / category
                            aug_subdir.mkdir(parents=True, exist_ok=True)
                            aug_name = f"{source_stem}__aug_{aug_idx:04d}.wav"
                            aug_path = aug_subdir / aug_name
                            sf.write(str(aug_path), processed, sample_rate)
                            aug_relpath = str(aug_path)

                        clap_ds[record_idx] = clap_raw
                        style_ds[record_idx] = style_label
                        params_ds[record_idx] = norm_params
                        active_ds[record_idx] = active_mask
                        order_ds[record_idx] = effect_order
                        if delta_norm_ds is not None:
                            delta_norm_ds[record_idx] = np.float32(delta_norm)

                        if manifest_fp is not None:
                            # Keep a lightweight trace so generated audio can be audited later.
                            manifest_fp.write(json.dumps({
                                'record_index': record_idx,
                                'source_audio_path': audio_path,
                                'augmented_audio_path': aug_relpath,
                                'active_effects': active_effects,
                                'effect_order': [int(x) for x in effect_order],
                                'sample_rate': sample_rate,
                                'params': params_dict,
                                'clap_delta_norm': delta_norm if use_delta_clap else None,
                                'style_entropy_bits': label_entropy,
                                'style_top1_mass': label_top1,
                                'style_margin_top1_top2': label_margin,
                            }) + "\n")

                        record_idx += 1

                except Exception as e:
                    print(f"Warning: Failed {audio_path}: {e}")
                    continue

                # Save checkpoint after each audio file completes.
                f.attrs['actual_records'] = record_idx
                f.flush()
                if manifest_fp is not None:
                    manifest_fp.flush()
                _save_checkpoint(
                    checkpoint_path, audio_idx + 1, record_idx, rng, active_effect_counts,
                )

        f.attrs['actual_records'] = record_idx
        if use_delta_clap:
            f.attrs['delta_low_norm_rejected'] = int(delta_low_norm_rejected)
            f.attrs['delta_low_norm_fallback'] = int(delta_low_norm_fallback)
            mean_delta = (delta_norm_sum / max(delta_norm_count, 1)) if delta_norm_count > 0 else 0.0
            f.attrs['delta_norm_mean'] = float(mean_delta)
        f.attrs['label_low_quality_rejected'] = int(label_low_quality_rejected)
        f.attrs['label_low_quality_fallback'] = int(label_low_quality_fallback)
        f.attrs['style_entropy_mean'] = (
            float(label_entropy_sum / max(label_count, 1)) if label_count > 0 else 0.0
        )
        f.attrs['style_top1_mass_mean'] = (
            float(label_top1_sum / max(label_count, 1)) if label_count > 0 else 0.0
        )
        f.attrs['style_margin_mean'] = (
            float(label_margin_sum / max(label_count, 1)) if label_count > 0 else 0.0
        )

    # Clean up checkpoint on successful completion.
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Checkpoint file removed (build complete).")

    print(f"Saved {record_idx} records to {output_path}")
    if record_idx > 0:
        print("Active effect selection counts:")
        total_selected = sum(active_effect_counts.values())
        for name in effect_names:
            count = active_effect_counts[name]
            frac = (count / total_selected) if total_selected > 0 else 0.0
            print(f"  {name}: {count} ({frac:.4f})")
        if use_delta_clap:
            mean_delta = (delta_norm_sum / max(delta_norm_count, 1)) if delta_norm_count > 0 else 0.0
            print(
                "Delta norm stats: "
                f"mean={mean_delta:.6f}, rejected_low_norm={delta_low_norm_rejected}, "
                f"fallback_low_norm={delta_low_norm_fallback}"
            )
        mean_entropy = (label_entropy_sum / max(label_count, 1)) if label_count > 0 else 0.0
        mean_top1 = (label_top1_sum / max(label_count, 1)) if label_count > 0 else 0.0
        mean_margin = (label_margin_sum / max(label_count, 1)) if label_count > 0 else 0.0
        print(
            "Style label stats: "
            f"mean_entropy={mean_entropy:.6f}, mean_top1={mean_top1:.6f}, "
            f"mean_margin={mean_margin:.6f}, rejected_low_quality={label_low_quality_rejected}, "
            f"fallback_low_quality={label_low_quality_fallback}"
        )
    return InverseMappingDB(output_path)
