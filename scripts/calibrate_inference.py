#!/usr/bin/env python
"""
Calibration script for image-domain inference.

Two-stage calibration:
  Stage 1 – Joint (T*, norm_confidence_threshold, norm_confidence_scale):
    Generate (I, wand(I)) pairs. Compute CLIP delta cosine sims (T-independent)
    AND delta norms (||CLIP(I') - CLIP(I)||, before normalization).

    The style label uses norm-based confidence mixing:
      confidence = sigmoid((delta_norm - threshold) / scale)
      style_label = confidence * softmax(sims/T) + (1-confidence) * uniform(1/V)

    Joint grid search over (T, threshold, scale) minimizes |E[top1] - target_top1|.
    Tiebreaker: maximize std(confidence) so the threshold sits at a discriminative
    point in the norm distribution (not all 0 or all 1).

  Stage 2 – Activity threshold:
    Run controller with confidence-adjusted style labels.
    Find threshold where E[N_active] ≈ training mean.

Writes T*, norm_confidence_threshold, norm_confidence_scale, activity_threshold_override
to configs/pipeline.yaml.

Usage:
    python scripts/calibrate_inference.py --config configs/pipeline.yaml
"""

import argparse
import sys
import json
import random
import numpy as np
import yaml
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def softmax_top1(sims_matrix: np.ndarray, T: float) -> np.ndarray:
    """(N, V) raw sims -> (N,) top-1 softmax probability."""
    logits = sims_matrix / T
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs.max(axis=1)


def softmax_probs(sims_matrix: np.ndarray, T: float) -> np.ndarray:
    """(N, V) raw sims -> (N, V) softmax probabilities."""
    logits = sims_matrix / T
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Stage 1: collect raw cosine sims from image pairs
# ---------------------------------------------------------------------------

def collect_raw_sims(
    image_paths: list,
    img_vocab_embeddings: np.ndarray,
    clip_embedder,
    effect_names: list,
    intensity_range: tuple = (0.3, 1.0),
    seed: int = 42,
    batch_size: int = 32,
    device: str = "cuda",
):
    """
    For each (image, effect) pair:
      - apply wand effect at random intensity
      - embed original and edited with CLIP
      - compute cosine sims of normalized delta to IMG vocab
      - record delta norm BEFORE normalizing (used for confidence calibration)

    Returns:
        raw_sims:    (N, V) float32 -- T-independent cosine sims
        delta_norms: (N,)   float32 -- ||normalize(CLIP(I')) - normalize(CLIP(I))||
    """
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image as PILImage
    from src.effects.wand_image_effects import apply_effect, get_effect_types

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    available_effects = get_effect_types()
    effects_to_use = [e for e in effect_names if e in available_effects]
    if not effects_to_use:
        effects_to_use = available_effects
    print(f"Wand effects for calibration ({len(effects_to_use)}): {effects_to_use}")

    # Build all (image_path, effect, intensity) triples
    pairs = []
    for img_path in image_paths:
        for eff in effects_to_use:
            intensity = float(np_rng.uniform(intensity_range[0], intensity_range[1]))
            pairs.append((img_path, eff, intensity))
    rng.shuffle(pairs)
    print(f"Total calibration pairs: {len(pairs)}")

    all_sims = []
    all_norms = []
    skipped = 0

    # Process in batches of pairs
    i = 0
    while i < len(pairs):
        batch = pairs[i: i + batch_size]
        orig_tensors, edit_tensors = [], []

        for img_path, eff, intensity in batch:
            try:
                orig_pil = PILImage.open(img_path).convert("RGB")
                edit_pil = apply_effect(orig_pil, eff, intensity=intensity)
                orig_tensors.append(TF.to_tensor(orig_pil))
                edit_tensors.append(TF.to_tensor(edit_pil))
            except Exception:
                skipped += 1
                orig_tensors.append(None)
                edit_tensors.append(None)

        valid_idx = [j for j, t in enumerate(orig_tensors) if t is not None]
        if not valid_idx:
            i += batch_size
            continue

        import torch
        orig_batch = torch.stack([orig_tensors[j] for j in valid_idx]).to(device)
        edit_batch = torch.stack([edit_tensors[j] for j in valid_idx]).to(device)

        with torch.no_grad():
            e_orig = clip_embedder.embed_images(orig_batch).cpu().numpy().astype(np.float32)
            e_edit = clip_embedder.embed_images(edit_batch).cpu().numpy().astype(np.float32)

        # L2-normalize CLIP embeddings (unit sphere)
        e_orig = e_orig / np.maximum(np.linalg.norm(e_orig, axis=1, keepdims=True), 1e-8)
        e_edit = e_edit / np.maximum(np.linalg.norm(e_edit, axis=1, keepdims=True), 1e-8)

        # Delta of unit-sphere embeddings -- norm captured BEFORE second normalization
        delta = e_edit - e_orig  # (B, D)
        delta_norm_vals = np.linalg.norm(delta, axis=1)  # (B,)  ← norm saved here

        # Skip near-zero deltas (effect had no visual impact on CLIP)
        valid_mask = delta_norm_vals > 1e-6
        delta_dir = delta / np.maximum(delta_norm_vals[:, None], 1e-8)  # normalized direction

        # Cosine sims of normalized delta to IMG vocab: (B, V)
        sims = (delta_dir @ img_vocab_embeddings.T)[valid_mask]  # (B_valid, V)
        norms = delta_norm_vals[valid_mask]                       # (B_valid,)

        if len(sims) > 0:
            all_sims.append(sims)
            all_norms.append(norms)

        i += batch_size
        if (i // batch_size) % 10 == 0:
            n_done = sum(len(s) for s in all_sims)
            print(f"  Processed {i}/{len(pairs)} pairs, collected {n_done} valid sims", flush=True)

    if skipped > 0:
        print(f"  Skipped {skipped} pairs (effect errors)")

    raw_sims = np.concatenate(all_sims, axis=0).astype(np.float32)
    delta_norms = np.concatenate(all_norms, axis=0).astype(np.float32)
    print(f"Collected {len(raw_sims)} valid calibration pairs")
    return raw_sims, delta_norms


# ---------------------------------------------------------------------------
# Stage 1: Joint (T*, norm_confidence_threshold, norm_confidence_scale) grid search
# ---------------------------------------------------------------------------

def _apply_confidence_mix(
    peaked: np.ndarray,   # (N, V) softmax distributions
    delta_norms: np.ndarray,  # (N,)
    threshold: float,
    scale: float,
) -> np.ndarray:
    """Mix peaked distribution with uniform prior via sigmoid confidence."""
    V = peaked.shape[1]
    uniform = np.ones(V, dtype=np.float32) / V
    conf = 1.0 / (1.0 + np.exp(-(delta_norms - threshold) / scale))  # (N,)
    return conf[:, None] * peaked + (1.0 - conf[:, None]) * uniform   # (N, V)


def find_optimal_params_joint(
    raw_sims: np.ndarray,      # (N, V)
    delta_norms: np.ndarray,   # (N,)
    target_top1_mean: float,
    T_grid: list = None,
    thr_percentiles: list = None,
    scale_alphas: list = None,
) -> dict:
    """
    Joint grid search over (T, norm_confidence_threshold, norm_confidence_scale).

    style_label = confidence * softmax(sims/T) + (1-confidence) * uniform
    confidence  = sigmoid((delta_norm - threshold) / scale)

    Primary objective:   minimize |E[top1(style_label)] - target_top1_mean|
    Secondary objective: maximize std(confidence)  -> threshold sits at a
                         discriminative point in the norm distribution.

    scale is parameterized as alpha * threshold (proportional), so a bigger
    threshold has a proportionally wider transition band.

    Returns:
        dict with T_star, norm_confidence_threshold, norm_confidence_scale,
        mean_top1, error, conf_std.
    """
    if T_grid is None:
        T_grid = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.20]
    if thr_percentiles is None:
        thr_percentiles = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    if scale_alphas is None:
        scale_alphas = [0.05, 0.10, 0.20, 0.35, 0.60, 1.00, 2.00]

    V = raw_sims.shape[1]

    # Precompute peaked distributions for each T (reused across threshold/scale loops)
    peaked_cache = {}
    for T in T_grid:
        peaked_cache[T] = softmax_probs(raw_sims, T)  # (N, V)

    # Threshold values from norm percentiles
    thr_values = [float(np.percentile(delta_norms, p)) for p in thr_percentiles]

    print(f"\nDelta norm stats:")
    for p in [5, 25, 50, 75, 95]:
        print(f"  p{p:2d}: {np.percentile(delta_norms, p):.4f}")
    print(f"  mean={delta_norms.mean():.4f}, std={delta_norms.std():.4f}")
    print(f"\nTarget top-1 mean: {target_top1_mean:.4f}")
    print(f"Grid: {len(T_grid)} T × {len(thr_values)} threshold × {len(scale_alphas)} scale "
          f"= {len(T_grid)*len(thr_values)*len(scale_alphas)} combinations")

    candidates = []
    for T in T_grid:
        peaked = peaked_cache[T]  # (N, V)
        for thr, thr_pct in zip(thr_values, thr_percentiles):
            for alpha in scale_alphas:
                sc = max(thr * alpha, 1e-8)
                labels = _apply_confidence_mix(peaked, delta_norms, thr, sc)
                top1 = labels.max(axis=1)
                error = abs(float(top1.mean()) - target_top1_mean)
                conf = 1.0 / (1.0 + np.exp(-(delta_norms - thr) / sc))
                conf_std = float(conf.std())
                candidates.append((error, -conf_std, T, thr, sc, thr_pct, alpha, float(top1.mean())))

    # Sort: primary = error, secondary = -conf_std (higher std is better)
    candidates.sort(key=lambda x: (x[0], x[1]))

    print(f"\n{'T':>7} {'thr_pct':>8} {'thr':>8} {'scale_a':>8} {'top1_mean':>10} {'error':>8} {'conf_std':>9}")
    print("-" * 70)
    for row in candidates[:20]:
        err, neg_cs, T, thr, sc, thr_pct, alpha, mean_t1 = row
        mark = " <--" if row is candidates[0] else ""
        print(f"{T:>7.4f} {thr_pct:>8.0f} {thr:>8.4f} {alpha:>8.2f} {mean_t1:>10.4f} {err:>8.4f} {-neg_cs:>9.4f}{mark}")

    best = candidates[0]
    _, _, T_star, thr_star, sc_star, thr_pct_star, alpha_star, mean_t1_star = best
    print(f"\nBest: T*={T_star}, threshold*={thr_star:.4f} (p{thr_pct_star}), "
          f"scale*={sc_star:.4f} (alpha={alpha_star})")

    return {
        'T_star': T_star,
        'norm_confidence_threshold': round(float(thr_star), 6),
        'norm_confidence_scale': round(float(sc_star), 6),
        'norm_confidence_threshold_percentile': thr_pct_star,
        'norm_confidence_scale_alpha': alpha_star,
        'mean_top1': mean_t1_star,
        'error': best[0],
        'conf_std': -best[1],
    }


# ---------------------------------------------------------------------------
# Stage 2: threshold calibration via controller
# ---------------------------------------------------------------------------

def calibrate_threshold(
    raw_sims: np.ndarray,
    delta_norms: np.ndarray,
    T_star: float,
    norm_confidence_threshold: float,
    norm_confidence_scale: float,
    controller,
    clap_embedder,
    audio_paths: list,
    effect_names: list,
    target_mean_n_active: float,
    device: str = "cuda",
    sample_size: int = 1000,
    batch_size: int = 128,
    seed: int = 42,
) -> tuple:
    """
    Run controller on sample of (style_label, clap_emb) pairs.
    Find threshold t* where E[N_active(t*)] ≈ target_mean_n_active.

    Returns:
        (threshold_star, activity_prob_matrix)
    """
    import torch
    import librosa

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # Sample style labels using confidence-adjusted distribution (matches inference path)
    idx = np_rng.choice(len(raw_sims), size=min(sample_size, len(raw_sims)), replace=False)
    peaked = softmax_probs(raw_sims[idx], T_star)                                  # (S, V)
    style_labels = _apply_confidence_mix(                                           # (S, V)
        peaked, delta_norms[idx], norm_confidence_threshold, norm_confidence_scale
    )

    # Sample audio embeddings (random audio files)
    audio_sample_paths = rng.choices(audio_paths, k=min(sample_size, len(audio_paths)))
    print(f"\nLoading {len(audio_sample_paths)} audio files for threshold calibration...")

    clap_embs = []
    for ap in audio_sample_paths:
        try:
            audio, _ = librosa.load(ap, sr=48000, mono=True, duration=10.0)
            at = torch.from_numpy(audio).float().unsqueeze(0).to(device)
            with torch.no_grad():
                emb = clap_embedder.embed_audio(at, 48000)
            # embed_audio returns (1, D) -- store as (D,) so stack gives (N, D)
            clap_embs.append(emb.cpu().numpy().squeeze(0).astype(np.float32))
        except Exception:
            clap_embs.append(None)

    # Filter valid
    valid_pairs = [(sl, ce) for sl, ce in zip(style_labels, clap_embs) if ce is not None]
    if not valid_pairs:
        raise RuntimeError("No valid audio embeddings for threshold calibration.")
    style_labels_v = np.stack([p[0] for p in valid_pairs])  # (N, V)
    clap_embs_v = np.stack([p[1] for p in valid_pairs])     # (N, D)
    N = len(valid_pairs)
    print(f"  Valid pairs for threshold calibration: {N}")

    # Run controller in batches to get activity logits
    controller.eval()
    all_activity_probs = []

    for b_start in range(0, N, batch_size):
        b_end = min(b_start + batch_size, N)
        sl_batch = torch.from_numpy(style_labels_v[b_start:b_end]).float().to(device)
        ce_batch = torch.from_numpy(clap_embs_v[b_start:b_end]).float().to(device)

        with torch.no_grad():
            _, activity_logits = controller.forward_with_activity(ce_batch, sl_batch)

        if activity_logits is not None:
            probs = torch.sigmoid(activity_logits).cpu().numpy()  # (B, n_effects)
            all_activity_probs.append(probs)

    if not all_activity_probs:
        raise RuntimeError("Controller did not return activity logits.")

    activity_probs = np.concatenate(all_activity_probs, axis=0)  # (N, n_effects)
    print(f"  Activity prob matrix: {activity_probs.shape}")
    print(f"  Per-effect mean prob: {activity_probs.mean(axis=0).round(3)}")

    # Grid search threshold
    t_grid = np.arange(0.40, 0.96, 0.01)
    print(f"\n{'Threshold':>10}  {'E[N_active]':>12}  {'error':>8}")
    print("-" * 35)

    best_t = 0.5
    best_err = float('inf')
    results = {}

    for t in t_grid:
        n_active = (activity_probs >= t).sum(axis=1).astype(float)
        mean_n = float(n_active.mean())
        err = abs(mean_n - target_mean_n_active)
        results[float(t)] = {'mean_n_active': mean_n, 'error': err}
        if err < best_err:
            best_err = err
            best_t = float(t)

    # Print bracketing range
    for t in t_grid:
        mean_n = results[float(t)]['mean_n_active']
        err = results[float(t)]['error']
        if abs(mean_n - target_mean_n_active) < 0.1:
            print(f"{t:>10.2f}  {mean_n:>12.4f}  {err:>8.4f}  ←")
        elif 1.0 <= mean_n <= 2.0:
            print(f"{t:>10.2f}  {mean_n:>12.4f}  {err:>8.4f}")

    print(f"\nBest threshold* = {best_t:.2f}  (E[N_active]={results[best_t]['mean_n_active']:.4f}, error={best_err:.4f})")
    return round(best_t, 2), activity_probs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Calibrate inference T and activity threshold")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--n-images", type=int, default=None, help="Max images (null=all)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-sims", type=str, default=None, help="Path to cache raw sims (.npy)")
    parser.add_argument("--load-sims", type=str, default=None, help="Path to load cached raw sims (.npy)")
    parser.add_argument("--skip-threshold", action="store_true", help="Only run T calibration")
    parser.add_argument("--target-top1", type=float, default=0.1412,
                        help="Training style_label top-1 mean to match")
    parser.add_argument("--target-n-active", type=float, default=1.25,
                        help="Training mean N_active to match (0.75*1+0.25*2=1.25)")
    args = parser.parse_args()

    # --- Load config ---
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = config.get("device", "cuda")
    output_dir = Path(config["output"]["dir"])
    effect_names = config["effects"]["active"]
    wand_effects = config.get("image_inverse_mapping", {}).get("effect_types", [
        "adaptive_blur", "motion_blur", "adaptive_sharpen",
        "add_noise", "spread", "sepia_tone", "solarize",
    ])

    # --- Load CLIP ---
    import sys as _sys
    _saved = _sys.argv[:]
    _sys.argv = [_sys.argv[0]]
    try:
        from src.models import CLIPEmbedder, CLAPEmbedder
    finally:
        _sys.argv = _saved

    clip_cfg = config["model"]["clip"]
    print(f"Loading CLIP {clip_cfg['name']}...")
    clip = CLIPEmbedder(model_name=clip_cfg["name"], pretrained=clip_cfg["pretrained"], device=device)

    # --- Load IMG vocab ---
    from src.vocab.style_vocab import StyleVocabulary
    vocab = StyleVocabulary()
    vocab.load(str(output_dir))
    img_vocab = vocab.img_vocab.embeddings  # (V, D)
    print(f"IMG vocab: {img_vocab.shape}")

    # --- Collect image paths ---
    image_dir = Path(config["data"]["image_dir"])
    image_paths = sorted(image_dir.rglob("*.jpg")) + \
                  sorted(image_dir.rglob("*.jpeg")) + \
                  sorted(image_dir.rglob("*.png"))
    image_paths = [str(p) for p in image_paths]
    if args.n_images:
        image_paths = image_paths[:args.n_images]
    print(f"Images: {len(image_paths)}")

    # -----------------------------------------------------------------------
    # Stage 1: raw sims + delta norms
    # -----------------------------------------------------------------------
    # Auto-save paths in output_dir (always written after collection)
    sims_cache = output_dir / "calib_raw_sims.npy"
    norms_cache = output_dir / "calib_raw_norms.npy"

    load_path = Path(args.load_sims) if args.load_sims else sims_cache
    load_norms_path = load_path.parent / (load_path.stem + "_norms.npy") \
        if args.load_sims else norms_cache

    if load_path.exists() and load_norms_path.exists():
        print(f"\nLoading cached raw sims from {load_path}...")
        raw_sims = np.load(str(load_path)).astype(np.float32)
        delta_norms = np.load(str(load_norms_path)).astype(np.float32)
        print(f"Loaded: sims={raw_sims.shape}, norms={delta_norms.shape}")
    else:
        if load_path.exists() and not load_norms_path.exists():
            print("Warning: sims cache found but norms cache missing -- re-running Stage 1a.")
        print("\n" + "=" * 60)
        print("STAGE 1a: Collecting raw CLIP delta cosine sims + norms")
        print("=" * 60)
        raw_sims, delta_norms = collect_raw_sims(
            image_paths=image_paths,
            img_vocab_embeddings=img_vocab,
            clip_embedder=clip,
            effect_names=wand_effects,
            intensity_range=(0.3, 1.0),
            seed=args.seed,
            batch_size=args.batch_size,
            device=device,
        )
        np.save(str(sims_cache), raw_sims)
        np.save(str(norms_cache), delta_norms)
        print(f"Cached sims -> {sims_cache}")
        print(f"Cached norms -> {norms_cache}")
        if args.save_sims:
            np.save(args.save_sims, raw_sims)
            print(f"Also saved sims to {args.save_sims}")

    # Raw sim / norm stats
    per_sample_max = raw_sims.max(axis=1)
    print(f"\nRaw sim stats (T-independent):")
    print(f"  per-sample max sim: mean={per_sample_max.mean():.4f}, std={per_sample_max.std():.4f}")
    print(f"  per-sample max sim: min={per_sample_max.min():.4f}, max={per_sample_max.max():.4f}")
    print(f"  per-sample spread (std across vocab): mean={raw_sims.std(axis=1).mean():.6f}")
    print(f"  delta norm: mean={delta_norms.mean():.4f}, std={delta_norms.std():.4f}, "
          f"min={delta_norms.min():.4f}, max={delta_norms.max():.4f}")

    # Stage 1b: Joint (T*, norm_confidence_threshold, norm_confidence_scale) grid search
    print("\n" + "=" * 60)
    print("STAGE 1b: Joint grid search (T, norm_confidence_threshold, norm_confidence_scale)")
    print("=" * 60)
    joint_result = find_optimal_params_joint(
        raw_sims=raw_sims,
        delta_norms=delta_norms,
        target_top1_mean=args.target_top1,
    )
    T_star = joint_result['T_star']
    norm_conf_threshold = joint_result['norm_confidence_threshold']
    norm_conf_scale = joint_result['norm_confidence_scale']

    if args.skip_threshold:
        print(f"\nSkipping activity threshold calibration (--skip-threshold).")
        print(f"T*={T_star}, norm_confidence_threshold={norm_conf_threshold}, "
              f"norm_confidence_scale={norm_conf_scale}")
        return

    # -----------------------------------------------------------------------
    # Stage 2: threshold calibration
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 2: Activity threshold calibration")
    print("=" * 60)

    # Load CLAP + controller
    clap_cfg = config["model"]["clap"]
    print(f"Loading CLAP...")
    clap = CLAPEmbedder(
        model_id=clap_cfg["model_id"],
        enable_fusion=clap_cfg["enable_fusion"],
        max_duration=clap_cfg["max_duration"],
        device=device,
    )

    from src.inference import DeltaV2APipeline
    pipeline = DeltaV2APipeline.load(
        artifacts_dir=str(output_dir),
        clip_embedder=clip,
        clap_embedder=clap,
        device=device,
        use_siamese_visual_encoder=False,
        style_temperature=T_star,
        activity_threshold_override=None,  # Use original thresholds for analysis
    )
    controller = pipeline.controller
    print(f"  Original calibrated thresholds: {pipeline.activity_thresholds}")

    # Audio paths for calibration
    audio_dir = Path(config["data"]["audio_dir"])
    audio_paths = sorted(audio_dir.rglob("*.mp3")) + \
                  sorted(audio_dir.rglob("*.wav")) + \
                  sorted(audio_dir.rglob("*.flac"))
    audio_paths = [str(p) for p in audio_paths]
    print(f"Audio files for threshold calibration: {len(audio_paths)}")

    threshold_star, activity_probs = calibrate_threshold(
        raw_sims=raw_sims,
        delta_norms=delta_norms,
        T_star=T_star,
        norm_confidence_threshold=norm_conf_threshold,
        norm_confidence_scale=norm_conf_scale,
        controller=controller,
        clap_embedder=clap,
        audio_paths=audio_paths,
        effect_names=effect_names,
        target_mean_n_active=args.target_n_active,
        device=device,
        sample_size=1000,
        batch_size=128,
        seed=args.seed,
    )

    # -----------------------------------------------------------------------
    # Save results and update config
    # -----------------------------------------------------------------------
    calib_result = {
        "T_star": T_star,
        "norm_confidence_threshold": norm_conf_threshold,
        "norm_confidence_scale": norm_conf_scale,
        "norm_confidence_threshold_percentile": joint_result['norm_confidence_threshold_percentile'],
        "norm_confidence_scale_alpha": joint_result['norm_confidence_scale_alpha'],
        "threshold_star": threshold_star,
        "target_top1_mean": args.target_top1,
        "target_n_active": args.target_n_active,
        "achieved_top1_mean": joint_result['mean_top1'],
        "achieved_n_active": float((activity_probs >= threshold_star).sum(axis=1).mean()),
        "n_calibration_pairs": len(raw_sims),
        "joint_grid_best": joint_result,
    }
    calib_path = output_dir / "inference_calibration.json"
    with open(calib_path, "w") as f:
        json.dump(calib_result, f, indent=2)
    print(f"\nCalibration results saved to {calib_path}")

    # Update pipeline.yaml
    print(f"\nUpdating {args.config}...")
    with open(args.config, "r", encoding="utf-8") as f:
        cfg_text = f.read()

    import re
    cfg_text = re.sub(r"(style_temperature:\s*)[\d.]+", f"\\g<1>{T_star}", cfg_text)
    cfg_text = re.sub(r"(activity_threshold_override:\s*)[\d.]+", f"\\g<1>{threshold_star}", cfg_text)
    # norm_confidence_threshold / scale: may be null initially, update to float value
    cfg_text = re.sub(
        r"(norm_confidence_threshold:\s*)(?:null|[\d.]+)",
        f"\\g<1>{norm_conf_threshold}",
        cfg_text,
    )
    cfg_text = re.sub(
        r"(norm_confidence_scale:\s*)(?:null|[\d.]+)",
        f"\\g<1>{norm_conf_scale}",
        cfg_text,
    )
    with open(args.config, "w", encoding="utf-8") as f:
        f.write(cfg_text)

    print(f"\n{'='*60}")
    print("CALIBRATION COMPLETE")
    print(f"{'='*60}")
    print(f"  T*                        = {T_star}")
    print(f"  norm_confidence_threshold = {norm_conf_threshold:.4f}  "
          f"(p{joint_result['norm_confidence_threshold_percentile']} of delta_norms)")
    print(f"  norm_confidence_scale     = {norm_conf_scale:.4f}  "
          f"(alpha={joint_result['norm_confidence_scale_alpha']})")
    print(f"  activity_threshold*       = {threshold_star}")
    print(f"  E[top1]    = {calib_result['achieved_top1_mean']:.4f}  (target {args.target_top1:.4f})")
    print(f"  E[N_active]= {calib_result['achieved_n_active']:.4f}  (target {args.target_n_active:.4f})")
    print(f"\npipeline.yaml updated.")


if __name__ == "__main__":
    main()
