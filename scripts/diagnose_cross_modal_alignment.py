#!/usr/bin/env python
"""
Cross-Modal Style Alignment Diagnostic

Audio side:  pedalboard effects → CLAP delta → AUD_VOCAB projection (from existing HDF5 DB)
Image side:  wand effects → CLIP delta → IMG_VOCAB projection (computed on-the-fly)

핵심 질문:
  1. 각 이펙트가 어떤 vocab term으로 투영되는가?
  2. 의미적으로 유사한 이미지↔오디오 이펙트 쌍이 같은 vocab term을 가리키는가?
  3. IMG_VOCAB vs AUD_VOCAB의 cross-modal alignment가 얼마나 강한가?

Usage:
  python scripts/diagnose_cross_modal_alignment.py
  python scripts/diagnose_cross_modal_alignment.py \\
      --h5 outputs/attempts/attempt11_.../run/inverse_mapping.h5 \\
      --img-vocab outputs/attempts/attempt11_.../run/img_vocab.npz \\
      --aud-vocab outputs/attempts/attempt11_.../run/aud_vocab.npz \\
      --image-dir data/original/images \\
      --n-images 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)  # flush every line
sys.path.append(str(Path(__file__).parent.parent))

WAND_EFFECTS = [
    "adaptive_blur",
    "motion_blur",
    "adaptive_sharpen",
    "add_noise",
    "spread",
    "sepia_tone",
    "solarize",
]

PEDALBOARD_EFFECTS = [
    "lowpass",
    "bitcrush",
    "reverb",
    "highpass",
    "distortion",
    "playback_rate",
    "delay",
]

# 의미적으로 연결될 것으로 예상되는 페어 (image effect → audio effect)
EXPECTED_PAIRS = [
    ("adaptive_blur",    "reverb"),       # blurry ↔ spacious/reverberant
    ("adaptive_blur",    "delay"),        # blur ↔ echo
    ("adaptive_sharpen", "highpass"),     # sharp ↔ bright/crisp
    ("adaptive_sharpen", "distortion"),   # sharpen ↔ distortion (edge emphasis)
    ("add_noise",        "bitcrush"),     # noise ↔ bitcrush/digital noise
    ("add_noise",        "distortion"),   # noise ↔ distortion
    ("sepia_tone",       "lowpass"),      # vintage/warm ↔ lowpass/muffled
    ("spread",           "reverb"),       # spread/scatter ↔ reverb
    ("spread",           "delay"),        # scatter ↔ delay
    ("solarize",         "bitcrush"),     # surreal/inverted ↔ bitcrush
    ("motion_blur",      "playback_rate"),# motion ↔ pitch shift
]


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose cross-modal style alignment.")
    p.add_argument("--h5",
        default="outputs/attempts/attempt11_20260224_130839/run/inverse_mapping.h5")
    p.add_argument("--img-vocab",
        default="outputs/attempts/attempt11_20260224_130839/run/img_vocab.npz")
    p.add_argument("--aud-vocab",
        default="outputs/attempts/attempt11_20260224_130839/run/aud_vocab.npz")
    p.add_argument("--image-dir", default="data/original/images")
    p.add_argument("--out-dir", default="outputs/diagnostics/cross_modal_alignment")
    p.add_argument("--n-images", type=int, default=30,
        help="Number of test images for image-side computation")
    p.add_argument("--intensities", nargs="+", type=float,
        default=[0.2, 0.4, 0.6, 0.8, 1.0],
        help="Wand effect intensities to test")
    p.add_argument("--style-temperature", type=float, default=0.015,
        help="Softmax temperature for img_style_scores (matches pipeline.yaml)")
    p.add_argument("--norm-confidence-threshold", type=float, default=0.529277)
    p.add_argument("--norm-confidence-scale", type=float, default=0.529277)
    p.add_argument("--device", default="cpu")
    p.add_argument("--top-k", type=int, default=5,
        help="Top vocab terms to display per effect")
    return p.parse_args()


# ── Audio side: load from HDF5 ───────────────────────────────────────────────

def load_audio_profiles(h5_path: str, aud_vocab_path: str) -> tuple[dict, list[str], list[str]]:
    """
    Read existing inverse mapping DB and compute per-effect style profiles.
    Only uses single-effect records (exactly one effect active) for clean signal.
    """
    import h5py

    print(f"\n[Audio] Loading DB: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        n = int(f.attrs.get("actual_records", f["style_labels"].shape[0]))
        style_labels = f["style_labels"][:n].astype(np.float32)
        effect_mask  = f["effect_active_mask"][:n].astype(np.float32)
        effect_names_str = str(f.attrs.get("effect_names", ""))
        effect_names = [x for x in effect_names_str.split(",") if x]

    d = np.load(aud_vocab_path, allow_pickle=True)
    keywords = d["keywords"].tolist() if "keywords" in d else d["terms"].tolist()

    print(f"  Records: {n}, Effects: {effect_names}")
    print(f"  Vocab: {keywords}")

    profiles: dict[str, np.ndarray] = {}
    for i, ename in enumerate(effect_names):
        # Single-effect records only: this effect active, all others inactive
        single_effect = (effect_mask[:, i] > 0.5) & (effect_mask.sum(axis=1) == 1)
        n_single = int(single_effect.sum())
        if n_single < 5:
            print(f"  WARNING: {ename} has only {n_single} single-effect records")
            # Fall back to all records where this effect is active
            single_effect = effect_mask[:, i] > 0.5
            n_single = int(single_effect.sum())
        avg = style_labels[single_effect].mean(axis=0)
        profiles[ename] = avg
        print(f"  {ename}: n={n_single}")

    return profiles, effect_names, keywords


# ── Image side: on-the-fly computation ───────────────────────────────────────

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    logits = logits / temperature
    logits = logits - logits.max()
    exp_l = np.exp(logits)
    return exp_l / exp_l.sum()


def compute_img_style_score(
    clip_orig: np.ndarray,
    clip_edit: np.ndarray,
    img_vocab_emb: np.ndarray,
    temperature: float,
    conf_threshold: float,
    conf_scale: float,
) -> np.ndarray:
    """
    Exact inference pipeline logic from pipeline.py:
      1. CLIP delta (both inputs are L2-normalized)
      2. Normalize delta
      3. softmax(IMG_VOCAB @ norm_delta / T)
      4. Confidence mixing based on delta norm
    """
    delta = clip_edit - clip_orig
    delta_norm = float(np.linalg.norm(delta))

    eps = 1e-8
    delta_normalized = delta / (delta_norm + eps)

    sims = img_vocab_emb @ delta_normalized
    peaked = _softmax(sims, temperature)

    # Confidence mixing: low delta norm → mix toward uniform
    confidence = _sigmoid((delta_norm - conf_threshold) / conf_scale)
    vocab_size = img_vocab_emb.shape[0]
    uniform = np.full(vocab_size, 1.0 / vocab_size, dtype=np.float32)
    style_score = confidence * peaked + (1.0 - confidence) * uniform

    return style_score.astype(np.float32)


def collect_image_paths(image_dir: str, n_images: int) -> list[str]:
    """Collect image paths from directory (recursive)."""
    root = Path(image_dir)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = [str(p) for p in sorted(root.rglob("*")) if p.suffix.lower() in exts]
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    if len(paths) > n_images:
        # Evenly sample across the sorted list for diversity
        indices = np.linspace(0, len(paths) - 1, n_images, dtype=int)
        paths = [paths[i] for i in indices]
    print(f"  Using {len(paths)} images from {image_dir}")
    return paths


def _embed_pil(clip_model, pil_img, device: str) -> np.ndarray:
    """Embed a single PIL image using CLIP preprocess pipeline."""
    import torch
    tensor = clip_model.preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.model.encode_image(tensor)[0].cpu().numpy()
    return emb.astype(np.float32)


def load_image_profiles(
    image_dir: str,
    n_images: int,
    img_vocab_path: str,
    intensities: list[float],
    temperature: float,
    conf_threshold: float,
    conf_scale: float,
    device: str,
) -> tuple[dict, list[str]]:
    """
    For each wand effect × intensity level, compute avg img_style_scores.
    Final profile = average over all images and intensity levels.
    """
    from src.models.clip_embedder import CLIPEmbedder
    from src.effects.wand_image_effects import apply_effect
    from PIL import Image

    print(f"\n[Image] Loading CLIP + IMG_VOCAB...")
    clip = CLIPEmbedder(device=device)
    clip.eval()

    d = np.load(img_vocab_path, allow_pickle=True)
    keywords = d["keywords"].tolist() if "keywords" in d else d["terms"].tolist()
    img_vocab_emb = d["embeddings"].astype(np.float32)
    # Ensure L2 normalized
    norms = np.linalg.norm(img_vocab_emb, axis=1, keepdims=True)
    img_vocab_emb = img_vocab_emb / np.maximum(norms, 1e-8)

    print(f"  IMG_VOCAB: {len(keywords)} terms, dim={img_vocab_emb.shape[1]}")

    image_paths = collect_image_paths(image_dir, n_images)

    # Pre-compute CLIP embeddings for original images
    print(f"  Embedding {len(image_paths)} original images...")
    orig_pils: list = []
    clip_origs: list[np.ndarray] = []

    for path in image_paths:
        try:
            pil = Image.open(path).convert("RGB")
            orig_pils.append(pil)
            emb = _embed_pil(clip, pil, device)
            norm = np.linalg.norm(emb)
            clip_origs.append(emb / max(norm, 1e-8))
        except Exception as e:
            print(f"  Warning: skipping {path}: {e}")

    if not clip_origs:
        raise RuntimeError("No valid images could be loaded")

    print(f"  Successfully embedded {len(clip_origs)} images")

    # Per-effect accumulation
    profiles: dict[str, np.ndarray] = {}
    delta_norms_per_effect: dict[str, list[float]] = {}

    for eff in WAND_EFFECTS:
        print(f"  Processing wand effect: {eff} ...")
        scores_all: list[np.ndarray] = []
        delta_norms: list[float] = []

        for img_idx, (pil_orig, clip_orig) in enumerate(zip(orig_pils, clip_origs)):
            for intensity in intensities:
                try:
                    pil_edit = apply_effect(pil_orig, eff, intensity)
                    emb_edit = _embed_pil(clip, pil_edit, device)
                    norm_edit = np.linalg.norm(emb_edit)
                    clip_edit = emb_edit / max(norm_edit, 1e-8)

                    delta_norm = float(np.linalg.norm(clip_edit - clip_orig))
                    delta_norms.append(delta_norm)

                    score = compute_img_style_score(
                        clip_orig, clip_edit, img_vocab_emb,
                        temperature, conf_threshold, conf_scale,
                    )
                    scores_all.append(score)
                except Exception as e:
                    print(f"    Warning: {eff} intensity={intensity} img={img_idx}: {e}")

        if scores_all:
            profiles[eff] = np.stack(scores_all).mean(axis=0)
            delta_norms_per_effect[eff] = delta_norms
            mean_dn = np.mean(delta_norms)
            print(f"    n_samples={len(scores_all)}, mean_delta_norm={mean_dn:.4f}")
        else:
            profiles[eff] = np.full(len(keywords), 1.0 / len(keywords), dtype=np.float32)
            print(f"    WARNING: no valid samples for {eff}")

    # Print per-effect delta norm stats
    print("\n  Delta norm summary (higher = stronger CLIP response to effect):")
    for eff in WAND_EFFECTS:
        if eff in delta_norms_per_effect and delta_norms_per_effect[eff]:
            dn = np.array(delta_norms_per_effect[eff])
            print(f"    {eff:20s}: mean={dn.mean():.4f}, std={dn.std():.4f}, "
                  f"min={dn.min():.4f}, max={dn.max():.4f}")

    return profiles, keywords


# ── Analysis ──────────────────────────────────────────────────────────────────

def top_k_terms(profile: np.ndarray, keywords: list[str], k: int) -> list[tuple[str, float]]:
    idx = np.argsort(profile)[-k:][::-1]
    return [(keywords[i], float(profile[i])) for i in idx]


def profile_cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float((a / na) @ (b / nb))


def compute_cross_modal_matrix(
    img_profiles: dict[str, np.ndarray],
    aud_profiles: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Compute cosine similarity between each (wand_effect, pedalboard_effect) pair
    using their avg style profiles in the shared vocab space.
    """
    wand_effs = list(img_profiles.keys())
    ped_effs  = list(aud_profiles.keys())
    mat = np.zeros((len(wand_effs), len(ped_effs)), dtype=np.float32)
    for i, we in enumerate(wand_effs):
        for j, pe in enumerate(ped_effs):
            mat[i, j] = profile_cosine_sim(img_profiles[we], aud_profiles[pe])
    return mat, wand_effs, ped_effs


def print_comparison_table(
    img_profiles: dict,
    aud_profiles: dict,
    img_keywords: list[str],
    aud_keywords: list[str],
    top_k: int,
):
    print("\n" + "=" * 80)
    print("IMAGE EFFECTS → IMG_VOCAB top terms")
    print("=" * 80)
    for eff, profile in img_profiles.items():
        terms = top_k_terms(profile, img_keywords, top_k)
        terms_str = ", ".join(f"{t}({v:.3f})" for t, v in terms)
        print(f"  {eff:20s}: {terms_str}")

    print("\n" + "=" * 80)
    print("AUDIO EFFECTS → AUD_VOCAB top terms")
    print("=" * 80)
    for eff, profile in aud_profiles.items():
        terms = top_k_terms(profile, aud_keywords, top_k)
        terms_str = ", ".join(f"{t}({v:.3f})" for t, v in terms)
        print(f"  {eff:20s}: {terms_str}")

    print("\n" + "=" * 80)
    print("CROSS-MODAL SIMILARITY MATRIX (wand × pedalboard)")
    print("(higher = both effects map to similar vocab distributions)")
    print("=" * 80)
    mat, wand_effs, ped_effs = compute_cross_modal_matrix(img_profiles, aud_profiles)

    # Header
    col_w = 14
    header = " " * 22 + "".join(f"{e[:col_w]:>{col_w}}" for e in ped_effs)
    print(header)
    for i, we in enumerate(wand_effs):
        row_vals = "".join(f"{mat[i, j]:>{col_w}.3f}" for j in range(len(ped_effs)))
        print(f"  {we:20s}{row_vals}")

    print("\n" + "=" * 80)
    print("EXPECTED PAIRS: how well do semantically related effects align?")
    print("=" * 80)
    for img_eff, aud_eff in EXPECTED_PAIRS:
        if img_eff in img_profiles and aud_eff in aud_profiles:
            sim = profile_cosine_sim(img_profiles[img_eff], aud_profiles[aud_eff])
            print(f"  {img_eff:20s} ↔ {aud_eff:15s}: cosine = {sim:.4f}")

    # Best pairs
    print("\n  Best actual pairs (top 5 by cosine sim):")
    pairs = []
    for i, we in enumerate(wand_effs):
        for j, pe in enumerate(ped_effs):
            pairs.append((we, pe, float(mat[i, j])))
    pairs.sort(key=lambda x: -x[2])
    for we, pe, sim in pairs[:5]:
        print(f"  {we:20s} ↔ {pe:15s}: {sim:.4f}")

    return mat, wand_effs, ped_effs


# ── Visualization ─────────────────────────────────────────────────────────────

def save_plots(
    out_dir: Path,
    img_profiles: dict,
    aud_profiles: dict,
    img_keywords: list[str],
    aud_keywords: list[str],
    cross_mat: np.ndarray,
    wand_effs: list[str],
    ped_effs: list[str],
    top_k: int,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Plot 1: Image effect heatmap ──
    img_mat = np.stack([img_profiles[e] for e in WAND_EFFECTS if e in img_profiles])
    img_eff_names = [e for e in WAND_EFFECTS if e in img_profiles]

    fig, ax = plt.subplots(figsize=(max(12, len(img_keywords) * 0.6), max(4, len(img_eff_names) * 0.7)))
    im = ax.imshow(img_mat, aspect="auto", cmap="Blues", vmin=0)
    ax.set_xticks(range(len(img_keywords)))
    ax.set_xticklabels(img_keywords, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(img_eff_names)))
    ax.set_yticklabels(img_eff_names, fontsize=9)
    ax.set_title("Image Effects → IMG_VOCAB Style Scores (avg over test images)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "1_image_effect_vocab_heatmap.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: Audio effect heatmap ──
    aud_mat = np.stack([aud_profiles[e] for e in PEDALBOARD_EFFECTS if e in aud_profiles])
    aud_eff_names = [e for e in PEDALBOARD_EFFECTS if e in aud_profiles]

    fig, ax = plt.subplots(figsize=(max(12, len(aud_keywords) * 0.6), max(4, len(aud_eff_names) * 0.7)))
    im = ax.imshow(aud_mat, aspect="auto", cmap="Oranges", vmin=0)
    ax.set_xticks(range(len(aud_keywords)))
    ax.set_xticklabels(aud_keywords, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(aud_eff_names)))
    ax.set_yticklabels(aud_eff_names, fontsize=9)
    ax.set_title("Audio Effects → AUD_VOCAB Style Scores (avg over DB records)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "2_audio_effect_vocab_heatmap.png", dpi=150)
    plt.close(fig)

    # ── Plot 3: Cross-modal similarity matrix ──
    fig, ax = plt.subplots(figsize=(max(8, len(ped_effs) * 1.2), max(6, len(wand_effs) * 0.9)))
    im = ax.imshow(cross_mat, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(ped_effs)))
    ax.set_xticklabels(ped_effs, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(wand_effs)))
    ax.set_yticklabels(wand_effs, fontsize=9)
    ax.set_title("Cross-Modal Profile Similarity\n(cosine sim between avg style profiles)")
    for i in range(len(wand_effs)):
        for j in range(len(ped_effs)):
            ax.text(j, i, f"{cross_mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "3_cross_modal_similarity.png", dpi=150)
    plt.close(fig)

    # ── Plot 4: Top-K bar charts side by side ──
    n_wand = len(WAND_EFFECTS)
    n_ped  = len(PEDALBOARD_EFFECTS)
    fig, axes = plt.subplots(max(n_wand, n_ped), 2,
                             figsize=(18, 2.2 * max(n_wand, n_ped)))
    if max(n_wand, n_ped) == 1:
        axes = axes[np.newaxis, :]

    for row, (weff, peff) in enumerate(zip(
        WAND_EFFECTS + [""] * (n_ped - n_wand),
        PEDALBOARD_EFFECTS + [""] * (n_wand - n_ped),
    )):
        # Left: image effect
        ax_l = axes[row, 0]
        if weff and weff in img_profiles:
            profile = img_profiles[weff]
            idx = np.argsort(profile)[-top_k:][::-1]
            vals = profile[idx]
            kws  = [img_keywords[i] for i in idx]
            ax_l.barh(range(len(kws)), vals[::-1], color="#4472C4", alpha=0.8)
            ax_l.set_yticks(range(len(kws)))
            ax_l.set_yticklabels(kws[::-1], fontsize=8)
            ax_l.set_title(f"[IMG] {weff}", fontsize=9)
            ax_l.set_xlim(0, max(vals) * 1.3 if vals.max() > 0 else 0.1)
        else:
            ax_l.axis("off")

        # Right: audio effect
        ax_r = axes[row, 1]
        if peff and peff in aud_profiles:
            profile = aud_profiles[peff]
            idx = np.argsort(profile)[-top_k:][::-1]
            vals = profile[idx]
            kws  = [aud_keywords[i] for i in idx]
            ax_r.barh(range(len(kws)), vals[::-1], color="#ED7D31", alpha=0.8)
            ax_r.set_yticks(range(len(kws)))
            ax_r.set_yticklabels(kws[::-1], fontsize=8)
            ax_r.set_title(f"[AUD] {peff}", fontsize=9)
            ax_r.set_xlim(0, max(vals) * 1.3 if vals.max() > 0 else 0.1)
        else:
            ax_r.axis("off")

    fig.suptitle(f"Top-{top_k} Vocab Terms per Effect (left=image, right=audio)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "4_per_effect_top_terms.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved 4 plots to {out_dir}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Audio side ──
    aud_profiles, aud_effect_names, aud_keywords = load_audio_profiles(
        args.h5, args.aud_vocab,
    )

    # ── Image side ──
    img_profiles, img_keywords = load_image_profiles(
        image_dir=args.image_dir,
        n_images=args.n_images,
        img_vocab_path=args.img_vocab,
        intensities=args.intensities,
        temperature=args.style_temperature,
        conf_threshold=args.norm_confidence_threshold,
        conf_scale=args.norm_confidence_scale,
        device=args.device,
    )

    # ── Analysis ──
    cross_mat, wand_effs, ped_effs = print_comparison_table(
        img_profiles, aud_profiles, img_keywords, aud_keywords, args.top_k,
    )

    # ── Verdict ──
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    # 1. Are image effects discriminative? (pairwise similarity between img profiles)
    img_eff_list = [e for e in WAND_EFFECTS if e in img_profiles]
    if len(img_eff_list) >= 2:
        img_sims = []
        for i in range(len(img_eff_list)):
            for j in range(i + 1, len(img_eff_list)):
                img_sims.append(profile_cosine_sim(
                    img_profiles[img_eff_list[i]], img_profiles[img_eff_list[j]]
                ))
        mean_img_sim = float(np.mean(img_sims))
        print(f"\n1. Image effect discriminativeness:")
        print(f"   Mean pairwise cosine sim between img profiles: {mean_img_sim:.4f}")
        if mean_img_sim > 0.9:
            print("   CRITICAL: Image effects are nearly indistinguishable in vocab space.")
            print("   → Different wand effects produce almost identical style distributions.")
        elif mean_img_sim > 0.7:
            print("   WARNING: Image effects have weak discrimination.")
        else:
            print("   OK: Image effects are reasonably discriminative.")

    # 2. Are audio effects discriminative?
    aud_eff_list = [e for e in PEDALBOARD_EFFECTS if e in aud_profiles]
    if len(aud_eff_list) >= 2:
        aud_sims = []
        for i in range(len(aud_eff_list)):
            for j in range(i + 1, len(aud_eff_list)):
                aud_sims.append(profile_cosine_sim(
                    aud_profiles[aud_eff_list[i]], aud_profiles[aud_eff_list[j]]
                ))
        mean_aud_sim = float(np.mean(aud_sims))
        print(f"\n2. Audio effect discriminativeness:")
        print(f"   Mean pairwise cosine sim between aud profiles: {mean_aud_sim:.4f}")
        if mean_aud_sim > 0.9:
            print("   CRITICAL: Audio effects are nearly indistinguishable in vocab space.")
        elif mean_aud_sim > 0.7:
            print("   WARNING: Audio effects have weak discrimination.")
        else:
            print("   OK: Audio effects are reasonably discriminative.")

    # 3. Cross-modal alignment for expected pairs
    print(f"\n3. Cross-modal alignment for expected pairs:")
    pair_sims = []
    for img_eff, aud_eff in EXPECTED_PAIRS:
        if img_eff in img_profiles and aud_eff in aud_profiles:
            sim = profile_cosine_sim(img_profiles[img_eff], aud_profiles[aud_eff])
            pair_sims.append(sim)
    if pair_sims:
        mean_pair_sim = float(np.mean(pair_sims))
        print(f"   Mean cosine sim for expected pairs: {mean_pair_sim:.4f}")
        if mean_pair_sim > 0.7:
            print("   GOOD: Cross-modal alignment is strong for expected pairs.")
        elif mean_pair_sim > 0.5:
            print("   FAIR: Cross-modal alignment is moderate. Identity mapping is imperfect but usable.")
        else:
            print("   POOR: Cross-modal alignment is weak.")
            print("   → Identity mapping (img_style → aud_style) is unlikely to work well.")
            print("   → Consider: learned cross-modal bridge, or redesign vocab for better CLIP-CLAP alignment.")

    # 4. Dominant term check
    print(f"\n4. Dominant vocab term check:")
    all_img = np.stack(list(img_profiles.values()))  # (n_effects, vocab)
    per_term_mean = all_img.mean(axis=0)
    top3_idx = np.argsort(per_term_mean)[-3:][::-1]
    print(f"   Top terms by mean score across ALL image effects:")
    for idx in top3_idx:
        print(f"   → '{img_keywords[idx]}': mean={per_term_mean[idx]:.4f}")
    if per_term_mean[top3_idx[0]] > 0.15:
        print("   WARNING: One or few terms dominate across all effects — vocab has structural bias.")

    print("\n" + "=" * 80)

    # ── Save JSON report ──
    report = {
        "args": vars(args),
        "img_effect_profiles": {
            eff: {kw: float(v) for kw, v in zip(img_keywords, prof)}
            for eff, prof in img_profiles.items()
        },
        "aud_effect_profiles": {
            eff: {kw: float(v) for kw, v in zip(aud_keywords, prof)}
            for eff, prof in aud_profiles.items()
        },
        "cross_modal_matrix": {
            wand_effs[i]: {ped_effs[j]: float(cross_mat[i, j]) for j in range(len(ped_effs))}
            for i in range(len(wand_effs))
        },
        "expected_pair_sims": {
            f"{ie}↔{ae}": float(profile_cosine_sim(img_profiles[ie], aud_profiles[ae]))
            for ie, ae in EXPECTED_PAIRS
            if ie in img_profiles and ae in aud_profiles
        },
        "img_discriminativeness_mean_cosine": float(np.mean(img_sims)) if img_eff_list else None,
        "aud_discriminativeness_mean_cosine": float(np.mean(aud_sims)) if aud_eff_list else None,
    }
    report_path = out_dir / "cross_modal_alignment_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")

    # ── Plots ──
    print("Generating plots...")
    save_plots(
        out_dir=out_dir,
        img_profiles=img_profiles,
        aud_profiles=aud_profiles,
        img_keywords=img_keywords,
        aud_keywords=aud_keywords,
        cross_mat=cross_mat,
        wand_effs=wand_effs,
        ped_effs=ped_effs,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
