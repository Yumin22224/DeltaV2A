#!/usr/bin/env python
"""
Quick image-side discriminativeness test for candidate style vocabularies.

Builds CLIP vocab embeddings on-the-fly with candidate vocab + templates,
then measures how well 7 wand effects are separated in vocab space.

Metric: mean pairwise cosine similarity between per-effect style profiles.
  Lower = effects are more distinguishable = BETTER.

Usage:
  python scripts/test_vocab_discriminativeness.py --device cuda
  python scripts/test_vocab_discriminativeness.py --device cuda --n-images 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.append(str(Path(__file__).parent.parent))

# ─────────────────────────────────────────────────────────────────────────────
# CANDIDATE VOCAB DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

# Attempt 13 candidate (Gemini-inspired, reviewed)
VOCAB_CANDIDATE = {
    "name": "attempt13_candidate",
    "axes": [
        # Phase 1: Texture & Surface
        ("granular",   "polished"),    # micro-texture: grain vs smooth
        ("piercing",   "muffled"),     # spectral: high-freq vs low-freq emphasis
        ("clipped",    "pristine"),    # amplitude: signal overload vs original quality
        ("fluid",      "rigid"),       # modulation: LFO flowing vs stiff
        # Phase 2: Space & Dimensionality
        ("expansive",  "confined"),    # spatial scale: reverberant vs dry/small
        ("layered",    "flat"),        # depth: multi-voice vs single-plane
        ("dense",      "hollow"),      # density: full-spectrum vs sparse/empty
        ("fragmented", "cohesive"),    # continuity: broken vs seamless
        # Phase 3: Energy & Form
        ("agitated",   "calm"),        # energy/modulation speed
        ("warped",     "linear"),      # signal distortion: bent vs straight
        ("crisp",      "muddy"),       # clarity: defined vs blurred/muddy
        ("deep",       "thin"),        # weight: bass-heavy vs high-pass/thin
    ],
    "img_templates": [
        "a {word} image",
        "a {word} visual texture",
        "an image with a {word} quality",
        "something that looks {word}",
        "an image that appears more {word}",
        "a {word} visual effect on an image",
        "image processing with a {word} result",
        "the {word} character of a processed image",
    ],
}

# Attempt 12 vocab (for comparison baseline)
VOCAB_ATTEMPT12 = {
    "name": "attempt12_baseline",
    "axes": [
        ("warm",        "cold"),
        ("tense",       "relaxed"),
        ("rough",       "smooth"),
        ("heavy",       "light"),
        ("dense",       "sparse"),
        ("distant",     "intimate"),
        ("aggressive",  "gentle"),
        ("static",      "dynamic"),
        ("clean",       "dirty"),
        ("meditative",  "urgent"),
        ("archaic",     "contemporary"),
        ("organic",     "synthetic"),
    ],
    "img_templates": [
        "a {word} image",
        "a {word} photograph",
        "an image with a {word} mood",
        "a {word} visual style",
        "a {word} scene",
        "a {word} picture",
        "a photo that feels {word}",
        "artwork with a {word} atmosphere",
    ],
}

# Attempt 11 vocab (original, for reference)
# attempt13 vocab with STANDARD templates (to isolate template effect)
VOCAB_CANDIDATE_STD_TEMPLATES = {
    "name": "attempt13_std_templates",
    "axes": [
        ("granular",   "polished"),
        ("piercing",   "muffled"),
        ("clipped",    "pristine"),
        ("fluid",      "rigid"),
        ("expansive",  "confined"),
        ("layered",    "flat"),
        ("dense",      "hollow"),
        ("fragmented", "cohesive"),
        ("agitated",   "calm"),
        ("warped",     "linear"),
        ("crisp",      "muddy"),
        ("deep",       "thin"),
    ],
    "img_templates": [
        "a {word} image",
        "a {word} photograph",
        "an image with a {word} mood",
        "a {word} visual style",
        "a {word} scene",
        "a {word} picture",
        "a photo that feels {word}",
        "artwork with a {word} atmosphere",
    ],
}

# attempt13 vocab WITHOUT "warped" - replace with "distorted" to see if warped is the issue
VOCAB_CANDIDATE_NO_WARPED = {
    "name": "attempt13_no_warped",
    "axes": [
        ("granular",   "polished"),
        ("piercing",   "muffled"),
        ("clipped",    "pristine"),
        ("fluid",      "rigid"),
        ("expansive",  "confined"),
        ("layered",    "flat"),
        ("dense",      "hollow"),
        ("fragmented", "cohesive"),
        ("agitated",   "calm"),
        ("distorted",  "stable"),   # replace warped/linear
        ("crisp",      "muddy"),
        ("deep",       "thin"),
    ],
    "img_templates": [
        "a {word} image",
        "a {word} visual texture",
        "an image with a {word} quality",
        "something that looks {word}",
        "an image that appears more {word}",
        "a {word} visual effect on an image",
        "image processing with a {word} result",
        "the {word} character of a processed image",
    ],
}

# attempt13 vocab with "oscillating/steady" replacing "warped/linear"
VOCAB_OSCILLATING = {
    "name": "attempt13_oscillating",
    "axes": [
        ("granular",    "polished"),
        ("piercing",    "muffled"),
        ("clipped",     "pristine"),
        ("fluid",       "rigid"),
        ("expansive",   "confined"),
        ("layered",     "flat"),
        ("dense",       "hollow"),
        ("fragmented",  "cohesive"),
        ("agitated",    "calm"),
        ("oscillating", "steady"),   # replace warped/linear: periodic variation vs stability
        ("crisp",       "muddy"),
        ("deep",        "thin"),
    ],
    "img_templates": [
        "a {word} image",
        "a {word} photograph",
        "an image with a {word} mood",
        "a {word} visual style",
        "a {word} scene",
        "a {word} picture",
        "a photo that feels {word}",
        "artwork with a {word} atmosphere",
    ],
}

# attempt13 vocab with "undulating/uniform" replacing "warped/linear" (second candidate)
VOCAB_UNDULATING = {
    "name": "attempt13_undulating",
    "axes": [
        ("granular",    "polished"),
        ("piercing",    "muffled"),
        ("clipped",     "pristine"),
        ("fluid",       "rigid"),
        ("expansive",   "confined"),
        ("layered",     "flat"),
        ("dense",       "hollow"),
        ("fragmented",  "cohesive"),
        ("agitated",    "calm"),
        ("undulating",  "uniform"),  # replace warped/linear: wave-like vs constant/even
        ("crisp",       "muddy"),
        ("deep",        "thin"),
    ],
    "img_templates": [
        "a {word} image",
        "a {word} photograph",
        "an image with a {word} mood",
        "a {word} visual style",
        "a {word} scene",
        "a {word} picture",
        "a photo that feels {word}",
        "artwork with a {word} atmosphere",
    ],
}

VOCAB_ATTEMPT11 = {
    "name": "attempt11_original",
    "axes": [
        ("warm",       "cold"),
        ("bright",     "dark"),
        ("rough",      "smooth"),
        ("heavy",      "light"),
        ("thick",      "thin"),
        ("distant",    "intimate"),
        ("soft",       "hard"),
        ("static",     "dynamic"),
        ("clean",      "dirty"),
        ("dreamy",     "realistic"),
        ("vintage",    "modern"),
        ("natural",    "surreal"),
    ],
    "img_templates": [
        "a {word} image",
        "a {word} photograph",
        "an image with a {word} mood",
        "a {word} visual style",
        "a {word} scene",
        "a {word} picture",
        "a photo that feels {word}",
        "artwork with a {word} atmosphere",
    ],
}

WAND_EFFECTS = [
    "adaptive_blur",
    "motion_blur",
    "adaptive_sharpen",
    "add_noise",
    "spread",
    "sepia_tone",
    "solarize",
]

INTENSITIES = [0.2, 0.4, 0.6, 0.8, 1.0]


# ─────────────────────────────────────────────────────────────────────────────
# CLIP helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_vocab_embeddings(clip_model, axes: list, templates: list, device: str) -> tuple[np.ndarray, list[str]]:
    """Build L2-normalized vocab embeddings using prompt ensemble."""
    import torch

    keywords = [w for pair in axes for w in pair]
    prompts = []
    for kw in keywords:
        for tmpl in templates:
            prompts.append(tmpl.format(word=kw))

    # Tokenize and embed in batches
    batch_size = 128
    all_embs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i: i + batch_size]
        tokens = clip_model.tokenizer(batch).to(device)
        with torch.no_grad():
            emb = clip_model.model.encode_text(tokens).float()
        emb = emb / emb.norm(dim=-1, keepdim=True)
        all_embs.append(emb.cpu().numpy())

    all_embs = np.concatenate(all_embs, axis=0)  # (n_words * n_templates, dim)
    n_words = len(keywords)
    n_templates = len(templates)
    all_embs = all_embs.reshape(n_words, n_templates, -1)

    # Average across templates, then re-normalize
    mean_emb = all_embs.mean(axis=1)  # (n_words, dim)
    mean_emb = mean_emb / np.maximum(np.linalg.norm(mean_emb, axis=1, keepdims=True), 1e-8)
    return mean_emb.astype(np.float32), keywords


def _embed_pil(clip_model, pil_img, device: str) -> np.ndarray:
    import torch
    tensor = clip_model.preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.model.encode_image(tensor).float()
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze(0).cpu().numpy().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Style score computation
# ─────────────────────────────────────────────────────────────────────────────

def _softmax(logits: np.ndarray, T: float) -> np.ndarray:
    logits = logits / T
    logits -= logits.max()
    e = np.exp(logits)
    return e / e.sum()


def compute_style_score(clip_orig: np.ndarray, clip_edit: np.ndarray,
                         vocab_emb: np.ndarray, T: float = 0.015) -> np.ndarray:
    """Compute style score from CLIP delta (no confidence mixing for simplicity)."""
    delta = clip_edit - clip_orig
    norm = float(np.linalg.norm(delta))
    if norm < 1e-6:
        return np.full(vocab_emb.shape[0], 1.0 / vocab_emb.shape[0], dtype=np.float32)
    delta_n = delta / norm
    sims = vocab_emb @ delta_n
    return _softmax(sims, T).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Per-effect profile computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_effect_profiles(
    image_paths: list[str],
    vocab_emb: np.ndarray,
    clip_model,
    device: str,
    intensities: list[float],
    temperature: float = 0.015,
) -> dict[str, np.ndarray]:
    """Compute mean style profile for each wand effect."""
    from PIL import Image as PILImage
    from src.effects.wand_image_effects import apply_effect

    print(f"  Embedding {len(image_paths)} original images...")
    orig_embs = {}
    for path in image_paths:
        try:
            pil = PILImage.open(path).convert("RGB")
            orig_embs[path] = _embed_pil(clip_model, pil, device)
        except Exception as e:
            print(f"  [WARN] skip {path}: {e}")
    print(f"  Successfully embedded {len(orig_embs)} images")

    profiles: dict[str, np.ndarray] = {}
    for eff in WAND_EFFECTS:
        scores = []
        for path, orig_emb in orig_embs.items():
            try:
                pil_orig = PILImage.open(path).convert("RGB")
            except Exception:
                continue
            for intensity in intensities:
                try:
                    pil_edit = apply_effect(pil_orig, eff, intensity)
                    edit_emb = _embed_pil(clip_model, pil_edit, device)
                    s = compute_style_score(orig_emb, edit_emb, vocab_emb, T=temperature)
                    scores.append(s)
                except Exception as e:
                    pass  # skip failed augmentations

        if scores:
            profile = np.stack(scores).mean(axis=0)
            profiles[eff] = profile
            delta_norms = []
            print(f"  {eff}: n_samples={len(scores)}")
        else:
            print(f"  [WARN] {eff}: no valid samples")

    return profiles


# ─────────────────────────────────────────────────────────────────────────────
# Discriminativeness metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_discriminativeness(profiles: dict[str, np.ndarray]) -> dict:
    """
    Mean pairwise cosine similarity between effect profiles.
    Lower = more discriminative = BETTER.
    """
    keys = list(profiles.keys())
    n = len(keys)
    if n < 2:
        return {"mean_pairwise_cos": 1.0, "pairwise_matrix": {}}

    pairwise = {}
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = profiles[keys[i]], profiles[keys[j]]
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
            pairwise[(keys[i], keys[j])] = cos
            sims.append(cos)

    return {
        "mean_pairwise_cos": float(np.mean(sims)),
        "std_pairwise_cos": float(np.std(sims)),
        "min_pairwise_cos": float(np.min(sims)),
        "max_pairwise_cos": float(np.max(sims)),
        "pairwise_matrix": pairwise,
    }


def top_terms_per_effect(profiles: dict[str, np.ndarray], keywords: list[str], top_k: int = 5) -> dict:
    result = {}
    for eff, prof in profiles.items():
        top_idx = np.argsort(prof)[-top_k:][::-1]
        result[eff] = [(keywords[i], float(prof[i])) for i in top_idx]
    return result


def dominant_terms(profiles: dict[str, np.ndarray], keywords: list[str]) -> list:
    all_profiles = np.stack(list(profiles.values()))
    mean_across_effects = all_profiles.mean(axis=0)
    top_idx = np.argsort(mean_across_effects)[-5:][::-1]
    return [(keywords[i], float(mean_across_effects[i])) for i in top_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Run test for a single vocab configuration
# ─────────────────────────────────────────────────────────────────────────────

def test_vocab(vocab_cfg: dict, image_paths: list[str], clip_model, device: str,
               temperature: float = 0.015) -> dict:
    name = vocab_cfg["name"]
    axes = vocab_cfg["axes"]
    templates = vocab_cfg["img_templates"]

    print(f"\n{'='*70}")
    print(f"Testing vocab: {name}")
    print(f"  {len(axes)} axes, {len(axes)*2} terms, {len(templates)} templates")
    print(f"{'='*70}")
    print(f"  Terms: {[w for pair in axes for w in pair]}")

    # Build vocab embeddings
    print(f"\n  Building vocab embeddings...")
    vocab_emb, keywords = build_vocab_embeddings(clip_model, axes, templates, device)
    print(f"  Vocab: {vocab_emb.shape[0]} terms, {vocab_emb.shape[1]}d")

    # Compute per-effect style profiles
    print(f"\n  Computing image effect profiles...")
    profiles = compute_effect_profiles(image_paths, vocab_emb, clip_model, device,
                                        intensities=INTENSITIES, temperature=temperature)

    # Discriminativeness metrics
    disc = compute_discriminativeness(profiles)
    top_terms = top_terms_per_effect(profiles, keywords, top_k=3)
    dom = dominant_terms(profiles, keywords)

    # Report
    print(f"\n  RESULT [{name}]:")
    print(f"  Mean pairwise cosine sim: {disc['mean_pairwise_cos']:.4f}  "
          f"(std={disc['std_pairwise_cos']:.4f}, "
          f"min={disc['min_pairwise_cos']:.4f}, "
          f"max={disc['max_pairwise_cos']:.4f})")
    print(f"  → Lower is BETTER (more discriminative)")

    print(f"\n  Top terms per effect:")
    for eff, terms in top_terms.items():
        terms_str = ", ".join(f"{t}({s:.3f})" for t, s in terms)
        print(f"    {eff:20s}: {terms_str}")

    print(f"\n  Dominant terms (mean across all effects):")
    for term, score in dom:
        print(f"    {term:20s}: {score:.4f}")

    return {
        "name": name,
        "mean_pairwise_cos": disc["mean_pairwise_cos"],
        "std_pairwise_cos": disc["std_pairwise_cos"],
        "min_pairwise_cos": disc["min_pairwise_cos"],
        "max_pairwise_cos": disc["max_pairwise_cos"],
        "top_terms": top_terms,
        "dominant_terms": dom,
        "profiles": {k: v.tolist() for k, v in profiles.items()},
        "keywords": keywords,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", default="data/original/images")
    p.add_argument("--n-images", type=int, default=30)
    p.add_argument("--device", default="cpu")
    p.add_argument("--temperature", type=float, default=0.015)
    p.add_argument("--vocabs", nargs="+",
                   choices=["candidate", "candidate_std", "candidate_no_warped",
                             "oscillating", "undulating", "attempt12", "attempt11", "all"],
                   default=["all"],
                   help="Which vocab(s) to test")
    args = p.parse_args()

    # Select vocabs to test
    vocab_map = {
        "candidate": VOCAB_CANDIDATE,
        "candidate_std": VOCAB_CANDIDATE_STD_TEMPLATES,
        "candidate_no_warped": VOCAB_CANDIDATE_NO_WARPED,
        "oscillating": VOCAB_OSCILLATING,
        "undulating": VOCAB_UNDULATING,
        "attempt12": VOCAB_ATTEMPT12,
        "attempt11": VOCAB_ATTEMPT11,
    }
    if "all" in args.vocabs:
        vocabs_to_test = list(vocab_map.values())
    else:
        vocabs_to_test = [vocab_map[v] for v in args.vocabs]

    # Load CLIP
    print(f"\nLoading CLIP ViT-L-14...")
    from src.models.clip_embedder import CLIPEmbedder
    clip = CLIPEmbedder(model_name="ViT-L-14", pretrained="openai", device=args.device)
    print(f"CLIP loaded on {args.device}")

    # Collect images
    from pathlib import Path
    root = Path(args.image_dir)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    all_paths = [str(p2) for p2 in sorted(root.rglob("*")) if p2.suffix.lower() in exts]
    if not all_paths:
        print(f"ERROR: No images found in {args.image_dir}")
        sys.exit(1)
    if len(all_paths) > args.n_images:
        indices = np.linspace(0, len(all_paths) - 1, args.n_images, dtype=int)
        image_paths = [all_paths[i] for i in indices]
    else:
        image_paths = all_paths
    print(f"\nUsing {len(image_paths)} images from {args.image_dir}")

    # Run tests
    results = []
    for vocab_cfg in vocabs_to_test:
        r = test_vocab(vocab_cfg, image_paths, clip, args.device, args.temperature)
        results.append(r)

    # Final comparison
    print(f"\n{'='*70}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Vocab':<30} {'Mean Pairwise Cos':>18}  {'Verdict'}")
    print(f"{'-'*70}")
    for r in sorted(results, key=lambda x: x["mean_pairwise_cos"]):
        verdict = "← BEST" if r == min(results, key=lambda x: x["mean_pairwise_cos"]) else ""
        print(f"{r['name']:<30} {r['mean_pairwise_cos']:>18.4f}  {verdict}")
    print(f"\nNOTE: Lower mean pairwise cosine = more discriminative = BETTER")
    print(f"Reference: attempt11≈0.83, attempt12≈0.94 (image side)")


if __name__ == "__main__":
    main()
