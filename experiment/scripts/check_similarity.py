#!/usr/bin/env python
"""
CLIP+CLAP Similarity Checker

Compute pairwise cosine similarity between files using CLIP (image) and CLAP (audio) embeddings.
Supports image-image, audio-audio, and cross-modal (image↔audio) comparison with CCA alignment.

Usage:
    # Image-image similarity (category-level + pairwise)
    python experiment/scripts/check_similarity.py --images data/original/images/

    # Audio-audio similarity
    python experiment/scripts/check_similarity.py --audio data/original/audio/

    # Cross-modal (image ↔ audio)
    python experiment/scripts/check_similarity.py --images data/original/images/ --audio data/original/audio/

    # Specific files only
    python experiment/scripts/check_similarity.py --files img1.jpg img2.jpg audio1.wav

    # Category-level only (faster for large datasets)
    python experiment/scripts/check_similarity.py --images data/original/images/ --mode category

    # Save heatmap
    python experiment/scripts/check_similarity.py --images data/original/images/ --save heatmap.png
"""

import argparse
import sys
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models import CLIPEmbedder, CLAPEmbedder, MultimodalEmbedder, CCAAlignment

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between two embedding matrices. a: (N, D), b: (M, D) -> (N, M)"""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T


def collect_files(directory: Path, extensions: set) -> Dict[str, List[Path]]:
    """
    Collect files grouped by category (subfolder).

    If directory has subfolders, each subfolder is a category.
    If directory is flat, all files go under a single "_all" category.

    Returns:
        {category_name: [file_paths]}
    """
    groups = OrderedDict()

    subdirs = sorted([d for d in directory.iterdir() if d.is_dir()])

    if subdirs:
        for subdir in subdirs:
            files = sorted([
                f for f in subdir.iterdir()
                if f.is_file()
                and f.suffix.lower() in extensions
                and not f.name.startswith("._")
            ])
            if files:
                groups[subdir.name] = files
    else:
        files = sorted([
            f for f in directory.iterdir()
            if f.is_file()
            and f.suffix.lower() in extensions
            and not f.name.startswith("._")
        ])
        if files:
            groups["_all"] = files

    return groups


def detect_modality(path: Path) -> str:
    if path.suffix.lower() in IMAGE_EXTS:
        return "image"
    elif path.suffix.lower() in AUDIO_EXTS:
        return "audio"
    else:
        raise ValueError(f"Unknown file type: {path.suffix}")


def embed_files(
    embedder: MultimodalEmbedder,
    paths: List[Path],
    modality: str,
    batch_size: int = 16,
) -> np.ndarray:
    """Embed a list of files. Returns (N, D) numpy array."""
    all_embeddings = []

    for i in range(0, len(paths), batch_size):
        batch_paths = [str(p) for p in paths[i:i + batch_size]]

        if modality == "image":
            embs = embedder.embed_image_paths(batch_paths)
        else:
            embs = embedder.embed_audio_paths(batch_paths)

        all_embeddings.append(embs.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def print_pairwise_matrix(
    sim_matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str = "",
    max_label_len: int = 20,
):
    """Print a formatted similarity matrix."""
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")

    # Truncate labels
    row_labels = [l[:max_label_len] for l in row_labels]
    col_labels = [l[:max_label_len] for l in col_labels]

    label_w = max(len(l) for l in row_labels) + 2

    # Header
    header = " " * label_w
    for cl in col_labels:
        header += f"{cl:>8}"
    print(header)
    print(" " * label_w + "-" * (8 * len(col_labels)))

    # Rows
    for i, rl in enumerate(row_labels):
        row = f"{rl:<{label_w}}"
        for j in range(len(col_labels)):
            row += f"{sim_matrix[i, j]:>8.3f}"
        print(row)


def print_top_pairs(
    sim_matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    top_k: int = 20,
    title: str = "",
    exclude_diagonal: bool = True,
):
    """Print top-k most similar pairs."""
    if title:
        print(f"\n{'-' * 50}")
        print(f"  {title}")
        print(f"{'-' * 50}")

    pairs = []
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            if exclude_diagonal and i == j and row_labels[i] == col_labels[j]:
                continue
            pairs.append((sim_matrix[i, j], row_labels[i], col_labels[j]))

    pairs.sort(key=lambda x: x[0], reverse=True)

    print(f"  {'Rank':>4}  {'Sim':>7}  {'Item A':<30}  {'Item B':<30}")
    print(f"  {'-' * 4}  {'-' * 7}  {'-' * 30}  {'-' * 30}")
    for rank, (sim, a, b) in enumerate(pairs[:top_k], 1):
        print(f"  {rank:>4}  {sim:>7.4f}  {a:<30}  {b:<30}")

    if len(pairs) > top_k:
        print(f"\n  ... and {len(pairs) - top_k} more pairs")
        # Also show bottom-k
        print(f"\n  Bottom {min(5, len(pairs))} pairs:")
        for sim, a, b in pairs[-5:]:
            print(f"         {sim:>7.4f}  {a:<30}  {b:<30}")


def run_category_similarity(
    embedder: MultimodalEmbedder,
    groups: Dict[str, List[Path]],
    modality: str,
    batch_size: int = 16,
) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
    """
    Compute category-level similarity.

    Returns:
        sim_matrix: (C, C) category similarity matrix
        category_names: list of category names
        category_embeddings: {name: (N, D) individual embeddings}
    """
    category_names = list(groups.keys())
    category_means = []
    category_embeddings = {}

    for name in category_names:
        paths = groups[name]
        print(f"  Embedding {name}: {len(paths)} files...")
        embs = embed_files(embedder, paths, modality, batch_size)
        category_embeddings[name] = embs
        category_means.append(embs.mean(axis=0))

    means = np.stack(category_means)  # (C, D)
    sim_matrix = cosine_sim(means, means)

    return sim_matrix, category_names, category_embeddings


def save_heatmap(
    sim_matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    save_path: str,
    title: str = "Cosine Similarity",
):
    """Save a heatmap of the similarity matrix."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 0.8), max(6, len(row_labels) * 0.6)))
    im = ax.imshow(sim_matrix, cmap="RdYlBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(row_labels, fontsize=9)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(j, i, f"{sim_matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nHeatmap saved to {save_path}")
    plt.close()


def run_single_modality(
    embedder: MultimodalEmbedder,
    directory: Path,
    modality: str,
    mode: str,
    batch_size: int,
    top_k: int,
    save_path: Optional[str],
):
    """Run similarity analysis for a single modality."""
    exts = IMAGE_EXTS if modality == "image" else AUDIO_EXTS
    groups = collect_files(directory, exts)

    if not groups:
        print(f"No {modality} files found in {directory}")
        return

    total_files = sum(len(v) for v in groups.values())
    print(f"\nFound {total_files} {modality} files in {len(groups)} group(s)")

    if mode in ("category", "both") and len(groups) > 1:
        print(f"\n--- Category-level {modality.upper()} similarity ---")
        sim_matrix, names, cat_embs = run_category_similarity(
            embedder, groups, modality, batch_size
        )
        print_pairwise_matrix(
            sim_matrix, names, names,
            title=f"{modality.upper()} Category Similarity (mean embedding)"
        )

        # Stats
        triu = sim_matrix[np.triu_indices(len(names), k=1)]
        if len(triu) > 0:
            print(f"\n  Off-diagonal stats: mean={triu.mean():.4f}, "
                  f"std={triu.std():.4f}, min={triu.min():.4f}, max={triu.max():.4f}")

        if save_path:
            save_heatmap(sim_matrix, names, names, save_path,
                         title=f"{modality.capitalize()} Category Similarity")

    if mode in ("pairwise", "both"):
        # For pairwise, embed all files together
        all_paths = []
        all_labels = []
        for name, paths in groups.items():
            for p in paths:
                all_paths.append(p)
                prefix = f"{name}/" if len(groups) > 1 else ""
                all_labels.append(f"{prefix}{p.name}")

        if len(all_paths) > 500:
            print(f"\n  WARNING: {len(all_paths)} files - pairwise will be slow. "
                  f"Consider --mode category")

        print(f"\n--- File-level {modality.upper()} pairwise similarity ---")
        print(f"  Embedding {len(all_paths)} files...")
        all_embs = embed_files(embedder, all_paths, modality, batch_size)

        sim_matrix = cosine_sim(all_embs, all_embs)

        # Show top-k pairs
        print_top_pairs(
            sim_matrix, all_labels, all_labels, top_k=top_k,
            title=f"Top-{top_k} most similar {modality} pairs"
        )

        # Within-category vs across-category stats
        if len(groups) > 1:
            within_sims = []
            across_sims = []
            group_indices = {}
            idx = 0
            for name, paths in groups.items():
                group_indices[name] = list(range(idx, idx + len(paths)))
                idx += len(paths)

            for name, indices in group_indices.items():
                for i in indices:
                    for j in indices:
                        if i < j:
                            within_sims.append(sim_matrix[i, j])

            for n1, idx1 in group_indices.items():
                for n2, idx2 in group_indices.items():
                    if n1 < n2:
                        for i in idx1:
                            for j in idx2:
                                across_sims.append(sim_matrix[i, j])

            within_sims = np.array(within_sims)
            across_sims = np.array(across_sims)

            print(f"\n  Within-category:  mean={within_sims.mean():.4f}, std={within_sims.std():.4f}")
            print(f"  Across-category:  mean={across_sims.mean():.4f}, std={across_sims.std():.4f}")
            print(f"  Gap: {within_sims.mean() - across_sims.mean():.4f}")


def run_cross_modal(
    embedder: MultimodalEmbedder,
    image_dir: Path,
    audio_dir: Path,
    mode: str,
    batch_size: int,
    top_k: int,
    save_path: Optional[str],
):
    """Run cross-modal (image ↔ audio) similarity analysis."""
    image_groups = collect_files(image_dir, IMAGE_EXTS)
    audio_groups = collect_files(audio_dir, AUDIO_EXTS)

    if not image_groups or not audio_groups:
        print("Need both image and audio files for cross-modal comparison")
        return

    total_images = sum(len(v) for v in image_groups.values())
    total_audio = sum(len(v) for v in audio_groups.values())
    print(f"\nFound {total_images} images ({len(image_groups)} groups), "
          f"{total_audio} audio ({len(audio_groups)} groups)")

    if mode in ("category", "both"):
        print(f"\n--- Cross-modal category similarity ---")

        # Embed image categories
        img_names = []
        img_means = []
        for name, paths in image_groups.items():
            print(f"  [IMG] Embedding {name}: {len(paths)} files...")
            embs = embed_files(embedder, paths, "image", batch_size)
            img_means.append(embs.mean(axis=0))
            img_names.append(f"img:{name}")

        # Embed audio categories
        aud_names = []
        aud_means = []
        for name, paths in audio_groups.items():
            print(f"  [AUD] Embedding {name}: {len(paths)} files...")
            embs = embed_files(embedder, paths, "audio", batch_size)
            aud_means.append(embs.mean(axis=0))
            aud_names.append(f"aud:{name}")

        img_mat = np.stack(img_means)
        aud_mat = np.stack(aud_means)
        cross_sim = cosine_sim(img_mat, aud_mat)

        print_pairwise_matrix(
            cross_sim, img_names, aud_names,
            title="Cross-modal Category Similarity (Image ↔ Audio)"
        )

        if save_path:
            save_heatmap(cross_sim, img_names, aud_names, save_path,
                         title="Cross-modal Similarity (Image ↔ Audio)")

    if mode in ("pairwise", "both"):
        print(f"\n--- Cross-modal file-level pairwise similarity ---")

        # Embed all images
        img_paths = []
        img_labels = []
        for name, paths in image_groups.items():
            for p in paths:
                img_paths.append(p)
                prefix = f"{name}/" if len(image_groups) > 1 else ""
                img_labels.append(f"img:{prefix}{p.name}")

        # Embed all audio
        aud_paths = []
        aud_labels = []
        for name, paths in audio_groups.items():
            for p in paths:
                aud_paths.append(p)
                prefix = f"{name}/" if len(audio_groups) > 1 else ""
                aud_labels.append(f"aud:{prefix}{p.name}")

        total = len(img_paths) + len(aud_paths)
        if total > 500:
            print(f"  WARNING: {total} total files - this may be slow")

        print(f"  Embedding {len(img_paths)} images...")
        img_embs = embed_files(embedder, img_paths, "image", batch_size)

        print(f"  Embedding {len(aud_paths)} audio files...")
        aud_embs = embed_files(embedder, aud_paths, "audio", batch_size)

        cross_sim = cosine_sim(img_embs, aud_embs)

        print_top_pairs(
            cross_sim, img_labels, aud_labels, top_k=top_k,
            title=f"Top-{top_k} cross-modal pairs (image ↔ audio)",
            exclude_diagonal=False,
        )

        print(f"\n  Overall cross-modal stats: "
              f"mean={cross_sim.mean():.4f}, std={cross_sim.std():.4f}, "
              f"min={cross_sim.min():.4f}, max={cross_sim.max():.4f}")


def run_files_mode(
    embedder: MultimodalEmbedder,
    file_paths: List[str],
    batch_size: int,
):
    """Compute pairwise similarity between specific files."""
    paths = [Path(p) for p in file_paths]

    # Separate by modality
    image_paths = [p for p in paths if p.suffix.lower() in IMAGE_EXTS]
    audio_paths = [p for p in paths if p.suffix.lower() in AUDIO_EXTS]

    all_embs = []
    all_labels = []

    if image_paths:
        print(f"  Embedding {len(image_paths)} images...")
        img_embs = embed_files(embedder, image_paths, "image", batch_size)
        all_embs.append(img_embs)
        all_labels.extend([f"img:{p.name}" for p in image_paths])

    if audio_paths:
        print(f"  Embedding {len(audio_paths)} audio files...")
        aud_embs = embed_files(embedder, audio_paths, "audio", batch_size)
        all_embs.append(aud_embs)
        all_labels.extend([f"aud:{p.name}" for p in audio_paths])

    if not all_embs:
        print("No valid files found")
        return

    embeddings = np.concatenate(all_embs, axis=0)
    sim_matrix = cosine_sim(embeddings, embeddings)

    if len(all_labels) <= 30:
        print_pairwise_matrix(sim_matrix, all_labels, all_labels,
                              title="File Pairwise Similarity")
    else:
        print_top_pairs(sim_matrix, all_labels, all_labels, top_k=20,
                        title="Top-20 most similar pairs")


def main():
    parser = argparse.ArgumentParser(description="CLIP+CLAP Similarity Checker")

    parser.add_argument("--images", type=str, help="Image directory")
    parser.add_argument("--audio", type=str, help="Audio directory")
    parser.add_argument("--files", nargs="+", type=str, help="Specific file paths")

    parser.add_argument(
        "--mode", choices=["category", "pairwise", "both"],
        default="both",
        help="Analysis mode (default: both)",
    )
    parser.add_argument("--device", type=str, default="mps", help="Device (default: mps)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k pairs to show (default: 20)")
    parser.add_argument("--save", type=str, help="Save heatmap to file (PNG)")
    parser.add_argument("--alignment", type=str, help="Path to CCA alignment model (for cross-modal)")

    args = parser.parse_args()

    if not args.images and not args.audio and not args.files:
        parser.error("Provide --images, --audio, --files, or a combination")

    # Load models
    print(f"Loading CLIP and CLAP on {args.device}...")
    clip_embedder = CLIPEmbedder(device=args.device)
    clap_embedder = CLAPEmbedder(device=args.device)
    embedder = MultimodalEmbedder(clip_embedder, clap_embedder)

    if args.files:
        # Specific files mode
        run_files_mode(embedder, args.files, args.batch_size)

    elif args.images and args.audio:
        # Cross-modal mode
        run_cross_modal(
            embedder,
            Path(args.images), Path(args.audio),
            args.mode, args.batch_size, args.top_k, args.save,
        )

    elif args.images:
        run_single_modality(
            embedder, Path(args.images), "image",
            args.mode, args.batch_size, args.top_k, args.save,
        )

    elif args.audio:
        run_single_modality(
            embedder, Path(args.audio), "audio",
            args.mode, args.batch_size, args.top_k, args.save,
        )


if __name__ == "__main__":
    main()
