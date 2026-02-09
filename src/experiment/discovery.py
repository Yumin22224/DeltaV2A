"""
Phase 1: Effect Correspondence Discovery (Text Anchor Ensemble)

Discovers cross-modal effect correspondences using a 3-way similarity metric:
- Image prototype ↔ Image text anchor
- Audio prototype ↔ Audio text anchor
- Image text anchor ↔ Audio text anchor

Final score uses multiplication (AND logic - all three must be high).
"""

import numpy as np
import torch
import json
from typing import Dict, List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from .text_anchor import generate_text_anchors, compute_text_anchor_similarity, TextAnchor


def load_deltas(deltas_path: str) -> List[Dict]:
    """Load delta embeddings from JSON."""
    with open(deltas_path, 'r') as f:
        data = json.load(f)
    # Handle both formats: {"deltas": [...]} and [...]
    if isinstance(data, dict) and 'deltas' in data:
        return data['deltas']
    return data


def compute_prototypes(
    deltas: List[Dict],
    modality: str,
) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Compute prototype vectors for each (effect_type, intensity) combination.

    Prototype = E[Δe / ||Δe||]  (mean of normalized deltas)

    Args:
        deltas: List of delta dictionaries
        modality: "image" or "audio"

    Returns:
        Dictionary mapping (effect_type, intensity) to prototype vector
    """
    # Group deltas by (effect_type, intensity)
    grouped = {}
    for delta_dict in deltas:
        if delta_dict['modality'] != modality:
            continue

        key = (delta_dict['effect_type'], delta_dict['intensity'])
        delta = np.array(delta_dict['delta'])

        # Normalize delta
        norm = np.linalg.norm(delta)
        if norm > 0:
            normalized_delta = delta / norm
        else:
            normalized_delta = delta

        if key not in grouped:
            grouped[key] = []
        grouped[key].append(normalized_delta)

    # Compute prototype (mean of normalized deltas)
    prototypes = {}
    for key, deltas_list in grouped.items():
        prototypes[key] = np.mean(deltas_list, axis=0)

    return prototypes


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(v1, v2) / (norm1 * norm2))


def compute_discovery_matrix(
    image_prototypes: Dict[Tuple[str, str], np.ndarray],
    audio_prototypes: Dict[Tuple[str, str], np.ndarray],
    image_text_anchors: Dict[str, TextAnchor],
    audio_text_anchors: Dict[str, TextAnchor],
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Compute discovery matrix using 3-way similarity.

    For each (image_effect, audio_effect) pair:
        score = Sim(img_proto, img_text) × Sim(aud_proto, aud_text) × Sim(img_text, aud_text)

    Args:
        image_prototypes: (effect_type, intensity) -> prototype vector
        audio_prototypes: (effect_type, intensity) -> prototype vector
        image_text_anchors: effect_type -> TextAnchor
        audio_text_anchors: effect_type -> TextAnchor

    Returns:
        matrix: (n_image_effects, n_audio_effects) discovery scores
        image_labels: List of image effect labels
        audio_labels: List of audio effect labels
    """
    # Get unique image and audio effects (with intensities)
    image_keys = sorted(image_prototypes.keys())
    audio_keys = sorted(audio_prototypes.keys())

    # Create labels: "effect_type (intensity)"
    image_labels = [f"{effect} ({intensity})" for effect, intensity in image_keys]
    audio_labels = [f"{effect} ({intensity})" for effect, intensity in audio_keys]

    # Initialize matrix
    n_image = len(image_keys)
    n_audio = len(audio_keys)
    matrix = np.zeros((n_image, n_audio))

    # Compute 3-way similarity for each pair
    for i, (img_effect, img_intensity) in enumerate(image_keys):
        for j, (aud_effect, aud_intensity) in enumerate(audio_keys):
            # Skip if text anchors don't exist
            if img_effect not in image_text_anchors or aud_effect not in audio_text_anchors:
                matrix[i, j] = 0.0
                continue

            # Get vectors
            img_proto = image_prototypes[(img_effect, img_intensity)]
            aud_proto = audio_prototypes[(aud_effect, aud_intensity)]
            img_text = image_text_anchors[img_effect].delta
            aud_text = audio_text_anchors[aud_effect].delta

            # Compute 3 similarities
            sim_img = cosine_similarity(img_proto, img_text)  # Image proto ↔ Image text
            sim_aud = cosine_similarity(aud_proto, aud_text)  # Audio proto ↔ Audio text
            sim_cross = compute_text_anchor_similarity(
                image_text_anchors[img_effect],
                audio_text_anchors[aud_effect],
            )  # Image text ↔ Audio text

            # Multiply (AND logic)
            score = sim_img * sim_aud * sim_cross
            matrix[i, j] = score

    return matrix, image_labels, audio_labels


def plot_discovery_heatmap(
    matrix: np.ndarray,
    image_labels: List[str],
    audio_labels: List[str],
    output_path: str,
):
    """
    Plot discovery matrix as a heatmap.

    Args:
        matrix: (n_image_effects, n_audio_effects) discovery scores
        image_labels: Image effect labels (y-axis)
        audio_labels: Audio effect labels (x-axis)
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 10))

    sns.heatmap(
        matrix,
        xticklabels=audio_labels,
        yticklabels=image_labels,
        cmap='YlOrRd',
        annot=True,
        fmt='.3f',
        cbar_kws={'label': '3-Way Similarity Score'},
        vmin=0,
        vmax=1,
    )

    plt.title('Effect Correspondence Discovery Matrix\n(Image Effects × Audio Effects)',
              fontsize=14, pad=20)
    plt.xlabel('Audio Effects', fontsize=12)
    plt.ylabel('Image Effects', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved discovery heatmap to {output_path}")
    plt.close()


def run_discovery(
    image_deltas_path: str,
    audio_deltas_path: str,
    embedder,  # MultimodalEmbedder
    output_dir: str,
    image_categories: List[str] = None,
    audio_genres: List[str] = None,
) -> Dict:
    """
    Run Phase 1: Effect Correspondence Discovery.

    Args:
        image_deltas_path: Path to image_deltas.json
        audio_deltas_path: Path to audio_deltas.json
        embedder: MultimodalEmbedder instance
        output_dir: Directory to save results
        image_categories: Categories for text anchor generation
        audio_genres: Genres for text anchor generation

    Returns:
        results: Dictionary with discovery matrix and metadata
    """
    print("\n" + "="*60)
    print("PHASE 1: EFFECT CORRESPONDENCE DISCOVERY")
    print("="*60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load deltas
    print("\n1. Loading delta embeddings...")
    image_deltas = load_deltas(image_deltas_path)
    audio_deltas = load_deltas(audio_deltas_path)
    print(f"   Loaded {len(image_deltas)} image deltas")
    print(f"   Loaded {len(audio_deltas)} audio deltas")

    # Compute prototypes
    print("\n2. Computing prototype vectors...")
    image_prototypes = compute_prototypes(image_deltas, "image")
    audio_prototypes = compute_prototypes(audio_deltas, "audio")
    print(f"   Image prototypes: {len(image_prototypes)}")
    print(f"   Audio prototypes: {len(audio_prototypes)}")

    # Get effect types from deltas
    image_effect_types = list(set(d['effect_type'] for d in image_deltas))
    audio_effect_types = list(set(d['effect_type'] for d in audio_deltas))

    # Generate text anchors
    print("\n3. Generating text anchors...")
    print("   Image text anchors:")
    image_text_anchors = generate_text_anchors(
        embedder=embedder.clip,
        modality="image",
        effect_types=image_effect_types,
        categories=image_categories,
    )
    print(f"      Generated {len(image_text_anchors)} anchors")

    print("   Audio text anchors:")
    audio_text_anchors = generate_text_anchors(
        embedder=embedder.clap,
        modality="audio",
        effect_types=audio_effect_types,
        genres=audio_genres,
    )
    print(f"      Generated {len(audio_text_anchors)} anchors")

    # Compute discovery matrix
    print("\n4. Computing discovery matrix...")
    matrix, image_labels, audio_labels = compute_discovery_matrix(
        image_prototypes=image_prototypes,
        audio_prototypes=audio_prototypes,
        image_text_anchors=image_text_anchors,
        audio_text_anchors=audio_text_anchors,
    )
    print(f"   Matrix shape: {matrix.shape}")

    # Find top correspondences
    print("\n5. Top correspondences:")
    flat_indices = np.argsort(matrix.flatten())[::-1][:10]  # Top 10
    for rank, flat_idx in enumerate(flat_indices, 1):
        i = flat_idx // matrix.shape[1]
        j = flat_idx % matrix.shape[1]
        score = matrix[i, j]
        print(f"   {rank:2d}. {image_labels[i]:25s} ↔ {audio_labels[j]:25s}  (score: {score:.4f})")

    # Save results
    print("\n6. Saving results...")

    # Save matrix as numpy
    matrix_path = output_dir / "discovery_matrix.npy"
    np.save(matrix_path, matrix)
    print(f"   Saved matrix to {matrix_path}")

    # Save labels
    labels_path = output_dir / "discovery_labels.json"
    with open(labels_path, 'w') as f:
        json.dump({
            'image_labels': image_labels,
            'audio_labels': audio_labels,
        }, f, indent=2)
    print(f"   Saved labels to {labels_path}")

    # Plot heatmap
    heatmap_path = output_dir / "discovery_heatmap.png"
    plot_discovery_heatmap(matrix, image_labels, audio_labels, str(heatmap_path))

    print("\n" + "="*60)
    print("PHASE 1 COMPLETE")
    print("="*60)

    return {
        'matrix': matrix,
        'image_labels': image_labels,
        'audio_labels': audio_labels,
    }
