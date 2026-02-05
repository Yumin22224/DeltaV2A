#!/usr/bin/env python
"""
Delta Correspondence Experiment Runner

Usage:
    # Extract deltas from data
    python scripts/run_experiment.py extract --config configs/experiment.yaml

    # Phase 1: Prototype similarity matrix
    python scripts/run_experiment.py phase1 --config configs/experiment.yaml

    # Phase 2: Retrieval evaluation
    python scripts/run_experiment.py phase2 --config configs/experiment.yaml

    # Full pipeline
    python scripts/run_experiment.py all --config configs/experiment.yaml
"""

import argparse
import sys
from pathlib import Path
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from src.models.embedder import ImageBindEmbedder
from src.experiment.delta_extraction import DeltaExtractor, DeltaDataset
from src.experiment.prototype import compute_prototypes, compute_effect_type_similarity
from src.experiment.retrieval import DeltaRetrieval
from src.experiment.statistics import (
    retrieval_permutation_test,
    trivial_confound_baseline,
    norm_monotonicity_analysis,
    cross_intensity_alignment,
)


def load_config(config_path: str) -> dict:
    """Load experiment configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_data_paths(config: dict) -> tuple:
    """
    Get image and audio paths from data directory.

    TODO: Implement based on your data structure.
    """
    data_dir = Path(config['data']['root_dir'])
    image_dir = data_dir / config['data']['image_subdir']
    audio_dir = data_dir / config['data']['audio_subdir']

    # Get image paths
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    image_paths = []
    if image_dir.exists():
        for ext in image_extensions:
            image_paths.extend(image_dir.glob(f'*{ext}'))
            image_paths.extend(image_dir.glob(f'**/*{ext}'))
    image_paths = [str(p) for p in sorted(image_paths)]

    # Get audio paths
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
    audio_paths = []
    if audio_dir.exists():
        for ext in audio_extensions:
            audio_paths.extend(audio_dir.glob(f'*{ext}'))
            audio_paths.extend(audio_dir.glob(f'**/*{ext}'))
    audio_paths = [str(p) for p in sorted(audio_paths)]

    return image_paths, audio_paths


def extract_deltas(config: dict):
    """Extract delta embeddings from data."""
    print("\n" + "="*60)
    print("DELTA EXTRACTION")
    print("="*60)

    image_paths, audio_paths = get_data_paths(config)

    if not image_paths:
        print(f"WARNING: No images found in {config['data']['root_dir']}/{config['data']['image_subdir']}")
    if not audio_paths:
        print(f"WARNING: No audio found in {config['data']['root_dir']}/{config['data']['audio_subdir']}")

    print(f"\nFound {len(image_paths)} images, {len(audio_paths)} audio files")

    if not image_paths and not audio_paths:
        print("ERROR: No data found. Please prepare your dataset first.")
        return

    # Initialize embedder
    device = config.get('device', 'cpu')
    print(f"\nLoading ImageBind on {device}...")
    embedder = ImageBindEmbedder(device=device)

    extractor = DeltaExtractor(embedder, device=device)

    output_dir = Path(config['output']['dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract image deltas
    if image_paths:
        print(f"\nExtracting image deltas...")
        image_dataset = extractor.extract_image_deltas(
            image_paths[:config['data'].get('max_images', 100)],
            effect_types=config['effects']['image']['types'],
            intensities=config['effects']['intensities'],
        )
        image_dataset.save(str(output_dir / 'image_deltas.json'))
        print(f"  Saved {len(image_dataset.deltas)} image deltas")

    # Extract audio deltas
    if audio_paths:
        print(f"\nExtracting audio deltas...")
        audio_dataset = extractor.extract_audio_deltas(
            audio_paths[:config['data'].get('max_audio', 100)],
            effect_types=config['effects']['audio']['types'],
            intensities=config['effects']['intensities'],
        )
        audio_dataset.save(str(output_dir / 'audio_deltas.json'))
        print(f"  Saved {len(audio_dataset.deltas)} audio deltas")

    print("\nDelta extraction complete!")


def run_phase1(config: dict):
    """Phase 1: Prototype similarity matrix analysis."""
    print("\n" + "="*60)
    print("PHASE 1: PROTOTYPE SIMILARITY MATRIX")
    print("="*60)

    output_dir = Path(config['output']['dir'])

    # Load deltas
    image_delta_path = output_dir / 'image_deltas.json'
    audio_delta_path = output_dir / 'audio_deltas.json'

    if not image_delta_path.exists() or not audio_delta_path.exists():
        print("ERROR: Delta files not found. Run 'extract' first.")
        return

    image_dataset = DeltaDataset.load(str(image_delta_path))
    audio_dataset = DeltaDataset.load(str(audio_delta_path))

    print(f"\nLoaded {len(image_dataset.deltas)} image deltas")
    print(f"Loaded {len(audio_dataset.deltas)} audio deltas")

    # Compute prototypes
    print("\nComputing prototypes...")
    image_protos = compute_prototypes(image_dataset, normalize_deltas=True)
    audio_protos = compute_prototypes(audio_dataset, normalize_deltas=True)

    # Compute effect-type similarity matrix
    print("\nComputing effect-type similarity matrix...")
    sim_matrix, img_effects, aud_effects = compute_effect_type_similarity(
        image_protos, audio_protos, aggregate="mean"
    )

    print(f"\nSimilarity matrix ({len(img_effects)} x {len(aud_effects)}):")
    print(f"  Image effects: {img_effects}")
    print(f"  Audio effects: {aud_effects}")
    print(f"\nMatrix:\n{sim_matrix}")

    # Save results
    results = {
        'similarity_matrix': sim_matrix.tolist(),
        'image_effects': img_effects,
        'audio_effects': aud_effects,
    }

    with open(output_dir / 'phase1_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)

    ax.set_xticks(range(len(aud_effects)))
    ax.set_yticks(range(len(img_effects)))
    ax.set_xticklabels(aud_effects, rotation=45, ha='right')
    ax.set_yticklabels(img_effects)

    ax.set_xlabel('Audio Effects')
    ax.set_ylabel('Image Effects')
    ax.set_title('Prototype Similarity Matrix (Image ↔ Audio)')

    plt.colorbar(im, ax=ax, label='Cosine Similarity')

    # Add values
    for i in range(len(img_effects)):
        for j in range(len(aud_effects)):
            ax.text(j, i, f'{sim_matrix[i, j]:.2f}',
                   ha='center', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'phase1_heatmap.png', dpi=150)
    print(f"\nSaved heatmap to {output_dir / 'phase1_heatmap.png'}")

    # Identify strong candidates
    print("\n" + "-"*40)
    print("STRONG CANDIDATE PAIRS (|sim| > 0.3):")
    print("-"*40)
    for i, img_e in enumerate(img_effects):
        for j, aud_e in enumerate(aud_effects):
            if abs(sim_matrix[i, j]) > 0.3:
                print(f"  {img_e} <-> {aud_e}: {sim_matrix[i, j]:.3f}")


def run_phase2(config: dict):
    """Phase 2: Retrieval evaluation and statistical tests."""
    print("\n" + "="*60)
    print("PHASE 2: RETRIEVAL EVALUATION")
    print("="*60)

    output_dir = Path(config['output']['dir'])

    # Load deltas
    image_dataset = DeltaDataset.load(str(output_dir / 'image_deltas.json'))
    audio_dataset = DeltaDataset.load(str(output_dir / 'audio_deltas.json'))

    # Effect type mapping (from config or default)
    effect_mapping = config.get('effect_mapping', {
        'blur': 'lpf',
        'brightness': 'highshelf',
        'contrast': 'saturation',
        'saturation': 'reverb',
    })

    print(f"\nEffect mapping: {effect_mapping}")

    # Initialize retrieval
    retrieval = DeltaRetrieval(image_dataset, audio_dataset, effect_mapping)

    # Evaluate
    print("\n--- Retrieval Metrics ---")
    metrics = retrieval.evaluate(k_values=[1, 3, 5, 10])
    print(f"Total queries: {metrics.total_queries}")
    print(f"Mean Reciprocal Rank (MRR): {metrics.mean_reciprocal_rank:.4f}")
    print(f"Mean Rank: {metrics.mean_rank:.2f}")
    for k, acc in metrics.top_k_accuracy.items():
        print(f"Top-{k} Accuracy: {acc:.4f}")

    # Evaluation by intensity
    print("\n--- Metrics by Intensity ---")
    by_intensity = retrieval.evaluate_by_intensity()
    for intensity, m in by_intensity.items():
        print(f"\n{intensity.upper()}:")
        print(f"  MRR: {m.mean_reciprocal_rank:.4f}")
        print(f"  Top-5 Accuracy: {m.top_k_accuracy.get(5, 0):.4f}")

    # Permutation test
    print("\n--- Permutation Test ---")
    perm_result = retrieval_permutation_test(
        image_dataset, audio_dataset, effect_mapping,
        metric="mrr",
        n_permutations=config.get('n_permutations', 1000),
    )
    print(f"Observed MRR: {perm_result.observed_statistic:.4f}")
    print(f"Null mean: {np.mean(perm_result.null_distribution):.4f}")
    print(f"Null std: {np.std(perm_result.null_distribution):.4f}")
    print(f"p-value: {perm_result.p_value:.4f}")
    print(f"Effect size: {perm_result.effect_size:.2f}")

    # Trivial confound baseline
    print("\n--- Trivial Confound Baseline ---")
    baseline = trivial_confound_baseline(
        image_dataset, audio_dataset, effect_mapping
    )
    print(f"Baseline MRR (using original embeddings): {baseline.mean_reciprocal_rank:.4f}")
    print(f"Baseline Top-5 Accuracy: {baseline.top_k_accuracy.get(5, 0):.4f}")

    # Norm monotonicity
    print("\n--- Norm Monotonicity Analysis ---")
    for modality in ['image', 'audio']:
        print(f"\n{modality.upper()} effects:")
        mono_results = norm_monotonicity_analysis(
            image_dataset if modality == 'image' else audio_dataset,
            modality
        )
        for effect, result in mono_results.items():
            status = "✓" if result.is_monotonic else "✗"
            print(f"  {effect}: rho={result.spearman_rho:.3f}, p={result.spearman_p_value:.3f} {status}")
            print(f"    norms: {result.intensity_to_norm}")

    # Cross-intensity alignment
    print("\n--- Cross-Intensity Alignment ---")
    alignment = cross_intensity_alignment(
        image_dataset, audio_dataset, effect_mapping
    )
    for pair, rho in alignment.items():
        print(f"  {pair}: Spearman rho = {rho:.3f}")

    # Save results
    results = {
        'metrics': {
            'mrr': metrics.mean_reciprocal_rank,
            'mean_rank': metrics.mean_rank,
            'top_k_accuracy': metrics.top_k_accuracy,
        },
        'permutation_test': {
            'observed': perm_result.observed_statistic,
            'p_value': perm_result.p_value,
            'effect_size': perm_result.effect_size,
        },
        'baseline_mrr': baseline.mean_reciprocal_rank,
    }

    with open(output_dir / 'phase2_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'phase2_results.json'}")


def main():
    parser = argparse.ArgumentParser(description="Delta Correspondence Experiment")
    parser.add_argument(
        'command',
        choices=['extract', 'phase1', 'phase2', 'all'],
        help="Command to run"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/experiment.yaml',
        help="Path to config file"
    )

    args = parser.parse_args()

    config = load_config(args.config)

    if args.command == 'extract':
        extract_deltas(config)
    elif args.command == 'phase1':
        run_phase1(config)
    elif args.command == 'phase2':
        run_phase2(config)
    elif args.command == 'all':
        extract_deltas(config)
        run_phase1(config)
        run_phase2(config)


if __name__ == '__main__':
    main()
