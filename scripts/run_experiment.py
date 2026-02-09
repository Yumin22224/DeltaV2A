#!/usr/bin/env python
"""
Delta Correspondence Experiment Runner (Phase 0-2)

Complete pipeline using CLIP+CLAP with CCA alignment.

Commands:
  extract         - Extract deltas using CLIP+CLAP
  sensitivity     - Phase 0-a: Sensitivity check
  linearity       - Phase 0-b: Linearity/consistency check
  fit_alignment   - Phase 2-a: Fit CCA on original embeddings
  phase1          - Phase 3-a: Discovery (prototype similarity heatmap)
  phase2          - Phase 3-b: Statistical validation (retrieval, permutation, Spearman, norm)
  all             - Run full pipeline

Usage:
    # Full pipeline
    python scripts/run_experiment.py all --config configs/experiment.yaml

    # Individual stages
    python scripts/run_experiment.py extract --config configs/experiment.yaml
    python scripts/run_experiment.py sensitivity --config configs/experiment.yaml
    python scripts/run_experiment.py linearity --config configs/experiment.yaml
    python scripts/run_experiment.py fit_alignment --config configs/experiment.yaml
    python scripts/run_experiment.py phase1 --config configs/experiment.yaml
    python scripts/run_experiment.py phase2 --config configs/experiment.yaml
"""

import argparse
import sys
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

# Model imports
from src.models import CLIPEmbedder, CLAPEmbedder, MultimodalEmbedder, CCAAlignment

# Experiment imports
from src.experiment.delta_extraction import DeltaExtractor, DeltaDataset
from src.experiment.sensitivity import sensitivity_check, print_sensitivity_report, get_insensitive_effects
from src.experiment.linearity import (
    linearity_analysis,
    cross_category_variance_check,
    print_linearity_report,
    get_inconsistent_effects,
)
from src.experiment.prototype import compute_prototypes, compute_similarity_matrix, compute_effect_type_similarity
from src.experiment.retrieval import DeltaRetrieval
from src.experiment.statistics import (
    retrieval_permutation_test,
    norm_monotonicity_analysis,
    spearman_intensity_correlation,
    norm_intensity_analysis,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML file."""
    print(f"Loading config from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_data_paths(config: dict) -> Tuple[List[str], List[str]]:
    """
    Get image and audio paths from data directory.

    Supports category filtering:
    - Image categories: folder names under images/
    - Audio categories: currently treats all as single category

    Returns:
        image_paths: List of image file paths
        audio_paths: List of audio file paths
    """
    data_dir = Path(config['data']['root_dir'])
    image_dir = data_dir / config['data']['image_subdir']
    audio_dir = data_dir / config['data']['audio_subdir']

    # Get category filters
    categories_config = config['data'].get('categories', {})
    image_categories = categories_config.get('image', None) if categories_config else None
    audio_categories = categories_config.get('audio', None) if categories_config else None

    # Get image paths
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    image_paths = []
    if image_dir.exists():
        if image_categories is not None and len(image_categories) > 0:
            # Filter by specific categories (folders)
            print(f"  Filtering images by categories: {image_categories}")
            for category in image_categories:
                category_dir = image_dir / category
                if category_dir.exists() and category_dir.is_dir():
                    for ext in image_extensions:
                        image_paths.extend(category_dir.glob(f'*{ext}'))
                else:
                    print(f"    WARNING: Category '{category}' not found in {image_dir}")
        else:
            # Get all images
            for ext in image_extensions:
                image_paths.extend(image_dir.glob(f'*{ext}'))
                image_paths.extend(image_dir.glob(f'**/*{ext}'))
    image_paths = [str(p) for p in sorted(set(image_paths))]

    # Get audio paths
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    audio_paths = []
    if audio_dir.exists():
        if audio_categories is not None and len(audio_categories) > 0:
            # Filter by specific categories (folders if exist, or filename patterns)
            print(f"  Filtering audio by categories: {audio_categories}")
            for category in audio_categories:
                category_dir = audio_dir / category
                if category_dir.exists() and category_dir.is_dir():
                    # Category is a folder
                    for ext in audio_extensions:
                        audio_paths.extend(category_dir.glob(f'*{ext}'))
                else:
                    # Try filename pattern matching
                    for ext in audio_extensions:
                        for path in audio_dir.glob(f'*{ext}'):
                            if category.lower() in path.name.lower():
                                audio_paths.append(path)
        else:
            # Get all audio files
            for ext in audio_extensions:
                audio_paths.extend(audio_dir.glob(f'*{ext}'))
                audio_paths.extend(audio_dir.glob(f'**/*{ext}'))
    audio_paths = [str(p) for p in sorted(set(audio_paths))]

    # Apply limits if specified
    max_images = config['data'].get('max_images', None)
    max_audio = config['data'].get('max_audio', None)

    if max_images and len(image_paths) > max_images:
        image_paths = image_paths[:max_images]
    if max_audio and len(audio_paths) > max_audio:
        audio_paths = audio_paths[:max_audio]

    return image_paths, audio_paths


def load_embedders(config: dict) -> MultimodalEmbedder:
    """
    Load CLIP and CLAP embedders.

    Args:
        config: Experiment configuration

    Returns:
        MultimodalEmbedder instance
    """
    device = config.get('device', 'cpu')
    print(f"\nInitializing embedders on {device}...")

    # Initialize CLIP
    clip_config = config['model']['clip']
    print(f"  Loading CLIP ({clip_config['name']}/{clip_config['pretrained']})...")
    clip_embedder = CLIPEmbedder(
        model_name=clip_config['name'],
        pretrained=clip_config['pretrained'],
        device=device,
    )

    # Initialize CLAP
    clap_config = config['model']['clap']
    print(f"  Loading CLAP (model_id={clap_config['model_id']}, sample_rate=48000Hz, max_duration={clap_config['max_duration']}s)...")
    clap_embedder = CLAPEmbedder(
        model_id=clap_config['model_id'],
        enable_fusion=clap_config['enable_fusion'],
        max_duration=clap_config['max_duration'],
        device=device,
    )

    # Create multimodal embedder
    embedder = MultimodalEmbedder(clip_embedder, clap_embedder)
    print("  Embedders loaded successfully!")

    return embedder


# =============================================================================
# COMMAND: EXTRACT
# =============================================================================

def extract_deltas(config: dict):
    """Extract delta embeddings from image and audio data."""
    print("\n" + "="*80)
    print("  DELTA EXTRACTION")
    print("="*80)

    # Get data paths
    image_paths, audio_paths = get_data_paths(config)

    if not image_paths:
        print(f"WARNING: No images found in {config['data']['root_dir']}/{config['data']['image_subdir']}")
    if not audio_paths:
        print(f"WARNING: No audio found in {config['data']['root_dir']}/{config['data']['audio_subdir']}")

    print(f"\nFound {len(image_paths)} images, {len(audio_paths)} audio files")

    if not image_paths and not audio_paths:
        print("ERROR: No data found. Please prepare your dataset first.")
        return

    # Load embedders
    embedder = load_embedders(config)

    # Initialize extractor
    extractor = DeltaExtractor(embedder, device=config.get('device', 'cpu'))

    # Create output directory
    output_dir = Path(config['output']['dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if augmented files should be saved
    save_augmented = config['data'].get('save_augmented', False)
    augmented_dir = config['data'].get('augmented_dir', 'outputs/augmented') if save_augmented else None

    if save_augmented:
        print(f"\n⚠️  Augmented files will be saved to: {augmented_dir}")
        Path(augmented_dir).mkdir(parents=True, exist_ok=True)

    # Extract image deltas
    if image_paths:
        print(f"\nExtracting image deltas...")
        print(f"  Effects: {config['effects']['image']['types']}")
        print(f"  Intensities: {config['effects']['intensities']}")
        if save_augmented:
            print(f"  Saving augmented images to: {augmented_dir}/images/")

        image_dataset = extractor.extract_image_deltas(
            image_paths,
            effect_types=config['effects']['image']['types'],
            intensities=config['effects']['intensities'],
            save_augmented=save_augmented,
            augmented_dir=augmented_dir,
        )

        image_save_path = output_dir / 'image_deltas.json'
        image_dataset.save(str(image_save_path))
        print(f"  Saved {len(image_dataset.deltas)} image deltas to {image_save_path}")

    # Extract audio deltas
    if audio_paths:
        print(f"\nExtracting audio deltas...")
        print(f"  Effects: {config['effects']['audio']['types']}")
        print(f"  Intensities: {config['effects']['intensities']}")
        if save_augmented:
            print(f"  Saving augmented audio to: {augmented_dir}/audio/")

        audio_dataset = extractor.extract_audio_deltas(
            audio_paths,
            effect_types=config['effects']['audio']['types'],
            intensities=config['effects']['intensities'],
            sample_rate=embedder.audio_sample_rate,  # Use CLAP's sample rate (48000)
            save_augmented=save_augmented,
            augmented_dir=augmented_dir,
        )

        audio_save_path = output_dir / 'audio_deltas.json'
        audio_dataset.save(str(audio_save_path))
        print(f"  Saved {len(audio_dataset.deltas)} audio deltas to {audio_save_path}")

    print("\n" + "="*80)
    print("  Delta extraction complete!")
    print("="*80)


# =============================================================================
# COMMAND: SENSITIVITY
# =============================================================================

def run_sensitivity(config: dict):
    """Phase 0-a: Sensitivity check."""
    print("\n" + "="*80)
    print("  PHASE 0-a: SENSITIVITY CHECK")
    print("="*80)

    output_dir = Path(config['output']['dir'])
    threshold = config['thresholds']['sensitivity']['min_distance']

    results_all = []

    # Check image deltas
    image_delta_path = output_dir / 'image_deltas.json'
    if image_delta_path.exists():
        print(f"\nLoading image deltas from {image_delta_path}...")
        image_dataset = DeltaDataset.load(str(image_delta_path))
        print(f"  Loaded {len(image_dataset.deltas)} image deltas")

        results = sensitivity_check(image_dataset, threshold=threshold)
        results_all.extend(results)
    else:
        print(f"\nWARNING: Image deltas not found at {image_delta_path}")

    # Check audio deltas
    audio_delta_path = output_dir / 'audio_deltas.json'
    if audio_delta_path.exists():
        print(f"\nLoading audio deltas from {audio_delta_path}...")
        audio_dataset = DeltaDataset.load(str(audio_delta_path))
        print(f"  Loaded {len(audio_dataset.deltas)} audio deltas")

        results = sensitivity_check(audio_dataset, threshold=threshold)
        results_all.extend(results)
    else:
        print(f"\nWARNING: Audio deltas not found at {audio_delta_path}")

    if not results_all:
        print("\nERROR: No deltas found. Run 'extract' first.")
        return

    # Print report
    print_sensitivity_report(results_all)

    # Save results
    save_path = output_dir / 'sensitivity_results.json'
    results_dict = {
        'threshold': threshold,
        'results': [
            {
                'modality': r.modality,
                'effect_type': r.effect_type,
                'intensity': r.intensity,
                'mean_distance': r.mean_distance,
                'std_distance': r.std_distance,
                'min_distance': r.min_distance,
                'max_distance': r.max_distance,
                'num_samples': r.num_samples,
                'is_sensitive': r.is_sensitive,
            }
            for r in results_all
        ],
    }

    with open(save_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to {save_path}")

    # Flag insensitive effects
    insensitive = get_insensitive_effects(results_all)
    if insensitive:
        print("\n" + "-"*80)
        print("  WARNING: Insensitive effects detected (exclude from analysis):")
        print("-"*80)
        for modality, effects in insensitive.items():
            print(f"\n{modality.upper()}:")
            for effect_type, intensity in effects:
                print(f"  - {effect_type}/{intensity}")


# =============================================================================
# COMMAND: LINEARITY
# =============================================================================

def run_linearity(config: dict):
    """Phase 0-b: Linearity and consistency check."""
    print("\n" + "="*80)
    print("  PHASE 0-b: LINEARITY & CONSISTENCY CHECK")
    print("="*80)

    output_dir = Path(config['output']['dir'])
    cosine_threshold = config['thresholds']['linearity']['min_cosine']
    variance_threshold = config['thresholds']['linearity']['max_variance']

    results_all = []
    variance_all = []

    # Check image deltas
    image_delta_path = output_dir / 'image_deltas.json'
    if image_delta_path.exists():
        print(f"\nLoading image deltas from {image_delta_path}...")
        image_dataset = DeltaDataset.load(str(image_delta_path))
        print(f"  Loaded {len(image_dataset.deltas)} image deltas")

        print("\nAnalyzing linearity...")
        results = linearity_analysis(image_dataset, cosine_threshold=cosine_threshold)
        results_all.extend(results)

        print("Checking cross-category variance...")
        variance = cross_category_variance_check(results, variance_threshold=variance_threshold)
        variance_all.extend(variance)
    else:
        print(f"\nWARNING: Image deltas not found at {image_delta_path}")

    # Check audio deltas
    audio_delta_path = output_dir / 'audio_deltas.json'
    if audio_delta_path.exists():
        print(f"\nLoading audio deltas from {audio_delta_path}...")
        audio_dataset = DeltaDataset.load(str(audio_delta_path))
        print(f"  Loaded {len(audio_dataset.deltas)} audio deltas")

        print("\nAnalyzing linearity...")
        results = linearity_analysis(audio_dataset, cosine_threshold=cosine_threshold)
        results_all.extend(results)

        print("Checking cross-category variance...")
        variance = cross_category_variance_check(results, variance_threshold=variance_threshold)
        variance_all.extend(variance)
    else:
        print(f"\nWARNING: Audio deltas not found at {audio_delta_path}")

    if not results_all:
        print("\nERROR: No deltas found. Run 'extract' first.")
        return

    # Print report
    print_linearity_report(results_all, variance_all)

    # Save results
    save_path = output_dir / 'linearity_results.json'
    results_dict = {
        'thresholds': {
            'cosine': cosine_threshold,
            'variance': variance_threshold,
        },
        'linearity': [
            {
                'modality': r.modality,
                'effect_type': r.effect_type,
                'intensity': r.intensity,
                'category': r.category,
                'mean_pairwise_cosine': r.mean_pairwise_cosine,
                'std_pairwise_cosine': r.std_pairwise_cosine,
                'mean_norm': r.mean_norm,
                'cv_norm': r.cv_norm,
                'num_samples': r.num_samples,
                'is_consistent': r.is_consistent,
            }
            for r in results_all
        ],
        'cross_category_variance': [
            {
                'modality': v.modality,
                'effect_type': v.effect_type,
                'intensity': v.intensity,
                'variance_cosine': v.variance_cosine,
                'mean_cosine': v.mean_cosine,
                'is_context_invariant': v.is_context_invariant,
            }
            for v in variance_all
        ],
    }

    with open(save_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to {save_path}")

    # Flag inconsistent effects
    inconsistent = get_inconsistent_effects(variance_all)
    if inconsistent:
        print("\n" + "-"*80)
        print("  WARNING: Context-dependent effects detected:")
        print("-"*80)
        for modality, effects in inconsistent.items():
            print(f"\n{modality.upper()}:")
            for effect_type, intensity in effects:
                print(f"  - {effect_type}/{intensity}")


# =============================================================================
# COMMAND: FIT_ALIGNMENT
# =============================================================================

def fit_alignment(config: dict):
    """Phase 2-a: Fit CCA alignment on original embeddings."""
    print("\n" + "="*80)
    print("  PHASE 2-a: FIT CCA ALIGNMENT")
    print("="*80)

    output_dir = Path(config['output']['dir'])

    # Load deltas to get original embeddings
    image_delta_path = output_dir / 'image_deltas.json'
    audio_delta_path = output_dir / 'audio_deltas.json'

    if not image_delta_path.exists() or not audio_delta_path.exists():
        print("ERROR: Delta files not found. Run 'extract' first.")
        return

    print(f"\nLoading delta datasets...")
    image_dataset = DeltaDataset.load(str(image_delta_path))
    audio_dataset = DeltaDataset.load(str(audio_delta_path))
    print(f"  Image deltas: {len(image_dataset.deltas)}")
    print(f"  Audio deltas: {len(audio_dataset.deltas)}")

    # Extract original embeddings (e_0)
    print("\nExtracting original embeddings...")
    image_originals = {}
    for delta in image_dataset.deltas:
        path = delta.original_path
        if path not in image_originals:
            image_originals[path] = delta.original_embedding

    audio_originals = {}
    for delta in audio_dataset.deltas:
        path = delta.original_path
        if path not in audio_originals:
            audio_originals[path] = delta.original_embedding

    # Stack embeddings
    image_embeds = np.stack(list(image_originals.values()))
    audio_embeds = np.stack(list(audio_originals.values()))

    print(f"  Unique image embeddings: {image_embeds.shape}")
    print(f"  Unique audio embeddings: {audio_embeds.shape}")

    # Handle different sample sizes
    n_samples = min(len(image_embeds), len(audio_embeds))
    if len(image_embeds) != len(audio_embeds):
        print(f"\nWARNING: Different number of samples. Using first {n_samples} from each.")
        image_embeds = image_embeds[:n_samples]
        audio_embeds = audio_embeds[:n_samples]

    # Fit CCA
    n_components = config['model']['alignment']['n_components']
    print(f"\nFitting CCA with {n_components} components...")

    alignment = CCAAlignment(n_components=n_components)
    alignment.fit(image_embeds, audio_embeds)

    # Save alignment
    save_path = output_dir / 'cca_alignment.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(alignment, f)
    print(f"\nCCA alignment saved to {save_path}")

    print("\n" + "="*80)
    print("  CCA alignment complete!")
    print("="*80)


# =============================================================================
# COMMAND: PHASE1
# =============================================================================

def run_phase1(config: dict):
    """Phase 3-a: Discovery (prototype similarity heatmap)."""
    print("\n" + "="*80)
    print("  PHASE 3-a: DISCOVERY (Prototype Similarity)")
    print("="*80)

    output_dir = Path(config['output']['dir'])

    # Load deltas
    image_delta_path = output_dir / 'image_deltas.json'
    audio_delta_path = output_dir / 'audio_deltas.json'

    if not image_delta_path.exists() or not audio_delta_path.exists():
        print("ERROR: Delta files not found. Run 'extract' first.")
        return

    print(f"\nLoading delta datasets...")
    image_dataset = DeltaDataset.load(str(image_delta_path))
    audio_dataset = DeltaDataset.load(str(audio_delta_path))
    print(f"  Image deltas: {len(image_dataset.deltas)}")
    print(f"  Audio deltas: {len(audio_dataset.deltas)}")

    # Load CCA alignment
    alignment_path = output_dir / 'cca_alignment.pkl'
    if not alignment_path.exists():
        print(f"ERROR: CCA alignment not found at {alignment_path}. Run 'fit_alignment' first.")
        return

    print(f"\nLoading CCA alignment from {alignment_path}...")
    with open(alignment_path, 'rb') as f:
        alignment = pickle.load(f)

    # Compute prototypes
    print("\nComputing prototypes...")
    image_prototypes = compute_prototypes(image_dataset, normalize_deltas=True)
    audio_prototypes = compute_prototypes(audio_dataset, normalize_deltas=True)
    print(f"  Image prototypes: {len(image_prototypes.prototypes)}")
    print(f"  Audio prototypes: {len(audio_prototypes.prototypes)}")

    # Compute effect-type similarity matrix
    print("\nComputing effect-type similarity matrix...")
    sim_matrix, img_effects, aud_effects = compute_effect_type_similarity(
        image_prototypes,
        audio_prototypes,
        alignment,
        aggregate="mean",
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

    save_path = output_dir / 'phase1_results.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {save_path}")

    # Plot heatmap
    if config['output'].get('save_plots', True):
        print("\nGenerating heatmap...")
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(sim_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)

        ax.set_xticks(range(len(aud_effects)))
        ax.set_yticks(range(len(img_effects)))
        ax.set_xticklabels(aud_effects, rotation=45, ha='right')
        ax.set_yticklabels(img_effects)

        ax.set_xlabel('Audio Effects', fontsize=12)
        ax.set_ylabel('Image Effects', fontsize=12)
        ax.set_title('Prototype Similarity Matrix (Image ↔ Audio)', fontsize=14)

        plt.colorbar(im, ax=ax, label='Cosine Similarity')

        # Add values
        for i in range(len(img_effects)):
            for j in range(len(aud_effects)):
                text = ax.text(j, i, f'{sim_matrix[i, j]:.2f}',
                             ha='center', va='center', fontsize=10,
                             color='white' if abs(sim_matrix[i, j]) > 0.5 else 'black')

        plt.tight_layout()

        plot_path = output_dir / 'phase1_heatmap.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Heatmap saved to {plot_path}")
        plt.close()

    # Identify strong candidate pairs
    print("\n" + "-"*80)
    print("  STRONG CANDIDATE PAIRS (|sim| > 0.3):")
    print("-"*80)
    candidates = []
    for i, img_e in enumerate(img_effects):
        for j, aud_e in enumerate(aud_effects):
            if abs(sim_matrix[i, j]) > 0.3:
                candidates.append((img_e, aud_e, sim_matrix[i, j]))
                print(f"  {img_e} <-> {aud_e}: {sim_matrix[i, j]:.3f}")

    if not candidates:
        print("  (None found)")


# =============================================================================
# COMMAND: PHASE2
# =============================================================================

def run_phase2(config: dict):
    """Phase 3-b: Statistical validation (retrieval, permutation, Spearman, norm)."""
    print("\n" + "="*80)
    print("  PHASE 3-b: STATISTICAL VALIDATION")
    print("="*80)

    output_dir = Path(config['output']['dir'])

    # Load deltas
    image_delta_path = output_dir / 'image_deltas.json'
    audio_delta_path = output_dir / 'audio_deltas.json'

    if not image_delta_path.exists() or not audio_delta_path.exists():
        print("ERROR: Delta files not found. Run 'extract' first.")
        return

    print(f"\nLoading delta datasets...")
    image_dataset = DeltaDataset.load(str(image_delta_path))
    audio_dataset = DeltaDataset.load(str(audio_delta_path))
    print(f"  Image deltas: {len(image_dataset.deltas)}")
    print(f"  Audio deltas: {len(audio_dataset.deltas)}")

    # Load CCA alignment
    alignment_path = output_dir / 'cca_alignment.pkl'
    if not alignment_path.exists():
        print(f"ERROR: CCA alignment not found. Run 'fit_alignment' first.")
        return

    print(f"\nLoading CCA alignment...")
    with open(alignment_path, 'rb') as f:
        alignment = pickle.load(f)

    # Get effect mapping
    effect_mapping = config.get('effect_mapping', {})
    print(f"\nEffect mapping (hypothesis): {effect_mapping}")

    # Initialize retrieval
    print("\nInitializing retrieval system...")
    retrieval = DeltaRetrieval(
        image_dataset,
        audio_dataset,
        effect_mapping,
        alignment=alignment,
    )

    # === Retrieval Metrics ===
    print("\n" + "-"*80)
    print("  RETRIEVAL METRICS")
    print("-"*80)

    metrics = retrieval.evaluate(k_values=[1, 3, 5, 10])
    print(f"\nTotal queries: {metrics.total_queries}")
    print(f"Mean Reciprocal Rank (MRR): {metrics.mean_reciprocal_rank:.4f}")
    print(f"Mean Rank: {metrics.mean_rank:.2f}")
    for k, acc in metrics.top_k_accuracy.items():
        print(f"Top-{k} Accuracy: {acc:.4f}")

    # Metrics by intensity
    print("\n--- By Intensity ---")
    by_intensity = retrieval.evaluate_by_intensity()
    for intensity, m in by_intensity.items():
        print(f"\n{intensity.upper()}:")
        print(f"  MRR: {m.mean_reciprocal_rank:.4f}")
        print(f"  Top-5 Accuracy: {m.top_k_accuracy.get(5, 0):.4f}")

    # === Permutation Test ===
    print("\n" + "-"*80)
    print("  PERMUTATION TEST")
    print("-"*80)

    n_permutations = config.get('n_permutations', 1000)
    print(f"\nRunning permutation test ({n_permutations} permutations)...")

    perm_result = retrieval_permutation_test(
        image_dataset,
        audio_dataset,
        effect_mapping,
        metric="mrr",
        n_permutations=n_permutations,
        alignment=alignment,
    )

    print(f"\nObserved MRR: {perm_result.observed_statistic:.4f}")
    print(f"Null mean: {np.mean(perm_result.null_distribution):.4f}")
    print(f"Null std: {np.std(perm_result.null_distribution):.4f}")
    print(f"p-value: {perm_result.p_value:.4f}")
    print(f"Effect size (Cohen's d): {perm_result.effect_size:.2f}")

    if perm_result.p_value < 0.05:
        print("  ✓ Result is statistically significant (p < 0.05)")
    else:
        print("  ✗ Result is NOT statistically significant (p >= 0.05)")

    # === Norm Monotonicity ===
    print("\n" + "-"*80)
    print("  NORM MONOTONICITY ANALYSIS")
    print("-"*80)

    for modality, dataset in [("image", image_dataset), ("audio", audio_dataset)]:
        print(f"\n{modality.upper()} effects:")
        mono_results = norm_monotonicity_analysis(dataset, modality)

        for effect, result in mono_results.items():
            status = "✓" if result.is_monotonic else "✗"
            print(f"  {effect}: ρ={result.spearman_rho:.3f}, p={result.spearman_p_value:.3f} {status}")
            print(f"    Norms by intensity: {result.intensity_to_norm}")

    # === Spearman Correlation (intensity vs norm) ===
    print("\n" + "-"*80)
    print("  SPEARMAN CORRELATION (Intensity vs Norm)")
    print("-"*80)

    for modality, dataset in [("image", image_dataset), ("audio", audio_dataset)]:
        print(f"\n{modality.upper()}:")
        spearman_results = spearman_intensity_correlation(dataset)

        for effect_type, (rho, p_value) in spearman_results.items():
            status = "✓ Sig" if p_value < 0.05 else "✗ NS"
            print(f"  {effect_type}: ρ={rho:.3f}, p={p_value:.3f} {status}")

    # === Norm-Intensity Analysis ===
    print("\n" + "-"*80)
    print("  NORM-INTENSITY ANALYSIS")
    print("-"*80)

    for modality, dataset in [("image", image_dataset), ("audio", audio_dataset)]:
        print(f"\n{modality.upper()}:")
        norm_analysis = norm_intensity_analysis(dataset)

        for effect_type, stats in norm_analysis.items():
            print(f"\n  {effect_type}:")
            for intensity, (mean, std) in stats.items():
                print(f"    {intensity}: {mean:.4f} ± {std:.4f}")

    # === Save Results ===
    print("\n" + "-"*80)
    print("  SAVING RESULTS")
    print("-"*80)

    results = {
        'retrieval_metrics': {
            'mrr': metrics.mean_reciprocal_rank,
            'mean_rank': metrics.mean_rank,
            'top_k_accuracy': metrics.top_k_accuracy,
            'total_queries': metrics.total_queries,
        },
        'retrieval_by_intensity': {
            intensity: {
                'mrr': m.mean_reciprocal_rank,
                'top_k_accuracy': m.top_k_accuracy,
            }
            for intensity, m in by_intensity.items()
        },
        'permutation_test': {
            'observed_mrr': perm_result.observed_statistic,
            'null_mean': float(np.mean(perm_result.null_distribution)),
            'null_std': float(np.std(perm_result.null_distribution)),
            'p_value': perm_result.p_value,
            'effect_size': perm_result.effect_size,
            'n_permutations': n_permutations,
        },
    }

    save_path = output_dir / 'phase2_results.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {save_path}")

    print("\n" + "="*80)
    print("  Phase 2 complete!")
    print("="*80)


# =============================================================================
# COMMAND: ALL
# =============================================================================

def run_all(config: dict):
    """Run full pipeline."""
    print("\n" + "="*80)
    print("  RUNNING FULL PIPELINE (Phase 0-2)")
    print("="*80)

    extract_deltas(config)
    run_sensitivity(config)
    run_linearity(config)
    fit_alignment(config)
    run_phase1(config)
    run_phase2(config)

    print("\n" + "="*80)
    print("  FULL PIPELINE COMPLETE!")
    print("="*80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Delta Correspondence Experiment (Phase 0-2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        'command',
        choices=['extract', 'sensitivity', 'linearity', 'fit_alignment', 'phase1', 'phase2', 'all'],
        help="Command to run",
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/experiment.yaml',
        help="Path to configuration file (default: configs/experiment.yaml)",
    )

    args = parser.parse_args()

    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        sys.exit(1)

    # Run command
    try:
        if args.command == 'extract':
            extract_deltas(config)
        elif args.command == 'sensitivity':
            run_sensitivity(config)
        elif args.command == 'linearity':
            run_linearity(config)
        elif args.command == 'fit_alignment':
            fit_alignment(config)
        elif args.command == 'phase1':
            run_phase1(config)
        elif args.command == 'phase2':
            run_phase2(config)
        elif args.command == 'all':
            run_all(config)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
