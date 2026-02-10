#!/usr/bin/env python
"""
Delta Correspondence Experiment Runner (Phase 0-3)

Complete pipeline using CLIP+CLAP with Text Anchor Ensemble and Decoder Learning.

Commands:
  extract         - Extract deltas using CLIP+CLAP
  sensitivity     - Phase 0-a: Sensitivity check
  linearity       - Phase 0-b: Linearity/consistency check
  fit_alignment   - Fit CCA on original embeddings
  phase1          - Phase 1: Discovery (Text Anchor Ensemble with 3-way similarity)
  phase3          - Phase 3: Learning (Train DSP parameter decoder)
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
    python scripts/run_experiment.py phase3 --config configs/experiment.yaml
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

# Model imports (CLAP import delayed to avoid argparse conflict)
from src.models import CLIPEmbedder, MultimodalEmbedder, CCAAlignment

# Experiment imports
from src.experiment.delta_extraction import DeltaExtractor, DeltaDataset
from src.experiment.sensitivity import sensitivity_check, print_sensitivity_report, get_insensitive_effects
from src.experiment.linearity import (
    linearity_analysis,
    cross_category_variance_check,
    print_linearity_report,
    get_inconsistent_effects,
)
from src.experiment.discovery import run_discovery
from src.experiment.phase3_dataset import load_phase3_dataset
from src.experiment.phase3_training import run_phase3_training


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
    # Import CLAP here to avoid argparse conflict
    # Save sys.argv and temporarily replace it to prevent CLAP's argparse from interfering
    import sys as _sys
    _saved_argv = _sys.argv[:]
    _sys.argv = [_sys.argv[0]]  # Keep only script name

    try:
        from src.models import CLAPEmbedder
    finally:
        _sys.argv = _saved_argv  # Restore original argv

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
        variance = cross_category_variance_check(image_dataset, cosine_threshold=cosine_threshold)
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
        variance = cross_category_variance_check(audio_dataset, cosine_threshold=cosine_threshold)
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
    """Phase 1: Discovery (Text Anchor Ensemble)."""
    output_dir = Path(config['output']['dir'])

    # Check delta files exist
    image_delta_path = output_dir / 'image_deltas.json'
    audio_delta_path = output_dir / 'audio_deltas.json'

    if not image_delta_path.exists() or not audio_delta_path.exists():
        print("ERROR: Delta files not found. Run 'extract' first.")
        return

    # Load embedders (needed for text embedding)
    print("\nLoading embedders for text anchor generation...")
    device = config['device']
    embedder = load_embedders(config)

    # Get categories/genres from config
    image_categories = config['data'].get('categories', {}).get('image')
    audio_genres = config['data'].get('categories', {}).get('audio')

    # Run discovery
    run_discovery(
        image_deltas_path=str(image_delta_path),
        audio_deltas_path=str(audio_delta_path),
        embedder=embedder,
        output_dir=str(output_dir),
        image_categories=image_categories,
        audio_genres=audio_genres,
    )


# =============================================================================
# COMMAND: PHASE3
# =============================================================================

def run_phase3(config: dict):
    """Phase 3: Learning (The Decoder)."""
    output_dir = Path(config['output']['dir'])

    # Check discovery results exist
    discovery_matrix_path = output_dir / 'discovery_matrix.npy'
    discovery_labels_path = output_dir / 'discovery_labels.json'

    if not discovery_matrix_path.exists() or not discovery_labels_path.exists():
        print("ERROR: Discovery results not found. Run 'phase1' first.")
        return

    # Get phase3 config
    phase3_config = config.get('phase3', {})
    batch_size = phase3_config.get('batch_size', 32)
    num_epochs = phase3_config.get('num_epochs', 100)
    learning_rate = phase3_config.get('learning_rate', 1e-4)
    val_split = phase3_config.get('val_split', 0.2)
    audio_duration = phase3_config.get('audio_duration', 10.0)
    max_files = phase3_config.get('max_audio_files', None)

    # Audio directory
    audio_dir = Path(config['data']['root_dir']) / config['data']['audio_subdir']

    # Load dataset
    print("\nLoading Phase 3 dataset...")
    dataset = load_phase3_dataset(
        audio_dir=str(audio_dir),
        discovery_matrix_path=str(discovery_matrix_path),
        discovery_labels_path=str(discovery_labels_path),
        sample_rate=config['model']['clap']['sample_rate'],
        duration=audio_duration,
        max_files=max_files,
    )

    # Load CLAP embedder
    print("\nLoading CLAP embedder...")
    from src.models.clap_embedder import CLAPEmbedder
    device = config['device']

    clap_embedder = CLAPEmbedder(
        model_id=config['model']['clap']['model_id'],
        enable_fusion=config['model']['clap']['enable_fusion'],
        max_duration=config['model']['clap']['max_duration'],
        device=device,
    )

    # Run training
    run_phase3_training(
        dataset=dataset,
        embedder=clap_embedder,
        output_dir=str(output_dir / 'phase3'),
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        val_split=val_split,
        device=device,
    )


# =============================================================================
# COMMAND: ALL
# =============================================================================

def run_all(config: dict):
    """Run full pipeline."""
    print("\n" + "="*80)
    print("  RUNNING FULL PIPELINE (Phase 0-3)")
    print("="*80)

    extract_deltas(config)
    run_sensitivity(config)
    run_linearity(config)
    run_phase1(config)
    run_phase3(config)

    print("\n" + "="*80)
    print("  FULL PIPELINE COMPLETE!")
    print("="*80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Delta Correspondence Experiment (Phase 0-3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        'command',
        choices=['extract', 'sensitivity', 'linearity', 'fit_alignment', 'phase1', 'phase3', 'all'],
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
        elif args.command == 'phase3':
            run_phase3(config)
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
