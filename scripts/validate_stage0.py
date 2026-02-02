"""
Stage 0 Validation Script

Validates Prior estimation on high-similarity pairs
Tests:
1. C_prior estimation quality
2. Hard/Soft prior properties
3. Entropy and sparsity constraints

Based on System Specification v2.2, Section 3
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

# Suppress ObjC class conflict warnings on macOS (cv2 vs av dylib collision)
# This must be set before importing cv2 or av
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

# Suppress known harmless Python warnings
# - pkg_resources deprecation from imagebind/data.py (upstream issue)
# - open_clip QuickGELU mismatch (cosmetic, does not affect output)
# - pyloudnorm clipping warning (expected for some audio normalization)
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*QuickGELU mismatch.*")
warnings.filterwarnings("ignore", message=".*Possible clipped samples.*")

import torch
import torch.nn as nn
from omegaconf import OmegaConf
import wandb
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import HardPrior, SoftPrior, PriorEstimator
from src.data import PriorDataset, get_audio_transform, get_image_transform
from src.utils.prior_utils import (
    validate_c_prior,
    test_prior_estimator,
    visualize_coupling,
    visualize_hard_soft_comparison,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate Stage 0 Prior")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage0_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--test-pairs",
        type=str,
        default=None,
        help="Path to test image-audio pairs directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to validate",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations",
    )
    return parser.parse_args()


def build_prior_estimator(config, device):
    """
    Build PriorEstimator from config

    Returns:
        prior_estimator: PriorEstimator model
    """
    print("\nBuilding Prior Estimator...")

    # 1. Hard Prior
    print("  Initializing Hard Prior...")
    hard_prior = HardPrior(
        rules=config.hard_prior.rules,
        num_heads=config.model.num_heads,
    )
    print(f"    {len(hard_prior.rules)} rules loaded")

    # 2. Soft Prior
    print("  Initializing Soft Prior...")
    soft_prior = SoftPrior(
        model_name=config.soft_prior.model_name,
        num_heads=config.model.num_heads,
        head_dim=config.model.head_dim,
        freeze=config.soft_prior.freeze,
    ).to(device)

    # Initialize head queries
    if config.soft_prior.head_queries.init_method == "clip_text":
        soft_prior.initialize_head_queries(
            config.soft_prior.head_queries.text_prompts
        )
    print("    Head queries initialized")

    # 3. Prior Estimator
    print("  Combining Hard and Soft Priors...")
    prior_estimator = PriorEstimator(
        hard_prior=hard_prior,
        soft_prior=soft_prior,
        alpha=config.model.prior.alpha,
        entropy_min=config.model.prior.entropy_min,
        sparsity_max=config.model.prior.sparsity_max,
    ).to(device)

    print(f"  Prior Estimator ready (α={config.model.prior.alpha})")

    return prior_estimator


def validate_dataset(args, config, prior_estimator):
    """
    Validate on Prior dataset
    """
    print(f"\n{'='*60}")
    print("Validating on Prior Dataset")
    print(f"{'='*60}")

    device = torch.device(args.device)

    # Create dataset
    audio_transform = get_audio_transform(config.data.audio)
    image_transform = get_image_transform(config.data.image)

    dataset = PriorDataset(
        data_dir=config.data.dataset_path,
        audio_transform=audio_transform,
        image_transform=image_transform,
    )

    print(f"Dataset size: {len(dataset)}")

    # Sample validation
    num_samples = min(args.num_samples, len(dataset))
    indices = torch.randperm(len(dataset))[:num_samples]

    print(f"Validating {num_samples} samples...\n")

    all_validations = []
    all_c_priors = []

    prior_estimator.eval()

    with torch.no_grad():
        for i, idx in enumerate(tqdm(indices, desc="Validating")):
            sample = dataset[int(idx)]

            image = sample['image'].unsqueeze(0).to(device)
            audio_mel = sample['audio_mel'].unsqueeze(0).to(device)

            # Estimate C_prior
            C_prior = prior_estimator(image, audio_mel)

            # Validate
            validation = validate_c_prior(
                C_prior,
                entropy_min=config.model.prior.entropy_min,
                sparsity_max=config.model.prior.sparsity_max,
            )

            all_validations.append(validation)
            all_c_priors.append(C_prior)

            # Visualize first few samples
            if args.visualize and i < 5:
                save_dir = Path(config.output.output_path) / "visualizations"
                save_dir.mkdir(parents=True, exist_ok=True)

                visualize_coupling(
                    C_prior[0],
                    save_path=str(save_dir / f"sample_{i:03d}_coupling.png"),
                    title=f"Sample {i} Coupling Distribution"
                )

    # Summary statistics
    print(f"\n{'='*60}")
    print("Validation Summary")
    print(f"{'='*60}")

    passed_count = sum(v['passed'] for v in all_validations)
    pass_rate = passed_count / len(all_validations) * 100

    print(f"Pass rate: {passed_count}/{len(all_validations)} ({pass_rate:.1f}%)")

    avg_entropy = sum(v['entropy_mean'] for v in all_validations) / len(all_validations)
    avg_sparsity = sum(v['sparsity_mean'] for v in all_validations) / len(all_validations)

    print(f"Average entropy: {avg_entropy:.3f} (min: {config.model.prior.entropy_min})")
    print(f"Average sparsity: {avg_sparsity:.3f} (max: {config.model.prior.sparsity_max})")

    # Head-wise statistics
    print(f"\nHead-wise coupling statistics:")
    head_names = ['rhythm', 'harmony', 'energy', 'timbre', 'space', 'texture']

    all_c_stacked = torch.cat(all_c_priors, dim=0)  # (N, N_v, 6)
    head_means = all_c_stacked.mean(dim=(0, 1))  # (6,)
    head_stds = all_c_stacked.std(dim=(0, 1))  # (6,)

    for h, name in enumerate(head_names):
        print(f"  {name:10s}: {head_means[h]:.3f} ± {head_stds[h]:.3f}")

    print(f"{'='*60}\n")

    # Check if validation passed
    if pass_rate < 95.0:
        print("⚠️  WARNING: Pass rate < 95%, Prior may need tuning")
    else:
        print("✓ Validation PASSED")

    return {
        'pass_rate': pass_rate,
        'avg_entropy': avg_entropy,
        'avg_sparsity': avg_sparsity,
        'head_means': head_means.tolist(),
        'head_stds': head_stds.tolist(),
    }


def test_custom_pairs(args, config, prior_estimator):
    """
    Test on user-provided image-audio pairs
    """
    if args.test_pairs is None:
        print("\nNo test pairs provided, skipping custom test")
        return

    print(f"\n{'='*60}")
    print("Testing on Custom Pairs")
    print(f"{'='*60}")

    test_dir = Path(args.test_pairs)
    if not test_dir.exists():
        print(f"Error: Test directory {test_dir} not found")
        return

    # Find image-audio pairs
    image_exts = ['.jpg', '.jpeg', '.png']
    audio_exts = ['.mp3', '.wav', '.flac']

    images = sorted([f for f in test_dir.glob('*') if f.suffix.lower() in image_exts])
    audios = sorted([f for f in test_dir.glob('*') if f.suffix.lower() in audio_exts])

    if len(images) == 0 or len(audios) == 0:
        print(f"Error: No valid pairs found in {test_dir}")
        return

    # Pair up (assume alphabetical matching)
    pairs = list(zip(images, audios))[:min(len(images), len(audios))]

    print(f"Found {len(pairs)} pairs")

    # Test each pair
    save_dir = Path(config.output.output_path) / "test_pairs"

    from src.utils.prior_utils import batch_test_prior

    results = batch_test_prior(
        prior_estimator,
        [(str(img), str(aud)) for img, aud in pairs],
        device=args.device,
        save_dir=str(save_dir) if args.visualize else None,
    )

    print(f"\nResults saved to {save_dir}")

    return results


def main():
    args = parse_args()

    # Load config with defaults
    config_path = Path(args.config)
    default_config_path = config_path.parent / "default.yaml"

    # Load default config first
    if default_config_path.exists():
        config = OmegaConf.load(default_config_path)
        # Merge with stage-specific config
        stage_config = OmegaConf.load(args.config)
        config = OmegaConf.merge(config, stage_config)
    else:
        config = OmegaConf.load(args.config)

    print(f"Loaded config from {args.config}")

    # Resolve variable interpolations
    OmegaConf.resolve(config)

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Build Prior Estimator
    prior_estimator = build_prior_estimator(config, device)

    # Validate on dataset
    if config.data.dataset_path and Path(config.data.dataset_path).exists():
        dataset_results = validate_dataset(args, config, prior_estimator)
    else:
        print(f"\nDataset path {config.data.dataset_path} not found, skipping dataset validation")
        dataset_results = None

    # Test on custom pairs
    custom_results = test_custom_pairs(args, config, prior_estimator)

    # Save Prior Estimator
    save_dir = Path(config.output.output_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save model components
    torch.save({
        'hard_prior': {
            'rules': prior_estimator.hard_prior.rules,
            'W_hard': prior_estimator.hard_prior.W_hard,
        },
        'soft_prior': prior_estimator.soft_prior.state_dict(),
        'config': OmegaConf.to_container(config),
    }, save_dir / "prior_estimator.pt")

    print(f"\nPrior Estimator saved to {save_dir / 'prior_estimator.pt'}")

    # Save validation results
    if dataset_results:
        import json
        with open(save_dir / "validation_results.json", 'w') as f:
            json.dump(dataset_results, f, indent=2)
        print(f"Validation results saved to {save_dir / 'validation_results.json'}")

    print(f"\n{'='*60}")
    print("Stage 0 Validation Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
