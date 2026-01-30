"""
Utility functions for Prior validation and testing

Provides tools to:
1. Test C_prior estimation
2. Visualize coupling distributions
3. Validate Hard/Soft Prior properties
4. Preprocess arbitrary images/audios for testing
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2


def preprocess_test_image(image_path: str, resolution: int = 512) -> torch.Tensor:
    """
    Preprocess arbitrary image for Prior testing

    Args:
        image_path: Path to image file
        resolution: Target resolution

    Returns:
        image: (1, 3, H, W) tensor
    """
    from PIL import Image
    import torchvision.transforms as T

    # Load image
    img = Image.open(image_path).convert('RGB')

    # Transform
    transform = T.Compose([
        T.Resize(resolution),
        T.CenterCrop(resolution),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    img_tensor = transform(img).unsqueeze(0)  # (1, 3, H, W)

    return img_tensor


def preprocess_test_audio(
    audio_path: str,
    sample_rate: int = 16000,
    n_mels: int = 64,
    mel_time_steps: int = 800,
) -> torch.Tensor:
    """
    Preprocess arbitrary audio for Prior testing
    Supports mp3, wav, flac, etc.

    Args:
        audio_path: Path to audio file (mp3, wav, etc.)
        sample_rate: Target sample rate
        n_mels: Number of mel bands
        mel_time_steps: Target time steps

    Returns:
        audio_mel: (1, 1, T, F) mel spectrogram
    """
    import torchaudio
    import librosa

    # Try torchaudio first (supports mp3)
    try:
        waveform, sr = torchaudio.load(audio_path)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)

    except Exception as e:
        print(f"torchaudio failed ({e}), trying librosa...")
        # Fallback to librosa
        waveform_np, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        waveform = torch.from_numpy(waveform_np).unsqueeze(0).float()

    # Mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=160,
        n_mels=n_mels,
    )

    mel_spec = mel_transform(waveform)  # (1, n_mels, T)

    # Log scale
    mel_spec = torch.log(mel_spec + 1e-8)

    # Normalize
    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)

    # Adjust time dimension
    if mel_spec.shape[-1] < mel_time_steps:
        padding = mel_time_steps - mel_spec.shape[-1]
        mel_spec = F.pad(mel_spec, (0, padding))
    elif mel_spec.shape[-1] > mel_time_steps:
        mel_spec = mel_spec[..., :mel_time_steps]

    # Reshape to (1, 1, T, F)
    mel_spec = mel_spec.transpose(1, 2).unsqueeze(0)  # (1, 1, T, n_mels)

    return mel_spec


def validate_c_prior(
    C_prior: torch.Tensor,
    entropy_min: float = 0.5,
    sparsity_max: float = 5.0,
) -> Dict[str, any]:
    """
    Validate C_prior properties according to Spec v2

    C_prior should satisfy:
    1. Entropy constraint: H(C_prior) > H_min
    2. Sparsity constraint: ||C_prior||_1 < S_max
    3. Probability distribution: sum(C_prior[i, :]) = 1 for each token

    Args:
        C_prior: (B, N_v, 6) coupling tensor
        entropy_min: Minimum entropy threshold
        sparsity_max: Maximum sparsity threshold

    Returns:
        validation_results: Dictionary with validation metrics
    """
    B, N_v, K = C_prior.shape

    results = {
        'passed': True,
        'entropy': [],
        'sparsity': [],
        'sum_check': [],
        'violations': [],
    }

    for b in range(B):
        C = C_prior[b]  # (N_v, 6)

        # 1. Entropy check (per token)
        # H = -sum(p * log(p))
        eps = 1e-8
        entropy_per_token = -(C * torch.log(C + eps)).sum(dim=-1)  # (N_v,)
        avg_entropy = entropy_per_token.mean().item()

        results['entropy'].append(avg_entropy)

        if avg_entropy < entropy_min:
            results['passed'] = False
            results['violations'].append(
                f"Batch {b}: Low entropy {avg_entropy:.3f} < {entropy_min}"
            )

        # 2. Sparsity check
        sparsity = torch.norm(C, p=1).item()
        results['sparsity'].append(sparsity)

        if sparsity > sparsity_max:
            results['passed'] = False
            results['violations'].append(
                f"Batch {b}: High sparsity {sparsity:.3f} > {sparsity_max}"
            )

        # 3. Probability sum check (should be ~1 per token)
        token_sums = C.sum(dim=-1)  # (N_v,)
        sum_error = (token_sums - 1.0).abs().max().item()
        results['sum_check'].append(sum_error)

        if sum_error > 0.01:
            results['violations'].append(
                f"Batch {b}: Probability sum error {sum_error:.4f}"
            )

    # Summary statistics
    results['entropy_mean'] = np.mean(results['entropy'])
    results['entropy_std'] = np.std(results['entropy'])
    results['sparsity_mean'] = np.mean(results['sparsity'])
    results['sparsity_std'] = np.std(results['sparsity'])

    return results


def visualize_coupling(
    C: torch.Tensor,
    head_names: List[str] = None,
    save_path: Optional[str] = None,
    title: str = "Coupling Distribution",
):
    """
    Visualize coupling distribution C

    Args:
        C: (N_v, 6) coupling for single sample
        head_names: List of head names
        save_path: Path to save figure
        title: Plot title
    """
    if head_names is None:
        head_names = ['rhythm', 'harmony', 'energy', 'timbre', 'space', 'texture']

    C_np = C.detach().cpu().numpy()
    N_v, K = C_np.shape

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for h, head_name in enumerate(head_names):
        ax = axes[h]

        # Plot distribution of coupling weights for this head
        weights = C_np[:, h]

        ax.hist(weights, bins=50, alpha=0.7, color=f'C{h}', edgecolor='black')
        ax.axvline(weights.mean(), color='red', linestyle='--',
                   label=f'Mean: {weights.mean():.3f}')
        ax.set_title(f'{head_name.capitalize()} Head')
        ax.set_xlabel('Coupling Weight')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved coupling visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_hard_soft_comparison(
    C_hard: torch.Tensor,
    C_soft: torch.Tensor,
    C_prior: torch.Tensor,
    alpha: float = 0.3,
    save_path: Optional[str] = None,
):
    """
    Visualize comparison between Hard Prior, Soft Prior, and Combined Prior

    Args:
        C_hard: (N_v, 6) hard prior contribution
        C_soft: (N_v, 6) soft prior
        C_prior: (N_v, 6) combined prior
        alpha: Hard/soft balance weight
        save_path: Path to save figure
    """
    head_names = ['rhythm', 'harmony', 'energy', 'timbre', 'space', 'texture']

    # Convert to numpy
    C_hard_np = C_hard.detach().cpu().numpy()
    C_soft_np = C_soft.detach().cpu().numpy()
    C_prior_np = C_prior.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for h, head_name in enumerate(head_names):
        ax = axes[h]

        # Plot distributions
        ax.hist(C_hard_np[:, h], bins=30, alpha=0.5, color='red',
                label=f'Hard (α={alpha})', edgecolor='black')
        ax.hist(C_soft_np[:, h], bins=30, alpha=0.5, color='blue',
                label=f'Soft (1-α={1-alpha})', edgecolor='black')
        ax.hist(C_prior_np[:, h], bins=30, alpha=0.7, color='green',
                label='Combined', edgecolor='black')

        ax.set_title(f'{head_name.capitalize()} Head', fontsize=12, fontweight='bold')
        ax.set_xlabel('Coupling Weight')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f'Hard vs Soft Prior Comparison (α={alpha})',
        fontsize=16,
        fontweight='bold'
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    else:
        plt.show()

    plt.close()


def test_prior_estimator(
    prior_estimator,
    image_path: str,
    audio_path: str,
    device: str = 'cpu',
    visualize: bool = True,
    save_dir: Optional[str] = None,
) -> Dict[str, any]:
    """
    End-to-end test of Prior estimator

    Args:
        prior_estimator: PriorEstimator model
        image_path: Path to test image
        audio_path: Path to test audio (mp3, wav, etc.)
        device: Device to use
        visualize: Whether to create visualizations
        save_dir: Directory to save outputs

    Returns:
        test_results: Dictionary with test outputs
    """
    print(f"\n{'='*60}")
    print(f"Testing Prior Estimator")
    print(f"{'='*60}")
    print(f"Image: {Path(image_path).name}")
    print(f"Audio: {Path(audio_path).name}")
    print(f"Device: {device}\n")

    # Prepare save directory
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Preprocess inputs
    print("Preprocessing inputs...")
    image = preprocess_test_image(image_path).to(device)
    audio_mel = preprocess_test_audio(audio_path).to(device)

    print(f"  Image shape: {image.shape}")
    print(f"  Audio shape: {audio_mel.shape}")

    # Estimate C_prior
    print("\nEstimating C_prior...")
    prior_estimator.eval()
    with torch.no_grad():
        C_prior = prior_estimator(image, audio_mel)

    print(f"  C_prior shape: {C_prior.shape}")
    print(f"  C_prior range: [{C_prior.min():.4f}, {C_prior.max():.4f}]")

    # Validate
    print("\nValidating C_prior properties...")
    validation = validate_c_prior(C_prior)

    print(f"  ✓ Validation {'PASSED' if validation['passed'] else 'FAILED'}")
    print(f"  - Entropy: {validation['entropy_mean']:.3f} ± {validation['entropy_std']:.3f}")
    print(f"  - Sparsity: {validation['sparsity_mean']:.3f} ± {validation['sparsity_std']:.3f}")

    if validation['violations']:
        print(f"\n  Violations:")
        for v in validation['violations']:
            print(f"    - {v}")

    # Head-wise statistics
    print("\nHead-wise statistics:")
    head_names = ['rhythm', 'harmony', 'energy', 'timbre', 'space', 'texture']
    C_sample = C_prior[0]  # (N_v, 6)

    for h, name in enumerate(head_names):
        weights = C_sample[:, h]
        print(f"  {name:10s}: mean={weights.mean():.3f}, "
              f"std={weights.std():.3f}, "
              f"max={weights.max():.3f}")

    # Visualize
    if visualize:
        print("\nGenerating visualizations...")

        # Coupling distribution
        viz_path = save_dir / "coupling_distribution.png" if save_dir else None
        visualize_coupling(
            C_sample,
            head_names=head_names,
            save_path=str(viz_path) if viz_path else None,
            title=f"C_prior Distribution - {Path(image_path).stem}"
        )

    # Prepare results
    results = {
        'C_prior': C_prior,
        'validation': validation,
        'image_path': image_path,
        'audio_path': audio_path,
        'head_stats': {
            name: {
                'mean': C_sample[:, h].mean().item(),
                'std': C_sample[:, h].std().item(),
                'max': C_sample[:, h].max().item(),
                'min': C_sample[:, h].min().item(),
            }
            for h, name in enumerate(head_names)
        },
    }

    print(f"\n{'='*60}")
    print("Test complete!")
    print(f"{'='*60}\n")

    return results


def batch_test_prior(
    prior_estimator,
    test_pairs: List[Tuple[str, str]],
    device: str = 'cpu',
    save_dir: Optional[str] = None,
) -> List[Dict]:
    """
    Test Prior estimator on multiple image-audio pairs

    Args:
        prior_estimator: PriorEstimator model
        test_pairs: List of (image_path, audio_path) tuples
        device: Device to use
        save_dir: Directory to save outputs

    Returns:
        all_results: List of test results for each pair
    """
    all_results = []

    for i, (img_path, aud_path) in enumerate(test_pairs):
        print(f"\n[{i+1}/{len(test_pairs)}] Testing pair...")

        pair_save_dir = None
        if save_dir:
            pair_save_dir = Path(save_dir) / f"pair_{i:03d}"
            pair_save_dir.mkdir(parents=True, exist_ok=True)

        results = test_prior_estimator(
            prior_estimator,
            img_path,
            aud_path,
            device=device,
            visualize=True,
            save_dir=str(pair_save_dir) if pair_save_dir else None,
        )

        all_results.append(results)

    # Summary statistics
    print(f"\n{'='*60}")
    print(f"Batch Test Summary ({len(test_pairs)} pairs)")
    print(f"{'='*60}")

    all_passed = all(r['validation']['passed'] for r in all_results)
    print(f"Overall validation: {'PASSED' if all_passed else 'FAILED'}")

    avg_entropy = np.mean([r['validation']['entropy_mean'] for r in all_results])
    avg_sparsity = np.mean([r['validation']['sparsity_mean'] for r in all_results])

    print(f"Average entropy: {avg_entropy:.3f}")
    print(f"Average sparsity: {avg_sparsity:.3f}")
    print(f"{'='*60}\n")

    return all_results


# Quick test function for development
def quick_test():
    """
    Quick test with dummy data (for development)
    """
    print("Running quick test with dummy data...")

    # Create dummy C_prior
    B, N_v, K = 2, 256, 6
    C_prior = torch.softmax(torch.randn(B, N_v, K), dim=-1)

    # Validate
    results = validate_c_prior(C_prior)

    print(f"\nValidation: {'PASSED' if results['passed'] else 'FAILED'}")
    print(f"Entropy: {results['entropy_mean']:.3f} ± {results['entropy_std']:.3f}")
    print(f"Sparsity: {results['sparsity_mean']:.3f} ± {results['sparsity_std']:.3f}")

    # Visualize
    visualize_coupling(C_prior[0], title="Dummy C_prior Test")

    print("\nQuick test complete!")


if __name__ == "__main__":
    quick_test()
