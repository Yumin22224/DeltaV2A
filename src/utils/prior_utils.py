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

from src.data.transforms import AudioTransform, ImageTransform


def preprocess_test_image(image_path: str, resolution: int = 512) -> torch.Tensor:
    """
    Preprocess arbitrary image for Prior testing

    Delegates to ImageTransform for consistent preprocessing.

    Args:
        image_path: Path to image file
        resolution: Target resolution

    Returns:
        image: (1, 3, H, W) tensor
    """
    transform = ImageTransform(resolution=resolution)
    img_tensor = transform(image_path).unsqueeze(0)  # (1, 3, H, W)
    return img_tensor


def preprocess_test_audio(
    audio_path: str,
    sample_rate: int = 16000,
    n_mels: int = 64,
    mel_time_steps: int = 800,
    return_waveform: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Preprocess arbitrary audio for Prior testing
    Supports mp3, wav, flac, etc.

    Delegates to AudioTransform for consistent preprocessing.

    Args:
        audio_path: Path to audio file (mp3, wav, etc.)
        sample_rate: Target sample rate
        n_mels: Number of mel bands
        mel_time_steps: Target time steps
        return_waveform: Whether to also return raw waveform (for ImageBind)

    Returns:
        Dictionary with:
            - mel: (1, 1, T, F) mel spectrogram
            - waveform: (1, T_samples) raw waveform (if return_waveform=True)
    """
    transform = AudioTransform(
        sample_rate=sample_rate,
        n_mels=n_mels,
        mel_time_steps=mel_time_steps,
    )
    audio_result = transform(audio_path, return_waveform=return_waveform)
    mel_spec = audio_result['mel']  # (1, n_mels, T) from AudioTransform

    # Reshape to (1, 1, T, F) expected by Prior models
    mel_spec = mel_spec.transpose(1, 2).unsqueeze(0)  # (1, 1, T, n_mels)

    result = {'mel': mel_spec}
    if return_waveform and 'waveform' in audio_result:
        result['waveform'] = audio_result['waveform']  # (1, T_samples)

    return result


def validate_c_prior(
    C_prior: torch.Tensor,
    entropy_min: float = 0.5,
    sparsity_max: float = 5.0,
) -> Dict[str, any]:
    """
    Validate C_prior properties according to Spec v2.2

    C_prior should satisfy:
    1. Entropy constraint: H(C_prior) > H_min = 0.5 (prevents one-hot)
    2. Sparsity constraint: IPR(C_prior) < S_max = 5.0 (prevents uniform)
       IPR = Inverse Participation Ratio = 1 / sum(p_i^2) per token
       - Uniform over K=6 heads: IPR = 6 (maximum)
       - One-hot: IPR = 1 (minimum)
    3. Probability distribution: sum(C_prior[i, :]) = 1 for each token

    Args:
        C_prior: (B, N_v, 6) coupling tensor
        entropy_min: Minimum entropy threshold
        sparsity_max: Maximum IPR threshold

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

        # 2. Sparsity check via IPR (Inverse Participation Ratio)
        # IPR = 1 / sum(p_i^2) per token, averaged over all tokens
        # High IPR = uniform (bad), Low IPR = sparse/one-hot
        ipr_per_token = 1.0 / ((C ** 2).sum(dim=-1) + eps)  # (N_v,)
        avg_ipr = ipr_per_token.mean().item()
        results['sparsity'].append(avg_ipr)

        if avg_ipr > sparsity_max:
            results['passed'] = False
            results['violations'].append(
                f"Batch {b}: High IPR {avg_ipr:.3f} > {sparsity_max} (too uniform)"
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

        # Plot distributions (use adaptive bins to handle low-variance data)
        for data, color, label, a in [
            (C_hard_np[:, h], 'red', f'Hard (α={alpha})', 0.5),
            (C_soft_np[:, h], 'blue', f'Soft (1-α={1-alpha:.1f})', 0.5),
            (C_prior_np[:, h], 'green', 'Combined', 0.7),
        ]:
            data_range = data.max() - data.min()
            if data_range < 1e-6:
                # Near-constant data: single bar
                ax.axvline(data.mean(), color=color, alpha=a, linewidth=2, label=label)
            else:
                n_bins = min(30, max(5, int(len(data) ** 0.5)))
                ax.hist(data, bins=n_bins, alpha=a, color=color,
                        label=label, edgecolor='black')

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


def export_rule_activations_json(
    activation_detail: Dict,
    image_path: str = "",
    audio_path: str = "",
    save_path: Optional[str] = None,
) -> Dict:
    """
    Export rule activation diagnostics to JSON.

    Shows which rules were activated for a given (image, audio) pair,
    with visual feature values, audio descriptor values, and contributions.

    Args:
        activation_detail: Single-sample activation dict from HardPrior
        image_path: Path to the image (for reference)
        audio_path: Path to the audio (for reference)
        save_path: Path to save JSON file (optional)

    Returns:
        json_data: Structured dict suitable for JSON serialization
    """
    import json

    head_names = ['rhythm', 'harmony', 'energy', 'timbre', 'space', 'texture']

    # Aggregate per-head contributions
    head_summary = {h: {'total_weight': 0.0, 'activated_rules': [], 'inactive_rules': []}
                    for h in head_names}

    for rule_info in activation_detail.get('rules', []):
        head = rule_info['target_head']
        if head not in head_summary:
            continue

        entry = {
            'rule': rule_info['name'],
            'visual_feature': rule_info['visual_feature'],
            'audio_feature': rule_info['audio_feature'],
            'f_v': rule_info['f_v_value'],
            'f_a': rule_info['f_a_value'],
            'correlation': rule_info['correlation'],
            'weight': rule_info['weight'],
        }

        if rule_info.get('activation_score') is not None:
            entry['activation_score'] = rule_info['activation_score']

        if rule_info['activated']:
            head_summary[head]['activated_rules'].append(entry)
            head_summary[head]['total_weight'] += rule_info.get('contribution', rule_info['weight'])
        else:
            head_summary[head]['inactive_rules'].append(entry)

    json_data = {
        'image': str(Path(image_path).name) if image_path else '',
        'audio': str(Path(audio_path).name) if audio_path else '',
        'tau': 0.5,
        'summary': {
            head: {
                'total_activated_weight': round(info['total_weight'], 4),
                'num_activated': len(info['activated_rules']),
                'num_total': len(info['activated_rules']) + len(info['inactive_rules']),
            }
            for head, info in head_summary.items()
        },
        'rules': activation_detail.get('rules', []),
        'head_details': head_summary,
    }

    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"Saved rule activations to {save_path}")

    return json_data


def _print_activation_summary(activation_detail: Dict):
    """Print a concise summary of rule activations."""
    head_names = ['rhythm', 'harmony', 'energy', 'timbre', 'space', 'texture']
    head_weights = {h: 0.0 for h in head_names}
    activated_count = 0
    total_count = 0

    for rule_info in activation_detail.get('rules', []):
        total_count += 1
        if rule_info['activated']:
            activated_count += 1
            head = rule_info['target_head']
            if head in head_weights:
                head_weights[head] += rule_info.get('contribution', rule_info['weight'])

    print(f"\nRule activations: {activated_count}/{total_count} rules activated")
    print(f"  Per-head activated W_hard:")
    for h in head_names:
        w = head_weights[h]
        indicator = " *" if w > 0 else ""
        print(f"    {h:10s}: {w:.2f}{indicator}")


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
    audio_result = preprocess_test_audio(audio_path, return_waveform=True)
    audio_mel = audio_result['mel'].to(device)
    waveform = audio_result.get('waveform', None)
    if waveform is not None:
        waveform = waveform.to(device)

    print(f"  Image shape: {image.shape}")
    print(f"  Audio shape: {audio_mel.shape}")

    # Estimate C_prior (with rule activation details)
    print("\nEstimating C_prior...")
    prior_estimator.eval()
    activation_details = None
    with torch.no_grad():
        result = prior_estimator(
            image, audio_mel, waveforms=waveform, return_details=True
        )
        if isinstance(result, tuple):
            C_prior, activation_details = result
        else:
            C_prior = result

    print(f"  C_prior shape: {C_prior.shape}")
    print(f"  C_prior range: [{C_prior.min():.4f}, {C_prior.max():.4f}]")

    # Print rule activation summary
    if activation_details:
        _print_activation_summary(activation_details[0])

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

    # Compute C_soft and C_hard separately for comparison
    C_soft = None
    C_hard = None
    if hasattr(prior_estimator, 'soft_prior') and hasattr(prior_estimator, 'hard_prior'):
        with torch.no_grad():
            C_soft = prior_estimator.soft_prior(image, audio_mel, waveforms=waveform)
            # C_hard: saliency * activated W_hard, normalized
            N_v = C_soft.shape[1]
            saliency = prior_estimator.hard_prior.compute_saliency(image, N_v=N_v)
            W_hard_activated, _ = prior_estimator.hard_prior.compute_activated_weights(
                image, audio_mel, waveforms=waveform
            )  # (B, 6)
            C_hard_raw = saliency.unsqueeze(-1) * W_hard_activated.unsqueeze(1)
            C_hard = C_hard_raw / (C_hard_raw.sum(dim=-1, keepdim=True) + 1e-8)

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

        # Hard vs Soft comparison
        if C_soft is not None and C_hard is not None:
            comp_path = save_dir / "hard_soft_comparison.png" if save_dir else None
            visualize_hard_soft_comparison(
                C_hard[0],
                C_soft[0],
                C_sample,
                alpha=getattr(prior_estimator, 'alpha', 0.3),
                save_path=str(comp_path) if comp_path else None,
            )

    # Save rule activation JSON
    if activation_details and save_dir:
        json_path = save_dir / "rule_activations.json"
        export_rule_activations_json(
            activation_details[0],
            image_path=image_path,
            audio_path=audio_path,
            save_path=str(json_path),
        )

    # Prepare results
    results = {
        'C_prior': C_prior,
        'C_soft': C_soft,
        'C_hard': C_hard,
        'activation_details': activation_details,
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
