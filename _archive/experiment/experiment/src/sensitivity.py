"""
Sensitivity Check (Phase 0-a)

Verifies that CLIP/CLAP models detect effects by measuring
the distance between original and augmented embeddings.
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

from .delta_extraction import DeltaDataset


@dataclass
class SensitivityResult:
    """Result of sensitivity check for an effect."""
    modality: str  # "image" or "audio"
    effect_type: str
    intensity: str
    mean_distance: float  # E[||e(x_eff) - e(x_0)||]
    std_distance: float
    min_distance: float
    max_distance: float
    num_samples: int
    is_sensitive: bool  # mean_distance > threshold


def sensitivity_check(
    delta_dataset: DeltaDataset,
    threshold: float = 0.01,
) -> List[SensitivityResult]:
    """
    Check if models are sensitive to effects.

    For each (modality, effect_type, intensity):
    - Compute ||delta|| = ||e(x_eff) - e(x_0)||
    - Check if mean distance > threshold

    Args:
        delta_dataset: Dataset of extracted deltas
        threshold: Minimum mean distance to be considered sensitive

    Returns:
        List of SensitivityResult for each (modality, effect, intensity)
    """
    results = []

    # Group by (modality, effect_type, intensity)
    groups: Dict[Tuple[str, str, str], List] = {}

    for delta_result in delta_dataset.deltas:
        key = (delta_result.modality, delta_result.effect_type, delta_result.intensity)
        if key not in groups:
            groups[key] = []
        groups[key].append(delta_result)

    # Compute statistics for each group
    for (modality, effect_type, intensity), deltas in groups.items():
        # Compute norms
        norms = [np.linalg.norm(d.delta) for d in deltas]

        mean_dist = float(np.mean(norms))
        std_dist = float(np.std(norms))
        min_dist = float(np.min(norms))
        max_dist = float(np.max(norms))

        is_sensitive = mean_dist > threshold

        results.append(SensitivityResult(
            modality=modality,
            effect_type=effect_type,
            intensity=intensity,
            mean_distance=mean_dist,
            std_distance=std_dist,
            min_distance=min_dist,
            max_distance=max_dist,
            num_samples=len(deltas),
            is_sensitive=is_sensitive,
        ))

    return results


def print_sensitivity_report(results: List[SensitivityResult]) -> None:
    """Print formatted sensitivity report."""
    print("\n" + "=" * 80)
    print("  SENSITIVITY CHECK REPORT (Phase 0-a)")
    print("=" * 80)

    # Group by modality
    for modality in ["image", "audio"]:
        modality_results = [r for r in results if r.modality == modality]
        if not modality_results:
            continue

        print(f"\n{modality.upper()} Effects:")
        print(f"  {'Effect':<15} {'Intensity':<10} {'Mean Dist':<12} {'Std':<10} "
              f"{'Samples':<8} {'Sensitive?':<10}")
        print(f"  {'-' * 15} {'-' * 10} {'-' * 12} {'-' * 10} {'-' * 8} {'-' * 10}")

        for r in sorted(modality_results, key=lambda x: (x.effect_type, x.intensity)):
            status = "✓ YES" if r.is_sensitive else "✗ NO"
            print(f"  {r.effect_type:<15} {r.intensity:<10} {r.mean_distance:<12.4f} "
                  f"{r.std_distance:<10.4f} {r.num_samples:<8} {status:<10}")

    # Summary
    total = len(results)
    sensitive = sum(1 for r in results if r.is_sensitive)
    print(f"\n" + "-" * 80)
    print(f"  Total: {total} effect configurations")
    print(f"  Sensitive: {sensitive} ({sensitive/total*100:.1f}%)")
    print(f"  Insensitive: {total-sensitive} ({(total-sensitive)/total*100:.1f}%)")
    print("=" * 80)


def get_insensitive_effects(
    results: List[SensitivityResult],
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Get list of insensitive effects to exclude from experiment.

    Returns:
        {modality: [(effect_type, intensity), ...]}
    """
    insensitive = {}

    for r in results:
        if not r.is_sensitive:
            if r.modality not in insensitive:
                insensitive[r.modality] = []
            insensitive[r.modality].append((r.effect_type, r.intensity))

    return insensitive
