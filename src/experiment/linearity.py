"""
Linearity Analysis (Phase 0-b)

Verifies context invariance: effects should produce consistent delta directions
regardless of source image/audio category.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path

from .delta_extraction import DeltaDataset


@dataclass
class LinearityResult:
    """
    Result of linearity analysis for a single (modality, effect_type, intensity, category).
    """
    modality: str
    effect_type: str
    intensity: str
    category: str

    # Delta direction consistency within category
    mean_pairwise_cosine: float
    std_pairwise_cosine: float
    min_cosine: float
    max_cosine: float

    # Delta magnitude consistency within category
    mean_norm: float
    cv_norm: float  # Coefficient of Variation (std/mean)

    num_samples: int
    is_consistent: bool  # mean_pairwise_cosine > threshold


@dataclass
class CrossCategoryVariance:
    """
    Variance of delta consistency across categories.

    High variance → context-dependent (violates linearity assumption)
    """
    modality: str
    effect_type: str
    intensity: str

    # Variance of mean_pairwise_cosine across categories
    variance_cosine: float
    mean_cosine: float
    categories: List[str]

    is_context_invariant: bool  # variance_cosine < threshold


def get_category_from_path(path: str) -> str:
    """
    Extract category from file path.

    Assumes directory structure: .../category/file.ext

    Args:
        path: File path

    Returns:
        category: Category name (parent directory)
    """
    return Path(path).parent.name


def linearity_analysis(
    delta_dataset: DeltaDataset,
    category_mapping: Dict[str, str] = None,
    cosine_threshold: float = 0.8,
) -> List[LinearityResult]:
    """
    Analyze delta linearity within each category.

    For each (modality, effect_type, intensity, category):
    - Compute pairwise cosine similarity of normalized deltas
    - Compute CV of delta norms

    Args:
        delta_dataset: Dataset of extracted deltas
        category_mapping: Optional path -> category mapping (if None, extract from path)
        cosine_threshold: Minimum mean cosine for consistency

    Returns:
        List of LinearityResult
    """
    results = []

    # Add category field if missing
    for delta_result in delta_dataset.deltas:
        if not hasattr(delta_result, 'category') or delta_result.category is None:
            if category_mapping:
                delta_result.category = category_mapping.get(
                    delta_result.original_path,
                    get_category_from_path(delta_result.original_path)
                )
            else:
                delta_result.category = get_category_from_path(delta_result.original_path)

    # Group by (modality, effect_type, intensity, category)
    groups: Dict[Tuple[str, str, str, str], List] = {}

    for delta_result in delta_dataset.deltas:
        key = (
            delta_result.modality,
            delta_result.effect_type,
            delta_result.intensity,
            delta_result.category,
        )
        if key not in groups:
            groups[key] = []
        groups[key].append(delta_result)

    # Analyze each group
    for (modality, effect_type, intensity, category), deltas in groups.items():
        if len(deltas) < 2:
            # Need at least 2 samples for pairwise comparison
            continue

        # Get delta vectors
        delta_vectors = np.stack([d.delta for d in deltas])

        # Normalize deltas
        norms = np.linalg.norm(delta_vectors, axis=1)
        safe_norms = np.where(norms > 0, norms, 1.0)
        normalized_deltas = delta_vectors / safe_norms[:, np.newaxis]

        # Compute pairwise cosine similarity
        similarity_matrix = normalized_deltas @ normalized_deltas.T

        # Extract upper triangle (exclude diagonal)
        mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
        pairwise_cosines = similarity_matrix[mask]

        if len(pairwise_cosines) == 0:
            continue

        mean_cosine = float(np.mean(pairwise_cosines))
        std_cosine = float(np.std(pairwise_cosines))
        min_cosine = float(np.min(pairwise_cosines))
        max_cosine = float(np.max(pairwise_cosines))

        # Compute norm statistics
        mean_norm = float(np.mean(norms))
        std_norm = float(np.std(norms))
        cv_norm = std_norm / mean_norm if mean_norm > 0 else float('inf')

        is_consistent = mean_cosine > cosine_threshold

        results.append(LinearityResult(
            modality=modality,
            effect_type=effect_type,
            intensity=intensity,
            category=category,
            mean_pairwise_cosine=mean_cosine,
            std_pairwise_cosine=std_cosine,
            min_cosine=min_cosine,
            max_cosine=max_cosine,
            mean_norm=mean_norm,
            cv_norm=cv_norm,
            num_samples=len(deltas),
            is_consistent=is_consistent,
        ))

    return results


def cross_category_variance_check(
    delta_dataset: DeltaDataset,
    cosine_threshold: float = 0.8,
) -> List[CrossCategoryVariance]:
    """
    Check if effects produce consistent delta directions across categories.

    Context invariance: Same effect should produce similar delta directions
    regardless of source category.

    Method:
    1. For each (modality, effect, intensity):
       - Compute prototype vector for each category: p_c = mean(normalized deltas)
       - Compute pairwise cosine similarity between category prototypes
       - High similarity → context-invariant (✓)

    Args:
        delta_dataset: Dataset of extracted deltas
        cosine_threshold: Minimum cosine similarity for context invariance (default: 0.8)

    Returns:
        List of CrossCategoryVariance for each (modality, effect, intensity)
    """
    variance_results = []

    # Group deltas by (modality, effect_type, intensity, category)
    groups: Dict[Tuple[str, str, str, str], List] = {}

    for delta in delta_dataset.deltas:
        key = (delta.modality, delta.effect_type, delta.intensity, delta.category)
        if key not in groups:
            groups[key] = []
        groups[key].append(delta.delta)

    # Group by (modality, effect_type, intensity) to compare across categories
    effect_groups: Dict[Tuple[str, str, str], Dict[str, np.ndarray]] = {}

    for (modality, effect_type, intensity, category), deltas in groups.items():
        effect_key = (modality, effect_type, intensity)
        if effect_key not in effect_groups:
            effect_groups[effect_key] = {}

        # Compute prototype: mean of normalized deltas for this category
        deltas_array = np.stack(deltas)  # (N, D)
        norms = np.linalg.norm(deltas_array, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        normalized = deltas_array / norms
        prototype = np.mean(normalized, axis=0)  # (D,)

        effect_groups[effect_key][category] = prototype

    # Compute cross-category similarity for each effect
    for (modality, effect_type, intensity), category_prototypes in effect_groups.items():
        if len(category_prototypes) < 2:
            # Need at least 2 categories to compare
            continue

        categories = list(category_prototypes.keys())
        prototypes = [category_prototypes[c] for c in categories]

        # Compute pairwise cosine similarity between category prototypes
        similarities = []
        for i in range(len(prototypes)):
            for j in range(i + 1, len(prototypes)):
                p1 = prototypes[i]
                p2 = prototypes[j]

                # Normalize
                p1_norm = np.linalg.norm(p1)
                p2_norm = np.linalg.norm(p2)

                if p1_norm > 0 and p2_norm > 0:
                    cosine = np.dot(p1, p2) / (p1_norm * p2_norm)
                    similarities.append(float(cosine))

        if len(similarities) == 0:
            continue

        # Statistics
        mean_similarity = float(np.mean(similarities))
        variance_similarity = float(np.var(similarities))

        # Context-invariant if mean similarity is high
        is_invariant = mean_similarity >= cosine_threshold

        variance_results.append(CrossCategoryVariance(
            modality=modality,
            effect_type=effect_type,
            intensity=intensity,
            variance_cosine=variance_similarity,
            mean_cosine=mean_similarity,
            categories=categories,
            is_context_invariant=is_invariant,
        ))

    return variance_results


def print_linearity_report(
    results: List[LinearityResult],
    variance_results: List[CrossCategoryVariance],
) -> None:
    """Print formatted linearity analysis report."""
    print("\n" + "=" * 100)
    print("  LINEARITY ANALYSIS REPORT (Phase 0-b)")
    print("=" * 100)

    # Within-category consistency
    print("\n--- Within-Category Consistency ---")
    for modality in ["image", "audio"]:
        modality_results = [r for r in results if r.modality == modality]
        if not modality_results:
            continue

        print(f"\n{modality.upper()} Effects:")
        print(f"  {'Effect':<15} {'Intensity':<10} {'Category':<20} {'Mean Cos':<10} "
              f"{'CV Norm':<10} {'N':<5} {'OK?':<5}")
        print(f"  {'-' * 15} {'-' * 10} {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 5} {'-' * 5}")

        for r in sorted(modality_results, key=lambda x: (x.effect_type, x.intensity, x.category)):
            status = "✓" if r.is_consistent else "✗"
            print(f"  {r.effect_type:<15} {r.intensity:<10} {r.category:<20} "
                  f"{r.mean_pairwise_cosine:<10.3f} {r.cv_norm:<10.3f} {r.num_samples:<5} {status:<5}")

    # Cross-category variance
    print("\n--- Cross-Category Variance ---")
    for modality in ["image", "audio"]:
        modality_variance = [v for v in variance_results if v.modality == modality]
        if not modality_variance:
            continue

        print(f"\n{modality.upper()} Effects:")
        print(f"  {'Effect':<15} {'Intensity':<10} {'Variance':<12} {'Mean Cosine':<12} "
              f"{'# Categories':<12} {'Invariant?':<10}")
        print(f"  {'-' * 15} {'-' * 10} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 10}")

        for v in sorted(modality_variance, key=lambda x: (x.effect_type, x.intensity)):
            status = "✓ YES" if v.is_context_invariant else "✗ NO"
            print(f"  {v.effect_type:<15} {v.intensity:<10} {v.variance_cosine:<12.4f} "
                  f"{v.mean_cosine:<12.3f} {len(v.categories):<12} {status:<10}")

    # Summary
    total_configs = len(variance_results)
    invariant = sum(1 for v in variance_results if v.is_context_invariant)

    print("\n" + "-" * 100)
    print(f"  Total effect configurations: {total_configs}")

    if total_configs > 0:
        print(f"  Context-invariant: {invariant} ({invariant/total_configs*100:.1f}%)")
        print(f"  Context-dependent: {total_configs-invariant} ({(total_configs-invariant)/total_configs*100:.1f}%)")
    else:
        print("  ⚠️  Cross-category variance cannot be computed (need at least 2 categories per modality)")
        print("     Single-category data detected. Consider adding more categories for full validation.")

    print("=" * 100)


def get_inconsistent_effects(
    variance_results: List[CrossCategoryVariance],
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Get list of context-dependent effects to exclude or handle specially.

    Returns:
        {modality: [(effect_type, intensity), ...]}
    """
    inconsistent = {}

    for v in variance_results:
        if not v.is_context_invariant:
            if v.modality not in inconsistent:
                inconsistent[v.modality] = []
            inconsistent[v.modality].append((v.effect_type, v.intensity))

    return inconsistent
