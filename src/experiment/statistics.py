"""
Statistical Analysis Module

Permutation tests, p-values, and statistical baselines.
"""

from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from scipy import stats
from dataclasses import dataclass

from .delta_extraction import DeltaDataset
from .retrieval import DeltaRetrieval, RetrievalMetrics


@dataclass
class PermutationTestResult:
    """Result of a permutation test."""
    observed_statistic: float
    null_distribution: np.ndarray
    p_value: float
    n_permutations: int
    effect_size: float  # (observed - null_mean) / null_std


@dataclass
class MonotonicityResult:
    """Result of monotonicity analysis."""
    effect_type: str
    spearman_rho: float
    spearman_p_value: float
    is_monotonic: bool  # rho > 0 and p < 0.05
    intensity_to_norm: Dict[str, float]  # mean norm per intensity


def permutation_test(
    observed: float,
    null_samples: np.ndarray,
    alternative: str = "greater",
) -> float:
    """
    Compute permutation p-value.

    Args:
        observed: Observed test statistic
        null_samples: Array of test statistics under null hypothesis
        alternative: "greater", "less", or "two-sided"

    Returns:
        p-value
    """
    n = len(null_samples)

    if alternative == "greater":
        p = (np.sum(null_samples >= observed) + 1) / (n + 1)
    elif alternative == "less":
        p = (np.sum(null_samples <= observed) + 1) / (n + 1)
    else:  # two-sided
        p = (np.sum(np.abs(null_samples) >= np.abs(observed)) + 1) / (n + 1)

    return float(p)


def retrieval_permutation_test(
    query_dataset: DeltaDataset,
    database_dataset: DeltaDataset,
    effect_type_mapping: Dict[str, str],
    metric: str = "mrr",  # "mrr" or "top_k_accuracy"
    k: int = 5,
    n_permutations: int = 1000,
    seed: int = 42,
) -> PermutationTestResult:
    """
    Permutation test for retrieval performance.

    Null hypothesis: Effect type labels in database are random
    (no semantic correspondence between image and audio effects).

    Args:
        query_dataset: Image deltas
        database_dataset: Audio deltas
        effect_type_mapping: Mapping from image to audio effect types
        metric: Metric to test ("mrr" or "top_k_accuracy")
        k: Value of k for top-k accuracy
        n_permutations: Number of permutations
        seed: Random seed

    Returns:
        PermutationTestResult
    """
    rng = np.random.RandomState(seed)

    # Compute observed metric
    retrieval = DeltaRetrieval(
        query_dataset, database_dataset, effect_type_mapping
    )
    observed_metrics = retrieval.evaluate(k_values=[k])

    if metric == "mrr":
        observed = observed_metrics.mean_reciprocal_rank
    else:
        observed = observed_metrics.top_k_accuracy[k]

    # Generate null distribution by permuting effect labels
    null_samples = []

    original_labels = [d.effect_type for d in database_dataset.deltas]

    for _ in range(n_permutations):
        # Shuffle labels
        permuted_labels = rng.permutation(original_labels)

        # Create permuted dataset (modify labels in-place temporarily)
        for i, delta in enumerate(database_dataset.deltas):
            delta._original_effect_type = delta.effect_type
            delta.effect_type = permuted_labels[i]

        # Evaluate
        retrieval = DeltaRetrieval(
            query_dataset, database_dataset, effect_type_mapping
        )
        perm_metrics = retrieval.evaluate(k_values=[k])

        if metric == "mrr":
            null_samples.append(perm_metrics.mean_reciprocal_rank)
        else:
            null_samples.append(perm_metrics.top_k_accuracy[k])

        # Restore original labels
        for delta in database_dataset.deltas:
            delta.effect_type = delta._original_effect_type

    null_samples = np.array(null_samples)

    # Compute p-value and effect size
    p_value = permutation_test(observed, null_samples, alternative="greater")

    null_mean = np.mean(null_samples)
    null_std = np.std(null_samples)
    effect_size = (observed - null_mean) / null_std if null_std > 0 else 0.0

    return PermutationTestResult(
        observed_statistic=observed,
        null_distribution=null_samples,
        p_value=p_value,
        n_permutations=n_permutations,
        effect_size=effect_size,
    )


def trivial_confound_baseline(
    query_dataset: DeltaDataset,
    database_dataset: DeltaDataset,
    effect_type_mapping: Dict[str, str],
    k_values: List[int] = [1, 3, 5, 10],
) -> RetrievalMetrics:
    """
    Baseline using original embeddings instead of deltas.

    Tests whether retrieval works just because of static similarity
    (e.g., same content class) rather than effect correspondence.

    Args:
        query_dataset: Image deltas (will use original_embedding)
        database_dataset: Audio deltas (will use original_embedding)
        effect_type_mapping: Effect mapping

    Returns:
        RetrievalMetrics for the baseline
    """
    # Create temporary datasets with original embeddings as "deltas"
    from .delta_extraction import DeltaResult

    query_originals = DeltaDataset()
    for d in query_dataset.deltas:
        query_originals.add(DeltaResult(
            modality=d.modality,
            effect_type=d.effect_type,
            intensity=d.intensity,
            original_path=d.original_path,
            delta=d.original_embedding,  # Use original instead of delta
            original_embedding=d.original_embedding,
            augmented_embedding=d.augmented_embedding,
        ))

    db_originals = DeltaDataset()
    for d in database_dataset.deltas:
        db_originals.add(DeltaResult(
            modality=d.modality,
            effect_type=d.effect_type,
            intensity=d.intensity,
            original_path=d.original_path,
            delta=d.original_embedding,  # Use original instead of delta
            original_embedding=d.original_embedding,
            augmented_embedding=d.augmented_embedding,
        ))

    retrieval = DeltaRetrieval(query_originals, db_originals, effect_type_mapping)
    return retrieval.evaluate(k_values=k_values)


def norm_monotonicity_analysis(
    delta_dataset: DeltaDataset,
    modality: str,
) -> Dict[str, MonotonicityResult]:
    """
    Analyze if delta norm increases monotonically with intensity.

    For each effect type, check if ||Δe|| increases as intensity goes
    from "low" -> "mid" -> "high".

    Args:
        delta_dataset: Dataset with deltas
        modality: "image" or "audio"

    Returns:
        Dict mapping effect_type -> MonotonicityResult
    """
    intensity_order = {"low": 0, "mid": 1, "high": 2}
    results = {}

    # Get effect types
    effect_types = list(set(
        d.effect_type for d in delta_dataset.deltas
        if d.modality == modality
    ))

    for effect_type in effect_types:
        # Get deltas for this effect
        deltas = delta_dataset.filter_by(
            modality=modality,
            effect_type=effect_type,
        )

        # Group by intensity and compute mean norm
        intensity_norms = {}
        for intensity in ["low", "mid", "high"]:
            intensity_deltas = [d for d in deltas if d.intensity == intensity]
            if intensity_deltas:
                norms = [np.linalg.norm(d.delta) for d in intensity_deltas]
                intensity_norms[intensity] = float(np.mean(norms))

        # Compute Spearman correlation
        intensities = []
        norms = []
        for intensity in ["low", "mid", "high"]:
            if intensity in intensity_norms:
                intensities.append(intensity_order[intensity])
                norms.append(intensity_norms[intensity])

        if len(intensities) >= 3:
            rho, p_value = stats.spearmanr(intensities, norms)
        else:
            rho, p_value = 0.0, 1.0

        results[effect_type] = MonotonicityResult(
            effect_type=effect_type,
            spearman_rho=float(rho),
            spearman_p_value=float(p_value),
            is_monotonic=rho > 0 and p_value < 0.05,
            intensity_to_norm=intensity_norms,
        )

    return results


def cross_intensity_alignment(
    query_dataset: DeltaDataset,
    database_dataset: DeltaDataset,
    effect_type_mapping: Dict[str, str],
) -> Dict[str, float]:
    """
    Check if intensity alignment is preserved across modalities.

    For matched effect pairs, does intensity_image correlate with
    best-matching intensity_audio?

    Args:
        query_dataset: Image deltas
        database_dataset: Audio deltas
        effect_type_mapping: Effect mapping

    Returns:
        Dict with Spearman rho for each effect pair
    """
    from .retrieval import cosine_similarity

    intensity_order = {"low": 0, "mid": 1, "high": 2}
    results = {}

    for img_effect, aud_effect in effect_type_mapping.items():
        query_intensities = []
        best_match_intensities = []

        for intensity in ["low", "mid", "high"]:
            # Get image deltas for this effect/intensity
            img_deltas = query_dataset.filter_by(
                modality="image",
                effect_type=img_effect,
                intensity=intensity,
            )

            if not img_deltas:
                continue

            # Get all audio deltas for the matching effect
            aud_deltas = database_dataset.filter_by(
                modality="audio",
                effect_type=aud_effect,
            )

            if not aud_deltas:
                continue

            # For each image delta, find best matching audio delta intensity
            for img_d in img_deltas:
                best_sim = -float('inf')
                best_intensity = None

                for aud_d in aud_deltas:
                    sim = cosine_similarity(img_d.delta, aud_d.delta)
                    if sim > best_sim:
                        best_sim = sim
                        best_intensity = aud_d.intensity

                if best_intensity:
                    query_intensities.append(intensity_order[intensity])
                    best_match_intensities.append(intensity_order[best_intensity])

        if len(query_intensities) >= 3:
            rho, _ = stats.spearmanr(query_intensities, best_match_intensities)
            results[f"{img_effect}->{aud_effect}"] = float(rho)

    return results
