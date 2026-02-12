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
class DeltaConsistencyResult:
    """Result of delta consistency analysis for a single (effect_type, intensity)."""
    modality: str
    effect_type: str
    intensity: str
    n_samples: int

    # Direction consistency (pairwise cosine sim of normalized deltas)
    mean_pairwise_cosine: float
    std_pairwise_cosine: float
    min_pairwise_cosine: float
    max_pairwise_cosine: float

    # Magnitude consistency (delta norms)
    mean_norm: float
    std_norm: float
    cv_norm: float  # coefficient of variation = std/mean

    @property
    def is_direction_consistent(self) -> bool:
        """Direction consistent if mean pairwise cosine > 0.5"""
        return self.mean_pairwise_cosine > 0.5

    @property
    def is_magnitude_consistent(self) -> bool:
        """Magnitude consistent if CV < 0.5"""
        return self.cv_norm < 0.5


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


def delta_consistency_analysis(
    delta_dataset: DeltaDataset,
    modality: Optional[str] = None,
) -> Dict[str, DeltaConsistencyResult]:
    """
    Analyze if deltas are consistent across different source samples
    for each fixed (effect_type, intensity).

    For a given effect and intensity, all N source samples should produce
    delta vectors pointing in a similar direction with similar magnitude.

    Measures:
        - Direction: pairwise cosine similarity of normalized deltas
        - Magnitude: coefficient of variation of delta norms

    Args:
        delta_dataset: Dataset with deltas
        modality: Filter by modality ("image" or "audio"), or None for both

    Returns:
        Dict mapping "modality/effect_type/intensity" -> DeltaConsistencyResult
    """
    # Group deltas by (modality, effect_type, intensity)
    groups: Dict[str, list] = {}
    for d in delta_dataset.deltas:
        if modality and d.modality != modality:
            continue
        key = f"{d.modality}/{d.effect_type}/{d.intensity}"
        if key not in groups:
            groups[key] = []
        groups[key].append(d.delta)

    results = {}

    for key, deltas_list in groups.items():
        mod, effect_type, intensity = key.split("/")
        deltas = np.stack(deltas_list)  # (N, D)
        n = len(deltas)

        # --- Magnitude analysis ---
        norms = np.linalg.norm(deltas, axis=1)  # (N,)
        mean_norm = float(np.mean(norms))
        std_norm = float(np.std(norms))
        cv_norm = std_norm / mean_norm if mean_norm > 0 else float('inf')

        # --- Direction analysis ---
        # Normalize deltas to unit vectors
        safe_norms = np.where(norms > 0, norms, 1.0)
        normed = deltas / safe_norms[:, np.newaxis]  # (N, D)

        if n >= 2:
            # Pairwise cosine similarity (= dot product of unit vectors)
            sim_matrix = normed @ normed.T  # (N, N)

            # Extract upper triangle (exclude diagonal)
            triu_indices = np.triu_indices(n, k=1)
            pairwise_sims = sim_matrix[triu_indices]

            mean_cos = float(np.mean(pairwise_sims))
            std_cos = float(np.std(pairwise_sims))
            min_cos = float(np.min(pairwise_sims))
            max_cos = float(np.max(pairwise_sims))
        else:
            mean_cos = std_cos = min_cos = max_cos = float('nan')

        results[key] = DeltaConsistencyResult(
            modality=mod,
            effect_type=effect_type,
            intensity=intensity,
            n_samples=n,
            mean_pairwise_cosine=mean_cos,
            std_pairwise_cosine=std_cos,
            min_pairwise_cosine=min_cos,
            max_pairwise_cosine=max_cos,
            mean_norm=mean_norm,
            std_norm=std_norm,
            cv_norm=cv_norm,
        )

    return results


def print_consistency_report(
    results: Dict[str, 'DeltaConsistencyResult'],
) -> None:
    """Print a formatted consistency report."""
    # Group by modality
    by_modality: Dict[str, list] = {}
    for key, r in results.items():
        if r.modality not in by_modality:
            by_modality[r.modality] = []
        by_modality[r.modality].append(r)

    for modality, items in sorted(by_modality.items()):
        print(f"\n{'='*70}")
        print(f"  {modality.upper()} DELTA CONSISTENCY")
        print(f"{'='*70}")
        print(f"  {'effect':<14} {'intens':<6} {'N':>4}  "
              f"{'cos_mean':>8} {'cos_std':>8} {'cos_min':>8}  "
              f"{'norm_mean':>9} {'norm_cv':>8}  {'dir':>3} {'mag':>3}")
        print(f"  {'-'*14} {'-'*6} {'-'*4}  "
              f"{'-'*8} {'-'*8} {'-'*8}  "
              f"{'-'*9} {'-'*8}  {'-'*3} {'-'*3}")

        # Sort by effect_type, then intensity
        intensity_order = {"low": 0, "mid": 1, "high": 2}
        items.sort(key=lambda r: (r.effect_type, intensity_order.get(r.intensity, 9)))

        for r in items:
            dir_ok = "O" if r.is_direction_consistent else "X"
            mag_ok = "O" if r.is_magnitude_consistent else "X"
            print(f"  {r.effect_type:<14} {r.intensity:<6} {r.n_samples:>4}  "
                  f"{r.mean_pairwise_cosine:>8.4f} {r.std_pairwise_cosine:>8.4f} {r.min_pairwise_cosine:>8.4f}  "
                  f"{r.mean_norm:>9.5f} {r.cv_norm:>8.4f}  {dir_ok:>3} {mag_ok:>3}")


def spearman_intensity_correlation(
    retrieval_metrics: Dict[str, RetrievalMetrics],
) -> Tuple[float, float]:
    """
    Spearman's ρ: Intensity 증가에 따라 Similarity/Matching Score가 단조증가하는지 확인

    Args:
        retrieval_metrics: {intensity: metrics} from evaluate_by_intensity()

    Returns:
        (rho, p_value)
    """
    intensities = ["low", "mid", "high"]
    intensity_map = {i: idx for idx, i in enumerate(intensities)}

    x = [intensity_map[i] for i in retrieval_metrics.keys() if i in intensity_map]
    y = [retrieval_metrics[i].mean_reciprocal_rank for i in retrieval_metrics.keys() if i in intensity_map]

    if len(x) >= 3:
        return stats.spearmanr(x, y)
    else:
        return 0.0, 1.0


def norm_intensity_analysis(
    delta_dataset: DeltaDataset,
) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """
    Norm Analysis: 벡터 크기가 이펙트 강도와 상관관계가 있는지 확인

    Returns:
        {(modality, effect_type): (spearman_rho, p_value)}
    """
    intensity_map = {"low": 0, "mid": 1, "high": 2}
    results = {}

    for modality in ["image", "audio"]:
        effect_types = set(d.effect_type for d in delta_dataset.deltas if d.modality == modality)

        for effect_type in effect_types:
            deltas = [
                d for d in delta_dataset.deltas
                if d.modality == modality and d.effect_type == effect_type
            ]

            x = [intensity_map[d.intensity] for d in deltas]
            y = [np.linalg.norm(d.delta) for d in deltas]

            if len(x) >= 3:
                rho, pval = stats.spearmanr(x, y)
                results[(modality, effect_type)] = (float(rho), float(pval))

    return results
