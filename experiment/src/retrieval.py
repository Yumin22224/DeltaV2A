"""
Retrieval Module

Delta-to-delta retrieval and evaluation metrics.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

from .delta_extraction import DeltaDataset, DeltaResult


@dataclass
class RetrievalResult:
    """Result of a single retrieval query."""
    query_modality: str
    query_effect_type: str
    query_intensity: str
    query_path: str
    retrieved_indices: List[int]  # indices in database, sorted by similarity
    similarities: List[float]  # corresponding similarities
    ground_truth_effect_type: str  # expected matching effect type
    ground_truth_indices: List[int]  # indices of ground truth matches in database


@dataclass
class RetrievalMetrics:
    """Aggregated retrieval metrics."""
    top_k_accuracy: Dict[int, float]  # {k: accuracy}
    mean_reciprocal_rank: float
    mean_rank: float
    hit_at_k: Dict[int, int]  # {k: count of hits}
    total_queries: int


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_matrix(queries: np.ndarray, database: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix.

    Args:
        queries: (N, D) query vectors
        database: (M, D) database vectors

    Returns:
        similarities: (N, M) similarity matrix
    """
    # Normalize
    q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    d_norms = np.linalg.norm(database, axis=1, keepdims=True)

    q_norms = np.where(q_norms > 0, q_norms, 1.0)
    d_norms = np.where(d_norms > 0, d_norms, 1.0)

    queries_norm = queries / q_norms
    database_norm = database / d_norms

    return queries_norm @ database_norm.T


class DeltaRetrieval:
    """
    Delta-to-delta retrieval system.

    Query: Image delta (Δe_V)
    Database: Audio deltas (Δe_A)
    """

    def __init__(
        self,
        query_dataset: DeltaDataset,  # Image deltas
        database_dataset: DeltaDataset,  # Audio deltas
        alignment,  # CCAAlignment object
        effect_type_mapping: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            query_dataset: Dataset of image deltas (queries)
            database_dataset: Dataset of audio deltas (database)
            alignment: CCAAlignment for cross-modal alignment (required)
            effect_type_mapping: Mapping from image effect to expected audio effect
                                 e.g., {"blur": "lpf", "brightness": "highshelf"}
        """
        self.query_deltas = query_dataset.deltas
        self.database_deltas = database_dataset.deltas
        self.alignment = alignment
        self.effect_type_mapping = effect_type_mapping or {}

        # Build database matrix and align
        database_matrix_raw = np.stack([d.delta for d in self.database_deltas])
        self.database_matrix = self.alignment.transform_audio(database_matrix_raw)  # Align audio

    def retrieve(
        self,
        query_delta: np.ndarray,
        top_k: int = 10,
        normalize: bool = True,
    ) -> Tuple[List[int], List[float]]:
        """
        Retrieve top-k most similar audio deltas.

        Args:
            query_delta: (D,) query delta vector (image delta, 768d)
            top_k: Number of results to return
            normalize: Whether to L2-normalize before computing similarity

        Returns:
            indices: Top-k indices in database
            similarities: Corresponding similarity scores
        """
        # Align query (image delta) to shared space
        query_aligned = self.alignment.transform_image(query_delta.reshape(1, -1))[0]

        if normalize:
            query_norm = np.linalg.norm(query_aligned)
            if query_norm > 0:
                query_aligned = query_aligned / query_norm

            db_norms = np.linalg.norm(self.database_matrix, axis=1, keepdims=True)
            db_norms = np.where(db_norms > 0, db_norms, 1.0)
            database = self.database_matrix / db_norms
        else:
            database = self.database_matrix

        # Compute similarities
        similarities = database @ query_aligned

        # Sort by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1]
        top_indices = sorted_indices[:top_k].tolist()
        top_sims = similarities[sorted_indices[:top_k]].tolist()

        return top_indices, top_sims

    def evaluate(
        self,
        query_effect_types: Optional[List[str]] = None,
        query_intensities: Optional[List[str]] = None,
        k_values: List[int] = [1, 3, 5, 10],
        normalize: bool = True,
    ) -> RetrievalMetrics:
        """
        Evaluate retrieval performance.

        Ground truth: Retrieved delta should have matching effect type
        (based on effect_type_mapping).

        Args:
            query_effect_types: Filter queries by effect type
            query_intensities: Filter queries by intensity
            k_values: Values of k for top-k accuracy
            normalize: Whether to normalize deltas

        Returns:
            RetrievalMetrics with all metrics
        """
        # Filter queries
        queries = self.query_deltas
        if query_effect_types:
            queries = [q for q in queries if q.effect_type in query_effect_types]
        if query_intensities:
            queries = [q for q in queries if q.intensity in query_intensities]

        if not queries:
            return RetrievalMetrics(
                top_k_accuracy={k: 0.0 for k in k_values},
                mean_reciprocal_rank=0.0,
                mean_rank=float('inf'),
                hit_at_k={k: 0 for k in k_values},
                total_queries=0,
            )

        hits = {k: 0 for k in k_values}
        reciprocal_ranks = []
        ranks = []

        for query in queries:
            # Get ground truth effect type
            gt_effect_type = self.effect_type_mapping.get(
                query.effect_type,
                query.effect_type  # Use same name if no mapping
            )

            # Find ground truth indices in database
            gt_indices = [
                i for i, d in enumerate(self.database_deltas)
                if d.effect_type == gt_effect_type
            ]

            if not gt_indices:
                continue

            # Retrieve
            retrieved_indices, _ = self.retrieve(
                query.delta,
                top_k=max(k_values),
                normalize=normalize,
            )

            # Check hits at various k
            for k in k_values:
                top_k_retrieved = set(retrieved_indices[:k])
                if top_k_retrieved.intersection(gt_indices):
                    hits[k] += 1

            # Find rank of first ground truth hit
            for rank, idx in enumerate(retrieved_indices, 1):
                if idx in gt_indices:
                    reciprocal_ranks.append(1.0 / rank)
                    ranks.append(rank)
                    break
            else:
                reciprocal_ranks.append(0.0)
                ranks.append(len(self.database_deltas))

        total = len(queries)

        return RetrievalMetrics(
            top_k_accuracy={k: hits[k] / total for k in k_values},
            mean_reciprocal_rank=float(np.mean(reciprocal_ranks)),
            mean_rank=float(np.mean(ranks)),
            hit_at_k=hits,
            total_queries=total,
        )

    def evaluate_by_intensity(
        self,
        k_values: List[int] = [1, 3, 5, 10],
        normalize: bool = True,
    ) -> Dict[str, RetrievalMetrics]:
        """
        Evaluate retrieval separately for each intensity level.

        Returns:
            Dict mapping intensity -> RetrievalMetrics
        """
        intensities = list(set(q.intensity for q in self.query_deltas))
        results = {}

        for intensity in intensities:
            results[intensity] = self.evaluate(
                query_intensities=[intensity],
                k_values=k_values,
                normalize=normalize,
            )

        return results

    def get_retrieval_matrix(
        self,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, List[DeltaResult], List[DeltaResult]]:
        """
        Get full similarity matrix between all queries and database.

        Returns:
            sim_matrix: (num_queries, num_database)
            query_results: List of query DeltaResults
            database_results: List of database DeltaResults
        """
        query_matrix = np.stack([q.delta for q in self.query_deltas])

        sim_matrix = cosine_similarity_matrix(query_matrix, self.database_matrix)

        return sim_matrix, self.query_deltas, self.database_deltas
