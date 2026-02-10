"""
SBERT Correspondence Matrix (Phase A - Step 2)

Computes semantic similarity between IMG_VOCAB and AUD_VOCAB terms
using Sentence-BERT, creating a cross-modal correspondence bridge.

Matrix C has shape (|IMG_VOCAB|, |AUD_VOCAB|) where C[i,j] is
the SBERT cosine similarity between image term i and audio term j.
"""

import numpy as np
from typing import List, Tuple
from pathlib import Path


_sbert_model = None


def _get_sbert_model(model_name: str = 'all-MiniLM-L6-v2'):
    """Load SBERT model (cached singleton)."""
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer(model_name)
        print(f"Loaded SBERT model ({model_name})")
    return _sbert_model


class CorrespondenceMatrix:
    """
    IMG_VOCAB <-> AUD_VOCAB correspondence matrix.

    Computed once during pre-computation (Phase A),
    used during inference (Phase C, step 3) for cross-modal mapping.
    """

    def __init__(
        self,
        matrix: np.ndarray,
        img_terms: List[str],
        aud_terms: List[str],
    ):
        self.matrix = matrix.astype(np.float32)
        self.img_terms = img_terms
        self.aud_terms = aud_terms

    @property
    def shape(self) -> Tuple[int, int]:
        return self.matrix.shape

    def map_visual_to_audio(self, img_scores: np.ndarray) -> np.ndarray:
        """
        Map visual style scores to audio style scores via correspondence.

        Args:
            img_scores: (|IMG_VOCAB|,) similarity scores from style retrieval

        Returns:
            aud_scores: (|AUD_VOCAB|,) weighted audio style scores (normalized)
        """
        aud_scores = img_scores @ self.matrix
        total = np.abs(aud_scores).sum()
        if total > 0:
            aud_scores = aud_scores / total
        return aud_scores

    def save(self, path: str):
        np.savez(
            path,
            matrix=self.matrix,
            img_terms=np.array(self.img_terms, dtype=object),
            aud_terms=np.array(self.aud_terms, dtype=object),
        )
        print(f"Saved correspondence matrix {self.shape} to {path}")

    @classmethod
    def load(cls, path: str) -> 'CorrespondenceMatrix':
        data = np.load(path, allow_pickle=True)
        return cls(
            matrix=data['matrix'],
            img_terms=data['img_terms'].tolist(),
            aud_terms=data['aud_terms'].tolist(),
        )


def compute_correspondence_matrix(
    img_keywords: List[str],
    aud_keywords: List[str],
    sbert_model_name: str = 'all-MiniLM-L6-v2',
) -> CorrespondenceMatrix:
    """
    Compute SBERT correspondence matrix between vocabularies.

    Uses modality-neutral keywords (not full phrases) so that
    SBERT similarity reflects semantic overlap without being
    biased by modality words ("scene", "audio", etc.).

    Args:
        img_keywords: Image vocabulary keywords (modality-neutral)
        aud_keywords: Audio vocabulary keywords (modality-neutral)
        sbert_model_name: SBERT model to use

    Returns:
        CorrespondenceMatrix instance
    """
    from sentence_transformers import util

    sbert = _get_sbert_model(sbert_model_name)

    print(f"Computing correspondence matrix ({len(img_keywords)} x {len(aud_keywords)})...")

    img_embeddings = sbert.encode(img_keywords, convert_to_tensor=True, show_progress_bar=False)
    aud_embeddings = sbert.encode(aud_keywords, convert_to_tensor=True, show_progress_bar=False)

    sim_matrix = util.cos_sim(img_embeddings, aud_embeddings).cpu().numpy().astype(np.float32)

    print(f"  Shape: {sim_matrix.shape}")
    print(f"  Range: [{sim_matrix.min():.4f}, {sim_matrix.max():.4f}], Mean: {sim_matrix.mean():.4f}")

    return CorrespondenceMatrix(matrix=sim_matrix, img_terms=img_keywords, aud_terms=aud_keywords)
