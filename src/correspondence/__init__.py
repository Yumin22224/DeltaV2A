"""Cross-modal correspondence via SBERT semantic similarity."""
from .sbert_matrix import CorrespondenceMatrix, compute_correspondence_matrix

__all__ = ["CorrespondenceMatrix", "compute_correspondence_matrix"]
