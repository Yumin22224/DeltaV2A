"""
Model modules for DeltaV2A
"""

from .clip_embedder import CLIPEmbedder
from .clap_embedder import CLAPEmbedder
from .multimodal_embedder import MultimodalEmbedder
from .alignment import CCAAlignment

__all__ = [
    "CLIPEmbedder",
    "CLAPEmbedder",
    "MultimodalEmbedder",
    "CCAAlignment",
]
