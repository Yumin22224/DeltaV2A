"""
Model modules for DeltaV2A
"""

from .clip_embedder import CLIPEmbedder
from .clap_embedder import CLAPEmbedder
from .multimodal_embedder import MultimodalEmbedder
from .alignment import CCAAlignment

# Legacy (ImageBind moved to legacy/)
# from .embedder import ImageBindEmbedder

__all__ = [
    "CLIPEmbedder",
    "CLAPEmbedder",
    "MultimodalEmbedder",
    "CCAAlignment",
]
