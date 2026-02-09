"""
Legacy embedders (preserved but not actively used)

ImageBind weights are preserved in ~/.cache/imagebind/
"""

from .embedder import ImageBindEmbedder

__all__ = ["ImageBindEmbedder"]
