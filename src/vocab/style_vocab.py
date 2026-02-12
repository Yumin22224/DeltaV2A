"""
Style Vocabulary (Phase A - Step 1)

IMG_VOCAB: Visual style terms embedded via CLIP text encoder -> (|V_img|, 768)
AUD_VOCAB: Audio style terms embedded via CLAP text encoder -> (|V_aud|, 512)

Each term has:
  - phrase: Full sentence for CLIP/CLAP embedding (modality-specific context)
  - keyword: Core concept for SBERT correspondence (modality-neutral)
"""

import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass


# === Visual style vocabulary: (phrase, keyword) ===
# Broad visual descriptors (not tightly bound to specific processing operators).
IMG_VOCAB: List[Tuple[str, str]] = [
    ("a soft-focus dreamy photograph", "soft-focus dreamy"),
    ("a motion-streaked dynamic photograph", "motion-streaked dynamic"),
    ("a crisp and highly detailed image", "crisp high detail"),
    ("a grainy and textured image", "grainy textured"),
    ("a smeared and painterly image", "smeared painterly"),
    ("a vintage warm-toned image", "vintage warm-toned"),
    ("a surreal color-shifted image", "surreal color-shifted"),
    ("a clear and neutral realistic image", "clear neutral realistic"),
    ("a clean low-noise image", "clean low-noise"),
    ("a high-contrast dramatic image", "dramatic high-contrast"),
    ("a low-contrast flat image", "flat low-contrast"),
    ("a vivid and saturated color image", "vivid saturated color"),
    ("a muted and desaturated color image", "muted desaturated color"),
    ("a bright luminous exposure", "bright luminous exposure"),
    ("a dim and moody exposure", "dim moody exposure"),
    ("a cool bluish color cast", "cool bluish color cast"),
    ("a warm golden color cast", "warm golden color cast"),
    ("a hazy atmospheric image", "hazy atmospheric"),
    ("an edge-sharp high-clarity image", "edge-sharp high-clarity"),
    ("a smooth and refined texture image", "smooth refined texture"),
    ("a monochrome black-and-white image", "monochrome black-and-white"),
    ("a faded film-like image", "faded film-like"),
    ("a glossy polished image finish", "glossy polished finish"),
    ("a matte soft image finish", "matte soft finish"),
]

# === Audio style vocabulary: (phrase, keyword) ===
# Broad audio descriptors (not tightly bound to specific processing operators).
AUD_VOCAB: List[Tuple[str, str]] = [
    ("warm and full-bodied audio", "warm full-bodied"),
    ("bright and airy audio", "bright airy"),
    ("dark and muffled audio", "dark muffled"),
    ("clean and transparent audio", "clean transparent"),
    ("gritty and rough audio texture", "gritty rough texture"),
    ("punchy and transient-forward audio", "punchy transient-forward"),
    ("soft and rounded transients", "soft rounded transients"),
    ("tight dry and close audio", "tight dry close"),
    ("spacious wide and ambient audio", "spacious wide ambient"),
    ("distant and washed-out audio", "distant washed-out"),
    ("thick dense layered audio", "thick dense layered"),
    ("thin and lightweight audio", "thin lightweight"),
    ("smooth polished audio finish", "smooth polished finish"),
    ("noisy textured lo-fi audio", "noisy textured lo-fi"),
    ("stable and centered audio image", "stable centered"),
    ("swirling moving stereo image", "swirling moving stereo"),
    ("resonant and ringing audio", "resonant ringing"),
    ("echoing and repeating tails", "echoing repeating tails"),
    ("deep heavy low-end audio", "deep heavy low-end"),
    ("light and lean low-end audio", "light lean low-end"),
    ("forward upfront audio presence", "forward upfront presence"),
    ("recessed background audio presence", "recessed background presence"),
    ("sharp and edgy timbre", "sharp edgy timbre"),
    ("gentle mellow timbre", "gentle mellow timbre"),
]

# Convenience accessors
IMG_VOCAB_PHRASES: List[str] = [phrase for phrase, _ in IMG_VOCAB]
IMG_VOCAB_KEYWORDS: List[str] = [kw for _, kw in IMG_VOCAB]
AUD_VOCAB_PHRASES: List[str] = [phrase for phrase, _ in AUD_VOCAB]
AUD_VOCAB_KEYWORDS: List[str] = [kw for _, kw in AUD_VOCAB]


@dataclass
class VocabEmbeddings:
    """Pre-computed vocabulary embeddings."""
    terms: List[str]       # phrases (for CLIP/CLAP embedding)
    keywords: List[str]    # keywords (for SBERT correspondence)
    embeddings: np.ndarray  # (|V|, embed_dim)
    modality: str  # "image" or "audio"

    @property
    def size(self) -> int:
        return len(self.terms)

    @property
    def embed_dim(self) -> int:
        return self.embeddings.shape[1]

    def save(self, path: str):
        np.savez(
            path,
            terms=np.array(self.terms, dtype=object),
            keywords=np.array(self.keywords, dtype=object),
            embeddings=self.embeddings,
            modality=self.modality,
        )
        print(f"Saved {self.modality} vocab ({self.size} terms, {self.embed_dim}d) to {path}")

    @classmethod
    def load(cls, path: str) -> 'VocabEmbeddings':
        data = np.load(path, allow_pickle=True)
        keywords = data['keywords'].tolist() if 'keywords' in data else data['terms'].tolist()
        return cls(
            terms=data['terms'].tolist(),
            keywords=keywords,
            embeddings=data['embeddings'].astype(np.float32),
            modality=str(data['modality']),
        )


class StyleVocabulary:
    """
    Manages style vocabularies and their embeddings.

    Computes CLIP embeddings for IMG_VOCAB and CLAP embeddings for AUD_VOCAB.
    """

    def __init__(self):
        self.img_vocab: Optional[VocabEmbeddings] = None
        self.aud_vocab: Optional[VocabEmbeddings] = None

    def build_img_vocab(
        self,
        clip_embedder,
        vocab: Optional[List[Tuple[str, str]]] = None,
    ) -> VocabEmbeddings:
        """Build IMG_VOCAB by embedding visual style phrases with CLIP."""
        if vocab is None:
            vocab = IMG_VOCAB
        phrases = [p for p, _ in vocab]
        keywords = [k for _, k in vocab]

        print(f"Building IMG_VOCAB ({len(phrases)} terms)...")
        embeddings = clip_embedder.embed_text(phrases).cpu().numpy().astype(np.float32)

        # L2 normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-8)

        self.img_vocab = VocabEmbeddings(terms=phrases, keywords=keywords, embeddings=embeddings, modality="image")
        print(f"  IMG_VOCAB: {self.img_vocab.size} terms, {self.img_vocab.embed_dim}d")
        return self.img_vocab

    def build_aud_vocab(
        self,
        clap_embedder,
        vocab: Optional[List[Tuple[str, str]]] = None,
    ) -> VocabEmbeddings:
        """Build AUD_VOCAB by embedding audio style phrases with CLAP."""
        if vocab is None:
            vocab = AUD_VOCAB
        phrases = [p for p, _ in vocab]
        keywords = [k for _, k in vocab]

        print(f"Building AUD_VOCAB ({len(phrases)} terms)...")
        embeddings = clap_embedder.embed_text(phrases).cpu().numpy().astype(np.float32)

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-8)

        self.aud_vocab = VocabEmbeddings(terms=phrases, keywords=keywords, embeddings=embeddings, modality="audio")
        print(f"  AUD_VOCAB: {self.aud_vocab.size} terms, {self.aud_vocab.embed_dim}d")
        return self.aud_vocab

    def retrieve_img_style(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[int, str, float]]:
        """
        Retrieve top-k image styles from IMG_VOCAB.

        Args:
            query_embedding: (embed_dim,) L2-normalized query vector
            top_k: Number of results

        Returns:
            List of (index, term, similarity_score)
        """
        if self.img_vocab is None:
            raise RuntimeError("IMG_VOCAB not built. Call build_img_vocab() first.")

        sims = self.img_vocab.embeddings @ query_embedding
        top_indices = np.argsort(sims)[-top_k:][::-1]
        return [(int(i), self.img_vocab.terms[i], float(sims[i])) for i in top_indices]

    def save(self, output_dir: str):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if self.img_vocab is not None:
            self.img_vocab.save(str(out / "img_vocab.npz"))
        if self.aud_vocab is not None:
            self.aud_vocab.save(str(out / "aud_vocab.npz"))

    def load(self, output_dir: str):
        out = Path(output_dir)
        img_path = out / "img_vocab.npz"
        aud_path = out / "aud_vocab.npz"
        if img_path.exists():
            self.img_vocab = VocabEmbeddings.load(str(img_path))
            print(f"Loaded IMG_VOCAB: {self.img_vocab.size} terms, {self.img_vocab.embed_dim}d")
        if aud_path.exists():
            self.aud_vocab = VocabEmbeddings.load(str(aud_path))
            print(f"Loaded AUD_VOCAB: {self.aud_vocab.size} terms, {self.aud_vocab.embed_dim}d")
