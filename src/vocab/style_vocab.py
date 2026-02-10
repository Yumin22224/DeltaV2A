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
IMG_VOCAB: List[Tuple[str, str]] = [
    ("a bright and luminous scene", "bright and luminous"),
    ("a dark and moody scene", "dark and moody"),
    ("a warm-toned photograph", "warm-toned"),
    ("a cool-toned photograph", "cool-toned"),
    ("a hazy and dreamlike scene", "hazy and dreamlike"),
    ("a sharp and crisp photograph", "sharp and crisp"),
    ("a soft and diffused scene", "soft and diffused"),
    ("a high contrast dramatic scene", "high contrast dramatic"),
    ("a muted and desaturated scene", "muted and desaturated"),
    ("a vibrant and colorful scene", "vibrant and colorful"),
    ("a grainy vintage photograph", "grainy vintage"),
    ("a smooth and clean image", "smooth and clean"),
    ("a blurry out-of-focus scene", "blurry out-of-focus"),
    ("an overexposed washed-out scene", "overexposed washed-out"),
    ("an underexposed dark scene", "underexposed dark"),
    ("a vast expansive landscape", "vast and expansive"),
    ("a foggy atmospheric scene", "foggy and atmospheric"),
    ("a clear and detailed scene", "clear and detailed"),
    ("a minimalist sparse scene", "minimalist and sparse"),
    ("a noisy and textured scene", "noisy and textured"),
]

# === Audio style vocabulary: (phrase, keyword) ===
AUD_VOCAB: List[Tuple[str, str]] = [
    ("bright and crisp sounding audio", "bright and crisp"),
    ("dark and muffled sounding audio", "dark and muffled"),
    ("warm and smooth sounding audio", "warm and smooth"),
    ("cold and thin sounding audio", "cold and thin"),
    ("harsh and distorted audio", "harsh and distorted"),
    ("clean and pristine audio", "clean and pristine"),
    ("rich and full-bodied audio", "rich and full-bodied"),
    ("tinny and hollow audio", "tinny and hollow"),
    ("spacious and reverberant audio", "spacious and reverberant"),
    ("dry and intimate audio", "dry and intimate"),
    ("echoey and ambient sound", "echoey and ambient"),
    ("close and direct sound", "close and direct"),
    ("loud and compressed audio", "loud and compressed"),
    ("soft and dynamic audio", "soft and dynamic"),
    ("punchy and impactful audio", "punchy and impactful"),
    ("gentle and subtle audio", "gentle and subtle"),
    ("grainy and lo-fi audio", "grainy and lo-fi"),
    ("smooth and polished audio", "smooth and polished"),
    ("noisy and gritty audio", "noisy and gritty"),
    ("quiet and minimal audio", "quiet and minimal"),
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
