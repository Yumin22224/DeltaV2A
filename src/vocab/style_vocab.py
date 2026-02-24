"""
Style Vocabulary (Phase A - Step 1)

IMG_VOCAB: Visual style terms embedded via CLIP text encoder -> (|V_img|, 768)
AUD_VOCAB: Audio style terms embedded via CLAP text encoder -> (|V_aud|, 512)

Each term has:
  - phrase: Full sentence for CLIP/CLAP embedding (modality-specific context)
  - keyword: Core concept token (modality-neutral)
"""

import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass


# Common cross-modal style axes.
# We keep identical concept order for image/audio vocabularies.
STYLE_AXES_PHASES: List[Tuple[str, List[Tuple[str, str]]]] = [
    ("phase_1", [
        ("warm", "cold"),
        ("bright", "dark"),
        ("rough", "smooth"),
        ("heavy", "light"),
    ]),
    ("phase_2", [
        ("thick", "thin"),
        ("distant", "intimate"),
        ("soft", "hard"),
        ("static", "dynamic"),
    ]),
    ("phase_3_abstract", [
        ("clean", "dirty"),
        ("dreamy", "realistic"),
        ("vintage", "modern"),
        ("natural", "surreal"),
    ]),
]


def _flatten_axes_words(phases: List[Tuple[str, List[Tuple[str, str]]]]) -> List[str]:
    words: List[str] = []
    for _, pairs in phases:
        for left, right in pairs:
            words.extend([left, right])
    return words


COMMON_STYLE_WORDS: List[str] = _flatten_axes_words(STYLE_AXES_PHASES)

# Prompt templates for text-embedding ensemble.
# `IMG_VOCAB` / `AUD_VOCAB` below keep the canonical single-template phrase
# for backwards compatibility and readable term names.
DEFAULT_IMG_PROMPT_TEMPLATES: List[str] = [
    "a {word} image",
    "a {word} photograph",
    "an image with a {word} mood",
    "a {word} visual style",
    "a {word} scene",
    "a {word} picture",
    "a photo that feels {word}",
    "artwork with a {word} atmosphere",
]

DEFAULT_AUD_PROMPT_TEMPLATES: List[str] = [
    "a {word} sound",
    "a {word} music track",
    "audio with a {word} mood",
    "a {word} sonic texture",
    "a {word} musical atmosphere",
    "a {word} sounding track",
    "music that feels {word}",
    "a {word} audio style",
]

# === Visual style vocabulary: (phrase, keyword) ===
IMG_VOCAB: List[Tuple[str, str]] = [
    (f"a {word} image", word) for word in COMMON_STYLE_WORDS
]

# === Audio style vocabulary: (phrase, keyword) ===
AUD_VOCAB: List[Tuple[str, str]] = [
    (f"a {word} sound", word) for word in COMMON_STYLE_WORDS
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
    keywords: List[str]    # modality-neutral concept tokens
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

    def _normalize_prompt_templates(self, templates: Optional[List[str]]) -> List[str]:
        if templates is None:
            return []
        out: List[str] = []
        for t in templates:
            s = str(t).strip()
            if not s:
                continue
            if "{word}" not in s:
                raise ValueError(f"Prompt template must include '{{word}}': {s}")
            out.append(s)
        return out

    def _build_ensemble_embeddings(
        self,
        text_embedder,
        keywords: List[str],
        prompt_templates: List[str],
    ) -> np.ndarray:
        if len(prompt_templates) == 0:
            raise ValueError("prompt_templates must be non-empty for ensemble build")
        prompts: List[str] = []
        for kw in keywords:
            for tmpl in prompt_templates:
                prompts.append(tmpl.format(word=kw))

        prompt_emb = text_embedder.embed_text(prompts).cpu().numpy().astype(np.float32)
        n_words = len(keywords)
        n_templates = len(prompt_templates)
        prompt_emb = prompt_emb.reshape(n_words, n_templates, -1)

        # Normalize each prompt embedding, then average across templates.
        prompt_emb = prompt_emb / np.maximum(np.linalg.norm(prompt_emb, axis=2, keepdims=True), 1e-8)
        emb = prompt_emb.mean(axis=1)
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
        return emb

    def build_img_vocab(
        self,
        clip_embedder,
        vocab: Optional[List[Tuple[str, str]]] = None,
        prompt_templates: Optional[List[str]] = None,
    ) -> VocabEmbeddings:
        """Build IMG_VOCAB by embedding visual style phrases with CLIP."""
        if vocab is None:
            vocab = IMG_VOCAB
        keywords = [k for _, k in vocab]
        templates = self._normalize_prompt_templates(prompt_templates)

        if len(templates) == 0:
            phrases = [p for p, _ in vocab]
            print(f"Building IMG_VOCAB ({len(phrases)} terms)...")
            embeddings = clip_embedder.embed_text(phrases).cpu().numpy().astype(np.float32)

            # L2 normalize for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-8)
        else:
            print(
                f"Building IMG_VOCAB ({len(keywords)} terms) with prompt ensemble "
                f"({len(templates)} templates)..."
            )
            embeddings = self._build_ensemble_embeddings(clip_embedder, keywords, templates)
            # Keep a readable representative phrase per term.
            phrases = [templates[0].format(word=k) for k in keywords]

        self.img_vocab = VocabEmbeddings(
            terms=phrases,
            keywords=keywords,
            embeddings=embeddings,
            modality="image",
        )
        print(f"  IMG_VOCAB: {self.img_vocab.size} terms, {self.img_vocab.embed_dim}d")
        return self.img_vocab

    def build_aud_vocab(
        self,
        clap_embedder,
        vocab: Optional[List[Tuple[str, str]]] = None,
        prompt_templates: Optional[List[str]] = None,
    ) -> VocabEmbeddings:
        """Build AUD_VOCAB by embedding audio style phrases with CLAP."""
        if vocab is None:
            vocab = AUD_VOCAB
        keywords = [k for _, k in vocab]
        templates = self._normalize_prompt_templates(prompt_templates)

        if len(templates) == 0:
            phrases = [p for p, _ in vocab]
            print(f"Building AUD_VOCAB ({len(phrases)} terms)...")
            embeddings = clap_embedder.embed_text(phrases).cpu().numpy().astype(np.float32)

            # L2 normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-8)
        else:
            print(
                f"Building AUD_VOCAB ({len(keywords)} terms) with prompt ensemble "
                f"({len(templates)} templates)..."
            )
            embeddings = self._build_ensemble_embeddings(clap_embedder, keywords, templates)
            phrases = [templates[0].format(word=k) for k in keywords]

        self.aud_vocab = VocabEmbeddings(
            terms=phrases,
            keywords=keywords,
            embeddings=embeddings,
            modality="audio",
        )
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
