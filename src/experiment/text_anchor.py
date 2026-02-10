"""
Text Anchor Ensemble Generation (Phase 1)

Creates text-based delta embeddings using synonym expansion and template injection.
"""

import os
# Fix CLAP tokenizer deadlock issue
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class TextAnchor:
    """Text anchor delta embedding."""
    effect_type: str
    modality: str  # "image" or "audio"
    delta: np.ndarray  # (embed_dim,)
    synonyms_used: List[str]
    templates_used: List[str]
    # Representative texts for SBERT cross-modal comparison
    representative_texts: List[str] = field(default_factory=list)


# Global SBERT model cache
_sbert_model = None


def _get_sbert_model():
    """Load SBERT model (cached)."""
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("      Loaded SBERT model (all-MiniLM-L6-v2)")
    return _sbert_model


# Synonym sets for image effects
IMAGE_SYNONYMS = {
    "blur": ["blurry", "out of focus", "fuzzy", "unclear", "hazy", "soft"],
    "brightness": ["bright", "luminous", "vivid", "light", "radiant", "brilliant"],
    "contrast": ["high contrast", "sharp", "crisp", "defined", "distinct"],
    "saturation": ["saturated", "vibrant", "colorful", "vivid", "rich", "intense"],
}

# Synonym sets for audio effects
AUDIO_SYNONYMS = {
    "lpf": ["muffled", "soft", "distant", "low fidelity", "dampened", "dull"],
    "highshelf": ["bright", "crisp", "airy", "clear", "sharp", "brilliant"],
    "saturation": ["distorted", "overdriven", "gritty", "harsh", "aggressive"],
    "reverb": ["spacious", "echoey", "reverberant", "ambient", "roomy"],
}


def generate_text_anchors(
    embedder,  # MultimodalEmbedder or specific embedder
    modality: str,
    effect_types: List[str],
    categories: List[str] = None,
    genres: List[str] = None,
) -> Dict[str, TextAnchor]:
    """
    Generate text anchor delta embeddings for given effect types.

    Method:
    1. Synonym Expansion: Use multiple synonyms for each effect
    2. Context Injection: Use templates with category/genre context
    3. Delta Computation: "A {effect} photo" - "A photo"
    4. Ensemble: Average over synonyms and contexts

    Args:
        embedder: Text embedder (CLIP or CLAP)
        modality: "image" or "audio"
        effect_types: List of effect types to generate anchors for
        categories: Image categories (e.g., ["ocean", "mountain"])
        genres: Audio genres (e.g., ["techno", "ambient"])

    Returns:
        Dictionary mapping effect_type to TextAnchor
    """
    if modality == "image":
        return _generate_image_text_anchors(embedder, effect_types, categories)
    elif modality == "audio":
        return _generate_audio_text_anchors(embedder, effect_types, genres)
    else:
        raise ValueError(f"Unknown modality: {modality}")


def _generate_image_text_anchors(
    embedder,
    effect_types: List[str],
    categories: List[str] = None,
) -> Dict[str, TextAnchor]:
    """Generate text anchors for image effects."""
    anchors = {}

    # Default categories if not provided
    if categories is None:
        categories = ["scene", "landscape", "photo"]

    for effect_type in effect_types:
        if effect_type not in IMAGE_SYNONYMS:
            print(f"Warning: No synonyms defined for '{effect_type}', skipping")
            continue

        synonyms = IMAGE_SYNONYMS[effect_type]
        all_deltas = []
        templates_used = []

        # Generate deltas for each synonym × category combination
        for synonym in synonyms:
            for category in categories:
                # Effected text: "A blurry photo of ocean"
                text_effect = f"A {synonym} photo of {category}"
                # Neutral text: "A photo of ocean"
                text_neutral = f"A photo of {category}"

                try:
                    # Embed both (explicitly pass as list)
                    emb_effect_tensor = embedder.embed_text([text_effect])
                    emb_neutral_tensor = embedder.embed_text([text_neutral])

                    # Convert to numpy
                    if isinstance(emb_effect_tensor, torch.Tensor):
                        emb_effect = emb_effect_tensor[0].detach().cpu().numpy()
                    else:
                        emb_effect = emb_effect_tensor[0]

                    if isinstance(emb_neutral_tensor, torch.Tensor):
                        emb_neutral = emb_neutral_tensor[0].detach().cpu().numpy()
                    else:
                        emb_neutral = emb_neutral_tensor[0]

                    # Compute delta
                    delta = emb_effect - emb_neutral
                    all_deltas.append(delta)

                    templates_used.append(f"{text_effect} - {text_neutral}")

                except Exception as e:
                    print(f"      Warning: Failed to embed '{text_effect}': {e}")
                    continue

        # Average over all deltas to get ensemble
        ensemble_delta = np.mean(all_deltas, axis=0)

        # Representative texts for SBERT (synonym descriptions)
        rep_texts = [f"a {syn} image" for syn in synonyms]

        anchors[effect_type] = TextAnchor(
            effect_type=effect_type,
            modality="image",
            delta=ensemble_delta,
            synonyms_used=synonyms,
            templates_used=templates_used,
            representative_texts=rep_texts,
        )

    return anchors


def _generate_audio_text_anchors(
    embedder,
    effect_types: List[str],
    genres: List[str] = None,
) -> Dict[str, TextAnchor]:
    """Generate text anchors for audio effects."""
    anchors = {}

    # Default genres if not provided
    if genres is None:
        genres = ["music", "song", "track"]

    for effect_type in effect_types:
        if effect_type not in AUDIO_SYNONYMS:
            print(f"Warning: No synonyms defined for '{effect_type}', skipping")
            continue

        synonyms = AUDIO_SYNONYMS[effect_type]
        all_deltas = []
        templates_used = []

        # Generate deltas for each synonym × genre combination
        for synonym in synonyms:
            for genre in genres:
                print(f"      Processing: {synonym} - {genre}...", end="", flush=True)

                # Effected text: "A muffled techno song"
                text_effect = f"A {synonym} {genre} song"
                # Neutral text: "A techno song"
                text_neutral = f"A {genre} song"

                try:
                    # Embed both (explicitly pass as list)
                    emb_effect_tensor = embedder.embed_text([text_effect])
                    emb_neutral_tensor = embedder.embed_text([text_neutral])

                    # Convert to numpy
                    if isinstance(emb_effect_tensor, torch.Tensor):
                        emb_effect = emb_effect_tensor[0].detach().cpu().numpy()
                    else:
                        emb_effect = emb_effect_tensor[0]

                    if isinstance(emb_neutral_tensor, torch.Tensor):
                        emb_neutral = emb_neutral_tensor[0].detach().cpu().numpy()
                    else:
                        emb_neutral = emb_neutral_tensor[0]

                    # Compute delta
                    delta = emb_effect - emb_neutral
                    all_deltas.append(delta)

                    templates_used.append(f"{text_effect} - {text_neutral}")
                    print(" ✓")

                except Exception as e:
                    print(f" ✗ Error: {e}")
                    continue

        # Average over all deltas to get ensemble
        ensemble_delta = np.mean(all_deltas, axis=0)

        # Representative texts for SBERT (synonym descriptions)
        rep_texts = [f"a {syn} sound" for syn in synonyms]

        anchors[effect_type] = TextAnchor(
            effect_type=effect_type,
            modality="audio",
            delta=ensemble_delta,
            synonyms_used=synonyms,
            templates_used=templates_used,
            representative_texts=rep_texts,
        )

    return anchors


def compute_text_anchor_similarity(
    text_anchor1: TextAnchor,
    text_anchor2: TextAnchor,
) -> float:
    """
    Compute cross-modal text anchor similarity using SBERT.

    Since CLIP (768d) and CLAP (512d) embeddings live in different spaces,
    we use SBERT to compare the semantic meaning of the text descriptions.

    Method:
    - Embed representative texts from both anchors with SBERT
    - Compute mean SBERT embedding for each anchor
    - Return cosine similarity between the two means
    """
    from sentence_transformers import util

    sbert = _get_sbert_model()

    texts1 = text_anchor1.representative_texts
    texts2 = text_anchor2.representative_texts

    if not texts1 or not texts2:
        return 0.0

    # Encode with SBERT
    emb1 = sbert.encode(texts1, convert_to_tensor=True)  # (N1, sbert_dim)
    emb2 = sbert.encode(texts2, convert_to_tensor=True)  # (N2, sbert_dim)

    # Mean pooling
    mean1 = emb1.mean(dim=0)  # (sbert_dim,)
    mean2 = emb2.mean(dim=0)  # (sbert_dim,)

    # Cosine similarity
    sim = util.cos_sim(mean1.unsqueeze(0), mean2.unsqueeze(0))

    return float(sim.item())
