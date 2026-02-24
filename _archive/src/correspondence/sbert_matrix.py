"""
SBERT Correspondence Matrix (Phase A - Step 2)

Computes semantic similarity between IMG_VOCAB and AUD_VOCAB terms
using Sentence-BERT, creating a cross-modal correspondence bridge.

Matrix C has shape (|IMG_VOCAB|, |AUD_VOCAB|) where C[i,j] is
the SBERT cosine similarity between image term i and audio term j.
"""

import json
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


def _compute_tfidf_fallback_similarity(
    img_keywords: List[str],
    aud_keywords: List[str],
) -> np.ndarray:
    """
    Fallback similarity when SBERT is unavailable (e.g., offline environment).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    corpus = img_keywords + aud_keywords
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        analyzer="word",
    )
    tfidf = vectorizer.fit_transform(corpus)
    img_vecs = tfidf[:len(img_keywords)]
    aud_vecs = tfidf[len(img_keywords):]
    return cosine_similarity(img_vecs, aud_vecs).astype(np.float32)


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
    print(f"Computing correspondence matrix ({len(img_keywords)} x {len(aud_keywords)})...")
    try:
        from sentence_transformers import util

        sbert = _get_sbert_model(sbert_model_name)
        img_embeddings = sbert.encode(img_keywords, convert_to_tensor=True, show_progress_bar=False)
        aud_embeddings = sbert.encode(aud_keywords, convert_to_tensor=True, show_progress_bar=False)
        sim_matrix = util.cos_sim(img_embeddings, aud_embeddings).cpu().numpy().astype(np.float32)
        print(f"  Similarity backend: SBERT ({sbert_model_name})")
    except Exception as e:
        print(f"Warning: SBERT unavailable ({e}). Falling back to TF-IDF cosine similarity.")
        sim_matrix = _compute_tfidf_fallback_similarity(img_keywords, aud_keywords)
        print("  Similarity backend: TF-IDF fallback")

    print(f"  Shape: {sim_matrix.shape}")
    print(f"  Range: [{sim_matrix.min():.4f}, {sim_matrix.max():.4f}], Mean: {sim_matrix.mean():.4f}")

    return CorrespondenceMatrix(matrix=sim_matrix, img_terms=img_keywords, aud_terms=aud_keywords)


def save_correspondence_heatmap(
    correspondence: CorrespondenceMatrix,
    output_path: str,
    title: str = "IMG-Vocab vs AUD-Vocab Similarity (SBERT)",
):
    """
    Save a heatmap image for the correspondence matrix.

    Args:
        correspondence: CorrespondenceMatrix to visualize.
        output_path: PNG output path.
        title: Plot title.
    """
    from PIL import Image, ImageDraw, ImageFont

    mat = correspondence.matrix
    img_labels = correspondence.img_terms
    aud_labels = correspondence.aud_terms

    # Normalize to [0, 1] for color mapping.
    vmin = float(np.min(mat))
    vmax = float(np.max(mat))
    vmean = float(np.mean(mat))
    denom = (vmax - vmin) if vmax > vmin else 1.0
    norm = np.clip((mat - vmin) / denom, 0.0, 1.0)

    # Lightweight viridis-like colormap (piecewise linear RGB interpolation).
    anchors = np.array([
        [0.267, 0.005, 0.329],  # dark purple
        [0.283, 0.141, 0.458],
        [0.254, 0.265, 0.530],
        [0.207, 0.372, 0.553],
        [0.164, 0.471, 0.558],
        [0.128, 0.567, 0.551],
        [0.135, 0.659, 0.518],
        [0.267, 0.749, 0.441],
        [0.478, 0.821, 0.318],
        [0.741, 0.873, 0.150],  # yellow
    ], dtype=np.float32)

    x = norm * (len(anchors) - 1)
    lo = np.floor(x).astype(np.int32)
    hi = np.clip(lo + 1, 0, len(anchors) - 1)
    w = (x - lo)[..., None]
    rgb = (1.0 - w) * anchors[lo] + w * anchors[hi]
    rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    # --- Render annotated heatmap canvas ---
    rows, cols = mat.shape
    cell = 28 if max(rows, cols) <= 30 else 22
    heat_w = cols * cell
    heat_h = rows * cell

    # Conservative margins sized for label readability.
    left_margin = max(260, max(len(t) for t in img_labels) * 7 + 30)
    top_margin = 90
    bottom_margin = max(260, max(len(t) for t in aud_labels) * 6 + 40)
    right_margin = 160

    canvas_w = left_margin + heat_w + right_margin
    canvas_h = top_margin + heat_h + bottom_margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    # Heatmap image (nearest-neighbor upscale).
    heat = Image.fromarray(rgb_u8, mode="RGB").resize(
        (heat_w, heat_h),
        resample=Image.Resampling.NEAREST,
    )
    x0, y0 = left_margin, top_margin
    x1, y1 = x0 + heat_w, y0 + heat_h
    canvas.paste(heat, (x0, y0))

    # Grid lines and frame.
    for r in range(rows + 1):
        yy = y0 + r * cell
        draw.line([(x0, yy), (x1, yy)], fill=(215, 215, 215), width=1)
    for c in range(cols + 1):
        xx = x0 + c * cell
        draw.line([(xx, y0), (xx, y1)], fill=(215, 215, 215), width=1)
    draw.rectangle([x0, y0, x1, y1], outline=(80, 80, 80), width=2)

    # Cell numeric annotations.
    for r in range(rows):
        for c in range(cols):
            val = float(mat[r, c])
            label = f"{val:.2f}"
            cx = x0 + c * cell + cell // 2
            cy = y0 + r * cell + cell // 2
            tw, th = draw.textbbox((0, 0), label, font=font)[2:]
            # Use contrasting text color against cell brightness.
            rr, gg, bb = rgb_u8[r, c]
            luminance = 0.2126 * rr + 0.7152 * gg + 0.0722 * bb
            txt_color = (20, 20, 20) if luminance > 150 else (245, 245, 245)
            draw.text((cx - tw / 2, cy - th / 2), label, fill=txt_color, font=font)

    # Title and axis names.
    title_text = f"{title}"
    stats_text = f"min={vmin:.4f}  max={vmax:.4f}  mean={vmean:.4f}"
    draw.text((x0, 20), title_text, fill=(20, 20, 20), font=font)
    draw.text((x0, 40), stats_text, fill=(40, 40, 40), font=font)
    draw.text((x0 + heat_w // 2 - 55, y1 + bottom_margin - 24), "Audio Keywords", fill=(20, 20, 20), font=font)
    draw.text((20, y0 - 22), "Image Keywords", fill=(20, 20, 20), font=font)

    # Y-axis tick labels.
    for r, label in enumerate(img_labels):
        yy = y0 + r * cell + cell // 2
        draw.line([(x0 - 6, yy), (x0, yy)], fill=(60, 60, 60), width=1)
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        draw.text((x0 - 10 - tw, yy - th / 2), label, fill=(25, 25, 25), font=font)

    # X-axis tick labels (vertical to avoid overlap).
    for c, label in enumerate(aud_labels):
        xx = x0 + c * cell + cell // 2
        draw.line([(xx, y1), (xx, y1 + 6)], fill=(60, 60, 60), width=1)
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        txt = Image.new("RGBA", (tw + 4, th + 4), (255, 255, 255, 0))
        txt_draw = ImageDraw.Draw(txt)
        txt_draw.text((2, 2), label, fill=(25, 25, 25), font=font)
        rot = txt.rotate(270, expand=True)
        canvas.paste(rot, (int(xx - rot.width / 2), y1 + 10), rot)

    # Right color bar.
    bar_w = 26
    bar_x0 = x1 + 36
    bar_x1 = bar_x0 + bar_w
    bar = np.linspace(1.0, 0.0, heat_h, dtype=np.float32)[:, None]
    bx = bar * (len(anchors) - 1)
    blo = np.floor(bx).astype(np.int32)
    bhi = np.clip(blo + 1, 0, len(anchors) - 1)
    bw = (bx - blo)[..., None]
    brgb = (1.0 - bw) * anchors[blo] + bw * anchors[bhi]
    brgb_u8 = np.clip(brgb * 255.0, 0, 255).astype(np.uint8).reshape(heat_h, 1, 3)
    bar_img = Image.fromarray(brgb_u8, mode="RGB").resize((bar_w, heat_h), resample=Image.Resampling.BILINEAR)
    canvas.paste(bar_img, (bar_x0, y0))
    draw.rectangle([bar_x0, y0, bar_x1, y1], outline=(80, 80, 80), width=1)
    draw.text((bar_x1 + 8, y0 - 2), f"{vmax:.3f}", fill=(20, 20, 20), font=font)
    draw.text((bar_x1 + 8, y0 + heat_h // 2 - 6), f"{vmean:.3f}", fill=(20, 20, 20), font=font)
    draw.text((bar_x1 + 8, y1 - 10), f"{vmin:.3f}", fill=(20, 20, 20), font=font)
    draw.text((bar_x0 - 4, y0 - 20), "Cosine Sim", fill=(20, 20, 20), font=font)

    # Save row/column labels to sidecar JSON for traceability.
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    sidecar = out_path.with_suffix(".labels.json")
    np.savez(out_path.with_suffix(".stats.npz"), vmin=np.array([vmin]), vmax=np.array([vmax]))
    sidecar.write_text(
        json.dumps(
            {
                "title": title,
                "rows_image_keywords": correspondence.img_terms,
                "cols_audio_keywords": correspondence.aud_terms,
                "vmin": vmin,
                "vmax": vmax,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Saved correspondence heatmap to {out_path}")
