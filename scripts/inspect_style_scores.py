#!/usr/bin/env python
"""
Inspect soft style labels and similarity scores for inverse-mapping records.

Supports two views:
1) Read stored soft labels from inverse_mapping.h5 (`style_labels`)
2) Recompute CLAP(audio) vs AUD_VOCAB similarities from source/augmented audio

Example:
  ./venv_DeltaV2A/bin/python scripts/inspect_style_scores.py \
    --record-indices 0,1,2 \
    --top-k 8 \
    --mode both
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import sys

sys.path.append(str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect style soft labels/similarity scores.")
    p.add_argument("--h5", type=str, default="outputs/pipeline/inverse_mapping.h5")
    p.add_argument("--aud-vocab", type=str, default="outputs/pipeline/aud_vocab.npz")
    p.add_argument("--manifest", type=str, default="data/augmented/pipeline/audio/manifest.jsonl")
    p.add_argument(
        "--mode",
        type=str,
        choices=["h5", "recompute", "both"],
        default="both",
        help="h5: read stored style_labels, recompute: CLAP+AUD_VOCAB, both: compare.",
    )
    p.add_argument(
        "--record-indices",
        type=str,
        default="0",
        help="Comma-separated record indices (e.g., 0,15,1024).",
    )
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x)
    ex = np.exp(x)
    return (ex / np.sum(ex)).astype(np.float32)


def load_vocab(aud_vocab_path: Path) -> Dict[str, np.ndarray]:
    data = np.load(str(aud_vocab_path), allow_pickle=True)
    terms = data["terms"].tolist()
    keywords = data["keywords"].tolist() if "keywords" in data else terms
    embeddings = data["embeddings"].astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-8)
    return {"terms": np.array(terms, dtype=object), "keywords": np.array(keywords, dtype=object), "embeddings": embeddings}


def load_manifest(manifest_path: Path) -> Dict[int, dict]:
    rows: Dict[int, dict] = {}
    if not manifest_path.exists():
        return rows
    with manifest_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            idx = int(row.get("record_index", -1))
            if idx >= 0:
                rows[idx] = row
    return rows


def read_h5_style_labels(h5_path: Path, indices: List[int]) -> Dict[int, np.ndarray]:
    import h5py

    out: Dict[int, np.ndarray] = {}
    try:
        with h5py.File(str(h5_path), "r") as f:
            ds = f["style_labels"]
            n = ds.shape[0]
            for idx in indices:
                if 0 <= idx < n:
                    out[idx] = ds[idx].astype(np.float32)
    except BlockingIOError:
        raise RuntimeError(
            f"Could not open {h5_path} due to file lock. "
            "If precompute is still running, wait until it finishes and retry."
        )
    return out


def recompute_scores(
    manifest_rows: Dict[int, dict],
    vocab_embeddings: np.ndarray,
    indices: List[int],
    temperature: float,
    device: str,
) -> Dict[int, Dict[str, np.ndarray]]:
    from src.models.clap_embedder import CLAPEmbedder

    clap = CLAPEmbedder(model_id=1, enable_fusion=False, device=device)
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for idx in indices:
        row = manifest_rows.get(idx)
        if row is None:
            continue
        p_aug = row.get("augmented_audio_path")
        p_src = row.get("source_audio_path")
        if not p_aug:
            continue
        aug_path = Path(p_aug)
        if not aug_path.is_absolute():
            aug_path = Path.cwd() / aug_path
        if not aug_path.exists():
            continue

        emb_aug = clap.embed_audio_paths([str(aug_path)])[0].cpu().numpy().astype(np.float32)
        emb_aug = emb_aug / max(float(np.linalg.norm(emb_aug)), 1e-8)
        sims_aug = vocab_embeddings @ emb_aug
        probs_aug = _softmax(sims_aug / float(temperature))

        rec: Dict[str, np.ndarray] = {
            "sims_aug": sims_aug.astype(np.float32),
            "probs_aug": probs_aug,
        }

        if p_src:
            src_path = Path(p_src)
            if not src_path.is_absolute():
                src_path = Path.cwd() / src_path
            if src_path.exists():
                emb_src = clap.embed_audio_paths([str(src_path)])[0].cpu().numpy().astype(np.float32)
                emb_src = emb_src / max(float(np.linalg.norm(emb_src)), 1e-8)
                sims_src = vocab_embeddings @ emb_src
                probs_src = _softmax(sims_src / float(temperature))
                rec["sims_src"] = sims_src.astype(np.float32)
                rec["probs_src"] = probs_src
                rec["delta_sims"] = (sims_aug - sims_src).astype(np.float32)
                rec["delta_probs"] = (probs_aug - probs_src).astype(np.float32)

        out[idx] = rec
    return out


def topk_report(
    vec: np.ndarray,
    sims: Optional[np.ndarray],
    terms: np.ndarray,
    keywords: np.ndarray,
    k: int,
) -> List[str]:
    k = max(1, min(k, vec.shape[0]))
    order = np.argsort(vec)[-k:][::-1]
    lines: List[str] = []
    for rank, i in enumerate(order, start=1):
        if sims is None:
            lines.append(
                f"{rank:>2}. idx={int(i):>2} kw={keywords[i]} prob={float(vec[i]):.6f} term={terms[i]}"
            )
        else:
            lines.append(
                f"{rank:>2}. idx={int(i):>2} kw={keywords[i]} prob={float(vec[i]):.6f} sim={float(sims[i]):.6f} term={terms[i]}"
            )
    return lines


def topk_delta_report(
    delta: np.ndarray,
    terms: np.ndarray,
    keywords: np.ndarray,
    k: int,
    largest: bool = True,
) -> List[str]:
    k = max(1, min(k, delta.shape[0]))
    order = np.argsort(delta)
    if largest:
        picks = order[-k:][::-1]
    else:
        picks = order[:k]
    lines: List[str] = []
    for rank, i in enumerate(picks, start=1):
        lines.append(
            f"{rank:>2}. idx={int(i):>2} kw={keywords[i]} delta={float(delta[i]):+.6f} term={terms[i]}"
        )
    return lines


def main() -> None:
    args = parse_args()
    indices = [int(x.strip()) for x in args.record_indices.split(",") if x.strip()]
    if not indices:
        raise ValueError("--record-indices is empty")

    vocab = load_vocab(Path(args.aud_vocab))
    terms = vocab["terms"]
    keywords = vocab["keywords"]
    vemb = vocab["embeddings"]

    manifest_rows = load_manifest(Path(args.manifest))
    h5_labels: Dict[int, np.ndarray] = {}
    recalc: Dict[int, Dict[str, np.ndarray]] = {}

    if args.mode in ("h5", "both"):
        h5_labels = read_h5_style_labels(Path(args.h5), indices)
    if args.mode in ("recompute", "both"):
        recalc = recompute_scores(
            manifest_rows=manifest_rows,
            vocab_embeddings=vemb,
            indices=indices,
            temperature=args.temperature,
            device=args.device,
        )

    print(f"Indices: {indices}")
    for idx in indices:
        print("\n" + "=" * 80)
        print(f"record_index={idx}")
        row = manifest_rows.get(idx)
        if row:
            print(f"  source_audio_path: {row.get('source_audio_path')}")
            print(f"  augmented_audio_path: {row.get('augmented_audio_path')}")
            print(f"  active_effects: {row.get('active_effects')}")
        else:
            print("  manifest row: not found")

        if idx in h5_labels:
            print("\n  [Stored soft label from H5]")
            for line in topk_report(h5_labels[idx], None, terms, keywords, args.top_k):
                print("   ", line)
        elif args.mode in ("h5", "both"):
            print("\n  [Stored soft label from H5] missing")

        if idx in recalc:
            rec = recalc[idx]

            if "probs_src" in rec and "sims_src" in rec:
                print("\n  [Recomputed from source audio]")
                for line in topk_report(rec["probs_src"], rec["sims_src"], terms, keywords, args.top_k):
                    print("   ", line)

            print("\n  [Recomputed from augmented audio]")
            probs_aug = rec["probs_aug"]
            sims_aug = rec["sims_aug"]
            for line in topk_report(probs_aug, sims_aug, terms, keywords, args.top_k):
                print("   ", line)

            if "delta_probs" in rec and "delta_sims" in rec:
                print("\n  [Delta: augmented - source | prob]")
                for line in topk_delta_report(rec["delta_probs"], terms, keywords, args.top_k, largest=True):
                    print("   ", line)
                print("\n  [Delta: augmented - source | sim]")
                for line in topk_delta_report(rec["delta_sims"], terms, keywords, args.top_k, largest=True):
                    print("   ", line)
        elif args.mode in ("recompute", "both"):
            print("\n  [Recomputed from augmented audio] missing")

        if idx in h5_labels and idx in recalc:
            p1 = h5_labels[idx]
            p2 = recalc[idx]["probs_aug"]
            l1 = float(np.mean(np.abs(p1 - p2)))
            l2 = float(np.sqrt(np.mean((p1 - p2) ** 2)))
            print("\n  [Stored vs Recomputed]")
            print(f"    mean_abs_diff: {l1:.8f}")
            print(f"    rmse:          {l2:.8f}")


if __name__ == "__main__":
    main()
