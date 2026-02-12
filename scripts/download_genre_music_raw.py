#!/usr/bin/env python
"""
Download a small multi-genre raw music set into data/raw using GTZAN.

GTZAN contains 30-second excerpts and is commonly used for MIR research.
This script downloads once into a cache directory, then copies per-genre
subsets into an easy-to-inspect folder structure.

Example:
  ./venv_DeltaV2A/bin/python scripts/download_genre_music_raw.py \
      --count-per-genre 50
"""

import argparse
import json
import random
import shutil
from pathlib import Path

from torchaudio.datasets import GTZAN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download genre-balanced raw music subset (GTZAN).")
    p.add_argument("--cache-dir", type=str, default="data/raw/gtzan_cache")
    p.add_argument("--output-dir", type=str, default="data/raw/genre_music")
    p.add_argument("--genres", type=str, default="classical,jazz,pop,rock")
    p.add_argument("--count-per-genre", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clean-output", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.output_dir)
    genres = [g.strip().lower() for g in args.genres.split(",") if g.strip()]
    count = int(args.count_per_genre)
    rng = random.Random(args.seed)

    if count <= 0:
        raise ValueError("--count-per-genre must be > 0")
    if not genres:
        raise ValueError("--genres is empty")

    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_output:
        for p in out_dir.iterdir():
            if p.is_file():
                p.unlink()
            else:
                shutil.rmtree(p)

    print("Preparing GTZAN dataset download/cache...")
    # Trigger download if missing.
    GTZAN(root=str(cache_dir), download=True)
    gtzan_root = cache_dir / "genres"
    if not gtzan_root.exists():
        raise FileNotFoundError(f"Expected GTZAN folder not found: {gtzan_root}")

    manifest_rows = []
    summary = {}

    for genre in genres:
        src_dir = gtzan_root / genre
        if not src_dir.exists():
            print(f"[warn] genre not found in GTZAN: {genre}")
            continue

        files = sorted(src_dir.glob("*.wav"))
        if not files:
            print(f"[warn] no files in {src_dir}")
            continue

        choose_n = min(count, len(files))
        selected = rng.sample(files, choose_n)

        genre_out = out_dir / genre
        genre_out.mkdir(parents=True, exist_ok=True)

        for i, src in enumerate(sorted(selected), start=1):
            dst = genre_out / f"{genre}_{i:03d}.wav"
            shutil.copy2(src, dst)
            manifest_rows.append(
                {
                    "dataset": "GTZAN",
                    "genre": genre,
                    "source_path": str(src),
                    "output_path": str(dst),
                }
            )

        summary[genre] = {
            "available": len(files),
            "selected": choose_n,
            "output_dir": str(genre_out),
        }
        print(f"[ok] {genre}: selected {choose_n}/{len(files)}")

    (out_dir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in manifest_rows) + ("\n" if manifest_rows else "")
    )
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "dataset": "GTZAN",
                "genres": genres,
                "count_per_genre_requested": count,
                "seed": args.seed,
                "license_note": "Use according to GTZAN dataset terms; intended for MIR research.",
                "summary": summary,
                "total_selected": len(manifest_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print(f"[done] output: {out_dir}")
    print(f"[done] files: {len(manifest_rows)}")
    print(f"[done] manifest: {out_dir / 'manifest.jsonl'}")
    print(f"[done] summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
