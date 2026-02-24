#!/usr/bin/env python
"""
Download genre-balanced tracks from Jamendo using MTG metadata (raw_30s.tsv).

Downloads are saved under data/raw so you can inspect by genre.

Example:
  ./venv_DeltaV2A/bin/python scripts/download_jamendo_genres.py \
      --count-per-genre 30 \
      --genres classical,jazz,pop,rock
"""

import argparse
import json
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Jamendo tracks by genre.")
    p.add_argument("--metadata-tsv", type=str, default="data/raw/mtg_jamendo/.cache/raw_30s.tsv")
    p.add_argument("--output-dir", type=str, default="data/raw/genre_music")
    p.add_argument("--genres", type=str, default="classical,jazz,pop,rock")
    p.add_argument("--count-per-genre", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout-sec", type=int, default=30)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--max-checks-per-genre", type=int, default=800)
    p.add_argument("--order", type=str, default="ascending", choices=["ascending", "random"])
    p.add_argument("--clean-output", action="store_true")
    return p.parse_args()


def _parse_track_number(track_id: str) -> int:
    # track_0000382 -> 382
    return int(track_id.split("_")[-1])


def _download_file(url: str, dst: Path, timeout_sec: int, max_retries: int) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(1, max_retries + 1):
        try:
            # Use curl instead of urllib for more reliable network behavior here.
            cmd = [
                "curl",
                "-L",
                "--fail",
                "--silent",
                "--show-error",
                "--max-time",
                str(timeout_sec),
                "-o",
                str(tmp),
                url,
            ]
            res = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if res.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
                tmp.replace(dst)
                return True
        except OSError:
            pass
        # Cleanup + retry path
        try:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        except OSError:
            pass
        if attempt < max_retries:
            time.sleep(0.5 * attempt)
        else:
            return False
    return False


def _load_genre_candidates(metadata_tsv: Path, target_genres: List[str]) -> Dict[str, List[str]]:
    candidates = {g: [] for g in target_genres}
    with metadata_tsv.open("r", encoding="utf-8") as f:
        header = next(f, None)
        if header is None:
            raise ValueError(f"empty metadata file: {metadata_tsv}")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            track_id = parts[0]
            tag_text = "\t".join(parts[5:])
            for genre in target_genres:
                if f"genre---{genre}" in tag_text:
                    candidates[genre].append(track_id)
    return candidates


def main():
    args = parse_args()
    metadata_tsv = Path(args.metadata_tsv)
    out_dir = Path(args.output_dir)
    genres = [g.strip().lower() for g in args.genres.split(",") if g.strip()]
    count_per_genre = int(args.count_per_genre)
    rng = random.Random(args.seed)

    if not metadata_tsv.exists():
        raise FileNotFoundError(f"metadata tsv not found: {metadata_tsv}")
    if count_per_genre <= 0:
        raise ValueError("--count-per-genre must be > 0")
    if not genres:
        raise ValueError("--genres is empty")

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_output:
        for p in out_dir.iterdir():
            if p.is_file():
                p.unlink()
            else:
                shutil.rmtree(p)

    candidates = _load_genre_candidates(metadata_tsv, genres)
    manifest_rows = []
    summary = {}

    total_ok = 0
    total_fail = 0

    for genre in genres:
        pool = list(dict.fromkeys(candidates.get(genre, [])))
        if args.order == "ascending":
            pool.sort(key=_parse_track_number)
        else:
            rng.shuffle(pool)
        genre_dir = out_dir / genre
        genre_dir.mkdir(parents=True, exist_ok=True)

        ok = 0
        failed = 0
        checked = 0
        print(f"[{genre}] candidates: {len(pool)}")
        for track_id in pool:
            if ok >= count_per_genre:
                break
            if checked >= args.max_checks_per_genre:
                break
            checked += 1
            track_num = _parse_track_number(track_id)
            url = f"https://prod-1.storage.jamendo.com/download/track/{track_num}/mp32/"
            dst = genre_dir / f"{track_id}.mp3"

            if dst.exists() and dst.stat().st_size > 0:
                ok += 1
                manifest_rows.append(
                    {
                        "genre": genre,
                        "track_id": track_id,
                        "url": url,
                        "output_path": str(dst),
                        "status": "existing",
                    }
                )
                continue

            success = _download_file(
                url=url,
                dst=dst,
                timeout_sec=args.timeout_sec,
                max_retries=args.max_retries,
            )
            if success:
                ok += 1
                manifest_rows.append(
                    {
                        "genre": genre,
                        "track_id": track_id,
                        "url": url,
                        "output_path": str(dst),
                        "status": "downloaded",
                    }
                )
                if ok % 10 == 0:
                    print(f"[{genre}] downloaded {ok}/{count_per_genre}")
            else:
                failed += 1
                if checked % 25 == 0:
                    print(f"[{genre}] checked {checked}, success {ok}, failed {failed}")

        total_ok += ok
        total_fail += failed
        summary[genre] = {
            "candidates_in_metadata": len(pool),
            "checked": checked,
            "selected_target": count_per_genre,
            "downloaded_or_existing": ok,
            "failed_attempts": failed,
            "output_dir": str(genre_dir),
        }
        print(f"[done] {genre}: {ok}/{count_per_genre} (failed attempts: {failed})")

    manifest_path = out_dir / "manifest.jsonl"
    summary_path = out_dir / "summary.json"
    manifest_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in manifest_rows) + ("\n" if manifest_rows else "")
    )
    summary_path.write_text(
        json.dumps(
            {
                "source": "Jamendo direct track download",
                "metadata_tsv": str(metadata_tsv),
                "genres": genres,
                "count_per_genre_requested": count_per_genre,
                "seed": args.seed,
                "total_downloaded_or_existing": total_ok,
                "total_failed_attempts": total_fail,
                "summary": summary,
                "license_note": "Check Jamendo terms and track licenses before redistribution/commercial use.",
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print(f"[done] output dir: {out_dir}")
    print(f"[done] total files: {total_ok}")
    print(f"[done] manifest: {manifest_path}")
    print(f"[done] summary: {summary_path}")


if __name__ == "__main__":
    main()
