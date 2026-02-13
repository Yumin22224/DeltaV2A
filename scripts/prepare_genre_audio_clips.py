#!/usr/bin/env python
"""
Create 10-15s clips from genre raw audio and place them under data/original/audio.

Input layout:
  data/raw/genre_music/<genre>/*.(mp3|wav|...)

Output layout:
  data/original/audio/<genre>/<genre> - <track_name> - <index>.mp3

The filename format intentionally follows "artist - title - index" so
scripts/build_audio_splits.py can infer shared track_id across indices.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aif", ".aiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare short genre clips for training splits.")
    p.add_argument("--raw-dir", type=str, default="data/raw/genre_music")
    p.add_argument("--output-dir", type=str, default="data/original/audio")
    p.add_argument("--genres", type=str, default="classical,jazz,pop,rock")
    p.add_argument("--min-sec", type=float, default=10.0)
    p.add_argument("--max-sec", type=float, default=15.0)
    p.add_argument(
        "--clips-per-track",
        type=int,
        default=None,
        help="Legacy fixed clip count per track. If set, min/max clip args are ignored.",
    )
    p.add_argument("--min-clips-per-track", type=int, default=2)
    p.add_argument("--max-clips-per-track", type=int, default=3)
    p.add_argument(
        "--segmentation-mode",
        type=str,
        choices=["random", "uniform"],
        default="random",
        help="random: random anchored segments, uniform: equal-length sequential segments.",
    )
    p.add_argument(
        "--uniform-clip-sec",
        type=float,
        default=15.0,
        help="Clip length used when --segmentation-mode uniform.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clean-genres", action="store_true", help="Remove existing output genre folders before export.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing clip files.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def discover_audio_files(folder: Path) -> List[Path]:
    return [
        p for p in sorted(folder.rglob("*"))
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]


def probe_duration_seconds(path: Path) -> Optional[float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        value = float(out)
        if value <= 0:
            return None
        return value
    except Exception:
        return None


def normalize_track_name(stem: str) -> str:
    name = stem.replace("_", " ").strip()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"[\\/]+", " ", name)
    name = re.sub(r"\s*-\s*", " ", name)
    return name.strip(" .") or "track"


def choose_segments(
    duration_sec: float,
    min_clips_per_track: int,
    max_clips_per_track: int,
    min_sec: float,
    max_sec: float,
    rng: random.Random,
) -> List[Tuple[float, float]]:
    if duration_sec < min_sec or min_clips_per_track <= 0 or max_clips_per_track <= 0:
        return []

    max_possible = max(1, int(duration_sec // min_sec))
    lo = max(1, min(min_clips_per_track, max_clips_per_track))
    hi = max(lo, max(min_clips_per_track, max_clips_per_track))
    lo = min(lo, max_possible)
    hi = min(hi, max_possible)
    if lo > hi:
        return []

    n_clips = rng.randint(lo, hi)

    # Sample clip lengths with feasibility constraints so clips remain non-overlapping.
    lengths: List[float] = []
    budget = float(duration_sec)
    for i in range(n_clips):
        remaining = n_clips - i - 1
        min_remaining = remaining * min_sec
        len_lo = min_sec
        len_hi = min(max_sec, budget - min_remaining)
        if len_hi < len_lo:
            return []
        seg_len = rng.uniform(len_lo, len_hi)
        lengths.append(seg_len)
        budget -= seg_len

    # Distribute the remaining slack as random gaps (n_clips + 1 buckets).
    slack = max(duration_sec - sum(lengths), 0.0)
    gap_weights = [rng.random() for _ in range(n_clips + 1)]
    wsum = sum(gap_weights)
    if wsum <= 0:
        gaps = [0.0 for _ in range(n_clips + 1)]
    else:
        gaps = [slack * (w / wsum) for w in gap_weights]

    segments: List[Tuple[float, float]] = []
    cur = gaps[0]
    for i, seg_len in enumerate(lengths):
        start = max(min(cur, duration_sec - seg_len), 0.0)
        segments.append((start, seg_len))
        cur = start + seg_len + gaps[i + 1]

    segments.sort(key=lambda x: x[0])
    return segments


def choose_uniform_segments(
    duration_sec: float,
    min_sec: float,
    max_sec: float,
    uniform_clip_sec: float,
) -> List[Tuple[float, float]]:
    if duration_sec < min_sec:
        return []
    clip_sec = float(uniform_clip_sec)
    if clip_sec < min_sec or clip_sec > max_sec:
        raise ValueError(
            f"--uniform-clip-sec ({clip_sec}) must be within [--min-sec, --max-sec] = [{min_sec}, {max_sec}]"
        )
    n = int(duration_sec // clip_sec)
    if n <= 0:
        # Fallback for short-but-valid files.
        clip_len = min(max(duration_sec, min_sec), max_sec)
        start = max((duration_sec - clip_len) / 2.0, 0.0)
        return [(start, clip_len)]

    return [(i * clip_sec, clip_sec) for i in range(n)]


def export_clip(
    src: Path,
    dst: Path,
    start_sec: float,
    duration_sec: float,
    overwrite: bool,
) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    cmd.append("-y" if overwrite else "-n")
    cmd.extend(
        [
            "-ss",
            f"{start_sec:.3f}",
            "-i",
            str(src),
            "-t",
            f"{duration_sec:.3f}",
            "-vn",
            "-acodec",
            "libmp3lame",
            "-q:a",
            "2",
            str(dst),
        ]
    )
    subprocess.run(cmd, check=True)


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    if rows:
        text += "\n"
    path.write_text(text)


def main() -> None:
    args = parse_args()
    if args.min_sec <= 0 or args.max_sec <= 0:
        raise ValueError("--min-sec and --max-sec must be > 0")
    if args.min_sec > args.max_sec:
        raise ValueError("--min-sec must be <= --max-sec")
    if args.uniform_clip_sec <= 0:
        raise ValueError("--uniform-clip-sec must be > 0")
    if args.clips_per_track is not None and args.clips_per_track <= 0:
        raise ValueError("--clips-per-track must be > 0 when set")
    if args.min_clips_per_track <= 0 or args.max_clips_per_track <= 0:
        raise ValueError("--min-clips-per-track and --max-clips-per-track must be > 0")
    if args.min_clips_per_track > args.max_clips_per_track:
        raise ValueError("--min-clips-per-track must be <= --max-clips-per-track")

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    genres = [g.strip().lower() for g in args.genres.split(",") if g.strip()]
    rng = random.Random(args.seed)

    if not genres:
        raise ValueError("No genres provided")

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[dict] = []
    summary: Dict[str, dict] = {}

    total_input = 0
    total_clips = 0
    total_skipped_short = 0
    total_probe_fail = 0

    for genre in genres:
        genre_in = raw_dir / genre
        genre_out = output_dir / genre
        if not genre_in.exists():
            summary[genre] = {
                "error": "missing_input_dir",
                "input_dir": str(genre_in),
            }
            continue

        if args.clean_genres and genre_out.exists() and not args.dry_run:
            shutil.rmtree(genre_out)
        if not args.dry_run:
            genre_out.mkdir(parents=True, exist_ok=True)

        files = discover_audio_files(genre_in)
        genre_input = len(files)
        genre_output = 0
        genre_short = 0
        genre_probe_fail = 0
        total_input += genre_input

        for src in files:
            source_duration = probe_duration_seconds(src)
            if source_duration is None:
                genre_probe_fail += 1
                total_probe_fail += 1
                continue

            if args.segmentation_mode == "uniform":
                segments = choose_uniform_segments(
                    duration_sec=source_duration,
                    min_sec=args.min_sec,
                    max_sec=args.max_sec,
                    uniform_clip_sec=args.uniform_clip_sec,
                )
            else:
                if args.clips_per_track is not None:
                    min_cpt = int(args.clips_per_track)
                    max_cpt = int(args.clips_per_track)
                else:
                    min_cpt = int(args.min_clips_per_track)
                    max_cpt = int(args.max_clips_per_track)
                segments = choose_segments(
                    duration_sec=source_duration,
                    min_clips_per_track=min_cpt,
                    max_clips_per_track=max_cpt,
                    min_sec=args.min_sec,
                    max_sec=args.max_sec,
                    rng=rng,
                )
            if not segments:
                genre_short += 1
                total_skipped_short += 1
                continue

            track_name = normalize_track_name(src.stem)
            for idx, (start_sec, clip_sec) in enumerate(segments, start=1):
                out_name = f"{genre} - {track_name} - {idx}.mp3"
                out_path = genre_out / out_name
                row = {
                    "genre": genre,
                    "source_path": str(src),
                    "output_path": str(out_path),
                    "source_duration_sec": round(float(source_duration), 4),
                    "clip_index": idx,
                    "clip_start_sec": round(float(start_sec), 4),
                    "clip_duration_sec": round(float(clip_sec), 4),
                }
                manifest_rows.append(row)

                if not args.dry_run:
                    export_clip(
                        src=src,
                        dst=out_path,
                        start_sec=start_sec,
                        duration_sec=clip_sec,
                        overwrite=args.overwrite,
                    )

                genre_output += 1
                total_clips += 1

        summary[genre] = {
            "input_tracks": genre_input,
            "clips_created": genre_output,
            "skipped_too_short": genre_short,
            "probe_failures": genre_probe_fail,
            "input_dir": str(genre_in),
            "output_dir": str(genre_out),
        }

    manifest_path = output_dir / "genre_clip_manifest.jsonl"
    summary_path = output_dir / "genre_clip_summary.json"
    write_jsonl(manifest_path, manifest_rows)
    summary_path.write_text(
        json.dumps(
            {
                "genres": genres,
                "seed": args.seed,
                "min_sec": args.min_sec,
                "max_sec": args.max_sec,
                "clips_per_track": args.clips_per_track,
                "min_clips_per_track": args.min_clips_per_track,
                "max_clips_per_track": args.max_clips_per_track,
                "segmentation_mode": args.segmentation_mode,
                "uniform_clip_sec": args.uniform_clip_sec,
                "dry_run": args.dry_run,
                "totals": {
                    "input_tracks": total_input,
                    "clips_created": total_clips,
                    "skipped_too_short": total_skipped_short,
                    "probe_failures": total_probe_fail,
                },
                "by_genre": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print(f"[done] manifest: {manifest_path}")
    print(f"[done] summary:  {summary_path}")
    print(
        "[done] totals: "
        f"tracks={total_input}, clips={total_clips}, "
        f"short={total_skipped_short}, probe_fail={total_probe_fail}"
    )


if __name__ == "__main__":
    main()
