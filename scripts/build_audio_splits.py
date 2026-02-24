#!/usr/bin/env python
"""
Build audio dataset splits with:
1) Artist-track grouping from filenames ("artist - title - index")
2) Split allocation (track-isolated or per-track index distribution)
3) Optional BPM metadata/filtering
4) Optional per-track clip cap

Outputs:
  - <output_dir>/manifest.jsonl
  - <output_dir>/summary.json
  - <output_dir>/train.txt
  - <output_dir>/val.txt
  - <output_dir>/test.txt
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aif", ".aiff"}

BPM_RE = re.compile(r"(?<!\d)(\d{2,3}(?:\.\d+)?)\s*bpm(?!\w)", re.IGNORECASE)

# Strip segment-style suffixes from clip filenames to recover parent track id.
TRACK_SUFFIX_RES = [
    re.compile(r"(?:[_\-\s]|__)(?:clip|seg|segment|part|take|loop|excerpt|slice|chunk)\d+$", re.IGNORECASE),
    re.compile(r"(?:[_\-\s]|__)\d{1,3}(?:bars?|bar)$", re.IGNORECASE),
    re.compile(r"(?:[_\-\s]|__)bars?\d{1,3}$", re.IGNORECASE),
]


@dataclass
class ClipRecord:
    path: str
    track_id: str
    bpm: Optional[float]
    bpm_source: str
    bpm_bin: str


def discover_audio_files(audio_dir: Path) -> List[Path]:
    files = [
        p for p in sorted(audio_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]
    return files


def _normalize_token(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "track"


def _strip_artist_title_index_suffix(stem: str) -> str:
    """
    For names like "Artist - Title - 03", return "Artist - Title".

    Only applies when:
      - there are at least 3 hyphen-separated parts
      - the last part is purely numeric (clip index)
    """
    parts = [p.strip() for p in re.split(r"\s*-\s*", stem) if p.strip()]
    if len(parts) >= 3 and re.fullmatch(r"\d{1,4}", parts[-1]):
        return " - ".join(parts[:-1]).strip()
    return stem


def infer_track_id(path: Path, audio_dir: Path) -> str:
    rel = path.relative_to(audio_dir)
    stem = rel.stem
    stem = BPM_RE.sub("", stem)
    stem = stem.strip(" _-.")

    # Preferred naming convention: "artist - title - index"
    stem = _strip_artist_title_index_suffix(stem)

    for pat in TRACK_SUFFIX_RES:
        prev = None
        while prev != stem:
            prev = stem
            stem = pat.sub("", stem).strip(" _-.")

    stem = _normalize_token(stem)
    parent = rel.parent.as_posix()
    parent_key = "" if parent == "." else _normalize_token(parent)
    return f"{parent_key}::{stem}" if parent_key else stem


def parse_bpm_from_name(path: Path) -> Optional[float]:
    for text in (path.stem, path.name):
        match = BPM_RE.search(text)
        if not match:
            continue
        bpm = float(match.group(1))
        if 40.0 <= bpm <= 220.0:
            return bpm
    return None


def estimate_bpm(path: Path, estimate_seconds: float) -> Optional[float]:
    try:
        import librosa
    except Exception:
        return None

    try:
        y, sr = librosa.load(str(path), sr=22050, mono=True, duration=estimate_seconds)
        if y.size == 0:
            return None
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo.squeeze())
        tempo = float(tempo)
        if not math.isfinite(tempo):
            return None
        if 40.0 <= tempo <= 220.0:
            return tempo
        return None
    except Exception:
        return None


def bpm_to_bin(bpm: Optional[float], min_bpm: int, max_bpm: int, bin_width: int) -> str:
    if bpm is None:
        return "unknown"
    bpm_i = int(round(float(bpm)))
    if bpm_i < min_bpm or bpm_i > max_bpm:
        return "out_of_range"
    idx = (bpm_i - min_bpm) // bin_width
    low = min_bpm + idx * bin_width
    high = min(low + bin_width - 1, max_bpm)
    return f"{low:03d}-{high:03d}"


def allocate_counts(n: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
    if n <= 0:
        return (0, 0, 0)
    if n == 1:
        return (1, 0, 0)

    raw = [n * ratios[0], n * ratios[1], n * ratios[2]]
    counts = [int(math.floor(v)) for v in raw]
    remain = n - sum(counts)
    frac_order = sorted(range(3), key=lambda i: (raw[i] - counts[i]), reverse=True)
    for i in range(remain):
        counts[frac_order[i % 3]] += 1

    # Prefer at least one val/test sample when enough tracks exist.
    if n >= 3:
        for idx in (1, 2):
            if counts[idx] == 0:
                donor = int(np.argmax(counts))
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[idx] += 1
    return (counts[0], counts[1], counts[2])


def build_track_split(
    track_to_bin: Dict[str, str],
    rng: np.random.Generator,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, str]:
    by_bin: Dict[str, List[str]] = {}
    for track_id, bpm_bin in track_to_bin.items():
        by_bin.setdefault(bpm_bin, []).append(track_id)

    track_split: Dict[str, str] = {}
    ratios = (train_ratio, val_ratio, test_ratio)

    for bpm_bin in sorted(by_bin.keys()):
        ids = by_bin[bpm_bin]
        ids = [ids[i] for i in rng.permutation(len(ids))]
        n_train, n_val, n_test = allocate_counts(len(ids), ratios)
        for track_id in ids[:n_train]:
            track_split[track_id] = "train"
        for track_id in ids[n_train:n_train + n_val]:
            track_split[track_id] = "val"
        for track_id in ids[n_train + n_val:n_train + n_val + n_test]:
            track_split[track_id] = "test"

    return track_split


def build_distributed_clip_split(
    track_to_records: Dict[str, List[ClipRecord]],
    rng: np.random.Generator,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> List[Tuple[ClipRecord, str]]:
    """Split clips within each artist-track group across train/val/test."""
    ratios = (train_ratio, val_ratio, test_ratio)
    rows: List[Tuple[ClipRecord, str]] = []
    two_clip_toggle = 0
    for track_id in sorted(track_to_records.keys()):
        clips = track_to_records[track_id]
        perm = rng.permutation(len(clips))
        clips = [clips[int(i)] for i in perm]
        n = len(clips)
        if n == 2:
            # Force different splits for 2-index tracks as well.
            if two_clip_toggle % 2 == 0:
                n_train, n_val, n_test = (1, 1, 0)
            else:
                n_train, n_val, n_test = (1, 0, 1)
            two_clip_toggle += 1
        else:
            n_train, n_val, n_test = allocate_counts(n, ratios)
        for rec in clips[:n_train]:
            rows.append((rec, "train"))
        for rec in clips[n_train:n_train + n_val]:
            rows.append((rec, "val"))
        for rec in clips[n_train + n_val:n_train + n_val + n_test]:
            rows.append((rec, "test"))
    return rows


def clip_path_for_manifest(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def select_clips_per_track(
    track_to_records: Dict[str, List[ClipRecord]],
    max_clips_per_track: Optional[int],
    rng: np.random.Generator,
) -> Dict[str, List[ClipRecord]]:
    if not max_clips_per_track:
        return track_to_records

    selected: Dict[str, List[ClipRecord]] = {}
    for track_id, records in track_to_records.items():
        if len(records) <= max_clips_per_track:
            selected[track_id] = records
            continue
        order = rng.permutation(len(records))[:max_clips_per_track]
        picks = [records[int(i)] for i in sorted(order)]
        selected[track_id] = picks
    return selected


def write_outputs(
    output_dir: Path,
    records: Iterable[Tuple[ClipRecord, str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    summary_path = output_dir / "summary.json"

    split_paths: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    split_track_ids: Dict[str, set] = {"train": set(), "val": set(), "test": set()}
    split_bins: Dict[str, Dict[str, int]] = {"train": {}, "val": {}, "test": {}}

    rows = list(records)
    with open(manifest_path, "w") as f:
        for rec, split in rows:
            split_paths[split].append(rec.path)
            split_track_ids[split].add(rec.track_id)
            split_bins[split][rec.bpm_bin] = split_bins[split].get(rec.bpm_bin, 0) + 1
            f.write(json.dumps({
                "path": rec.path,
                "track_id": rec.track_id,
                "bpm": rec.bpm,
                "bpm_source": rec.bpm_source,
                "bpm_bin": rec.bpm_bin,
                "split": split,
            }) + "\n")

    for split in ("train", "val", "test"):
        split_file = output_dir / f"{split}.txt"
        with open(split_file, "w") as f:
            for p in sorted(split_paths[split]):
                f.write(f"{p}\n")

    summary = {
        "total_clips": len(rows),
        "splits": {
            split: {
                "num_clips": len(split_paths[split]),
                "num_tracks": len(split_track_ids[split]),
                "bpm_bins": dict(sorted(split_bins[split].items())),
            }
            for split in ("train", "val", "test")
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved manifest: {manifest_path}")
    print(f"Saved summary:  {summary_path}")
    for split in ("train", "val", "test"):
        print(
            f"  {split:5s} clips={len(split_paths[split]):4d} "
            f"tracks={len(split_track_ids[split]):4d} file={output_dir / (split + '.txt')}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Build audio splits with artist-track grouping.",
    )
    parser.add_argument("--audio-dir", type=str, default="data/original/audio")
    parser.add_argument("--output-dir", type=str, default="data/original/splits")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--split-mode",
        choices=["track_isolated", "distribute_indices"],
        default="distribute_indices",
        help=(
            "track_isolated: all indices of a track stay in one split. "
            "distribute_indices: indices of same track are distributed across splits."
        ),
    )

    parser.add_argument("--min-bpm", type=int, default=120)
    parser.add_argument("--max-bpm", type=int, default=140)
    parser.add_argument("--bpm-bin-width", type=int, default=5)
    parser.add_argument(
        "--bpm-mode",
        choices=["none", "filename", "estimate", "filename_or_estimate"],
        default="none",
    )
    parser.add_argument("--estimate-seconds", type=float, default=30.0)
    parser.add_argument("--drop-out-of-range", action="store_true")
    parser.add_argument("--drop-unknown-bpm", action="store_true")
    parser.add_argument(
        "--max-clips-per-track",
        type=int,
        default=0,
        help="Maximum clips per track_id (0 = no limit).",
    )

    args = parser.parse_args()

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(total_ratio, 1.0, atol=1e-6):
        raise ValueError("train/val/test ratios must sum to 1.0")
    if args.min_bpm >= args.max_bpm:
        raise ValueError("min_bpm must be smaller than max_bpm")
    if args.bpm_bin_width <= 0:
        raise ValueError("bpm_bin_width must be > 0")

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    rng = np.random.default_rng(args.seed)

    files = discover_audio_files(audio_dir)
    if not files:
        raise FileNotFoundError(f"No audio files found under: {audio_dir}")

    if args.bpm_mode == "none" and (args.drop_out_of_range or args.drop_unknown_bpm):
        print("Warning: bpm_mode=none, so drop-out-of-range / drop-unknown-bpm options are ignored.")

    records: List[ClipRecord] = []
    for path in files:
        track_id = infer_track_id(path, audio_dir)
        bpm = None
        source = "disabled"

        if args.bpm_mode != "none":
            source = "none"
            if args.bpm_mode in ("filename", "filename_or_estimate"):
                bpm = parse_bpm_from_name(path)
                if bpm is not None:
                    source = "filename"
            if bpm is None and args.bpm_mode in ("estimate", "filename_or_estimate"):
                bpm = estimate_bpm(path, args.estimate_seconds)
                if bpm is not None:
                    source = "estimate"

            bpm_bin = bpm_to_bin(bpm, args.min_bpm, args.max_bpm, args.bpm_bin_width)

            if args.drop_out_of_range and bpm_bin == "out_of_range":
                continue
            if args.drop_unknown_bpm and bpm_bin == "unknown":
                continue
        else:
            bpm_bin = "all"

        records.append(
            ClipRecord(
                path=clip_path_for_manifest(path),
                track_id=track_id,
                bpm=bpm,
                bpm_source=source,
                bpm_bin=bpm_bin,
            )
        )

    if not records:
        raise RuntimeError("No records left after BPM filtering options.")

    track_to_records: Dict[str, List[ClipRecord]] = {}
    for rec in records:
        track_to_records.setdefault(rec.track_id, []).append(rec)
    track_to_records = select_clips_per_track(
        track_to_records=track_to_records,
        max_clips_per_track=args.max_clips_per_track,
        rng=rng,
    )

    if args.split_mode == "track_isolated":
        track_to_bin: Dict[str, str] = {}
        for track_id, clips in track_to_records.items():
            if args.bpm_mode == "none":
                track_to_bin[track_id] = "all"
                continue
            known_bpms = [c.bpm for c in clips if c.bpm is not None]
            median_bpm = float(np.median(np.array(known_bpms))) if known_bpms else None
            track_to_bin[track_id] = bpm_to_bin(
                median_bpm, args.min_bpm, args.max_bpm, args.bpm_bin_width
            )

        track_split = build_track_split(
            track_to_bin=track_to_bin,
            rng=rng,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )

        final_rows: List[Tuple[ClipRecord, str]] = []
        for track_id, clips in track_to_records.items():
            split = track_split[track_id]
            for rec in clips:
                final_rows.append((rec, split))
    else:
        final_rows = build_distributed_clip_split(
            track_to_records=track_to_records,
            rng=rng,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )

    write_outputs(output_dir, final_rows)


if __name__ == "__main__":
    main()
