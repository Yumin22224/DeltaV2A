"""
MTG-Jamendo Dataset Downloader

Downloads electronic music tracks from the MTG-Jamendo dataset.
https://github.com/MTG/mtg-jamendo-dataset

Usage:
    python scripts/download_jamendo.py --help
    python scripts/download_jamendo.py --list-genres
    python scripts/download_jamendo.py --genre electronic --limit 100
    python scripts/download_jamendo.py --genre electronic,house,techno --limit 500
"""

import os
import sys
import argparse
import json
import urllib.request
import ssl
import tarfile
from pathlib import Path
from typing import List, Set, Dict, Optional

# Handle SSL certificate issues on macOS
ssl._create_default_https_context = ssl._create_unverified_context

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


# MTG-Jamendo URLs
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data"
AUDIO_BASE_URL = "https://cdn.freesound.org/mtg-jamendo"


def download_file(url: str, output_path: str, show_progress: bool = True) -> bool:
    """Download a file from URL"""
    try:
        if show_progress:
            print(f"  Downloading: {url}")
        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as e:
        print(f"  Error downloading {url}: {e}")
        return False


def load_raw_30s_tsv(cache_dir: Path) -> Dict[str, Dict]:
    """
    Load raw_30s.tsv for track metadata including tags.

    Format: TRACK_ID, ARTIST_ID, ALBUM_ID, PATH, DURATION, TAGS
    Tags format: genre---electronic, mood/theme---happy, etc.

    Returns:
        Dict mapping track_id to metadata (including parsed genres)
    """
    tsv_path = cache_dir / "raw_30s.tsv"

    if not tsv_path.exists():
        print("Downloading track metadata...")
        url = f"{GITHUB_RAW_BASE}/raw_30s.tsv"
        if not download_file(url, str(tsv_path)):
            raise RuntimeError("Failed to download metadata")

    tracks = {}

    with open(tsv_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        # TRACK_ID, ARTIST_ID, ALBUM_ID, PATH, DURATION, TAGS

        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 6:
                continue

            track_id = parts[0]
            tags_str = parts[5] if len(parts) > 5 else ''

            # Parse tags: "genre---electronic, mood/theme---happy"
            genres = []
            moods = []
            instruments = []

            for tag in tags_str.split(', '):
                if tag.startswith('genre---'):
                    genres.append(tag.replace('genre---', ''))
                elif tag.startswith('mood/theme---'):
                    moods.append(tag.replace('mood/theme---', ''))
                elif tag.startswith('instrument---'):
                    instruments.append(tag.replace('instrument---', ''))

            tracks[track_id] = {
                'track_id': track_id,
                'artist_id': parts[1],
                'album_id': parts[2],
                'path': parts[3],
                'duration': float(parts[4]) if parts[4] else 0,
                'genres': genres,
                'moods': moods,
                'instruments': instruments,
                'tags_raw': tags_str,
            }

    return tracks


def get_available_genres(tracks: Dict[str, Dict]) -> Set[str]:
    """Extract all unique genres from tracks"""
    genres = set()
    for meta in tracks.values():
        genres.update(meta.get('genres', []))
    return genres


def filter_tracks_by_genre(
    tracks: Dict[str, Dict],
    target_genres: Set[str],
    require_all: bool = False,
) -> List[str]:
    """
    Filter tracks by genre.

    Args:
        tracks: Dict mapping track_id to metadata
        target_genres: Set of genres to filter by
        require_all: If True, track must have ALL target genres

    Returns:
        List of matching track IDs
    """
    matching = []

    for track_id, meta in tracks.items():
        track_genres = set(meta.get('genres', []))

        if require_all:
            if target_genres.issubset(track_genres):
                matching.append(track_id)
        else:
            if target_genres.intersection(track_genres):
                matching.append(track_id)

    return matching


def download_tracks(
    track_ids: List[str],
    tracks_meta: Dict[str, Dict],
    output_dir: Path,
    limit: Optional[int] = None,
) -> List[str]:
    """
    Download audio files for given track IDs.

    Args:
        track_ids: List of track IDs to download
        tracks_meta: Metadata dict with file paths
        output_dir: Directory to save files
        limit: Maximum number of tracks to download

    Returns:
        List of successfully downloaded file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if limit:
        track_ids = track_ids[:limit]

    print(f"\nDownloading {len(track_ids)} tracks to {output_dir}")

    downloaded = []
    failed = []

    for i, track_id in enumerate(track_ids):
        meta = tracks_meta.get(track_id)
        if not meta:
            print(f"  [{i+1}/{len(track_ids)}] {track_id}: No metadata, skipping")
            continue

        rel_path = meta.get('path', '')
        if not rel_path:
            continue

        output_file = output_dir / f"{track_id}.mp3"

        if output_file.exists():
            print(f"  [{i+1}/{len(track_ids)}] {track_id}: Already exists")
            downloaded.append(str(output_file))
            continue

        # MTG-Jamendo audio URL
        url = f"{AUDIO_BASE_URL}/raw_30s/audio/{rel_path}"

        print(f"  [{i+1}/{len(track_ids)}] {track_id} ", end="", flush=True)

        if download_file(url, str(output_file), show_progress=False):
            downloaded.append(str(output_file))
            print("✓")
        else:
            failed.append(track_id)
            print("✗")

    print(f"\nDownload complete: {len(downloaded)} successful, {len(failed)} failed")

    return downloaded


def save_track_list(
    track_ids: List[str],
    tracks_meta: Dict[str, Dict],
    output_path: Path,
):
    """Save track list with metadata to JSON"""
    data = {
        'total': len(track_ids),
        'tracks': []
    }

    for track_id in track_ids:
        meta = tracks_meta.get(track_id, {})

        data['tracks'].append({
            'track_id': track_id,
            'genres': meta.get('genres', []),
            'moods': meta.get('moods', []),
            'duration': meta.get('duration', 0),
            'path': meta.get('path', ''),
        })

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved track list to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download electronic music from MTG-Jamendo dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/mtg_jamendo",
        help="Output directory for downloaded files",
    )
    parser.add_argument(
        "--genre",
        type=str,
        default="electronic",
        help="Comma-separated list of genres to download (e.g., 'electronic,house,techno')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of tracks to download",
    )
    parser.add_argument(
        "--list-genres",
        action="store_true",
        help="List all available genres and exit",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only create track list, don't download audio",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Require ALL specified genres (instead of ANY)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = output_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    print("Loading metadata...")
    tracks_meta = load_raw_30s_tsv(cache_dir)
    print(f"  Total tracks: {len(tracks_meta)}")

    # List genres mode
    if args.list_genres:
        genres = get_available_genres(tracks_meta)

        electronic_genres = {
            'electronic', 'electropop', 'house', 'techno', 'trance',
            'drumnbass', 'ambient', 'chillout', 'downtempo', 'dub',
            'triphop', 'minimal', 'dance', 'club', 'breakbeat', 'synthpop'
        }

        print("\n  Electronic-related:")
        for g in sorted(genres):
            if g in electronic_genres:
                # Count tracks
                count = sum(1 for t in tracks_meta.values() if g in t.get('genres', []))
                print(f"    {g}: {count} tracks")

        print("\n  All genres:")
        for g in sorted(genres):
            count = sum(1 for t in tracks_meta.values() if g in t.get('genres', []))
            print(f"    {g}: {count} tracks")

        return

    # Parse target genres
    target_genres = set(g.strip() for g in args.genre.split(','))
    print(f"\nFiltering by genres: {target_genres}")

    # Filter tracks
    matching_tracks = filter_tracks_by_genre(
        tracks_meta, target_genres, require_all=args.require_all
    )
    print(f"  Matching tracks: {len(matching_tracks)}")

    # Apply limit
    if args.limit:
        matching_tracks = matching_tracks[:args.limit]
        print(f"  After limit: {len(matching_tracks)}")

    if len(matching_tracks) == 0:
        print("\nNo matching tracks found!")
        return

    # Save track list
    audio_dir = output_dir / "audio"
    track_list_path = output_dir / "track_list.json"
    save_track_list(matching_tracks, tracks_meta, track_list_path)

    # Download if not list-only
    if not args.list_only:
        downloaded = download_tracks(
            matching_tracks,
            tracks_meta,
            audio_dir,
            limit=None,
        )

        print(f"\n{'='*50}")
        print(f"Downloaded {len(downloaded)} tracks to {audio_dir}")
        print(f"Track list saved to {track_list_path}")
        print(f"{'='*50}")
    else:
        print(f"\nTrack list saved to {track_list_path}")
        print("Use without --list-only to download audio files")


if __name__ == "__main__":
    main()
