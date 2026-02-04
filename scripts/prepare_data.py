"""
Data Preparation Script

Utilities for preparing delta pair dataset:
1. Extract 4-bar segments from electronic music
2. Create pairs.json metadata file
3. Validate dataset structure
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import shutil

sys.path.append(str(Path(__file__).parent.parent))


def create_pairs_json(
    pairs: List[Dict],
    output_path: str,
) -> None:
    """
    Create pairs.json metadata file.

    Args:
        pairs: List of pair dictionaries
        output_path: Path to save pairs.json
    """
    data = {"pairs": pairs}

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Created {output_path} with {len(pairs)} pairs")


def validate_dataset(data_dir: str) -> Dict:
    """
    Validate dataset structure and files.

    Args:
        data_dir: Path to dataset directory

    Returns:
        Validation report
    """
    data_dir = Path(data_dir)
    report = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {},
    }

    # Check directory structure
    required_dirs = ['images', 'audios']
    for d in required_dirs:
        if not (data_dir / d).exists():
            report['errors'].append(f"Missing directory: {d}")
            report['valid'] = False

    # Check pairs.json
    pairs_file = data_dir / 'pairs.json'
    if not pairs_file.exists():
        report['errors'].append("Missing pairs.json")
        report['valid'] = False
        return report

    # Load and validate pairs
    with open(pairs_file) as f:
        data = json.load(f)

    pairs = data.get('pairs', [])
    report['stats']['total_pairs'] = len(pairs)

    valid_pairs = 0
    missing_files = []

    for i, pair in enumerate(pairs):
        # Check required fields
        required_fields = ['i_init', 'i_edit', 'a_init', 'a_edit']
        for field in required_fields:
            if field not in pair:
                report['errors'].append(f"Pair {i}: missing field '{field}'")
                report['valid'] = False
                continue

        # Check files exist
        files_exist = True
        for field in required_fields:
            if field.startswith('i_'):
                filepath = data_dir / 'images' / pair[field]
            else:
                filepath = data_dir / 'audios' / pair[field]

            if not filepath.exists():
                missing_files.append(str(filepath))
                files_exist = False

        if files_exist:
            valid_pairs += 1

    report['stats']['valid_pairs'] = valid_pairs
    report['stats']['invalid_pairs'] = len(pairs) - valid_pairs

    if missing_files:
        report['warnings'].append(f"Missing {len(missing_files)} files")
        if len(missing_files) <= 10:
            for f in missing_files:
                report['warnings'].append(f"  - {f}")
        else:
            for f in missing_files[:5]:
                report['warnings'].append(f"  - {f}")
            report['warnings'].append(f"  ... and {len(missing_files) - 5} more")

    return report


def extract_4bar_segments(
    audio_path: str,
    output_dir: str,
    bpm: Optional[float] = None,
    sample_rate: int = 16000,
) -> List[str]:
    """
    Extract 4-bar segments from audio file.

    Args:
        audio_path: Path to source audio
        output_dir: Directory to save segments
        bpm: BPM (will be estimated if not provided)
        sample_rate: Target sample rate

    Returns:
        List of output file paths
    """
    import librosa
    import soundfile as sf
    import numpy as np

    # Load audio
    y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)

    # Estimate BPM if not provided
    if bpm is None:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo)
        print(f"  Estimated BPM: {bpm:.1f}")

    # Calculate 4-bar duration in samples
    beat_duration = 60.0 / bpm  # seconds per beat
    bar_duration = beat_duration * 4  # 4 beats per bar
    segment_duration = bar_duration * 4  # 4 bars
    segment_samples = int(segment_duration * sr)

    # Extract segments
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(audio_path).stem
    output_files = []

    num_segments = len(y) // segment_samples

    for i in range(num_segments):
        start = i * segment_samples
        end = start + segment_samples
        segment = y[start:end]

        output_path = output_dir / f"{base_name}_seg{i:03d}.wav"
        sf.write(str(output_path), segment, sr)
        output_files.append(str(output_path))

    print(f"  Extracted {len(output_files)} segments from {audio_path}")
    return output_files


def create_example_dataset(output_dir: str) -> None:
    """
    Create example dataset structure with placeholder files.

    Args:
        output_dir: Directory to create example dataset
    """
    output_dir = Path(output_dir)
    (output_dir / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'audios').mkdir(parents=True, exist_ok=True)

    # Create example pairs.json
    example_pairs = [
        {
            "i_init": "scene_001.jpg",
            "i_edit": "scene_001_bright.jpg",
            "a_init": "track_001_bar1.wav",
            "a_edit": "track_001_bar2.wav",
            "valid": True,
            "description": "Brightness increase → energy change"
        },
        {
            "i_init": "scene_002.jpg",
            "i_edit": "scene_002_blur.jpg",
            "a_init": "track_002_bar1.wav",
            "a_edit": "track_002_bar2.wav",
            "valid": True,
            "description": "Blur increase → reverb change"
        },
    ]

    pairs_file = output_dir / 'pairs.json'
    with open(pairs_file, 'w') as f:
        json.dump({"pairs": example_pairs}, f, indent=2)

    print(f"Created example dataset structure at {output_dir}")
    print(f"  - {pairs_file}")
    print(f"  - {output_dir / 'images'}/ (add your image files here)")
    print(f"  - {output_dir / 'audios'}/ (add your audio files here)")
    print(f"\nEdit pairs.json to match your actual file names.")


def main():
    parser = argparse.ArgumentParser(description="Data Preparation Tools")
    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate dataset')
    validate_parser.add_argument('data_dir', type=str, help='Dataset directory')

    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract 4-bar segments')
    extract_parser.add_argument('audio_path', type=str, help='Source audio file')
    extract_parser.add_argument('output_dir', type=str, help='Output directory')
    extract_parser.add_argument('--bpm', type=float, default=None, help='BPM (optional)')

    # Example command
    example_parser = subparsers.add_parser('example', help='Create example dataset structure')
    example_parser.add_argument('output_dir', type=str, help='Output directory')

    args = parser.parse_args()

    if args.command == 'validate':
        report = validate_dataset(args.data_dir)

        print(f"\n{'='*50}")
        print("Dataset Validation Report")
        print(f"{'='*50}")

        if report['valid']:
            print("✓ Dataset is valid")
        else:
            print("✗ Dataset has errors")

        print(f"\nStats:")
        for k, v in report['stats'].items():
            print(f"  {k}: {v}")

        if report['errors']:
            print(f"\nErrors:")
            for e in report['errors']:
                print(f"  ✗ {e}")

        if report['warnings']:
            print(f"\nWarnings:")
            for w in report['warnings']:
                print(f"  ⚠ {w}")

    elif args.command == 'extract':
        print(f"Extracting 4-bar segments from {args.audio_path}")
        extract_4bar_segments(args.audio_path, args.output_dir, bpm=args.bpm)

    elif args.command == 'example':
        create_example_dataset(args.output_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
