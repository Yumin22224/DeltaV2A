#!/usr/bin/env python
"""
SUN397 Landscape Filter

Downloads SUN397 dataset and filters for natural landscape images.

Usage:
    # Download SUN397 (if not already downloaded)
    python scripts/prepare_sun397_landscape.py --download --download-dir data/raw/SUN397

    # List matching classes
    python scripts/prepare_sun397_landscape.py --sun397-root data/raw/SUN397 --list-only

    # Copy filtered images to experiment directory
    python scripts/prepare_sun397_landscape.py --sun397-root data/raw/SUN397 --output data/experiment/images --limit 100

    # All in one: download + filter
    python scripts/prepare_sun397_landscape.py --download --download-dir data/raw/SUN397 --output data/experiment/images --limit 100
"""

import argparse
import shutil
import os
import sys
import tarfile
from pathlib import Path
from typing import List, Set
import re


# Include keywords (natural landscape)
INCLUDE_KEYWORDS = [
    # Natural/wilderness
    "natural", "nature", "wilderness",
    # Mountains
    "mountain", "hill", "valley", "canyon", "cliff", "ridge", "peak",
    # Forest
    "forest", "woods", "jungle", "rainforest", "tree",
    # Water
    "river", "lake", "pond", "waterfall", "stream", "creek",
    "ocean", "sea", "coast", "beach", "shore", "bay", "cove",
    # Arid
    "desert", "dune", "badlands",
    # Fields
    "field", "meadow", "grassland", "farmland", "pasture", "prairie",
    # Cold
    "glacier", "ice", "snow", "tundra",
    # Sky/weather
    "sky", "cloud", "sunset", "sunrise",
    # Other nature
    "swamp", "marsh", "wetland", "savanna", "steppe",
]

# Exclude keywords (urban/artificial)
EXCLUDE_KEYWORDS = [
    # Urban
    "street", "alley", "highway", "bridge", "city", "urban", "road", "intersection",
    # Buildings
    "building", "house", "castle", "church", "temple", "tower", "skyscraper",
    "apartment", "office", "shop", "store", "mall", "hotel", "hospital",
    "school", "university", "museum", "library", "station", "airport",
    # Semi-natural (exclude for purity)
    "park", "garden", "yard", "lawn", "golf", "cemetery",
    # Indoor
    "indoor", "interior", "room", "lobby", "corridor", "hallway",
    # Other artificial
    "parking", "garage", "warehouse", "factory", "construction",
    "stadium", "arena", "court", "pool", "gym",
]


def download_sun397(download_dir: Path) -> Path:
    """
    Download and extract SUN397 dataset.

    Returns path to extracted SUN397 directory.
    """
    import urllib.request
    import ssl

    # SUN397 download URL
    url = "http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz"
    tar_path = download_dir / "SUN397.tar.gz"
    extract_path = download_dir / "SUN397"

    download_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    if extract_path.exists():
        # Check if it has content
        subdirs = list(extract_path.iterdir())
        if len(subdirs) > 0:
            print(f"SUN397 already exists at {extract_path}")
            return extract_path

    # Check if tar already downloaded
    if not tar_path.exists():
        print(f"Downloading SUN397 from {url}")
        print("This may take a while (~37GB compressed)...")

        # Handle SSL
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        try:
            # Download with progress
            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100, downloaded * 100 / total_size)
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(f"\r  Downloading: {mb_downloaded:.1f}/{mb_total:.1f} MB ({percent:.1f}%)", end="", flush=True)
                else:
                    mb_downloaded = downloaded / (1024 * 1024)
                    print(f"\r  Downloading: {mb_downloaded:.1f} MB", end="", flush=True)

            # Try with urllib
            urllib.request.urlretrieve(url, tar_path, reporthook=report_progress)
            print("\n  Download complete!")

        except Exception as e:
            print(f"\nDownload failed: {e}")
            print("\nTrying alternative method with requests...")

            try:
                import requests

                response = requests.get(url, stream=True, verify=False)
                total_size = int(response.headers.get('content-length', 0))

                with open(tar_path, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = downloaded * 100 / total_size
                            mb = downloaded / (1024 * 1024)
                            print(f"\r  Downloading: {mb:.1f} MB ({percent:.1f}%)", end="", flush=True)
                print("\n  Download complete!")

            except Exception as e2:
                print(f"\nAlternative download also failed: {e2}")
                print("\nPlease download manually from:")
                print(f"  {url}")
                print(f"And save to: {tar_path}")
                sys.exit(1)

    # Extract
    print(f"\nExtracting to {extract_path}...")
    print("This may take a while...")

    with tarfile.open(tar_path, 'r:gz') as tar:
        # Get total members for progress
        members = tar.getmembers()
        total = len(members)

        for i, member in enumerate(members):
            if i % 1000 == 0:
                print(f"\r  Extracting: {i}/{total} files ({100*i/total:.1f}%)", end="", flush=True)
            tar.extract(member, download_dir)

    print(f"\n  Extraction complete!")

    # Handle nested directory structure if needed
    # SUN397.tar.gz extracts to SUN397/SUN397 sometimes
    nested = download_dir / "SUN397" / "SUN397"
    if nested.exists() and nested.is_dir():
        # Move contents up one level
        for item in nested.iterdir():
            shutil.move(str(item), str(extract_path))
        nested.rmdir()

    return extract_path


def get_sun397_classes(sun397_root: Path) -> List[str]:
    """Get all class names from SUN397 directory structure."""
    classes = []

    # SUN397 structure: SUN397/a/abbey, SUN397/b/beach, etc.
    for letter_dir in sorted(sun397_root.iterdir()):
        if letter_dir.is_dir() and len(letter_dir.name) == 1:
            for class_dir in sorted(letter_dir.iterdir()):
                if class_dir.is_dir():
                    classes.append(class_dir.name)

    return classes


def matches_keywords(class_name: str, keywords: List[str]) -> bool:
    """Check if class name contains any of the keywords."""
    class_lower = class_name.lower().replace("_", " ")

    for keyword in keywords:
        # Check for word boundary match
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        if re.search(pattern, class_lower):
            return True
        # Also check without word boundary for compound words
        if keyword.lower() in class_lower:
            return True

    return False


def filter_landscape_classes(classes: List[str]) -> List[str]:
    """Filter classes to get natural landscape only."""
    landscape_classes = []

    for cls in classes:
        # Check include
        if matches_keywords(cls, INCLUDE_KEYWORDS):
            # Check exclude
            if not matches_keywords(cls, EXCLUDE_KEYWORDS):
                landscape_classes.append(cls)

    return landscape_classes


def get_class_path(sun397_root: Path, class_name: str) -> Path:
    """Get the full path to a class directory."""
    first_letter = class_name[0].lower()
    return sun397_root / first_letter / class_name


def get_images_from_class(class_path: Path, extensions: Set[str] = {'.jpg', '.jpeg', '.png'}) -> List[Path]:
    """Get all image files from a class directory."""
    images = []
    if class_path.exists():
        for ext in extensions:
            images.extend(class_path.glob(f'*{ext}'))
            images.extend(class_path.glob(f'*{ext.upper()}'))
    return sorted(images)


def main():
    parser = argparse.ArgumentParser(description="Download and filter SUN397 for landscape images")

    # Download options
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download SUN397 dataset",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default="data/raw",
        help="Directory to download/extract SUN397 (default: data/raw)",
    )

    # Filter options
    parser.add_argument(
        "--sun397-root",
        type=str,
        help="Path to SUN397 root directory (auto-detected if --download used)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/experiment/images",
        help="Output directory for filtered images",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list matching classes, don't copy",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=15,
        help="Number of classes to select (default: 15, range: 10-20)",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=50,
        help="Images per class (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for class selection",
    )

    args = parser.parse_args()

    # Download if requested
    if args.download:
        download_dir = Path(args.download_dir)
        sun397_root = download_sun397(download_dir)
    elif args.sun397_root:
        sun397_root = Path(args.sun397_root)
    else:
        # Try default location
        sun397_root = Path("data/raw/SUN397")
        if not sun397_root.exists():
            print("ERROR: SUN397 not found. Use --download or --sun397-root")
            return

    if not sun397_root.exists():
        print(f"ERROR: SUN397 root not found: {sun397_root}")
        return

    # Get all classes
    print(f"\nScanning SUN397 at {sun397_root}...")
    all_classes = get_sun397_classes(sun397_root)
    print(f"Found {len(all_classes)} total classes")

    # Filter for landscape
    landscape_classes = filter_landscape_classes(all_classes)
    print(f"Found {len(landscape_classes)} landscape classes")

    # Print classes
    print("\n" + "="*60)
    print("LANDSCAPE CLASSES:")
    print("="*60)
    for cls in landscape_classes:
        class_path = get_class_path(sun397_root, cls)
        images = get_images_from_class(class_path)
        print(f"  {cls}: {len(images)} images")

    if args.list_only:
        return

    # Select classes with enough images
    import random
    random.seed(args.seed)

    # Filter classes that have at least per_class images
    viable_classes = []
    for cls in landscape_classes:
        class_path = get_class_path(sun397_root, cls)
        images = get_images_from_class(class_path)
        if len(images) >= args.per_class:
            viable_classes.append((cls, len(images)))

    print(f"\n{len(viable_classes)} classes have >= {args.per_class} images")

    if len(viable_classes) < args.num_classes:
        print(f"WARNING: Only {len(viable_classes)} viable classes, requested {args.num_classes}")
        selected_classes = [c[0] for c in viable_classes]
    else:
        # Randomly select num_classes
        random.shuffle(viable_classes)
        selected_classes = [c[0] for c in viable_classes[:args.num_classes]]

    print(f"\nSelected {len(selected_classes)} classes:")
    for cls in selected_classes:
        print(f"  - {cls}")

    # Copy images
    print("\n" + "="*60)
    print("COPYING IMAGES")
    print("="*60)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_copied = 0

    for cls in selected_classes:
        class_path = get_class_path(sun397_root, cls)
        images = get_images_from_class(class_path)

        # Uniform sampling: select per_class images evenly
        if len(images) > args.per_class:
            # Sample uniformly
            step = len(images) / args.per_class
            indices = [int(i * step) for i in range(args.per_class)]
            images_to_copy = [images[i] for i in indices]
        else:
            images_to_copy = images[:args.per_class]

        print(f"\n  {cls}: copying {len(images_to_copy)} images")

        for img_path in images_to_copy:
            # Create unique filename with class prefix
            new_name = f"{cls}_{img_path.name}"
            dest_path = output_dir / new_name

            if not dest_path.exists():
                shutil.copy2(img_path, dest_path)
                total_copied += 1

    print(f"\n" + "="*60)
    print(f"Total copied: {total_copied} images")
    print(f"Classes: {len(selected_classes)}")
    print(f"Per class: ~{args.per_class}")
    print(f"Output: {output_dir}")
    print("="*60)

    # Save class list
    class_list_path = output_dir / "_classes.txt"
    with open(class_list_path, 'w') as f:
        for cls in landscape_classes:
            f.write(f"{cls}\n")
    print(f"Class list saved to {class_list_path}")


if __name__ == "__main__":
    main()
