#!/usr/bin/env python
"""
Places365 Landscape Filter

Downloads Places365 dataset and filters for natural landscape images.

Usage:
    # Download Places365 validation set (smaller, ~900MB)
    python scripts/prepare_places365_landscape.py --download --split val --download-dir data/raw

    # Download Places365 train set (larger, ~24GB for standard)
    python scripts/prepare_places365_landscape.py --download --split train --download-dir data/raw

    # List matching classes
    python scripts/prepare_places365_landscape.py --places365-root data/raw/places365 --list-only

    # Copy filtered images to experiment directory
    python scripts/prepare_places365_landscape.py --places365-root data/raw/places365 --output data/experiment/images

    # All in one
    python scripts/prepare_places365_landscape.py --download --split val --download-dir data/raw --output data/experiment/images
"""

import argparse
import shutil
import sys
import tarfile
from pathlib import Path
from typing import List, Tuple
import re


# Exact category names to include (natural landscapes, minimal human presence)
# Based on Places365 categories_places365.txt
INCLUDE_CATEGORIES = {
    # Arid/Desert
    "badlands",
    "butte",
    "canyon",
    "desert/sand",
    "desert/vegetation",
    "rock_arch",
    "volcano",

    # Mountains/Hills
    "cliff",
    "crevasse",
    "mountain",
    "mountain_path",
    "mountain_snowy",
    "valley",

    # Forest/Woods
    "bamboo_forest",
    "forest/broadleaf",
    "forest_path",
    "rainforest",

    # Water bodies
    "coast",
    "creek",
    "glacier",
    "hot_spring",
    "ice_floe",
    "ice_shelf",
    "iceberg",
    "islet",
    "lagoon",
    "lake/natural",
    "marsh",
    "ocean",
    "pond",
    "river",
    "swamp",
    "waterfall",
    "wave",

    # Fields/Plains
    "corn_field",
    "field/cultivated",
    "field/wild",
    "hayfield",
    "pasture",
    "rice_paddy",
    "snowfield",
    "tundra",
    "wheat_field",

    # Sky
    "sky",

    # Underwater
    "underwater/ocean_deep",
}


def download_places365(download_dir: Path, split: str = "val") -> Path:
    """
    Download Places365 dataset.

    Args:
        download_dir: Directory to download to
        split: "val" (validation, ~500MB) or "train" (training, ~24GB)

    Returns:
        Path to places365 directory
    """
    import ssl

    # Disable SSL verification for macOS
    ssl._create_default_https_context = ssl._create_unverified_context

    download_dir.mkdir(parents=True, exist_ok=True)

    # URLs from http://places2.csail.mit.edu/download.html (use HTTPS)
    if split == "val":
        url = "https://data.csail.mit.edu/places/places365/val_256.tar"
        tar_name = "val_256.tar"
        extract_name = "val_256"
        size_desc = "~500MB"
    else:  # train
        url = "https://data.csail.mit.edu/places/places365/train_256_places365standard.tar"
        tar_name = "train_256_places365standard.tar"
        extract_name = "data_256"
        size_desc = "~24GB"

    tar_path = download_dir / tar_name
    places_root = download_dir / "places365"

    # Check if already extracted
    extract_path = places_root / extract_name
    if extract_path.exists() and any(extract_path.iterdir()):
        print(f"Places365 {split} already exists at {extract_path}")
        return places_root

    # Also need categories file
    categories_url = "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt"
    categories_path = places_root / "categories_places365.txt"

    places_root.mkdir(parents=True, exist_ok=True)

    # Download categories if needed
    if not categories_path.exists():
        print("Downloading categories file...")
        try:
            import requests
            response = requests.get(categories_url, verify=False)
            categories_path.write_text(response.text)
            print("  Categories downloaded")
        except Exception as e:
            print(f"  Warning: Could not download categories: {e}")

    # Download dataset if needed
    if not tar_path.exists():
        print(f"Downloading Places365 {split} ({size_desc}) from:")
        print(f"  {url}")
        print("This may take a while...")

        try:
            import requests

            response = requests.get(url, stream=True, verify=False)
            total = int(response.headers.get('content-length', 0))

            with open(tar_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=65536):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        mb = downloaded / (1024 * 1024)
                        mb_total = total / (1024 * 1024)
                        percent = 100 * downloaded / total
                        print(f"\r  {mb:.1f}/{mb_total:.1f} MB ({percent:.1f}%)", end="", flush=True)
                    else:
                        mb = downloaded / (1024 * 1024)
                        print(f"\r  {mb:.1f} MB", end="", flush=True)

            print("\n  Download complete!")

        except Exception as e:
            print(f"\nDownload failed: {e}")
            print("\nPlease download manually with curl:")
            print(f"  curl -L -o {tar_path} '{url}'")
            sys.exit(1)

    # Extract
    print(f"\nExtracting to {places_root}...")

    with tarfile.open(tar_path, 'r') as tar:
        members = tar.getmembers()
        total = len(members)
        for i, member in enumerate(members):
            if i % 5000 == 0:
                print(f"\r  Extracting: {i}/{total} ({100*i/total:.1f}%)", end="", flush=True)
            tar.extract(member, places_root)

    print(f"\n  Extraction complete!")

    return places_root


def load_category_mapping(places_root: Path) -> dict:
    """Load category ID to name mapping from categories_places365.txt."""
    categories_file = places_root / "categories_places365.txt"
    id_to_name = {}

    if categories_file.exists():
        with open(categories_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    # Format: /a/abbey 0
                    full_name = parts[0]
                    cat_id = int(parts[1])
                    # Extract category name (remove /a/ prefix)
                    cat_name = full_name.split('/', 2)[-1] if '/' in full_name else full_name
                    id_to_name[cat_id] = cat_name

    return id_to_name


def load_val_labels(places_root: Path) -> dict:
    """Load validation image to category ID mapping."""
    labels_file = places_root / "places365_val.txt"
    image_to_cat = {}

    if labels_file.exists():
        with open(labels_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    image_name = parts[0]
                    cat_id = int(parts[1])
                    image_to_cat[image_name] = cat_id

    return image_to_cat


def get_places365_images_by_category(
    places_root: Path,
    split: str = "val"
) -> dict:
    """
    Get images organized by category.

    Returns dict: {category_name: [image_paths]}
    """
    # Load mappings
    id_to_name = load_category_mapping(places_root)

    if split == "val":
        data_dir = places_root / "val_256"
        image_to_cat = load_val_labels(places_root)

        # Group images by category
        cat_to_images = {}
        for image_name, cat_id in image_to_cat.items():
            cat_name = id_to_name.get(cat_id, f"unknown_{cat_id}")
            image_path = data_dir / image_name

            if image_path.exists() and not image_path.name.startswith("._"):
                if cat_name not in cat_to_images:
                    cat_to_images[cat_name] = []
                cat_to_images[cat_name].append(image_path)

        return cat_to_images

    else:  # train - has directory structure
        data_dir = places_root / "data_256"
        cat_to_images = {}

        if data_dir.exists():
            for cat_dir in data_dir.iterdir():
                if cat_dir.is_dir():
                    images = [
                        p for p in cat_dir.glob("*.jpg")
                        if not p.name.startswith("._")
                    ]
                    if images:
                        cat_to_images[cat_dir.name] = images

        return cat_to_images


def filter_landscape_categories(cat_to_images: dict) -> dict:
    """Filter to get natural landscape categories only."""
    return {
        cat: images
        for cat, images in cat_to_images.items()
        if cat in INCLUDE_CATEGORIES
    }


def main():
    parser = argparse.ArgumentParser(description="Download and filter Places365 for landscape images")

    # Download options
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download Places365 dataset",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default="data/raw",
        help="Directory to download Places365 (default: data/raw)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "train"],
        default="val",
        help="Dataset split: val (~900MB) or train (~24GB)",
    )

    # Filter options
    parser.add_argument(
        "--places365-root",
        type=str,
        help="Path to Places365 root directory",
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
        help="Number of classes to select (default: 15)",
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

    # Determine places365 root
    if args.download:
        download_dir = Path(args.download_dir)
        places_root = download_places365(download_dir, args.split)
    elif args.places365_root:
        places_root = Path(args.places365_root)
    else:
        places_root = Path("data/raw/places365")

    if not places_root.exists():
        print(f"ERROR: Places365 not found at {places_root}")
        print("Use --download to download, or --places365-root to specify location")
        return

    # Get all images by category
    print(f"\nScanning Places365 at {places_root}...")
    cat_to_images = get_places365_images_by_category(places_root, args.split)
    print(f"Found {len(cat_to_images)} total categories")

    # Filter for landscape
    landscape_cats = filter_landscape_categories(cat_to_images)
    print(f"Found {len(landscape_cats)} landscape categories")

    # Print categories
    print("\n" + "="*60)
    print("LANDSCAPE CATEGORIES:")
    print("="*60)
    for cat_name in sorted(landscape_cats.keys()):
        images = landscape_cats[cat_name]
        print(f"  {cat_name}: {len(images)} images")

    if args.list_only:
        return

    # Select categories with enough images
    import random
    random.seed(args.seed)

    viable_cats = [
        (cat, images)
        for cat, images in landscape_cats.items()
        if len(images) >= args.per_class
    ]

    print(f"\n{len(viable_cats)} categories have >= {args.per_class} images")

    if len(viable_cats) < args.num_classes:
        print(f"WARNING: Only {len(viable_cats)} viable categories")
        selected = viable_cats
    else:
        random.shuffle(viable_cats)
        selected = viable_cats[:args.num_classes]

    print(f"\nSelected {len(selected)} categories:")
    for cat_name, images in selected:
        print(f"  - {cat_name} ({len(images)} images)")

    # Copy images
    print("\n" + "="*60)
    print("COPYING IMAGES")
    print("="*60)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_copied = 0

    for cat_name, images in selected:
        # Sort for consistent ordering
        images = sorted(images)

        # Uniform sampling
        if len(images) > args.per_class:
            step = len(images) / args.per_class
            indices = [int(i * step) for i in range(args.per_class)]
            images_to_copy = [images[i] for i in indices]
        else:
            images_to_copy = images[:args.per_class]

        # Create category subfolder
        safe_cat = cat_name.replace("/", "_").replace(" ", "_")
        cat_dir = output_dir / safe_cat
        cat_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  {cat_name}: copying {len(images_to_copy)} images -> {safe_cat}/")

        for img_path in images_to_copy:
            dest_path = cat_dir / img_path.name

            if not dest_path.exists():
                shutil.copy2(img_path, dest_path)
                total_copied += 1

    print(f"\n" + "="*60)
    print(f"Total copied: {total_copied} images")
    print(f"Categories: {len(selected)}")
    print(f"Per category: ~{args.per_class}")
    print(f"Output: {output_dir}")
    print("="*60)

    # Save category list
    class_list_path = output_dir / "_categories.txt"
    with open(class_list_path, 'w') as f:
        for cat_name, _ in selected:
            f.write(f"{cat_name}\n")
    print(f"Category list saved to {class_list_path}")


if __name__ == "__main__":
    main()
