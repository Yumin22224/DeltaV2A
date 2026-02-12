"""
Image inverse-mapping database for Siamese training (Phase A-4).

Builds deterministic-supervised records from random Wand edits:
  - input context: CLIP(I), CLIP(I'), CLIP(I'-I)
  - target label:  Sim(I', IMG_VOCAB) - Sim(I, IMG_VOCAB)

Each record:
  - clip_original: (D,) float32
  - clip_edited: (D,) float32
  - clip_diff: (D,) float32
  - style_delta: (|IMG_VOCAB|,) float32
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF

from ..effects.wand_image_effects import apply_effect


class ImageInverseMappingDB:
    """HDF5-backed database of image inverse-mapping records."""

    def __init__(self, path: str):
        self.path = path

    @property
    def exists(self) -> bool:
        return Path(self.path).exists()

    def get_size(self) -> int:
        with h5py.File(self.path, "r") as f:
            return int(f.attrs.get("actual_records", f["clip_original"].shape[0]))

    def get_metadata(self) -> Dict:
        with h5py.File(self.path, "r") as f:
            return dict(f.attrs)


class ImageInverseMappingDataset:
    """PyTorch dataset for image inverse-mapping records."""

    def __init__(self, db: ImageInverseMappingDB):
        self.db = db
        with h5py.File(db.path, "r") as f:
            actual = int(f.attrs.get("actual_records", f["clip_original"].shape[0]))
            self.clip_original = f["clip_original"][:actual]
            self.clip_edited = f["clip_edited"][:actual]
            self.clip_diff = f["clip_diff"][:actual]
            self.style_delta = f["style_delta"][:actual]

        self._size = int(self.clip_original.shape[0])

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int):
        return {
            "clip_original": torch.from_numpy(self.clip_original[idx]),
            "clip_edited": torch.from_numpy(self.clip_edited[idx]),
            "clip_diff": torch.from_numpy(self.clip_diff[idx]),
            "style_delta": torch.from_numpy(self.style_delta[idx]),
        }


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec


def _style_sim_delta(
    clip_orig: np.ndarray,
    clip_edit: np.ndarray,
    img_vocab_embeddings: np.ndarray,
) -> np.ndarray:
    sim_orig = img_vocab_embeddings @ clip_orig
    sim_edit = img_vocab_embeddings @ clip_edit
    return (sim_edit - sim_orig).astype(np.float32)


def build_image_inverse_mapping_db(
    image_paths: List[str],
    clip_embedder,
    img_vocab_embeddings: np.ndarray,
    output_path: str,
    augmentations_per_image: int = 2,
    effect_types: Optional[List[str]] = None,
    intensities: Optional[List[str]] = None,
    seed: int = 42,
    save_augmented_images: bool = False,
    augmented_image_dir: Optional[str] = None,
) -> ImageInverseMappingDB:
    """Build image inverse-mapping DB for Siamese training."""
    if effect_types is None:
        effect_types = [
            "adaptive_blur",
            "motion_blur",
            "adaptive_sharpen",
            "add_noise",
            "spread",
            "sepia_tone",
            "solarize",
        ]
    if intensities is None:
        intensities = ["low", "mid", "high"]
    if augmentations_per_image <= 0:
        raise ValueError("augmentations_per_image must be > 0")
    if not image_paths:
        raise ValueError("image_paths is empty")

    rng = np.random.default_rng(seed)
    img_vocab = img_vocab_embeddings.astype(np.float32)
    # Ensure cosine similarity semantics.
    img_vocab = img_vocab / np.maximum(np.linalg.norm(img_vocab, axis=1, keepdims=True), 1e-8)

    total_records = len(image_paths) * augmentations_per_image
    vocab_size = int(img_vocab.shape[0])
    clip_dim = int(img_vocab.shape[1])

    print("Building image inverse-mapping database...")
    print(f"  Images: {len(image_paths)}")
    print(f"  Augmentations/image: {augmentations_per_image}")
    print(f"  Total records (planned): {total_records}")
    print(f"  Effect types: {effect_types}")
    print(f"  Intensities: {intensities}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    aug_root = None
    manifest_path = None
    if save_augmented_images:
        if not augmented_image_dir:
            raise ValueError("augmented_image_dir must be set when save_augmented_images=True")
        aug_root = Path(augmented_image_dir)
        aug_root.mkdir(parents=True, exist_ok=True)
        manifest_path = aug_root / "manifest.jsonl"
        print(f"  Augmented image dir: {aug_root}")

    with h5py.File(output_path, "w") as f:
        clip_orig_ds = f.create_dataset("clip_original", shape=(total_records, clip_dim), dtype="float32")
        clip_edit_ds = f.create_dataset("clip_edited", shape=(total_records, clip_dim), dtype="float32")
        clip_diff_ds = f.create_dataset("clip_diff", shape=(total_records, clip_dim), dtype="float32")
        style_delta_ds = f.create_dataset("style_delta", shape=(total_records, vocab_size), dtype="float32")

        f.attrs["clip_dim"] = clip_dim
        f.attrs["vocab_size"] = vocab_size
        f.attrs["augmentations_per_image"] = int(augmentations_per_image)
        f.attrs["effect_types"] = json.dumps(effect_types)
        f.attrs["intensities"] = json.dumps(intensities)

        record_idx = 0
        with open(manifest_path, "w") if manifest_path else nullcontext() as manifest_fp:
            for image_path in tqdm(image_paths, desc="Building image inverse mapping DB"):
                source_path = Path(image_path)
                try:
                    original_pil = Image.open(source_path).convert("RGB")
                except Exception as e:
                    print(f"Warning: Failed to open image {image_path}: {e}")
                    continue

                original_tensor = TF.to_tensor(original_pil).unsqueeze(0)

                for aug_idx in range(augmentations_per_image):
                    effect = str(effect_types[int(rng.integers(0, len(effect_types)))])
                    intensity = str(intensities[int(rng.integers(0, len(intensities)))])

                    try:
                        edited_pil = apply_effect(original_pil, effect, intensity)
                        edited_tensor = TF.to_tensor(edited_pil).unsqueeze(0)
                        diff_tensor = torch.clamp((edited_tensor - original_tensor + 1.0) / 2.0, 0.0, 1.0)

                        with torch.no_grad():
                            clip_orig = clip_embedder.embed_images(original_tensor)[0].detach().cpu().numpy()
                            clip_edit = clip_embedder.embed_images(edited_tensor)[0].detach().cpu().numpy()
                            clip_diff = clip_embedder.embed_images(diff_tensor)[0].detach().cpu().numpy()

                        clip_orig = _l2_normalize(clip_orig)
                        clip_edit = _l2_normalize(clip_edit)
                        clip_diff = _l2_normalize(clip_diff)
                        style_delta = _style_sim_delta(clip_orig, clip_edit, img_vocab)

                        clip_orig_ds[record_idx] = clip_orig
                        clip_edit_ds[record_idx] = clip_edit
                        clip_diff_ds[record_idx] = clip_diff
                        style_delta_ds[record_idx] = style_delta

                        aug_relpath = None
                        if aug_root is not None:
                            category = source_path.parent.name
                            aug_subdir = aug_root / category / f"{effect}_{intensity}"
                            aug_subdir.mkdir(parents=True, exist_ok=True)
                            aug_name = f"{source_path.stem}__img_aug_{aug_idx:04d}.png"
                            aug_path = aug_subdir / aug_name
                            edited_pil.save(str(aug_path))
                            aug_relpath = str(aug_path)

                        if manifest_fp is not None:
                            manifest_fp.write(json.dumps({
                                "record_index": record_idx,
                                "source_image_path": str(source_path),
                                "augmented_image_path": aug_relpath,
                                "effect": effect,
                                "intensity": intensity,
                            }) + "\n")

                        record_idx += 1
                    except Exception as e:
                        print(f"Warning: Failed augmentation for {image_path}: {e}")
                        continue

        f.attrs["actual_records"] = int(record_idx)

    print(f"Saved {record_idx} image records to {output_path}")
    return ImageInverseMappingDB(output_path)
