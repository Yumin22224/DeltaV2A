"""
Delta Extraction Module

Computes delta embeddings: Δe = e(augmented) - e(original)
"""

from typing import Dict, List, Tuple, Optional
from pathlib import Path
import torch
import numpy as np
from dataclasses import dataclass, field
from PIL import Image
import torchaudio

from src.models.multimodal_embedder import MultimodalEmbedder
from .effects import image_effects, audio_effects


@dataclass
class DeltaResult:
    """Container for delta extraction results."""
    modality: str  # "image" or "audio"
    effect_type: str
    intensity: str
    original_path: str
    category: str  # NEW: category from folder structure
    delta: np.ndarray  # (embed_dim,)
    original_embedding: np.ndarray  # (embed_dim,)
    augmented_embedding: np.ndarray  # (embed_dim,)
    embedding_dim: int  # NEW: 768 for image, 512 for audio


@dataclass
class DeltaDataset:
    """Collection of delta results."""
    deltas: List[DeltaResult] = field(default_factory=list)

    def add(self, result: DeltaResult):
        self.deltas.append(result)

    def filter_by(
        self,
        modality: Optional[str] = None,
        effect_type: Optional[str] = None,
        intensity: Optional[str] = None,
    ) -> List[DeltaResult]:
        """Filter deltas by criteria."""
        results = self.deltas
        if modality:
            results = [d for d in results if d.modality == modality]
        if effect_type:
            results = [d for d in results if d.effect_type == effect_type]
        if intensity:
            results = [d for d in results if d.intensity == intensity]
        return results

    def get_delta_matrix(
        self,
        modality: str,
        effect_type: str,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Get delta matrix for a specific effect type.

        Args:
            modality: "image" or "audio"
            effect_type: Effect type name
            normalize: Whether to L2-normalize deltas

        Returns:
            deltas: (N, embed_dim) array
            intensities: List of intensity labels
        """
        results = self.filter_by(modality=modality, effect_type=effect_type)
        if not results:
            return np.array([]), []

        deltas = np.stack([r.delta for r in results])
        intensities = [r.intensity for r in results]

        if normalize:
            norms = np.linalg.norm(deltas, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)
            deltas = deltas / norms

        return deltas, intensities

    def save(self, path: str):
        """Save dataset to file."""
        data = {
            'deltas': [
                {
                    'modality': d.modality,
                    'effect_type': d.effect_type,
                    'intensity': d.intensity,
                    'original_path': d.original_path,
                    'category': d.category,
                    'delta': d.delta.tolist(),
                    'original_embedding': d.original_embedding.tolist(),
                    'augmented_embedding': d.augmented_embedding.tolist(),
                    'embedding_dim': d.embedding_dim,
                }
                for d in self.deltas
            ]
        }
        import json
        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> 'DeltaDataset':
        """Load dataset from file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)

        dataset = cls()
        for d in data['deltas']:
            # Backward compatibility: extract category if missing
            category = d.get('category')
            if category is None:
                category = get_category_from_path(d['original_path'])

            # Backward compatibility: infer embedding_dim if missing
            embedding_dim = d.get('embedding_dim')
            if embedding_dim is None:
                embedding_dim = len(d['delta'])

            dataset.add(DeltaResult(
                modality=d['modality'],
                effect_type=d['effect_type'],
                intensity=d['intensity'],
                original_path=d['original_path'],
                category=category,
                delta=np.array(d['delta']),
                original_embedding=np.array(d['original_embedding']),
                augmented_embedding=np.array(d['augmented_embedding']),
                embedding_dim=embedding_dim,
            ))
        return dataset


def get_category_from_path(path: str) -> str:
    """Extract category from folder structure."""
    return Path(path).parent.name


class DeltaExtractor:
    """
    Extracts delta embeddings for images and audio.

    Delta = e(augmented) - e(original)
    """

    def __init__(
        self,
        embedder: MultimodalEmbedder,
        device: str = "cpu",
    ):
        self.embedder = embedder
        self.device = device

    def extract_image_deltas(
        self,
        image_paths: List[str],
        effect_types: Optional[List[str]] = None,
        intensities: Optional[List[str]] = None,
        save_augmented: bool = False,
        augmented_dir: Optional[str] = None,
    ) -> DeltaDataset:
        """
        Extract delta embeddings for images.

        Args:
            image_paths: List of original image paths
            effect_types: List of effects to apply (default: all)
            intensities: List of intensities (default: all)
            save_augmented: Whether to save augmented images to disk
            augmented_dir: Directory to save augmented images (required if save_augmented=True)

        Returns:
            DeltaDataset with all delta results
        """
        if effect_types is None:
            effect_types = image_effects.get_effect_types()
        if intensities is None:
            intensities = image_effects.get_intensity_levels()

        if save_augmented and augmented_dir is None:
            raise ValueError("augmented_dir must be provided when save_augmented=True")

        if save_augmented:
            augmented_root = Path(augmented_dir) / "images"
            augmented_root.mkdir(parents=True, exist_ok=True)

        dataset = DeltaDataset()

        for path in image_paths:
            # Load and embed original
            original_emb = self.embedder.embed_image_paths([path])[0].cpu().numpy()

            # Extract category
            category = get_category_from_path(path)
            filename = Path(path).name

            for effect_type in effect_types:
                for intensity in intensities:
                    # Apply effect
                    img = Image.open(path).convert("RGB")
                    augmented = image_effects.apply_effect(img, effect_type, intensity)

                    # Save augmented image if requested
                    if save_augmented:
                        aug_subdir = augmented_root / category / f"{effect_type}_{intensity}"
                        aug_subdir.mkdir(parents=True, exist_ok=True)
                        aug_path = aug_subdir / filename
                        augmented.save(str(aug_path))

                    # Convert to tensor for embedding
                    import torchvision.transforms.functional as TF
                    aug_tensor = TF.to_tensor(augmented).unsqueeze(0)

                    # Embed augmented
                    aug_emb = self.embedder.embed_images(aug_tensor)[0].cpu().numpy()

                    # Compute delta
                    delta = aug_emb - original_emb

                    dataset.add(DeltaResult(
                        modality="image",
                        effect_type=effect_type,
                        intensity=intensity,
                        original_path=path,
                        category=category,
                        delta=delta,
                        original_embedding=original_emb,
                        augmented_embedding=aug_emb,
                        embedding_dim=self.embedder.image_dim,
                    ))

        return dataset

    def extract_audio_deltas(
        self,
        audio_paths: List[str],
        effect_types: Optional[List[str]] = None,
        intensities: Optional[List[str]] = None,
        sample_rate: int = 48000,  # Updated to CLAP's 48kHz
        save_augmented: bool = False,
        augmented_dir: Optional[str] = None,
    ) -> DeltaDataset:
        """
        Extract delta embeddings for audio.

        Args:
            audio_paths: List of original audio paths
            effect_types: List of effects to apply (default: all)
            intensities: List of intensities (default: all)
            sample_rate: Target sample rate
            save_augmented: Whether to save augmented audio to disk
            augmented_dir: Directory to save augmented audio (required if save_augmented=True)

        Returns:
            DeltaDataset with all delta results
        """
        if effect_types is None:
            effect_types = audio_effects.get_effect_types()
        if intensities is None:
            intensities = audio_effects.get_intensity_levels()

        if save_augmented and augmented_dir is None:
            raise ValueError("augmented_dir must be provided when save_augmented=True")

        if save_augmented:
            augmented_root = Path(augmented_dir) / "audio"
            augmented_root.mkdir(parents=True, exist_ok=True)

        dataset = DeltaDataset()

        for path in audio_paths:
            # Load audio
            waveform, sr = audio_effects.load_audio(path, sample_rate)

            # Embed original
            original_emb = self.embedder.embed_audio(waveform, sample_rate)[0].cpu().numpy()

            # Extract category
            category = get_category_from_path(path)
            filename = Path(path).stem + ".wav"  # Save as WAV

            for effect_type in effect_types:
                for intensity in intensities:
                    # Apply effect
                    augmented = audio_effects.apply_effect(
                        waveform, sample_rate, effect_type, intensity
                    )

                    # Save augmented audio if requested
                    if save_augmented:
                        aug_subdir = augmented_root / category / f"{effect_type}_{intensity}"
                        aug_subdir.mkdir(parents=True, exist_ok=True)
                        aug_path = aug_subdir / filename

                        # Save as WAV
                        import soundfile as sf
                        sf.write(str(aug_path), augmented.cpu().numpy().T, sample_rate)

                    # Embed augmented
                    aug_emb = self.embedder.embed_audio(augmented, sample_rate)[0].cpu().numpy()

                    # Compute delta
                    delta = aug_emb - original_emb

                    dataset.add(DeltaResult(
                        modality="audio",
                        effect_type=effect_type,
                        intensity=intensity,
                        original_path=path,
                        category=category,
                        delta=delta,
                        original_embedding=original_emb,
                        augmented_embedding=aug_emb,
                        embedding_dim=self.embedder.audio_dim,
                    ))

        return dataset
