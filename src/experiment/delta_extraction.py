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

from ..models.embedder import ImageBindEmbedder
from ..effects import image_effects, audio_effects


@dataclass
class DeltaResult:
    """Container for delta extraction results."""
    modality: str  # "image" or "audio"
    effect_type: str
    intensity: str
    original_path: str
    delta: np.ndarray  # (embed_dim,)
    original_embedding: np.ndarray  # (embed_dim,)
    augmented_embedding: np.ndarray  # (embed_dim,)


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
                    'delta': d.delta.tolist(),
                    'original_embedding': d.original_embedding.tolist(),
                    'augmented_embedding': d.augmented_embedding.tolist(),
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
            dataset.add(DeltaResult(
                modality=d['modality'],
                effect_type=d['effect_type'],
                intensity=d['intensity'],
                original_path=d['original_path'],
                delta=np.array(d['delta']),
                original_embedding=np.array(d['original_embedding']),
                augmented_embedding=np.array(d['augmented_embedding']),
            ))
        return dataset


class DeltaExtractor:
    """
    Extracts delta embeddings for images and audio.

    Delta = e(augmented) - e(original)
    """

    def __init__(
        self,
        embedder: ImageBindEmbedder,
        device: str = "cpu",
    ):
        self.embedder = embedder
        self.device = device

    def extract_image_deltas(
        self,
        image_paths: List[str],
        effect_types: Optional[List[str]] = None,
        intensities: Optional[List[str]] = None,
    ) -> DeltaDataset:
        """
        Extract delta embeddings for images.

        Args:
            image_paths: List of original image paths
            effect_types: List of effects to apply (default: all)
            intensities: List of intensities (default: all)

        Returns:
            DeltaDataset with all delta results
        """
        if effect_types is None:
            effect_types = image_effects.get_effect_types()
        if intensities is None:
            intensities = image_effects.get_intensity_levels()

        dataset = DeltaDataset()

        for path in image_paths:
            # Load and embed original
            original_emb = self.embedder.embed_image_paths([path])[0].cpu().numpy()

            for effect_type in effect_types:
                for intensity in intensities:
                    # Apply effect
                    img = Image.open(path).convert("RGB")
                    augmented = image_effects.apply_effect(img, effect_type, intensity)

                    # Convert to tensor for embedding
                    import torchvision.transforms.functional as TF
                    aug_tensor = TF.to_tensor(augmented).unsqueeze(0)

                    # Embed augmented
                    aug_emb = self.embedder.embed_image(aug_tensor)[0].cpu().numpy()

                    # Compute delta
                    delta = aug_emb - original_emb

                    dataset.add(DeltaResult(
                        modality="image",
                        effect_type=effect_type,
                        intensity=intensity,
                        original_path=path,
                        delta=delta,
                        original_embedding=original_emb,
                        augmented_embedding=aug_emb,
                    ))

        return dataset

    def extract_audio_deltas(
        self,
        audio_paths: List[str],
        effect_types: Optional[List[str]] = None,
        intensities: Optional[List[str]] = None,
        sample_rate: int = 16000,
    ) -> DeltaDataset:
        """
        Extract delta embeddings for audio.

        Args:
            audio_paths: List of original audio paths
            effect_types: List of effects to apply (default: all)
            intensities: List of intensities (default: all)
            sample_rate: Target sample rate

        Returns:
            DeltaDataset with all delta results
        """
        if effect_types is None:
            effect_types = audio_effects.get_effect_types()
        if intensities is None:
            intensities = audio_effects.get_intensity_levels()

        dataset = DeltaDataset()

        for path in audio_paths:
            # Load audio
            waveform, sr = audio_effects.load_audio(path, sample_rate)

            # Embed original
            original_emb = self.embedder.embed_audio(waveform, sample_rate)[0].cpu().numpy()

            for effect_type in effect_types:
                for intensity in intensities:
                    # Apply effect
                    augmented = audio_effects.apply_effect(
                        waveform, sample_rate, effect_type, intensity
                    )

                    # Embed augmented
                    aug_emb = self.embedder.embed_audio(augmented, sample_rate)[0].cpu().numpy()

                    # Compute delta
                    delta = aug_emb - original_emb

                    dataset.add(DeltaResult(
                        modality="audio",
                        effect_type=effect_type,
                        intensity=intensity,
                        original_path=path,
                        delta=delta,
                        original_embedding=original_emb,
                        augmented_embedding=aug_emb,
                    ))

        return dataset
