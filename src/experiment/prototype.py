"""
Prototype Computation Module

Computes prototype representations for each (effect_type, intensity) combination.

Prototype: p(t, s) = E_n[Δe_n / ||Δe_n||]
(Mean of normalized deltas across samples)
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

from .delta_extraction import DeltaDataset


@dataclass
class Prototype:
    """Prototype representation for an effect."""
    modality: str
    effect_type: str
    intensity: str
    vector: np.ndarray  # (embed_dim,) normalized prototype vector
    count: int  # number of samples used
    mean_norm: float  # mean delta norm before normalization


@dataclass
class PrototypeSet:
    """Collection of prototypes."""
    prototypes: Dict[str, Prototype]  # key: "modality/effect_type/intensity"

    def get(
        self,
        modality: str,
        effect_type: str,
        intensity: str,
    ) -> Optional[Prototype]:
        """Get a specific prototype."""
        key = f"{modality}/{effect_type}/{intensity}"
        return self.prototypes.get(key)

    def get_all_by_modality(self, modality: str) -> List[Prototype]:
        """Get all prototypes for a modality."""
        return [p for p in self.prototypes.values() if p.modality == modality]

    def get_effect_types(self, modality: str) -> List[str]:
        """Get unique effect types for a modality."""
        return list(set(
            p.effect_type for p in self.prototypes.values()
            if p.modality == modality
        ))

    def get_matrix(
        self,
        modality: str,
        effect_types: Optional[List[str]] = None,
        intensities: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Get prototype matrix.

        Args:
            modality: "image" or "audio"
            effect_types: List of effect types (default: all)
            intensities: List of intensities (default: all)

        Returns:
            matrix: (len(effect_types) * len(intensities), embed_dim)
            labels: List of "effect_type/intensity" labels
            effect_order: List of effect types
        """
        protos = self.get_all_by_modality(modality)

        if effect_types is None:
            effect_types = sorted(set(p.effect_type for p in protos))
        if intensities is None:
            intensities = ["low", "mid", "high"]

        vectors = []
        labels = []

        for effect_type in effect_types:
            for intensity in intensities:
                proto = self.get(modality, effect_type, intensity)
                if proto is not None:
                    vectors.append(proto.vector)
                    labels.append(f"{effect_type}/{intensity}")

        return np.stack(vectors), labels, effect_types


def compute_prototypes(
    delta_dataset: DeltaDataset,
    normalize_deltas: bool = True,
) -> PrototypeSet:
    """
    Compute prototype representations for each (modality, effect_type, intensity).

    p(t, s) = E_n[Δe_n / ||Δe_n||]  (if normalize_deltas=True)
    p(t, s) = E_n[Δe_n]             (if normalize_deltas=False)

    Args:
        delta_dataset: Dataset containing all deltas
        normalize_deltas: Whether to normalize individual deltas before averaging

    Returns:
        PrototypeSet containing all prototypes
    """
    # Group deltas by (modality, effect_type, intensity)
    groups: Dict[str, List] = {}

    for delta_result in delta_dataset.deltas:
        key = f"{delta_result.modality}/{delta_result.effect_type}/{delta_result.intensity}"
        if key not in groups:
            groups[key] = []
        groups[key].append(delta_result)

    # Compute prototype for each group
    prototypes = {}

    for key, results in groups.items():
        modality, effect_type, intensity = key.split("/")

        # Stack deltas
        deltas = np.stack([r.delta for r in results])

        # Compute norms
        norms = np.linalg.norm(deltas, axis=1)
        mean_norm = float(np.mean(norms))

        if normalize_deltas:
            # Normalize each delta before averaging
            safe_norms = np.where(norms > 0, norms, 1.0)
            normalized = deltas / safe_norms[:, np.newaxis]
            prototype_vec = np.mean(normalized, axis=0)
        else:
            prototype_vec = np.mean(deltas, axis=0)

        # Normalize the final prototype
        proto_norm = np.linalg.norm(prototype_vec)
        if proto_norm > 0:
            prototype_vec = prototype_vec / proto_norm

        prototypes[key] = Prototype(
            modality=modality,
            effect_type=effect_type,
            intensity=intensity,
            vector=prototype_vec,
            count=len(results),
            mean_norm=mean_norm,
        )

    return PrototypeSet(prototypes=prototypes)


def compute_similarity_matrix(
    image_prototypes: PrototypeSet,
    audio_prototypes: PrototypeSet,
    image_effect_types: Optional[List[str]] = None,
    audio_effect_types: Optional[List[str]] = None,
    intensities: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Compute cosine similarity matrix between image and audio prototypes.

    Args:
        image_prototypes: PrototypeSet for images
        audio_prototypes: PrototypeSet for audio
        image_effect_types: List of image effect types (default: all)
        audio_effect_types: List of audio effect types (default: all)
        intensities: List of intensities to include

    Returns:
        sim_matrix: (num_image_effects, num_audio_effects) similarity matrix
        image_labels: Row labels
        audio_labels: Column labels
    """
    if intensities is None:
        intensities = ["low", "mid", "high"]

    # Get all image prototypes
    image_matrix, image_labels, _ = image_prototypes.get_matrix(
        "image", image_effect_types, intensities
    )

    # Get all audio prototypes
    audio_matrix, audio_labels, _ = audio_prototypes.get_matrix(
        "audio", audio_effect_types, intensities
    )

    # Compute cosine similarity (prototypes are already normalized)
    sim_matrix = image_matrix @ audio_matrix.T

    return sim_matrix, image_labels, audio_labels


def compute_effect_type_similarity(
    image_prototypes: PrototypeSet,
    audio_prototypes: PrototypeSet,
    aggregate: str = "mean",
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Compute similarity matrix between effect types (aggregated across intensities).

    Args:
        image_prototypes: PrototypeSet for images
        audio_prototypes: PrototypeSet for audio
        aggregate: How to aggregate intensities ("mean", "max")

    Returns:
        sim_matrix: (num_image_effects, num_audio_effects)
        image_effect_types: Row labels
        audio_effect_types: Column labels
    """
    intensities = ["low", "mid", "high"]

    image_effect_types = image_prototypes.get_effect_types("image")
    audio_effect_types = audio_prototypes.get_effect_types("audio")

    sim_matrix = np.zeros((len(image_effect_types), len(audio_effect_types)))

    for i, img_effect in enumerate(image_effect_types):
        for j, aud_effect in enumerate(audio_effect_types):
            sims = []
            for intensity in intensities:
                img_proto = image_prototypes.get("image", img_effect, intensity)
                aud_proto = audio_prototypes.get("audio", aud_effect, intensity)

                if img_proto is not None and aud_proto is not None:
                    sim = float(np.dot(img_proto.vector, aud_proto.vector))
                    sims.append(sim)

            if sims:
                if aggregate == "mean":
                    sim_matrix[i, j] = np.mean(sims)
                elif aggregate == "max":
                    sim_matrix[i, j] = np.max(sims)

    return sim_matrix, image_effect_types, audio_effect_types
