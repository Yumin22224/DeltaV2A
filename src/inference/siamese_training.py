"""
Siamese visual encoder training (Phase B-2).

Trains VisualEncoder on (I, I') pairs so z_visual aligns with IMG_VOCAB
text embedding space.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms.functional as TF

from .visual_encoder import VisualEncoder
from ..effects.wand_image_effects import apply_effect


class SiameseImagePairDataset(Dataset):
    """Builds deterministic (I, I') pairs by applying Wand effects."""

    def __init__(
        self,
        image_paths: List[str],
        effect_types: List[str],
        intensities: List[str],
        augmentations_per_image: int = 2,
        seed: int = 42,
        save_augmented: bool = False,
        augmented_dir: Optional[str] = None,
    ):
        self.image_paths = image_paths
        self.effect_types = effect_types
        self.intensities = intensities
        self.augmentations_per_image = augmentations_per_image
        self.save_augmented = save_augmented
        self.augmented_dir = Path(augmented_dir) if augmented_dir else None
        self.effect_to_idx = {e: i for i, e in enumerate(effect_types)}

        rng = np.random.default_rng(seed)
        self.samples = []
        sample_id = 0
        for path in image_paths:
            for _ in range(augmentations_per_image):
                effect = effect_types[int(rng.integers(0, len(effect_types)))]
                intensity = intensities[int(rng.integers(0, len(intensities)))]
                self.samples.append({
                    "sample_id": sample_id,
                    "image_path": path,
                    "effect": effect,
                    "intensity": intensity,
                })
                sample_id += 1

        if self.save_augmented and self.augmented_dir is None:
            raise ValueError("augmented_dir is required when save_augmented=True")
        if self.save_augmented:
            self.augmented_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        image_path = item["image_path"]
        effect = item["effect"]
        intensity = item["intensity"]

        original = Image.open(image_path).convert("RGB")
        edited = apply_effect(original, effect, intensity)

        if self.save_augmented:
            src = Path(image_path)
            category = src.parent.name
            out_dir = self.augmented_dir / category / f"{effect}_{intensity}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_name = f"{src.stem}__siamese_{item['sample_id']:06d}.png"
            out_path = out_dir / out_name
            if not out_path.exists():
                edited.save(str(out_path))

        return {
            "original": TF.to_tensor(original),  # [0,1], (3,H,W)
            "edited": TF.to_tensor(edited),      # [0,1], (3,H,W)
            "effect_idx": torch.tensor(self.effect_to_idx[effect], dtype=torch.long),
        }


@dataclass
class SiameseTrainConfig:
    batch_size: int = 32
    num_epochs: int = 40
    learning_rate: float = 1e-4
    val_split: float = 0.2
    style_temperature: float = 0.07
    contrastive_margin: float = 0.3
    loss_weight_align: float = 1.0
    loss_weight_style: float = 0.5
    loss_weight_contrastive: float = 0.2
    seed: int = 42
    device: str = "cpu"


class VisualEncoderTrainer:
    def __init__(
        self,
        model: VisualEncoder,
        clip_embedder,
        img_vocab_embeddings: np.ndarray,
        config: SiameseTrainConfig,
    ):
        self.model = model.to(config.device)
        self.clip = clip_embedder
        self.cfg = config
        vocab = torch.from_numpy(img_vocab_embeddings).float().to(config.device)
        self.img_vocab = F.normalize(vocab, dim=-1)
        self.optimizer = torch.optim.AdamW(self.model.projection.parameters(), lr=config.learning_rate)
        self.history = {
            "train_total": [],
            "train_align": [],
            "train_style": [],
            "train_contrastive": [],
            "val_total": [],
            "best_val": float("inf"),
        }

    @torch.no_grad()
    def _compute_style_targets(self, edited: torch.Tensor):
        clip_edit = self.clip.embed_images(edited).to(self.cfg.device)
        clip_edit = F.normalize(clip_edit, dim=-1)
        sims = clip_edit @ self.img_vocab.t()
        sims = torch.clamp(sims, min=0.0)
        sums = sims.sum(dim=-1, keepdim=True)
        style_soft = torch.where(
            sums > 0,
            sims / (sums + 1e-8),
            torch.full_like(sims, 1.0 / sims.shape[-1]),
        )
        target = F.normalize(style_soft @ self.img_vocab, dim=-1)
        return style_soft, target

    def _contrastive_loss(self, z: torch.Tensor, effect_idx: torch.Tensor) -> torch.Tensor:
        if z.shape[0] < 2:
            return torch.zeros((), device=z.device)
        sim = z @ z.t()
        mask = ~torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
        same = effect_idx[:, None] == effect_idx[None, :]
        pos = sim[same & mask]
        neg = sim[(~same) & mask]
        loss_pos = (1.0 - pos).pow(2).mean() if pos.numel() > 0 else torch.zeros((), device=z.device)
        loss_neg = F.relu(neg - self.cfg.contrastive_margin).pow(2).mean() if neg.numel() > 0 else torch.zeros((), device=z.device)
        return loss_pos + loss_neg

    def _step(self, batch: Dict[str, torch.Tensor], train: bool = True):
        original = batch["original"].to(self.cfg.device)
        edited = batch["edited"].to(self.cfg.device)
        effect_idx = batch["effect_idx"].to(self.cfg.device)

        style_soft, target = self._compute_style_targets(edited)
        z = self.model(original, edited)

        loss_align = (1.0 - (z * target).sum(dim=-1)).mean()
        logits = (z @ self.img_vocab.t()) / self.cfg.style_temperature
        loss_style = F.kl_div(F.log_softmax(logits, dim=-1), style_soft, reduction="batchmean")
        loss_ctr = self._contrastive_loss(z, effect_idx)

        total = (
            self.cfg.loss_weight_align * loss_align
            + self.cfg.loss_weight_style * loss_style
            + self.cfg.loss_weight_contrastive * loss_ctr
        )

        if train:
            self.optimizer.zero_grad()
            total.backward()
            self.optimizer.step()

        return {
            "total": float(total.item()),
            "align": float(loss_align.item()),
            "style": float(loss_style.item()),
            "contrastive": float(loss_ctr.item()),
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_path: str,
    ):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        log_path = save_path.with_name("visual_encoder_training_log.json")

        for epoch in range(1, self.cfg.num_epochs + 1):
            self.model.train()
            train_stats = {"total": 0.0, "align": 0.0, "style": 0.0, "contrastive": 0.0}
            n_train = 0
            for batch in train_loader:
                out = self._step(batch, train=True)
                for k in train_stats:
                    train_stats[k] += out[k]
                n_train += 1

            for k in train_stats:
                train_stats[k] /= max(n_train, 1)

            self.model.eval()
            val_total = 0.0
            n_val = 0
            with torch.no_grad():
                for batch in val_loader:
                    out = self._step(batch, train=False)
                    val_total += out["total"]
                    n_val += 1
            val_total /= max(n_val, 1)

            self.history["train_total"].append(train_stats["total"])
            self.history["train_align"].append(train_stats["align"])
            self.history["train_style"].append(train_stats["style"])
            self.history["train_contrastive"].append(train_stats["contrastive"])
            self.history["val_total"].append(val_total)

            print(
                f"Visual Epoch {epoch}/{self.cfg.num_epochs} | "
                f"train={train_stats['total']:.5f} val={val_total:.5f}"
            )

            if val_total < self.history["best_val"]:
                self.history["best_val"] = val_total
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "val_total": val_total,
                    },
                    save_path,
                )

            with open(log_path, "w") as f:
                json.dump(self.history, f, indent=2)


def train_visual_encoder(
    image_paths: List[str],
    clip_embedder,
    img_vocab_embeddings: np.ndarray,
    save_path: str,
    projection_dim: int = 768,
    dropout: float = 0.1,
    batch_size: int = 32,
    num_epochs: int = 40,
    learning_rate: float = 1e-4,
    val_split: float = 0.2,
    augmentations_per_image: int = 2,
    effect_types: Optional[List[str]] = None,
    intensities: Optional[List[str]] = None,
    save_augmented: bool = False,
    augmented_dir: Optional[str] = None,
    style_temperature: float = 0.07,
    contrastive_margin: float = 0.3,
    loss_weight_align: float = 1.0,
    loss_weight_style: float = 0.5,
    loss_weight_contrastive: float = 0.2,
    seed: int = 42,
    device: str = "cpu",
):
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

    dataset = SiameseImagePairDataset(
        image_paths=image_paths,
        effect_types=effect_types,
        intensities=intensities,
        augmentations_per_image=augmentations_per_image,
        seed=seed,
        save_augmented=save_augmented,
        augmented_dir=augmented_dir,
    )
    if len(dataset) < 2:
        raise ValueError(f"Need at least 2 Siamese samples, got {len(dataset)}")

    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = VisualEncoder(
        clip_embedder=clip_embedder,
        projection_dim=projection_dim,
        dropout=dropout,
    )

    cfg = SiameseTrainConfig(
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        val_split=val_split,
        style_temperature=style_temperature,
        contrastive_margin=contrastive_margin,
        loss_weight_align=loss_weight_align,
        loss_weight_style=loss_weight_style,
        loss_weight_contrastive=loss_weight_contrastive,
        seed=seed,
        device=device,
    )

    trainer = VisualEncoderTrainer(
        model=model,
        clip_embedder=clip_embedder,
        img_vocab_embeddings=img_vocab_embeddings,
        config=cfg,
    )
    trainer.train(train_loader=train_loader, val_loader=val_loader, save_path=save_path)
