"""
Siamese visual encoder training (Phase B-2).

Trains VisualEncoder on precomputed image inverse-mapping DB:
  input  : CLIP(I), CLIP(I'), CLIP(I'-I)
  target : S_delta = Sim(I', IMG_VOCAB) - Sim(I, IMG_VOCAB)

Loss:
  MSE( Sim(z_pred, IMG_VOCAB), S_delta )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from .visual_encoder import VisualEncoder
from ..database import ImageInverseMappingDB, ImageInverseMappingDataset


@dataclass
class SiameseTrainConfig:
    batch_size: int = 32
    num_epochs: int = 40
    learning_rate: float = 1e-4
    val_split: float = 0.2
    seed: int = 42
    device: str = "cpu"


class VisualEncoderTrainer:
    def __init__(
        self,
        model: VisualEncoder,
        img_vocab_embeddings: np.ndarray,
        config: SiameseTrainConfig,
    ):
        self.model = model.to(config.device)
        self.cfg = config

        vocab = torch.from_numpy(img_vocab_embeddings).float().to(config.device)
        self.img_vocab = F.normalize(vocab, dim=-1)

        self.optimizer = torch.optim.AdamW(self.model.projection.parameters(), lr=config.learning_rate)
        self.history = {
            "train_mse": [],
            "val_mse": [],
            "best_val": float("inf"),
        }

    def _step(self, batch, train: bool = True) -> float:
        clip_orig = batch["clip_original"].to(self.cfg.device).float()
        clip_edit = batch["clip_edited"].to(self.cfg.device).float()
        clip_diff = batch["clip_diff"].to(self.cfg.device).float()
        style_delta = batch["style_delta"].to(self.cfg.device).float()

        z_pred = self.model.forward_from_clip_embeddings(clip_orig, clip_edit, clip_diff)
        pred_sim = z_pred @ self.img_vocab.t()
        loss = F.mse_loss(pred_sim, style_delta)

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return float(loss.item())

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
            train_total = 0.0
            n_train = 0
            for batch in train_loader:
                train_total += self._step(batch, train=True)
                n_train += 1
            train_mse = train_total / max(n_train, 1)

            self.model.eval()
            val_total = 0.0
            n_val = 0
            with torch.no_grad():
                for batch in val_loader:
                    val_total += self._step(batch, train=False)
                    n_val += 1
            val_mse = val_total / max(n_val, 1)

            self.history["train_mse"].append(train_mse)
            self.history["val_mse"].append(val_mse)

            print(
                f"Visual Epoch {epoch}/{self.cfg.num_epochs} | "
                f"train_mse={train_mse:.6f} val_mse={val_mse:.6f}"
            )

            if val_mse < self.history["best_val"]:
                self.history["best_val"] = val_mse
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "val_mse": val_mse,
                    },
                    save_path,
                )

            with open(log_path, "w") as f:
                json.dump(self.history, f, indent=2)


def train_visual_encoder(
    image_db_path: str,
    clip_embedder,
    img_vocab_embeddings: np.ndarray,
    save_path: str,
    projection_dim: int = 768,
    dropout: float = 0.1,
    batch_size: int = 32,
    num_epochs: int = 40,
    learning_rate: float = 1e-4,
    val_split: float = 0.2,
    seed: int = 42,
    device: str = "cpu",
):
    db = ImageInverseMappingDB(image_db_path)
    if not db.exists:
        raise FileNotFoundError(f"Image inverse mapping DB not found: {image_db_path}")

    dataset = ImageInverseMappingDataset(db)
    if len(dataset) < 2:
        raise ValueError(f"Need at least 2 Siamese samples, got {len(dataset)}")

    vocab_dim = int(img_vocab_embeddings.shape[1])
    if projection_dim != vocab_dim:
        raise ValueError(
            f"projection_dim ({projection_dim}) must match IMG_VOCAB dim ({vocab_dim}) "
            "to compare z_pred with IMG_VOCAB using cosine similarity."
        )

    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    if train_size < 1:
        raise ValueError(
            f"Not enough training samples after split: total={len(dataset)}, val_size={val_size}"
        )

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
        seed=seed,
        device=device,
    )

    trainer = VisualEncoderTrainer(
        model=model,
        img_vocab_embeddings=img_vocab_embeddings,
        config=cfg,
    )
    trainer.train(train_loader=train_loader, val_loader=val_loader, save_path=save_path)
