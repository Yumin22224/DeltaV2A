"""
Post-training analysis for AudioController (Phase B-1).

Generates:
1) training loss curve
2) validation parameter metrics (MAE/RMSE/MSE)
3) A/B render bundle (input/pred/target) using controller_best.pt
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, random_split

from .model import AudioController
from ..database.inverse_mapping import InverseMappingDB, InverseMappingDataset
from ..effects.pedalboard_effects import EFFECT_CATALOG, PedalboardRenderer, denormalize_params


def _to_serializable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    return obj


def _load_manifest_map(manifest_path: Optional[Path]) -> Dict[int, Dict]:
    if manifest_path is None or not manifest_path.exists():
        return {}
    out: Dict[int, Dict] = {}
    with manifest_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[int(row["record_index"])] = row
    return out


def _plot_loss_curve(training_log_path: Path, out_path: Path) -> Dict:
    with training_log_path.open("r") as f:
        history = json.load(f)

    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    epochs = np.arange(1, len(train_loss) + 1)

    os.environ.setdefault("MPLCONFIGDIR", str(out_path.parent / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best_idx = int(np.argmin(val_loss))
    best_val = float(val_loss[best_idx])

    plt.figure(figsize=(10, 5), dpi=160)
    plt.plot(epochs, train_loss, label="train_loss", linewidth=2.0)
    plt.plot(epochs, val_loss, label="val_loss", linewidth=2.0)
    plt.scatter([best_idx + 1], [best_val], s=40, label=f"best val (ep {best_idx+1})")
    plt.title("Controller Training Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    return {
        "epochs": len(train_loss),
        "best_epoch": best_idx + 1,
        "best_val_loss": best_val,
        "first_train_loss": float(train_loss[0]),
        "last_train_loss": float(train_loss[-1]),
        "first_val_loss": float(val_loss[0]),
        "last_val_loss": float(val_loss[-1]),
    }


def _build_param_names(effect_names: List[str]) -> List[str]:
    names = []
    for effect_name in effect_names:
        spec = EFFECT_CATALOG[effect_name]
        for ps in spec.params:
            names.append(f"{effect_name}.{ps.name}")
    return names


def _evaluate_controller(
    db_path: Path,
    checkpoint_path: Path,
    effect_names: List[str],
    style_vocab_size: int,
    val_split: float,
    split_seed: int,
    batch_size: int,
    hidden_dims: List[int],
    dropout: float,
    device: str,
):
    db = InverseMappingDB(str(db_path))
    dataset = InverseMappingDataset(db)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    _, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(split_seed),
    )

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_cfg = ckpt.get("model_config", {})
    total_params = int(model_cfg.get("total_params", sum(EFFECT_CATALOG[name].num_params for name in effect_names)))
    model = AudioController(
        audio_embed_dim=int(model_cfg.get("audio_embed_dim", 512)),
        style_vocab_size=int(model_cfg.get("style_vocab_size", style_vocab_size)),
        total_params=total_params,
        hidden_dims=model_cfg.get("hidden_dims", hidden_dims),
        dropout=float(model_cfg.get("dropout", dropout)),
        use_activity_head=bool(model_cfg.get("use_activity_head", False)),
        num_effects=int(model_cfg.get("num_effects", len(effect_names))),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    preds = []
    gts = []
    with torch.no_grad():
        for batch in val_loader:
            clap_emb = batch["clap_embedding"].to(device)
            style = batch["style_label"].to(device)
            gt = batch["normalized_params"].to(device)
            pred = model(clap_emb, style)
            preds.append(pred.cpu().numpy())
            gts.append(gt.cpu().numpy())

    pred_arr = np.concatenate(preds, axis=0)
    gt_arr = np.concatenate(gts, axis=0)
    err = pred_arr - gt_arr

    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(np.square(err), axis=0))
    mse = np.mean(np.square(err), axis=0)

    param_names = _build_param_names(effect_names)
    rows = []
    for i, name in enumerate(param_names):
        rows.append(
            {
                "param": name,
                "mae": float(mae[i]),
                "rmse": float(rmse[i]),
                "mse": float(mse[i]),
            }
        )

    rows_sorted = sorted(rows, key=lambda x: x["rmse"], reverse=True)
    summary = {
        "dataset_size": len(dataset),
        "val_size": len(val_ds),
        "val_split": val_split,
        "split_seed": split_seed,
        "overall_mae": float(np.mean(np.abs(err))),
        "overall_rmse": float(np.sqrt(np.mean(np.square(err)))),
        "overall_mse": float(np.mean(np.square(err))),
        "top5_highest_rmse": rows_sorted[:5],
    }

    return {
        "summary": summary,
        "rows": rows,
        "val_indices": list(val_ds.indices),
        "pred_arr": pred_arr,
        "gt_arr": gt_arr,
        "sample_mae": np.mean(np.abs(err), axis=1),
        "sample_rmse": np.sqrt(np.mean(np.square(err), axis=1)),
    }


def _save_metrics_csv(rows: List[Dict], path: Path):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["param", "mae", "rmse", "mse"])
        writer.writeheader()
        writer.writerows(rows)


def _render_ab_bundle(
    out_dir: Path,
    val_indices: List[int],
    pred_arr: np.ndarray,
    gt_arr: np.ndarray,
    sample_mae: np.ndarray,
    sample_rmse: np.ndarray,
    effect_names: List[str],
    sample_rate: int,
    max_duration: float,
    manifest_map: Dict[int, Dict],
    num_renders: int,
):
    render_dir = out_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    renderer = PedalboardRenderer(sample_rate=sample_rate)

    order = np.argsort(sample_mae)[::-1]
    selected_local = order[:num_renders]

    render_manifest = []
    for rank, local_idx in enumerate(selected_local, start=1):
        record_index = int(val_indices[local_idx])
        meta = manifest_map.get(record_index, {})
        source_audio_path = meta.get("source_audio_path")
        augmented_audio_path = meta.get("augmented_audio_path")

        if not source_audio_path:
            continue
        source_path = Path(source_audio_path)
        if not source_path.exists():
            continue

        waveform, _ = librosa.load(
            str(source_path),
            sr=sample_rate,
            mono=True,
            duration=max_duration,
        )

        pred_norm = pred_arr[local_idx]
        gt_norm = gt_arr[local_idx]
        pred_params = denormalize_params(pred_norm, effect_names)
        gt_params = denormalize_params(gt_norm, effect_names)

        pred_audio = renderer.render(waveform, pred_params)
        gt_audio = renderer.render(waveform, gt_params)

        stem = f"val_rank_{rank:02d}_record_{record_index:04d}"
        in_path = render_dir / f"{stem}__input.wav"
        pred_path = render_dir / f"{stem}__pred.wav"
        target_path = render_dir / f"{stem}__target.wav"
        meta_path = render_dir / f"{stem}__meta.json"

        sf.write(str(in_path), waveform, sample_rate)
        sf.write(str(pred_path), pred_audio, sample_rate)
        sf.write(str(target_path), gt_audio, sample_rate)

        sample_info = {
            "rank_by_sample_mae": rank,
            "record_index": record_index,
            "sample_mae": float(sample_mae[local_idx]),
            "sample_rmse": float(sample_rmse[local_idx]),
            "source_audio_path": str(source_path),
            "augmented_audio_path": augmented_audio_path,
            "active_effects": meta.get("active_effects"),
            "input_wav": str(in_path),
            "pred_wav": str(pred_path),
            "target_wav": str(target_path),
            "pred_params": pred_params,
            "target_params": gt_params,
        }
        with meta_path.open("w") as f:
            json.dump(_to_serializable(sample_info), f, indent=2)
        render_manifest.append(sample_info)

    with (render_dir / "render_manifest.json").open("w") as f:
        json.dump(_to_serializable(render_manifest), f, indent=2)

    return render_manifest


def run_controller_post_train_analysis(
    artifacts_dir: str,
    out_dir: Optional[str] = None,
    val_split: float = 0.2,
    split_seed: int = 42,
    batch_size: int = 128,
    num_renders: int = 5,
    sample_rate: int = 48000,
    max_duration: float = 20.0,
    device: str = "cpu",
    manifest_path: Optional[str] = None,
    hidden_dims: Optional[List[int]] = None,
    dropout: float = 0.1,
) -> Dict:
    """
    Run post-train analysis for controller and save artifacts.
    """
    artifacts = Path(artifacts_dir)
    analysis_dir = Path(out_dir) if out_dir else (artifacts / "controller" / "analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    if hidden_dims is None:
        hidden_dims = [512, 256, 128]

    pipeline_cfg_path = artifacts / "pipeline_config.json"
    db_path = artifacts / "inverse_mapping.h5"
    ckpt_path = artifacts / "controller_best.pt"
    training_log_path = artifacts / "controller" / "training_log.json"

    if not pipeline_cfg_path.exists():
        raise FileNotFoundError(f"Missing {pipeline_cfg_path}")
    if not db_path.exists():
        raise FileNotFoundError(f"Missing {db_path}")
    if not ckpt_path.exists():
        legacy = artifacts / "controller" / "controller_best.pt"
        if legacy.exists():
            ckpt_path = legacy
        else:
            raise FileNotFoundError(f"Missing controller checkpoint at {ckpt_path}")
    if not training_log_path.exists():
        raise FileNotFoundError(f"Missing {training_log_path}")

    with pipeline_cfg_path.open("r") as f:
        pipeline_cfg = json.load(f)

    effect_names = list(pipeline_cfg["effect_names"])
    style_vocab_size = int(pipeline_cfg["aud_vocab_size"])

    loss_plot_path = analysis_dir / "controller_loss_curve.png"
    curve_summary = _plot_loss_curve(training_log_path, loss_plot_path)

    eval_out = _evaluate_controller(
        db_path=db_path,
        checkpoint_path=ckpt_path,
        effect_names=effect_names,
        style_vocab_size=style_vocab_size,
        val_split=val_split,
        split_seed=split_seed,
        batch_size=batch_size,
        hidden_dims=hidden_dims,
        dropout=dropout,
        device=device,
    )
    _save_metrics_csv(eval_out["rows"], analysis_dir / "val_param_metrics.csv")
    with (analysis_dir / "val_metrics_summary.json").open("w") as f:
        json.dump(_to_serializable(eval_out["summary"]), f, indent=2)
    with (analysis_dir / "val_param_metrics.json").open("w") as f:
        json.dump(_to_serializable(eval_out["rows"]), f, indent=2)

    manifest_file = Path(manifest_path) if manifest_path else None
    manifest_map = _load_manifest_map(manifest_file)
    render_manifest = _render_ab_bundle(
        out_dir=analysis_dir,
        val_indices=eval_out["val_indices"],
        pred_arr=eval_out["pred_arr"],
        gt_arr=eval_out["gt_arr"],
        sample_mae=eval_out["sample_mae"],
        sample_rmse=eval_out["sample_rmse"],
        effect_names=effect_names,
        sample_rate=sample_rate,
        max_duration=max_duration,
        manifest_map=manifest_map,
        num_renders=num_renders,
    )

    report = {
        "curve_summary": curve_summary,
        "val_summary": eval_out["summary"],
        "loss_curve_path": str(loss_plot_path),
        "val_metrics_csv": str(analysis_dir / "val_param_metrics.csv"),
        "val_metrics_summary_json": str(analysis_dir / "val_metrics_summary.json"),
        "render_manifest_path": str(analysis_dir / "renders" / "render_manifest.json"),
        "num_rendered_examples": len(render_manifest),
        "notes": {
            "val_split_reconstruction": "random_split with seed=42 (same as trainer default).",
            "render_selection": "Top-N validation samples by sample-level MAE.",
        },
    }
    with (analysis_dir / "analysis_report.json").open("w") as f:
        json.dump(_to_serializable(report), f, indent=2)

    return report
