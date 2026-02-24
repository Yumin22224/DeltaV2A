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
from ..effects.pedalboard_effects import (
    EFFECT_CATALOG,
    PedalboardRenderer,
    denormalize_params,
    normalize_params,
)


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


def _effect_param_slices(effect_names: List[str]) -> List[slice]:
    slices: List[slice] = []
    start = 0
    for effect_name in effect_names:
        width = EFFECT_CATALOG[effect_name].num_params
        slices.append(slice(start, start + width))
        start += width
    return slices


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
    activity_thresholds: Optional[np.ndarray] = None,
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
        fusion_mode=str(model_cfg.get("fusion_mode", "concat")),
        audio_gate_bias=float(model_cfg.get("audio_gate_bias", -2.0)),
        use_activity_head=bool(model_cfg.get("use_activity_head", False)),
        num_effects=int(model_cfg.get("num_effects", len(effect_names))),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    preds = []
    gts = []
    effect_masks = []
    activity_probs_all = []
    activity_pred_all = []
    with torch.no_grad():
        for batch in val_loader:
            clap_emb = batch["clap_embedding"].to(device)
            style = batch["style_label"].to(device)
            gt = batch["normalized_params"].to(device)
            eff_mask = batch.get("effect_active_mask")
            pred, activity_logits = model.forward_with_activity(clap_emb, style)
            preds.append(pred.cpu().numpy())
            gts.append(gt.cpu().numpy())
            if eff_mask is not None:
                effect_masks.append(eff_mask.cpu().numpy())
            if activity_logits is not None:
                probs = torch.sigmoid(activity_logits).cpu().numpy()
                if activity_thresholds is not None and activity_thresholds.shape[0] == probs.shape[1]:
                    thresholds = activity_thresholds[None, :]
                else:
                    thresholds = np.full((1, probs.shape[1]), 0.5, dtype=np.float32)
                activity_probs_all.append(probs)
                activity_pred_all.append((probs >= thresholds).astype(np.float32))

    pred_arr = np.concatenate(preds, axis=0)
    gt_arr = np.concatenate(gts, axis=0)
    err = pred_arr - gt_arr
    effect_mask_arr = np.concatenate(effect_masks, axis=0) if effect_masks else None

    pred_gated_arr = pred_arr.copy()
    if effect_mask_arr is not None and activity_pred_all:
        activity_pred_arr = np.concatenate(activity_pred_all, axis=0)
        bypass_norm = normalize_params({}, effect_names)
        slices = _effect_param_slices(effect_names)
        for i, sl in enumerate(slices):
            inactive = activity_pred_arr[:, i] < 0.5
            if np.any(inactive):
                pred_gated_arr[inactive, sl] = bypass_norm[sl]
    else:
        activity_pred_arr = np.concatenate(activity_pred_all, axis=0) if activity_pred_all else None

    err_gated = pred_gated_arr - gt_arr

    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(np.square(err), axis=0))
    mse = np.mean(np.square(err), axis=0)

    mae_gated = np.mean(np.abs(err_gated), axis=0)
    rmse_gated = np.sqrt(np.mean(np.square(err_gated), axis=0))
    mse_gated = np.mean(np.square(err_gated), axis=0)

    param_names = _build_param_names(effect_names)
    rows = []
    active_param_mask = None
    active_counts = None
    if effect_mask_arr is not None:
        slices = _effect_param_slices(effect_names)
        chunks = []
        for i, sl in enumerate(slices):
            width = sl.stop - sl.start
            chunks.append(np.repeat(effect_mask_arr[:, i:i + 1], width, axis=1))
        active_param_mask = np.concatenate(chunks, axis=1).astype(np.float32)
        active_counts = active_param_mask.sum(axis=0)

    for i, name in enumerate(param_names):
        row = {
            "param": name,
            "mae": float(mae[i]),
            "rmse": float(rmse[i]),
            "mse": float(mse[i]),
            "mae_gated": float(mae_gated[i]),
            "rmse_gated": float(rmse_gated[i]),
            "mse_gated": float(mse_gated[i]),
        }
        if active_param_mask is not None and active_counts is not None and active_counts[i] > 0:
            mask_i = active_param_mask[:, i] > 0.5
            row["active_count"] = int(active_counts[i])
            row["active_mae"] = float(np.mean(np.abs(err[mask_i, i])))
            row["active_rmse"] = float(np.sqrt(np.mean(np.square(err[mask_i, i]))))
            row["active_mse"] = float(np.mean(np.square(err[mask_i, i])))
            row["active_mae_gated"] = float(np.mean(np.abs(err_gated[mask_i, i])))
            row["active_rmse_gated"] = float(np.sqrt(np.mean(np.square(err_gated[mask_i, i]))))
            row["active_mse_gated"] = float(np.mean(np.square(err_gated[mask_i, i])))
        else:
            row["active_count"] = 0
            row["active_mae"] = None
            row["active_rmse"] = None
            row["active_mse"] = None
            row["active_mae_gated"] = None
            row["active_rmse_gated"] = None
            row["active_mse_gated"] = None

        rows.append(
            row
        )

    rows_sorted = sorted(rows, key=lambda x: x["rmse"], reverse=True)
    rows_sorted_active = sorted(
        [r for r in rows if r.get("active_rmse_gated") is not None],
        key=lambda x: float(x["active_rmse_gated"]),
        reverse=True,
    )

    if active_param_mask is not None:
        denom_active = float(np.maximum(active_param_mask.sum(), 1.0))
        active_mse = float((np.square(err) * active_param_mask).sum() / denom_active)
        active_mse_gated = float((np.square(err_gated) * active_param_mask).sum() / denom_active)
        active_mae = float((np.abs(err) * active_param_mask).sum() / denom_active)
        active_mae_gated = float((np.abs(err_gated) * active_param_mask).sum() / denom_active)
        active_rmse = float(np.sqrt(active_mse))
        active_rmse_gated = float(np.sqrt(active_mse_gated))
    else:
        active_mae = None
        active_rmse = None
        active_mse = None
        active_mae_gated = None
        active_rmse_gated = None
        active_mse_gated = None

    summary = {
        "dataset_size": len(dataset),
        "val_size": len(val_ds),
        "val_split": val_split,
        "split_seed": split_seed,
        "overall_mae": float(np.mean(np.abs(err))),
        "overall_rmse": float(np.sqrt(np.mean(np.square(err)))),
        "overall_mse": float(np.mean(np.square(err))),
        "overall_mae_gated": float(np.mean(np.abs(err_gated))),
        "overall_rmse_gated": float(np.sqrt(np.mean(np.square(err_gated)))),
        "overall_mse_gated": float(np.mean(np.square(err_gated))),
        "active_only_mae": active_mae,
        "active_only_rmse": active_rmse,
        "active_only_mse": active_mse,
        "active_only_mae_gated": active_mae_gated,
        "active_only_rmse_gated": active_rmse_gated,
        "active_only_mse_gated": active_mse_gated,
        "selection_aligned_active_param_rmse": (
            active_rmse_gated if active_rmse_gated is not None else active_rmse
        ),
        "top5_highest_rmse": rows_sorted[:5],
        "top5_highest_active_rmse_gated": rows_sorted_active[:5],
    }

    return {
        "summary": summary,
        "rows": rows,
        "val_indices": list(val_ds.indices),
        "pred_arr": pred_arr,
        "pred_gated_arr": pred_gated_arr,
        "gt_arr": gt_arr,
        "effect_active_mask_arr": effect_mask_arr,
        "activity_probs_arr": np.concatenate(activity_probs_all, axis=0) if activity_probs_all else None,
        "activity_pred_arr": activity_pred_arr,
        "sample_mae": np.mean(np.abs(err), axis=1),
        "sample_rmse": np.sqrt(np.mean(np.square(err), axis=1)),
        "sample_mae_gated": np.mean(np.abs(err_gated), axis=1),
        "sample_rmse_gated": np.sqrt(np.mean(np.square(err_gated), axis=1)),
    }


def _save_metrics_csv(rows: List[Dict], path: Path):
    if not rows:
        return
    preferred = [
        "param",
        "mae", "rmse", "mse",
        "mae_gated", "rmse_gated", "mse_gated",
        "active_count",
        "active_mae", "active_rmse", "active_mse",
        "active_mae_gated", "active_rmse_gated", "active_mse_gated",
    ]
    row_keys = set()
    for row in rows:
        row_keys.update(row.keys())
    fieldnames = [k for k in preferred if k in row_keys]
    fieldnames.extend(sorted(k for k in row_keys if k not in fieldnames))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
    activity_probs_arr: Optional[np.ndarray] = None,
    activity_pred_arr: Optional[np.ndarray] = None,
    activity_thresholds: Optional[np.ndarray] = None,
    render_selection: str = "best_worst",
    render_seed: int = 42,
):
    render_dir = out_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    for old_file in render_dir.glob("*"):
        if old_file.is_file():
            old_file.unlink()
    renderer = PedalboardRenderer(sample_rate=sample_rate)

    if len(val_indices) == 0:
        return []
    n_pick = min(int(num_renders), len(val_indices))
    mode = str(render_selection).lower()
    if mode == "top_mae":
        order = np.argsort(sample_mae)[::-1]
        selected_local = order[:n_pick]
        stem_prefix = "val_rank"
        bucket_labels = ["worst"] * len(selected_local)
    elif mode == "best_worst":
        n_worst = min(max(1, n_pick // 2), len(val_indices))
        n_best = min(n_pick - n_worst, len(val_indices) - n_worst)
        order_desc = np.argsort(sample_mae)[::-1]
        order_asc = np.argsort(sample_mae)
        worst = list(order_desc[:n_worst])
        best = []
        for idx in order_asc:
            if idx in worst:
                continue
            best.append(idx)
            if len(best) >= n_best:
                break
        selected_local = np.array(worst + best, dtype=np.int64)
        stem_prefix = "val_extreme"
        bucket_labels = (["worst"] * len(worst)) + (["best"] * len(best))
    else:
        rng = np.random.default_rng(int(render_seed))
        selected_local = rng.choice(len(val_indices), size=n_pick, replace=False)
        stem_prefix = "val_rand"
        bucket_labels = ["random"] * len(selected_local)

    param_names = _build_param_names(effect_names)
    bucket_counts: Dict[str, int] = {}

    render_manifest = []
    for rank, local_idx in enumerate(selected_local, start=1):
        bucket = bucket_labels[rank - 1] if rank - 1 < len(bucket_labels) else "unknown"
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
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

        stem = f"{stem_prefix}_{rank:02d}_record_{record_index:04d}"
        in_path = render_dir / f"{stem}__input.wav"
        pred_path = render_dir / f"{stem}__pred.wav"
        target_path = render_dir / f"{stem}__target.wav"
        meta_path = render_dir / f"{stem}__meta.json"

        sf.write(str(in_path), waveform, sample_rate)
        sf.write(str(pred_path), pred_audio, sample_rate)
        sf.write(str(target_path), gt_audio, sample_rate)

        abs_diff = np.abs(pred_norm - gt_norm)
        top_idx = np.argsort(abs_diff)[-3:][::-1]
        param_error_top3 = [
            {
                "param": param_names[int(i)],
                "abs_error": float(abs_diff[int(i)]),
                "pred_norm": float(pred_norm[int(i)]),
                "target_norm": float(gt_norm[int(i)]),
            }
            for i in top_idx
        ]

        sample_info = {
            "rank_by_sample_mae": rank,
            "render_selection": str(render_selection),
            "selection_bucket": bucket,
            "rank_within_bucket": int(bucket_counts[bucket]),
            "record_index": record_index,
            "sample_mae": float(sample_mae[local_idx]),
            "sample_rmse": float(sample_rmse[local_idx]),
            "mean_abs_param_error": float(np.mean(abs_diff)),
            "param_error_top3": param_error_top3,
            "source_audio_path": str(source_path),
            "augmented_audio_path": augmented_audio_path,
            "active_effects": meta.get("active_effects"),
            "input_wav": str(in_path),
            "pred_wav": str(pred_path),
            "target_wav": str(target_path),
            "pred_params": pred_params,
            "target_params": gt_params,
            "predicted_activity_probs": (
                activity_probs_arr[local_idx].tolist() if activity_probs_arr is not None else None
            ),
            "predicted_activity_mask": (
                activity_pred_arr[local_idx].astype(bool).tolist() if activity_pred_arr is not None else None
            ),
            "activity_thresholds": (
                activity_thresholds.tolist() if activity_thresholds is not None else None
            ),
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
    num_renders: int = 4,
    sample_rate: int = 48000,
    max_duration: float = 20.0,
    device: str = "cpu",
    manifest_path: Optional[str] = None,
    hidden_dims: Optional[List[int]] = None,
    dropout: float = 0.1,
    render_selection: str = "best_worst",
    render_seed: int = 42,
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

    activity_thresholds = None
    thresholds_dict = None
    threshold_candidates = [
        artifacts / "controller" / "activity_thresholds.json",
        artifacts / "controller_activity_thresholds.json",
    ]
    for cand in threshold_candidates:
        if cand.exists():
            with cand.open("r") as f:
                payload = json.load(f)
            thresholds = payload.get("thresholds", {})
            if isinstance(thresholds, dict):
                thresholds_dict = thresholds
            break

    with pipeline_cfg_path.open("r") as f:
        pipeline_cfg = json.load(f)

    effect_names = list(pipeline_cfg["effect_names"])
    style_vocab_size = int(pipeline_cfg["aud_vocab_size"])
    if thresholds_dict is not None:
        activity_thresholds = np.array(
            [float(thresholds_dict.get(name, 0.5)) for name in effect_names],
            dtype=np.float32,
        )

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
        activity_thresholds=activity_thresholds,
    )
    _save_metrics_csv(eval_out["rows"], analysis_dir / "val_param_metrics.csv")
    with (analysis_dir / "val_metrics_summary.json").open("w") as f:
        json.dump(_to_serializable(eval_out["summary"]), f, indent=2)
    with (analysis_dir / "val_param_metrics.json").open("w") as f:
        json.dump(_to_serializable(eval_out["rows"]), f, indent=2)
    active_rows = []
    for row in eval_out["rows"]:
        if row.get("active_count", 0) <= 0:
            continue
        active_rows.append(
            {
                "param": row["param"],
                "active_count": int(row.get("active_count", 0)),
                "active_mae": row.get("active_mae"),
                "active_rmse": row.get("active_rmse"),
                "active_mae_gated": row.get("active_mae_gated"),
                "active_rmse_gated": row.get("active_rmse_gated"),
            }
        )
    active_rows = sorted(
        active_rows,
        key=lambda x: float(x["active_rmse_gated"]) if x.get("active_rmse_gated") is not None else -1.0,
        reverse=True,
    )
    with (analysis_dir / "active_only_param_metrics.json").open("w") as f:
        json.dump(_to_serializable(active_rows), f, indent=2)

    manifest_file = Path(manifest_path) if manifest_path else None
    manifest_map = _load_manifest_map(manifest_file)
    render_manifest = _render_ab_bundle(
        out_dir=analysis_dir,
        val_indices=eval_out["val_indices"],
        pred_arr=eval_out["pred_gated_arr"],
        gt_arr=eval_out["gt_arr"],
        sample_mae=eval_out["sample_mae_gated"],
        sample_rmse=eval_out["sample_rmse_gated"],
        effect_names=effect_names,
        sample_rate=sample_rate,
        max_duration=max_duration,
        manifest_map=manifest_map,
        num_renders=num_renders,
        activity_probs_arr=eval_out.get("activity_probs_arr"),
        activity_pred_arr=eval_out.get("activity_pred_arr"),
        activity_thresholds=activity_thresholds,
        render_selection=render_selection,
        render_seed=render_seed,
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
            "render_selection": str(render_selection),
            "render_seed": int(render_seed),
        },
    }
    with (analysis_dir / "analysis_report.json").open("w") as f:
        json.dump(_to_serializable(report), f, indent=2)

    return report
