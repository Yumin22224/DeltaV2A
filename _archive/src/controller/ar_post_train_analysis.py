"""
Post-training analysis for ARController (Phase B-AR).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import random_split

from .ar_model import ARController
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


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _load_manifest_map(manifest_path: Optional[Path]) -> Dict[int, Dict]:
    if manifest_path is None or not manifest_path.exists():
        return {}
    out: Dict[int, Dict] = {}
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[int(row["record_index"])] = row
    return out


def _plot_loss_curves(
    epochs,
    train_total,
    val_total,
    train_effect,
    val_effect,
    train_param,
    val_param,
    out_path: Path,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, train_total, label="train_total")
    axes[0].plot(epochs, val_total, label="val_total")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_effect, label="train_effect")
    axes[1].plot(epochs, val_effect, label="val_effect")
    axes[1].set_title("Effect Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs, train_param, label="train_param")
    axes[2].plot(epochs, val_param, label="val_param")
    axes[2].set_title("Param Loss")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _effect_param_slices(effect_names: List[str]) -> List[slice]:
    slices: List[slice] = []
    start = 0
    for effect_name in effect_names:
        width = EFFECT_CATALOG[effect_name].num_params
        slices.append(slice(start, start + width))
        start += width
    return slices


def _build_param_names(effect_names: List[str]) -> List[str]:
    names: List[str] = []
    for effect_name in effect_names:
        spec = EFFECT_CATALOG[effect_name]
        for ps in spec.params:
            names.append(f"{effect_name}.{ps.name}")
    return names


def _evaluate_ar_controller(
    db_path: Path,
    checkpoint_path: Path,
    effect_names: List[str],
    val_split: float,
    split_seed: int,
    device: str,
) -> Dict:
    db = InverseMappingDB(str(db_path))
    dataset = InverseMappingDataset(db)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    _, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(split_seed),
    )
    val_indices = list(val_ds.indices)
    if not val_indices:
        return {
            "summary": {
                "dataset_size": len(dataset),
                "val_size": 0,
                "val_split": val_split,
                "split_seed": split_seed,
            },
            "rows": [],
        }

    model = _load_ar_model(checkpoint_path, device=device)
    effect_slices = _effect_param_slices(effect_names)
    pred_out = _predict_ar_for_val(
        dataset=dataset,
        val_indices=val_indices,
        model=model,
        effect_names=effect_names,
        effect_slices=effect_slices,
        device=device,
    )
    pred_arr = pred_out["pred_arr"]
    gt_arr = pred_out["gt_arr"]
    err = pred_arr - gt_arr

    effect_mask_arr = dataset.effect_active_mask[np.asarray(val_indices, dtype=np.int64)]
    chunks: List[np.ndarray] = []
    for i, sl in enumerate(effect_slices):
        width = sl.stop - sl.start
        chunks.append(np.repeat(effect_mask_arr[:, i:i + 1], width, axis=1))
    active_param_mask = np.concatenate(chunks, axis=1).astype(np.float32)
    active_counts = active_param_mask.sum(axis=0)

    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(np.square(err), axis=0))
    mse = np.mean(np.square(err), axis=0)

    param_names = _build_param_names(effect_names)
    rows: List[Dict] = []
    for i, name in enumerate(param_names):
        row = {
            "param": name,
            "mae": float(mae[i]),
            "rmse": float(rmse[i]),
            "mse": float(mse[i]),
        }
        if active_counts[i] > 0:
            mask_i = active_param_mask[:, i] > 0.5
            row["active_count"] = int(active_counts[i])
            row["active_mae"] = float(np.mean(np.abs(err[mask_i, i])))
            row["active_rmse"] = float(np.sqrt(np.mean(np.square(err[mask_i, i]))))
            row["active_mse"] = float(np.mean(np.square(err[mask_i, i])))
        else:
            row["active_count"] = 0
            row["active_mae"] = None
            row["active_rmse"] = None
            row["active_mse"] = None
        rows.append(row)

    rows_sorted = sorted(rows, key=lambda x: float(x["rmse"]), reverse=True)
    rows_sorted_active = sorted(
        [r for r in rows if r.get("active_rmse") is not None],
        key=lambda x: float(x["active_rmse"]),
        reverse=True,
    )

    denom_active = float(np.maximum(active_param_mask.sum(), 1.0))
    active_mse = float((np.square(err) * active_param_mask).sum() / denom_active)
    active_mae = float((np.abs(err) * active_param_mask).sum() / denom_active)
    active_rmse = float(np.sqrt(active_mse))

    summary = {
        "dataset_size": len(dataset),
        "val_size": len(val_indices),
        "val_split": val_split,
        "split_seed": split_seed,
        "overall_mae": float(np.mean(np.abs(err))),
        "overall_rmse": float(np.sqrt(np.mean(np.square(err)))),
        "overall_mse": float(np.mean(np.square(err))),
        "active_only_mae": active_mae,
        "active_only_rmse": active_rmse,
        "active_only_mse": active_mse,
        # For AR we don't have an activity-gated variant in this report.
        "selection_aligned_active_param_rmse": active_rmse,
        "top5_highest_rmse": rows_sorted[:5],
        "top5_highest_active_rmse": rows_sorted_active[:5],
    }
    return {
        "summary": summary,
        "rows": rows,
    }


def _build_pred_norm_from_chain(
    chain: List[Tuple[str, Dict[str, float]]],
    effect_names: List[str],
    effect_slices: List[slice],
) -> np.ndarray:
    pred_norm = normalize_params({}, effect_names)
    effect_to_idx = {name: i for i, name in enumerate(effect_names)}
    for effect_name, params_norm in chain:
        idx = effect_to_idx.get(effect_name)
        if idx is None:
            continue
        spec = EFFECT_CATALOG[effect_name]
        sl = effect_slices[idx]
        vec = pred_norm[sl].copy()
        for j, ps in enumerate(spec.params):
            if ps.name in params_norm:
                vec[j] = float(np.clip(params_norm[ps.name], 0.0, 1.0))
        pred_norm[sl] = vec
    return pred_norm.astype(np.float32)


def _load_ar_model(ckpt_path: Path, device: str) -> ARController:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = ckpt.get("model_config", {})
    model = ARController(
        effect_names=list(cfg["effect_names"]),
        style_vocab_size=int(cfg.get("style_vocab_size", 24)),
        condition_dim=int(cfg.get("condition_dim", 128)),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        dropout=float(cfg.get("dropout", 0.1)),
        max_steps=int(cfg.get("max_steps", 2)),
        clap_embed_dim=int(cfg.get("clap_embed_dim", 0)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _select_local_indices(
    sample_mae: np.ndarray,
    num_renders: int,
    render_selection: str,
    render_seed: int,
) -> Tuple[np.ndarray, str, List[str]]:
    n = int(sample_mae.shape[0])
    if n <= 0:
        return np.array([], dtype=np.int64), "none"
    n_pick = min(int(num_renders), n)
    mode = str(render_selection).lower()
    if mode == "top_mae":
        order = np.argsort(sample_mae)[::-1]
        picked = order[:n_pick]
        return picked, "val_rank", ["worst"] * len(picked)
    if mode == "best_worst":
        n_worst = min(max(1, n_pick // 2), n)
        n_best = min(n_pick - n_worst, n - n_worst)
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
        picked = np.array(worst + best, dtype=np.int64)
        labels = (["worst"] * len(worst)) + (["best"] * len(best))
        return picked, "val_extreme", labels
    rng = np.random.default_rng(int(render_seed))
    picked = rng.choice(n, size=n_pick, replace=False)
    return picked, "val_rand", ["random"] * len(picked)


def _predict_ar_for_val(
    dataset: InverseMappingDataset,
    val_indices: List[int],
    model: ARController,
    effect_names: List[str],
    effect_slices: List[slice],
    device: str,
) -> Dict[str, np.ndarray]:
    n = len(val_indices)
    total_params = len(normalize_params({}, effect_names))
    pred_arr = np.zeros((n, total_params), dtype=np.float32)
    gt_arr = np.zeros((n, total_params), dtype=np.float32)
    sample_mae = np.zeros((n,), dtype=np.float32)
    sample_rmse = np.zeros((n,), dtype=np.float32)

    for local_idx, record_index in enumerate(val_indices):
        style_np = dataset.style_labels[record_index].astype(np.float32)
        clap_np = dataset.clap_embeddings[record_index].astype(np.float32)
        gt_norm = dataset.normalized_params[record_index].astype(np.float32)

        style = torch.from_numpy(style_np).to(device)
        clap = torch.from_numpy(clap_np).to(device)
        if model.clap_embed_dim > 0:
            chain = model.infer(style, clap_embedding=clap)
        else:
            chain = model.infer(style, clap_embedding=None)

        pred_norm = _build_pred_norm_from_chain(chain, effect_names, effect_slices)
        pred_arr[local_idx] = pred_norm
        gt_arr[local_idx] = gt_norm
        diff = pred_norm - gt_norm
        sample_mae[local_idx] = float(np.mean(np.abs(diff)))
        sample_rmse[local_idx] = float(np.sqrt(np.mean(np.square(diff))))

    return {
        "pred_arr": pred_arr,
        "gt_arr": gt_arr,
        "sample_mae": sample_mae,
        "sample_rmse": sample_rmse,
    }


def _render_ab_bundle(
    out_dir: Path,
    db_path: Path,
    ckpt_path: Path,
    effect_names: List[str],
    val_split: float,
    split_seed: int,
    sample_rate: int,
    max_duration: float,
    manifest_map: Dict[int, Dict],
    num_renders: int,
    render_seed: int,
    render_selection: str,
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
    val_indices = list(val_ds.indices)
    if not val_indices:
        return []

    renderer = PedalboardRenderer(sample_rate=sample_rate)
    render_dir = out_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    for old_file in render_dir.glob("*"):
        if old_file.is_file():
            old_file.unlink()

    model = _load_ar_model(ckpt_path, device=device)
    effect_slices = _effect_param_slices(effect_names)
    pred_out = _predict_ar_for_val(
        dataset=dataset,
        val_indices=val_indices,
        model=model,
        effect_names=effect_names,
        effect_slices=effect_slices,
        device=device,
    )
    picked_local, stem_prefix, bucket_labels = _select_local_indices(
        sample_mae=pred_out["sample_mae"],
        num_renders=num_renders,
        render_selection=render_selection,
        render_seed=render_seed,
    )
    param_names = []
    for effect_name in effect_names:
        spec = EFFECT_CATALOG[effect_name]
        for ps in spec.params:
            param_names.append(f"{effect_name}.{ps.name}")
    bucket_counts: Dict[str, int] = {}

    render_manifest = []
    for rank, local_idx in enumerate(picked_local, start=1):
        bucket = bucket_labels[rank - 1] if rank - 1 < len(bucket_labels) else "unknown"
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        record_index = int(val_indices[local_idx])
        meta = manifest_map.get(record_index, {})
        source_audio_path = meta.get("source_audio_path")
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

        style_np = dataset.style_labels[record_index].astype(np.float32)
        clap_np = dataset.clap_embeddings[record_index].astype(np.float32)
        gt_norm = pred_out["gt_arr"][local_idx]
        pred_norm = pred_out["pred_arr"][local_idx]

        style = torch.from_numpy(style_np).to(device)
        clap = torch.from_numpy(clap_np).to(device)
        if model.clap_embed_dim > 0:
            chain = model.infer(style, clap_embedding=clap)
        else:
            chain = model.infer(style, clap_embedding=None)

        pred_params = denormalize_params(pred_norm, effect_names)
        gt_params = denormalize_params(gt_norm, effect_names)

        pred_audio = renderer.render(waveform, pred_params)
        gt_audio = renderer.render(waveform, gt_params)

        sample_mae = float(pred_out["sample_mae"][local_idx])
        sample_rmse = float(pred_out["sample_rmse"][local_idx])

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
            "render_selection": str(render_selection),
            "selection_bucket": bucket,
            "rank_within_bucket": int(bucket_counts[bucket]),
            "record_index": record_index,
            "sample_mae": sample_mae,
            "sample_rmse": sample_rmse,
            "mean_abs_param_error": float(np.mean(abs_diff)),
            "param_error_top3": param_error_top3,
            "source_audio_path": str(source_path),
            "augmented_audio_path": meta.get("augmented_audio_path"),
            "active_effects": meta.get("active_effects"),
            "target_effect_order": meta.get("effect_order"),
            "pred_effect_order": [effect_names.index(eff) for eff, _ in chain if eff in effect_names],
            "pred_chain_normalized": [
                {"effect": eff, "params": params}
                for eff, params in chain
            ],
            "input_wav": str(in_path),
            "pred_wav": str(pred_path),
            "target_wav": str(target_path),
            "pred_params": pred_params,
            "target_params": gt_params,
        }
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(_to_serializable(sample_info), f, indent=2)
        render_manifest.append(sample_info)

    with (render_dir / "render_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(render_manifest), f, indent=2)
    return render_manifest


def run_ar_post_train_analysis(
    artifacts_dir: str,
    out_dir: Optional[str] = None,
    val_split: float = 0.2,
    split_seed: int = 42,
    sample_rate: int = 48000,
    max_duration: float = 20.0,
    device: str = "cpu",
    manifest_path: Optional[str] = None,
    num_renders: int = 4,
    render_seed: int = 42,
    render_selection: str = "best_worst",
) -> Dict:
    """
    Analyze AR training artifacts and save report + A/B render bundle.
    """
    artifacts = Path(artifacts_dir)
    ar_dir = artifacts / "ar_controller"
    log_path = ar_dir / "ar_training_log.json"
    if not log_path.exists():
        raise FileNotFoundError(f"AR training log not found: {log_path}")

    with log_path.open("r", encoding="utf-8") as f:
        history = json.load(f)
    if not history:
        raise RuntimeError(f"Empty AR training log: {log_path}")

    analysis_dir = Path(out_dir) if out_dir else (ar_dir / "analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    epochs = [int(row.get("epoch", i + 1)) for i, row in enumerate(history)]
    train_total = [_safe_float(row.get("train_total_loss")) for row in history]
    val_total = [_safe_float(row.get("val_total_loss")) for row in history]
    train_effect = [_safe_float(row.get("train_effect_loss")) for row in history]
    val_effect = [_safe_float(row.get("val_effect_loss")) for row in history]
    train_param = [_safe_float(row.get("train_param_loss")) for row in history]
    val_param = [_safe_float(row.get("val_param_loss")) for row in history]

    best_idx = min(range(len(history)), key=lambda i: val_total[i])
    best_epoch = epochs[best_idx]
    best_val_loss = val_total[best_idx]

    csv_path = analysis_dir / "ar_loss_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "lr",
                "train_total_loss",
                "train_effect_loss",
                "train_param_loss",
                "val_total_loss",
                "val_effect_loss",
                "val_param_loss",
            ]
        )
        for row in history:
            writer.writerow(
                [
                    int(row.get("epoch", 0)),
                    _safe_float(row.get("lr")),
                    _safe_float(row.get("train_total_loss")),
                    _safe_float(row.get("train_effect_loss")),
                    _safe_float(row.get("train_param_loss")),
                    _safe_float(row.get("val_total_loss")),
                    _safe_float(row.get("val_effect_loss")),
                    _safe_float(row.get("val_param_loss")),
                ]
            )

    plot_path = analysis_dir / "ar_loss_curve.png"
    _plot_loss_curves(
        epochs=epochs,
        train_total=train_total,
        val_total=val_total,
        train_effect=train_effect,
        val_effect=val_effect,
        train_param=train_param,
        val_param=val_param,
        out_path=plot_path,
    )

    pipeline_cfg_path = artifacts / "pipeline_config.json"
    if not pipeline_cfg_path.exists():
        raise FileNotFoundError(f"Missing pipeline config: {pipeline_cfg_path}")
    with pipeline_cfg_path.open("r", encoding="utf-8") as f:
        pipeline_cfg = json.load(f)
    effect_names = list(pipeline_cfg["effect_names"])

    db_path = artifacts / "inverse_mapping.h5"
    ckpt_path = ar_dir / "ar_controller_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing AR checkpoint: {ckpt_path}")
    if not db_path.exists():
        raise FileNotFoundError(f"Missing DB: {db_path}")

    eval_out = _evaluate_ar_controller(
        db_path=db_path,
        checkpoint_path=ckpt_path,
        effect_names=effect_names,
        val_split=val_split,
        split_seed=split_seed,
        device=device,
    )
    val_summary_json = analysis_dir / "val_metrics_summary.json"
    with val_summary_json.open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(eval_out["summary"]), f, indent=2)
    active_only_metrics_json = analysis_dir / "active_only_param_metrics.json"
    with active_only_metrics_json.open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(eval_out["rows"]), f, indent=2)

    manifest_file = Path(manifest_path) if manifest_path else None
    manifest_map = _load_manifest_map(manifest_file)
    render_manifest = _render_ab_bundle(
        out_dir=analysis_dir,
        db_path=db_path,
        ckpt_path=ckpt_path,
        effect_names=effect_names,
        val_split=val_split,
        split_seed=split_seed,
        sample_rate=sample_rate,
        max_duration=max_duration,
        manifest_map=manifest_map,
        num_renders=num_renders,
        render_seed=render_seed,
        render_selection=render_selection,
        device=device,
    )

    report = {
        "curve_summary": {
            "epochs": len(history),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "first_train_loss": float(train_total[0]),
            "last_train_loss": float(train_total[-1]),
            "first_val_loss": float(val_total[0]),
            "last_val_loss": float(val_total[-1]),
        },
        "loss_curve_path": str(plot_path),
        "loss_metrics_csv": str(csv_path),
        "val_metrics_summary_json": str(val_summary_json),
        "active_only_param_metrics_json": str(active_only_metrics_json),
        "val_summary": eval_out["summary"],
        "training_log_path": str(log_path),
        "best_checkpoint": str(ar_dir / "ar_controller_best.pt"),
        "last_checkpoint": str(ar_dir / "ar_controller_last.pt"),
        "render_manifest_path": str(analysis_dir / "renders" / "render_manifest.json"),
        "num_rendered_examples": len(render_manifest),
        "notes": {
            "val_split_reconstruction": "random_split with seed=42 (same as trainer default).",
            "render_selection": str(render_selection),
            "render_seed": int(render_seed),
        },
    }

    report_path = analysis_dir / "analysis_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(report), f, indent=2)

    return report
