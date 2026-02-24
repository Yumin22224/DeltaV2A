#!/usr/bin/env python
"""
Finalize a training attempt:
- run style diagnosis (with controller report)
- compare against baseline diagnosis (optional)
- collect best checkpoints/reports into attempt folder
"""

from __future__ import annotations

import argparse
import json
import csv
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# AR controller archived (attempt9+)


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _maybe_copy(src: Path, dst: Path, copied: List[str], missing: List[str]):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(str(dst))
    else:
        missing.append(str(src))


def _maybe_copytree(src: Path, dst: Path, copied: List[str], missing: List[str]):
    if src.exists() and src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
        copied.append(str(dst))
    else:
        missing.append(str(src))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Finalize a per-attempt training bundle.")
    p.add_argument("--attempt-dir", required=True, type=str)
    p.add_argument("--baseline-diagnosis", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _write_loss_metric_summary(
    path: Path,
    mlp_cfg: Dict,
    ar_cfg: Dict,
    mlp_report: Dict,
    ar_report: Dict,
):
    ctrl = mlp_cfg.get("controller", {})
    ar = ar_cfg.get("ar_controller", {})
    mlp_val = mlp_report.get("val_summary", {})
    ar_val = ar_report.get("val_summary", {})
    lines = [
        "# Model Loss / Metric Summary",
        "",
        "## MLP Controller",
        "",
        "### Training Loss",
        "- total_loss = param_loss_weight * param_loss + activity_loss_weight * activity_loss",
        f"- param_loss_type: `{ctrl.get('param_loss_type', 'mse')}` (huber_delta={ctrl.get('huber_delta', 'n/a')})",
        f"- activity_loss_type: `{ctrl.get('activity_loss_type', 'bce')}`",
        f"- activity_mismatch_weight: {ctrl.get('activity_mismatch_weight', 0.0)}",
        f"- activity_mismatch_gamma: {ctrl.get('activity_mismatch_gamma', 2.0)}",
        "",
        "### Model Selection Metric",
        f"- selection_metric: `{ctrl.get('selection_metric', 'val_param_loss')}`",
        "",
        "### Key Validation Metrics",
        f"- selection_aligned_active_param_rmse: {mlp_val.get('selection_aligned_active_param_rmse')}",
        f"- overall_rmse: {mlp_val.get('overall_rmse')}",
        f"- active_only_rmse: {mlp_val.get('active_only_rmse')}",
        f"- active_only_rmse_gated: {mlp_val.get('active_only_rmse_gated')}",
        "",
        "## AR Controller",
        "",
        "### Training Loss",
        "- total_loss = effect_loss_weight * CrossEntropy(effect) + param_loss_weight * Huber(param)",
        f"- effect_loss_weight: {ar.get('effect_loss_weight', 1.0)}",
        f"- param_loss_weight: {ar.get('param_loss_weight', 1.0)}",
        f"- huber_delta: {ar.get('huber_delta', 0.02)}",
        "",
        "### Current Logged Metric",
        "- best_val_loss (AR trainer objective, same combined loss scale)",
        f"- best_val_loss: {ar_report.get('curve_summary', {}).get('best_val_loss')}",
        f"- best_epoch: {ar_report.get('curve_summary', {}).get('best_epoch')}",
        "",
        "### Key Validation Metrics",
        f"- selection_aligned_active_param_rmse: {ar_val.get('selection_aligned_active_param_rmse')}",
        f"- overall_rmse: {ar_val.get('overall_rmse')}",
        f"- active_only_rmse: {ar_val.get('active_only_rmse')}",
        "",
        "### Notes",
        "- AR checkpoint selection is still based on combined training objective (best_val_loss).",
        "- AR now also reports active/overall param RMSE on reconstructed validation split for side-by-side comparison.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _safe_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _write_intra_attempt_model_comparison(
    attempt_dir: Path,
    mlp_report: Dict,
    ar_report: Dict,
) -> Dict[str, str]:
    out_dir = attempt_dir / "comparisons" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "model_comparison_summary.json"
    csv_path = out_dir / "model_comparison_table.csv"

    mlp_curve = mlp_report.get("curve_summary", {})
    ar_curve = ar_report.get("curve_summary", {})
    mlp_val = mlp_report.get("val_summary", {})
    ar_val = ar_report.get("val_summary", {})

    metric_rows: List[Dict[str, object]] = []

    def _add_metric(name: str, mlp_value, ar_value, comparable: bool = True):
        mlp_f = _safe_float(mlp_value)
        ar_f = _safe_float(ar_value)
        delta = None
        better = "n/a"
        if mlp_f is not None and ar_f is not None:
            delta = ar_f - mlp_f
            if comparable:
                better = "mlp" if mlp_f < ar_f else ("ar" if ar_f < mlp_f else "tie")
        metric_rows.append(
            {
                "metric": name,
                "mlp": mlp_f,
                "ar": ar_f,
                "delta_ar_minus_mlp": delta,
                "better": better,
                "comparable": bool(comparable),
            }
        )

    _add_metric(
        "active_only_rmse",
        mlp_val.get("active_only_rmse"),
        ar_val.get("active_only_rmse"),
        comparable=True,
    )
    _add_metric(
        "overall_rmse",
        mlp_val.get("overall_rmse"),
        ar_val.get("overall_rmse"),
        comparable=True,
    )
    _add_metric(
        "selection_aligned_active_param_rmse",
        mlp_val.get("selection_aligned_active_param_rmse"),
        ar_val.get("selection_aligned_active_param_rmse"),
        comparable=True,
    )
    _add_metric(
        "best_val_loss_raw_training_objective",
        mlp_curve.get("best_val_loss"),
        ar_curve.get("best_val_loss"),
        comparable=False,
    )

    summary = {
        "metrics": metric_rows,
        "notes": [
            "Lower is better for RMSE metrics.",
            (
                "selection_aligned_active_param_rmse may reflect different gating behaviors: "
                "MLP uses activity-threshold gating, AR currently has no separate activity-head gating path."
            ),
            "best_val_loss_raw_training_objective is not directly comparable across MLP/AR due to different loss definitions.",
        ],
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["metric", "mlp", "ar", "delta_ar_minus_mlp", "better", "comparable"],
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    return {
        "summary_json": str(json_path),
        "table_csv": str(csv_path),
    }


def main():
    args = parse_args()
    attempt_dir = Path(args.attempt_dir).resolve()
    if not attempt_dir.exists():
        raise FileNotFoundError(f"Attempt directory not found: {attempt_dir}")

    metadata_path = attempt_dir / "attempt_metadata.json"
    metadata = _load_json(metadata_path) if metadata_path.exists() else {}

    mlp_cfg_path = attempt_dir / "configs" / "mlp.yaml"
    ar_cfg_path = attempt_dir / "configs" / "ar.yaml"
    if not mlp_cfg_path.exists() or not ar_cfg_path.exists():
        raise FileNotFoundError("Expected configs/mlp.yaml and configs/ar.yaml inside attempt dir.")

    mlp_cfg = _load_yaml(mlp_cfg_path)
    ar_cfg = _load_yaml(ar_cfg_path)
    run_dir = Path(mlp_cfg["output"]["dir"]).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    copied: List[str] = []
    missing: List[str] = []

    mlp_model_dir = attempt_dir / "models" / "mlp"
    mlp_model_dir.mkdir(parents=True, exist_ok=True)

    def _pick(cfg: Dict, key: str, default):
        v = cfg.get(key, None)
        return default if v is None else v

    # MLP artifact collection
    _maybe_copy(
        run_dir / "controller" / "controller_best.pt",
        mlp_model_dir / "best" / "controller_best.pt",
        copied,
        missing,
    )
    _maybe_copy(
        run_dir / "controller" / "controller_best_activity.pt",
        mlp_model_dir / "best" / "controller_best_activity.pt",
        copied,
        missing,
    )
    _maybe_copy(
        run_dir / "controller" / "training_log.json",
        mlp_model_dir / "training_log.json",
        copied,
        missing,
    )
    _maybe_copy(
        run_dir / "controller" / "analysis" / "analysis_report.json",
        mlp_model_dir / "analysis" / "analysis_report.json",
        copied,
        missing,
    )
    _maybe_copy(
        run_dir / "controller" / "analysis" / "val_metrics_summary.json",
        mlp_model_dir / "analysis" / "val_metrics_summary.json",
        copied,
        missing,
    )
    _maybe_copy(
        run_dir / "controller" / "analysis" / "active_only_param_metrics.json",
        mlp_model_dir / "analysis" / "active_only_param_metrics.json",
        copied,
        missing,
    )
    _maybe_copytree(
        run_dir / "controller" / "analysis" / "renders",
        mlp_model_dir / "analysis" / "renders",
        copied,
        missing,
    )

    # Style diagnosis (current run)
    diagnosis_dir = attempt_dir / "comparisons" / "style_diagnosis_current"
    diagnosis_cmd = [
        sys.executable,
        "scripts/diagnose_style_labels.py",
        "--h5",
        str(run_dir / "inverse_mapping.h5"),
        "--aud-vocab",
        str(run_dir / "aud_vocab.npz"),
        "--controller-report",
        str(run_dir / "controller" / "analysis" / "analysis_report.json"),
        "--out-dir",
        str(diagnosis_dir),
    ]
    subprocess.run(diagnosis_cmd, check=True, cwd=str(ROOT))

    baseline = args.baseline_diagnosis or metadata.get("baseline_diagnosis")
    comparison_summary_path: Optional[Path] = None
    if baseline:
        compare_dir = attempt_dir / "comparisons" / "vs_baseline"
        compare_cmd = [
            sys.executable,
            "scripts/compare_diagnosis_reports.py",
            "--baseline-report",
            str(Path(baseline).resolve()),
            "--new-report",
            str(diagnosis_dir / "diagnosis_report.json"),
            "--out-dir",
            str(compare_dir),
            "--comparison-id",
            attempt_dir.name,
        ]
        subprocess.run(compare_cmd, check=True, cwd=str(ROOT))
        comparison_summary_path = compare_dir / "comparison_summary.json"

    mlp_report_path = run_dir / "controller" / "analysis" / "analysis_report.json"
    mlp_report = _load_json(mlp_report_path) if mlp_report_path.exists() else {}
    loss_metric_summary_path = attempt_dir / "notes" / "model_loss_metric_summary.md"
    _write_loss_metric_summary(
        path=loss_metric_summary_path,
        mlp_cfg=mlp_cfg,
        ar_cfg=ar_cfg,
        mlp_report=mlp_report,
        ar_report=ar_analysis_report,
    )
    intra_model_cmp = _write_intra_attempt_model_comparison(
        attempt_dir=attempt_dir,
        mlp_report=mlp_report,
        ar_report=ar_analysis_report,
    )

    summary = {
        "attempt_id": metadata.get("attempt_id", attempt_dir.name),
        "attempt_dir": str(attempt_dir),
        "run_dir": str(run_dir),
        "mlp": {
            "best_checkpoint": str(run_dir / "controller" / "controller_best.pt"),
            "analysis_report": str(mlp_report_path),
            "best_epoch": mlp_report.get("curve_summary", {}).get("best_epoch"),
            "best_val_loss": mlp_report.get("curve_summary", {}).get("best_val_loss"),
            "selection_aligned_active_param_rmse": mlp_report.get("val_summary", {}).get(
                "selection_aligned_active_param_rmse"
            ),
        },
        "ar": {
            "best_checkpoint": str(run_dir / "ar_controller" / "ar_controller_best.pt"),
            "analysis_report": str(run_dir / "ar_controller" / "analysis" / "analysis_report.json"),
            "best_epoch": ar_analysis_report.get("curve_summary", {}).get("best_epoch"),
            "best_val_loss": ar_analysis_report.get("curve_summary", {}).get("best_val_loss"),
        },
        "diagnosis_report": str(diagnosis_dir / "diagnosis_report.json"),
        "baseline_comparison_summary": str(comparison_summary_path) if comparison_summary_path else None,
        "model_comparison_summary": intra_model_cmp["summary_json"],
        "model_comparison_table": intra_model_cmp["table_csv"],
        "loss_metric_summary": str(loss_metric_summary_path),
        "copied_artifacts": copied,
        "missing_artifacts": missing,
    }

    summary_path = attempt_dir / "attempt_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] attempt summary: {summary_path}")
    print(f"[done] diagnosis report: {diagnosis_dir / 'diagnosis_report.json'}")
    if comparison_summary_path:
        print(f"[done] baseline comparison: {comparison_summary_path}")


if __name__ == "__main__":
    main()
