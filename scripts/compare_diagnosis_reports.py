#!/usr/bin/env python
"""
Compare two style-diagnosis JSON reports and save a compact diff bundle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, Optional


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_triplet(b: Optional[float], n: Optional[float]) -> Dict[str, Optional[float]]:
    if b is None or n is None:
        return {"baseline": b, "new": n, "delta": None}
    return {"baseline": b, "new": n, "delta": n - b}


def _mean_f1(probe: Dict) -> Optional[float]:
    if not isinstance(probe, dict) or not probe:
        return None
    vals = [_to_float(v.get("f1")) for v in probe.values() if isinstance(v, dict)]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(mean(vals))


def _top_pair_cosine(effect_profiles: Dict) -> Optional[float]:
    pairs = effect_profiles.get("pairwise_profile_similarity", [])
    if not pairs:
        return None
    vals = [_to_float(p.get("cosine_sim")) for p in pairs if isinstance(p, dict)]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return max(vals)


def _active_only_rmse(report: Dict, key: str) -> Optional[float]:
    probe_params = report.get("probe_params", {})
    if isinstance(probe_params.get(key), dict):
        row = probe_params.get(key, {})
    else:
        row = probe_params.get("probe_results", {}).get(key, {})
    return _to_float(row.get("active_only_rmse"))


def _global_metrics(baseline: Dict, new: Dict) -> Dict[str, Dict[str, Optional[float]]]:
    b_dataset = baseline.get("dataset", {})
    n_dataset = new.get("dataset", {})
    b_entropy = baseline.get("entropy", {})
    n_entropy = new.get("entropy", {})
    b_consistency = baseline.get("consistency", {})
    n_consistency = new.get("consistency", {})

    out = {
        "records": _metric_triplet(_to_float(b_dataset.get("n_records")), _to_float(n_dataset.get("n_records"))),
        "entropy.mean": _metric_triplet(
            _to_float(b_entropy.get("mean_entropy")),
            _to_float(n_entropy.get("mean_entropy")),
        ),
        "entropy.effective_dims": _metric_triplet(
            _to_float(b_entropy.get("mean_effective_dims")),
            _to_float(n_entropy.get("mean_effective_dims")),
        ),
        "entropy.top1_mass": _metric_triplet(
            _to_float(b_entropy.get("mean_top1_mass")),
            _to_float(n_entropy.get("mean_top1_mass")),
        ),
        "consistency.separation": _metric_triplet(
            _to_float(b_consistency.get("separation")),
            _to_float(n_consistency.get("separation")),
        ),
        "effect.top_pair_cosine": _metric_triplet(
            _top_pair_cosine(baseline.get("effect_profiles", {})),
            _top_pair_cosine(new.get("effect_profiles", {})),
        ),
        "probe.mean_baseline.active_only_rmse": _metric_triplet(
            _active_only_rmse(baseline, "mean_baseline"),
            _active_only_rmse(new, "mean_baseline"),
        ),
        "probe.style_only.active_only_rmse": _metric_triplet(
            _active_only_rmse(baseline, "style_only"),
            _active_only_rmse(new, "style_only"),
        ),
        "probe.clap_only.active_only_rmse": _metric_triplet(
            _active_only_rmse(baseline, "clap_only"),
            _active_only_rmse(new, "clap_only"),
        ),
        "probe.clap+style.active_only_rmse": _metric_triplet(
            _active_only_rmse(baseline, "clap+style"),
            _active_only_rmse(new, "clap+style"),
        ),
        "probe.style_f1_mean": _metric_triplet(
            _mean_f1(baseline.get("probe_style_to_effect", {})),
            _mean_f1(new.get("probe_style_to_effect", {})),
        ),
        "probe.clap_style_f1_mean": _metric_triplet(
            _mean_f1(baseline.get("probe_clap_style_to_effect", {})),
            _mean_f1(new.get("probe_clap_style_to_effect", {})),
        ),
    }

    b_ctrl = baseline.get("controller_vs_baselines", {})
    n_ctrl = new.get("controller_vs_baselines", {})
    if not b_ctrl.get("skipped", False) and not n_ctrl.get("skipped", False):
        out["controller.active_only_rmse"] = _metric_triplet(
            _to_float(b_ctrl.get("controller_active_only_rmse")),
            _to_float(n_ctrl.get("controller_active_only_rmse")),
        )
        out["controller.overall_rmse"] = _metric_triplet(
            _to_float(b_ctrl.get("controller_overall_rmse")),
            _to_float(n_ctrl.get("controller_overall_rmse")),
        )
    return out


def _per_effect(baseline: Dict, new: Dict) -> Dict[str, Dict]:
    b_style = baseline.get("probe_style_to_effect", {})
    n_style = new.get("probe_style_to_effect", {})
    b_clap_style = baseline.get("probe_clap_style_to_effect", {})
    n_clap_style = new.get("probe_clap_style_to_effect", {})
    b_profiles = baseline.get("effect_profiles", {}).get("per_effect", {})
    n_profiles = new.get("effect_profiles", {}).get("per_effect", {})

    effect_names = sorted(
        set(b_style.keys())
        | set(n_style.keys())
        | set(b_clap_style.keys())
        | set(n_clap_style.keys())
        | set(b_profiles.keys())
        | set(n_profiles.keys())
    )

    out = {}
    for name in effect_names:
        out[name] = {
            "style_f1": _metric_triplet(
                _to_float(b_style.get(name, {}).get("f1")),
                _to_float(n_style.get(name, {}).get("f1")),
            ),
            "clap_style_f1": _metric_triplet(
                _to_float(b_clap_style.get(name, {}).get("f1")),
                _to_float(n_clap_style.get(name, {}).get("f1")),
            ),
            "profile_max_delta": _metric_triplet(
                _to_float(b_profiles.get(name, {}).get("max_delta")),
                _to_float(n_profiles.get(name, {}).get("max_delta")),
            ),
        }
    return out


def _flatten_global_rows(global_metrics: Dict[str, Dict[str, Optional[float]]]) -> Iterable[Dict[str, Optional[float]]]:
    for metric, row in global_metrics.items():
        yield {
            "section": "global",
            "name": metric,
            "baseline": row.get("baseline"),
            "new": row.get("new"),
            "delta": row.get("delta"),
        }


def _flatten_effect_rows(per_effect: Dict[str, Dict]) -> Iterable[Dict[str, Optional[float]]]:
    for effect, metrics in per_effect.items():
        for metric_name, row in metrics.items():
            yield {
                "section": f"effect:{effect}",
                "name": metric_name,
                "baseline": row.get("baseline"),
                "new": row.get("new"),
                "delta": row.get("delta"),
            }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two style diagnosis reports.")
    p.add_argument("--baseline-report", required=True, type=str)
    p.add_argument("--new-report", required=True, type=str)
    p.add_argument("--out-dir", required=True, type=str)
    p.add_argument("--comparison-id", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    baseline_path = Path(args.baseline_report).resolve()
    new_path = Path(args.new_report).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with baseline_path.open("r", encoding="utf-8") as f:
        baseline = json.load(f)
    with new_path.open("r", encoding="utf-8") as f:
        new = json.load(f)

    comparison_id = args.comparison_id or out_dir.name
    global_metrics = _global_metrics(baseline, new)
    per_effect = _per_effect(baseline, new)

    summary = {
        "comparison_id": comparison_id,
        "baseline": {
            "path": str(baseline_path),
            "sha256": _sha256(baseline_path),
        },
        "new": {
            "path": str(new_path),
            "sha256": _sha256(new_path),
        },
        "global_metrics": global_metrics,
        "per_effect": per_effect,
        "notes": {
            "baseline_controller_skipped": bool(baseline.get("controller_vs_baselines", {}).get("skipped", False)),
            "new_controller_skipped": bool(new.get("controller_vs_baselines", {}).get("skipped", False)),
        },
    }

    summary_path = out_dir / "comparison_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    table_path = out_dir / "comparison_table.csv"
    with table_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["section", "name", "baseline", "new", "delta"],
        )
        writer.writeheader()
        for row in _flatten_global_rows(global_metrics):
            writer.writerow(row)
        for row in _flatten_effect_rows(per_effect):
            writer.writerow(row)

    print(f"[done] summary: {summary_path}")
    print(f"[done] table:   {table_path}")


if __name__ == "__main__":
    main()
