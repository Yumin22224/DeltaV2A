#!/usr/bin/env python
"""
Analyze trained controller artifacts.

Usage:
  ./venv_DeltaV2A/bin/python scripts/analyze_controller_training.py \
      --artifacts-dir outputs/pipeline \
      --out-dir outputs/pipeline/controller/analysis \
      --num-renders 5
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.controller import run_controller_post_train_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze controller training artifacts.")
    parser.add_argument("--artifacts-dir", type=str, default="outputs/pipeline")
    parser.add_argument("--out-dir", type=str, default="outputs/pipeline/controller/analysis")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-renders", type=int, default=5)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--manifest-path", type=str, default="data/augmented/pipeline/audio/manifest.jsonl")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    report = run_controller_post_train_analysis(
        artifacts_dir=args.artifacts_dir,
        out_dir=args.out_dir,
        val_split=args.val_split,
        split_seed=args.split_seed,
        batch_size=args.batch_size,
        num_renders=args.num_renders,
        sample_rate=args.sample_rate,
        max_duration=args.max_duration,
        device=args.device,
        manifest_path=args.manifest_path,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    )
    print(f"[done] loss plot: {report['loss_curve_path']}")
    print(f"[done] val metrics: {report['val_metrics_csv']}")
    print(f"[done] renders: {Path(report['render_manifest_path']).parent}")
    print(f"[done] report: {Path(report['val_metrics_summary_json']).parent / 'analysis_report.json'}")


if __name__ == "__main__":
    main()
