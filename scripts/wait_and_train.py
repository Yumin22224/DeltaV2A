#!/usr/bin/env python
"""
Wait for attempt13 precompute to finish, then run training + finalize.

Usage:
    python scripts/wait_and_train.py [--attempt-dir ATTEMPT_DIR] [--device cuda]

Polls for pipeline_config.json (written at end of precompute) every 60s.
When precompute is done, runs:
  1. python scripts/run_pipeline.py train --config ... --device ...
  2. python scripts/finalize_training_attempt.py --attempt-dir ... --device ...
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wait for precompute then train.")
    p.add_argument(
        "--attempt-dir",
        type=str,
        default="outputs/attempts/attempt13",
        help="Attempt directory (relative to project root or absolute).",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--baseline-diagnosis",
        type=str,
        default=None,
        help="Path to baseline diagnosis_report.json for comparison.",
    )
    p.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds between precompute-done checks.",
    )
    return p.parse_args()


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (ROOT / p).resolve()


def run(cmd: list[str]) -> int:
    print(f"\n[wait_and_train] Running: {' '.join(cmd)}\n{'=' * 70}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    return result.returncode


def main():
    args = parse_args()
    attempt_dir = resolve(args.attempt_dir)
    if not attempt_dir.exists():
        print(f"[ERROR] Attempt dir not found: {attempt_dir}")
        sys.exit(1)

    # Locate config
    mlp_config = attempt_dir / "configs" / "mlp.yaml"
    if not mlp_config.exists():
        print(f"[ERROR] Config not found: {mlp_config}")
        sys.exit(1)

    # Load run_dir from config
    import yaml
    with open(mlp_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_dir_raw = cfg.get("output", {}).get("dir", "")
    run_dir = resolve(run_dir_raw)

    pipeline_config_path = run_dir / "pipeline_config.json"
    controller_best_path = run_dir / "controller" / "controller_best.pt"

    # ── Phase 1: Wait for precompute ─────────────────────────────────────────
    if pipeline_config_path.exists():
        print(f"[wait_and_train] pipeline_config.json already exists -- precompute done.")
    else:
        print(
            f"[wait_and_train] Waiting for precompute to finish...\n"
            f"  Watching: {pipeline_config_path}\n"
            f"  Poll interval: {args.poll_interval}s\n"
        )
        while not pipeline_config_path.exists():
            time.sleep(args.poll_interval)
            # Quick progress hint from checkpoint
            ckpt = run_dir / "inverse_mapping.checkpoint.json"
            if ckpt.exists():
                try:
                    import json
                    with open(ckpt) as f:
                        ck = json.load(f)
                    print(
                        f"  [{time.strftime('%H:%M:%S')}] "
                        f"Precompute progress: {ck.get('completed_audio_files', '?')} files, "
                        f"{ck.get('record_idx', '?')} records"
                    )
                except Exception:
                    print(f"  [{time.strftime('%H:%M:%S')}] still running...")
        print(f"[wait_and_train] Precompute DONE -- {pipeline_config_path}")

    # ── Phase 2: Train ────────────────────────────────────────────────────────
    if controller_best_path.exists():
        print(f"[wait_and_train] Controller checkpoint already exists — skipping train.")
    else:
        print(f"\n[wait_and_train] Starting MLP training...")
        rc = run([
            sys.executable,
            "scripts/run_pipeline.py",
            "train",
            "--config", str(mlp_config),
            "--device", args.device,
        ])
        if rc != 0:
            print(f"[ERROR] Training failed with exit code {rc}")
            sys.exit(rc)
        print(f"[wait_and_train] Training DONE.")

    # ── Phase 3: Finalize (style diagnosis + baseline comparison) ─────────────
    print(f"\n[wait_and_train] Starting finalize...")

    # Auto-detect baseline from attempt12 if not provided
    baseline = args.baseline_diagnosis
    if baseline is None:
        auto_baseline = (
            ROOT
            / "outputs"
            / "attempts"
            / "attempt12_20260225_085718"
            / "comparisons"
            / "style_diagnosis_current"
            / "diagnosis_report.json"
        )
        if auto_baseline.exists():
            baseline = str(auto_baseline)
            print(f"  Auto-detected baseline: {baseline}")

    finalize_cmd = [
        sys.executable,
        "scripts/finalize_training_attempt.py",
        "--attempt-dir", str(attempt_dir),
        "--device", args.device,
    ]
    if baseline:
        finalize_cmd += ["--baseline-diagnosis", baseline]

    rc = run(finalize_cmd)
    if rc != 0:
        print(f"[ERROR] Finalize failed with exit code {rc}")
        sys.exit(rc)

    print(f"\n[wait_and_train] ALL DONE for {attempt_dir.name}.")
    attempt_summary = attempt_dir / "attempt_summary.json"
    if attempt_summary.exists():
        import json
        with open(attempt_summary) as f:
            s = json.load(f)
        mlp_s = s.get("mlp", {})
        print(f"  best_epoch = {mlp_s.get('best_epoch')}")
        print(f"  best_val_loss = {mlp_s.get('best_val_loss')}")
        print(f"  active_param_rmse = {mlp_s.get('selection_aligned_active_param_rmse')}")
        print(f"  diagnosis_report = {s.get('diagnosis_report')}")
        if s.get("baseline_comparison_summary"):
            print(f"  comparison = {s.get('baseline_comparison_summary')}")


if __name__ == "__main__":
    main()
