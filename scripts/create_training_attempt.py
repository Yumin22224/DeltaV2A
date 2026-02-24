#!/usr/bin/env python
"""
Create a new training-attempt workspace with model-specific config snapshots.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import yaml


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump_yaml(path: Path, data: Dict):
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _prepare_config(cfg: Dict, attempt_id: str, run_dir: str) -> Dict:
    cfg = dict(cfg)
    cfg.setdefault("output", {})
    cfg["output"]["dir"] = run_dir

    cfg.setdefault("data", {})
    cfg["data"]["augmented_audio_dir"] = f"data/augmented/attempts/{attempt_id}/audio"
    cfg["data"]["save_augmented_audio"] = True
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create per-attempt training workspace.")
    p.add_argument("--attempt-id", type=str, default=None)
    p.add_argument("--root", type=str, default="outputs/attempts")
    p.add_argument("--mlp-config", type=str, default="configs/model_mlp.yaml")
    p.add_argument("--ar-config", type=str, default="configs/model_ar.yaml")
    p.add_argument("--baseline-diagnosis", type=str, default=None)
    p.add_argument("--change-note", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    now = datetime.now()
    attempt_id = args.attempt_id or now.strftime("attempt_%Y%m%d_%H%M%S")
    root = Path(args.root)
    attempt_dir = root / attempt_id

    if attempt_dir.exists():
        raise FileExistsError(f"Attempt directory already exists: {attempt_dir}")

    mlp_src = Path(args.mlp_config)
    ar_src = Path(args.ar_config)
    if not mlp_src.exists():
        raise FileNotFoundError(f"MLP config not found: {mlp_src}")
    if not ar_src.exists():
        raise FileNotFoundError(f"AR config not found: {ar_src}")

    run_dir = f"{attempt_dir.as_posix()}/run"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    (attempt_dir / "configs").mkdir(parents=True, exist_ok=True)
    (attempt_dir / "notes").mkdir(parents=True, exist_ok=True)
    (attempt_dir / "models" / "mlp").mkdir(parents=True, exist_ok=True)
    (attempt_dir / "models" / "ar").mkdir(parents=True, exist_ok=True)
    (attempt_dir / "comparisons").mkdir(parents=True, exist_ok=True)

    mlp_cfg = _prepare_config(_load_yaml(mlp_src), attempt_id=attempt_id, run_dir=run_dir)
    ar_cfg = _prepare_config(_load_yaml(ar_src), attempt_id=attempt_id, run_dir=run_dir)

    mlp_out = attempt_dir / "configs" / "mlp.yaml"
    ar_out = attempt_dir / "configs" / "ar.yaml"
    _dump_yaml(mlp_out, mlp_cfg)
    _dump_yaml(ar_out, ar_cfg)

    changes_path = attempt_dir / "notes" / "changes.md"
    note = args.change_note.strip() or "- TODO: describe this attempt's key changes."
    baseline_text = args.baseline_diagnosis or "null"
    changes_path.write_text(
        "\n".join(
            [
                f"# {attempt_id}",
                "",
                f"- created_at: {now.strftime('%Y-%m-%d %H:%M:%S')}",
                f"- baseline_diagnosis: {baseline_text}",
                "",
                "## Changes",
                note,
                "",
                "## Run Commands",
                f"- python scripts/run_pipeline.py precompute --config {mlp_out.as_posix()} --device cuda",
                f"- python scripts/run_pipeline.py train --config {mlp_out.as_posix()} --device cuda",
                f"- python scripts/run_pipeline.py train_ar --config {ar_out.as_posix()} --device cuda",
                f"- python scripts/finalize_training_attempt.py --attempt-dir {attempt_dir.as_posix()}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    metadata = {
        "attempt_id": attempt_id,
        "created_at": now.isoformat(),
        "paths": {
            "attempt_dir": str(attempt_dir),
            "run_dir": run_dir,
            "mlp_config": str(mlp_out),
            "ar_config": str(ar_out),
            "changes_note": str(changes_path),
        },
        "source_configs": {
            "mlp": str(mlp_src),
            "ar": str(ar_src),
        },
        "baseline_diagnosis": args.baseline_diagnosis,
        "change_note": args.change_note,
    }
    metadata_path = attempt_dir / "attempt_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[done] attempt directory: {attempt_dir}")
    print(f"[done] mlp config snapshot: {mlp_out}")
    print(f"[done] ar config snapshot:  {ar_out}")
    print(f"[done] change note file:    {changes_path}")
    print(f"[done] metadata:            {metadata_path}")


if __name__ == "__main__":
    main()
