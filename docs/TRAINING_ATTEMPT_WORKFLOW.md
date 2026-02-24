# Training Attempt Workflow

This project now supports per-attempt experiment bundles so MLP/AR iterations do not get mixed.

## 1) Create a new attempt folder

```bash
python scripts/create_training_attempt.py \
  --attempt-id attempt_YYYYMMDD_HHMMSS \
  --mlp-config configs/model_mlp.yaml \
  --ar-config configs/model_ar.yaml \
  --baseline-diagnosis <path-to-baseline-diagnosis-report.json> \
  --change-note "- short summary of what changed"
```

This creates:

- `outputs/attempts/<attempt_id>/configs/mlp.yaml`
- `outputs/attempts/<attempt_id>/configs/ar.yaml`
- `outputs/attempts/<attempt_id>/notes/changes.md`
- `outputs/attempts/<attempt_id>/attempt_metadata.json`

Both config snapshots point to the same run directory:

- `outputs/attempts/<attempt_id>/run`

so precompute DB is shared across MLP/AR training.

## 2) Run training

```bash
python scripts/run_pipeline.py precompute --config outputs/attempts/<attempt_id>/configs/mlp.yaml --device cuda
python scripts/run_pipeline.py train --config outputs/attempts/<attempt_id>/configs/mlp.yaml --device cuda
python scripts/run_pipeline.py train_ar --config outputs/attempts/<attempt_id>/configs/ar.yaml --device cuda
```

Notes:

- MLP training generates `controller/analysis/analysis_report.json`.
- AR training now also generates `ar_controller/analysis/analysis_report.json`.
- Both analyses save `input/pred/target` audio bundles with `best_worst` selection by default:
  - worst 2 validation samples + best 2 validation samples.

## 3) Finalize and compare

```bash
python scripts/finalize_training_attempt.py \
  --attempt-dir outputs/attempts/<attempt_id>
```

This will:

- regenerate AR post-train analysis (safe idempotent),
- run style diagnosis with MLP controller report,
- compare against baseline diagnosis (if provided),
- copy key artifacts (best pt, logs, analysis) under:
  - `outputs/attempts/<attempt_id>/models/mlp`
  - `outputs/attempts/<attempt_id>/models/ar`
- write `outputs/attempts/<attempt_id>/attempt_summary.json`.

## Directory purpose

- `outputs/pipeline`: legacy/default output path (single rolling workspace).
- `outputs/runs`: manual isolated run folders (legacy transition phase).
- `outputs/comparisons`: manual comparison bundles (legacy transition phase).
- `outputs/attempts`: recommended per-training-attempt bundles (new standard).
