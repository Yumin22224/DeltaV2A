#!/usr/bin/env python
"""
Style Label Informativeness Diagnostic

Answers the core question: do style labels carry enough information
about the applied effects/parameters for the Controller to learn?

Tests:
  1. Entropy analysis    - How peaky are style labels? (T=0.1 concern)
  2. Effect→Style map    - Do different effects produce distinct style profiles?
  3. Style→Effect probe  - Can a simple linear model predict active effects from style labels?
  4. Style→Param probe   - Can a simple linear model predict parameters from style labels?
  5. Consistency check   - Same effect on different audio → similar style labels?
  6. Baseline comparison - Is the trained Controller better than trivial baselines?

Usage:
  python scripts/diagnose_style_labels.py
  python scripts/diagnose_style_labels.py --h5 outputs/pipeline/inverse_mapping.h5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose style label informativeness.")
    p.add_argument("--h5", type=str, default="outputs/pipeline/inverse_mapping.h5")
    p.add_argument("--aud-vocab", type=str, default="outputs/pipeline/aud_vocab.npz")
    p.add_argument("--out-dir", type=str, default="outputs/pipeline/controller/analysis/style_diagnosis")
    p.add_argument(
        "--controller-report",
        type=str,
        default=None,
        help=(
            "Path to controller analysis_report.json for baseline comparison. "
            "If omitted, controller-vs-baselines section is skipped."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Data loading ─────────────────────────────────────────────────────────────

def load_db(h5_path: str):
    import h5py

    with h5py.File(h5_path, "r") as f:
        n = int(f.attrs.get("actual_records", f["clap_embeddings"].shape[0]))
        data = {
            "clap_embeddings": f["clap_embeddings"][:n].astype(np.float32),
            "style_labels": f["style_labels"][:n].astype(np.float32),
            "normalized_params": f["normalized_params"][:n].astype(np.float32),
            "effect_active_mask": f["effect_active_mask"][:n].astype(np.float32),
        }
        effect_names = [x for x in str(f.attrs.get("effect_names", "")).split(",") if x]
        temperature = float(f.attrs.get("temperature", 0.1))
    return data, effect_names, temperature


def load_vocab_keywords(aud_vocab_path: str) -> list[str]:
    d = np.load(aud_vocab_path, allow_pickle=True)
    if "keywords" in d:
        return d["keywords"].tolist()
    return d["terms"].tolist()


# ── 1. Entropy Analysis ─────────────────────────────────────────────────────

def analyze_entropy(style_labels: np.ndarray, vocab_size: int) -> dict:
    """Shannon entropy of each style label. Max entropy = log2(vocab_size)."""
    eps = 1e-12
    log_p = np.log2(style_labels + eps)
    entropy_per_sample = -np.sum(style_labels * log_p, axis=1)
    max_entropy = np.log2(vocab_size)

    # Effective dimensionality: 2^entropy (how many dims are "active")
    eff_dim = np.power(2.0, entropy_per_sample)

    # Argmax concentration: how much mass is in the top-1 term
    top1_mass = np.max(style_labels, axis=1)

    return {
        "max_possible_entropy": float(max_entropy),
        "mean_entropy": float(np.mean(entropy_per_sample)),
        "median_entropy": float(np.median(entropy_per_sample)),
        "std_entropy": float(np.std(entropy_per_sample)),
        "min_entropy": float(np.min(entropy_per_sample)),
        "max_entropy_observed": float(np.max(entropy_per_sample)),
        "mean_effective_dims": float(np.mean(eff_dim)),
        "median_effective_dims": float(np.median(eff_dim)),
        "mean_top1_mass": float(np.mean(top1_mass)),
        "median_top1_mass": float(np.median(top1_mass)),
        "pct_top1_above_0.5": float(np.mean(top1_mass > 0.5) * 100),
        "pct_top1_above_0.9": float(np.mean(top1_mass > 0.9) * 100),
        "entropy_histogram": np.histogram(entropy_per_sample, bins=30)[0].tolist(),
    }


# ── 2. Effect → Style Profile ───────────────────────────────────────────────

def effect_style_profiles(
    style_labels: np.ndarray,
    effect_mask: np.ndarray,
    effect_names: list[str],
    vocab_keywords: list[str],
) -> dict:
    """Average style profile per effect. Shows which vocab terms each effect maps to."""
    profiles = {}
    for i, ename in enumerate(effect_names):
        active_idx = effect_mask[:, i] > 0.5
        inactive_idx = ~active_idx
        n_active = int(active_idx.sum())
        n_inactive = int(inactive_idx.sum())

        if n_active < 10:
            continue

        mean_active = style_labels[active_idx].mean(axis=0)
        mean_inactive = style_labels[inactive_idx].mean(axis=0) if n_inactive > 10 else np.zeros_like(mean_active)
        delta = mean_active - mean_inactive

        # Top terms that shift most when this effect is on
        top_up = np.argsort(delta)[-5:][::-1]
        top_down = np.argsort(delta)[:5]

        profiles[ename] = {
            "n_active": n_active,
            "n_inactive": n_inactive,
            "top_up": [(vocab_keywords[j], f"{delta[j]:+.6f}") for j in top_up],
            "top_down": [(vocab_keywords[j], f"{delta[j]:+.6f}") for j in top_down],
            "max_delta": float(np.max(np.abs(delta))),
            "mean_abs_delta": float(np.mean(np.abs(delta))),
        }

    # Inter-effect discriminability: pairwise cosine distance between mean profiles
    mean_profiles = []
    profile_names = []
    for ename in effect_names:
        active_idx = effect_mask[:, effect_names.index(ename)] > 0.5
        if active_idx.sum() < 10:
            continue
        mean_profiles.append(style_labels[active_idx].mean(axis=0))
        profile_names.append(ename)

    if len(mean_profiles) >= 2:
        M = np.stack(mean_profiles)
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        M_norm = M / np.maximum(norms, 1e-8)
        cos_sim = M_norm @ M_norm.T
        # Report off-diagonal similarities
        n_eff = len(profile_names)
        pairs = []
        for ii in range(n_eff):
            for jj in range(ii + 1, n_eff):
                pairs.append({
                    "effect_a": profile_names[ii],
                    "effect_b": profile_names[jj],
                    "cosine_sim": float(cos_sim[ii, jj]),
                })
        pairs.sort(key=lambda x: x["cosine_sim"], reverse=True)
    else:
        pairs = []

    return {
        "per_effect": profiles,
        "pairwise_profile_similarity": pairs,
    }


# ── 3. Style → Effect Linear Probe ──────────────────────────────────────────

def linear_probe_effects(
    style_labels: np.ndarray,
    effect_mask: np.ndarray,
    effect_names: list[str],
    seed: int,
) -> dict:
    """Can a logistic regression predict active effects from style labels alone?"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, accuracy_score

    X_tr, X_te, Y_tr, Y_te = train_test_split(
        style_labels, effect_mask, test_size=0.2, random_state=seed,
    )

    results = {}
    for i, ename in enumerate(effect_names):
        y_tr = (Y_tr[:, i] > 0.5).astype(int)
        y_te = (Y_te[:, i] > 0.5).astype(int)

        clf = LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs")
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)

        results[ename] = {
            "accuracy": float(accuracy_score(y_te, y_pred)),
            "f1": float(f1_score(y_te, y_pred, zero_division=0)),
            "positive_rate": float(y_te.mean()),
            "baseline_accuracy": float(max(y_te.mean(), 1 - y_te.mean())),
        }

    # Also try with CLAP embedding as input (the full input the Controller sees)
    return results


def linear_probe_effects_with_clap(
    clap_embeddings: np.ndarray,
    style_labels: np.ndarray,
    effect_mask: np.ndarray,
    effect_names: list[str],
    seed: int,
) -> dict:
    """Same probe but with [CLAP, style_label] concatenated — the Controller's actual input."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, accuracy_score

    X = np.concatenate([clap_embeddings, style_labels], axis=1)
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X, effect_mask, test_size=0.2, random_state=seed,
    )

    results = {}
    for i, ename in enumerate(effect_names):
        y_tr = (Y_tr[:, i] > 0.5).astype(int)
        y_te = (Y_te[:, i] > 0.5).astype(int)

        clf = LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs")
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)

        results[ename] = {
            "accuracy": float(accuracy_score(y_te, y_pred)),
            "f1": float(f1_score(y_te, y_pred, zero_division=0)),
            "positive_rate": float(y_te.mean()),
            "baseline_accuracy": float(max(y_te.mean(), 1 - y_te.mean())),
        }

    return results


# ── 4. Style → Param Linear Probe ───────────────────────────────────────────

def linear_probe_params(
    clap_embeddings: np.ndarray,
    style_labels: np.ndarray,
    normalized_params: np.ndarray,
    effect_mask: np.ndarray,
    effect_names: list[str],
    seed: int,
) -> dict:
    """Can a linear regression predict params? Compare: style-only, CLAP-only, both."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    # Build param name list
    from src.effects.pedalboard_effects import EFFECT_CATALOG
    param_names = []
    param_effect_idx = []
    for ei, ename in enumerate(effect_names):
        for ps in EFFECT_CATALOG[ename].params:
            param_names.append(f"{ename}.{ps.name}")
            param_effect_idx.append(ei)

    # Only evaluate on active params per sample
    idx_tr, idx_te = train_test_split(
        np.arange(len(style_labels)), test_size=0.2, random_state=seed,
    )
    mask_te = effect_mask[idx_te]

    # Expand effect-level active mask to param-level mask.
    param_active_mask_te = np.zeros((len(idx_te), len(param_names)), dtype=np.float32)
    for pi, ei in enumerate(param_effect_idx):
        param_active_mask_te[:, pi] = (mask_te[:, ei] > 0.5).astype(np.float32)

    inputs = {
        "style_only": style_labels,
        "clap_only": clap_embeddings,
        "clap+style": np.concatenate([clap_embeddings, style_labels], axis=1),
    }

    results = {}
    for input_name, X_all in inputs.items():
        X_tr = X_all[idx_tr]
        X_te = X_all[idx_te]
        Y_tr = normalized_params[idx_tr]
        Y_te = normalized_params[idx_te]

        model = Ridge(alpha=1.0)
        model.fit(X_tr, Y_tr)
        Y_pred = model.predict(X_te)

        # Overall RMSE
        overall_rmse = float(np.sqrt(np.mean((Y_pred - Y_te) ** 2)))
        active_denom = float(np.maximum(param_active_mask_te.sum(), 1.0))
        active_only_rmse = float(
            np.sqrt(np.sum(((Y_pred - Y_te) ** 2) * param_active_mask_te) / active_denom)
        )

        # Per-param RMSE (active only)
        per_param = {}
        for pi, pname in enumerate(param_names):
            ei = param_effect_idx[pi]
            active = mask_te[:, ei] > 0.5
            if active.sum() < 10:
                continue
            rmse = float(np.sqrt(np.mean((Y_pred[active, pi] - Y_te[active, pi]) ** 2)))
            per_param[pname] = {"active_rmse": rmse, "n_active": int(active.sum())}

        results[input_name] = {
            "overall_rmse": overall_rmse,
            "active_only_rmse": active_only_rmse,
            "per_param": per_param,
        }

    # Mean predictor baseline
    Y_tr_base = normalized_params[idx_tr]
    Y_te_base = normalized_params[idx_te]
    mean_pred = np.tile(Y_tr_base.mean(axis=0), (len(idx_te), 1))
    baseline_rmse = float(np.sqrt(np.mean((mean_pred - Y_te_base) ** 2)))
    active_denom = float(np.maximum(param_active_mask_te.sum(), 1.0))
    baseline_active_rmse = float(
        np.sqrt(np.sum(((mean_pred - Y_te_base) ** 2) * param_active_mask_te) / active_denom)
    )

    per_param_baseline = {}
    for pi, pname in enumerate(param_names):
        ei = param_effect_idx[pi]
        active = effect_mask[idx_te][:, ei] > 0.5
        if active.sum() < 10:
            continue
        rmse = float(np.sqrt(np.mean((mean_pred[active, pi] - Y_te_base[active, pi]) ** 2)))
        per_param_baseline[pname] = {"active_rmse": rmse, "n_active": int(active.sum())}

    results["mean_baseline"] = {
        "overall_rmse": baseline_rmse,
        "active_only_rmse": baseline_active_rmse,
        "per_param": per_param_baseline,
    }

    return {"param_names": param_names, "probe_results": results}


# ── 5. Consistency: same effect on different audio ───────────────────────────

def consistency_check(
    style_labels: np.ndarray,
    effect_mask: np.ndarray,
    effect_names: list[str],
    n_bootstrap: int = 500,
    seed: int = 42,
) -> dict:
    """
    For each effect, sample pairs with the same active effect set.
    Compare their style label similarity to random pairs.
    If style labels are informative, same-effect pairs should be more similar.
    """
    rng = np.random.default_rng(seed)

    # Group samples by their active effect combination (as tuple of bools)
    combo_map: dict[tuple, list[int]] = {}
    for i in range(len(effect_mask)):
        key = tuple((effect_mask[i] > 0.5).astype(int).tolist())
        combo_map.setdefault(key, []).append(i)

    # Only keep combos with enough samples
    valid_combos = {k: v for k, v in combo_map.items() if len(v) >= 10}

    if not valid_combos:
        return {"error": "Not enough samples per effect combination"}

    # Sample same-combo pairs
    same_sims = []
    for combo, indices in valid_combos.items():
        idx_arr = np.array(indices)
        for _ in range(min(n_bootstrap, len(indices) * 2)):
            a, b = rng.choice(idx_arr, size=2, replace=False)
            cos = float(np.dot(style_labels[a], style_labels[b]) / (
                max(np.linalg.norm(style_labels[a]), 1e-8) *
                max(np.linalg.norm(style_labels[b]), 1e-8)
            ))
            same_sims.append(cos)

    # Sample random pairs
    all_indices = np.arange(len(style_labels))
    rand_sims = []
    for _ in range(len(same_sims)):
        a, b = rng.choice(all_indices, size=2, replace=False)
        cos = float(np.dot(style_labels[a], style_labels[b]) / (
            max(np.linalg.norm(style_labels[a]), 1e-8) *
            max(np.linalg.norm(style_labels[b]), 1e-8)
        ))
        rand_sims.append(cos)

    same_arr = np.array(same_sims)
    rand_arr = np.array(rand_sims)

    return {
        "same_combo_pairs": len(same_sims),
        "random_pairs": len(rand_sims),
        "same_combo_cosine_mean": float(same_arr.mean()),
        "same_combo_cosine_std": float(same_arr.std()),
        "random_cosine_mean": float(rand_arr.mean()),
        "random_cosine_std": float(rand_arr.std()),
        "separation": float(same_arr.mean() - rand_arr.mean()),
        "num_valid_combos": len(valid_combos),
        "largest_combo_sizes": sorted(
            [(str(k), len(v)) for k, v in valid_combos.items()],
            key=lambda x: x[1], reverse=True,
        )[:10],
    }


# ── 6. Controller vs Baselines ──────────────────────────────────────────────

def controller_vs_baselines(
    controller_report_path: str,
    probe_results: dict,
) -> dict:
    """Compare trained Controller's RMSE against linear probes and mean baseline."""
    report_path = Path(controller_report_path)
    if not report_path.exists():
        return {"error": f"Controller report not found at {controller_report_path}"}

    with open(report_path) as f:
        report = json.load(f)

    val_summary = report.get("val_summary", {})
    ctrl_rmse_aligned = val_summary.get(
        "selection_aligned_active_param_rmse",
        val_summary.get("active_only_rmse_gated", val_summary.get("active_only_rmse")),
    )
    if ctrl_rmse_aligned is None:
        ctrl_rmse_aligned = val_summary.get("overall_rmse")

    probes = probe_results["probe_results"]

    comparison = {
        "controller_rmse": ctrl_rmse_aligned,
        "controller_active_only_rmse": ctrl_rmse_aligned,
        "controller_overall_rmse": val_summary.get("overall_rmse"),
        "mean_baseline_rmse": probes["mean_baseline"]["overall_rmse"],
        "linear_style_only_rmse": probes["style_only"]["overall_rmse"],
        "linear_clap_only_rmse": probes["clap_only"]["overall_rmse"],
        "linear_clap_style_rmse": probes["clap+style"]["overall_rmse"],
        "mean_baseline_active_only_rmse": probes["mean_baseline"].get("active_only_rmse"),
        "linear_style_only_active_only_rmse": probes["style_only"].get("active_only_rmse"),
        "linear_clap_only_active_only_rmse": probes["clap_only"].get("active_only_rmse"),
        "linear_clap_style_active_only_rmse": probes["clap+style"].get("active_only_rmse"),
    }

    # Per-param comparison
    per_param = {}
    param_names = probe_results["param_names"]
    ctrl_top = val_summary.get("top5_highest_active_rmse_gated", val_summary.get("top5_highest_rmse", []))
    ctrl_params = {}
    for p in ctrl_top:
        pname = p.get("param")
        if not pname:
            continue
        ctrl_params[pname] = p.get("active_rmse_gated", p.get("active_rmse", p.get("rmse")))

    for pname in param_names:
        entry = {}
        if pname in ctrl_params:
            entry["controller"] = ctrl_params[pname]
        for probe_name in ["mean_baseline", "style_only", "clap_only", "clap+style"]:
            pp = probes[probe_name]["per_param"]
            if pname in pp:
                entry[probe_name] = pp[pname]["active_rmse"]
        if entry:
            per_param[pname] = entry

    comparison["per_param_sample"] = per_param
    return comparison


# ── Visualization ────────────────────────────────────────────────────────────

def save_plots(
    out_dir: Path,
    style_labels: np.ndarray,
    effect_mask: np.ndarray,
    effect_names: list[str],
    vocab_keywords: list[str],
    entropy_result: dict,
    probe_params_result: dict,
):
    """Generate diagnostic plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Plot 1: Entropy histogram ──
    eps = 1e-12
    entropies = -np.sum(style_labels * np.log2(style_labels + eps), axis=1)
    max_ent = np.log2(style_labels.shape[1])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(entropies, bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axvline(max_ent, color="red", linestyle="--", label=f"Max entropy ({max_ent:.2f})")
    ax.axvline(np.mean(entropies), color="orange", linestyle="--", label=f"Mean ({np.mean(entropies):.2f})")
    ax.set_xlabel("Shannon Entropy (bits)")
    ax.set_ylabel("Count")
    ax.set_title("Style Label Entropy Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "1_entropy_histogram.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: Mean style profile per effect ──
    fig, axes = plt.subplots(len(effect_names), 1, figsize=(12, 2.5 * len(effect_names)), sharex=True)
    if len(effect_names) == 1:
        axes = [axes]

    for i, (ax, ename) in enumerate(zip(axes, effect_names)):
        active = effect_mask[:, i] > 0.5
        inactive = ~active
        mean_on = style_labels[active].mean(axis=0) if active.sum() > 0 else np.zeros(style_labels.shape[1])
        mean_off = style_labels[inactive].mean(axis=0) if inactive.sum() > 0 else np.zeros(style_labels.shape[1])
        delta = mean_on - mean_off

        x = np.arange(len(vocab_keywords))
        ax.bar(x, delta, color=["tab:blue" if d >= 0 else "tab:red" for d in delta], alpha=0.7)
        ax.set_ylabel("Delta")
        ax.set_title(f"{ename} ON vs OFF (n_on={int(active.sum())})")
        ax.axhline(0, color="gray", linewidth=0.5)

    axes[-1].set_xticks(np.arange(len(vocab_keywords)))
    axes[-1].set_xticklabels(vocab_keywords, rotation=45, ha="right", fontsize=8)
    fig.suptitle("Style Label Shift per Effect (active - inactive)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "2_effect_style_profiles.png", dpi=150)
    plt.close(fig)

    # ── Plot 3: Top-1 mass histogram ──
    top1 = np.max(style_labels, axis=1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(top1, bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Top-1 probability mass")
    ax.set_ylabel("Count")
    ax.set_title("Argmax Concentration (how peaky is each label?)")
    ax.axvline(np.mean(top1), color="orange", linestyle="--", label=f"Mean={np.mean(top1):.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "3_top1_mass_histogram.png", dpi=150)
    plt.close(fig)

    # ── Plot 4: Probe comparison bar chart ──
    probes = probe_params_result["probe_results"]
    methods = ["mean_baseline", "style_only", "clap_only", "clap+style"]
    labels = ["Mean baseline", "Style only", "CLAP only", "CLAP+Style"]
    rmses = [probes[m].get("active_only_rmse", probes[m]["overall_rmse"]) for m in methods]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, rmses, color=["gray", "tab:orange", "tab:blue", "tab:green"], alpha=0.8)
    for bar, v in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Active-only RMSE")
    ax.set_title("Linear Probe: Param Prediction (Active-only RMSE)")
    fig.tight_layout()
    fig.savefig(out_dir / "4_probe_comparison.png", dpi=150)
    plt.close(fig)

    # ── Plot 5: Per-param RMSE comparison ──
    param_names = probe_params_result["param_names"]
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(param_names))
    width = 0.2
    for offset, (method, label, color) in enumerate(zip(methods, labels, ["gray", "tab:orange", "tab:blue", "tab:green"])):
        vals = []
        for pn in param_names:
            pp = probes[method]["per_param"]
            vals.append(pp[pn]["active_rmse"] if pn in pp else 0.0)
        ax.bar(x + offset * width, vals, width, label=label, color=color, alpha=0.8)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Active-only RMSE")
    ax.set_title("Per-Parameter RMSE: Linear Probes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "5_per_param_probe_rmse.png", dpi=150)
    plt.close(fig)

    # ── Plot 6: Style label correlation matrix ──
    # Correlation between style label dimensions across dataset
    corr = np.corrcoef(style_labels.T)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(vocab_keywords)))
    ax.set_yticks(np.arange(len(vocab_keywords)))
    ax.set_xticklabels(vocab_keywords, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(vocab_keywords, fontsize=7)
    ax.set_title("Style Label Dimension Correlations")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "6_style_label_correlations.png", dpi=150)
    plt.close(fig)

    print(f"Saved 6 diagnostic plots to {out_dir}/")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data, effect_names, temperature = load_db(args.h5)
    vocab_keywords = load_vocab_keywords(args.aud_vocab)
    n = len(data["style_labels"])
    print(f"  Records: {n}, Effects: {effect_names}, Vocab: {len(vocab_keywords)}, T={temperature}")

    all_results = {
        "dataset": {
            "n_records": n,
            "n_effects": len(effect_names),
            "effect_names": effect_names,
            "vocab_size": len(vocab_keywords),
            "vocab_keywords": vocab_keywords,
            "temperature": temperature,
        },
    }

    # ── Test 1: Entropy ──
    print("\n[1/6] Entropy analysis...")
    ent = analyze_entropy(data["style_labels"], len(vocab_keywords))
    all_results["entropy"] = ent
    print(f"  Mean entropy: {ent['mean_entropy']:.3f} / {ent['max_possible_entropy']:.3f} bits")
    print(f"  Mean effective dims: {ent['mean_effective_dims']:.2f} / {len(vocab_keywords)}")
    print(f"  Mean top-1 mass: {ent['mean_top1_mass']:.4f}")
    print(f"  % labels with top1 > 0.9: {ent['pct_top1_above_0.9']:.1f}%")

    # ── Test 2: Effect → Style profiles ──
    print("\n[2/6] Effect → Style profiles...")
    profiles = effect_style_profiles(
        data["style_labels"], data["effect_active_mask"], effect_names, vocab_keywords,
    )
    all_results["effect_profiles"] = profiles
    for ename, prof in profiles["per_effect"].items():
        print(f"  {ename}: max_delta={prof['max_delta']:.6f}, mean_abs_delta={prof['mean_abs_delta']:.6f}")
        print(f"    top_up:   {prof['top_up'][:3]}")
        print(f"    top_down: {prof['top_down'][:3]}")
    if profiles["pairwise_profile_similarity"]:
        print("  Pairwise profile similarity (top-3 most similar):")
        for pair in profiles["pairwise_profile_similarity"][:3]:
            print(f"    {pair['effect_a']} ↔ {pair['effect_b']}: cos={pair['cosine_sim']:.4f}")

    # ── Test 3: Style → Effect linear probe ──
    print("\n[3/6] Linear probe: style → effect activity...")
    probe_eff_style = linear_probe_effects(
        data["style_labels"], data["effect_active_mask"], effect_names, args.seed,
    )
    all_results["probe_style_to_effect"] = probe_eff_style
    for ename, res in probe_eff_style.items():
        gain = res["accuracy"] - res["baseline_accuracy"]
        print(f"  {ename}: F1={res['f1']:.3f}, Acc={res['accuracy']:.3f} "
              f"(baseline={res['baseline_accuracy']:.3f}, gain={gain:+.3f})")

    print("\n  ...with [CLAP + style] input:")
    probe_eff_both = linear_probe_effects_with_clap(
        data["clap_embeddings"], data["style_labels"],
        data["effect_active_mask"], effect_names, args.seed,
    )
    all_results["probe_clap_style_to_effect"] = probe_eff_both
    for ename, res in probe_eff_both.items():
        gain = res["accuracy"] - res["baseline_accuracy"]
        print(f"  {ename}: F1={res['f1']:.3f}, Acc={res['accuracy']:.3f} "
              f"(baseline={res['baseline_accuracy']:.3f}, gain={gain:+.3f})")

    # ── Test 4: Style → Param linear probe ──
    print("\n[4/6] Linear probe: inputs → params...")
    probe_params = linear_probe_params(
        data["clap_embeddings"], data["style_labels"],
        data["normalized_params"], data["effect_active_mask"],
        effect_names, args.seed,
    )
    all_results["probe_params"] = probe_params
    for method in ["mean_baseline", "style_only", "clap_only", "clap+style"]:
        rmse_overall = probe_params["probe_results"][method]["overall_rmse"]
        rmse_active = probe_params["probe_results"][method].get("active_only_rmse", rmse_overall)
        print(
            f"  {method:20s}: overall RMSE = {rmse_overall:.4f}, "
            f"active-only RMSE = {rmse_active:.4f}"
        )

    # ── Test 5: Consistency ──
    print("\n[5/6] Consistency check...")
    consist = consistency_check(
        data["style_labels"], data["effect_active_mask"], effect_names,
        seed=args.seed,
    )
    all_results["consistency"] = consist
    if "error" not in consist:
        print(f"  Same-combo cosine: {consist['same_combo_cosine_mean']:.4f} "
              f"± {consist['same_combo_cosine_std']:.4f}")
        print(f"  Random cosine:     {consist['random_cosine_mean']:.4f} "
              f"± {consist['random_cosine_std']:.4f}")
        print(f"  Separation:        {consist['separation']:.4f}")
        print(f"  Valid combos: {consist['num_valid_combos']}")

    # ── Test 6: Controller vs baselines ──
    print("\n[6/6] Controller vs baselines...")
    if args.controller_report:
        comparison = controller_vs_baselines(args.controller_report, probe_params)
        all_results["controller_vs_baselines"] = comparison
        if "error" not in comparison:
            print(f"  Controller active RMSE: {comparison['controller_active_only_rmse']:.4f}")
            print(f"  Mean baseline active RMSE: {comparison['mean_baseline_active_only_rmse']:.4f}")
            print(f"  Linear style active RMSE:  {comparison['linear_style_only_active_only_rmse']:.4f}")
            print(f"  Linear CLAP active RMSE:   {comparison['linear_clap_only_active_only_rmse']:.4f}")
            print(f"  Linear both active RMSE:   {comparison['linear_clap_style_active_only_rmse']:.4f}")
    else:
        all_results["controller_vs_baselines"] = {"skipped": True, "reason": "controller report not provided"}
        print("  Skipped (no --controller-report provided).")

    # ── Save results ──
    report_out = out_dir / "diagnosis_report.json"
    with open(report_out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull report saved to {report_out}")

    # ── Plots ──
    print("\nGenerating plots...")
    save_plots(
        out_dir=out_dir,
        style_labels=data["style_labels"],
        effect_mask=data["effect_active_mask"],
        effect_names=effect_names,
        vocab_keywords=vocab_keywords,
        entropy_result=ent,
        probe_params_result=probe_params,
    )

    # ── Summary verdict ──
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)

    issues = []
    if ent["mean_effective_dims"] < 3.0:
        issues.append(
            f"CRITICAL: Style labels are extremely peaky (eff. dims={ent['mean_effective_dims']:.1f}). "
            f"T={temperature} makes softmax near-argmax. Most of the 24-dim label is wasted."
        )
    elif ent["mean_effective_dims"] < 6.0:
        issues.append(
            f"WARNING: Style labels are peaky (eff. dims={ent['mean_effective_dims']:.1f}). "
            f"Consider raising temperature."
        )

    max_profile_delta = max(
        (p["max_delta"] for p in profiles["per_effect"].values()), default=0,
    )
    if max_profile_delta < 0.01:
        issues.append(
            "CRITICAL: Effect ON/OFF produces nearly identical style profiles. "
            "Style labels carry almost no effect-discriminative information."
        )
    elif max_profile_delta < 0.03:
        issues.append(
            f"WARNING: Effect→Style delta is very small (max={max_profile_delta:.4f}). "
            "Style labels weakly encode effect information."
        )

    probe_rmse = probe_params["probe_results"]
    baseline_rmse = probe_rmse["mean_baseline"].get("active_only_rmse", probe_rmse["mean_baseline"]["overall_rmse"])
    style_rmse = probe_rmse["style_only"].get("active_only_rmse", probe_rmse["style_only"]["overall_rmse"])
    both_rmse = probe_rmse["clap+style"].get("active_only_rmse", probe_rmse["clap+style"]["overall_rmse"])
    if style_rmse > baseline_rmse * 0.95:
        issues.append(
            f"CRITICAL: Linear probe with style labels (RMSE={style_rmse:.4f}) "
            f"is no better than mean baseline (RMSE={baseline_rmse:.4f}). "
            f"Style labels contain negligible param information."
        )
    if both_rmse > baseline_rmse * 0.90:
        issues.append(
            f"WARNING: Even [CLAP+style] linear probe (RMSE={both_rmse:.4f}) "
            f"barely beats mean baseline (RMSE={baseline_rmse:.4f})."
        )

    if "error" not in consist and consist["separation"] < 0.01:
        issues.append(
            f"WARNING: Same-effect pairs have nearly identical similarity to random pairs "
            f"(separation={consist['separation']:.4f}). Style labels are not consistent across audio."
        )

    if not issues:
        print("No critical issues detected. Style labels appear informative.")
    else:
        for i, issue in enumerate(issues, 1):
            print(f"\n{i}. {issue}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
