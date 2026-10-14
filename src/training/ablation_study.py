"""PMSM ablation study — EWMA feature contribution with multi-seed evaluation.

Runs the full HPO + retrain + test pipeline for each (config, seed) pair
and aggregates results into a summary table + box-plot figure.

Ablation axis
-------------
  Condition A — raw inputs only          (no EWMA, no physics)
  Condition B — raw + physics            (i_squared, power_elec, …)
  Condition C — raw + EWMA              (temporal features, no physics)
  Condition D — raw + physics + EWMA    (full feature set — your baseline)

Each condition is evaluated on N seeds (default 5) so you get mean ± std
across random HPO initialisations and profile splits.

Usage
-----
    # Full ablation, 5 seeds, all configs:
    uv run python -m src.training.ablation_ewma \
        --csv path/to/measures_v2.csv

    # Quick smoke test (2 seeds, 10 trials, 30 epochs):
    uv run python -m src.training.ablation_ewma \
        --csv measures_v2.csv \
        --seeds 0 1 \
        --n-trials 10 --max-epochs 30 --retrain-epochs 40 \
        --hpo-pool-size 5000

    # Single condition, custom seeds:
    uv run python -m src.training.ablation_study `
        --csv measures_v2.csv `
        --conditions A_raw B_physics `
        --seeds 0 1 2 3 4 `
        --n-trials 30 --max-epochs 300 --patience 40 --hpo-pool-size 40000

Outputs
-------
    results/ablation_ewma/
        ablation_results.csv        — one row per (condition, config, seed)
        ablation_summary.csv        — mean ± std across seeds
        ablation_boxplot.png        — RMSE°C distribution per condition
        ablation_r2_boxplot.png     — R² distribution per condition
        ablation_mae_boxplot.png    — MAE°C distribution per condition
        ablation_improvement_heatmap.png — ΔRMSE vs A_raw
        runs/
            <condition>_<config>_seed<N>/
                config.json
                model.pt
                metrics.json
                training_curves.png
                hpo_results.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# ── project imports ────────────────────────────────────────────────────────────
from src.models.mlp import MLP                                       # noqa: E402
from src.data.pmsm_configs import ALL_CONFIG_NAMES, get_config       # noqa: E402

# loader API — everything comes from pmsm_loader_ewma
from src.data.pmsm_loader_ewma import (                              # noqa: E402
    make_profile_split,          # random / chronological split
    add_physics_features,        # (df, phys_cols) → (df, list[str])
    add_ewma_features,           # (df, cols, spans) → (df, list[str])
    _compute_norm_stats,         # train-only z-score stats
    _normalise,                  # (arr, mean, std) → arr
    DEFAULT_EWMA_COLS,
    DEFAULT_EWMA_SPANS,
    PHYSICS_VAR,                # ["power_dq", "torque_speed", "i_cross", "i_squared"]
)

# training API — everything comes from train_pmsm_baseline
from src.training.train_pmsm_baseline import (                       # noqa: E402
    train_minibatch,
    make_objective,
    select_best_trial,
    save_training_curves,
    save_hpo_csv,
    DEFAULT_N_TRIALS,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_PATIENCE,
    DEFAULT_RETRAIN_EPOCHS,
    DEFAULT_RETRAIN_PAT,
    DEFAULT_HPO_POOL_SIZE,
    PARAM_BUDGET_SOFT_CAP,
    SELECTION_TOLERANCE,
)

# ── paths ──────────────────────────────────────────────────────────────────────
ABLATION_DIR = REPO_ROOT / "results" / "ablation_ewma"

# ── ablation conditions ────────────────────────────────────────────────────────
ABLATION_CONDITIONS: dict[str, dict] = {
    "A_raw": {
        "label":       "Raw inputs only",
        "add_physics": False,
        "add_ewma":    False,
        "ewma_spans":  [],
    },
    "B_physics": {
        "label":       "Raw + physics",
        "add_physics": True,
        "add_ewma":    False,
        "ewma_spans":  [],
    },
    "C_ewma": {
        "label":       "Raw + EWMA",
        "add_physics": False,
        "add_ewma":    True,
        "ewma_spans":  [60, 300, 1200, 6000, 12000],
    },
    "D_full": {
        "label":       "Raw + physics + EWMA  (full)",
        "add_physics": True,
        "add_ewma":    True,
        "ewma_spans":  [60, 300, 1200, 6000, 12000],
    },
}

DEFAULT_CONDITIONS = list(ABLATION_CONDITIONS.keys())
DEFAULT_SEEDS      = [0, 1, 2, 3, 4]

# colour palette — one per condition, used in all plots
_PALETTE: dict[str, str] = {
    "A_raw":     "#adb5bd",
    "B_physics": "#4895ef",
    "C_ewma":    "#f77f00",
    "D_full":    "#2dc653",
}


# ══════════════════════════════════════════════════════════════════════════════
# Data preparation — one (condition × seed) pair
# ══════════════════════════════════════════════════════════════════════════════

def prepare_tensors_for_condition(
    csv_path: Path,
    config_name: str,
    condition: dict,
    seed: int,
    train_frac: float = 0.70,
    val_frac: float = 0.10,
) -> tuple[
    torch.Tensor, torch.Tensor,   # x_tr, y_tr
    torch.Tensor, torch.Tensor,   # x_val, y_val
    torch.Tensor, torch.Tensor,   # x_te, y_te
    dict,                         # norm_stats
]:
    """Build normalised train/val/test tensors for one (condition, seed) pair.

    Uses the public API of pmsm_loader_ewma — no private internals except
    _compute_norm_stats / _normalise which are the canonical helpers.

    Each seed gets its own random profile split so variance across seeds
    reflects both HPO randomness AND data-split randomness.
    """
    # ── 1. raw CSV ─────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)

    # ── 2. physics features (before EWMA so ewma(i_squared) is valid) ──────
    phys_new_cols: list[str] = []
    if condition["add_physics"]:
        df, phys_new_cols = add_physics_features(df)

    # ── 3. EWMA features, grouped by profile_id ────────────────────────────
    ewma_new_cols: list[str] = []
    if condition["add_ewma"] and condition["ewma_spans"]:
        # Base cols: standard signals + any physics cols we just created
        ewma_base = list(DEFAULT_EWMA_COLS) + [
            c for c in phys_new_cols if c not in DEFAULT_EWMA_COLS
        ]
        # Keep only cols that actually exist in df
        ewma_base = [c for c in ewma_base if c in df.columns]
        df, ewma_new_cols = add_ewma_features(
            df, ewma_base, condition["ewma_spans"]
        )

    # ── 4. Build full input column list ────────────────────────────────────
    # get_config returns the BASE columns defined in pmsm_configs.py.
    # We append: physics cols not already in base, then all EWMA cols.
    base_input_cols, target_col = get_config(config_name)

    extra_phys = [
        c for c in phys_new_cols
        if c in df.columns and c not in base_input_cols
    ]
    input_cols = list(base_input_cols) + extra_phys + ewma_new_cols

    # Guard: every requested column must exist in df
    missing = [c for c in input_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"prepare_tensors_for_condition: {len(missing)} columns missing "
            f"from df.\nMissing: {missing[:15]}"
        )

    # ── 5. Random profile split (seeded) ───────────────────────────────────
    # make_profile_split from pmsm_loader_ewma — random_split=True
    split_ids = make_profile_split(
        df,
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
        random_split=True,
    )

    masks = {
        split: df["profile_id"].isin(pid_list).to_numpy()
        for split, pid_list in split_ids.items()
    }

    # ── 6. Normalisation (train stats only) ────────────────────────────────
    norm_stats = _compute_norm_stats(
        df, masks["train"], input_cols, target_col
    )
    x_mean = np.array(norm_stats["x_mean"], dtype=np.float64)
    x_std  = np.array(norm_stats["x_std"],  dtype=np.float64)
    y_mean = norm_stats["y_mean"]
    y_std  = norm_stats["y_std"]

    # ── 7. Build tensors ────────────────────────────────────────────────────
    def _to_tensors(mask: np.ndarray):
        sub   = df.loc[mask]
        x_raw = sub[input_cols].to_numpy(dtype=np.float64)
        y_raw = sub[[target_col]].to_numpy(dtype=np.float64)
        return (
            torch.tensor(_normalise(x_raw, x_mean, x_std), dtype=torch.float32),
            torch.tensor(_normalise(y_raw, y_mean, y_std), dtype=torch.float32),
        )

    x_tr,  y_tr  = _to_tensors(masks["train"])
    x_val, y_val = _to_tensors(masks["val"])
    x_te,  y_te  = _to_tensors(masks["test"])

    # Annotate norm_stats with provenance info for later inspection
    norm_stats.update({
        "input_cols":   input_cols,
        "target_col":   target_col,
        "condition":    condition["label"],
        "seed":         seed,
        "add_physics":  condition["add_physics"],
        "add_ewma":     condition["add_ewma"],
        "ewma_spans":   condition["ewma_spans"],
        "phys_cols":    phys_new_cols,
        "ewma_new_cols": ewma_new_cols,
        "train_profiles": split_ids["train"],
        "val_profiles":   split_ids["val"],
        "test_profiles":  split_ids["test"],
    })

    return x_tr, y_tr, x_val, y_val, x_te, y_te, norm_stats


# ══════════════════════════════════════════════════════════════════════════════
# Single (condition × config × seed) run
# ══════════════════════════════════════════════════════════════════════════════

def run_single(
    csv_path: Path,
    condition_key: str,
    config_name: str,
    seed: int,
    n_trials: int,
    max_epochs: int,
    patience: int,
    retrain_epochs: int,
    retrain_patience: int,
    hpo_pool_size: int,
    device: torch.device,
    out_dir: Path,
    train_frac: float = 0.70,
    val_frac: float = 0.10,
) -> dict:
    """Full HPO → retrain → locked test for one (condition, config, seed).

    Returns a flat metrics dict suitable for appending to the results list.
    """
    import optuna

    condition = ABLATION_CONDITIONS[condition_key]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  ── {condition_key}  |  {config_name}  |  seed={seed} ──")
    print(f"     {condition['label']}")
    print(f"     add_physics={condition['add_physics']}  "
          f"add_ewma={condition['add_ewma']}  "
          f"spans={condition['ewma_spans']}")

    # ── data ───────────────────────────────────────────────────────────────
    t0 = time.time()
    (x_tr, y_tr,
     x_val, y_val,
     x_te, y_te,
     norm_stats) = prepare_tensors_for_condition(
        csv_path, config_name, condition, seed,
        train_frac=train_frac,
        val_frac=val_frac,
    )
    input_dim = x_tr.shape[1]
    print(f"     input_dim={input_dim}  "
          f"train={x_tr.shape[0]:,}  "
          f"val={x_val.shape[0]:,}  "
          f"test={x_te.shape[0]:,}  "
          f"({time.time()-t0:.1f}s)")

    # ── HPO subsample ──────────────────────────────────────────────────────
    if hpo_pool_size and x_tr.shape[0] > hpo_pool_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(x_tr.shape[0], hpo_pool_size, replace=False)
        idx.sort()
        x_pool, y_pool = x_tr[idx], y_tr[idx]
        print(f"     HPO pool: {hpo_pool_size:,} rows (subsampled)")
    else:
        x_pool, y_pool = x_tr, y_tr
        print(f"     HPO pool: {x_pool.shape[0]:,} rows (full train)")

    # ── Optuna ─────────────────────────────────────────────────────────────
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            multivariate=True, group=True, seed=seed,
        ),
    )
    objective = make_objective(
        x_pool, y_pool, x_val, y_val,
        input_dim=input_dim,
        max_epochs=max_epochs,
        patience=patience,
        device=device,
    )

    t_hpo = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    hpo_time = time.time() - t_hpo

    n_complete = sum(
        1 for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    )
    n_pruned = sum(
        1 for t in study.trials
        if t.state == optuna.trial.TrialState.PRUNED
    )
    print(f"     HPO: {n_complete} complete  {n_pruned} pruned  {hpo_time:.0f}s")

    if n_complete == 0:
        raise RuntimeError(
            f"All {n_trials} trials were pruned — "
            f"try raising PARAM_BUDGET_SOFT_CAP or reducing n_neurons range."
        )

    # ── select + retrain ───────────────────────────────────────────────────
    chosen, overall_best = select_best_trial(study)
    p = chosen.params
    print(f"     chosen: trial #{chosen.number}  "
          f"val_mse={chosen.value:.4e}  "
          f"n_params={chosen.user_attrs['n_params']}  "
          f"arch={p['n_layers']}×{p['n_neurons']} {p['activation']}")

    model = MLP(
        n_layers=int(p["n_layers"]),
        n_neurons=int(p["n_neurons"]),
        activation=p["activation"],
        input_dim=input_dim,
        output_dim=1,
    )
    n_params_final = model.get_weight_count()

    t_retrain = time.time()
    retrain_val_mse, retrain_info = train_minibatch(
        model, x_tr, y_tr, x_val, y_val,
        lr=float(p["lr"]),
        weight_decay=float(p["weight_decay"]),
        batch_size=int(p["batch_size"]),
        max_epochs=retrain_epochs,
        patience=retrain_patience,
        device=device,
        record_curves=True,
        verbose=False,
    )
    retrain_time = time.time() - t_retrain
    print(f"     retrain: val_mse={retrain_val_mse:.4e}  "
          f"epochs={retrain_info['epoch']}  {retrain_time:.0f}s")

    # ── locked test (evaluated ONCE) ───────────────────────────────────────
    model.eval()
    x_te_d = x_te.to(device)
    y_te_d = y_te.to(device)
    with torch.no_grad():
        preds    = model.to(device)(x_te_d)
        test_mse = nn.functional.mse_loss(preds, y_te_d).item()

    y_std    = norm_stats["y_std"]
    y_mean_c = norm_stats["y_mean"]
    preds_c  = preds.cpu().numpy()  * y_std + y_mean_c
    actual_c = y_te_d.cpu().numpy() * y_std + y_mean_c
    errors   = np.abs(preds_c - actual_c)
    mae_c    = float(errors.mean())
    rmse_c   = float(np.sqrt(((preds_c - actual_c) ** 2).mean()))
    max_c    = float(errors.max())
    ss_res   = float(np.sum((actual_c - preds_c) ** 2))
    ss_tot   = float(np.sum((actual_c - actual_c.mean()) ** 2))
    r2       = float(1.0 - ss_res / ss_tot)

    print(f"     TEST → RMSE={rmse_c:.2f}°C  MAE={mae_c:.2f}°C  "
          f"R²={r2:.4f}  max={max_c:.2f}°C")

    # ── persist artefacts ──────────────────────────────────────────────────
    torch.save(model.state_dict(), out_dir / "model.pt")

    if "train_hist" in retrain_info:
        save_training_curves(
            retrain_info["train_hist"],
            retrain_info["val_hist"],
            out_dir,
        )

    save_hpo_csv(study, out_dir)

    # metrics dict — flat, one row in results CSV
    arch_str = f"{p['n_layers']}×{p['n_neurons']} {p['activation']}"
    run_metrics = {
        "condition":         condition_key,
        "condition_label":   condition["label"],
        "config_name":       config_name,
        "seed":              seed,
        "add_physics":       condition["add_physics"],
        "add_ewma":          condition["add_ewma"],
        "ewma_spans":        str(condition["ewma_spans"]),
        "input_dim":         input_dim,
        "architecture":      arch_str,
        "n_params":          n_params_final,
        "hpo_best_val_mse":  overall_best.value,
        "chosen_val_mse":    chosen.value,
        "retrain_val_mse":   retrain_val_mse,
        "test_mse_norm":     test_mse,
        "test_rmse_c":       rmse_c,
        "test_mae_c":        mae_c,
        "test_max_c":        max_c,
        "test_r2":           r2,
        "hpo_time_s":        round(hpo_time, 1),
        "retrain_time_s":    round(retrain_time, 1),
        "n_complete_trials": n_complete,
        "n_pruned_trials":   n_pruned,
        "retrain_epochs":    retrain_info["epoch"],
    }

    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(run_metrics, fh, indent=2)

    with open(out_dir / "config.json", "w") as fh:
        json.dump({
            "dataset":      config_name,
            "architecture": {
                "n_layers":   int(p["n_layers"]),
                "n_neurons":  int(p["n_neurons"]),
                "activation": p["activation"],
                "input_dim":  input_dim,
                "output_dim": 1,
            },
            "training_hparams": {
                "lr":           float(p["lr"]),
                "weight_decay": float(p["weight_decay"]),
                "batch_size":   int(p["batch_size"]),
                "max_epochs":   retrain_epochs,
                "patience":     retrain_patience,
            },
            "n_params": n_params_final,
            "metrics":  run_metrics,
            "normalisation": {
                "y_mean_celsius": y_mean_c,
                "y_std_celsius":  y_std,
            },
            "data_provenance": {
                "input_cols":  norm_stats["input_cols"],
                "phys_cols":   norm_stats["phys_cols"],
                "ewma_new_cols": norm_stats["ewma_new_cols"],
                "train_profiles": norm_stats["train_profiles"],
                "val_profiles":   norm_stats["val_profiles"],
                "test_profiles":  norm_stats["test_profiles"],
            },
        }, fh, indent=2)

    return run_metrics


# ══════════════════════════════════════════════════════════════════════════════
# Aggregation
# ══════════════════════════════════════════════════════════════════════════════

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std per (condition, config_name) across seeds."""
    numeric_cols = [
        "test_rmse_c", "test_mae_c", "test_r2",
        "test_max_c", "n_params", "input_dim",
        "retrain_val_mse", "hpo_time_s", "retrain_time_s",
    ]
    # Only aggregate columns that are actually present
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    agg = (
        df.groupby(["condition", "condition_label", "config_name"])[numeric_cols]
        .agg(["mean", "std"])
    )
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg.reset_index()


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def _boxplot_metric(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    fname: str,
    out_dir: Path,
    conditions: list[str],
    configs: list[str],
) -> None:
    cond_labels = {k: ABLATION_CONDITIONS[k]["label"] for k in conditions
                   if k in ABLATION_CONDITIONS}

    n_configs = len(configs)
    fig, axes = plt.subplots(
        1, n_configs,
        figsize=(max(5, 4 * n_configs), 5),
        sharey=False,
    )
    if n_configs == 1:
        axes = [axes]

    rng = np.random.default_rng(0)   # reproducible jitter

    for ax, cfg in zip(axes, configs):
        sub            = df[df["config_name"] == cfg]
        data_per_cond  = [
            sub[sub["condition"] == cond][metric].dropna().values
            for cond in conditions
        ]
        bp = ax.boxplot(
            data_per_cond,
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
        )
        for patch, cond in zip(bp["boxes"], conditions):
            patch.set_facecolor(_PALETTE.get(cond, "#999"))
            patch.set_alpha(0.85)

        for i, (cond, vals) in enumerate(zip(conditions, data_per_cond), start=1):
            jitter = rng.uniform(-0.08, 0.08, len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter,
                vals,
                color=_PALETTE.get(cond, "#999"),
                edgecolors="black",
                s=45, zorder=5, linewidths=0.7,
            )

        ax.set_xticks(range(1, len(conditions) + 1))
        ax.set_xticklabels(
            [cond_labels.get(c, c) for c in conditions],
            rotation=22, ha="right", fontsize=8,
        )
        ax.set_title(cfg, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel if ax is axes[0] else "")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    n_seeds = df["seed"].nunique()
    fig.suptitle(
        f"Ablation: {ylabel} by condition\n"
        f"{n_seeds} seeds — each dot = one seed",
        fontsize=11, fontweight="bold",
    )

    handles = [
        plt.Rectangle((0, 0), 1, 1,
                       facecolor=_PALETTE.get(c, "#999"),
                       edgecolor="black", alpha=0.85)
        for c in conditions
    ]
    fig.legend(
        handles,
        [cond_labels.get(c, c) for c in conditions],
        loc="lower center",
        ncol=min(len(conditions), 4),
        fontsize=8,
        bbox_to_anchor=(0.5, -0.04),
    )
    plt.tight_layout()
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved → {out_dir / fname}")


def plot_boxplots(
    df: pd.DataFrame,
    out_dir: Path,
    conditions: list[str],
    configs: list[str],
) -> None:
    for metric, ylabel, fname in [
        ("test_rmse_c", "RMSE (°C)",  "ablation_boxplot.png"),
        ("test_r2",     "R²",         "ablation_r2_boxplot.png"),
        ("test_mae_c",  "MAE (°C)",   "ablation_mae_boxplot.png"),
    ]:
        _boxplot_metric(df, metric, ylabel, fname, out_dir, conditions, configs)


def plot_improvement_heatmap(
    summary: pd.DataFrame,
    out_dir: Path,
    conditions: list[str],
    configs: list[str],
) -> None:
    """ΔRMSE heatmap: improvement over A_raw per (config, condition)."""
    import matplotlib.colors as mcolors

    compare_conds = [c for c in conditions if c != "A_raw"]
    if not compare_conds:
        return

    matrix = np.full((len(configs), len(compare_conds)), np.nan)
    for i, cfg in enumerate(configs):
        base_row = summary[
            (summary["config_name"] == cfg) &
            (summary["condition"] == "A_raw")
        ]
        if base_row.empty:
            continue
        base_rmse = float(base_row["test_rmse_c_mean"].iloc[0])
        for j, cond in enumerate(compare_conds):
            row = summary[
                (summary["config_name"] == cfg) &
                (summary["condition"] == cond)
            ]
            if not row.empty:
                matrix[i, j] = base_rmse - float(row["test_rmse_c_mean"].iloc[0])

    valid = matrix[~np.isnan(matrix)]
    if len(valid) == 0:
        return

    vmin, vmax = valid.min(), valid.max()
    # If all values are the same sign avoid TwoSlopeNorm edge case
    vcenter = 0.0 if vmin < 0 < vmax else (vmin + vmax) / 2
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    fig, ax = plt.subplots(
        figsize=(max(5, len(compare_conds) * 2.5),
                 max(3, len(configs) * 1.4))
    )
    im = ax.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="auto")
    plt.colorbar(im, ax=ax, label="ΔRMSE vs A_raw (°C)  — green = better")

    ax.set_xticks(range(len(compare_conds)))
    ax.set_yticks(range(len(configs)))
    ax.set_xticklabels(
        [ABLATION_CONDITIONS.get(c, {}).get("label", c) for c in compare_conds],
        rotation=20, ha="right", fontsize=9,
    )
    ax.set_yticklabels(configs, fontsize=9)

    for i in range(len(configs)):
        for j in range(len(compare_conds)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(
                    j, i, f"{val:+.2f}°C",
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color="black",
                )

    ax.set_title(
        "RMSE improvement over raw-only baseline (A_raw)\n"
        "Positive = lower RMSE than A_raw  |  mean across seeds",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    path = out_dir / "ablation_improvement_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PMSM EWMA ablation study — multi-seed evaluation.",
    )
    # ── data ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--csv", required=True,
        help="Path to measures_v2.csv.",
    )
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac",   type=float, default=0.10)

    # ── ablation scope ─────────────────────────────────────────────────────
    parser.add_argument(
        "--conditions", nargs="+", default=DEFAULT_CONDITIONS,
        choices=list(ABLATION_CONDITIONS.keys()),
        help=(
            "Ablation conditions to run. Default: all four. "
            "Example: --conditions A_raw D_full"
        ),
    )
    parser.add_argument(
        "--configs", nargs="+", default=ALL_CONFIG_NAMES,
        help=f"PMSM configs to train. Default: all. Options: {ALL_CONFIG_NAMES}",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
        help=(
            "Random seeds. Controls both the profile split AND the Optuna sampler. "
            "Default: 0 1 2 3 4.  Example: --seeds 7 42 123"
        ),
    )

    # ── HPO / training ─────────────────────────────────────────────────────
    parser.add_argument("--n-trials",         type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--max-epochs",       type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience",         type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--retrain-epochs",   type=int, default=DEFAULT_RETRAIN_EPOCHS)
    parser.add_argument("--retrain-patience", type=int, default=DEFAULT_RETRAIN_PAT)
    parser.add_argument("--hpo-pool-size",    type=int, default=DEFAULT_HPO_POOL_SIZE)
    parser.add_argument("--device", default=None,
                        help="e.g. 'cuda' or 'cpu'. Auto-detected if omitted.")

    # ── resume ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip runs where metrics.json already exists (resume a crashed study).",
    )

    args = parser.parse_args()

    device = (
        torch.device(args.device) if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)

    # ── run plan ───────────────────────────────────────────────────────────
    total_runs = len(args.conditions) * len(args.configs) * len(args.seeds)
    print("=" * 70)
    print("  PMSM EWMA Ablation Study")
    print("=" * 70)
    print(f"  Conditions  : {args.conditions}")
    print(f"  Configs     : {args.configs}")
    print(f"  Seeds       : {args.seeds}")
    print(f"  Total runs  : {total_runs}")
    print(f"  Split       : random (per-seed, train={args.train_frac} "
          f"val={args.val_frac})")
    print(f"  Device      : {device}")
    print(f"  Output      : {ABLATION_DIR}/")
    print("=" * 70)

    # ── execute ────────────────────────────────────────────────────────────
    all_results: list[dict] = []
    run_number = 0

    for condition_key in args.conditions:
        for config_name in args.configs:
            if config_name not in ALL_CONFIG_NAMES:
                print(f"  WARNING: unknown config {config_name!r} — skipping.")
                continue
            for seed in args.seeds:
                run_number += 1
                run_tag = f"{condition_key}_{config_name}_seed{seed}"
                out_dir = ABLATION_DIR / "runs" / run_tag

                print(f"\n[{run_number}/{total_runs}]  {run_tag}")

                if args.skip_existing and (out_dir / "metrics.json").exists():
                    print("  → skipping (metrics.json exists)")
                    with open(out_dir / "metrics.json") as fh:
                        all_results.append(json.load(fh))
                    continue

                try:
                    result = run_single(
                        csv_path         = Path(args.csv),
                        condition_key    = condition_key,
                        config_name      = config_name,
                        seed             = seed,
                        n_trials         = args.n_trials,
                        max_epochs       = args.max_epochs,
                        patience         = args.patience,
                        retrain_epochs   = args.retrain_epochs,
                        retrain_patience = args.retrain_patience,
                        hpo_pool_size    = args.hpo_pool_size,
                        device           = device,
                        out_dir          = out_dir,
                        train_frac       = args.train_frac,
                        val_frac         = args.val_frac,
                    )
                    all_results.append(result)

                except Exception as exc:
                    print(f"  ERROR in {run_tag}: {exc}")
                    import traceback
                    traceback.print_exc()
                    # do not abort — continue remaining runs
                    continue

    if not all_results:
        print("No results collected — aborting.")
        sys.exit(1)

    # ── aggregate + save ───────────────────────────────────────────────────
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(ABLATION_DIR / "ablation_results.csv", index=False)
    print(f"\n  saved → {ABLATION_DIR}/ablation_results.csv  "
          f"({len(df_results)} rows)")

    summary = build_summary(df_results)
    summary.to_csv(ABLATION_DIR / "ablation_summary.csv", index=False)
    print(f"  saved → {ABLATION_DIR}/ablation_summary.csv")

    # ── plots ──────────────────────────────────────────────────────────────
    conditions_present = [
        c for c in args.conditions
        if c in df_results["condition"].values
    ]
    configs_present = [
        c for c in args.configs
        if c in df_results["config_name"].values
    ]

    plot_boxplots(df_results, ABLATION_DIR, conditions_present, configs_present)

    if "A_raw" in conditions_present:
        plot_improvement_heatmap(
            summary, ABLATION_DIR, conditions_present, configs_present
        )

    # ── console summary ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  ABLATION SUMMARY  (mean ± std across seeds)")
    print(f"{'='*70}")
    print(f"  {'Condition':<14s}  {'Config':<28s}  "
          f"{'RMSE°C':>14s}  {'MAE°C':>11s}  {'R²':>10s}  {'Params':>8s}")
    print(f"  {'-'*14}  {'-'*28}  {'-'*14}  {'-'*11}  {'-'*10}  {'-'*8}")

    for _, row in summary.sort_values(["config_name", "condition"]).iterrows():
        rmse_s  = f"{row['test_rmse_c_mean']:.2f} ± {row['test_rmse_c_std']:.2f}"
        mae_s   = f"{row['test_mae_c_mean']:.2f} ± {row['test_mae_c_std']:.2f}"
        r2_s    = f"{row['test_r2_mean']:.4f} ± {row['test_r2_std']:.4f}"
        par_s   = f"{row['n_params_mean']:.0f}"
        print(f"  {row['condition']:<14s}  {row['config_name']:<28s}  "
              f"{rmse_s:>14s}  {mae_s:>11s}  {r2_s:>10s}  {par_s:>8s}")
    print()


if __name__ == "__main__":
    main()