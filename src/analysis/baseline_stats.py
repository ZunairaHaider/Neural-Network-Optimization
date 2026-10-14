"""
Unified box-plots: ablation conditions (A_raw, B_physics, …) +
baseline split modes (diversity, random) on the same axes.

One figure per dataset, one panel per metric.

Usage
-----
    python -m src.analysis.unified_boxplot
    python -m src.analysis.unified_boxplot \
        --results-dir  results/baseline \
        --ablation-dir results/ablation_ewma \
        --out-dir      results/stats
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_RESULTS_DIR  = REPO_ROOT / "results" / "baseline"
DEFAULT_ABLATION_DIR = REPO_ROOT / "results" / "ablation_ewma"
DEFAULT_OUT_DIR      = REPO_ROOT / "results" / "stats"

# ── Ablation condition order + labels ─────────────────────────────────────────
ABLATION_ORDER = ["A_raw", "B_physics", "C_ewma", "D_full"]
ABLATION_LABELS = {
    "A_raw":     "Raw only",
    "B_physics": "Raw + physics",
    "C_ewma":    "Raw + EWMA",
    "D_full":    "Raw + physics\n+ EWMA",
}

# ── Baseline split modes ───────────────────────────────────────────────────────
BASELINE_ORDER  = ["diversity", "random"]
BASELINE_LABELS = {
    "diversity": "Baseline\n(diversity)",
    "random":    "Baseline\n(random)",
}

# ── Colours ────────────────────────────────────────────────────────────────────
PALETTE = {
    # ablation
    "A_raw":     "#adb5bd",
    "B_physics": "#e07b54",
    "C_ewma":    "#f77f00",
    "D_full":    "#9b2dca",
    # baseline
    "diversity": "#2dc653",
    "random":    "#4895ef",
}

# ── Metrics to show (column_name, y-axis label) ────────────────────────────────
PLOT_METRICS = [
    ("test_rmse_c",     "RMSE (°C)"),
    ("test_mae_c",      "MAE (°C)"),
    ("test_r2",         "R²"),
    ("test_max_c",      "Max abs err (°C)"),
]


# ══════════════════════════════════════════════════════════════════════════════
# Scanners — produce a single unified DataFrame
# ══════════════════════════════════════════════════════════════════════════════
def _normalise_dataset_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip trailing '_ewma' (case-insensitive) from dataset names so that
    'PMSM_all_in_ewma' and 'PMSM_all_in' are treated as the same dataset.
    """
    df = df.copy()
    df["dataset"] = (
        df["dataset"]
        .str.strip()
        .str.replace(r"_ewma$", "", regex=True, case=False)
    )
    return df

def _scan_baseline(results_dir: Path) -> pd.DataFrame:
    """
    Scan results/baseline/{split_mode}_{seed}/{dataset}/config.json.

    The baseline config.json stores test metrics under metrics.test_rmse_celsius
    etc.  We rename them to the shared column names used everywhere else.
    """
    rows: list[dict] = []

    if not results_dir.exists():
        return pd.DataFrame()

    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        # folder name must be "{split_mode}_{seed}"
        parts = run_dir.name.rsplit("_", 1)
        if len(parts) != 2:
            continue
        split_mode, seed_str = parts
        if split_mode not in BASELINE_ORDER:
            continue
        try:
            seed = int(seed_str)
        except ValueError:
            continue

        for cfg_path in sorted(run_dir.rglob("config.json")):
            try:
                with open(cfg_path) as fh:
                    cfg = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue

            m    = cfg.get("metrics", {})
            arch = cfg.get("architecture", {})

            rows.append({
                "source":      "baseline",
                "group":       split_mode,          # key for colour / x-pos
                "label":       BASELINE_LABELS.get(split_mode, split_mode),
                "dataset":     cfg.get("dataset", cfg_path.parent.name),
                "seed":        seed,
                # ── shared metric names ───────────────────────────────────
                "test_rmse_c":     m.get("test_rmse_celsius"),
                "test_mae_c":      m.get("test_mae_celsius"),
                "test_max_c":      m.get("test_max_abs_celsius"),
                "test_r2":         m.get("test_r2"),
                "test_mse_norm":   m.get("test_mse_locked"),
                "n_params":        cfg.get("n_params"),
                "retrain_val_mse": m.get("retrain_val_mse"),
                "input_dim":       arch.get("input_dim"),
            })

    df = pd.DataFrame(rows)
    _cast_numeric(df)
    return df


def _scan_ablation(ablation_dir: Path) -> pd.DataFrame:
    """
    Scan results/ablation_ewma/runs/{condition}_{dataset}_seed{N}/metrics.json.

    The ablation metrics.json already uses the shared column names
    (test_rmse_c, test_mae_c, …) so no renaming is needed.
    """
    runs_dir = ablation_dir / "runs"
    rows: list[dict] = []

    if not runs_dir.exists():
        print(f"  [ablation] runs/ not found at {runs_dir} — skipping.")
        return pd.DataFrame()

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue

        try:
            with open(metrics_path) as fh:
                m = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue

        condition = m.get("condition", "")
        if condition not in ABLATION_ORDER:
            continue

        rows.append({
            "source":      "ablation",
            "group":       condition,
            "label":       ABLATION_LABELS.get(condition, condition),
            "dataset":     m.get("config_name", ""),
            "seed":        m.get("seed"),
            # ── already the shared names ──────────────────────────────────
            "test_rmse_c":     m.get("test_rmse_c"),
            "test_mae_c":      m.get("test_mae_c"),
            "test_max_c":      m.get("test_max_c"),
            "test_r2":         m.get("test_r2"),
            "test_mse_norm":   m.get("test_mse_norm"),
            "n_params":        m.get("n_params"),
            "retrain_val_mse": m.get("retrain_val_mse"),
            "input_dim":       m.get("input_dim"),
        })

    df = pd.DataFrame(rows)
    _cast_numeric(df)
    return df


def _cast_numeric(df: pd.DataFrame) -> None:
    """Cast metric columns to float in-place."""
    num_cols = [col for col, _ in PLOT_METRICS] + ["seed", "input_dim"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def load_unified(results_dir: Path, ablation_dir: Path) -> pd.DataFrame:
    """Merge baseline and ablation into one tidy DataFrame."""
    parts = []

    bl = _scan_baseline(results_dir)
    if not bl.empty:
        parts.append(bl)
        print(f"  baseline : {len(bl)} rows  "
              f"— groups {sorted(bl['group'].unique())}")

    ab = _scan_ablation(ablation_dir)
    if not ab.empty:
        parts.append(ab)
        print(f"  ablation : {len(ab)} rows  "
              f"— conditions {sorted(ab['group'].unique())}")

    if not parts:
        raise FileNotFoundError(
            "No data found.\n"
            f"  baseline  : {results_dir}\n"
            f"  ablation  : {ablation_dir / 'runs'}"
        )

    df = pd.concat(parts, ignore_index=True)

    # ── normalise dataset names BEFORE anything else ───────────────────────
    df = _normalise_dataset_names(df)

    print(f"  datasets after normalisation : {sorted(df['dataset'].unique())}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Group ordering helper
# ══════════════════════════════════════════════════════════════════════════════

def _ordered_groups(df: pd.DataFrame) -> list[tuple[str, str]]:
    """
    Return (group, label) pairs in the canonical display order:
        ablation conditions first (A_raw → D_full)
        then baseline split modes (diversity, random)

    Only includes groups that actually appear in df.
    """
    present = set(df["group"].unique())
    order: list[tuple[str, str]] = []

    for g in ABLATION_ORDER:
        if g in present:
            order.append((g, ABLATION_LABELS.get(g, g)))

    for g in BASELINE_ORDER:
        if g in present:
            order.append((g, BASELINE_LABELS.get(g, g)))

    return order


# ══════════════════════════════════════════════════════════════════════════════
# Main plot
# ══════════════════════════════════════════════════════════════════════════════

def plot_unified_per_dataset(df: pd.DataFrame, out_dir: Path) -> None:
    """
    One PNG per dataset.

    Layout
    ------
    2 rows × 3 cols grid of metric panels.
    All panels share the same x-axis groups.

        [RMSE]        [MAE]         [R²]
        [Max abs err] [# params]    [Retrain MSE]

    Groups on x-axis (left → right):
        A_raw | B_physics | C_ewma | D_full  ┆  diversity | random
        ←────────── ablation ──────────────→  ←── baseline ───────→
    """
    datasets = sorted(df["dataset"].unique())
    rng      = np.random.default_rng(0)

    # 2-row × 3-col grid — fits comfortably on screen and in memory
    N_ROWS, N_COLS = 2, 3
    assert len(PLOT_METRICS) <= N_ROWS * N_COLS, \
        "Add more grid cells if you add more metrics."

    for dataset in datasets:
        sub    = df[df["dataset"] == dataset]
        groups = _ordered_groups(sub)

        if not groups:
            print(f"  [skip] {dataset} — no groups found")
            continue

        n_groups   = len(groups)
        n_ablation = sum(1 for g, _ in groups if g in ABLATION_ORDER)
        x_pos      = np.arange(n_groups)

        # ── figure ────────────────────────────────────────────────────────
        fig, axes = plt.subplots(
            N_ROWS, N_COLS,
            figsize=(max(10, 2.2 * n_groups) , 9),   # fixed height
            constrained_layout=True,                  # replaces tight_layout
        )
        # flatten to 1-D so we can zip easily
        axes_flat = axes.flatten()

        for ax_idx, (metric_col, metric_ylabel) in enumerate(PLOT_METRICS):
            ax = axes_flat[ax_idx]

            # ── collect values in group order ──────────────────────────────
            box_data: list[np.ndarray] = []
            colors:   list[str]        = []
            for group, label in groups:
                vals = (
                    sub.loc[sub["group"] == group, metric_col]
                    .dropna().values.astype(float)
                )
                box_data.append(vals if len(vals) > 0 else np.array([np.nan]))
                colors.append(PALETTE.get(group, "#888888"))

            # ── box-plot ──────────────────────────────────────────────────
            bp = ax.boxplot(
                box_data,
                positions=x_pos,
                widths=0.52,
                patch_artist=True,
                medianprops=dict(color="black", linewidth=2.0),
                whiskerprops=dict(linewidth=1.1),
                capprops=dict(linewidth=1.1),
                flierprops=dict(
                    marker="x", markersize=4,
                    markeredgewidth=0.9, color="grey",
                ),
                zorder=2,
            )
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.72)

            # ── jittered seed dots ─────────────────────────────────────────
            for pos, (group, _), vals in zip(x_pos, groups, box_data):
                real = vals[~np.isnan(vals)]
                if len(real) == 0:
                    continue
                jitter = rng.uniform(-0.13, 0.13, len(real))
                ax.scatter(
                    np.full(len(real), pos) + jitter, real,
                    color=PALETTE.get(group, "#888"),
                    edgecolors="black",
                    s=44, zorder=5, linewidths=0.75,
                )

            # ── ablation | baseline separator ──────────────────────────────
            if 0 < n_ablation < n_groups:
                ax.axvline(
                    x=n_ablation - 0.5,
                    color="dimgray", linewidth=1.1,
                    linestyle="--", alpha=0.65, zorder=1,
                )

            # ── x-axis ticks ───────────────────────────────────────────────
            ax.set_xticks(x_pos)
            ax.set_xticklabels(
                [lbl for _, lbl in groups],
                rotation=25, ha="right", fontsize=8,
            )

            ax.set_ylabel(metric_ylabel, fontsize=9)
            ax.set_title(metric_ylabel, fontsize=9, fontweight="bold", pad=4)
            ax.grid(True, axis="y", alpha=0.25, linestyle="--", zorder=0)
            ax.set_xlim(-0.6, n_groups - 0.4)

            # ── μ ± σ text above each box ──────────────────────────────────
            y_top = ax.get_ylim()[1]
            for pos, (group, _), vals in zip(x_pos, groups, box_data):
                real = vals[~np.isnan(vals)]
                if len(real) < 1:
                    continue
                mean = np.mean(real)
                std  = np.std(real, ddof=1) if len(real) > 1 else 0.0
                ax.text(
                    pos, y_top,
                    f"μ={mean:.2g}\nσ={std:.1g}",
                    ha="center", va="bottom",
                    fontsize=6, color="dimgray",
                )

        # hide any unused grid cells (if PLOT_METRICS < N_ROWS*N_COLS)
        for ax_idx in range(len(PLOT_METRICS), len(axes_flat)):
            axes_flat[ax_idx].set_visible(False)

        # ── section labels in first row, left and right halves ────────────
        ax_first = axes_flat[0]
        if n_ablation > 0:
            ax_first.annotate(
                "◀  ablation conditions  ▶",
                xy=((n_ablation - 1) / 2 / max(n_groups - 1, 1), 1.04),
                xycoords="axes fraction",
                ha="center", va="bottom",
                fontsize=8, color="dimgray", fontstyle="italic",
            )
        n_baseline = n_groups - n_ablation
        if n_baseline > 0 and n_groups > 1:
            ax_first.annotate(
                "◀  full-EWMA baseline  ▶",
                xy=(
                    (n_ablation + (n_baseline - 1) / 2) / max(n_groups - 1, 1),
                    1.04,
                ),
                xycoords="axes fraction",
                ha="center", va="bottom",
                fontsize=8, color="dimgray", fontstyle="italic",
            )

        # ── legend ────────────────────────────────────────────────────────
        handles = [
            plt.Rectangle(
                (0, 0), 1, 1,
                facecolor=PALETTE.get(g, "#888"),
                edgecolor="black", alpha=0.80,
                label=lbl,
            )
            for g, lbl in groups
        ]
        fig.legend(
            handles=handles,
            labels=[lbl for _, lbl in groups],
            loc="lower center",
            ncol=min(n_groups, 6),
            fontsize=8,
            bbox_to_anchor=(0.5, 0.00),
            framealpha=0.9,
        )

        # ── title ─────────────────────────────────────────────────────────
        n_seeds = sub["seed"].nunique()
        fig.suptitle(
            f"{dataset}  —  unified comparison\n"
            f"{n_seeds} seeds  ·  each dot = one seed  ·  "
            "dashed line = ablation | baseline boundary",
            fontsize=11, fontweight="bold",
        )

        fname = out_dir / f"unified_boxplot_{dataset}.png"
        fig.savefig(fname, dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  saved → {fname}")

def plot_unified_per_metric(df: pd.DataFrame, out_dir: Path) -> None:
    """
    One PNG per metric.

    Layout
    ------
    1 row × n_datasets panels, shared y-axis scale optional.
    Each panel has one box per group (ablation + baseline).

    Example for RMSE:

        ┌─────────────────────────────────────────────────────────────┐
        │              RMSE (°C) — all datasets                       │
        ├──────────────────┬──────────────────┬────────────────────── │
        │  PMSM_all_in     │  PMSM_no_temps   │  PMSM_no_temps_pm    │
        │ [A][B][C][D]┆[dv][rn] │ same │ same │
        └──────────────────┴──────────────────┴────────────────────── │
    """
    datasets   = sorted(df["dataset"].unique())
    all_groups = _ordered_groups(df)          # global order across full df
    n_datasets = len(datasets)
    n_groups   = len(all_groups)
    n_ablation = sum(1 for g, _ in all_groups if g in ABLATION_ORDER)
    rng        = np.random.default_rng(0)

    x_pos = np.arange(n_groups)

    for metric_col, metric_ylabel in PLOT_METRICS:

        fig, axes = plt.subplots(
            1, n_datasets,
            figsize=(max(5, 2.4 * n_groups) * n_datasets, 5),
            sharey=False,                     # each dataset may have diff scale
        )
        if n_datasets == 1:
            axes = [axes]

        for ax, dataset in zip(axes, datasets):
            sub = df[df["dataset"] == dataset]

            # ── collect values in the global group order ───────────────────
            box_data: list[np.ndarray] = []
            colors:   list[str]        = []

            for group, label in all_groups:
                vals = (
                    sub.loc[sub["group"] == group, metric_col]
                    .dropna().values.astype(float)
                )
                box_data.append(vals if len(vals) > 0 else np.array([np.nan]))
                colors.append(PALETTE.get(group, "#888888"))

            # ── box-plot ──────────────────────────────────────────────────
            bp = ax.boxplot(
                box_data,
                positions=x_pos,
                widths=0.52,
                patch_artist=True,
                medianprops=dict(color="black", linewidth=2.0),
                whiskerprops=dict(linewidth=1.1),
                capprops=dict(linewidth=1.1),
                flierprops=dict(
                    marker="x", markersize=4,
                    markeredgewidth=0.9, color="grey",
                ),
                zorder=2,
            )
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)

            # ── jittered seed dots ─────────────────────────────────────────
            for pos, (group, _), vals in zip(x_pos, all_groups, box_data):
                real = vals[~np.isnan(vals)]
                if len(real) == 0:
                    continue
                jitter = rng.uniform(-0.13, 0.13, len(real))
                ax.scatter(
                    np.full(len(real), pos) + jitter, real,
                    color=PALETTE.get(group, "#888"),
                    edgecolors="black",
                    s=44, zorder=5, linewidths=0.75,
                )

            # ── ablation | baseline separator ──────────────────────────────
            if 0 < n_ablation < n_groups:
                ax.axvline(
                    x=n_ablation - 0.5,
                    color="dimgray", linewidth=1.1,
                    linestyle="--", alpha=0.65, zorder=1,
                )

            # ── μ ± σ below each box ───────────────────────────────────────
            for pos, (group, _), vals in zip(x_pos, all_groups, box_data):
                real = vals[~np.isnan(vals)]
                if len(real) < 1:
                    continue
                mean = np.mean(real)
                std  = np.std(real, ddof=1) if len(real) > 1 else 0.0
                ax.text(
                    pos, ax.get_ylim()[0],
                    f"μ={mean:.2g}\nσ={std:.1g}",
                    ha="center", va="top",
                    fontsize=6, color="dimgray",
                    transform=ax.get_xaxis_transform(),
                )

            # ── axes cosmetics ─────────────────────────────────────────────
            ax.set_xticks(x_pos)
            ax.set_xticklabels(
                [lbl for _, lbl in all_groups],
                rotation=25, ha="right", fontsize=8,
            )
            ax.set_xlim(-0.6, n_groups - 0.4)
            ax.set_title(dataset, fontsize=10, fontweight="bold", pad=6)
            ax.set_ylabel(metric_ylabel if ax is axes[0] else "", fontsize=9)
            ax.grid(True, axis="y", alpha=0.25, linestyle="--", zorder=0)

            # ── section labels (first dataset panel only) ──────────────────
            if ax is axes[0]:
                if n_ablation > 0:
                    ax.annotate(
                        "ablation",
                        xy=((n_ablation - 1) / 2 / max(n_groups - 1, 1), 1.04),
                        xycoords="axes fraction",
                        ha="center", va="bottom",
                        fontsize=8, color="dimgray", fontstyle="italic",
                    )
                n_baseline = n_groups - n_ablation
                if n_baseline > 0 and n_groups > 1:
                    ax.annotate(
                        "baseline",
                        xy=(
                            (n_ablation + (n_baseline - 1) / 2)
                            / max(n_groups - 1, 1),
                            1.04,
                        ),
                        xycoords="axes fraction",
                        ha="center", va="bottom",
                        fontsize=8, color="dimgray", fontstyle="italic",
                    )

        # ── shared legend ──────────────────────────────────────────────────
        handles = [
            plt.Rectangle(
                (0, 0), 1, 1,
                facecolor=PALETTE.get(g, "#888"),
                edgecolor="black", alpha=0.80,
            )
            for g, _ in all_groups
        ]
        fig.legend(
            handles=handles,
            labels=[lbl for _, lbl in all_groups],
            loc="lower center",
            ncol=min(n_groups, 6),
            fontsize=8,
            bbox_to_anchor=(0.5, 0.00),
            framealpha=0.9,
        )

        n_seeds = df["seed"].nunique()
        fig.suptitle(
            f"{metric_ylabel}  —  all datasets\n"
            f"{n_seeds} seeds  ·  each dot = one seed  ·  "
            "dashed line = ablation | baseline",
            fontsize=12, fontweight="bold",
        )

        plt.tight_layout(rect=[0, 0.08, 1, 0.95])

        # safe filename from ylabel
        safe = metric_ylabel.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace("°", "deg")
        fname = out_dir / f"metric_{safe}.png"
        fig.savefig(fname, dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  saved → {fname}")

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified box-plots: ablation conditions + baseline splits.",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
        help=f"Baseline results root. Default: {DEFAULT_RESULTS_DIR}",
    )
    parser.add_argument(
        "--ablation-dir", type=Path, default=DEFAULT_ABLATION_DIR,
        help=f"Ablation results root (contains runs/). "
             f"Default: {DEFAULT_ABLATION_DIR}",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help=f"Output directory for PNGs. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Restrict to specific dataset names. Default: all found.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading data …")
    df = load_unified(args.results_dir, args.ablation_dir)

    if args.datasets:
        df = df[df["dataset"].isin(args.datasets)]
        if df.empty:
            print(f"No data for datasets {args.datasets} — aborting.")
            sys.exit(1)

    print(f"\n  Datasets : {sorted(df['dataset'].unique())}")
    print(f"  Groups   : {sorted(df['group'].unique())}")
    print(f"  Seeds    : {sorted(df['seed'].dropna().unique())}")
    print(f"  Total    : {len(df)} rows\n")

    print("Generating unified box-plots …")
    print("Generating plots …")
    plot_unified_per_metric(df, args.out_dir) 

    print(f"\nDone. Figures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()