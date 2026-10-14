"""Architecture comparison — patent topology search vs pruning workstream.

Loads results from:
  results/baseline/{A,B}/config.json
  results/comparison/comparison_table.csv   (pruning, all methods × sparsities)
  patenting/outputs/dataset_{A,B}/metrics.json

Produces:
  results/comparison/architecture_comparison.csv   — one row per (method, dataset)
  results/comparison/architecture_comparison_A.png — params vs test_mse, dataset A
  results/comparison/architecture_comparison_B.png — params vs test_mse, dataset B

Note on scale:
  Pruning methods use PyTorch ReLU networks trained on raw (un-normalised) data.
  The patent method uses pure-NumPy tanh networks trained on min-max normalised
  data (zero-mean, unit-std per feature and target).  MSE values are therefore
  on different absolute scales for dataset B; the plot renders them in log-space
  and annotates the normalisation difference.  Dataset A is a simple 1-D
  function whose two methods happen to yield broadly comparable MSEs.

Usage:
    python src/analysis/architecture_compare.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR        = REPO_ROOT / "results" / "comparison"
PRUNING_CSV    = OUT_DIR / "comparison_table.csv"
BASELINE_DIR   = REPO_ROOT / "results" / "baseline"
PATENT_DIR     = REPO_ROOT / "patenting" / "outputs"

DATASETS       = ["A", "B"]
BASELINE_KEY   = "test_mse_locked"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_baseline(dataset: str) -> dict:
    path = BASELINE_DIR / dataset / "config.json"
    with open(path) as f:
        return json.load(f)


def load_patent(dataset: str) -> dict | None:
    path = PATENT_DIR / f"dataset_{dataset}" / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_pruning_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Build unified comparison table
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_table(datasets: list[str]) -> list[dict]:
    """One row per (method, dataset) representing the Pareto-efficient point.

    For pruning methods: pick the sparsity with the lowest test_mse (best
    achievable accuracy) — the 'champion' checkpoint.  This deliberately
    over-samples the best pruning result so the comparison is generous to
    the pruning workstream.

    For the patent method: single topology per dataset (no sparsity sweep).
    For the baseline: 0% sparsity, full network.
    """
    rows_pruning = load_pruning_csv(PRUNING_CSV)

    table: list[dict] = []

    for ds in datasets:
        baseline = load_baseline(ds)
        baseline_mse    = baseline["metrics"][BASELINE_KEY]
        baseline_params = baseline["n_params"]
        topology        = (f"{baseline['architecture']['n_layers']} hidden layers, "
                           f"{baseline['architecture']['n_neurons']} neurons/layer, "
                           f"{baseline['architecture']['activation']}")

        # ── Baseline ─────────────────────────────────────────────────────────
        table.append({
            "method":        "baseline",
            "method_type":   "baseline",
            "dataset":       ds,
            "n_params":      baseline_params,
            "test_mse":      baseline_mse,
            "mse_ratio":     1.0,
            "topology_desc": topology,
            "note":          "ReLU, Adam, HPO",
        })

        # ── Pruning — one row per method (best sparsity) ──────────────────
        method_best: dict[str, dict] = {}
        for r in rows_pruning:
            if r["dataset"] != ds:
                continue
            m = r["method"]
            mse_val = float(r["test_mse"])
            if m not in method_best or mse_val < float(method_best[m]["test_mse"]):
                method_best[m] = r

        for m, r in method_best.items():
            mse_val = float(r["test_mse"])
            n_params = int(r["param_count"])
            table.append({
                "method":        m,
                "method_type":   r.get("pruning_type", "pruning"),
                "dataset":       ds,
                "n_params":      n_params,
                "test_mse":      mse_val,
                "mse_ratio":     mse_val / baseline_mse,
                "topology_desc": f"pruned, sparsity={r['sparsity_pct']}%",
                "note":          "best sparsity checkpoint",
            })

        # ── Patent method ─────────────────────────────────────────────────
        patent = load_patent(ds)
        if patent is None:
            continue

        patent_mse    = patent["test_mse"]
        patent_params = patent["n_params"]
        layers_str    = "→".join(str(x) for x in patent["final_topology"])
        table.append({
            "method":        "patent (topology search)",
            "method_type":   "patent",
            "dataset":       ds,
            "n_params":      patent_params,
            "test_mse":      patent_mse,
            "mse_ratio":     patent_mse / baseline_mse,
            "topology_desc": f"[{layers_str}] tanh, from [1→h→1]",
            "note":          "normalised targets — see scale note",
        })

    return table


# ─────────────────────────────────────────────────────────────────────────────
# CSV output
# ─────────────────────────────────────────────────────────────────────────────

def write_csv(table: list[dict], path: Path) -> None:
    fields = ["method", "method_type", "dataset", "n_params",
              "test_mse", "mse_ratio", "topology_desc", "note"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    print(f"CSV  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

PRUNING_TYPE_STYLE = {
    "weight-removal": {"marker": "o", "color": "#5B9BD5", "alpha": 0.70, "s": 60},
    "structured":     {"marker": "s", "color": "#ED7D31", "alpha": 0.70, "s": 60},
    "low-rank":       {"marker": "D", "color": "#70AD47", "alpha": 0.70, "s": 60},
    "pruning":        {"marker": "o", "color": "#999999", "alpha": 0.60, "s": 50},
}


def plot_comparison(table: list[dict], dataset: str, out_path: Path) -> None:
    ds_rows = [r for r in table if r["dataset"] == dataset]

    fig, ax = plt.subplots(figsize=(9, 6))

    # ── Pruning points ────────────────────────────────────────────────────
    plotted_types: set = set()
    for r in ds_rows:
        if r["method_type"] in ("baseline", "patent"):
            continue
        ptype = r["method_type"]
        style = PRUNING_TYPE_STYLE.get(ptype, PRUNING_TYPE_STYLE["pruning"])
        label = ptype if ptype not in plotted_types else None
        ax.scatter(r["n_params"], r["test_mse"],
                   marker=style["marker"], color=style["color"],
                   alpha=style["alpha"], s=style["s"], label=label,
                   zorder=2)
        plotted_types.add(ptype)

    # ── Baseline ──────────────────────────────────────────────────────────
    for r in ds_rows:
        if r["method_type"] == "baseline":
            ax.scatter(r["n_params"], r["test_mse"],
                       marker="X", color="crimson", s=150, zorder=4,
                       label="baseline (ReLU, Adam)")

    # ── Patent method ─────────────────────────────────────────────────────
    for r in ds_rows:
        if r["method_type"] == "patent":
            ax.scatter(r["n_params"], r["test_mse"],
                       marker="*", color="gold", edgecolors="black",
                       linewidths=0.8, s=300, zorder=5,
                       label="patent (topology search, tanh, norm. Y)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameter count  (log scale)", fontsize=12)
    ax.set_ylabel("Test MSE  (log scale)", fontsize=12)
    ax.set_title(f"Architecture Efficiency — Dataset {dataset}\n"
                 "Each point = one (method, best-sparsity) combination",
                 fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", alpha=0.25)

    if dataset == "B":
        ax.text(0.02, 0.02,
                "Note: patent uses normalised targets;\n"
                "MSE is NOT directly comparable to pruning methods.",
                transform=ax.transAxes, fontsize=8, color="gray",
                verticalalignment="bottom")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    table = build_comparison_table(DATASETS)

    write_csv(table, OUT_DIR / "architecture_comparison.csv")

    for ds in DATASETS:
        plot_comparison(table, ds, OUT_DIR / f"architecture_comparison_{ds}.png")

    # ── Print summary ────────────────────────────────────────────────────
    print("\n=== ARCHITECTURE COMPARISON SUMMARY ===")
    header = f"{'Method':<30} {'Dataset':<8} {'Params':>8} {'Test MSE':>14} {'MSE/Baseline':>14}"
    print(header)
    print("-" * len(header))
    for r in sorted(table, key=lambda x: (x["dataset"], x["n_params"])):
        print(f"{r['method']:<30} {r['dataset']:<8} {r['n_params']:>8} "
              f"{r['test_mse']:>14.6e} {r['mse_ratio']:>14.4f}")


if __name__ == "__main__":
    main()
