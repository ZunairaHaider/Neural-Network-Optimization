"""Scaling-law analysis — aggregates pruning sweeps across (method, dataset, dim).

Produces the Phase 2 deliverables (handoff §4) for the pruning workstream only.
The patent track is deferred and gets a written summary instead of figures.

Inputs (read-only)
------------------
  results/baseline/{dataset}[_{dim}d]/config.json            (per-dim baseline)
  results/pruning/{method}/{dataset}[_{dim}d]/sparsity_*.pt  (per-target checkpoints)

Outputs
-------
  results/scaling_law/
    pruning_results_all.csv       — unified long-format table
    fig_sparsity_vs_mse_{ds}_{dim}d.png
    fig_params_vs_mse_{ds}_{dim}d.png
    fig_dim_vs_best_mse_{ds}.png             ← the headline scaling-law plot
    fig_dim_vs_max_sparsity_{ds}.png         ← compressibility scaling-law plot
    fig_dim_vs_flops_at_target_{ds}.png      ← FLOPs scaling
    summary.md                     — narrative-ready summary of findings

FLOPs
-----
Per-forward FLOPs for a pruned MLP, weight-removal accounting:
  FLOPs = Σ_layers (2 * alive_weights_in_layer) + Σ_layers alive_biases
Multiply-add counted as 2 ops; bias add as 1 op. This understates structured
pruning slightly (a dropped neuron's column in the next layer is also zero,
which is captured because those weights are masked too).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.models.mlp import MLP  # noqa: E402

# ── config ────────────────────────────────────────────────────────────────────

METHODS = [
    "obd", "obs", "mag", "struct", "activ",
    "redundancy", "lowrank",
    "taylor_fo", "snip", "imp_rewind",
]

DATASETS = ["A", "B"]
DIMS     = [1, 2, 3]

METHOD_STYLES = {
    "obd":        {"linestyle": "-",  "marker": "o", "color": "steelblue"},
    "obs":        {"linestyle": "--", "marker": "s", "color": "darkorange"},
    "mag":        {"linestyle": ":",  "marker": "^", "color": "seagreen"},
    "struct":     {"linestyle": "-.", "marker": "D", "color": "crimson"},
    "activ":      {"linestyle": (0, (3, 1, 1, 1)), "marker": "P", "color": "purple"},
    "redundancy": {"linestyle": "-",  "marker": "v", "color": "teal"},
    "lowrank":    {"linestyle": "--", "marker": "X", "color": "goldenrod"},
    "taylor_fo":  {"linestyle": ":",  "marker": "<", "color": "indianred"},
    "snip":       {"linestyle": "-.", "marker": ">", "color": "slategrey"},
    "imp_rewind": {"linestyle": (0, (5, 2)), "marker": "p", "color": "darkgreen"},
}

OUT_DIR = REPO_ROOT / "results" / "scaling_law"


# ── helpers ───────────────────────────────────────────────────────────────────

def _dataset_tag(dataset: str, dim: int) -> str:
    return dataset if dim == 1 else f"{dataset}_{dim}d"


def _baseline_dir(dataset: str, dim: int) -> Path:
    return REPO_ROOT / "results" / "baseline" / _dataset_tag(dataset, dim)


def _pruning_dir(method: str, dataset: str, dim: int) -> Path:
    return REPO_ROOT / "results" / "pruning" / method / _dataset_tag(dataset, dim)


def _load_baseline(dataset: str, dim: int) -> dict | None:
    cfg_path = _baseline_dir(dataset, dim) / "config.json"
    if not cfg_path.exists():
        return None
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    arch = cfg["architecture"]
    model = MLP(
        n_layers=arch["n_layers"], n_neurons=arch["n_neurons"],
        activation=arch["activation"],
        input_dim=arch.get("input_dim", 1),
        output_dim=arch.get("output_dim", 1),
    )
    n_params = sum(p.numel() for p in model.parameters())
    flops = _model_flops_dense(model)
    return {
        "n_params": n_params,
        "flops": flops,
        "test_mse": cfg["metrics"]["test_mse_locked"],
        "arch": arch,
    }


def _model_flops_dense(model: MLP) -> int:
    """Dense forward FLOPs: 2 * weights + biases per Linear layer."""
    total = 0
    for layer in model.layers:
        total += 2 * layer.weight.numel() + layer.bias.numel()
    return total


def _checkpoint_flops(state_dict: dict) -> int:
    """FLOPs after pruning: 2 * nonzero weights + (assumed-full) biases."""
    total = 0
    for k, v in state_dict.items():
        if k.endswith(".weight"):
            total += 2 * int((v != 0).sum().item())
        elif k.endswith(".bias"):
            total += v.numel()
    return total


def _load_checkpoints(method: str, dataset: str, dim: int) -> list[dict]:
    base = _pruning_dir(method, dataset, dim)
    if not base.exists():
        return []
    rows = []
    for pt_path in sorted(base.glob("sparsity_*.pt")):
        if "_config" in pt_path.stem:
            continue
        blob = torch.load(pt_path, map_location="cpu", weights_only=False)
        rows.append(blob)
    rows.sort(key=lambda r: r["sparsity"])
    return rows


# ── build the unified table ───────────────────────────────────────────────────

def build_long_table() -> list[dict]:
    rows = []
    for ds in DATASETS:
        for dim in DIMS:
            base = _load_baseline(ds, dim)
            if base is None:
                continue
            # Baseline row at sparsity 0
            rows.append({
                "method": "baseline",
                "dataset": ds,
                "dim": dim,
                "sparsity_pct": 0.0,
                "test_mse": base["test_mse"],
                "test_mse_ratio": 1.0,
                "param_count": base["n_params"],
                "param_ratio": 1.0,
                "flops": base["flops"],
                "flops_ratio": 1.0,
                "inference_ms": "",
            })
            for method in METHODS:
                for ckpt in _load_checkpoints(method, ds, dim):
                    flops = _checkpoint_flops(ckpt["state_dict"])
                    rows.append({
                        "method": method,
                        "dataset": ds,
                        "dim": dim,
                        "sparsity_pct": round(ckpt["sparsity"] * 100, 1),
                        "test_mse": ckpt["test_mse"],
                        "test_mse_ratio": ckpt["test_mse"] / base["test_mse"]
                                          if base["test_mse"] else float("nan"),
                        "param_count": ckpt["param_count"],
                        "param_ratio": ckpt["param_count"] / base["n_params"]
                                        if base["n_params"] else float("nan"),
                        "flops": flops,
                        "flops_ratio": flops / base["flops"] if base["flops"] else float("nan"),
                        "inference_ms": round(float(ckpt.get("inference_ms", 0.0)), 4),
                    })
    return rows


# ── plots ─────────────────────────────────────────────────────────────────────

def _style(method: str) -> dict:
    return METHOD_STYLES.get(method, {"linestyle": "-", "marker": "o", "color": "gray"})


def plot_sparsity_vs_mse(rows: list[dict], ds: str, dim: int) -> Path:
    sub = [r for r in rows if r["dataset"] == ds and r["dim"] == dim]
    base = next((r for r in sub if r["method"] == "baseline"), None)
    fig, ax = plt.subplots(figsize=(9, 5))
    if base:
        ax.scatter([0], [base["test_mse"]], marker="*", s=200, color="black",
                   zorder=5, label="Baseline (dense)")
    for m in METHODS:
        ms = [r for r in sub if r["method"] == m]
        if not ms:
            continue
        s = _style(m)
        xs = [r["sparsity_pct"] for r in ms]
        ys = [r["test_mse"] for r in ms]
        ax.plot(xs, ys, label=m, **s, linewidth=1.6, markersize=6)
    ax.set_xlabel("Sparsity (%)")
    ax.set_ylabel("Test MSE")
    ax.set_yscale("log")
    ax.set_title(f"Sparsity vs test MSE — Dataset {ds}, d={dim}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    out = OUT_DIR / f"fig_sparsity_vs_mse_{ds}_{dim}d.png"
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return out


def plot_params_vs_mse(rows: list[dict], ds: str, dim: int) -> Path:
    sub = [r for r in rows if r["dataset"] == ds and r["dim"] == dim]
    base = next((r for r in sub if r["method"] == "baseline"), None)
    fig, ax = plt.subplots(figsize=(9, 5))
    if base:
        ax.scatter([base["param_count"]], [base["test_mse"]], marker="*", s=200,
                   color="black", zorder=5, label="Baseline (dense)")
    for m in METHODS:
        ms = [r for r in sub if r["method"] == m]
        if not ms:
            continue
        s = _style(m)
        xs = [r["param_count"] for r in ms]
        ys = [r["test_mse"] for r in ms]
        ax.plot(xs, ys, label=m, **s, linewidth=1.6, markersize=6)
    ax.set_xlabel("Surviving params")
    ax.set_ylabel("Test MSE")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"Params vs test MSE — Dataset {ds}, d={dim}")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2)
    out = OUT_DIR / f"fig_params_vs_mse_{ds}_{dim}d.png"
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return out


def plot_dim_scaling(rows: list[dict], ds: str, target_sparsity: float = 0.5) -> Path:
    """Headline scaling-law plot: at a fixed sparsity target, how does
    test_mse_ratio (test MSE relative to dense baseline at the same dim)
    change with input dim, per method?"""
    fig, ax = plt.subplots(figsize=(9, 5))
    for m in METHODS:
        xs, ys = [], []
        for dim in DIMS:
            sub = [r for r in rows if r["dataset"] == ds and r["dim"] == dim
                   and r["method"] == m]
            if not sub:
                continue
            # Pick the row closest to target_sparsity that's <= target_sparsity
            below = [r for r in sub if r["sparsity_pct"] <= target_sparsity * 100 + 1e-6]
            if not below:
                continue
            r = max(below, key=lambda r: r["sparsity_pct"])
            xs.append(dim); ys.append(r["test_mse_ratio"])
        if not xs:
            continue
        s = _style(m)
        ax.plot(xs, ys, label=m, **s, linewidth=1.8, markersize=8)
    ax.set_xlabel("Input dimension d")
    ax.set_ylabel(f"Test MSE / dense baseline MSE  (at ≤ {target_sparsity*100:.0f}% sparsity)")
    ax.set_xticks(DIMS)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_title(f"Dimension scaling — Dataset {ds}, sparsity ≤ {target_sparsity*100:.0f}%")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    out = OUT_DIR / f"fig_dim_vs_best_mse_{ds}.png"
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return out


def plot_dim_max_sparsity(rows: list[dict], ds: str, mse_budget: float = 2.0) -> Path:
    """Per method, the highest sparsity where test_mse_ratio <= mse_budget."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for m in METHODS:
        xs, ys = [], []
        for dim in DIMS:
            sub = [r for r in rows if r["dataset"] == ds and r["dim"] == dim
                   and r["method"] == m and r["test_mse_ratio"] <= mse_budget]
            if not sub:
                continue
            best = max(sub, key=lambda r: r["sparsity_pct"])
            xs.append(dim); ys.append(best["sparsity_pct"])
        if not xs:
            continue
        s = _style(m)
        ax.plot(xs, ys, label=m, **s, linewidth=1.8, markersize=8)
    ax.set_xlabel("Input dimension d")
    ax.set_ylabel(f"Max sparsity (%) with test MSE ≤ {mse_budget:.1f}× baseline")
    ax.set_xticks(DIMS)
    ax.set_title(f"Compressibility scaling — Dataset {ds}, budget = {mse_budget:.1f}× MSE")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    out = OUT_DIR / f"fig_dim_vs_max_sparsity_{ds}.png"
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return out


def plot_dim_flops(rows: list[dict], ds: str, mse_budget: float = 2.0) -> Path:
    """Per method, FLOPs of the smallest network meeting the MSE budget, vs dim."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for m in METHODS:
        xs, ys = [], []
        for dim in DIMS:
            sub = [r for r in rows if r["dataset"] == ds and r["dim"] == dim
                   and r["method"] == m and r["test_mse_ratio"] <= mse_budget]
            if not sub:
                continue
            best = min(sub, key=lambda r: r["flops"])
            xs.append(dim); ys.append(best["flops"])
        if not xs:
            continue
        s = _style(m)
        ax.plot(xs, ys, label=m, **s, linewidth=1.8, markersize=8)
    # Dense baselines for reference
    base_x, base_y = [], []
    for dim in DIMS:
        b = next((r for r in rows if r["dataset"] == ds and r["dim"] == dim
                  and r["method"] == "baseline"), None)
        if b:
            base_x.append(dim); base_y.append(b["flops"])
    if base_x:
        ax.plot(base_x, base_y, "k:*", label="dense baseline", markersize=10, linewidth=1.5)
    ax.set_xlabel("Input dimension d")
    ax.set_ylabel(f"Forward FLOPs of smallest model with test MSE ≤ {mse_budget:.1f}× baseline")
    ax.set_xticks(DIMS); ax.set_yscale("log")
    ax.set_title(f"FLOPs scaling — Dataset {ds}, budget = {mse_budget:.1f}× MSE")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2)
    out = OUT_DIR / f"fig_dim_vs_flops_at_target_{ds}.png"
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return out


# ── summary text ──────────────────────────────────────────────────────────────

def write_summary(rows: list[dict]) -> Path:
    """Markdown summary suitable as the seed of the narrative deliverable."""
    out = OUT_DIR / "summary.md"
    lines: list[str] = []
    lines.append("# Pruning scaling-law summary\n")
    lines.append("Automatically generated from `results/pruning/**` checkpoints. "
                 "Re-run via `python src/analysis/scaling_law.py`.\n")

    lines.append("## Baseline architectures\n")
    lines.append("| Dataset | dim | n_layers × n_neurons | activation | params | FLOPs | test MSE |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        if r["method"] != "baseline":
            continue
        cfg_path = _baseline_dir(r["dataset"], r["dim"]) / "config.json"
        a = json.load(open(cfg_path))["architecture"]
        lines.append(f"| {r['dataset']} | {r['dim']} | "
                     f"{a['n_layers']}×{a['n_neurons']} | {a['activation']} | "
                     f"{r['param_count']} | {r['flops']} | {r['test_mse']:.4e} |")
    lines.append("")

    for budget in (1.5, 2.0, 3.0):
        lines.append(f"## Max sparsity at test MSE ≤ {budget:.1f}× baseline\n")
        lines.append("Best surviving sparsity per (method, dataset, dim). Empty = the "
                     "method never satisfied the MSE budget.\n")
        for ds in DATASETS:
            lines.append(f"### Dataset {ds}\n")
            lines.append("| method | d=1 | d=2 | d=3 |")
            lines.append("|---|---|---|---|")
            for m in METHODS:
                cells = []
                for dim in DIMS:
                    sub = [r for r in rows if r["dataset"] == ds and r["dim"] == dim
                           and r["method"] == m and r["test_mse_ratio"] <= budget]
                    if not sub:
                        cells.append("—")
                    else:
                        best = max(sub, key=lambda r: r["sparsity_pct"])
                        cells.append(f"{best['sparsity_pct']:.0f}%")
                lines.append(f"| {m} | " + " | ".join(cells) + " |")
            lines.append("")
    out.write_text("\n".join(lines))
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_long_table()
    if not rows:
        print("No results found. Run the pruning sweep first.")
        return

    # Write unified CSV
    csv_path = OUT_DIR / "pruning_results_all.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  CSV → {csv_path}  ({len(rows)} rows)")

    # Per-cell plots
    for ds in DATASETS:
        for dim in DIMS:
            if not any(r["dataset"] == ds and r["dim"] == dim and r["method"] != "baseline"
                       for r in rows):
                continue
            p = plot_sparsity_vs_mse(rows, ds, dim);  print(f"  → {p.name}")
            p = plot_params_vs_mse(rows, ds, dim);    print(f"  → {p.name}")

    # Dim-scaling plots
    for ds in DATASETS:
        if not any(r["dataset"] == ds and r["dim"] > 1 and r["method"] != "baseline"
                   for r in rows):
            continue
        p = plot_dim_scaling(rows, ds);       print(f"  → {p.name}")
        p = plot_dim_max_sparsity(rows, ds);  print(f"  → {p.name}")
        p = plot_dim_flops(rows, ds);         print(f"  → {p.name}")

    p = write_summary(rows); print(f"  summary → {p}")
    print(f"\nDone. All outputs → {OUT_DIR}/")


if __name__ == "__main__":
    main()
