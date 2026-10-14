"""Comparison framework — aggregates pruning results across ALL methods and datasets.

Usage
-----
    python src/analysis/compare.py

Outputs  (results/comparison/)
--------------------------------
  sparsity_vs_mse_{dataset}.png
  sparsity_vs_params_{dataset}.png
  memory_vs_mse_{dataset}.png
  comparison_table.csv
  comparison_table_rich.txt
  summary_statistics.txt
  architecture_summary.txt

11 Comparison metrics
----------------------
  method            : pruning algorithm
  pruning_type      : weight-removal vs structured
  sparsity_pct      : % of weights zeroed / neurons removed
  test_mse          : test MSE after fine-tuning
  mse_ratio         : test_mse / baseline_mse  (1.00 = no degradation)
  param_count       : surviving non-zero parameters
  compression_ratio : baseline_params / param_count  (e.g. 2.23x)
  model_size_kb     : param_count * 4 bytes / 1024
  memory_saved_pct  : % of baseline memory recovered
  inference_ms      : median inference latency over 100 runs (ms)
  training_time_s   : total pruning + fine-tuning wall-clock time (seconds)
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.models.mlp import MLP

OUT_DIR  = REPO_ROOT / "results" / "comparison"
DATASETS = {
    "A": "Clean quadratic",
    "B": "Noisy quadratic",
}

METHODS = [
    "obd", "obs", "mag", "struct", "activ",
    "redundancy", "lowrank",
    "taylor_fo", "snip", "imp_rewind",
]

PRUNING_TYPE = {
    "obd":        "weight-removal",
    "obs":        "weight-removal",
    "mag":        "weight-removal",
    "struct":     "structured",
    "activ":      "structured",
    "redundancy": "structured",
    "lowrank":    "low-rank",
    "taylor_fo":  "weight-removal",
    "snip":       "weight-removal",
    "imp_rewind": "weight-removal",
}

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

BYTES_PER_PARAM = 4


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_baseline_config(dataset: str) -> dict | None:
    path = REPO_ROOT / "results" / "baseline" / dataset / "config.json"
    if not path.exists():
        return None
    with open(path) as fh:
        raw = json.load(fh)
    arch    = raw.get("architecture", raw)
    hparams = raw.get("training_hparams", raw)
    return {
        "n_layers":   arch["n_layers"],
        "n_neurons":  arch["n_neurons"],
        "activation": arch["activation"],
        "lr":         hparams["lr"],
        "batch_size": 256,
        "test_mse":     raw.get("metrics", {}).get("test_mse_locked") or raw.get("test_mse"),
        "inference_ms": raw.get("inference_ms"),
    }


def _baseline_param_count(dataset: str) -> int | None:
    cfg = _load_baseline_config(dataset)
    if cfg is None:
        return None
    model = MLP(
        n_layers=cfg["n_layers"],
        n_neurons=cfg["n_neurons"],
        activation=cfg["activation"],
    )
    return sum(p.numel() for p in model.parameters())


def _load_pruning_checkpoints(method: str, dataset: str) -> list[dict]:
    base = REPO_ROOT / "results" / "pruning" / method / dataset
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


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_sparsity_vs_mse(dataset: str, label: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    cfg = _load_baseline_config(dataset)
    baseline_mse = cfg.get("test_mse") if cfg else None
    if baseline_mse:
        ax.scatter([0], [baseline_mse], marker="*", s=200, color="black",
                   zorder=5, label="Baseline (dense)")
    all_mses = [baseline_mse] if baseline_mse else []
    for method in METHODS:
        rows = _load_pruning_checkpoints(method, dataset)
        if not rows:
            continue
        xs = [r["sparsity"] * 100 for r in rows]
        ys = [r["test_mse"] for r in rows]
        all_mses.extend(ys)
        s = METHOD_STYLES[method]
        ax.plot(xs, ys, linestyle=s["linestyle"], marker=s["marker"],
                color=s["color"], linewidth=1.8, markersize=6,
                label=f"{method.upper()} ({PRUNING_TYPE[method]})")
    valid = [v for v in all_mses if v and v > 0]
    if valid and max(valid) / min(valid) > 10:
        ax.set_yscale("log")
    ax.set_xlabel("Sparsity (%)"); ax.set_ylabel("Test MSE")
    ax.set_title(f"Sparsity vs Test MSE — Dataset {dataset} ({label})")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); fig.tight_layout()
    out = OUT_DIR / f"sparsity_vs_mse_{dataset}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved -> {out}")


def plot_sparsity_vs_params(dataset: str, label: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for method in METHODS:
        rows = _load_pruning_checkpoints(method, dataset)
        if not rows:
            continue
        s = METHOD_STYLES[method]
        ax.plot([r["sparsity"]*100 for r in rows],
                [r["param_count"] for r in rows],
                linestyle=s["linestyle"], marker=s["marker"],
                color=s["color"], linewidth=1.8, markersize=6,
                label=f"{method.upper()} ({PRUNING_TYPE[method]})")
    ax.set_xlabel("Sparsity (%)"); ax.set_ylabel("Non-zero parameters")
    ax.set_title(f"Sparsity vs Param count — Dataset {dataset} ({label})")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); fig.tight_layout()
    out = OUT_DIR / f"sparsity_vs_params_{dataset}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved -> {out}")


def plot_memory_vs_mse(dataset: str, label: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    baseline_params = _baseline_param_count(dataset)
    if not baseline_params:
        return
    for method in METHODS:
        rows = _load_pruning_checkpoints(method, dataset)
        if not rows:
            continue
        s = METHOD_STYLES[method]
        xs = [(baseline_params - r["param_count"]) * BYTES_PER_PARAM / 1024
              for r in rows]
        ys = [r["test_mse"] for r in rows]
        ax.scatter(xs, ys, color=s["color"], marker=s["marker"],
                   s=60, label=method.upper(), zorder=4)
        ax.plot(xs, ys, color=s["color"], linestyle=":", alpha=0.4)
    ax.set_xlabel("Memory saved (KB) vs baseline"); ax.set_ylabel("Test MSE")
    ax.set_title(f"Memory Saved vs Test MSE — Dataset {dataset} ({label})")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); fig.tight_layout()
    out = OUT_DIR / f"memory_vs_mse_{dataset}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved -> {out}")


# ── comparison table (11 metrics) ────────────────────────────────────────────

def write_comparison_table() -> None:
    all_rows: list[dict] = []

    for dataset in DATASETS:
        baseline_params = _baseline_param_count(dataset)
        cfg             = _load_baseline_config(dataset)
        baseline_mse    = cfg.get("test_mse") if cfg else None
        if not baseline_params:
            print(f"  [skip] No baseline for dataset {dataset}.")
            continue

        baseline_size_kb = baseline_params * BYTES_PER_PARAM / 1024

        for method in METHODS:
            for ck in _load_pruning_checkpoints(method, dataset):
                pc           = ck["param_count"]
                mse          = ck["test_mse"]
                infer_ms     = ck.get("inference_ms", 0)
                training_s   = ck.get("training_time_s")   # may be None for old checkpoints
                sparsity_pct = round(ck["sparsity"] * 100, 1)

                compression     = (baseline_params / pc) if pc > 0 else float("inf")
                model_size_kb   = pc * BYTES_PER_PARAM / 1024
                memory_saved_kb = (baseline_params - pc) * BYTES_PER_PARAM / 1024
                mem_saved_pct   = memory_saved_kb / baseline_size_kb * 100
                mse_ratio       = (mse / baseline_mse) if baseline_mse else None

                all_rows.append({
                    "method":           method,
                    "pruning_type":     PRUNING_TYPE[method],
                    "dataset":          dataset,
                    "sparsity_pct":     sparsity_pct,
                    "test_mse":         round(mse, 8),
                    "mse_ratio":        round(mse_ratio, 4) if mse_ratio else "",
                    "param_count":      pc,
                    "compression_ratio":round(compression, 2),
                    "model_size_kb":    round(model_size_kb, 3),
                    "memory_saved_pct": round(mem_saved_pct, 1),
                    "inference_ms":     round(infer_ms, 4),
                    "training_time_s":  round(training_s, 2) if training_s else "N/A",
                })

    if not all_rows:
        print("  No pruning results found — run pruner.py first.")
        return

    # CSV
    csv_path = OUT_DIR / "comparison_table.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n  CSV saved -> {csv_path}")

    # Rich text table
    lines = []
    lines.append("")
    lines.append("=" * 125)
    lines.append("PRUNING COMPARISON TABLE  —  11 Metrics")
    lines.append("=" * 125)

    header = (
        f"{'Method':>7} | {'Type':<15} | {'Dataset':>7} | {'Sparsity':>9} | "
        f"{'Test MSE':>12} | {'MSE/Base':>9} | {'Params':>7} | "
        f"{'Compress':>9} | {'Size KB':>8} | {'MemSaved%':>10} | "
        f"{'Infer ms':>9} | {'Train s':>8}"
    )
    sep = "-" * len(header)
    lines.append(header)
    lines.append(sep)

    prev_dataset = None
    for r in all_rows:
        if prev_dataset and r["dataset"] != prev_dataset:
            lines.append(sep)
        prev_dataset = r["dataset"]

        mse_ratio_s = f"{r['mse_ratio']:.3f}x" if r["mse_ratio"] != "" else "   -   "
        flag = " *" if r["mse_ratio"] != "" and float(r["mse_ratio"]) > 1.5 else "  "

        lines.append(
            f"{r['method']:>7} | {r['pruning_type']:<15} | {r['dataset']:>7} | "
            f"{r['sparsity_pct']:>8.1f}% | {r['test_mse']:>12.4e} | "
            f"{mse_ratio_s:>9} | {r['param_count']:>7d} | "
            f"{r['compression_ratio']:>8.2f}x | {r['model_size_kb']:>8.3f} | "
            f"{r['memory_saved_pct']:>9.1f}% | {r['inference_ms']:>9.3f} | "
            f"{str(r['training_time_s']):>8}{flag}"
        )

    lines.append(sep)
    lines.append("* MSE degraded > 1.5x baseline")
    lines.append("")
    lines.append("METRIC GLOSSARY")
    lines.append("-" * 70)
    lines.append("  Method        : OBD/OBS/MAG = weight-removal  |  STRUCT/ACTIV = structured (whole neurons)")
    lines.append("  Sparsity      : % of total parameters zeroed or removed")
    lines.append("  Test MSE      : Mean squared error on test set after fine-tuning")
    lines.append("  MSE/Base      : test_mse / baseline_mse  (1.00 = no accuracy loss)")
    lines.append("  Params        : Surviving non-zero parameter count")
    lines.append("  Compress      : baseline_params / param_count  (higher = more compressed)")
    lines.append("  Size KB       : model footprint in KB  (float32: param_count * 4 / 1024)")
    lines.append("  MemSaved%     : % of baseline memory recovered after pruning")
    lines.append("  Infer ms      : Median inference latency over 100 runs on full test set")
    lines.append("  Train s       : Total pruning + fine-tuning wall-clock time in seconds")
    lines.append("=" * 125)

    rich_text = "\n".join(lines)
    print(rich_text)
    txt_path = OUT_DIR / "comparison_table_rich.txt"
    with open(txt_path, "w") as fh:
        fh.write(rich_text + "\n")
    print(f"\n  Rich table saved -> {txt_path}")


# ── best checkpoint summary ───────────────────────────────────────────────────

def write_summary_statistics() -> None:
    lines = ["\nBEST CHECKPOINT SUMMARY (lowest MSE per method per dataset)"]
    lines.append("=" * 85)
    lines.append(f"{'Method':<8} {'Type':<15} {'Dataset':<8} {'BestSpars%':>10} "
                 f"{'BestMSE':>12} {'MemSaved%':>10} {'SizeKB':>8} {'CompressX':>10}")
    lines.append("-" * 85)

    for dataset in DATASETS:
        baseline_params = _baseline_param_count(dataset)
        cfg             = _load_baseline_config(dataset)
        baseline_mse    = cfg.get("test_mse") if cfg else None
        if not baseline_params:
            continue
        baseline_size_kb = baseline_params * BYTES_PER_PARAM / 1024

        for method in METHODS:
            rows = _load_pruning_checkpoints(method, dataset)
            if not rows:
                continue
            best = min(rows, key=lambda r: r["test_mse"])
            pc           = best["param_count"]
            mem_saved_pct = (baseline_params - pc) * BYTES_PER_PARAM / 1024 / baseline_size_kb * 100
            size_kb      = pc * BYTES_PER_PARAM / 1024
            compress     = baseline_params / pc if pc > 0 else float("inf")
            lines.append(
                f"{method.upper():<8} {PRUNING_TYPE[method]:<15} {dataset:<8} "
                f"{best['sparsity']*100:>9.1f}% {best['test_mse']:>12.4e} "
                f"{mem_saved_pct:>9.1f}% {size_kb:>7.3f} {compress:>9.2f}x"
            )

    lines.append("=" * 85)
    summary = "\n".join(lines)
    print(summary)
    txt_path = OUT_DIR / "summary_statistics.txt"
    with open(txt_path, "w") as fh:
        fh.write(summary + "\n")
    print(f"\n  Summary saved -> {txt_path}")


# ── architecture summary ──────────────────────────────────────────────────────

def write_architecture_summary() -> None:
    lines = []
    for dataset in DATASETS:
        for method in METHODS:
            for ck in _load_pruning_checkpoints(method, dataset):
                sparsity_pct = round(ck["sparsity"] * 100, 1)
                lines.append(
                    f"\n[{method.upper()}] ({PRUNING_TYPE[method]}) "
                    f"Dataset {dataset} -- {sparsity_pct:.1f}% sparsity"
                )
                cfg  = ck.get("config", {})
                mask = ck["mask"]
                sd   = ck["state_dict"]
                model = MLP(
                    n_layers=cfg.get("n_layers", 1),
                    n_neurons=cfg.get("n_neurons", 64),
                    activation=cfg.get("activation", "relu"),
                )
                model.load_state_dict(sd)
                offset = 0
                for i, layer in enumerate(model.layers):
                    w_size = layer.weight.numel()
                    b_size = layer.bias.numel()
                    layer_size = w_size + b_size
                    alive = int(mask[offset: offset + layer_size].sum().item())
                    lines.append(
                        f"  layer {i:2d} ({layer.in_features}->{layer.out_features}): "
                        f"{alive:5d} / {layer_size:5d} non-zero  ({alive/layer_size*100:.1f}%)"
                    )
                    offset += layer_size
    out = OUT_DIR / "architecture_summary.txt"
    with open(out, "w") as fh:
        fh.write("\n".join(lines))
    print(f"  Saved -> {out}")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating sparsity-vs-MSE plots...")
    for dataset, label in DATASETS.items():
        plot_sparsity_vs_mse(dataset, label)

    print("\nGenerating sparsity-vs-params plots...")
    for dataset, label in DATASETS.items():
        plot_sparsity_vs_params(dataset, label)

    print("\nGenerating memory-saved-vs-MSE plots...")
    for dataset, label in DATASETS.items():
        plot_memory_vs_mse(dataset, label)

    print("\nBuilding comparison table (11 metrics)...")
    write_comparison_table()

    print("\nBuilding best-checkpoint summary...")
    write_summary_statistics()

    print("\nBuilding architecture summary...")
    write_architecture_summary()

    print("\nAll outputs -> results/comparison/")


if __name__ == "__main__":
    main()
