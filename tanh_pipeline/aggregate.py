"""Aggregate tanh-only baseline + pruning results into a comparison table.

Reads:
  tanh_pipeline/results/baseline/<tag>/config.json
  tanh_pipeline/results/pruning/<method>/<tag>/sparsity_*_config.json

Writes:
  tanh_pipeline/results/comparison/comparison.csv
  tanh_pipeline/results/comparison/summary.txt

Two per-method picks are reported:
  - best_mse : checkpoint with the lowest test MSE (matches the original
    architecture_comparison.csv selection — generous on accuracy).
  - best_compression : the most-pruned checkpoint whose test MSE stays within
    1.5x baseline (the parsimony story — fewest params at acceptable accuracy).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

OUT_ROOT = Path(__file__).resolve().parents[0]
BASE_DIR = OUT_ROOT / "results" / "baseline"
PRUNE_DIR = OUT_ROOT / "results" / "pruning"
COMP_DIR = OUT_ROOT / "results" / "comparison"

MSE_BLOWUP_FACTOR = 1.5

DATASET_TAGS = ["A", "B", "A_2d", "A_3d", "B_2d", "B_3d", "D"]
METHODS = ["obd", "obs", "mag", "struct", "activ", "redundancy", "lowrank",
           "taylor_fo", "snip", "imp_rewind"]


def load_baseline(tag: str) -> dict | None:
    p = BASE_DIR / tag / "config.json"
    if not p.exists():
        return None
    with open(p) as fh:
        return json.load(fh)


def load_method_checkpoints(method: str, tag: str) -> list[dict]:
    d = PRUNE_DIR / method / tag
    if not d.exists():
        return []
    rows = []
    for cfg in sorted(d.glob("sparsity_*_config.json")):
        with open(cfg) as fh:
            rows.append(json.load(fh))
    return rows


def main() -> None:
    COMP_DIR.mkdir(parents=True, exist_ok=True)
    table: list[dict] = []

    for tag in DATASET_TAGS:
        base = load_baseline(tag)
        if base is None:
            continue
        baseline_mse = float(base["metrics"]["test_mse_locked"])
        arch = base["architecture"]
        baseline_params = int(base["n_params"])
        table.append({
            "dataset": tag,
            "method": "baseline",
            "method_type": "baseline",
            "n_params": baseline_params,
            "test_mse": baseline_mse,
            "mse_ratio": 1.0,
            "sparsity_pct": 0.0,
            "topology_desc": f"{arch['n_layers']}x{arch['n_neurons']} {arch['activation']}",
            "note": "tanh HPO baseline",
        })

        for method in METHODS:
            ckpts = load_method_checkpoints(method, tag)
            if not ckpts:
                continue

            best_mse = min(ckpts, key=lambda r: float(r["test_mse"]))
            table.append({
                "dataset": tag,
                "method": method,
                "method_type": "pruning",
                "n_params": int(best_mse["param_count"]),
                "test_mse": float(best_mse["test_mse"]),
                "mse_ratio": float(best_mse["test_mse"]) / baseline_mse,
                "sparsity_pct": float(best_mse["sparsity_pct"]),
                "topology_desc": f"pruned, sparsity={best_mse['sparsity_pct']}%",
                "note": "best test MSE",
            })

            acceptable = [r for r in ckpts
                          if float(r["test_mse"]) <= MSE_BLOWUP_FACTOR * baseline_mse]
            if acceptable:
                best_comp = min(acceptable, key=lambda r: int(r["param_count"]))
                table.append({
                    "dataset": tag,
                    "method": method,
                    "method_type": "pruning",
                    "n_params": int(best_comp["param_count"]),
                    "test_mse": float(best_comp["test_mse"]),
                    "mse_ratio": float(best_comp["test_mse"]) / baseline_mse,
                    "sparsity_pct": float(best_comp["sparsity_pct"]),
                    "topology_desc": f"pruned, sparsity={best_comp['sparsity_pct']}%",
                    "note": "best compression (<=1.5x baseline MSE)",
                })

    fields = ["dataset", "method", "method_type", "n_params", "test_mse",
              "mse_ratio", "sparsity_pct", "topology_desc", "note"]
    csv_path = COMP_DIR / "comparison.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(table)

    lines = ["TANH-ONLY PIPELINE — comparison summary", "=" * 60, ""]
    for tag in DATASET_TAGS:
        rows = [r for r in table if r["dataset"] == tag]
        if not rows:
            continue
        base_row = next((r for r in rows if r["method"] == "baseline"), None)
        lines.append(f"Dataset {tag}")
        if base_row:
            lines.append(f"  baseline: {base_row['n_params']:>5d} params  "
                         f"MSE={base_row['test_mse']:.4e}  ({base_row['topology_desc']})")
        comp_rows = [r for r in rows if r["note"].startswith("best compression")]
        comp_rows.sort(key=lambda r: r["n_params"])
        for r in comp_rows:
            lines.append(f"    {r['method']:>11s}: {r['n_params']:>5d} params  "
                         f"MSE={r['test_mse']:.4e}  ({r['mse_ratio']:.2f}x)  "
                         f"@ {r['sparsity_pct']:.0f}% sparsity")
        lines.append("")
    summary = "\n".join(lines)
    with open(COMP_DIR / "summary.txt", "w") as fh:
        fh.write(summary)

    print(summary)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {COMP_DIR / 'summary.txt'}")


if __name__ == "__main__":
    main()
