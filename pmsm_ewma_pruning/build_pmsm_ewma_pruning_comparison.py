"""
Build a comparison table of every pruning method x sparsity level x PMSM EWMA
baseline, with MSE, FLOPs, param count, and inference time.

Run this AFTER scripts/run_pruning_pmsm_ewma.sh has completed and populated
results/pruning/<method>/<dataset>/sparsity_*.pt

Usage:
    python scripts/build_pmsm_pruning_comparison.py
    python scripts/build_pmsm_pruning_comparison.py --best-only   # 1 row/method/dataset

Outputs:
    results/pruning/pmsm_ewma_comparison_full.csv     (every sparsity level)
    results/pruning/pmsm_ewma_comparison_best.csv      (best row per method/dataset)
    Also prints a markdown table to stdout for the best-per-method view.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.mlp import MLP  # noqa: E402
from runtime_analysis.flops_estimator import estimate_flops, dense_flops  # noqa: E402

DATASETS = ["PMSM_no_temps_ewma", "PMSM_no_temps_pm_ewma", "PMSM_all_in_ewma"]
PRUNING_ROOT = REPO_ROOT / "results" / "pruning"
BASELINE_ROOT = REPO_ROOT / "results" / "baseline"


def _load_baseline_info(dataset: str) -> dict:
    cfg_path = BASELINE_ROOT / dataset / "config.json"
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    return cfg


def _rebuild_model(config: dict) -> MLP:
    arch = config["architecture"] if "architecture" in config else config
    return MLP(
        n_layers=arch["n_layers"],
        n_neurons=arch["n_neurons"],
        activation=arch["activation"],
        input_dim=arch.get("input_dim", 1),
        output_dim=arch.get("output_dim", 1),
    )


def collect_rows() -> list[dict]:
    rows: list[dict] = []

    for dataset in DATASETS:
        try:
            base_cfg = _load_baseline_info(dataset)
        except FileNotFoundError:
            print(f"[skip] no baseline config for {dataset}")
            continue
        baseline_mse = base_cfg["metrics"]["test_mse_locked"]
        baseline_params = base_cfg["n_params"]
        dense_model = _rebuild_model(base_cfg)
        baseline_flops = dense_flops(dense_model)

        rows.append({
            "method": "baseline (unpruned)",
            "dataset": dataset,
            "sparsity_pct": 0.0,
            "test_mse": baseline_mse,
            "mse_ratio_vs_baseline": 1.0,
            "param_count": baseline_params,
            "flops": baseline_flops,
            "inference_ms": None,
        })

        if not PRUNING_ROOT.exists():
            continue

        for method_dir in sorted(PRUNING_ROOT.iterdir()):
            if not method_dir.is_dir():
                continue
            result_dir = method_dir / dataset
            if not result_dir.exists():
                continue

            for ckpt_path in sorted(result_dir.glob("sparsity_*.pt")):
                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                except Exception as e:
                    print(f"[warn] could not load {ckpt_path}: {e}")
                    continue

                model = _rebuild_model(ckpt["config"])
                mask = ckpt["mask"]
                flops = estimate_flops(model, mask=mask)
                test_mse = ckpt["test_mse"]

                rows.append({
                    "method": method_dir.name,
                    "dataset": dataset,
                    "sparsity_pct": round(ckpt["sparsity"] * 100, 1),
                    "test_mse": test_mse,
                    "mse_ratio_vs_baseline": test_mse / baseline_mse,
                    "param_count": ckpt["param_count"],
                    "flops": flops,
                    "inference_ms": ckpt.get("inference_ms"),
                })

    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    import csv
    if not rows:
        print(f"[warn] no rows to write to {path}")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {path}")


def best_per_method(rows: list[dict], mse_blowup_factor: float = 1.5) -> list[dict]:
    """Pick, per (method, dataset), the highest-sparsity row that still keeps
    test MSE within `mse_blowup_factor` x the baseline MSE. Falls back to the
    lowest-MSE row if nothing qualifies."""
    best: dict[tuple[str, str], dict] = {}
    for r in rows:
        if r["method"] == "baseline (unpruned)":
            key = (r["method"], r["dataset"])
            best[key] = r
            continue
        key = (r["method"], r["dataset"])
        qualifies = r["mse_ratio_vs_baseline"] <= mse_blowup_factor
        current = best.get(key)
        if current is None:
            best[key] = r
        else:
            cur_qualifies = current["mse_ratio_vs_baseline"] <= mse_blowup_factor
            if qualifies and cur_qualifies:
                if r["sparsity_pct"] > current["sparsity_pct"]:
                    best[key] = r
            elif qualifies and not cur_qualifies:
                best[key] = r
            elif not qualifies and not cur_qualifies:
                if r["mse_ratio_vs_baseline"] < current["mse_ratio_vs_baseline"]:
                    best[key] = r
    return list(best.values())


def print_markdown(rows: list[dict]) -> None:
    rows_sorted = sorted(rows, key=lambda r: (r["dataset"], r["method"]))
    print("\n| Dataset | Method | Sparsity % | Test MSE | MSE ratio vs baseline | Params | FLOPs |")
    print("|---|---|---|---|---|---|---|")
    for r in rows_sorted:
        print(
            f"| {r['dataset']} | {r['method']} | {r['sparsity_pct']:.1f} | "
            f"{r['test_mse']:.4e} | {r['mse_ratio_vs_baseline']:.2f}x | "
            f"{r['param_count']} | {r['flops']:.0f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-only", action="store_true",
                         help="Only print/save the best row per method/dataset")
    parser.add_argument("--mse-blowup-factor", type=float, default=1.5)
    args = parser.parse_args()

    rows = collect_rows()
    if not rows:
        print("No results found. Did you run scripts/run_pruning_pmsm_ewma.sh first?")
        return

    write_csv(rows, PRUNING_ROOT / "pmsm_ewma_comparison_full.csv")
    best_rows = best_per_method(rows, args.mse_blowup_factor)
    write_csv(best_rows, PRUNING_ROOT / "pmsm_ewma_comparison_best.csv")

    print_markdown(best_rows if args.best_only else rows)


if __name__ == "__main__":
    main()
