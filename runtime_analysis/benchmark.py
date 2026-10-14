"""
runtime_analysis/benchmark.py
-------------------------------
Benchmarks all pruning methods across datasets and dimensions,
recording real wall-clock timing into:

    runtime_analysis/pruning_runtime_costs.csv

Only sparsity targets 50%, 70%, 90% are evaluated per run to save time.
All combinations of dataset × dim are covered by default.

Usage
-----
    # Full benchmark — all methods, all datasets, all dims (~45-90 min):
    python runtime_analysis/benchmark.py --reset

    # Quick test — 3 fast methods, Dataset A only:
    python runtime_analysis/benchmark.py --methods mag struct activ --datasets A --dims 1

    # All methods, both datasets, dim=1 only:
    python runtime_analysis/benchmark.py --datasets A B --dims 1

    # Override device:
    python runtime_analysis/benchmark.py --device cpu

What this script does
---------------------
1. Loops over every (dataset, dim) combination specified.
2. For each combination, runs all specified pruning methods.
3. Monkeypatches pruner.SPARSITY_STEP and pruner.MAX_SPARSITY temporarily
   so only the benchmark targets are visited — no pruning logic changed.
4. Timing hooks already in pruner.py write rows to the CSV automatically.
5. Restores original constants after every run.

IMPORTANT: pruning logic, seeds, and results are completely unchanged.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# ── benchmark settings ────────────────────────────────────────────────────────
BENCHMARK_TARGETS = [0.50, 0.70, 0.90]   # sparsity levels to visit per run

ALL_METHODS  = ["mag", "struct", "activ", "taylor_fo",
                "obd", "obs", "redundancy", "snip", "imp_rewind", "lowrank"]
ALL_DATASETS = ["A", "B"]
ALL_DIMS     = [1, 2, 3]

# Estimated seconds per method per dataset/dim (for ETA display)
ETA_ESTIMATE = {
    "mag": 8, "struct": 11, "activ": 14, "taylor_fo": 8,
    "obd": 10, "obs": 20, "redundancy": 11, "snip": 14,
    "imp_rewind": 15, "lowrank": 24,
}


def _sparsity_step(targets: list[float]) -> float:
    """Return the smallest gap between consecutive sparsity targets."""
    s = sorted(targets)
    gaps = [round(s[i+1] - s[i], 2) for i in range(len(s) - 1)]
    return min(gaps) if gaps else s[0]


def _baseline_exists(dataset: str, dim: int) -> bool:
    suffix = "" if dim == 1 else f"_{dim}d"
    return (REPO_ROOT / "results" / "baseline" / f"{dataset}{suffix}").exists()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Runtime benchmark — all methods, datasets, dims.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--methods",  nargs="+", default=ALL_METHODS,
                        choices=ALL_METHODS, metavar="METHOD",
                        help=f"Methods to benchmark. Default: all. Choices: {ALL_METHODS}")
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                        choices=ALL_DATASETS, metavar="DATASET",
                        help="Datasets to benchmark (A and/or B). Default: both.")
    parser.add_argument("--dims",     nargs="+", type=int, default=ALL_DIMS,
                        choices=ALL_DIMS, metavar="DIM",
                        help="Input dimensions to benchmark (1, 2, 3). Default: all.")
    parser.add_argument("--device",   default=None,
                        help="Override torch device (e.g. cpu, cuda).")
    parser.add_argument("--reset",    action="store_true",
                        help="Clear existing CSV before running.")
    args = parser.parse_args()

    import src.pruning.pruner as pruner_mod
    from runtime_analysis.csv_logger import RuntimeCSVLogger

    logger = RuntimeCSVLogger()
    if args.reset:
        logger.reset()

    orig_step = pruner_mod.SPARSITY_STEP
    orig_max  = pruner_mod.MAX_SPARSITY

    # ── compute total combinations and ETA ────────────────────────────────────
    combos = [(d, dim) for d in args.datasets for dim in args.dims
              if _baseline_exists(d, dim)]
    n_total = len(combos) * len(args.methods) * len(BENCHMARK_TARGETS)
    eta_s   = sum(ETA_ESTIMATE.get(m, 15) for m in args.methods) * len(combos)

    print(f"\n{'='*65}")
    print(f"  Pruning Runtime Benchmark — Extended (all datasets + dims)")
    print(f"  Methods:  {args.methods}")
    print(f"  Datasets: {args.datasets}   Dims: {args.dims}")
    print(f"  Sparsity targets per run: {[int(t*100) for t in BENCHMARK_TARGETS]}%")
    print(f"  Combinations: {len(combos)} dataset×dim  ×  {len(args.methods)} methods")
    print(f"  Estimated total time: ~{eta_s//60:.0f}–{(eta_s*1.5)//60:.0f} min")
    print(f"  CSV: {logger.path()}")
    print(f"{'='*65}\n")

    step = _sparsity_step(BENCHMARK_TARGETS)
    max_t = max(BENCHMARK_TARGETS)

    t_all = time.perf_counter()
    completed = 0
    failed = []

    for dataset, dim in combos:
        suffix = "" if dim == 1 else f"_{dim}d"
        print(f"\n{'█'*65}")
        print(f"  Dataset: {dataset}   dim={dim}   ({dataset}{suffix})")
        print(f"{'█'*65}")

        for method in args.methods:
            print(f"\n  ── {method.upper()} | {dataset}{suffix} ──────────────────────")

            pruner_mod.SPARSITY_STEP = step
            pruner_mod.MAX_SPARSITY  = max_t

            t_method = time.perf_counter()
            try:
                pruner_mod.run_pruning(method, dataset, dim, args.device)
                elapsed = time.perf_counter() - t_method
                completed += 1
                print(f"  ✓ {method.upper()} | {dataset}{suffix} done in {elapsed:.1f}s")
            except Exception as e:
                elapsed = time.perf_counter() - t_method
                failed.append(f"{method}/{dataset}{suffix}: {e}")
                print(f"  ✗ {method.upper()} | {dataset}{suffix} FAILED in {elapsed:.1f}s: {e}")
            finally:
                pruner_mod.SPARSITY_STEP = orig_step
                pruner_mod.MAX_SPARSITY  = orig_max

    total = time.perf_counter() - t_all
    print(f"\n{'='*65}")
    print(f"  Benchmark complete!")
    print(f"  Completed: {completed} / {len(combos) * len(args.methods)} runs")
    print(f"  Total time: {total:.1f}s  ({total/60:.1f} min)")
    if failed:
        print(f"  Failed ({len(failed)}):")
        for f in failed:
            print(f"    ✗ {f}")
    print(f"  CSV → {logger.path()}")
    print(f"{'='*65}\n")
    print("  Next step → python runtime_analysis/runtime_summary.py --timed-only")


if __name__ == "__main__":
    main()
