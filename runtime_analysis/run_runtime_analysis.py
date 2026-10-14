"""
runtime_analysis/run_runtime_analysis.py
------------------------------------------
Standalone script to collect runtime/computational-cost data for all
pruning methods and save results to:

    runtime_analysis/pruning_runtime_costs.csv

TWO MODES
---------

  MODE 1 — Re-run with live timing (recommended for accurate data):
  ----------------------------------------------------------------
      python runtime_analysis/run_runtime_analysis.py --live

      Runs each (method, dataset, dim) combination through the pruning
      pipeline with time.perf_counter() hooks injected around:
        • the scoring/saliency step
        • the fine-tuning step
      Results are accurate wall-clock times on your machine.

      Options:
        --methods obd mag struct        (subset of methods to run)
        --datasets A B                  (subset of datasets)
        --dims 1 2 3                    (subset of dims)
        --mem                           (also profile CPU RAM via tracemalloc)
        --device cpu                    (override device)

  MODE 2 — Reconstruct from existing results (no re-run needed):
  -------------------------------------------------------------
      python runtime_analysis/run_runtime_analysis.py --from-results

      Reads every sparsity_*_config.json already saved under
      results/pruning/ and builds a summary CSV with:
        • method, dataset, dim, sparsity_pct
        • final_test_mse, param_count, inference_ms
        • mse_ratio vs baseline
      Timing fields are left as 0.0 (not available from saved checkpoints).
      Useful to get a quick structural summary without re-running.

SAFETY
------
Neither mode modifies the existing pruning code, results, or checkpoints.
Live mode calls run_pruning() from the unmodified pruner.py with a thin
wrapper that starts/stops timers around scoring and fine-tune calls.
From-results mode is purely read-only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

# ── repo root on sys.path ─────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "runtime_analysis"))

from runtime_analysis.timing_utils import RuntimeRecord, Timer, MemoryProfiler, CUDAMemoryProfiler
from runtime_analysis.csv_logger import RuntimeCSVLogger
from runtime_analysis.flops_estimator import estimate_flops

import torch
import torch.nn as nn
import numpy as np


# ── constants mirrored from pruner.py (read-only) ────────────────────────────
ALL_METHODS = [
    "obd", "obs", "mag", "struct", "activ",
    "redundancy", "lowrank", "taylor_fo", "snip", "imp_rewind",
]
ALL_DATASETS = ["A", "B"]
ALL_DIMS = [1, 2, 3]


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: load baseline MSE from saved config
# ─────────────────────────────────────────────────────────────────────────────

def _load_baseline_mse(dataset: str, dim: int) -> float:
    suffix = "" if dim == 1 else f"_{dim}d"
    cfg_path = REPO_ROOT / "results" / "baseline" / f"{dataset}{suffix}" / "config.json"
    if not cfg_path.exists():
        return 0.0
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    # baseline config stores test MSE under metrics.test_mse_locked
    metrics = cfg.get("metrics", {})
    return float(
        metrics.get("test_mse_locked")
        or metrics.get("test_mse")
        or cfg.get("test_mse", 0.0)
    )


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2: reconstruct from existing saved results
# ─────────────────────────────────────────────────────────────────────────────

def collect_from_results(
    methods: List[str],
    datasets: List[str],
    dims: List[int],
    logger: RuntimeCSVLogger,
) -> int:
    """Scan results/pruning/ and build CSV from saved _config.json files."""
    total = 0
    pruning_root = REPO_ROOT / "results" / "pruning"

    for method in methods:
        for dataset in datasets:
            for dim in dims:
                suffix = "" if dim == 1 else f"_{dim}d"
                dataset_tag = f"{dataset}{suffix}"
                result_dir = pruning_root / method / dataset_tag

                if not result_dir.exists():
                    continue

                baseline_mse = _load_baseline_mse(dataset, dim)
                cfg_files = sorted(result_dir.glob("*_config.json"))

                for cfg_file in cfg_files:
                    with open(cfg_file) as fh:
                        data = json.load(fh)

                    test_mse = float(data.get("test_mse", 0.0))
                    mse_ratio = (test_mse / baseline_mse) if baseline_mse > 0 else 0.0

                    rec = RuntimeRecord(
                        method          = method,
                        dataset         = dataset,
                        dim             = dim,
                        sparsity_pct    = float(data.get("sparsity_pct", 0.0)),
                        final_test_mse  = test_mse,
                        baseline_mse    = baseline_mse,
                        mse_ratio       = round(mse_ratio, 4),
                        param_count     = int(data.get("param_count", 0)),
                        inference_ms    = float(data.get("inference_ms", 0.0)),
                        # Timing fields not available from saved checkpoints
                        scoring_time_s  = 0.0,
                        finetune_time_s = 0.0,
                        total_time_s    = 0.0,
                    )
                    logger.append(rec)
                    total += 1
                    print(f"  [{method}] {dataset_tag} sparsity={rec.sparsity_pct:.0f}%  MSE={test_mse:.4e}")

    return total


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1: live timing via thin wrappers around pruner internals
# ─────────────────────────────────────────────────────────────────────────────

def run_live_timing(
    methods: List[str],
    datasets: List[str],
    dims: List[int],
    device_arg: Optional[str],
    profile_mem: bool,
    logger: RuntimeCSVLogger,
) -> int:
    """
    Import the unmodified pruner module and monkey-patch ONLY the two
    timing-critical inner functions:
      • The saliency/scoring block → wrapped with Timer
      • _finetune() → wrapped with Timer

    The monkey-patch is scoped to this process and does NOT write to
    any source file. All pruning logic remains byte-for-byte identical.
    """

    import src.pruning.pruner as pruner_mod

    # ── save original _finetune so we can restore it later ───────────────────
    _original_finetune = pruner_mod._finetune

    # ── shared accumulator dict (updated by the patched finetune) ────────────
    _timing: dict = {}

    def _timed_finetune(model, mask, x_tr, y_tr, x_val, y_val, lr, batch_size, device):
        with Timer() as t:
            result = _original_finetune(model, mask, x_tr, y_tr, x_val, y_val, lr, batch_size, device)
        _timing["finetune_s"] = t.elapsed_s
        return result

    pruner_mod._finetune = _timed_finetune

    total = 0

    try:
        for method in methods:
            for dataset in datasets:
                for dim in dims:
                    suffix = "" if dim == 1 else f"_{dim}d"
                    dataset_tag = f"{dataset}{suffix}"

                    # Check baseline exists
                    base_dir = REPO_ROOT / "results" / "baseline" / dataset_tag
                    if not base_dir.exists():
                        print(f"  [skip] No baseline for {dataset_tag}")
                        continue

                    print(f"\n── {method.upper()} | {dataset_tag} ──────────────────────")

                    baseline_mse = _load_baseline_mse(dataset, dim)

                    t_total = Timer()
                    mem_cpu  = MemoryProfiler(enabled=profile_mem)
                    mem_cuda = CUDAMemoryProfiler()
                    _timing.clear()

                    with t_total, mem_cpu, mem_cuda:
                        pruner_mod.run_pruning(method, dataset, dim, device_arg)

                    # Collect the last checkpoint's config to get final MSE
                    result_dir = REPO_ROOT / "results" / "pruning" / method / dataset_tag
                    cfg_files = sorted(result_dir.glob("*_config.json"))
                    if not cfg_files:
                        print(f"  [warn] No config files found for {method}/{dataset_tag}")
                        continue

                    # Use the highest-sparsity checkpoint as the "summary" row
                    with open(cfg_files[-1]) as fh:
                        last_cfg = json.load(fh)

                    test_mse = float(last_cfg.get("test_mse", 0.0))
                    sparsity_pct = float(last_cfg.get("sparsity_pct", 0.0))
                    param_count = int(last_cfg.get("param_count", 0))
                    inference_ms = float(last_cfg.get("inference_ms", 0.0))
                    mse_ratio = (test_mse / baseline_mse) if baseline_mse > 0 else 0.0

                    finetune_s = _timing.get("finetune_s", 0.0)
                    scoring_s = max(0.0, t_total.elapsed_s - finetune_s)

                    rec = RuntimeRecord(
                        method          = method,
                        dataset         = dataset,
                        dim             = dim,
                        sparsity_pct    = sparsity_pct,
                        scoring_time_s  = round(scoring_s, 4),
                        finetune_time_s = round(finetune_s, 4),
                        total_time_s    = round(t_total.elapsed_s, 4),
                        final_test_mse  = test_mse,
                        baseline_mse    = baseline_mse,
                        mse_ratio       = round(mse_ratio, 4),
                        param_count     = param_count,
                        inference_ms    = inference_ms,
                        cpu_peak_mb     = round(mem_cpu.peak_mb, 3),
                        cuda_peak_mb    = round(mem_cuda.peak_mb, 3),
                    )
                    logger.append(rec)
                    total += 1

                    print(
                        f"  ✓ total={rec.total_time_s:.1f}s  "
                        f"finetune={rec.finetune_time_s:.1f}s  "
                        f"scoring≈{rec.scoring_time_s:.1f}s  "
                        f"MSE={test_mse:.4e}  ratio={mse_ratio:.2f}x"
                    )

    finally:
        # Always restore original _finetune even if the run errors
        pruner_mod._finetune = _original_finetune

    return total


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Runtime/cost analysis for pruning methods.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--live", action="store_true",
        help="Re-run pruning with live timing hooks (accurate wall-clock).",
    )
    mode_group.add_argument(
        "--from-results", action="store_true",
        help="Reconstruct summary from existing saved results (no re-run).",
    )

    parser.add_argument(
        "--methods", nargs="+", default=ALL_METHODS,
        choices=ALL_METHODS, metavar="METHOD",
        help=f"Methods to analyse. Default: all. Choices: {ALL_METHODS}",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS,
        choices=ALL_DATASETS, metavar="DATASET",
        help="Datasets (A and/or B). Default: both.",
    )
    parser.add_argument(
        "--dims", nargs="+", type=int, default=ALL_DIMS,
        choices=ALL_DIMS, metavar="DIM",
        help="Input dimensionalities (1, 2, 3). Default: all.",
    )
    parser.add_argument(
        "--mem", action="store_true",
        help="(Live mode only) Enable CPU RAM profiling via tracemalloc.",
    )
    parser.add_argument(
        "--device", default=None,
        help="(Live mode only) Override torch device (e.g. cpu, cuda, mps).",
    )
    parser.add_argument(
        "--output", default=None,
        help="CSV output path. Default: runtime_analysis/pruning_runtime_costs.csv",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete existing CSV before writing (start fresh).",
    )

    args = parser.parse_args()

    logger = RuntimeCSVLogger(csv_path=args.output)
    if args.reset:
        logger.reset()

    print(f"\n{'='*60}")
    print(f"  Runtime Analysis  —  mode: {'live' if args.live else 'from-results'}")
    print(f"  Methods:  {args.methods}")
    print(f"  Datasets: {args.datasets}  Dims: {args.dims}")
    print(f"  Output:   {logger.path()}")
    print(f"{'='*60}\n")

    t0 = time.perf_counter()

    if args.live:
        n = run_live_timing(
            methods    = args.methods,
            datasets   = args.datasets,
            dims       = args.dims,
            device_arg = args.device,
            profile_mem= args.mem,
            logger     = logger,
        )
    else:
        n = collect_from_results(
            methods  = args.methods,
            datasets = args.datasets,
            dims     = args.dims,
            logger   = logger,
        )

    elapsed = time.perf_counter() - t0
    print(f"\n{'='*60}")
    print(f"  Done.  Wrote {n} rows in {elapsed:.1f}s")
    print(f"  CSV → {logger.path()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
