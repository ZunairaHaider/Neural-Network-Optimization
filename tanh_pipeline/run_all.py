"""Orchestrate the full tanh-only pipeline: baselines -> pruning -> aggregate.

Runs each step in its own subprocess (fresh module-level seeds = deterministic),
logging stdout/stderr to tanh_pipeline/results/logs/. Continues past a failed
run and reports a status table at the end.

Usage
-----
    python tanh_pipeline/run_all.py                 # everything
    python tanh_pipeline/run_all.py --only-baselines
    python tanh_pipeline/run_all.py --only-pruning
    python tanh_pipeline/run_all.py --datasets A B  # subset (tags, e.g. A B D A_2d)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

OUT_ROOT = Path(__file__).resolve().parents[0]
PY = sys.executable
LOG_DIR = OUT_ROOT / "results" / "logs"

# (dataset, dim) -> tag
DATASETS = [
    ("A", 1), ("B", 1),
    ("A", 2), ("A", 3),
    ("B", 2), ("B", 3),
    ("D", 1),
]
METHODS = ["obd", "obs", "mag", "struct", "activ", "redundancy", "lowrank",
           "taylor_fo", "snip", "imp_rewind"]


def tag_of(dataset: str, dim: int) -> str:
    return dataset if dim == 1 else f"{dataset}_{dim}d"


def run(cmd: list[str], log_path: Path) -> tuple[bool, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    return proc.returncode == 0, dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-baselines", action="store_true")
    ap.add_argument("--only-pruning", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Subset of tags to run (e.g. A B D). Default: all.")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    selected = DATASETS
    if args.datasets:
        wanted = set(args.datasets)
        selected = [(d, dim) for (d, dim) in DATASETS if tag_of(d, dim) in wanted]

    do_base = not args.only_pruning
    do_prune = not args.only_baselines

    results: list[tuple[str, bool, float]] = []
    overall_t0 = time.time()

    if do_base:
        print("=" * 70)
        print("STAGE 1 — baselines")
        print("=" * 70, flush=True)
        for dataset, dim in selected:
            tag = tag_of(dataset, dim)
            print(f"[baseline] {tag} ...", flush=True)
            ok, dt = run(
                [PY, str(OUT_ROOT / "train_baseline.py"),
                 "--dataset", dataset, "--dim", str(dim), "--device", args.device],
                LOG_DIR / f"baseline_{tag}.log",
            )
            results.append((f"baseline/{tag}", ok, dt))
            print(f"    {'OK' if ok else 'FAIL'}  ({dt:.0f}s)", flush=True)

    if do_prune:
        print("=" * 70)
        print("STAGE 2 — pruning (10 methods x datasets)")
        print("=" * 70, flush=True)
        for dataset, dim in selected:
            tag = tag_of(dataset, dim)
            base_cfg = OUT_ROOT / "results" / "baseline" / tag / "config.json"
            if not base_cfg.exists():
                print(f"[skip] {tag}: no baseline yet", flush=True)
                continue
            for method in METHODS:
                print(f"[prune] {method} / {tag} ...", flush=True)
                ok, dt = run(
                    [PY, str(OUT_ROOT / "pruner.py"),
                     "--method", method, "--dataset", dataset, "--dim", str(dim),
                     "--device", args.device],
                    LOG_DIR / f"prune_{method}_{tag}.log",
                )
                results.append((f"{method}/{tag}", ok, dt))
                print(f"    {'OK' if ok else 'FAIL'}  ({dt:.0f}s)", flush=True)

    # Aggregate
    print("=" * 70)
    print("STAGE 3 — aggregate", flush=True)
    ok, dt = run([PY, str(OUT_ROOT / "aggregate.py")], LOG_DIR / "aggregate.log")
    results.append(("aggregate", ok, dt))

    total = time.time() - overall_t0
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print("\n" + "=" * 70)
    print(f"DONE in {total/60:.1f} min — {len(results)} steps, {n_fail} failed")
    print("=" * 70)
    for name, ok, dt in results:
        if not ok:
            print(f"  FAIL  {name}  ({dt:.0f}s)  -> see logs/")
    if n_fail == 0:
        print("  all steps OK")
    print(f"\nComparison: tanh_pipeline/results/comparison/summary.txt")


if __name__ == "__main__":
    main()
