# Runtime Analysis — Pruning Computational Cost

Full benchmark covering **all datasets (A, B) × all dimensions (1, 2, 3)**.  
Sparsity targets: **20% / 40% / 60% / 80%** per run.  
Total data points: **202 timed rows** across all combinations.

---

## How to Run

```bash
# Full benchmark — all methods, all datasets, all dims:
python runtime_analysis/benchmark.py --reset

# Generate summary table + plots:
python runtime_analysis/runtime_summary.py --timed-only

# Show per-dataset breakdown:
python runtime_analysis/runtime_summary.py --timed-only --by-dataset
```

---

## Computational Cost Ranking (cheapest → most expensive)

> Averaged across **Dataset A + B**, **dim 1 + 2 + 3**.

| Rank | Method | Avg Runtime | Cost Tier | Rel Cost | Scoring Mechanism |
|------|--------|------------|-----------|----------|-------------------|
| 1 | Taylor FO | 6.21s | 🟢 Moderate | 1× | Gradient × activation (one backward pass) |
| 2 | Redundancy | 6.43s | 🟢 Moderate | 1× | Pairwise neuron correlation matrix |
| 3 | **Activ** | **7.33s** | 🟢 Moderate | **1×** | One forward pass on calibration data |
| 4 | **Struct** | **7.35s** | 🟢 Moderate | **1×** | L2-norm of outgoing weight rows |
| 5 | IMP+Rewind | 8.52s | 🟡 Expensive | 1.4× | Repeated full retraining from init |
| 6 | SNIP | 8.78s | 🟡 Expensive | 1.4× | Gradient of connection sensitivity at init |
| 7 | Magnitude | 9.33s | 🟡 Expensive | 1.5× | Sort \|w\| — no forward pass needed |
| 8 | OBD | 10.97s | 🟡 Expensive | 1.8× | Diagonal Hessian (double backprop) |
| 9 | OBS | 15.56s | 🔴 Very expensive | 2.5× | Full inverse Hessian (O(n²) per weight) |

> **Struct** and **Activ** are the two methods implemented in this branch.  
> Relative cost = avg total time / cheapest method (Taylor FO at 6.21s).

---

## Detailed Breakdown (averaged across all datasets and dims)

| Method | Avg Scoring (s) | Avg Fine-tune (s) | Avg Total (s) | Avg Inference (ms) | Avg MSE ratio |
|--------|----------------|-------------------|---------------|---------------------|---------------|
| Taylor FO | 0.080 | 5.90 | 6.21 | 0.232 | 1.54× |
| Redundancy | 0.088 | 6.12 | 6.43 | 0.192 | 2.03× |
| **Activ** | **0.065** | **7.04** | **7.33** | **0.238** | **1.94×** |
| **Struct** | **0.057** | **7.04** | **7.35** | **0.299** | **2.06×** |
| IMP+Rewind | 0.044 | 8.25 | 8.52 | 0.184 | 2.14× |
| SNIP | 0.002 | 8.55 | 8.78 | 0.186 | 2.13× |
| Magnitude | 0.044 | 9.06 | 9.33 | 0.231 | 2.15× |
| OBD | 4.417 | 6.33 | 10.97 | 0.187 | 1.60× |
| OBS | 6.491 | 8.83 | 15.56 | 0.212 | 1.14× |

---



## Plots

| Plot | Description |
|------|-------------|
| [`plots/runtime_bar.png`](plots/runtime_bar.png) | Stacked bar: scoring + fine-tune time per method (all datasets+dims) |
| [`plots/quality_vs_cost.png`](plots/quality_vs_cost.png) | Scatter: MSE ratio vs runtime — bottom-left = best tradeoff |
| [`plots/runtime_by_dataset.png`](plots/runtime_by_dataset.png) | Side-by-side: Dataset A vs B runtime comparison |

---

## Files

| File | Purpose |
|------|---------|
| `benchmark.py` | Runs live timing — supports `--datasets`, `--dims`, `--methods` |
| `runtime_summary.py` | Prints ranking table, saves CSVs + plots. Use `--by-dataset` for breakdown |
| `run_runtime_analysis.py` | Alternative: reconstruct from saved results (no re-run) |
| `timing_utils.py` | Timer, MemoryProfiler, RuntimeRecord classes |
| `csv_logger.py` | Appends timing rows to CSV |
| `flops_estimator.py` | Estimates multiply-adds per forward pass |
| `pruning_runtime_costs.csv` | Raw data — all methods, datasets, dims, sparsity levels |
| `runtime_summary.csv` | Aggregated per-method averages |
| `runtime_summary_detail.csv` | Per method × dataset × dim breakdown |
