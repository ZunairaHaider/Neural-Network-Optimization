"""
runtime_analysis/runtime_summary.py
--------------------------------------
Loads the runtime CSV, computes averages, prints comparison tables,
generates a computational-cost ranking, and saves:

    runtime_analysis/runtime_summary.csv        — per-method averages
    runtime_analysis/runtime_summary_detail.csv — per method+dataset+dim
    runtime_analysis/plots/runtime_bar.png
    runtime_analysis/plots/quality_vs_cost.png

Usage
-----
    # Full summary (all data including zeros):
    python runtime_analysis/runtime_summary.py

    # Only rows with live timing data:
    python runtime_analysis/runtime_summary.py --timed-only

    # Show per-dataset breakdown instead of global averages:
    python runtime_analysis/runtime_summary.py --timed-only --by-dataset

    # Custom CSV path:
    python runtime_analysis/runtime_summary.py --csv runtime_analysis/pruning_runtime_costs.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CSV_IN         = REPO_ROOT / "runtime_analysis" / "pruning_runtime_costs.csv"
CSV_OUT        = REPO_ROOT / "runtime_analysis" / "runtime_summary.csv"
CSV_OUT_DETAIL = REPO_ROOT / "runtime_analysis" / "runtime_summary_detail.csv"
PLOT_DIR       = REPO_ROOT / "runtime_analysis" / "plots"

METHOD_LABELS = {
    "obd": "OBD", "obs": "OBS", "mag": "Magnitude",
    "struct": "Struct", "activ": "Activ", "redundancy": "Redundancy",
    "taylor_fo": "Taylor FO", "snip": "SNIP",
    "imp_rewind": "IMP+Rewind", "lowrank": "LowRank",
}

COMPLEXITY_DESC = {
    "mag":        "Sort |w| — no forward pass needed",
    "struct":     "L2-norm of outgoing weight rows",
    "activ":      "One forward pass on calibration data",
    "lowrank":    "SVD decomposition of weight matrices",
    "redundancy": "Pairwise neuron correlation matrix",
    "taylor_fo":  "Gradient × activation (one backward pass)",
    "snip":       "Gradient of connection sensitivity at init",
    "obd":        "Diagonal Hessian (double backprop)",
    "obs":        "Full inverse Hessian (O(n²) per weight)",
    "imp_rewind": "Repeated full retraining from init",
}


def load_csv(path: Path, timed_only: bool) -> list[dict]:
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if timed_only and float(r["total_time_s"]) == 0.0:
                continue
            rows.append(r)
    return rows


def _avg(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def aggregate_by_method(rows: list[dict]) -> dict[str, dict]:
    """Average across all datasets, dims, sparsity levels per method."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[r["method"]].append(r)

    summary = {}
    for method, group in buckets.items():
        def avg(key, g=group):
            vals = [float(r[key]) for r in g if float(r[key]) != 0.0]
            if key == "mse_ratio":
                vals = [float(r[key]) for r in g if 0 < float(r[key]) < 10.0]
            return _avg(vals)

        datasets = sorted(set(r["dataset"] for r in group))
        dims     = sorted(set(int(r["dim"]) for r in group))

        summary[method] = {
            "method":           method,
            "label":            METHOD_LABELS.get(method, method),
            "n_rows":           len(group),
            "datasets":         ",".join(datasets),
            "dims":             ",".join(str(d) for d in dims),
            "avg_scoring_s":    avg("scoring_time_s"),
            "avg_finetune_s":   avg("finetune_time_s"),
            "avg_total_s":      avg("total_time_s"),
            "avg_inference_ms": avg("inference_ms"),
            "avg_flops":        avg("flops_estimate"),
            "avg_mse_ratio":    avg("mse_ratio"),
        }
    return summary


def aggregate_by_method_dataset(rows: list[dict]) -> list[dict]:
    """Detailed breakdown: one row per (method, dataset, dim)."""
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["method"], r["dataset"], int(r["dim"]))
        buckets[key].append(r)

    results = []
    for (method, dataset, dim), group in sorted(buckets.items()):
        def avg(key, g=group):
            vals = [float(r[key]) for r in g if float(r[key]) != 0.0]
            if key == "mse_ratio":
                vals = [float(r[key]) for r in g if 0 < float(r[key]) < 10.0]
            return _avg(vals)

        results.append({
            "method":           method,
            "label":            METHOD_LABELS.get(method, method),
            "dataset":          dataset,
            "dim":              dim,
            "n_rows":           len(group),
            "avg_scoring_s":    avg("scoring_time_s"),
            "avg_finetune_s":   avg("finetune_time_s"),
            "avg_total_s":      avg("total_time_s"),
            "avg_inference_ms": avg("inference_ms"),
            "avg_flops":        avg("flops_estimate"),
            "avg_mse_ratio":    avg("mse_ratio"),
        })
    return results


def cost_tier(total_s: float, max_s: float) -> str:
    if max_s == 0 or total_s == 0: return "N/A"
    f = total_s / max_s
    if f < 0.10: return "Very cheap"
    if f < 0.25: return "Cheap"
    if f < 0.50: return "Moderate"
    if f < 0.75: return "Expensive"
    return "Very expensive"


def relative_cost(total_s: float, min_s: float) -> str:
    if min_s == 0 or total_s == 0: return "—"
    r = total_s / min_s
    if r < 1.5: return "1×"
    if r < 3:   return f"{r:.1f}×"
    return f"{r:.0f}×"


def print_table(summary: dict[str, dict], title: str = "") -> None:
    rows = sorted(summary.values(), key=lambda r: r["avg_total_s"])
    max_s = max(r["avg_total_s"] for r in rows) or 1.0
    min_s = next((r["avg_total_s"] for r in rows if r["avg_total_s"] > 0), 1.0)

    W = {"method": 12, "total": 12, "score": 12, "ft": 13, "infer": 10, "mse": 8, "tier": 16, "rel": 8}

    def fmt(m, tot, sc, ft, inf, mse, tier, rel):
        return (f"  {m:<{W['method']}} {tot:>{W['total']}} {sc:>{W['score']}} "
                f"{ft:>{W['ft']}} {inf:>{W['infer']}} {mse:>{W['mse']}} "
                f"{tier:<{W['tier']}} {rel:>{W['rel']}}")

    sep = "  " + "─" * (sum(W.values()) + len(W))
    hdr = fmt("Method", "Avg total(s)", "Avg score(s)", "Avg finetune(s)",
              "Infer(ms)", "MSE×", "Cost tier", "Rel cost")

    heading = title or "Pruning Method Runtime Summary"
    print(f"\n  ┌─ {heading} " + "─" * max(0, 60 - len(heading)) + "┐")
    print(hdr)
    print(sep)

    for r in rows:
        tot = r["avg_total_s"];  sc = r["avg_scoring_s"];  ft = r["avg_finetune_s"]
        inf = r["avg_inference_ms"];  mse = r["avg_mse_ratio"]
        tier = cost_tier(tot, max_s);  rel = relative_cost(tot, min_s)

        print(fmt(
            r["label"],
            f"{tot:.3f}s" if tot > 0 else "N/A",
            f"{sc:.3f}s"  if sc  > 0 else "N/A",
            f"{ft:.3f}s"  if ft  > 0 else "N/A",
            f"{inf:.4f}"  if inf > 0 else "N/A",
            f"{mse:.3f}×" if mse > 0 else "N/A",
            tier, rel,
        ))

    print(sep)
    print("  Rel cost = avg_total / cheapest method.  N/A = no live timing data.")
    print()


def print_ranking(summary: dict[str, dict]) -> None:
    rows = sorted(summary.values(), key=lambda r: r["avg_total_s"])
    max_s = max(r["avg_total_s"] for r in rows) or 1.0
    min_s = next((r["avg_total_s"] for r in rows if r["avg_total_s"] > 0), 1.0)

    print("  ┌─ Computational Cost Ranking (cheapest → most expensive) " + "─" * 22 + "┐")
    print(f"  {'Method':<14} {'Avg Runtime':>12}  {'Relative Cost':<16}  Scoring mechanism")
    print("  " + "─" * 78)
    for r in rows:
        tot  = r["avg_total_s"]
        tier = cost_tier(tot, max_s)
        rel  = relative_cost(tot, min_s)
        tot_s = f"{tot:.2f}s" if tot > 0 else "N/A (run benchmark)"
        desc  = COMPLEXITY_DESC.get(r["method"], "")
        print(f"  {r['label']:<14} {tot_s:>12}  {tier:<16}  {desc}")
    print()


def print_by_dataset(detail_rows: list[dict]) -> None:
    """Print a breakdown table per method × dataset × dim."""
    print("\n  ┌─ Runtime Breakdown by Dataset & Dimension " + "─" * 36 + "┐")
    print(f"  {'Method':<14} {'Dataset':>8} {'Dim':>4} {'Total(s)':>10} "
          f"{'Score(s)':>10} {'Finetune(s)':>12} {'MSE×':>8}")
    print("  " + "─" * 72)

    prev_method = None
    for r in detail_rows:
        if r["method"] != prev_method:
            if prev_method is not None:
                print()
            prev_method = r["method"]

        tot = r["avg_total_s"];  sc = r["avg_scoring_s"];  ft = r["avg_finetune_s"]
        mse = r["avg_mse_ratio"]

        print(f"  {r['label']:<14} {r['dataset']:>8} {r['dim']:>4} "
              f"{f'{tot:.2f}s' if tot > 0 else 'N/A':>10} "
              f"{f'{sc:.3f}s' if sc > 0 else 'N/A':>10} "
              f"{f'{ft:.2f}s' if ft > 0 else 'N/A':>12} "
              f"{f'{mse:.2f}×' if mse > 0 else 'N/A':>8}")
    print()


def save_summary_csv(summary: dict[str, dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "label", "datasets", "dims", "n_rows",
              "avg_scoring_s", "avg_finetune_s", "avg_total_s",
              "avg_inference_ms", "avg_flops", "avg_mse_ratio"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in sorted(summary.values(), key=lambda x: x["avg_total_s"]):
            w.writerow({k: round(r[k], 6) if isinstance(r[k], float) else r[k]
                        for k in fields})
    print(f"  Summary CSV saved       → {path}")


def save_detail_csv(detail_rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "label", "dataset", "dim", "n_rows",
              "avg_scoring_s", "avg_finetune_s", "avg_total_s",
              "avg_inference_ms", "avg_flops", "avg_mse_ratio"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in detail_rows:
            w.writerow({k: round(r[k], 6) if isinstance(r[k], float) else r[k]
                        for k in fields})
    print(f"  Detail CSV saved        → {path}")


def plot_bar(summary: dict[str, dict], detail_rows: list[dict],
             plot_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [skip] matplotlib not available.")
        return

    plot_dir.mkdir(parents=True, exist_ok=True)

    rows = sorted(summary.values(), key=lambda r: r["avg_total_s"])
    labels  = [r["label"] for r in rows]
    score_s = [r["avg_scoring_s"]  for r in rows]
    ft_s    = [r["avg_finetune_s"] for r in rows]
    total_s = [r["avg_total_s"]    for r in rows]
    has_live = any(t > 0 for t in total_s)

    x = list(range(len(labels)))

    # ── Plot 1: stacked runtime bar ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Pruning Method — Computational Cost Comparison\n"
        "(averaged across all datasets and dimensions)",
        fontsize=12, fontweight="bold"
    )

    ax = axes[0]
    if has_live:
        ax.bar(x, score_s, label="Scoring time",    color="#378ADD")
        ax.bar(x, ft_s,    label="Fine-tune time",  color="#D85A30", bottom=score_s)
        ax.set_ylabel("Seconds")
        ax.legend(fontsize=9)
    else:
        ax.bar(x, [1]*len(labels), color="#B4B2A9")
        ax.text(0.5, 0.5, "Run benchmark first\npython runtime_analysis/benchmark.py",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)

    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_title("Avg runtime per target (scoring + fine-tune)\naveraged across Dataset A+B, dim 1+2+3", fontsize=10)
    ax.spines[["top","right"]].set_visible(False)

    # ── Plot 2: inference latency ─────────────────────────────────────────────
    ax2 = axes[1]
    infer = [r["avg_inference_ms"] for r in rows]
    if any(i > 0 for i in infer):
        ax2.bar(x, infer, color="#1D9E75")
        ax2.set_ylabel("Milliseconds")
    else:
        ax2.bar(x, [1]*len(labels), color="#B4B2A9")
        ax2.text(0.5, 0.5, "No inference data", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=11)

    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax2.set_title("Avg inference latency per method", fontsize=10)
    ax2.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = plot_dir / "runtime_bar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Bar chart saved         → {out}")

    # ── Plot 3: quality vs cost scatter ───────────────────────────────────────
    if has_live:
        fig2, ax3 = plt.subplots(figsize=(7, 5))
        mse_vals = [r["avg_mse_ratio"] for r in rows]
        colors   = ["#1D9E75" if m < 1.5 else "#BA7517" if m < 3 else "#E24B4A"
                    for m in mse_vals]
        ax3.scatter(total_s, mse_vals, c=colors, s=100, zorder=3)
        for r, t, m in zip(rows, total_s, mse_vals):
            ax3.annotate(r["label"], (t, m), textcoords="offset points",
                         xytext=(5, 3), fontsize=8)
        ax3.axhline(1.3, color="#E24B4A", linestyle="--", linewidth=0.8,
                    label="1.3× MSE threshold")
        ax3.set_xlabel("Avg total runtime per target (s)")
        ax3.set_ylabel("Avg MSE ratio (×baseline)")
        ax3.set_title("Quality vs computational cost\n(bottom-left = best tradeoff, averaged across all datasets+dims)",
                      fontsize=10)
        ax3.legend(fontsize=8)
        ax3.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        out2 = plot_dir / "quality_vs_cost.png"
        plt.savefig(out2, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Scatter plot saved      → {out2}")

    # ── Plot 4: per-dataset bar chart (if detail data exists) ─────────────────
    if has_live and detail_rows:
        datasets = sorted(set(r["dataset"] for r in detail_rows))
        methods_sorted = [r["label"] for r in rows]

        fig3, axes3 = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5),
                                   sharey=True)
        if len(datasets) == 1:
            axes3 = [axes3]

        fig3.suptitle("Runtime per method — Dataset A vs Dataset B comparison",
                      fontsize=12, fontweight="bold")

        for ax_i, (ax_d, ds) in enumerate(zip(axes3, datasets)):
            ds_rows = {r["method"]: r for r in detail_rows
                       if r["dataset"] == ds and r["dim"] == 1}
            totals = [ds_rows.get(r["method"], {}).get("avg_total_s", 0)
                      for r in rows]
            colors_d = ["#378ADD" if ds == "A" else "#1D9E75"] * len(methods_sorted)
            ax_d.bar(x, totals, color=colors_d)
            ax_d.set_xticks(x)
            ax_d.set_xticklabels(methods_sorted, rotation=35, ha="right", fontsize=9)
            ax_d.set_title(f"Dataset {ds}, dim=1", fontsize=11)
            ax_d.set_ylabel("Avg total runtime (s)" if ax_i == 0 else "")
            ax_d.spines[["top","right"]].set_visible(False)

        plt.tight_layout()
        out3 = plot_dir / "runtime_by_dataset.png"
        plt.savefig(out3, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Dataset comparison plot → {out3}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Runtime summary and cost ranking.")
    parser.add_argument("--csv",        default=str(CSV_IN))
    parser.add_argument("--timed-only", action="store_true",
                        help="Only use rows with live timing (total_time_s > 0).")
    parser.add_argument("--by-dataset", action="store_true",
                        help="Also print per-dataset × dim breakdown table.")
    parser.add_argument("--no-plot",    action="store_true")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        print("Run first:  python runtime_analysis/benchmark.py --reset")
        sys.exit(1)

    rows = load_csv(csv_path, timed_only=args.timed_only)
    if not rows:
        print("No rows found. If using --timed-only, run the benchmark first.")
        sys.exit(1)

    # Show coverage info
    from collections import Counter
    coverage = Counter(f"{r['dataset']}_d{r['dim']}" for r in rows)
    timed_coverage = Counter(f"{r['dataset']}_d{r['dim']}"
                             for r in rows if float(r["total_time_s"]) > 0)

    print(f"\n  Loaded {len(rows)} rows from {csv_path.name}")
    print(f"  Coverage: {dict(coverage)}")
    if timed_coverage:
        print(f"  With live timing: {dict(timed_coverage)}")
    if args.timed_only:
        print("  (timed-only mode: rows with total_time_s=0 excluded)")

    summary     = aggregate_by_method(rows)
    detail_rows = aggregate_by_method_dataset(rows)

    print_table(summary, title="Runtime Summary — Averaged Across All Datasets & Dims")
    print_ranking(summary)

    if args.by_dataset:
        print_by_dataset(detail_rows)

    save_summary_csv(summary, CSV_OUT)
    save_detail_csv(detail_rows, CSV_OUT_DETAIL)

    if not args.no_plot:
        plot_bar(summary, detail_rows, PLOT_DIR)

    print()


if __name__ == "__main__":
    main()
