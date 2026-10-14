# Handoff — tanh-only baseline + pruning rerun

**Folder:** `tanh_pipeline/`
**Date:** 2026-06-12
**Branch:** `main`

---

## Why this rerun existed

The previous study's baselines were **overparametrized**: the tanh HPO selected
`[3×32] = 2209` params for the clean 1-D dataset A, where a sub-30-param tanh net
fits near-perfectly. This was tracked as a "patience" bug, but the real cause was
deeper.

**Root cause (confirmed):** the HPO trials never trained small nets *to
convergence*. Full-batch Adam (even at 20 000 epochs) leaves tiny smooth-regression
tanh nets far from their minimum, so the selector ranked architectures on
under-trained trial MSE and kept climbing to bigger nets. It was an **optimizer
problem, not a selection-rule problem.** Proof (dataset A, 1-D), trial MSE vs.
converged MSE: a 13-param net trained to convergence reaches 2.6e-4 — matching the
2209-param net; a 19-param net reaches 2.6e-7, the best of all architectures.

## What was done

1. **Tanh-only search space** (prior study mixed ReLU/GELU/tanh; ReLU won only
   because of this bug).
2. **LBFGS strong-Wolfe convergence polish** added after Adam (`lbfgs_polish()` in
   `train_baseline.py`), applied in **both** the HPO trial eval and the final
   retrain, with a keep-only-if-val-improves guard. This is the core fix.
3. **Patience raised to 100** (trial + retrain).
4. **All 7 datasets** (A/B at d=1,2,3 plus D) × **all 10 pruning methods**.
5. **OBS crash fixed:** once OBS falls back to OBD-diagonal saliency
   (ill-conditioned Hessian), it now stays in fallback — a later iteration whose
   Hessian inverts cleanly no longer leaves `h_diag` unset (this had crashed obs on
   B, A_3d, B_3d with `float * NoneType`).
6. **Aggregation filter set to 1.5× baseline MSE** for the "best compression" pick
   (`aggregate.py`, `MSE_BLOWUP_FACTOR`).

## Results

### Corrected baselines
| Dataset | Arch | Params | Test MSE | vs. before |
|---|---|---:|---:|---|
| A d1 | 1×8 | 25 | 1.3e-5 | was 2209 → **88× smaller** |
| A d2 | 1×16 | 65 | 2.5e-5 | small |
| A d3 | 3×32 | 2273 | 5.8e-4 | **genuine** (see below) |
| B d1 | 1×4 | 13 | 3.95 | small (noise-limited) |
| B d2 | 1×4 | 17 | 27.4 | small |
| B d3 | 1×6 | 31 | 622 | small |
| D d1 | 1×4 | 13 | 9.3e-3 | small |

Six of seven baselines are now small. **A_3d's `[3×32]` is legitimate, not an
artifact:** a convergence test showed no small tanh net fits the 3-D quadratic
below the 1e-3 target (single-hidden-layer nets fail entirely — they cannot
represent the cross-terms; the smallest net clearing 1e-3 is ~2×32). Its pruning
fragility (10% pruning → 128× MSE) confirms it densely uses capacity.

### Pruning (best compression within 1.5× baseline MSE)
- Noisy datasets prune hard: B → 6 params (snip, 50%), B_2d → 8 (obd, 50%),
  B_3d → 9 (obd, 70%), D → 7 (struct, 40%).
- Clean low-dim barely prunes: A → 22 (obs, 10%), A_2d → 45 (obs, 30%).
- A_3d: nothing within 1.5× (too fragile).
- Second-order (OBD/OBS) + Taylor-FO are the most reliable compressors.

Full per-method tables: `results.ipynb`, `results/comparison/summary.txt`,
`results/comparison/comparison.csv`.

## Repo state (this folder)

```
tanh_pipeline/
├── train_baseline.py   ← tanh-only HPO + LBFGS convergence polish (the fix)
├── pruner.py           ← 10-method pruner (obs fallback bug fixed)
├── aggregate.py        ← comparison.csv + summary.txt (1.5× filter)
├── run_all.py          ← orchestrator: 7 datasets × 10 methods → aggregate
├── results.ipynb       ← readable single-file summary (this study)
├── HANDOFF.md          ← this file
└── results/
    ├── baseline/<tag>/         config.json, model.pt, curves, hpo_results
    ├── pruning/<method>/<tag>/ checkpoints + topology plots
    ├── comparison/             summary.txt, comparison.csv
    └── logs/                   per-step run logs
```

## How to reproduce

```bash
source .venv/bin/activate
python tanh_pipeline/run_all.py --device cpu     # ~75 min, full pipeline + aggregate
# single cell:
python tanh_pipeline/train_baseline.py --dataset A --dim 1 --device cpu
python tanh_pipeline/pruner.py --method obs --dataset A --dim 1 --device cpu
python tanh_pipeline/aggregate.py
```

## Open / optional follow-ups

- **Noise-floor stop on clean data is a judgment call.** For A the 1e-3 floor is
  arbitrary (true min ≈ 0). If you'd accept ~5e-2 on A_3d, the HPO would pick a
  ~113-param net (2×8) instead of `[3×32]`. Left strict for now.
- **Single seed per cell** — no error bars (matches the prior study's scope).
- The `results.ipynb` is all-markdown (no Jupyter kernel installed in `.venv`);
  tables are generated from the live CSV by `/tmp/build_nb.py` if a rebuild is
  needed.
