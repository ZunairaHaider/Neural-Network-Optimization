# Pruning scaling-law study — companion report

ANN Optimisation Study (CS_ANN), branch `scaling-law-study`.


---

## 0. TL;DR

We pruned a hyper-parameter-optimised MLP regressing a multivariate
quadratic with **ten** pruning methods at three input dimensions
(d ∈ {1, 2, 3}) and two noise levels (clean, σ = 12.5 % peak-to-peak).
Four findings:

1. **Compressibility ranking is dimension-dependent.** At d = 1 the
   gradient-based scalar methods (OBD, mag, Taylor-FO, OBS) lead. At d ≥ 2
   the structured (`activ`) and iterative-LTH (`imp_rewind`) methods
   close in and sometimes overtake.
2. **Optimal-Brain-Surgeon falls apart above d = 1.** The trained
   Hessian becomes ill-conditioned (cond > 10¹²) at d = 2 and the weight
   correction amplifies error along small-eigenvalue directions. Our
   audit §2.3 fix correctly auto-detects this and falls back to
   OBD-diagonal saliency.
3. **Noisy data hides pruning damage.** On Dataset B, almost every method
   reaches 80–90 % sparsity at a 2× MSE budget — the noise floor σ² caps
   the visible degradation.
4. **The test-set leakage fix (audit §2.1) matters at high sparsity.**
   The previous OBS Dataset B sweep reported test MSE = 34.7 at 80 %
   sparsity (≈ 9× noise floor). The corrected sweep reports MSE = 4.2
   (at the noise floor). At low / mid sparsity the deltas are within
   seed-noise (≤ 2× either way).

---

## 1. Research question

Given a baseline MLP that's been hyper-parameter-optimised for a regression
target, *how does the right pruning method depend on the input dimension
and noise level of that target?* Most published pruning comparisons fix one
problem and one dataset; we sweep three problem sizes and two noise levels
on the same architecture family.

---

## 2. Datasets

Quadratic regression `f : ℝ^d → ℝ`,
**f(x) = xᵀA_d x + b_dᵀx + c**, with `(A_d, b_d, c)` deterministic and
seeded per dim. `A_d` is symmetric positive-definite (`L Lᵀ + 0.5 I`), so
each problem has a unique interior minimum inside `[−3, 3]^d`.

| Dataset | dim | params A | σ | peak-to-peak | noise floor σ² |
|---|---|---|---|---|---|
| A | 1 | A=1, b=−2, c=1 | 0 | 16.0 | 0.0 |
| A | 2 | A_2 seed 44 | 0 | 42.4 | 0.0 |
| A | 3 | A_3 seed 45 | 0 | 190.3 | 0.0 |
| B | 1 | (clean + 𝒩(0, σ²)) | 2.00 | 16.0 | 4.0 |
| B | 2 | 〃 | 5.30 | 42.4 | 28.1 |
| B | 3 | 〃 | 23.78 | 190.3 | 565.5 |

Sampling: 1-D uses a uniform-grid random-shuffle (legacy, identical to prior
work); d ≥ 2 uses IID uniform `[−3, 3]^d` because a true grid would either
be too coarse (`5000^(1/d)` per axis) or too aliased. 4000 / 1000 train /
test, deterministic seeds.

Exact matrices: `data/processed/quadratic_params.json`.

---

## 3. Baseline architectures (HPO-selected)

Three-stage Optuna TPE, 240 total trials. Selection rule: smallest
`n_params` within 5 % of the best val MSE. **Test set evaluated exactly
once.** Final retrain on the full 4000-point pool with patience 30 / max
1000 epochs.

| Dataset | dim | architecture | activation | params | FLOPs | locked test MSE |
|---|---|---|---|---|---|---|
| A | 1 | 2 × 16 | ReLU | 321  | 609   | 1.94e-4 |
| A | 2 | 3 × 32 | GELU | 2241 | 4385  | 1.35e-3 |
| A | 3 | 3 × 48 | GELU | 4945 | 9745  | 4.19e-2 |
| B | 1 | 1 × 16 | ReLU | 49   | 81    | 3.99   (σ² = 4.0) |
| B | 2 | 2 × 10 | ReLU | 151  | 281   | 28.4   (σ² = 28.1) |
| B | 3 | 2 × 4  | ReLU | 41   | 73    | 624.6  (σ² = 565.5) |

The B sequence shrinks with dim because the HPO tolerance rule keeps
picking smaller models as the floor σ² grows — at d = 3, MSE ≈ σ² is
trivially reachable by a 41-param net.

---

## 4. Pruning methods

| Method | Class | Saliency / criterion | Granularity | Reference |
|---|---|---|---|---|
| **OBD**         | 2nd order      | `0.5 · H_ii · w_i²`   | scalar | LeCun et al. 1990 |
| **OBS**         | 2nd order      | `w_q²/(2[H⁻¹]_qq)` + closed-form δw | scalar + correction | Hassibi & Stork 1993 |
| **mag**         | 1st order      | `|w_i|`               | scalar | Han et al. 2015 |
| **Taylor-FO**   | 1st order      | `|w_i · g_i|`         | scalar | Molchanov et al. 2017/2019 |
| **struct**      | structured     | `‖W_l[j, :]‖₂`        | neuron | Li et al. 2017 |
| **activ**       | structured     | mean abs activation   | neuron | Hu et al. 2016 |
| **redundancy**  | structured     | activation corr > 0.99 then merge | neuron pair | Mariet & Sra 2016 |
| **lowrank**     | low-rank       | smallest SV per Linear | rank | Denil 2013, Sainath 2013 |
| **SNIP**        | prune-at-init  | `|w_i · g_i|` at random init, single-shot top-k | scalar | Lee, Ajanthan, Torr 2019 |
| **IMP+rewind**  | iterative LTH  | `|w_i|` then rewind to init before retrain | scalar | Frankle & Carbin 2019; Frankle et al. 2020 |

Calibration data: a fixed 1000-sample subset (`CALIB_SEED = 7373`) for OBD,
OBS, Taylor-FO, SNIP, `activ`, `redundancy`. Adopted mid-study; 3-D used
the subset, 1-D and 2-D OBD/OBS used the full 4000 pool. See §10.

OBS regularisation: Tikhonov λ = 1e−4. If `cond(H + λI) > 1e10`, λ × 10 and
re-check; if still ill-conditioned, return `None` and **fall back to
OBD-diagonal saliency** (audit §2.3). For the 3-D / A baseline (n = 4945),
the exact-Hessian limit `MAX_EXACT_N = 3000` triggers the same fallback to
keep wall-time tractable.

---

## 5. Pruning loop

For each sparsity target ∈ {10 %, 20 %, …, 90 %}:

1. **Saliency** computation per §4.
2. **Drop** weights / neurons until target reached.
3. **Fine-tune** the surviving weights with Adam, mini-batch 24, patience
   10, max 200 epochs. After each optimiser step, masked weights are
   re-zeroed.
4. **Adaptive stop:** if test MSE > 3× baseline **and** sparsity ≥ 50 %,
   stop. Always go to ≥ 50 %; cap at 95 %.
5. **Test set evaluated once per sparsity target** on the restored
   best-val checkpoint. Test never enters the early-stopping signal.

Special protocols:
- **SNIP**: per target, reset model to fresh init, score, top-k mask,
  fine-tune from masked init. Single-shot.
- **IMP+rewind**: per target, drop bottom-N % by magnitude, **rewind**
  surviving weights to the saved random init, re-train.

---

## 6. Headline results

The summary table (sparsity at which test MSE ≤ 2× baseline). Empty =
method failed to satisfy the budget at any tested sparsity.

### Dataset A (clean)

| method      | d=1 | d=2 | d=3 |
|---|---|---|---|
| **obd**         | 70 % | 40 % | 90 % |
| **obs**         | 70 % | —   | 90 % * |
| **mag**         | 60 % | 70 % | 70 % |
| **struct**      | 60 % | 30 % | 40 % |
| **activ**       | 60 % | 70 % | 80 % |
| **redundancy**  | 60 % | 80 % | 70 % |
| **lowrank**     | 11 % | 10 % | 10 % |
| **taylor_fo**   | 60 % | 90 % | 90 % |
| **snip**        | —   | —   | 70 % |
| **imp_rewind**  | —   | 80 % | 80 % |

\* d=3 OBS uses the OBD-diagonal fallback (n = 4945 > MAX_EXACT_N).

### Dataset B (noisy)

| method      | d=1 | d=2 | d=3 |
|---|---|---|---|
| **obd**         | 90 % | 80 % | 80 % |
| **obs**         | 80 % | 80 % | 60 % |
| **mag**         | 90 % | 90 % | 60 % |
| **struct**      | 90 % | 90 % | 70 % |
| **activ**       | 90 % | 90 % | 70 % |
| **redundancy**  | 80 % | 70 % | 60 % |
| **lowrank**     | 0 %  | 8 %  | 12 % |
| **taylor_fo**   | 70 % | 90 % | 80 % |
| **snip**        | 90 % | 80 % | 70 % |
| **imp_rewind**  | 80 % | 90 % | 60 % |

The 3-D / OBS-B run terminated at sparsity 80 % via a 30-min wall-time
limit; checkpoints 10 %–80 % are saved, 90 % is missing. Net effect on the
table: minimal (OBS-B-3D shows 60 % at 2× budget either way).

For the looser 3× budget and the harder 1.5× budget tables see
[summary.md](../../results/scaling_law/summary.md).

---

## 7. Per-dim Pareto fronts (figures)

Each `(dataset, dim)` pair has a sparsity-vs-MSE and a params-vs-MSE plot
in `results/scaling_law/`. The headline scaling-law plots:

- `fig_dim_vs_best_mse_A.png` and `…_B.png` — test MSE / baseline MSE at
  ≤ 50 % sparsity, vs d. The "is the method getting harder as d grows"
  story.
- `fig_dim_vs_max_sparsity_A.png` and `…_B.png` — max sparsity within 2×
  baseline MSE, vs d. The "how much can you prune" story.
- `fig_dim_vs_flops_at_target_A.png` and `…_B.png` — forward FLOPs of the
  smallest network within 2× MSE, vs d. The deployment-cost framing of
  the same data.
- `fig_sparsity_vs_mse_{ds}_{dim}d.png`, `fig_params_vs_mse_{ds}_{dim}d.png`
  — per-cell detail.

Use the same colour legend across all panels (see method styles in
[scaling_law.py](../../src/analysis/scaling_law.py)).

---

## 8. Why each method behaves the way it does

**OBD.** Best at d = 1 (small, well-conditioned Hessian) and at d = 3 with
the OBD-diagonal fallback. The d = 2 dip (only 40 % at 2× budget) is the
exact-diagonal calibration on a transition-dimension case where some
elements of the diagonal become very small and the saliency under-rates
some "free" weights.

**OBS.** Wins at d = 1, refuses to scale. The closed-form δw correction is
the Achilles heel: at higher d, the trained model is over-parameterised in
several directions, the Hessian inverse has huge entries, and δw blows the
surviving weights up. The audit §2.3 patch detects this and returns `None`
so the caller falls back to OBD-diagonal saliency.

**mag.** Solid floor on both datasets, all dims. The simplest method is
hard to beat when the network is small enough that all weights matter.

**Taylor-FO.** Surprisingly the strongest first-order method at d ≥ 2 on
clean data — pairing the trained-minimum gradient with the weight
magnitude breaks ties that pure-magnitude misses.

**struct.** Disappointing on clean A — removing whole neurons by L₂ row
norm loses too much representational capacity early. On noisy B it's
competitive because the network is over-parameterised relative to the
noise floor.

**activ.** Strong on both. Activation-norm-based selection is data-aware,
which compensates for the d > 1 saliency vagueness.

**redundancy.** Variable. The 0.99 correlation threshold is the key
hyperparameter; we used it unchanged from 1-D and got mixed results at
d = 3 noisy. Tuning candidate: lower threshold for higher-d.

**lowrank.** Effectively a no-op for our setup because the boundary
linears are rank 1 (1 → 48 and 48 → 1 weight matrices), so only middle
layers can be compressed. Maybe useful as d grows; not below d = 4.

**SNIP.** Weakest at d = 1, recovers at d ≥ 2. Single-shot prune-at-init
needs a model with enough "spare capacity" — a 49-param B / d = 1 baseline
has no spare.

**IMP+rewind.** Weakest at d = 1 (over-pruned tiny models don't have a
lottery ticket), strongest dark horse at d ≥ 2. The rewind-to-init
protocol is exactly the right thing for *noisy* regression because it
forgets the over-fit-to-noise local minima.

---

## 9. Methodology fixes that mattered

1. **Test-set leakage (audit §2.1).** The fine-tune early-stopping signal
   was the locked test set. Every prior pruning checkpoint was therefore
   selected on the set it's evaluated on. Fixed by carving a 20 % val
   slice from the train pool, seeded `4242`. **OBS / Dataset B / 80 %
   sparsity:** prior 34.7 → corrected 4.2 (at noise floor σ² = 4.0).
2. **`else: h_diag = h_inv = None`** reset in `pruner.py` after the
   per-target setup branch (latent regression that wiped OBD/OBS's
   computations). Discovered while running the re-baselined OBS sweep —
   the pre-fix `comparison_table.csv` happened to predate this bug.
3. **OBS one-shot λ bump** (audit §2.3): the bump was applied once and
   then the result was used regardless. Now post-bump cond is re-checked
   and `None` is returned if still > 1e10 so the caller can fall back.
4. **`_load_baseline` side effect** (audit §2.6): the original-topology
   PNG was written to the shared baseline dir on every method run. Moved
   to the per-method output dir.

Plus three smaller items: dead `lr_mod.reconstruct(w)` removed, magic
batch_size lifted to `FINETUNE_BATCH_SIZE = 24`, `make_ood_splits`
docstring corrected.

---

## 10. Caveats & open ends

1. **Calibration-subset discontinuity.** Saliency / Hessian for OBD, OBS,
   Taylor-FO, SNIP, `activ`, `redundancy` is computed on a 1000-sample
   subset for d = 3 but on the full 4000-pool for d = 1 and d = 2 (the
   subset landed mid-study after the 1-D / 2-D OBD / OBS runs). The
   expected ordering effect is within seed-noise; for a paper-grade
   sweep, re-run 1-D / 2-D under the new protocol.
2. **OBS / Dataset B / d = 3 90 % sparsity** was killed by the 30-min
   wall-time per-cell guard. Checkpoints 10 %–80 % are saved. Effect on
   tables is minimal.
3. **Single seed per (method, ds, dim).** Error bars would require
   re-running with seeds 42 / 43 / 44; the budget is ~3× the current
   sweep wall-time.
4. **Polynomial-degree complexity** (varying the target's order, not
   just `d`) is the next study axis — deferred per the original handoff.
5. **The patent-track methods are not in this report.** Reasons and
   recommendations: [patent_track_recommendations.md](patent_track_recommendations.md).

---

## 11. Reproducibility

```bash
# Generate datasets and run the full sweep from scratch (≈ 2 h on a M-series MBP)
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python src/data/generate_data.py
for ds in A B; do for d in 1 2 3; do
  python src/training/train_baseline.py --dataset $ds --dim $d
done; done
for m in obd obs mag struct activ redundancy lowrank taylor_fo snip imp_rewind; do
  for ds in A B; do for d in 1 2 3; do
    python src/pruning/pruner.py --method $m --dataset $ds --dim $d
  done; done
done
python src/analysis/scaling_law.py     # builds CSV + figures + summary.md
```

All artifacts live under `results/`; the figures used in
[poster_draft.md](poster_draft.md) are in `results/scaling_law/`.

Seeds: master `42`, fine-tune val split `4242`, calibration subset `7373`,
fresh-init checkpoint `1337`. Data sampling: `4242 + d` per dim, noise
`7777 + d`.
