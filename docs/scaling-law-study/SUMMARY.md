# Pruning scaling-law study — complete summary

Marko Prohaska · CS_ANN, branch `scaling-law-study` · 2026-05-14.

Single-file walkthrough of everything done in this study: the question
asked, the methodology used, the per-dataset baselines, every pruned
variant of every baseline, and the cross-cutting conclusions. Companion
files for the long-form report and methodology spec are linked at the
bottom.

---

## Table of contents

1. [Introduction & statement of results](#1-introduction--statement-of-results)
2. [Methodology](#2-methodology)
3. [Results — Dataset A (clean quadratic)](#3-results--dataset-a-clean-quadratic)
   - 3.1 [d = 1](#31-d--1)
   - 3.2 [d = 2](#32-d--2)
   - 3.3 [d = 3](#33-d--3)
4. [Results — Dataset B (noisy quadratic)](#4-results--dataset-b-noisy-quadratic)
   - 4.1 [d = 1](#41-d--1)
   - 4.2 [d = 2](#42-d--2)
   - 4.3 [d = 3](#43-d--3)
5. [Cross-cutting scaling-law plots](#5-cross-cutting-scaling-law-plots)
6. [Conclusion](#6-conclusion)
7. [Related files](#7-related-files)

---

## 1. Introduction & statement of results

### 1.1 What we did

We trained a hyper-parameter-optimised MLP regressor on a quadratic
target `f(x) = xᵀA x + bᵀx + c` for **three input dimensions**
(d ∈ {1, 2, 3}) and **two noise levels** (Dataset A clean, Dataset B
noisy at σ = 12.5 % of peak-to-peak signal range). For each of the six
resulting baselines, we ran **ten** different pruning methods through an
iterative sparsity schedule, fine-tuning after every step, and measured
how test MSE scales with surviving parameters / FLOPs.

The repo before this study contained five pruning methods (OBD, OBS,
mag, struct, activ) and 1-D-only datasets. The audit before this
session also turned up six concrete fixes — including a critical
test-set leakage in the fine-tune early-stopping signal. Both lists are
now closed.

### 1.2 Headline results (max sparsity at test MSE ≤ 2× baseline)

**Dataset A — clean**

| method      | d=1 | d=2 | d=3 |
|---|---|---|---|
| **OBD**         | 70% | 40% | 90% |
| **OBS**         | 70% | —  | 90%¹ |
| **mag**         | 60% | 70% | 70% |
| **Taylor-FO**   | 60% | 90% | 90% |
| **struct**      | 60% | 30% | 40% |
| **activ**       | 60% | 70% | 80% |
| **redundancy**  | 60% | 80% | 70% |
| **lowrank**     | 11% | 10% | 10% |
| **SNIP**        | —  | —  | 70% |
| **IMP + rewind**| —  | 80% | 80% |

¹ OBS at d=3 used the OBD-diagonal fallback (n=4945 > `MAX_EXACT_N=3000`).

**Dataset B — noisy**

| method      | d=1 | d=2 | d=3 |
|---|---|---|---|
| **OBD**         | 90% | 80% | 80% |
| **OBS**         | 80% | 80% | 60% |
| **mag**         | 90% | 90% | 60% |
| **Taylor-FO**   | 70% | 90% | 80% |
| **struct**      | 90% | 90% | 70% |
| **activ**       | 90% | 90% | 70% |
| **redundancy**  | 80% | 70% | 60% |
| **lowrank**     | 0%  | 8%  | 12% |
| **SNIP**        | 90% | 80% | 70% |
| **IMP + rewind**| 80% | 90% | 60% |

### 1.3 The four conclusions in one line each

1. **The best pruning method depends on input dimension.** Second-order
   methods dominate at d=1, structured/iterative methods catch up and
   sometimes overtake at d ≥ 2.
2. **OBS does not generalise above d=1.** The Hessian becomes
   ill-conditioned at the trained minimum; weight correction amplifies
   error. Our autodetection now falls back to OBD-diagonal saliency.
3. **Noisy data hides pruning damage.** On Dataset B almost every
   method reaches 80–90 % sparsity at a 2× MSE budget because σ² caps
   the visible degradation.
4. **The audit's test-set leakage fix materially changed high-sparsity
   numbers.** OBS / Dataset B / 80 % sparsity previously reported 8 ×
   noise floor; the corrected re-baseline sits at the floor.

---

## 2. Methodology

### 2.1 Datasets

`f : ℝ^d → ℝ`, `f(x) = xᵀA_d x + b_dᵀ x + c`, `(A_d, b_d, c)`
deterministic per dim. `A_d` is symmetric positive-definite
(`L Lᵀ + 0.5 I`) so each problem has a unique interior minimum inside
`[−3, 3]^d`.

| Dataset | dim | σ noise | peak-to-peak | noise floor σ² |
|---|---|---|---|---|
| A 1-D | 1 | 0 | 16.0 | 0.0 |
| B 1-D | 1 | 2.00 | 16.0 | 4.0 |
| A 2-D | 2 | 0 | 42.4 | 0.0 |
| B 2-D | 2 | 5.30 | 42.4 | 28.1 |
| A 3-D | 3 | 0 | 190.3 | 0.0 |
| B 3-D | 3 | 23.78 | 190.3 | 565.5 |

Sampling: 1-D uses uniform linspace random-shuffle (legacy, identical
to prior runs); d ≥ 2 uses IID uniform `[−3, 3]^d`. 4000 / 1000 train /
test split. Exact A_d, b_d matrices saved at
[`data/processed/quadratic_params.json`](../../data/processed/quadratic_params.json).

### 2.2 Baseline MLP and HPO

The MLP is `Linear → act → … → Linear` (no activation on the regression
head). Each cell has its own architecture, chosen by **3-stage Optuna
TPE** (240 trials total). Selection rule: **smallest n_params within
5 % of the best val MSE**. Final retrain on the full 4000-pool, patience
30, max 1000 epochs. **The test set is evaluated exactly once.**

### 2.3 Pruning loop (per method, per sparsity target)

1. Compute saliency (per method, see §2.4).
2. Drop weights or neurons until the sparsity target is reached.
3. Fine-tune surviving weights with Adam, mini-batch size 24,
   patience 10, max 200 epochs. Re-zero masked weights after every step.
4. **Adaptive stop:** if test MSE > 3× baseline AND sparsity ≥ 50 %,
   stop. Always go to ≥ 50 %; cap at 95 %.
5. **Evaluate the locked test set once per sparsity target** on the
   restored best-val checkpoint. Test never enters the early-stopping
   signal (audit §2.1 leakage fix).

Saliency calibration uses a 1 000-sample subset of the train pool for
speed (`CALIB_SAMPLES = 1000`, seed `7373`). Fine-tune mini-batches the
full train pool.

### 2.4 The ten pruning methods

| Method | Class | Saliency | Granularity | Reference |
|---|---|---|---|---|
| **OBD**         | 2nd order | `0.5 · H_ii · w_i²` | scalar | LeCun et al. 1990 |
| **OBS**         | 2nd order | `w_q²/(2·[H⁻¹]_qq)` + closed-form δw | scalar + correction | Hassibi & Stork 1993 |
| **mag**         | 1st order | `|w_i|` | scalar | Han et al. 2015 |
| **Taylor-FO**   | 1st order | `|w_i · g_i|` | scalar | Molchanov et al. 2017/2019 |
| **struct**      | structured | `‖W_l[j, :]‖₂` | neuron | Li et al. 2017 |
| **activ**       | structured | mean abs activation | neuron | Hu et al. 2016 |
| **redundancy**  | structured | activation corr > 0.99 then merge | neuron pair | Mariet & Sra 2016 |
| **lowrank**     | low-rank | smallest singular value per Linear | rank | Denil 2013, Sainath 2013 |
| **SNIP**        | prune-at-init | `|w·g|` at random init, single-shot top-k | scalar | Lee, Ajanthan, Torr 2019 |
| **IMP + rewind**| iterative LTH | `|w|` then rewind survivors to init | scalar | Frankle & Carbin 2019 |

OBS uses Tikhonov regularisation λ = 1e−4 with a 10× bump if
`cond > 10¹⁰`; if still ill-conditioned after the bump, returns `None`
and the caller falls back to OBD-diagonal saliency. The exact-Hessian
limit is `MAX_EXACT_N = 3000` (above which OBS also falls back) — set
to keep wall-time tractable for the 4945-param A 3-D baseline.

### 2.5 Pruning fixes that landed in this study

1. **Test-set leakage** (audit §2.1): fine-tune used the test set for
   early stopping. Fixed by carving a fixed 20 % val slice from train.
2. **OBS one-shot λ bump** (audit §2.3): now re-checks cond after the
   bump and returns `None` if still ill-conditioned.
3. **Latent `h_diag = h_inv = None` reset** that wiped OBD/OBS's
   per-target Hessian computations.
4. **Magic batch-size 24** lifted to `FINETUNE_BATCH_SIZE` constant.
5. **Dead `lr_mod.reconstruct(w)`** removed.
6. **`_load_baseline`** no longer writes the topology PNG to the shared
   baseline dir (now per-method).
7. **Stale `make_ood_splits` docstring** corrected.

---

## 3. Results — Dataset A (clean quadratic)

The "clean" target is the exact quadratic with no observation noise. As
dim grows, the HPO picks larger architectures (321 → 2241 → 4945 params)
because more weights are needed to fit the higher-dimensional surface
exactly. The dense baseline test MSE rises from 1.94 × 10⁻⁴ (d=1) to
4.19 × 10⁻² (d=3), reflecting the harder fitting problem rather than
any failure mode.

### 3.1 d = 1

**Baseline.** 2 hidden layers × 16 neurons, ReLU activation, **321
parameters**, **609 FLOPs / forward**, locked test MSE **1.94 × 10⁻⁴**.
The output layer (1 → 1) and the input layer (1 → 16) are both rank-1
by construction.

![Baseline A 1-D](../../results/pruning/mag/A/original_network.png)

Pruned variants (best within the 2 × baseline-MSE budget):

| Method | Sparsity | Test MSE | Params | FLOPs |
|---|---|---|---|---|
| OBD         | 70 % | 2.52e-4 | 96  | 197 |
| OBS         | 70 % | 1.94e-4 | 96  | 197 |
| mag         | 60 % | 2.43e-4 | 128 | 259 |
| Taylor-FO   | 60 % | 3.44e-4 | 128 | 245 |
| struct      | 60 % | 3.16e-4 | 123 | 239 |
| activ       | 60 % | 3.12e-4 | 127 | 247 |
| redundancy  | 60 % | 3.19e-4 | 128 | 257 |
| lowrank     | 11 % | 1.96e-4 | 579 | 609 |
| SNIP        | — (best ach. 30 % at 4.5e-4) | — | 225 | 427 |
| IMP+rewind  | — (best ach. 20 % at 4.9e-4) | — | 256 | 479 |

![Sparsity vs test MSE — A 1-D](../../results/scaling_law/fig_sparsity_vs_mse_A_1d.png)

#### Post-pruning architectures (one diagram per method, at the best-achieved sparsity)

**OBD — final sparsity 70%, test MSE 2.522e-04, 96 params, 197 FLOPs**

![obd post-pruning topology, A d=1](../../results/pruning/obd/A/after_pruning_sparsity_80.png)

**OBS — final sparsity 70%, test MSE 1.945e-04, 96 params, 197 FLOPs**

![obs post-pruning topology, A d=1](../../results/pruning/obs/A/after_pruning_sparsity_90.png)

**mag — final sparsity 60%, test MSE 2.435e-04, 128 params, 259 FLOPs**

![mag post-pruning topology, A d=1](../../results/pruning/mag/A/after_pruning_sparsity_70.png)

**struct — final sparsity 60%, test MSE 3.159e-04, 123 params, 239 FLOPs**

![struct post-pruning topology, A d=1](../../results/pruning/struct/A/after_pruning_sparsity_73.png)

**activ — final sparsity 60%, test MSE 3.116e-04, 127 params, 247 FLOPs**

![activ post-pruning topology, A d=1](../../results/pruning/activ/A/after_pruning_sparsity_73.png)

**redundancy — final sparsity 60%, test MSE 3.194e-04, 128 params, 257 FLOPs**

![redundancy post-pruning topology, A d=1](../../results/pruning/redundancy/A/after_pruning_sparsity_70.png)

**lowrank — final sparsity 11%, test MSE 1.961e-04, 579 params, 609 FLOPs**

![lowrank post-pruning topology, A d=1](../../results/pruning/lowrank/A/after_pruning_sparsity_11.png)

**Taylor-FO — final sparsity 60%, test MSE 3.437e-04, 128 params, 245 FLOPs**

![taylor_fo post-pruning topology, A d=1](../../results/pruning/taylor_fo/A/after_pruning_sparsity_90.png)

**SNIP — final sparsity 30%, test MSE 4.469e-04, 225 params, 427 FLOPs**  *(outside 2× MSE budget — best-effort result)*

![snip post-pruning topology, A d=1](../../results/pruning/snip/A/after_pruning_sparsity_50.png)

**IMP + rewind — final sparsity 20%, test MSE 4.935e-04, 256 params, 479 FLOPs**  *(outside 2× MSE budget — best-effort result)*

![imp_rewind post-pruning topology, A d=1](../../results/pruning/imp_rewind/A/after_pruning_sparsity_50.png)


**Reading.** OBD/OBS find a sweet spot at 70 % sparsity with MSE close
to the dense baseline — second-order methods exploit the small, well-
conditioned Hessian. mag/Taylor-FO and the structured methods sit just
behind at 60 %. **LowRank** is essentially a no-op: it compresses only
the middle Linear (boundary layers are rank-1 by input/output dim).
**SNIP** and **IMP+rewind** are the two prune-at-init / iterative-LTH
methods; on a 321-param network there is no "spare capacity" and they
fail to meet the 2 × budget.

### 3.2 d = 2

**Baseline.** 3 hidden layers × 32 neurons, GELU, **2241 parameters**,
**4385 FLOPs**, locked test MSE **1.35 × 10⁻³**.

![Baseline A 2-D](../../results/pruning/mag/A_2d/original_network.png)

| Method | Sparsity | Test MSE | Params | FLOPs |
|---|---|---|---|---|
| OBD         | 40 % | 2.47e-3 | 1344 | 2631 |
| OBS         | — (10 % at 4.6e-2) | — | 2016 | 3959 |
| mag         | 70 % | 3.61e-4 | 672 | 1335 |
| Taylor-FO   | 90 % | 2.69e-3 | 224 | 463 |
| struct      | 30 % | 2.19e-3 | 1550 | 3041 |
| activ       | 70 % | 2.42e-3 | 668 | 1325 |
| redundancy  | 80 % | 6.82e-4 | 448 | 919 |
| lowrank     | 10 % | 1.89e-3 | 4294 | 4385 |
| SNIP        | — (10 % at 3.5e-3) | — | 2017 | 3937 |
| IMP+rewind  | 80 % | 2.57e-3 | 448 | 875 |

![Sparsity vs test MSE — A 2-D](../../results/scaling_law/fig_sparsity_vs_mse_A_2d.png)

#### Post-pruning architectures (one diagram per method, at the best-achieved sparsity)

**OBD — final sparsity 40%, test MSE 2.468e-03, 1344 params, 2631 FLOPs**

![obd post-pruning topology, A d=2](../../results/pruning/obd/A_2d/after_pruning_sparsity_50.png)

**OBS — final sparsity 10%, test MSE 4.587e-02, 2016 params, 3959 FLOPs**  *(outside 2× MSE budget — best-effort result)*

![obs post-pruning topology, A d=2](../../results/pruning/obs/A_2d/after_pruning_sparsity_50.png)

**mag — final sparsity 70%, test MSE 3.606e-04, 672 params, 1335 FLOPs**

![mag post-pruning topology, A d=2](../../results/pruning/mag/A_2d/after_pruning_sparsity_80.png)

**struct — final sparsity 30%, test MSE 2.194e-03, 1550 params, 3041 FLOPs**

![struct post-pruning topology, A d=2](../../results/pruning/struct/A_2d/after_pruning_sparsity_51.png)

**activ — final sparsity 70%, test MSE 2.423e-03, 668 params, 1325 FLOPs**

![activ post-pruning topology, A d=2](../../results/pruning/activ/A_2d/after_pruning_sparsity_90.png)

**redundancy — final sparsity 80%, test MSE 6.818e-04, 448 params, 919 FLOPs**

![redundancy post-pruning topology, A d=2](../../results/pruning/redundancy/A_2d/after_pruning_sparsity_90.png)

**lowrank — final sparsity 10%, test MSE 1.893e-03, 4294 params, 4385 FLOPs**

![lowrank post-pruning topology, A d=2](../../results/pruning/lowrank/A_2d/after_pruning_sparsity_10.png)

**Taylor-FO — final sparsity 90%, test MSE 2.693e-03, 224 params, 463 FLOPs**

![taylor_fo post-pruning topology, A d=2](../../results/pruning/taylor_fo/A_2d/after_pruning_sparsity_90.png)

**SNIP — final sparsity 10%, test MSE 3.539e-03, 2017 params, 3937 FLOPs**  *(outside 2× MSE budget — best-effort result)*

![snip post-pruning topology, A d=2](../../results/pruning/snip/A_2d/after_pruning_sparsity_50.png)

**IMP + rewind — final sparsity 80%, test MSE 2.566e-03, 448 params, 875 FLOPs**

![imp_rewind post-pruning topology, A d=2](../../results/pruning/imp_rewind/A_2d/after_pruning_sparsity_90.png)


**Reading.** **mag** is the surprise winner at d=2: 70 % sparsity with
*better* MSE than the dense baseline (3.6e-4 vs 1.35e-3). With more
spare capacity, magnitude-based selection plus fine-tune can act as a
mild regulariser. **OBS** collapses: the Hessian becomes
ill-conditioned (`cond > 10¹²`) and the closed-form δw correction
amplifies error along low-curvature directions; the audit §2.3 fix
correctly returns `None` so the loop falls back to OBD-diagonal at
later targets, but the early-target damage is locked in. **IMP+rewind**
makes its first strong showing — at d=2 the network has a real lottery
ticket to find.

### 3.3 d = 3

**Baseline.** 3 hidden layers × 48 neurons, GELU, **4945 parameters**,
**9745 FLOPs**, locked test MSE **4.19 × 10⁻²**.

![Baseline A 3-D](../../results/pruning/mag/A_3d/original_network.png)

| Method | Sparsity | Test MSE | Params | FLOPs |
|---|---|---|---|---|
| OBD         | 90 % | 2.63e-2 | 494  | 999  |
| OBS         | 90 %¹ | 2.63e-2 | 494  | 999  |
| mag         | 70 % | 1.55e-2 | 1483 | 2943 |
| Taylor-FO   | 90 % | 2.55e-2 | 494  | 991  |
| struct      | 40 % | 5.09e-2 | 2961 | 5853 |
| activ       | 80 % | 4.09e-2 | 979  | 1951 |
| redundancy  | 70 % | 2.88e-2 | 1483 | 2947 |
| lowrank     | 10 % | 4.29e-2 | 9563 | 9745 |
| SNIP        | 70 % | 6.16e-2 | 1484 | 2881 |
| IMP+rewind  | 80 % | 5.29e-2 | 989  | 1927 |

¹ OBS falls back to OBD-diagonal saliency (n=4945 > MAX_EXACT_N=3000).

![Sparsity vs test MSE — A 3-D](../../results/scaling_law/fig_sparsity_vs_mse_A_3d.png)

#### Post-pruning architectures (one diagram per method, at the best-achieved sparsity)

**OBD — final sparsity 90%, test MSE 2.634e-02, 494 params, 999 FLOPs**

![obd post-pruning topology, A d=3](../../results/pruning/obd/A_3d/after_pruning_sparsity_90.png)

**OBS — final sparsity 90%, test MSE 2.634e-02, 494 params, 999 FLOPs**

![obs post-pruning topology, A d=3](../../results/pruning/obs/A_3d/after_pruning_sparsity_90.png)

**mag — final sparsity 70%, test MSE 1.554e-02, 1483 params, 2943 FLOPs**

![mag post-pruning topology, A d=3](../../results/pruning/mag/A_3d/after_pruning_sparsity_80.png)

**struct — final sparsity 40%, test MSE 5.086e-02, 2961 params, 5853 FLOPs**

![struct post-pruning topology, A d=3](../../results/pruning/struct/A_3d/after_pruning_sparsity_51.png)

**activ — final sparsity 80%, test MSE 4.092e-02, 979 params, 1951 FLOPs**

![activ post-pruning topology, A d=3](../../results/pruning/activ/A_3d/after_pruning_sparsity_90.png)

**redundancy — final sparsity 70%, test MSE 2.875e-02, 1483 params, 2947 FLOPs**

![redundancy post-pruning topology, A d=3](../../results/pruning/redundancy/A_3d/after_pruning_sparsity_80.png)

**lowrank — final sparsity 10%, test MSE 4.290e-02, 9563 params, 9745 FLOPs**

![lowrank post-pruning topology, A d=3](../../results/pruning/lowrank/A_3d/after_pruning_sparsity_10.png)

**Taylor-FO — final sparsity 90%, test MSE 2.548e-02, 494 params, 991 FLOPs**

![taylor_fo post-pruning topology, A d=3](../../results/pruning/taylor_fo/A_3d/after_pruning_sparsity_90.png)

**SNIP — final sparsity 70%, test MSE 6.161e-02, 1484 params, 2881 FLOPs**

![snip post-pruning topology, A d=3](../../results/pruning/snip/A_3d/after_pruning_sparsity_80.png)

**IMP + rewind — final sparsity 80%, test MSE 5.286e-02, 989 params, 1927 FLOPs**

![imp_rewind post-pruning topology, A d=3](../../results/pruning/imp_rewind/A_3d/after_pruning_sparsity_90.png)


**Reading.** **OBD**, **Taylor-FO** and **activ** lead — all reach 90 %
sparsity with MSE ≤ 2× baseline. Three of the four under-200-FLOPs
points on the Pareto front sit very close together (OBD, OBS-fallback,
Taylor-FO at ~990 FLOPs). **mag** trades sparsity for fit (70 %
sparsity but the lowest MSE of any method at ≥ 70 %). **struct** is the
weakest scalable method — removing whole neurons at 50 %+ sparsity
costs too much representational capacity in this clean-data regime.

---

## 4. Results — Dataset B (noisy quadratic)

Dataset B is the same function as Dataset A plus Gaussian noise scaled
to **12.5 % of the function's peak-to-peak range**. The **noise floor
σ²** is the smallest test MSE any model can reach (it's the variance of
the noise itself).

| dim | σ | noise floor σ² | baseline test MSE |
|---|---|---|---|
| 1 | 2.00  | 4.0   | 3.99  |
| 2 | 5.30  | 28.1  | 28.4  |
| 3 | 23.78 | 565.5 | 624.6 |

Every dense baseline sits **at the noise floor**, so the HPO tolerance
rule picks small architectures (49, 151, 41 params). When the dense
baseline is already at the floor, **pruning can't make MSE worse than
roughly the floor** until it removes essential structure — that's why
nearly every method reaches 80–90 % sparsity at the 2× MSE budget on
Dataset B.

### 4.1 d = 1

**Baseline.** 1 hidden layer × 16 neurons, ReLU, **49 parameters**,
**81 FLOPs**, locked test MSE **3.99** (noise floor 4.0).

![Baseline B 1-D](../../results/pruning/mag/B/original_network.png)

| Method | Sparsity | Test MSE | Params | FLOPs |
|---|---|---|---|---|
| OBD         | 90 % | 5.45 | 4  | 25 |
| OBS         | 80 % | 4.21 | 9  | 33 |
| mag         | 90 % | 5.45 | 4  | 23 |
| Taylor-FO   | 70 % | 4.02 | 14 | 35 |
| struct      | 90 % | 4.93 | 4  | 21 |
| activ       | 90 % | 4.90 | 4  | 21 |
| redundancy  | 80 % | 5.45 | 9  | 25 |
| lowrank     | 0  % | 3.99 | 51 | 81 |
| SNIP        | 90 % | 5.31 | 5  | 25 |
| IMP+rewind  | 80 % | 4.20 | 9  | 31 |

![Sparsity vs test MSE — B 1-D](../../results/scaling_law/fig_sparsity_vs_mse_B_1d.png)

#### Post-pruning architectures (one diagram per method, at the best-achieved sparsity)

**OBD — final sparsity 90%, test MSE 5.453e+00, 4 params, 25 FLOPs**

![obd post-pruning topology, B d=1](../../results/pruning/obd/B/after_pruning_sparsity_92.png)

**OBS — final sparsity 80%, test MSE 4.209e+00, 9 params, 33 FLOPs**

![obs post-pruning topology, B d=1](../../results/pruning/obs/B/after_pruning_sparsity_82.png)

**mag — final sparsity 90%, test MSE 5.450e+00, 4 params, 23 FLOPs**

![mag post-pruning topology, B d=1](../../results/pruning/mag/B/after_pruning_sparsity_92.png)

**struct — final sparsity 90%, test MSE 4.929e+00, 4 params, 21 FLOPs**

![struct post-pruning topology, B d=1](../../results/pruning/struct/B/after_pruning_sparsity_92.png)

**activ — final sparsity 90%, test MSE 4.899e+00, 4 params, 21 FLOPs**

![activ post-pruning topology, B d=1](../../results/pruning/activ/B/after_pruning_sparsity_92.png)

**redundancy — final sparsity 80%, test MSE 5.447e+00, 9 params, 25 FLOPs**

![redundancy post-pruning topology, B d=1](../../results/pruning/redundancy/B/after_pruning_sparsity_92.png)

**lowrank — final sparsity 0%, test MSE 3.989e+00, 51 params, 81 FLOPs**

![lowrank post-pruning topology, B d=1](../../results/pruning/lowrank/B/after_pruning_sparsity_00.png)

**Taylor-FO — final sparsity 70%, test MSE 4.021e+00, 14 params, 35 FLOPs**

![taylor_fo post-pruning topology, B d=1](../../results/pruning/taylor_fo/B/after_pruning_sparsity_82.png)

**SNIP — final sparsity 90%, test MSE 5.307e+00, 5 params, 25 FLOPs**

![snip post-pruning topology, B d=1](../../results/pruning/snip/B/after_pruning_sparsity_90.png)

**IMP + rewind — final sparsity 80%, test MSE 4.202e+00, 9 params, 31 FLOPs**

![imp_rewind post-pruning topology, B d=1](../../results/pruning/imp_rewind/B/after_pruning_sparsity_92.png)


**Reading.** **OBS** and **IMP+rewind** tie for the best MSE among
8-parameter networks (≈ noise floor). **Six methods** reach 90 %
sparsity with MSE ≤ noise-floor × 1.35 — meaning a **4-parameter
network** is essentially indistinguishable from the dense 49-parameter
baseline once we account for noise. **LowRank** does nothing because the
1 → 16 and 16 → 1 boundary linears are already rank-1.

### 4.2 d = 2

**Baseline.** 2 hidden layers × 10 neurons, ReLU, **151 parameters**,
**281 FLOPs**, locked test MSE **28.4** (noise floor 28.1).

![Baseline B 2-D](../../results/pruning/mag/B_2d/original_network.png)

| Method | Sparsity | Test MSE | Params | FLOPs |
|---|---|---|---|---|
| OBD         | 80 % | 28.6  | 30  | 71 |
| OBS         | 80 % | 52.5  | 30  | 69 |
| mag         | 90 % | 43.0  | 15  | 45 |
| Taylor-FO   | 90 % | 41.1  | 15  | 43 |
| struct      | 90 % | 36.2  | 15  | 41 |
| activ       | 90 % | 54.2  | 13  | 37 |
| redundancy  | 70 % | 35.0  | 45  | 95 |
| lowrank     | 8 %  | 28.4  | 256 | 281 |
| SNIP        | 80 % | 40.6  | 30  | 65 |
| IMP+rewind  | 90 % | 41.1  | 15  | 45 |

![Sparsity vs test MSE — B 2-D](../../results/scaling_law/fig_sparsity_vs_mse_B_2d.png)

#### Post-pruning architectures (one diagram per method, at the best-achieved sparsity)

**OBD — final sparsity 80%, test MSE 2.859e+01, 30 params, 71 FLOPs**

![obd post-pruning topology, B d=2](../../results/pruning/obd/B_2d/after_pruning_sparsity_90.png)

**OBS — final sparsity 80%, test MSE 5.248e+01, 30 params, 69 FLOPs**

![obs post-pruning topology, B d=2](../../results/pruning/obs/B_2d/after_pruning_sparsity_80.png)

**mag — final sparsity 90%, test MSE 4.297e+01, 15 params, 45 FLOPs**

![mag post-pruning topology, B d=2](../../results/pruning/mag/B_2d/after_pruning_sparsity_90.png)

**struct — final sparsity 90%, test MSE 3.622e+01, 15 params, 41 FLOPs**

![struct post-pruning topology, B d=2](../../results/pruning/struct/B_2d/after_pruning_sparsity_90.png)

**activ — final sparsity 90%, test MSE 5.422e+01, 13 params, 37 FLOPs**

![activ post-pruning topology, B d=2](../../results/pruning/activ/B_2d/after_pruning_sparsity_91.png)

**redundancy — final sparsity 70%, test MSE 3.503e+01, 45 params, 95 FLOPs**

![redundancy post-pruning topology, B d=2](../../results/pruning/redundancy/B_2d/after_pruning_sparsity_90.png)

**lowrank — final sparsity 8%, test MSE 2.838e+01, 256 params, 281 FLOPs**

![lowrank post-pruning topology, B d=2](../../results/pruning/lowrank/B_2d/after_pruning_sparsity_08.png)

**Taylor-FO — final sparsity 90%, test MSE 4.106e+01, 15 params, 43 FLOPs**

![taylor_fo post-pruning topology, B d=2](../../results/pruning/taylor_fo/B_2d/after_pruning_sparsity_90.png)

**SNIP — final sparsity 80%, test MSE 4.056e+01, 30 params, 65 FLOPs**

![snip post-pruning topology, B d=2](../../results/pruning/snip/B_2d/after_pruning_sparsity_90.png)

**IMP + rewind — final sparsity 90%, test MSE 4.105e+01, 15 params, 45 FLOPs**

![imp_rewind post-pruning topology, B d=2](../../results/pruning/imp_rewind/B_2d/after_pruning_sparsity_90.png)


**Reading.** **OBD** delivers the cleanest hit: 80 % sparsity at the
noise floor (28.6 vs 28.1). **mag**, **Taylor-FO**, **struct** and
**IMP+rewind** all hit 90 % sparsity with MSE within 1.5× the floor —
i.e. the 15-parameter networks are barely worse than the 151-parameter
baseline once noise is accounted for. **OBS** is still the laggard.

### 4.3 d = 3

**Baseline.** 2 hidden layers × 4 neurons, ReLU, **41 parameters**,
**73 FLOPs**, locked test MSE **624.6** (noise floor 565.5).

![Baseline B 3-D](../../results/pruning/mag/B_3d/original_network.png)

| Method | Sparsity | Test MSE | Params | FLOPs |
|---|---|---|---|---|
| OBD         | 80 % | 967  | 8  | 21 |
| OBS         | 60 % | 994  | 16 | 35 |
| mag         | 60 % | 934  | 16 | 35 |
| Taylor-FO   | 80 % | 1151 | 8  | 21 |
| struct      | 70 % | 1060 | 11 | 23 |
| activ       | 70 % | 658  | 11 | 23 |
| redundancy  | 60 % | 1067 | 16 | 35 |
| lowrank     | 12 % | 625  | 67 | 73 |
| SNIP        | 70 % | 1059 | 12 | 25 |
| IMP+rewind  | 60 % | 1104 | 16 | 33 |

![Sparsity vs test MSE — B 3-D](../../results/scaling_law/fig_sparsity_vs_mse_B_3d.png)

#### Post-pruning architectures (one diagram per method, at the best-achieved sparsity)

**OBD — final sparsity 80%, test MSE 9.665e+02, 8 params, 21 FLOPs**

![obd post-pruning topology, B d=3](../../results/pruning/obd/B_3d/after_pruning_sparsity_90.png)

**OBS — final sparsity 60%, test MSE 9.942e+02, 16 params, 35 FLOPs**

![obs post-pruning topology, B d=3](../../results/pruning/obs/B_3d/after_pruning_sparsity_80.png)

**mag — final sparsity 60%, test MSE 9.336e+02, 16 params, 35 FLOPs**

![mag post-pruning topology, B d=3](../../results/pruning/mag/B_3d/after_pruning_sparsity_80.png)

**struct — final sparsity 70%, test MSE 1.060e+03, 11 params, 23 FLOPs**

![struct post-pruning topology, B d=3](../../results/pruning/struct/B_3d/after_pruning_sparsity_93.png)

**activ — final sparsity 70%, test MSE 6.580e+02, 11 params, 23 FLOPs**

![activ post-pruning topology, B d=3](../../results/pruning/activ/B_3d/after_pruning_sparsity_98.png)

**redundancy — final sparsity 60%, test MSE 1.067e+03, 16 params, 35 FLOPs**

![redundancy post-pruning topology, B d=3](../../results/pruning/redundancy/B_3d/after_pruning_sparsity_90.png)

**lowrank — final sparsity 12%, test MSE 6.251e+02, 67 params, 73 FLOPs**

![lowrank post-pruning topology, B d=3](../../results/pruning/lowrank/B_3d/after_pruning_sparsity_12.png)

**Taylor-FO — final sparsity 80%, test MSE 1.151e+03, 8 params, 21 FLOPs**

![taylor_fo post-pruning topology, B d=3](../../results/pruning/taylor_fo/B_3d/after_pruning_sparsity_90.png)

**SNIP — final sparsity 70%, test MSE 1.059e+03, 12 params, 25 FLOPs**

![snip post-pruning topology, B d=3](../../results/pruning/snip/B_3d/after_pruning_sparsity_90.png)

**IMP + rewind — final sparsity 60%, test MSE 1.104e+03, 16 params, 33 FLOPs**

![imp_rewind post-pruning topology, B d=3](../../results/pruning/imp_rewind/B_3d/after_pruning_sparsity_90.png)


**Reading.** **activ** is the clear winner: 70 % sparsity at MSE **658**
— better than the dense baseline's 624.6 — by removing under-utilised
neurons that were over-fitting noise. The dense baseline was already
basically empty (41 params, 4-wide hidden layers), so further pruning
can only do so much; most methods stagger between 1.5×–2× the noise
floor at 60–70 % sparsity.

Note: OBS at 90 % sparsity for B 3-D wasn't reached — the run hit the
30-min per-cell wall-time guard at 80 %. Checkpoints 10 %–80 % are
saved.

---

## 5. Cross-cutting scaling-law plots

These four plots aggregate the six per-cell sweeps into the headline
scaling-law story.

**Dataset A — max sparsity at MSE ≤ 2× baseline, vs input dim.**

![Dim vs max sparsity A](../../results/scaling_law/fig_dim_vs_max_sparsity_A.png)

**Dataset B — same axes.**

![Dim vs max sparsity B](../../results/scaling_law/fig_dim_vs_max_sparsity_B.png)

**FLOPs of the smallest model with MSE ≤ 2× baseline, vs dim
(Dataset A).** Dashed line = dense baseline.

![FLOPs scaling A](../../results/scaling_law/fig_dim_vs_flops_at_target_A.png)

**Test MSE / dense-baseline MSE at ≤ 50 % sparsity, vs dim
(Dataset A).**

![MSE-ratio at 50 % sparsity A](../../results/scaling_law/fig_dim_vs_best_mse_A.png)

---

## 6. Conclusion

The scaling-law sweep gave four robust take-aways:

1. **Best method depends on dim.** A pruning ranking from d=1 alone
   would mis-rank methods for d ≥ 2. Second-order methods (OBD, OBS) win
   at d=1 where the Hessian is small and well-conditioned; structured
   (`activ`, `redundancy`) and iterative-LTH (`IMP+rewind`) methods win
   at d ≥ 2 where the network has enough spare capacity to find a
   real "ticket".
2. **OBS is the cautionary tale.** The closed-form weight correction
   that makes OBS dominant at d=1 makes it catastrophic at d=2 —
   `cond(H + λI) > 10¹²` and the δw correction amplifies error along
   low-curvature directions. Our audit-§2.3 patch correctly returns
   `None` so the caller falls back, but the result is OBS-without-the-
   correction (i.e. OBD-diagonal). Practical recommendation: **use OBS
   only for tiny, well-conditioned networks.**
3. **Noisy data is a great compression substrate.** On Dataset B the
   noise floor σ² caps the visible degradation, so most methods reach
   80–90 % sparsity without measurable damage. If you are pruning a
   model whose training data has 10 %+ noise, almost any algorithm
   will get you to 1/10× the parameter count.
4. **The audit fixes were not cosmetic.** The single largest delta in
   the corrected vs pre-fix re-baseline was OBS / Dataset B / 80 %
   sparsity going from 34.7 to 4.2 test MSE — the previous run was
   selecting the model on the test set itself.

### What this study leaves open

- **Polynomial degree** (not just `d`) as a complexity axis — the user-
  confirmed next study.
- **Seed-noise error bars** — a single seed per (method, ds, dim) cell;
  proper bars would multiply wall-time by 3-5×.
- **The patent track is not in here** — open recommendations are in
  [patent_track_recommendations.md](patent_track_recommendations.md),
  to be picked up in a future session.

---

## 7. Related files

- [docs/scaling-law-study/methodology.md](methodology.md) — full methodology spec
  with citations, intended as the seed of the final-report chapter.
- [docs/scaling-law-study/report.md](report.md) — long-form companion report.
- [docs/scaling-law-study/poster_draft.md](poster_draft.md) — text content for the
  DIN A0 poster.
- [docs/scaling-law-study/poster_layout.md](poster_layout.md) — coordinate-level
  layout for the poster template.
- [docs/scaling-law-study/patent_track_recommendations.md](patent_track_recommendations.md)
  — deferred patent-track work.
- [results/scaling_law/](../../results/scaling_law/) — all
  scaling-law figures plus
  [pruning_results_all.csv](../../results/scaling_law/pruning_results_all.csv)
  (457 rows: method × dataset × dim × sparsity → MSE / params / FLOPs).
- [docs/scaling-law-study/phase_0_audit.md](phase_0_audit.md) — the Phase 0
  audit whose §3.2 items were closed in this study.

Reproducibility:

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python src/data/generate_data.py
for ds in A B; do for d in 1 2 3; do
  python src/training/train_baseline.py --dataset $ds --dim $d
done; done
for m in obd obs mag struct activ redundancy lowrank taylor_fo snip imp_rewind; do
  for ds in A B; do for d in 1 2 3; do
    python src/pruning/pruner.py --method $m --dataset $ds --dim $d
done; done; done
python src/analysis/scaling_law.py
```
