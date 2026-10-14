# OBS and GeLU: Why It Can't Be Used, and a Comparison of Four Approaches

This report answers four things: (1) why two of the three baselines use GeLU and one uses tanh; (2) which "Hessian" each of the four OBS approaches uses, how they perform, and their drawbacks; (3) the mathematical procedure of each approach; and (4) **why, on GeLU, "exact-Hessian OBS" is mathematically unusable.**

---

## 1. Why some baselines use GeLU and one uses tanh

**It was not hand-picked — it was chosen automatically by hyperparameter search (HPO).** The three baseline architectures were found with Optuna (TPE sampler, seed=42), and the activation was one of the searched hyperparameters:

```python
activation = trial.suggest_categorical("activation", ["relu", "tanh", "gelu"])
```

The selection rule was "smallest architecture within 5% of the best validation MSE." The search independently picked, per task, whichever activation minimized validation error:

| Config | Chosen arch | Inputs | Difficulty | Test R² | Test MSE (z) |
|---|---|---|---|---:|---:|
| `PMSM_no_temps` | 1×32 **GeLU** | 8 (no temp sensors) | hard | 0.853 | 1.56×10⁻¹ |
| `PMSM_no_temps_pm` | 2×32 **GeLU** | 9 (no temp sensors) | hard | 0.905 | 1.00×10⁻¹ |
| `PMSM_all_in` | 1×32 **Tanh** | 11 (incl. stator temps) | very easy | 0.997 | 2.84×10⁻³ |

So strictly there is no hand-chosen rationale: the search found GeLU gave lower validation error on the two hard "no-temperature-sensor, infer from current/voltage" tasks, and tanh won on the near-linear `all_in` task that has temperature inputs.

> This hard/easy split is not just an activation issue — it also determines whether OBS works later (see Section 5). The key number is R²: `all_in` fits to R²=0.997 (tiny residuals); the two GeLU tasks only reach 0.85 / 0.90 (large residuals).

---

## 2. Overview of the four OBS approaches

All OBS variants **share the same selection/correction formulas** (Section 4); the only difference is **how the matrix H (or H⁻¹) is obtained.**

| Approach | Matrix used | Who uses it | Key implementation |
|---|---|---|---|
| **① Exact · full-batch** | exact Hessian  | `hessian(…, vectorize=True)`, whole train set in one call |
| **② Exact · chunked** | exact Hessian (numerically identical to ①) |  same, but accumulated over 8192-row chunks |
| **③ Exact · no-vec + OBD fallback** | exact Hessian; degrades to OBD-diagonal when ill-conditioned/indefinite | `vectorize=False` + guard that degrades |
| **④ Fisher / Gauss-Newton** | empirical Fisher (**PSD**)  | `MAX_EXACT_N=0`, force Fisher |

**Capability matrix (does it finish vs. are results valid):**

| Approach | tanh finishes | tanh result | GeLU finishes | GeLU result | Main drawback |
|---|:---:|:---:|:---:|:---:|---|
| ① Exact · full-batch | ✅ | ✅ | ❌ full batch → OOM/hang | ❌ | both hangs and diverges |
| ② Exact · chunked  | ✅ | ✅ | ✅*(modern torch) | ❌ **diverges** | finishes, but GeLU result is still garbage |
| ③ Exact + OBD fallback | ✅ | ✅ | ✅ | ⚠️ marginally usable | after degrading it is no longer true OBS; mediocre, higher MSE |
| ④ Fisher | ✅ | ✅ | ✅ | ✅ **stable** | an approximation, not the true Hessian, but numerically stable and near-baseline |

\* If still on PyTorch 2.2, GeLU's `erf` double-backward lacks a vmap rule, so ② may still hang even when chunked; modern torch has fixed this.

**In one line:** ① and ② use the exact Hessian and therefore necessarily diverge on GeLU; on GeLU the only usable ones are ④ (recommended) and ③ (marginal).

---

## 3. Results comparison

Because ② (chunked) and ① (full-batch) compute a **numerically identical** matrix, their MSE values are the same (they differ only in *whether they finish*), so the numeric column is merged as "Exact." Units: Test MSE (z space); `—` = stopped by the "MSE > 3× baseline at sparsity ≥ 50%" rule, or missing.

### `PMSM_no_temps` (baseline 1.555e-01, 321p)
| Sparsity | Exact (①/②) | Exact+OBD (③) | Fisher (④) |
|---:|---|---|---|
| 10% | 9.65e+03 | 1.72e-01 | 1.69e-01 |
| 20% | 6.81e+06 | 1.69e-01 | 1.48e-01 |
| 30% | 6.47e+05 | 1.54e-01 | 1.51e-01 |
| 40% | 3.01e+04 | 1.40e-01 | 1.58e-01 |
| 50% | 1.81e+01 | 1.50e-01 | 1.37e-01 |
| 60% | — (diverges) | 2.87e-01 | 1.50e-01 |
| 70% | — | 3.25e-01 | 1.64e-01 |
| 80% | — | 3.02e-01 | 1.93e-01 |
| 90% | — | 5.61e-01 | 2.06e-01 |

### `PMSM_no_temps_pm` (baseline 1.005e-01, 1409p)
| Sparsity | Exact (①/②) | Exact+OBD (③) | Fisher (④) |
|---:|---|---|---|
| 10% | 3.27e+06 | 1.23e-01 | 1.02e-01 |
| 20% | 1.13e+03 | 1.21e-01 | 1.04e-01 |
| 30% | 2.25e+00 | 1.28e-01 | 1.05e-01 |
| 40% | 1.50e+00 | 1.27e-01 | 1.13e-01 |
| 50% | 1.17e+00 | 1.14e-01 | 1.22e-01 |
| 60% | — (diverges) | 1.26e-01 | 1.22e-01 |
| 70% | — | 1.26e-01 | 1.22e-01 |
| 80% | — | 1.22e-01 | 1.15e-01 |
| 90% | — | 1.16e-01 | 1.10e-01 |

### `PMSM_all_in` (baseline 2.840e-03, 417p, **tanh**)
| Sparsity | Exact (①/②) | Exact+OBD (③) | Fisher (④) |
|---:|---|---|---|
| 10% | 2.79e-03 | 2.95e-03 | 2.80e-03 |
| 20% | 3.70e-03 | 2.65e-02 | 2.82e-03 |
| 30% | 3.33e-03 | 1.54e-02 | 2.77e-03 |
| 40% | 6.59e-03 | 3.00e-02 | 2.77e-03 |
| 50% | 4.52e-03 | 7.85e-03 | 2.89e-03 |
| 60% | 1.63e-01 | 1.15e-02 | 3.06e-03 |
| 70% | — | — | 1.35e-02 |
| 80% | — | — | — |
| 90% | — | — | — |

**How to read these:**
- The two **GeLU** configs: the Exact column is already garbage at 10% (10³–10⁶ magnitude); ③ (Exact+OBD) is caught by the OBD fallback and reaches 90% but with higher MSE; ④ (Fisher) is stable, stays near baseline all the way to 90%.
- The **tanh** `all_in` config: all three are usable, including exact OBS (to ~50–60%), because this task's Hessian is nearly PSD (see Section 5). Notably ④ Fisher is the most stable here (reaches 70%), while exact OBS on this tiny net becomes unstable past 60%.

**Deepest usable sparsity on GeLU (MSE ≤ 3× baseline):**

| Config | Exact (①/②) | Exact+OBD (③) | Fisher (④) |
|---|:---:|:---:|:---:|
| `PMSM_no_temps` | unusable | 80% | **90%** |
| `PMSM_no_temps_pm` | unusable | 90% | **90%** |

---

## 4. The mathematical procedure of each approach

### 4.0 The OBS core shared by all variants (only H differs)

For a trained network, expand the loss to second order around the current weights **w**. Deleting one weight `w_q` (setting it to 0) increases the loss by approximately

$$\Delta L \approx \tfrac{1}{2}\,\delta w^\top H\,\delta w,\quad \text{subject to } \delta w_q = -w_q.$$

From this (Hassibi & Stork, 1993):

- **Saliency (deletion cost):** $L_q = \dfrac{w_q^2}{2\,[H^{-1}]_{qq}}$ — delete the weight with the smallest `L_q`.
- **Weight correction:** $\delta w = -\dfrac{w_q}{[H^{-1}]_{qq}}\,H^{-1}e_q$ — adjust the remaining weights to compensate.
- **Rank-1 update:** $H^{-1}\leftarrow H^{-1}-\dfrac{H^{-1}e_q e_q^\top H^{-1}}{[H^{-1}]_{qq}}$ — update the inverse after each deletion, then continue.

**This machinery assumes H is positive semi-definite (PSD)** — only then are `ΔL ≥ 0` and `[H⁻¹]_qq > 0` meaningful. The four approaches differ only in how they obtain H:

### 4.1 ① Exact · full-batch
$$H=\nabla^2_w\Big(\tfrac{1}{N}\sum_{i=1}^{N}(f_w(x_i)-y_i)^2\Big),\quad \text{via } \texttt{autograd.functional.hessian(fn, w, vectorize=True)} \text{ over all } N \text{ rows at once.}$$
`vectorize=True` uses vmap to batch the n directional second derivatives. **Drawback:** O(N·n) memory; the full train set OOMs/hangs (empirically GeLU is OOM-killed at N=100k).

### 4.2 ② Exact · chunked (z4)
Using "the Hessian of a mean loss = the sample-fraction-weighted mean of per-chunk Hessians":
$$H=\sum_{\text{chunk}}\frac{n_{\text{chunk}}}{N}\,H_{\text{chunk}},\quad H_{\text{chunk}}=\texttt{hessian}(fn_{\text{chunk}}, w, \texttt{vectorize=True}),\ n_{\text{chunk}}\le 8192.$$
Numerically **identical** to ① (measured difference ~6e-8, pure float error), but each call processes ≤8192 rows, so memory is bounded. **Drawback:** it only fixes "memory / can it finish." The matrix obtained is still the exact one — and on GeLU that matrix is still **indefinite** (Section 5), so the result still diverges.

### 4.3 ③ Exact + `vectorize=False` + OBD fallback
Build the exact Hessian **row by row** with `hessian(fn, w, vectorize=False)` (n ordinary backward passes, no vmap — avoids the PyTorch-2.2 erf gap). Then add a guard: if `(H+λI)` is ill-conditioned (bump λ by 10×; if still bad, give up), or the positive-diagonal set is exhausted mid-run, **degrade to OBD-diagonal saliency**:
$$L_q^{\text{OBD}}=\tfrac{1}{2}\,w_q^2\,H_{qq}\quad(\text{diagonal } H_{qq} \text{ only, no full-matrix correction}).$$
**Drawback:** once degraded it is no longer true OBS (no correction term); accuracy drops to OBD level, MSE higher.

### 4.4 ④ Fisher / Gauss-Newton
Replace the Hessian with the empirical Fisher matrix:
$$F=\frac{1}{N}\sum_{i=1}^{N} g_i\,g_i^\top,\quad g_i=\nabla_w\,\ell_i(w).$$
`F` is a sum of outer products and is therefore **PSD by construction**. Then run the same OBS machinery (saliency / correction / rank-1 update) with F in place of H. Near a minimum $F\approx$ Gauss-Newton $\approx H$, so this is a principled approximation, not an arbitrary swap. **Drawback:** it only approximates the true Hessian (slightly off where residuals are large), but in return it is PSD and numerically stable.

---

## 5. Why exact OBS can't be used on GeLU — the clear reason

**Bottom line first: because the exact Hessian of the two GeLU tasks is *indefinite* (has negative eigenvalues), while OBS's formulas require the Hessian to be PSD by derivation. When that premise fails, both the selection and the correction it produces become meaningless — this is a mathematical problem, not a code bug, and no implementation change (chunking ②, the exact part of ③, upgrading torch) can fix it.**

### 5.1 Why the exact Hessian is indefinite

The exact Hessian of the MSE loss decomposes exactly into two terms:
$$H=\underbrace{\frac{2}{N}\sum_i J_i^\top J_i}_{\text{Gauss-Newton, always PSD}}\;+\;\underbrace{\frac{2}{N}\sum_i r_i\,\nabla^2_w f_w(x_i)}_{\text{residual term, sign-indefinite}},\quad r_i=f_w(x_i)-y_i.$$

- The first term (Gauss-Newton, `JᵀJ`) is always PSD.
- The second term is driven by the **residuals `r_i`** times the **second derivative of the network output w.r.t. parameters** `∇²f`. That `∇²f` contains the activation's second derivative, and **GeLU's second derivative changes sign** (non-monotone). When residuals are large, this term can overpower the first and **inject negative eigenvalues → indefinite.**

This explains "why exactly the two GeLU ones":
- `all_in` (tanh) **fits to R²=0.997, residuals ≈ 0** → the residual term ≈ 0 → `H ≈ Gauss-Newton` → **nearly PSD** → OBS premise holds → exact OBS is usable.
- The two no_temps (GeLU) are **hard tasks with large residuals** (R²=0.85/0.90, MSE~0.1) → the residual term is significant → **H indefinite** → the OBS premise fails.

> In other words: indefiniteness = (large residual) × (activation curvature). The two high-residual hard tasks happen to have been assigned GeLU by HPO, so the phenomenon shows up as "the GeLU configs diverge."

### 5.2 Indefiniteness breaks both OBS formulas at once

1. **Deletion cost becomes meaningless; the loop stalls.** `L_q = w_q²/(2[H⁻¹]_qq)`. When H is indefinite, `[H⁻¹]_qq` can be negative → the cost goes negative, so "which weight hurts least" is undefined. OBS can only pick weights with `[H⁻¹]_qq > 0`; those get used up quickly → the removal loop can't progress (another reason it "hangs").
2. **The correction amplifies error; MSE explodes.** The correction `δw` moves along **negative-curvature directions**. The Taylor term `ΔL = ½δwᵀHδw` is negative there — the formula "believes" the loss is decreasing while it actually increases → error accumulates and is amplified at each step → MSE blows up to **10³–10⁶× baseline** (the Exact column in Section 3).

### 5.3 Why "changing the implementation" cannot help

**Indefiniteness is a property of the matrix itself, determined by (residual × GeLU curvature), independent of how you compute it.**
- ② chunked: computes the same matrix → same indefiniteness → same divergence.
- ③'s `vectorize=False`: only changes how the Hessian is built (avoids the vmap gap); the matrix is still that indefinite one — which is exactly why ③ must rely on "degrade to OBD" to avoid diverging.
- Upgrading torch: affects only whether it finishes, not the indefiniteness.

**Only "change the matrix" or "degrade" works:**
- ④ **Fisher** uses `F = Σ gᵢgᵢᵀ`, a sum of outer products that is **always PSD** → `[F⁻¹]_qq ≥ 0` → deletion cost is always non-negative and the correction always compensates rather than amplifies → the OBS premise is automatically restored; and `F ≈` the true Hessian, so it is both stable and accurate. **This is the fundamental reason Fisher works and exact does not.**
- ③ **OBD fallback**: abandons the full matrix and uses only the (positive) diagonal — also avoids divergence, but is no longer true OBS and is mediocre.

---

## 6. Conclusions and recommendations

- **tanh (`all_in`)**: all four approaches are usable, including exact OBS (tiny residuals → nearly-PSD Hessian).
- **GeLU (`no_temps` / `no_temps_pm`)**: **exact-Hessian OBS  is unusable** — the indefinite Hessian breaks OBS's mathematical premise and the result diverges. **Only ④ Fisher (recommended — stable, near-baseline, to 90%) or ③ Exact+OBD (usable but mediocre) work.**
- **What ② is**: it fixed the **engineering** problem of "feeding the Hessian the full batch → OOM/hang" (chunking → bounded memory), letting the exact Hessian be computed quickly and cheaply — **but it does nothing for GeLU's divergence**; on the two GeLU configs OBS is still garbage.

> In one line: **② made it "finish," but not "compute correctly." To be correct on GeLU you must replace the indefinite exact Hessian with the PSD Fisher matrix (or degrade to OBD) — a consequence of the mathematical fact that OBS requires a PSD Hessian while the exact Hessian of the hard GeLU tasks is indefinite.**
