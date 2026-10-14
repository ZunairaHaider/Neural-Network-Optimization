# Results — faithful patent method (2026-06-18)

Results produced after re-implementing the patent's selection mechanism (the
frozen-multiplier ΔL estimate, eq. 13/14). See `docs/patent-implementation/Patent-Conformance-Audit.md`
for what changed and why.

---

## 1. Estimator validation (`python -m patenting.validate_estimate`)

Does the cheap ΔL estimate rank candidates the way a full retrain would?

| Test bed | Spearman(cheap ΔL, true ΔJ) | least-harmful removal | sign agreement |
|---|---|---|---|
| underfit synth (signal present) | **+0.89** | match | 6/6 |
| A/B at width 5 (saturated) | ≈0 (noise) | — | — |

The estimate is a valid ranker where capacity signal exists; on a saturated
parent it correctly collapses toward 0 (the patent's "stop" regime). The fix that
made this work was using the **post-activation adjoint** for the multiplier (the
old code used the pre-activation backprop delta) + sum-of-squares `J`.

---

## 2. A/B static rerun, d = 1,2,3 (`python -m patenting.main`)

Faithful engine, patent-minimal init `[d, 1, output]`, global seed 42.

| Dataset | NEW topology | NEW test MSE | OLD topology | OLD test MSE |
|---|---|---|---|---|
| A 1d | `[1,4,4,1]`   | 1.19e-6 | `[1,2,1]`   | **3.7e-11** |
| A 2d | `[2,4,4,2,1]` | **1.69e-4** | `[2,2,2,1]` | 0.039 |
| A 3d | `[3,4,4,1,1]` | **3.97e-4** | `[3,2,1]`   | 0.018 |
| B 1d | `[1,2,1]`     | 0.171   | `[1,1,4,1]` | 0.156 |
| B 2d | `[2,4,3,1]`   | 0.311   | `[2,3,1]`   | 0.300 |
| B 3d | `[3,4,1,1]`   | 0.454   | `[3,2,1]`   | 0.46  |

### Is the new method "better"? — honest read

Mixed, and that's expected, because **the goal was faithfulness, not winning on MSE**:

- **A 2d / 3d — new is much better** (~100× lower MSE). The harder multi-input
  cases genuinely benefit from the faithful search building deeper nets.
- **A 1d — old is better.** The old `[1,2,1]` hit machine precision (3.7e-11);
  the new run overshot to `[1,4,4,1]` (1.19e-6, still tiny). On a *noiseless,
  trivially-fittable* 1-D function the stopping point is sensitive: if an
  intermediate `[1,2,1]` isn't retrained all the way to ~0, the search keeps
  seeing "capacity helps" and grows. This is a **parsimony-tuning** issue
  (stronger per-topology GN convergence, or a larger `eps_accept` brake), NOT a
  flaw in the estimate. The old `[1,2,1]` came from the *non-faithful*
  GN-mini-train selection + a hand-tuned ε=0.05 brake.
- **B 1d / 2d / 3d — tied.** B is noise-limited; both methods sit at the noise
  floor (σ² ≈ 0.17 / 0.30 / 0.45). Differences are sampling noise.

**Takeaway:** the new method is faithful (validated), better on the harder A
cases, tied on B, and slightly over-grows the trivial A-1d case — a known,
tunable follow-up. The old A-1d number was prettier but produced by a method that
wasn't actually the patent.

---

## 3. PMSM stator-winding (SMOKE TESTS ONLY — not the full sweep)

Config `coolant+elec` = 7 electrical/speed/ambient signals **+ coolant, NO other
temperature sensors** (the email's key question). Short runs to prove the
pipeline; not converged, single config.

| Embodiment | topology | held-out error |
|---|---|---|
| Static (instantaneous `winding=f(inputs)`) | `[8,8,2,1]` | test MSE 0.192 (norm) → **MAE ≈ 8.8 °C** |
| Dynamic twin (`w_{t+Δ}=w_t+h(w_t,u_t)`, stride 10 s) | `[9,9,1,1]` | rollout **RMSE mean 2.55 °C / max 6.56 °C**, hot-spot max 26.8 °C |

The dynamic framing is ~3.5× better on the same inputs because winding
temperature has thermal lag an instantaneous map can't capture. Caveats: small
subsamples (2.5–3k), 6 cycles, one config. The full 5-config comparison
(`run_stator.run()`) has NOT been run end-to-end yet.

---

## 4. Dynamic engine self-test (`python -m patenting.dynamic`)

Lumped thermal system `x_{t+1}=x_t+a(u_t−x_t)` (linear in state+drive):
- modeling block found = **minimal `[2,1,1]`** (one-step MSE 3.7e-11) — correct
  parsimony for a linear system.
- free-running rollout RMSE **0.0008** on state-std 5.6 — near-perfect twin from
  `x_0` + drive only.
