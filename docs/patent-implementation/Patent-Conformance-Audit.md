# Patent Conformance Audit — US 12,124,779 B2 (ADAGOS)

**Audited:** `patenting/` against the granted patent (24 pp., 16 claims).
**Rewritten:** 2026-06-18 — describes what the code **actually does** after the
faithful-implementation pass. (The prior version of this file overstated
fidelity; see "What changed" below.)

---

## 1. Verdict

The constructive search now realises the patent's **signature mechanism**: candidate
topology edits are ranked by the cheap **frozen-multiplier error-variation estimate**
(eq. 13/14), computed from the converged parent's frozen activations `X*` and
multipliers `Λ*`, **without retraining any candidate**. Only the selected winner is
fully retrained with Gauss-Newton (Claim 7).

Three embodiments are implemented and validated:

| Embodiment | Claims | Module | Status |
|---|---|---|---|
| Static regression | 1–8 | `lagrangian_estimate.py` + `architecture_search.py` | ✅ faithful |
| Dynamic / recurrent twin | 11, 14–15 | `dynamic.py` | ✅ faithful |
| Compression / decompression | 9, 16 | (intentionally OFF for low-dim data) | ➖ N/A by scope |

---

## 2. The faithful estimate (the core fix)

The patent's cheap estimate (p.14–16) is

```
ΔL = L(Γ'', W'', X*, Λ*) − J(parent)  =  Σ_i  Λ*_i · ( X*_i − φ_i(W''_{i-1} X*_{i-1}) )
```

with `X*`, `Λ*` frozen from the parent. The J terms cancel (the output state is
frozen), so only the layers whose weights changed contribute — O(layers·N), no
candidate forward/GN pass. Per op (`lagrangian_estimate.py`):

- **REMOVE_NEURON** — zero the dropped neuron's outgoing column; read off the
  layer-(L+1) constraint term. Exact (eq. 13).
- **REMOVE_LAYER** — linear-merge `W_L W_{L-1}` (eq. 13 restriction) + the patent's
  "one or two iterations" at frozen `X*,Λ*` (the merge skips a nonlinearity, so the
  raw restriction is not first-order).
- **ADD_NEURON / ADD_SKIP / ADD_LAYER** — function-preserving / zero init, then the
  Pontryagin warm-up (col. 16: "minimize L w.r.t. W'' with X and Λ fixed"). With the
  state frozen the layers decouple, so this is a tiny per-candidate fit of only the
  new weights — far cheaper than the Gauss-Newton mini-train it replaces.

**Two bugs fixed to make this work:**

1. **Multiplier definition.** The patent's `Λ_i` is the adjoint of the
   *post-activation* state (`Λ_i = W_i^T(φ'(z_{i+1})⊙Λ_{i+1})`, appendix p.22).
   The old `lagrangian.calculate_lagrange_multipliers` returned the *pre-activation*
   backprop delta `δ_i = φ'(z_i)⊙(W_i^T δ_{i+1})` — a different quantity — which made
   the frozen estimate unusable (and is almost certainly why it was abandoned).
   `lagrangian_estimate.post_adjoint` computes the correct quantity.
2. **Error normalisation.** The estimate and `J` both use the SUM-of-squares error
   (patent eq. 4 / Claim 6), so `Λ_nc = 2(out−Y)` makes `L ≡ J` at consistency.

**Validation** (`python -m patenting.validate_estimate`): on a controlled underfit
parent (signal present) the cheap ΔL and the true post-retrain ΔJ agree at
**Spearman ≈ 0.89**, the cheap "least-harmful removal" matches the true one, and
sign agreement is **6/6** (ADD<0, REMOVE>0). On saturated parents (A/B at width 5,
already at the noise floor) the estimate correctly collapses toward 0 — the patent's
"stop" regime.

---

## 3. Claim-by-claim

| Claim | Status | Notes |
|---|---|---|
| 1 | ◐ | Add/remove neurons, layers, and skip connections. Edge-granularity single links not implemented (scope). |
| 2 | ✅ | Selects the most-negative ΔL. |
| 3 | ✅ | Lagrange operator `L(Γ,W,X,Λ)` **drives ranking** (post-activation adjoint). |
| 4 | ✅ | Additive estimate = zero/preserving init + frozen-`X,Λ` Pontryagin warm-up. |
| 5 | ✅ | Subtractive estimate = restriction (zero column / linear merge). |
| 6 | ✅ | `J = Σ‖f(Xᵢ)−Yᵢ‖²`. |
| 7 | ✅ | Winner retrained by damped Gauss-Newton (analytical multi-output Jacobian). Not the zero-memory variant (forms JᵀJ; fine at this scale). |
| 8 | ✅ | Additive accepted only on a meaningful train-error cut (parsimony brake); overfit (train≪val) restricts to subtractive; val gate; patience stop. |
| 9, 16 | ➖ | Compression OFF by scope — low-dim data has nothing to compress. |
| 11, 14–15 | ✅ | `dynamic.py`: recurrent block `x_{t+1}=x_t+h(x_t,u_t)`, one-step teacher-forced (reduces to static), free-running rollout on held-out sequences. |
| 12, 13 | ✅ | Network object / runnable module. |

---

## 4. What changed vs. the prior implementation

> Concrete, code-level before/after (exact functions and lines):
> [`Patent-Method-Changes.md`](Patent-Method-Changes.md).

The old selection **Gauss-Newton mini-trained every candidate and ranked by MSE** —
exactly the per-candidate cost the patent's estimate exists to avoid — and discarded
the frozen multipliers entirely (`evaluate_candidate` recomputed fresh multipliers,
collapsing `L` to the candidate's own MSE). The audit then claimed "scoring uses the
Lagrangian," which the code did not do. It is now replaced by
`lagrangian_estimate.rank_candidates`. Dead acceptance branches (the unconditional
`current = retrained`, the no-op complexity rejection) were removed. The old
GN-mini-train path remains in `training.py` only as `train_network` (the winner's
retrainer); the candidate-ranking functions there are no longer used.

---

## 5. Reproduce

```bash
python -m patenting.validate_estimate          # estimate vs ground-truth (Spearman)
python -m patenting.dynamic                     # dynamic engine self-test (lumped thermal)
python -m patenting.motor.run_stator            # PMSM stator-winding config sweep (static)
python -m patenting.main                         # A/B static, d=1,2,3 (compression skipped <4-D)
```
