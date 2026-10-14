# Patent Method — Concrete Change Record

**What this is.** A precise, code-level record of what changed when the patent
method (US 12,124,779 B2, ADAGOS) was made *faithful* (the 2026-06-18 pass). The
claim-by-claim conformance is in [`Patent-Conformance-Audit.md`](Patent-Conformance-Audit.md);
the before/after results are in [`results-faithful-method.md`](results-faithful-method.md).
This file pins down the *mechanism* changes — what the old code did, what the new
code does, and the exact functions/lines involved.

---

## TL;DR

The old code **did not use the patent's signature mechanism.** Candidate topology
edits were ranked by **fully mini-training each candidate with Gauss-Newton and
reading its MSE** — exactly the per-candidate cost the patent's cheap estimate
exists to avoid — and the frozen Lagrange multipliers were discarded. Two bugs
made the genuine estimate unusable. Both are fixed; selection now ranks candidates
by the frozen-multiplier error-variation estimate (eq. 13/14) with **no candidate
retraining**, and only the winner is Gauss-Newton retrained (Claim 7).

---

## Change 1 — Selection mechanism (the core change)

| | Before | After |
|---|---|---|
| Candidate scoring | GN mini-train **every** candidate, rank by resulting MSE | Frozen-multiplier ΔL estimate (eq. 13/14), **no retraining** |
| Multipliers | recomputed fresh per candidate (so `L` collapsed to the candidate's own MSE) | frozen from the converged parent `X*, Λ*` |
| Cost per cycle | O(candidates · GN-iterations · N) | O(layers · N) + a tiny warm-up of only the *new* weights |
| Winner | re-used the already-trained candidate | the single selected edit is GN-retrained from the merged init |
| Where | `training.py` candidate-ranking helpers | `lagrangian_estimate.rank_candidates` |

The estimate (frozen `X*, Λ*`; the `J` terms cancel because the output state is
frozen, so only changed layers contribute):

```
ΔL  =  Σ_i  Λ*_i · ( X*_i − φ_i(W''_{i-1} X*_{i-1}) )        (eq. 13/14)
```

Per operation, in `lagrangian_estimate.py`:
- **REMOVE_NEURON** — zero the dropped neuron's outgoing column, read off the
  layer-(L+1) constraint term (`delta_remove_neuron` / `best_remove_neuron`). Exact.
- **REMOVE_LAYER** — linear-merge `W_L W_{L-1}` + 1–2 frozen-state iterations
  (`delta_remove_layer`).
- **ADD_NEURON / ADD_SKIP / ADD_LAYER** — function-preserving / zero init, then a
  Pontryagin warm-up of only the new weights at frozen `X*, Λ*` (`delta_add_neurons`).

Dead branches removed from the old loop: the unconditional `current = retrained`
acceptance and the no-op complexity rejection.

---

## Change 2 — Multiplier definition (bug fix)

The patent's `Λ_i` is the adjoint of the **post-activation** state
(`Λ_i = W_iᵀ (φ'(z_{i+1}) ⊙ Λ_{i+1})`, appendix p.22). The old code returned the
**pre-activation** backprop delta — a different quantity — which is almost
certainly why the frozen estimate was abandoned and replaced by mini-training.

**Before** — `lagrangian.calculate_lagrange_multipliers` applies `σ'(z_i)` *after*
`Wᵀ`, at layer `i`:

```python
# Λ_i = (W_i^T Λ_{i+1}) * φ'(z_i)   — φ' applied AFTER Wᵀ, at layer i
grad   = net.weights[i].T @ lam[i + 1]
deriv  = jacobian_layer(net.layer_types[i - 1], pre_acts[i - 1])   # φ'(z_i)
lam[i] = grad * deriv                                              # pre-activation delta
```

**After** — `lagrangian_estimate.post_adjoint` applies `φ'(z_{i+1})` *inside*,
before `W_iᵀ`:

```python
p[nc] = 2.0 * (acts[nc] - Y)
for i in range(nc - 1, 0, -1):
    phi_d = jacobian_layer(net.layer_types[i], pre_acts[i])   # φ'(z_{i+1})
    p[i]  = net.weights[i].T @ (phi_d * p[i + 1])             # post-activation adjoint
```

This single change is what turns the estimate from noise into a valid ranker.

---

## Change 3 — Error normalisation (consistency fix)

The old code mixed **mean** and **sum** of squared errors between the cost `J` and
the multiplier seed, so `L` and `J` were not on the same scale and never cancelled.
Now both use the **sum**-of-squares error (patent eq. 4 / Claim 6), and the output
multiplier is seeded `Λ_nc = 2(out − Y)`, so `L ≡ J` at consistency and the ΔL
differences are meaningful.

```
J = Σ_i ‖ f(X_i) − Y_i ‖²            Λ_nc = 2 (out − Y)
```

---

## Change 4 — Search-loop discipline (determinism + parsimony)

To make the constructive search reproducible and to stop genuine over-growth:
- **Incremental growth** — add **one** node at a time (`add_widths=(1,)`), instead
  of jumping widths.
- **Parsimony brake** — an additive edit is accepted only if it cuts the train
  error by a meaningful margin (`eps_accept`); otherwise it is rejected.
- **Claim-8 validation gate** — an overfitting parent (train ≪ val) is restricted
  to *subtractive* edits; a relative val-slack gate + `patience` stop end the search.
- **Single global seed + monotonic retraining** — removes run-to-run drift that the
  old code showed on noiseless data.

(Signature: `architecture_search(..., eps_accept=0.01, val_tol=0.02, patience=3, add_widths=(1,), selection="faithful")`.)

---

## Files

| File | Change |
|---|---|
| `patenting/lagrangian_estimate.py` | **NEW** — `post_adjoint`, `parent_state`, per-op ΔL (`delta_*`), `rank_candidates` (eq. 13/14) |
| `patenting/architecture_search.py` | **REWRITTEN** — ranks by faithful ΔL; parsimony brake + Claim-8 gates; dead branches removed |
| `patenting/lagrangian.py` | superseded multiplier (pre-activation delta); kept for reference, no longer drives selection |
| `patenting/training.py` | **SLIMMED** — only `train_network` (the winner's GN retrainer) remains; candidate-ranking helpers retired |
| `patenting/dynamic.py` | **NEW** — recurrent twin reusing the same search (Claims 11, 14–15) |
| `patenting/validate_estimate.py` | **NEW** — proves the estimate ranks like a full retrain |

---

## Validation

`python -m patenting.validate_estimate`: on a controlled underfit parent (signal
present) the cheap ΔL and the true post-retrain ΔJ agree at **Spearman ≈ 0.89**,
the cheap "least-harmful removal" matches the true one, and sign agreement is
**6/6** (ADD < 0, REMOVE > 0). On saturated parents the estimate correctly collapses
toward 0 — the patent's "stop" regime.

---

## Intentionally NOT changed / out of scope

- **Compression / decompression (Claims 9, 16)** — OFF by scope: low-dim tabular
  data has nothing to compress; the patent only warrants it for high-dim `X`.
- **Edge-granularity single-link edits (part of Claim 1)** — not implemented; the
  search operates at neuron/layer/skip granularity.
- **Zero-memory Gauss-Newton (Fehrenbach 2006)** — the winner retrain forms `JᵀJ`
  explicitly; fine at this problem scale.
