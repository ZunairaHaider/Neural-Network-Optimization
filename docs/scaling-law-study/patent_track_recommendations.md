# Patent-track recommendations — deferred work

Author: scaling-law-study session, 2026-05-14.
Scope: changes / improvements I recommend for the `patenting/` (Aryan & Yanqin)
and `patent-method-revised/` (this branch) packages. Code is **not** changed in
this session — pruning is the priority. This is a written checklist for the
patent owners and for a future session that picks the patent track back up.

The Phase 1 sweep delivered for pruning therefore does **not** include the
patent track in the headline scaling-law tables. When that work resumes, this
doc is the entry point.

---

## 1. Hard blockers found in the smoke test

`python patent-method-revised/main.py --dataset A` runs end-to-end and writes
the expected outputs, but the result quality is far below the HPO + pruning
baseline. Specifically (smoke run, 2026-05-14):

| | smoke run | HPO baseline (1-D Dataset A) |
|---|---|---|
| Architecture | `[8]` (1 hidden layer) | `[16, 16]` |
| Params | 25 | 321 |
| Test MSE | 7.36 × 10⁻¹ | 1.94 × 10⁻⁴ |

That is ≈ 4000× worse than the dense baseline. The search bails on cycle 12
with “no candidate improved val MSE.” The architecture got stuck at a
single-hidden-layer width = 8 model that under-fits a clean quadratic.

The most likely root causes — all **fixable without algorithmic change**:

1. **Short-fine-tune budget too small.** `error_estimation._short_finetune`
   trains each candidate for **30 epochs** of full-batch Adam at `lr=1e-3`.
   That is enough to rank candidates *consistently* but not enough for the
   ranking score to actually track post-retrain performance. The post-
   selection retrain is only 500 epochs (vs. 1000 for the HPO baseline). Both
   should be increased.

   - **Recommendation:** raise short-fine-tune to ≥ 100 epochs with patience
     = 20; raise post-selection retrain to 2000 epochs with patience = 50.
     Document the cost increase in cycle wall-clock.

2. **Greedy local search is trapped at small architectures.** Once the
   network shrinks to `[1]` via a `remove_layer` (cycle 4 in the smoke run),
   every subsequent cycle picks `add_neuron@0` and never `add_layer`. That
   is correct local greedy behaviour, but it means the search is biased
   toward wide-shallow over deep-narrow.

   - **Recommendation:** in `topology_search.py`, enforce a non-greedy
     **temperature schedule** on candidate acceptance for the first K cycles,
     or interleave forced `add_layer` candidates every M cycles. The patent
     col. 13 “broad sampling” clause is on our side — we are within scope.

3. **`K_ADD_SAMPLE = 32` is never the binding constraint** in 1-D because
   the strict-MLP candidate set is small. In 2-D / 3-D it will be, so the
   parameter is still right.

   - **Recommendation:** keep `K_ADD_SAMPLE = 32` but log how many were
     actually enumerated so future debugging is easier.

These are all hyperparameter tuning, not algorithmic. **Estimated effort: 1
focused session.**

---

## 2. Algorithmic deltas already closed (D2 / D4 / D5 / D7)

These are the audit-flagged deltas vs. the patent that `patent-method-revised/`
**already implements**. No further work needed:

- **D2** — Cheap Lagrangian via warm-start + short fine-tune (`error_estimation.py`).
- **D4** — Broad subtractive enumeration (`candidates.py` enumerates every
  neuron with width ≥ 2 and every removable layer).
- **D5** — Phase selection by train/val tolerance (`topology_search.py` col. 6).
- **D7** — Val-set stopping criterion (the search stops when val MSE fails to
  improve; not test-set as in `patenting/`).

---

## 3. Algorithmic deltas explicitly deferred (D1 / D3 / D8)

These are the audit-flagged items we agreed not to close in Phase 1:

- **D1** — Skip connections. **Permanently out of scope** for this study
  (strict-MLP only). Document this clearly in the final-report methodology
  section.
- **D3** — Zero-memory Gauss-Newton training. The patent describes a
  Gauss-Newton optimiser that requires no Hessian storage; we use plain Adam
  in both `patenting/` and `patent-method-revised/`. For 1-D / 2-D / 3-D
  quadratics, Adam is more than adequate and the deviation is not a fairness
  problem. **Defer to a follow-up paper.**
- **D8** — Incremental compression-block construction. The patent (col. 14)
  describes building the compression block by recursive subdivision of the
  central hidden layer. We currently don’t implement compression in
  `patent-method-revised/`, and `patenting/`’s compression block is a
  standalone autoencoder whose reconstruction loss isn't used in the
  objective (logged-only). For a faithful re-implementation, **D8 should be
  closed before the patent track is compared head-to-head.**

---

## 4. Recommended changes to `patenting/` (Aryan & Yanqin)

The audit (§4) flagged these in the original package. Two of them are
**trivial to port** and would make the head-to-head comparison fairer:

1. **Val-set stopping (D7).** `patenting/topology_search.py` currently uses
   test MSE in the stopping rule. This is a leakage issue analogous to the
   pruning one I fixed in §2.1 of this branch. **One-line fix:** carve a val
   split inside the search loop and use it for the stopping rule.
2. **Cycle cap.** `patenting/` caps at 15 cycles, `patent-method-revised/` at
   30. Pick one and use both for parity.

The other items in the audit (compression-block role, complexity penalty
form) are deeper and require coordination — flag for the next sync with
Aryan & Yanqin.

---

## 5. Pre-flight checklist for resuming the patent track

A future session that picks this back up should:

1. Read this doc.
2. Apply the §1 hyperparameter changes and re-smoke `patent-method-revised`
   on Dataset A. Expect test MSE to drop from ≈ 0.74 to below 10⁻³.
3. Add FLOPs + inference-latency + per-layer width counters to both patent
   packages so they emit the same 11-metric table the pruning track does.
   The pruning-side hook is `src/analysis/scaling_law.py`; extend it with a
   patent-track ingestion path.
4. Run both `patenting/` and `patent-method-revised/` on Dataset A & B at
   d ∈ {1, 2, 3} once `patent-method-revised/`’s outputs match the pruning
   track schema. **Estimated runtime budget:** 5–15 min per (method, ds, dim)
   cell × 12 cells = ~1–3 hours.
5. Update `docs/scaling-law-study/narrative.md` with the patent-vs-pruning comparison.

---

## 6. Open questions for the team

These are explicitly noted by the audit (§5–6) and need human-in-the-loop
decisions:

- Which patent implementation goes into the final report? `patent-method-
  revised/` (faithful to the patent claims) or `patenting/` (Aryan/Yanqin's
  original, simpler greedy search)? Choice depends on (a) how strict we want
  the “patent re-implementation” framing to be and (b) which team has time to
  bring the chosen implementation to parity.
- Whether to close **D3** (Gauss-Newton training) — currently the patent
  track uses Adam, which is a deviation. For a thesis-grade comparison this
  matters; for a poster, probably fine to defer with a footnote.
