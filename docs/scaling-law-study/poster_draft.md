# Poster draft — DIN A0 landscape

> Drop-in content for the 6-block layout below. Each block has a target word
> budget for a DIN A0 poster (≈ 1100 × 840 mm landscape). Total ≈ 350 words
> on the poster itself — the rest of the detail lives in
> [report.md](report.md). Figures referenced are in
> `results/scaling_law/*.png`.
>
> Style note for the layout step: use a single sans-serif family
> (e.g. Inter) at heading 60 pt / body 24 pt. Two-row header strip; six
> body blocks below; method legend along the bottom.

---

## ⚙ Title block (top strip)

**Pruning a small regression MLP scales differently than you'd guess**

*Ten pruning methods, two quadratic regression datasets, three input
dimensions — same architecture story, different headlines.*

> Marko Prohaska · ANN Optimisation Study · 2026
> Repo `CS_ANN`, branch `scaling-law-study`

---

## 1️⃣ Problem & question  (~50 words)

We train an MLP to fit a multivariate quadratic
**f(x) = xᵀA x + bᵀx + c** over `[−3, 3]^d`, d ∈ {1, 2, 3}, clean (A) and
noisy (B, σ = 12.5 % peak-to-peak). For each setup we tune an HPO-picked
baseline and prune it with **ten** algorithms. The question:

> *Does the same method dominate at every dimension and noise level?*

---

## 2️⃣ Methodology  (~80 words)

10 pruning methods → 6 (d × dataset) cells → up to 9 sparsity targets
(10 %–90 %) → 540 data points after fine-tune.

- Baselines selected by 3-stage Optuna TPE; smallest model within 5 % of
  best val MSE.
- Fine-tune mini-batch (size 24), early-stopped on a held-out val slice
  (audit §2.1 leakage fix landed mid-study).
- Saliency / Hessian / activations computed on a 1 000-sample calibration
  subset for speed.
- FLOPs counted as 2 × alive weights + biases.

See [docs/scaling-law-study/methodology.md](methodology.md) for full detail.

---

## 3️⃣ Pruning methods (table)  (~60 words)

| Class | Methods | Granularity |
|---|---|---|
| Second-order | **OBD**, **OBS** | scalar |
| First-order  | **Taylor-FO**, **mag** | scalar |
| Structured   | **struct**, **activ**, **redundancy** | neuron |
| Low-rank     | **lowrank** | rank |
| Prune-at-init / iterative LTH | **SNIP**, **IMP + rewind** | scalar (different protocol) |

Hessian for OBD/OBS exact ≤ 3 000 params, OBD-diagonal fallback above.

---

## 4️⃣ Headline charts (left → right)

1. **Fig. A** — [`fig_dim_vs_max_sparsity_A.png`](../../results/scaling_law/fig_dim_vs_max_sparsity_A.png):
   Max sparsity with test MSE ≤ 2× baseline, **Dataset A**, vs d.
   The compressibility ranking *changes* with dimension.
2. **Fig. B** — [`fig_dim_vs_max_sparsity_B.png`](../../results/scaling_law/fig_dim_vs_max_sparsity_B.png):
   Same, **Dataset B**. Noise floor caps the achievable test MSE → almost
   every method pushes to 80–90 %.
3. **Fig. C** — [`fig_params_vs_mse_A_3d.png`](../../results/scaling_law/fig_params_vs_mse_A_3d.png):
   Test MSE vs surviving params at d=3, A. The “Pareto front” reveals which
   method is cheapest at a given accuracy budget.
4. **Fig. D** (optional) — `fig_dim_vs_flops_at_target_A.png`:
   FLOPs of the smallest network meeting a 2× MSE budget, vs d. Frames the
   pruning win in deployment terms.

If the printer can fit two more columns, also drop in
`fig_sparsity_vs_mse_A_2d.png` (mid-dim Pareto sanity check).

---

## 5️⃣ Five headline findings  (~120 words)

1. **Clean ⇒ second-order, noisy ⇒ everything plateaus at the noise floor.**
   On Dataset A, OBD / Taylor-FO / activ dominate the high-d Pareto. On
   Dataset B, ~all methods reach 80–90 % sparsity at the 2× MSE budget
   because the floor σ² limits how much accuracy degradation is even
   visible.
2. **OBS does not generalise above d = 1.** The Hessian becomes
   ill-conditioned at the trained minimum (cond > 10¹²) and the closed-form
   weight correction amplifies error along low-curvature directions. Falls
   back to OBD-diagonal at d ≥ 3 via our auto-detection.
3. **IMP + weight rewinding is the d ≥ 2 dark horse** — weakest at d = 1
   (over-parameterised → lottery tickets don't help), strongest at d ≥ 2.
4. **SNIP (prune-at-init) is the weakest single-shot baseline** but
   recovers as d grows (more weights to score → better init mask).
5. **Low-rank pruning fails for low-d regression**: the input / output
   linear is rank 1 by construction; only the middle layers compress.

---

## 6️⃣ Take-aways for the field  (~50 words)

- *Method choice is dimension-dependent.* A poster ranking from d=1 alone
  would mis-rank methods for d ≥ 2.
- *Noise floor masks pruning damage* — high-noise regression makes pruning
  look effortless.
- *Audit-fixed leakage materially changes high-sparsity numbers* — old
  OBS-on-B 80 % MSE was 8× the noise floor; corrected, it sits at the floor.

> Companion report: [docs/scaling-law-study/report.md](report.md).
> Code & figures: `github.com/sima2006/CS_ANN`, branch
> `scaling-law-study`.

---

## Layout footer

Method legend strip (consistent colours across all panels):

OBD ▮ steelblue   OBS ▮ darkorange   mag ▮ seagreen   struct ▮ crimson
activ ▮ purple   redundancy ▮ teal   lowrank ▮ goldenrod
Taylor-FO ▮ indianred   SNIP ▮ slategrey   IMP+rewind ▮ darkgreen
