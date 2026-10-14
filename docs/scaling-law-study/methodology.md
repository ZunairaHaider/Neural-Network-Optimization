# Methodology — pruning workstream

> **Final-report seed.** This file is meant to be expanded with paper
> citations (Frankle & Carbin 2019; Lee, Ajanthan & Torr 2019; Molchanov et
> al. 2017/2019; LeCun, Denker & Solla 1990; Hassibi & Stork 1993; Denil et
> al. 2013; Sainath et al. 2013; Mariet & Sra 2016; Srinivas & Babu 2015)
> and figures, then dropped into the methodology chapter of the written
> report due ~31 July 2026. The patent-track methodology lives in
> `patent_track_recommendations.md` (deferred to a later session).

## 1. Datasets

We study a single regression problem at three input dimensionalities so
that scaling-law trends can be read off directly:

| Dataset | dim | target f(x) | N_train / N_test | noise σ | noise floor σ² |
|---|---|---|---|---|---|
| A | 1 | (x − 1)² = x² − 2x + 1 | 4000 / 1000 | 0.0 | 0.0 |
| B | 1 | A + 𝒩(0, σ²) | 4000 / 1000 | 2.0 | 4.0 |
| A | 2 | xᵀA₂x + b₂ᵀx + c, A₂ SPD (seed 44) | 4000 / 1000 | 0.0 | 0.0 |
| B | 2 | A_2D + 𝒩(0, σ²) | 4000 / 1000 | 5.30 | 28.1 |
| A | 3 | xᵀA₃x + b₃ᵀx + c, A₃ SPD (seed 45) | 4000 / 1000 | 0.0 | 0.0 |
| B | 3 | A_3D + 𝒩(0, σ²) | 4000 / 1000 | 23.78 | 565.5 |

- Domain `x ∈ [−3, 3]^d`; 1-D uses a uniform linspace grid that's then
  random-shuffled; d ∈ {2, 3} use IID uniform sampling.
- `A_d, b_d` for d > 1 are constructed with a deterministic generator
  (`A = L Lᵀ + 0.5 I` from `torch.randn(d, d, generator=seed+d)`), guaranteed
  symmetric positive-definite → unique interior minimum. Exact values are
  persisted to `data/processed/quadratic_params.json` for reproducibility.
- The noise σ for Dataset B is scaled to keep the **noise-to-signal-range
  ratio at 12.5 %** of the empirical peak-to-peak (16 in 1-D, 42.4 in 2-D,
  190.3 in 3-D). The 1-D σ is unchanged from prior work.
- Generation is deterministic; seeds `42` (master), `44/45` (A_d/b_d for
  d=2/3), `4242+d` (sampling), `7777+d` (noise). 1-D .pt files are
  bit-identical to the pre-extension dataset (verified).

## 2. Baseline MLP

`MLP(n_layers, n_neurons, activation, input_dim=d, output_dim=1)`:
`Linear → act → … → Linear` (no activation on the regression head).
Activations: ReLU / Tanh / GELU / Sigmoid; the HPO picks per-dataset.

Flat-vector accessors `get_weight_vector()` / `set_weights_from_vector()`
exist for OBD and OBS, which treat all parameters as one concatenated vector.
Per-layer parameter iteration order is fixed
(`layers[0].weight, layers[0].bias, layers[1].weight, …`) — every pruning
method depends on this layout.

## 3. HPO + baseline training

Optimiser: **full-batch Adam** (the dataset is small enough for this to be
the right default; mini-batches are used only during pruning fine-tunes).
Loss: MSE. Early stopping on validation MSE with patience 20 (HPO) / 30
(final retrain), max epochs 500 (HPO) / 1000 (retrain).

Three-stage Optuna TPE search:
- Stage 1: `n_layers ∈ {1, 2}`, `n_neurons ∈ {4..16}` — 120 trials.
- Stage 2: `n_layers ∈ {2, 3}`, `n_neurons ∈ {12..32}` — 80 trials,
  conditional on improvement potential.
- Stage 3: `n_layers ∈ {2, 3}`, `n_neurons ∈ {32, 48}` — 40 trials,
  conditional on ≥ 5 % stage-2 improvement.

Each trial also samples `activation`, `lr` (log-uniform), `weight_decay`.
Trials with > 5 000 params are pruned early. Validation uses a single random
IID 20 % split per trial (re-seeded per stage); no repeated CV — speed-for-
seed trade-off compensated by the tolerance-based selection rule:
**smallest n_params within 5 %** of the best val MSE. Final retrain on the
full 4000-point pool. **Test set is evaluated exactly once.**

Selected architectures (post-Phase-0 audit):

| Dataset | d=1 | d=2 | d=3 |
|---|---|---|---|
| A | 2×16 ReLU (321 p) | 3×32 GELU (2241 p) | 3×48 GELU (4945 p) |
| B | 1×16 ReLU (49 p)  | 2×10 ReLU (151 p)  | 2×4 ReLU (41 p)    |

The d=3 / B architecture being smaller than d=1 / A reflects the HPO selection
rule prioritising small models near the noise floor: at σ² ≈ 566, even tiny
models reach it.

## 4. Pruning loop

A single iterative loop (`src/pruning/pruner.py::run_pruning`) supports all
ten methods. For each sparsity target ∈ {10 %, 20 %, …}:

1. Per-method **saliency** computation (see §5).
2. **Drop** weights / neurons until the target sparsity is hit; the
   structured methods (`struct`, `activ`, `redundancy`) drop whole neurons.
3. **Fine-tune** the surviving weights: Adam, mini-batch
   `FINETUNE_BATCH_SIZE = 24`, patience 10, max 200 epochs. After each
   optimiser step, masked weights are zeroed so they stay pruned.
4. **Adaptive stop**: if test MSE > 3 × baseline **and** sparsity ≥ 50 %,
   stop. Always go to at least 50 %; cap at 95 %.

**Validation for fine-tune early stopping** uses a held-out 20 % split of the
training pool (audit §2.1 leakage fix: prior pruning runs used the test set
for early stopping). The split is computed once per run with a dedicated
seed (4242).

**Calibration subset for saliency / Hessian / activation calibration.** For
the per-target saliency computations (OBD, OBS, Taylor-FO, SNIP) and the
activation-based methods (`activ`, `redundancy`), we score on a fixed random
subset of 1 000 training samples (`CALIB_SAMPLES = 1000`, seed `7373`). The
saliency is itself an *estimate*; 1 000 samples are enough to rank weights
within noise, and the change cuts per-target wall-time roughly 4× compared
to the full 4 000-pool. The fine-tune still mini-batches the full pool.
A note on continuity: the 1-D and 2-D OBD/OBS sweeps in this branch were
launched **before** this change landed and therefore used the full pool; the
3-D sweep uses the calibration subset. The expected effect on saliency
ordering is small and within seed-noise, but a future re-run of 1-D / 2-D
on the new protocol is recommended for a paper-grade comparison.

## 5. Pruning methods

| Method | Saliency / criterion | Granularity | Reference |
|---|---|---|---|
| **obd** | 0.5 · H_ii · w_i² (diag Hessian, exact for n ≤ 5 000; Fisher fallback) | scalar weight | LeCun et al. 1990 |
| **obs** | w_q²/(2·[H⁻¹]_qq) + closed-form δw; rank-1 H⁻¹ update | scalar weight + correction | Hassibi & Stork 1993 |
| **mag** | \|w_i\| | scalar weight | (folklore — see Han et al. 2015) |
| **struct** | ‖W_l[j, :]‖₂ (outgoing-row L2) | whole neuron | Li et al. 2017 |
| **activ** | mean-absolute-activation of each hidden unit | whole neuron | Hu et al. 2016 |
| **redundancy** | activation correlation > 0.99, merge w/ scaled transfer | neuron pair | Mariet & Sra 2016; Srinivas & Babu 2015 |
| **lowrank** | smallest singular value first (per-Linear SVD) | rank | Denil et al. 2013; Sainath et al. 2013 |
| **taylor_fo** | \|w_i · g_i\| (first-order Taylor) | scalar weight | Molchanov et al. 2017/2019 |
| **snip** | \|w_i · g_i\| at *random init*, single-shot top-k mask | scalar weight (prune-at-init) | Lee, Ajanthan, Torr 2019 |
| **imp_rewind** | \|w_i\| then rewind survivors to init before retrain | scalar weight (iterative LTH) | Frankle & Carbin 2019; Frankle et al. 2020 |

The fresh-init checkpoint used by SNIP and IMP+rewind is a Kaiming-uniform
draw with seed `INIT_SEED = 1337`, deterministic and reproducible.

### OBS regularisation

Tikhonov λ = 1e-4; if `cond(H + λI) > 1e10`, λ is bumped 10× **and**
re-checked. If still ill-conditioned, `compute_hessian_inv` returns `None`
and the caller falls back to OBD-diagonal saliency (audit §2.3 fix —
previous code did a one-shot bump without re-checking).

### Lowrank specifics

The low-rank module re-parameterises each `Linear` as `x → Vᵀ → diag(S) → U`,
zeroes the smallest singular value, fine-tunes the UV-factored model, and
writes the reconstructed dense weight back. For input or output dim = 1, the
factoring is degenerate (rank already 1 in the boundary layers), so the
method is most informative on the middle layer.

## 6. FLOPs accounting

For a pruned MLP with arbitrary weight mask:
`FLOPs_forward = Σ_layers (2 · alive_weights) + Σ_layers alive_biases`
(multiply-add counted as 2, bias add as 1). For structured methods, dropped
neurons take both their incoming and outgoing weights to zero, so the count
remains correct without special-casing.

## 7. Reproducibility

- Master seed: `torch.manual_seed(42)` / `np.random.seed(42)` / `random.seed(42)`
  at module load of every entry script.
- Data generation: deterministic generators per (dim, role); see §1.
- HPO: each Optuna stage re-seeded.
- Pruning: fine-tune val split seeded `4242`; fresh-init checkpoint seeded
  `1337`.
- All artifacts are written to `results/{baseline,pruning,scaling_law}/`;
  re-running any cell overwrites just that cell.
