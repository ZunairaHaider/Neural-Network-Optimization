# Phase 0 Audit — CS_ANN

**Branch:** `scaling-law-study` (off latest `main` at `eb7227a`)
**Date:** 2026-05-14
**Patent of record:** US12124779B2 — Bompard et al., ADAGOS, "Method of neural network construction for the simulation of physical systems" (issued 2024-10-22)

This audit is the Phase 0 deliverable. **No code has been changed.** Sign-off requested before any implementation work begins.

---

## 1. Branch hygiene

- Current branch is `scaling-law-study`, created from `main` at `eb7227a small fixes on redundancy`.
- The previously-active branch `baseline-hpo` is stale (0 commits ahead of `main`, missing the patent workstream and 4 of the 7 pruning methods). All work on this study should ignore it.
- Untracked artifacts that were on `baseline-hpo` (`results/pruning/mag/{A,B}/sparsity_*.{pt,json}`) have been **moved aside, not deleted**, to `.local-backup/mag-from-baseline-hpo/`. They were blocking the fast-forward to `main`. Restore or discard at your discretion.

---

## 2. Inventory — what exists today on `main`

### 2.1 MLP / training
- `src/models/mlp.py` — uniform-width feedforward MLP, torch. Hard-coded **input dim = 1, output dim = 1** (line 52: `sizes = [1] + [n_neurons]*n_layers + [1]`). Flat-vector accessors used by OBD/OBS.
- `src/training/train_baseline.py` — staged Optuna TPE HPO (3 stages, 120+80+40 trials), tolerance-based selection, retrain + locked-test eval.
- `src/training/hpo_utils.py` — `make_ood_splits` (despite the name now does random IID; docstring is stale), `train_one`.

### 2.2 Data
- `src/data/generate_data.py` — Dataset A (clean `(x−1)²`) and B (noisy, σ=2), x ∈ [−3, 3], N=5000, IID 4000/1000 split. **1-D inputs only.**
- `data/processed/` — `.pt` blobs for A and B + preview plot.

### 2.3 Pruning workstream
Seven methods registered in `src/pruning/pruner.py`:
| Method | File | Type |
|---|---|---|
| `obd` | `obd.py` | weight (diagonal Hessian) |
| `obs` | `obs.py` | weight (full H⁻¹ + correction) |
| `mag` | `mag.py` | weight (|w|) |
| `struct` | `struct_pruning.py` | structured neuron (L2 of outgoing row) |
| `activ` | `activ_pruning.py` | structured neuron (Mean Absolute Activation) |
| `redundancy` | `redundancy_pruning.py` | neuron merging by activation correlation > 0.99 |
| `lowrank` | `low_rank_pruning.py` | per-Linear SVD truncation |

Analysis: `src/analysis/compare.py` (11-metric table: sparsity, MSE, MSE/baseline, params, compression ratio, model size KB, memory saved %, inference ms, training time), `src/analysis/topology.py` (network graph plot).

### 2.4 Patent workstream (separate package, pure NumPy)
`patenting/` contains a self-contained pipeline:
- `main.py` — runs both datasets end-to-end.
- `network.py` — `DynamicTopologyNetwork(layer_sizes)` (dense fully-connected layers, ReLU hidden, identity output, He init).
- `topology.py` — `TopologyEditor`: `add_neuron(layer_idx=1)`, `remove_neuron(layer_idx=1)` (argmin Σ|W_row|), `add_layer(position=1, size=8)`.
- `architecture_search.py` — greedy cycle: try ADD_NEURON / REMOVE_NEURON / ADD_LAYER, train each candidate from scratch (300 / 300 / 1000 epochs), pick the best by `total_objective`. Stops when no candidate improves.
- `compression.py` — fixed-shape linear autoencoder (`latent_dim=6`), trained once on inputs, used only for *logging* `reconstruction_loss` per cycle. Not in the objective.
- `training.py` — vanilla gradient descent (`net.weights[i] -= lr * gW[i]`), fixed `lr=0.001`, no early stopping.
- `total_objective = MSE + 0.001 · complexity_cost` where `complexity_cost = 0.01·neurons + 0.0001·connections + 0.01·layers`.

Outputs already in `patenting/outputs/dataset_{A,B}/`: `mse_plot`, `recon_plot`, `neuron_plot`, `layer_plot`, `{initial,final}_architecture`, `metrics.txt` (4 lines), `history.npy`.

---

## 3. Pruning methodology review

### 3.1 What's solid
- HPO pipeline is well-staged with TPE, noise-floor early-exit, and a tolerance-based "smallest within X%" selection rule — appropriate for the budget.
- OBD: exact diagonal Hessian via double-backprop for n ≤ 5000, Fisher fallback otherwise with a warning. Mathematically clean.
- OBS: full Hessian via `torch.autograd.functional.hessian` with Tikhonov regularization (λ=1e-4, 10× bump if cond > 1e10). Rank-1 update of H⁻¹ within a sparsity step. Correct per Hassibi & Stork (1993).
- Fine-tune loop re-zeros pruned weights after every optimiser step — correct mask-projection.
- `compare.py` is dataset-agnostic and metric-rich.

### 3.2 Issues to fix (ordered by severity)

1. **TEST-SET LEAKAGE in fine-tuning early stopping.** `pruner.py:_finetune` passes `x_te, y_te` as the early-stopping criterion (lines 245, 139–140). Every pruned model is therefore *selected on the test set*, then *reported on that same test set*. This contaminates every sparsity number in the existing results. Fix: carve a held-out val split from the train pool inside the fine-tune loop (same OOD-flavoured logic as HPO), use it for early stopping, evaluate test once at the end of each sparsity target.

2. **`batch_size = 24` hard-coded in `pruner.py:218`.** Ignores the HPO-chosen batch size in `config["training_hparams"]`. Fix: read `config["training_hparams"].get("batch_size", <full-batch>)`.

3. **OBS condition-number check is one-shot.** After bumping λ by 10× the code does not re-check whether `cond` is now acceptable. If still bad, `torch.linalg.inv` silently returns a noisy inverse. Fix: log post-bump cond; if still > 1e10, fall back to OBD-diagonal saliency (already supported by the caller) instead of inverting.

4. **Dead-code call to `lr_mod.reconstruct(w)` in `pruner.py:389`.** `LowRankPruning` has no `reconstruct` method. The branch is unreachable because the lowrank path `continue`s earlier, but the line should be removed for clarity.

5. **`hpo_utils.make_ood_splits` docstring is stale.** Claims to bias splits toward the right edge; implementation is plain `np.permutation`. Either fix the docstring or actually implement OOD-flavoured splits (the latter is more in line with the project's "scaling law" framing). Recommend: fix docstring only — IID is the right choice for the current question.

6. **`select_neuron_weights` in `struct_pruning.py` and `activ_pruning.py` are O(neurons × out_features × loops in Python)**. Fine at current sizes (≤ 50 neurons) but won't scale to the 2D/3D nets we're about to build. Vectorise before higher dimensions.

7. **`_load_baseline` always writes `original_network.png` to the shared baseline dir.** Cross-method side-effect. Move to method-specific output dir.

### 3.3 Fine-tuning protocol vs. spec
You asked specifically that fine-tuning be "resume training after each pruning step with early stopping." The current loop *does* resume from the post-prune weights with patience=10 / max 200 epochs — that part is correct. The only blocker is the test-set leakage above; fixing it preserves the protocol.

---

## 4. Patent workstream faithfulness to US12124779B2

The repo's `patenting/` package captures the **spirit** of the patent (additive/subtractive topology search with candidate-selection by error variation) but **misses the algorithmic innovations** that make the patent novel. Discrepancies, cited to the patent text (OCR'd from the PDF; page refs below):

| # | Patent (US12124779B2) | Repo `patenting/` | Severity |
|---|---|---|---|
| **D1** | "The input of a processing node of one layer can be connected to the output of a node of *any of the previously calculated layers*" — i.e. **skip connections across non-adjacent layers** are the headline structural feature (col. 11–12, claim 1). The weight matrix W_i has columns indexed over Σ_{k≤i} n_k. | No skip-connection support. `TopologyEditor` only widens layer `layer_idx=1` or inserts a full layer between two. | **Out of scope — documented** — the study is restricted to strict-MLP architectures (fully layered, no skips). Skip connections turn an MLP into a more general feedforward DAG. This delta is therefore not a defect of our implementation but a deliberate scope choice; downstream we cannot claim full patent-faithfulness on this point. |
| **D2** | Error variation is **estimated** by a Lagrangian: `ℒ(Γⁿ, Wⁿ, X, Λ) − J(Γⁿ⁻¹, Wⁿ⁻¹*)` (col. 5). For an additive step, Wⁿ is initialised from Wⁿ⁻¹* on shared edges and zero on new edges, then `Wⁿ` is updated by minimising `ℒ` w.r.t. only the new weights. This makes candidate ranking **cheap** — no full retrain per candidate. | Repo trains every candidate from scratch for 300 / 300 / 1000 epochs and reads the resulting loss. Computationally orders of magnitude more expensive and methodologically different. | **Critical** — core algorithmic claim. |
| **D3** | Training uses the **zero-memory Gauss-Newton method** (Fehrenbach et al. 2006), explicitly justified as needed for "quantitative and deep phenomena" (col. 13). | `training.py` uses vanilla full-batch gradient descent, fixed `lr=0.001`, no momentum, no early stopping. | **Major** — affects every fit quality measurement. |
| **D4** | Subtractive phase: **all nodes and all links are candidates in turn**, each scored by the Lagrangian (col. 13). | `remove_neuron` removes exactly one hidden neuron, chosen by argmin Σ \|W_row\|. No link removal. Single fixed `layer_idx=1`. | **Major** |
| **D5** | Additive phase: for large networks, ~1000 random candidate links/nodes per cycle (col. 13). | Three fixed candidates per cycle (add_neuron, remove_neuron, add_layer of size 8). | **Major** |
| **D6** | Bias cell **only in the input layer**; sigmoid activation hidden, identity output (col. 12). | Bias in every layer; ReLU hidden, identity output. | **Minor** — choice of activation is arguably fine, but the bias-only-on-input is part of the patent's parsimony argument. |
| **D7** | Stopping uses both **training and validation** errors. A subtractive step is triggered when train-error drops below val-error beyond a tolerance (col. 6). | No validation set inside the architecture search. Objective is `train_MSE + 0.001 · ad-hoc complexity_cost`. | **Major** — wrong stopping criterion. |
| **D8** | Compression block is built **incrementally** by recursive subdivision of a central hidden layer + topological optimisation of the compression and decompression blocks (col. 6–7). | A flat `CompressionSystem(input_dim → latent=6 → input_dim)` linear AE is trained once. Reconstruction loss is logged but not in the objective. | **Major** — the patent's compression construction isn't implemented; current AE is unrelated machinery. |
| **D9** | Initial topology is minimal: input + **single hidden layer with one neuron** + output (col. 13). | `[X.shape[0], 2, Y.shape[0]]` — close, but hard-coded width 2 and no minimality argument. | **Minor** |
| **D10** | `compression_system.reconstruction_loss(X_test)` is computed in the search loop using **test data**. | Same as patent? — No, the patent's compression block has its own train/val regime; the repo uses test set for the reported loss, which is at least bookkeeping noise but reads suspicious. | **Minor** |

**Overall judgement:** the current `patenting/` package is a baseline implementation of the *idea* (greedy add/remove topology search with a complexity penalty) but is not faithful to the patent's distinctive contributions. **Within the strict-MLP scope of this study**, D1 is consciously out of scope (skip connections require leaving MLP territory). The other distinguishing deltas — D2 (Lagrangian estimate), D4 (all-nodes/all-links candidates), D5 (broad candidate sampling), D7 (validation-set stopping) — are MLP-compatible and should be closed in the revised implementation. D3 (Gauss-Newton) and D8 (incremental compression block) remain larger scope items that can be flagged as bounded follow-ups.

**Implementation strategy.** Rather than modifying the existing `patenting/` package (which Aryan and Yanqin own), a **new top-level package `patent-method-revised/`** will be added that re-implements the patent method within MLP constraints, with explicit patent column citations. The original `patenting/` package is left untouched as the reference implementation for comparison.

> OCR caveat: the patent PDF is image-only (24 pages, scanned via tiff2pdf). I OCR'd with tesseract; equation rendering is partial. The algorithmic claims above were extracted from prose in cols. 4–13. Recommend a human re-read of cols. 6 (Lagrangian) and 14 (compression block) before we commit to the implementation details of D2 and D8.

---

## 5. Metrics & visualisation parity (pruning vs. patent track)

What pruning has and patent track lacks today:

| Artifact | Pruning | Patent |
|---|---|---|
| Comparison table (CSV + rich txt) with sparsity, MSE, params, compression ratio, model size, memory saved %, inference ms, training time | ✅ 11-metric `comparison_table.csv` + rich txt | ❌ 4-line `metrics.txt` per dataset |
| Sparsity-vs-MSE curve | ✅ | ❌ (only cycle-vs-MSE) |
| Memory-saved-vs-MSE plot | ✅ | ❌ |
| Per-layer alive-weight counts | ✅ `architecture_summary.txt` | Partial — `final_architecture.png` only |
| Network topology graph | ✅ `analysis/topology.py` | ✅ NetworkX-based (`visualization.py`) |
| MSE history | n/a (single-shot) | ✅ |
| Reconstruction loss history | n/a | ✅ |
| Inference latency | ✅ median over 100 runs | ❌ |
| Training wall-clock | ✅ (when checkpointed) | ❌ (per-cycle time not logged) |
| **FLOPs** | ❌ — you asked for this | ❌ |

Plan (in Phase 1, not now): add inference-latency + wall-clock + FLOPs to the patent track, and add a sparsity-equivalent x-axis (`1 − params / baseline_params`) so the two tracks can share a comparison table. FLOPs needs adding for both.

---

## 6. Additional pruning methods worth adding

With 7 methods already in place, I'd cap additions at **3 high-value ones** rather than reaching for 5. Each justified against what's already present:

1. **Iterative Magnitude Pruning with Weight Rewinding (lottery-ticket style — Frankle & Carbin, 2019; Frankle et al. 2020).** *Rewinds* surviving weights to a stored early-training snapshot after each prune, instead of fine-tuning from the post-prune state. Different recovery dynamics from anything currently in the repo. Cheap given the model size. **Worth adding** — it is one of the few canonical compressive baselines we are missing.

2. **SNIP (Lee, Ajanthan, Torr, 2019) — Single-shot connection sensitivity at initialisation.** Computes a per-weight saliency from one mini-batch gradient at init, prunes once, then trains. Different paradigm from everything else (everything in the repo is *post-training* pruning). Adds a "prune-at-init" datapoint to the scaling-law comparison. **Worth adding.**

3. **Taylor-FO (first-order Taylor expansion criterion — Molchanov et al. 2017/2019).** Per-neuron / per-filter saliency `|g · w|` averaged over a mini-batch. Cheaper than OBD's diagonal Hessian, often competitive. Provides a first-order analogue alongside OBD's second-order. Pairs naturally with `struct_pruning` and `activ_pruning`. **Worth adding.**

Methods I considered and **dropped**:

- **GraSP (Wang, Zhang, Grosse, 2020)** — saliency via gradient flow; redundant given SNIP captures the at-init regime and our models are tiny.
- **Movement pruning (Sanh, Wolf, Rush, 2020)** — designed for large LM fine-tuning; out of scope for an MLP regression study.
- **L0-regularised pruning (Louizos et al. 2018)** — would force a training-loop rewrite; overkill for the budget.

Recommendation: add IMP+rewinding, SNIP, Taylor-FO. Skip the rest.

---

## 7. Dataset extension to 2D / 3D (Phase 1 scope-setter)

Datasets A and B today: `f(x) = (x−1)²`, x ∈ ℝ.

Per your confirmation, the 2D/3D versions should be **general multivariate quadratics** with cross terms:

  `f(x) = xᵀ A x + bᵀ x + c`,   x ∈ ℝ^d

with `A` a fixed, **reproducibly random** symmetric PSD matrix (seed 42), `b` and `c` likewise fixed. PSD ensures a unique minimum; reproducibility means every run sees the same target function. For Dataset B variants, `y = f(x) + ε`, ε ~ 𝒩(0, σ²), with σ scaled to match the existing 12.5%-of-peak-to-peak ratio in d=1.

Sampling grid for d > 1: a uniform IID sample over the box `[−3, 3]^d`, N=5000, IID 4000/1000 split (consistent with the current 1-D pipeline). No tensor-grid blow-up.

The MLP needs an `input_dim` parameter (currently hard-coded to 1 in `mlp.py:52`). The patent's `DynamicTopologyNetwork` already accepts arbitrary `layer_sizes[0]`, so no change there.

---

## 8. Open questions / risks before Phase 1

1. **Patent faithfulness scope.** Closing D1+D2+D4+D7 is non-trivial work — it's effectively a rewrite of `architecture_search.py` and `topology.py`. **Are you OK with that scope, or would you rather I document the deltas and run the existing implementation as-is for Phase 1, then close gaps as a Phase 1.5?** Strong recommendation: at minimum close D1 (skip connections) and D7 (val-set stopping) — without those, the patent track is materially mis-implemented.

2. **OBS Hessian stability at higher dimensions.** Current baseline has 321 params (Dataset A). Multi-D quadratics may push HPO to pick larger models; if n > ~500 OBS already falls back to Fisher diagonal (with a warning). I will flag any case where this fallback triggers, per your request.

3. **HPO budget × cells.** Six (dataset, dim) cells × ~240 trials = 1440 HPO trials. At ~1–2s per trial on CPU/MPS this is 30–60 minutes total — fine on the MacBook. Pruning runs are ~2-5 min each × 7 methods × 6 cells = 1–3.5 hours. Doable in one session; no Colab needed unless OBS hits the O(n²) wall.

4. **Test-set leakage fix (Issue 3.2.1) will move existing numbers.** Once fixed, the headline figures in `results/pruning/` will change. We should re-baseline before claiming any scaling-law trend.

5. **OCR confidence on patent equations.** The Lagrangian (col. 5) and the compression-block recursion (col. 14) deserve a human re-read before I write the corresponding code. I'll flag specific equation OCRs to you when implementing D2 / D8.

---

## 9. Proposed Phase 1 execution plan (after sign-off)

In order, each as its own commit:

1. **Build `patent-method-revised/`** — strict-MLP-compatible re-implementation closing D2, D4, D5, D7 (and optionally D6). Document D1, D3, D8 as deliberately deferred. Patent column references inline.
2. **Add documentation** to `low_rank_pruning.py` and `redundancy_pruning.py` (the two pruning files with the lightest module-level docstrings).
3. **Hand off** to a follow-up session via `handoff.md` covering: state of the branch, remaining Phase 1 work (data extension to 2D/3D, test-set leakage fix, new pruning methods, FLOPs, full sweep), and Phase 2 deliverables.

Items deferred to the next session (after handoff):

- Fix test-set leakage in `_finetune` (Issue 3.2.1). Held-out val from train pool. Re-baseline existing methods on existing 1-D datasets.
- Add `input_dim` to `MLP`. Verify 1-D Dataset A/B baselines reproduce.
- Extend `generate_data.py` to 2-D and 3-D general multivariate quadratics.
- Add 3 new pruning methods: IMP+rewinding, SNIP, Taylor-FO.
- Add FLOPs metric. Add inference-latency, wall-clock, FLOPs to patent track. Unify comparison schema.
- Run the full sweep (10 pruning methods + patent revised + original patent) × {A,B} × {1D, 2D, 3D}.
- Phase 2: scaling-law plots, comparison tables, narrative markdown, poster-ready figures for the DIN-A0 landscape template.

Larger patent items D3 (Gauss-Newton), D8 (incremental compression) remain queued — out of scope for this session and the immediate next one.

---

## 10. Sign-off checklist

Please confirm or push back on each before I write code:

- [ ] **Patent faithfulness scope** — close D1 + D7 in Phase 1, defer D2 / D3 / D8 to Phase 1.5? (Or do all in Phase 1?)
- [ ] **Pruning methods to add** — IMP+rewinding, SNIP, Taylor-FO (3, not 5)?
- [ ] **Dataset form for d > 1** — general multivariate quadratic with cross terms, fixed PSD A, fixed b, c, seed 42?
- [ ] **Test-set leakage fix** — agreed that this is fix-first-then-re-baseline, even if it changes existing numbers?
- [ ] **`.local-backup/`** — leave it where it is, or you'll clean up?
- [ ] **`baseline-hpo` branch** — ignore for the rest of this study?
- [ ] **Anything else surprising in the audit you want corrected before I start?**
