# Improvement & Reference Package

**Patent:** US 12,124,779 B2  

## Contents

- [Part I — Patent Implementation Improvement Analysis](#part-i--patent-implementation-improvement-analysis)
- [Part II — Pruning Improvement Analysis](#part-ii--pruning-improvement-analysis)
- [Part III — Reference Group Assignment](#part-iii--reference-group-assignment)

---

# Part I — Patent Implementation Improvement Analysis

## 1. Patent core mechanism and current implementation status

### 1.1 Core technical approach

US 12,124,779 B2 builds a parsimonious feed-forward network from a minimal initial topology through iterative additive and subtractive topology-optimization phases, mainly for digital-twin modelling of physical systems.

The main distinguishing innovations are:

- **Lagrangian error-variation estimation**  
  Uses equations 5–14 and Claims 3–5 to rank candidate topologies without full retraining.

- **Zero-memory Gauss–Newton optimiser**  
  Claim 7 supports fast local convergence after each topology selection.

- **Validation-based stopping and structural extensions**  
  Claim 8 introduces validation-based stopping. Claims 9–11 introduce compression–decompression blocks. Claim 1 includes cross-layer skip connections.

### 1.2 Key gaps from the current claim mapping

| Claim | Status | Gap |
|---|---|---|
| Claim 1 | Partial | `ADD_SKIP` in `topology.py` generates only one source–target pair; exhaustive cross-layer enumeration is absent. |
| Claim 10 | Not implemented | The modelling block and compression/decompression are separate; the compressed latent representation is never fed into the main network. |
| Claim 11 | Not implemented | The dynamic-system recurrent form `X_{t+1}=F(X_t,P_t)+G_t` is absent. Current datasets are static `Y=f(X)`. |
| Claims 9/15/16 | Partial | The compression block exists but is disabled by default and is not connected to the prediction network. |

---

## 2. Improvement directions

## 2.1 Exhaustive cross-layer skip-connection enumeration

**Related claim:** Claim 1  
**References:**  
- US 2019/0130278 — Karras, ProGAN  
- US 2019/0171936 — Karras, progressive modification of NN topology  
- US 2019/0080240 — Andoni, genetic-algorithm NN generation  

### Why this helps

ProGAN and US 2019/0171936 both grow a topology during training and retain additions that improve the model. This is conceptually close to the additive phase of US 12,124,779 B2.

Andoni’s genetic-algorithm approach generates and mutates many topology candidates, then selects them according to a fitness score. This provides a useful analogy for enumerating and ranking candidate network modifications.

### Implementation direction

In `topology.py`, update `generate_candidates()` so that `ADD_SKIP` enumerates all valid cross-layer pairs:

```text
ADD_SKIP_{i}_{j}, where source layer i < target layer j
```

For larger networks, cap the enumeration at about `K ≈ 50` sampled pairs.

Each candidate should then be scored using the Lagrangian estimate `ΔL`, and the candidate with the most negative estimated value should be selected, following the optimal-candidate principle of Claim 2.

### Expected benefit

This makes `ADD_SKIP` consistent with the patent’s optimal-candidate selection logic and may find better sparse shortcuts, especially for high-dimensional datasets.

---

## 2.2 Connect the modelling block into the compression–decompression pipeline

**Related claim:** Claim 10  
**References:**  
- US 2018/0322388 — O’Shea, end-to-end autoencoder over a channel  
- US 6,038,389 — Rahon & Masmoudi, reduced-order modelling  
- US 2021/0357555 — Liu, clustering reduced-order modelling  

### Why this helps

O’Shea trains an encoder and decoder jointly end-to-end. This matches the required structure of Claim 10:

```text
compress → model → decompress
```

Rahon and Liu both perform modelling inside a reduced or compressed representation and then reconstruct the full output. This is exactly what the modelling block in Claim 10 should do.

### Implementation direction

1. Add a `modeling_block` attribute to `CompressionSystem` in `compression.py`.
2. The modelling block should take the compressed latent representation `x_i` as input.
3. In `architecture_search()`, first optimise the compression block `C` and decompression block `D` separately.
4. Insert the modelling block between them.
5. Run a joint fixed-topology fine-tuning stage.
6. Update the data flow in `main.py`:

```text
X → C → x_compressed → modelling → x_pred → D → Y_pred
```

### Expected benefit

This completes Claim 10 and enables the implementation to handle genuinely high-dimensional physical states, especially in very-low-data regimes such as those described in the patent’s Table 2.

---

## 2.3 Dynamic-system recurrent modelling

**Related claim:** Claim 11  
**Reference status:** No cited patent is a direct recurrent-dynamic-NN match. This should therefore be implemented mainly from the patent’s own equations 16–18.

**Domain-support references:**  
- US 6,038,389 — Rahon, dynamic reservoir simulation  
- US 2021/0357555 — Liu, PDE-based physical-system prediction  
- US 2022/0092240 — Chi, iterative physical optimisation  

### Implementation direction

1. Add time-series support in `dataset_loader.py`:

```text
(X_t, P_t, G_t) → X_{t+1}
```

2. Implement recurrent rollout in `network.py`:

```python
recurrent_forward(X0, P_seq, G_seq, k)
```

3. Add the dynamic objective in `architecture_search.py`:

```text
min Σ || h(x_i, p_i) + g_i − x_{i+1} ||²
```

4. Score topology candidates against multi-step error using the Lagrangian estimate.

### Expected benefit

This would reproduce the dynamic digital-twin functionality described around the sodium-melting-front experiments in Figures 7–9 of the patent and would implement the core dynamic-system part of Claim 11.

---

## 2.4 Use an ML surrogate to accelerate or enrich candidate scoring

**References:**  
- US 2022/0092240 — Chi  
- US 2019/0080240 — Andoni  

### Why this helps

Chi uses a DNN to predict expensive sensitivity values inside an optimisation loop. This is analogous to learning a cheap predictor for expensive Lagrangian estimates.

Andoni’s adaptive mutation probabilities suggest that candidate types can be weighted according to their recent success.

### Implementation direction

Add a small learned surrogate model that predicts `ΔL` for candidate topology changes. This surrogate can pre-rank candidates before the exact Lagrangian estimate is applied.

An optional physics-informed training objective can also be added:

```text
L_total = L_data + λ · L_physics
```

The value of `λ` can be scheduled inside the Claim-8 validation-based stopping logic.

### Expected benefit

This can reduce the number of expensive candidate evaluations and improve candidate ranking in small-data physical-system settings.

---

## 2.5 Candidate pre-screening to accelerate the search

**References:**  
- US 2019/0080240 — Andoni  
- US 2020/0167659 — Moon  

### Implementation direction

After `generate_candidates()` and before Lagrangian estimation, apply a cheap activation-variance or residual-based filter:

| Candidate type | Pre-screening rule |
|---|---|
| `ADD_NEURON` | Prefer layers with high activation variance. |
| `REMOVE_NEURON` | Prefer neurons with the lowest activation variance. |
| `ADD_SKIP` | Prefer layer pairs with the largest residuals. |

Only the top candidates should be passed into the Lagrangian estimation stage.

### Expected benefit

This could reduce Lagrangian evaluations by approximately 60–80% while preserving most of the candidate-selection quality.

---

## 2.6 Deploy the sparse network efficiently

**References:**  
- US 10,614,354 — Aydonat, FPGA CNN accelerator  
- US 11,544,539 — Zhang, NN compilation/conversion for constrained hardware  
- US 2019/0130246 — Katayama, reconfigurable virtual neurons  
- US 2015/0106311 — Birdwell, DANNA neuromorphic hardware  

### Implementation direction

In `network.py`, update `forward()` so that it switches to `scipy.sparse` or `torch.sparse` when sparsity exceeds 80%.

After training:

1. Zero weights with `|w| < ε`.
2. Store sparse matrices in CSR/CSC format.
3. Add a sparsity report to `visualization.py`.
4. Add an inference-time benchmark.

### Expected benefit

This can reduce inference memory and computation, supporting the patent’s reported “laptop, 1 min 13 s” regime and making embedded deployment more realistic.

---

## 2.7 Online adaptive update for evolving digital twins

**Reference:**  
- US 9,336,483 — Abeysooriya  

### Implementation direction

Add an `--online` flag to `architecture_search.py`.

When new data arrives:

1. Run a lightweight Lagrangian update round.
2. Add or remove only the connections required by the observed data shift.
3. Use the magnitude of the Lagrange multipliers to protect the most critical neurons during the update.

### Expected benefit

This allows the digital twin to adapt online as the physical system changes over time.

---

## 3. Patent implementation priority summary

| Priority | Improvement | Reference(s) | Files |
|---|---|---|---|
| P1 | Claim 10: compression–modelling–decompression | O’Shea; Rahon; Liu | `compression.py`, `architecture_search.py`, `main.py` |
| P1 | Claim 11: dynamic recurrent modelling | Patent equations 16–18; Rahon; Liu; Chi | `network.py`, `dataset_loader.py`, `training.py` |
| P2 | Full `ADD_SKIP` enumeration | Karras 0130278; Karras 0171936; Andoni | `topology.py`, `architecture_search.py` |
| P2 | ML-surrogate candidate scoring and optional physics term | Chi; Andoni; general physics-informed learning | `training.py`, `lagrangian.py` |
| P3 | Candidate pre-screening | Andoni; Moon | `topology.py` |
| P3 | Sparse/hardware deployment | Aydonat; Zhang; Katayama; Birdwell | `network.py`, `visualization.py` |
| P4 | Online adaptive update | Abeysooriya | `architecture_search.py`, `network.py` |

---

# Part II — Pruning Improvement Analysis

## 0. Executive summary

The repository implements the following pruning methods in `src/pruning/`:

- Magnitude pruning, or `MAG`
- Optimal Brain Damage, or `OBD`
- Optimal Brain Surgeon, or `OBS`
- Structural pruning
- Activation-based pruning
- Redundancy pruning
- Low-rank pruning
- SNIP
- Taylor-first-order pruning
- Iterative magnitude pruning with rewind, or lottery-ticket-style `IMP-with-rewind`

The primary reference for OBS is **US 5,636,326**, by Stork and Hassibi. This patent is the authoritative reference for `src/pruning/obs.py`, because it describes full-Hessian saliency and closed-form weight correction.

---

## 1. Observed limitations and fixes

| Method / area | Limitation | Suggested fix |
|---|---|---|
| MAG | Ignores weight interdependence. | Use iterative percentile-threshold pruning plus recalibration. |
| OBD / OBS | Hessian is recomputed too infrequently. | Recompute every 5% for small networks, use rank-1 inverse update in `obs.py`, and diagonal correction for `obd.py`. |
| Structural / Activation pruning | Granularity mismatch makes it hard to hit exact sparsity targets. | Combine structural pruning with a small unstructured step. |
| Redundancy pruning | Fully recomputes correlations. | Use incremental correlation updates and invalidate only entries touching the merged pair. |
| Low-rank pruning | Reporting is not directly comparable to weight sparsity. | Report weight-equivalent sparsity: `1 − rank(rows + cols)/(rows × cols)`. |
| Fine-tuning | No knowledge transfer from the dense model. | Use distillation-style fine-tuning from the dense teacher. |

---

## 2. Additions from cited references

## 2.1 OBS / OBD — US 5,636,326

**Reference:** US 5,636,326 — Stork / Hassibi  
**Relevant files:** `obs.py`, `obd.py`, `pruner.py`

Use the Stork / Hassibi patent as the authoritative specification for `obs.py`.

The OBS saliency is:

```text
L_q = w_q² / (2[H⁻¹]_{qq})
```

The closed-form correction is:

```text
δW = −(w_q / [H⁻¹]_{qq}) H⁻¹ e_q
```

This correction is applied to surviving weights after each deletion.

### Why this matters

This validates:

- More frequent Hessian updates
- Rank-1 inverse maintenance in `obs.py`
- Closed-form correction to reduce retraining

`obd.py` should remain the diagonal-Hessian companion to OBS.

---

## 2.2 Structural pruning by sub-structure extraction

**Reference:** US 2015/0106311 — Birdwell  
**Relevant files:** `struct_pruning.py`, `redundancy_pruning.py`

Birdwell identifies useful sub-structures and extracts unsuccessful ones. This is an evolutionary structural-pruning idea.

### Implementation direction

For `struct_pruning.py` and `redundancy_pruning.py`, score whole neurons or sub-structures by their contribution to a subtask, rather than using only L2 norm.

### Expected benefit

This makes structural pruning more semantically meaningful and better aligned with actual network function.

---

## 2.3 Hardware target for structured pruning

**Reference:** US 10,614,354 — Aydonat  
**Relevant file:** `src/analysis/compare.py`

Aydonat defines which sparsity patterns yield real speed-ups on an FPGA PE-array.

### Implementation direction

Use this reference to choose structured-pruning granularity and to define the inference-time benchmark in `compare.py`.

---

## 2.4 Inference benchmark for pruned networks

**Reference:** US 11,544,539 — Zhang  
**Relevant file:** `compare.py`

Zhang describes a neural-network-to-hardware compilation or conversion route. This supports adding inference-time benchmarking to the pruning analysis.

---

## 2.5 Realistic physical-system test data

**Reference:** US 11,616,471 — Narayanaswamy  
**Relevant files:** `data/`, `benchmarks`

This reference describes a physical-system NN application involving photovoltaic-array connection-topology optimisation.

### Implementation direction

Use it as a realistic source or design inspiration for Dataset D pruning experiments.

---

## 2.6 General-literature techniques

These are useful improvements, but they are not from the cited patent set.

### BN-scaling pruner

**Reference:** Liu et al., 2017  
**Relevant files:** `src/models/mlp.py`, new `bn_scaling.py`

Add a `LearnedGating` module after each hidden activation. The gate parameter `γ` should receive an L1 penalty.

Then:

1. Sort channels by `|γ|`.
2. Zero the smallest channels.
3. Fine-tune the model.

### Knowledge-distillation fine-tuning

**Reference:** Hinton et al., 2015  
**Relevant file:** `pruner.py`

Use the dense network as a teacher. The pruned student should minimise:

```text
α · L_MSE + (1 − α) · L_KL(student, teacher)
```

---

## 3. Pruning priority summary

| Priority | Improvement | Reference / source | Files |
|---|---|---|---|
| P1 | Strengthen OBS / OBD | US 5,636,326, Stork / Hassibi | `obs.py`, `obd.py`, `pruner.py` |
| P1 | Knowledge-distillation fine-tune | Hinton et al., 2015 | `pruner.py`, all methods |
| P2 | Sub-structure structural pruning | US 2015/0106311, Birdwell | `struct_pruning.py`, `redundancy_pruning.py` |
| P2 | Mixed structural + unstructured pruning; low-rank reporting | Code-driven | `struct`, `activ`, `low_rank`, `pruner.py` |
| P2 | FPGA-aware granularity and benchmark | Aydonat; Zhang | `compare.py` |
| P2 | BN-scaling pruner, optional | Liu et al., 2017 | `models/mlp.py`, new `bn_scaling.py` |
| P3 | Physical-system test data, Dataset D | US 11,616,471, Narayanaswamy | `data/`, `benchmarks` |
| P3 | Incremental correlation and adaptive rate | Code-driven | `redundancy_pruning.py`, `pruner.py` |
| P4 | Lottery-ticket reset | Already in `imp_rewind.py` | `imp_rewind.py`, `pruner.py` |

---

# Part III — Reference Group Assignment

## 1. References most useful to the Patent team

## 1.1 References that directly improve the implementation

| Reference | Help for the team | Improvement |
|---|---|---|
| US 2019/0130278 — Karras, ProGAN | Incrementally grows topology and keeps useful additions. | `ADD_SKIP` / additive phase |
| US 2019/0171936 — Karras | Grow-topology-during-training strategy. | Additive enumeration |
| US 2019/0080240 — Andoni | Generates and mutates topology candidates, then selects by fitness. | `ADD_SKIP` / pre-screening |
| US 2018/0322388 — O’Shea | Jointly trains encoder and decoder. | Implement Claim 10 |
| US 6,038,389 — Rahon & Masmoudi | Models inside a reduced basis and reconstructs. | Claim 10 / compression |
| US 2021/0357555 — Liu | Solves governing equations in a reduced space. | Claim 10 / domain |
| US 2022/0092240 — Chi | Uses a DNN to predict expensive sensitivities in an optimisation loop. | Candidate scoring |
| US 9,336,483 — Abeysooriya | Monitors a live model and trains/promotes replacement candidates. | Online adaptive update |
| US 10,614,354 — Aydonat | Defines deployable sparsity patterns. | Inference deployment |
| US 11,544,539 — Zhang | Compiles sparse networks to constrained hardware. | Inference deployment |

## 1.2 Background or weak-match references

| Reference | Help for the team | Improvement |
|---|---|---|
| US 2014/0067741 — Katayama | Hardware interconnect background for large NN systems. | Deployment background |
| US 2019/0130246 — Katayama | Maps NN topology onto reconfigurable hardware. | Deployment background |
| US 2021/0279553 — Zhu | AI-workflow orchestration and MLOps. | Deployment tooling |
| US 7,814,038 — Repici | Feedback-tolerant weight-update rule; contrast with Gauss–Newton. | Background |
| US 11,604,957 — Schubert | Hybrid optical and digital NN design. | Hardware background |
| US 11,158,096 — Schubert | Straight-through estimators for discrete design choices. | Discretisation idea |

---

## 2. References most useful to the Pruning team

| Reference | Help for the team | Improvement |
|---|---|---|
| US 5,636,326 — Stork / Hassibi, OBS | The method in `obs.py`: full-Hessian saliency plus closed-form correction. | Strengthen `obs.py` / `obd.py` |
| US 2015/0106311 — Birdwell, NIDA/DANNA | Identifies useful sub-structures and extracts unsuccessful ones. | `struct_pruning.py`, `redundancy_pruning.py` |
| US 10,614,354 — Aydonat, FPGA | Shows which sparsity patterns accelerate on hardware. | `compare.py` granularity |
| US 11,544,539 — Zhang | Deployment/conversion route and inference-time benchmark. | `compare.py` |
| US 11,616,471 — Narayanaswamy | Physical-system NN data source for pruning experiments. | Dataset D design |

---

## 2.1 General-literature techniques for the Pruning team

These are not from the cited patent set, but they are useful implementation additions.

- **BN-scaling pruner:** add `LearnedGating γ` and an L1 penalty in `src/models/mlp.py`, then create a new `bn_scaling.py`.
- **Knowledge-distillation fine-tuning:** use a dense teacher to guide the pruned student during fine-tuning.

---

## 3. References valuable to both teams

| Reference | Value to Patent team | Value to Pruning team |
|---|---|---|
| US 5,636,326 — Stork, OBS | Provides second-order saliency context for the subtractive phase and Lagrangian reasoning. | Authoritative OBS reference for `obs.py`. |
| US 2015/0106311 — Birdwell | Automatic sub-structure construction; related to add/remove topology search. | Sub-structure extraction for structural and redundancy pruning. |
| US 10,614,354 — Aydonat | Deployment substrate for the sparse network. | Accelerable sparsity patterns and benchmark target. |
| US 11,544,539 — Zhang | Compiles sparse networks for inference. | Inference-time benchmark for pruned networks. |
