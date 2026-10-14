# Pruning Methods Summary
**Reference:** `src/pruning/` — iterative pruning suite

---

## The Setup

Start from a trained baseline MLP. Apply each pruning method across a shared sparsity schedule (10%, 20%, ..., up to 95%). At each sparsity target:

1. Remove weights/neurons until that sparsity level is reached
2. Fine-tune surviving weights (Adam, up to 200 epochs, early stopping on a held-out val split)
3. Evaluate test MSE once
4. Stop early if `test_mse > 3 × baseline_mse` and sparsity ≥ 50%

For each method you get a **MSE vs. sparsity curve**. The "best model" per method is the highest-sparsity checkpoint that stays within your acceptable error budget.

**No fixed thresholds** — the schedule is the control knob, not a threshold value.

**Calibration set:** 1000 samples drawn from the training pool, used for saliency/Hessian/activation scoring. Fine-tuning still uses the full training pool via mini-batches.

---

## The Taylor Expansion Foundation

All second-order methods start from: if we zero weight $w_q$ (i.e. $\delta w_q = -w_q$), what is the change in loss?

$$\delta \mathcal{L} \approx \underbrace{g_q (-w_q)}_{\text{first order}} + \underbrace{\frac{1}{2} H_{qq} w_q^2}_{\text{second order}}$$

At a trained optimum, $g_q \approx 0$, so the first-order term vanishes and:

$$\delta \mathcal{L} \approx \frac{1}{2} H_{qq} w_q^2$$

This is the basis for OBD. The goal across all methods: **remove the weight whose removal causes the smallest change in loss** — i.e. causes the least damage.

---

## Method 1 — Magnitude Pruning

**Criterion:** $S_i = |w_i|$

**Rule:** remove the smallest-magnitude unmasked weight each step.

**Intuition:** small weights contribute little to the output regardless of curvature. The implicit prior is $w \approx 0$ means the connection is near-irrelevant. This is wrong when a small weight sits at a sharp Hessian diagonal — OBD would catch that, magnitude won't.

**Cost:** O(P) — no gradients or Hessian needed.

---

## Method 2 — Taylor First-Order (Taylor FO)

**Criterion:**
$$S_i = |w_i \cdot g_i|$$

**Rule:** remove the weight with the smallest $|w_i g_i|$.

**Where it comes from:** first-order Taylor expansion of the loss change when zeroing $w_i$:
$$|\delta \mathcal{L}| \approx |g_i \cdot (-w_i)| = |g_i w_i|$$

So you are directly estimating: *which weight's removal causes the smallest first-order change in loss?* That is exactly the right question.

**Caveat:** at a true optimum $g_i = 0$ and all saliencies collapse to zero — the signal comes from the fact that practical training leaves residual gradient magnitude differences across weights.

**Cost:** O(P) — one forward + one backward pass.

---

## Method 3 — OBD (Optimal Brain Damage)

**Criterion:**
$$S_i = \frac{1}{2} H_{ii} w_i^2$$

**Rule:** remove the weight with the smallest $S_i$.

**Where it comes from:** same Taylor expansion but keeping the second-order term (valid at the optimum where $g \approx 0$). Uses only the **diagonal** of the Hessian.

**Why diagonal only?** Faster (O(P) vs O(P²) for the full inverse) and more stable — a diagonal matrix is singular only if an entry is exactly zero, which essentially never happens at a trained optimum. The off-diagonal terms (weight interactions) are discarded.

**Cost:** O(P) — diagonal Hessian via double backprop.

---

## Method 4 — OBS (Optimal Brain Surgeon)

**Criterion:**
$$S_q = \frac{w_q^2}{2 [H^{-1}]_{qq}}$$

**Weight correction:** after removing $w_q$, adjust all surviving weights by:
$$\delta W = -\frac{w_q}{[H^{-1}]_{qq}} H^{-1} e_q$$

**Where it comes from:** OBD discards off-diagonal Hessian terms. OBS instead solves the exact constrained problem: *minimize* $\frac{1}{2}\delta W^\top H \, \delta W$ *subject to* $e_q^\top \delta W = -w_q$ (i.e. weight $q$ is forced to zero).

Using a Lagrange multiplier $\lambda$ on the constraint:
$$\mathcal{L} = \frac{1}{2}\delta W^\top H \, \delta W + \lambda (e_q^\top \delta W + w_q)$$
$$\frac{\partial}{\partial \delta W} = 0 \implies H\,\delta W + \lambda e_q = 0 \implies \delta W = -\lambda H^{-1} e_q$$

Apply constraint: $e_q^\top \delta W = -w_q \implies \lambda = \frac{w_q}{[H^{-1}]_{qq}}$.

Substitute back: $S_q = \frac{w_q^2}{2[H^{-1}]_{qq}}$.

**The key difference from OBD:** OBS accounts for weight interactions (off-diagonal Hessian) and compensates surviving weights to partially undo the damage. More accurate, but O(P²) — needs the full Hessian inverse.

**In the code:** $H^{-1}$ is updated after each removal via a rank-1 update (no full recompute per step).

---

## Method 5 — Structured Pruning

**Granularity:** removes entire **neurons**, not individual weights. Result is a smaller dense network — real memory and speed savings on any hardware (unstructured sparsity requires special sparse kernels).

**Criterion:** L2 norm of the neuron's outgoing weight row:
$$S_j = \|W_l[j, :]\|_2$$

**Rule:** remove the neuron with the smallest outgoing L2 norm. One "removal step" zeros: the full row of $W_l$ (outgoing weights), the corresponding column of $W_{l+1}$ (incoming weights to the next layer), and the bias $b_j$.

**Intuition:** a neuron with small outgoing weights has little influence on the next layer regardless of what it computes. This is the weight-magnitude criterion applied at the neuron level.

**Cost:** O(P) — no Hessian, no forward pass needed.

---

## Method 6 — Activation Pruning

**Granularity:** same as Structured — removes entire neurons.

**Criterion:** Mean Absolute Activation (MAA) over the calibration set:
$$S_j = \frac{1}{N}\sum_{i=1}^N |h_j(x_i)|$$

**Rule:** remove the neuron with the smallest MAA.

**Why not just weight norm?** A neuron can have large weights but near-zero output if its inputs consistently cancel in the pre-activation sum, or if it is a dead ReLU (pre-activation always negative). Structured pruning misses these — activation stats catch them directly.

**Calibration set:** one forward pass over 1000 training samples collects post-activation outputs $h_j(x)$ for every hidden neuron. The average absolute firing magnitude is the score.

---

## Method 7 — Redundancy Pruning

**Idea:** if two neurons in the same layer have Pearson correlation $|r| > 0.99$ across the calibration set, they compute nearly the same feature — one is redundant.

**Procedure per step:**
1. Compute correlation matrix of post-activation outputs per hidden layer on calibration data
2. Find all pairs with $|r| > 0.99$, sort by descending correlation
3. For the highest-correlation pair: keep the neuron with the larger outgoing weight norm (survivor), remove the other (redundant)
4. **Weight transfer before zeroing:** add a scaled contribution from the redundant neuron's downstream column into the survivor's:
$$\gamma = \frac{\langle h_{\text{surv}}, h_{\text{redund}} \rangle}{\langle h_{\text{surv}}, h_{\text{surv}} \rangle}, \qquad W_{l+1}[:, \text{surv}] \mathrel{+}= \gamma \cdot W_{l+1}[:, \text{redund}]$$
5. Correlations are recomputed at the next sparsity target (the merge changes activations, invalidating stale estimates)

**Fallback:** if no pairs above 0.99 remain, falls back to magnitude pruning.

**Intuition:** high correlation = same feature computed twice. The weight transfer ensures the survivor absorbs the redundant neuron's downstream contribution before it is removed, minimising the loss change.

---

## Method 8 — SNIP

**Key distinction:** prunes **before any training**, at random initialization.

**Criterion:**
$$s_i = |w_i \cdot g_i| \quad \text{evaluated at the untrained init weights}$$

**Procedure per sparsity target:**
1. Reset model to saved random init
2. One forward + backward pass on calibration data to get $g_i$
3. Compute $s_i = |w_i g_i|$ for every weight
4. Keep top-$(1-t)$ fraction by score — **one-shot**, no iterative removal
5. Zero the pruned weights, then train the masked network from the same init

**Same formula as Taylor FO, completely different meaning:** Taylor FO uses gradients from a *trained* network where the loss surface reflects learned representations. SNIP asks: *which connections are sensitive to the loss before training?* — a structural question about the network's inductive bias at init.

**The bet:** the connections that matter for the loss landscape at init are the ones that will matter after training. This is the only prune-before-train method in the suite.

---

## Method 9 — IMP + Rewind

**Idea:** the Lottery Ticket Hypothesis (Frankle & Carlin, 2019) — within a dense network there exists a sparse subnetwork ("winning ticket") that, when trained from the same random init, matches the dense network's performance.

**Procedure per sparsity target:**
1. Prune smallest-$|w|$ weights to reach the target (same selection rule as Magnitude)
2. **Rewind:** copy the saved random init weights back into surviving positions; pruned positions stay zero
3. Retrain from scratch (fine-tune loop is effectively full training since weights just rewound)

**Why rewind?** Plain iterative magnitude pruning keeps the trained weights in surviving positions. IMP+Rewind tests whether the *mask* — the subgraph structure — is the real discovery, not the weight values. If the rewound sparse network trains just as well as plain magnitude pruning, the Lottery Ticket Hypothesis holds for our networks.

**In code:** the selection function is identical to Magnitude (`select_weight` = argmin $|w|$); the rewind is applied in `pruner.py` before the fine-tune loop at each sparsity target.

---

## Summary Table

| Method | Criterion | Granularity | Hessian | When scored |
|---|---|---|---|---|
| Magnitude | $\|w_i\|$ | weight | none | per target |
| Taylor FO | $\|g_i w_i\|$ at trained weights | weight | none | per target |
| OBD | $\frac{1}{2}H_{ii}w_i^2$ | weight | diagonal | per target |
| OBS | $\frac{w_q^2}{2[H^{-1}]_{qq}}$ + weight correction | weight | full inverse | per removal |
| Structured | $\|W_l[j,:]\|_2$ | neuron | none | per step |
| Activation | MAA of $h_j$ over calibration set | neuron | none | per step |
| Redundancy | Pearson $\|r\| > 0.99$ + weight transfer | neuron pair | none | per target |
| SNIP | $\|g_i w_i\|$ at init — single shot | weight | none | once, before train |
| IMP+Rewind | $\|w_i\|$ + rewind to init | weight | none | per target |

---

## The Shared Evaluation Framework

```
for target in [10%, 20%, 30%, ..., 95%]:
    while current_sparsity < target:
        score weights/neurons by saliency criterion
        remove the least salient unit
    fine-tune surviving weights (Adam, patience=10 on val)
    evaluate test MSE once
    if test_mse > 3 × baseline_mse and sparsity ≥ 50%: stop

→ MSE vs. sparsity curve per method
→ best model = highest sparsity within acceptable error budget
```

**SNIP exception:** mask is set in one shot at the target level, then trains from init — no iterative inner loop.

**OBS exception:** $H^{-1}$ is rank-1 updated and weight correction applied after each individual removal — more continuous than the rest.
