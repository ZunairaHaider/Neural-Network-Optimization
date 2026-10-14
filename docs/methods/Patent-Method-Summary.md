# Dynamic Neural Architecture Search — Patent Method Summary
**Reference:** US12124779B2 (ADAGOS)

---

## The Problem

We want to fit a dataset with an MLP but don't know the right architecture. Instead of guessing upfront, we grow and shrink the network one structural change at a time, guided by a cheap error estimate at each step.

---

## Step 1 — Initialise

Start from a minimal network Γ*: one hidden layer, one neuron. Train to convergence with Gauss-Newton.

At convergence the network is **consistent**: every hidden activation satisfies its forward-pass equation exactly.

---

## Step 2 — Compute the Lagrangian

The Lagrangian augments MSE with equality constraints enforcing the forward pass:

$$\mathcal{L}(\Gamma, W, X, \Lambda) = \underbrace{\|a^{(L)} - Y\|^2}_{\text{MSE}} + \sum_{i=1}^{L} \Lambda^{(i)\top}\bigl(a^{(i)} - \phi_i(z^{(i)})\bigr)$$

**Key property:** at the current consistent network every constraint term is zero, so L = MSE.

---

## Step 3 — Compute and Freeze the Multipliers

The multipliers Λ(i) are derived from the KKT stationarity condition dL/da(i) = 0.

For a hidden layer i, two terms in the Lagrangian depend on a(i):
- Constraint i directly: contributes +Λ(i)
- Constraint i+1 through z(i+1) = W_{i+1} a(i): contributes -W_{i+1}^T (φ'_{i+1}(z(i+1)) ⊙ Λ(i+1))

Setting to zero and absorbing the activation derivative gives the recurrence:

$$\Lambda^{(L)} = 2(a^{(L)} - Y)$$

$$\Lambda^{(i)} = \bigl(W_{i+1}^\top \Lambda^{(i+1)}\bigr) \odot \phi_i'(z^{(i)})$$

This is exactly backprop — the multipliers are the gradients of the loss w.r.t. each layer's pre-activation. **Freeze** Λ* — reused for all candidates this cycle.

**Intuition:** Λ(i)_j tells you how sensitive the loss is to neuron j at layer i. Large |Λ(i)_j| = that neuron matters a lot. The sign tells you which direction of change reduces the loss.

---

## Step 4 — Overfitting Check

    train_MSE < (1 - 0.2) × val_MSE?

- **Yes** → overfitting → allow only subtractive candidates
- **No** → allow additive and subtractive

This couples topology complexity directly to generalisation without explicit regularisation.

---

## Step 5 — Generate Candidates

One structural change at a time. Never combinations.

**Additive** (up to 32 sampled):
- `ADD_NEURON` — append one neuron to a hidden layer, zero-init incoming and outgoing weights
- `ADD_LAYER` — insert a new layer at position p, initialised via SVD so W2 W1 ≈ W_orig (identity preservation). Non-linear layers use scaled SVD with ε=1e-4 so tanh(z) ≈ z at init
- `ADD_SKIP` — insert a zero-weight skip connection s → t

**Subtractive** (all enumerated):
- `REMOVE_NEURON` — ranked by ||Λ(i)||, drop least sensitive neuron
- `REMOVE_LAYER` — try each, score by ΔL
- `REMOVE_SKIP` — drop skip with lowest mean |W_skip|

---

## Step 6 — Score Each Candidate Cheaply

For each candidate Γ'':

1. Warm-start weights from W*, zero-init new weights
2. Run k=15 gradient descent steps on MSE — just enough to move new weights off zero
3. Compute:

$$\Delta\mathcal{L} = \mathcal{L}(\Gamma'', W'', X, \Lambda^*) - \mathcal{L}(\Gamma^*, W^*, X, \Lambda^*)$$

The constraint term Λ(i)*^T (a(i)* - φ_i(W_i'' a(i-1)*)) measures how much the structural change shifts activations away from the old solution, weighted by loss sensitivity.

**Intuition:** the constraint term is a first-order Taylor expansion of the loss change. If the structural change shifts neuron j's activation in the direction the multiplier says is downhill (opposite sign to Λ_j), the term is negative → ΔL goes down → promising candidate.

**Why not just MSE?**
- **Zero-init problem:** new neurons have zero output and zero gradient after k steps — all ADD_NEURON candidates look identical under MSE
- **k steps too few:** MSE after k steps reflects initialisation noise, not structural quality

The Lagrangian gives signal immediately via sensitivity-weighted constraint terms — before weights have converged.

**Layer type registry:** a lookup table so forward pass, backprop, and Lagrangian all use the same activation and derivative with no hardcoding:

| Key | φ(z) | φ'(z) |
|---|---|---|
| tanh | tanh z | 1 − tanh²z |
| relu | max(0,z) | 1_{z>0} |
| linear | z | 1 |

---

## Step 7 — Pick the Winner and Fully Retrain

Sort all candidates by ΔL. Take the most negative.

If best ΔL ≥ 0 → no candidate improves → patience++. Three cycles in a row → **stop**.

If best ΔL < 0 → fully retrain the winner with **Gauss-Newton**.

### Why Gauss-Newton?

Minimise ||r||² where r = f(W) - Y. Linearise the residual (first-order Taylor):

    r(W + ΔW) ≈ r(W) + J ΔW

Minimise over ΔW, set derivative to zero:

$$\Delta W = -(\mathbf{J}^\top \mathbf{J})^{-1}\mathbf{J}^\top r$$

J^T J is a quasi-Hessian — second-order curvature from first-order Jacobians, no learning rate needed. Quadratic convergence near the optimum. With Levenberg-Marquardt damping for stability:

$$\Delta W = -(\mathbf{J}^\top \mathbf{J} + \lambda I)^{-1}\mathbf{J}^\top r$$

Run with multiple random restarts, keep the best Lagrangian across all restarts.

---

## Step 8 — Accept or Revert

After full retraining, the winner is consistent again → constraint terms vanish → L = new MSE.

- New MSE < old MSE → **accept**: Γ* ← winner, recompute Λ*, next cycle
- New MSE ≥ old MSE → **revert**: discard winner, increment patience

**Monotone descent guarantee:** because you only accept on strict improvement, the sequence of best-network losses never increases.

---

## The Full Loop

```
Initialise Γ* (1 neuron), train to convergence
Compute Λ* ← freeze

For each cycle:
    Check overfitting → restrict to subtractive if train << val
    Generate candidates (additive + subtractive)
    For each candidate:
        Warm-start + zero-init new weights
        k=15 GD steps on MSE
        Compute ΔL using frozen Λ*
    Pick candidate with lowest ΔL
    If best ΔL ≥ 0 → patience++; if patience = 3 → stop
    Fully retrain winner (Gauss-Newton, multi-restart)
    If new MSE < old MSE → accept, recompute Λ*
    Else → revert
```

---

## Key Comparisons

| Aspect | Patent Method | Classical NAS |
|---|---|---|
| Candidate evaluation | Cheap ΔL (k steps + Lagrangian) | Full retrain per candidate |
| Cost | O(M·k) passes | O(M·k_full) passes |
| Topology + weight search | Decoupled | Coupled (weight sharing) |
| Overfitting control | Built-in via subtractive phase | Requires explicit regularisation |
| Convergence guarantee | Monotone descent | Not guaranteed |
