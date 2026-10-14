"""Optimal Brain Surgeon — Hassibi & Stork (1993).

To delete weight w_q while minimising the resulting increase in loss, OBS
solves a constrained quadratic optimisation problem.  At a trained minimum
(where the gradient is zero), the second-order approximation gives:

  q*   = argmin_q  w_q^2 / (2 * [H^{-1}]_{qq})      saliency (Eq. 7)
  δw   = -(w_q* / [H^{-1}]_{q*q*}) * H^{-1} e_{q*}  weight correction (Eq. 8)
  H^{-1}_new = H^{-1} - (H^{-1} e_q e_q^T H^{-1}) / [H^{-1}]_{qq}  (rank-1 update)

The rank-1 update means the full Hessian is computed ONCE per sparsity
checkpoint; all subsequent removals within that checkpoint re-use the
updated inverse without recomputing.

Hessian computation
-------------------
We use torch.autograd.functional.hessian (exact, second-order autograd) over
the full training set via a functional closure — same pattern as OBD but
computing the full matrix, not just the diagonal.

Practical size limit
--------------------
Materialising an (n x n) Hessian requires O(n^2) memory.  For n > MAX_EXACT_N
(default 15 000), we fall back to the empirical Fisher matrix:
  F = (1/N) sum_i  g_i g_i^T       (outer products of per-sample gradients)
Under the Gauss-Newton approximation, F = H at a loss minimum, so this is
theoretically consistent with OBS while keeping memory feasible.  A warning
is printed whenever the fallback is used.

Tikhonov regularisation
-----------------------
H_reg = H + lambda * I,   lambda = 1e-4
If H_reg is still ill-conditioned (cond > 1e10), lambda is increased by 10x
and a warning is emitted.  This is documented in the report.

Public API (called from pruner.py)
-----------------------------------
  compute_hessian_inv(model, x_all, y_all, device, regularisation=1e-4)
      → (n x n) tensor: (H + lambda*I)^{-1}
  select_and_correct(w, mask, h_inv)
      → (q, new_w): index of pruned weight + weight-corrected vector
  update_h_inv(h_inv, q)
      → updated (n x n) inverse Hessian after removing weight q

Adapted from pruning_study/pruning/obs.py: _argmin_saliency, _apply_obs_step,
_update_h_inv reused near-verbatim; BCELoss → MSELoss; Fisher closure adapted
for our MLP; exact Hessian now via torch.autograd.functional.hessian.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F

from src.models.mlp import MLP

MAX_EXACT_N = 3_000    # above this, return None → pruner falls back to OBD-diagonal.
# Was 15_000; the inversion cost is O(n³) and the per-target recompute makes
# n ~ 5000 take several hours of CPU. The 2-D A baseline (n=2241) still uses
# OBS exact; 3-D A (n=4945) falls back. Documented in scaling-law narrative.

_ACT_FNS: dict = {
    "relu": F.relu,
    "tanh": torch.tanh,
    "gelu": F.gelu,
    "sigmoid": torch.sigmoid,
}


# ── shared functional forward ─────────────────────────────────────────────────

def _functional_mse(
    flat: torch.Tensor,
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    shapes: list,
    sizes: list,
    act_fn,
    n_linear: int,
) -> torch.Tensor:
    offset = 0
    params = []
    for shape, size in zip(shapes, sizes):
        params.append(flat[offset : offset + size].view(shape))
        offset += size
    h = x_all
    for i in range(n_linear - 1):
        h = act_fn(F.linear(h, params[2 * i], params[2 * i + 1]))
    h = F.linear(h, params[2 * (n_linear - 1)], params[2 * (n_linear - 1) + 1])
    return F.mse_loss(h, y_all)


# ── Hessian computation ───────────────────────────────────────────────────────

def _exact_hessian(
    flat0: torch.Tensor,
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    shapes: list,
    sizes: list,
    act_fn,
    n_linear: int,
) -> torch.Tensor:
    """Full exact Hessian via torch.autograd.functional.hessian (vectorized)."""
    fn = lambda w: _functional_mse(w, x_all, y_all, shapes, sizes, act_fn, n_linear)
    H = torch.autograd.functional.hessian(fn, flat0, vectorize=True)
    return H.detach()


def _fisher_hessian(
    flat0: torch.Tensor,
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    shapes: list,
    sizes: list,
    act_fn,
    n_linear: int,
    device: torch.device,
) -> torch.Tensor:
    """Empirical Fisher matrix F = (1/N) sum_i g_i g_i^T as H proxy."""
    n = flat0.numel()
    F_mat = torch.zeros(n, n, device=device, dtype=torch.float32)

    for xi, yi in zip(x_all, y_all):
        xi = xi.unsqueeze(0)
        yi = yi.unsqueeze(0)
        w = flat0.detach().requires_grad_(True)
        loss_i = _functional_mse(w, xi, yi, shapes, sizes, act_fn, n_linear)
        (g,) = torch.autograd.grad(loss_i, w)
        F_mat += torch.outer(g.detach(), g.detach())

    F_mat /= len(x_all)
    return F_mat


# ── public API ────────────────────────────────────────────────────────────────

def compute_hessian_inv(
    model: MLP,
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    device: torch.device,
    regularisation: float = 1e-4,
) -> torch.Tensor | None:
    """Compute (H + λI)^{-1} for OBS.

    Uses the exact Hessian when n <= MAX_EXACT_N.  When n > MAX_EXACT_N the
    Fisher matrix approach requires O(n²) memory (e.g. 22 GB for n=74k,
    279 GB for n=264k) which exceeds device limits.  In that case this
    function returns **None** and the caller (pruner.py) falls back to the
    Fisher-diagonal OBD saliency without weight correction.  The limitation
    is documented in the report.

    Args:
        model:          Trained MLP.
        x_all:          Full training inputs on `device`.
        y_all:          Full training targets on `device`.
        device:         Compute device.
        regularisation: Tikhonov λ (default 1e-4, increased 10× if still
                        ill-conditioned).

    Returns:
        (n x n) inverse regularised Hessian tensor, or None if infeasible.
    """
    model.eval()
    act_fn   = _ACT_FNS[model.config["activation"]]
    n_linear = len(model.layers)
    p_list   = list(model.parameters())
    shapes   = [p.shape for p in p_list]
    sizes    = [p.numel() for p in p_list]
    n        = sum(sizes)

    flat0 = model.get_weight_vector().to(device)

    if n <= MAX_EXACT_N:
        H = _exact_hessian(flat0, x_all, y_all, shapes, sizes, act_fn, n_linear)
    else:
        warnings.warn(
            f"[OBS] n={n} > {MAX_EXACT_N}: full H⁻¹ requires O(n²) memory "
            f"({n**2 * 4 / 1e9:.1f} GB); returning None. "
            "pruner.py will fall back to Fisher-diagonal OBD saliency (no weight correction). "
            "Documented in report.",
            stacklevel=2,
        )
        return None

    # Tikhonov regularisation
    lam = regularisation
    H_reg = H + lam * torch.eye(n, device=device, dtype=H.dtype)

    cond = torch.linalg.cond(H_reg).item()
    if cond > 1e10:
        lam *= 10.0
        warnings.warn(
            f"[OBS] H_reg ill-conditioned (cond={cond:.2e}); "
            f"increasing λ to {lam:.2e}.",
            stacklevel=2,
        )
        H_reg = H + lam * torch.eye(n, device=device, dtype=H.dtype)
        cond_post = torch.linalg.cond(H_reg).item()
        if cond_post > 1e10:
            # Audit §2.3: after the one-shot bump, still ill-conditioned →
            # bail out and let the caller fall back to OBD-diagonal saliency
            # rather than silently degrading.
            warnings.warn(
                f"[OBS] λ bump did not restore conditioning (cond={cond_post:.2e}); "
                "returning None — caller will fall back to OBD-diagonal saliency.",
                stacklevel=2,
            )
            return None

    return torch.linalg.inv(H_reg)


def select_and_correct(
    w: torch.Tensor,
    mask: torch.Tensor,
    h_inv: torch.Tensor,
) -> tuple[int, torch.Tensor]:
    """Select the weight to prune and apply the OBS weight correction.

    Implements Hassibi & Stork (1993) Eq. 7 (saliency) and Eq. 8 (δw).

    Saliency:          L_q = w_q^2 / (2 * [H^{-1}]_{qq})
    Weight correction: δw  = -(w_q / [H^{-1}]_{qq}) * H^{-1} e_q

    Args:
        w:     Current flat weight vector (length n).
        mask:  Boolean tensor, True = weight is still alive.
        h_inv: Current (n x n) inverse Hessian.

    Returns:
        (q, new_w) where q is the pruned index and new_w has w[q] == 0
        and the remaining weights corrected to compensate.
    """
    diag = torch.diagonal(h_inv)
    saliency = w.detach() ** 2 / (2.0 * diag.clamp(min=1e-12))
    saliency = torch.where(
        mask & (diag > 0),
        saliency,
        torch.full_like(saliency, float("inf")),
    )
    q = int(saliency.argmin().item())

    # Weight correction (Eq. 8)
    h_col = h_inv[:, q]
    epsilon=10e-6
    denom = float(h_inv[q, q].item())+epsilon
    delta = -(float(w[q].item()) / denom) * h_col
    new_w = w + delta
    new_w[q] = 0.0  # force exact zero
    return q, new_w


def update_h_inv(h_inv: torch.Tensor, q: int) -> torch.Tensor:
    """Rank-1 update of H^{-1} after removing weight q.

    H^{-1}_new = H^{-1} - (H^{-1} e_q e_q^T H^{-1}) / [H^{-1}]_{qq}

    The q-th row and column are zeroed afterward so the removed weight cannot
    influence future saliency comparisons (the alive mask is the source of
    truth, but zeroing the row/col is an extra safety guard).

    Adapted verbatim from pruning_study/pruning/obs.py::_update_h_inv().
    """
    h_col = h_inv[:, q : q + 1]    # (n, 1)
    h_row = h_inv[q : q + 1, :]    # (1, n)
    denom = float(h_inv[q, q].item())
    h_new = h_inv - (h_col @ h_row) / denom
    h_new[q, :] = 0.0
    h_new[:, q] = 0.0
    return h_new
