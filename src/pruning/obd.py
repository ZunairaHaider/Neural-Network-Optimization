"""Optimal Brain Damage — LeCun, Denker, Solla (1990).

Saliency of weight w_i:
    s_i = 0.5 * H_ii * w_i^2

where H_ii = d^2 L / dw_i^2 is the DIAGONAL of the Hessian.

Hessian diagonal computation
----------------------------
Exact method (n <= MAX_EXACT_N = 5 000):
  Double backpropagation via torch.autograd.grad with create_graph=True.
  Procedure:
    1. Build a flat weight vector that tracks gradients.
    2. Forward pass → scalar MSE loss.
    3. autograd.grad(loss, flat, create_graph=True) → first-order gradient g.
    4. For each element g_i: autograd.grad(g_i, flat)[i] = H_ii.
  Time: O(n) backward passes.  Memory: O(n).

Fisher fallback (n > MAX_EXACT_N):
  Empirical Fisher diagonal F_ii = (1/N) * sum_k (∂L_k/∂w_i)^2,
  where L_k is the per-sample MSE loss.  Under the Gauss-Newton
  approximation, F = H at a trained loss minimum, so this is theoretically
  consistent with OBD for well-trained networks.  Requires only N forward+
  backward passes (one per training sample) rather than n, giving O(n*N)
  instead of O(n^2*N).  A warning is printed whenever the fallback is used.

Public API (called from pruner.py)
-----------------------------------
  compute_hessian_diag(model, x_all, y_all, device)
      → 1-D tensor of shape (n,), clamped to >= 1e-12.
  select_weight(w, mask, h_diag)
      → integer index q of the unmasked weight with the smallest saliency.

Adapted from pruning_study/pruning/obd.py: replaced BCELoss with MSELoss,
added mask support; exact double-backprop path added for small networks,
Fisher diagonal retained as fallback for large networks.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F

from src.models.mlp import MLP

MAX_EXACT_N = 5_000   # above this, fall back to Fisher diagonal + warning
# The Fisher-diagonal fallback loops once per training ROW (not once per
# parameter) -- O(N) backward passes in plain Python. On PMSM's real data
# (N up to ~970k rows) that is many hours, not minutes. Cap the number of
# rows used for this curvature estimate; a few thousand rows is a standard,
# widely-used calibration-set size for this kind of saliency estimation
# (same idea SNIP/Taylor-FO already use "a calibration batch", per their
# own docstrings) and gives a stable estimate without the full-dataset cost.
MAX_FISHER_CALIB_ROWS = 4_000
_FISHER_CALIB_SEED = 777

# Functional equivalents of supported activations (needed for the closure that
# passes parameters as a flat tensor rather than using model.parameters()).
_ACT_FNS: dict = {
    "relu": F.relu,
    "tanh": torch.tanh,
    "gelu": F.gelu,
    "sigmoid": torch.sigmoid,
}


# ── internal helpers ──────────────────────────────────────────────────────────

def _functional_mse(
    flat: torch.Tensor,
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    shapes: list,
    sizes: list,
    act_fn,
    n_linear: int,
) -> torch.Tensor:
    """MSE loss as a differentiable function of a flat weight vector.

    Runs a manual forward pass using parameters unpacked from `flat` so that
    autograd can differentiate through both the loss and the weights.
    """
    offset = 0
    params = []
    for shape, size in zip(shapes, sizes):
        params.append(flat[offset : offset + size].view(shape))
        offset += size

    h = x_all
    for i in range(n_linear - 1):                         # hidden layers
        h = act_fn(F.linear(h, params[2 * i], params[2 * i + 1]))
    h = F.linear(h, params[2 * (n_linear - 1)], params[2 * (n_linear - 1) + 1])  # output
    return F.mse_loss(h, y_all)


def _exact_hessian_diag(
    flat: torch.Tensor,
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    shapes: list,
    sizes: list,
    act_fn,
    n_linear: int,
    n: int,
    device: torch.device,
) -> torch.Tensor:
    """Exact Hessian diagonal via double backprop (create_graph=True).

    O(n) backward passes — feasible only for n <= MAX_EXACT_N.
    """
    loss = _functional_mse(flat, x_all, y_all, shapes, sizes, act_fn, n_linear)
    (g,) = torch.autograd.grad(loss, flat, create_graph=True)

    hdiag = torch.zeros(n, device=device)
    for i in range(n):
        (h_row,) = torch.autograd.grad(g[i], flat, retain_graph=(i < n - 1))
        hdiag[i] = h_row[i].detach()
    return hdiag


def _fisher_hessian_diag(
    flat0: torch.Tensor,
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    shapes: list,
    sizes: list,
    act_fn,
    n_linear: int,
    n: int,
    device: torch.device,
) -> torch.Tensor:
    """Fisher / Gauss-Newton diagonal: F_ii = (1/N) * sum_k (∂L_k/∂w_i)^2.

    Requires N first-order backward passes (one per sample) → O(n*N) total.
    At a trained loss minimum this equals the true Hessian diagonal.

    Runs on CPU regardless of `device` to avoid MPS buffer-size errors that
    occur for large n when autograd allocates internal intermediates.
    """
    cpu = torch.device("cpu")
    flat_cpu = flat0.detach().cpu()
    x_cpu = x_all.cpu()
    y_cpu = y_all.cpu()

    F_diag = torch.zeros(n, device=cpu)
    N = len(x_cpu)
    for xi, yi in zip(x_cpu, y_cpu):
        xi = xi.unsqueeze(0)
        yi = yi.unsqueeze(0)
        w = flat_cpu.detach().requires_grad_(True)
        loss_i = _functional_mse(w, xi, yi, shapes, sizes, act_fn, n_linear)
        (g,) = torch.autograd.grad(loss_i, w)
        F_diag += g.detach() ** 2
    return (F_diag / N).to(device)


# ── public API ────────────────────────────────────────────────────────────────

def compute_hessian_diag(
    model: MLP,
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Diagonal of the Hessian for OBD saliency computation.

    Uses exact double-backprop when n <= MAX_EXACT_N; otherwise falls back to
    the empirical Fisher diagonal with a printed warning (same pattern as OBS).

    Args:
        model:  Trained MLP (eval mode set internally).
        x_all:  All training inputs, shape (N, 1), on `device`.
        y_all:  All training targets, shape (N, 1), on `device`.
        device: Compute device.

    Returns:
        1-D tensor of length n (= model.get_weight_count()) with H_ii values,
        clamped to >= 1e-12 (OBD assumes positive curvature at a trained minimum).
    """
    model.eval()
    act_fn   = _ACT_FNS[model.config["activation"]]
    n_linear = len(model.layers)
    p_list   = list(model.parameters())
    shapes   = [p.shape for p in p_list]
    sizes    = [p.numel() for p in p_list]
    n        = sum(sizes)

    flat = model.get_weight_vector().to(device)

    if n <= MAX_EXACT_N:
        flat = flat.requires_grad_(True)
        hdiag = _exact_hessian_diag(
            flat, x_all, y_all, shapes, sizes, act_fn, n_linear, n, device
        )
    else:
        warnings.warn(
            f"[OBD] n={n} > {MAX_EXACT_N}: using Fisher approximation for Hessian diagonal. "
            "Consider retraining with a smaller architecture for exact OBD.",
            stacklevel=2,
        )
        if x_all.shape[0] > MAX_FISHER_CALIB_ROWS:
            gen = torch.Generator(device="cpu").manual_seed(_FISHER_CALIB_SEED)
            idx = torch.randperm(x_all.shape[0], generator=gen)[:MAX_FISHER_CALIB_ROWS]
            warnings.warn(
                f"[OBD] Fisher fallback: subsampling {MAX_FISHER_CALIB_ROWS} of "
                f"{x_all.shape[0]} training rows as a calibration set (the per-row "
                "loop is O(N) backward passes and is infeasible on the full set).",
                stacklevel=2,
            )
            x_calib = x_all[idx]
            y_calib = y_all[idx]
        else:
            x_calib = x_all
            y_calib = y_all
        hdiag = _fisher_hessian_diag(
            flat, x_calib, y_calib, shapes, sizes, act_fn, n_linear, n, device
        )

    return hdiag.clamp(min=1e-12)


def select_weight(
    w: torch.Tensor,
    mask: torch.Tensor,
    h_diag: torch.Tensor,
) -> int:
    """Return the index of the unmasked weight with the smallest OBD saliency.

    Saliency: s_i = 0.5 * H_ii * w_i^2  (LeCun et al. 1990, Eq. 5)
    Masked-out (already pruned) weights are excluded via +inf.

    Args:
        w:      Current flat weight vector (on any device).
        mask:   Boolean tensor, True = weight is alive.
        h_diag: Hessian diagonal from compute_hessian_diag().

    Returns:
        Integer index q of the weight to prune next.
    """
    saliency = 0.5 * h_diag * w.detach() ** 2
    saliency = torch.where(mask, saliency, torch.full_like(saliency, float("inf")))
    return int(saliency.argmin().item())
