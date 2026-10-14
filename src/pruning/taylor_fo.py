"""Taylor first-order saliency — Molchanov, Tyree, Karras, Aila, Kautz (2017);
Molchanov, Mallya, Tyree, Frosio, Kautz (2019).

Saliency of weight w_i:
    s_i = | w_i * g_i |          where g_i = ∂L/∂w_i

This is the first-order Taylor expansion of the loss around the trained weights:
the change in loss from zeroing w_i is approximately |w_i * g_i| (the higher-order
terms drop out and the absolute value avoids sign cancellation across batch
elements). Note that at an exact loss minimum g ≈ 0 → saliency degenerates;
in practice the trained network is at a *practical* minimum and gradients on
the training set carry meaningful magnitude differences across weights.

Pairs naturally with OBD's diagonal Hessian (second-order Taylor) as a cheap
first-order alternative — no Hessian computation required.

Public API (called from pruner.py)
-----------------------------------
  compute_saliency(model, x_all, y_all, device) → 1-D tensor of length n
  select_weight(w, mask, saliency) → int

References
----------
Molchanov et al. (2017), *Pruning Convolutional Neural Networks for Resource
Efficient Inference*, ICLR.
Molchanov et al. (2019), *Importance Estimation for Neural Network Pruning*,
CVPR.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.mlp import MLP

_ACT_FNS: dict = {
    "relu": F.relu,
    "tanh": torch.tanh,
    "gelu": F.gelu,
    "sigmoid": torch.sigmoid,
}


def _functional_mse(flat, x_all, y_all, shapes, sizes, act_fn, n_linear):
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


def compute_saliency(
    model: MLP,
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Per-weight Taylor-FO saliency on the full training pool.

    Returns a length-n tensor of non-negative saliencies; recompute after each
    fine-tune since gradients shift with the weights.
    """
    model.eval()
    act_fn   = _ACT_FNS[model.config["activation"]]
    n_linear = len(model.layers)
    p_list   = list(model.parameters())
    shapes   = [p.shape for p in p_list]
    sizes   = [p.numel() for p in p_list]

    flat = model.get_weight_vector().to(device).requires_grad_(True)
    loss = _functional_mse(flat, x_all, y_all, shapes, sizes, act_fn, n_linear)
    (g,) = torch.autograd.grad(loss, flat)
    return (g * flat).detach().abs()


def select_weight(
    w: torch.Tensor,
    mask: torch.Tensor,
    saliency: torch.Tensor,
) -> int:
    """Argmin saliency among alive weights."""
    s = torch.where(mask, saliency, torch.full_like(saliency, float("inf")))
    return int(s.argmin().item())
