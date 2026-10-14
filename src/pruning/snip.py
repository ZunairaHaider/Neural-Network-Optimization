"""SNIP — Single-shot Network Pruning based on connection sensitivity.

Lee, Ajanthan, Torr (ICLR 2019).

SNIP prunes the network **at initialisation** — before any training — based on
how sensitive the loss is to removing each weight. The sensitivity is computed
by introducing per-weight gating variables c_i (all initialised to 1) and
measuring |∂L/∂c_i|. Since at c=1 the network output equals f(w⊙c=w) we have
∂L/∂c_i = w_i · ∂L/∂w_i, so the criterion reduces to:

    s_i = | w_i * g_i |        evaluated at the **untrained** init

Only the *highest-sensitivity* connections are kept; the rest are pruned in a
single shot before training begins. The remaining network is then trained
from the same init. This is the only "prune-at-init" datapoint in the suite.

Pruner integration
------------------
Per sparsity target `t`:
  1. Reset the model to a saved fresh init (`init_state`).
  2. Compute SNIP scores on a calibration batch (the full train pool here).
  3. Keep the top-k connections by score where k = (1 - t) * n.
  4. Zero pruned weights, then train (fine-tune) the masked network from init.
  5. Evaluate test once.

References
----------
Lee, Ajanthan, Torr (2019), *SNIP: Single-shot Network Pruning Based on
Connection Sensitivity*, ICLR.
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


def compute_scores(
    model: MLP,
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """SNIP connection-sensitivity scores on the model's *current* weights.

    Intended to be called after the model has been reset to a fresh
    initialisation (see ``snip_mask_for_target`` for the pruner integration).
    """
    model.eval()
    act_fn   = _ACT_FNS[model.config["activation"]]
    n_linear = len(model.layers)
    p_list   = list(model.parameters())
    shapes   = [p.shape for p in p_list]
    sizes    = [p.numel() for p in p_list]

    flat = model.get_weight_vector().to(device).requires_grad_(True)
    loss = _functional_mse(flat, x_all, y_all, shapes, sizes, act_fn, n_linear)
    (g,) = torch.autograd.grad(loss, flat)
    return (g * flat).detach().abs()


def snip_mask_for_target(
    scores: torch.Tensor,
    target_sparsity: float,
) -> torch.Tensor:
    """Boolean mask keeping the top-(1 - target_sparsity) fraction by score.

    Returns a tensor where True = alive, False = pruned. Ties are broken in
    favour of keeping (we use floor(k_keep), so we may slightly under-prune
    rather than over-prune — matters only at corner cases).
    """
    n = scores.numel()
    n_keep = max(1, int(round(n * (1.0 - target_sparsity))))
    # topk indices of the n_keep highest scores
    top = torch.topk(scores, n_keep, largest=True).indices
    mask = torch.zeros(n, dtype=torch.bool, device=scores.device)
    mask[top] = True
    return mask
