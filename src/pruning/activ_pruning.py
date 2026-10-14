"""Activation-based structured pruning — removes neurons by activation statistics.

Unlike weight-magnitude methods (mag.py, struct_pruning.py) that score neurons
purely from their parameter values, activation-based pruning looks at how much
each neuron *actually fires* on the training data.  Neurons that are dormant
or near-zero across all inputs contribute almost nothing to the network's
function and are safe to remove.

Motivation
----------
A neuron can have large weights yet produce near-zero post-activation outputs
if its inputs consistently cancel in the pre-activation sum, or if it is
chronically below the activation threshold (e.g. dead ReLU neurons whose
pre-activation is always negative).  Weight-norm criteria miss these neurons.
Activation statistics catch them directly.

Saliency criterion
------------------
For every hidden neuron j in layer l, collect the post-activation output
h_j(x) across all N training samples.  Score the neuron by the L1-mean of
the absolute activation:

    saliency_j = (1/N) * sum_i |h_j(x_i)|     (Mean Absolute Activation, MAA)

Neurons with small MAA have near-zero influence on downstream layers regardless
of their weight magnitudes and are removed first.

Why MAA instead of variance?
    Variance-based criteria (e.g. APoZ-style) discard neurons with low output
    variance, which can accidentally remove bias neurons with large constant
    output.  MAA is robust: a neuron must have both large AND consistent output
    to survive.  For ReLU networks MAA == average of positive pre-activations,
    so it directly measures how often and how strongly the neuron fires.

Adaptation to the flat-vector API
----------------------------------
Same as struct_pruning.py — the function returns the flat-vector indices of
ALL weights belonging to the chosen neuron (outgoing row, bias, incoming column
in the next layer).

Public API (mirrors struct_pruning.py / obd.py for pruner.py)
-------------------------------------------------------------
  compute_saliencies(model, x, y, device) → 1-D tensor, one score per weight.
      Weight-level saliency: each weight's score = MAA of its neuron.
      Output-layer weights receive +inf (never pruned).

  select_neuron_weights(model, x_all, device, mask) → list[int]
      Returns flat-vector indices of ALL weights belonging to the hidden neuron
      with the smallest MAA.  Call once per pruning step.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from src.models.mlp import MLP


# ── internal helpers ──────────────────────────────────────────────────────────

def _layer_param_ranges(model: MLP) -> list[tuple[int, int, int, int]]:
    """Return (w_start, w_end, b_start, b_end) for each layer in the flat vector."""
    ranges = []
    offset = 0
    for layer in model.layers:
        w_size = layer.weight.numel()
        b_size = layer.bias.numel()
        ranges.append((offset, offset + w_size,
                       offset + w_size, offset + w_size + b_size))
        offset += w_size + b_size
    return ranges


_ACT_FNS: dict[str, Callable] = {
    "relu":    F.relu,
    "tanh":    torch.tanh,
    "gelu":    F.gelu,
    "sigmoid": torch.sigmoid,
}


@torch.no_grad()
def _collect_activations(
    model: MLP,
    x_all: torch.Tensor,
    device: torch.device,
) -> list[torch.Tensor]:
    """Run a forward pass and collect post-activation outputs for every hidden layer.

    Returns
    -------
    list of tensors, one per hidden layer, shape (N, hidden_width).
    """
    model.eval()
    act_fn = _ACT_FNS[model.config["activation"]]
    x = x_all.to(device)
    activations: list[torch.Tensor] = []

    h = x
    for layer in model.layers[:-1]:   # all hidden layers
        pre = layer(h)
        h   = act_fn(pre)
        activations.append(h.detach().cpu())   # (N, out_features)
    return activations                          # skip output layer


def _maa_scores(activations: list[torch.Tensor]) -> list[torch.Tensor]:
    """Compute Mean Absolute Activation for each neuron in each hidden layer.

    Returns a list of 1-D tensors, one per hidden layer, length = layer width.
    """
    return [act.abs().mean(dim=0) for act in activations]   # (out_features,) each


# ── public API ────────────────────────────────────────────────────────────────

def compute_saliencies(
    model: MLP,
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Return a weight-level saliency tensor for activation-based pruning.

    Every scalar weight belonging to hidden neuron j in layer l receives
    saliency = MAA_j(l).  Output-layer weights get +inf.

    Args:
        model:  Trained MLP.
        x_all:  All training inputs (N, 1).  Used to collect activations.
        y_all:  Unused — kept for uniform pruner.py API.
        device: Compute device.

    Returns:
        1-D tensor of length n = model.get_weight_count().
    """
    model.eval()
    ranges     = _layer_param_ranges(model)
    n          = model.get_weight_count()
    saliency   = torch.full((n,), float("inf"))

    layer_acts = _collect_activations(model, x_all, device)
    maa_list   = _maa_scores(layer_acts)        # one per hidden layer

    n_hidden = len(model.layers) - 1
    for l_idx in range(n_hidden):
        layer      = model.layers[l_idx]
        out_f      = layer.out_features
        in_f       = layer.in_features
        w_start    = ranges[l_idx][0]
        bias_start = ranges[l_idx][2]

        next_layer   = model.layers[l_idx + 1]
        next_w_start = ranges[l_idx + 1][0]
        next_in      = next_layer.in_features
        next_out     = next_layer.out_features

        maa = maa_list[l_idx]   # (out_f,)

        for j in range(out_f):
            score = maa[j].item()

            # Outgoing row of this layer
            row_start = w_start + j * in_f
            saliency[row_start : row_start + in_f] = score

            # Bias
            saliency[bias_start + j] = score

            # Incoming column of next layer
            for r in range(next_out):
                saliency[next_w_start + r * next_in + j] = score

    return saliency


def select_neuron_weights(
    model: MLP,
    x_all: torch.Tensor,
    device: torch.device,
    mask: torch.Tensor,
) -> list[int]:
    """Return flat-vector indices of ALL weights belonging to the least-active
    unmasked hidden neuron (lowest MAA).

    Args:
        model:  MLP with current weights.
        x_all:  All training inputs — used to compute activations.
        device: Compute device.
        mask:   Boolean flat-vector mask, True = alive.

    Returns:
        List of flat-vector indices to prune.  Empty if no neurons remain.
    """
    model.eval()
    ranges     = _layer_param_ranges(model)
    layer_acts = _collect_activations(model, x_all, device)
    maa_list   = _maa_scores(layer_acts)

    n_hidden     = len(model.layers) - 1
    best_score   = float("inf")
    best_indices: list[int] = []

    for l_idx in range(n_hidden):
        layer      = model.layers[l_idx]
        out_f      = layer.out_features
        in_f       = layer.in_features
        w_start    = ranges[l_idx][0]
        bias_start = ranges[l_idx][2]

        next_layer   = model.layers[l_idx + 1]
        next_w_start = ranges[l_idx + 1][0]
        next_in      = next_layer.in_features
        next_out     = next_layer.out_features

        maa = maa_list[l_idx]

        for j in range(out_f):
            row_start   = w_start + j * in_f
            row_indices = list(range(row_start, row_start + in_f))
            bias_idx    = bias_start + j
            col_indices = [next_w_start + r * next_in + j for r in range(next_out)]
            all_idx     = row_indices + [bias_idx] + col_indices

            # Skip already fully-pruned neurons
            outgoing = torch.tensor(row_indices + [bias_idx])
            if not mask[outgoing].any():
                continue

            score = maa[j].item()
            if score < best_score:
                best_score   = score
                best_indices = all_idx

    return best_indices
