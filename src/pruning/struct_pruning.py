"""Structured neuron pruning — removes entire neurons (rows of weight matrix).

Unlike weight-removal methods (OBD, OBS, magnitude) which zero individual
scalar weights, structured pruning removes WHOLE NEURONS from hidden layers.
This means entire rows of a weight matrix (outgoing connections) and the
corresponding column in the next layer (incoming connections) are removed,
along with the neuron's bias term.

Motivation
----------
Unstructured weight sparsity (e.g. magnitude, OBD) creates irregular zero
patterns that do not reduce actual memory or FLOPS without special sparse
hardware/software support.  Structural pruning produces a genuinely *smaller*
dense network — every surviving neuron is fully connected — so memory and
inference speed benefits are real and hardware-agnostic.

Saliency criterion
------------------
Each hidden neuron j in layer l is scored by the L2-norm of its outgoing
weight vector (the j-th row of W_l):

    saliency_j = ||W_l[j, :]||_2

Neurons with the smallest outgoing L2-norm have the least influence on the
next layer's activations and are removed first.  The corresponding incoming
weights (column j of W_{l+1}) and bias b_l[j] are also zeroed.

This is equivalent to the classic "weight-magnitude per neuron" criterion and
is the simplest, fastest structured saliency that needs no Hessian.

Adaptation to the flat-vector API
----------------------------------
pruner.py works exclusively with a 1-D flat weight vector and a boolean mask
of the same length.  Neuron pruning is expressed in this framework by masking
ALL scalar weights associated with the chosen neuron:
  - The neuron's own bias in layer l.
  - The row of W_l corresponding to the neuron (outgoing weights from l-1).
  - The column of W_{l+1} corresponding to the neuron (incoming weights to l+1).

Because the pruner loop removes one "unit" per iteration, here one "unit" is
one neuron's full set of connections.  The function returns the indices (in
the flat vector) to be masked in this single call.

Public API (mirrors obd.py / mag.py conventions for pruner.py)
--------------------------------------------------------------
  compute_saliencies(model, x, y, device) → 1-D tensor, one score per weight.
      Weight-level saliency: the score of a weight equals the L2-norm of the
      neuron it belongs to.  Pruner.py will still call argmin on this, so the
      entire neuron's weights share the same (low) score.  To ensure the whole
      neuron is removed atomically, use select_neuron_weights() instead of
      argmin directly.

  select_neuron_weights(model, mask) → list[int]
      Returns the flat-vector indices of ALL weights belonging to the hidden
      neuron with the smallest outgoing L2-norm (ignoring already-masked
      neurons).  Call this once per pruning step instead of selecting one
      scalar weight.

Note on compatibility with pruner.py's loop
-------------------------------------------
pruner.py's inner while-loop removes one weight at a time (calling argmin on
the saliency vector).  For structured pruning we override this at the
call-site in pruner.py by calling select_neuron_weights() which masks the
whole neuron in one shot.  See README section 7 ("How to add a new pruning
method") for how METHOD_MAP is extended.
"""

from __future__ import annotations

import torch

from src.models.mlp import MLP


# ── internal helpers ──────────────────────────────────────────────────────────

def _layer_param_ranges(model: MLP) -> list[tuple[int, int, int, int]]:
    """Return the flat-vector index ranges for (weight_start, weight_end,
    bias_start, bias_end) for every Linear layer in model.layers.

    The flat vector is ordered:
      layers[0].weight (row-major), layers[0].bias,
      layers[1].weight (row-major), layers[1].bias, ...
    """
    ranges = []
    offset = 0
    for layer in model.layers:
        w_size = layer.weight.numel()   # out_features × in_features
        b_size = layer.bias.numel()     # out_features
        ranges.append((offset, offset + w_size,
                       offset + w_size, offset + w_size + b_size))
        offset += w_size + b_size
    return ranges


# ── public API ────────────────────────────────────────────────────────────────

def compute_saliencies(
    model: MLP,
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Return a weight-level saliency tensor for structured pruning.

    Every scalar weight belonging to hidden neuron j in layer l receives the
    same score: the L2-norm of that neuron's outgoing weight row ||W_l[j,:]||_2.
    Output-layer weights (last layer) receive +inf — we never prune the output.

    Args:
        model:  Trained MLP (only parameters are read; x, y unused but kept
                for API symmetry with obd.compute_saliencies).
        x_all:  Unused — kept for uniform pruner.py API.
        y_all:  Unused — kept for uniform pruner.py API.
        device: Unused — kept for uniform pruner.py API.

    Returns:
        1-D tensor of length n = model.get_weight_count() with per-weight
        saliency scores.  Pruner.py's argmin will select the flat index q;
        structured pruning then converts q → neuron → all associated indices.
    """
    model.eval()
    ranges = _layer_param_ranges(model)
    n = model.get_weight_count()
    saliency = torch.full((n,), float("inf"))

    # Only score hidden layers (all layers except the last)
    n_hidden_layers = len(model.layers) - 1
    for l_idx in range(n_hidden_layers):
        layer = model.layers[l_idx]
        w_start, w_end, _, _ = ranges[l_idx]
        out_features = layer.out_features
        in_features  = layer.in_features

        # outgoing weights: shape (out_features, in_features)
        W = layer.weight.detach()   # (out_features, in_features)

        for j in range(out_features):
            # L2 norm of row j — saliency of neuron j in this layer
            neuron_score = W[j].norm(p=2).item()

            # Assign score to the row's flat positions
            row_start = w_start + j * in_features
            row_end   = row_start + in_features
            saliency[row_start:row_end] = neuron_score

            # Also assign to bias index
            bias_start = ranges[l_idx][2]
            saliency[bias_start + j] = neuron_score

            # Assign to the corresponding column in the NEXT layer
            next_w_start = ranges[l_idx + 1][0]
            next_layer   = model.layers[l_idx + 1]
            next_in      = next_layer.in_features
            # Column j of next layer's weight matrix
            col_indices  = [next_w_start + row * next_in + j
                            for row in range(next_layer.out_features)]
            for ci in col_indices:
                saliency[ci] = neuron_score

    return saliency


def select_neuron_weights(
    model: MLP,
    mask: torch.Tensor,
) -> list[int]:
    """Return flat-vector indices of ALL weights belonging to the least-salient
    unmasked hidden neuron.

    This should be called ONCE per pruning step in place of the scalar argmin
    used by weight-removal methods.  It masks the entire neuron atomically.

    Args:
        model: MLP with current (possibly partially pruned) weights.
        mask:  Boolean tensor, True = weight is alive.

    Returns:
        List of flat-vector integer indices to set to False in mask.
        Returns [] if no hidden neurons remain active.
    """
    model.eval()
    ranges = _layer_param_ranges(model)
    n_hidden_layers = len(model.layers) - 1

    best_score  = float("inf")
    best_indices: list[int] = []

    for l_idx in range(n_hidden_layers):
        layer      = model.layers[l_idx]
        W          = layer.weight.detach()            # (out_features, in_features)
        out_f      = layer.out_features
        in_f       = layer.in_features
        w_start    = ranges[l_idx][0]
        bias_start = ranges[l_idx][2]

        next_layer   = model.layers[l_idx + 1]
        next_w_start = ranges[l_idx + 1][0]
        next_in      = next_layer.in_features
        next_out     = next_layer.out_features

        for j in range(out_f):
            # Collect all flat indices for this neuron
            row_start   = w_start + j * in_f
            row_indices = list(range(row_start, row_start + in_f))
            bias_idx    = bias_start + j
            col_indices = [next_w_start + r * next_in + j for r in range(next_out)]
            all_idx     = row_indices + [bias_idx] + col_indices

            # Skip if ALL of this neuron's outgoing weights are already pruned
            outgoing = torch.tensor(row_indices + [bias_idx])
            if not mask[outgoing].any():
                continue

            # Score: L2 norm of outgoing row (live weights only)
            row_slice = W[j]
            score = row_slice.norm(p=2).item()

            if score < best_score:
                best_score   = score
                best_indices = all_idx

    return best_indices
