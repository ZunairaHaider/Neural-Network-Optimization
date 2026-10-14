# ============================================================================
# lagrangian.py
# ============================================================================
#
# PATENT ALIGNMENT NOTES (US 12,124,779 B2)
# ------------------------------------------
#
# The Lagrangian is defined as (patent eq. 5):
#
#   L(Γ, W, X, Λ) = J(Γ, W)
#                   + Σ_i  Λ_i^T (X^i - f_{σ}(W_{i-1} * X^{i-1}))
#
# where f_{σ}(W_{i-1} * X^{i-1}) is the FULL pre-activation of layer i,
# including ALL incoming connections — standard weights AND skip connections.
#
# Key property (eq. 6): when X is produced by the forward pass, every
# constraint term is exactly zero, so L == J (MSE loss).
#
# Bug fixed: the constraint term must reconstruct the pre-activation using
# both the standard weight matrix AND any skip connections targeting layer i.
# Omitting skip contributions makes the constraint non-zero even for the
# current consistent network, causing L ≠ MSE and wildly incorrect estimates.
#
# ============================================================================

import numpy as np

from patenting.activations import sigmoid, sigmoid_deriv, tanh_deriv, tanh
from patenting.layer_types import LAYER_REGISTRY, forward_layer, jacobian_layer

# ============================================================================
# HELPERS
# ============================================================================

def _full_preactivation(net, acts, layer_idx):
    """
    Reconstructs the full pre-activation z^i = W_{i-1} X^{i-1} + b_{i-1}
    + Σ_{skip sources} skip_W @ X^{src}

    This must match exactly what forward() computes so that the constraint
    error (acts[i] - predicted) is zero for the current consistent network.
    """
    i = layer_idx                          # 1-based layer index in acts[]
    weight_idx = i - 1                     # index into net.weights[]

    z = net.weights[weight_idx] @ acts[i - 1] + net.biases[weight_idx]

    # Add skip connection contributions targeting this layer
    if i in net.skip_connections:
        for src_layer, skip_W, skip_b in net.skip_connections[i]:
            z += skip_W @ acts[src_layer] + skip_b

    return z


# ============================================================================
# COMPUTE LAGRANGE MULTIPLIERS  (solving ∂_{X^i} L = 0, patent eq. 7)
# ============================================================================

def calculate_lagrange_multipliers(net, acts, pre_acts, Y):
    """
    Backpropagate Lagrange multipliers Λ_i through the network.

    At the output layer (i = nc):
        Λ_{nc} = 2 (X^{nc} - Y)

    For hidden layers i = nc-1, …, 1:
        Λ_i = (W_i^T Λ_{i+1}) * σ'(z_i)

    Skip connections contribute additional terms: if layer j has a skip
    connection from layer i, then Λ_i receives W_skip^T Λ_j * σ'(z_i).
    This ensures the multipliers correctly account for all paths through
    which layer i's activations affect the output.
    """
    assert len(net.layer_types) == len(net.weights)
    num_layers = len(net.weights)
    lam = [None] * (num_layers + 1)

    # Output multiplier
    lam[num_layers] = 2.0 * (acts[-1] - Y)

    # Backpropagate through hidden layers
    for i in range(num_layers - 1, 0, -1):

        # Standard backprop from the next layer
        grad = net.weights[i].T @ lam[i + 1]

        # Add contributions from skip connections that originate at layer i
        # (i.e., skip connections from layer i to some later layer j)
        for target_layer, conns in net.skip_connections.items():
            for src_layer, skip_W, skip_b in conns:
                if src_layer == i and target_layer <= num_layers:
                    grad = grad + skip_W.T @ lam[target_layer]
        deriv  = jacobian_layer(net.layer_types[i - 1], pre_acts[i - 1])
        lam[i] = grad * deriv

    return lam


# ============================================================================
# FULL LAGRANGIAN  L(Γ, W, X, Λ)   (patent eq. 5)
# ============================================================================

def calculate_lagrangian(net, X, Y, lam=None):
    """
    Computes L(Γ, W, X, Λ) per patent eq. 5.

    For the CURRENT network (acts produced by forward pass):
        All constraint terms are zero → L == MSE (eq. 6).

    For a CANDIDATE network evaluated with OLD multipliers from the previous
    topology, the constraint terms are non-zero and give the cheap error
    variation estimate (eq. 13 / 14).

    Critical: _full_preactivation() must account for skip connections so
    the constraint error is zero for the current consistent network.
    """

    acts, pre_acts = net.forward(X)

    # Prediction loss J = ||X^{nc} - Y||²
    pred_loss = np.mean((acts[-1] - Y) ** 2)

    # Lagrange multipliers
    if lam is None:
        lam = calculate_lagrange_multipliers(net, acts, pre_acts, Y)

    # Constraint penalty  Σ_i  Λ_i^T (X^i - f_σ(W_{i-1} X^{i-1} + skip))
    lagrange_term = 0.0

    for i in range(1, len(acts)):

        # Full predicted pre-activation including skip connections
        predicted = _full_preactivation(net, acts, i)

        predicted = forward_layer(net.layer_types[i-1],predicted)

        constraint_error = acts[i] - predicted
        lagrange_term += np.sum(lam[i] * constraint_error)

    return pred_loss + lagrange_term, lam
