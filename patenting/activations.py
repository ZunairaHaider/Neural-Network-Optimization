# ============================================================================
# activations.py
# ============================================================================
#
# PATENT ALIGNMENT NOTES (US 12,124,779 B2)
# ------------------------------------------
#
# The patent specifies sigmoid as the hidden-layer activation (col.12):
#
#   f_{σ}(x) = 1 / (1 + exp(-x))
#
# and the Lagrange-multiplier backprop in the appendix is derived explicitly
# for sigmoid derivatives.  Using ReLU (previous implementation) breaks the
# zero-gradient assumption for inactive neurons, which can produce ΔL ≈ 0
# for any additive candidate regardless of its true utility, distorting
# candidate ranking and causing premature stopping.
#
# Sigmoid does not suffer from dead-neuron collapse, so the multi-restart
# heuristic in train_network can be reduced (though kept for robustness).
# Xavier/Glorot initialisation is used instead of He, as is standard for
# sigmoid networks.
#
# ============================================================================

import numpy as np


# ============================================================================
# SIGMOID  (patent activation, col.12)
# ============================================================================

def sigmoid(x):
    """Numerically stable sigmoid: clips large |x| to avoid overflow."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
        np.exp(np.clip(x, -500, 500)) / (1.0 + np.exp(np.clip(x, -500, 500))),
    )


def sigmoid_deriv(x):
    """σ'(x) = σ(x)(1 − σ(x))"""
    s = sigmoid(x)
    return s * (1.0 - s)


# ============================================================================
# RELU  (kept for reference / ablation; no longer used in main pipeline)
# ============================================================================

def relu(x):
    return np.maximum(0, x)


def relu_deriv(x):
    return (x > 0).astype(float)

# ========================================================
# Tanh
# ========================================================


def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.tanh(x) ** 2