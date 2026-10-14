"""
runtime_analysis/flops_estimator.py
-------------------------------------
Estimates multiply-add (MAC) operations for one forward pass through
the pruned MLP, taking into account the weight mask (zero weights are
skipped in sparse execution models).

This is an *estimate* — real FLOP savings depend on hardware sparsity
support. The number here is: non-zero weights × 2 (multiply + add).

Usage
-----
    from runtime_analysis.flops_estimator import estimate_flops

    flops = estimate_flops(model, mask, input_dim=1)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Allow importing from repo root even when called standalone
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def estimate_flops(model: nn.Module, mask: torch.Tensor | None = None) -> float:
    """Return estimated multiply-add count for one forward pass.

    Parameters
    ----------
    model : nn.Module
        The (possibly sparse) MLP.
    mask : torch.BoolTensor | None
        If provided, only non-masked (non-zero) weights are counted.
        If None, counts all weights (dense baseline FLOPs).

    Returns
    -------
    float
        Total multiply-adds. Multiply by 2 for FLOP count.
    """
    total_macs: float = 0.0
    param_idx = 0  # position in the flat mask vector

    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue

        weight = module.weight  # shape: (out_features, in_features)
        n_weights = weight.numel()

        if mask is None:
            # Dense: every weight contributes one MAC per output element
            total_macs += n_weights
        else:
            # Sparse: count only live (True) mask entries for this layer
            layer_mask = mask[param_idx: param_idx + n_weights]
            total_macs += float(layer_mask.sum().item())

        param_idx += n_weights

        # Bias doesn't count as MACs in the standard convention
        if module.bias is not None:
            param_idx += module.bias.numel()

    return total_macs


def dense_flops(model: nn.Module) -> float:
    """Convenience wrapper — returns dense (no mask) FLOPs estimate."""
    return estimate_flops(model, mask=None)
