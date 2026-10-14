"""Magnitude-based weight pruning — L1 absolute value, no Hessian.

Selects the unmasked weight with the smallest |w_i| for removal.  No
second-order information, no closed-form correction — pruner.py applies
fine-tuning afterward to recover.

Public API mirrors obd.py for interchangeability in pruner.py:
  select_weight(w, mask) → int

(Adapted from archive/pruning_study/pruning/magnitude.py: removed the
classification-loop machinery, kept the saliency selection logic.)
"""

from __future__ import annotations

import torch


def select_weight(w: torch.Tensor, mask: torch.Tensor) -> int:
    """Return the index of the unmasked weight with the smallest |w|."""
    saliency = w.detach().abs()
    saliency = torch.where(mask, saliency, torch.full_like(saliency, float("inf")))
    return int(saliency.argmin().item())
