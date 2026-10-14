"""IMP with weight rewinding — Frankle & Carbin (2019); Frankle et al. (2020).

Iterative Magnitude Pruning (IMP) prunes the smallest-magnitude weights, then
*rewinds* the surviving weights to an early-training checkpoint and retrains.
The result tests the Lottery Ticket Hypothesis: that the dense network
contains a sparse subnetwork which, when trained from the same starting
weights, can match the dense model's accuracy.

Variants
--------
- Frankle & Carbin (2019): rewind to **iteration 0** (the random init).
- Frankle, Dziugaite, Roy, Carbin (2020): rewind to an **early-training**
  checkpoint (e.g. after a few epochs of warm-up), which the authors showed
  is more stable for larger networks.

In this repo we rewind to **iteration 0** (the saved random init). For our
tiny MLPs (≲ 300 params) there is no benefit to a warm-up checkpoint —
training is fast enough that the random init is itself a clean start.

Pruner integration
------------------
At the top of the run, we save the random init that the baseline MLP was
spawned with (re-instantiated deterministically). Per sparsity target:
  1. From the current model state (trained from a prior rewound init),
     compute |w| saliency.
  2. Mask the smallest-magnitude alive weights to additionally reach the
     target sparsity (cumulative: same schedule as the iterative loop).
  3. **Rewind**: copy init weights back into surviving positions; masked
     positions stay zero.
  4. Re-train via the standard fine-tune loop (which is effectively training
     from init since the weights have just been rewound).
  5. Evaluate test once.

This module exposes a single saliency function; the rewind itself is handled
in pruner.py since it requires access to the saved init state.

References
----------
Frankle & Carbin (2019), *The Lottery Ticket Hypothesis: Finding Sparse,
Trainable Neural Networks*, ICLR.
Frankle, Dziugaite, Roy, Carbin (2020), *Linear Mode Connectivity and the
Lottery Ticket Hypothesis*, ICML.
"""

from __future__ import annotations

import torch


def select_weight(w: torch.Tensor, mask: torch.Tensor) -> int:
    """Return the index of the unmasked weight with the smallest |w|.

    Same selection rule as plain magnitude pruning; the IMP-specific part is
    the rewind-then-retrain protocol, applied at the pruner level.
    """
    saliency = w.detach().abs()
    saliency = torch.where(mask, saliency, torch.full_like(saliency, float("inf")))
    return int(saliency.argmin().item())
