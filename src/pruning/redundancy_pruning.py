"""Redundancy pruning via neuron-pair merging.

Background
----------
Two neurons in the same hidden layer are *redundant* if their
post-activation outputs are highly correlated across a calibration
dataset — informally, they compute "the same feature". A correlation of
exactly 1 means one is a linear function of the other (up to bias), so
keeping both adds capacity without representation: their downstream
contribution can be absorbed into a single survivor.

This module:
  1. Runs a forward pass to collect post-activation outputs for every
     hidden layer on a held-out calibration set ``x_calib``.
  2. Computes the per-layer Pearson correlation matrix between alive
     neurons.
  3. Treats every pair with ``|corr| > 0.99`` as a redundancy candidate.
     Ranks pairs by descending correlation.
  4. For each candidate: keeps the neuron with the larger outgoing
     weight norm (the "survivor"), zeros the other's incoming weights
     and bias, and transfers a scaled contribution
     ``γ = ⟨h_i, h_j⟩ / ⟨h_i, h_i⟩`` from the redundant to the survivor
     in the next layer's incoming column.

References:
  - Mariet & Sra (2016), *Diversity networks: Neural network compression
    using determinantal point processes*. ICLR.
  - Srinivas & Babu (2015), *Data-free parameter pruning for deep neural
    networks*. BMVC.

The hard 0.99 threshold is tight by design — only near-duplicate neurons
qualify, so the method tends to remove fewer parameters than magnitude
or structured pruning but with very little quality loss when it does
fire. ``pruner.py`` falls back to magnitude pruning when no pairs remain
above threshold, so the method gracefully degrades.

Caveats
-------
The merge is applied **with weight transfer**, not pure deletion. Once a
pair is merged, the activations of the survivor change, which can
invalidate the correlation estimates for *other* pending pairs. To stay
safe, ``_merge_neurons`` clears ``self._pairs`` after each merge and the
correlations are recomputed at the next sparsity target by
``pruner.py``.

Public API
----------
  RedundancyPruning(model, x_calib, layer_names=None, device=None)
  compute_correlations(target_sparsity, current_sparsity, total_params)
      Refresh activations and the candidate pair list. Called once per
      sparsity target by ``pruner.py``.
  select_weight(w, mask) -> tuple[int, int]
      Pop the highest-correlation pair, apply the merge, and return the
      flat-vector index range of the redundant neuron's incoming weights
      (so ``pruner.py`` can update its mask). Falls back to magnitude
      pruning when no pairs remain.
  n_pairs_remaining() -> int
"""

from __future__ import annotations

import warnings
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class _NeuronPair:
    layer_name: str
    survivor_idx: int
    redundant_idx: int
    correlation: float
    redundant_flat_start: int
    redundant_flat_end: int
    next_layer: Optional[nn.Linear]
    next_layer_w_offset: int
    activations: torch.Tensor


class RedundancyPruning:
    def __init__(
        self,
        model: nn.Module,
        x_calib: torch.Tensor,
        layer_names=None,
        device=None,
    ):
        self.model = model
        self.x_calib = x_calib
        self.device = device or x_calib.device

        self._ordered_layers = [
            (name, mod)
            for name, mod in model.named_modules()
            if isinstance(mod, nn.Linear)
            and (layer_names is None or name in layer_names)
        ]

        self._pairs = []
        self._param_offsets = {}
        self._total_params = 0

    # ------------------------------------------------------------
    # Keep same API
    # ------------------------------------------------------------
    def compute_correlations(
        self,
        target_sparsity: float,
        current_sparsity: float,
        total_params: int,
    ):
        self._pairs.clear()
        self._param_offsets.clear()

        activations = {}
        hooks = []

        for name, layer in self._ordered_layers:
            def hook(mod, inp, out, layer_name=name):
                activations[layer_name] = out.detach()

            hooks.append(layer.register_forward_hook(hook))

        self.model.eval()
        with torch.no_grad():
            self.model(self.x_calib.to(self.device))

        for h in hooks:
            h.remove()

        # parameter offsets
        offset = 0
        for pname, p in self.model.named_parameters():
            self._param_offsets[pname] = offset
            offset += p.numel()
        self._total_params = offset

        next_layer_map = {
            name: (
                self._ordered_layers[i + 1]
                if i + 1 < len(self._ordered_layers)
                else None
            )
            for i, (name, _) in enumerate(self._ordered_layers)
        }

        all_candidates = []

        for layer_name, layer in self._ordered_layers[:-1]:
            if layer_name not in activations:
                continue

            H = activations[layer_name]
            N, hidden_dim = H.shape

            H_centered = H - H.mean(0, keepdim=True)
            std = H_centered.std(0)

            alive = std > 1e-6
            if alive.sum() < 2:
                continue

            H_alive = H_centered[:, alive]
            std_alive = std[alive]

            H_norm = H_alive / std_alive.unsqueeze(0)

            corr = (H_norm.T @ H_norm) / N
            corr.fill_diagonal_(0)

            original_idx = alive.nonzero(as_tuple=True)[0]

            weight_name = f"{layer_name}.weight"
            w_offset = self._param_offsets[weight_name]
            in_features = layer.in_features

            next_info = next_layer_map[layer_name]
            next_layer = None
            next_w_offset = -1

            if next_info is not None:
                next_name, next_layer = next_info
                next_w_offset = self._param_offsets.get(
                    f"{next_name}.weight",
                    -1
                )

            for i in range(corr.shape[0]):
                for j in range(i + 1, corr.shape[1]):
                    c = corr[i, j].abs().item()

                    if c < 0.99:
                        continue

                    orig_i = original_idx[i].item()
                    orig_j = original_idx[j].item()

                    norm_i = layer.weight.data[orig_i].norm()
                    norm_j = layer.weight.data[orig_j].norm()

                    if norm_i >= norm_j:
                        survivor = orig_i
                        redundant = orig_j
                    else:
                        survivor = orig_j
                        redundant = orig_i

                    redundant_start = (
                        w_offset + redundant * in_features
                    )

                    all_candidates.append(
                        _NeuronPair(
                            layer_name=layer_name,
                            survivor_idx=survivor,
                            redundant_idx=redundant,
                            correlation=c,
                            redundant_flat_start=redundant_start,
                            redundant_flat_end=redundant_start + in_features,
                            next_layer=next_layer,
                            next_layer_w_offset=next_w_offset,
                            activations=H
                        )
                    )

        all_candidates.sort(
            key=lambda p: -p.correlation
        )

        self._pairs = all_candidates

        if not self._pairs:
            warnings.warn(
                "No redundant pairs found.",
                UserWarning
            )

    # ------------------------------------------------------------
    # Keep same API
    # ------------------------------------------------------------
    def select_weight(
        self,
        w: torch.Tensor,
        mask: torch.Tensor
    ):
        while self._pairs:
            pair = self._pairs.pop(0)

            rs = pair.redundant_flat_start
            re = pair.redundant_flat_end

            if mask[rs:re].any():
                self._merge_neurons(w, pair)

                print(
                    f"[Redundancy] merged "
                    f"{pair.layer_name}: "
                    f"{pair.survivor_idx} <- {pair.redundant_idx}"
                )

                return rs, re

        # fallback to magnitude pruning
        scores = w.abs().clone()
        scores[~mask] = float("inf")

        q = int(scores.argmin())
        return q, q + 1

    def n_pairs_remaining(self):
        return len(self._pairs)

    # ------------------------------------------------------------
    # FIXED merge logic
    # ------------------------------------------------------------
    def _merge_neurons(
        self,
        w: torch.Tensor,
        pair: _NeuronPair
    ):
        i = pair.survivor_idx
        j = pair.redundant_idx

        H = pair.activations

        h_i = H[:, i]
        h_j = H[:, j]

        eps = 1e-8

        gamma = torch.dot(h_i, h_j) / (
            torch.dot(h_i, h_i) + eps
        )
        gamma = gamma.clamp(-1.0, 1.0)

        # ---------------------------
        # DO NOT average incoming weights
        # ---------------------------
        rs = pair.redundant_flat_start
        re = pair.redundant_flat_end
        w[rs:re] = 0.0

        # zero bias
        layer = dict(self.model.named_modules())[pair.layer_name]

        if layer.bias is not None:
            layer.bias.data[j] = 0.0

        # ---------------------------
        # transfer downstream contribution
        # ---------------------------
        if pair.next_layer is not None:
            pair.next_layer.weight.data[:, i] += (
                gamma * pair.next_layer.weight.data[:, j]
            )

            pair.next_layer.weight.data[:, j] = 0.0

        # recompute future correlations next iteration
        self._pairs.clear()