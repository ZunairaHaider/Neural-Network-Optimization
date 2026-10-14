"""Low-rank pruning via per-Linear SVD truncation.

Background
----------
Every fully-connected layer ``y = W x + b`` can be factored
``W = U · diag(S) · Vᵀ`` (singular value decomposition). The smallest
singular values contribute the least to the layer's input → output map,
so zeroing them is a principled "compression" that does not look at
individual scalar weights at all. Equivalent operations: think of the
layer's effective rank.

References:
  - Denil, Shakibi, Dinh, Ranzato, de Freitas (2013), *Predicting parameters
    in deep learning*. NeurIPS.
  - Sainath, Kingsbury, Sindhwani, Arisoy, Ramabhadran (2013), *Low-rank
    matrix factorization for deep neural network training with high-
    dimensional output targets*. ICASSP.

This module implements low-rank pruning **without** the early-exit
re-parameterisation typically used at inference time — the pruner keeps
the dense ``nn.Linear`` shape but truncates SVs in place via the
``write_back`` mechanism. ``get_reparameterised`` is provided as an
optional UV-factored fine-tuning step (used inside ``pruner.py`` to
preserve the low-rank structure during recovery).

Why it pairs well with the other pruning methods
------------------------------------------------
Low-rank is unique in this repo: every other method drops scalar weights
or whole neurons, while SVD truncation reduces the effective rank without
producing sparsity in the per-weight sense. This makes the per-weight
"sparsity" metric reported in the compare-table somewhat misleading for
low-rank — the relevant metric is the surviving rank. See ``pruner.py``'s
``lr_mod.total_sv_params()`` for the corresponding parameter count.

Public API (consumed by ``src/pruning/pruner.py``)
--------------------------------------------------
  LowRankPruning(model, layer_names=None, device=None)
      Constructor — decomposes every targeted Linear's weight matrix.
  rank_sparsity() -> float
      Fraction of singular values currently zeroed across all layers.
  svs_to_prune_for_target(target_sparsity, current_sparsity) -> int
      How many additional SVs to remove to hit a given rank-sparsity.
  prune_one_sv() -> None
      Zero out the next-smallest surviving SV across all layers.
  get_reparameterised() -> nn.Module
      Build a fresh module where each Linear is split into
      ``x → Vᵀ → diag(S) → U``, dropping zeroed SVs (yields a smaller
      dense net for fine-tuning).
  finetune(rep_model, x_tr, y_tr, x_val, y_val, lr, batch_size, device)
      Train the UV-factored model with Adam + early stopping on the held-out
      val split (audit §2.1 leakage fix landed).
  write_back() -> None
      Reconstruct each original Linear's weight from the surviving SVs
      and write it back into the model.
  enforce_lowrank() -> None
      Re-SVD after fine-tuning to project away any drift.
  total_sv_params() -> int
      Effective parameter count, treating each surviving SV as
      ``rank · (rows + cols) + bias``.
  get_rank_info() -> dict[str, tuple[int, int]]
      ``{layer_name: (surviving_rank, full_rank)}``.
"""

from __future__ import annotations

import math
import heapq
import copy
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from dataclasses import dataclass


# ------------------------------------------------------------
# Dataclass
# ------------------------------------------------------------

@dataclass
class _SVDLayer:
    layer_name: str
    layer_ref: nn.Linear

    rows: int
    cols: int
    full_rank: int

    U: torch.Tensor
    S: torch.Tensor
    Vh: torch.Tensor


# ------------------------------------------------------------
# Low-rank pruner
# ------------------------------------------------------------

class LowRankPruning:
    def __init__(
        self,
        model: nn.Module,
        layer_names: list[str] | None = None,
        device: torch.device | None = None,
    ):
        self.model = model
        self.device = device or torch.device("cpu")

        self._target_layers = {
            name: mod
            for name, mod in model.named_modules()
            if isinstance(mod, nn.Linear)
            and (layer_names is None or name in layer_names)
        }

        self._svd_layers: dict[str, _SVDLayer] = {}
        self._heap = []

        self._decompose_once()

    # --------------------------------------------------------
    # Initial decomposition
    # --------------------------------------------------------

    def _decompose_once(self):
        print("[LowRank] Initial decomposition...", flush=True)

        self._svd_layers.clear()
        self._heap.clear()

        for layer_name, layer in self._target_layers.items():
            W = layer.weight.data.float().to(self.device)

            rows, cols = W.shape
            full_rank = min(rows, cols)

            U, S, Vh = torch.linalg.svd(W, full_matrices=False)

            svd_layer = _SVDLayer(
                layer_name=layer_name,
                layer_ref=layer,
                rows=rows,
                cols=cols,
                full_rank=full_rank,
                U=U,
                S=S,
                Vh=Vh,
            )

            self._svd_layers[layer_name] = svd_layer

            for k in range(full_rank):
                heapq.heappush(
                    self._heap,
                    (
                        float(S[k] ** 2),   # prune smallest singular values first
                        layer_name,
                        k
                    )
                )

            print(
                f"[LowRank] {layer_name}: "
                f"{rows}x{cols}, rank={full_rank}",
                flush=True
            )

    # --------------------------------------------------------
    # Current sparsity
    # --------------------------------------------------------

    def rank_sparsity(self) -> float:
        total = sum(layer.full_rank for layer in self._svd_layers.values())

        removed = 0
        for layer in self._svd_layers.values():
            removed += (layer.S == 0).sum().item()

        return removed / total

    # --------------------------------------------------------
    # How many SVs to prune
    # --------------------------------------------------------

    def svs_to_prune_for_target(
        self,
        target_sparsity: float,
        current_sparsity: float,
    ) -> int:
        total_rank = sum(
            layer.full_rank
            for layer in self._svd_layers.values()
        )

        current_pruned = round(current_sparsity * total_rank)
        target_pruned = round(target_sparsity * total_rank)

        return max(0, target_pruned - current_pruned)

    # --------------------------------------------------------
    # Prune one singular value
    # --------------------------------------------------------

    def prune_one_sv(self):
        while self._heap:
            sv_val, layer_name, k = heapq.heappop(self._heap)

            layer = self._svd_layers[layer_name]
            active_rank = (layer.S > 0).sum().item()
            if active_rank <= 1:

                continue

            print(
                f"[LowRank] Pruning: "
                f"{layer_name} | k={k} | σ={sv_val:.6f}",
                flush=True
            )

            layer.S[k] = 0.0
            return

        print("[LowRank] No singular values left to prune.")

    # --------------------------------------------------------
    # Build low-rank trainable model
    # --------------------------------------------------------

    def get_reparameterised(self) -> nn.Module:
        """
        Rebuilds model where each linear layer becomes:

        x -> Vᵀ -> diag(S) -> U
        """

        model_copy = copy.deepcopy(self.model)

        for name, module in model_copy.named_modules():
            if name not in self._svd_layers:
                continue

            svd_layer = self._svd_layers[name]

            rank = int((svd_layer.S > 0).sum().item())

            if rank == 0:
                rank = 1

            U = svd_layer.U[:, :rank]
            S = svd_layer.S[:rank]
            Vh = svd_layer.Vh[:rank, :]

            first = nn.Linear(
                svd_layer.cols,
                rank,
                bias=False
            )

            second = nn.Linear(
                rank,
                svd_layer.rows,
                bias=True
            )

            first.weight.data = Vh.clone()

            second.weight.data = (U * S.unsqueeze(0)).clone()

            if module.bias is not None:
                second.bias.data = module.bias.data.clone()

            self._replace_module(
                model_copy,
                name,
                nn.Sequential(first, second)
            )

        return model_copy.to(self.device)
    
    def enforce_lowrank(self):
        """
        Re-project current weights back into SVD space.
        This prevents drift after fine-tuning.
        """
        for layer_name, layer in self._svd_layers.items():
            W = layer.layer_ref.weight.data.float().to(self.device)

            U, S, Vh = torch.linalg.svd(W, full_matrices=False)

            layer.U = U
            layer.S = S
            layer.Vh = Vh

        # IMPORTANT: rebuild heap consistency
            self._heap.clear()

            full_rank = min(W.shape)

            for k in range(full_rank):
                heapq.heappush(
                self._heap,
                (float(S[k]), layer_name, k)
                )
    # --------------------------------------------------------
    # Write trained low-rank weights back
    # --------------------------------------------------------

    def write_back(self):
        """
        Write low-rank reconstruction back to model,
        using ONLY retained singular values.
        """
        for layer_name, svd_layer in self._svd_layers.items():

            mask = svd_layer.S > 0

            U = svd_layer.U[:, mask]
            S = svd_layer.S[mask]
            Vh = svd_layer.Vh[mask, :]

            W = (U * S.unsqueeze(0)) @ Vh

            layer = svd_layer.layer_ref
            layer.weight.data.copy_(W)

    # --------------------------------------------------------
    # Effective parameter count
    # --------------------------------------------------------

    def total_sv_params(self) -> int:
        total = 0

        for layer in self._svd_layers.values():
            rank = int((layer.S > 0).sum().item())

            total += rank * (layer.rows + layer.cols)

            if layer.layer_ref.bias is not None:
                total += layer.rows

        return total

    # --------------------------------------------------------
    # Rank info
    # --------------------------------------------------------

    def get_rank_info(self):
        info = {}

        for name, layer in self._svd_layers.items():
            current_rank = int((layer.S > 0).sum().item())

            info[name] = (
                current_rank,
                layer.full_rank
            )

        return info

    # --------------------------------------------------------
    # Replace nested modules
    # --------------------------------------------------------

    def _replace_module(
        self,
        model: nn.Module,
        module_name: str,
        new_module: nn.Module
    ):
        parts = module_name.split(".")
        parent = model

        for p in parts[:-1]:
            parent = getattr(parent, p)

        setattr(parent, parts[-1], new_module)


    def finetune(
        self,
        rep_model,
        x_tr,
        y_tr,
        x_val,
        y_val,
        lr: float,
        batch_size: int,
        device,
        max_epochs: int = 200,
        patience: int = 10,
    ):
        """
        Finetune the reparameterised low-rank model.
        No mask logic, no sparsity projection — structure is enforced by design.
        Early stopping uses the held-out val split (audit §2.1 leakage fix).
        """

        rep_model = rep_model.to(device)
        rep_model.train()

        optimiser = torch.optim.Adam(rep_model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        loader = DataLoader(
            TensorDataset(x_tr.to(device), y_tr.to(device)),
            batch_size=batch_size,
            shuffle=True,
        )

        best_mse = float("inf")
        best_state = None
        no_improve = 0

        for _ in range(max_epochs):
            for xb, yb in loader:
                optimiser.zero_grad()
                loss = criterion(rep_model(xb), yb)
                loss.backward()
                optimiser.step()

            mse = criterion(rep_model(x_val.to(device)),
                        y_val.to(device)).item()

            if mse < best_mse - 1e-9:
                best_mse = mse
                best_state = copy.deepcopy(rep_model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                break

        if best_state is not None:
            rep_model.load_state_dict(best_state)

        return best_mse