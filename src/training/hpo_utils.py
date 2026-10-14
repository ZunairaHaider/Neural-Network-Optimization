"""Shared utilities for HPO: random IID validation splits and training loops.

The 4000-point HPO pool is split each repeat into train + val by random
permutation. Train and val are drawn from the same distribution as the
locked 1000-point test set (which is shuffled IID upstream in
``generate_data.py``). Different seeds per repeat give different partitions.

The function is named ``make_ood_splits`` for historical reasons (an earlier
version biased the val toward the rightmost-x extrapolation region); the
implementation is now plain IID — see audit §2.5.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def make_ood_splits(
    x: torch.Tensor,
    n_repeats: int = 1,
    val_frac: float = 0.2,
    base_seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Random in-distribution val splits (standard ML practice).

    Train and val are both drawn from the same x distribution (the HPO pool).
    The locked test set is separate. Function name kept for backward
    compatibility with older callers; despite the name it now returns random
    IID splits, not OOD splits.

    For multi-repeat (e.g., k-fold-ish behavior) use n_repeats > 1; different
    seeds per repeat produce different random partitions.
    """
    n = len(x)
    n_val = int(round(n * val_frac))
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for r in range(max(1, n_repeats)):
        rng = np.random.default_rng(base_seed + r)
        perm = rng.permutation(n)
        val_idx = np.sort(perm[:n_val])
        train_idx = np.sort(perm[n_val:])
        splits.append((train_idx, val_idx))
    return splits


def train_one(
    model: nn.Module,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_va: torch.Tensor,
    y_va: torch.Tensor,
    lr: float,
    weight_decay: float,
    max_epochs: int = 500,
    patience: int = 20,
    device: torch.device | None = None,
    record_curves: bool = False,
) -> tuple[float, dict]:
    """Full-batch Adam training with early stopping on val MSE.

    Returns (best_val_mse, info_dict). info_dict contains 'epoch' and (if
    record_curves) 'train_hist' / 'val_hist'.
    """
    device = device or torch.device("cpu")
    model = model.to(device)
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_va, y_va = x_va.to(device), y_va.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = float("inf")
    best_state = None
    no_imp = 0
    train_hist: list[float] = []
    val_hist: list[float] = []
    last_epoch = 0

    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad()
        pred = model(x_tr)
        loss = nn.functional.mse_loss(pred, y_tr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            vm = nn.functional.mse_loss(model(x_va), y_va).item()

        if record_curves:
            train_hist.append(loss.item())
            val_hist.append(vm)

        last_epoch = epoch + 1
        if vm < best_val - 1e-9:
            best_val = vm
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    info: dict = {"epoch": last_epoch}
    if record_curves:
        info["train_hist"] = train_hist
        info["val_hist"] = val_hist
    return best_val, info


def n_params_uniform_mlp(n_layers: int, n_neurons: int) -> int:
    """Param count for input=1, n_layers hidden of width n_neurons, output=1."""
    h = n_neurons
    # input layer: (1*h + h); intermediate hidden-hidden: (n_layers-1) * (h*h + h);
    # output: h*1 + 1
    return (1 * h + h) + (n_layers - 1) * (h * h + h) + (h + 1)


def n_params_mlp(
    n_layers: int,
    n_neurons: int,
    input_dim: int = 1,
    output_dim: int = 1,
) -> int:
    """Param count for a uniform-width MLP with arbitrary input/output dims.

    Architecture: Linear(input_dim, h) → [Linear(h, h)] × (n_layers-1) → Linear(h, output_dim)
    Each Linear includes a bias vector.

    When input_dim=1 and output_dim=1 this returns the same value as
    ``n_params_uniform_mlp``.
    """
    h = n_neurons
    # input layer:  input_dim * h  weights  +  h biases
    # hidden layers: (n_layers - 1) * (h * h weights + h biases)
    # output layer: h * output_dim weights + output_dim biases
    return (input_dim * h + h) + (n_layers - 1) * (h * h + h) + (h * output_dim + output_dim)
