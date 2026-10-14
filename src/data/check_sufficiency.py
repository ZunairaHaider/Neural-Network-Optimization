"""Task 0 — Data sufficiency check.

Reports:
  * Trivial predictor (mean of y_train) test MSE for A and B.
  * Noise-floor-normalised MSE for B (test MSE / sigma^2).
  * Test-MSE std across 5 seeds for a fixed small MLP — proxy for training
    stochasticity. If std/mean > 5% on either dataset, regenerate at N=5000.

Run from repo root:
    python src/data/check_sufficiency.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.generate_data import load_split
from src.models.mlp import MLP

SIGMA_B_SQ = 4.0
SEEDS = [42, 7, 123, 2024, 31]


def _set_seed(s: int) -> None:
    torch.manual_seed(s); np.random.seed(s); random.seed(s)


def _trivial_mse(dataset: str) -> tuple[float, float]:
    x_tr, y_tr = load_split(dataset, "train")
    x_te, y_te = load_split(dataset, "test")
    yhat = y_tr.mean()
    mse = float(((y_te - yhat) ** 2).mean())
    return mse, float(yhat)


def _train_once(dataset: str, seed: int) -> float:
    _set_seed(seed)
    x_tr, y_tr = load_split(dataset, "train")
    x_te, y_te = load_split(dataset, "test")
    model = MLP(n_layers=2, n_neurons=24, activation="tanh")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=32, shuffle=True)
    criterion = nn.MSELoss()
    best = float("inf"); no_imp = 0
    for _ in range(300):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); criterion(model(xb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            m = criterion(model(x_te), y_te).item()
        if m < best - 1e-9:
            best = m; no_imp = 0
        else:
            no_imp += 1
        if no_imp >= 15: break
    return best


def main() -> None:
    print("=" * 60)
    print("TASK 0 — Data sufficiency report")
    print("=" * 60)
    for ds in ["A", "B"]:
        mse, mean_y = _trivial_mse(ds)
        ratio = mse / SIGMA_B_SQ if ds == "B" else None
        print(f"\nDataset {ds}:")
        print(f"  mean(y_train)              = {mean_y:.4f}")
        print(f"  trivial-predictor test MSE = {mse:.4f}")
        if ratio is not None:
            print(f"  / noise floor (σ²=4)       = {ratio:.2f}×")

        test_mses = [_train_once(ds, s) for s in SEEDS]
        mean_m = float(np.mean(test_mses))
        std_m  = float(np.std(test_mses))
        rel    = std_m / mean_m if mean_m > 0 else float("inf")
        print(f"  5-seed test MSE: mean={mean_m:.4e}  std={std_m:.2e}  rel_std={rel*100:.2f}%")
        print(f"  → {'OK (rel_std ≤ 5%)' if rel <= 0.05 else 'NEEDS MORE DATA (rel_std > 5%)'}")


if __name__ == "__main__":
    main()
