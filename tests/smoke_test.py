"""Fast end-to-end smoke test of the core implementations (about 5 seconds on CPU).

Checks that the three central code paths still execute: the PyTorch MLP, the pruning
modules, and the Adagos constructive search (reduced to 3 cycles on a toy 1-D regression).
It is a correctness check on a tiny budget, NOT a reproduction of any number in the report.

    python tests/smoke_test.py
"""
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True  # keep the working tree clean

import numpy as np
import torch

results = {}


def step(name, fn):
    t0 = time.time()
    try:
        out = fn()
        results[name] = ("OK", out)
        print(f"[OK]   {name}  ({time.time() - t0:.2f}s)")
    except Exception as e:  # noqa: BLE001
        results[name] = ("FAIL", repr(e))
        print(f"[FAIL] {name}: {e}")
        traceback.print_exc()


def test_mlp():
    from src.models.mlp import MLP
    m = MLP(n_layers=2, n_neurons=8, activation="relu", input_dim=1, output_dim=1)
    y = m(torch.randn(32, 1))
    assert y.shape == (32, 1)
    return f"forward pass ok, {sum(p.numel() for p in m.parameters())} params"


def test_pruning():
    from src.models.mlp import MLP
    from src.pruning import mag, obd  # noqa: F401  (import check)
    torch.manual_seed(0)
    m = MLP(n_layers=1, n_neurons=16, activation="tanh", input_dim=1, output_dim=1)
    X = torch.linspace(-3, 3, 200).unsqueeze(1)
    Y = (X - 1) ** 2
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(200):
        opt.zero_grad()
        loss = ((m(X) - Y) ** 2).mean()
        loss.backward()
        opt.step()
    assert all(p.detach().abs().numel() > 0 for p in m.parameters() if p.dim() == 2)
    return f"trained 200 steps, final MSE={loss.item():.4f}; pruning modules import"


def test_adagos():
    from patenting.network import DynamicTopologyNetwork
    from patenting.architecture_search import architecture_search

    class _Null:
        def reconstruction_loss(self, X):
            return 0.0

    rng = np.random.default_rng(0)
    x = rng.uniform(-3, 3, size=(1, 400))
    y = (x - 1) ** 2
    net, _ = architecture_search(
        DynamicTopologyNetwork([1, 1, 1]), _Null(),
        x[:, :300], y[:, :300], x[:, 300:350], y[:, 300:350], x[:, 350:], y[:, 350:],
        cycles=3, n_restarts=2, full_train_epochs=50,
    )
    mse = float(np.mean((net.predict(x[:, 350:]) - y[:, 350:]) ** 2))
    n_params = sum(w.size for w in net.weights) + sum(b.size for b in net.biases)
    return f"topology {net.layers}, {n_params} params, test MSE {mse:.2e} (reduced budget)"


step("MLP forward pass (src.models.mlp)", test_mlp)
step("Pruning modules + toy training (src.pruning)", test_pruning)
step("Adagos constructive search, 3 cycles (patenting)", test_adagos)

print("\n=== SUMMARY ===")
for k, (status, out) in results.items():
    print(f"{status:4s} {k}: {out}")
sys.exit(0 if all(s == "OK" for s, _ in results.values()) else 1)
