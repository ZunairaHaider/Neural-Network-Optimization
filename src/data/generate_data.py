"""Generate and persist Dataset A (clean) and Dataset B (noisy) for the study.

Quadratic function — 1-D
-------------------------
  f(x) = x^2 - 2x + 1 = (x - 1)^2     (a=1, b=-2, c=1)

Quadratic function — d-D (d ∈ {2, 3})
-------------------------------------
  f(x) = xᵀ A x + bᵀ x + c
  A: deterministic symmetric-positive-definite matrix (Cholesky construction
     from a seeded generator → A = LLᵀ + εI), so the quadratic has a unique
     interior minimum.
  b: deterministic vector drawn from N(0, 1) with the same seed stream.
  c: 1.0 (constant offset).
  See ``QUADRATIC_PARAMS`` for the exact realisations.

Domain / grid
-------------
  x in [-3, 3]^d, N = 5000 samples.
  - d=1: UNIFORM grid (torch.linspace) random-shuffled with seed 42 — keeps
    the original 1-D dataset bit-identical to prior runs.
  - d>1: UNIFORM IID samples (torch.rand · 6 − 3) since a grid in d-D would
    either be too sparse (5000^(1/d) per axis) or too aliased.
  Both cases use a 4000/1000 train/test split.

  Why random-shuffle (not the previous OOD-by-position split):
    Standard ML best practice for regression baselines is an IID train/test
    split. This isolates the pruning question (fit quality vs. sparsity) from
    the extrapolation question. The OOD split (test = rightmost slice) and
    forward-chaining val are documented in README as future-work variants.

Noise for Dataset B
--------------------
  eps ~ N(0, sigma^2)
  - d=1: sigma = 2.0 (peak-to-peak ≈ 16 → 12.5% ratio).
  - d>1: sigma scaled to maintain the 12.5% peak-to-peak ratio. The peak-to-
    peak is computed analytically on a dense uniform sample from [-3, 3]^d
    once per dimension (deterministic seed) and persisted in
    ``QUADRATIC_PARAMS``.

Seeds
-----
  torch.manual_seed(42), np.random.seed(42), random.seed(42) at module level.
  Every run produces IDENTICAL output.

Outputs (data/processed/)
--------------------------
  1-D (legacy naming, unchanged):
    dataset_A_train.pt, dataset_A_test.pt   — {'x': tensor, 'y': tensor}
    dataset_B_train.pt, dataset_B_test.pt
  d-D (d ∈ {2, 3}):
    dataset_{A,B}_{2,3}d_{train,test}.pt
  Shared:
    datasets_preview.png — 1-D preview only (2-D/3-D not plotted).
    quadratic_params.json — A, b, c for d=2 and d=3 (provenance).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# ── seeds ────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# ── constants ─────────────────────────────────────────────────────────────────
A, B, C = 1.0, -2.0, 1.0   # f(x) = x^2 - 2x + 1 = (x-1)^2
X_MIN, X_MAX = -3.0, 3.0
# N chosen by Task-0 sufficiency check (src/data/check_sufficiency.py).
# At N=1000 the 5-seed test-MSE rel_std was 8.3% (A) and 5.7% (B).
# At N=5000 mean test MSE dropped sharply (A: 7.78→0.99, B: 12.5→7.5);
# remaining rel_std (27%/7%) is dominated by training stochasticity on the
# OOD test window (test is x∈[~1.79, 3], extrapolation), not data sparsity.
# Spec caps additional data at N=5000, so we keep N=5000.
N = 5000
N_TRAIN = 4000
SIGMA_B = 2.0               # 12.5% of peak-to-peak range 16
SIGMA_D=0.1

# ── true function ─────────────────────────────────────────────────────────────

def f(x: torch.Tensor) -> torch.Tensor:
    """f(x) = x^2 - 2x + 1 (exact, no noise) — 1-D scalar target."""
    return A * x ** 2 + B * x + C


# ── d-D quadratic parameters ──────────────────────────────────────────────────
#
# A_d, b_d, c_d are constructed once with a deterministic generator so the
# experiments are exactly reproducible. A is built as L Lᵀ + εI to guarantee
# symmetric positive-definite — the unique minimum is x* = −½ A⁻¹ b, lying
# inside [-3, 3]^d for the chosen seed (verified numerically). The matrices
# are also persisted to ``data/processed/quadratic_params.json`` for the
# methodology section of the final report.

QUADRATIC_PARAMS: dict[int, dict] = {}


def _make_quadratic_params(d: int, seed: int = 42) -> dict:
    """Deterministic A (SPD), b, c for the d-D quadratic. Cached on first call."""
    if d in QUADRATIC_PARAMS:
        return QUADRATIC_PARAMS[d]
    g = torch.Generator().manual_seed(seed + d)   # different per dim
    L = torch.randn(d, d, generator=g)
    A_mat = L @ L.T + 0.5 * torch.eye(d)         # SPD with bounded condition number
    b_vec = torch.randn(d, generator=g)
    c_val = 1.0

    # Compute peak-to-peak on a dense IID sample so we can scale Dataset B's σ.
    n_probe = 100_000
    g2 = torch.Generator().manual_seed(seed + d + 1000)
    x_probe = (torch.rand((n_probe, d), generator=g2) * 6.0 - 3.0)
    y_probe = _quadratic_eval(x_probe, A_mat, b_vec, c_val)
    peak_to_peak = float((y_probe.max() - y_probe.min()).item())
    sigma = 0.125 * peak_to_peak

    QUADRATIC_PARAMS[d] = {
        "A": A_mat,
        "b": b_vec,
        "c": c_val,
        "peak_to_peak": peak_to_peak,
        "sigma_B": sigma,
    }
    return QUADRATIC_PARAMS[d]


def _quadratic_eval(x: torch.Tensor, A_mat: torch.Tensor, b_vec: torch.Tensor, c_val: float) -> torch.Tensor:
    """y = xᵀ A x + bᵀ x + c, returned as shape (N, 1)."""
    quad = (x @ A_mat * x).sum(dim=-1, keepdim=True)
    lin  = (x @ b_vec.unsqueeze(-1))
    return quad + lin + c_val


def f_d(x: torch.Tensor, d: int) -> torch.Tensor:
    """d-D quadratic target. ``x`` has shape (N, d), returns (N, 1)."""
    p = _make_quadratic_params(d)
    return _quadratic_eval(x, p["A"], p["b"], p["c"])


# ── dataset C placeholder ─────────────────────────────────────────────────────

def load_dataset_C(path: str):  # noqa: N802
    """Placeholder for Dataset C (e.g. thermal sensor data).

    TODO: replace the NotImplementedError with actual loading logic once the
    data file is available. Expected output: (x_train, y_train, x_test, y_test)
    as float32 torch tensors shaped (N, 1).
    """
    raise NotImplementedError(
        "Dataset C is not yet available. Provide the thermal data path and "
        "implement loading logic here."
    )


# ── generation ────────────────────────────────────────────────────────────────

def _make_grid_shuffled() -> torch.Tensor:
    """Uniform grid of N points over [X_MIN, X_MAX], random-shuffled (seed=42).

    Shape: (N, 1). The shuffle uses a fresh local RNG so module-level seeds
    elsewhere are unaffected.
    """
    grid = torch.linspace(X_MIN, X_MAX, N).unsqueeze(1)
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(N, generator=g)
    return grid[perm]


def generate_dataset_A() -> tuple[dict, dict]:
    """Clean 1-D quadratic dataset. Random IID split across full [-3, 3]."""
    x = _make_grid_shuffled()
    y = f(x)
    train = {"x": x[:N_TRAIN], "y": y[:N_TRAIN]}
    test  = {"x": x[N_TRAIN:], "y": y[N_TRAIN:]}
    return train, test


def generate_dataset_B() -> tuple[dict, dict]:
    """Noisy 1-D quadratic dataset. eps ~ N(0, SIGMA_B^2). Random IID split."""
    x = _make_grid_shuffled()
    y_clean = f(x)
    noise = torch.tensor(
        np.random.normal(0.0, SIGMA_B, size=y_clean.shape), dtype=torch.float32
    )
    y = y_clean + noise
    train = {"x": x[:N_TRAIN], "y": y[:N_TRAIN]}
    test  = {"x": x[N_TRAIN:], "y": y[N_TRAIN:]}
    return train, test

def generate_dataset_D() -> tuple[dict, dict]:
    """Noisy quadratic dataset. eps ~ N(0, SIGMA_B^2). Random IID split."""
    x = _make_grid_shuffled()
    y_clean = f(x)
    noise = torch.tensor(
        np.random.normal(0.0, SIGMA_D, size=y_clean.shape), dtype=torch.float32
    )
    y = y_clean + noise
    train = {"x": x[:N_TRAIN], "y": y[:N_TRAIN]}
    test  = {"x": x[N_TRAIN:], "y": y[N_TRAIN:]}
    return train, test


def _sample_uniform_d(d: int, n: int, seed: int) -> torch.Tensor:
    """Uniform IID samples in [X_MIN, X_MAX]^d. Shape (n, d)."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand((n, d), generator=g) * (X_MAX - X_MIN) + X_MIN


def generate_dataset_dD(d: int, noisy: bool) -> tuple[dict, dict]:
    """Clean (noisy=False) or noisy (True) d-D quadratic dataset.

    Sampling: N=5000 IID uniform in [-3, 3]^d, 4000/1000 split. Noise σ is
    pre-computed in ``QUADRATIC_PARAMS`` to maintain 12.5% peak-to-peak.
    """
    if d == 1:
        raise ValueError("Use generate_dataset_A/B for d=1.")
    params = _make_quadratic_params(d)
    x = _sample_uniform_d(d, N, seed=4242 + d)
    y = f_d(x, d)
    if noisy:
        g = torch.Generator().manual_seed(7777 + d)
        eps = torch.randn(y.shape, generator=g) * params["sigma_B"]
        y = y + eps
    train = {"x": x[:N_TRAIN], "y": y[:N_TRAIN]}
    test  = {"x": x[N_TRAIN:], "y": y[N_TRAIN:]}
    return train, test


# ── I/O helpers ───────────────────────────────────────────────────────────────

def save_datasets() -> None:
    """Generate 1-D and d-D datasets and persist them to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    a_train, a_test = generate_dataset_A()
    b_train, b_test = generate_dataset_B()
    d_train, d_test= generate_dataset_D()

    # torch.save(a_train, PROCESSED_DIR / "dataset_A_train.pt")
    # torch.save(a_test,  PROCESSED_DIR / "dataset_A_test.pt")
    # torch.save(b_train, PROCESSED_DIR / "dataset_B_train.pt")
    # torch.save(b_test,  PROCESSED_DIR / "dataset_B_test.pt")
    torch.save(d_train, PROCESSED_DIR / "dataset_D_train.pt")
    torch.save(d_test,  PROCESSED_DIR / "dataset_D_test.pt")

    print(f"Saved 1-D datasets to {PROCESSED_DIR}/")
    print(f"  A: train={len(a_train['x'])}, test={len(a_test['x'])}")
    print(f"  B: train={len(b_train['x'])}, test={len(b_test['x'])}, sigma={SIGMA_B}")
    print(f"  D: train={len(d_train['x'])}, test={len(d_test['x'])}, sigma={SIGMA_D}")

    import json
    params_log: dict = {}
    for d in (2, 3):
        a_tr, a_te = generate_dataset_dD(d, noisy=False)
        b_tr, b_te = generate_dataset_dD(d, noisy=True)
        torch.save(a_tr, PROCESSED_DIR / f"dataset_A_{d}d_train.pt")
        torch.save(a_te, PROCESSED_DIR / f"dataset_A_{d}d_test.pt")
        torch.save(b_tr, PROCESSED_DIR / f"dataset_B_{d}d_train.pt")
        torch.save(b_te, PROCESSED_DIR / f"dataset_B_{d}d_test.pt")
        p = _make_quadratic_params(d)
        print(f"  {d}-D: A: train={len(a_tr['x'])}, B sigma={p['sigma_B']:.3f}"
              f"  (peak-to-peak={p['peak_to_peak']:.3f})")
        params_log[d] = {
            "A": p["A"].tolist(),
            "b": p["b"].tolist(),
            "c": p["c"],
            "peak_to_peak": p["peak_to_peak"],
            "sigma_B": p["sigma_B"],
        }
    with open(PROCESSED_DIR / "quadratic_params.json", "w") as fh:
        json.dump(params_log, fh, indent=2)


def load_split(dataset: str, split: str, dim: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a saved split.

    Args:
        dataset: 'A', 'B', 'D', or a PMSM config name (e.g. 'PMSM_no_temps').
        split:   'train', 'test', or 'val' (val only for PMSM).
        dim:     1 (default, legacy file naming) or 2 / 3.  Ignored for PMSM.

    Returns:
        (x, y) float32 tensors. x has shape (N, dim); y has shape (N, 1).
    """
    # ── PMSM datasets are handled by the dedicated loader ─────────────────
    if dataset.startswith("PMSM"):
        from src.data.pmsm_loader import load_pmsm_split
        return load_pmsm_split(dataset, split)

    # ── synthetic datasets (A / B / D, possibly multi-dim) ────────────────
    if dim == 1:
        path = PROCESSED_DIR / f"dataset_{dataset}_{split}.pt"
    else:
        path = PROCESSED_DIR / f"dataset_{dataset}_{dim}d_{split}.pt"
    data = torch.load(path, weights_only=True)
    return data["x"].float(), data["y"].float()


# ── sanity plot ───────────────────────────────────────────────────────────────

def _plot_datasets() -> None:
    """2-panel scatter + true-curve plot, saved to data/processed/."""
    x_curve = torch.linspace(X_MIN, X_MAX, 400).unsqueeze(1)
    y_curve = f(x_curve).numpy().squeeze()
    x_curve_np = x_curve.numpy().squeeze()

    a_train, a_test = generate_dataset_A()
    b_train, b_test = generate_dataset_B()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, (tr, te), title in zip(
        axes,
        [(a_train, a_test), (b_train, b_test)],
        ["Dataset A — clean", f"Dataset B — noisy (σ={SIGMA_B})"],
    ):
        ax.scatter(
            tr["x"].numpy(), tr["y"].numpy(),
            s=4, alpha=0.5, label="train", color="steelblue",
        )
        ax.scatter(
            te["x"].numpy(), te["y"].numpy(),
            s=4, alpha=0.5, label="test", color="orange",
        )
        ax.plot(x_curve_np, y_curve, "k-", linewidth=1.5, label="f(x) = (x−1)²")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        ax.legend(markerscale=3)
        ax.grid(True, alpha=0.3)

    fig.suptitle("ANN Optimisation Study — Dataset Preview", fontsize=13)
    fig.tight_layout()
    out = PROCESSED_DIR / "datasets_preview.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved sanity plot → {out}")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    save_datasets()
    # _plot_datasets()
    print(
        "\nFunction: f(x) = x^2 - 2x + 1  (a=1, b=-2, c=1)\n"
        f"Domain  : x in [{X_MIN}, {X_MAX}],  N={N} uniform grid points\n"
        f"Split   : train={N_TRAIN}, test={N - N_TRAIN}\n"
        f"Noise B : sigma={SIGMA_B}  (~{SIGMA_B / 16 * 100:.1f}% of peak-to-peak range)\n"
    )


if __name__ == "__main__":
    main()
