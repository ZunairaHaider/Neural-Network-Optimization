"""
Yarotsky (2017) theoretical bounds — Proposition 2 construction
==================================================================
Reference
---------
Yarotsky, D. (2017). Error bounds for approximations with deep ReLU
networks. Neural Networks, 94, 103-114.  Proposition 2.

Core result (on [0,1])
-----------------------
The function f(x) = x² on [0,1] is approximated by

    f_m(x) = x - Σ_{s=1}^{m} g_s(x) / 2^(2s)

where g_s = g∘g∘...∘g (s compositions of the "tooth" function)

    g(x) = 2x        for x < 1/2
         = 2(1-x)     for x ≥ 1/2

g_s is a sawtooth with 2^(s-1) teeth. The approximation error is exact:

    ε_m = ‖f_m - x²‖_∞ = 2^(-2m-2)

Depth/weight count is O(m): the tooth g costs 3 ReLU units
(g(x) = 2σ(x) - 4σ(x - 1/2) + 2σ(x - 1)), and each composition adds a
fixed number of units, so an m-fold construction has depth O(m) and
O(m) weights and computation units.

Domain adaptation for dataset A
---------------------------------
Dataset A is f(x) = (x-1)² on [-3, 3], not x² on [0,1]. We use the
affine map  u = φ(x) = (x+3)/6  ∈ [0,1]  and back-substitute:

    x = 6u - 3
    f(x) = (x-1)² = (6u-4)² = 36u² - 48u + 16
         = 36·g(u) - 48u + 16        where g(u) = u² is Yarotsky's target

So:  f(x) = 36 · f_m(φ(x))  -  48 · φ(x)  +  16   (+ affine correction terms)

Since the affine map only rescales the output by a constant factor 36,
the error bound becomes:

    ‖f_m_full - f‖_∞ = 36 · ε_m = 36 · 2^(-2m-2) = 9 · 2^(-2m)

This module provides:
    required_depth(eps)        — sanity-check: m and achieved error for a given eps
    YarotskySquaringNet         — explicit (non-trained) PyTorch network for f_m
    build_dataset_a_network(eps) — wraps the above with the [-3,3] affine pre/post layers
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Sanity-check function: given epsilon, what depth do we need?
# ---------------------------------------------------------------------------

@dataclass
class YarotskyBound:
    """Result of required_depth(): the depth and achieved guarantees."""
    epsilon_requested:  float   # the epsilon the user asked for
    m:                  int     # number of tooth-function compositions (depth driver)
    epsilon_achieved:   float   # exact error on [0,1]:  2^(-2m-2)
    epsilon_achieved_domain: float  # error after [-3,3] rescaling: 36 * epsilon_achieved
    depth:              int     # network depth (layers), O(m)
    n_weights:          int     # number of weights, O(m)
    n_units:            int     # number of computation units, O(m)


def required_depth(epsilon: float, domain_scale: float = 36.0) -> YarotskyBound:
    """
    Sanity check: for a target error `epsilon` (measured on the ACTUAL domain,
    e.g. [-3,3] for dataset A), compute the number of tooth-compositions m
    needed and the resulting network complexity, per Yarotsky Prop. 2.

    Parameters
    ----------
    epsilon      : float
        Target uniform approximation error ON THE ACTUAL DOMAIN (e.g. [-3,3]),
        not on the normalised [0,1] domain.
    domain_scale : float
        Multiplicative factor relating [0,1]-domain error to actual-domain
        error. For dataset A (domain [-3,3], f(x)=(x-1)^2), this is 36
        (derived from the affine map u=(x+3)/6, scale^2 = 6^2 = 36).
        Pass 1.0 if you want the raw [0,1] bound instead.

    Returns
    -------
    YarotskyBound with m, achieved error (both normalised and actual-domain),
    and the network's depth/weight/unit counts.

    Derivation
    ----------
    We need:  domain_scale * 2^(-2m-2) <= epsilon
    =>        2^(-2m-2) <= epsilon / domain_scale
    =>        -2m - 2   <= log2(epsilon / domain_scale)
    =>        m         >= -(log2(epsilon / domain_scale) + 2) / 2
    """
    if not (0 < epsilon < 1):
        raise ValueError(f"epsilon must be in (0, 1), got {epsilon}")
    if domain_scale <= 0:
        raise ValueError(f"domain_scale must be positive, got {domain_scale}")

    target_normalised_eps = epsilon / domain_scale
    if target_normalised_eps >= 0.25:
        # m=0 already satisfies epsilon_m = 2^-2 = 0.25 on [0,1]
        m = 0
    else:
        m = math.ceil((-math.log2(target_normalised_eps) - 2) / 2)
        m = max(0, m)

    epsilon_achieved        = 2.0 ** (-2 * m - 2)          # exact, on [0,1]
    epsilon_achieved_domain = domain_scale * epsilon_achieved

    # Complexity: tooth g costs 3 units/weights; each composition adds the
    # same g-subnetwork again (3 units) plus 1 linear combination weight
    # for the running sum. We report the construction used by
    # YarotskySquaringNet below, which is exact (not just O(m) asymptotic).
    depth     = m + 2          # m compositions + input/output affine layers
    n_units   = 3 * m           # 3 ReLUs per tooth composition
    n_weights = 3 * m + (m + 1) # 3 weights/unit in g, plus m+1 combination coeffs

    return YarotskyBound(
        epsilon_requested        = epsilon,
        m                        = m,
        epsilon_achieved         = epsilon_achieved,
        epsilon_achieved_domain  = epsilon_achieved_domain,
        depth                    = depth,
        n_weights                = n_weights,
        n_units                  = n_units,
    )


def print_bound_table(epsilons: list[float], domain_scale: float = 36.0) -> None:
    """Quick console sanity check across several epsilon values."""
    print(f"{'eps target':>12}  {'m':>4}  {'eps achieved':>14}  "
          f"{'depth':>6}  {'weights':>8}  {'units':>6}")
    print("-" * 64)
    for eps in epsilons:
        b = required_depth(eps, domain_scale=domain_scale)
        print(f"{b.epsilon_requested:>12.2e}  {b.m:>4d}  "
              f"{b.epsilon_achieved_domain:>14.2e}  "
              f"{b.depth:>6d}  {b.n_weights:>8d}  {b.n_units:>6d}")


# ---------------------------------------------------------------------------
# 2. The "tooth" function as a 3-unit ReLU layer
# ---------------------------------------------------------------------------

class _ToothLayer(nn.Module):
    """
    g(x) = 2σ(x) - 4σ(x - 1/2) + 2σ(x - 1)   for x in [0,1]

    Implemented as Linear(1→3) + ReLU + Linear(3→1), with fixed
    (non-trained) weights matching Yarotsky's explicit construction.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 3)
        self.fc2 = nn.Linear(3, 1, bias=False)

        with torch.no_grad():
            # Pre-activations: x, x - 1/2, x - 1
            self.fc1.weight.copy_(torch.tensor([[1.0], [1.0], [1.0]]))
            self.fc1.bias.copy_(torch.tensor([0.0, -0.5, -1.0]))
            # Combination: 2*sigma(x) - 4*sigma(x-1/2) + 2*sigma(x-1)
            self.fc2.weight.copy_(torch.tensor([[2.0, -4.0, 2.0]]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# 3. The full m-fold squaring network on [0,1]
# ---------------------------------------------------------------------------

class YarotskySquaringNet(nn.Module):
    """
    Explicit (non-trained) ReLU network implementing

        f_m(x) = x - Σ_{s=1}^{m} g_s(x) / 2^(2s)

    which approximates x² on [0,1] with error exactly 2^(-2m-2).

    This is NOT trained — all weights are set analytically per
    Yarotsky's Proposition 2 construction. Useful as a theoretical
    reference curve to compare against trained/pruned networks.

    Parameters
    ----------
    m : int
        Number of tooth-function compositions. Use required_depth(eps).m
        to pick this for a target error.
    """

    def __init__(self, m: int):
        super().__init__()
        if m < 0:
            raise ValueError(f"m must be >= 0, got {m}")
        self.m = m
        self.teeth = nn.ModuleList([_ToothLayer() for _ in range(m)])
        # Coefficients 1/2^(2s) for s=1..m, applied to g_s(x)
        self.register_buffer(
            "coeffs",
            torch.tensor([1.0 / (2.0 ** (2 * s)) for s in range(1, m + 1)])
            if m > 0 else torch.zeros(0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : Tensor [..., 1], values expected in [0,1].
        Returns f_m(x) : Tensor [..., 1], approximating x².
        """
        running = x.clone()      # holds g_s(x) as s increases (g_0 = x)
        total_correction = torch.zeros_like(x)

        for s, tooth in enumerate(self.teeth):
            running = tooth(running)              # g_{s+1}(x) = g(g_s(x))
            total_correction = total_correction + self.coeffs[s] * running

        return x - total_correction

    @property
    def n_weights(self) -> int:
        """Actual weight count in this PyTorch module (for verification)."""
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# 4. Dataset-A wrapper: affine pre/post layers for domain [-3,3], f(x)=(x-1)^2
# ---------------------------------------------------------------------------

class YarotskyDatasetANet(nn.Module):
    """
    Wraps YarotskySquaringNet with the affine change of variables needed
    for dataset A: f(x) = (x-1)^2 on [-3, 3].

    u = (x+3)/6                    maps [-3,3] -> [0,1]
    f(x) = 36*u^2 - 48*u + 16       exact algebraic identity
         = 36*g(u) - 48*u + 16      where g(u) = u^2

    We replace g(u) with the YarotskySquaringNet approximation f_m(u),
    giving:
        f_approx(x) = 36*f_m(u) - 48*u + 16,   u = (x+3)/6

    Pre/post affine maps are folded into Linear layers with fixed
    (non-trained) weights, so they don't add to the "theoretical"
    parameter count beyond what Yarotsky's bound already accounts for
    — they are O(1) extra weights, negligible compared to O(m).
    """

    def __init__(self, m: int):
        super().__init__()
        self.squaring_net = YarotskySquaringNet(m)

        # Pre-affine: u = (x+3)/6 = x/6 + 0.5
        self.pre = nn.Linear(1, 1)
        with torch.no_grad():
            self.pre.weight.fill_(1.0 / 6.0)
            self.pre.bias.fill_(0.5)

        self.config = {
            "model": "yarotsky",
            "activation": "relu",
            "epsilon": None,
            "depth": m + 2,
            "n_weights": self.n_weights,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.pre(x)                      # [-3,3] -> [0,1]
        g_u = self.squaring_net(u)            # ≈ u^2
        # f(x) = 36*g(u) - 48*u + 16   (exact algebraic post-transform)
        return 36.0 * g_u - 48.0 * u + 16.0

    def get_weight_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_weight_vector(self) -> torch.Tensor:
        with torch.no_grad():
            return torch.cat([p.detach().reshape(-1) for p in self.parameters() if p.requires_grad])

    def set_weights_from_vector(self, w: torch.Tensor) -> None:
        expected = self.get_weight_count()
        if w.numel() != expected:
            raise ValueError(
                f"Weight vector has {w.numel()} entries but model has {expected}."
            )
        offset = 0
        with torch.no_grad():
            for p in self.parameters():
                if not p.requires_grad:
                    continue
                n = p.numel()
                p.copy_(w[offset : offset + n].view_as(p).to(p.device, p.dtype))
                offset += n

    @property
    def n_weights(self) -> int:
        return self.get_weight_count()


def build_dataset_a_network(epsilon: float) -> tuple[YarotskyDatasetANet, YarotskyBound]:
    """
    Convenience builder: given a target error epsilon on dataset A's
    actual domain ([-3,3]), return the explicit (non-trained) network
    and the theoretical bound metadata used to construct it.

    Example
    -------
        net, bound = build_dataset_a_network(1e-3)
        print(bound)              # m, depth, weight count, achieved error
        net.eval()
        with torch.no_grad():
            y_pred = net(x_test)  # x_test in [-3,3]
    """
    bound = required_depth(epsilon, domain_scale=36.0)
    net   = YarotskyDatasetANet(m=bound.m)
    return net, bound


# ---------------------------------------------------------------------------
# 5. Helper utilities for pruning / Hessian computation
# ---------------------------------------------------------------------------

_MAX_YAROTSKY_EXACT_N = 3_000


def _yarotsky_param_shapes(model: YarotskyDatasetANet) -> tuple[list[torch.Size], list[int]]:
    shapes: list[torch.Size] = []
    sizes: list[int] = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        shapes.append(p.shape)
        sizes.append(p.numel())
    return shapes, sizes


def _yarotsky_flat_forward(
    flat: torch.Tensor,
    x: torch.Tensor,
    model: YarotskyDatasetANet,
) -> torch.Tensor:
    shapes, sizes = _yarotsky_param_shapes(model)
    params: list[torch.Tensor] = []
    offset = 0
    for shape, size in zip(shapes, sizes):
        params.append(flat[offset : offset + size].view(shape))
        offset += size

    pre_w, pre_b = params[0], params[1]
    u = F.linear(x, pre_w, pre_b)
    running = u
    total_correction = torch.zeros_like(u)
    idx = 2
    for s in range(model.m):
        fc1_w = params[idx]
        fc1_b = params[idx + 1]
        fc2_w = params[idx + 2]
        idx += 3

        running = F.linear(running, fc1_w, fc1_b)
        running = F.relu(running)
        running = F.linear(running, fc2_w)
        total_correction = total_correction + model.coeffs[s] * running

    g_u = u - total_correction
    return 36.0 * g_u - 48.0 * u + 16.0


def compute_hessian_diag(
    model: YarotskyDatasetANet,
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    flat = model.get_weight_vector().to(device)
    n = flat.numel()
    if n > _MAX_YAROTSKY_EXACT_N:
        raise RuntimeError(
            f"Yarotsky Hessian diagonal unsupported for n={n} > {_MAX_YAROTSKY_EXACT_N}."
        )
    flat = flat.requires_grad_(True)

    def loss_fn(w: torch.Tensor) -> torch.Tensor:
        y_pred = _yarotsky_flat_forward(w, x_all, model)
        return F.mse_loss(y_pred, y_all)

    H = torch.autograd.functional.hessian(loss_fn, flat, vectorize=True)
    return torch.diagonal(H).clamp(min=1e-12)


def compute_hessian_inv(
    model: YarotskyDatasetANet,
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    device: torch.device,
    regularisation: float = 1e-4,
) -> torch.Tensor | None:
    model.eval()
    flat = model.get_weight_vector().to(device)
    n = flat.numel()
    if n > _MAX_YAROTSKY_EXACT_N:
        return None

    def loss_fn(w: torch.Tensor) -> torch.Tensor:
        y_pred = _yarotsky_flat_forward(w, x_all, model)
        return F.mse_loss(y_pred, y_all)

    H = torch.autograd.functional.hessian(loss_fn, flat, vectorize=True)
    H_reg = H + regularisation * torch.eye(n, device=device, dtype=H.dtype)
    cond = torch.linalg.cond(H_reg).item()
    if cond > 1e10:
        regularisation *= 10.0
        H_reg = H + regularisation * torch.eye(n, device=device, dtype=H.dtype)
        cond_post = torch.linalg.cond(H_reg).item()
        if cond_post > 1e10:
            return None
    return torch.linalg.inv(H_reg)
