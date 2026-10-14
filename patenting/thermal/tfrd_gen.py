# ============================================================================
# tfrd_gen.py  —  FDM generator for ADlet-style temperature fields
# ============================================================================
#
# Reproduces the ADlet sub-task of TFRD (arXiv:2108.08298) with a lightweight
# finite-difference solver (numpy/scipy only — no FEniCS).
#
# Physics: steady-state heat conduction on the unit square with a volumetric
# heat source and an ALL-DIRICHLET boundary (the field is held at a fixed
# temperature on every edge — this is what "ADlet" means).
#
#     -k ∇²T(x,y) = φ(x,y)        in  (0,1)²
#            T    = T_bc           on  ∂(0,1)²
#
# Each "layout" places a few rectangular heat-generating components at random
# positions with random power densities → one source field φ → one solved
# temperature field T. Sparse sensor readings are sampled on a fixed uniform
# grid, mimicking the monitoring points of the TFR-HSS reconstruction task.
#
# The discrete Laplacian depends only on (grid, boundary condition), so it is
# assembled and LU-factorised ONCE and reused across all layouts — only the
# right-hand side (the source) changes per layout.
# ============================================================================

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


def _build_laplacian(n):
    """5-point -∇² on an n×n grid with Dirichlet rows on the boundary.

    Returns a CSC matrix A such that A @ T_flat = rhs enforces
    -∇²T = φ on the interior and T = T_bc on the boundary.
    Spacing h is folded into the interior rows.
    """
    h = 1.0 / (n - 1)
    inv_h2 = 1.0 / (h * h)

    idx = lambda i, j: i * n + j
    rows, cols, vals = [], [], []

    for i in range(n):
        for j in range(n):
            k = idx(i, j)
            if i == 0 or i == n - 1 or j == 0 or j == n - 1:
                # Dirichlet boundary row: T = rhs (rhs will carry T_bc)
                rows.append(k); cols.append(k); vals.append(1.0)
            else:
                rows.append(k); cols.append(k); vals.append(4.0 * inv_h2)
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rows.append(k); cols.append(idx(i + di, j + dj))
                    vals.append(-1.0 * inv_h2)

    A = sp.csc_matrix((vals, (rows, cols)), shape=(n * n, n * n))
    return A


def _sample_layout(rng, n_components_range, power_range, half_size_range, margin):
    """Sample a list of rectangular components: (cx, cy, hx, hy, power)."""
    m = rng.integers(n_components_range[0], n_components_range[1] + 1)
    comps = []
    for _ in range(m):
        hx = rng.uniform(*half_size_range)
        hy = rng.uniform(*half_size_range)
        cx = rng.uniform(margin + hx, 1.0 - margin - hx)
        cy = rng.uniform(margin + hy, 1.0 - margin - hy)
        power = rng.uniform(*power_range)
        comps.append((cx, cy, hx, hy, power))
    return comps


def _source_field(n, comps):
    """Rasterise the components onto an n×n volumetric source field φ."""
    xs = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    phi = np.zeros((n, n))
    for (cx, cy, hx, hy, power) in comps:
        mask = (np.abs(X - cx) <= hx) & (np.abs(Y - cy) <= hy)
        phi[mask] += power
    return phi


def _sensor_indices(n, k_side):
    """Indices (i, j) of a uniform k_side × k_side sensor grid, inset from the
    boundary (sensors sit on the interior, like real monitoring points)."""
    pos = np.linspace(0, n - 1, k_side + 2)[1:-1]
    pos = np.round(pos).astype(int)
    ii, jj = np.meshgrid(pos, pos, indexing="ij")
    return ii.ravel(), jj.ravel()


def generate(
    n_layouts=150,
    grid=48,
    k_side=4,
    t_bc=298.0,
    n_components_range=(2, 4),
    power_range=(500.0, 3000.0),
    half_size_range=(0.05, 0.12),
    margin=0.06,
    seed=42,
):
    """Generate a set of ADlet layouts.

    Returns a dict with:
      fields   : (n_layouts, grid, grid)  solved temperature fields
      sensors  : (n_layouts, k)           sensor readings (k = k_side²)
      coords   : (grid*grid, 2)           normalised [-1,1] (x,y) per grid node
      sens_ij  : (k, 2)                   sensor (i,j) grid indices
      meta      : generation parameters
    """
    rng = np.random.default_rng(seed)
    n = grid

    A = _build_laplacian(n)
    lu = splu(A)

    si, sj = _sensor_indices(n, k_side)
    sensor_flat = si * n + sj

    # boundary mask (flattened) for assembling the RHS
    bmask = np.zeros((n, n), dtype=bool)
    bmask[0, :] = bmask[-1, :] = bmask[:, 0] = bmask[:, -1] = True
    bmask_flat = bmask.ravel()

    fields = np.empty((n_layouts, n, n))
    sensors = np.empty((n_layouts, sensor_flat.size))

    for L in range(n_layouts):
        comps = _sample_layout(rng, n_components_range, power_range,
                               half_size_range, margin)
        phi = _source_field(n, comps)

        rhs = phi.ravel().copy()
        rhs[bmask_flat] = t_bc          # Dirichlet rows carry the boundary temp

        T = lu.solve(rhs).reshape(n, n)
        fields[L] = T
        sensors[L] = T.ravel()[sensor_flat]

    xs = np.linspace(-1.0, 1.0, n)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    coords = np.stack([gx.ravel(), gy.ravel()], axis=1)

    return {
        "fields": fields,
        "sensors": sensors,
        "coords": coords,
        "sens_ij": np.stack([si, sj], axis=1),
        "meta": {
            "grid": n, "k_side": k_side, "k": int(sensor_flat.size),
            "t_bc": t_bc, "n_layouts": n_layouts, "seed": seed,
        },
    }


def save(path, data):
    np.savez_compressed(
        path,
        fields=data["fields"],
        sensors=data["sensors"],
        coords=data["coords"],
        sens_ij=data["sens_ij"],
        meta=np.array([str(data["meta"])], dtype=object),
    )


if __name__ == "__main__":
    import os
    out = os.path.join(os.path.dirname(__file__), "tfrd_adlet.npz")
    data = generate()
    save(out, data)
    f = data["fields"]
    print(f"Generated {data['meta']['n_layouts']} ADlet layouts "
          f"on {data['meta']['grid']}² grid, {data['meta']['k']} sensors.")
    print(f"Field temperature range: [{f.min():.2f}, {f.max():.2f}] K")
    print(f"Saved → {out}")
