# ============================================================================
# tfrd_loader.py  —  build the few-sensors reconstruction dataset
# ============================================================================
#
# Turns generated ADlet layouts into the coordinate-MLP reconstruction task:
#
#       f(x, y, s_1 … s_k)  →  T(x, y)
#
# where (s_1 … s_k) are the sparse sensor readings of the SAME layout. The
# network must reconstruct the full field of layouts it has never seen, using
# only their coordinates and sensor vector — this is the TFR-HSS task.
#
# Split is BY LAYOUT (held-out layouts for test) so the test measures genuine
# generalisation, not interpolation within a memorised field. Normalisation
# uses training-layout statistics only (leakage-safe), mirroring the A/B study.
#
# All arrays follow the patenting convention: X is (input_dim, n_samples),
# Y is (output_dim, n_samples)  — features as rows, samples as columns.
# ============================================================================

import numpy as np


def _rows_for_layouts(coords, fields, sensors, layout_ids, points_per_layout, rng):
    """Stack (x, y, s_1..s_k) -> T rows for the given layout ids.

    Returns X (input_dim, N), Y (1, N).
    """
    n_grid = coords.shape[0]
    cols_X, cols_Y = [], []
    for L in layout_ids:
        T_flat = fields[L].ravel()
        s = sensors[L]                                  # (k,)
        if points_per_layout is None or points_per_layout >= n_grid:
            pts = np.arange(n_grid)
        else:
            pts = rng.choice(n_grid, size=points_per_layout, replace=False)
        xy = coords[pts]                                # (P, 2)
        s_tile = np.tile(s, (pts.size, 1))              # (P, k)
        X = np.concatenate([xy, s_tile], axis=1)        # (P, 2+k)
        cols_X.append(X)
        cols_Y.append(T_flat[pts][:, None])             # (P, 1)
    X = np.concatenate(cols_X, axis=0).T                # (2+k, N)
    Y = np.concatenate(cols_Y, axis=0).T                # (1, N)
    return X, Y


def load_reconstruction_dataset(
    npz_path,
    n_train_layouts=110,
    n_val_layouts=15,
    n_test_layouts=25,
    points_per_layout=48,
    seed=0,
):
    """Build normalised train/val/test arrays for the reconstruction task.

    Returns a dict with normalised column-major X_*/Y_*, the normalisation
    stats, and the raw test layouts (full fields) for reconstruction figures.
    """
    data = np.load(npz_path, allow_pickle=True)
    coords  = data["coords"]      # (grid², 2), already in [-1,1]
    fields  = data["fields"]      # (n_layouts, grid, grid)
    sensors = data["sensors"]     # (n_layouts, k)
    sens_ij = data["sens_ij"]     # (k, 2) sensor (i,j) grid indices

    n_layouts = fields.shape[0]
    need = n_train_layouts + n_val_layouts + n_test_layouts
    if need > n_layouts:
        raise ValueError(f"Requested {need} layouts but only {n_layouts} generated.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_layouts)
    tr_ids = perm[:n_train_layouts]
    va_ids = perm[n_train_layouts:n_train_layouts + n_val_layouts]
    te_ids = perm[n_train_layouts + n_val_layouts:need]

    X_train, Y_train = _rows_for_layouts(coords, fields, sensors, tr_ids,
                                         points_per_layout, rng)
    X_val,   Y_val   = _rows_for_layouts(coords, fields, sensors, va_ids,
                                         points_per_layout, rng)
    X_test,  Y_test  = _rows_for_layouts(coords, fields, sensors, te_ids,
                                         points_per_layout, rng)

    # --- leakage-safe normalisation: training stats only -----------------
    x_mean = X_train.mean(axis=1, keepdims=True)
    x_std  = X_train.std(axis=1,  keepdims=True) + 1e-8
    y_mean = Y_train.mean(axis=1, keepdims=True)
    y_std  = Y_train.std(axis=1,  keepdims=True) + 1e-8

    norm = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}

    def nx(X): return (X - x_mean) / x_std
    def ny(Y): return (Y - y_mean) / y_std

    grid = fields.shape[1]
    test_layouts = [
        {"sensors": sensors[L], "field": fields[L], "coords": coords}
        for L in te_ids
    ]

    return {
        "X_train": nx(X_train), "Y_train": ny(Y_train),
        "X_val":   nx(X_val),   "Y_val":   ny(Y_val),
        "X_test":  nx(X_test),  "Y_test":  ny(Y_test),
        "norm": norm,
        "test_layouts": test_layouts,
        "sens_ij": sens_ij,
        "grid": grid,
        "k": sensors.shape[1],
        "n_train_layouts": int(n_train_layouts),
        "n_test_layouts": int(n_test_layouts),
        "points_per_layout": int(points_per_layout),
    }


def build_field_input(coords, sensors, norm):
    """Normalised full-grid input (input_dim, grid²) for one test layout,
    so a trained net can reconstruct its entire field."""
    n_grid = coords.shape[0]
    s_tile = np.tile(sensors, (n_grid, 1))              # (grid², k)
    X = np.concatenate([coords, s_tile], axis=1).T      # (2+k, grid²)
    return (X - norm["x_mean"]) / norm["x_std"]


def denorm_y(y_norm, norm):
    """Map normalised network output back to Kelvin."""
    return y_norm * norm["y_std"] + norm["y_mean"]
