# ============================================================================
# utils.py
# ============================================================================

import numpy as np


def mse(a, b):
    if not np.isfinite(a).all():
        return float('nan')
    return float(np.mean((a - b) ** 2))


def normalize_data(X_train, X_test):
    """
    Normalises using training-set statistics only (no data leakage).
    Applied to both train and test to keep the same scale.
    """
    mean = X_train.mean(axis=1, keepdims=True)
    std  = X_train.std(axis=1,  keepdims=True) + 1e-8

    X_train = (X_train - mean) / std
    X_test  = (X_test  - mean) / std

    return X_train, X_test, std

def check_lagrangian_consistency(net, X, Y, label=""):
    """
    Verifies that constraint terms are ≈ 0 for the current network,
    i.e. L ≈ MSE. If |L - MSE| > tol the Lagrangian is broken.
    """
    from patenting.lagrangian import calculate_lagrangian
    from patenting.utils import mse

    L,_  = calculate_lagrangian(net, X, Y)
    mse_val = mse(net.predict(X), Y)
    diff = abs(L - mse_val)

    print(
        f"  [Consistency {label}] "
        f"Lagrangian={L:.8f} | MSE={mse_val:.8f} | "
        f"|L-MSE|={diff:.2e} "
        f"{'✓' if diff < 1e-6 else '✗ BROKEN'}"
    )
    return diff