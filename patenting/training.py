# ============================================================================
# training.py
# ============================================================================
#
# Full Gauss-Newton retraining of a fixed topology (patent Claim 7: retrain the
# selected topology by a rapid descent method — the "zero-memory Gauss-Newton").
# This is the WINNER retrainer used by architecture_search after the faithful
# frozen-multiplier estimate has selected a candidate.
#
# NOTE: the old per-candidate scoring helpers that lived here
# (estimate_error_variation, evaluate_candidate, total_objective, …) have been
# removed — candidate ranking is now the faithful eq.13/14 estimate in
# patenting/lagrangian_estimate.py. The superseded code is in
# archive/patent_search_v1_superseded/.
# ============================================================================

import numpy as np

from patenting.lagrangian import calculate_lagrangian
from patenting.optimizer import gauss_newton_step


# ============================================================================
# FULL RETRAINING  (Gauss-Newton with multi-restart, patent Claim 7)
# ============================================================================

def train_network(
    net,
    X,
    Y,
    epochs=200,
    n_restarts=1,
    tol=1e-4,          # relative change threshold
    patience=20,       # check every N steps
):
    best_net = None
    best_L   = np.inf

    for restart in range(n_restarts):
        candidate = net.copy()

        if restart > 0:
            scale = 0.01 * (10 ** (restart / n_restarts * 3))
            for i in range(len(candidate.weights)):
                candidate.weights[i] = (
                    np.random.randn(*candidate.weights[i].shape) * scale
                )
                candidate.biases[i] = (
                    np.random.randn(*candidate.biases[i].shape) * scale
                )

        # ── per-restart convergence tracking ──────────────────────────
        L_checkpoint, _ = calculate_lagrangian(candidate, X, Y)
        converged    = False

        for epoch in range(epochs):
            try:
                candidate = gauss_newton_step(candidate, X, Y, damping=1e-3)
                pred      = candidate.predict(X)
                if not np.isfinite(pred).all():
                    raise ValueError("diverged")
            except Exception:
                break   # this restart failed, move on

            # ── convergence check every `patience` steps ───────────────
            if (epoch + 1) % patience == 0:
                L_now, _      = calculate_lagrangian(candidate, X, Y)
                rel_change = abs(L_checkpoint - L_now) / (abs(L_checkpoint) + 1e-12)

                if rel_change < tol:
                    if np.isfinite(L_now) and L_now < best_L:
                        best_L   = L_now
                        best_net = candidate.copy()
                    converged = True
                    break

                L_checkpoint = L_now   # move window forward

        # ── if max epochs hit without convergence, still record best ───
        if not converged:
            L_final, _ = calculate_lagrangian(candidate, X, Y)
            if np.isfinite(L_final) and L_final < best_L:
                best_L   = L_final
                best_net = candidate.copy()

    if best_net is None:
        return net.copy()
    return best_net
