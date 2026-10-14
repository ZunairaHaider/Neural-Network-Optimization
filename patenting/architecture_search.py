# ============================================================================
# architecture_search.py
# ============================================================================
#
# PATENT-FAITHFUL constructive topology search (US 12,124,779 B2, FIG.3, Claims 1-8).
# ----------------------------------------------------------------------------
#
#   1. Initialise with a minimal topology and train to convergence (GN).
#   2. Each topological-optimisation cycle:
#        a. From the converged parent, compute the frozen activations X* and
#           multipliers Λ* ONCE (parent_state).
#        b. Rank ALL candidate edits by the cheap frozen-multiplier estimate
#           ΔL = Σ_i Λ*_i·(X*_i − φ(W''_i X*_{i-1}))  (eq.13/14) — NO candidate
#           is retrained to rank it (lagrangian_estimate.rank_candidates).
#        c. Select the most-negative ΔL (the patent's "optimises the variation").
#        d. FULLY retrain only the winner with Gauss-Newton (Claim 7).
#        e. Accept/stop on training+validation error (Claim 8): additive steps
#           must reduce train error; subtractive steps are taken when train<val
#           (overfit) and must not worsen val; stop when nothing helps.
#
# This REPLACES the earlier selection that Gauss-Newton mini-trained every
# candidate and ranked by MSE — exactly the per-candidate cost the patent's
# estimate exists to avoid, and which discarded the frozen multipliers.
# ============================================================================

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from patenting.training import train_network
from patenting.utils import mse
from patenting.lagrangian_estimate import rank_candidates
from patenting.pruning import prune_small_weights


def _Jsum(net, X, Y):
    return float(np.sum((net.predict(X) - Y) ** 2))


# ============================================================================
# ARCHITECTURE SEARCH
# ============================================================================

def architecture_search(
    net,
    compression_system,
    X_train, Y_train,
    X_val,   Y_val,
    X_test,  Y_test,
    cycles=15,
    fine_tune_epochs=2,        # frozen-Λ warm-up iterations for the estimate
    full_train_epochs=400,     # Gauss-Newton steps for the winner's full retrain
    n_restarts=8,
    n_restarts_estimate=2,
    rel_stop=0.005,            # stop when best ΔL > -rel_stop·J(train)
    patience=3,
    val_tol=0.02,              # relative val slack on the Claim-8 gate
    eps_accept=0.01,           # parsimony brake: additive must cut train ≥1%
    add_widths=(1,),           # incremental construction: add 1 node at a time
    selection="faithful",      # "faithful" (eq.13/14) — the patent method
    **_ignored,
):
    history = {k: [] for k in (
        "cycle", "mse", "unnormalized_mse", "hidden_neurons", "num_layers",
        "topology", "recon_loss", "operation", "train_mse", "val_mse",
    )}

    # --- initial training (step 100/101) ----------------------------------
    current = train_network(net, X_train, Y_train, epochs=3000,
                            n_restarts=n_restarts)
    J_cur = _Jsum(current, X_train, Y_train)

    print("\n=== ARCHITECTURE SEARCH (faithful US 12,124,779 B2) ===")
    print(f"Initial topology  : {current.layers}")
    print(f"Initial train MSE : {mse(current.predict(X_train), Y_train):.6e}")
    print(f"Initial val   MSE : {mse(current.predict(X_val), Y_val):.6e}\n")

    no_improve = 0
    overfit_hist = []
    OVERFIT_WINDOW = 3

    for cycle in range(cycles):
        # --- Claim-8 overfitting check → restrict to subtractive ----------
        total_hidden = sum(current.layers[1:-1])
        overfit = False
        if total_hidden > 4:
            v = mse(current.predict(X_val), Y_val)
            t = mse(current.predict(X_train), Y_train)
            overfit_hist.append((v - t) / (v + 1e-12) > 0.20)
            if len(overfit_hist) > OVERFIT_WINDOW:
                overfit_hist.pop(0)
            overfit = (len(overfit_hist) == OVERFIT_WINDOW and all(overfit_hist))
        else:
            overfit_hist = []

        # --- rank every candidate by the cheap frozen-multiplier ΔL -------
        _, _, J_cur, ranked = rank_candidates(
            current, X_train, Y_train, seed=cycle,
            allow_remove=True, add_widths=add_widths,
        )
        if overfit:
            ranked = [r for r in ranked if r[0].startswith("REMOVE")]
        if not ranked:
            print(f"Cycle {cycle+1}: no candidates → stopping.")
            break

        print(f"Cycle {cycle+1:2d}: " + " | ".join(
            f"{n}:{d:+.3e}" for n, d, _ in ranked[:6]))

        best_name, best_dL, best_cand = ranked[0]

        # --- stopping: best estimated ΔL not meaningfully negative --------
        if best_dL > -rel_stop * abs(J_cur):
            no_improve += 1
            print(f"  no meaningful candidate (ΔL={best_dL:+.3e}, "
                  f"{no_improve}/{patience})")
            if no_improve >= patience:
                print("  patience exhausted → stopping.")
                break
            continue

        # --- full Gauss-Newton retrain of the winner only (Claim 7) -------
        epochs = full_train_epochs // 2 if best_name.startswith("REMOVE") \
            else full_train_epochs
        retrained = train_network(best_cand, X_train, Y_train, epochs=epochs,
                                  n_restarts=n_restarts_estimate,
                                  tol=1e-3, patience=max(20, fine_tune_epochs))
        J_new = _Jsum(retrained, X_train, Y_train)
        val_new = mse(retrained.predict(X_val), Y_val)
        val_cur = mse(current.predict(X_val), Y_val)
        is_add = best_name.startswith("ADD")

        # Claim-8 acceptance: additive must lower TRAIN error; subtractive may
        # raise train but must not worsen VAL beyond tolerance.
        reject = False
        if is_add and J_new > J_cur * (1.0 - eps_accept):
            # parsimony brake: an added node must EARN its place with a
            # meaningful (≥eps_accept) relative cut in train error.
            reject, why = True, (f"additive gain below brake "
                                 f"(J {J_cur:.4e}→{J_new:.4e})")
        if val_new > val_cur * (1.0 + val_tol):
            reject, why = True, f"val worsened {val_cur:.4e}→{val_new:.4e}"

        if reject:
            no_improve += 1
            print(f"  ✗ reject {best_name}: {why} ({no_improve}/{patience})")
            if no_improve >= patience:
                print("  patience exhausted → stopping.")
                break
            continue

        no_improve = 0
        current, J_cur = retrained, J_new
        test_mse = mse(current.predict(X_test), Y_test)
        recon = compression_system.reconstruction_loss(X_test)
        history["cycle"].append(cycle + 1)
        history["mse"].append(test_mse)
        history["train_mse"].append(mse(current.predict(X_train), Y_train))
        history["val_mse"].append(val_new)
        history["recon_loss"].append(recon)
        history["hidden_neurons"].append(sum(current.layers[1:-1]))
        history["num_layers"].append(len(current.layers))
        history["topology"].append(current.layers.copy())
        history["operation"].append(best_name)
        print(f"  ✓ {best_name:<16} topology={current.layers} "
              f"train={mse(current.predict(X_train),Y_train):.4e} "
              f"val={val_new:.4e} test={test_mse:.4e}")

    print("\nPost-construction pruning...")
    current = prune_small_weights(current, X_train, Y_train, X_val, Y_val,
                                  threshold=0.01, max_prune_fraction=0.5)
    return current, history
