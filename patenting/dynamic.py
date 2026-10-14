# ============================================================================
# dynamic.py  —  faithful dynamic-system embodiment (US 12,124,779 B2, Claim 11)
# ============================================================================
#
# The patent's dynamic embodiment models a system evolving as
#
#       X_{i+1} = F(X_i, P_i) + G_i                                    (eq.16)
#
# with a recurrent modeling block h such that x_{i+1} = h(x_i, p_i) + g_i
# (eq.17).  We model a measurable thermal state x (e.g. a winding temperature)
# as a residual update driven by exogenous signals u:
#
#       x_{t+Δ} = x_t + h(x_t, u_t)
#
# The modeling block h is built by the SAME constructive search used for the
# static case (architecture_search), trained TEACHER-FORCED on one-step pairs —
# which reduces exactly to a static regression, so Claims 1-8 apply unchanged.
# The constructed h is then run FREE-RUNNING over held-out sequences to
# reconstruct whole trajectories from x_0 + the drive (Claims 14-15: applying
# the constructed network to new data to simulate the physical system).
#
# Compression of the state (Claim 9) is OFF here by design: for a low-dim
# thermal state there is nothing to compress — the patent only compresses
# when X_i is of large dimension.  The drive u may carry its own thermal-memory
# features, supplied by the caller.
# ============================================================================

import numpy as np

from patenting.network import DynamicTopologyNetwork
from patenting.architecture_search import architecture_search
from patenting.utils import mse


# ----------------------------------------------------------------------------
# One-step teacher-forced pairs:  [x_t ; u_t]  ->  x_{t+stride} - x_t
# ----------------------------------------------------------------------------

def build_onestep_pairs(sessions, stride=1, subsample=None, seed=0):
    """sessions : list of dicts with
         "x" : (dx, T) state sequence (time order)
         "u" : (du, T) exogenous drive sequence (time order)
    Returns X (dx+du, N), Y (dx, N): the input is the current state stacked
    with the current drive; the target is the state CHANGE over `stride`."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for s in sessions:
        x = s["x"][:, ::stride]
        u = s["u"][:, ::stride]
        if x.shape[1] < 2:
            continue
        xs.append(np.vstack([x[:, :-1], u[:, :-1]]))     # [x_t ; u_t]
        ys.append(x[:, 1:] - x[:, :-1])                  # Δx over stride
    X = np.concatenate(xs, axis=1)
    Y = np.concatenate(ys, axis=1)
    if subsample is not None and X.shape[1] > subsample:
        idx = rng.choice(X.shape[1], subsample, replace=False)
        X, Y = X[:, idx], Y[:, idx]
    return X, Y


# ----------------------------------------------------------------------------
# Learn the modeling block h via the faithful constructive search.
# ----------------------------------------------------------------------------

def learn_dynamics(train_sessions, val_sessions, stride=1, cycles=10,
                   full_train_epochs=400, subsample=8000, seed=0, **search_kw):
    Xtr, Ytr = build_onestep_pairs(train_sessions, stride, subsample, seed)
    Xva, Yva = build_onestep_pairs(val_sessions, stride,
                                   max(1000, subsample // 5), seed + 1)
    dx, du = train_sessions[0]["x"].shape[0], train_sessions[0]["u"].shape[0]

    class _Null:
        def reconstruction_loss(self, X): return 0.0

    init = DynamicTopologyNetwork([dx + du, 1, dx])
    h, history = architecture_search(
        init, _Null(), Xtr, Ytr, Xva, Yva, Xva, Yva,
        cycles=cycles, full_train_epochs=full_train_epochs, **search_kw)
    onestep = mse(h.predict(Xva), Yva)
    return h, {"history": history, "onestep_val_mse": float(onestep),
               "dx": dx, "du": du, "stride": stride}


# ----------------------------------------------------------------------------
# Free-running rollout (the dynamic twin):  x_0 + drive  ->  trajectory
# ----------------------------------------------------------------------------

def rollout(h, session, stride=1, denorm=None):
    """Reconstruct the state trajectory from the true initial state and the
    drive alone. `denorm` (mean, std) maps normalised state back to units."""
    x = session["x"][:, ::stride]
    u = session["u"][:, ::stride]
    T = x.shape[1]
    xhat = np.empty_like(x)
    xhat[:, 0] = x[:, 0]
    for t in range(T - 1):
        inp = np.vstack([xhat[:, t:t + 1], u[:, t:t + 1]])
        xhat[:, t + 1] = xhat[:, t] + h.predict(inp)[:, 0]
    if denorm is not None:
        m, sd = denorm
        xhat = xhat * sd + m
        x = x * sd + m
    err = xhat - x
    return {"x_true": x, "x_hat": xhat,
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "max_abs": float(np.max(np.abs(err)))}


# ----------------------------------------------------------------------------
# Self-test: a 1-D lumped thermal system  x_{t+1} = x_t + a (u_t - x_t)
# (Newton cooling toward a load-driven setpoint) — verifies the twin tracks
# a known dynamical system from x_0 + drive only.
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    def make_session(T=600, a=0.05, seed=0):
        r = np.random.default_rng(seed)
        u = np.cumsum(r.standard_normal(T) * 0.3)[None, :]   # smooth drive
        x = np.zeros((1, T))
        for t in range(T - 1):
            x[0, t + 1] = x[0, t] + a * (u[0, t] - x[0, t])
        return {"x": x, "u": u}

    raw = [make_session(seed=i) for i in range(12)]
    # normalise by pooled train stats (leakage-safe-ish for the self-test)
    allx = np.concatenate([s["x"] for s in raw[:8]], axis=1)
    allu = np.concatenate([s["u"] for s in raw[:8]], axis=1)
    mx, sx = allx.mean(), allx.std()
    mu, su = allu.mean(), allu.std()
    for s in raw:
        s["x"] = (s["x"] - mx) / sx
        s["u"] = (s["u"] - mu) / su
    tr, va, te = raw[:8], raw[8:10], raw[10:]

    h, meta = learn_dynamics(tr, va, stride=1, cycles=8, subsample=4000,
                             n_restarts=4)
    print(f"\nmodeling block h topology : {h.layers}")
    print(f"one-step val MSE          : {meta['onestep_val_mse']:.3e}")
    rolls = [rollout(h, s, denorm=(mx, sx)) for s in te]
    print(f"rollout RMSE (held-out)   : "
          f"{np.mean([r['rmse'] for r in rolls]):.4f} "
          f"(state std {sx:.3f})")
