# ============================================================================
# lagrangian_estimate.py
# ============================================================================
#
# FAITHFUL cheap error-variation estimate (US 12,124,779 B2, eq. 12-14).
# ------------------------------------------------------------------------
# This module implements the patent's *signature* contribution: ranking
# candidate topology modifications by the change in the Lagrangian, evaluated
# with the PARENT network's FROZEN activations X* and FROZEN multipliers Λ*,
# WITHOUT retraining each candidate.  Derivation (patent p.14-16):
#
#   For W close to W0,   J(Γ, W)  =  L(Γ, W, X_{W0}, Λ_{W0})            (eq.12)
#   so for a candidate topology Γ'' with weights W'':
#
#     ΔL  =  L(Γ'', W'', X*, Λ*) − J(parent)
#         =  Σ_i  Λ*_i · ( X*_i − φ_i( W''_{i-1} · X*_{i-1} ) )         (eq.13/14)
#
#   because the J terms cancel (the output state X*_{nc} is frozen at the
#   parent's value).  Only the layers whose incoming weights changed
#   contribute a non-zero term — the rest are consistent (≈0).  Cost is one
#   matmul at the affected layer, O(layers·N), NOT a candidate forward+GN pass.
#
# Per the patent (col.16): the additive estimate is sharpened by "one or two
# learning iterations on W'' with X and Λ fixed" — a Pontryagin-style update.
# With X*,Λ* frozen the layers DECOUPLE, so this is a tiny per-candidate fit of
# only the new weights, far cheaper than the Gauss-Newton mini-train the old
# evaluate_candidate did (which is exactly the cost this estimate avoids).
#
# Convention: J and the Lagrangian use the SUM-of-squares error (patent eq.4,
# Claim 6) so that λ_nc = 2(out − Y) makes L ≡ J at consistency. Rankings are
# invariant to the global scale; only internal consistency matters.
# ============================================================================

import numpy as np

from patenting.layer_types import forward_layer, jacobian_layer


# ----------------------------------------------------------------------------
# Post-activation adjoint  Λ_i = ∂J/∂X^i  (the patent's multiplier, p.22)
# ----------------------------------------------------------------------------
#
#   Λ_nc = 2 (X^nc − Y)
#   Λ_i  = W_i^T ( φ'(z_{i+1}) ⊙ Λ_{i+1} )       i = nc-1 … 1
#
# This is the adjoint of the *post-activation* state X^i, because the patent's
# constraint is X^i = f_σ(W_{i-1} X^{i-1}).  (The existing
# lagrangian.calculate_lagrange_multipliers instead returns the *pre-activation*
# backprop delta δ_i = φ'(z_i)⊙(W_i^T δ_{i+1}) — a different quantity — which is
# why the frozen-multiplier estimate built on it was unusable. See module head.)

def post_adjoint(net, acts, pre_acts, Y):
    nc = len(net.weights)
    p = [None] * (nc + 1)
    p[nc] = 2.0 * (acts[nc] - Y)
    for i in range(nc - 1, 0, -1):
        # z_{i+1} = pre_acts[i];  φ' of the activation producing layer i+1
        phi_d = jacobian_layer(net.layer_types[i], pre_acts[i])   # (n_{i+1}, N)
        p[i] = net.weights[i].T @ (phi_d * p[i + 1])              # (n_i, N)
    return p


# ----------------------------------------------------------------------------
# Parent frozen state:  X* (activations), Λ* (multipliers), J(parent)
# ----------------------------------------------------------------------------

def parent_state(net, X, Y):
    """Forward+backward pass on the converged parent.

    Returns
    -------
    acts : list, X*_0 .. X*_nc        (acts[i] is the state of layer i)
    lam  : list, Λ*_1 .. Λ*_nc        (lam[i] = ∂J/∂X^i; lam[0]=None)
    J    : float, SUM-of-squares error of the parent
    """
    acts, pre_acts = net.forward(X)
    lam = post_adjoint(net, acts, pre_acts, Y)
    J = float(np.sum((acts[-1] - Y) ** 2))
    return acts, lam, J


def _skip_into(net, acts, target_layer):
    """Sum of skip-connection contributions feeding `target_layer` (or 0)."""
    if target_layer in net.skip_connections:
        z = 0.0
        for src, W, b in net.skip_connections[target_layer]:
            z = z + W @ acts[src] + b
        return z
    return 0.0


# ----------------------------------------------------------------------------
# SUBTRACTIVE estimate (eq.13): remove neuron k of hidden layer L
# ----------------------------------------------------------------------------

def delta_remove_neuron(net, acts, lam, L, k):
    """ΔL for deleting neuron k of layer L, by zeroing its OUTGOING column
    (patent col.14: "setting the removed links to zero"). Exact, no fit."""
    Wl = net.weights[L].copy()        # weights[L]: layer L -> L+1
    Wl[:, k] = 0.0
    z = Wl @ acts[L] + net.biases[L] + _skip_into(net, acts, L + 1)
    pred = forward_layer(net.layer_types[L], z)
    return float(np.sum(lam[L + 1] * (acts[L + 1] - pred)))


def best_remove_neuron(net, acts, lam, L):
    """(k*, ΔL*) for the least-harmful neuron to remove from layer L."""
    deltas = [delta_remove_neuron(net, acts, lam, L, k)
              for k in range(net.layers[L])]
    k = int(np.argmin(deltas))
    return k, deltas[k]


# ----------------------------------------------------------------------------
# ADDITIVE estimate (eq.14): add m neurons to hidden layer L, with the
# patent's "one or two iterations minimizing L at fixed X*, Λ*" warm-up.
# ----------------------------------------------------------------------------

def delta_add_neurons(net, acts, lam, L, m=1, n_warm=2, lr=0.5,
                      init_scale=0.1, seed=None):
    """ΔL for adding m neurons to hidden layer L (1 <= L <= nc-1).

    New neuron states are determined by their own (new) weights from the
    frozen lower activation X*_{L-1}; their layer-L constraint is identically
    zero (consistent + λ_new = 0), so they affect the estimate ONLY through the
    layer L+1 reconstruction.  We minimise that reconstruction penalty over the
    new weights with X*,Λ* frozen — the layers decouple, so this is a tiny
    nonlinear least-squares in (w_in, b_in, w_out) alone.

    n_warm = number of Pontryagin warm-up steps (the patent's "one or two").
    Returns (ΔL, params) where params = (w_in, b_in, w_out).
    """
    rng = np.random.default_rng(seed)
    a_prev = acts[L - 1]                    # X*_{L-1}   (n_{L-1}, N)
    a_next = acts[L + 1]                    # X*_{L+1}   (n_{L+1}, N)
    lam_next = lam[L + 1]                   # Λ*_{L+1}
    type_L = net.layer_types[L - 1]        # activation producing layer L
    type_Lp1 = net.layer_types[L]          # activation producing layer L+1

    n_prev = a_prev.shape[0]
    n_next = a_next.shape[0]

    # frozen parent pre-activation of layer L+1 (everything except the new path)
    z_base = net.weights[L] @ acts[L] + net.biases[L] + _skip_into(net, acts, L + 1)

    # init: small-random incoming (so the new unit carries signal), zero outgoing
    w_in = rng.standard_normal((m, n_prev)) * init_scale
    b_in = np.zeros((m, 1))
    w_out = np.zeros((n_next, m))

    def forward_new():
        u = w_in @ a_prev + b_in           # (m, N)
        a_new = forward_layer(type_L, u)   # (m, N)
        z = z_base + w_out @ a_new         # (n_next, N)
        pred = forward_layer(type_Lp1, z)
        return u, a_new, z, pred

    def dL(pred):
        return float(np.sum(lam_next * (a_next - pred)))

    for _ in range(n_warm):
        u, a_new, z, pred = forward_new()
        # loss = Σ λ_next·(a_next − φ(z));  ∂loss/∂z = −λ_next·φ'(z)
        G = -lam_next * jacobian_layer(type_Lp1, z)        # (n_next, N)
        g_wout = G @ a_new.T                               # (n_next, m)
        # back into the new unit
        g_anew = w_out.T @ G                               # (m, N)
        g_u = g_anew * jacobian_layer(type_L, u)           # (m, N)
        g_win = g_u @ a_prev.T                             # (m, n_prev)
        g_bin = g_u.sum(axis=1, keepdims=True)             # (m, 1)
        # normalised gradient step (bounded, scale-free across candidates)
        for g, p in ((g_wout, w_out), (g_win, w_in), (g_bin, b_in)):
            nrm = np.linalg.norm(g)
            if nrm > 1e-12:
                p -= lr * g / nrm

    _, _, _, pred = forward_new()
    return dL(pred), (w_in, b_in, w_out)


# ----------------------------------------------------------------------------
# SUBTRACTIVE estimate (eq.13): remove hidden layer L (linear-merge restriction)
# ----------------------------------------------------------------------------

def delta_remove_layer(net, acts, lam, L, n_warm=2, lr=0.5):
    """ΔL for deleting hidden layer L by merging W_L @ W_{L-1} (patent: W''=
    restriction of W* to the shrunk topology). Layer L+1 is then fed from layer
    L-1 through the merged map. Because the merge skips a nonlinearity it is a
    large (not first-order) perturbation, so we apply the patent's sanctioned
    "one or two iterations" to the merged weights at frozen X*,Λ* — giving the
    fair best-single-layer reconstruction rather than a spurious raw value."""
    a_prev, a_next = acts[L - 1], acts[L + 1]
    lam_next = lam[L + 1]
    type_next = net.layer_types[L]
    skip = _skip_into(net, acts, L + 1)
    Wm = net.weights[L] @ net.weights[L - 1]
    bm = net.weights[L] @ net.biases[L - 1] + net.biases[L]

    def fwd():
        z = Wm @ a_prev + bm + skip
        return z, forward_layer(type_next, z)

    for _ in range(n_warm):
        z, pred = fwd()
        G = -lam_next * jacobian_layer(type_next, z)
        g_W = G @ a_prev.T
        g_b = G.sum(axis=1, keepdims=True)
        for g, p in ((g_W, Wm), (g_b, bm)):
            nrm = np.linalg.norm(g)
            if nrm > 1e-12:
                p -= lr * g / nrm

    _, pred = fwd()
    return float(np.sum(lam_next * (a_next - pred)))


# ----------------------------------------------------------------------------
# ADDITIVE estimate (eq.14): add a skip connection src S -> target T
# ----------------------------------------------------------------------------

def delta_add_skip(net, acts, lam, S, T, n_warm=2, lr=0.5, seed=None):
    """ΔL for a new connection from layer S to layer T (S<T), warmed up at
    frozen X*,Λ*. Like delta_add_neurons but the 'feature' is the frozen state
    a_S itself (no new activation), fed into T via new weights (zero-init)."""
    rng = np.random.default_rng(seed)
    a_src = acts[S]                                  # (n_S, N)
    a_tgt = acts[T]                                  # (n_T, N)
    lam_t = lam[T]
    type_T = net.layer_types[T - 1]                  # activation producing T
    z_base = net.weights[T - 1] @ acts[T - 1] + net.biases[T - 1] \
        + _skip_into(net, acts, T)
    sW = np.zeros((a_tgt.shape[0], a_src.shape[0]))
    sb = np.zeros((a_tgt.shape[0], 1))

    def fwd():
        z = z_base + sW @ a_src + sb
        return z, forward_layer(type_T, z)

    for _ in range(n_warm):
        z, pred = fwd()
        G = -lam_t * jacobian_layer(type_T, z)
        g_sW = G @ a_src.T
        g_sb = G.sum(axis=1, keepdims=True)
        for g, p in ((g_sW, sW), (g_sb, sb)):
            nrm = np.linalg.norm(g)
            if nrm > 1e-12:
                p -= lr * g / nrm

    _, pred = fwd()
    return float(np.sum(lam_t * (a_tgt - pred))), (S, sW, sb)


# ----------------------------------------------------------------------------
# ADDITIVE estimate (eq.14): insert a function-preserving hidden layer
# ----------------------------------------------------------------------------

def delta_add_layer(net, acts, lam, pos, width, new_type="tanh",
                    n_warm=2, lr=0.5, seed=None):
    """ΔL for inserting a hidden layer of `width` between layers pos and pos+1.

    Init is function-preserving (SVD split of W_pos), so ΔL≈0 at start; the
    inserted layer's weights are then warmed up at frozen X*,Λ*. The downstream
    layer pos+1 is fed ONLY through the inserted layer (the direct W_pos path is
    replaced), so the estimate reads off the pos+1 reconstruction."""
    from patenting.inits_func import get_init, DEFAULT_INIT_FOR_LAYER_TYPE
    a_prev = acts[pos]                               # X*_pos
    a_next = acts[pos + 1]                           # X*_{pos+1}
    lam_next = lam[pos + 1]
    type_next = net.layer_types[pos]                 # activation producing pos+1

    init_fn = get_init(DEFAULT_INIT_FOR_LAYER_TYPE.get(new_type, "scaled_svd"))
    W1, W2 = init_fn(net.weights[pos], width)
    b1 = np.zeros((width, 1))
    b2 = net.biases[pos].copy()

    def fwd():
        a_ins = forward_layer(new_type, W1 @ a_prev + b1)
        z = W2 @ a_ins + b2 + _skip_into(net, acts, pos + 1)
        return a_ins, z, forward_layer(type_next, z)

    for _ in range(n_warm):
        a_ins, z, pred = fwd()
        G = -lam_next * jacobian_layer(type_next, z)          # (n_next, N)
        g_W2 = G @ a_ins.T
        g_b2 = G.sum(axis=1, keepdims=True)
        g_ains = W2.T @ G
        g_zins = g_ains * jacobian_layer(new_type, W1 @ a_prev + b1)
        g_W1 = g_zins @ a_prev.T
        g_b1 = g_zins.sum(axis=1, keepdims=True)
        for g, p in ((g_W2, W2), (g_b2, b2), (g_W1, W1), (g_b1, b1)):
            nrm = np.linalg.norm(g)
            if nrm > 1e-12:
                p -= lr * g / nrm

    _, _, pred = fwd()
    return float(np.sum(lam_next * (a_next - pred))), (W1, b1, W2, b2)


# ============================================================================
# Candidate builders (apply the structural edit AND the warm-up weights, so
# the winner's full retrain starts from the function-improving init the
# estimate found — the patent's "W'' initialised then a few iterations").
# ============================================================================

def _build_remove_neuron(net, L, k):
    n = net.copy()
    n.weights[L - 1] = np.delete(n.weights[L - 1], k, axis=0)
    n.biases[L - 1] = np.delete(n.biases[L - 1], k, axis=0)
    n.weights[L] = np.delete(n.weights[L], k, axis=1)
    for tgt, conns in list(n.skip_connections.items()):
        new = []
        for src, W, b in conns:
            if tgt == L:
                W = np.delete(W, k, axis=0); b = np.delete(b, k, axis=0)
            if src == L:
                W = np.delete(W, k, axis=1)
            new.append((src, W, b))
        n.skip_connections[tgt] = new
    n.layers[L] -= 1
    n.validate()
    return n


def _build_add_neurons(net, L, params):
    w_in, b_in, w_out = params
    m = w_in.shape[0]
    n = net.copy()
    n.weights[L - 1] = np.vstack([n.weights[L - 1], w_in])
    n.biases[L - 1] = np.vstack([n.biases[L - 1], b_in])
    n.weights[L] = np.hstack([n.weights[L], w_out])
    for tgt, conns in list(n.skip_connections.items()):
        new = []
        for src, W, b in conns:
            if src == L:
                W = np.hstack([W, np.zeros((W.shape[0], m))])
            if tgt == L:
                W = np.vstack([W, np.zeros((m, W.shape[1]))])
                b = np.vstack([b, np.zeros((m, 1))])
            new.append((src, W, b))
        n.skip_connections[tgt] = new
    n.layers[L] += m
    n.validate()
    return n


# ============================================================================
# RANK ALL CANDIDATES by the faithful frozen-multiplier ΔL estimate.
# Returns (acts, lam, J, ranked) with ranked = [(name, ΔL, candidate_net), …]
# sorted most-negative first.  No candidate is retrained here.
# ============================================================================

def rank_candidates(net, X, Y, add_widths=(1, 2, 4), max_hidden=3,
                    allow_remove=True, seed=0):
    from patenting.topology import TopologyEditor

    acts, lam, J = parent_state(net, X, Y)
    nc = len(net.weights)
    num_hidden = len(net.layers) - 2
    out = []

    for L in range(1, len(net.layers) - 1):           # hidden layers
        # ADD_NEURON{m}
        for m in add_widths:
            dL, params = delta_add_neurons(net, acts, lam, L, m=m, seed=seed + m)
            name = f"ADD_NEURON{m}" + ("" if num_hidden == 1 else f"_L{L}")
            out.append((name, dL, _build_add_neurons(net, L, params)))
        # REMOVE_NEURON (least-harmful), only if width allows
        if allow_remove and net.layers[L] > 1:
            k, dL = best_remove_neuron(net, acts, lam, L)
            name = "REMOVE_NEURON" + ("" if num_hidden == 1 else f"_L{L}")
            out.append((name, dL, _build_remove_neuron(net, L, k)))
        # REMOVE_LAYER (only if more than one hidden layer)
        if allow_remove and num_hidden > 1:
            dL = delta_remove_layer(net, acts, lam, L)
            cand = TopologyEditor.remove_layer(net, layer_idx=L)
            if cand is not None:
                out.append((f"REMOVE_LAYER_L{L}", dL, cand))

    # ADD_LAYER at each valid position (cap depth)
    if num_hidden < max_hidden:
        width = max(max(net.layers[1:-1]), net.layers[0], 4)
        for pos in range(0, len(net.layers) - 1):
            if net.layers[pos] < 2:        # need context to split
                continue
            dL, (W1, b1, W2, b2) = delta_add_layer(net, acts, lam, pos, width,
                                                   seed=seed)
            cand = TopologyEditor.add_layer(net, position=pos, size=width,
                                            new_layer_type="tanh")
            cand.weights[pos] = W1; cand.biases[pos] = b1
            cand.weights[pos + 1] = W2; cand.biases[pos + 1] = b2
            cand.validate()
            out.append((f"ADD_LAYER_p{pos}_w{width}", dL, cand))

    # ADD_SKIP (only when deep enough to be meaningful)
    if len(net.layers) >= 4:
        dL, (S, sW, sb) = delta_add_skip(net, acts, lam, 0, 2, seed=seed)
        cand = TopologyEditor.add_skip_connection(net, 0, 2)
        if 2 in cand.skip_connections:
            cand.skip_connections[2] = [
                (s, sW.copy(), sb.copy()) if s == 0 else (s, W, b)
                for s, W, b in cand.skip_connections[2]
            ]
        out.append(("ADD_SKIP", dL, cand))

    out.sort(key=lambda r: r[1])
    return acts, lam, J, out


