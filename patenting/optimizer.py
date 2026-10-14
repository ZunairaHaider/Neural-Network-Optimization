# ============================================================================
# optimizer.py
# ============================================================================
#
# PATENT ALIGNMENT NOTES (US 12,124,779 B2)
# ------------------------------------------
#
# The patent explicitly requires the "zero-memory Gauss-Newton method" for
# weight updates (col.13, col.16 eq.15, Claims 7, 8).  The Gauss-Newton
# method combines gradient backpropagation with forward propagation to
# achieve rapid local convergence, avoiding the slow convergence of plain
# gradient descent.  Levenberg-Marquardt damping (damping * I) is added for
# numerical stability when the approximate Hessian is near-singular.
#
# ============================================================================

import numpy as np


# ============================================================================
# FLATTEN PARAMETERS
# ============================================================================

def flatten_weights(net):
    flat   = []
    shapes = []

    for W in net.weights:
        shapes.append(('W', W.shape))
        flat.append(W.flatten())

    for b in net.biases:
        shapes.append(('b', b.shape))
        flat.append(b.flatten())

    for target, conns in net.skip_connections.items():
        for idx, (src, W, b) in enumerate(conns):
            shapes.append(('skip_W', W.shape, target, idx, src))
            flat.append(W.flatten())
            shapes.append(('skip_b', b.shape, target, idx, src))
            flat.append(b.flatten())

    return np.concatenate(flat), shapes


# ============================================================================
# RESTORE PARAMETERS
# ============================================================================

def restore_weights(net, flat_vector, shapes):
    idx       = 0
    w_idx     = 0
    b_idx     = 0

    for meta in shapes:
        kind  = meta[0]
        shape = meta[1]
        size  = np.prod(shape)
        chunk = flat_vector[idx : idx + size].reshape(shape)
        idx  += size

        if kind == 'W':
            net.weights[w_idx] = chunk
            w_idx += 1

        elif kind == 'b':
            net.biases[b_idx] = chunk
            b_idx += 1

        elif kind == 'skip_W':
            _, _, target, conn_idx, src = meta
            src_old, _, b_old = net.skip_connections[target][conn_idx]
            net.skip_connections[target][conn_idx] = (src_old, chunk, b_old)

        elif kind == 'skip_b':
            _, _, target, conn_idx, src = meta
            src_old, W_cur, _ = net.skip_connections[target][conn_idx]
            net.skip_connections[target][conn_idx] = (src_old, W_cur, chunk)


# ============================================================================
# APPROXIMATE JACOBIAN  (finite-difference)
# ============================================================================

def approximate_jacobian(net, X, Y, epsilon=1e-5):
    """
    Numerical Jacobian ∂r/∂w where r = (f(X) - Y).flatten().
    Used to build the approximate Hessian J^T J for Gauss-Newton.
    """

    params, shapes = flatten_weights(net)

    base_pred = net.predict(X)
    residual  = (base_pred - Y).flatten()

    J = np.zeros((residual.size, params.size))

    for i in range(params.size):
        perturbed    = params.copy()
        perturbed[i] += epsilon
        restore_weights(net, perturbed, shapes)
        residual_eps = (net.predict(X) - Y).flatten()
        J[:, i] = (residual_eps - residual) / epsilon

    restore_weights(net, params, shapes)

    return J, residual

# ============================================================================
# JACOBIAN Computation  (anatycal version)
# ============================================================================

def analytical_jacobian(net, X, Y):
    """Jacobian ∂r/∂w of the flattened residual r = (f(X) - Y).

    Supports arbitrary output dimension m (= net.layers[-1]). The residual is
    flattened output-major: row (o, n) = o*N + n, matching ``(f-Y).flatten()``.
    Per-layer deltas carry an explicit output axis ``(layer_size, m, N)``; for
    m == 1 this reduces exactly to the original scalar-output construction.
    """
    from patenting.layer_types import jacobian_layer

    acts, pre_acts = net.forward(X)
    residual = (acts[-1] - Y).flatten()
    N = X.shape[1]
    m = net.layers[-1]
    params, shapes = flatten_weights(net)
    J = np.zeros((m * N, params.size))

    num_layers = len(net.weights)
    deltas = [None] * num_layers

    # Output-layer delta: ∂(output o)/∂z_last = deriv_last · I[o, j].
    # Output activation is typically linear (deriv = 1); apply it for safety.
    deriv_last = jacobian_layer(net.layer_types[-1], pre_acts[-1])  # (m, N)
    eye = np.eye(m)                                                 # (m, m)
    deltas[-1] = eye[:, :, None] * deriv_last[:, None, :]          # (m, m, N)

    # Backprop deltas (extra output axis is broadcast through unchanged).
    for i in reversed(range(num_layers - 1)):
        deriv = jacobian_layer(net.layer_types[i], pre_acts[i])    # (size_i, N)
        # (size_i, size_{i+1}) @ (size_{i+1}, m, N) -> (size_i, m, N)
        propagated = np.einsum(
            "ji,jon->ion", net.weights[i + 1], deltas[i + 1]
        )
        deltas[i] = propagated * deriv[:, None, :]

    col = 0

    # Standard weights: J_W[(o,n),(r,c)] = deltas[i][r,o,n] * acts[i][c,n]
    for i in range(num_layers):
        n_out, n_in = net.weights[i].shape
        J_W = np.einsum(
            "ron,cn->onrc", deltas[i], acts[i]
        ).reshape(m * N, n_out * n_in)
        J[:, col:col + n_out * n_in] = J_W
        col += n_out * n_in

    # Standard biases: J_b[(o,n),r] = deltas[i][r,o,n]
    for i in range(num_layers):
        n_out = net.biases[i].shape[0]
        J[:, col:col + n_out] = np.einsum("ron->onr", deltas[i]).reshape(
            m * N, n_out
        )
        col += n_out

    for target, conns in net.skip_connections.items():
        for idx, (src, W, b) in enumerate(conns):
            n_out, n_in = W.shape
            target_w_idx = target - 1
            J_W = np.einsum(
                "ron,cn->onrc", deltas[target_w_idx], acts[src]
            ).reshape(m * N, n_out * n_in)
            J[:, col:col + n_out * n_in] = J_W
            col += n_out * n_in

            J[:, col:col + n_out] = np.einsum(
                "ron->onr", deltas[target_w_idx]
            ).reshape(m * N, n_out)
            col += n_out

    return J, residual

# ============================================================================
# GAUSS-NEWTON STEP  (patent: "zero-memory Gauss-Newton method", col.13/16)
# ============================================================================

def gauss_newton_step(net, X, Y, damping=1e-3):
    """
    One damped Gauss-Newton update (Levenberg-Marquardt stabilisation).

    H  = J^T J + λI     (approximate Hessian with damping)
    g  = J^T r           (gradient)
    δw = H^{-1} g        (Newton step)
    W  ← W - δw
    """

    params, shapes = flatten_weights(net)

    #J, residual = approximate_jacobian(net, X, Y)
    J, residual = analytical_jacobian(net, X, Y)
    damping = max(damping, 1e-2 * np.mean(np.diag(J.T @ J)))
    H = J.T @ J + damping * np.eye(params.size)
    g  = J.T @ residual
    delta = np.linalg.solve(H, g)
    max_step = 1.0

    norm = np.linalg.norm(delta)
    if norm > max_step:
        delta = delta * (max_step / norm)
    
    updated=net.copy()
    restore_weights(updated, params - delta, shapes)

    return updated
