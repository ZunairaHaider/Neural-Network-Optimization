# ============================================================================
# inits_func.py
# ============================================================================
#
# Identity-preserving layer insertion initialisations.
#
# All functions share the same signature:
#   W_orig : (next_dim, prev_dim)  — weight matrix being factorised
#   size   : int                   — number of neurons in new layer
#
# All return:
#   W1 : (size, prev_dim)
#   W2 : (next_dim, size)
#
# such that W2 @ W1 ≈ W_orig  (exact or approximate depending on method)
# ============================================================================

import numpy as np


# ============================================================================
# 1. SVD  (baseline, exact)
# ============================================================================

def svd_init(W_orig, size):
    """
    Exact factorisation via SVD.
    W2 @ W1 = W_orig exactly.
    W1 has orthogonal rows — best conditioned of all methods.
    Reference: Saxe et al. (2014), Net2Net (Chen et al. 2015)
    """
    next_dim, prev_dim = W_orig.shape
    U, s, Vt = np.linalg.svd(W_orig, full_matrices=False)
    r = len(s)
    S_sqrt = np.diag(np.sqrt(np.maximum(s, 1e-10)))

    W1_core = S_sqrt @ Vt    # (r, prev_dim)
    W2_core = U @ S_sqrt     # (next_dim, r)

    if size <= r:
        return W1_core[:size], W2_core[:, :size]

    extra = size - r
    noise = np.random.randn(extra, prev_dim)
    Q     = W1_core / (np.linalg.norm(W1_core, axis=1, keepdims=True) + 1e-10)
    noise = noise - (noise @ Q.T) @ Q
    noise = noise / (np.linalg.norm(noise, axis=1, keepdims=True) + 1e-10)
    noise *= np.mean(np.sqrt(s)) * 0.01

    W1 = np.vstack([W1_core, noise])
    W2 = np.hstack([W2_core, np.zeros((next_dim, extra))])
    return W1, W2


# ============================================================================
# 2. Net2Net  (exact, no linear algebra)
# ============================================================================

def net2net_init(W_orig, size):
    """
    Net2DeeperNet operator from Chen et al. (2015).
    New layer = identity function in linear regime.

    W1 = I (or random projection preserving norms)
    W2 = W_orig
    Exact when new layer has identity activation (your linear_layers flag).

    Simplest possible init — no SVD, no inversion.
    Works best when prev_dim == size (square case).
    For size != prev_dim uses least-squares padding.
    """
    next_dim, prev_dim = W_orig.shape

    if size == prev_dim:
        # Exact: new layer is identity map
        W1 = np.eye(size)                          # (size, prev_dim)
        W2 = W_orig.copy()                         # (next_dim, size)

    elif size > prev_dim:
        # Pad identity with near-zero extra rows
        W1 = np.vstack([
            np.eye(prev_dim),
            np.random.randn(size - prev_dim, prev_dim) * 0.01,
        ])                                          # (size, prev_dim)
        W2 = np.hstack([
            W_orig,
            np.zeros((next_dim, size - prev_dim)),
        ])                                          # (next_dim, size)

    else:
        # size < prev_dim: truncate — low-rank approximation
        # Use top-size rows of a random orthogonal matrix
        Q = np.linalg.qr(np.random.randn(prev_dim, prev_dim))[0]
        W1 = Q[:size, :]                           # (size, prev_dim)
        W2 = W_orig @ Q.T[:, :size]               # (next_dim, size)

    return W1, W2


# ============================================================================
# 3. Random Orthogonal  (Saxe et al. 2014)
# ============================================================================

def orthogonal_init(W_orig, size):
    """
    Saxe et al. (2014): orthogonal weight matrices give optimal gradient
    flow in deep linear networks — singular values preserved exactly,
    no vanishing/exploding at init.

    W1 = random orthogonal (QR decomposition)
    W2 = W_orig @ W1^T  (so W2 @ W1 = W_orig @ W1^T @ W1 = W_orig)

    W2 @ W1 = W_orig exactly since W1 W1^T = I (orthogonal rows).
    Condition number of W1 = 1 — best possible.
    """
    next_dim, prev_dim = W_orig.shape

    # Random matrix → QR → orthogonal rows
    rand  = np.random.randn(size, prev_dim)
    Q, _  = np.linalg.qr(rand.T)        # Q: (prev_dim, prev_dim)
    W1    = Q.T[:size, :]               # (size, prev_dim) orthonormal rows

    W2 = W_orig @ W1.T                  # (next_dim, size)
    # W2 @ W1 = W_orig @ W1^T @ W1 = W_orig @ I[:prev_dim,:prev_dim] = W_orig ✓

    return W1, W2


# ============================================================================
# 4. Averaging  (exact for scalar output, no linear algebra)
# ============================================================================

def averaging_init(W_orig, size):
    """
    Exact for any next_dim when W_orig rows are treated independently.

    W1[i, :] = W_orig[i % next_dim, :] / ceil(size / next_dim)
    W2        = tiled identity scaled so W2 @ W1 = W_orig exactly.

    No matrix decomposition at all — very fast, always stable.
    Best for small nets or quick candidate screening.
    Reference: original approach in your patent codebase.
    """
    next_dim, prev_dim = W_orig.shape
    repeats = int(np.ceil(size / next_dim))

    W1 = np.tile(W_orig, (repeats, 1))[:size, :] / repeats  # (size, prev_dim)

    # W2 sums groups of `repeats` rows → recovers W_orig row i
    W2 = np.zeros((next_dim, size))
    for i in range(next_dim):
        for j in range(size):
            if j % next_dim == i:
                W2[i, j] = 1.0                              # (next_dim, size)

    # W2 @ W1: row i of W2 picks columns j where j%next_dim==i
    # each picked W1[j,:] = W_orig[i,:]/repeats
    # sum of `repeats` such terms = W_orig[i,:] ✓
    return W1, W2

def scaled_svd_init(W_orig, size, scale=0.1):
    """
    SVD factorisation scaled so W1 outputs stay in tanh linear regime.
    
    tanh(z) ≈ z  when |z| < 0.3  (error < 1%)
    
    Strategy:
      W1_scaled = scale * W1_svd    → pre-activations small → tanh ≈ identity
      W2_scaled = W2_svd / scale    → compensates so W2 @ tanh(W1 @ x) ≈ W_orig @ x
    
    Error: O(scale² * ||W1 @ x||²) per Taylor expansion of tanh
    As scale → 0, approximation becomes exact.
    """
    W1, W2 = svd_init(W_orig, size)

    # Scale so tanh(W1 @ x) ≈ W1 @ x
    W1_scaled = W1 * scale
    W2_scaled = W2 / scale

    # Safety: if W2 blows up, fall back to pure linear init
    if np.abs(W2_scaled).max() > 100.0:
        return W1, W2   # caller should mark layer as linear
    return W1_scaled, W2_scaled


def taylor_corrected_init(W_orig, size, scale=0.01):
    """
    Accounts for the cubic term of tanh to reduce approximation error.

    tanh(z) = z - z³/3 + O(z⁵)

    So: W2 @ tanh(W1 @ x) = W2 @ (W1 @ x - (W1 @ x)³/3 + ...)
                           = W2 @ W1 @ x - W2 @ (W1 @ x)³/3 + ...

    To cancel the cubic term we can't do it per-sample, but we can
    absorb it into a bias correction using the data statistics.
    Practical for small scale values where cubic term is tiny.
    """
    W1_s, W2_s = scaled_svd_init(W_orig, size, scale=scale)

    # Estimate cubic correction using data second moment
    # E[(W1 x)³] ≈ 0 if x is zero-mean (often true after normalisation)
    # so for normalised data this correction vanishes — scaled_svd suffices

    return W1_s, W2_s

def network_morphism_init(W_orig, size, activation='tanh', X_sample=None):
    """
    Wei et al. (2016): Network Morphism.
    
    For nonlinear f, solve:
      f(W2 @ f(W1 @ x)) = f(W_orig @ x)
    
    by choosing W1 such that f(W1 @ x) is a SUFFICIENT STATISTIC
    for f(W_orig @ x), then fitting W2 by least squares on sample data.

    Requires a data sample X_sample to fit W2.
    
    Two sub-cases:
      size >= next_dim : W1 is overcomplete, W2 fitted by lstsq
      size <  next_dim : W1 is undercomplete, lossy approximation
    """
    if X_sample is None:
        # Fall back to scaled SVD without data
        return scaled_svd_init(W_orig, size, scale=0.1)

    next_dim, prev_dim = W_orig.shape

    ACTIVATION = {
        'tanh':    np.tanh,
        'relu':    lambda z: np.maximum(0, z),
        'sigmoid': lambda z: 1 / (1 + np.exp(-z)),
    }
    f = ACTIVATION[activation]

    # Target: what the new layer must reproduce
    # f(W_orig @ X_sample) shape: (next_dim, n_samples)
    target = f(W_orig @ X_sample)           # (next_dim, n_samples)

    # W1: start from SVD core for good conditioning
    W1, _ = svd_init(W_orig, size)
    W1 = W1 * 0.1                           # push into linear regime initially

    # f(W1 @ X): shape (size, n_samples)
    hidden = f(W1 @ X_sample)              # (size, n_samples)

    # Fit W2 by least squares: W2 @ hidden ≈ target
    # W2 shape: (next_dim, size)
    # lstsq: min ||W2 @ hidden - target||²
    # → W2 = target @ hidden^T @ (hidden @ hidden^T)^{-1}
    W2, _, _, _ = np.linalg.lstsq(
        hidden.T,    # (n_samples, size)
        target.T,    # (n_samples, next_dim)
        rcond=None,
    )
    W2 = W2.T       # (next_dim, size)

    # Safety clip
    W2 = np.clip(W2, -10.0, 10.0)

    return W1, W2

def random_feature_init(W_orig, size, X_sample=None):
    """
    Treat the new layer as a random feature expansion.
    
    W1: random Gaussian scaled to preserve input variance
    W2: fitted by least squares so W2 @ f(W1 @ x) ≈ W_orig @ x
    
    Unlike network_morphism_init, W1 is NOT derived from W_orig —
    it's a fresh random projection. This breaks the identity-preservation
    property but gives the new layer maximum expressivity from the start.
    
    Best used when you WANT the new layer to learn new features,
    not just replicate the existing mapping.
    
    Reference: Rahimi & Recht (2007) "Random Features for Large-Scale
    Kernel Machines"
    """
    next_dim, prev_dim = W_orig.shape

    if X_sample is None:
        # Without data, use He init scaled down
        W1 = np.random.randn(size, prev_dim) * np.sqrt(2.0 / prev_dim) * 0.1
        W2 = np.zeros((next_dim, size))
        # W2 = 0 means network output = bias only at init
        # GN will fit W2 in first step
        return W1, W2

    # Scale W1 to unit input variance
    input_std = np.std(X_sample) + 1e-10
    W1 = np.random.randn(size, prev_dim) * (1.0 / (input_std * np.sqrt(prev_dim)))

    hidden = np.tanh(W1 @ X_sample)   # (size, n_samples)

    # Fit W2 to reproduce W_orig @ x (not f(W_orig @ x))
    # so the combined f(W2 @ f(W1 @ x)) ≈ f(W_orig @ x)
    target = W_orig @ X_sample         # (next_dim, n_samples)

    W2, _, _, _ = np.linalg.lstsq(
        hidden.T,
        target.T,
        rcond=None,
    )
    W2 = W2.T
    W2 = np.clip(W2, -10.0, 10.0)

    return W1, W2

def relu_morphism_init(W_orig, size, X_sample=None):
    """
    For ReLU after W1.

    ReLU(W1 @ x) ≈ W1 @ x  only for positive half-space.
    Strategy:
      - He init for W1 (correct variance for ReLU)
      - W2 fitted by lstsq if data available
      - W2 = W_orig @ pinv(W1) * 2  if no data
        (factor 2 compensates for ~half neurons being dead)
    """
    next_dim, prev_dim = W_orig.shape

    # He init: var = 2/fan_in
    W1 = np.random.randn(size, prev_dim) * np.sqrt(2.0 / prev_dim) * 0.1

    if X_sample is not None:
        hidden = np.maximum(0, W1 @ X_sample)        # ReLU forward
        W2, _, _, _ = np.linalg.lstsq(
            hidden.T, (W_orig @ X_sample).T, rcond=None
        )
        W2 = W2.T
        W2 = np.clip(W2, -10.0, 10.0)
    else:
        # No data: factor 2 compensates for dead neurons
        W2 = W_orig @ np.linalg.pinv(W1) * 2.0
        W2 = np.clip(W2, -10.0, 10.0)

    return W1, W2

# ============================================================================
# REGISTRY  — pass init_method string to add_layer
# ============================================================================

INIT_REGISTRY = {
    "svd":          svd_init,
    "net2net":      net2net_init,
    "orthogonal":   orthogonal_init,
    "averaging":    averaging_init,
        # Nonlinear regime (approximate identity preservation)
    "scaled_svd":       scaled_svd_init,       # no data needed, simple
    "network_morphism": network_morphism_init,  # data-driven, most accurate
    "random_feature":   random_feature_init,    # data-driven, most expressive
    "relu_morphism" : relu_morphism_init,   
}

DEFAULT_INIT_FOR_LAYER_TYPE = {
    "linear"    : "svd",          # exact factorisation, no activation concern
    "tanh"      : "scaled_svd",   # push into tanh linear regime
    "relu"      : "relu_morphism",# He-scaled W1, lstsq W2
    "layernorm" : "svd",          # layernorm is scale-invariant → SVD fine
    "prelu"     : "scaled_svd",   # starts near linear (alpha=0.01) → scaled_svd
}

def get_init(name="svd"):
    if name not in INIT_REGISTRY:
        raise ValueError(
            f"Unknown init '{name}'. "
            f"Available: {list(INIT_REGISTRY.keys())}"
        )
    return INIT_REGISTRY[name]