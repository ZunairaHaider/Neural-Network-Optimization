# layer_types.py
import numpy as np

LAYERS_TYPES=["tanh"]

def _tanh_forward(z):
    return np.tanh(z)

def _tanh_jacobian(z):
    return 1.0 - np.tanh(z) ** 2      # elementwise, (dim, batch)

def _relu_forward(z):
    return np.maximum(0.0, z)

def _relu_jacobian(z):
    return (z > 0).astype(float)

def _layernorm_forward(z, eps=1e-5):
    mu    = z.mean(axis=0, keepdims=True)
    sigma = z.std(axis=0, keepdims=True) + eps
    return (z - mu) / sigma

def _layernorm_jacobian(z, eps=1e-5):
    sigma = z.std(axis=0, keepdims=True) + eps
    return np.ones_like(z) / sigma     # diagonal approx, good enough for GN

def _prelu_forward(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def _prelu_jacobian(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)


# Registry: name → (forward_fn, jacobian_fn)
LAYER_REGISTRY = {
    "tanh"      : (_tanh_forward,      _tanh_jacobian),
    "relu"      : (_relu_forward,      _relu_jacobian),
    "layernorm" : (_layernorm_forward,  _layernorm_jacobian),
    "prelu"     : (_prelu_forward,      _prelu_jacobian),
    "linear"    : (lambda z: z,         lambda z: np.ones_like(z)),
}


def forward_layer(layer_type, z):
    fn, _ = LAYER_REGISTRY[layer_type]
    return fn(z)


def jacobian_layer(layer_type, z):
    """
    Returns elementwise derivative d_activation/d_z.
    Shape matches z: (dim, batch).
    Used in lagrangian.py for the backprop chain.
    """
    _, jac = LAYER_REGISTRY[layer_type]
    return jac(z)