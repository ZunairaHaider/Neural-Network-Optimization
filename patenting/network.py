# ============================================================================
# network.py
# ============================================================================
#
# PATENT ALIGNMENT NOTES (US 12,124,779 B2)
# ------------------------------------------
#
# The patent describes a feedforward network where:
#   - Each layer is a set of simultaneously calculable nodes (col.11).
#   - A processing node in layer i can receive inputs from ANY preceding
#     layer, not just layer i-1 (Claim 1, col.8 col.2, col.11).
#     This is implemented via skip_connections.
#   - He initialisation gives sensible starting weights for the
#     Gauss-Newton optimiser (rapid local convergence).
#   - The complexity cost provides the parsimony objective (col.8, col.9):
#     the patent explicitly aims to MINIMISE the number of nodes and
#     connections while minimising the network error.
#
# ============================================================================

import numpy as np

from patenting.activations import sigmoid, sigmoid_deriv, tanh
from patenting.lagrangian import calculate_lagrange_multipliers
from patenting.layer_types import LAYER_REGISTRY, forward_layer, jacobian_layer

# ============================================================================
# DYNAMIC TOPOLOGY NETWORK
# ============================================================================

class DynamicTopologyNetwork:

    def __init__(self, layer_sizes, skip_connections=None, layer_types=None):
        """
        Parameters
        ----------
        layer_sizes      : list of ints, e.g. [9, 4, 1]
        skip_connections : dict  {target_layer_idx: [(src_layer_idx, W, b), …]}
                           Allows any preceding layer to connect directly to
                           a later layer (patent Claim 1).
        """

        self.layers = list(layer_sizes)
        n_weights = len(layer_sizes) - 1
        if layer_types is not None:
            assert len(layer_types) == n_weights
            self.layer_types = list(layer_types)
        else:
            # default: tanh hidden layers, linear output
            self.layer_types = ["tanh"] * (n_weights - 1) + ["linear"]

        # Xavier/Glorot initialisation scaled down for sigmoid + Gauss-Newton.
        #
        # Standard Xavier (√2/(fan_in+fan_out)) saturates sigmoid when fan_in
        # is small (e.g. 1D input): pre-activations reach ±1–2 where σ'≈0,
        # collapsing the Jacobian and causing GN to predict the constant mean.
        # Scaling by 0.1 keeps pre-activations in [-0.3, +0.3] at init, where
        # σ'≈0.25 (its maximum), giving GN a full-rank, well-conditioned
        # Jacobian from the first step.  The train_network multi-restart loop
        # then reliably escapes any remaining saddle points.
        SIGMOID_GN_SCALE = 0.1
        self.weights = []
        self.biases  = []

        for i in range(len(layer_sizes) - 1):
            fan_in  = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            # W = (
            #     np.random.randn(fan_out, fan_in)
            #     * np.sqrt(2.0 / (fan_in + fan_out))
            #     * SIGMOID_GN_SCALE
            # )
            W = (
                 np.random.randn(fan_out, fan_in)
                 * np.sqrt(2.0 / (fan_in + fan_out)))
            b = np.zeros((fan_out, 1))
            self.weights.append(W)
            self.biases.append(b)

        # skip_connections: {target_layer: [(src_layer, W, b), …]}
        self.skip_connections = {} if skip_connections is None \
                                else skip_connections

    # ======================================================================
    # FORWARD PASS
    # ======================================================================

    def forward(self, X):
        acts     = [X]
        pre_acts = []

        for i in range(len(self.weights)):
            z = self.weights[i] @ acts[-1] + self.biases[i]

            target_layer = i + 1
            if target_layer in self.skip_connections:
                for src_layer, skip_W, skip_b in self.skip_connections[target_layer]:
                
                    skip_contrib = skip_W @ acts[src_layer] + skip_b
                
                # Shape guard — skip silently if stale
                    if skip_contrib.shape != z.shape:
                        continue   # or raise, depending on strictness
                
                    z = z + skip_contrib   # non in-place avoids broadcast cache issue

            pre_acts.append(z)
            acts.append(forward_layer(self.layer_types[i], z))

        return acts, pre_acts

    # ======================================================================
    # PREDICT
    # ======================================================================

    def predict(self, X):
        acts, _ = self.forward(X)
        return acts[-1]

    # ======================================================================
    # COPY
    # ======================================================================

    def copy(self):
        n = DynamicTopologyNetwork(self.layers.copy(), skip_connections={},layer_types=self.layer_types.copy())
        n.weights = [w.copy() for w in self.weights]
        n.biases  = [b.copy() for b in self.biases]
        for target, conns in self.skip_connections.items():
            n.skip_connections[target] = [
                (src, W.copy(), b.copy()) for src, W, b in conns
            ]

        return n

    # ======================================================================
    # COMPLEXITY COST  (patent parsimony objective, col.8-9)
    # ======================================================================

    def complexity_cost(
        self,
        neuron_weight=0.0001,
        connection_weight=0.0001,
        layer_weight=0.007,
        skip_weight=0.005,
    ):
        """
        Topology complexity penalty ensuring the search favours parsimonious
        networks (patent col.8-9: "a parsimonious neural network, meaning
        where any redundancy is eliminated, optimised for the intended task").

        Components:
          - neuron_weight    × total neuron count
          - connection_weight× total standard weight count
          - layer_weight     × number of layers
          - skip_weight      × total skip-connection weight count
        """

        neurons     = sum(self.layers)
        connections = sum(w.size for w in self.weights)
        layers      = len(self.layers)

        skip_connections = sum(
            W.size
            for conns in self.skip_connections.values()
            for _, W, _ in conns
        )

        return (
            neuron_weight     * neurons
            + connection_weight * connections
            + layer_weight      * layers
            + skip_weight       * skip_connections
        )

    # ======================================================================
    # NEURON SENSITIVITY  (patent: Lagrange-based sensitivity analysis)
    # ======================================================================

    def compute_neuron_loss_sensitivity(self, X, Y, layer_idx):
        """
        Computes the mean absolute Lagrange multiplier for each neuron in
        layer_idx, used to rank neurons by their error contribution.
        Higher sensitivity → more important to keep.
        """

        acts, pre_acts = self.forward(X)
        lam = calculate_lagrange_multipliers(self, acts, pre_acts, Y)
        score = np.linalg.norm(lam[layer_idx], axis=1) * np.var(lam[layer_idx], axis=1)
        return score #np.mean(np.abs(lam[layer_idx]), axis=1)

    # ======================================================================
    # ADD SKIP CONNECTION (instance method)
    # ======================================================================

    def add_skip_connection(self, source_layer, target_layer):
        """
        Adds a zero-initialised skip connection (additive phase).
        """

        if source_layer >= target_layer:
            raise ValueError("source_layer must precede target_layer")

        source_dim = self.layers[source_layer]
        target_dim = self.layers[target_layer]

        skip_W = np.zeros((target_dim, source_dim))
        skip_b = np.zeros((target_dim, 1))

        if target_layer not in self.skip_connections:
            self.skip_connections[target_layer] = []

        self.skip_connections[target_layer].append(
            (source_layer, skip_W, skip_b)
        )

    # ======================================================================
    # REMOVE SKIP CONNECTION (instance method)
    # ======================================================================

    def remove_skip_connection(self):
        """
        Removes the weakest skip connection (lowest mean |weight|).
        """

        all_connections = []

        for target, conns in self.skip_connections.items():
            for idx, (src, W, b) in enumerate(conns):
                all_connections.append((np.mean(np.abs(W)), target, idx))

        if not all_connections:
            return

        _, target, idx = min(all_connections, key=lambda x: x[0])

        del self.skip_connections[target][idx]

        if not self.skip_connections[target]:
            del self.skip_connections[target]

    def validate(self):
        n_w = len(self.layers) - 1
        assert len(self.weights) == n_w
        assert len(self.biases)  == n_w
        assert len(self.layer_types) == n_w, f"layer_types: {len(self.layer_types)} != {n_w}"
        for i in range(len(self.weights)):
            expected_W = (self.layers[i+1], self.layers[i])
            expected_b = (self.layers[i+1], 1)
            if self.weights[i].shape != expected_W:
                raise ValueError(f"W{i}: got {self.weights[i].shape}, expected {expected_W}")
            if self.biases[i].shape != expected_b:
                raise ValueError(f"b{i}: got {self.biases[i].shape}, expected {expected_b}")

    # Validate skip connections
        for target, conns in self.skip_connections.items():
            for src, W, b in conns:
                expected_skip = (self.layers[target], self.layers[src])
                if W.shape != expected_skip:
                    raise ValueError(
                    f"Skip {src}→{target}: W shape {W.shape} "
                    f"!= expected {expected_skip}. "
                    f"Layers: {self.layers}"
                )