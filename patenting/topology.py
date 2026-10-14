# ============================================================================
# topology.py
# ============================================================================
#
# PATENT ALIGNMENT NOTES (US 12,124,779 B2)
# ------------------------------------------
#
# Candidate generation (generate_candidates) must cover ALL mutation types
# described in the patent:
#
#   Additive    : ADD_NEURON, ADD_LAYER, ADD_SKIP   (claim 1, para 4 col.13)
#   Subtractive : REMOVE_NEURON, REMOVE_LAYER       (claim 1, para 4 col.13)
#
# For additive mutations new-link weights are initialised to ZERO so that
# J(Γ'', W'') == J(Γ^{n-1}, W^{n-1}*) at initialisation (patent col. 13-14).
#
# For subtractive mutations W'' is the restriction of W^{n-1}* to the
# surviving links (patent col.14 / eq.13).
#
# The sensitivity-based neuron removal (patent: "based on an estimation of
# the variation in the network error") uses Lagrange multipliers to rank
# neurons by their error contribution, removing the least important one.
#
# ============================================================================

import numpy as np

from patenting.lagrangian import calculate_lagrange_multipliers
from patenting.inits_func import get_init, DEFAULT_INIT_FOR_LAYER_TYPE
from patenting.layer_types import LAYER_REGISTRY, forward_layer, jacobian_layer, LAYERS_TYPES
# ============================================================================
# TOPOLOGY EDITOR
# ============================================================================

class TopologyEditor:

    # ======================================================================
    # ADD NEURON  (additive phase)
    # ======================================================================

    @staticmethod
    def add_neuron(net, layer_idx=1):
        """
        Adds one neuron to a hidden layer.

        New incoming/outgoing weights are initialised to ZERO per the patent
        (additive phase: new-link weights start at zero so the Lagrangian
        estimate equals the current error).
        """

        n = net.copy()
        n.layers[layer_idx] += 1

        incoming = n.weights[layer_idx - 1]
        outgoing = n.weights[layer_idx]

        # Xavier/Glorot init for sigmoid: small perturbation so GN fine-tuning
        # can move the new neuron, while keeping the change to J minimal at t=0.
        fan_in  = incoming.shape[1]
        fan_out = outgoing.shape[0]
        new_row = np.random.randn(1, incoming.shape[1]) * 0.01
        new_col = np.random.randn(outgoing.shape[0], 1) * 0.01

        n.weights[layer_idx - 1] = np.vstack([incoming, new_row])
        n.weights[layer_idx]     = np.hstack([outgoing, new_col])
        n.biases[layer_idx - 1]  = np.vstack([
            n.biases[layer_idx - 1],
            np.zeros((1, 1)),
        ])
        # ----------------------------------------------------------------
        # Update skip connections affected by the new neuron
        # ----------------------------------------------------------------
           # Update skip connections that touch this layer
        for target, conns in n.skip_connections.items():
            new_conns = []
            for src, W, b in conns:
                if src == layer_idx:
                # Source grew: add zero column to skip_W
                    new_W = np.hstack([W, np.zeros((W.shape[0], 1))])
                    new_conns.append((src, new_W, b))
                elif target == layer_idx:
                    # Target grew: add zero row to skip_W and skip_b
                    new_W = np.vstack([W, np.zeros((1, W.shape[1]))])
                    new_b = np.vstack([b, np.zeros((1, 1))])
                    new_conns.append((src, new_W, new_b))
                else:
                    new_conns.append((src, W, b))
            n.skip_connections[target] = new_conns
        n.validate()
        return n

    # ======================================================================
    # REMOVE NEURON  (subtractive phase, patent sensitivity pruning)
    # ======================================================================

    @staticmethod
    def remove_neuron(net, X, Y, layer_idx=1):
        """
        Patent-style sensitivity pruning.

        Computes the Lagrange multipliers for all neurons in layer_idx and
        removes the neuron with the LOWEST mean absolute multiplier value,
        i.e. the one that contributes least to the error.

        The surviving weights are an exact restriction of W^{n-1}* to the
        new topology (patent eq. 13 / col.14).
        """
        if net is None:
            return None
        if net.layers[layer_idx] <= 1:
            return None

        n = net.copy()

        acts, pre_acts = n.forward(X)

        lam = calculate_lagrange_multipliers(n, acts, pre_acts, Y)

        neuron_sensitivity = np.mean(np.abs(lam[layer_idx]), axis=1)

        remove_idx = np.argmin(neuron_sensitivity)

        # Restrict weights (subtractive initialisation)
        n.weights[layer_idx - 1] = np.delete(
            n.weights[layer_idx - 1], remove_idx, axis=0
        )
        n.weights[layer_idx] = np.delete(
            n.weights[layer_idx], remove_idx, axis=1
        )
        n.biases[layer_idx - 1] = np.delete(
            n.biases[layer_idx - 1], remove_idx, axis=0
        )

        n.layers[layer_idx] -= 1
        # ----------------------------------------------------------------
        # Update skip connections affected by the new neuron
        # ----------------------------------------------------------------
        updated_skips = {}
        for target, conns in n.skip_connections.items():
            new_conns = []
            for src, skip_W, skip_b in conns:

                if target == layer_idx:
                    skip_W = np.delete(skip_W, remove_idx, axis=0)
                    skip_b = np.delete(skip_b, remove_idx, axis=0)

                if src == layer_idx:
                    skip_W = np.delete(skip_W, remove_idx, axis=1)

                new_conns.append((src, skip_W, skip_b))
            updated_skips[target] = new_conns

        n.skip_connections = updated_skips
        n.validate()
        return n

    # ======================================================================
    # ADD LAYER  (additive phase)
    # ======================================================================

    @staticmethod
    def add_layer(net, position=1, size=4,
              init_method=None, new_layer_type="tanh"):
        n = net.copy()

        W_orig = n.weights[position].copy()
        b_orig = n.biases[position].copy()

        # choose init based on layer type
        if init_method is None:
            init_method = DEFAULT_INIT_FOR_LAYER_TYPE.get(
            new_layer_type, "scaled_svd"   # safe fallback
            )

        init_fn = get_init(init_method)
        W1,W2=init_fn(W_orig, size)

        b1 = np.zeros((size, 1))
        b2 = b_orig.copy()

        # shift skip indices
        new_skip = {}
        for target, conns in n.skip_connections.items():
            new_conns = []
            for src, W, b in conns:
                new_src    = src    + 1 if src    > position else src
                new_target = target + 1 if target > position else target
                new_conns.append((new_src, W, b))
            new_key = target + 1 if target > position else target
            new_skip[new_key] = new_conns
        n.skip_connections = new_skip

        # insert new layer
        n.layers.insert(position + 1, size)

        # layer_types: W1 gets new_layer_type, W2 inherits old type at position
        old_type = n.layer_types[position]
        n.layer_types.insert(position,     new_layer_type)
        n.layer_types.insert(position + 1, old_type)
        # then remove the original (now at position+2)
        del n.layer_types[position + 2]

        del n.weights[position]
        del n.biases[position]
        n.weights.insert(position,     W1)
        n.weights.insert(position + 1, W2)
        n.biases.insert(position,      b1)
        n.biases.insert(position + 1,  b2)

        n.validate()
        return n

    # ======================================================================
    # REMOVE LAYER  (subtractive phase)
    # ======================================================================

    @staticmethod
    def remove_layer(net, layer_idx=1):
        if len(net.layers) <= 3:
            return None

        n = net.copy()

        W_prev = n.weights[layer_idx - 1]
        b_prev = n.biases[layer_idx - 1]
        W_next = n.weights[layer_idx]
        b_next = n.biases[layer_idx]

        merged_W = W_next @ W_prev
        merged_b = W_next @ b_prev + b_next

        # merged layer keeps the outer activation
        merged_type = n.layer_types[layer_idx]
        del n.layer_types[layer_idx]
        del n.layer_types[layer_idx - 1]
        n.layer_types.insert(layer_idx - 1, merged_type)

        # skip connections (unchanged logic)
        new_skip = {}
        for target, conns in n.skip_connections.items():
            new_conns = []
            for src, W, b in conns:
                if src == layer_idx or target == layer_idx:
                    continue
                new_src    = src    - 1 if src    > layer_idx else src
                new_target = target - 1 if target > layer_idx else target
                new_conns.append((new_src, W, b))
            if new_conns:
                new_key = target - 1 if target > layer_idx else target
                new_skip[new_key] = new_conns
        n.skip_connections = new_skip

        del n.layers[layer_idx]
        del n.weights[layer_idx]
        del n.weights[layer_idx - 1]
        del n.biases[layer_idx]
        del n.biases[layer_idx - 1]
        n.weights.insert(layer_idx - 1, merged_W)
        n.biases.insert(layer_idx - 1,  merged_b)

        n.validate()
        return n

    # ======================================================================
    # ADD SKIP CONNECTION  (additive cross-layer connection, Claim 1)
    # ======================================================================

    @staticmethod
    def add_skip_connection(net, source_layer, target_layer):
        """
        Adds a residual-style skip connection from source_layer to
        target_layer (patent Claim 1: "connection link between the input of
        a node of a layer and the output of a node of any preceding layer").

        New weights are initialised to ZERO (additive phase).
        Uses the dict-based skip_connections format of DynamicTopologyNetwork.
        """

        if source_layer >= target_layer:
            raise ValueError("source_layer must precede target_layer")
        if target_layer in net.skip_connections:
            for src, W, b in net.skip_connections[target_layer]:
                if src == source_layer:
                    return net.copy()  # already exists, return unchanged
        n = net.copy()

        source_dim = n.layers[source_layer]
        target_dim = n.layers[target_layer]

        # Small random init (avoids dead-gradient from pure zero init with ReLU)
        skip_W = np.zeros((target_dim, source_dim))
        skip_b = np.zeros((target_dim, 1))

        if target_layer not in n.skip_connections:
            n.skip_connections[target_layer] = []

        n.skip_connections[target_layer].append(
            (source_layer, skip_W, skip_b)
        )

        return n

    # ======================================================================
    # REMOVE SKIP CONNECTION  (subtractive phase)
    # ======================================================================

    @staticmethod
    def remove_skip_connection(net):
        """
        Removes the weakest skip connection (lowest mean |weight|).
        Returns None if there are no skip connections to remove.
        """

        all_connections = []

        for target in net.skip_connections:
            for idx, (src, W, b) in enumerate(
                net.skip_connections[target]
            ):
                score = np.mean(np.abs(W))
                all_connections.append((score, target, idx))

        if not all_connections:
            return None

        n = net.copy()

        _, target, idx = min(all_connections, key=lambda x: x[0])

        del n.skip_connections[target][idx]

        if len(n.skip_connections[target]) == 0:
            del n.skip_connections[target]

        return n

    # ======================================================================
    # GENERATE ALL CANDIDATES  (patent FIG. 3, Claims 1, 8)
    # ======================================================================

    @staticmethod
    def generate_candidates(net, X, Y):
        """
        Generates the full set of topology mutation candidates covering both
        additive and subtractive phases as required by the patent.

        ADD_LAYER candidates are generated at every valid insertion position
        and at multiple widths so the Lagrangian estimator can select the
        best configuration.  Inserting a layer next to a bottleneck neuron
        (e.g. [1,1,4,1]) prevents learning; we therefore propose insertions
        at every position and let the estimator discard bad ones.

        Returns a list of (operation_name, candidate_network) tuples.
        Each candidate is ready for cheap Lagrangian estimation.
        """

        candidates = []
        num_hidden = len(net.layers) - 2   # layers between input and output
        max_hidden_width = max(net.layers[1:-1]) if num_hidden > 0 else 1
        input_dim        = net.layers[0]
        min_useful_width = max(max_hidden_width, input_dim, 4)
        # ------------------------------------------------------------------
        # ADDITIVE: ADD_NEURON
        # Add one neuron to each hidden layer separately as a candidate.
        # (Usually just layer_idx=1 for shallow nets; all layers for deep.)
        # ------------------------------------------------------------------
        # n_weights=0
        # weights=net.weights
        # for i in range(num_hidden):
        #     n_weights+=weights[i].shape[0]
        # n_neurons_set={2 ** i for i in range(n_weights.bit_length())}
        n_neurons_set={1, 2, 4, 8}
        for layer_idx in range(1, len(net.layers) - 1):
            for n in n_neurons_set:
                cand=net.copy()
                for _ in range(n):
                    cand = TopologyEditor.add_neuron(cand, layer_idx=layer_idx)
                label = f"ADD_NEURON{n}" if num_hidden == 1 else f"ADD_NEURON{n}_L{layer_idx}"
                candidates.append((label, cand))

        # ------------------------------------------------------------------
        # ADDITIVE: ADD_LAYER
        # Propose inserting a new hidden layer at every valid position and
        # at two widths (narrow and wide).  The Lagrangian estimator will
        # select whichever combination best reduces the error.
        #
        # Width heuristic: match the widest existing hidden layer (so the
        # new layer is not a bottleneck), and also try double that width.
        # ------------------------------------------------------------------

        seen_layer_configs = set()

        for pos in range(0, len(net.layers) - 1):
            if num_hidden>=3:
                continue
            for layer_type in LAYERS_TYPES:
                for width in sorted({
                min_useful_width
                }):
                    key = (pos, width, layer_type)
                    if key in seen_layer_configs:     
                        continue
                    seen_layer_configs.add(key)

                    prev_width = net.layers[pos]
                    next_width = net.layers[pos + 1] if (pos+1!=len(net.layers) - 1) else np.inf
                    MIN_CONTEXT = max(4, width // 2)

                    if prev_width < MIN_CONTEXT or next_width < MIN_CONTEXT:
                        continue
                    if width < prev_width:
                        if (pos+1<(len(net.layers) - 1)) & (width<next_width):
                            continue
                    if (layer_type=="linear") & (pos==len(net.layers) - 2):
                        continue
                    cand  = TopologyEditor.add_layer(net, position=pos, size=width,new_layer_type=layer_type)
                    label = f"ADD_LAYER_p{pos}_w{width}_l{layer_type}"
                    candidates.append((label, cand))

        # ------------------------------------------------------------------
        # ADDITIVE: ADD_SKIP (only when >= 4 layers, so skip is meaningful)
        # ------------------------------------------------------------------

        if len(net.layers) >= 4:
            candidates.append((
                "ADD_SKIP",
                TopologyEditor.add_skip_connection(net, 0, 2),
            ))

        # ------------------------------------------------------------------
        # SUBTRACTIVE: REMOVE_NEURON from each hidden layer
        # ------------------------------------------------------------------

        for layer_idx in range(1, len(net.layers) - 1):
            for n in [1, 2]:
                removed=net.copy()
                for _ in range(n):
                    removed = TopologyEditor.remove_neuron(removed, X, Y, layer_idx=layer_idx)
                if removed is not None and net.layers[layer_idx]>=3:
                    label = f"REMOVE_NEURON{n}" if num_hidden == 1 else f"REMOVE_NEURON{n}_L{layer_idx}"
                    candidates.append((label, removed))

        # SUBTRACTIVE: REMOVE_LAYER
        for layer_idx in range(1, len(net.layers) - 1):
            removed_layer = TopologyEditor.remove_layer(net, layer_idx=layer_idx)
            if removed_layer is not None:
                candidates.append((f"REMOVE_LAYER_L{layer_idx}", removed_layer))

        # SUBTRACTIVE: REMOVE_SKIP
        if net.skip_connections:
            removed_skip = TopologyEditor.remove_skip_connection(net)
            if removed_skip is not None:
                candidates.append(("REMOVE_SKIP", removed_skip))

        return candidates
