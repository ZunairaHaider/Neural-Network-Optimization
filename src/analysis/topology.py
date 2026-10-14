import torch
import networkx as nx
import matplotlib.pyplot as plt
import os


def build_network_graph(model):
    G = nx.DiGraph()

    linear_layers = [
        module for module in model.modules()
        if isinstance(module, torch.nn.Linear)
    ]

    layer_sizes = []

    # input layer
    layer_sizes.append(linear_layers[0].in_features)

    # hidden + output layers
    for layer in linear_layers:
        layer_sizes.append(layer.out_features)

    node_positions = {}
    layer_nodes = []

    # ---------------------
    # Create all nodes
    # ---------------------
    for layer_idx, size in enumerate(layer_sizes):
        current_nodes = []

        for neuron_idx in range(size):
            node_name = f"L{layer_idx}_N{neuron_idx}"

            G.add_node(node_name)

            node_positions[node_name] = (
                layer_idx,
                -neuron_idx
            )

            current_nodes.append(node_name)

        layer_nodes.append(current_nodes)

    # ---------------------
    # Create edges
    # ---------------------
    for layer_idx, layer in enumerate(linear_layers):
        weights = layer.weight.data

        for out_idx in range(weights.shape[0]):
            for in_idx in range(weights.shape[1]):

                w = weights[out_idx, in_idx].item()

                if abs(w) > 1e-8:
                    source = layer_nodes[layer_idx][in_idx]
                    target = layer_nodes[layer_idx + 1][out_idx]

                    G.add_edge(
                        source,
                        target,
                        weight=w
                    )

    # ---------------------
    # Remove isolated nodes
    # ---------------------
    #isolated_nodes = list(nx.isolates(G))

    #G.remove_nodes_from(isolated_nodes)
    # ---------------------
# Remove dead hidden neurons
# ---------------------
    G = prune_dead_nodes(G)
    # also remove positions of deleted nodes
    node_positions = {
        k: v for k, v in node_positions.items()
        if k in G.nodes
    }

    return G, node_positions

def plot_network_topology(model, save_path=None, title="Network Topology"):
    G, pos = build_network_graph(model)

    plt.figure(figsize=(10, 6))

    nx.draw(
        G,
        pos,
        with_labels=False,
        node_size=300,
        arrows=False
    )

    n_edges = G.number_of_edges()
    plt.title(f"{title} — {n_edges} nonzero weights")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Topology saved to {save_path} ({n_edges} nonzero weights)")
        plt.close()

    #plt.show()

#----- Helpers for removing dead neurons in plot --------

def prune_dead_nodes(G):
    changed = True

    while changed:
        changed = False
        nodes_to_remove = []

        # determine final layer index
        max_layer = max(
            int(node.split("_")[0][1:])
            for node in G.nodes()
        )

        for node in list(G.nodes()):
            layer = int(node.split("_")[0][1:])

            # keep input layer
            if layer == 0:
                continue

            # keep output layer
            if layer == max_layer:
                continue

            in_deg = G.in_degree(node)
            out_deg = G.out_degree(node)

            # hidden neuron must have both
            if in_deg == 0 or out_deg == 0:
                nodes_to_remove.append(node)

        if nodes_to_remove:
            G.remove_nodes_from(nodes_to_remove)
            changed = True

    return G