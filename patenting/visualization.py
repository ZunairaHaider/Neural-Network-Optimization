# ============================================================================
# visualization.py
# ============================================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os


# ============================================================================
# NETWORK ARCHITECTURE DRAWING
# ============================================================================

def draw_network_architecture(net, title, save_path=None):

    G   = nx.DiGraph()
    pos = {}
    layers = []

    for layer_idx, layer_size in enumerate(net.layers):
        current_layer = []
        y_offset = -(layer_size - 1) / 2
        for neuron_idx in range(layer_size):
            node = f"L{layer_idx}N{neuron_idx}"
            G.add_node(node)
            pos[node] = (layer_idx, y_offset + neuron_idx)
            current_layer.append(node)
        layers.append(current_layer)

    # Standard edges
    for i in range(len(layers) - 1):
        for src in layers[i]:
            for dst in layers[i + 1]:
                G.add_edge(src, dst)

    # Skip connection edges
    for target_layer, conns in net.skip_connections.items():
        for src_layer, _, _ in conns:
            for src in layers[src_layer]:
                for dst in layers[target_layer]:
                    G.add_edge(src, dst)

    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, node_size=300, with_labels=False, arrows=False)
    plt.title(title)
    plt.axis("off")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.close()


# ============================================================================
# HISTORY PLOTS
# ============================================================================

def plot_history(history, output_dir=None):

    cycles = history["cycle"]
    if not cycles:
        return

    def _save(name):
        if output_dir:
            plt.savefig(os.path.join(output_dir, name))
        plt.close()

    # MSE (test, train, val)
    plt.figure(figsize=(10, 5))
    plt.plot(cycles, history["mse"],       marker="o", label="Test MSE")
    if history.get("train_mse"):
        plt.plot(cycles, history["train_mse"], marker="s", label="Train MSE")
    if history.get("val_mse"):
        plt.plot(cycles, history["val_mse"],   marker="^", label="Val MSE")
    plt.title("MSE over Topology Cycles")
    plt.xlabel("Cycle")
    plt.ylabel("MSE (log scale)")
    plt.yscale("log")  
    plt.legend()
    plt.grid(True, which="both")  
    _save("mse_plot.png")

    # Reconstruction loss
    plt.figure(figsize=(10, 5))
    plt.plot(cycles, history["recon_loss"], marker="o")
    plt.title("Compression Reconstruction Loss")
    plt.xlabel("Cycle")
    plt.grid(True)
    _save("recon_plot.png")

    # Hidden neurons
    plt.figure(figsize=(10, 5))
    plt.plot(cycles, history["hidden_neurons"], marker="o")
    plt.title("Hidden Neuron Count")
    plt.xlabel("Cycle")
    plt.grid(True)
    _save("neuron_plot.png")

    # Layer count
    plt.figure(figsize=(10, 5))
    plt.plot(cycles, history["num_layers"], marker="o")
    plt.title("Number of Layers")
    plt.xlabel("Cycle")
    plt.grid(True)
    _save("layer_plot.png")

    # Operations performed
    if history.get("operation"):
        plt.figure(figsize=(10, 3))
        plt.scatter(cycles, [0] * len(cycles), c="steelblue", zorder=3)
        for x, op in zip(cycles, history["operation"]):
            plt.text(x, 0.02, op, rotation=45, ha="center", fontsize=7)
        plt.title("Topology Operations per Cycle")
        plt.yticks([])
        plt.xlabel("Cycle")
        plt.grid(axis="x")
        _save("operations_plot.png")

    # Save raw history
    if output_dir:
        np.save(
            os.path.join(output_dir, "history.npy"),
            history,
            allow_pickle=True,
        )
