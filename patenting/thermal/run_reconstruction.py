# ============================================================================
# run_reconstruction.py  —  constructive search on the TFR-HSS task (ADlet)
# ============================================================================
#
# Drives the patent-faithful constructive topology search on the few-sensors
# temperature-field reconstruction task:
#
#       f(x, y, s_1 … s_k)  →  T(x, y)
#
# Trains across many ADlet layouts, then reconstructs the FULL field of
# held-out layouts from their sensors alone. Reports field MSE / RMSE / hot-spot
# error in Kelvin and saves reconstruction figures.
#
# Run:  python -m patenting.thermal.run_reconstruction
# ============================================================================

import os
import sys

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pickle

from patenting.network import DynamicTopologyNetwork
from patenting.architecture_search import architecture_search
from patenting.visualization import draw_network_architecture
from patenting.utils import mse
from patenting.thermal import tfrd_gen
from patenting.thermal.tfrd_loader import (
    load_reconstruction_dataset, build_field_input, denorm_y,
)


class _NullCompression:
    """The reconstruction task has no compression block; the search only
    calls reconstruction_loss() for logging, so return 0."""
    def reconstruction_loss(self, X):
        return 0.0


def count_params(net):
    """Total and non-zero parameters (pruning zeros weights without removing)."""
    total, nz = 0, 0
    for arr in net.weights + net.biases:
        total += arr.size
        nz += int(np.count_nonzero(arr))
    for conns in net.skip_connections.values():
        for _, W, b in conns:
            total += W.size + b.size
            nz += int(np.count_nonzero(W)) + int(np.count_nonzero(b))
    return total, nz


def evaluate_fields(net, ds):
    """Reconstruct each held-out field from its sensors; return per-layout
    field MSE and hot-spot (max-abs) error in Kelvin."""
    norm = ds["norm"]
    grid = ds["grid"]
    field_mse, hot_err = [], []
    recons = []
    for lay in ds["test_layouts"]:
        X = build_field_input(lay["coords"], lay["sensors"], norm)   # (2+k, grid²)
        pred_K = denorm_y(net.predict(X), norm).reshape(grid, grid)
        true_K = lay["field"]
        field_mse.append(float(np.mean((pred_K - true_K) ** 2)))
        hot_err.append(float(np.max(np.abs(pred_K - true_K))))
        recons.append((true_K, pred_K))
    return np.array(field_mse), np.array(hot_err), recons


def _overlay_sensors(ax, sens_ij):
    """Mark the input sensor locations (the only data the net sees) on a panel.
    Field arrays are indexed [i, j] with imshow(origin='lower') → x=j, y=i."""
    si, sj = sens_ij[:, 0], sens_ij[:, 1]
    ax.scatter(sj, si, s=42, marker="s", facecolors="none",
               edgecolors="lime", linewidths=1.4, label="sensors (input)")


def save_recon_figures(recons, sens_ij, out_dir, n=3):
    os.makedirs(out_dir, exist_ok=True)
    k = sens_ij.shape[0]
    for idx in range(min(n, len(recons))):
        true_K, pred_K = recons[idx]
        err = pred_K - true_K
        rmse = float(np.sqrt(np.mean(err ** 2)))
        maxe = float(np.max(np.abs(err)))
        vmin, vmax = true_K.min(), true_K.max()

        fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
        im0 = ax[0].imshow(true_K, origin="lower", vmin=vmin, vmax=vmax, cmap="inferno")
        ax[0].set_title(f"Ground truth (K)\npeak {true_K.max():.1f} K")
        im1 = ax[1].imshow(pred_K, origin="lower", vmin=vmin, vmax=vmax, cmap="inferno")
        ax[1].set_title(f"Reconstruction from {k} sensors (K)\npeak {pred_K.max():.1f} K")
        im2 = ax[2].imshow(err, origin="lower", cmap="coolwarm")
        ax[2].set_title(f"Error (K)\nRMSE {rmse:.2f} K | max |err| {maxe:.1f} K")
        for a, im in zip(ax, (im0, im1, im2)):
            _overlay_sensors(a, sens_ij)
            a.set_xticks([]); a.set_yticks([])
            fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
        ax[0].legend(loc="upper left", fontsize=8, framealpha=0.7)
        fig.suptitle(f"Held-out layout #{idx}  —  green squares = sensor inputs "
                     f"(the rest of the field is reconstructed)", y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"reconstruction_{idx}.png"),
                    dpi=120, bbox_inches="tight")
        plt.close(fig)


def save_summary_figure(field_mse, hot_err, out_dir):
    """Per-layout RMSE across all held-out layouts + summary statistics."""
    rmse = np.sqrt(field_mse)
    order = np.argsort(rmse)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))

    ax[0].bar(range(len(rmse)), rmse[order], color="steelblue")
    ax[0].axhline(rmse.mean(), color="crimson", ls="--",
                  label=f"mean {rmse.mean():.2f} K")
    ax[0].set_xlabel("held-out layout (sorted)")
    ax[0].set_ylabel("field RMSE (K)")
    ax[0].set_title("Reconstruction RMSE per held-out layout")
    ax[0].legend(fontsize=9)

    ax[1].scatter(rmse, hot_err, color="darkorange", edgecolors="k", s=40)
    ax[1].set_xlabel("field RMSE (K)")
    ax[1].set_ylabel("hot-spot |error| (K)")
    ax[1].set_title("Field error vs. hot-spot error")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "summary_metrics.png"), dpi=120,
                bbox_inches="tight")
    plt.close(fig)


def run(
    n_train_layouts=110,
    n_val_layouts=15,
    n_test_layouts=25,
    points_per_layout=48,
    cycles=8,
    fine_tune_epochs=40,
    full_train_epochs=2000,
    seed=42,
):
    np.random.seed(seed)

    npz = os.path.join(CURRENT_DIR, "tfrd_adlet.npz")
    if not os.path.exists(npz):
        print("Generating ADlet dataset ...")
        tfrd_gen.save(npz, tfrd_gen.generate())

    ds = load_reconstruction_dataset(
        npz,
        n_train_layouts=n_train_layouts,
        n_val_layouts=n_val_layouts,
        n_test_layouts=n_test_layouts,
        points_per_layout=points_per_layout,
        seed=seed,
    )

    input_dim = ds["X_train"].shape[0]
    print("=" * 70)
    print("TFR-HSS reconstruction (ADlet) — constructive topology search")
    print("=" * 70)
    print(f"Input dim         : {input_dim}  (2 coords + {ds['k']} sensors)")
    print(f"Train rows        : {ds['X_train'].shape[1]}  "
          f"({ds['n_train_layouts']} layouts × {ds['points_per_layout']} pts)")
    print(f"Test  layouts     : {ds['n_test_layouts']}  (full-field reconstruction)")
    print(f"Field grid        : {ds['grid']}²")

    # Minimal initial topology (patent col.12: single hidden neuron).
    net = DynamicTopologyNetwork([input_dim, 1, 1])

    optimised, history = architecture_search(
        net,
        _NullCompression(),
        ds["X_train"], ds["Y_train"],
        ds["X_val"],   ds["Y_val"],
        ds["X_test"],  ds["Y_test"],
        cycles=cycles,
        fine_tune_epochs=fine_tune_epochs,
        full_train_epochs=full_train_epochs,
    )

    total_p, nz_p = count_params(optimised)
    field_mse, hot_err, recons = evaluate_fields(optimised, ds)

    out_dir = os.path.join(CURRENT_DIR, "outputs")
    save_recon_figures(recons, ds["sens_ij"], out_dir)
    save_summary_figure(field_mse, hot_err, out_dir)

    draw_network_architecture(
        optimised,
        f"ADlet reconstruction — final topology {optimised.layers}",
        save_path=os.path.join(out_dir, "final_architecture.png"),
    )
    with open(os.path.join(out_dir, "final_net.pkl"), "wb") as fh:
        pickle.dump(optimised, fh)

    rmse = np.sqrt(field_mse)
    k_side = int(round(np.sqrt(ds["k"])))
    sij = ds["sens_ij"]
    lines = [
        "=" * 70,
        "RESULT — TFR-HSS reconstruction (ADlet)",
        "=" * 70,
        f"Sensors (input)   : {ds['k']} = {k_side}×{k_side} uniform grid",
        f"  grid indices i  : {sorted(set(sij[:, 0].tolist()))}  (of 0..{ds['grid']-1})",
        f"  grid indices j  : {sorted(set(sij[:, 1].tolist()))}",
        f"Final topology    : {optimised.layers}  (+{len(optimised.skip_connections)} skip)",
        f"Parameters        : {nz_p} non-zero / {total_p} total",
        f"Held-out layouts  : {ds['n_test_layouts']}",
        f"Field RMSE (K)    : mean {rmse.mean():.3f} | median {np.median(rmse):.3f} "
        f"| min {rmse.min():.3f} | max {rmse.max():.3f}",
        f"Field MSE (K²)    : {field_mse.mean():.4f}",
        f"Hot-spot |err| (K): mean {hot_err.mean():.3f} | median {np.median(hot_err):.3f} "
        f"| max {hot_err.max():.3f}",
        f"Test-point MSE    : {mse(optimised.predict(ds['X_test']), ds['Y_test']):.5e} (normalised)",
    ]
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(os.path.join(out_dir, "metrics.txt"), "w") as fh:
        fh.write(summary + "\n")
        fh.write("\nPer-layout field RMSE (K):\n")
        for i, r in enumerate(rmse):
            fh.write(f"  layout {i:2d}: RMSE {r:6.3f} K | hot-spot |err| {hot_err[i]:6.3f} K\n")
    print(f"\nFigures + metrics saved to: {out_dir}")

    return optimised, history, ds


if __name__ == "__main__":
    run()
