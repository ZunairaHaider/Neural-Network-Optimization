# ============================================================================
# main.py
# ============================================================================
#
# PATENT ALIGNMENT NOTES (US 12,124,779 B2)
# ------------------------------------------
#
# Key changes vs. original:
#
#   1. Validation split  — required for Claim 8 (overfitting detection and
#      stopping criterion).  We carve 10 % of training data as val set.
#
#   2. Compression topology optimisation — Claims 9-11 specify that the
#      compression and decompression blocks are themselves topology-searched.
#      CompressionSystem.build_topology_optimised() is called when
#      USE_TOPOLOGY_COMPRESSION = True.
#
#   3. architecture_search now receives X_val, Y_val so that the Claim 8
#      stopping criterion can be applied inside the search loop.
#
# ============================================================================

import os
import sys

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from patenting.dataset_loader import load_pt_dataset
from patenting.utils          import normalize_data, mse
from patenting.network        import DynamicTopologyNetwork
from patenting.compression    import CompressionSystem
from patenting.architecture_search import architecture_search
from patenting.visualization  import draw_network_architecture, plot_history

# ============================================================================
# CONFIGURATION
# ============================================================================

# Set True to run full topology search on the compression/decompression
# blocks (patent Claims 9-11).  Now enabled by default for patent fidelity.
USE_TOPOLOGY_COMPRESSION = True
NORMALIZE = True
DIMENSION=0
N_TOP= 1
N_HIDDEN_LAYERS=2
OUTPUT_ROOT = os.path.join(CURRENT_DIR, f"outputs_{N_TOP}_{N_TOP}_40")
os.makedirs(OUTPUT_ROOT, exist_ok=True)


# ============================================================================
# TOPOLOGY SEARCH WRAPPER for use inside CompressionSystem
# ============================================================================

def _compression_topology_search_fn(
    net,
    X_train, Y_train,
    X_val,   Y_val,
    cycles=10,
    full_train_epochs=200,
):
    """
    Thin wrapper so CompressionSystem.build_topology_optimised() can call
    architecture_search without needing a compression_system reference
    (reconstruction loss is irrelevant inside the compression blocks).
    """

    class _NullCompression:
        """Dummy compression system that always returns 0 recon loss."""
        def reconstruction_loss(self, X):
            return 0.0

    optimised, history = architecture_search(
        net,
        _NullCompression(),
        X_train, Y_train,
        X_val,   Y_val,
        X_val,   Y_val,   # use val as test (no separate test here)
        cycles=cycles,
        fine_tune_epochs=20,    # GN steps for cheap Lagrangian estimation
        full_train_epochs=full_train_epochs,
    )

    return optimised, history


# ============================================================================
# MAIN DEMO
# ============================================================================

def run_demo():

    np.random.seed(42)

    DATASETS = {
        "dataset_A": {

            "train":
                "CS_ANN/data/processed/dataset_A_train.pt",

            "test":
                "CS_ANN/data/processed/dataset_A_test.pt",
        },
        "dataset_B": {

            "train":
                "CS_ANN/data/processed/dataset_B_train.pt",

            "test":
                "CS_ANN/data/processed/dataset_B_test.pt",
        },
                        "dataset_A_2d": {

            "train":
                "CS_ANN/data/processed/dataset_A_2d_train.pt",

            "test":
                "CS_ANN/data/processed/dataset_A_2d_test.pt",
        },
        "dataset_B_2d": {

            "train":
                "CS_ANN/data/processed/dataset_B_2d_train.pt",

            "test":
                "CS_ANN/data/processed/dataset_B_2d_test.pt",
        },
                "dataset_A_3d": {

            "train":
                "CS_ANN/data/processed/dataset_A_3d_train.pt",

            "test":
                "CS_ANN/data/processed/dataset_A_3d_test.pt",
        },
        "dataset_B_3d": {

            "train":
                "CS_ANN/data/processed/dataset_B_3d_train.pt",

            "test":
                "CS_ANN/data/processed/dataset_B_3d_test.pt",
        },

    }
    if DIMENSION>1:
        DATASETS={dataset_name : dataset for dataset_name, dataset in DATASETS.items() if dataset_name.endswith(f"_{DIMENSION}d")}
    for dataset_name, paths in DATASETS.items():


        print("\n" + "=" * 70)
        print(f"RUNNING {dataset_name}")
        print("=" * 70)

        dataset_output_dir = os.path.join(OUTPUT_ROOT, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)

        # ==================================================================
        # LOAD & NORMALISE
        # ==================================================================

        def _resolve(p):
            # paths are written as "CS_ANN/data/..."; resolve relative to the
            # repo root regardless of the current working directory.
            if p.startswith("CS_ANN/"):
                p = p[len("CS_ANN/"):]
            return os.path.join(PROJECT_ROOT, p)

        X_train_full, Y_train_full = load_pt_dataset(_resolve(paths["train"]))
        X_test,       Y_test       = load_pt_dataset(_resolve(paths["test"]))
        if NORMALIZE:
            X_train_full, X_test, std_x = normalize_data(X_train_full, X_test)
            Y_train_full, Y_test, std_y = normalize_data(Y_train_full, Y_test)

        # ==================================================================
        # VALIDATION SPLIT  (patent Claim 8: requires validation data)
        # Carve 10 % of the training set as a validation set.
        # ==================================================================

        N       = X_train_full.shape[1]
        n_val   = max(1, int(N * 0.10))
        n_train = N - n_val

        # Shuffle before splitting
        perm    = np.random.permutation(N)
        X_train = X_train_full[:, perm[:n_train]]
        Y_train = Y_train_full[:, perm[:n_train]]
        X_val   = X_train_full[:, perm[n_train:]]
        Y_val   = Y_train_full[:, perm[n_train:]]

        print(f"\nTrain samples : {n_train}")
        print(f"Val   samples : {n_val}")
        print(f"Test  samples : {X_test.shape[1]}")
        print(f"Input  dim    : {X_train.shape[0]}")
        print(f"Output dim    : {Y_train.shape[0]}")

        # ==================================================================
        # INITIAL NETWORK  (minimal topology: input → 1 hidden → output)
        # Patent col.12: "a possibly minimal initial topology, comprising
        # a single hidden layer comprising a single neuron"
        # ==================================================================

        # Patent col.12: "a possibly minimal initial topology, comprising a
        # single hidden layer comprising a single neuron." The constructive
        # search grows from here; it must NOT be pre-seeded with extra capacity.
        net = DynamicTopologyNetwork([
            X_train.shape[0],
            1,
            Y_train.shape[0],
        ])

        print(f"\nInitial topology: {net.layers}")

        draw_network_architecture(
            net,
            f"{dataset_name} Initial Architecture",
            save_path=os.path.join(dataset_output_dir, "initial_architecture.png"),
        )

        # ==================================================================
        # COMPRESSION SYSTEM
        # Patent Claims 9-11: compression/decompression blocks are built
        # with their own topology optimisation, then combined with the
        # modeling block.
        #
        # Compression is only meaningful when input_dim is large enough to
        # have a latent space strictly smaller than the input.  When
        # input_dim <= MIN_DIM_FOR_COMPRESSION we skip topology compression
        # and use the fast linear autoencoder baseline instead, since a
        # 1→6→1 "compression" is degenerate and wastes significant time.
        # ==================================================================

        MIN_DIM_FOR_COMPRESSION = 4
        input_dim = X_train.shape[0]

        # Latent dim must be strictly smaller than input to be meaningful
        latent_dim = max(1, min(6, input_dim - 1)) if input_dim > 1 else 1
        use_compression = USE_TOPOLOGY_COMPRESSION and (input_dim >= MIN_DIM_FOR_COMPRESSION)

        compression_system = CompressionSystem(
            input_dim=input_dim,
            latent_dim=latent_dim,
        )

        if use_compression:
            # Full patent-faithful path: topology search on C and D blocks
            compression_system.build_topology_optimised(
                X_train,
                topology_search_fn=_compression_topology_search_fn,
                val_split=0.1,
                initial_train_epochs=500,
                search_cycles=8,
                full_train_epochs=200,
            )
        else:
            if USE_TOPOLOGY_COMPRESSION and input_dim < MIN_DIM_FOR_COMPRESSION:
                print(
                    f"\n[Compression] Skipping topology search: input_dim={input_dim} "
                    f"< {MIN_DIM_FOR_COMPRESSION} — compression is degenerate at this "
                    f"dimensionality. Using linear autoencoder baseline."
                )
            # Fast path: simple gradient-descent autoencoder (baseline)
            compression_system.train(
                X_train,
                epochs=10000,
                lr=0.0003,
            )

        print(
            f"\nCompression recon loss: "
            f"{compression_system.reconstruction_loss(X_val):.6f}"
        )

        # ==================================================================
        # ARCHITECTURE SEARCH  (patent FIG. 3, Claims 1-8)
        # ==================================================================

        optimised, history = architecture_search(
            net,
            compression_system,
            X_train,
            Y_train,
            X_val,
            Y_val,
            X_test,
            Y_test,
            cycles=10,
        )

        # ==================================================================
        # OUTPUTS
        # ==================================================================

        draw_network_architecture(
            optimised,
            f"{dataset_name} Final Architecture",
            save_path=os.path.join(dataset_output_dir, "final_architecture.png"),
        )

        plot_history(history, output_dir=dataset_output_dir)

        final_mse = mse(optimised.predict(X_test), Y_test)
        if NORMALIZE:
            unnormalized_mse = mse(optimised.predict(X_test) * std_y, Y_test * std_y)

        print(f"\nFinal topology : {optimised.layers}")
        print(f"Final Test MSE : {final_mse:.4e}")
        if NORMALIZE:
            print(f"Final unnormalized Test MSE : {unnormalized_mse:.4e}")
        # Save metrics
        metrics_path = os.path.join(dataset_output_dir, "metrics.txt")
        with open(metrics_path, "w") as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Final Topology: {optimised.layers}\n")
            f.write(f"Final Test MSE: {final_mse}\n")
            if history["recon_loss"]:
                f.write(f"Final Recon Loss: {history['recon_loss'][-1]}\n")
            if history["val_mse"]:
                f.write(f"Final Val  MSE:  {history['val_mse'][-1]}\n")
            if NORMALIZE:
                f.write(f"Final Unnormalized Test MSE: {unnormalized_mse}\n")
        print(f"\nOutputs saved to: {dataset_output_dir}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_demo()
