# ============================================================================
# pruning.py
# ============================================================================

import numpy as np

def prune_small_weights(net, X_train, Y_train, X_val, Y_val,
                        threshold=0.01, max_prune_fraction=0.5):
    """
    Post-construction pruning phase.
    Zeros weights below threshold, one layer at a time,
    verifying MSE doesn't degrade beyond tolerance after each step.
    """
    from patenting.utils import mse
    from patenting.lagrangian import calculate_lagrangian

    current     = net.copy()
    base_mse    = mse(current.predict(X_val), Y_val)
    MSE_TOLERANCE = 1.05  # allow 5% MSE degradation

    print(f"\n=== POST-CONSTRUCTION PRUNING ===")
    print(f"Base val MSE: {base_mse:.6f}")

    total_weights  = sum(w.size for w in current.weights)
    pruned_total   = 0

    for layer_idx in range(len(current.weights)):

        W     = current.weights[layer_idx]
        sizes = np.abs(W)

        # rank weights by magnitude, prune smallest first
        flat_order = np.argsort(sizes.flatten())

        pruned_this_layer = 0
        max_prune = int(W.size * max_prune_fraction)

        for flat_i in flat_order:
            row, col = np.unravel_index(flat_i, W.shape)

            # skip if already zero
            if W[row, col] == 0.0:
                continue

            # skip if below threshold
            if abs(W[row, col]) > threshold:
                break  # sorted, so remaining are all larger

            # check neuron won't become fully disconnected
            incoming_active = np.count_nonzero(current.weights[layer_idx][row, :])
            outgoing_active = (
                np.count_nonzero(current.weights[layer_idx + 1][:, row])
                if layer_idx + 1 < len(current.weights)
                else 1
            )
            if incoming_active <= 1 or outgoing_active <= 1:
                continue  # would disconnect neuron entirely

            # try zeroing this weight
            candidate = current.copy()
            candidate.weights[layer_idx][row, col] = 0.0

            new_mse = mse(candidate.predict(X_val), Y_val)

            if new_mse <= base_mse * MSE_TOLERANCE:
                current = candidate
                pruned_this_layer += 1
                pruned_total += 1

                if pruned_this_layer >= max_prune:
                    break
            # if MSE degrades, skip this weight and continue

        remaining = np.count_nonzero(current.weights[layer_idx])
        print(
            f"  Layer {layer_idx}: pruned {pruned_this_layer} weights | "
            f"active={remaining}/{W.size} | "
            f"val_mse={mse(current.predict(X_val), Y_val):.6f}"
        )

    final_mse  = mse(current.predict(X_val), Y_val)
    sparsity   = pruned_total / total_weights

    print(f"Pruned {pruned_total}/{total_weights} weights ({sparsity*100:.1f}% sparse)")
    print(f"Final val MSE after pruning: {final_mse:.6f}")
    if base_mse > 0:
        print(f"MSE change: {(final_mse/base_mse - 1)*100:+.2f}%")
    else:
        print(f"MSE change: base_mse=0 (Δ={final_mse - base_mse:+.3e})")

    return current