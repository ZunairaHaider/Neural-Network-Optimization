# ============================================================================
# start_topology_robustness.py
# ============================================================================
#
# STARTING-TOPOLOGY robustness for the faithful constructive search.
#
# The patent method is meant to START SMALL and GROW: you hand it a minimal
# network and it constructs the topology the data needs.  A good method should
# therefore land on (roughly) the SAME final architecture regardless of which
# *small* starting topology you give it.  This study probes exactly that — it
# fixes the random seed and varies only the INITIAL topology over a handful of
# small nets:
#
#       [1, 1, 1]       # "1"     : one single-neuron hidden layer ([d,1,output])
#       [1, 1, 1, 1]    # "1-1"   : two  single-neuron hidden layers
#       [1, 1, 1, 1, 1] # "1-1-1" : three single-neuron hidden layers
#
# i.e. we keep every hidden layer one neuron wide and add depth, asking whether
# handing the search a deeper (but still minimal-width) start changes where it
# ends up.  (A net with NO hidden layer, e.g. [1, 1], is not a valid starting
# point: the search grows FROM an existing hidden layer, so it needs at least
# one.  The patent's own minimal start is [d, 1, output].)
#
# The seed is fixed throughout, so any difference in the final architecture is
# attributable to the STARTING topology alone.
#
#   python -m patenting.start_topology_robustness
#
# Writes results/start_topology_robustness.json for the convergence notebook.
# ============================================================================

import os
import sys
import json

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from patenting.dataset_loader      import load_pt_dataset
from patenting.utils               import normalize_data, mse
from patenting.network             import DynamicTopologyNetwork
from patenting.architecture_search import architecture_search

SEED  = 42                      # fixed: isolate the effect of the starting topology
STARTS = [[1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1, 1]]   # "1", "1-1", "1-1-1"
DATASETS = {
    "dataset_A": "data/processed/dataset_A_{}.pt",   # 1-D, noiseless
    "dataset_B": "data/processed/dataset_B_{}.pt",   # 1-D, noisy (noise floor)
}


class _NullCompression:
    def reconstruction_loss(self, X):
        return 0.0


def _resolve(p):
    return os.path.join(PROJECT_ROOT, p)


def _load(train_path, test_path):
    X_tr_full, Y_tr_full = load_pt_dataset(_resolve(train_path))
    X_test,    Y_test    = load_pt_dataset(_resolve(test_path))
    X_tr_full, X_test, _     = normalize_data(X_tr_full, X_test)
    Y_tr_full, Y_test, _     = normalize_data(Y_tr_full, Y_test)
    N       = X_tr_full.shape[1]
    n_val   = max(1, int(N * 0.10))
    n_train = N - n_val
    perm    = np.random.permutation(N)
    return (X_tr_full[:, perm[:n_train]], Y_tr_full[:, perm[:n_train]],
            X_tr_full[:, perm[n_train:]], Y_tr_full[:, perm[n_train:]],
            X_test, Y_test)


def run_one(data, start):
    Xtr, Ytr, Xva, Yva, Xte, Yte = data
    net = DynamicTopologyNetwork(list(start))
    optimised, _ = architecture_search(
        net, _NullCompression(), Xtr, Ytr, Xva, Yva, Xte, Yte, cycles=10)
    return list(optimised.layers), float(mse(optimised.predict(Xte), Yte))


def main():
    results = {}
    for name, tmpl in DATASETS.items():
        print("\n" + "=" * 64)
        print(f"{name}: starting-topology robustness (seed {SEED} fixed)")
        print("=" * 64)
        runs = []
        for start in STARTS:
            np.random.seed(SEED)                       # identical seed each start
            data = _load(tmpl.format("train"), tmpl.format("test"))
            final, tmse = run_one(data, start)
            runs.append({"start": list(start), "final_topology": final,
                         "test_mse": tmse})
            print(f"  start {str(start):16s} -> final {str(final):18s} "
                  f"test_mse={tmse:.3e}")
        finals = [tuple(r["final_topology"]) for r in runs]
        unique = sorted(set(finals), key=finals.index)
        results[name] = {
            "seed": SEED, "starts": STARTS, "runs": runs,
            "unique_final_topologies": [list(t) for t in unique],
            "all_agree": len(unique) == 1,
        }
        print(f"  -> {len(unique)} distinct final topology(ies); "
              f"{'ALL AGREE' if len(unique) == 1 else 'DISAGREE'}")

    out_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "start_topology_robustness.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
