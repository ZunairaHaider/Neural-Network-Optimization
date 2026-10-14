# ============================================================================
# validate_estimate.py
# ============================================================================
# Proves the faithful cheap ΔL estimate (lagrangian_estimate.py) ranks
# candidate topology edits the SAME way a full Gauss-Newton retrain would —
# i.e. that the patent's signature estimate (eq.13/14) is a valid selection
# signal, WITHOUT retraining every candidate.
#
# For a converged parent net on dataset A/B we compare, per candidate:
#   cheap ΔL   — frozen-multiplier estimate (no candidate retrain)
#   true ΔJ    — SUM-of-squares train error change after FULL GN retrain
# and report Spearman rank correlation + whether the cheap argmin matches.
#
# Run:  python -m patenting.validate_estimate
# ============================================================================

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from patenting.dataset_loader import load_pt_dataset
from patenting.utils import normalize_data
from patenting.network import DynamicTopologyNetwork
from patenting.training import train_network
from patenting.lagrangian_estimate import (
    parent_state, delta_remove_neuron, delta_add_neurons,
)

DATA = "data/processed"


def Jsum(net, X, Y):
    return float(np.sum((net.predict(X) - Y) ** 2))


def remove_specific_neuron(net, L, k):
    """Exact-restriction candidate: delete neuron k of hidden layer L."""
    n = net.copy()
    n.weights[L - 1] = np.delete(n.weights[L - 1], k, axis=0)
    n.biases[L - 1] = np.delete(n.biases[L - 1], k, axis=0)
    n.weights[L] = np.delete(n.weights[L], k, axis=1)
    n.layers[L] -= 1
    n.validate()
    return n


def add_m_neurons(net, L, m, seed):
    """Grow hidden layer L by m neurons (small-random new links)."""
    rng = np.random.default_rng(seed)
    n = net.copy()
    n_prev, n_next = net.layers[L - 1], net.layers[L + 1]
    n.weights[L - 1] = np.vstack([n.weights[L - 1],
                                  rng.standard_normal((m, n_prev)) * 0.1])
    n.biases[L - 1] = np.vstack([n.biases[L - 1], np.zeros((m, 1))])
    n.weights[L] = np.hstack([n.weights[L], np.zeros((n_next, m))])
    n.layers[L] += m
    n.validate()
    return n


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def make_synth(n=400, seed=1):
    """A 1-D target that genuinely needs ~6 tanh neurons, so an underfit parent
    leaves real, graded signal for the estimate to rank."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-2.0, 2.0, n))[None, :]
    y = (np.sin(3.0 * x) + 0.3 * np.sin(7.0 * x))
    return x, y


def run(dataset="dataset_A", parent_topo=(1, 5, 1), L=1,
        retrain_epochs=400, n_restarts=6, seed=0):
    np.random.seed(seed)
    if dataset == "synth":
        Xtr, Ytr = make_synth(seed=1)
        Xte, Yte = make_synth(seed=2)
    else:
        Xtr, Ytr = load_pt_dataset(f"{DATA}/{dataset}_train.pt")
        Xte, Yte = load_pt_dataset(f"{DATA}/{dataset}_test.pt")
    Xtr, Xte, _ = normalize_data(Xtr, Xte)
    Ytr, Yte, _ = normalize_data(Ytr, Yte)

    print(f"\n=== {dataset}  parent {list(parent_topo)} ===")
    parent = DynamicTopologyNetwork(list(parent_topo))
    parent = train_network(parent, Xtr, Ytr, epochs=2000, n_restarts=n_restarts)
    acts, lam, Jp = parent_state(parent, Xtr, Ytr)
    print(f"parent J(sum)={Jp:.6e}   width(L={L})={parent.layers[L]}")

    rows = []   # (name, cheap_dL, true_dJ)

    # ---- SUBTRACTIVE: remove each neuron of layer L -----------------------
    for k in range(parent.layers[L]):
        cheap = delta_remove_neuron(parent, acts, lam, L, k)
        cand = remove_specific_neuron(parent, L, k)
        cand = train_network(cand, Xtr, Ytr, epochs=retrain_epochs,
                             n_restarts=n_restarts)
        true = Jsum(cand, Xtr, Ytr) - Jp
        rows.append((f"REMOVE_n{k}", cheap, true))

    # ---- ADDITIVE: add m neurons to layer L -------------------------------
    for m in (1, 2, 4):
        cheap, _ = delta_add_neurons(parent, acts, lam, L, m=m, n_warm=2, seed=seed)
        cand = add_m_neurons(parent, L, m, seed=seed + m)
        cand = train_network(cand, Xtr, Ytr, epochs=retrain_epochs,
                             n_restarts=n_restarts)
        true = Jsum(cand, Xtr, Ytr) - Jp
        rows.append((f"ADD_{m}", cheap, true))

    # ---- report -----------------------------------------------------------
    print(f"\n{'candidate':<12}{'cheap ΔL':>16}{'true ΔJ':>16}")
    for name, c, t in rows:
        print(f"{name:<12}{c:>16.6e}{t:>16.6e}")

    names = [r[0] for r in rows]
    cheap = [r[1] for r in rows]
    true = [r[2] for r in rows]
    rho = spearman(cheap, true)

    # removals only: does the cheap argmin (least-harmful) match the true one?
    rem = [(n, c, t) for n, c, t in rows if n.startswith("REMOVE")]
    cheap_best = min(rem, key=lambda r: r[1])[0]
    true_best = min(rem, key=lambda r: r[2])[0]

    print(f"\nSpearman(cheap ΔL, true ΔJ) over all candidates : {rho:+.3f}")
    print(f"least-harmful removal  cheap={cheap_best}  true={true_best}  "
          f"{'MATCH' if cheap_best == true_best else 'MISS'}")
    print(f"sign agreement (ADD<0, REMOVE>0): "
          f"{sum((c<0)==(t<0) for _,c,t in rows)}/{len(rows)}")
    return rho


if __name__ == "__main__":
    # synth = underfit parent (real graded signal); A/B = saturated (≈0, stop)
    run("synth", (1, 3, 1))
    run("dataset_A", (1, 5, 1))
    run("dataset_B", (1, 5, 1))
