# ============================================================================
# run_stator.py  —  PMSM STATOR-WINDING prediction (Schaeffler use case)
# ============================================================================
#
# Corrects the earlier motor phase (which predicted the rotor magnet temp `pm`
# via a manufactured 48-dim compression block + dynamic twin).  Per Fabian's
# brief the task is: predict the STATOR WINDING temperature from the remaining
# signals, with `pm` usable as an INPUT and `coolant` expected to be important —
# and to compare different INPUT-OUTPUT CONFIGURATIONS (all signals vs. only a
# subset of the temperature measurements).
#
# This is the patent's static-regression embodiment (Claims 1-8): for each input
# configuration the FAITHFUL constructive search (architecture_search) finds the
# smallest network fitting winding = f(inputs), and we report topology, params
# and held-out error.  Comparing configs answers "can we drop the other
# temperature sensors (keeping coolant)?" and "does pm help?".
#
# Run:  python -m patenting.motor.run_stator
# ============================================================================

import os
import json
import numpy as np
import pandas as pd

from patenting.network import DynamicTopologyNetwork
from patenting.architecture_search import architecture_search
from patenting.visualization import draw_network_architecture
from patenting.utils import mse

CSV = os.path.expanduser(
    "~/.cache/kagglehub/datasets/wkirgsn/electric-motor-temperature/"
    "versions/3/measures_v2.csv")
TARGET = "stator_winding"
OUTDIR = os.path.join(os.path.dirname(__file__), "outputs")

# input-output configurations to compare (the email's central question)
ELEC = ["ambient", "u_d", "u_q", "motor_speed", "i_d", "i_q", "torque"]
CONFIGS = {
    "all_signals":      ELEC + ["coolant", "pm", "stator_tooth", "stator_yoke"],
    "coolant+elec":     ELEC + ["coolant"],            # drop ALL other temps
    "coolant+elec+pm":  ELEC + ["coolant", "pm"],      # pm allowed as input
    "elec_only":        ELEC,                          # also drop coolant
    "temps_only":       ["coolant", "pm", "stator_tooth", "stator_yoke"],
}


class _Null:
    def reconstruction_loss(self, X):
        return 0.0


def load_split(seed=42, train_frac=0.7, val_frac=0.1):
    """Read once, split by profile_id (leakage-safe), return per-split row
    indices and the dataframe."""
    df = pd.read_csv(CSV)
    rng = np.random.default_rng(seed)
    pids = np.array(sorted(df["profile_id"].unique()))
    perm = rng.permutation(len(pids))
    n_tr = int(round(len(pids) * train_frac))
    n_va = int(round(len(pids) * val_frac))
    split = {"train": set(pids[perm[:n_tr]]),
             "val":   set(pids[perm[n_tr:n_tr + n_va]]),
             "test":  set(pids[perm[n_tr + n_va:]])}
    masks = {k: df["profile_id"].isin(v).to_numpy() for k, v in split.items()}
    return df, masks


def make_arrays(df, masks, cols, subsample, seed=0):
    """Column-major (features, samples) arrays, normalised by TRAIN stats.
    Returns arrays plus the target's °C std (for unit-MAE reporting)."""
    rng = np.random.default_rng(seed)
    Xall = df[cols].to_numpy(np.float64).T          # (d, Nall)
    Yall = df[[TARGET]].to_numpy(np.float64).T      # (1, Nall)
    tr = masks["train"]
    xm = Xall[:, tr].mean(1, keepdims=True); xs = Xall[:, tr].std(1, keepdims=True) + 1e-8
    ym = Yall[:, tr].mean(1, keepdims=True); ys = Yall[:, tr].std(1, keepdims=True) + 1e-8
    Xn = (Xall - xm) / xs
    Yn = (Yall - ym) / ys

    def grab(split, cap):
        idx = np.where(masks[split])[0]
        if cap is not None and idx.size > cap:
            idx = rng.choice(idx, cap, replace=False)
        return Xn[:, idx], Yn[:, idx]

    Xtr, Ytr = grab("train", subsample)
    Xva, Yva = grab("val", max(2000, subsample // 5))
    Xte, Yte = grab("test", 20000)
    return Xtr, Ytr, Xva, Yva, Xte, Yte, float(ys)


def make_sessions(df, masks, cols, split, seed=0, max_sessions=None):
    """Per-profile sequences for the dynamic twin: state x = stator_winding,
    drive u = the config inputs. Normalised by TRAIN stats."""
    tr = masks["train"]
    Xall = df[cols].to_numpy(np.float64).T
    Yall = df[[TARGET]].to_numpy(np.float64).T
    xm = Xall[:, tr].mean(1, keepdims=True); xs = Xall[:, tr].std(1, keepdims=True) + 1e-8
    ym = float(Yall[:, tr].mean()); ys = float(Yall[:, tr].std()) + 1e-8
    pid = df["profile_id"].to_numpy()
    want = {p for p, m in zip(pid, masks[split]) if m}
    sessions = []
    for p in sorted(want):
        idx = np.where(pid == p)[0]
        u = (Xall[:, idx] - xm) / xs
        x = (Yall[:, idx] - ym) / ys
        sessions.append({"x": x, "u": u, "pid": int(p)})
    if max_sessions:
        sessions = sessions[:max_sessions]
    return sessions, (ym, ys)


def run_dynamic(config="coolant+elec", stride=10, cycles=10, subsample=6000,
                seed=42, save_dir=OUTDIR):
    """Dynamic twin for stator winding: x_{t+Δ}=x_t+h(x_t,u_t), free-running
    rollout on held-out profiles (captures the thermal lag the static map can't)."""
    from patenting.dynamic import learn_dynamics, rollout
    df, masks = load_split(seed=seed)
    cols = CONFIGS[config]
    tr, _ = make_sessions(df, masks, cols, "train", seed)
    va, _ = make_sessions(df, masks, cols, "val", seed)
    te, denorm = make_sessions(df, masks, cols, "test", seed)
    print(f"\nDYNAMIC twin | config {config} | drive {len(cols)} signals | "
          f"stride {stride} ({0.5*stride:.1f}s)")
    h, meta = learn_dynamics(tr, va, stride=stride, cycles=cycles,
                             subsample=subsample, seed=seed, n_restarts=6)
    rolls = [rollout(h, s, stride=stride, denorm=denorm) for s in te]
    rmse = np.array([r["rmse"] for r in rolls])
    maxe = np.array([r["max_abs"] for r in rolls])
    print(f"  modeling block h : {h.layers}  one-step val MSE "
          f"{meta['onestep_val_mse']:.3e}")
    print(f"  rollout RMSE (°C): mean {rmse.mean():.2f} | max {rmse.max():.2f}  "
          f"| hot-spot |err| max {maxe.max():.2f}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        params = sum(w.size for w in h.weights) + sum(b.size for b in h.biases)
        result = {
            "config": config, "drive_inputs": cols, "n_drive": len(cols),
            "stride": stride, "step_seconds": 0.5 * stride,
            "modeling_block_topology": list(h.layers),
            "skip_connections": {str(k): [int(s[0]) for s in v]
                                 for k, v in h.skip_connections.items()},
            "params": int(params),
            "onestep_val_mse": float(meta["onestep_val_mse"]),
            "rollout_rmse_mean_c": float(rmse.mean()),
            "rollout_rmse_max_c": float(rmse.max()),
            "hotspot_abs_max_c": float(maxe.max()),
            "n_test_profiles": len(te),
            "per_profile": [{"pid": int(s["pid"]), "rmse_c": r["rmse"],
                             "max_abs_c": r["max_abs"]}
                            for s, r in zip(te, rolls)],
        }
        with open(os.path.join(save_dir, "dynamic_results.json"), "w") as f:
            json.dump(result, f, indent=2)
        draw_network_architecture(
            h, f"dynamic block h  {h.layers}  (rollout {rmse.mean():.2f} °C RMSE)",
            os.path.join(save_dir, "dynamic_block_arch.png"))
        # save trajectories for best / median / worst held-out profiles
        order = np.argsort(rmse)
        picks = {"best": int(order[0]),
                 "median": int(order[len(order) // 2]),
                 "worst": int(order[-1])}
        traj = {kind: {"pid": int(te[i]["pid"]),
                       "rmse_c": rolls[i]["rmse"],
                       "max_abs_c": rolls[i]["max_abs"],
                       "x_true": rolls[i]["x_true"][0].tolist(),
                       "x_hat": rolls[i]["x_hat"][0].tolist()}
                for kind, i in picks.items()}
        np.save(os.path.join(save_dir, "dynamic_trajectories.npy"),
                traj, allow_pickle=True)
        print(f"saved → {os.path.join(save_dir, 'dynamic_results.json')}")
    return h, rolls


def run(subsample=6000, cycles=12, seed=42, save_dir=OUTDIR):
    df, masks = load_split(seed=seed)
    print(f"rows {len(df):,}  | split profiles "
          f"tr/va/te = {sum(masks['train'])//1000}k/"
          f"{sum(masks['val'])//1000}k/{sum(masks['test'])//1000}k samples")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    rows = []
    results = []
    for name, cols in CONFIGS.items():
        np.random.seed(seed)
        Xtr, Ytr, Xva, Yva, Xte, Yte, ys = make_arrays(
            df, masks, cols, subsample, seed)
        print("\n" + "=" * 70)
        print(f"CONFIG {name:<16} | {len(cols)} inputs: {cols}")
        print("=" * 70)
        init = DynamicTopologyNetwork([len(cols), 1, 1])
        net, _ = architecture_search(
            init, _Null(), Xtr, Ytr, Xva, Yva, Xte, Yte,
            cycles=cycles, n_restarts=6, full_train_epochs=300)
        pred = net.predict(Xte)
        test_mse = mse(pred, Yte)
        mae_c = float(np.mean(np.abs(pred - Yte))) * ys
        rmse_c = float(np.sqrt(np.mean((pred - Yte) ** 2))) * ys
        maxe_c = float(np.max(np.abs(pred - Yte))) * ys
        params = sum(w.size for w in net.weights) + sum(b.size for b in net.biases)
        rows.append((name, len(cols), net.layers, params, test_mse, mae_c))
        results.append({
            "config": name, "inputs": cols, "n_inputs": len(cols),
            "topology": list(net.layers),
            "skip_connections": {str(k): [int(s[0]) for s in v]
                                 for k, v in net.skip_connections.items()},
            "params": int(params), "test_mse": float(test_mse),
            "mae_c": mae_c, "rmse_c": rmse_c, "max_abs_c": maxe_c,
            "target_std_c": float(ys),
        })
        print(f"  -> {name}: {net.layers}  {params}p  "
              f"test MSE {test_mse:.4e}  MAE {mae_c:.3f} °C")
        if save_dir:
            draw_network_architecture(
                net, f"static {name}  {net.layers}  ({mae_c:.2f} °C MAE)",
                os.path.join(save_dir, f"static_arch_{name}.png"))

    print("\n" + "=" * 78)
    print(f"{'config':<18}{'#in':>4}{'topology':>16}{'params':>8}"
          f"{'test MSE':>12}{'MAE °C':>9}")
    print("-" * 78)
    for name, nin, topo, p, m, mae in rows:
        print(f"{name:<18}{nin:>4}{str(topo):>16}{p:>8}{m:>12.3e}{mae:>9.3f}")
    print("=" * 78)
    if save_dir:
        with open(os.path.join(save_dir, "static_results.json"), "w") as f:
            json.dump({"seed": seed, "subsample": subsample, "cycles": cycles,
                       "target": TARGET, "elec": ELEC, "configs": results}, f,
                      indent=2)
        print(f"saved → {os.path.join(save_dir, 'static_results.json')}")
    return rows


if __name__ == "__main__":
    run()
    run_dynamic()
