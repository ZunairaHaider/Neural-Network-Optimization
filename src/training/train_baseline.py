"""Staged HPO + final retrain for the baseline MLP.

Pipeline
--------
1. Pilot: compare CV stability at N=1000 vs N=4000 on 5 random configs.
   Use N=1000 if rel_std < 10% and ranking is preserved.
2. Stage 1: TPE over (n_layers in {1,2}, n_neurons in {4,6,8,10,12,16},
   activation, lr, weight_decay). 5 OOD-flavored val repeats per trial.
3. Stage 2: advance if Stage 1 best > target floor and improvement room exists.
   TPE over (n_layers in {2,3}, n_neurons in {12,16,24,32}).
4. Stage 3 (only if needed): n_neurons in {32, 48}.
5. 1-SE rule across all stages: smallest model within (best_mean + SE_of_best).
6. Retrain winner on full 4000 with single OOD-flavored val split (seed 99) for
   early stopping. Evaluate on locked OOD test set EXACTLY ONCE.

Seeds: torch/np/random = 42 at module top. Model init seed reset per trial.

Usage
-----
    python src/training/train_baseline.py --dataset A
    python src/training/train_baseline.py --dataset B
    python src/training/train_baseline.py --dataset A --skip-pilot --n-hpo 4000
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch

# ── seeds ─────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.generate_data import load_split  # noqa: E402
from src.models.mlp import MLP  # noqa: E402
from src.training.hpo_utils import (  # noqa: E402
    make_ood_splits,
    n_params_uniform_mlp,
    train_one,
)

# ── stage config ──────────────────────────────────────────────────────────────
STAGES = [
    {
        "name": "stage1",
        "n_layers": [1, 2],
        "n_neurons": [4, 6, 8, 10, 12, 16],
        "trials": 120,
    },
    {
        "name": "stage2",
        "n_layers": [2, 3],
        "n_neurons": [12, 16, 24, 32],
        "trials": 80,
    },
    {
        "name": "stage3",
        "n_layers": [2, 3],
        "n_neurons": [32, 48],
        "trials": 40,
    },
]
ACTIVATIONS = ["relu", "tanh", "gelu", "sigmoid"]
WEIGHT_DECAYS = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
LR_RANGE = (1e-4, 1e-2)
MAX_TRIAL_EPOCHS = 500
TRIAL_PATIENCE = 20
N_REPEATS = 1  # single OOD-flavored split per trial (no CV)
HPO_POOL_SIZE = 1000  # fixed N=1000 subsample of the 4000-point pool

MAX_RETRAIN_EPOCHS = 1000
RETRAIN_PATIENCE = 20

PARAM_BUDGET_SOFT = 5_000  # reject during HPO to keep models small (not hard cap)
STAGE_ADVANCE_REL_IMPROVEMENT = 0.05  # advance only if best drops >=5% rel
NOISE_FLOOR_STOP_B = 4.8  # 1.2 * sigma^2 = 4.8 for Dataset B
NOISE_FLOOR_STOP_A = 1e-3  # generous floor for clean A
SELECTION_TOLERANCE = 0.05  # 5% relative tolerance (no SE since single split)


# ── device ────────────────────────────────────────────────────────────────────
def pick_device(preferred: str) -> torch.device:
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── objective ─────────────────────────────────────────────────────────────────
def evaluate_config(
    x_pool: torch.Tensor,
    y_pool: torch.Tensor,
    splits,
    n_layers: int,
    n_neurons: int,
    activation: str,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> dict:
    val_mses: list[float] = []
    epochs_used: list[int] = []
    for tr_idx, va_idx in splits:
        torch.manual_seed(42)  # fixed init seed; data varies
        model = MLP(n_layers=n_layers, n_neurons=n_neurons, activation=activation)
        x_tr = x_pool[tr_idx]
        y_tr = y_pool[tr_idx]
        x_va = x_pool[va_idx]
        y_va = y_pool[va_idx]
        best_val, info = train_one(
            model, x_tr, y_tr, x_va, y_va,
            lr=lr, weight_decay=weight_decay,
            max_epochs=MAX_TRIAL_EPOCHS, patience=TRIAL_PATIENCE, device=device,
        )
        val_mses.append(best_val)
        epochs_used.append(info["epoch"])
    arr = np.asarray(val_mses, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "se": float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
        "val_mses": val_mses,
        "epochs": epochs_used,
    }


def make_objective(stage_cfg, x_pool, y_pool, splits, device):
    def objective(trial: optuna.Trial) -> float:
        n_layers = trial.suggest_categorical("n_layers", stage_cfg["n_layers"])
        n_neurons = trial.suggest_categorical("n_neurons", stage_cfg["n_neurons"])
        activation = trial.suggest_categorical("activation", ACTIVATIONS)
        lr = trial.suggest_float("lr", LR_RANGE[0], LR_RANGE[1], log=True)
        weight_decay = trial.suggest_categorical("weight_decay", WEIGHT_DECAYS)

        n_p = n_params_uniform_mlp(n_layers, n_neurons)
        if n_p > PARAM_BUDGET_SOFT:
            raise optuna.TrialPruned(f"n_params={n_p} > budget {PARAM_BUDGET_SOFT}")

        result = evaluate_config(
            x_pool, y_pool, splits,
            n_layers=n_layers, n_neurons=n_neurons, activation=activation,
            lr=lr, weight_decay=weight_decay, device=device,
        )
        trial.set_user_attr("mean", result["mean"])
        trial.set_user_attr("se", result["se"])
        trial.set_user_attr("std", result["std"])
        trial.set_user_attr("val_mses", result["val_mses"])
        trial.set_user_attr("n_params", n_p)
        return result["mean"]

    return objective


# ── stage runner ──────────────────────────────────────────────────────────────
def run_stage(
    stage_cfg,
    x_pool: torch.Tensor,
    y_pool: torch.Tensor,
    splits,
    device: torch.device,
    seed: int = 42,
) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True)
    study = optuna.create_study(direction="minimize", sampler=sampler,
                                study_name=stage_cfg["name"])
    objective = make_objective(stage_cfg, x_pool, y_pool, splits, device)

    log_every = max(1, stage_cfg["trials"] // 10)

    def _cb(study, trial):
        n = len(study.trials)
        if n == 1 or n % log_every == 0:
            try:
                bv = study.best_value
                bp = study.best_trial.params
                print(f"    trial {n:3d}  best={bv:.4e}  params={bp}")
            except ValueError:
                pass

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0 = time.time()
    study.optimize(objective, n_trials=stage_cfg["trials"], callbacks=[_cb])
    dt = time.time() - t0
    print(f"  {stage_cfg['name']} done in {dt:.1f}s "
          f"({stage_cfg['trials']} trials)")
    return study


def best_trial_summary(study: optuna.Study) -> dict:
    t = study.best_trial
    return {
        "params": t.params,
        "mean": t.user_attrs["mean"],
        "se": t.user_attrs["se"],
        "std": t.user_attrs["std"],
        "n_params": t.user_attrs["n_params"],
        "trial_number": t.number,
    }


def all_complete_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]


def apply_tolerance_rule(
    all_trials: list[optuna.trial.FrozenTrial],
    tolerance: float = SELECTION_TOLERANCE,
) -> optuna.trial.FrozenTrial:
    """Select smallest model whose val MSE <= best_val_mse * (1 + tolerance).

    With a single OOD-flavored val split (no repeats) we cannot compute SE, so
    we use a fixed relative tolerance (default 5%) as a proxy for 'within noise'.
    """
    best = min(all_trials, key=lambda t: t.user_attrs["mean"])
    threshold = best.user_attrs["mean"] * (1.0 + tolerance)
    candidates = [t for t in all_trials if t.user_attrs["mean"] <= threshold]

    def size_key(t):
        p = t.params
        return (t.user_attrs["n_params"], int(p["n_layers"]), int(p["n_neurons"]))

    return min(candidates, key=size_key)


# ── final retrain ─────────────────────────────────────────────────────────────
def final_retrain(
    params: dict,
    x_pool: torch.Tensor,
    y_pool: torch.Tensor,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    final_splits = make_ood_splits(x_pool, n_repeats=1, base_seed=99)
    tr_idx, va_idx = final_splits[0]
    x_tr = x_pool[tr_idx]
    y_tr = y_pool[tr_idx]
    x_va = x_pool[va_idx]
    y_va = y_pool[va_idx]

    torch.manual_seed(42)
    model = MLP(
        n_layers=int(params["n_layers"]),
        n_neurons=int(params["n_neurons"]),
        activation=params["activation"],
    )
    best_val, info = train_one(
        model, x_tr, y_tr, x_va, y_va,
        lr=float(params["lr"]), weight_decay=float(params["weight_decay"]),
        max_epochs=MAX_RETRAIN_EPOCHS, patience=RETRAIN_PATIENCE,
        device=device, record_curves=True,
    )
    return model, {
        "val_mse": best_val,
        "epochs": info["epoch"],
        "train_hist": info["train_hist"],
        "val_hist": info["val_hist"],
        "val_idx": va_idx.tolist(),
    }


# ── plotting & saving ─────────────────────────────────────────────────────────
def save_training_curves(train_hist, val_hist, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_hist) + 1)
    ax.plot(epochs, train_hist, label="Train MSE", linewidth=1)
    ax.plot(epochs, val_hist, label="Val MSE (OOD-flavored)", linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.set_title("Final retraining curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def save_hpo_results(stage_studies: dict, out_dir):
    rows = []
    for stage_name, study in stage_studies.items():
        for t in all_complete_trials(study):
            rows.append({
                "stage": stage_name,
                "trial_number": t.number,
                **t.params,
                "mean_val_mse": t.user_attrs["mean"],
                "se": t.user_attrs["se"],
                "std": t.user_attrs["std"],
                "n_params": t.user_attrs["n_params"],
            })
    if rows:
        with open(out_dir / "hpo_results.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"stage1": "tab:blue", "stage2": "tab:orange", "stage3": "tab:green"}
    offset = 0
    for stage_name, study in stage_studies.items():
        trials = all_complete_trials(study)
        if not trials:
            continue
        xs = [offset + t.number for t in trials]
        ys = [t.user_attrs["mean"] for t in trials]
        ax.scatter(xs, ys, s=15, alpha=0.6,
                   color=colors.get(stage_name, "gray"), label=stage_name)
        offset += len(trials) + 5
    ax.set_xlabel("Trial (concatenated across stages)")
    ax.set_ylabel("Mean CV val MSE")
    ax.set_yscale("log")
    ax.set_title("HPO trial history by stage")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "hpo_results.png", dpi=150)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────
def run_for_dataset(dataset: str, args) -> dict:
    device = pick_device(args.device)
    print(f"\n=========== Dataset {dataset}  | device={device} ===========")

    x_full, y_full = load_split(dataset, "train")
    print(f"HPO pool: x.shape={tuple(x_full.shape)}, "
          f"x range=[{x_full.min().item():.3f}, {x_full.max().item():.3f}]")

    n_hpo = args.n_hpo if args.n_hpo is not None else HPO_POOL_SIZE
    if n_hpo < len(x_full):
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(len(x_full), size=n_hpo, replace=False))
        x_pool = x_full[idx]
        y_pool = y_full[idx]
        print(f"\nUsing N={n_hpo} subsample for HPO trials (single OOD-flavored split)")
    else:
        x_pool = x_full
        y_pool = y_full
        print(f"\nUsing full N={n_hpo} for HPO trials")

    splits = make_ood_splits(x_pool, n_repeats=N_REPEATS)
    tr_idx0, va_idx0 = splits[0]
    print(f"Random IID val split: "
          f"train={len(tr_idx0)} (x in [{x_pool[tr_idx0].min().item():.3f}, "
          f"{x_pool[tr_idx0].max().item():.3f}]), "
          f"val={len(va_idx0)} (x in [{x_pool[va_idx0].min().item():.3f}, "
          f"{x_pool[va_idx0].max().item():.3f}])")

    noise_floor = NOISE_FLOOR_STOP_B if dataset == "B" else NOISE_FLOOR_STOP_A

    stage_studies: dict[str, optuna.Study] = {}
    prev_best = float("inf")
    for i, stage_cfg in enumerate(STAGES):
        print(f"\n--- {stage_cfg['name']}: layers={stage_cfg['n_layers']} "
              f"neurons={stage_cfg['n_neurons']} trials={stage_cfg['trials']} ---")
        study = run_stage(stage_cfg, x_pool, y_pool, splits, device, seed=42 + i)
        stage_studies[stage_cfg["name"]] = study
        bs = best_trial_summary(study)
        print(f"  best mean CV val MSE = {bs['mean']:.4e} +/- {bs['se']:.4e}  "
              f"params={bs['params']}  n_params={bs['n_params']}")

        if bs["mean"] <= noise_floor:
            print(f"  noise floor reached ({bs['mean']:.4e} <= {noise_floor}). "
                  f"Skipping further stages.")
            break
        rel_improvement = (prev_best - bs["mean"]) / prev_best if prev_best != float("inf") else 1.0
        if i > 0 and rel_improvement < STAGE_ADVANCE_REL_IMPROVEMENT:
            print(f"  improvement {rel_improvement:.2%} < "
                  f"{STAGE_ADVANCE_REL_IMPROVEMENT:.0%}. Stopping stage advancement.")
            break
        prev_best = bs["mean"]

    # Tolerance-based selection across all completed stages
    all_trials = [t for s in stage_studies.values() for t in all_complete_trials(s)]
    overall_best = min(all_trials, key=lambda t: t.user_attrs["mean"])
    chosen = apply_tolerance_rule(all_trials)

    print("\n--- selection ---")
    print(f"  overall best: trial #{overall_best.number}  "
          f"val_mse={overall_best.user_attrs['mean']:.4e}  "
          f"params={overall_best.params}  "
          f"n_params={overall_best.user_attrs['n_params']}")
    print(f"  tolerance choice ({SELECTION_TOLERANCE:.0%} of best, smallest): "
          f"trial #{chosen.number}  "
          f"val_mse={chosen.user_attrs['mean']:.4e}  "
          f"params={chosen.params}  n_params={chosen.user_attrs['n_params']}")

    # Final retrain on FULL 4000 (not subsample)
    print(f"\n--- final retrain on full N={len(x_full)} pool ---")
    model, retr = final_retrain(chosen.params, x_full, y_full, device)
    n_params_final = model.get_weight_count()
    print(f"  retrain val MSE = {retr['val_mse']:.4e} after {retr['epochs']} epochs"
          f"  | n_params = {n_params_final}")

    # Locked test set: SINGLE EVAL
    x_te, y_te = load_split(dataset, "test")
    model.eval()
    with torch.no_grad():
        test_mse = torch.nn.functional.mse_loss(
            model(x_te.to(device)), y_te.to(device)
        ).item()
    print(f"  LOCKED test MSE = {test_mse:.4e} (evaluated once)")

    # Persist
    out_dir = REPO_ROOT / "results" / "baseline" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    save_training_curves(retr["train_hist"], retr["val_hist"], out_dir)
    save_hpo_results(stage_studies, out_dir)

    config = {
        "dataset": dataset,
        "architecture": {
            "n_layers": int(chosen.params["n_layers"]),
            "n_neurons": int(chosen.params["n_neurons"]),
            "activation": chosen.params["activation"],
        },
        "training_hparams": {
            "lr": float(chosen.params["lr"]),
            "weight_decay": float(chosen.params["weight_decay"]),
            "optimizer": "Adam",
            "batch_mode": "full-batch",
            "max_epochs": MAX_RETRAIN_EPOCHS,
            "patience": RETRAIN_PATIENCE,
        },
        "n_params": n_params_final,
        "metrics": {
            "hpo_val_mse": chosen.user_attrs["mean"],
            "overall_best_val_mse": overall_best.user_attrs["mean"],
            "retrain_val_mse": retr["val_mse"],
            "test_mse_locked": test_mse,
        },
        "hpo": {
            "split_type": "random IID (val = random 20% of HPO pool)",
            "n_repeats": N_REPEATS,
            "val_frac": 0.2,
            "stages_completed": list(stage_studies.keys()),
            "trials_per_stage": {k: len(v.trials) for k, v in stage_studies.items()},
            "n_hpo_pool": int(n_hpo),
            "noise_floor_target": noise_floor,
        },
        "selection_rule": f"smallest n_params within {SELECTION_TOLERANCE:.0%} of best val MSE",
        "selected_trial_number": chosen.number,
        "overall_best_trial_number": overall_best.number,
    }
    with open(out_dir / "config.json", "w") as fh:
        json.dump(config, fh, indent=2)
    print(f"  saved -> {out_dir}/")

    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["A", "B", "D", "both"], default="both")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu",
                        help="Tiny MLPs run fastest on CPU due to dispatch overhead.")
    parser.add_argument("--n-hpo", type=int, default=None,
                        help=f"Override HPO pool size (default {HPO_POOL_SIZE}).")
    args = parser.parse_args()

    datasets = ["A", "B"] if args.dataset == "both" else [args.dataset]
    for d in datasets:
        run_for_dataset(d, args)


if __name__ == "__main__":
    main()
