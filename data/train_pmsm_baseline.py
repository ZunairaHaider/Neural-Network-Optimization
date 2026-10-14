"""PMSM baseline training pipeline — Optuna HPO with mini-batch Adam.

Standalone script that trains baseline MLPs for the three PMSM input
configurations.  It does NOT import from or modify the existing Dataset A/B/D
training script (``train_baseline.py``).

Outputs per config are saved to ``results/baseline/PMSM_<config>/``:
    config.json         — architecture + hparams (pruner-compatible)
    model.pt            — state_dict of the final retrained model
    hpo_results.csv     — all completed Optuna trials
    training_curves.png — train / val loss during final retrain
    metrics.json        — test metrics including denormalised °C values

Usage
-----
    # Train all three configs:
    python -m src.training.train_pmsm_baseline --csv path/to/measures_v2.csv

    # Train a single config:
    python -m src.training.train_pmsm_baseline --csv path/to/measures_v2.csv \\
        --configs PMSM_no_temps

    # Quick smoke test (10 trials, 50 epochs, 10k rows):
    python -m src.training.train_pmsm_baseline --csv path/to/measures_v2.csv \\
        --n-trials 10 --max-epochs 50 --hpo-pool-size 10000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.models.mlp import MLP                               # noqa: E402
from src.data.generate_data import load_split                 # noqa: E402
from src.data.pmsm_configs import ALL_CONFIG_NAMES, get_config  # noqa: E402
from src.data.pmsm_loader import (                            # noqa: E402
    load_norm_stats,
    prepare_all_configs as prepare_pmsm_data,
)
from src.training.hpo_utils import n_params_mlp               # noqa: E402

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_N_TRIALS       = 100
DEFAULT_MAX_EPOCHS     = 500
DEFAULT_PATIENCE       = 30
DEFAULT_HPO_POOL_SIZE  = 100_000   # subsample train split for HPO speed
DEFAULT_RETRAIN_EPOCHS = 800
DEFAULT_RETRAIN_PAT    = 50
SELECTION_TOLERANCE    = 0.05      # 5% of best val MSE
PARAM_BUDGET_SOFT_CAP  = 100_000   # prune trials > this early

# ══════════════════════════════════════════════════════════════════════════════
# Mini-batch training loop
# ══════════════════════════════════════════════════════════════════════════════


def train_minibatch(
    model: nn.Module,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_va: torch.Tensor,
    y_va: torch.Tensor,
    lr: float,
    weight_decay: float,
    batch_size: int = 1024,
    max_epochs: int = 500,
    patience: int = 30,
    device: torch.device | None = None,
    record_curves: bool = False,
    verbose: bool = False,
) -> tuple[float, dict]:
    """Mini-batch Adam training with early stopping on val MSE.

    This is the PMSM equivalent of ``hpo_utils.train_one`` — a separate
    function (not a modification of the original) because PMSM trains
    with mini-batches on ~1 M rows while the synthetic pipeline is full-batch
    on 4 000 rows.

    Returns:
        (best_val_mse, info_dict) where info_dict contains 'epoch' and
        optionally 'train_hist' / 'val_hist'.
    """
    device = device or torch.device("cpu")
    model = model.to(device)
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_va, y_va = x_va.to(device), y_va.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    n = x_tr.shape[0]
    best_val = float("inf")
    best_state = None
    no_imp = 0
    train_hist: list[float] = []
    val_hist: list[float] = []
    last_epoch = 0

    for epoch in range(max_epochs):
        # ── mini-batch training pass ──────────────────────────────────────
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss_sum = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = x_tr[idx], y_tr[idx]

            opt.zero_grad()
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            opt.step()

            epoch_loss_sum += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss_sum / n_batches

        # ── validation (full-batch, no grad) ──────────────────────────────
        model.eval()
        with torch.no_grad():
            vm = nn.functional.mse_loss(model(x_va), y_va).item()

        if record_curves:
            train_hist.append(avg_train_loss)
            val_hist.append(vm)

        last_epoch = epoch + 1

        if vm < best_val - 1e-9:
            best_val = vm
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if verbose and (epoch + 1) % 25 == 0:
            print(f"    epoch {epoch+1:4d}  train={avg_train_loss:.4e}  "
                  f"val={vm:.4e}  best={best_val:.4e}  noimp={no_imp}")

        if no_imp >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    info: dict = {"epoch": last_epoch}
    if record_curves:
        info["train_hist"] = train_hist
        info["val_hist"] = val_hist
    return best_val, info


# ══════════════════════════════════════════════════════════════════════════════
# Optuna objective
# ══════════════════════════════════════════════════════════════════════════════


def make_objective(
    x_pool: torch.Tensor,
    y_pool: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    input_dim: int,
    max_epochs: int,
    patience: int,
    device: torch.device,
):
    """Return an Optuna objective function (closure over data)."""
    import optuna

    def objective(trial: optuna.Trial) -> float:
        n_layers    = trial.suggest_int("n_layers", 1, 3)
        n_neurons   = trial.suggest_categorical("n_neurons", [32, 64, 128, 256])
        activation  = trial.suggest_categorical("activation", ["relu", "tanh"])
        lr          = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_categorical(
            "weight_decay", [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
        )
        batch_size  = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])

        # Soft param budget: skip clearly oversized models.
        n_p = n_params_mlp(n_layers, n_neurons, input_dim=input_dim, output_dim=1)
        if n_p > PARAM_BUDGET_SOFT_CAP:
            raise optuna.TrialPruned(f"n_params={n_p} > soft cap {PARAM_BUDGET_SOFT_CAP}")

        model = MLP(
            n_layers=n_layers,
            n_neurons=n_neurons,
            activation=activation,
            input_dim=input_dim,
            output_dim=1,
        )

        val_mse, info = train_minibatch(
            model, x_pool, y_pool, x_val, y_val,
            lr=lr, weight_decay=weight_decay, batch_size=batch_size,
            max_epochs=max_epochs, patience=patience, device=device,
        )

        trial.set_user_attr("n_params", n_p)
        trial.set_user_attr("epochs", info["epoch"])
        return val_mse

    return objective


# ══════════════════════════════════════════════════════════════════════════════
# Selection rule
# ══════════════════════════════════════════════════════════════════════════════


def select_best_trial(study) -> tuple:
    """Tolerance-based selection: smallest model within 5% of best val MSE.

    Same principle as the A/B/D pipeline in ``train_baseline.py``.
    Returns (chosen_trial, overall_best_trial).
    """
    import optuna
    complete = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise RuntimeError("No completed trials in the study.")

    overall_best = min(complete, key=lambda t: t.value)
    threshold = overall_best.value * (1.0 + SELECTION_TOLERANCE)

    within_tol = [t for t in complete if t.value <= threshold]
    chosen = min(within_tol, key=lambda t: t.user_attrs["n_params"])

    return chosen, overall_best


# ══════════════════════════════════════════════════════════════════════════════
# Saving results
# ══════════════════════════════════════════════════════════════════════════════


def save_training_curves(train_hist, val_hist, out_dir: Path) -> None:
    """Plot and save training/validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(train_hist, label="train MSE", alpha=0.7)
    ax.semilogy(val_hist,   label="val MSE",   alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (normalised)")
    ax.legend()
    ax.set_title("Final retrain — training curves")
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def save_hpo_csv(study, out_dir: Path) -> None:
    """Save all Optuna trials to CSV."""
    import optuna
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {
            "trial": t.number,
            "val_mse": t.value,
            "n_params": t.user_attrs.get("n_params", ""),
            "epochs": t.user_attrs.get("epochs", ""),
        }
        row.update(t.params)
        rows.append(row)

    import csv
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(out_dir / "hpo_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Per-config runner
# ══════════════════════════════════════════════════════════════════════════════


def run_for_config(
    config_name: str,
    n_trials: int,
    max_epochs: int,
    patience: int,
    retrain_epochs: int,
    retrain_patience: int,
    hpo_pool_size: int,
    device: torch.device,
) -> dict:
    """Full HPO → retrain → locked test evaluation for one PMSM config.

    Returns the saved config dict.
    """
    import optuna

    input_cols, target_col = get_config(config_name)
    input_dim = len(input_cols)
    norm_stats = load_norm_stats(config_name)

    print(f"\n{'='*70}")
    print(f"  Config: {config_name}")
    print(f"  Inputs ({input_dim}): {input_cols}")
    print(f"  Target: {target_col}")
    print(f"{'='*70}")

    # ── Load data ─────────────────────────────────────────────────────────
    x_tr_full, y_tr_full = load_split(config_name, "train")
    x_val, y_val         = load_split(config_name, "val")
    x_test, y_test       = load_split(config_name, "test")
    print(f"  Train: {x_tr_full.shape[0]:,} rows")
    print(f"  Val:   {x_val.shape[0]:,} rows")
    print(f"  Test:  {x_test.shape[0]:,} rows")

    # ── Subsample train for HPO speed ─────────────────────────────────────
    if hpo_pool_size and x_tr_full.shape[0] > hpo_pool_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(x_tr_full.shape[0], hpo_pool_size, replace=False)
        idx.sort()
        x_pool = x_tr_full[idx]
        y_pool = y_tr_full[idx]
        print(f"  HPO pool: {hpo_pool_size:,} rows (subsampled from train)")
    else:
        x_pool = x_tr_full
        y_pool = y_tr_full
        print(f"  HPO pool: {x_pool.shape[0]:,} rows (full train)")

    # ── Optuna study ──────────────────────────────────────────────────────
    print(f"\n  Starting Optuna HPO ({n_trials} trials, "
          f"max {max_epochs} epochs, patience {patience}) ...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            multivariate=True, group=True, seed=42,
        ),
    )

    objective = make_objective(
        x_pool, y_pool, x_val, y_val,
        input_dim=input_dim,
        max_epochs=max_epochs,
        patience=patience,
        device=device,
    )

    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    hpo_time = time.time() - t0

    n_complete = len([t for t in study.trials
                      if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned   = len([t for t in study.trials
                      if t.state == optuna.trial.TrialState.PRUNED])
    print(f"  HPO done in {hpo_time:.0f}s  "
          f"({n_complete} complete, {n_pruned} pruned)")

    # ── Select best trial ─────────────────────────────────────────────────
    chosen, overall_best = select_best_trial(study)
    print(f"\n  Overall best:  trial #{overall_best.number}  "
          f"val_mse={overall_best.value:.4e}  "
          f"n_params={overall_best.user_attrs['n_params']}  "
          f"params={overall_best.params}")
    print(f"  Tolerance pick ({SELECTION_TOLERANCE:.0%}): trial #{chosen.number}  "
          f"val_mse={chosen.value:.4e}  "
          f"n_params={chosen.user_attrs['n_params']}  "
          f"params={chosen.params}")

    # ── Final retrain on FULL train split ─────────────────────────────────
    print(f"\n  Final retrain on full train ({x_tr_full.shape[0]:,} rows), "
          f"max {retrain_epochs} epochs, patience {retrain_patience} ...")

    p = chosen.params
    model = MLP(
        n_layers=int(p["n_layers"]),
        n_neurons=int(p["n_neurons"]),
        activation=p["activation"],
        input_dim=input_dim,
        output_dim=1,
    )
    n_params_final = model.get_weight_count()

    t1 = time.time()
    retrain_val_mse, retrain_info = train_minibatch(
        model, x_tr_full, y_tr_full, x_val, y_val,
        lr=float(p["lr"]),
        weight_decay=float(p["weight_decay"]),
        batch_size=int(p["batch_size"]),
        max_epochs=retrain_epochs,
        patience=retrain_patience,
        device=device,
        record_curves=True,
        verbose=True,
    )
    retrain_time = time.time() - t1
    print(f"  Retrain val MSE = {retrain_val_mse:.4e} "
          f"after {retrain_info['epoch']} epochs ({retrain_time:.0f}s)")

    # ── Locked test evaluation (SINGLE EVAL) ──────────────────────────────
    model.eval()
    x_te_d, y_te_d = x_test.to(device), y_test.to(device)
    with torch.no_grad():
        preds = model.to(device)(x_te_d)
        test_mse = nn.functional.mse_loss(preds, y_te_d).item()

    # Denormalise to °C
    y_std  = norm_stats["y_std"]
    y_mean = norm_stats["y_mean"]
    preds_c  = preds.cpu().numpy() * y_std + y_mean
    actual_c = y_te_d.cpu().numpy() * y_std + y_mean
    errors_c = np.abs(preds_c - actual_c)
    mae_c    = float(errors_c.mean())
    rmse_c   = float(np.sqrt(((preds_c - actual_c) ** 2).mean()))
    max_abs_c = float(errors_c.max())
    r2 = float(1.0 - np.sum((actual_c - preds_c)**2) / np.sum((actual_c - actual_c.mean())**2))

    print(f"\n  LOCKED TEST (evaluated once):")
    print(f"    MSE (norm):    {test_mse:.4e}")
    print(f"    MAE  (°C):     {mae_c:.2f}")
    print(f"    RMSE (°C):     {rmse_c:.2f}")
    print(f"    Max abs (°C):  {max_abs_c:.2f}")
    print(f"    R²:            {r2:.4f}")

    # ── Save outputs ──────────────────────────────────────────────────────
    out_dir = REPO_ROOT / "results" / "baseline" / config_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # model.pt
    torch.save(model.state_dict(), out_dir / "model.pt")

    # training_curves.png
    if "train_hist" in retrain_info:
        save_training_curves(
            retrain_info["train_hist"],
            retrain_info["val_hist"],
            out_dir,
        )

    # hpo_results.csv
    save_hpo_csv(study, out_dir)

    # config.json  — format matches what pruner.py::_load_baseline expects
    #
    # NOTE: ``input_dim`` and ``output_dim`` are included inside
    # ``architecture``.  The current pruner does not read them (it defaults
    # to 1), so a one-line fix will be needed in ``_load_baseline`` before
    # pruning PMSM baselines:
    #     input_dim=architecture.get("input_dim", 1),
    #     output_dim=architecture.get("output_dim", 1),
    config = {
        "dataset": config_name,
        "architecture": {
            "n_layers": int(p["n_layers"]),
            "n_neurons": int(p["n_neurons"]),
            "activation": p["activation"],
            "input_dim": input_dim,
            "output_dim": 1,
        },
        "training_hparams": {
            "lr": float(p["lr"]),
            "weight_decay": float(p["weight_decay"]),
            "optimizer": "Adam",
            "batch_size": int(p["batch_size"]),
            "batch_mode": "mini-batch",
            "max_epochs": retrain_epochs,
            "patience": retrain_patience,
        },
        "n_params": n_params_final,
        "metrics": {
            "hpo_val_mse": chosen.value,
            "overall_best_val_mse": overall_best.value,
            "retrain_val_mse": retrain_val_mse,
            "test_mse_locked": test_mse,
            "test_mae_celsius": mae_c,
            "test_rmse_celsius": rmse_c,
            "test_max_abs_celsius": max_abs_c,
            "test_r2": r2,
        },
        "hpo": {
            "split_type": "profile-based (train / val / test by profile_id)",
            "n_trials": n_trials,
            "n_complete": n_complete,
            "n_pruned_by_budget": n_pruned,
            "hpo_pool_size": x_pool.shape[0],
            "val_size": x_val.shape[0],
            "test_size": x_test.shape[0],
            "param_budget_soft_cap": PARAM_BUDGET_SOFT_CAP,
            "hpo_time_seconds": round(hpo_time, 1),
        },
        "normalisation": {
            "y_mean_celsius": y_mean,
            "y_std_celsius": y_std,
            "note": "predictions are in z-score space; multiply by y_std and add y_mean for °C",
        },
        "selection_rule": f"smallest n_params within {SELECTION_TOLERANCE:.0%} of best val MSE",
        "selected_trial_number": chosen.number,
    }
    with open(out_dir / "config.json", "w") as fh:
        json.dump(config, fh, indent=2)

    # metrics.json — lightweight separate file for quick inspection
    metrics = {
        "config_name": config_name,
        "input_dim": input_dim,
        "n_params": n_params_final,
        "architecture": f"{p['n_layers']}×{p['n_neurons']} {p['activation']}",
        "test_mse_normalised": test_mse,
        "test_mae_celsius": mae_c,
        "test_rmse_celsius": rmse_c,
        "test_max_abs_celsius": max_abs_c,
        "test_r2": r2,
    }
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"\n  Saved to {out_dir}/")
    print(f"    config.json, model.pt, hpo_results.csv, "
          f"training_curves.png, metrics.json")
    return config


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PMSM baseline MLPs with Optuna HPO.",
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to measures_v2.csv. If .pt splits don't exist yet, they "
             "are generated automatically.",
    )
    parser.add_argument(
        "--configs", nargs="+", default=ALL_CONFIG_NAMES,
        help=f"Which configs to train (default: all). "
             f"Options: {ALL_CONFIG_NAMES}",
    )
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--retrain-epochs", type=int, default=DEFAULT_RETRAIN_EPOCHS)
    parser.add_argument("--retrain-patience", type=int, default=DEFAULT_RETRAIN_PAT)
    parser.add_argument("--hpo-pool-size", type=int, default=DEFAULT_HPO_POOL_SIZE)
    parser.add_argument("--device", default=None, help="e.g. 'cuda' or 'cpu'")
    args = parser.parse_args()

    # Ensure .pt splits exist (idempotent — skips if already on disk).
    from src.data.pmsm_loader import PROCESSED_DIR
    sample_pt = PROCESSED_DIR / f"dataset_{args.configs[0]}_train.pt"
    if not sample_pt.exists():
        print("PMSM .pt splits not found — generating from CSV ...")
        prepare_pmsm_data(args.csv)
    else:
        print(f"PMSM .pt splits already exist in {PROCESSED_DIR}/")

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    results = {}
    for config_name in args.configs:
        if config_name not in ALL_CONFIG_NAMES:
            print(f"WARNING: skipping unknown config {config_name!r}. "
                  f"Known: {ALL_CONFIG_NAMES}")
            continue
        cfg = run_for_config(
            config_name,
            n_trials=args.n_trials,
            max_epochs=args.max_epochs,
            patience=args.patience,
            retrain_epochs=args.retrain_epochs,
            retrain_patience=args.retrain_patience,
            hpo_pool_size=args.hpo_pool_size,
            device=device,
        )
        results[config_name] = cfg

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Config':<22s}  {'Arch':>12s}  {'Params':>7s}  "
          f"{'RMSE°C':>7s}  {'MAE°C':>6s}  {'R²':>6s}")
    print(f"  {'-'*22}  {'-'*12}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*6}")
    for name, c in results.items():
        m = c["metrics"]
        a = c["architecture"]
        arch_str = f"{a['n_layers']}×{a['n_neurons']} {a['activation']}"
        print(f"  {name:<22s}  {arch_str:>12s}  {c['n_params']:>7d}  "
              f"{m['test_rmse_celsius']:>7.2f}  {m['test_mae_celsius']:>6.2f}  "
              f"{m['test_r2']:>6.4f}")
    print()


if __name__ == "__main__":
    main()
