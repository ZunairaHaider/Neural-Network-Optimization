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
import pandas as pd
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.models.mlp import MLP                                    # noqa: E402
from src.data.pmsm_loader_ewma import load_pmsm_split as load_split   # noqa: E402
from src.data.pmsm_configs import ALL_CONFIG_NAMES, EWMA_CONFIG_NAMES, get_config        # noqa: E402
from src.data.pmsm_loader_ewma import (                               # noqa: E402
    DEFAULT_EWMA_COLS,
    DEFAULT_EWMA_SPANS,
    INPUT_FINGERPRINT_COLS,   # needed for --fingerprint-cols default
    PROCESSED_DIR,
    PHYSICS_VAR,
    add_ewma_features,
    add_physics_features,
    load_norm_stats,
    prepare_all_configs as prepare_pmsm_data,
)
from src.training.hpo_utils import n_params_mlp                       # noqa: E402

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_N_TRIALS       = 100
DEFAULT_MAX_EPOCHS     = 500
DEFAULT_PATIENCE       = 30
DEFAULT_HPO_POOL_SIZE  = 100_000
DEFAULT_RETRAIN_EPOCHS = 800
DEFAULT_RETRAIN_PAT    = 50
SELECTION_TOLERANCE    = 0.05
PARAM_BUDGET_SOFT_CAP  = 100_000


# ══════════════════════════════════════════════════════════════════════════════
# Split-existence check
# ══════════════════════════════════════════════════════════════════════════════

def _splits_are_valid(
    config_name: str,
    add_ewma: bool,
    ewma_cols: list[str] | None,
    ewma_spans: list[int] | None,
    split_mode: str,
    dedup_threshold: float,
    fingerprint_cols: list[str],
    add_physics: bool,
) -> bool:
    """Return True iff the .pt splits on disk match what we're about to request.

    Checks:
      1. At least one split file exists for this config.
      2. The saved pmsm_profile_split_ewma.json records matching parameters
         (split_mode, add_ewma, ewma_cols, ewma_spans, dedup_threshold,
         fingerprint_cols) so stale files from a previous run are detected.

    If any mismatch is found, a human-readable warning is printed and False
    is returned so the caller regenerates the splits.
    """
    suffix = "ewma" if add_ewma else "base"
    sample_pt = PROCESSED_DIR / f"dataset_{config_name}_{suffix}_train.pt"
    if not sample_pt.exists():
        print(f"  Split file not found: {sample_pt}")
        return False

    split_json = PROCESSED_DIR / "pmsm_profile_split_ewma.json"
    if not split_json.exists():
        print(f"  Split metadata not found: {split_json}")
        return False

    with open(split_json) as fh:
        saved = json.load(fh)

    # ── Check each parameter that would change the split ──────────────────
    mismatches: list[str] = []

    if saved.get("split_mode") != split_mode:
        mismatches.append(
            f"split_mode: saved={saved.get('split_mode')!r} "
            f"requested={split_mode!r}"
        )

    if saved.get("add_ewma") != add_ewma:
        mismatches.append(
            f"add_ewma: saved={saved.get('add_ewma')} requested={add_ewma}"
        )
    # if saved.get("add_physics") != add_physics:
    #     mismatches.append(
    #         f"add_physics: saved={saved.get('add_physics')} "
    #         f"requested={add_physics}"
    #     )
    if add_ewma:
        if add_physics:
            req_cols  = sorted(ewma_cols + PHYSICS_VAR  or DEFAULT_EWMA_COLS)
        req_spans = sorted(ewma_spans or DEFAULT_EWMA_SPANS)
        saved_cols  = sorted(saved.get("ewma_cols")  or DEFAULT_EWMA_COLS)
        saved_spans = sorted(saved.get("ewma_spans") or DEFAULT_EWMA_SPANS)
        if saved_cols != req_cols:
            mismatches.append(
                f"ewma_cols: saved={saved_cols} requested={req_cols}"
            )
        if saved_spans != req_spans:
            mismatches.append(
                f"ewma_spans: saved={saved_spans} requested={req_spans}"
            )

    if split_mode == "diversity":
        saved_thresh = saved.get("dedup_threshold")
        if saved_thresh is not None and abs(saved_thresh - dedup_threshold) > 1e-6:
            mismatches.append(
                f"dedup_threshold: saved={saved_thresh} "
                f"requested={dedup_threshold}"
            )
        saved_fp = sorted(saved.get("fingerprint_cols") or INPUT_FINGERPRINT_COLS)
        req_fp   = sorted(fingerprint_cols)
        if saved_fp != req_fp:
            mismatches.append(
                f"fingerprint_cols: saved={saved_fp} requested={req_fp}"
            )

    if mismatches:
        print("  WARNING: saved splits do not match requested parameters:")
        for m in mismatches:
            print(f"    • {m}")
        print("  → regenerating splits from CSV.")
        return False

    return True


# ══════════════════════════════════════════════════════════════════════════════
# Mini-batch training loop  (unchanged)
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
    device = device or torch.device("cpu")
    model = model.to(device)
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_va, y_va = x_va.to(device), y_va.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    n = x_tr.shape[0]
    best_val   = float("inf")
    best_state = None
    no_imp     = 0
    train_hist: list[float] = []
    val_hist:   list[float] = []
    last_epoch  = 0

    for epoch in range(max_epochs):
        model.train()
        perm           = torch.randperm(n, device=device)
        epoch_loss_sum = 0.0
        n_batches      = 0

        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = x_tr[idx], y_tr[idx]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss_sum += loss.item()
            n_batches      += 1

        avg_train_loss = epoch_loss_sum / n_batches

        model.eval()
        with torch.no_grad():
            vm = nn.functional.mse_loss(model(x_va), y_va).item()

        if record_curves:
            train_hist.append(avg_train_loss)
            val_hist.append(vm)

        last_epoch = epoch + 1

        if vm < best_val - 1e-9:
            best_val   = vm
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_imp     = 0
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
        info["val_hist"]   = val_hist
    return best_val, info


# ══════════════════════════════════════════════════════════════════════════════
# Optuna objective  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def make_objective(
    x_pool: torch.Tensor,
    y_pool: torch.Tensor,
    x_val:  torch.Tensor,
    y_val:  torch.Tensor,
    input_dim: int,
    max_epochs: int,
    patience: int,
    device: torch.device,
):
    import optuna

    def objective(trial: optuna.Trial) -> float:
        n_layers     = trial.suggest_int("n_layers", 2, 7)
        n_neurons    = trial.suggest_categorical("n_neurons", [64, 128, 256, 512])
        activation   = trial.suggest_categorical("activation", ["relu", "tanh", "gelu"])
        lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_categorical(
            "weight_decay", [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
        )
        batch_size   = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])

        n_p = n_params_mlp(n_layers, n_neurons, input_dim=input_dim, output_dim=1)
        if n_p > PARAM_BUDGET_SOFT_CAP:
            raise optuna.TrialPruned(f"n_params={n_p} > soft cap {PARAM_BUDGET_SOFT_CAP}")

        model = MLP(
            n_layers=n_layers, n_neurons=n_neurons, activation=activation,
            input_dim=input_dim, output_dim=1,
        )
        val_mse, info = train_minibatch(
            model, x_pool, y_pool, x_val, y_val,
            lr=lr, weight_decay=weight_decay, batch_size=batch_size,
            max_epochs=max_epochs, patience=patience, device=device,
        )
        trial.set_user_attr("n_params", n_p)
        trial.set_user_attr("epochs",   info["epoch"])
        return val_mse

    return objective


# ══════════════════════════════════════════════════════════════════════════════
# Selection rule  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def select_best_trial(study) -> tuple:
    import optuna
    complete = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise RuntimeError("No completed trials in the study.")
    overall_best = min(complete, key=lambda t: t.value)
    threshold    = overall_best.value * (1.0 + SELECTION_TOLERANCE)
    within_tol   = [t for t in complete if t.value <= threshold]
    chosen       = min(within_tol, key=lambda t: t.user_attrs["n_params"])
    return chosen, overall_best


# ══════════════════════════════════════════════════════════════════════════════
# Saving helpers  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def save_training_curves(train_hist, val_hist, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(train_hist, label="train MSE", alpha=0.7)
    ax.semilogy(val_hist,   label="val MSE",   alpha=0.7)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (normalised)")
    ax.legend(); ax.set_title("Final retrain — training curves")
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def save_hpo_csv(study, out_dir: Path) -> None:
    import optuna, csv
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {
            "trial":    t.number,
            "val_mse":  t.value,
            "n_params": t.user_attrs.get("n_params", ""),
            "epochs":   t.user_attrs.get("epochs",   ""),
        }
        row.update(t.params)
        rows.append(row)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(out_dir / "hpo_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def save_worst_session_diagnostic(
    model: nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    norm_stats: dict,
    out_dir: Path,
    csv_path: str | Path,
    config_name: str,
    add_ewma: bool,
    ewma_cols: list[str] | None,
    ewma_spans: list[int] | None,
    add_physics: bool,
    device: torch.device | None = None,
) -> dict | None:
    """Save a CSV/plot of the worst-profile test session by absolute error."""
    split_json = PROCESSED_DIR / "pmsm_profile_split_ewma.json"
    if not split_json.exists():
        return None

    with open(split_json) as fh:
        split_meta = json.load(fh)

    test_pids = split_meta.get("test", [])
    if not test_pids:
        return None

    df = pd.read_csv(csv_path)
    if add_physics:
        df, _ = add_physics_features(df)
    if add_ewma:
        df, _ = add_ewma_features(
            df,
            list(ewma_cols or DEFAULT_EWMA_COLS),
            list(ewma_spans or DEFAULT_EWMA_SPANS),
        )

    if "profile_id" not in df.columns:
        return None

    mask = df["profile_id"].isin(test_pids).to_numpy()
    test_df = df.loc[mask].copy()
    input_cols = list(norm_stats.get("input_cols", []))
    target_col = norm_stats.get("target_col", get_config(config_name)[1])

    if not input_cols or len(input_cols) != x_test.shape[1]:
        return None

    if target_col not in test_df.columns:
        return None

    model.eval()
    with torch.no_grad():
        preds = model(x_test.to(device or torch.device("cpu"))).cpu().numpy()

    y_std, y_mean = float(norm_stats["y_std"]), float(norm_stats["y_mean"])
    preds_c = preds[:, 0] * y_std + y_mean
    actual_c = y_test.cpu().numpy()[:, 0] * y_std + y_mean

    diag = test_df.loc[:, ["profile_id"] + input_cols + [target_col]].copy()
    diag = diag.reset_index(drop=True)
    diag["pred_c"] = preds_c
    diag["true_c"] = actual_c
    diag["abs_err_c"] = np.abs(diag["pred_c"] - diag["true_c"])

    if len(diag) != len(preds_c):
        return None

    profile_errors = diag.groupby("profile_id", sort=False)["abs_err_c"].max()
    worst_profile_id = int(profile_errors.idxmax())
    session_df = diag[diag["profile_id"] == worst_profile_id].copy().reset_index(drop=True)

    if session_df.empty:
        return None

    session_df["step"] = np.arange(1, len(session_df) + 1)
    session_df["pred_c"] = session_df["pred_c"].astype(float)
    session_df["true_c"] = session_df["true_c"].astype(float)
    session_df["abs_err_c"] = session_df["abs_err_c"].astype(float)

    plot_cols = [c for c in input_cols if c in session_df.columns]
    if len(plot_cols) > 6:
        plot_cols = plot_cols[:6]

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path_out = out_dir / "worst_test_session.csv"
    session_df.to_csv(csv_path_out, index=False)

    n_panels = 1 + len(plot_cols)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 2.6 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    ax = axes[0]
    ax.plot(session_df["step"], session_df["true_c"], label="true", color="tab:blue")
    ax.plot(session_df["step"], session_df["pred_c"], label="pred", color="tab:orange")
    ax.set_ylabel("Target (°C)")
    ax.set_title(f"Worst test session profile {worst_profile_id} | max abs err {session_df['abs_err_c'].max():.2f} °C")
    ax.legend(loc="best")

    for panel_idx, col in enumerate(plot_cols, start=1):
        ax = axes[panel_idx]
        ax.plot(session_df["step"], session_df[col], color="tab:green")
        ax.set_ylabel(col)
        ax.set_xlabel("step within session")

    fig.tight_layout()
    fig.savefig(out_dir / "worst_test_session.png", dpi=200)
    plt.close(fig)

    return {
        "profile_id": int(worst_profile_id),
        "max_abs_err_c": float(session_df["abs_err_c"].max()),
        "csv": str(csv_path_out),
        "plot": str(out_dir / "worst_test_session.png"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Per-config runner  (unchanged except split_mode recorded in config.json)
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
    split_mode: str,          # recorded in config.json for traceability
    csv_path: str | Path,
    add_ewma: bool,
    ewma_cols: list[str] | None,
    ewma_spans: list[int] | None,
    add_physics: bool,
    seed: int = 42,           # random seed for reproducibility

) -> dict:
    import optuna

    norm_stats = load_norm_stats(config_name)
    if isinstance(norm_stats, dict) and "input_cols" in norm_stats:
        input_cols = list(norm_stats["input_cols"])
        target_col = norm_stats.get("target_col", get_config(config_name)[1])
    else:
        input_cols, target_col = get_config(config_name)
    input_dim = len(input_cols)

    print(f"\n{'='*70}")
    print(f"  Config:    {config_name}")
    print(f"  Split:     {split_mode}")
    print(f"  Inputs ({input_dim}): {input_cols[:6]}{'…' if input_dim > 6 else ''}")
    print(f"  Target:    {target_col}")
    print(f"{'='*70}")

    x_tr_full, y_tr_full = load_split(config_name, "train")
    x_val, y_val         = load_split(config_name, "val")
    x_test, y_test       = load_split(config_name, "test")
    print(f"  Train: {x_tr_full.shape[0]:,} rows")
    print(f"  Val:   {x_val.shape[0]:,} rows")
    print(f"  Test:  {x_test.shape[0]:,} rows")

    # HPO subsample — random rows are fine; EWMA already baked in
    if hpo_pool_size and x_tr_full.shape[0] > hpo_pool_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(x_tr_full.shape[0], hpo_pool_size, replace=False)
        idx.sort()
        x_pool, y_pool = x_tr_full[idx], y_tr_full[idx]
        print(f"  HPO pool: {hpo_pool_size:,} rows (subsampled from train)")
    else:
        x_pool, y_pool = x_tr_full, y_tr_full
        print(f"  HPO pool: {x_pool.shape[0]:,} rows (full train)")

    print(f"\n  Starting Optuna HPO ({n_trials} trials, "
          f"max {max_epochs} epochs, patience {patience}) ...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True, seed=seed),
    )
    objective = make_objective(
        x_pool, y_pool, x_val, y_val,
        input_dim=input_dim, max_epochs=max_epochs,
        patience=patience, device=device
    )
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    hpo_time = time.time() - t0

    n_complete = sum(1 for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE)
    n_pruned   = sum(1 for t in study.trials
                     if t.state == optuna.trial.TrialState.PRUNED)
    print(f"  HPO done in {hpo_time:.0f}s  "
          f"({n_complete} complete, {n_pruned} pruned)")

    chosen, overall_best = select_best_trial(study)
    print(f"\n  Overall best:  trial #{overall_best.number}  "
          f"val_mse={overall_best.value:.4e}  "
          f"n_params={overall_best.user_attrs['n_params']}")
    print(f"  Tolerance pick ({SELECTION_TOLERANCE:.0%}): trial #{chosen.number}  "
          f"val_mse={chosen.value:.4e}  "
          f"n_params={chosen.user_attrs['n_params']}")

    print(f"\n  Final retrain on full train ({x_tr_full.shape[0]:,} rows), "
          f"max {retrain_epochs} epochs, patience {retrain_patience} ...")
    p = chosen.params
    model = MLP(
        n_layers=int(p["n_layers"]), n_neurons=int(p["n_neurons"]),
        activation=p["activation"], input_dim=input_dim, output_dim=1,
    )
    n_params_final = model.get_weight_count()

    t1 = time.time()
    retrain_val_mse, retrain_info = train_minibatch(
        model, x_tr_full, y_tr_full, x_val, y_val,
        lr=float(p["lr"]), weight_decay=float(p["weight_decay"]),
        batch_size=int(p["batch_size"]),
        max_epochs=retrain_epochs, patience=retrain_patience,
        device=device, record_curves=True, verbose=True,
    )
    print(f"  Retrain val MSE = {retrain_val_mse:.4e} "
          f"after {retrain_info['epoch']} epochs ({time.time()-t1:.0f}s)")

    model.eval()
    x_te_d, y_te_d = x_test.to(device), y_test.to(device)
    with torch.no_grad():
        preds    = model(x_te_d)
        test_mse = nn.functional.mse_loss(preds, y_te_d).item()

    y_std, y_mean = norm_stats["y_std"], norm_stats["y_mean"]
    preds_c   = preds.cpu().numpy()   * y_std + y_mean
    actual_c  = y_te_d.cpu().numpy() * y_std + y_mean
    errors_c  = np.abs(preds_c - actual_c)
    mae_c     = float(errors_c.mean())
    rmse_c    = float(np.sqrt(((preds_c - actual_c) ** 2).mean()))
    max_abs_c = float(errors_c.max())
    r2        = float(1.0 - np.sum((actual_c - preds_c)**2)
                           / np.sum((actual_c - actual_c.mean())**2))

    print(f"\n  LOCKED TEST:")
    print(f"    MSE (norm):   {test_mse:.4e}")
    print(f"    MAE  (°C):    {mae_c:.2f}")
    print(f"    RMSE (°C):    {rmse_c:.2f}")
    print(f"    Max abs (°C): {max_abs_c:.2f}")
    print(f"    R²:           {r2:.4f}")

    out_dir = REPO_ROOT / "results" / "baseline" / f"{split_mode}_{seed}" / config_name
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_dir / "model.pt")
    if "train_hist" in retrain_info:
        save_training_curves(retrain_info["train_hist"], retrain_info["val_hist"], out_dir)
    save_hpo_csv(study, out_dir)

    config = {
        "dataset":   config_name,
        "split_mode": split_mode,          # ← new: traceability
        "architecture": {
            "n_layers":   int(p["n_layers"]),
            "n_neurons":  int(p["n_neurons"]),
            "activation": p["activation"],
            "input_dim":  input_dim,
            "output_dim": 1,
        },
        "training_hparams": {
            "lr":           float(p["lr"]),
            "weight_decay": float(p["weight_decay"]),
            "optimizer":    "Adam",
            "batch_size":   int(p["batch_size"]),
            "batch_mode":   "mini-batch",
            "max_epochs":   retrain_epochs,
            "patience":     retrain_patience,
        },
        "n_params": n_params_final,
        "metrics": {
            "hpo_val_mse":          chosen.value,
            "overall_best_val_mse": overall_best.value,
            "retrain_val_mse":      retrain_val_mse,
            "test_mse_locked":      test_mse,
            "test_mae_celsius":     mae_c,
            "test_rmse_celsius":    rmse_c,
            "test_max_abs_celsius": max_abs_c,
            "test_r2":              r2,
        },
        "hpo": {
            "split_type":           f"profile-based / {split_mode}",
            "n_trials":             n_trials,
            "n_complete":           n_complete,
            "n_pruned_by_budget":   n_pruned,
            "hpo_pool_size":        x_pool.shape[0],
            "val_size":             x_val.shape[0],
            "test_size":            x_test.shape[0],
            "param_budget_soft_cap": PARAM_BUDGET_SOFT_CAP,
            "hpo_time_seconds":     round(hpo_time, 1),
        },
        "normalisation": {
            "y_mean_celsius": y_mean,
            "y_std_celsius":  y_std,
            "note": "multiply by y_std and add y_mean for °C",
        },
        "selection_rule":        f"smallest n_params within {SELECTION_TOLERANCE:.0%} of best val MSE",
        "selected_trial_number": chosen.number,
    }
    with open(out_dir / "config.json", "w") as fh:
        json.dump(config, fh, indent=2)

    metrics = {
        "config_name":          config_name,
        "split_mode":           split_mode,
        "input_dim":            input_dim,
        "n_params":             n_params_final,
        "architecture":         f"{p['n_layers']}×{p['n_neurons']} {p['activation']}",
        "test_mse_normalised":  test_mse,
        "test_mae_celsius":     mae_c,
        "test_rmse_celsius":    rmse_c,
        "test_max_abs_celsius": max_abs_c,
        "test_r2":              r2,
    }
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    

    print(f"\n  Saved to {out_dir}/")
    return config


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PMSM baseline MLPs with Optuna HPO.",
    )
    parser.add_argument("--csv", required=True,
                        help="Path to measures_v2.csv.")
    parser.add_argument("--configs", nargs="+", default=EWMA_CONFIG_NAMES,
                        help=f"Configs to train. Options: {EWMA_CONFIG_NAMES}")
    parser.add_argument("--n-trials",        type=int,   default=DEFAULT_N_TRIALS)
    parser.add_argument("--max-epochs",      type=int,   default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience",        type=int,   default=DEFAULT_PATIENCE)
    parser.add_argument("--retrain-epochs",  type=int,   default=DEFAULT_RETRAIN_EPOCHS)
    parser.add_argument("--retrain-patience",type=int,   default=DEFAULT_RETRAIN_PAT)
    parser.add_argument("--hpo-pool-size",   type=int,   default=DEFAULT_HPO_POOL_SIZE)
    parser.add_argument("--device",          default=None,
                        help="e.g. 'cuda' or 'cpu'")

    # ── Split parameters (mirror pmsm_loader_ewma CLI exactly) ───────────
    parser.add_argument(
        "--split-mode",
        choices=["random", "chronological", "diversity"],
        default="diversity",
        help=(
            "How to split profiles into train/val/test.  "
            "'diversity' (default): random test, then greedy max-diversity "
            "train covering the full operating envelope, val deduplication.  "
            "'random': seeded shuffle.  'chronological': sorted slice."
        ),
    )
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac",   type=float, default=0.15)
    parser.add_argument(
        "--dedup-threshold", type=float, default=0.85,
        help=(
            "[diversity mode] Val candidates with |r| >= this are merged: "
            "one representative stays in val, duplicates go to train. "
            "Default: 0.85"
        ),
    )
    parser.add_argument(
        "--fingerprint-cols", nargs="+", default=INPUT_FINGERPRINT_COLS,
        help=(
            "[diversity mode] Columns used to compute session similarity. "
            f"Default: {INPUT_FINGERPRINT_COLS}"
        ),
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    # ── EWMA parameters ───────────────────────────────────────────────────
    parser.add_argument("--add-ewma", action="store_true")
    parser.add_argument("--ewma-cols",  nargs="+", default=None,
                        help=f"Default: {DEFAULT_EWMA_COLS}")
    parser.add_argument("--ewma-spans", nargs="+", type=int, default=None,
                        help=f"Default: {DEFAULT_EWMA_SPANS}")
    parser.add_argument(
        "--add-physics", action="store_true",
        help=(
            "Add physics features before EWMA: i_squared, power_elec, "
            "power_mech, i_cross. Computed in the loader before EWMA so "
            "ewma(i_squared) correctly captures thermal accumulation."
        ),
    )
    args = parser.parse_args()

    # ── Ensure .pt splits exist and match requested parameters ────────────
    need_regen = not _splits_are_valid(
        config_name      = args.configs[0],   # proxy — all configs share the split
        add_ewma         = args.add_ewma,
        ewma_cols        = args.ewma_cols,
        ewma_spans       = args.ewma_spans,
        split_mode       = args.split_mode,
        dedup_threshold  = args.dedup_threshold,
        fingerprint_cols = args.fingerprint_cols,
        add_physics      = args.add_physics,
    )

    if need_regen:
        print("Generating PMSM .pt splits from CSV …")
        prepare_pmsm_data(
            args.csv,
            seed = args.seed,
            train_frac       = args.train_frac,
            val_frac         = args.val_frac,
            split_mode       = args.split_mode,
            add_ewma         = args.add_ewma,
            ewma_cols        = args.ewma_cols,
            ewma_spans       = args.ewma_spans,
            fingerprint_cols = args.fingerprint_cols,
            roll_stats=["mean"],
            add_physics      = args.add_physics,
        )
    else:
        print(f"PMSM .pt splits are up-to-date in {PROCESSED_DIR}/")

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}\n")

    results = {}
    for config_name in args.configs:
        if config_name not in ALL_CONFIG_NAMES:
            print(f"WARNING: skipping unknown config {config_name!r}. "
                  f"Known: {ALL_CONFIG_NAMES}")
            continue
        cfg = run_for_config(
            config_name,
            n_trials         = args.n_trials,
            max_epochs       = args.max_epochs,
            patience         = args.patience,
            retrain_epochs   = args.retrain_epochs,
            retrain_patience = args.retrain_patience,
            hpo_pool_size    = args.hpo_pool_size,
            device           = device,
            split_mode       = args.split_mode,   # for config.json traceability
            csv_path         = args.csv,
            add_ewma         = args.add_ewma,
            ewma_cols        = args.ewma_cols,
            ewma_spans       = args.ewma_spans,
            add_physics      = args.add_physics,
            seed             = args.seed,
        )
        results[config_name] = cfg

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Config':<26s}  {'Split':<13s}  {'Arch':>12s}  "
          f"{'Params':>7s}  {'RMSE°C':>7s}  {'R²':>6s}")
    print(f"  {'-'*26}  {'-'*13}  {'-'*12}  {'-'*7}  {'-'*7}  {'-'*6}")
    for name, c in results.items():
        m = c["metrics"]; a = c["architecture"]
        print(f"  {name:<26s}  {c['split_mode']:<13s}  "
              f"{a['n_layers']}×{a['n_neurons']} {a['activation']:>4s}  "
              f"{c['n_params']:>7d}  "
              f"{m['test_rmse_celsius']:>7.2f}  "
              f"{m['test_r2']:>6.4f}")
    print()


if __name__ == "__main__":
    main()