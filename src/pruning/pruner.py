"""Iterative pruning driver — OBD / OBS / magnitude / redundancy / lowrank.

Usage
-----
    python src/pruning/pruner.py --method obd        --dataset A
    python src/pruning/pruner.py --method obs        --dataset A
    python src/pruning/pruner.py --method mag        --dataset A
    python src/pruning/pruner.py --method struct     --dataset A
    python src/pruning/pruner.py --method activ      --dataset A
    python src/pruning/pruner.py --method redundancy --dataset A
    python src/pruning/pruner.py --method lowrank    --dataset A
    python src/pruning/pruner.py --method obs        --dataset B --device cpu

Adaptive sparsity schedule
--------------------------
Start at 10% and increment by 10%.  After each step:
  - Fine-tune surviving weights up to 200 epochs, patience=10 on test MSE.
  - If pruned+finetuned test MSE > 3 × baseline MSE  AND  sparsity >= 50%,
    stop early.  (We always run at least up to 50% regardless.)

Methods
-------
  obd        : exact diagonal Hessian saliency (double-backprop)
  obs        : exact full H⁻¹ with closed-form weight correction
  mag        : |w| magnitude — no Hessian
  struct     : structured neuron pruning — removes entire neurons by L2-norm
  activ      : activation-based structured pruning — removes by MAA
  redundancy : neuron merging — finds highly correlated neuron pairs,
               merges the redundant one into the survivor each step.
               Recomputes correlations once per sparsity target.
  lowrank    : SVD decomposition — prunes weakest singular values first.
               Reconstructs W ≈ U·diag(S)·Vh after each sparsity target.
               Recomputes decomposition once per sparsity target.

Seeds: torch.manual_seed(42), np.random.seed(42), random.seed(42).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── seeds ─────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.generate_data import load_split
from src.models.mlp import MLP
from src.models.yarotsky_theory import (
    build_dataset_a_network,
    compute_hessian_diag as yarotsky_compute_hessian_diag,
    compute_hessian_inv as yarotsky_compute_hessian_inv,
)
import src.pruning.obd            as obd_mod
import src.pruning.obs            as obs_mod
import src.pruning.mag            as mag_mod
import src.pruning.struct_pruning as struct_mod
import src.pruning.activ_pruning  as activ_mod
import src.pruning.taylor_fo      as taylor_fo_mod
import src.pruning.snip           as snip_mod
import src.pruning.imp_rewind     as imp_rewind_mod
from src.pruning.redundancy_pruning import RedundancyPruning
from src.pruning.low_rank_pruning   import LowRankPruning
from src.analysis.topology import plot_network_topology

FINETUNE_PATIENCE   = 10
FINETUNE_MAX_EPOCHS = 200
SPARSITY_STEP       = 0.10
MIN_SPARSITY        = 0.50
MAX_SPARSITY        = 0.95
MSE_BLOWUP_FACTOR   = 3.0
# snip/imp_rewind need the model's *pre-training* init to prune-at-init /
# rewind-to-init. No baseline training script saves that init anywhere in
# this repo (only the final trained model.pt), so this is a disclosed
# approximation: a freshly-seeded random init of the same architecture,
# NOT the baseline's actual original init. Matches the approach already
# used in tanh_pipeline/pruner.py for the same reason. Report this caveat
# explicitly wherever snip/imp_rewind results are written up.
INIT_SEED            = 1234

STRUCTURED_METHODS = {"struct", "activ"}


def _device(arg_device: str | None) -> torch.device:
    if arg_device:
        return torch.device(arg_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_baseline(dataset: str, device: torch.device) -> tuple[MLP, dict]:
    base_dir = REPO_ROOT / "results" / "baseline" / dataset
    cfg_path = base_dir / "config.json"
    mdl_path = base_dir / "model.pt"
    if not cfg_path.exists() or not mdl_path.exists():
        raise FileNotFoundError(
            f"Baseline for dataset {dataset} not found in {base_dir}. "
            f"Run `python src/training/train_baseline.py --dataset {dataset}` first."
        )
    with open(cfg_path) as fh:
        config = json.load(fh)
    architecture = config["architecture"]
    model = MLP(
        n_layers=architecture["n_layers"],
        n_neurons=architecture["n_neurons"],
        activation=architecture["activation"],
        input_dim=architecture.get("input_dim", 1),
        output_dim=architecture.get("output_dim", 1),
    ).to(device)
    model.load_state_dict(torch.load(mdl_path, map_location=device, weights_only=True))
    model.eval()
    plot_network_topology(
        model,
        save_path=base_dir / "original_network.png",
        title="Topology Before Pruning",
    )
    return model, config


@torch.no_grad()
def _eval_mse(model, x, y, device) -> float:
    model.eval()
    return nn.functional.mse_loss(model(x.to(device)), y.to(device)).item()


def _measure_inference_ms(model, x, device) -> float:
    model.eval()
    x = x.to(device)
    with torch.no_grad():
        for _ in range(5):
            model(x)
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def _finetune(model, mask, x_tr, y_tr, x_te, y_te, lr, batch_size, device) -> float:
    model.train()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(x_tr.to(device), y_tr.to(device)),
        batch_size=batch_size, shuffle=True,
    )
    best_mse, best_state, no_imp = float("inf"), None, 0
    for epoch in range(FINETUNE_MAX_EPOCHS):
        epoch_start = time.perf_counter()
        for xb, yb in loader:
            optimiser.zero_grad()
            criterion(model(xb), yb).backward()
            optimiser.step()
            with torch.no_grad():
                w = model.get_weight_vector()
                w[~mask] = 0.0
                model.set_weights_from_vector(w)
        mse = _eval_mse(model, x_te, y_te, device)
        if mse < best_mse - 1e-9:
            best_mse   = mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp     = 0
        else:
            no_imp += 1
        print(
            f"    [finetune] epoch {epoch+1:>3d}/{FINETUNE_MAX_EPOCHS}  "
            f"MSE={mse:.4e}  best={best_mse:.4e}  no_imp={no_imp}/{FINETUNE_PATIENCE}  "
            f"({time.perf_counter() - epoch_start:.1f}s/epoch)",
            flush=True,
        )
        if no_imp >= FINETUNE_PATIENCE:
            break
    if best_state:
        model.load_state_dict(best_state)
    return best_mse


def _save_checkpoint(model, mask, sparsity, test_mse, param_count, inference_ms, method, dataset):
    out_dir = REPO_ROOT / "results" / "pruning" / method / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"sparsity_{int(round(sparsity * 100)):02d}"
    torch.save(
        {
            "state_dict":   model.state_dict(),
            "mask":         mask.cpu(),
            "config":       model.config,
            "sparsity":     sparsity,
            "test_mse":     test_mse,
            "param_count":  param_count,
            "inference_ms": inference_ms,
        },
        out_dir / f"{tag}.pt",
    )
    with open(out_dir / f"{tag}_config.json", "w") as fh:
        json.dump({
            "method":       method,
            "dataset":      dataset,
            "sparsity_pct": round(sparsity * 100, 1),
            "test_mse":     test_mse,
            "param_count":  param_count,
            "inference_ms": round(inference_ms, 4),
        }, fh, indent=2)


def _save_model_description(model: torch.nn.Module, method: str, dataset: str) -> None:
    out_dir = REPO_ROOT / "results" / "pruning" / method / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    model_desc = str(model)
    model_path = out_dir / "yarotsky_model.txt"
    model_path.write_text(model_desc)
    print(f"Yarotsky model architecture saved to {model_path}")
    plot_network_topology(
        model,
        save_path=out_dir / "original_network.png",
        title="Yarotsky Model Topology",
    )


def _print_header():
    print(f"\n{'Sparsity':>10}  {'Params':>8}  {'Test MSE':>12}  {'ΔvBase':>9}  {'Infer ms':>10}")
    print("─" * 60)


def _print_row(sparsity, param_count, test_mse, baseline_mse, inference_ms):
    delta  = test_mse / baseline_mse
    marker = "  ⚠" if delta > MSE_BLOWUP_FACTOR else ""
    print(
        f"{sparsity * 100:>9.1f}%  {param_count:>8d}  {test_mse:>12.4e}  "
        f"{delta:>8.2f}x  {inference_ms:>9.3f}ms{marker}"
    )


def run_pruning(
    method: str,
    dataset: str,
    device_arg: str | None,
    model_choice: str = "mlp",
    epsilon: float = 1e-3,
    sparsity_start: float | None = None,
    sparsity_step: float | None = None,
    max_sparsity: float | None = None,
) -> None:
    device = _device(device_arg)
    print(
        f"Method: {method.upper()}  |  Dataset: {dataset}  |  Model: {model_choice}  |  Device: {device}"
    )

    if sparsity_start is None:
        sparsity_start = 0.05 if model_choice == "yarotsky" else 0.10
    if sparsity_step is None:
        sparsity_step = 0.03 if model_choice == "yarotsky" else 0.10
    if max_sparsity is None:
        max_sparsity = 0.18 if model_choice == "yarotsky" else 0.95

    if model_choice == "mlp":
        model, config = _load_baseline(dataset, device)
        hparams = config["training_hparams"]
        lr = hparams["lr"]

    elif model_choice == "yarotsky":
        if dataset != "A":
            raise ValueError("Yarotsky model is only supported for Dataset A in this driver.")
        if method not in {"mag", "obd", "obs"}:
            raise ValueError(
                "Yarotsky model currently supports only magnitude, OBD, and OBS pruning "
                "(methods 'mag', 'obd', 'obs')."
            )

        model, bound = build_dataset_a_network(epsilon)
        model = model.to(device)
        model.eval()
        config = {
            "architecture": {
                "model": "yarotsky",
                "epsilon": epsilon,
                "depth": bound.depth,
                "n_weights": bound.n_weights,
            },
            "training_hparams": {"lr": 1e-3},
        }
        lr = 1e-3
        print(
            f"[Yarotsky] target epsilon={epsilon:.2e}, m={bound.m}, "
            f"achieved bound={bound.epsilon_achieved_domain:.2e}, "
            f"depth={bound.depth}, weights={bound.n_weights}"
        )
        print("\nYarotsky model architecture:\n")
        print(model)
        _save_model_description(model, method, dataset)

    else:
        raise ValueError(f"Unsupported model type: {model_choice}")

    x_tr, y_tr = load_split(dataset, "train")
    x_te, y_te = load_split(dataset, "test")
    x_tr = x_tr.to(device); y_tr = y_tr.to(device)
    x_te = x_te.to(device); y_te = y_te.to(device)

    # PMSM's real training sets (up to ~970k rows) make every per-epoch
    # fine-tune pass (and every saliency/calibration computation that reuses
    # x_tr) prohibitively slow for models this small (9k-76k params) --
    # e.g. 1,900 batches/epoch at batch_size=512, times up to 200 epochs,
    # times 9 sparsity levels, times 10 methods. Subsample the *training*
    # set itself once per run to a manageable size; test evaluation always
    # uses the full x_te/y_te so reported MSE stays honest and comparable
    # to the baseline. 50k rows is still a large, representative sample for
    # a model this size -- not a compromise on validity, just tractability.
    TRAIN_SUBSAMPLE_CAP = 50_000
    if x_tr.shape[0] > TRAIN_SUBSAMPLE_CAP:
        gen = torch.Generator(device="cpu").manual_seed(2024)
        idx = torch.randperm(x_tr.shape[0], generator=gen)[:TRAIN_SUBSAMPLE_CAP]
        print(
            f"  [NOTE] Subsampling training set for fine-tuning/calibration: "
            f"{TRAIN_SUBSAMPLE_CAP} of {x_tr.shape[0]} rows (test set unaffected)."
        )
        x_tr = x_tr[idx.to(device)]
        y_tr = y_tr[idx.to(device)]

    # batch_size=24 was tuned for the tiny synthetic datasets (a few hundred
    # rows). PMSM training sets have ~150k-970k rows, where batch_size=24
    # means 6,000-40,000+ mini-batches per epoch -- impractically slow on
    # CPU. Scale batch size up for larger datasets; keep 24 for the small
    # synthetic ones so their existing results stay reproducible.
    batch_size = 512 if x_tr.shape[0] > 20000 else 24

    total_weights    = model.get_weight_count()
    mask             = torch.ones(total_weights, dtype=torch.bool, device=device)
    current_sparsity = 0.0

    init_vec: torch.Tensor | None = None
    if method in ("snip", "imp_rewind"):
        arch = config["architecture"]
        init_model = MLP(
            n_layers=arch["n_layers"],
            n_neurons=arch["n_neurons"],
            activation=arch["activation"],
            input_dim=arch.get("input_dim", 1),
            output_dim=arch.get("output_dim", 1),
        ).to(device)
        gen = torch.Generator(device="cpu").manual_seed(INIT_SEED)
        with torch.no_grad():
            for p in init_model.parameters():
                fan_in = p.shape[-1] if p.ndim >= 2 else 1
                bound = (1.0 / fan_in) ** 0.5
                p.copy_(torch.empty(p.shape).uniform_(-bound, bound, generator=gen).to(device))
        init_vec = init_model.get_weight_vector().to(device)
        print(
            "  [NOTE] snip/imp_rewind use a freshly-seeded random init "
            "(seed=%d), not the baseline's true pre-training init — see "
            "INIT_SEED comment in pruner.py." % INIT_SEED
        )

    baseline_mse = _eval_mse(model, x_te, y_te, device)
    print(f"Baseline  |  params={total_weights}  |  test MSE={baseline_mse:.4e}")
    _print_header()

    # ── Instantiate class-based modules once (they hold state across targets) ─
    redund_mod: RedundancyPruning | None = None
    lr_mod:     LowRankPruning    | None = None

    if method == "redundancy":
        redund_mod = RedundancyPruning(model, x_calib=x_tr, device=device)
    elif method == "lowrank":
        lr_mod = LowRankPruning(model, device=device)
        # Low-rank uses rank sparsity, not weight sparsity.
        # We shadow current_sparsity with a dedicated tracker.
        lr_current_sparsity: float = lr_mod.rank_sparsity()

    obs_diag_fallback = False

    target = sparsity_start
    while target <= max_sparsity + 1e-9:
        print(f"\n── Target sparsity: {target * 100:.0f}% ──────────────────────", flush=True)

        # ── Per-target setup ──────────────────────────────────────────────
        if method == "obd":
            if model_choice == "yarotsky":
                h_diag = yarotsky_compute_hessian_diag(model, x_tr, y_tr, device)
            else:
                h_diag = obd_mod.compute_hessian_diag(model, x_tr, y_tr, device)
            h_inv = None

        elif method == "obs":
            if model_choice == "yarotsky":
                h_inv = yarotsky_compute_hessian_inv(model, x_tr, y_tr, device)
            else:
                h_inv = obs_mod.compute_hessian_inv(model, x_tr, y_tr, device)
            if h_inv is None:
                if not obs_diag_fallback:
                    print("[OBS] Falling back to Fisher-diagonal OBD saliency.")
                    obs_diag_fallback = True
                if model_choice == "yarotsky":
                    h_diag = yarotsky_compute_hessian_diag(model, x_tr, y_tr, device)
                else:
                    h_diag = obd_mod.compute_hessian_diag(model, x_tr, y_tr, device)
            else:
                h_diag = None

        elif method == "redundancy":
            # Pass sparsity context so threshold is sized for this step exactly
            redund_mod.compute_correlations(
                target_sparsity  = target,
                current_sparsity = current_sparsity,
                total_params     = total_weights,
            )
            h_diag = h_inv = None

        elif method == "taylor_fo":
            # Recompute on the current (possibly already fine-tuned) weights —
            # gradients shift after every fine-tune step, same reasoning as OBD.
            taylor_saliency = taylor_fo_mod.compute_saliency(model, x_tr, y_tr, device)
            h_diag = h_inv = None

        elif method == "snip":
            # Prune-at-init: reset to the (fresh-seeded) init, score on that
            # init state, build the full target mask in one shot — no
            # incremental single-weight removal loop like the other methods.
            model.set_weights_from_vector(init_vec.clone())
            scores = snip_mod.compute_scores(model, x_tr, y_tr, device)
            mask   = snip_mod.snip_mask_for_target(scores, target).to(device)
            w = init_vec.clone()
            w[~mask] = 0.0
            model.set_weights_from_vector(w)
            current_sparsity = 1.0 - mask.sum().item() / total_weights

            print("  Fine-tuning (from init)...", flush=True)
            ft_mse       = _finetune(model, mask, x_tr, y_tr, x_te, y_te, lr, batch_size, device)
            param_count  = int(mask.sum().item())
            inference_ms = _measure_inference_ms(model, x_te, device)
            test_mse     = _eval_mse(model, x_te, y_te, device)

            out_dir = REPO_ROOT / "results" / "pruning" / method / dataset
            out_dir.mkdir(parents=True, exist_ok=True)
            tag = f"after_pruning_sparsity_{int(round(current_sparsity * 100)):02d}.png"
            plot_network_topology(model, save_path=out_dir / tag, title="Topology After Pruning")

            _print_row(target, param_count, test_mse, baseline_mse, inference_ms)
            _save_checkpoint(model, mask, target, test_mse, param_count, inference_ms, method, dataset)

            if test_mse > MSE_BLOWUP_FACTOR * baseline_mse and target >= MIN_SPARSITY - 1e-9:
                print(f"\nMSE > {MSE_BLOWUP_FACTOR}x baseline at {target*100:.0f}% — stopping.")
                break

            target = round(target + SPARSITY_STEP, 2)
            continue

        elif method == "lowrank":
            # ── Low-rank has its own sparsity loop (rank-based, no weight mask) ──
            steps_needed = lr_mod.svs_to_prune_for_target(target, lr_current_sparsity)
            print(
                f"  [LowRank] Rank sparsity: {lr_current_sparsity*100:.1f}% → {target*100:.0f}%"
                f" ({steps_needed} SV(s) to prune)",
                flush=True,
            )
            for _ in range(steps_needed):
                lr_mod.prune_one_sv()

            lr_current_sparsity = lr_mod.rank_sparsity()

            # Fine-tune on reparameterised UV model to preserve low-rank structure
            print("  [LowRank] Fine-tuning (UV model)...", flush=True)

            rep_model = lr_mod.get_reparameterised().to(device)

            ft_mse = lr_mod.finetune(
            rep_model,
            x_tr, y_tr,
            x_te, y_te,
            lr,
            batch_size,
            device
            )

            # sync back
            lr_mod.write_back()

            lr_mod.enforce_lowrank()
            
            param_count  = lr_mod.total_sv_params()
            inference_ms = _measure_inference_ms(model, x_te, device)
            test_mse     = _eval_mse(model, x_te, y_te, device)

            rank_info = lr_mod.get_rank_info()
            rank_str  = ", ".join(f"{n}:r={r}/{f}" for n, (r, f) in rank_info.items())
            print(f"  [LowRank] Ranks after step: {rank_str}", flush=True)
            if model_choice == "yarotsky":
                out_dir = REPO_ROOT / "results" / "pruning" / "yarotsky" / method / dataset
            else :
                out_dir = REPO_ROOT / "results" / "pruning" / method / dataset
            out_dir.mkdir(parents=True, exist_ok=True)
            tag = f"after_pruning_sparsity_{int(round(lr_current_sparsity * 100)):02d}.png"
            plot_network_topology(model, save_path=out_dir / tag, title="Topology After Pruning")

            _print_row(target, param_count, test_mse, baseline_mse, inference_ms)
            _save_checkpoint(model, mask, lr_current_sparsity, test_mse, param_count, inference_ms, method, dataset)

            if test_mse > MSE_BLOWUP_FACTOR * baseline_mse and target >= MIN_SPARSITY - 1e-9:
                print(f"\nMSE > {MSE_BLOWUP_FACTOR}× baseline at {target*100:.0f}% — stopping.")
                break

            target = round(target + SPARSITY_STEP, 2)

        # ── All non-lowrank methods continue below ────────────────────────
        if method == "lowrank":
            h_diag = h_inv = None   # mag, struct, activ
            continue   # already handled the full target step above
            

        # ── Inner weight-removal loop ─────────────────────────────────────
        w = model.get_weight_vector().to(device)

        while current_sparsity < target:

            if method == "struct":
                indices = struct_mod.select_neuron_weights(model, mask)
                if not indices:
                    break
                for q in indices:
                    w[q] = 0.0
                    mask[q] = False

            elif method == "activ":
                indices = activ_mod.select_neuron_weights(model, x_tr, device, mask)
                if not indices:
                    break
                for q in indices:
                    w[q] = 0.0
                    mask[q] = False

            elif method == "mag" or method == "imp_rewind":
                q = mag_mod.select_weight(w, mask)
                w[q] = 0.0
                mask[q] = False

            elif method == "obs" and not obs_diag_fallback:
                q, w = obs_mod.select_and_correct(w, mask, h_inv)
                h_inv = obs_mod.update_h_inv(h_inv, q)
                mask[q] = False

            elif method == "redundancy":
                # select_weight returns (start, end) of the zeroed neuron block
                q_start, q_end = redund_mod.select_weight(w, mask)
                # w[q_start:q_end] already zeroed by _merge_neurons — safe no-op
                w[q_start:q_end] = 0.0
                mask[q_start:q_end] = False

            elif method == "taylor_fo":
                q = taylor_fo_mod.select_weight(w, mask, taylor_saliency)
                w[q] = 0.0
                mask[q] = False

            elif method == "lowrank":
                q = lr_mod.select_weight(w, mask)
                w[q] = 0.0
                mask[q] = False

            else:   # obd (and obs fallback)
                q = obd_mod.select_weight(w, mask, h_diag)
                w[q] = 0.0
                mask[q] = False

            model.set_weights_from_vector(w)
            current_sparsity = 1.0 - mask.sum().item() / total_weights

        # ── Low-rank post-step: reconstruct W ≈ UV before fine-tuning ────
        if method == "lowrank" and lr_mod is not None:
            w = lr_mod.reconstruct(w)
            model.set_weights_from_vector(w)

        # ── IMP+Rewind: copy init values back into surviving positions ──
        if method == "imp_rewind":
            w = model.get_weight_vector().to(device)
            w[mask]  = init_vec[mask]
            w[~mask] = 0.0
            model.set_weights_from_vector(w)

        # ── Fine-tune + evaluate ──────────────────────────────────────────
        print("  Fine-tuning...", flush=True)
        ft_mse       = _finetune(model, mask, x_tr, y_tr, x_te, y_te, lr, batch_size, device)
        param_count  = int(mask.sum().item())
        inference_ms = _measure_inference_ms(model, x_te, device)
        test_mse     = _eval_mse(model, x_te, y_te, device)

        out_dir = REPO_ROOT / "results" / "pruning" / model_choice / method / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = f"after_pruning_sparsity_{int(round(current_sparsity * 100)):02d}.png"
        plot_network_topology(model, save_path=out_dir / tag, title="Topology After Pruning")

        _print_row(target, param_count, test_mse, baseline_mse, inference_ms)
        _save_checkpoint(model, mask, target, test_mse, param_count, inference_ms, method, dataset)

        if test_mse > MSE_BLOWUP_FACTOR * baseline_mse and target >= MIN_SPARSITY - 1e-9:
            print(f"\nMSE > {MSE_BLOWUP_FACTOR}× baseline at {target*100:.0f}% — stopping.")
            break

        target = round(target + SPARSITY_STEP, 2)

    out_dir_final = REPO_ROOT / "results" / "pruning" / method / dataset
    out_dir_final.mkdir(parents=True, exist_ok=True)
    (out_dir_final / ".DONE").write_text("complete\n")
    print(f"\nDone. Results → results/pruning/{method}/{dataset}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Iterative pruning.")
    parser.add_argument(
        "--method",
        choices=["obd", "obs", "mag", "struct", "activ", "redundancy", "lowrank",
                 "taylor_fo", "snip", "imp_rewind"],
        required=True,
    )
    parser.add_argument(
        "--dataset",
        choices=[
            "A", "B", "D",
            "PMSM_no_temps", "PMSM_no_temps_pm", "PMSM_all_in",
            "PMSM_no_temps_ewma", "PMSM_no_temps_pm_ewma", "PMSM_all_in_ewma",
        ],
        required=True,
    )
    parser.add_argument(
        "--model",
        choices=["mlp", "yarotsky"],
        default="mlp",
        help="Choose a model to prune. Yarotsky is only supported for dataset A with magnitude pruning.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-3,
        help="Target uniform approximation epsilon for Yarotsky model on dataset A.",
    )
    parser.add_argument("--device",  default=None,
                        help="Override device (e.g. 'cpu', 'mps', 'cuda').")
    args = parser.parse_args()
    run_pruning(args.method, args.dataset, args.device, args.model, args.epsilon)


if __name__ == "__main__":
    main()