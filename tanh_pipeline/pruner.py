"""Iterative pruning driver — TANH-ONLY rerun (all 10 methods).

Self-contained copy of the 10-method, dim-aware, leakage-fixed pruner
(``src/pruning/pruner.py`` at commit 2482f7c) with three changes:

  1. Reads baselines from ``tanh_pipeline/results/baseline/`` and writes pruning
     outputs under ``tanh_pipeline/results/pruning/`` — the original ``results/``
     tree is untouched.
  2. ``FINETUNE_PATIENCE`` 10 -> 30, consistent with the raised baseline patience
     so fine-tuning tanh subnetworks gets room to converge.
  3. Dataset ``D`` is accepted (1-D, like A/B).

Methods: obd, obs, mag, struct, activ, redundancy, lowrank, taylor_fo, snip,
imp_rewind.

Adaptive sparsity schedule: start 10%, +10% per step. After each step fine-tune
surviving weights (early stop on a held-out val split carved from train — no
test leakage). Stop once test MSE > 3x baseline AND sparsity >= 50%.

Usage
-----
    python tanh_pipeline/pruner.py --method obd --dataset A --dim 1
    python tanh_pipeline/pruner.py --method snip --dataset D

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

OUT_ROOT = Path(__file__).resolve().parents[0]
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.generate_data import load_split
from src.models.mlp import MLP
import src.pruning.obd            as obd_mod
import src.pruning.obs            as obs_mod
import src.pruning.mag            as mag_mod
import src.pruning.struct_pruning as struct_mod
import src.pruning.activ_pruning  as activ_mod
import src.pruning.taylor_fo      as taylor_mod
import src.pruning.snip           as snip_mod
import src.pruning.imp_rewind     as imp_mod
from src.pruning.redundancy_pruning import RedundancyPruning
from src.pruning.low_rank_pruning   import LowRankPruning
from src.analysis.topology import plot_network_topology

INIT_SEED = 1337   # seed for the fresh-init checkpoint used by SNIP and IMP+rewind

FINETUNE_PATIENCE   = 30
FINETUNE_MAX_EPOCHS = 200
FINETUNE_BATCH_SIZE = 24
FINETUNE_VAL_FRAC   = 0.20
FINETUNE_VAL_SEED   = 4242
CALIB_SAMPLES       = 1000
CALIB_SEED          = 7373
SPARSITY_STEP       = 0.10
MIN_SPARSITY        = 0.50
MAX_SPARSITY        = 0.95
MSE_BLOWUP_FACTOR   = 3.0

STRUCTURED_METHODS = {"struct", "activ"}


def _device(arg_device: str | None) -> torch.device:
    if arg_device:
        return torch.device(arg_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_baseline(dataset: str, dim: int, device: torch.device) -> tuple[MLP, dict]:
    suffix = "" if dim == 1 else f"_{dim}d"
    base_dir = OUT_ROOT / "results" / "baseline" / f"{dataset}{suffix}"
    cfg_path = base_dir / "config.json"
    mdl_path = base_dir / "model.pt"
    if not cfg_path.exists() or not mdl_path.exists():
        raise FileNotFoundError(
            f"Baseline for dataset {dataset}{suffix} not found in {base_dir}. "
            f"Run `python tanh_pipeline/train_baseline.py --dataset {dataset} --dim {dim}` first."
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
    return model, config


def _split_train_val(x_tr, y_tr, val_frac, seed):
    """Carve a deterministic val split from the train pool for fine-tune early stopping."""
    n = x_tr.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = int(round(n * val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    return x_tr[tr_idx], y_tr[tr_idx], x_tr[val_idx], y_tr[val_idx]


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


def _finetune(model, mask, x_tr, y_tr, x_val, y_val, lr, batch_size, device) -> float:
    """Fine-tune surviving weights; early-stop on a held-out val split."""
    model.train()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(x_tr.to(device), y_tr.to(device)),
        batch_size=batch_size, shuffle=True,
    )
    best_mse, best_state, no_imp = float("inf"), None, 0
    for _ in range(FINETUNE_MAX_EPOCHS):
        for xb, yb in loader:
            optimiser.zero_grad()
            criterion(model(xb), yb).backward()
            optimiser.step()
            with torch.no_grad():
                w = model.get_weight_vector()
                w[~mask] = 0.0
                model.set_weights_from_vector(w)
        mse = _eval_mse(model, x_val, y_val, device)
        if mse < best_mse - 1e-9:
            best_mse   = mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp     = 0
        else:
            no_imp += 1
        if no_imp >= FINETUNE_PATIENCE:
            break
    if best_state:
        model.load_state_dict(best_state)
    return best_mse


def _save_checkpoint(model, mask, sparsity, test_mse, param_count, inference_ms, method, dataset_tag):
    dataset = dataset_tag
    out_dir = OUT_ROOT / "results" / "pruning" / method / dataset_tag
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


def run_pruning(method: str, dataset: str, dim: int, device_arg: str | None) -> None:
    device = _device(device_arg)
    suffix = "" if dim == 1 else f"_{dim}d"
    dataset_tag = f"{dataset}{suffix}"
    print(f"Method: {method.upper()}  |  Dataset: {dataset_tag}  |  Device: {device}")

    model, config = _load_baseline(dataset, dim, device)
    hparams    = config["training_hparams"]
    lr         = hparams["lr"]
    batch_size = FINETUNE_BATCH_SIZE

    out_dir_base = OUT_ROOT / "results" / "pruning" / method / dataset_tag
    out_dir_base.mkdir(parents=True, exist_ok=True)
    plot_network_topology(
        model,
        save_path=out_dir_base / "original_network.png",
        title="Topology Before Pruning",
    )

    init_state: dict | None = None
    if method in ("snip", "imp_rewind"):
        arch = config["architecture"]
        gen = torch.Generator(device="cpu").manual_seed(INIT_SEED)
        init_model = MLP(
            n_layers=arch["n_layers"],
            n_neurons=arch["n_neurons"],
            activation=arch["activation"],
            input_dim=arch.get("input_dim", 1),
            output_dim=arch.get("output_dim", 1),
        )
        with torch.no_grad():
            for p in init_model.parameters():
                fan_in = p.shape[-1] if p.ndim >= 2 else 1
                bound = (1.0 / fan_in) ** 0.5
                p.copy_(torch.empty(p.shape).uniform_(-bound, bound, generator=gen))
        init_state = {k: v.clone().to(device) for k, v in init_model.state_dict().items()}

    x_tr_full, y_tr_full = load_split(dataset, "train", dim=dim)
    x_te, y_te = load_split(dataset, "test", dim=dim)
    x_tr_full = x_tr_full.to(device); y_tr_full = y_tr_full.to(device)
    x_te = x_te.to(device); y_te = y_te.to(device)

    x_tr, y_tr, x_val, y_val = _split_train_val(
        x_tr_full, y_tr_full, FINETUNE_VAL_FRAC, FINETUNE_VAL_SEED,
    )

    n_calib = min(CALIB_SAMPLES, x_tr.shape[0])
    g_calib = torch.Generator(device="cpu").manual_seed(CALIB_SEED)
    calib_idx = torch.randperm(x_tr.shape[0], generator=g_calib)[:n_calib]
    x_calib = x_tr[calib_idx]
    y_calib = y_tr[calib_idx]

    total_weights    = model.get_weight_count()
    mask             = torch.ones(total_weights, dtype=torch.bool, device=device)
    current_sparsity = 0.0

    baseline_mse = _eval_mse(model, x_te, y_te, device)
    print(f"Baseline  |  params={total_weights}  |  test MSE={baseline_mse:.4e}")
    _print_header()

    redund_mod: RedundancyPruning | None = None
    lr_mod:     LowRankPruning    | None = None

    if method == "redundancy":
        redund_mod = RedundancyPruning(model, x_calib=x_calib, device=device)
    elif method == "lowrank":
        lr_mod = LowRankPruning(model, device=device)
        lr_current_sparsity: float = lr_mod.rank_sparsity()

    obs_diag_fallback = False

    target = SPARSITY_STEP
    while target <= MAX_SPARSITY + 1e-9:
        print(f"\n── Target sparsity: {target * 100:.0f}% ──────────────────────", flush=True)

        if method == "obd":
            h_diag = obd_mod.compute_hessian_diag(model, x_calib, y_calib, device)
            h_inv  = None

        elif method == "obs":
            if obs_diag_fallback:
                # already committed to OBD-diagonal fallback for the rest of the
                # run; a later iteration's H_inv may invert fine, but we must
                # keep h_diag populated since selection routes through the OBD path.
                h_inv = None
                h_diag = obd_mod.compute_hessian_diag(model, x_calib, y_calib, device)
            else:
                h_inv = obs_mod.compute_hessian_inv(model, x_calib, y_calib, device)
                if h_inv is None:
                    print("[OBS] Falling back to Fisher-diagonal OBD saliency.")
                    obs_diag_fallback = True
                    h_diag = obd_mod.compute_hessian_diag(model, x_calib, y_calib, device)
                else:
                    h_diag = None

        elif method == "redundancy":
            redund_mod.compute_correlations(
                target_sparsity  = target,
                current_sparsity = current_sparsity,
                total_params     = total_weights,
            )
            h_diag = h_inv = None

        elif method == "taylor_fo":
            taylor_saliency = taylor_mod.compute_saliency(model, x_calib, y_calib, device)
            h_diag = h_inv = None

        elif method == "snip":
            # SNIP is single-shot: reset to fresh init, score, top-k mask in one shot.
            assert init_state is not None
            model.load_state_dict(init_state)
            scores = snip_mod.compute_scores(model, x_calib, y_calib, device)
            mask = snip_mod.snip_mask_for_target(scores, target).to(device)
            w = model.get_weight_vector().to(device)
            w[~mask] = 0.0
            model.set_weights_from_vector(w)
            current_sparsity = 1.0 - mask.sum().item() / total_weights

            print("  [SNIP] Fine-tuning from masked init...", flush=True)
            ft_mse       = _finetune(model, mask, x_tr, y_tr, x_val, y_val, lr, batch_size, device)
            param_count  = int(mask.sum().item())
            inference_ms = _measure_inference_ms(model, x_te, device)
            test_mse     = _eval_mse(model, x_te, y_te, device)

            tag = f"after_pruning_sparsity_{int(round(current_sparsity * 100)):02d}.png"
            plot_network_topology(model, save_path=out_dir_base / tag, title="Topology After Pruning")

            _print_row(target, param_count, test_mse, baseline_mse, inference_ms)
            _save_checkpoint(model, mask, target, test_mse, param_count, inference_ms, method, dataset_tag)

            if test_mse > MSE_BLOWUP_FACTOR * baseline_mse and target >= MIN_SPARSITY - 1e-9:
                print(f"\nMSE > {MSE_BLOWUP_FACTOR}× baseline at {target*100:.0f}% — stopping.")
                break

            target = round(target + SPARSITY_STEP, 2)
            continue

        elif method == "lowrank":
            steps_needed = lr_mod.svs_to_prune_for_target(target, lr_current_sparsity)
            print(
                f"  [LowRank] Rank sparsity: {lr_current_sparsity*100:.1f}% → {target*100:.0f}%"
                f" ({steps_needed} SV(s) to prune)",
                flush=True,
            )
            for _ in range(steps_needed):
                lr_mod.prune_one_sv()

            lr_current_sparsity = lr_mod.rank_sparsity()

            print("  [LowRank] Fine-tuning (UV model)...", flush=True)
            rep_model = lr_mod.get_reparameterised().to(device)
            ft_mse = lr_mod.finetune(
                rep_model, x_tr, y_tr, x_val, y_val, lr, batch_size, device
            )
            lr_mod.write_back()
            lr_mod.enforce_lowrank()

            param_count  = lr_mod.total_sv_params()
            inference_ms = _measure_inference_ms(model, x_te, device)
            test_mse     = _eval_mse(model, x_te, y_te, device)

            rank_info = lr_mod.get_rank_info()
            rank_str  = ", ".join(f"{n}:r={r}/{f}" for n, (r, f) in rank_info.items())
            print(f"  [LowRank] Ranks after step: {rank_str}", flush=True)

            tag = f"after_pruning_sparsity_{int(round(lr_current_sparsity * 100)):02d}.png"
            plot_network_topology(model, save_path=out_dir_base / tag, title="Topology After Pruning")

            _print_row(target, param_count, test_mse, baseline_mse, inference_ms)
            _save_checkpoint(model, mask, lr_current_sparsity, test_mse, param_count, inference_ms, method, dataset_tag)

            if test_mse > MSE_BLOWUP_FACTOR * baseline_mse and target >= MIN_SPARSITY - 1e-9:
                print(f"\nMSE > {MSE_BLOWUP_FACTOR}× baseline at {target*100:.0f}% — stopping.")
                break

            target = round(target + SPARSITY_STEP, 2)
            continue

        # ── Inner weight-removal loop (obd/obs/mag/struct/activ/redundancy/
        #     taylor_fo/imp_rewind) ──────────────────────────────────────────
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
                indices = activ_mod.select_neuron_weights(model, x_calib, device, mask)
                if not indices:
                    break
                for q in indices:
                    w[q] = 0.0
                    mask[q] = False

            elif method == "mag":
                q = mag_mod.select_weight(w, mask)
                w[q] = 0.0
                mask[q] = False

            elif method == "imp_rewind":
                q = imp_mod.select_weight(w, mask)
                w[q] = 0.0
                mask[q] = False

            elif method == "taylor_fo":
                q = taylor_mod.select_weight(w, mask, taylor_saliency)
                w[q] = 0.0
                mask[q] = False

            elif method == "obs" and not obs_diag_fallback:
                q, w = obs_mod.select_and_correct(w, mask, h_inv)
                h_inv = obs_mod.update_h_inv(h_inv, q)
                mask[q] = False

            elif method == "redundancy":
                q_start, q_end = redund_mod.select_weight(w, mask)
                w[q_start:q_end] = 0.0
                mask[q_start:q_end] = False

            else:   # obd (and obs fallback)
                q = obd_mod.select_weight(w, mask, h_diag)
                w[q] = 0.0
                mask[q] = False

            model.set_weights_from_vector(w)
            current_sparsity = 1.0 - mask.sum().item() / total_weights

        # ── IMP+rewind: rewind surviving weights to fresh init before retrain ──
        if method == "imp_rewind":
            assert init_state is not None
            model.load_state_dict(init_state)
            w = model.get_weight_vector().to(device)
            w[~mask] = 0.0
            model.set_weights_from_vector(w)

        # ── Fine-tune + evaluate ──────────────────────────────────────────
        print("  Fine-tuning...", flush=True)
        ft_mse       = _finetune(model, mask, x_tr, y_tr, x_val, y_val, lr, batch_size, device)
        param_count  = int(mask.sum().item())
        inference_ms = _measure_inference_ms(model, x_te, device)
        test_mse     = _eval_mse(model, x_te, y_te, device)

        tag = f"after_pruning_sparsity_{int(round(current_sparsity * 100)):02d}.png"
        plot_network_topology(model, save_path=out_dir_base / tag, title="Topology After Pruning")

        _print_row(target, param_count, test_mse, baseline_mse, inference_ms)
        _save_checkpoint(model, mask, target, test_mse, param_count, inference_ms, method, dataset_tag)

        if test_mse > MSE_BLOWUP_FACTOR * baseline_mse and target >= MIN_SPARSITY - 1e-9:
            print(f"\nMSE > {MSE_BLOWUP_FACTOR}× baseline at {target*100:.0f}% — stopping.")
            break

        target = round(target + SPARSITY_STEP, 2)

    print(f"\nDone. Results → tanh_pipeline/results/pruning/{method}/{dataset_tag}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Iterative pruning (tanh-only rerun).")
    parser.add_argument(
        "--method",
        choices=[
            "obd", "obs", "mag", "struct", "activ", "redundancy", "lowrank",
            "taylor_fo", "snip", "imp_rewind",
        ],
        required=True,
    )
    parser.add_argument("--dataset", choices=["A", "B", "D"], required=True)
    parser.add_argument("--dim", type=int, choices=[1, 2, 3], default=1,
                        help="Input dimensionality (must match the trained baseline).")
    parser.add_argument("--device",  default=None,
                        help="Override device (e.g. 'cpu', 'mps', 'cuda').")
    args = parser.parse_args()
    run_pruning(args.method, args.dataset, args.dim, args.device)


if __name__ == "__main__":
    main()
