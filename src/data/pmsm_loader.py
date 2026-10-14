"""PMSM dataset loader — profile-based split, z-score normalisation, .pt persistence.

This module handles the full pipeline from raw CSV to the same ``{'x': tensor,
'y': tensor}`` dictionaries that ``generate_data.py`` produces for the synthetic
datasets.  The key difference is the **profile-based** train / val / test split
which prevents temporal leakage between adjacent 1 Hz samples.

Normalisation
-------------
Z-score (mean=0, std=1) computed on the **training profiles only**.  The same
(mean, std) are applied to val and test.  Statistics are saved to a JSON
sidecar so downstream code can denormalise predictions back to °C.

Usage
-----
    # Prepare all three configs from a CSV path:
    python -m src.data.pmsm_loader --csv path/to/measures_v2.csv

    # Or from Python:
    from src.data.pmsm_loader import prepare_all_configs, load_pmsm_split
    prepare_all_configs("path/to/measures_v2.csv")
    x_tr, y_tr = load_pmsm_split("PMSM_no_temps", "train")
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.pmsm_configs import CONFIGS, TARGET, ALL_CONFIG_NAMES, get_config  # noqa: E402

PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# ── profile-based split ──────────────────────────────────────────────────────


def make_profile_split(
    df: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.10,
) -> dict[str, list[int]]:
    """Partition profile_ids into train / val / test sets (leakage-safe).

    Args:
        df:         DataFrame with a ``profile_id`` column.
        seed:       Random seed for the permutation.
        train_frac: Fraction of profiles for training.
        val_frac:   Fraction of profiles for validation.

    Returns:
        Dict mapping ``'train'``, ``'val'``, ``'test'`` to lists of profile_ids.
    """
    pids = np.array(sorted(df["profile_id"].unique()))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pids))

    n_tr = int(round(len(pids) * train_frac))
    n_va = int(round(len(pids) * val_frac))

    split_ids = {
        "train": sorted(pids[perm[:n_tr]].tolist()),
        "val":   sorted(pids[perm[n_tr : n_tr + n_va]].tolist()),
        "test":  sorted(pids[perm[n_tr + n_va :]].tolist()),
    }

    # Sanity: every profile assigned exactly once.
    all_assigned = set(split_ids["train"]) | set(split_ids["val"]) | set(split_ids["test"])
    assert all_assigned == set(pids.tolist()), "Profile partition is not exhaustive."

    return split_ids


# ── normalisation ─────────────────────────────────────────────────────────────


def _compute_norm_stats(
    df: pd.DataFrame,
    train_mask: np.ndarray,
    input_cols: list[str],
    target_col: str,
) -> dict:
    """Compute z-score (mean, std) on the training rows only.

    Returns a dict with keys ``'x_mean'``, ``'x_std'``, ``'y_mean'``, ``'y_std'``
    (all Python lists of floats) plus ``'input_cols'`` and ``'target_col'`` for
    provenance.
    """
    train_df = df.loc[train_mask]

    x_mean = train_df[input_cols].mean().values.astype(np.float64)
    x_std  = train_df[input_cols].std().values.astype(np.float64)
    y_mean = float(train_df[target_col].mean())
    y_std  = float(train_df[target_col].std())

    # Guard against zero-std columns (constant features).
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    if y_std < 1e-8:
        y_std = 1.0

    return {
        "input_cols": input_cols,
        "target_col": target_col,
        "x_mean": x_mean.tolist(),
        "x_std":  x_std.tolist(),
        "y_mean": y_mean,
        "y_std":  y_std,
    }


def _normalise(
    arr: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Element-wise z-score: (arr - mean) / std."""
    return (arr - mean) / std


# ── prepare one config ────────────────────────────────────────────────────────


def prepare_config(
    df: pd.DataFrame,
    config_name: str,
    split_ids: dict[str, list[int]],
    out_dir: Path | None = None,
) -> dict:
    """Build normalised train / val / test tensors for one PMSM config.

    Saves three ``.pt`` files (one per split) plus a ``_norm.json`` sidecar
    to ``out_dir`` (defaults to ``data/processed/``).

    Returns the norm_stats dict so callers can inspect it.
    """
    out_dir = out_dir or PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    input_cols, target_col = get_config(config_name)

    # Build boolean masks from profile_id lists.
    masks = {
        split: df["profile_id"].isin(pid_list).to_numpy()
        for split, pid_list in split_ids.items()
    }

    # Compute normalisation stats on training rows only.
    norm_stats = _compute_norm_stats(df, masks["train"], input_cols, target_col)
    x_mean = np.array(norm_stats["x_mean"], dtype=np.float64)
    x_std  = np.array(norm_stats["x_std"],  dtype=np.float64)
    y_mean = norm_stats["y_mean"]
    y_std  = norm_stats["y_std"]

    for split in ("train", "val", "test"):
        sub = df.loc[masks[split]]
        x_raw = sub[input_cols].to_numpy(dtype=np.float64)
        y_raw = sub[[target_col]].to_numpy(dtype=np.float64)

        x_norm = _normalise(x_raw, x_mean, x_std)
        y_norm = _normalise(y_raw, y_mean, y_std)

        data = {
            "x": torch.tensor(x_norm, dtype=torch.float32),
            "y": torch.tensor(y_norm, dtype=torch.float32),
        }

        pt_path = out_dir / f"dataset_{config_name}_{split}.pt"
        torch.save(data, pt_path)

    # Save normalisation stats so we can denormalise predictions to °C later.
    norm_path = out_dir / f"{config_name}_norm.json"
    with open(norm_path, "w") as fh:
        json.dump(norm_stats, fh, indent=2)

    return norm_stats


# ── prepare all configs ───────────────────────────────────────────────────────


def prepare_all_configs(
    csv_path: str | Path,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.10,
) -> dict[str, dict]:
    """End-to-end: read CSV once, split profiles, prepare all PMSM configs.

    Returns a dict mapping config_name → norm_stats.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"PMSM CSV not found: {csv_path}")

    print(f"Reading {csv_path} ...", flush=True)
    t0 = time.time()
    df = pd.read_csv(csv_path)
    print(f"  loaded {len(df):,} rows × {len(df.columns)} columns "
          f"in {time.time() - t0:.1f}s")

    # Verify expected columns exist.
    required = set(["profile_id", TARGET])
    for cols in CONFIGS.values():
        required.update(cols)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    # Profile-based split (computed once, shared by all configs).
    split_ids = make_profile_split(df, seed=seed,
                                   train_frac=train_frac, val_frac=val_frac)
    print(f"  profile split (seed={seed}):  "
          f"train={len(split_ids['train'])} profiles, "
          f"val={len(split_ids['val'])} profiles, "
          f"test={len(split_ids['test'])} profiles")

    # Save the split itself for reproducibility audits.
    split_path = PROCESSED_DIR / "pmsm_profile_split.json"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as fh:
        json.dump({"seed": seed, "train_frac": train_frac,
                    "val_frac": val_frac, **split_ids}, fh, indent=2)
    print(f"  saved profile split → {split_path}")

    # Prepare each config.
    all_stats: dict[str, dict] = {}
    for config_name in ALL_CONFIG_NAMES:
        input_cols, _ = get_config(config_name)
        print(f"\n  config: {config_name}  ({len(input_cols)} inputs → {TARGET})")

        t1 = time.time()
        stats = prepare_config(df, config_name, split_ids)
        dt = time.time() - t1

        # Report sizes.
        for split in ("train", "val", "test"):
            pt = PROCESSED_DIR / f"dataset_{config_name}_{split}.pt"
            d = torch.load(pt, weights_only=True)
            print(f"    {split:5s}: x={tuple(d['x'].shape)}  y={tuple(d['y'].shape)}")
        print(f"    done in {dt:.1f}s")

        all_stats[config_name] = stats

    print(f"\nAll PMSM configs saved to {PROCESSED_DIR}/")
    return all_stats


# ── load helper (called by generate_data.load_split) ─────────────────────────


def load_pmsm_split(
    config_name: str,
    split: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a saved PMSM split.  Drop-in replacement for the synthetic loader.

    Args:
        config_name: One of the keys in ``CONFIGS`` (e.g. ``'PMSM_no_temps'``).
        split:       ``'train'``, ``'val'``, or ``'test'``.

    Returns:
        ``(x, y)`` float32 tensors.  ``x`` has shape ``(N, n_inputs)``;
        ``y`` has shape ``(N, 1)``.

    Raises:
        FileNotFoundError with instructions if the .pt file doesn't exist.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")

    # prepare_all_configs (src/data/pmsm_loader_ewma.py) saves files as
    # dataset_{config_name}_ewma_{split}.pt or _base_{split}.pt depending on
    # whether --add-ewma was passed; older runs may have saved the plain
    # dataset_{config_name}_{split}.pt with no suffix. Check all three so
    # this loader (used by generate_data.load_split -> pruner.py) actually
    # finds files produced by the EWMA prep pipeline.
    candidates = [
        PROCESSED_DIR / f"dataset_{config_name}_ewma_{split}.pt",
        PROCESSED_DIR / f"dataset_{config_name}_base_{split}.pt",
        PROCESSED_DIR / f"dataset_{config_name}_{split}.pt",
    ]
    pt_path = next((p for p in candidates if p.exists()), None)
    if pt_path is None:
        raise FileNotFoundError(
            "PMSM split file not found. Checked:\n"
            + "\n".join(f"  {p}" for p in candidates)
            + "\nRun:  python -m src.data.pmsm_loader_ewma --csv <path/to/measures_v2.csv> "
              "--split-mode diversity --add-ewma ..."
        )
    data = torch.load(pt_path, weights_only=True)
    return data["x"].float(), data["y"].float()


def load_norm_stats(config_name: str) -> dict:
    """Load the z-score normalisation statistics for a PMSM config.

    Useful for denormalising predictions back to °C:
        ``y_celsius = y_norm * stats['y_std'] + stats['y_mean']``
    """
    norm_path = PROCESSED_DIR / f"{config_name}_norm.json"
    if not norm_path.exists():
        raise FileNotFoundError(
            f"Norm stats not found: {norm_path}\n"
            f"Run:  python -m src.data.pmsm_loader --csv <path/to/measures_v2.csv>"
        )
    with open(norm_path) as fh:
        return json.load(fh)


# ── CLI entry point ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare PMSM dataset splits for baseline training + pruning."
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to measures_v2.csv (the PMSM dataset).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for profile-based split (default: 42).",
    )
    args = parser.parse_args()

    prepare_all_configs(args.csv, seed=args.seed)


if __name__ == "__main__":
    main()
