# ══════════════════════════════════════════════════════════════════════════════
# Naming convention (used everywhere — single source of truth)
# ══════════════════════════════════════════════════════════════════════════════
# EWMA  : {col}_ewma_{span}
# ROLL  : {col}_roll_{stat}_{window}   ← underscore before stat
#
# Previously add_rolling_grouped used {col}_roll{stat}_{window} (no underscore)
# while _extend_input_cols_with_roll_delta used {col}_roll_{stat}_{window}.
# This single-character mismatch caused every rolling column to be missing
# from the DataFrame when prepare_config tried to select them.
# ══════════════════════════════════════════════════════════════════════════════

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

from src.data.pmsm_configs import CONFIGS, CONFIGS_EWMA, TARGET, ALL_CONFIG_NAMES, EWMA_CONFIG_NAMES, get_config  # noqa: E402

PROCESSED_DIR = REPO_ROOT / "data" / "processed"

DEFAULT_EWMA_COLS  = ["motor_speed", "torque", "i_d", "i_q", "u_d", "u_q"]
DEFAULT_EWMA_SPANS = [60, 300, 1200]
PHYSICS_VAR: list[str] = [
    "power_dq", "torque_speed", "i_cross", "i_squared"
]
# Rolling config — kept separate so it is easy to change independently
DEFAULT_ROLL_STATS   = ["mean", "max"]   # which stats to compute
# Windows are shared with EWMA spans (same timescales, complementary features)

INPUT_FINGERPRINT_COLS = ["motor_speed", "torque", "i_d", "i_q"]
FINGERPRINT_N_POINTS   = 200


# ══════════════════════════════════════════════════════════════════════════════
# Single naming helpers — call these everywhere instead of f-strings
# ══════════════════════════════════════════════════════════════════════════════

def ewma_col_name(col: str, span: int) -> str:
    """Canonical EWMA column name."""
    return f"{col}_ewma_{span}"


def roll_col_name(col: str, stat: str, window: int) -> str:
    """Canonical rolling column name  (note underscore before stat)."""
    return f"{col}_roll_{stat}_{window}"


# ══════════════════════════════════════════════════════════════════════════════
# Session fingerprint helpers  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def _session_fingerprint(
    sub: pd.DataFrame,
    cols: list[str],
    n_points: int = FINGERPRINT_N_POINTS,
) -> np.ndarray:
    x_old = np.linspace(0.0, 1.0, len(sub))
    x_new = np.linspace(0.0, 1.0, n_points)
    parts: list[np.ndarray] = []
    for c in cols:
        y    = sub[c].to_numpy(dtype=np.float64)
        y_rs = np.interp(x_new, x_old, y)
        std  = y_rs.std()
        parts.append((y_rs - y_rs.mean()) / (std if std > 1e-9 else 1.0))
    return np.concatenate(parts)


def _build_similarity_matrix(
    df: pd.DataFrame,
    pids: list[int],
    cols: list[str] = INPUT_FINGERPRINT_COLS,
    n_points: int = FINGERPRINT_N_POINTS,
) -> tuple[np.ndarray, list[int]]:
    fingerprints: list[np.ndarray] = []
    valid_pids:   list[int]        = []
    for pid in pids:
        sub = df[df["profile_id"] == pid]
        if len(sub) < 10:
            continue
        fingerprints.append(_session_fingerprint(sub, cols, n_points))
        valid_pids.append(pid)
    if len(fingerprints) < 2:
        raise ValueError("Need ≥ 2 valid sessions for similarity matrix.")
    sim = np.corrcoef(np.stack(fingerprints))
    sim = np.clip(sim, -1.0, 1.0)
    sim = np.nan_to_num(sim, nan=0.0)
    return sim, valid_pids


def _greedy_max_diversity_train(
    pool_pids: list[int],
    sim_matrix: np.ndarray,
    pid_to_idx: dict[int, int],
    n_train: int,
    seed: int,
) -> list[int]:
    rng      = np.random.default_rng(seed)
    pool_set = set(pool_pids)
    if n_train >= len(pool_pids):
        return list(pool_pids)

    seed_pid = int(rng.choice(list(pool_pids)))
    train_pids: list[int] = [seed_pid]
    pool_set.remove(seed_pid)
    seed_idx = pid_to_idx[seed_pid]
    print(f"    greedy step   1/{n_train}  added profile {seed_pid:>3d}  (seed)", flush=True)

    max_abs_sim: dict[int, float] = {
        pid: abs(float(sim_matrix[pid_to_idx[pid], seed_idx]))
        for pid in pool_set
    }

    for step in range(1, n_train):
        best_pid   = min(pool_set, key=lambda p: max_abs_sim[p])
        best_score = max_abs_sim[best_pid]
        best_idx   = pid_to_idx[best_pid]

        train_pids.append(best_pid)
        pool_set.remove(best_pid)

        for pid in pool_set:
            new_sim = abs(float(sim_matrix[pid_to_idx[pid], best_idx]))
            if new_sim > max_abs_sim[pid]:
                max_abs_sim[pid] = new_sim

        if step % 5 == 0 or step == n_train - 1:
            next_str = (
                f"  next={min(max_abs_sim[p] for p in pool_set):.3f}"
                if pool_set else ""
            )
            print(f"    greedy step {step+1:3d}/{n_train}  "
                  f"added profile {best_pid:>3d}  "
                  f"max|sim|={best_score:.3f}" + next_str, flush=True)

    return train_pids


def _print_diversity_report(
    split_ids: dict[str, list[int]],
    sim_matrix: np.ndarray,
    valid_pids: list[int],
) -> None:
    pid_to_idx = {pid: i for i, pid in enumerate(valid_pids)}
    print("\n  DIVERSITY REPORT")
    print(f"  {'Split':<8s}  {'n':>4s}  {'mean|r|':>8s}  {'max|r|':>7s}  {'min|r|':>7s}")
    print("  " + "-" * 45)
    for split_name, pids in split_ids.items():
        idx = [pid_to_idx[p] for p in pids if p in pid_to_idx]
        if len(idx) < 2:
            print(f"  {split_name:<8s}  {len(pids):>4d}  (too few for stats)")
            continue
        sub      = sim_matrix[np.ix_(idx, idx)]
        mask     = np.triu(np.ones_like(sub, dtype=bool), k=1)
        abs_vals = np.abs(sub[mask])
        note = ("⚠  near-duplicate pair" if abs_vals.max() > 0.85
                else "✓  well diversified"  if abs_vals.mean() < 0.30 else "")
        print(f"  {split_name:<8s}  {len(pids):>4d}  "
              f"{abs_vals.mean():>8.3f}  {abs_vals.max():>7.3f}  "
              f"{abs_vals.min():>7.3f}  {note}")
    print()


def make_diversity_split(
    df: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.10,
    fingerprint_cols: list[str] = INPUT_FINGERPRINT_COLS,
    n_fp_points: int = FINGERPRINT_N_POINTS,
) -> tuple[dict[str, list[int]], np.ndarray, list[int]]:
    pids_all = sorted(df["profile_id"].unique().tolist())
    rng      = np.random.default_rng(seed)
    n_total  = len(pids_all)
    n_test   = max(1, int(round(n_total * (1.0 - train_frac - val_frac))))
    n_train  = int(round(n_total * train_frac))

    test_pids = sorted(rng.choice(pids_all, size=n_test, replace=False).tolist())
    pool_pids = [p for p in pids_all if p not in set(test_pids)]

    print(f"  [diversity split]  total={n_total}  test={n_test}  pool={len(pool_pids)}")
    t0 = time.time()
    sim_matrix, valid_pool_pids = _build_similarity_matrix(
        df, pool_pids, cols=fingerprint_cols, n_points=n_fp_points,
    )
    print(f"    fingerprints done in {time.time()-t0:.1f}s")

    pid_to_idx = {pid: i for i, pid in enumerate(valid_pool_pids)}
    train_pids = _greedy_max_diversity_train(
        valid_pool_pids, sim_matrix, pid_to_idx, n_train, seed=seed + 1,
    )

    train_set = set(train_pids)
    val_pids  = [p for p in valid_pool_pids if p not in train_set]
    dropped   = set(pool_pids) - set(valid_pool_pids)
    if dropped:
        print(f"  WARNING: {len(dropped)} tiny profiles → val: {sorted(dropped)}")
        val_pids = sorted(val_pids + list(dropped))

    split_ids = {
        "train": sorted(train_pids),
        "val":   sorted(val_pids),
        "test":  sorted(test_pids),
    }
    all_assigned = set(split_ids["train"]) | set(split_ids["val"]) | set(split_ids["test"])
    assert all_assigned == set(pids_all), \
        f"Partition not exhaustive: missing {set(pids_all) - all_assigned}"

    print(f"  [diversity split result]  "
          f"train={len(split_ids['train'])}  "
          f"val={len(split_ids['val'])}  "
          f"test={len(split_ids['test'])}")
    return split_ids, sim_matrix, valid_pool_pids


# ══════════════════════════════════════════════════════════════════════════════
# Random / chronological split
# ══════════════════════════════════════════════════════════════════════════════

def make_profile_split(
    df: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.10,
    random_split: bool = True,
) -> dict[str, list[int]]:
    pids  = np.array(sorted(df["profile_id"].unique()))
    order = (np.random.default_rng(seed).permutation(len(pids))
             if random_split else np.arange(len(pids)))
    n_tr  = int(round(len(pids) * train_frac))
    n_va  = int(round(len(pids) * val_frac))
    split_ids = {
        "train": sorted(pids[order[:n_tr]].tolist()),
        "val":   sorted(pids[order[n_tr : n_tr + n_va]].tolist()),
        "test":  sorted(pids[order[n_tr + n_va :]].tolist()),
    }
    all_assigned = set(split_ids["train"]) | set(split_ids["val"]) | set(split_ids["test"])
    assert all_assigned == set(pids.tolist())
    return split_ids


# ══════════════════════════════════════════════════════════════════════════════
# Physics features
# ══════════════════════════════════════════════════════════════════════════════

def add_physics_features(
    df: pd.DataFrame,
    include_cross_terms: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    df       = df.copy()
    new_cols: list[str] = []

    df["i_squared"] = df["i_d"] ** 2 + df["i_q"] ** 2
    new_cols.append("i_squared")

    if "u_d" in df.columns and "u_q" in df.columns:
        df["power_dq"] = df["i_d"] * df["u_d"] + df["i_q"] * df["u_q"]
        new_cols.append("power_dq")

    if "torque" in df.columns and "motor_speed" in df.columns:
        df["torque_speed"] = df["torque"] * df["motor_speed"]
        new_cols.append("torque_speed")

    if include_cross_terms:
        df["i_cross"] = df["i_d"] * df["i_q"]
        new_cols.append("i_cross")

    return df, new_cols


# ══════════════════════════════════════════════════════════════════════════════
# EWMA features
# ══════════════════════════════════════════════════════════════════════════════

def add_ewma_features(
    df: pd.DataFrame,
    cols: list[str],
    spans: list[int],
) -> tuple[pd.DataFrame, list[str]]:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"add_ewma_features: columns not in df: {missing}")
    df      = df.copy()
    new_cols: list[str] = []
    grouped = df.groupby("profile_id")
    for col in cols:
        for span in spans:
            name       = ewma_col_name(col, span)   # ← uses helper
            df[name]   = grouped[col].transform(
                lambda x, s=span: x.ewm(span=s, adjust=False).mean()
            )
            new_cols.append(name)
    return df, new_cols


# ══════════════════════════════════════════════════════════════════════════════
# Rolling features  (vectorised, boundary-safe)
# ══════════════════════════════════════════════════════════════════════════════

def add_rolling_grouped(
    df: pd.DataFrame,
    cols: list[str],
    windows: list[int],
    stats: list[str] = ("mean", "max"),
) -> tuple[pd.DataFrame, list[str]]:
    """
    Vectorised rolling statistics, grouped by profile_id (no session bleed).

    Column naming:  {col}_roll_{stat}_{window}   (underscore before stat)

    Boundary correction:
        The rolling call operates on the FULL sorted array as if it were one
        continuous series.  For the first (window-1) rows of every profile
        after the first, the window overlaps the PREVIOUS profile.  We detect
        these rows via the boundary_indices array and overwrite them with
        correct expanding statistics computed using only the current profile's
        own data.
    """
    original_index = df.index
    df_sorted      = df.sort_values("profile_id").copy()

    profile_ids    = df_sorted["profile_id"].to_numpy().copy()
    is_new_profile = np.concatenate([[True], profile_ids[1:] != profile_ids[:-1]])
    boundary_indices = np.where(is_new_profile)[0]

    new_cols: list[str] = []

    for col in cols:
        if col not in df_sorted.columns:
            print(f"    [rolling] skip {col!r} — not in dataframe")
            continue

        values = df_sorted[col].to_numpy(dtype=np.float64).copy()
        s      = pd.Series(values)

        for window in windows:
            for stat in stats:
                name = roll_col_name(col, stat, window)   # ← uses helper

                # ── raw rolling on full array ─────────────────────────────
                if stat == "mean":
                    result = s.rolling(window, min_periods=1).mean().to_numpy().copy()
                elif stat == "std":
                    result = s.rolling(window, min_periods=1).std().to_numpy().copy()
                    result = np.nan_to_num(result, nan=0.0)
                elif stat == "max":
                    result = s.rolling(window, min_periods=1).max().to_numpy().copy()
                elif stat == "min":
                    result = s.rolling(window, min_periods=1).min().to_numpy().copy()
                elif stat == "jump":
                    padded = np.concatenate([np.zeros(window), values])
                    result = (values - padded[:-window]).copy()
                else:
                    raise ValueError(f"Unknown stat: {stat!r}")

                # ── boundary correction ───────────────────────────────────
                n_fix = window if stat == "jump" else window - 1
                for start_idx in boundary_indices:
                    end_idx = min(start_idx + n_fix, len(result))
                    if end_idx <= start_idx:
                        continue
                    prefix = values[start_idx:end_idx]
                    if stat == "mean":
                        counts = np.arange(1, len(prefix) + 1, dtype=np.float64)
                        result[start_idx:end_idx] = np.cumsum(prefix) / counts
                    elif stat == "std":
                        result[start_idx:end_idx] = 0.0
                    elif stat == "max":
                        result[start_idx:end_idx] = np.maximum.accumulate(prefix)
                    elif stat == "min":
                        result[start_idx:end_idx] = np.minimum.accumulate(prefix)
                    elif stat == "jump":
                        result[start_idx:end_idx] = 0.0

                df_sorted[name] = result
                new_cols.append(name)

    df_sorted = df_sorted.reindex(original_index)
    return df_sorted, new_cols


# ══════════════════════════════════════════════════════════════════════════════
# Input column extension helpers
# ══════════════════════════════════════════════════════════════════════════════

def _extend_input_cols_with_ewma(
    input_cols: list[str],
    ewma_cols: list[str],
    spans: list[int],
) -> list[str]:
    """Append EWMA column names for any base col present in both lists."""
    extended = list(input_cols)
    for col in input_cols:
        if col in ewma_cols:
            for span in spans:
                extended.append(ewma_col_name(col, span))   # ← uses helper
    return extended


def _extend_input_cols_with_rolling(
    input_cols: list[str],
    roll_cols: list[str],
    windows: list[int],
    stats: list[str] = DEFAULT_ROLL_STATS,
) -> list[str]:
    """Append rolling column names for any base col present in both lists."""
    extended = list(input_cols)
    for col in input_cols:
        if col in roll_cols:
            for window in windows:
                for stat in stats:
                    extended.append(roll_col_name(col, stat, window))  # ← uses helper
    return extended


def _build_full_input_cols(
    base_input_cols: list[str],
    ewma_cols: list[str] | None,
    ewma_spans: list[int] | None,
    roll_cols: list[str] | None,
    roll_windows: list[int] | None,
    roll_stats: list[str] = DEFAULT_ROLL_STATS,
) -> list[str]:
    """
    Single entry point that builds the complete input column list.
    Replaces the two separate _extend_* calls scattered across prepare_config
    and prepare_all_configs — previously these were called independently and
    could fall out of sync with what was actually computed.
    """
    cols = list(base_input_cols)
    if ewma_cols and ewma_spans:
        cols = _extend_input_cols_with_ewma(cols, ewma_cols, ewma_spans)
    # if roll_cols and roll_windows:
    #     cols = _extend_input_cols_with_rolling(cols, roll_cols, roll_windows, roll_stats)
    return cols


# ══════════════════════════════════════════════════════════════════════════════
# Normalisation
# ══════════════════════════════════════════════════════════════════════════════

def _compute_norm_stats(
    df: pd.DataFrame,
    train_mask: np.ndarray,
    input_cols: list[str],
    target_col: str,
) -> dict:
    # Verify all requested columns exist BEFORE trying to index
    missing = [c for c in input_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"_compute_norm_stats: {len(missing)} input columns missing from df.\n"
            f"First 10: {missing[:10]}\n"
            f"Available rolling-style cols: "
            f"{[c for c in df.columns if 'roll' in c][:10]}"
        )

    train_df = df.loc[train_mask]
    x_mean   = train_df[input_cols].mean().values.astype(np.float64)
    x_std    = train_df[input_cols].std().values.astype(np.float64)
    y_mean   = float(train_df[target_col].mean())
    y_std    = float(train_df[target_col].std())
    x_std    = np.where(x_std < 1e-8, 1.0, x_std)
    if y_std < 1e-8:
        y_std = 1.0
    return {
        "input_cols": input_cols,
        "target_col": target_col,
        "x_mean":     x_mean.tolist(),
        "x_std":      x_std.tolist(),
        "y_mean":     y_mean,
        "y_std":      y_std,
    }


def _normalise(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (arr - mean) / std


# ══════════════════════════════════════════════════════════════════════════════
# Prepare one config
# ══════════════════════════════════════════════════════════════════════════════

def prepare_config(
    df: pd.DataFrame,
    config_name: str,
    split_ids: dict[str, list[int]],
    out_dir: Path | None = None,
    ewma_cols: list[str] | None = None,
    ewma_spans: list[int] | None = None,
    roll_cols: list[str] | None = None,
    roll_windows: list[int] | None = None,
    roll_stats: list[str] = DEFAULT_ROLL_STATS,
    phys_cols: list[str] | None = None,
) -> dict:
    """
    Build normalised train/val/test tensors for one PMSM config.

    All column name construction goes through _build_full_input_cols so
    that the names are guaranteed to match what add_rolling_grouped and
    add_ewma_features actually created.
    """
    out_dir = out_dir or PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    base_input_cols, target_col = get_config(config_name)

    # Add physics columns to the base input list if they were computed
    if phys_cols:
        base_input_cols = list(base_input_cols) + [
            c for c in phys_cols if c not in base_input_cols
        ]

    # Build the full input column list — one place, consistent naming
    input_cols = _build_full_input_cols(
        base_input_cols,
        ewma_cols=ewma_cols,
        ewma_spans=ewma_spans,
        roll_cols=roll_cols,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
    )

    # Verify every column exists before going further
    missing = [c for c in input_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"prepare_config({config_name}): {len(missing)} columns not in df.\n"
            f"Missing: {missing[:20]}\n"
            f"Hint: rolling col names use 'roll_{{stat}}_{{window}}' format.\n"
            f"      Check that add_rolling_grouped was called with the same cols/windows."
        )

    masks = {
        split: df["profile_id"].isin(pid_list).to_numpy()
        for split, pid_list in split_ids.items()
    }

    norm_stats = _compute_norm_stats(df, masks["train"], input_cols, target_col)
    x_mean = np.array(norm_stats["x_mean"], dtype=np.float64)
    x_std  = np.array(norm_stats["x_std"],  dtype=np.float64)
    y_mean = norm_stats["y_mean"]
    y_std  = norm_stats["y_std"]

    norm_stats["add_ewma"]    = bool(ewma_cols and ewma_spans)
    norm_stats["ewma_cols"]   = ewma_cols   or []
    norm_stats["ewma_spans"]  = ewma_spans  or []
    norm_stats["roll_cols"]   = roll_cols   or []
    norm_stats["roll_windows"]= roll_windows or []
    norm_stats["roll_stats"]  = list(roll_stats)
    norm_stats["phys_cols"]   = phys_cols   or []

    for split in ("train", "val", "test"):
        sub   = df.loc[masks[split]]
        x_raw = sub[input_cols].to_numpy(dtype=np.float64)
        y_raw = sub[[target_col]].to_numpy(dtype=np.float64)
        data  = {
            "x": torch.tensor(_normalise(x_raw, x_mean, x_std), dtype=torch.float32),
            "y": torch.tensor(_normalise(y_raw, y_mean, y_std), dtype=torch.float32),
        }
        suffix  = "ewma" if (ewma_cols and ewma_spans) else "base"
        pt_path = out_dir / f"dataset_{config_name}_{suffix}_{split}.pt"
        torch.save(data, pt_path)

    norm_path = out_dir / f"{config_name}_ewma_norm.json"
    with open(norm_path, "w") as fh:
        json.dump(norm_stats, fh, indent=2)

    return norm_stats


# ══════════════════════════════════════════════════════════════════════════════
# Prepare all configs
# ══════════════════════════════════════════════════════════════════════════════

def prepare_all_configs(
    csv_path: str | Path,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.10,
    split_mode: str = "random",
    add_ewma: bool = False,
    add_physics: bool = False,
    ewma_cols: list[str] | None = None,
    ewma_spans: list[int] | None = None,
    roll_stats: list[str] = DEFAULT_ROLL_STATS,
    only_configs: list[str] | None = None,
    fingerprint_cols: list[str] = INPUT_FINGERPRINT_COLS,
) -> dict[str, dict]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if split_mode not in ("random", "chronological", "diversity"):
        raise ValueError(f"Unknown split_mode: {split_mode!r}")

    print(f"Reading {csv_path} …", flush=True)
    t0 = time.time()
    df = pd.read_csv(csv_path)
    print(f"  loaded {len(df):,} rows × {len(df.columns)} cols in {time.time()-t0:.1f}s")

    config_list = only_configs if only_configs is not None else EWMA_CONFIG_NAMES

    required = {"profile_id", TARGET}
    for name in config_list:
        required.update(CONFIGS_EWMA[name])
    missing_csv = required - set(df.columns)
    if missing_csv:
        raise ValueError(f"CSV missing columns: {sorted(missing_csv)}")

    # ── Physics features ──────────────────────────────────────────────────
    phys_cols: list[str] = []
    if add_physics:
        print("  computing physics features …", flush=True)
        df, phys_cols = add_physics_features(df)
        print(f"    added: {phys_cols}")

    # ── Resolve EWMA cols (base + optional physics) ───────────────────────
    resolved_ewma_cols:  list[str] | None = None
    resolved_ewma_spans: list[int] | None = None
    if add_ewma:
        resolved_ewma_cols  = list(ewma_cols  or DEFAULT_EWMA_COLS)
        resolved_ewma_spans = list(ewma_spans or DEFAULT_EWMA_SPANS)
        if add_physics:
            resolved_ewma_cols = resolved_ewma_cols + [
                c for c in phys_cols if c not in resolved_ewma_cols
            ]
        print(f"  EWMA cols : {resolved_ewma_cols}")
        print(f"  EWMA spans: {resolved_ewma_spans}")

    # ── Rolling cols (same set as EWMA cols, same windows as spans) ───────
    # Rolling and EWMA share the same base columns and timescale windows
    # so they are complementary features at the same resolution.
    resolved_roll_cols:    list[str] | None = None
    resolved_roll_windows: list[int] | None = None
    if add_ewma:   # rolling computed whenever EWMA is computed
        resolved_roll_cols    = list(resolved_ewma_cols)     # same base cols
        resolved_roll_windows = list(resolved_ewma_spans)    # same timescales

    # ── Compute EWMA features ─────────────────────────────────────────────
    if add_ewma:
        bad = [c for c in resolved_ewma_cols if c not in df.columns]
        if bad:
            raise ValueError(f"EWMA cols not in CSV: {bad}")
        print(f"  computing EWMA features …", flush=True)
        t1 = time.time()
        df, ewma_new = add_ewma_features(df, resolved_ewma_cols, resolved_ewma_spans)
        print(f"    {len(ewma_new)} EWMA cols in {time.time()-t1:.1f}s")

    # ── Compute rolling features ──────────────────────────────────────────
    if add_ewma:
        bad_roll = [c for c in resolved_roll_cols if c not in df.columns]
        if bad_roll:
            raise ValueError(f"Rolling cols not in df: {bad_roll}")
        print(f"  computing rolling features "
              f"(cols={len(resolved_roll_cols)}, "
              f"windows={resolved_roll_windows}, "
              f"stats={roll_stats}) …", flush=True)
        t1 = time.time()
        # df, roll_new = add_rolling_grouped(
        #     df,
        #     cols=resolved_roll_cols,
        #     windows=resolved_roll_windows,
        #     stats=roll_stats,
        # )
        # print(f"    {len(roll_new)} rolling cols in {time.time()-t1:.1f}s")

        # ── Sanity check: verify columns are actually in df ───────────────
        # expected_roll = [
        #     roll_col_name(col, stat, w)
        #     for col in resolved_roll_cols
        #     for w in resolved_roll_windows
        #     for stat in roll_stats
        #     if col in df.columns   # skip any that were skipped in add_rolling_grouped
        # ]
        # actually_missing = [c for c in expected_roll if c not in df.columns]
        # if actually_missing:
        #     raise RuntimeError(
        #         f"BUG: add_rolling_grouped did not create expected columns.\n"
        #         f"Missing: {actually_missing[:10]}\n"
        #         f"This should never happen — please file a bug report."
        #     )

    # ── Profile split ─────────────────────────────────────────────────────
    sim_matrix: np.ndarray | None = None
    sim_pids:   list[int]   | None = None

    if split_mode == "diversity":
        split_ids, sim_matrix, sim_pids = make_diversity_split(
            df, seed=seed, train_frac=train_frac, val_frac=val_frac,
            fingerprint_cols=fingerprint_cols,
        )
        _print_diversity_report(split_ids, sim_matrix, sim_pids)
        mode_str = "diversity (greedy max-diversity)"
    elif split_mode == "chronological":
        split_ids = make_profile_split(df, seed=seed, train_frac=train_frac,
                                        val_frac=val_frac, random_split=False)
        mode_str = "chronological"
    else:
        split_ids = make_profile_split(df, seed=seed, train_frac=train_frac,
                                        val_frac=val_frac, random_split=True)
        mode_str = "random (seeded)"

    print(f"\n  split [{mode_str}]:  "
          f"train={len(split_ids['train'])}  "
          f"val={len(split_ids['val'])}  "
          f"test={len(split_ids['test'])}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    split_path = PROCESSED_DIR / "pmsm_profile_split_ewma.json"
    with open(split_path, "w") as fh:
        json.dump({
            "seed": seed, "train_frac": train_frac, "val_frac": val_frac,
            "split_mode": split_mode, "add_ewma": add_ewma,
            "ewma_cols": resolved_ewma_cols, "ewma_spans": resolved_ewma_spans,
            "roll_cols": resolved_roll_cols, "roll_windows": resolved_roll_windows,
            "roll_stats": list(roll_stats),
            "fingerprint_cols": fingerprint_cols if split_mode == "diversity" else None,
            **split_ids,
        }, fh, indent=2)
    print(f"  saved split → {split_path}")

    # ── Prepare each config ───────────────────────────────────────────────
    all_stats: dict[str, dict] = {}
    for config_name in config_list:
        base_input_cols, _ = get_config(config_name)

        # Show what the full input list will look like before calling prepare_config
        preview_cols = _build_full_input_cols(
            list(base_input_cols) + [c for c in phys_cols if c not in base_input_cols],
            ewma_cols=resolved_ewma_cols,
            ewma_spans=resolved_ewma_spans,
            roll_cols=resolved_roll_cols,
            roll_windows=resolved_roll_windows,
            roll_stats=roll_stats,
        )
        print(f"\n  config: {config_name}  ({len(preview_cols)} inputs → {TARGET})")

        t1    = time.time()
        stats = prepare_config(
            df,
            config_name,
            split_ids,
            ewma_cols=resolved_ewma_cols,
            ewma_spans=resolved_ewma_spans,
            roll_cols=resolved_roll_cols,
            roll_windows=resolved_roll_windows,
            roll_stats=roll_stats,
            phys_cols=phys_cols if phys_cols else None,
        )

        suffix = "ewma" if add_ewma else "base"
        for split in ("train", "val", "test"):
            pt = PROCESSED_DIR / f"dataset_{config_name}_{suffix}_{split}.pt"
            d  = torch.load(pt, weights_only=True)
            print(f"    {split:5s}: x={tuple(d['x'].shape)}  y={tuple(d['y'].shape)}")
        print(f"    done in {time.time()-t1:.1f}s")

        all_stats[config_name] = stats

    print(f"\nAll configs saved to {PROCESSED_DIR}/")
    return all_stats


# ══════════════════════════════════════════════════════════════════════════════
# Load helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_pmsm_split(
    config_name: str,
    split: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be train/val/test, got {split!r}")
    candidates = [
        PROCESSED_DIR / f"dataset_{config_name}_ewma_{split}.pt",
        PROCESSED_DIR / f"dataset_{config_name}_base_{split}.pt",
        PROCESSED_DIR / f"dataset_{config_name}_{split}.pt",
    ]
    pt_path = next((p for p in candidates if p.exists()), None)
    if pt_path is None:
        raise FileNotFoundError(
            f"PMSM split not found. Checked:\n"
            + "\n".join(f"  {p}" for p in candidates)
        )
    data = torch.load(pt_path, weights_only=True)
    return data["x"].float(), data["y"].float()


def load_norm_stats(config_name: str) -> dict:
    norm_path = PROCESSED_DIR / f"{config_name}_ewma_norm.json"
    if not norm_path.exists():
        raise FileNotFoundError(f"Norm stats not found: {norm_path}")
    with open(norm_path) as fh:
        return json.load(fh)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PMSM dataset splits.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-mode",
                        choices=["random", "chronological", "diversity"],
                        default="diversity")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac",   type=float, default=0.15)
    parser.add_argument("--fingerprint-cols", nargs="+",
                        default=INPUT_FINGERPRINT_COLS)
    parser.add_argument("--add-ewma",    action="store_true")
    parser.add_argument("--add-physics", action="store_true")
    parser.add_argument("--ewma-cols",   nargs="+", default=None)
    parser.add_argument("--ewma-spans",  nargs="+", type=int, default=None)
    parser.add_argument("--roll-stats",  nargs="+",
                        default=DEFAULT_ROLL_STATS,
                        help=f"Rolling statistics to compute. Default: {DEFAULT_ROLL_STATS}")
    args = parser.parse_args()

    prepare_all_configs(
        args.csv,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        split_mode=args.split_mode,
        add_ewma=args.add_ewma,
        add_physics=args.add_physics,
        ewma_cols=args.ewma_cols,
        ewma_spans=args.ewma_spans,
        roll_stats=args.roll_stats,
        fingerprint_cols=args.fingerprint_cols,
    )


if __name__ == "__main__":
    main()