"""
=================================================================
  PMSM Electric Motor Temperature — Dataset Exploration (EWMA + Rolling)
=================================================================
"""

import os
import time
import textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Paths & config ───────────────────────────────────────────
DATA_PATH = "measures_v2.csv"
PLOTS_DIR = "plots_ewma"
os.makedirs(PLOTS_DIR, exist_ok=True)

INPUT_FEATURES  = ["u_q", "u_d", "i_q", "i_d", "motor_speed", "torque", "ambient", "coolant"]
TARGET_FEATURES = ["pm", "stator_winding", "stator_tooth", "stator_yoke"]

ORDERED_VARS = [
    "u_q", "u_d", "i_q", "i_d",
    "motor_speed", "torque",
    "ambient", "coolant",
    "pm", "stator_winding", "stator_tooth", "stator_yoke",
]

SAMPLING_HZ = 2
EWMA_SPAN_CANDIDATES = [60, 300, 1200, 3000, 6000, 12000]
EWMA_DEMO_COLS = ["motor_speed", "torque"]

EWMA_CORR_SPANS = [60, 300, 1200, 6000, 12000]   # ← added 6000, 12000

EWMA_CORR_COLS  = [
    "i_q", "i_d", "motor_speed", "torque",
    "ambient", "coolant",
    "i_squared", "power_proxy",               # ← physics cols added
]

# Physics cols that exist AFTER add_physics_cols() — used in rolling too
PHYSICS_COLS    = ["i_squared", "power_proxy"]

# Rolling feature config — mirrors pmsm_loader_ewma.py
# windows in samples; at 2 Hz: 60=30s, 300=150s, 1200=600s
ROLLING_WINDOWS   = [60, 300, 1200]
ROLLING_COLS      =  ["i_q", "i_d", "motor_speed", "torque"]   # subset: most physical
# Which rolling statistics to compute and show
ROLLING_STATS     = ["mean", "std", "max"]   # mean≈slow trend, std≈volatility, max≈peak stress

# Physics-based derived columns added before rolling
PHYSICS_COLS      = ["i_squared", "power_proxy"]

COVERAGE_COLS  = ["motor_speed", "torque", "ambient", "coolant", "stator_winding"]
SHAPE_COLS     = ["motor_speed", "torque", "stator_winding"]

INPUT_FINGERPRINT_COLS  = ["motor_speed", "torque", "i_q", "i_d", "ambient", "coolant"]
OUTPUT_FINGERPRINT_COLS = ["pm", "stator_winding", "stator_tooth", "stator_yoke"]

# ── Plot style ──────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})

C_ELEC = "#2e75b6"
C_MECH = "#1f4e79"
C_ENV  = "#0070c0"
C_TEMP = "#c00000"
C_WARN = "#e36c09"

VAR_COLORS = {
    "motor_speed": "#1f4e79", "torque": "#e36c09",
    "u_q": "#2e75b6", "u_d": "#c00000", "i_q": "#375623", "i_d": "#7030a0",
    "ambient": "#0070c0", "coolant": "#00b0f0",
    "pm": "#c00000", "stator_winding": "#e36c09",
    "stator_tooth": "#0070c0", "stator_yoke": "#7030a0",
    "i_squared": "#9b2226", "power_proxy": "#ae2012",
}

def banner(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)

def save(name):
    path = os.path.join(PLOTS_DIR, name)
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [saved] plots/{name}")


# ════════════════════════════════════════════════════════════
# HELPER — per-profile EWMA
# ════════════════════════════════════════════════════════════

def add_ewma_grouped(
    df: pd.DataFrame,
    cols: list[str],
    spans: list[int],
) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    new_cols = []
    grouped = df.groupby("profile_id")
    for col in cols:
        for span in spans:
            name = f"{col}_ewma_{span}"
            df[name] = grouped[col].transform(
                lambda x, s=span: x.ewm(span=s, adjust=False).mean()
            )
            new_cols.append(name)
    return df, new_cols

# ════════════════════════════════════════════════════════════
# HELPER — per-profile ROLLING features  (fully vectorised)
# ════════════════════════════════════════════════════════════

def add_rolling_grouped(
    df: pd.DataFrame,
    cols: list[str],
    windows: list[int],
    stats: list[str] = ("mean", "std", "max"),
) -> tuple[pd.DataFrame, list[str]]:
    """
    Vectorised rolling statistics grouped by profile_id.

    Key speedups vs the naive groupby+transform approach:
      1. Sort once by profile_id so each profile is a contiguous block.
      2. Compute rolling on the ENTIRE sorted array in one pandas call
         (no Python-level group iteration).
      3. Zero out the boundary region at the START of each new profile
         so values from the end of profile N never bleed into profile N+1.
         This is equivalent to resetting the rolling window at each
         profile boundary, but done with a vectorised mask instead of
         per-group transforms.
      4. Use numpy directly for 'jump' (np.diff + boundary zeroing)
         instead of groupby.diff().

    Stats supported: 'mean', 'std', 'max', 'min', 'jump'
    """
    # ── 1. Sort by profile_id so profiles are contiguous blocks ──────────
    # Keep original index to restore order at the end.
    original_index = df.index
    df_sorted = df.sort_values("profile_id").copy()

    # Profile boundary mask: True at the FIRST row of each new profile.
    # Used to zero out values that would otherwise bleed across sessions.
    profile_ids   = df_sorted["profile_id"].to_numpy().copy()
    is_new_profile = np.concatenate([[True], profile_ids[1:] != profile_ids[:-1]])

    new_cols: list[str] = []

    for col in cols:
        if col not in df_sorted.columns:
            continue

        values = df_sorted[col].to_numpy(dtype=np.float64).copy()

        for window in windows:
            # Build a pandas Series on the sorted array — ONE rolling call,
            # no groupby overhead.
            s = pd.Series(values)

            for stat in stats:
                name = f"{col}_roll{stat}_{window}"

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
                    # Signed difference vs `window` steps back.
                    # np.diff with prepend handles the boundary cleanly.
                    padded = np.concatenate([np.full(window, np.nan), values])
                    result = values - padded[:-window]
                    result = np.nan_to_num(result, nan=0.0)

                else:
                    raise ValueError(f"Unknown stat: {stat!r}")

                # ── 2. Zero out boundary contamination ───────────────────
                # For a window of size W, the first W-1 rows of each new
                # profile contain values mixed with the previous profile's
                # data. We find those rows and reset them.
                #
                # For 'jump': only the first `window` rows are contaminated.
                # For rolling: first `window-1` rows are contaminated.
                #
                # Strategy: for each profile start, overwrite the
                # contaminated prefix with the FIRST VALID value of that
                # profile (i.e. the single-sample rolling value, which is
                # always the raw value itself for mean/max/min, and 0 for std).

                boundary_indices = np.where(is_new_profile)[0]

                for start_idx in boundary_indices:
                    end_idx = min(start_idx + window - 1, len(result))
                    if end_idx <= start_idx:
                        continue

                    if stat == "mean":
                        # Fill with expanding mean within this profile's prefix
                        prefix_vals = values[start_idx:end_idx]
                        result[start_idx:end_idx] = np.cumsum(prefix_vals) / np.arange(
                            1, len(prefix_vals) + 1
                        )
                    elif stat == "std":
                        # std of 1 sample = 0; expanding std for the prefix
                        result[start_idx:end_idx] = 0.0
                    elif stat == "max":
                        prefix_vals = values[start_idx:end_idx]
                        result[start_idx:end_idx] = np.maximum.accumulate(prefix_vals)
                    elif stat == "min":
                        prefix_vals = values[start_idx:end_idx]
                        result[start_idx:end_idx] = np.minimum.accumulate(prefix_vals)
                    elif stat == "jump":
                        # No history → jump is 0
                        result[start_idx : start_idx + window] = 0.0

                df_sorted[name] = result
                new_cols.append(name)

    # ── 3. Restore original row order ────────────────────────────────────
    df_sorted = df_sorted.reindex(original_index)

    return df_sorted, new_cols




def add_physics_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add i_squared and power_proxy — the actual heat sources."""
    df = df.copy()
    if "i_d" in df.columns and "i_q" in df.columns:
        df["i_squared"] = df["i_d"] ** 2 + df["i_q"] ** 2
    if "u_d" in df.columns and "u_q" in df.columns and "i_squared" in df.columns:
        df["power_proxy"] = df["i_d"] * df["u_d"] + df["i_q"] * df["u_q"]
    return df


# ════════════════════════════════════════════════════════════
# HELPER — per-session z-score fingerprint
# ════════════════════════════════════════════════════════════

def session_fingerprint(
    sub: pd.DataFrame,
    cols: list[str],
    n_points: int = 200,
) -> np.ndarray:
    x_old = np.linspace(0.0, 1.0, len(sub))
    x_new = np.linspace(0.0, 1.0, n_points)
    parts = []
    for c in cols:
        y = sub[c].to_numpy(dtype=np.float64)
        y_rs = np.interp(x_new, x_old, y)
        std = y_rs.std()
        parts.append((y_rs - y_rs.mean()) / (std if std > 1e-9 else 1.0))
    return np.concatenate(parts)


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — DATASET OVERVIEW
# ═══════════════════════════════════════════════════════════════
banner("SECTION 1 — DATASET OVERVIEW")

df = pd.read_csv(DATA_PATH)
df = add_physics_cols(df)   # i_squared, power_proxy available throughout

n_samples  = len(df)
n_vars     = df.shape[1]
n_profiles = df["profile_id"].nunique()
hours      = n_samples / (SAMPLING_HZ * 3600)

profile_counts        = df["profile_id"].value_counts().sort_index()
profile_durations_hrs = profile_counts / (SAMPLING_HZ * 3600)

print(textwrap.dedent(f"""
  Total samples      : {n_samples:,}
  Total variables    : {n_vars}
  Operating profiles : {n_profiles}
  Sampling frequency : {SAMPLING_HZ} Hz
  Total duration     : {hours:.1f} hours ({hours/24:.1f} days)
  Missing values     : {df.isnull().sum().sum()}
  Physics cols added : i_squared = i_d²+i_q²,  power_proxy = i_d·u_d + i_q·u_q
"""))


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — CORRELATION HEATMAP (raw + EWMA)
# ═══════════════════════════════════════════════════════════════
banner("SECTION 6 — CORRELATION HEATMAP WITH EWMA FEATURES")

print(textwrap.dedent(f"""
  6a  raw variables only (baseline)
  6b  raw + EWMA features  (spans {EWMA_CORR_SPANS} samples =
      {[round(s/(SAMPLING_HZ*60),1) for s in EWMA_CORR_SPANS]} min)
  6c  correlation vs EWMA span line plot — empirical thermal time constant
"""))

# ── 6a raw + physics ─────────────────────────────────────────────────────────
# ORDERED_VARS_PHYS adds the two physics columns right after i_d/i_q
ORDERED_VARS_PHYS = [
    "u_q", "u_d", "i_q", "i_d",
    "i_squared", "power_proxy",               # ← physics derived
    "motor_speed", "torque",
    "ambient", "coolant",
    "pm", "stator_winding", "stator_tooth", "stator_yoke",
]
# Only keep cols that actually exist (physics cols might be absent if
# add_physics_cols was not called — but we called it in Section 1)
plot_vars_6a = [c for c in ORDERED_VARS_PHYS if c in df.columns]

corr_raw = df[plot_vars_6a].corr()
n6a = len(plot_vars_6a)
fig, ax = plt.subplots(figsize=(max(12, n6a * 0.75), max(10, n6a * 0.7)))
mask = np.triu(np.ones_like(corr_raw, dtype=bool), k=1)
sns.heatmap(
    corr_raw, mask=mask, annot=True, fmt=".2f",
    cmap="RdYlBu_r", center=0, vmin=-1, vmax=1,
    linewidths=0.5, ax=ax, annot_kws={"size": 8},
    cbar_kws={"label": "Pearson correlation"},
)
ax.set_title(
    "6a — Correlation Heatmap: Raw Variables + Physics Features\n"
    "(i_squared = i_d²+i_q²,  power_proxy = i_d·u_d + i_q·u_q)",
    fontsize=12, fontweight="bold", pad=12,
)

# Draw a box around the physics block so it stands out
# Physics cols sit at indices 4,5 in plot_vars_6a
phys_start = plot_vars_6a.index("i_squared")
phys_end   = plot_vars_6a.index("power_proxy") + 1
for spine in ["top", "bottom", "left", "right"]:
    ax.add_patch(plt.Rectangle(
        (phys_start, phys_start),
        phys_end - phys_start,
        phys_end - phys_start,
        fill=False, edgecolor="#9b2226", linewidth=2.5,
        transform=ax.transData, clip_on=False,
    ))
plt.tight_layout()
save("correlation_heatmap_raw.png")

# ── compute EWMA (longer spans now: up to 12000) ──────────────────────────────
print(f"  Computing grouped EWMA features (spans {EWMA_CORR_SPANS}) …", flush=True)
t_ewma = time.time()
df_ewma, ewma_col_names = add_ewma_grouped(df, EWMA_CORR_COLS, EWMA_CORR_SPANS)
print(f"  done in {time.time()-t_ewma:.1f}s — added {len(ewma_col_names)} EWMA cols")

# ── 6b extended heatmap: raw + physics + EWMA of all cols × all spans ────────
#
# Layout order:
#   [non-ewma raw inputs] | [physics raw] | [ewma per col per span] | [targets]
#
non_target_raw  = [c for c in plot_vars_6a if c not in TARGET_FEATURES]
ewma_ordered    = [
    f"{col}_ewma_{span}"
    for col in EWMA_CORR_COLS
    for span in EWMA_CORR_SPANS
]
extended_vars   = non_target_raw + ewma_ordered + TARGET_FEATURES

# Keep only cols present in df_ewma
extended_vars   = [c for c in extended_vars if c in df_ewma.columns]
corr_ext        = df_ewma[extended_vars].corr()
n_ext           = len(extended_vars)

fig, ax = plt.subplots(
    figsize=(max(18, n_ext * 0.35), max(16, n_ext * 0.33)),
)
mask_ext  = np.triu(np.ones_like(corr_ext, dtype=bool), k=1)
annot_ext = n_ext <= 35   # annotation only if readable
sns.heatmap(
    corr_ext, mask=mask_ext,
    annot=annot_ext, fmt=".2f",
    cmap="RdYlBu_r", center=0, vmin=-1, vmax=1,
    linewidths=0.3, ax=ax,
    annot_kws={"size": 6} if annot_ext else {},
    cbar_kws={"label": "Pearson correlation"},
)
ax.set_title(
    f"6b — Correlation Heatmap: Raw + Physics + EWMA Features\n"
    f"Physics: i_squared, power_proxy   "
    f"EWMA cols: {EWMA_CORR_COLS}\n"
    f"EWMA spans: {EWMA_CORR_SPANS} samples "
    f"= {[round(s/(SAMPLING_HZ*60),1) for s in EWMA_CORR_SPANS]} min",
    fontsize=11, fontweight="bold", pad=12,
)

# Separator lines: raw block | ewma block | targets
n_raw_block  = len(non_target_raw)
n_ewma_block = len(ewma_ordered)
n_tgt_block  = len(TARGET_FEATURES)

for x_sep in [n_raw_block, n_raw_block + n_ewma_block]:
    if 0 < x_sep < n_ext:
        ax.axvline(x=x_sep, color="black", linewidth=1.8,
                   linestyle="--", alpha=0.7)
        ax.axhline(y=x_sep, color="black", linewidth=1.8,
                   linestyle="--", alpha=0.7)

# Highlight physics rows/cols with a distinct tick color
phys_in_ext = [i for i, c in enumerate(extended_vars) if c in PHYSICS_COLS]
for i in phys_in_ext:
    ax.get_xticklabels()[i].set_color("#9b2226")
    ax.get_yticklabels()[i].set_color("#9b2226")

# EWMA span group separators (light grey lines every len(EWMA_CORR_COLS) rows)
n_cols_ewma = len(EWMA_CORR_COLS)
for k in range(1, len(EWMA_CORR_SPANS)):
    sep = n_raw_block + k * n_cols_ewma
    if sep < n_ext:
        ax.axvline(x=sep, color="grey", linewidth=0.7,
                   linestyle=":", alpha=0.5)
        ax.axhline(y=sep, color="grey", linewidth=0.7,
                   linestyle=":", alpha=0.5)

plt.tight_layout()
save("correlation_heatmap_ewma_extended.png")

# ── 6c correlation vs EWMA span — now includes physics cols ──────────────────
#
# X-axis: [raw=0, span_60, span_300, span_1200, span_6000, span_12000]
# One subplot per temperature target.
# Physics cols are plotted with thicker dashed lines to distinguish them.
#
span_axis   = [0] + EWMA_CORR_SPANS
span_labels = ["raw"] + [
    f"{s}\n({round(s/(SAMPLING_HZ*60),0):.0f}min)"
    for s in EWMA_CORR_SPANS
]

# Separate cols for visual grouping in the legend
base_cols   = [c for c in EWMA_CORR_COLS if c not in PHYSICS_COLS]
phys_cols_  = [c for c in EWMA_CORR_COLS if c in PHYSICS_COLS]

fig, axes = plt.subplots(
    len(TARGET_FEATURES), 1,
    figsize=(13, 4.0 * len(TARGET_FEATURES)),
    sharex=True,
)

for ax, target in zip(axes, TARGET_FEATURES):

    # ── regular input cols ────────────────────────────────────
    for col in base_cols:
        vals = [df_ewma[col].corr(df_ewma[target])] + [
            df_ewma[f"{col}_ewma_{s}"].corr(df_ewma[target])
            for s in EWMA_CORR_SPANS
            if f"{col}_ewma_{s}" in df_ewma.columns
        ]
        # Pad if some spans missing
        full_vals = [df_ewma[col].corr(df_ewma[target])]
        for s in EWMA_CORR_SPANS:
            ename = f"{col}_ewma_{s}"
            full_vals.append(
                df_ewma[ename].corr(df_ewma[target])
                if ename in df_ewma.columns else np.nan
            )
        ax.plot(
            range(len(span_axis)), full_vals,
            "o-", label=col,
            color=VAR_COLORS.get(col), linewidth=1.8, alpha=0.85,
        )

    # ── physics cols — thicker dashed lines ───────────────────
    for col in phys_cols_:
        full_vals = [df_ewma[col].corr(df_ewma[target])]
        for s in EWMA_CORR_SPANS:
            ename = f"{col}_ewma_{s}"
            full_vals.append(
                df_ewma[ename].corr(df_ewma[target])
                if ename in df_ewma.columns else np.nan
            )
        ax.plot(
            range(len(span_axis)), full_vals,
            "s--", label=f"{col} ★",          # star marks physics
            color=VAR_COLORS.get(col, "#9b2226"),
            linewidth=2.8, alpha=0.95,
            markersize=7,
        )

    ax.set_xticks(range(len(span_axis)))
    ax.set_xticklabels(span_labels, fontsize=8)
    ax.set_ylabel(f"r with {target}", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)

    # Shade region for long spans (6000, 12000) to visually mark the new range
    long_span_start = span_axis.index(6000)
    ax.axvspan(long_span_start - 0.4, len(span_axis) - 0.6,
               alpha=0.07, color="purple",
               label="_long-span region (≥6000)")

    ax.legend(loc="best", fontsize=7.5, ncol=3,
              title="★ = physics feature", title_fontsize=7)
    ax.set_title(
        f"→ {target}  "
        f"(physics cols in dashed — should peak at higher correlation)",
        fontsize=10, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel(
    "EWMA span (samples)  [0 = raw]  |  purple shade = new long spans",
    fontsize=10,
)
fig.suptitle(
    "6c — Correlation vs EWMA Span per Temperature Target\n"
    "Solid = raw inputs,  Dashed ★ = physics features (i_squared, power_proxy)\n"
    "Peak of curve = dominant thermal time constant for that input→target pair",
    fontsize=12, fontweight="bold", y=1.01,
)
plt.tight_layout()
save("correlation_vs_ewma_span.png")

print(textwrap.dedent(f"""
  INTERPRETATION 6b/6c:
  ─────────────────────
  Physics cols (★):
    i_squared and power_proxy should reach HIGHER peak correlation than raw
    i_d/i_q because they are the actual heat sources (R·I²).
    If they don't peak higher, the motor may be operating mostly in
    field-weakening where i_d ≈ 0 and i_q carries all the load.

  Long spans (6000={round(6000/(SAMPLING_HZ*60))} min, 12000={round(12000/(SAMPLING_HZ*60))} min):
    If correlation continues to rise at these spans, the dominant thermal
    time constant is > {round(6000/(SAMPLING_HZ*60))} min — relevant for slow components
    (housing, coolant circuit). EWMA at these spans may be valuable inputs.
    If correlation plateaus or falls, adding those spans adds noise.

  Dashed lines rising above solid lines:
    i_squared > i_q in correlation → the MLP genuinely cannot discover
    the quadratic relationship on its own — add i_squared as explicit input.
"""))

print(textwrap.dedent("""
  INTERPRETATION 6b/6c:
  Peak of each curve = empirical thermal time constant for that pair.
  Monotone increase → dominant tau > longest span tested.
  Flat curve → temperature driven by quasi-steady load level, not dynamics.
"""))


# ═══════════════════════════════════════════════════════════════
# SECTION 6.5 — ROLLING FEATURES ANALYSIS  (new)
# ═══════════════════════════════════════════════════════════════
banner("SECTION 6.5 — ROLLING FEATURES: WHAT THEY CAPTURE & HOW THEY CORRELATE")

print(textwrap.dedent(f"""
  MOTIVATION
  ----------
  EWMA features summarise the LEVEL of recent activity (exponentially
  weighted average). Rolling statistics complement this by capturing
  different aspects of the recent window:

    roll_mean  ≈ average load over last W samples  (similar to EWMA but
                 rectangular window — all past samples weighted equally)
    roll_std   = variability / volatility of the signal in the window.
                 High std = the motor is being commanded erratically.
                 Rapid oscillations cause MORE heating per unit average
                 current than a steady-state command of the same level.
    roll_max   = peak stress reached in the window. Relevant for thermal
                 hotspots: a brief current spike raises local temperature
                 even if the window mean looks moderate.
    jump       = signed difference between current value and value W
                 steps ago. Captures whether the motor is being ramped up
                 or down — positive = ramp-up (heating), negative = ramp-down.

  Physics-informed base:
    i_squared  = i_d² + i_q²  (actual Joule heating source, not raw current)
    power_proxy = i_d·u_d + i_q·u_q  (instantaneous electrical power)

  Rolling windows: {ROLLING_WINDOWS} samples
  = {[round(w/(SAMPLING_HZ*60),1) for w in ROLLING_WINDOWS]} minutes at {SAMPLING_HZ} Hz

  Columns rolled: {ROLLING_COLS + PHYSICS_COLS}
"""))

# ── Compute rolling features ──────────────────────────────────
print("  Computing grouped rolling features …", flush=True)
roll_base_cols = ROLLING_COLS + [c for c in PHYSICS_COLS if c in df.columns]
all_roll_stats = ROLLING_STATS + ["jump"]   # include jump for ramp detection

df_roll, roll_col_names = add_rolling_grouped(
    df_ewma,            # already has EWMA cols — add rolling on top
    roll_base_cols,
    ROLLING_WINDOWS,
    stats=all_roll_stats,
)
print(f"  Added {len(roll_col_names)} rolling columns.")

# ── 6.5a: correlation table — rolling features vs temperature targets ──
print("\n  ROLLING FEATURE CORRELATION WITH TEMPERATURE TARGETS\n")
print(f"  {'Feature':<35s}", end="")
for t in TARGET_FEATURES:
    print(f"  {t:>16s}", end="")
print()
print("  " + "-" * (35 + 18 * len(TARGET_FEATURES)))

# Group printout by base column for readability
for base_col in roll_base_cols:
    # Raw base first
    if base_col in df_roll.columns:
        raw_c = [df_roll[base_col].corr(df_roll[t]) for t in TARGET_FEATURES]
        print(f"  {base_col:<35s}", end="")
        for v in raw_c:
            print(f"  {v:>+16.3f}", end="")
        print()

    for window in ROLLING_WINDOWS:
        for stat in all_roll_stats:
            col_name = f"{base_col}_roll{stat}_{window}"
            if col_name not in df_roll.columns:
                continue
            corrs = [df_roll[col_name].corr(df_roll[t]) for t in TARGET_FEATURES]
            tag   = f"  └─ roll{stat}_{window}"
            print(f"  {tag:<35s}", end="")
            for v in corrs:
                print(f"  {v:>+16.3f}", end="")
            print()
    print()

# ── 6.5b: correlation vs window size — one panel per (stat, target) ──────
# Key plot: for each rolling statistic, does correlation with temperature
# INCREASE with window size (long memory) or PEAK (finite time constant)?
print("\n  Plotting correlation vs window size for each statistic …", flush=True)

stat_titles = {
    "mean": "roll_mean  (average load — like rectangular-window EWMA)",
    "std":  "roll_std   (signal volatility — oscillation-induced heating)",
    "max":  "roll_max   (peak stress — relevant for thermal hotspots)",
    "jump": "jump       (signed ramp — positive = heating ramp, negative = cool-down)",
}

for stat in all_roll_stats:
    fig, axes = plt.subplots(
        len(TARGET_FEATURES), 1,
        figsize=(11, 3.2 * len(TARGET_FEATURES)),
        sharex=True,
    )
    window_axis   = [0] + ROLLING_WINDOWS
    window_labels = ["raw"] + [str(w) for w in ROLLING_WINDOWS]

    for ax, target in zip(axes, TARGET_FEATURES):
        for base_col in roll_base_cols:
            if base_col not in df_roll.columns:
                continue
            raw_r  = df_roll[base_col].corr(df_roll[target])
            roll_r = []
            for w in ROLLING_WINDOWS:
                cname = f"{base_col}_roll{stat}_{w}"
                roll_r.append(
                    df_roll[cname].corr(df_roll[target])
                    if cname in df_roll.columns else np.nan
                )
            ax.plot(
                range(len(window_axis)),
                [raw_r] + roll_r,
                "o-",
                label=base_col,
                color=VAR_COLORS.get(base_col),
                linewidth=1.8,
            )
        ax.set_xticks(range(len(window_axis)))
        ax.set_xticklabels(window_labels, fontsize=9)
        ax.set_ylabel(f"r with {target}", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
        ax.legend(loc="best", fontsize=7.5, ncol=3)
        ax.set_title(
            f"→ {target}",
            fontsize=10, fontweight="bold",
        )
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Rolling window (samples)  [0 = raw]", fontsize=10)
    fig.suptitle(
        f"6.5 — {stat_titles[stat]}\nCorrelation with temperature targets vs window size",
        fontsize=11, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    save(f"rolling_corr_vs_window_{stat}.png")

# ── 6.5c: EWMA vs rolling_mean comparison — do they carry the same info? ──
# If roll_mean and EWMA are nearly perfectly correlated with each other,
# adding both to the model just inflates the input dimension without
# adding new information.  If they diverge at certain window sizes,
# each carries complementary information.
print("\n  Plotting EWMA vs roll_mean redundancy check …", flush=True)

fig, axes = plt.subplots(
    len(EWMA_CORR_COLS), 1,
    figsize=(10, 2.8 * len(EWMA_CORR_COLS)),
    sharex=False,
)
if len(EWMA_CORR_COLS) == 1:
    axes = [axes]

for ax, col in zip(axes, EWMA_CORR_COLS):
    if col not in df_roll.columns:
        ax.set_visible(False)
        continue

    # For each span/window, compute correlation between EWMA_span and roll_mean_window
    # of the SAME base column → how redundant are they?
    pairs_checked = []
    for span, window in zip(EWMA_CORR_SPANS, ROLLING_WINDOWS):
        ewma_col = f"{col}_ewma_{span}"
        roll_col = f"{col}_rollmean_{window}"
        if ewma_col in df_roll.columns and roll_col in df_roll.columns:
            r = df_roll[ewma_col].corr(df_roll[roll_col])
            pairs_checked.append((span, window, r))

    if not pairs_checked:
        ax.set_visible(False)
        continue

    spans_used   = [p[0] for p in pairs_checked]
    windows_used = [p[1] for p in pairs_checked]
    r_vals       = [p[2] for p in pairs_checked]

    x = np.arange(len(pairs_checked))
    bars = ax.bar(x, r_vals, color=VAR_COLORS.get(col, C_ELEC),
                  alpha=0.8, edgecolor="white")
    for bar, rv in zip(bars, r_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{rv:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"ewma_{s}\nvs\nrollmean_{w}" for s, w in zip(spans_used, windows_used)],
        fontsize=8,
    )
    ax.set_ylim(0, 1.15)
    ax.axhline(0.99, color="red", linestyle="--", linewidth=1,
               label="r=0.99 (effectively redundant)")
    ax.axhline(0.95, color="orange", linestyle="--", linewidth=1,
               label="r=0.95")
    ax.set_ylabel("Correlation", fontsize=9)
    ax.set_title(f"{col}:  EWMA_span  vs  roll_mean_window  (same timescale)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3, axis="y")

fig.suptitle(
    "6.5c — EWMA vs Rolling Mean Redundancy\n"
    "r > 0.99 → features are effectively duplicates (waste of input dimensions)\n"
    "r < 0.95 → they capture different aspects of temporal history",
    fontsize=11, fontweight="bold", y=1.01,
)
plt.tight_layout()
save("ewma_vs_rollmean_redundancy.png")

# ── 6.5d: visual demo on a representative session ─────────────
# Show raw signal + EWMA + roll_mean + roll_std + roll_max + jump
# for one column on one profile, so it's immediately obvious what
# each feature "looks like" over actual motor operation.
print("\n  Plotting rolling feature visual demo …", flush=True)

median_len  = profile_counts.median()
rep_profile = (profile_counts - median_len).abs().idxmin()
demo_col    = "i_squared"   # physics-based: most interpretable for heating

rep_sub = df_roll[df_roll["profile_id"] == rep_profile].reset_index(drop=True)
t_axis  = rep_sub.index.to_numpy()

demo_window = ROLLING_WINDOWS[1]   # middle window (300 samples = 2.5 min)
demo_span   = EWMA_CORR_SPANS[1]   # matching EWMA span

feature_specs = [
    (demo_col,                                  "Raw signal",      "#9ca3af", "-"),
    (f"{demo_col}_ewma_{demo_span}",            f"EWMA span={demo_span}",  "#2e75b6", "-"),
    (f"{demo_col}_rollmean_{demo_window}",      f"roll_mean w={demo_window}", "#1f4e79", "--"),
    (f"{demo_col}_rollstd_{demo_window}",       f"roll_std  w={demo_window}", C_WARN,   "-"),
    (f"{demo_col}_rollmax_{demo_window}",       f"roll_max  w={demo_window}", C_TEMP,   ":"),
    (f"{demo_col}_jump_{demo_window}",          f"jump      w={demo_window}", "#7030a0", "-"),
]

# Two panels: top = level features (raw/ewma/rollmean/rollmax),
#             bottom = change features (roll_std, jump)
level_specs  = [s for s in feature_specs if "std" not in s[0] and "jump" not in s[0]]
change_specs = [s for s in feature_specs if "std"  in s[0] or  "jump"  in s[0]]

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

for col_name, label, color, ls in level_specs:
    if col_name not in rep_sub.columns:
        continue
    y = rep_sub[col_name].to_numpy()
    # Normalise to [0,1] for overlay readability
    rng = y.max() - y.min()
    y_n = (y - y.min()) / (rng if rng > 1e-9 else 1.0)
    ax_top.plot(t_axis, y_n, color=color, linestyle=ls, linewidth=1.4,
                label=label, alpha=0.85)

ax_top.set_ylabel(f"Normalised  {demo_col}  (0–1)", fontsize=10)
ax_top.set_title(
    f"Level features on profile {rep_profile}  "
    f"(window={demo_window} samples = "
    f"{demo_window/(SAMPLING_HZ*60):.1f} min,  "
    f"EWMA span={demo_span})",
    fontsize=11, fontweight="bold",
)
ax_top.legend(loc="upper right", fontsize=8, ncol=2)
ax_top.grid(True, alpha=0.3)

for col_name, label, color, ls in change_specs:
    if col_name not in rep_sub.columns:
        continue
    y = rep_sub[col_name].to_numpy()
    ax_bot.plot(t_axis, y, color=color, linestyle=ls, linewidth=1.2,
                label=label, alpha=0.85)

ax_bot.axhline(0, color="black", linewidth=0.7, alpha=0.5)
ax_bot.set_ylabel("Change / volatility features  (original units)", fontsize=10)
ax_bot.set_xlabel(f"Sample index within profile {rep_profile}", fontsize=10)
ax_bot.set_title(
    "Change features: roll_std (volatility = oscillation-induced extra heating)  "
    "+ jump (signed ramp — heating direction)",
    fontsize=10, fontweight="bold",
)
ax_bot.legend(loc="upper right", fontsize=8)
ax_bot.grid(True, alpha=0.3)

fig.suptitle(
    f"6.5d — Rolling Feature Visual Demo on {demo_col}\n"
    "Top: level features (smooth differently) — "
    "Bottom: change features (detect transients)",
    fontsize=12, fontweight="bold",
)
plt.tight_layout()
save("rolling_feature_visual_demo.png")

# ── 6.5e: heatmap — ALL rolling feature correlations with ALL targets ────
# One compact heatmap rows=features, cols=targets.
# Organised as: raw → rollmean → rollstd → rollmax → jump per base col.
print("\n  Plotting rolling features correlation summary heatmap …", flush=True)

roll_heatmap_rows = []
roll_heatmap_index = []

for base_col in roll_base_cols:
    if base_col not in df_roll.columns:
        continue
    row = {t: df_roll[base_col].corr(df_roll[t]) for t in TARGET_FEATURES}
    roll_heatmap_rows.append(row)
    roll_heatmap_index.append(f"{base_col}  [raw]")

    for stat in all_roll_stats:
        for w in ROLLING_WINDOWS:
            cname = f"{base_col}_roll{stat}_{w}"
            if cname not in df_roll.columns:
                continue
            row = {t: df_roll[cname].corr(df_roll[t]) for t in TARGET_FEATURES}
            roll_heatmap_rows.append(row)
            roll_heatmap_index.append(f"  roll{stat}_{w}")

roll_heatmap_df = pd.DataFrame(roll_heatmap_rows, index=roll_heatmap_index)

fig, ax = plt.subplots(figsize=(9, max(8, len(roll_heatmap_df) * 0.38)))
sns.heatmap(
    roll_heatmap_df,
    annot=True, fmt=".2f",
    cmap="RdYlBu_r", center=0, vmin=-1, vmax=1,
    linewidths=0.4, ax=ax,
    annot_kws={"size": 8},
    cbar_kws={"label": "Pearson r with temperature target"},
)
ax.set_title(
    "6.5e — Rolling Feature Correlation Summary\n"
    "Rows: raw + every rolling statistic × window.  "
    "Cols: temperature targets.\n"
    "Bold rows where |r| increases over raw → feature adds information.",
    fontsize=11, fontweight="bold", pad=12,
)
ax.set_xlabel("Temperature target", fontsize=10)
ax.tick_params(axis="y", labelsize=8)

# Horizontal lines to separate base-column blocks
block_starts = [0]
for i, label in enumerate(roll_heatmap_index):
    if not label.startswith("  ") and i > 0:
        block_starts.append(i)
block_starts.append(len(roll_heatmap_index))
for bs in block_starts[1:-1]:
    ax.axhline(y=bs, color="black", linewidth=1.2, linestyle="--", alpha=0.5)

plt.tight_layout()
save("rolling_corr_summary_heatmap.png")

print(textwrap.dedent(f"""
  INTERPRETATION OF SECTION 6.5
  ──────────────────────────────
  roll_mean vs EWMA (6.5c):
    If r > 0.99 for matched timescales, they are near-identical.
    Use only ONE (prefer EWMA — no NaNs, handles session start better).
    If r < 0.95, they diverge: EWMA weights recent samples more heavily
    while roll_mean gives equal weight — both may be worth keeping.

  roll_std (volatility):
    If roll_std correlates positively with temperature ABOVE the
    roll_mean correlation, the motor's temperature depends not just
    on average load but on HOW ERRATICALLY the load oscillates.
    This is physically expected: high-frequency current ripple causes
    eddy-current and resistive heating beyond what DC equivalent predicts.

  roll_max (peak stress):
    Should correlate with temperature at least as well as roll_mean.
    If roll_max >> roll_mean in correlation, hotspot formation dominates
    and peak current matters more than average current.

  jump (signed ramp):
    Positive correlation with temperature → temperature lags behind
    current ramp-ups (expected from thermal inertia). If the jump
    feature's correlation is near zero, the model already captures
    ramps via EWMA/rollmean and the jump adds no extra information.

  Physics columns (i_squared, power_proxy):
    Should have HIGHER correlation with temperature than raw i_q, i_d
    because they are the actual heat source in the thermal ODE.
    If not, check whether your dataset has a significant portion of
    field-weakening operation where the relationship changes.
"""))


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — CORRELATION WITH STATOR_WINDING  (unchanged)
# ═══════════════════════════════════════════════════════════════
banner("SECTION 7 — CORRELATION WITH STATOR_WINDING")

# ── Section 7: correlation with stator_winding — now includes physics ─────────
banner("SECTION 7 — CORRELATION WITH STATOR_WINDING (incl. physics features)")

# Include all raw inputs + physics cols, exclude the other temperature targets
non_sw_vars = [
    v for v in plot_vars_6a          # plot_vars_6a already has physics cols
    if v != "stator_winding"
    and v not in ["pm", "stator_tooth", "stator_yoke"]
]

sw_corr     = df[non_sw_vars].corrwith(df["stator_winding"]).sort_values()
is_physics  = sw_corr.index.isin(PHYSICS_COLS)

# Color by sign; physics bars get a special hatch pattern
bar_colors_sw = []
bar_hatches   = []
for feat, is_phys in zip(sw_corr.index, is_physics):
    bar_colors_sw.append("#9b2226" if is_phys else
                         ("#2e75b6" if sw_corr[feat] >= 0 else "#c00000"))
    bar_hatches.append("///" if is_phys else "")

fig, ax = plt.subplots(figsize=(10, max(7, len(sw_corr) * 0.42)))
bars = ax.barh(
    sw_corr.index, sw_corr.values,
    color=bar_colors_sw, hatch=bar_hatches,
    edgecolor="white", alpha=0.9, height=0.65,
)
for bar, val, hatch in zip(bars, sw_corr.values, bar_hatches):
    bar.set_hatch(hatch)
    offset = 0.02 if val >= 0 else -0.02
    ha     = "left" if val >= 0 else "right"
    ax.text(
        val + offset, bar.get_y() + bar.get_height() / 2,
        f"{val:+.3f}", va="center", ha=ha,
        fontsize=9, fontweight="bold",
    )

ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Pearson Correlation with stator_winding", fontsize=11)
ax.set_title(
    "Correlation of Each Variable with Stator Winding Temperature\n"
    "Hatched bars (dark red) = physics-derived features (i_squared, power_proxy)",
    fontsize=12, fontweight="bold",
)
ax.set_xlim(-1.05, 1.05)
ax.legend(handles=[
    mpatches.Patch(color="#2e75b6",  label="Raw input (positive corr)"),
    mpatches.Patch(color="#c00000",  label="Raw input (negative corr)"),
    mpatches.Patch(color="#9b2226",  hatch="///", label="Physics feature"),
], loc="lower right", fontsize=10)
plt.tight_layout()
save("correlation_with_stator_winding.png")


# ═══════════════════════════════════════════════════════════════
# SECTION 12 — CROSS-SESSION BEHAVIOURAL CONSISTENCY
# (unchanged — paste your existing Section 12 here)
# ═══════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════
# SECTION 13 — FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════
banner("SECTION 13 — FINAL SUMMARY")

print(textwrap.dedent(f"""
  NEW IN THIS REPORT
  ------------------
  Section 6   — EWMA correlation heatmap + correlation-vs-span line plots.
  Section 6.5 — Rolling features (mean/std/max/jump) analysis:
                  6.5a  correlation table (all rolling features × all targets)
                  6.5b  correlation vs window size — 4 plots, one per stat
                  6.5c  EWMA vs roll_mean redundancy check
                  6.5d  visual demo on representative session
                  6.5e  compact summary heatmap (all rolling × all targets)
  Section 12  — Cross-session input/output consistency.
"""))

print("=" * 65)
print("  Done — all plots saved in plots_ewma/")
print("=" * 65)