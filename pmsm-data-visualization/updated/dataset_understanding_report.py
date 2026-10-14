"""
=================================================================
  PMSM Electric Motor Temperature — Dataset Exploration
  Case Study: Neural Network Pruning (CIT413058, TU Munich)
=================================================================
  This script explores the PMSM dataset: what it contains, what each
  variable means, and how the inputs and outputs relate to each other.
  The goal is to understand the data before using it for the pruning
  experiments later on.

  Run:  python dataset_understanding_report.py
  Deps: pandas, numpy, matplotlib, seaborn
  Data: place measures_v2.csv in data/ folder
=================================================================
"""

import os
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
DATA_PATH = "data/measures_v2.csv"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

INPUT_FEATURES  = ["u_q", "u_d", "i_q", "i_d", "motor_speed", "torque", "ambient", "coolant"]
TARGET_FEATURES = ["pm", "stator_winding", "stator_tooth", "stator_yoke"]

# Variables ordered by physical category — used for the correlation heatmap
# so related variables sit next to each other (Electrical, Mechanical,
# Environmental, then the four temperature Targets).
ORDERED_VARS = [
    "u_q", "u_d", "i_q", "i_d",          # Electrical
    "motor_speed", "torque",             # Mechanical
    "ambient", "coolant",                # Environmental
    "pm", "stator_winding", "stator_tooth", "stator_yoke",  # Targets
]

# ── Plot style ──────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})

# Color scheme — by physical meaning
C_ELEC = "#2e75b6"   # electrical (blue)
C_MECH = "#1f4e79"   # mechanical (dark blue)
C_ENV  = "#0070c0"   # environmental (light blue)
C_TEMP = "#c00000"   # temperature outputs (red)

# One consistent color per variable, reused across every figure so the
# same variable always looks the same wherever it appears.
VAR_COLORS = {
    "motor_speed": "#1f4e79", "torque": "#e36c09",
    "u_q": "#2e75b6", "u_d": "#c00000", "i_q": "#375623", "i_d": "#7030a0",
    "ambient": "#0070c0", "coolant": "#00b0f0",
    "pm": "#c00000", "stator_winding": "#e36c09",
    "stator_tooth": "#0070c0", "stator_yoke": "#7030a0",
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


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — DATASET OVERVIEW
# ═══════════════════════════════════════════════════════════════
banner("SECTION 1 — DATASET OVERVIEW")

df = pd.read_csv(DATA_PATH)

n_samples  = len(df)
n_vars     = df.shape[1]
n_inputs   = len(INPUT_FEATURES)
n_outputs  = len(TARGET_FEATURES)
n_profiles = df["profile_id"].nunique()
hours      = n_samples / (2 * 3600)  # 2 Hz sampling → 0.5 s per row

print(textwrap.dedent(f"""
  WHAT IS A PMSM?
  ---------------
  A PMSM (Permanent Magnet Synchronous Motor) is an electric motor that
  uses permanent magnets mounted on its rotating part (the rotor) to
  produce motion. When electric current is supplied to the surrounding
  coils (the stator), it creates a rotating magnetic field that drags the
  magnetic rotor around with it. PMSMs are highly efficient and are the
  standard motor type used in electric vehicles, drones, and industrial
  drives.

  When a PMSM runs, it generates heat. If it gets too hot, the permanent
  magnets can be permanently damaged (demagnetized). Measuring the
  internal temperatures directly is difficult and expensive in a real
  vehicle, so engineers want to PREDICT them from signals that are easy
  to measure (voltages, currents, speed, etc.).

  GOAL
  ----
  The primary objective is to predict the STATOR WINDING TEMPERATURE from
  the remaining available signals. The stator winding is especially hard
  to instrument with a physical sensor in production motors, so a neural
  network that predicts it accurately would eliminate the need for one.

  The available input signals include electrical measurements (u_q, u_d,
  i_q, i_d), mechanical measurements (motor_speed, torque), environmental
  measurements (ambient, coolant), and optionally the other temperature
  readings (pm, stator_tooth, stator_yoke). Part of the investigation is
  to determine WHICH of these signals are actually useful — in particular,
  whether accurate predictions are possible without relying on the other
  temperature sensors.

  DATASET SIZE
  ------------
  Total samples      : {n_samples:,}
  Total variables    : {n_vars}  (up to 11 inputs + 1 target + 1 metadata)
  Operating profiles : {n_profiles}
  Sampling frequency : 2 Hz
  Total duration     : {hours:.1f} hours ({hours/24:.1f} days)
  Missing values     : {df.isnull().sum().sum()}
"""))


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — VARIABLE EXPLANATION TABLE
# ═══════════════════════════════════════════════════════════════
banner("SECTION 2 — VARIABLE EXPLANATION TABLE")

var_table = pd.DataFrame([
    ("u_q",            "q-axis voltage",            "Voltage driving the torque-producing direction of the motor", "Electrical",    "Input"),
    ("u_d",            "d-axis voltage",            "Voltage in the magnetic-field direction of the motor",        "Electrical",    "Input"),
    ("i_q",            "q-axis current",            "Current that actually produces the rotating torque",          "Electrical",    "Input"),
    ("i_d",            "d-axis current",            "Current that controls the magnetic field strength",           "Electrical",    "Input"),
    ("motor_speed",    "Rotational speed",          "How fast the motor shaft spins (revolutions per minute)",      "Mechanical",    "Input"),
    ("torque",         "Motor torque",              "The turning force the motor produces (Newton-metres)",         "Mechanical",    "Input"),
    ("ambient",        "Ambient temperature",       "Temperature of the surrounding air",                          "Environmental", "Input"),
    ("coolant",        "Coolant temperature",       "Temperature of the liquid that cools the motor",              "Environmental", "Input"),
    ("pm",             "Permanent magnet temp.",    "Temperature of the rotor magnets",                            "Thermal",       "Input (optional)"),
    ("stator_tooth",   "Stator tooth temp.",        "Temperature of the iron teeth holding the coils",             "Thermal",       "Input (optional)"),
    ("stator_yoke",    "Stator yoke temp.",         "Temperature of the outer iron ring of the stator",            "Thermal",       "Input (optional)"),
    ("stator_winding", "Stator winding temp.",      "Temperature of the copper coils — PREDICTION TARGET",         "Thermal",       "Target"),
    ("profile_id",     "Operating session ID",      "Label marking which recording session a row belongs to",      "Metadata",      "Grouping"),
], columns=["Variable", "Full Name", "Physical Meaning (plain language)", "Category", "Role"])

# Print as readable text
with pd.option_context("display.max_colwidth", None, "display.width", 200):
    print("\n" + var_table.to_string(index=False))


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — EXPLAIN profile_id  (+ Plot 1)
# ═══════════════════════════════════════════════════════════════
banner("SECTION 3 — UNDERSTANDING profile_id")

profile_counts = df["profile_id"].value_counts().sort_index()
largest, smallest = profile_counts.idxmax(), profile_counts.idxmin()

print(textwrap.dedent(f"""
  WHAT IS profile_id?
  -------------------
  The data was NOT recorded in one long continuous run. Instead, engineers
  ran the motor through {n_profiles} separate test sessions. Each session
  used a different driving pattern (different speeds and loads over time).
  Every row is tagged with the session it came from — that tag is profile_id.

  WHY DOES IT EXIST?
  ------------------
  It lets us tell where one recording session ends and the next begins.
  Within a single profile the rows are in time order; across profiles they
  are independent experiments.

  WHY IS IT NOT A NORMAL SENSOR MEASUREMENT?
  ------------------------------------------
  profile_id is just a bookkeeping label (1, 2, 3, ...). It is not a
  physical quantity, it carries no units, and it should NEVER be fed into
  a model as an input feature. A higher profile_id does not mean "more"
  of anything.

  WHY IS IT USEFUL?
  -----------------
  It is essential for honest model evaluation. When splitting data into
  training and test sets, we must keep whole profiles together. If rows
  from the same session appear in both training and test sets, the model
  effectively "sees the answers" and scores look better than they really
  are. Splitting BY profile_id prevents this leakage.

  Largest profile : ID {largest} ({profile_counts.max():,} samples)
  Smallest profile: ID {smallest} ({profile_counts.min():,} samples)
"""))

# ── Plot 1: recording duration per profile ───────────────────
profile_durations_hrs = profile_counts / (2 * 3600)  # 2 Hz → hours

fig, ax = plt.subplots(figsize=(11, 14))
bars = ax.barh(profile_durations_hrs.index.astype(str),
               profile_durations_hrs.values,
               color=C_ELEC, edgecolor="white", linewidth=0.4, height=0.7)
for bar, val in zip(bars, profile_durations_hrs.values):
    ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
            f"{val:.2f} h", va="center", fontsize=7)
ax.set_xlabel("Recording Duration (hours)", fontsize=11)
ax.set_ylabel("Profile ID", fontsize=11)
ax.set_title("Recording Duration of Each Operating Profile",
             fontsize=13, fontweight="bold")
ax.invert_yaxis()

# Statistics box
stats_text = (
    f"Total profiles: {len(profile_durations_hrs)}\n"
    f"Total duration: {profile_durations_hrs.sum():.1f} hours\n"
    f"Longest: profile {profile_durations_hrs.idxmax()} ({profile_durations_hrs.max():.2f} h)\n"
    f"Shortest: profile {profile_durations_hrs.idxmin()} ({profile_durations_hrs.min():.2f} h)\n"
    f"Median: {profile_durations_hrs.median():.2f} hours\n"
    f"Sampling frequency: 2 Hz"
)
ax.text(0.97, 0.02, stats_text, transform=ax.transAxes,
        fontsize=9, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.5", fc="#f0f4f8", ec="#d1d5db", alpha=0.95))

plt.tight_layout()
save("profile_durations.png")

print("""
  INTERPRETATION: Sessions vary a lot in length — from about 18 minutes
  to over 6 hours. This is why a simple random split would be unfair:
  a few long profiles dominate the data. Splitting by whole profiles
  keeps evaluation honest.""")


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — PMSM SYSTEM FLOW DIAGRAM  (the main overview figure)
# ═══════════════════════════════════════════════════════════════
banner("SECTION 4 — PMSM SYSTEM FLOW DIAGRAM")

fig, ax = plt.subplots(figsize=(13, 9.5))
ax.set_xlim(0, 13); ax.set_ylim(0, 10); ax.axis("off")

def box(x, y, w, h, text, color, text_color="white", fs=10.5, bold=True):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                       linewidth=1.5, edgecolor="white", facecolor=color, alpha=0.95)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            color=text_color, fontsize=fs, fontweight="bold" if bold else "normal",
            linespacing=1.4)

# ── Title ──
ax.text(6.5, 9.5, "PMSM Dataset — Predicting Stator Winding Temperature",
        ha="center", fontsize=14, fontweight="bold", color="#1f2937")

# ── Available input signal groups (left column) ──
box(0.3, 7.2, 3.8, 1.3,
    "ELECTRICAL\n\nu_q   u_d   i_q   i_d", C_ELEC, fs=10)
box(0.3, 5.5, 3.8, 1.3,
    "MECHANICAL\n\nmotor_speed   torque", C_MECH, fs=10)
box(0.3, 3.8, 3.8, 1.3,
    "ENVIRONMENTAL\n\nambient   coolant", C_ENV, fs=10)
box(0.3, 2.1, 3.8, 1.3,
    "ADDITIONAL TEMPERATURES\n\npm   stator_tooth   stator_yoke", "#7c3aed", fs=10)

# ── Label above inputs ──
ax.text(2.2, 8.75, "Available Input Signals",
        ha="center", fontsize=11.5, fontweight="bold", color="#374151")

# ── Neural Network (center) ──
box(5.5, 4.2, 2.5, 2.5, "NEURAL\nNETWORK\n\n(baseline\nMLP)", "#374151", fs=12)

# ── Target (right) ──
box(9.2, 4.5, 3.3, 1.9,
    "TARGET\n\nstator_winding\n(°C)",
    C_TEMP, fs=12)

# ── Arrows from inputs to NN ──
arrow_kw = dict(arrowstyle="-|>", mutation_scale=22, linewidth=2, color="#6b7280")
for y_src in (7.85, 6.15, 4.45, 2.75):
    ax.add_patch(FancyArrowPatch((4.15, y_src), (5.45, 5.45),
                 connectionstyle="arc3,rad=0.0", **arrow_kw))

# ── Arrow NN -> target ──
ax.add_patch(FancyArrowPatch((8.05, 5.45), (9.15, 5.45),
             arrowstyle="-|>", mutation_scale=26, linewidth=2.5, color="#374151"))

# ── Captions ──
ax.text(4.8, 5.85, "feed into", ha="center", fontsize=8, color="#6b7280", style="italic")
ax.text(8.6, 5.85, "predicts", ha="center", fontsize=8.5, color="#374151", style="italic", fontweight="bold")

# ── Note about optional inputs ──
ax.text(2.2, 1.5, "* pm, stator_tooth, stator_yoke are optional inputs.\n"
        "  The investigation tests whether accurate predictions\n"
        "  are possible without them.",
        ha="center", fontsize=9, color="#7c3aed", style="italic")

# ── Bottom explanatory strip ──
ax.text(6.5, 0.5,
        "The goal is to predict stator_winding temperature from the available signals.\n"
        "Different input configurations are tested to determine which sensors are truly needed.",
        ha="center", fontsize=9.5, color="#4b5563", linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.6", fc="#f9fafb", ec="#e5e7eb"))

plt.tight_layout()
save("pmsm_system_flow.png")


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — REPRESENTATIVE OPERATING SESSION  (Plot 3)
# ═══════════════════════════════════════════════════════════════
banner("SECTION 5 — REPRESENTATIVE OPERATING SESSION")

# Auto-select profile closest to the median length
median_len = profile_counts.median()
rep_profile = (profile_counts - median_len).abs().idxmin()
prof = df[df["profile_id"] == rep_profile].reset_index(drop=True)
print(f"  Auto-selected representative profile: ID {rep_profile} ({len(prof):,} samples)")

# Three grouped bands sharing one x-axis
band_specs = [
    ("Mechanical signals",    ["motor_speed", "torque"]),
    ("Electrical signals",    ["u_q", "u_d", "i_q", "i_d"]),
    ("Temperature signals",   ["pm", "stator_winding", "stator_tooth", "stator_yoke"]),
]

fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
for ax, (band_name, cols) in zip(axes, band_specs):
    for col in cols:
        ax.plot(prof.index, prof[col], linewidth=0.9, label=col, color=VAR_COLORS[col])
    ax.set_ylabel(band_name, fontsize=10.5, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=len(cols))
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Sample index within profile (time \u2192)", fontsize=11)
fig.suptitle(f"Representative Operating Session — Profile {rep_profile}",
             fontsize=13, fontweight="bold", y=0.995)
plt.tight_layout()
save("causal_timeline_profile.png")
print("""
  This figure shows how the measured signals evolve during one
  representative operating session. Electrical and mechanical signals
  vary rapidly, while temperature signals change more gradually
  because of thermal inertia.""")


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — CORRELATION HEATMAP  (Plot 4)
# ═══════════════════════════════════════════════════════════════
banner("SECTION 6 — CORRELATION HEATMAP")

print(textwrap.dedent("""
  WHAT IS CORRELATION?
  --------------------
  Correlation measures how strongly two variables move together, on a
  scale from -1 to +1:
    +1  = they rise and fall perfectly together
     0  = no linear relationship
    -1  = when one rises, the other falls
  We calculate it to see, at a glance, which inputs are linked to which
  output temperatures, and how related the outputs are to each other.
"""))

# Correlate using the category-ordered variable list so related variables
# (electrical, mechanical, environmental, targets) sit next to each other.
corr = df[ORDERED_VARS].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
            center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
            annot_kws={"size": 9}, cbar_kws={"label": "Pearson correlation"})
ax.set_title("Correlation Heatmap of PMSM Variables (excluding profile_id)",
             fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
save("correlation_heatmap.png")

# ── Auto-generated insights ─────────────────────────────────
pairs = (corr.where(~np.eye(len(corr), dtype=bool)).stack())
# dedupe symmetric pairs
seen, uniq = set(), []
for (a, b), v in pairs.items():
    if (b, a) not in seen:
        seen.add((a, b)); uniq.append(((a, b), v))
uniq_s = sorted(uniq, key=lambda kv: kv[1])

print("  STRONGEST POSITIVE CORRELATIONS:")
for (a, b), v in uniq_s[-4:][::-1]:
    print(f"    {a:14s} <-> {b:14s}  r = {v:+.3f}")

print("\n  STRONGEST NEGATIVE CORRELATIONS:")
for (a, b), v in uniq_s[:4]:
    print(f"    {a:14s} <-> {b:14s}  r = {v:+.3f}")

print("\n  TARGET <-> TARGET CORRELATIONS:")
tcorr = corr.loc[TARGET_FEATURES, TARGET_FEATURES]
for i, a in enumerate(TARGET_FEATURES):
    for b in TARGET_FEATURES[i+1:]:
        print(f"    {a:14s} <-> {b:14s}  r = {tcorr.loc[a,b]:+.3f}")

print(textwrap.dedent("""
  KEY INSIGHTS:
  1. The three stator temperatures (winding, tooth, yoke) are very highly
     correlated with each other (r > 0.85) — they heat up almost together.
  2. pm (magnet temperature) is related to the stator temps but more
     loosely — it behaves the most independently of the four targets.
  3. Among inputs, the environmental temperatures (ambient, coolant) and
     motor_speed show the clearest links to the output temperatures.
  4. i_d tends to move opposite to magnet temperature (negative
     correlation), reflecting how field-weakening current is used at
     different operating points.
  5. No single input perfectly predicts the temperatures — the relationship
     is multi-variable and partly non-linear, which is exactly why a neural
     network is a sensible modelling choice.
"""))


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — CORRELATION WITH STATOR_WINDING  (Plot 5)
# ═══════════════════════════════════════════════════════════════
banner("SECTION 7 — CORRELATION WITH STATOR_WINDING")

print(textwrap.dedent("""
  Since stator_winding is the prediction target, it is useful to see how
  strongly each other variable correlates with it. This helps motivate
  which signals are likely to be the most informative inputs.
"""))

# Compute correlations with stator_winding, exclude self and profile_id
all_vars_no_target = [v for v in ORDERED_VARS if v != "stator_winding"]
sw_corr = df[all_vars_no_target].corrwith(df["stator_winding"]).sort_values()

# Color by sign: positive = blue, negative = orange-red
bar_colors_sw = ["#2e75b6" if v >= 0 else "#c00000" for v in sw_corr.values]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(sw_corr.index, sw_corr.values, color=bar_colors_sw,
               edgecolor="white", alpha=0.9, height=0.65)
for bar, val in zip(bars, sw_corr.values):
    offset = 0.02 if val >= 0 else -0.02
    ha = "left" if val >= 0 else "right"
    ax.text(val + offset, bar.get_y() + bar.get_height()/2,
            f"{val:+.3f}", va="center", ha=ha, fontsize=9, fontweight="bold")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Pearson Correlation with stator_winding", fontsize=11)
ax.set_ylabel("Variable", fontsize=11)
ax.set_title("Correlation of Each Variable with Stator Winding Temperature",
             fontsize=13, fontweight="bold")
ax.set_xlim(-1.05, 1.05)
ax.legend(handles=[mpatches.Patch(color="#2e75b6", label="Positive"),
                   mpatches.Patch(color="#c00000", label="Negative")],
          loc="lower right", fontsize=10)
plt.tight_layout()
save("correlation_with_stator_winding.png")

# Print ranking
print("\n  CORRELATION RANKING (strongest to weakest):")
for var, val in sw_corr.sort_values(ascending=False).items():
    marker = "***" if abs(val) > 0.7 else "**" if abs(val) > 0.4 else "*" if abs(val) > 0.2 else ""
    print(f"    {var:16s}  r = {val:+.3f}  {marker}")

print(textwrap.dedent("""
  INTERPRETATION:
  The three stator temperatures (stator_tooth, stator_yoke, pm) are the
  most strongly correlated with stator_winding — which makes physical
  sense since they are all measuring heat in nearby motor components.
  However, using them as inputs partly defeats the purpose of a virtual
  sensor (you already have temperature sensors installed).

  Among the non-temperature inputs, coolant, i_d, and motor_speed show
  the strongest relationships with stator_winding. The electrical signals
  (u_q, u_d, i_q) have weaker individual correlations, but in combination
  they may still carry useful predictive information — this is where the
  neural network comes in.
"""))


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — SIGNAL FLUCTUATION  (Plot 6)
# ═══════════════════════════════════════════════════════════════
banner("SECTION 8 — SIGNAL FLUCTUATION (STEP-TO-STEP VARIATION)")

print(textwrap.dedent("""
  SIGNAL FLUCTUATION
  ------------------
  This plot shows how much each signal changes from one timestep to the
  next. We measure it as the average absolute step change divided by the
  signal's own spread (standard deviation), so all signals compare fairly
  regardless of their units. Differences are computed WITHIN each profile,
  so the jump between separate sessions is never counted.

  Larger values mean the signal changes more rapidly from one sample to
  the next. Smaller values mean it changes more gradually.
"""))

# First differences within each profile (NaN at each session boundary -> ignored)
diffs = df.groupby("profile_id")[ORDERED_VARS].diff()

volatility = {}
for col in ORDERED_VARS:
    volatility[col] = diffs[col].abs().mean() / df[col].std()
vol_series = pd.Series(volatility).sort_values(ascending=True)

# Color by type: temperature signals red, other signals blue
bar_colors = [C_TEMP if v in TARGET_FEATURES else C_ELEC for v in vol_series.index]

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(vol_series.index, vol_series.values, color=bar_colors,
               edgecolor="white", alpha=0.9)
for bar, val in zip(bars, vol_series.values):
    ax.text(val + val*0.02, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=8.5)
ax.set_xlabel("Normalized step-to-step variation   (mean |Δ| ÷ standard deviation)", fontsize=11)
ax.set_ylabel("Variable", fontsize=11)
ax.set_title("Signal Fluctuation — Step-to-Step Variation", fontsize=13, fontweight="bold")
ax.legend(handles=[mpatches.Patch(color=C_ELEC, label="Electrical / Mechanical / Environmental"),
                   mpatches.Patch(color=C_TEMP, label="Temperature signals")],
          loc="lower right", fontsize=9)
plt.tight_layout()
save("bumpiness_analysis.png")
print("""
  The electrical and mechanical signals fluctuate the most — they react
  quickly to changes in the driving pattern. The temperature signals are
  the smoothest, which makes sense because temperatures are governed by
  thermal inertia and change gradually.""")


# ═══════════════════════════════════════════════════════════════
# SECTION 9 — FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════
banner("SECTION 9 — FINAL SUMMARY")

print(textwrap.dedent(f"""
  WHAT THE DATASET MEASURES
  -------------------------
  Real recordings from a Permanent Magnet Synchronous Motor on a test
  bench: {n_samples:,} samples across {n_profiles} operating sessions,
  sampled at 2 Hz ({hours:.1f} hours of motor operation).

  PREDICTION TARGET
  -----------------
  stator_winding — the temperature of the copper coils inside the
  stator. This is the hardest component to instrument with a physical
  sensor in production motors.

  AVAILABLE INPUT SIGNALS
  -----------------------
  Easy-to-measure signals describing how the motor is being driven:
    - Electrical    : u_q, u_d, i_q, i_d
    - Mechanical    : motor_speed, torque
    - Environmental : ambient, coolant
  Plus optionally the other temperature measurements:
    - Thermal       : pm, stator_tooth, stator_yoke

  WHAT profile_id REPRESENTS
  --------------------------
  A bookkeeping label for each recording session. Not a sensor value,
  never a model input, but essential for splitting data correctly.

  WHY IT SUITS NEURAL-NETWORK REGRESSION AND PRUNING
  --------------------------------------------------
  - Large, clean, real-world data with no missing values.
  - A regression task with a smooth, learnable target.
  - Relationships are multi-variable and partly non-linear, so a neural
    network is well justified.
  - Different input configurations can be tested to find the minimum set
    of sensors needed for accurate prediction.
  - Because the target is smooth and well-behaved, the trained network
    can likely be PRUNED (compressed) substantially while still predicting
    stator_winding accurately — which is exactly what this case study tests.
"""))

print("=" * 65)
print("  Done — all plots saved in plots/")
print("=" * 65)
