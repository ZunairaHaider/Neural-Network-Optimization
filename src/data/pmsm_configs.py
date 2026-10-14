"""PMSM baseline input-output configurations.

Single source of truth for which columns are inputs and which is the target
for each baseline experiment.  Every PMSM-aware module (loader, trainer,
pruner, comparison) imports from here so the column lists never go out of
sync.

Naming convention
-----------------
  PMSM_all_in      — all 11 sensor signals as inputs (upper bound)
  PMSM_no_temps    — only electrical + coolant (8 inputs, primary Schaeffler ask)
  PMSM_no_temps_pm — electrical + coolant + pm (9 inputs, tests pm value)

The target is always ``stator_winding`` (difficult-to-measure in practice).
"""

from __future__ import annotations

# ── target ────────────────────────────────────────────────────────────────────
TARGET = "stator_winding"

# ── shared input groups ───────────────────────────────────────────────────────
ELECTRICAL = ["ambient", "u_d", "u_q", "motor_speed", "i_d", "i_q", "torque"]

# ── per-baseline input lists ──────────────────────────────────────────────────
CONFIGS: dict[str, list[str]] = {
    "PMSM_all_in": ELECTRICAL + ["coolant", "pm", "stator_tooth", "stator_yoke"],
    "PMSM_no_temps": ELECTRICAL + ["coolant"],
    "PMSM_no_temps_pm": ELECTRICAL + ["coolant", "pm"],
}

# ── per-baseline input lists ──────────────────────────────────────────────────
CONFIGS_EWMA: dict[str, list[str]] = {
    "PMSM_all_in_ewma": ELECTRICAL + ["coolant", "pm", "stator_tooth", "stator_yoke"],
    "PMSM_no_temps_ewma": ELECTRICAL + ["coolant"],
    "PMSM_no_temps_pm_ewma": ELECTRICAL + ["coolant", "pm"],
}

# ── convenience ───────────────────────────────────────────────────────────────
ALL_CONFIG_NAMES: list[str] = list(CONFIGS.keys())
EWMA_CONFIG_NAMES: list[str] = list(CONFIGS_EWMA.keys())


def get_config(name: str) -> tuple[list[str], str]:
    """Return (input_columns, target_column) for a named configuration.

    Raises KeyError with a helpful message if the name is unknown.
    """
    if name not in CONFIGS:
        raise KeyError(
            f"Unknown PMSM config {name!r}. "
            f"Available: {ALL_CONFIG_NAMES}"
        )
    return CONFIGS[name], TARGET
