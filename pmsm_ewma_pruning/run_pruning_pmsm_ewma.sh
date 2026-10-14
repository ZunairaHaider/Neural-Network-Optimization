#!/usr/bin/env bash
# Run every pruning method currently wired into pruner.py against Simon's
# three new EWMA PMSM baselines (PMSM_no_temps_ewma, PMSM_no_temps_pm_ewma,
# PMSM_all_in_ewma).
#
# PREREQUISITES (must be done before running this script):
#   1. measures_v2.csv sitting in the repo root.
#   2. The processed EWMA dataset tensors regenerated, via:
#        python -m src.data.pmsm_loader_ewma --csv measures_v2.csv \
#            --split-mode diversity --add-ewma --add-physics \
#            --ewma-cols motor_speed i_d i_q u_d u_q coolant ambient \
#            --ewma-spans 60 300 1200 3000 6000 12000
#      (This is the exact command Simon pasted into docs/methods/PMSM-Baselines.md.
#       It must be re-run locally because the .pt tensor files are
#       git-ignored and are NOT present in the repo zip.)
#   3. results/baseline/PMSM_no_temps_ewma/, PMSM_no_temps_pm_ewma/,
#      PMSM_all_in_ewma/ each contain config.json + model.pt (already true
#      in the repo you gave me).
#
# All 10 documented methods are now wired into pruner.py:
#   obd, obs, mag, struct, activ, redundancy, lowrank, taylor_fo, snip, imp_rewind
#
# CAVEAT: snip and imp_rewind require the model's *pre-training* init.
# No baseline training script saves that init anywhere in this repo, so
# pruner.py uses a disclosed approximation: a freshly-seeded random init
# of the same architecture (seed=1234, see INIT_SEED in pruner.py) —
# NOT the baseline's true original init. State this explicitly in any
# report/writeup that includes snip or imp_rewind results.

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

METHODS=(obd obs mag struct activ redundancy lowrank taylor_fo snip imp_rewind)
DATASETS=(PMSM_no_temps_ewma PMSM_no_temps_pm_ewma PMSM_all_in_ewma)

for method in "${METHODS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    marker="results/pruning/${method}/${dataset}/.DONE"
    if [ -f "$marker" ]; then
      echo "[skip] ${method} / ${dataset} already completed (found ${marker})"
      continue
    fi

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  method=${method}   dataset=${dataset}"
    echo "════════════════════════════════════════════════════════════"
    python src/pruning/pruner.py --method "${method}" --dataset "${dataset}" --device cpu
  done
done

echo ""
echo "All done. Results saved under results/pruning/<method>/<dataset>/"
echo "Now run:  python scripts/build_pmsm_pruning_comparison.py"
