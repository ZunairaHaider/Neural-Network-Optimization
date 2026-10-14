# PowerShell equivalent of run_pruning_pmsm_ewma.sh
# Run every pruning method currently wired into pruner.py against Simon's
# three new EWMA PMSM baselines.
#
# PREREQUISITES (must already be done):
#   1. measures_v2.csv in the repo root.
#   2. EWMA dataset tensors regenerated (you already did this).
#   3. results/baseline/PMSM_*_ewma/ each contain config.json + model.pt.
#
# All 10 methods: obd, obs, mag, struct, activ, redundancy, lowrank,
# taylor_fo, snip, imp_rewind.
#
# CAVEAT: snip and imp_rewind use a freshly-seeded random init (seed=1234),
# not the baseline's true pre-training init -- state this in any report.
#
# Resumable: if interrupted, just run this script again -- it skips any
# method/dataset combination that already has a .DONE marker file.

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")   # repo root

$Methods  = @("obd", "obs", "mag", "struct", "activ", "redundancy", "lowrank", "taylor_fo", "snip", "imp_rewind")
$Datasets = @("PMSM_no_temps_ewma", "PMSM_no_temps_pm_ewma", "PMSM_all_in_ewma")

foreach ($method in $Methods) {
    foreach ($dataset in $Datasets) {
        $marker = "results\pruning\$method\$dataset\.DONE"
        if (Test-Path $marker) {
            Write-Host "[skip] $method / $dataset already completed (found $marker)"
            continue
        }

        Write-Host ""
        Write-Host "================================================================"
        Write-Host "  method=$method   dataset=$dataset"
        Write-Host "================================================================"
        python src/pruning/pruner.py --method $method --dataset $dataset --device cpu
    }
}

Write-Host ""
Write-Host "All done. Results saved under results/pruning/<method>/<dataset>/"
Write-Host "Now run:  python scripts/build_pmsm_pruning_comparison.py"
