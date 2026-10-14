# pmsm_ewma_pruning/ — PMSM EWMA pruning pipeline

This folder contains everything needed to re-run pruning on Simon Giuliani's new
EWMA PMSM baselines (`PMSM_no_temps_ewma`, `PMSM_no_temps_pm_ewma`,
`PMSM_all_in_ewma`) and rebuild the comparison report/CSVs from scratch.

## Files in this folder

| File | What it does |
|---|---|
| `run_pruning_pmsm_ewma.sh` | Bash version: loops over all 8 pruning methods x all 3 EWMA datasets and calls `pruner.py` for each combination. Resumable — skips any combination that already finished. |
| `run_pruning_pmsm_ewma.ps1` | Same as above, PowerShell version (use this on Windows if you don't have Git Bash). |
| `build_pmsm_ewma_pruning_comparison.py` | Reads everything under `results/pruning/` and builds the comparison CSVs (MSE, FLOPs, params per method/sparsity/dataset). Run this *after* the pruning script finishes. |

## Why this folder exists

Running all 8 methods x 3 datasets x 9 sparsity levels by hand is ~24
separate commands with a lot of room for typos or forgotten combinations.
These scripts make the whole sweep reproducible with two commands instead
of doing it manually, and let you resume cleanly if your machine needs to
shut down partway through (it took multiple sessions to run this the first
time — see the resumability note below).

## Prerequisites (must be done first)

1. **`measures_v2.csv`** must be sitting in the repo root. It is git-ignored
   (too large / not meant to be versioned), so it will not already be there
   after a fresh clone — get it from Simon Giuliani or wherever the team stores the
   raw PMSM dataset.

2. **Regenerate the EWMA dataset tensors.** These are also git-ignored
   (`data/processed/dataset_PMSM_*_ewma_*.pt`). Run Simon Giuliani's exact command
   from `docs/methods/PMSM-Baselines.md`:

   ```bash
   python -m src.data.pmsm_loader_ewma --csv measures_v2.csv \
       --split-mode diversity --add-ewma --add-physics \
       --ewma-cols motor_speed i_d i_q u_d u_q coolant ambient \
       --ewma-spans 60 300 1200 3000 6000 12000
   ```

   This takes a few minutes and creates the `.pt` train/val/test files for
   all three EWMA configs.

3. Confirm `results/baseline/PMSM_*_ewma/` already contains `config.json`
   and `model.pt` for all three configs (these ARE committed to the repo,
   so this should already be true after cloning).

## How to run the full pipeline

**On Mac/Linux/Git Bash:**
```bash
bash pmsm_ewma_pruning/run_pruning_pmsm_ewma.sh
```

**On Windows PowerShell** (if you don't have Git Bash, or `Set-ExecutionPolicy`
complains about running scripts, run this once first: `Set-ExecutionPolicy
-Scope Process -ExecutionPolicy Bypass`):
```powershell
.\pmsm_ewma_pruning\run_pruning_pmsm_ewma.ps1
```

Both scripts print progress as they go (per-epoch fine-tuning lines, per-method
headers). Expect the full sweep to take several hours on a normal laptop CPU.

**If you need to stop partway through** (close your laptop, etc.): press
`Ctrl+C`, then whenever you're ready, just run the exact same command again.
It checks for a `.DONE` marker file in each `results/pruning/<method>/<dataset>/`
folder and skips anything already finished — only the one combination that
was actually interrupted gets redone.

**Then build the comparison CSVs:**
```bash
python pmsm_ewma_pruning/build_pmsm_ewma_pruning_comparison.py
```
This writes:
- `results/pruning/pmsm_ewma_comparison_full.csv` — every method x every sparsity level
- `results/pruning/pmsm_ewma_comparison_best.csv` — best sparsity per method (highest compression within 1.5x baseline MSE)

## Known caveats / things to know before reusing these results

- **`snip` and `imp_rewind` are NOT included.** Both require the model's
  state *before* training (SNIP prunes at init; IMP+Rewind rewinds
  surviving weights to their initial values before retraining). No
  training script in this repo saves that pre-training init — only the
  final trained `model.pt` is saved — so there's no valid init to use.
  The team decided not to run these with a fake/freshly-seeded init.

- **Single run per (method, dataset, sparsity), seed=42.** No repeated
  trials were done, so there's no variance estimate on any of these
  numbers. If you want to run repeated trials (e.g. for a statistical
  comparison), `pruner.py` takes a `--seed` argument — vary it across runs
  (results save to a `seed_<N>` subfolder automatically so trials don't
  overwrite each other). Just don't reuse the same seed across "repeated"
  trials, or you'll get identical results every time.

- **Fine-tuning uses a 50,000-row random subsample** of the training set
  (not the full ~150k-970k rows) for speed. Test evaluation always uses
  the full, untouched test set, so reported MSE is not affected by this —
  only training speed is.

- **OBD's Hessian-diagonal fallback** (used automatically when a model has
  >5,000 parameters) subsamples 4,000 calibration rows rather than looping
  over the entire training set — otherwise this single step alone can take
  tens of minutes per sparsity level on the larger PMSM datasets.
