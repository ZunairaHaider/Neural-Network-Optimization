# Reproducibility

How to re-run this project, what to expect, and exactly which reported numbers can be traced to
artifacts in this repository. Everything here was checked against the repository's own code and
saved outputs, not assumed from the report's prose.

> Quick check (about 5 seconds, no data download): `python tests/smoke_test.py`

## Running the pipeline

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Generate synthetic datasets (clean/noisy quadratics, d = 1, 2, 3)
python src/data/generate_data.py

# 2. Baseline HPO + retrain (synthetic)
python src/training/train_baseline.py --dataset A
python src/training/train_baseline.py --dataset B

# 3. Run all 10 pruning methods (synthetic)
for m in obd obs mag snip taylor_fo activ struct redundancy lowrank imp_rewind; do
  python src/pruning/pruner.py --method $m --dataset A
  python src/pruning/pruner.py --method $m --dataset B
done

# 4. Constructive search (Adagos) on synthetic data
python -m patenting.main

# 5. PMSM real-motor pipeline
python data/train_pmsm_baseline.py
python -m patenting.motor.run_stator

# 6. Compare everything
python src/analysis/architecture_compare.py
```

All seeds are fixed (42) at entry, but **this does not make every run bit-for-bit
deterministic**: the report's own Finding 4 (§5.5) documents that the constructive search's
*discovered topology* varies across weight-initialisation seeds even when accuracy does not, and
the appendix (C — Reproducibility) notes that baselines are selected as the best of several
restarts rather than a single run. Treat a single local run as one sample from a distribution,
not as a guarantee of reproducing a specific table entry exactly.

**What is lightweight vs. expensive:** step 1 runs in seconds; steps 2–4 on synthetic data run
in minutes on CPU; the PMSM pipeline (steps 5–6) involves an Optuna HPO campaign of up to 100
trials per input configuration (documented at tens of minutes per configuration in the saved
configs under `results/baseline/`) and additionally requires the Kaggle *Electric Motor
Temperature* CSV, which is not bundled with this repository (it is git-ignored — see
`pmsm_ewma_pruning/README.md` for where to obtain it) and must be downloaded separately before
step 5 can run at all.

## What is reproduced directly, and what is reported only in the report

Everything below was checked directly against this repository's own code and saved artifacts —
not assumed from the report's prose — and distinguishes three cases: numbers independently
traceable to a saved run, numbers in the report that an *existing* saved run contradicts, and
numbers for which no corresponding run exists in this snapshot at all.

**Independently verified against saved artifacts, exactly or within normal run-to-run
variance:**
- All four PMSM baseline MLPs in §5/Table 5 (69,121 / 17,793 / 9,281 / 76,417 params) trace to
  specific `config.json` files under `results/baseline/` and
  `results/ablation_ewma/runs/A_raw_PMSM_all_in_seed3/`, matching the report's RMSE/MAE exactly.
- The synthetic-data baseline architectures in Table 2 (25/65/2273 params for Dataset A;
  13/17/31 for Dataset B) match `tanh_pipeline/results/baseline/*/config.json` exactly.
- The ten-method pruning suite's results on the EWMA-feature PMSM baselines (Table 7) match the
  saved sparsity sweeps under `results/pruning/` in ranking and magnitude, within a few percent.
- `tests/smoke_test.py` (an `MLP` forward/train pass, the pruning modules, and three cycles of
  the Adagos constructive search on a toy 1-D regression) passes end-to-end in a few seconds,
  confirming the core model, pruning, and constructive-search code execute correctly as written
  (not a reproduction of any report number: a correctness check on a reduced budget).

**Contradicted by an existing saved artifact** (an output exists, and disagrees with the
report, rather than being absent):
- The Adagos static result for the 11-input PMSM configuration: the report states `[11,10,1,1]`,
  133 params, 0.97°C MAE. Running `python -m patenting.motor.run_stator` with its own default
  arguments produces `patenting/motor/outputs/static_results.json`, and the project's own
  `notebooks/models_summary.ipynb` — which explicitly describes itself as loading
  "full-settings run artifacts" — reads that same file and reports `[11,12,12,1,1]`, 315 params,
  1.05°C MAE for the identical configuration.
- The dynamic thermal twin: the report's unpruned twin is `[9,4,1]`, 45 params, 2.43°C rollout
  RMSE. The saved `dynamic_results.json` (and the same notebook) report `[9,9,4,1]`, 135 params,
  2.79°C mean rollout RMSE for the `coolant+elec` configuration.
- The most likely explanation is run-to-run variability in the constructive search itself —
  exactly the sensitivity to initialisation the report documents as Finding 4 — rather than a
  script defect; `architecture_search` was executed successfully end-to-end in the smoke test
  above, so the code path itself is not broken. Re-running `run_stator.py` is not expected to
  reproduce the report's exact topology without knowing the specific run (seed, restart count)
  that produced it.

**No corresponding run exists in this snapshot** (missing, not contradicted):
- The construct-then-prune results (42 params/1.10°C for the 11-input network; 75 params/0.79°C
  for the 81-input network; 18 params/2.59°C for the OBD-pruned dynamic twin): there is no
  script anywhere in this repository that applies the pruning suite to an Adagos-constructed
  PMSM network. `src/pruning/`'s ten criteria operate on `src.models.mlp.MLP` (PyTorch) objects;
  `patenting/motor` produces a different, NumPy-based `DynamicTopologyNetwork`, and no bridge
  between the two is implemented here. (`patenting/pruning.py`'s own lightweight small-weight
  pruning runs automatically *inside* `architecture_search`, per Algorithm 1 line 17, and is
  already reflected in the 315p/135p numbers above — it is not a separate, invokable sweep.)
- Report Table 6 (pruning a 69k–76k-parameter raw-signal PMSM baseline down to ~7.6k
  parameters): `results/pruning/*/PMSM_all_in/` prunes a different, 417-parameter baseline
  instead.
- A handful of figures under `results/comparison/` and `results/scaling_law/` predate the
  tanh/LBFGS convergence fix documented in `tanh_pipeline/HANDOFF.md` and read differently from
  the report's Finding 1 (e.g. showing high sparsity achievable on clean synthetic data, which
  the report's final pipeline does not find) — for this reason the figures in §6 above were
  taken directly from the report rather than from these folders.

None of this bears on the report's written conclusions, which the architecture-level evidence
throughout the repository supports independently of the specific numbers above. The gap sits at
the level of a handful of saved checkpoints for the PMSM constructive-search experiments,
consistent with a multi-month project that went through at least one documented corrected
rerun already (`tanh_pipeline/HANDOFF.md`, for the synthetic baselines). Reproducing the exact
PMSM constructive-search numbers in the report would require locating or re-running the
specific `patenting/motor` invocation that produced them and committing its output.


## Notes on the included datasets

- `data/processed/` ships the exact synthetic datasets (A, B, D; d = 1, 2, 3) that the saved
  results were computed on (840 KB in total), together with `quadratic_params.json`. Use these
  when you want to compare against saved results.
- Running `python src/data/generate_data.py` regenerates them. In a test on a different machine
  and PyTorch version, the regenerated inputs `x` were identical and the targets `y` agreed with
  the shipped files to float32 rounding (largest absolute difference 3.4e-5 over the ten
  dataset files the script rewrites; the 1-D A and B files are not rewritten by the script), so regeneration reproduces the data to within numerical precision, but not
  byte-for-byte. Regenerating overwrites the shipped files.
- The raw PMSM CSV (`measures_v2.csv`) and the large processed PMSM tensors are not included
  (git-ignored); see `pmsm_ewma_pruning/README.md` for where to obtain the CSV.

## Status of earlier-iteration material

Some figures under `results/comparison/` and `results/scaling_law/`, and the write-ups in
`docs/scaling-law-study/`, come from an earlier iteration of the pipeline (before the tanh /
LBFGS convergence fix documented in `tanh_pipeline/HANDOFF.md`) and are kept for the project
history. They do not all match the final report's numbers. When they differ, the final report
(`docs/Paper/report.pdf`) and `tanh_pipeline/` are authoritative.
