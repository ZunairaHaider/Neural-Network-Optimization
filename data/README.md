# Data

`processed/` holds the synthetic regression datasets used in the study, shipped so that results
can be compared against the exact inputs:

- `dataset_{A,B}[_2d|_3d]_{train,test}.pt`: quadratic regression, clean (A) or noisy (B,
  12.5% peak-to-peak noise), input dimension 1, 2 or 3. `quadratic_params.json` holds the
  quadratic coefficients.
- `dataset_D_{train,test}.pt`: a further 1-D noisy quadratic with its own noise level
  (`SIGMA_D` in `src/data/generate_data.py`).
- `PMSM_*_norm.json`, `pmsm_profile_split*.json`: normalisation statistics and the profile-level
  train / validation / test split for the motor data.

The raw PMSM data (`measures_v2.csv`, public Kaggle *Electric Motor Temperature* dataset) and
the large processed PMSM tensors are **not** included. Download the CSV and see
[`../pmsm_ewma_pruning/README.md`](../pmsm_ewma_pruning/README.md).

`generate_data.py` (in `src/data/`) regenerates the synthetic sets; `pmsm_loader.py`,
`pmsm_configs.py` and `train_pmsm_baseline.py` here handle the PMSM pipeline.
