# Dataset visualisations

Figures showing the benchmark datasets used in the case study: quadratic
regression on `R^d` for `d ∈ {1, 2, 3}`, with Dataset A (clean) and Dataset B
(noisy, σ scaled to 12.5 % of peak-to-peak range).

## Files

- **`dataset_panel_v2.png`** — combined 2 × 3 grid (hero figure for the
  poster). Rows = noise axis (clean / noisy); columns = dimensionality
  axis (d = 1, 2, 3).
- **`dataset_1d_only.png`** — standalone 1-D, clean vs. noisy side-by-side.
- **`dataset_2d_only.png`** — standalone 2-D, with noise visualised as
  residuals `y - f(x)` so the noise is directly visible.
- **`dataset_3d_only.png`** — standalone 3-D, projected onto `(x₁, x₂)`
  with a slice of `f` at `x₃ = x₃*` as the background.

All four figures share the same colour scheme:
- Blue: Dataset A (clean)
- Red: Dataset B (noisy)
- Viridis colourmap: y-values (clean panels)
- Coolwarm diverging colourmap: residuals `y - f(x)` (noisy 2-D / 3-D panels)

## Regenerating

From the repo root:

```bash
python src/data/make_dataset_panel_v2.py
python src/data/make_standalone_panels.py
```

Both scripts read:
- `data/processed/dataset_{A,B}{,_2d,_3d}_{train,test}.pt`
- `data/processed/quadratic_params.json`

and write into this folder.

## Hardcoded assumptions

The scripts assume the current experimental design. If any of the following
changes, the relevant literals need to be edited in the script (each is
documented in the script's docstring):

- 1-D function form: `f(x) = (x - 1)^2`
- Input domain: `[-3, 3]^d`
- Noise levels in caption: `σ = 2.0, 5.3, 23.8` at `d = 1, 2, 3`
- Sample count: 5 000 per dataset (4 000 train + 1 000 test)
