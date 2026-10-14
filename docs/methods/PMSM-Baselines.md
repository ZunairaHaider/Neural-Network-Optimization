# PMSM Baseline MLPs
dataset with 81 input : CS_ANN\data\processed\dataset_PMSM_all_in_ewma_val.pt
CS_ANN\data\processed\dataset_PMSM_all_in_ewma_train.pt
CS_ANN\data\processed\dataset_PMSM_all_in_ewma_test.pt
```bash
python -m src.data.pmsm_loader_ewma --csv measures_v2.csv --split-mode diversity --add-ewma --add-physics --ewma-cols motor_speed torque i_d i_q u_d u_q ambient coolant  --ewma-spans 60 300 1200 3000 6000 12000 
```
## Objective

The goal is to train baseline MLP models for predicting the stator winding temperature of a
PMSM motor from different subsets of input signals. These baselines serve as the starting
point for applying the same 10 pruning methods used on Datasets A and B. The three input
configurations are designed to answer Schaeffler's question: which sensors are actually needed
to predict winding temperature?

## Dataset

We use the PMSM Electric Motor Temperature dataset (Kirchgässner et al., Paderborn
University) from Kaggle (`measures_v2.csv`). It contains 1,330,816 rows sampled at 2 Hz
across 69 drive sessions, totalling roughly 185 hours of motor operation. The target is
`stator_winding` — difficult to measure with a physical sensor in production motors.

The data is split at the profile level to prevent temporal leakage between adjacent 2 Hz
samples. Entire drive sessions go into one split:

| Split | Profiles | Rows |
|-------|----------|------|
| Train (70%) | 48 | 940,483 |
| Validation (10%) | 7 | 161,133 |
| Test (20%) | 14 | 229,200 |

All features are z-score normalised using training statistics only (y_mean = 66.15 °C,
y_std = 28.39 °C). Split seed: 42.

## Baseline configurations

| Config | Inputs | Count | Target | Purpose |
|--------|--------|------:|--------|---------|
| `PMSM_no_temps` | ambient, u_d, u_q, motor_speed, i_d, i_q, torque, coolant | 8 | stator_winding | Primary scenario — no temp sensors  |
| `PMSM_no_temps_pm` | above + pm | 9 | stator_winding | Tests whether permanent magnet temp helps |
| `PMSM_all_in` | above + pm, stator_tooth, stator_yoke | 11 | stator_winding | Upper bound — all available signals |

## Hyperparameter optimisation

Architectures were selected using Optuna (TPE sampler, seed 42). Search space:
`n_layers ∈ {1,2,3}`, `n_neurons ∈ {32,64,128,256}`, `activation ∈ {relu, tanh, gelu}`,
`lr ∈ [1e-4, 1e-2]` log-uniform, `weight_decay ∈ {0, 1e-5, 1e-4, 1e-3, 1e-2}`,
`batch_size ∈ {256, 512, 1024, 2048}`. Optimizer: Adam. Models exceeding 100,000 parameters
were pruned immediately.

Each trial trained on a 100,000-row subsample of the training split and evaluated on the full
validation split. Early stopping with patience 30. PMSM_no_temps and PMSM_no_temps_pm used
100 trials (max 500 epochs). PMSM_all_in used 50 trials (max 300 epochs) due to higher
per-trial cost from 11 inputs. All three used the same final retrain budget.

Selection rule: smallest architecture within 5% of best validation MSE. The selected model
was retrained from scratch on the full 940,483-row training set (800 epochs, patience 50),
then evaluated once on the locked test set.

## Final baseline architectures

| Config | Architecture | Params | FLOPs | Test MSE |
|--------|-------------|-------:|------:|---------:|
| `PMSM_no_temps` | 1×32, GeLU | 321 | 608 | 1.56 × 10⁻¹ |
| `PMSM_no_temps_pm` | 2×32, GeLU | 1,409 | 2,752 | 1.00 × 10⁻¹ |
| `PMSM_all_in` | 1×32, Tanh | 417 | 800 | 2.84 × 10⁻³ |

Denormalised test metrics:

| Config | RMSE (°C) | MAE (°C) | Max |e| (°C) | R² |
|--------|----------:|----------:|--------------:|------:|
| `PMSM_no_temps` | 11.20 | 8.25 | 208.78 | 0.8533 |
| `PMSM_no_temps_pm` | 9.00 | 6.30 | 73.91 | 0.9052 |
| `PMSM_all_in` | 1.51 | 1.01 | 11.18 | 0.9973 |

## Output files

Each baseline is saved to `results/baseline/<config_name>/`:

| File | Content |
|------|---------|
| `model.pt` | Trained weights (PyTorch state_dict) |
| `config.json` | Architecture, hyperparameters, metrics (pruner-compatible) |
| `metrics.json` | Test metrics summary for quick inspection |
| `hpo_results.csv` | All Optuna trials with hyperparameters and validation MSE |
| `training_curves.png` | Train/val loss over epochs during final retrain |

## How to reproduce

Place `measures_v2.csv` in the repo root. Run all commands from the **repo root** (`CS_ANN/`).

```bash
# 1. Generate PMSM train/val/test splits (~10 sec)
python -m src.data.pmsm_loader --csv measures_v2.csv

# 2. Baseline HPO + retrain  (~2-5 hrs each on CPU)
python -m src.training.train_pmsm_baseline --csv measures_v2.csv \
    --configs PMSM_no_temps --n-trials 100 --max-epochs 500 --patience 30 \
    --retrain-epochs 800 --retrain-patience 50

python -m src.training.train_pmsm_baseline --csv measures_v2.csv \
    --configs PMSM_no_temps_pm --n-trials 100 --max-epochs 500 --patience 30 \
    --retrain-epochs 800 --retrain-patience 50

python -m src.training.train_pmsm_baseline --csv measures_v2.csv \
    --configs PMSM_all_in --n-trials 50 --max-epochs 300 --patience 20 \
    --retrain-epochs 800 --retrain-patience 50
```
```bash
 python -m src.training.train_pmsm_baseline_ewma `
  --csv measures_v2.csv `                                                                                 
 --split-mode diversity  `                                                                                
--add-ewma `                                                                                               --add-physics`                                                                                  
--ewma-cols motor_speed i_d i_q u_d u_q coolant ambient `
--ewma-spans 60 300 1200 3000 6000 12000`
--n-trials 20 --max-epochs 300 --patience 50 --hpo-pool-size 40000 
```

Seeds: profile split uses NumPy seed 42, Optuna sampler uses seed 42. Final retrain budget
(800 epochs, patience 50) is the same across all three configs.

Quick smoke test (~2 min, just to verify the pipeline works):

```bash
python -m src.training.train_pmsm_baseline --csv measures_v2.csv \
    --configs PMSM_no_temps --n-trials 5 --max-epochs 20 --hpo-pool-size 5000
```

## Applying pruning methods

Before running pruning on PMSM baselines, two changes are needed in `src/pruning/pruner.py`:

**Change 1** — in `_load_baseline` (~line 103), add `input_dim` and `output_dim`:

```python
model = MLP(
    n_layers=architecture["n_layers"],
    n_neurons=architecture["n_neurons"],
    activation=architecture["activation"],
    input_dim=architecture.get("input_dim", 1),
    output_dim=architecture.get("output_dim", 1),
).to(device)
```

**Change 2** — in the argument parser (~line 422), add the PMSM config names:

```python
parser.add_argument("--dataset",
    choices=["A", "B", "D", "PMSM_no_temps", "PMSM_no_temps_pm", "PMSM_all_in"],
    required=True)
```

Both changes are backward-compatible — Dataset A/B/D works exactly as before.

After applying the changes:

```bash
# Pruning  (~1-5 min per method per dataset)
for method in obd obs mag snip taylor_fo activ struct redundancy lowrank imp_rewind; do
    python src/pruning/pruner.py --method $method --dataset PMSM_no_temps
    python src/pruning/pruner.py --method $method --dataset PMSM_no_temps_pm
    python src/pruning/pruner.py --method $method --dataset PMSM_all_in
done
```

Results are saved to `results/pruning/<method>/<dataset>/` in the same format as the
synthetic experiments.

## Notes

These baselines are the reference models for the PMSM pruning experiments. The same 10
pruning methods applied to Datasets A and B will be applied here to evaluate how each method
handles a real-world regression task with higher input dimensionality (8-11 features vs 1-3).

The PMSM code is fully contained in three new files (`src/data/pmsm_configs.py`,
`src/data/pmsm_loader.py`, `src/training/train_pmsm_baseline.py`) with minimal changes to
`generate_data.py` and `hpo_utils.py`. The synthetic Dataset A/B/D pipeline is unchanged.
