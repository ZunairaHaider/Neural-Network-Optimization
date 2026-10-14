# Pruning scaling-law summary

Automatically generated from `results/pruning/**` checkpoints. Re-run via `python src/analysis/scaling_law.py`.

## Baseline architectures

| Dataset | dim | n_layers × n_neurons | activation | params | FLOPs | test MSE |
|---|---|---|---|---|---|---|
| A | 1 | 2×16 | relu | 321 | 609 | 1.9350e-04 |
| A | 2 | 3×32 | gelu | 2241 | 4385 | 1.3546e-03 |
| A | 3 | 3×48 | gelu | 4945 | 9745 | 4.1879e-02 |
| B | 1 | 1×16 | relu | 49 | 81 | 3.9886e+00 |
| B | 2 | 2×10 | relu | 151 | 281 | 2.8385e+01 |
| B | 3 | 2×4 | relu | 41 | 73 | 6.2464e+02 |

## Max sparsity at test MSE ≤ 1.5× baseline

Best surviving sparsity per (method, dataset, dim). Empty = the method never satisfied the MSE budget.

### Dataset A

| method | d=1 | d=2 | d=3 |
|---|---|---|---|
| obd | 70% | 30% | 90% |
| obs | 70% | — | 90% |
| mag | 60% | 70% | 70% |
| struct | 50% | 20% | 40% |
| activ | 30% | 60% | 80% |
| redundancy | 20% | 80% | 70% |
| lowrank | 11% | 10% | 10% |
| taylor_fo | 50% | 80% | 90% |
| snip | — | — | 70% |
| imp_rewind | — | 70% | 80% |

### Dataset B

| method | d=1 | d=2 | d=3 |
|---|---|---|---|
| obd | 90% | 80% | 70% |
| obs | 80% | 70% | 50% |
| mag | 90% | 80% | 60% |
| struct | 90% | 90% | 40% |
| activ | 90% | 80% | 70% |
| redundancy | 80% | 70% | 50% |
| lowrank | 0% | 8% | 12% |
| taylor_fo | 70% | 90% | 50% |
| snip | 90% | 80% | 60% |
| imp_rewind | 80% | 90% | 30% |

## Max sparsity at test MSE ≤ 2.0× baseline

Best surviving sparsity per (method, dataset, dim). Empty = the method never satisfied the MSE budget.

### Dataset A

| method | d=1 | d=2 | d=3 |
|---|---|---|---|
| obd | 70% | 40% | 90% |
| obs | 70% | — | 90% |
| mag | 60% | 70% | 70% |
| struct | 60% | 30% | 40% |
| activ | 60% | 70% | 80% |
| redundancy | 60% | 80% | 70% |
| lowrank | 11% | 10% | 10% |
| taylor_fo | 60% | 90% | 90% |
| snip | — | — | 70% |
| imp_rewind | — | 80% | 80% |

### Dataset B

| method | d=1 | d=2 | d=3 |
|---|---|---|---|
| obd | 90% | 80% | 80% |
| obs | 80% | 80% | 60% |
| mag | 90% | 90% | 60% |
| struct | 90% | 90% | 70% |
| activ | 90% | 90% | 70% |
| redundancy | 80% | 70% | 60% |
| lowrank | 0% | 8% | 12% |
| taylor_fo | 70% | 90% | 80% |
| snip | 90% | 80% | 70% |
| imp_rewind | 80% | 90% | 60% |

## Max sparsity at test MSE ≤ 3.0× baseline

Best surviving sparsity per (method, dataset, dim). Empty = the method never satisfied the MSE budget.

### Dataset A

| method | d=1 | d=2 | d=3 |
|---|---|---|---|
| obd | 70% | 40% | 90% |
| obs | 80% | — | 90% |
| mag | 60% | 70% | 70% |
| struct | 60% | 40% | 40% |
| activ | 60% | 80% | 90% |
| redundancy | 60% | 80% | 70% |
| lowrank | 11% | 10% | 10% |
| taylor_fo | 80% | 90% | 90% |
| snip | 30% | 40% | 70% |
| imp_rewind | 20% | 80% | 90% |

### Dataset B

| method | d=1 | d=2 | d=3 |
|---|---|---|---|
| obd | 90% | 80% | 90% |
| obs | 80% | 80% | 80% |
| mag | 90% | 90% | 70% |
| struct | 90% | 90% | 90% |
| activ | 90% | 90% | 90% |
| redundancy | 80% | 80% | 80% |
| lowrank | 0% | 8% | 12% |
| taylor_fo | 70% | 90% | 90% |
| snip | 90% | 80% | 90% |
| imp_rewind | 80% | 90% | 90% |
