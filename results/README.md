# Results

Saved outputs of the experiments: metrics, configurations, model checkpoints and figures.
The final numbers and figures of the project are in the [report](../docs/Paper/report.pdf);
what can be traced to the files below is listed in
[`../docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md).

| Folder | Contents |
|---|---|
| `baseline/` | Tuned baseline MLP per dataset / input configuration: `config.json`, `hpo_results.csv`, `model.pt`, training curves. Synthetic sets `A`, `B`, `D` (with `_2d`, `_3d`) and PMSM configurations (`PMSM_all_in`, `PMSM_no_temps`, `PMSM_no_temps_pm`, each also with `_ewma`) |
| `pruning/<method>/<dataset>/` | Sparsity sweep per pruning method: a checkpoint and a small config (test MSE, parameter count, inference time) at each sparsity level. `pmsm_ewma_comparison_*.csv` compare methods on the EWMA-feature PMSM baselines |
| `ablation_ewma/` | EWMA feature ablation on the PMSM baseline (multiple seeds): per-run results, summary tables, box plots |
| `stats/` | Seed-stability plots, metric box plots, pairwise tests across methods |
| `comparison/` | Cross-method and constructive-vs-pruning comparison tables and plots |
| `dataset_viz/` | Figures of the synthetic datasets ([README](dataset_viz/README.md)) |
| `scaling_law/` | Results of the earlier scaling-law exploration (see [`../docs/scaling-law-study/`](../docs/scaling-law-study/README.md)); predates the final baselines |
| `demo_pruning_outputs/` | A small before / after pruning example (images and `metrics.yaml`) |
| `merge_results.py` | Helper that merges per-method result files |

Per-sparsity network diagrams were kept only where the documentation references them. The
sparsity sweeps themselves (checkpoints and configs) are complete.
