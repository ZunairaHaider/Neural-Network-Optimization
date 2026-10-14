# Scaling-law study (earlier exploration)

An earlier, pruning-only study of how the ranking of ten pruning methods changes with input
dimension (d = 1, 2, 3) and noise level on the synthetic quadratic problem. Its saved results
are in [`../../results/scaling_law/`](../../results/scaling_law/) and the code in
`src/analysis/scaling_law.py`.

This study was carried out before the optimizer / convergence fix documented in
[`../../tanh_pipeline/HANDOFF.md`](../../tanh_pipeline/HANDOFF.md), so its baseline architectures
(for example 321 / 2241 / 4945 parameters for Dataset A) differ from the corrected baselines
used in the final report (25 / 65 / 2273). It is kept as project history. For final numbers use
the [report](../Paper/report.pdf).

Files: `SUMMARY.md` (complete walkthrough), `report.md` (companion report), `methodology.md`,
`phase_0_audit.md`, `patent_track_recommendations.md`, `poster_draft.md` and `poster_layout.md`
(poster planning for this study), `readable.ipynb`.
