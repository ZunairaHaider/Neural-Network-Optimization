# Documentation index

The final written report is the authoritative description of the project. Everything else in
this folder is supporting material, grouped by purpose.

| Folder / file | What it contains | Read it if you want to... |
|---|---|---|
| [`Paper/report.pdf`](Paper/report.pdf) | Final report (41 pages): methods, derivations, all results | understand the whole project |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | How to re-run the code and which reported numbers trace to saved artifacts | re-run or verify results |
| [`methods/`](methods/) | Plain-language summaries: [pruning methods](methods/Pruning-Methods-Summary.md), [Adagos patent method](methods/Patent-Method-Summary.md), [OBS and GeLU](methods/OBS_GeLU_Report.md), [PMSM baseline models](methods/PMSM-Baselines.md) | learn the methods without the full report |
| [`patent-implementation/`](patent-implementation/) | Claim-by-claim conformance audit of the Adagos implementation against US 12,124,779 B2, the change record, and the results of the faithful re-implementation | check how faithful the implementation is |
| [`literature/`](literature/) | Review of related patents (`patent-summaries/`: one summary per patent; `improvement-analysis/`: comparison and improvement notes) | see the related-work background |
| [`scaling-law-study/`](scaling-law-study/) | An earlier, pruning-only exploration (branch `scaling-law-study`): report, summary, poster draft. Produced before the tanh / LBFGS convergence fix, so its numbers differ from the final report | follow the project history |
| [`project-notes/`](project-notes/) | Internal working notes: hand-off document and an early poster walkthrough | see how the work was organised |

If a number in a supporting document disagrees with the report, the report and
[`../tanh_pipeline/`](../tanh_pipeline/) are authoritative.
