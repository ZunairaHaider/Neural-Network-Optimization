# Handoff

**Branch:** `main`
**Date:** 2026-06-18

---

# FAITHFUL PATENT METHOD (2026-06-18) — current state, start here

> Most recent work. Supersedes the MOTOR PHASE section below (pm target +
> manufactured compression + dynamic twin), which is now archived.

## Why this phase happened

A close re-read of the patent (US 12,124,779 B2) against the code found the
implementation was **not actually using the patent's signature mechanism**.
Selection Gauss-Newton mini-trained every candidate and ranked by MSE — the exact
cost the patent's cheap estimate exists to avoid — and discarded the frozen
multipliers. Two underlying bugs made the genuine estimate unusable:
1. the multiplier was the **pre-activation** backprop delta, but the patent's is
   the **post-activation** adjoint `Λ_i = Wᵢᵀ(φ'(z_{i+1})⊙Λ_{i+1})` (appendix p.22);
2. inconsistent error normalisation (mean vs sum).
The conformance docs claimed fidelity the code didn't have.

## What was done (all faithful, all validated)

**Static engine (Claims 1-8).** Candidates are now ranked by the real
frozen-multiplier estimate `ΔL = Σ_i Λ*_i·(X*_i − φ(W''_i X*_{i-1}))` (eq.13/14),
**no candidate retraining**. Only the winner is fully GN-retrained (Claim 7).
Incremental 1-node growth + parsimony brake + Claim-8 val gate ⇒ genuine
grow-then-shrink. Validated: estimate-vs-true-retrain **Spearman 0.89**.

**Dynamic engine (Claim 11 + 14-15).** Recurrent `x_{t+1}=x_t+h(x_t,u_t)`,
one-step teacher-forced (reduces to the static search) + free-running rollout.
Low-dim state kept raw (compression OFF by scope — see below). Self-test on a
linear thermal system finds the minimal `[2,1,1]` and rolls out near-perfectly.

**PMSM retarget (Schaeffler use case).** Target is now **stator_winding** (was
pm), with pm/temps as OPTIONAL inputs, plus an input-configuration sweep
(all signals vs coolant-only vs +pm vs no-coolant). Both static and dynamic.

**Compression/decompression (Claim 9) — intentionally NOT implemented.** Low-dim
tabular data has nothing to compress; the patent only warrants it for high-dim X.
The old motor code manufactured a 48-dim EWMA bank just to give Claim 9 something
to do — circular, now removed. Build it only if a genuinely high-dim input
appears.

## Where the new code is

| File | What |
|---|---|
| `patenting/lagrangian_estimate.py` | **NEW** — faithful eq.13/14 estimate (post-adjoint, all op types, `rank_candidates`) |
| `patenting/architecture_search.py` | **REWRITTEN** — ranks by faithful ΔL; clean Claim-8 gates; parsimony brake |
| `patenting/dynamic.py` | **NEW** — Claim 11 recurrent twin (`build_onestep_pairs`, `learn_dynamics`, `rollout`) |
| `patenting/motor/run_stator.py` | **NEW** — PMSM stator-winding: static config sweep `run()` + dynamic `run_dynamic()` |
| `patenting/validate_estimate.py` | **NEW** — proves the estimate ranks like a full retrain |
| `patenting/training.py` | **SLIMMED** — only `train_network` (the winner retrainer) remains |
| `patenting/main.py` | fixed cwd-relative dataset paths + patent-minimal `[d,1,output]` init |
| `docs/patent-implementation/Patent-Conformance-Audit.md` | **REWRITTEN** to match the code |
| `docs/patent-implementation/results-faithful-method.md` | **NEW** — all results + honest old-vs-new comparison |
| `docs/patent-implementation/Patent-Method-Changes.md` | **NEW (2026-06-19)** — concrete code-level change record (before/after, exact functions) |
| `notebooks/models_summary.ipynb` | **NEW (2026-06-19)** — static + dynamic model architectures, results, network diagrams |

## What was archived (dead/superseded)

- `archive/motor_pm_phase_superseded/` — old pm/compression/twin motor code
  (`run_motor.py`, `motor_loader.py`, `motor_compression.py`), its outputs, and
  the `motor_results`/`motor_summary` notebooks.
- `archive/patent_search_v1_superseded/architecture_search2.py` — alternate loop,
  imported by nothing.
- `archive/old_patent_outputs_superseded/` — stale pre-faithful output dirs
  (`outputs`, `outputs.zip`, `outputs_1`, `outputs_1_1`, `outputs_2`).

**Current faithful A/B artifacts (kept):** `patenting/outputs_1_1_40/<dataset>/`
(metrics.txt + plots, from the 2026-06-18 faithful run). `results/` (top-level:
baseline/pruning/comparison/scaling_law) is KEPT — it holds the HPO + pruning
results the convergence analysis needs.

## Results (full table in `docs/patent-implementation/results-faithful-method.md`)

A/B d=1,2,3 rerun is done. **Honest "is it better?":** mixed — new is much better
on A-2d/3d (~100×), tied on B (noise floor), but slightly **over-grows the trivial
A-1d** case (`[1,4,4,1]` 1.2e-6 vs the old `[1,2,1]` 3.7e-11). The old A-1d number
was prettier but came from the non-faithful selection + a hand-tuned brake. The
goal here was faithfulness (achieved), not the MSE leaderboard. PMSM numbers are
SMOKE TESTS only (one config, short runs): static MAE ~8.8 °C, dynamic rollout
RMSE ~2.5 °C.

## Progress 2026-06-19 (items 1 & 2 DONE)

**Item 1 — convergence analysis redone (faithful patent vs. CORRECTED tanh pruning).**
`notebooks/convergence_analysis.ipynb` now compares the faithful patent runs
(`patenting/outputs_1_1_40/`) against the **`tanh_pipeline/` pruning results** (the
LBFGS-converged tanh baselines — NOT the old overparametrized ReLU pruning). Pick
rule = most-pruned within 1.5× baseline MSE (the tanh pipeline's own criterion).
**Verdict: 1/6 converge** — **B (1-D)**, both reach minimal `[1,2,1]` (pruning via
snip @50%). Two findings vs. the old ReLU study: (a) the order-of-magnitude param
gap **mostly disappears** once baselines are properly converged — both routes now
land in single-digit-to-low-tens params; (b) the remaining difference is
**structural (depth vs width)**: pruning stays one wide hidden layer, constructive
builds depth + a skip. **A-3d is where constructive genuinely wins**: pruning finds
NO accuracy-preserving compression (every checkpoint blows MSE ≫1.5×), constructive
fits with deep `[3,4,4,1,1]`+skip at 43 params. On noisy B-2d/B-3d constructive
slightly over-grows vs pruning. Figure: `results/comparison/convergence_architectures.png`.
(NB: an earlier 2026-06-19 pass mistakenly used the old ReLU `results/pruning/`,
which gave a different "1/6"; corrected to the tanh pipeline per request.)

**Plus (2026-06-19) three more deliverables:**
- **Init-robustness VISUALISED** — ~~§5 of the convergence notebook draws the
  per-seed selected topology~~ **SUPERSEDED 2026-06-19**: the seed study was
  replaced by the starting-topology study (see the later progress note); §5 is now
  the 3-start study, `results/comparison/start_topology_robustness.png`.
- **Models summary notebook** — `notebooks/models_summary.ipynb`: static A/B
  constructive models (table + diagrams + grow-then-shrink paths) and the dynamic
  twin (run inline: block `[2,1,1]`, rollout RMSE 0.0017 on std 5.6), plus the PMSM
  smoke-test summary. Figures: `static_models_architectures.png`,
  `dynamic_twin_rollout.png`.
- **Patent-method change record** — `docs/patent-implementation/Patent-Method-Changes.md`: concrete
  before/after of the selection mechanism, the post-adjoint multiplier bug (with
  the exact old/new code), error-normalisation fix, search-loop discipline, files.

**Item 2 — initialisation-robustness study DONE** — **SUPERSEDED 2026-06-19** by the
starting-topology study (`patenting/start_topology_robustness.py`); the seed-based
files were deleted. The original seed-study findings are kept below for history.
Original (`patenting/seed_robustness.py`, `results/seed_robustness.json`),
5 seeds {0,1,7,21,42} on the 1-D datasets:
- **B (1-D): deterministic** — all 5 seeds → `[1,2,1]` (the noise floor pins the
  stopping point).
- **A (1-D): seed-dependent** — 3 distinct topologies (`[1,2,4,4,1]`,`[1,2,4,1]`,
  `[1,4,4,4,1]`), all ~1e-6 MSE, none recovering `[1,2,1]`. **Confirms the
  noiseless-A over-growth is an init/convergence artifact, not a data property**
  — so item 3 below (parsimony/convergence tuning), NOT an estimate change, is
  the fix. (NB: isolated per-seed runs differ slightly from sequential `main.py`
  because main.py shares one global RNG stream across all 6 datasets.)

## Progress 2026-06-19 (items 0 & 4 DONE — full PMSM run + rebuilt notebook)

**Items 0 & 4 complete.** `run_stator.py` now **saves artifacts** (was print-only):
`run()`/`run_dynamic()` write `patenting/motor/outputs/{static_results.json,
dynamic_results.json,dynamic_trajectories.npy}` + per-model architecture PNGs.
`python -m patenting.motor.run_stator` now runs **both** static + dynamic at full
settings. `notebooks/models_summary.ipynb` was **rebuilt from scratch around the
two real stator-winding models only** (A/B content removed — it lives in
`convergence_analysis.ipynb`); it loads the JSON artifacts live, so re-running the
sweep refreshes every number. Generator: `notebooks/_build_models_summary.py`.

**Full-settings results (held-out, split by `profile_id`):**

*Static `winding = f(inputs)` (Claims 1-8), MAE in °C:*

| config | inputs | topology | params | MAE °C |
|---|---|---|---|---|
| all_signals | 11 | `[11,12,12,1,1]`+skip | 315 | **1.05** |
| temps_only | 4 | `[4,4,4,1,1]` | 47 | 1.64 |
| coolant+elec+pm | 9 | `[9,4,1]` | 45 | 7.24 |
| coolant+elec | 8 | `[8,8,8,1,1]` | 155 | 7.55 |
| elec_only | 7 | `[7,7,3,7,1]`+skip | 116 | 9.73 |

*Dynamic twin `x_{t+Δ}=x_t+h(x_t,u_t)` (Claims 11+14-15), drive=`coolant+elec`,
stride 10 (5 s):* block `[9,9,4,1]` (135 p), free-running **rollout RMSE mean
2.79 °C / max 4.44 °C**, hot-spot |err| max 12.42 °C over 14 held-out profiles.

**Findings:** (a) the two adjacent stator temps (`stator_tooth`,`stator_yoke`) are
nearly the answer — keeping them gives ~1 °C; **dropping them (coolant+elec) costs
+6.5 °C**. (b) **pm barely helps** the static map (7.55→7.24 °C, −0.30). (c) The
**dynamic twin recovers most of that gap from temporal structure alone**: 2.79 °C
RMSE vs the static 7.55 °C MAE on the *same* realistic `coolant+elec` sensor set —
because winding temperature has thermal lag an instantaneous map can't capture.
Figures: `results/comparison/stator_static_architectures.png`,
`stator_dynamic_rollout.png`.

## Progress 2026-06-19 (starting-topology robustness REPLACES seed robustness + skip-viz fix)

**Init-robustness reframed as a STARTING-TOPOLOGY study** — this **replaces** the
old 5-seed `seed_robustness` study (per user). The seed study + its artifacts were
deleted (`patenting/seed_robustness.py`, `results/seed_robustness.json`,
`results/comparison/seed_robustness_topologies.png`). The convergence notebook now
has a single init-robustness section (**§5**, no separate §6).
New study: `patenting/start_topology_robustness.py` → `results/start_topology_robustness.json`.
It **fixes the seed and varies the starting topology** over three single-neuron-chain
starts that differ only in initial depth: `1`=`[1,1,1]`, `1-1`=`[1,1,1,1]`,
`1-1-1`=`[1,1,1,1,1]` — does extra initial depth change where the search lands?
- **B (1-D, noisy): no — fully robust.** All three starts → `[1,2,1]` at the noise
  floor (matches the pruning route, §4). Strongest evidence the method finds the
  right size.
- **A (1-D, noiseless): yes — deeper starts get TRAPPED.** Only `1`=`[1,1,1]` grows
  (`[1,4,4,4,1]`, ~1e-6); `1-1` and `1-1-1` **never grow** (stuck ~3.4e-2, ~10⁴×
  worse) — a chain of single-neuron layers is an information bottleneck so poor the
  ΔL estimate sees no beneficial 1-node edit. **Takeaway: start small AND
  wide-enough (`[d,1,output]`), not deep-and-thin.** A net with no hidden layer
  (`[1,1]`) is not a valid start (search grows from an existing hidden layer).
- Figure: `results/comparison/start_topology_robustness.png`.

**Skip-connection visualisation fixed** in the model summary: `draw_topology` now
routes skips as a bold labelled red arc on a "skip bus" above the nodes (was an
invisible thin arc clipped by the axes). `all_signals` (skip L0→L3) and `elec_only`
(skip L0→L2) now render clearly. Regenerated `notebooks/models_summary.ipynb`.

## Pick up here (recommended order)

0. ~~**★ USER PRIORITY — rebuild the model-summary notebook around the PMSM
   stator-winding models ONLY.**~~ **DONE 2026-06-19** (see progress note above).
   Original brief kept below for reference. The current
   `notebooks/models_summary.ipynb` was built around the synthetic A/B models +
   the dynamic *self-test* — that was a misunderstanding. The user wants a summary
   of the **two real use-case models** we agreed on for predicting **stator winding
   temperature**: one **static** (snapshot-in-time `winding = f(inputs)`) and one
   **dynamic** (temporal `x_{t+Δ}=x_t+h(x_t,u_t)`, because the process is temporal).
   The A/B content does NOT belong here (it's covered in `convergence_analysis.ipynb`).

   **First run BOTH models at full settings** (this is item 4 — do it first; the
   current PMSM numbers are smoke tests only): `run_stator.run()` (5-config static
   sweep) + `run_stator.run_dynamic()` at full settings (~15-20 min). Note
   `run_stator.py` currently only **prints** — it saves no metrics/diagrams, so
   either capture the runs inline in the notebook or add artifact-saving to
   `run()`/`run_dynamic()` first.

   The notebook must make these four things explicit **for each of the two models**:
   - **Topology / architecture** — final layer sizes + a network diagram (reuse the
     `draw_topology` helper or `visualization.draw_network_architecture`, which
     renders skips). Static smoke gave `[8,8,2,1]`; dynamic smoke gave `[9,9,1,1]`
     — replace with the full-run topologies.
   - **Performance** — static: held-out **test MSE + MAE in °C** per input config;
     dynamic: free-running **rollout RMSE mean/max in °C** + hot-spot |err| on
     held-out profiles. Include the static-vs-dynamic comparison and *why* dynamic
     wins (winding thermal lag an instantaneous map can't capture; smoke: dynamic
     ~2.55 °C vs static ~8.8 °C MAE).
   - **Input variables** — list them concretely. `ELEC = [ambient, u_d, u_q,
     motor_speed, i_d, i_q, torque]`. The 5 static configs (`run_stator.CONFIGS`):
     `all_signals` (11: ELEC+coolant,pm,stator_tooth,stator_yoke), `coolant+elec`
     (8: ELEC+coolant — drops all other temps), `coolant+elec+pm` (9), `elec_only`
     (7), `temps_only` (4: coolant,pm,stator_tooth,stator_yoke). The config sweep
     answers the email's questions: *can we drop the other temperature sensors but
     keep coolant?* and *does pm help?* The dynamic drive `u` defaults to the
     `coolant+elec` config; state `x` = winding temp.
   - **Output** — `stator_winding` temperature (°C), single output. Data: Kaggle
     `electric-motor-temperature/measures_v2.csv`, split by `profile_id`
     (leakage-safe, 0.7/0.1/0.2), normalised by train stats; dynamic step = stride
     10 = 5 s.

   Conformance: static = Claims 1-8; dynamic = Claims 11 + 14-15. (Compression
   Claim 9 stays OFF — low-dim drive, nothing to compress.)

3. **Parsimony tuning on noiseless A-1d** if we care about recovering `[1,2,1]`:
   raise per-topology GN convergence (n_restarts/epochs) or `eps_accept`, so the
   intermediate `[1,2,1]` reaches ~0 and the brake stops growth there. (Relates to
   item 2 — the over-growth may be a convergence/seed artifact.)

4. ~~**Run the full PMSM sweep** end-to-end.~~ **DONE 2026-06-19** (results in the
   progress note above; artifacts in `patenting/motor/outputs/`). Original note:
   `python -m patenting.motor.run_stator`
   (all 5 configs, static) + `run_dynamic` at full settings — produces the table
   answering "can we predict winding without the other temperature sensors but
   keep coolant?" and "does pm help?". (~15-20 min.) Save the table.

5. Investigate the dynamic twin's **26.8 °C hot-spot** transient (one bad profile).
6. Decide on **EWMA thermal-memory features** for the static configs (physically
   justified vs. re-introduced engineered redundancy).
7. Optional: faithful **Claim 9** (per-block topology-optimised autoencoder) only
   if a high-dim input is introduced.

## Runnable

```bash
python -m patenting.validate_estimate     # estimate vs ground truth (Spearman)
python -m patenting.start_topology_robustness  # starting-topology robustness on A/B 1-D (1, 1-1, 1-1-1)
python -m patenting.dynamic                # dynamic engine self-test
python -m patenting.motor.run_stator       # PMSM stator-winding config sweep
python -m patenting.main                    # A/B static, d=1,2,3
```

---

# MOTOR PHASE (2026-06-12) — constructive twin of PMSM rotor temperature  [ARCHIVED]

> SUPERSEDED by the faithful phase above. Code moved to
> `archive/motor_pm_phase_superseded/`. Kept for history.

> Most recent work. Supersedes the "NEXT PHASE — TFRD" section further down
> (that plan was dropped in favour of a real dynamical dataset).

## What was done
Applied the patent method end-to-end to the Kaggle **Electric Motor Temperature**
dataset (Paderborn PMSM). Target: `pm` (rotor magnet temp), modelled as a thermal
**state** `pm_{t+1}=pm_t+f(pm_t, z_t)`. Three constructed blocks:

- **Compression (Claim 9):** constructive autoencoder `48 -> 14` latent on an EWMA
  feature bank (8 raw inputs x 5 time spans), recon MSE 0.006.
- **Regression (Claims 1-8):** constructive search builds `f(pm_t, z_t) -> dpm`,
  teacher-forced on one-step pairs. Grew `[15,1,1] -> [15,9,1]` then pruned to
  **`[15,6,1]`, 103 params** (genuine grow-then-shrink).
- **Dynamic twin (Claim 11 / 14-15):** free-running rollout over 14 held-out
  sessions from `pm_0` + inputs.

**Key fix — time decimation.** At the native 0.5 s step the per-step delta is too
small to learn (one-step MSE minimised by predicting "no change"); rollout drifted
(mean 9.7 C). Predicting `dpm` over a 5 s step (`stride=10`) fixed it:
**rollout RMSE mean 3.64 C / max 6.97 C** (~19% of pm's 19 C std), hot-spot
mean 7.68 / max 19.5 C.

## Patent conformance (verified against granted claims, pp. 23-26)
- **Claims 1-8** — faithful (same engine as the static audit).
- **Claims 14-15** (simulation on new data) — faithful (the rollout).
- **Claim 11** (dynamic modeling block) — **variant, and it's the correct one**:
  state `pm` is 1-D so we keep it raw and compress only the exogenous drive (the
  patent's `C_G`). Forcing state compression would be meaningless here.
- **Claim 9** (compression) — **PARTIAL / the one real gap.** The claim requires
  topology optimization *respectively on the compression block and on the
  decompression block* (spec Steps 103/200); we shrink the autoencoder as a whole
  and skip that. NOTE: the old code comment ("encoder has no well-posed isolated
  target") was a wrong justification — the gap IS closeable.
- **Claims 12-13/16** — product/medium/data-compression, trivially or in spirit.

> Earlier drafts mislabelled the dynamic twin as "Claims 12-16". The recurrence is
> **Claim 11**; held-out simulation is **Claims 14-15**. Source docstrings in
> `patenting/motor/` were corrected.

## Where to pick up (recommended order)
1. **Close the Claim 9 gap.** Subdivide `[48,14,48] -> [48,14,14,14,48]`
   (function-preserving), run `architecture_search` restricted to encoder layers,
   then to decoder layers (global reconstruction loss; filter candidates by layer
   position — the engine already tags them, e.g. `ADD_LAYER_p0...`), then whole-net
   fine-tune. Add as an opt-in path in `motor_compression.py` to compare head-to-
   head against the current symmetric shrink.
2. **Sweep compression tol / k0 / retrain epochs** so the bottleneck actually
   shrinks below k0 (currently stuck at 14 = intrinsic dim under the tolerance).
3. **Optional escalation:** true multi-step / BPTT training loss if lower rollout
   error is wanted (teacher-forced one-step is what we do now).
4. Extend `docs/patent-implementation/Patent-Conformance-Audit.md` (currently Claims 1-8 only) to cover
   9/11/14-15 with the motor verdicts above.

## Motor files & artifacts
- Code: `patenting/motor/{run_motor,motor_loader,motor_compression}.py`
- Run: `python -m patenting.motor.run_motor` (~7 min; downloads dataset via kagglehub)
- Outputs: `patenting/motor/outputs/` (`twin.pkl`, `metrics.txt`, `rollout_trajectories.png`)
- **Summary notebook: `notebooks/motor_summary.ipynb`** — what happened, results,
  conformance scorecard, where to improve. Re-run with
  `jupyter nbconvert --to notebook --execute --inplace notebooks/motor_summary.ipynb`.
- Multi-output Gauss-Newton Jacobian (`patenting/optimizer.py`) was generalised
  from scalar output to support the 48-dim autoencoder; verified vs finite-diff.

---

## What's done

All three workstreams are complete across all six datasets (A and B at d = 1, 2, 3).

### Baseline (HPO)
| Dataset | Architecture | Activation | Params | Test MSE |
|---|---|---|---|---|
| A | [1→16→16→1] | ReLU | 321 | 1.93e-4 |
| B | [1→16→1] | ReLU | 49 | 3.99 ≈ σ² |

ReLU won HPO because `patience=20` is too short for tanh to escape saturation
plateaus with Adam. The patent method (Gauss-Newton) has no such issue — a 7-param
tanh network fits dataset A near-exactly.

### Pruning (10 methods, both families)
Results in `results/pruning/{method}/{A,B}/`. Best checkpoints summarised in
`results/comparison/architecture_comparison.csv`.

### Patent method (US12124779B2 ADAGOS) — faithful, reproducible
Run via `python -m patenting.main` (global seed 42, all six datasets sequential).

| Dataset | Final topology | Params | Test MSE (normalised) | Cycles |
|---|---|---|---|---|
| A (1-D) | [1, 2, 1]    | 7  | 3.7e-11 | 1 |
| A (2-D) | [2, 2, 2, 1] | 15 | 0.039   | 2 |
| A (3-D) | [3, 2, 1]    | 11 | 0.018   | 1 |
| B (1-D) | [1, 1, 4, 1] | 15 | 0.156   | 1 |
| B (2-D) | [2, 3, 1]    | 13 | 0.300   | 2 |
| B (3-D) | [3, 2, 1]    | 11 | 0.46    | 1 |

Scoring is the faithful **cross-Lagrangian ΔL** (eq. 13/14), with an **ε = 0.05
parsimony brake** and **monotonic Levenberg–Marquardt** retraining. Claim-by-claim
conformance: `docs/patent-implementation/Patent-Conformance-Audit.md`.

### Convergence verdict
> **Updated 2026-06-19 (faithful method): 1/6, not 0/6.** With the faithful
> search the two routes now agree on **B (1-D)** — both land on the minimal
> `[1,2,1]`. See the "Progress 2026-06-19" section above and §4–§5 of
> `notebooks/convergence_analysis.ipynb`. The 0/6 below was the OLD non-faithful
> result, kept for history.

The two routes **do not converge** on a common architecture (0/6 datasets).
Pruning is constrained to the HPO baseline skeleton (it only zeros weights);
the constructive search builds from a minimal net and finds far smaller
topologies, adding depth when the data needs it (A 2-D and B 1-D each grew a
hidden layer). They agree on *performance* (both hit the noise floor on B) but
not *shape*. See `notebooks/convergence_analysis.ipynb`.

> **Note.** The earlier patent-method runs reported different topologies
> (`[1,2,1]` / `[1,5,1]` / `[1,6,1]` on dataset A). The reasons and the full
> bug-fix history are in `docs/patent-bugfixes.local.md` (git-ignored, local
> only). Short version: with no parsimony brake on noiseless data, growth was
> governed by estimation noise and shared global RNG state, so the stopping point
> drifted between runs and code versions. The ε-brake + deterministic ΔL +
> monotonic LM + single global seed remove all six causes.

---

## Repo state

```
CS_ANN/
├── patenting/                       ← patent method (the live implementation)
│   ├── main.py · network.py · architecture_search.py
│   ├── topology.py · training.py · lagrangian.py · optimizer.py
│   └── outputs/                     ← per-dataset metrics.json + plots
├── archive/
│   ├── patenting_v1/                ← original broken attempt
│   └── patent_method_revised/       ← Adam-based intermediate (superseded)
├── notebooks/
│   └── convergence_analysis.ipynb   ← visual architecture comparison (6 datasets)
├── results/comparison/              ← all plots + CSVs
├── docs/
│   ├── Patent-Conformance-Audit.md  ← claim-by-claim conformance
│   ├── handoff.md                   ← this file
│   └── patent-bugfixes.local.md     ← LOCAL ONLY (git-ignored) bug-fix history
├── src/                             ← baseline + pruning workstream
└── README.md
```

---

## How to reproduce

```bash
source .venv/bin/activate

python -m patenting.main                         # patent method, all 6 datasets (~1.5 min)
python src/analysis/architecture_compare.py      # comparison outputs
jupyter nbconvert --to notebook --execute notebooks/convergence_analysis.ipynb
```

Pruning results are already saved. To re-run one method:
`python src/pruning/pruner.py --method obs --dataset A`.

---

## NEXT PHASE — constructive thermal-field twin on TFRD

> **This section is a prompt to run in a fresh session. Read it back to me and
> ask me to confirm the open questions below BEFORE writing any code.**

### Overview (recap)

The goal is a **digital twin of an engine component that predicts the
temperatures within it**. CMAPSS (NASA's turbofan RUL benchmark) is **not** a fit:
it only logs ~4 gas-path *station* temperatures along the engine axis — point
scalars over flight cycles, not a spatial field. There is no field to reconstruct.

**TFRD** (Temperature Field Reconstruction Dataset, arXiv:2108.08298) is the right
proxy: it provides 2-D temperature fields `T(x, y)` over a component domain with
heat sources at varying layouts, plus sparse sensor samples. This maps cleanly
onto the constructive topology search as a **coordinate-MLP / neural field**:
input `(x, y[, conditioning])` → output `T`. The MLP *is* the field, and the
constructive search finds the **smallest** network that represents it to
tolerance — which is exactly what ADAGOS rewards.

### Why this also pulls in patent Claims 9–16

The current study is static regression (Claims 1–8). TFRD genuinely extends scope:

- **Data-compression embodiment (Claims 9–16) becomes directly relevant.** A
  coordinate-MLP that reproduces a full temperature grid (thousands of points)
  with a handful of weights is a *compressed encoding* of the field — precisely
  the compression/decompression block of the later claims. The parsimonious
  network is the compressed representation; evaluating it at a coordinate is the
  decompression. We should implement and measure this explicitly (field grid
  size vs. network params = compression ratio).
- **Dynamic-system embodiment (Claims 9–16) becomes relevant *if* we add time.**
  A true engine twin evolves: `T(x, y, t)` or a roll-forward over operating
  states. If we model temporal evolution, the recurrent/dynamic embodiment
  applies. For a single-snapshot reconstruction it does not — so whether we go
  static-field or dynamic-field is a scope decision (see Q1).

### Proposed pipeline

1. **Data** — pull one TFRD sub-set; start with a single fixed heat-source
   layout. Each sample = a grid of `(x, y, T)`. Normalise coords to `[-1, 1]`,
   z-score `T` from train statistics only (mirror the leakage-safe split we
   already use).
2. **Task** — coordinate regression `f(x, y) = T`. Split by held-out grid points
   (interpolation test) and optionally by held-out layouts (generalisation test).
3. **Constructive search** — run `architecture_search` with the faithful
   Lagrangian scoring; start from `[2, 1, 1]` and let it grow. Report selected
   topology + params vs. a fixed-width MLP baseline at equal MSE, and the
   **compression ratio** (grid points ÷ params) for the Claims 9–16 angle.
4. **Metrics** — field MSE, **max-abs / hot-spot error** (safety-relevant), and
   params. Reconstruction heatmap + error map per layout.
5. **Scale-up** — if single-layout works, condition on layout: input becomes
   `(x, y, source-params…)` for a multi-config twin.

### Open questions — please confirm before I build

1. **One layout or a family?** Single fixed geometry (cleanest first twin) vs.
   condition the MLP on source parameters for a multi-layout twin?
2. **Static field or dynamic (Claims 9–16 dynamic embodiment)?** Reconstruct a
   snapshot `T(x,y)`, or model temporal evolution `T(x,y,t)` / roll-forward?
3. **Which TFRD subset?** It ships several (varying source count / boundary
   conditions). Preference, or pick the simplest single-source case?
4. **Inputs = pure coordinates, or coordinates + sparse-sensor conditioning?**
   A true neural field `(x,y)→T`, or `(x,y, sensor-readings)→T` (the
   reconstruction-from-sensors task TFRD was designed for)?
5. **Success metric priority** — overall field MSE, or hot-spot / max-temperature
   accuracy (changes the loss; the latter is what matters for a real twin)?
6. **Compression embodiment — implement it explicitly?** Do you want the Claims
   9–16 compression framing measured and written up (grid → params ratio,
   decompression error), or is that a stretch goal?
7. **Stay patent-pure or allow baselines?** tanh + Gauss-Newton only, or also a
   ReLU+Adam fixed-MLP and the TFRD paper's CNN for "N× fewer params" comparison?
8. **Jacobian compute budget.** TFRD grids are far larger than A/B data, so the
   finite-difference GN Jacobian gets expensive. OK to cap grid resolution for
   the search, or implement the analytic / zero-memory GN (Fehrenbach 2006)
   first?

Answer Q1–Q5 and I can start the single-layout coordinate-MLP and produce a first
constructive-search result on a real thermal field; Q6–Q8 set how far we push the
Claims 9–16 compression story and the compute envelope.
