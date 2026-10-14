# Poster Walkthrough
**"How Lean Can We Go? — Comparing Pruning & Constructive Topology Search Across Dimensionality & Noise"**

---

## 1. Motivation

Real-world deployment of neural networks on thermal sensor hardware demands compact, fast, and interpretable models. We ask a concrete question:

> *How much can we compress an MLP regressor as the problem gets harder — more dimensions, more noise?*

Two complementary strategies: **pruning** (remove weights from a trained dense network) and **constructive topology search via ADAGOS** (grow from a minimal seed, no dense baseline needed). Benchmarked on synthetic data with known ground truth so we control exactly what's hard.

---

## 2. Two Axes of Difficulty

- **Dimensionality:** $d = 1, 2, 3$ inputs — controls model size and baseline complexity
- **Noise:** Dataset A (clean, near-zero residual achievable) vs. Dataset B (12.5% Gaussian noise, irreducible MSE floor)

The question becomes: *does the compression regime change as you move along these axes?*

---

## 3. Methods

### Pruning (post-training)
Start from a trained baseline. Score weights or neurons by importance. Iteratively remove the least significant. Fine-tune after each sparsity target.

- **Weight-level:** OBD, OBS, Magnitude, Taylor FO — differ in how they estimate the cost of removing a single weight (none/first/second-order, diagonal/full Hessian)
- **Neuron-level:** Structured (outgoing weight norm), Activation (mean firing magnitude), Redundancy (correlated neuron pairs) — remove whole neurons for real hardware speedup
- **Before/at init:** SNIP (single-shot mask at init, then train), IMP+Rewind (iterative magnitude pruning + rewind to init — tests the Lottery Ticket Hypothesis)

### ADAGOS — Constructive Topology Search
Fundamentally different: **no dense baseline**. Starts from a minimal seed $[d, 1, 1]$ and grows/shrinks topology cycle by cycle. A Lagrangian estimate scores candidates cheaply; a validation gate accepts a change only if it reduces generalisation error. Weights trained via Gauss-Newton.

---

## 4. Results

> **Note:** these are preliminary results on our synthetic benchmark (quadratic function, $d=1,2,3$, controlled noise). Takeaways should be read as hypotheses to test, not settled conclusions.

### The dominant finding: noise determines pruning budget, not method choice

**Noisy data (Dataset B):** almost every method reaches high sparsity at roughly $1.0\times$ baseline MSE — noise sets an irreducible floor, and once you accept you can't beat it, nearly everything compresses equally well. This effect is strongest at $d=1$ (80–90% sparsity across methods); it weakens as dimensionality increases, with $d=3$ noisy reaching only 30–70% depending on the method.

**Clean data (Dataset A):** much more scattered. The baseline error is already near zero, so any compression is fragile. Here method choice matters — and simple often wins: plain Magnitude is best at $d=2,3$; second-order methods (OBS, Taylor FO) are competitive at $d=1$.

### No universal winner
Under noise the "leader" rotates with dimension: Taylor FO at $d=1$, OBS at $d=2$, Structured at $d=3$. No single method dominates across the board.

### "Failures" are budget misses, not divergence
Dashes in the table mean the method never stayed within $1.5\times$ baseline MSE at any sparsity level — not that the network collapsed. This happens almost exclusively on **clean, low-$d$ tasks** where baseline error is already $\approx 0$: even modest compression hurts.

### ADAGOS: striking on clean, grows on noisy
On clean data ADAGOS achieves dramatically better MSE than the baseline with 78–98% fewer parameters. On noisy data it grew the network at $d=1$ and $d=3$ (−122%, −207%), finding it needed more capacity to model the noise — informative, not a bug. $d=2$ noisy is a mild exception (12% reduction).

---

## 5. Key Takeaways (on current synthetic dataset)

| Regime | Recommendation | Reason |
|---|---|---|
| Clean / low-noise | Start with Magnitude | Cheap, often sub-baseline MSE |
| Noisy | Structured or second-order (Taylor FO / OBS) | Everything hits the noise floor; winner rotates by $d$ |
| No dense baseline | ADAGOS | Discovers topology from scratch, self-regularises via validation gate |

---

## 7. Likely Audience Questions

### On methods

**Q: Why does plain Magnitude pruning perform so well if it uses no gradient or Hessian information?**
Because the networks are small and well-trained — at a good optimum, small weights genuinely contribute little. On clean low-dimensional data the loss landscape is smooth enough that the zero-order proxy is accurate. On harder tasks (noisy, high-$d$) the advantage of second-order methods becomes visible.

**Q: What's the practical difference between Structured and Activation pruning? They both remove neurons.**
Structured scores neurons by the L2 norm of their outgoing weight vector — purely parameter-based. Activation scores by mean absolute activation across data — a forward pass is required. Activation catches dead neurons (e.g. dead ReLUs with large weights but near-zero output) that Structured misses. In practice both are structured and give real hardware speedup unlike weight-level methods.

**Q: OBS is theoretically stronger than OBD — why doesn't it consistently win?**
OBS accounts for weight interactions (off-diagonal Hessian) and applies a closed-form weight correction after each removal. But on small networks the Hessian is well-conditioned and off-diagonal terms are small — the diagonal approximation in OBD is already good enough. OBS's advantage emerges on larger, more redundant networks. The added cost ($O(P^2)$ vs $O(P)$) isn't justified here.

**Q: What is the Lagrangian estimate in ADAGOS and why not just use MSE?**
After each structural change, new weights are initialised to zero. If you score candidates by MSE after $k$ gradient steps, all ADD_NEURON candidates look identical — zero output, zero gradient. The Lagrangian augments MSE with sensitivity-weighted constraint terms (the Lagrange multipliers are backprop gradients frozen from the previous network). These terms give signal immediately — before new weights have moved — by measuring how much the structural change shifts activations in a loss-reducing direction.

**Q: What's the Lottery Ticket Hypothesis and did you find evidence for it?**
Frankle & Carlin (2019): within a dense network there exists a sparse subnetwork ("winning ticket") that, when trained from the same random init, matches the dense network. IMP+Rewind tests this — prune by magnitude, rewind surviving weights to init, retrain. On our networks IMP+Rewind performs comparably to plain Magnitude, which is consistent with the hypothesis but not strong evidence given the network size.

**Q: Why do SNIP and IMP+Rewind have so many dashes on Dataset A?**
Both methods impose sparsity aggressively (SNIP in one shot at init, IMP+Rewind iteratively with rewind). On clean data the baseline error is already near zero — even a small structural constraint prevents full convergence to that near-zero floor. The 1.5× budget threshold is simply never met. On noisy data the irreducible floor is much higher, so both methods perform fine.

---

### On results

**Q: You say there's no clear winner — so what would you actually recommend in practice?**
It depends on the regime. On clean/low-noise data, start with Magnitude — it's the cheapest and often achieves sub-baseline MSE. On noisy data, use structured or second-order (the winner rotates by dimension, but all perform similarly at the noise floor). If you have no dense baseline or don't want to commit to an architecture upfront, use ADAGOS. These are recommendations from our synthetic benchmark — the real-data results may shift them.

**Q: Why does ADAGOS grow the network on noisy data?**
ADAGOS only accepts structural changes that strictly reduce validation error. On noisy data the irreducible noise floor means the network needs more capacity to approximate the signal amid the noise — every additive step that reduces validation error gets accepted, and subtractive steps can't reduce it further. It's the validation gate working as intended: the algorithm found it needed a bigger network, not a smaller one.

**Q: The "best method" rotates by dimension on noisy data — is that noise or a real signal?**
Honest answer: possibly noise, given single-seed results on small networks. The differences between Taylor FO (1.01×), OBS (1.00×), and Structured (1.00×) at their respective winning dimensions are within rounding. The pattern needs replication on more seeds and larger networks before claiming it's structural.

**Q: What does "sparsity" mean here — weight-level or neuron-level?**
It's counted uniformly as the fraction of scalar weights zeroed, regardless of method. For weight-level methods (Magnitude, OBD, etc.) individual scalars are zeroed. For neuron-level methods (Structured, Activation) an entire neuron's connections are zeroed at once — one pruning step removes more of the budget.

---

### On design choices

**Q: Why synthetic data and not real thermal sensor data from the start?**
Synthetic data gives us ground truth (known function, tunable noise) and lets us isolate the two axes of difficulty cleanly. Real sensor data has confounds — sensor drift, missing values, correlated inputs — that would make it hard to attribute performance differences to compression method vs. data properties. Synthetic first, real data as the next step.

**Q: How did you choose the 1.5× MSE budget threshold?**
It's a practical engineering tolerance — a model that regresses 50% worse than the dense baseline is arguably unusable for the thermal sensor application. The 3× early-stopping threshold in the pruning loop is a computational budget guard, not a selection criterion.

**Q: Did you run multiple seeds? Are the results statistically significant?**
Single seed per configuration in the current results. This is a limitation we acknowledge — the rankings, especially on clean data where differences are small, should be treated as preliminary. Replication across seeds is part of formalising the results in the next phase.

---

### On next steps

**Q: What would a hybrid method look like?**
One natural candidate: use ADAGOS to discover the topology (no need to commit to an architecture upfront), then apply a pruning pass to further compress the converged network. Another: use SNIP-style init-time pruning to constrain the search space before ADAGOS grows. We don't know yet whether the gains are additive — that's the experiment.

**Q: How will you compare architectures across methods?**
By recording the final hidden layer widths and depth for each method at each sparsity level, then checking overlap. If Magnitude and OBS consistently converge to the same $[16, 8]$ topology, the method choice is just a cost question. If they diverge, the method is shaping what the network learns to represent.

---

## 6. Next Steps

These results are a benchmark on controlled synthetic data. To turn them into actionable guidance for real thermal sensor deployment, four things remain:

1. **Formalise and deepen the current results** — analyse computational cost per method (training time, FLOPs, memory), not just accuracy vs. sparsity.

2. **Architecture convergence** — do different methods converge to the same final topology, or does the pruning method determine the structure? If they converge, method choice is really just a cost tradeoff, not a structural one.

3. **Scale up to a real dataset** — test on actual thermal sensor data (higher dimensionality, real noise distribution). The synthetic rankings may shift; ADAGOS in particular is compute-intensive and untested at scale.

4. **Explore hybrid / combined methods** — once we understand where each method excels, a natural next question is whether combinations work better than any single approach (e.g. ADAGOS for topology discovery, followed by pruning for further compression).
