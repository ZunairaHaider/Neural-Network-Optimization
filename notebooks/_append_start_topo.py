# Appends §6 (starting-topology robustness) to convergence_analysis.ipynb.
import os
import nbformat as nbf

P = os.path.join(os.path.dirname(__file__), 'convergence_analysis.ipynb')
nb = nbf.read(P, as_version=4)

# don't double-append
if any('Starting-topology robustness' in ''.join(c.source) for c in nb.cells):
    print('section already present; nothing appended'); raise SystemExit

md1 = r"""## 6. Starting-topology robustness (1-D datasets) — start small, vary the *start*

§5 fixed the *small* starting net and varied the random **seed**. This
complementary study does the opposite — it **fixes the seed and varies the
starting topology** over four small nets: the patent-canonical `[1, 1, 1]`
(`[d, 1, output]`), a same-depth width variant `[1, 2, 1]`, and two deeper
single-neuron nets `[1, 1, 1, 1]` and `[1, 1, 1, 1, 1]`. A method that genuinely
*starts small and grows to what the data needs* should reach the same final
architecture from any small start. Run with
`python -m patenting.start_topology_robustness`.

> A net with **no hidden layer** (e.g. `[1, 1]`) is not a valid starting point: the
> constructive search grows *from* an existing hidden layer, so it needs at least
> one — which is exactly why the patent's minimal start is `[d, 1, output]`."""

code1 = r"""st_path = os.path.join(ROOT, 'results', 'start_topology_robustness.json')
with open(st_path) as fh:
    start_rob = json.load(fh)

for name, d in start_rob.items():
    print(f"{name}  (seed {d['seed']} fixed)")
    for r in d['runs']:
        print(f"    start {str(r['start']):16s} -> final "
              f"{str(r['final_topology']):18s}  test_mse={r['test_mse']:.3e}")
    uq = d['unique_final_topologies']
    print(f"    => {len(uq)} distinct final topology(ies): {uq}  "
          f"[{'ALL AGREE' if d['all_agree'] else 'START-DEPENDENT'}]\n")"""

code2 = r"""# Visualise the final topology reached from each small start.
# Green = the modal final shape for that dataset; orange = a variant.
from collections import Counter

n_st  = max(len(d['runs']) for d in start_rob.values())
order = ['dataset_A', 'dataset_B']
names = [n for n in order if n in start_rob] + \
        [n for n in start_rob if n not in order]

fig, axes = plt.subplots(len(names), n_st,
                         figsize=(2.8 * n_st, 2.8 * len(names)), squeeze=False)
for r, name in enumerate(names):
    runs  = start_rob[name]['runs']
    modal = Counter(tuple(x['final_topology']) for x in runs).most_common(1)[0][0]
    for c, run in enumerate(runs):
        fin   = run['final_topology']
        color = '#55A868' if tuple(fin) == modal else '#DD8452'
        draw_topology(axes[r][c], fin,
                      f"start {run['start']} → {fin}\n({run['test_mse']:.1e})",
                      color)
    for c in range(len(runs), n_st):
        axes[r][c].axis('off')
    verdict = 'ALL AGREE' if start_rob[name]['all_agree'] else 'START-DEPENDENT'
    axes[r][0].set_ylabel(f"{name}\n{verdict}", fontsize=10, rotation=90, labelpad=12)
fig.suptitle('Starting-topology robustness: final topology per small start '
             '(green = modal shape, orange = variant)', y=1.01, fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, 'results', 'comparison',
            'start_topology_robustness.png'), dpi=130, bbox_inches='tight')
plt.show()"""

md2 = r"""### Interpretation — starting-topology robustness

* **B (1-D), noisy: fully robust.** All four small starts converge to the *same*
  minimal `[1, 2, 1]` at the noise-floor MSE (~0.171). The data's stopping signal —
  not the starting topology — fixes the architecture. This is the strongest single
  piece of evidence that the method "finds the right size": four different starts,
  one answer, and it matches the `[1, 2, 1]` the pruning route also reaches (§4).
* **A (1-D), noiseless: start-dependent, and two starts get *trapped*.** The
  canonical `[1, 1, 1]` and the width-2 `[1, 2, 1]` grow to good fits (~1e-6) but
  to *different* over-grown 5-layer shapes (`[1,4,4,4,1]`, `[1,2,4,4,1]`). The two
  deeper single-neuron starts `[1, 1, 1, 1]` and `[1, 1, 1, 1, 1]` **do not grow at
  all** — they stay stuck at ~3.4e-2 MSE, ~10⁴× worse. A chain of one-neuron hidden
  layers is an information bottleneck whose initial fit is so poor that the frozen-Λ
  ΔL estimate sees no single beneficial one-node edit, so patience/​the parsimony
  brake stop the search immediately.

**Takeaways.** (1) The patent-canonical minimal start `[d, 1, output]` is the right
default: start **small and wide-enough**, not **deep-and-thin** — a single-neuron
bottleneck can trap the search before it grows. (2) On a noiseless, trivially
fittable target there is no sharp stopping signal, so the exact final shape is
init/convergence-dependent (consistent with §5); recovering the minimal `[1, 2, 1]`
needs stronger per-topology Gauss-Newton convergence or a larger `eps_accept` brake
(handoff item 3), not a change to the ΔL estimate itself."""

nb.cells += [nbf.v4.new_markdown_cell(md1),
             nbf.v4.new_code_cell(code1),
             nbf.v4.new_code_cell(code2),
             nbf.v4.new_markdown_cell(md2)]
nbf.write(nb, P)
print('appended 4 cells; total now', len(nb.cells))
