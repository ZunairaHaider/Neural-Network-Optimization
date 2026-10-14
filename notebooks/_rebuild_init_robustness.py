# Replaces the init-robustness sections of convergence_analysis.ipynb with a
# single starting-topology study over the three single-neuron-chain starts
# "1" [1,1,1], "1-1" [1,1,1,1], "1-1-1" [1,1,1,1,1].
import os
import nbformat as nbf

P = os.path.join(os.path.dirname(__file__), 'convergence_analysis.ipynb')
nb = nbf.read(P, as_version=4)

# keep §1-§4 (cells 0..10), drop the old seed §5 + 4-start §6
keep = []
for c in nb.cells:
    s = ''.join(c.source)
    if s.startswith('## 5.') or s.startswith('## 6.'):
        break
    keep.append(c)
nb.cells = keep
print('kept', len(keep), 'cells (through §4)')

md1 = r"""## 5. Starting-topology robustness (1-D datasets) — start small, vary the *start*

The patent method is meant to *start small and grow to what the data needs*, so a
good method should reach the same final architecture from any small start. We test
that directly: **fix the random seed** and run the faithful search from three
minimal-width starting topologies that differ only in **initial depth** —

| label | starting topology | hidden layers |
|---|---|---|
| `1` | `[1, 1, 1]` | one single-neuron layer (`[d, 1, output]`) |
| `1-1` | `[1, 1, 1, 1]` | two single-neuron layers |
| `1-1-1` | `[1, 1, 1, 1, 1]` | three single-neuron layers |

— on the two 1-D datasets (A noiseless, B noisy). Run with
`python -m patenting.start_topology_robustness`.

> A net with **no hidden layer** (`[1, 1]`) is not a valid start: the constructive
> search grows *from* an existing hidden layer, so it needs at least one — which is
> why the patent's minimal start is `[d, 1, output]`."""

code1 = r"""st_path = os.path.join(ROOT, 'results', 'start_topology_robustness.json')
with open(st_path) as fh:
    start_rob = json.load(fh)

label = {3: '1', 4: '1-1', 5: '1-1-1'}   # by start length
for name, d in start_rob.items():
    print(f"{name}  (seed {d['seed']} fixed)")
    for r in d['runs']:
        lab = label.get(len(r['start']), '?')
        print(f"    {lab:6s} start {str(r['start']):16s} -> final "
              f"{str(r['final_topology']):18s}  test_mse={r['test_mse']:.3e}")
    uq = d['unique_final_topologies']
    print(f"    => {len(uq)} distinct final topology(ies): {uq}  "
          f"[{'ALL AGREE' if d['all_agree'] else 'START-DEPENDENT'}]\n")"""

code2 = r"""# Visualise the final topology reached from each start.
# Green = the modal final shape for that dataset; orange = a variant.
from collections import Counter

n_st  = max(len(d['runs']) for d in start_rob.values())
order = ['dataset_A', 'dataset_B']
names = [n for n in order if n in start_rob] + \
        [n for n in start_rob if n not in order]
label = {3: '1', 4: '1-1', 5: '1-1-1'}

fig, axes = plt.subplots(len(names), n_st,
                         figsize=(3.0 * n_st, 2.9 * len(names)), squeeze=False)
for r, name in enumerate(names):
    runs  = start_rob[name]['runs']
    modal = Counter(tuple(x['final_topology']) for x in runs).most_common(1)[0][0]
    for c, run in enumerate(runs):
        fin   = run['final_topology']
        color = '#55A868' if tuple(fin) == modal else '#DD8452'
        lab   = label.get(len(run['start']), '?')
        draw_topology(axes[r][c], fin,
                      f"start '{lab}' {run['start']} → {fin}\n({run['test_mse']:.1e})",
                      color)
    for c in range(len(runs), n_st):
        axes[r][c].axis('off')
    verdict = 'ALL AGREE' if start_rob[name]['all_agree'] else 'START-DEPENDENT'
    axes[r][0].set_ylabel(f"{name}\n{verdict}", fontsize=10, rotation=90, labelpad=12)
fig.suptitle("Starting-topology robustness: final topology per start "
             "(green = modal shape, orange = variant)", y=1.01, fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, 'results', 'comparison',
            'start_topology_robustness.png'), dpi=130, bbox_inches='tight')
plt.show()"""

md2 = r"""### Interpretation — does extra initial depth change anything?

* **B (1-D), noisy: no — fully robust.** All three starts (`1`, `1-1`, `1-1-1`)
  converge to the *same* minimal `[1, 2, 1]` at the noise-floor MSE (~0.171). The
  data's stopping signal — not the starting depth — fixes the architecture, and it
  matches the `[1, 2, 1]` the pruning route also reaches (§4). This is the strongest
  single piece of evidence that the method "finds the right size".
* **A (1-D), noiseless: yes — and deeper starts get *trapped*.** Only the canonical
  `1` = `[1, 1, 1]` grows to a good fit (`[1, 4, 4, 4, 1]`, ~1e-6). The deeper
  starts `1-1` and `1-1-1` **do not grow at all** — they stay stuck at their thin
  initial shape with ~3.4e-2 MSE, **~10⁴× worse**. A chain of one-neuron hidden
  layers is an information bottleneck whose initial fit is so poor that the frozen-Λ
  ΔL estimate sees no single beneficial one-node edit, so patience / the parsimony
  brake stop the search immediately.

**Takeaways.** (1) The patent-canonical minimal start `[d, 1, output]` is the right
default — start **small and wide-enough**, not **deep-and-thin**; a single-neuron
bottleneck can trap the search before it ever grows. (2) On a noiseless, trivially
fittable target there is no sharp stopping signal, so even the good start over-grows
and the exact final shape is init/convergence-dependent; recovering the minimal
`[1, 2, 1]` would need stronger per-topology Gauss-Newton convergence or a larger
`eps_accept` brake (handoff item 3), not a change to the ΔL estimate itself."""

nb.cells += [nbf.v4.new_markdown_cell(md1),
             nbf.v4.new_code_cell(code1),
             nbf.v4.new_code_cell(code2),
             nbf.v4.new_markdown_cell(md2)]
nbf.write(nb, P)
print('rebuilt; total now', len(nb.cells))
