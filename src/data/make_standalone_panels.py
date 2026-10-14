"""
Standalone per-dimension panels matching the combined v2 style.

Each output is poster-ready on its own. They share:
  - viridis colourmap for clean values
  - coolwarm diverging colourmap centred at 0 for noise residuals
  - blue/red colour-coding for Dataset A / Dataset B labels
  - same caption style and font sizes
  - same data loaded from data/processed/*.pt

Hardcoded assumptions (edit here if the experimental design changes):
  - 1-D function form is (x-1)^2
  - Domain is [-3, 3]^d
  - σ values shown in titles/captions: 2.0, 5.3, 23.8 at d=1,2,3

Run from the repo root:
  python src/data/make_standalone_panels.py

Outputs (in results/dataset_viz/):
  dataset_1d_only.png
  dataset_2d_only.png
  dataset_3d_only.png
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec  # noqa: F401

import torch


def load_pt(path):
    """Load a PyTorch tensor file saved with torch.save."""
    return torch.load(path, weights_only=True)


# Resolve paths relative to this script: src/data/script.py
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / 'data' / 'processed'
PARAMS_JSON = DATA_DIR / 'quadratic_params.json'
OUT = REPO_ROOT / 'results' / 'dataset_viz'
OUT.mkdir(parents=True, exist_ok=True)


def load_pair(ds, d):
    suffix = '' if d == 1 else f'_{d}d'
    tr = load_pt(DATA_DIR / f'dataset_{ds}{suffix}_train.pt')
    te = load_pt(DATA_DIR / f'dataset_{ds}{suffix}_test.pt')
    x = np.concatenate([tr['x'], te['x']], axis=0).astype(np.float32)
    y = np.concatenate([tr['y'], te['y']], axis=0).astype(np.float32).ravel()
    return x, y


data = {(ds, d): load_pair(ds, d) for ds in ('A', 'B') for d in (1, 2, 3)}

with open(PARAMS_JSON) as f:
    params = json.load(f)
A2, b2, c2 = np.array(params['2']['A']), np.array(params['2']['b']), params['2']['c']
A3, b3, c3 = np.array(params['3']['A']), np.array(params['3']['b']), params['3']['c']


def quad(x, A, b, c):
    return np.einsum('ni,ij,nj->n', x, A, x) + x @ b + c


# Shared y-range per dimension
y_ranges = {}
for d in (1, 2, 3):
    y_A = data[('A', d)][1]
    y_B = data[('B', d)][1]
    y_ranges[d] = (min(y_A.min(), y_B.min()), max(y_A.max(), y_B.max()))


# Pre-compute grids
G = 120
g1, g2 = np.meshgrid(np.linspace(-3, 3, G), np.linspace(-3, 3, G))
grid_2d = np.stack([g1.ravel(), g2.ravel()], axis=1)
z_2d = quad(grid_2d, A2, b2, c2).reshape(G, G)

x_star_3d = -0.5 * np.linalg.solve(A3, b3)
grid_3d = np.zeros((G * G, 3))
grid_3d[:, 0] = g1.ravel(); grid_3d[:, 1] = g2.ravel(); grid_3d[:, 2] = x_star_3d[2]
z_3d_slice = quad(grid_3d, A3, b3, c3).reshape(G, G)


# ============================================================================
# 1-D STANDALONE
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
x_curve = np.linspace(-3, 3, 400)
y_curve = (x_curve - 1.0) ** 2

for ax, ds in zip(axes, ('A', 'B')):
    x, y = data[(ds, 1)]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(x), size=500, replace=False)
    colour = '#1f77b4' if ds == 'A' else '#d62728'
    label = 'Dataset A (clean)' if ds == 'A' else 'Dataset B (noisy, σ=2.0)'
    ax.scatter(x[idx, 0], y[idx], s=14, alpha=0.55, color=colour,
               edgecolor='none', label='samples')
    ax.plot(x_curve, y_curve, '-', color='#0a2540', linewidth=2.2,
            label=r'true  $f(x) = (x-1)^2$', zorder=4)
    ax.set_xlabel(r'$x$', fontsize=12)
    if ds == 'A':
        ax.set_ylabel(r'$y$', fontsize=12)
    ax.set_xlim(-3.1, 3.1); ax.set_ylim(-8, 22)
    ax.set_title(label, fontsize=13, fontweight='bold', color=colour, pad=8)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper center', fontsize=10, framealpha=0.92)

fig.suptitle(r'1-D benchmark:  $f(x) = (x - 1)^2$,  $x \in [-3, 3]$',
             fontsize=15, fontweight='bold', y=1.02)
fig.text(0.5, -0.04,
         'Dataset B adds Gaussian noise with σ = 2.0 (12.5 % of peak-to-peak '
         'range), giving a Bayes-optimal test MSE of σ² = 4.0.',
         ha='center', va='top', fontsize=9.5, style='italic',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#f7f7f7',
                   edgecolor='#bbb', linewidth=0.6))
plt.tight_layout()
fig.savefig(OUT / 'dataset_1d_only.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print('wrote dataset_1d_only.png')


# ============================================================================
# 2-D STANDALONE
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))

for ax, ds in zip(axes, ('A', 'B')):
    x, y = data[(ds, 2)]
    # Grey contour of true f as background
    ax.contourf(g1, g2, z_2d, levels=20, cmap='Greys', alpha=0.30, zorder=0)
    ax.contour(g1, g2, z_2d, levels=8, colors='black', linewidths=0.5,
               alpha=0.35, zorder=1)
    rng = np.random.default_rng(1)
    idx = rng.choice(len(x), size=1100, replace=False)
    if ds == 'A':
        sc = ax.scatter(x[idx, 0], x[idx, 1], c=y[idx], s=16, cmap='viridis',
                        alpha=0.95, edgecolor='none',
                        vmin=y_ranges[2][0], vmax=y_ranges[2][1], zorder=2)
        cb_label = r'$y$'
        title_colour = '#1f77b4'
        title = 'Dataset A (clean) — colour shows $y$'
    else:
        f_true = quad(x[idx], A2, b2, c2)
        residual = y[idx] - f_true
        v = max(abs(residual.min()), abs(residual.max()))
        sc = ax.scatter(x[idx, 0], x[idx, 1], c=residual, s=16, cmap='coolwarm',
                        alpha=0.95, edgecolor='none', vmin=-v, vmax=v, zorder=2)
        cb_label = r'$y - f(\mathbf{x})$  (noise)'
        title_colour = '#d62728'
        title = 'Dataset B (noisy, σ=5.3) — colour shows residual'
    ax.set_xlabel(r'$x_1$', fontsize=12)
    ax.set_ylabel(r'$x_2$', fontsize=12)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect('equal')
    ax.set_title(title, fontsize=12, fontweight='bold', color=title_colour, pad=8)
    cb = plt.colorbar(sc, ax=ax, label=cb_label, shrink=0.85, pad=0.02)
    cb.ax.tick_params(labelsize=9)

fig.suptitle(r'2-D benchmark:  $f(\mathbf{x}) = \mathbf{x}^\top A\mathbf{x} '
             r'+ \mathbf{b}^\top \mathbf{x} + c$,  $\mathbf{x} \in [-3, 3]^2$',
             fontsize=15, fontweight='bold', y=1.02)
fig.text(0.5, -0.025,
         'Greyscale background shows the true function $f$. Clean samples '
         '(left) trace this surface; noisy samples (right) scatter around '
         'it — visualised as residuals so the noise is directly visible.',
         ha='center', va='top', fontsize=9.5, style='italic', wrap=True,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#f7f7f7',
                   edgecolor='#bbb', linewidth=0.6))
plt.tight_layout()
fig.savefig(OUT / 'dataset_2d_only.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print('wrote dataset_2d_only.png')


# ============================================================================
# 3-D STANDALONE
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))

for ax, ds in zip(axes, ('A', 'B')):
    x, y = data[(ds, 3)]
    ax.contourf(g1, g2, z_3d_slice, levels=20, cmap='Greys', alpha=0.30, zorder=0)
    ax.contour(g1, g2, z_3d_slice, levels=8, colors='black', linewidths=0.5,
               alpha=0.30, zorder=1)
    rng = np.random.default_rng(2)
    idx = rng.choice(len(x), size=1100, replace=False)
    if ds == 'A':
        sc = ax.scatter(x[idx, 0], x[idx, 1], c=y[idx], s=16, cmap='viridis',
                        alpha=0.95, edgecolor='none',
                        vmin=y_ranges[3][0], vmax=y_ranges[3][1], zorder=2)
        cb_label = r'$y$'
        title_colour = '#1f77b4'
        title = 'Dataset A (clean) — colour shows $y$'
    else:
        f_true = quad(x[idx], A3, b3, c3)
        residual = y[idx] - f_true
        v = max(abs(residual.min()), abs(residual.max()))
        sc = ax.scatter(x[idx, 0], x[idx, 1], c=residual, s=16, cmap='coolwarm',
                        alpha=0.95, edgecolor='none', vmin=-v, vmax=v, zorder=2)
        cb_label = r'$y - f(\mathbf{x})$  (noise)'
        title_colour = '#d62728'
        title = 'Dataset B (noisy, σ=23.8) — colour shows residual'
    ax.set_xlabel(r'$x_1$', fontsize=12)
    ax.set_ylabel(r'$x_2$', fontsize=12)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect('equal')
    ax.set_title(title, fontsize=12, fontweight='bold', color=title_colour, pad=8)
    cb = plt.colorbar(sc, ax=ax, label=cb_label, shrink=0.85, pad=0.02)
    cb.ax.tick_params(labelsize=9)
    ax.text(0.02, 0.97, r'3-D samples projected onto $(x_1,x_2)$',
            transform=ax.transAxes, fontsize=8, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor='gray', alpha=0.85))

fig.suptitle(r'3-D benchmark:  $f(\mathbf{x}) = \mathbf{x}^\top A\mathbf{x} '
             r'+ \mathbf{b}^\top \mathbf{x} + c$,  $\mathbf{x} \in [-3, 3]^3$',
             fontsize=15, fontweight='bold', y=1.02)
fig.text(0.5, -0.025,
         '3-D samples are projected onto $(x_1, x_2)$; greyscale shows a '
         r'slice of $f$ at $x_3 = x_3^*$ for orientation. '
         'Noise scale grows to ±70 (vs. ±18 at d=2) because σ scales with '
         'function peak-to-peak.',
         ha='center', va='top', fontsize=9.5, style='italic', wrap=True,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#f7f7f7',
                   edgecolor='#bbb', linewidth=0.6))
plt.tight_layout()
fig.savefig(OUT / 'dataset_3d_only.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print('wrote dataset_3d_only.png')

print('\nDone.')
