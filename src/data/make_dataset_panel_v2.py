"""
Dataset visualisation v2 — poster-ready.

Layout: 2 rows × 3 columns.
  Rows    = noise axis: clean (A) on top, noisy (B) on bottom
  Columns = dimensionality axis: d=1, d=2, d=3

Panels:
  d=1: scatter + true curve overlay
  d=2: contour of true f + scatter overlay
  d=3: contour slice through minimum (+ small 3-D scatter inset)

Caption baked into the figure. No train/test distinction (not visible at
poster size, not the point of this panel). No bar charts (those belong on
the separate complexity panel for Marko's item 3).

Built directly from data/processed/*.pt files via torch.load.

Hardcoded assumptions (edit here if the experimental design changes):
  - 1-D function form is (x-1)^2  (search for `y_curve` and `(x - 1.0)`)
  - Domain is [-3, 3] on every axis  (search for `np.linspace(-3, 3`)
  - σ values shown in caption: 2.0, 5.3, 23.8  (search for "sigma = 2.0")
  - Dataset names: A (clean), B (noisy)  (loop variable `ds`)
  - Dimensions: 1, 2, 3                  (loop variable `d`)
  - Sample count per panel: 4000 train + 1000 test = 5000  (caption text)

Run from the repo root:
  python src/data/make_dataset_panel_v2.py

Output:
  results/dataset_viz/dataset_panel_v2.png
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd  # noqa: F401  (kept for parity with sibling script)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import torch


def load_pt(path):
    """Load a PyTorch tensor file saved with torch.save."""
    return torch.load(path, weights_only=True)


# Resolve paths relative to this script: src/data/script.py
# parents[0] = src/data, parents[1] = src, parents[2] = repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / 'data' / 'processed'
PARAMS_JSON = DATA_DIR / 'quadratic_params.json'
OUT = REPO_ROOT / 'results' / 'dataset_viz'
OUT.mkdir(parents=True, exist_ok=True)


def load_pair(ds, d):
    suffix = '' if d == 1 else f'_{d}d'
    tr = load_pt(DATA_DIR / f'dataset_{ds}{suffix}_train.pt')
    te = load_pt(DATA_DIR / f'dataset_{ds}{suffix}_test.pt')
    # combine train and test for plotting — the panel is about the data
    # distribution, not the methodology
    x = np.concatenate([tr['x'], te['x']], axis=0).astype(np.float32)
    y = np.concatenate([tr['y'], te['y']], axis=0).astype(np.float32).ravel()
    return x, y


# Load all six datasets
data = {(ds, d): load_pair(ds, d) for ds in ('A', 'B') for d in (1, 2, 3)}

# Load quadratic A, b, c for plotting the true 2D/3D function
with open(PARAMS_JSON) as f:
    params = json.load(f)
A2, b2, c2 = np.array(params['2']['A']), np.array(params['2']['b']), params['2']['c']
A3, b3, c3 = np.array(params['3']['A']), np.array(params['3']['b']), params['3']['c']


def quad(x, A, b, c):
    return np.einsum('ni,ij,nj->n', x, A, x) + x @ b + c


# Shared colourmap range per column so clean/noisy are visually comparable
y_ranges = {}
for d in (1, 2, 3):
    y_A = data[('A', d)][1]
    y_B = data[('B', d)][1]
    y_ranges[d] = (min(y_A.min(), y_B.min()), max(y_A.max(), y_B.max()))


# ============================================================================
# Build the 2×3 figure
# ============================================================================
fig = plt.figure(figsize=(15.5, 9.3))
gs = GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.30,
              left=0.10, right=0.97, top=0.83, bottom=0.12)

# precompute the 1-D true curve
x_curve = np.linspace(-3, 3, 400)
y_curve = (x_curve - 1.0) ** 2

# precompute the 2-D contour grid
G = 120
g1, g2 = np.meshgrid(np.linspace(-3, 3, G), np.linspace(-3, 3, G))
grid_2d = np.stack([g1.ravel(), g2.ravel()], axis=1)
z_2d = quad(grid_2d, A2, b2, c2).reshape(G, G)

# precompute the 3-D slice through the minimum (slice at x3 = x3*)
x_star_3d = -0.5 * np.linalg.solve(A3, b3)
grid_3d = np.zeros((G * G, 3))
grid_3d[:, 0] = g1.ravel()
grid_3d[:, 1] = g2.ravel()
grid_3d[:, 2] = x_star_3d[2]
z_3d_slice = quad(grid_3d, A3, b3, c3).reshape(G, G)


def plot_1d(ax, ds):
    x, y = data[(ds, 1)]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(x), size=400, replace=False)
    colour = '#1f77b4' if ds == 'A' else '#d62728'
    ax.scatter(x[idx, 0], y[idx], s=12, alpha=0.55, color=colour,
               edgecolor='none', label='samples')
    ax.plot(x_curve, y_curve, '-', color='#0a2540', linewidth=2.2,
            label=r'true  $f(x)=(x-1)^2$', zorder=4)
    ax.set_xlabel(r'$x$', fontsize=11)
    ax.set_ylabel(r'$y$', fontsize=11)
    ax.set_xlim(-3.1, 3.1)
    ax.set_ylim(-8, 22)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper center', fontsize=9, framealpha=0.92)


def plot_2d(ax, ds):
    x, y = data[(ds, 2)]
    # contour of the TRUE function as background (same in both rows)
    ax.contourf(g1, g2, z_2d, levels=20, cmap='Greys', alpha=0.30,
                zorder=0)
    ax.contour(g1, g2, z_2d, levels=8, colors='black', linewidths=0.5,
               alpha=0.35, zorder=1)
    rng = np.random.default_rng(1)
    idx = rng.choice(len(x), size=900, replace=False)
    if ds == 'A':
        # clean → colour samples by their y value (matches the function colour)
        sc = ax.scatter(x[idx, 0], x[idx, 1], c=y[idx], s=14, cmap='viridis',
                        alpha=0.95, edgecolor='none',
                        vmin=y_ranges[2][0], vmax=y_ranges[2][1], zorder=2)
        cb_label = r'$y$'
    else:
        # noisy → colour samples by RESIDUAL y - f(x).  Clean = uniform 0,
        # noisy = visible scatter.  Diverging colormap centred at 0.
        f_true = quad(x[idx], A2, b2, c2)
        residual = y[idx] - f_true
        v = max(abs(residual.min()), abs(residual.max()))
        sc = ax.scatter(x[idx, 0], x[idx, 1], c=residual, s=14, cmap='coolwarm',
                        alpha=0.95, edgecolor='none',
                        vmin=-v, vmax=v, zorder=2)
        cb_label = r'$y - f(\mathbf{x})$  (noise)'
    ax.set_xlabel(r'$x_1$', fontsize=11)
    ax.set_ylabel(r'$x_2$', fontsize=11)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect('equal')
    cb = plt.colorbar(sc, ax=ax, label=cb_label, shrink=0.85, pad=0.02)
    cb.ax.tick_params(labelsize=8)
    return sc


def plot_3d(ax, ds):
    x, y = data[(ds, 3)]
    ax.contourf(g1, g2, z_3d_slice, levels=20, cmap='Greys', alpha=0.30,
                zorder=0)
    ax.contour(g1, g2, z_3d_slice, levels=8, colors='black', linewidths=0.5,
               alpha=0.30, zorder=1)
    rng = np.random.default_rng(2)
    idx = rng.choice(len(x), size=900, replace=False)
    if ds == 'A':
        sc = ax.scatter(x[idx, 0], x[idx, 1], c=y[idx], s=14, cmap='viridis',
                        alpha=0.95, edgecolor='none',
                        vmin=y_ranges[3][0], vmax=y_ranges[3][1], zorder=2)
        cb_label = r'$y$'
    else:
        f_true = quad(x[idx], A3, b3, c3)
        residual = y[idx] - f_true
        v = max(abs(residual.min()), abs(residual.max()))
        sc = ax.scatter(x[idx, 0], x[idx, 1], c=residual, s=14, cmap='coolwarm',
                        alpha=0.95, edgecolor='none',
                        vmin=-v, vmax=v, zorder=2)
        cb_label = r'$y - f(\mathbf{x})$  (noise)'
    ax.set_xlabel(r'$x_1$', fontsize=11)
    ax.set_ylabel(r'$x_2$', fontsize=11)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect('equal')
    cb = plt.colorbar(sc, ax=ax, label=cb_label, shrink=0.85, pad=0.02)
    cb.ax.tick_params(labelsize=8)
    ax.text(0.02, 0.97, r'3-D samples projected onto $(x_1,x_2)$',
            transform=ax.transAxes, fontsize=7.5, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor='gray', alpha=0.85))
    return sc


# Top row — Dataset A (clean)
plot_1d(fig.add_subplot(gs[0, 0]), 'A')
plot_2d(fig.add_subplot(gs[0, 1]), 'A')
plot_3d(fig.add_subplot(gs[0, 2]), 'A')

# Bottom row — Dataset B (noisy)
plot_1d(fig.add_subplot(gs[1, 0]), 'B')
plot_2d(fig.add_subplot(gs[1, 1]), 'B')
plot_3d(fig.add_subplot(gs[1, 2]), 'B')

# ── Column headers and row labels (the "two axes of difficulty") ─────────────
# Column headers above the top row — short labels
col_titles = [r'$d = 1$',
              r'$d = 2$',
              r'$d = 3$']
col_subtitles = [r'$f(x) = (x-1)^2$',
                 r'$f(\mathbf{x}) = \mathbf{x}^\top A\mathbf{x} + \mathbf{b}^\top\mathbf{x} + c$',
                 r'$f: \mathbb{R}^3 \to \mathbb{R}$  (slice shown)']
for col, (t, s) in enumerate(zip(col_titles, col_subtitles)):
    x_pos = 0.10 + (col + 0.5) * (0.87 / 3)
    fig.text(x_pos, 0.875, t, ha='center', va='bottom',
             fontsize=14, fontweight='bold')
    fig.text(x_pos, 0.852, s, ha='center', va='top', fontsize=10,
             color='#444')

# Row labels on the left, vertical and centred on each row
fig.text(0.038, 0.66, 'Dataset A\n(clean)', ha='center', va='center',
         fontsize=12, fontweight='bold', color='#1f77b4')
fig.text(0.038, 0.29, 'Dataset B\n(noisy)', ha='center', va='center',
         fontsize=12, fontweight='bold', color='#d62728')

# Two arrows indicating the difficulty axes
fig.text(0.52, 0.93,
         'dimensionality increases  →',
         ha='center', va='top', fontsize=10, color='#444',
         fontstyle='italic')
fig.text(0.013, 0.475,
         'noise increases  →',
         ha='center', va='center', fontsize=10, color='#444',
         fontstyle='italic', rotation=90)

# Main title
fig.suptitle('Two axes of difficulty: dimensionality and noise',
             fontsize=17, fontweight='bold', y=0.975)

# Caption at the bottom
caption = ('Increasing input dimensionality (left → right) and additive noise '
           '(top → bottom) create progressively harder regression tasks, '
           'motivating the question of how lean the network can remain.\n'
           r'Each panel shows 5 000 i.i.d. samples on $[-3, 3]^d$;  '
           r'noise scaled to 12.5 % of peak-to-peak range  '
           r'($\sigma = 2.0,\, 5.3,\, 23.8$ at $d = 1, 2, 3$).')
fig.text(0.52, 0.022, caption, ha='center', va='bottom', fontsize=9.5,
         style='italic', wrap=True,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#f7f7f7',
                   edgecolor='#bbb', linewidth=0.6))

fig.savefig(OUT / 'dataset_panel_v2.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print('wrote dataset_panel_v2.png')

print('\nDone.')
