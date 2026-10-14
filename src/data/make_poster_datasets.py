"""Generate poster dataset figure: 2x2 grid (1D + 2D surface, clean + noisy)."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
import torch

# ── reproduce quadratic params exactly as in generate_data.py ────────────────
def _make_quad_2d():
    g = torch.Generator().manual_seed(44)   # seed + d = 42 + 2
    L = torch.randn(2, 2, generator=g)
    A = (L @ L.T + 0.5 * torch.eye(2)).numpy()
    b = torch.randn(2, generator=g).numpy()
    c = 1.0
    return A, b, c

def f1d(x): return (x - 1.0) ** 2
def f2d(x1, x2, A, b, c):
    pts = np.stack([x1.ravel(), x2.ravel()], axis=1)   # (N,2)
    quad = (pts @ A * pts).sum(axis=1)
    lin  = pts @ b
    return (quad + lin + c).reshape(x1.shape)

# ── 1-D data (reproduce exactly) ─────────────────────────────────────────────
np.random.seed(42)
torch.manual_seed(42)
x1d = np.linspace(-3, 3, 5000)
np.random.shuffle(x1d)
y1d_A = f1d(x1d)
y1d_B = y1d_A + np.random.normal(0, 2.0, size=x1d.shape)

# ── 2-D data ──────────────────────────────────────────────────────────────────
A2, b2, c2 = _make_quad_2d()
rng = np.random.default_rng(4244)
x2d = rng.uniform(-3, 3, (5000, 2))
y2d_A = f2d(x2d[:, 0:1], x2d[:, 1:2], A2, b2, c2).ravel()
peak_to_peak = y2d_A.max() - y2d_A.min()
sigma_2d = 0.125 * peak_to_peak
rng2 = np.random.default_rng(7779)
y2d_B = y2d_A + rng2.normal(0, sigma_2d, size=y2d_A.shape)

# ── surface grid ──────────────────────────────────────────────────────────────
g1 = np.linspace(-3, 3, 60)
g2 = np.linspace(-3, 3, 60)
G1, G2 = np.meshgrid(g1, g2)
ZA = f2d(G1, G2, A2, b2, c2)
ZB = ZA  # true surface same; noise added to scatter only

# ── colour palette ────────────────────────────────────────────────────────────
TUM_BLUE   = "#0065BD"
TUM_ORANGE = "#E37222"
TUM_IVORY  = "#DAD7CB"
SURF_ALPHA = 0.70

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor("white")

TITLE_KW  = dict(fontsize=13, fontweight="bold", color=TUM_BLUE, pad=8)
LABEL_KW  = dict(fontsize=10)
TICK_KW   = dict(labelsize=9)
SCAT_KW   = dict(s=4, alpha=0.35, linewidths=0)
SURF_KW   = dict(alpha=SURF_ALPHA, linewidth=0, antialiased=True)

# ── (1,1) 1-D Dataset A ───────────────────────────────────────────────────────
ax11 = fig.add_subplot(2, 2, 1)
ax11.scatter(x1d[:400], y1d_A[:400], color=TUM_BLUE, **SCAT_KW, label="samples")
xc = np.linspace(-3, 3, 400)
ax11.plot(xc, f1d(xc), color="black", lw=2, label=r"$f(x)=(x-1)^2$")
ax11.set_xlim(-3.1, 3.1); ax11.set_ylim(-0.5, 17)
ax11.set_xlabel("$x$", **LABEL_KW); ax11.set_ylabel("$f(x)$", **LABEL_KW)
ax11.set_title("Dataset A — clean, $d=1$", **TITLE_KW)
ax11.legend(fontsize=9, markerscale=3); ax11.tick_params(**TICK_KW)
ax11.grid(True, alpha=0.25)

# ── (1,2) 2-D Dataset A — 3-D surface ────────────────────────────────────────
ax12 = fig.add_subplot(2, 2, 2, projection="3d")
ax12.plot_surface(G1, G2, ZA, cmap="Blues", **SURF_KW)
ax12.set_xlabel("$x_1$", **LABEL_KW, labelpad=4)
ax12.set_ylabel("$x_2$", **LABEL_KW, labelpad=4)
ax12.set_zlabel("$f(x_1,x_2)$", **LABEL_KW, labelpad=4)
ax12.set_title("Dataset A — clean, $d=2$", **TITLE_KW)
ax12.tick_params(**TICK_KW); ax12.view_init(elev=28, azim=-55)
ax12.set_box_aspect([1, 1, 0.7])

# ── (2,1) 1-D Dataset B ───────────────────────────────────────────────────────
ax21 = fig.add_subplot(2, 2, 3)
ax21.scatter(x1d[:400], y1d_B[:400], color=TUM_ORANGE, **SCAT_KW, label="samples")
ax21.plot(xc, f1d(xc), color="black", lw=2, label=r"$f(x)=(x-1)^2$")
ax21.set_xlim(-3.1, 3.1)
ax21.set_xlabel("$x$", **LABEL_KW); ax21.set_ylabel("$f(x)+\epsilon$", **LABEL_KW)
ax21.set_title(r"Dataset B — noisy ($\sigma=2$), $d=1$", **TITLE_KW)
ax21.legend(fontsize=9, markerscale=3); ax21.tick_params(**TICK_KW)
ax21.grid(True, alpha=0.25)

# ── (2,2) 2-D Dataset B — 3-D surface + noisy scatter ────────────────────────
ax22 = fig.add_subplot(2, 2, 4, projection="3d")
ax22.plot_surface(G1, G2, ZB, cmap="Oranges", **SURF_KW)
# scatter a subset of noisy points
sub = slice(0, 400)
ax22.scatter(x2d[sub, 0], x2d[sub, 1], y2d_B[sub],
             color=TUM_ORANGE, s=6, alpha=0.45, depthshade=False)
ax22.set_xlabel("$x_1$", **LABEL_KW, labelpad=4)
ax22.set_ylabel("$x_2$", **LABEL_KW, labelpad=4)
ax22.set_zlabel("$f+\epsilon$", **LABEL_KW, labelpad=4)
ax22.set_title(r"Dataset B — noisy, $d=2$", **TITLE_KW)
ax22.tick_params(**TICK_KW); ax22.view_init(elev=28, azim=-55)
ax22.set_box_aspect([1, 1, 0.7])

fig.suptitle("Two axes of difficulty: dimensionality  and  noise",
             fontsize=15, fontweight="bold", color=TUM_BLUE, y=1.01)

plt.tight_layout(rect=[0, 0, 1, 0.99])

out = Path(__file__).resolve().parents[2] / "poster" / "dataset_viz_poster.png"
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
