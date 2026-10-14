# Poster layout — DIN A0 landscape

Canvas: **1189 × 841 mm** (DIN A0 landscape). 12-column grid (≈ 99 mm
wide each) × 8-row grid (≈ 105 mm high each). Outer margin: 30 mm on all
sides. Inner gutter: 15 mm.

Drop-in for `~/Downloads/2111_Wissenschaftliches_Plakat_DIN-A0_Querformat.potx`.
Content blocks come from [poster_draft.md](poster_draft.md). Figures
referenced live in [`results/scaling_law/`](../../results/scaling_law/).

---

## Visual layout (proportional ASCII sketch)

```
+================================================================================+
| TITLE STRIP — height 80 mm                                                     |
|                                                                                |
|   Pruning a small regression MLP scales differently than you'd guess           |
|   Ten methods · two quadratics · three input dimensions                        |
|   Marko Prohaska · CS_ANN scaling-law-study · 2026                             |
+================================================================================+
| LEFT 280 mm        | CENTER 530 mm                       | RIGHT 280 mm        |
|                    |                                     |                     |
|  ┌──────────────┐  |  ┌─────────────────────────────────┐ | ┌────────────────┐ |
|  │ 1 Problem &  │  |  │ HERO FIG A                       │ | │ 5 Five findings│ |
|  │   question   │  |  │ fig_dim_vs_max_sparsity_A.png    │ | │  (5 bullets)   │ |
|  │  (~50 words) │  |  │  ~ 260 × 165 mm                  │ | │  ~120 words    │ |
|  └──────────────┘  |  └─────────────────────────────────┘ | │                │ |
|                    |  ┌─────────────────────────────────┐ | │                │ |
|  ┌──────────────┐  |  │ HERO FIG B                       │ | └────────────────┘ |
|  │ 2 Methodology│  |  │ fig_dim_vs_max_sparsity_B.png    │ | ┌────────────────┐ |
|  │ (~80 words)  │  |  │  ~ 260 × 165 mm                  │ | │ 6 Take-aways   │ |
|  │              │  |  └─────────────────────────────────┘ | │  (~50 words)   │ |
|  └──────────────┘  |  ┌──────────────┐  ┌──────────────┐  | │                │ |
|                    |  │ FIG C        │  │ FIG D        │  | └────────────────┘ |
|  ┌──────────────┐  |  │ Pareto       │  │ FLOPs scale  │  | ┌────────────────┐ |
|  │ 3 Methods    │  |  │ params_vs_   │  │ dim_vs_      │  | │ QR + repo URL  │ |
|  │   table      │  |  │ mse_A_3d     │  │ flops_A      │  | │ + thanks       │ |
|  │ (~60 words)  │  |  │ ~125 × 90 mm │  │ ~125 × 90 mm │  | └────────────────┘ |
|  └──────────────┘  |  └──────────────┘  └──────────────┘  |                     |
+================================================================================+
| METHOD LEGEND BAR — height 35 mm. Colour swatches + method names, left→right.  |
+================================================================================+
```

---

## Grid coordinates (mm from top-left, after 30 mm margin)

Every box below is given as `(x, y, w, h)` in mm, all relative to the
**1189 × 841 mm canvas**. The numbers assume 30 mm outer margin and 15 mm
inner gutters.

### Header strip — 1129 × 80 mm

| Element | x | y | w | h | Type / setting |
|---|---|---|---|---|---|
| Title             | 30  | 30  | 870 | 36 | 60 pt bold, dark navy |
| Sub-title         | 30  | 70  | 870 | 22 | 32 pt italic |
| Authors / repo    | 30  | 96  | 870 | 16 | 22 pt mono |
| Logo (optional)   | 950 | 30  | 209 | 80 | top-right |

### Left column — 280 × 660 mm (three stacked text blocks)

| Block | x | y | w | h | Words |
|---|---|---|---|---|---|
| 1 — Problem & question | 30 | 130 | 280 | 200 | ~50 |
| 2 — Methodology        | 30 | 345 | 280 | 200 | ~80 |
| 3 — Methods table      | 30 | 560 | 280 | 230 | ~60 + table |

Body font: 24 pt sans (Inter / DIN Next / Source Sans).
Section heading: 32 pt, same-family, brand-orange.

### Centre column — 530 × 660 mm (two hero figures + two small figures)

| Figure | x | y | w | h | Source PNG |
|---|---|---|---|---|---|
| FIG A — Dim vs max sparsity, A | 325 | 130 | 530 | 200 | `fig_dim_vs_max_sparsity_A.png` |
| FIG B — Dim vs max sparsity, B | 325 | 345 | 530 | 200 | `fig_dim_vs_max_sparsity_B.png` |
| FIG C — Params vs MSE, A 3-D   | 325 | 560 | 257 | 230 | `fig_params_vs_mse_A_3d.png` |
| FIG D — FLOPs scaling, A       | 597 | 560 | 258 | 230 | `fig_dim_vs_flops_at_target_A.png` |

All four figures: white background, 140 dpi already, leave a 5 mm pad
inside each frame and a 1 pt rule (light grey) around it. Caption (16 pt,
italic) directly below each figure.

### Right column — 280 × 660 mm (two blocks + QR)

| Block | x | y | w | h | Words |
|---|---|---|---|---|---|
| 5 — Five findings   | 870 | 130 | 290 | 320 | ~120 |
| 6 — Take-aways      | 870 | 470 | 290 | 180 | ~50 |
| QR + repo URL       | 870 | 670 | 290 | 120 | repo link + small QR (≥ 35 × 35 mm) |

### Method legend bar — 1129 × 35 mm

Place at `(30, 776, 1129, 35)`. Single row of ten swatch + name pairs,
left-justified, evenly spaced (≈ 112 mm per pair). 18 pt mono name. Colour
swatch: 18 × 18 mm rounded rectangle.

```
OBD ▮  OBS ▮  mag ▮  struct ▮  activ ▮  redundancy ▮  lowrank ▮  Taylor-FO ▮  SNIP ▮  IMP+rewind ▮
```

Colours: steelblue, darkorange, seagreen, crimson, purple, teal,
goldenrod, indianred, slategrey, darkgreen.

---

## Typography & colour

- Font family: a single sans (Inter, DIN Next, or Source Sans Pro).
- Heading colour: navy `#1F2A44`. Body: charcoal `#1A1A1A`. Brand accent:
  orange `#D97A2E` (matches the `darkorange` line in the figures).
- Background: warm white `#FAFAF7`.
- 1 pt rule (`#D0D0CC`) around each figure, no shadow.
- All text left-aligned. No justified text on a poster — kerning gaps
  hurt readability at 2 m viewing distance.

## Reading distance heuristics

| Element | Min font size (pt) | Min readable distance |
|---|---|---|
| Title  | 60 | 5 m |
| Section heads | 32 | 2 m |
| Body | 24 | 1 m |
| Captions | 16 | 0.5 m |

A viewer should be able to read the title and findings at 5 m. They walk
in and read the captions at 1 m. Hero figures are sized to be the focal
point at 2 m.

## Build steps in PowerPoint

1. Open the template:
   `~/Downloads/2111_Wissenschaftliches_Plakat_DIN-A0_Querformat.potx`.
2. Set the slide size to 1189 × 841 mm if not already.
3. Drop a 12 × 8 grid of guides at 99 × 105 mm cells with a 15 mm gutter
   (the math: 30 mm outer + (12 × 99) + (11 × 15) + 30 mm = ?, no — use
   the cell+gutter approach: cell 84 mm, gutter 15 mm, columns 12 →
   `30 + 12·84 + 11·15 = 1203 mm` → close enough; nudge to 1189 by
   shrinking the outer margin to 22 mm).
4. Paste in the six text blocks from [poster_draft.md](poster_draft.md).
5. Insert the four PNGs from
   [results/scaling_law/](../../results/scaling_law/) and resize to the
   `w × h` values above.
6. Add the method legend bar as a 10-cell row of swatches.
7. Add a QR pointing to
   `github.com/sima2006/CS_ANN/tree/scaling-law-study`.

## Optional: alternate layouts

- **3-column hero**: replace FIG D with a third hero figure
  (`fig_sparsity_vs_mse_A_3d.png`) for a "low → mid → high" Pareto story.
- **2-row hero**: drop FIG C/D and grow A & B to 800 × 250 mm each
  ("the only two charts we need").
- **Numbers-first variant**: replace FIG D with a poster-rendered version
  of the §6 results table from [report.md](report.md) — same data,
  bigger numbers.

---

**Total content budget on the poster:** ~360 body words + 4 figures + 1
table + 1 legend bar. Everything else lives in
[report.md](report.md).
