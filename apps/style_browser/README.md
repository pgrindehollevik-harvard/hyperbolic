# Style browser — Euclidean vs hyperbolic

An interactive demo that simulates Midjourney's style-references workflow
(`--sref` iterative refinement) using WikiArt paintings, comparing how a
Euclidean and a hyperbolic prototype-embedding model respond to the same
user clicks.

## What it does

Two columns side-by-side. Each starts from the same random batch of 16
WikiArt validation images. **Click images** to like / un-like them
(liked tiles get a red ring) — independently in each column, since the
columns will diverge. Click **Refine** under a column and the demo:

1. Computes the **Fréchet mean** of the liked images' embeddings, in
   the geometry of that column. (Riemannian iterative algorithm for
   the hyperbolic side, ordinary mean for Euclidean.)
2. Retrieves the 16 nearest images to that centroid, again in the
   column's native geometry.
3. Replaces the grid with that batch.

Repeat. The two columns drift apart over rounds — that's the artifact.

## Performance scorecard

At the top of the page, a side-by-side scorecard compares both geometries
on the *current* iteration. Five metrics, each annotated with whether
"better" has a defined direction:

| metric | direction | what it measures |
|---|---|---|
| **Style purity @16** | ↑ higher | fraction of the displayed batch matching the dominant liked style |
| **Tree distance to intent** | ↓ lower | mean tree-distance (in the hand-built hierarchy) between batch styles and the dominant liked style |
| **Batch diversity** (entropy) | ↔ contextual | normalized Shannon entropy of style labels in the batch |
| **Centroid drift / round** | ↔ contextual | distance the centroid moved on the most recent refine, geometry-native units |
| **Disk boundary distance** | ↑ higher | hyperbolic only: distance from centroid to Poincaré radius 1/√c (approaches 0 = failure mode) |
| **Cross-column batch overlap** | ↔ contextual | how many of the 16 retrieved are shared between the two columns |

The "better on this iteration" column flags the winner where the
direction is unambiguous. After three Cubism likes plus one refine in
our smoke test, the scorecard reports: Euclidean wins purity (75% vs
56%), Euclidean wins tree distance (1.19 vs 2.94), hyperbolic edges
out on diversity (0.31 vs 0.25), Euclidean wins stability by 24×
(drift 0.05 vs 1.2), and the hyperbolic centroid sits 0.011 from the
disk boundary — already in the failure zone after a single refine.
That's the artifact in numbers.

## Live metrics per column

Above each grid:

- **round** — refine count.
- **liked** — size of the current liked set.
- **centroid Δ** — distance moved by the centroid since the previous
  refine (geometry-native units; not directly comparable across
  columns, but the trajectory shape *within* each column is).
- **batch entropy** — Shannon entropy (normalized) of style labels in
  the displayed batch. Low entropy = the geometry is converging on one
  style; high entropy = it's still showing diverse cousins.

Below each grid:

- **centroid → nearest 3 prototypes**: which 27-class style prototypes
  is the centroid closest to, with the distance. Tells you what each
  geometry "thinks" you want.
- **dist to disk boundary** (hyperbolic only): how close the centroid
  is to the Poincaré radius $1/\sqrt{c}$. Centroids that drift to the
  boundary lose interpretability — Poincaré distance diverges there.

At the bottom: a centroid-drift trajectory chart showing distance
moved per round per geometry.

## Why this is interesting

It demonstrates two of the headline findings from our paper as a tactile
artifact rather than a number in a table:

- **The geometries return different recommendations from the same liked
  set.** In our smoke test (4 random likes, k=5 retrieved per side):
  zero overlap between Euclidean and hyperbolic top-5. So at d=16 the
  two embedding spaces are organising style structure measurably
  differently.
- **The hyperbolic centroid is dramatically less stable under iteration.**
  In the same smoke test, after liking 2 round-1 results in each column
  and recomputing: Euclidean centroid moved 0.33 units (L2), hyperbolic
  moved 10.32 units (Poincaré distance). The hyperbolic Fréchet mean
  shifts hard with each user click; the Euclidean one barely moves.

For an iterative-refinement UX (which is exactly what `--sref` is), the
second observation matters a lot: hyperbolic geometry's local-retrieval
advantage (which we document in the main paper) comes with an
iteration-stability cost that the user feels every click.

## Running it

From the repo root, with the venv that already has the rest of the
project's dependencies:

```bash
pip install streamlit altair st-clickable-images  # if not already in requirements
streamlit run apps/style_browser/app.py
```

Opens at <http://localhost:8501>.

Sidebar lets you:

- Pick the embedding dimension (8 / 16 / 32 / 64). Higher d emphasises
  the local-retrieval gap from Phase 1, so divergence shows up faster.
- Set the seed for the initial random batch (so two reviewers can see
  the same starting state).
- Reset to a fresh random batch.

## Requirements (data and weights)

The demo loads checkpoints from `data/runs/sweep/<config_hash>/ckpt.pt`,
chosen by best top-1 per (geometry, dim) from `data/runs/sweep.csv`. To
get those: run `python scripts/sweep.py --phase 1 --device mps` once
(~14 min). Image thumbnails come from `data/wikiart/wikiart/`, which the
project's setup step already produces.

## What it isn't

- Not a Midjourney clone. There is no diffusion. The "candidates" are
  retrieved from a fixed corpus of 24,421 WikiArt validation images, not
  generated. This is enough to expose the embedding-space behaviour;
  the generation step is orthogonal to the geometry question.
- Not a benchmark. The numbers in the drift panel are interpretable
  inside each geometry but not directly comparable across them
  (Euclidean L2 vs Poincaré distance). The visual comparison is the
  primary takeaway; the drift panel is a sanity check.
