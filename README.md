# Hyperbolic Embeddings for Hierarchical Style Representation

> **TL;DR.** Hyperbolic prototype embeddings preserve *local* hierarchy on artistic-style data more faithfully than Euclidean embeddings of matched dimension. The local advantage is real and robust across reference trees, regularizer strengths, embedding dimensions, and class-imbalance handling. The classification gap in Euclidean's favour is also robust — but a logistic-regression baseline on raw CLIP features matches the Euclidean prototype, suggesting Euclidean prototype geometry adds nothing over the encoder. Hyperbolic prototype geometry adds local-retrieval value the encoder doesn't already provide.

📄 **Paper:** [`docs/main.pdf`](docs/main.pdf)
🧪 **End-to-end notebook:** [`notebooks/ms4_main.ipynb`](notebooks/ms4_main.ipynb)
📊 **Sweep CSV (every config × every metric):** [`notebooks/sweep_results.csv`](notebooks/sweep_results.csv)

<p align="center">
  <img src="docs/figures/poincare_disk_d2.png" alt="Poincaré disk vs Euclidean prototype layouts at d=2" width="100%"/>
</p>

*Trained at d=2: hyperbolic prototypes (right) settle into a near-boundary ring — exactly where the exponential-volume regime lives — while Euclidean prototypes (left) scatter without structural pressure. The geometry behaves as theory predicts. Whether that geometric difference shows up in the downstream metrics that matter is what this paper actually measures.*

---

## What this project asks

Artistic style is hierarchical. Renaissance branches into Baroque, Baroque into Romanticism, Impressionism into Cubism. An embedding that respects this branching should be more useful for tasks like influence-aware retrieval, museum-catalog browsing, or pedagogical tools that organize collections by lineage rather than raw visual similarity.

Standard image embeddings live in $\mathbb{R}^d$ where the volume of a ball at radius $r$ grows polynomially. A tree's leaf count grows exponentially with depth. Forcing tree-like data into Euclidean coordinates therefore distorts pairwise distances, and the distortion widens as the tree deepens. The Poincaré ball, with its own exponential volume growth, embeds trees with provably lower distortion ([Nickel & Kiela 2017](https://arxiv.org/abs/1705.08039); [Sala et al. 2018](https://arxiv.org/abs/1804.03329)).

**The question:** does that mathematical advantage translate into an empirical one for representing artistic style?

## What we found

Across 150 seed-replicated runs spanning embedding dimension, curvature, and the strength $\lambda$ of a hierarchy-aware regularizer, the headline at the best per-geometry configuration is:

| | **Euclidean** | **Hyperbolic** | Winner |
|---|---:|---:|:---:|
| Top-1 accuracy | 65.9% ± 0.3 | 60.6% ± 0.2 | Eu |
| Top-5 accuracy | 95.9% ± 0.1 | 93.5% ± 0.1 | Eu |
| Class-center / tree Spearman (default tree) | 0.350 ± 0.027 | 0.242 ± 0.002 | Eu |
| **Sibling recall@5** | 0.136 ± 0.003 | **0.188 ± 0.004** | **Hy** |
| **Cousin recall@5** | 0.249 ± 0.003 | **0.296 ± 0.005** | **Hy** |

Two things sit underneath those numbers:

1. **The local advantage is robust.** The sibling/cousin recall gap holds across the full $d \in \{2,4,8,16,32,64\}$ sweep, every $\lambda \in \{0,0.1,0.3,1,3\}$, three different reference-tree definitions (lineage / chronological / flat null), and inverse-frequency class weighting. A paired comparison across 18 seed-config pairs gives +8.7 pp sibling and +15.2 pp cousin recall in hyperbolic's favour ($p<10^{-4}$).
2. **The Euclidean "win" on classification is calibration-free.** A logistic-regression baseline on raw CLIP features hits 64.1% top-1 — within noise of the Euclidean prototype's 64.3%. The Euclidean prototype geometry doesn't add classification value over the encoder; it just doesn't subtract any. The hyperbolic prototype geometry, by contrast, is the only model that meaningfully improves over k-NN-on-CLIP for sibling recall.

Global tree fidelity is the only axis where the comparison is fragile: prototype-tree Spearman across the full sweep is not significantly different from zero ($p=0.68$), and the small gap that does appear flips sign between empirical trees built from CLIP and DINOv2 feature centroids respectively. We therefore avoid making global-tree claims and frame the headline narrowly.

Full numbers, plots, and discussion are in [the paper](docs/main.pdf).

## What's in the repo

```
docs/
  main.tex          ← LaTeX source of the paper
  main.pdf          ← compiled paper
  sections/         ← background, problem, data_eda, methods,
                      results, conclusion, broader_impact, appendix
  preamble.tex
  references.bib
  figures/          ← all figures (Poincaré disks, scaling curves,
                      lambda sweep, tree-variant ablation, confusion
                      matrices, kNN grids, headline-results table,
                      DINOv2 cross-encoder check)

scripts/
  models.py                  ← MLP head + EuclideanHead + PoincareHead +
                               ClassifierLayer + Poincaré utilities
  hierarchy.py               ← STYLE_HIERARCHY (lineage, chronological,
                               flat) + tree-distance matrices
  dataset.py                 ← FeatureDataset over cached features
  train.py                   ← training loops with optional
                               hierarchy-aware regularizer
  eval.py                    ← run_evaluation() returns the full
                               metric dict; CLI also dumps per-run
                               artifacts to disk
  sweep.py                   ← grid-search runner with predefined
                               phases; resumable via single sweep.csv
  analysis.py                ← sweep loading, summarize_by_config,
                               best_per_geometry_dim, plot_*,
                               headline_table
  baselines.py               ← logistic-regression / kNN baselines on
                               raw encoder features
  empirical_recall.py        ← sibling/cousin recall against
                               empirical (data-driven) trees
  empirical_tree.py          ← agglomerative trees from class-mean
                               feature vectors
  significance.py            ← paired-t, sign agreement, Wilcoxon
                               across the seed-config grid
  extract_clip_features.py   ← cache CLIP ViT-B/16 features
  extract_dinov2_features.py ← cache DINOv2 features for the
                               cross-encoder check
  dinov2_analysis.py         ← Phase 5 DINOv2 cross-encoder analysis
  make_figures.py            ← regenerate docs/figures/* from sweep.csv

notebooks/
  ms2_data_wrangling.ipynb   ← data EDA
  ms3.ipynb                  ← cross-entropy baseline + pipeline
  ms4_main.ipynb             ← end-to-end reference run, sweep
                               analysis, qualitative figures
  sweep_results.csv          ← 150-row sweep snapshot

apps/
  style_browser/             ← interactive sibling/cousin browser
                               over a trained prototype model
```

## Reproduce the headline numbers

The cached CLIP features are 512-d float16 tensors, so a single training run takes ~10 seconds on an Apple-Silicon GPU. The full sweep finishes in under half an hour.

```bash
# (assuming setup from "Setup" below is done)
python scripts/sweep.py --phase 1 --device mps   # ~14 min, 90 configs
python scripts/sweep.py --phase 2 --device mps   # ~10 min, 48 configs
python scripts/sweep.py --phase 3 --device mps   # ~3 min,  18 configs
python scripts/sweep.py --phase 5 --device mps   # DINOv2 cross-encoder
python scripts/baselines.py                       # logreg, kNN baselines
python scripts/significance.py                    # paired-t, sign tests
python scripts/make_figures.py                    # regenerate docs/figures/*
```

The sweep is resumable via a single `data/runs/sweep.csv`. Each row is one (geometry, dim, curvature, λ, training tree, seed) configuration plus every evaluation metric. If a run dies halfway, re-running picks up where it left off.

For a single-cell reference run, open [`notebooks/ms4_main.ipynb`](notebooks/ms4_main.ipynb) and choose **Restart Kernel + Run All**. Section 4 trains a single Euclidean and a single hyperbolic model from scratch (~1 min) so the executable pipeline is verified; later sections load `notebooks/sweep_results.csv` and produce all the report figures.

## Setup

Requires Python 3.12+, ~25 GB of disk for WikiArt images.

```bash
# 1. Clone
git clone https://github.com/pgrindehollevik-harvard/hyperbolic.git
cd hyperbolic

# 2. Virtual env
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Dependencies (PyTorch, geoopt for Riemannian optimization, OpenCLIP, sklearn, matplotlib)
pip install -r requirements.txt

# 4. Jupyter kernel
python -m ipykernel install --user --name=hyperbolic --display-name="Python (hyperbolic)"

# 5. Dataset (~25 GB images + metadata CSVs)
pip install gdown
gdown "1vTChp3nU5GQeLkPwotrybpUGUXj12BTK" -O data/wikiart.zip
gdown "1uug57zp13wJDwb2nuHOQfR2Odr0hh1a8" -O data/wikiart_csvs.zip
unzip data/wikiart.zip -d data/wikiart
unzip data/wikiart_csvs.zip -d data/wikiart_csvs

# 6. CLIP feature extraction (one-time, ~30 min on a GPU). Outputs:
#    data/features/clip_vitb16.npy  -- (N, 512) float16
#    data/features/index.csv        -- row_idx, path, style_name
python scripts/extract_clip_features.py
```

## How the hierarchy-aware regularizer works

The cross-entropy baseline (and most prior hyperbolic-image work) uses the tree only at evaluation time. Cross-entropy on prototypes only requires that classes be separable; it never asks for the prototype layout to mirror the tree. So we can't fairly blame the geometry for failing on tree metrics if the loss never asked for it.

We add a single-term regularizer that pushes the prototype distance matrix toward the tree distance matrix:

$$\mathcal{L} = \mathcal{L}_{CE} + \lambda \cdot \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}} \left( \frac{d(p_i, p_j)}{\bar{d}} - \frac{T_{ij}}{\bar{T}} \right)^{\!2}$$

Each pair-distance vector is divided by its own mean before the squared difference, so the loss only constrains the *shape* of the prototype distance matrix rather than its absolute scale. The geometry picks the scale that classification likes. $\lambda = 0$ recovers cross-entropy exactly. Implementation: [`scripts/train.py`](scripts/train.py) (`tree_regularizer` function).

## Key references

| Paper | Link |
|---|---|
| Poincaré Embeddings for Learning Hierarchical Representations (Nickel & Kiela, NeurIPS 2017) | [arXiv 1705.08039](https://arxiv.org/abs/1705.08039) |
| Hyperbolic Neural Networks (Ganea et al., NeurIPS 2018) | [arXiv 1805.09112](https://arxiv.org/abs/1805.09112) |
| Representation Tradeoffs for Hyperbolic Embeddings (Sala et al., ICML 2018) | [arXiv 1804.03329](https://arxiv.org/abs/1804.03329) |
| Hyperbolic Image Embeddings (Khrulkov et al., CVPR 2020) | [arXiv 1904.02239](https://arxiv.org/abs/1904.02239) |
| From Trees to Continuous Embeddings and Back (Chami et al., NeurIPS 2020) | [arXiv 2010.00402](https://arxiv.org/abs/2010.00402) |
| Geoopt: Riemannian Optimization in PyTorch (Kochurov et al., GRL+ Workshop 2020) | [arXiv 2005.02819](https://arxiv.org/abs/2005.02819) |
| CLIP: Learning Transferable Visual Models from Natural Language Supervision (Radford et al., ICML 2021) | [arXiv 2103.00020](https://arxiv.org/abs/2103.00020) |
| Improved ArtGAN (Tan et al., IEEE TIP 2019) — original WikiArt-Refined dataset | [arXiv 1708.09533](https://arxiv.org/abs/1708.09533) |

## Authors

**Peter Flo (Grinde-Hollevik)** · [pgrindehollevik@g.harvard.edu](mailto:pgrindehollevik@g.harvard.edu) · [www.pflo.org](https://www.pflo.org)
**Luca Grossmann**
**Valerie Wang**
