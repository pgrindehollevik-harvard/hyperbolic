# Hyperbolic Embeddings for Hierarchical Style Representation

**Peter Flo (Grinde-Hollevik)**  
[pgrindehollevik@g.harvard.edu](mailto:pgrindehollevik@g.harvard.edu) | [www.pflo.org](https://www.pflo.org)

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/pgrindehollevik-harvard/hyperbolic.git
cd hyperbolic

# 2. Create and activate a virtual environment (requires Python 3.12+)
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Register the Jupyter kernel
python -m ipykernel install --user --name=hyperbolic --display-name="Python (hyperbolic)"

# 5. Download the dataset (~25 GB images + metadata CSVs)
pip install gdown
gdown "1vTChp3nU5GQeLkPwotrybpUGUXj12BTK" -O data/wikiart.zip
gdown "1uug57zp13wJDwb2nuHOQfR2Odr0hh1a8" -O data/wikiart_csvs.zip

# 6. Extract
unzip data/wikiart.zip -d data/wikiart
unzip data/wikiart_csvs.zip -d data/wikiart_csvs
```

When running notebooks, select the **"Python (hyperbolic)"** kernel.

## Data

We use the [WikiArt Refined Dataset](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset) (Tan et al., 2019), containing ~81,000 digitized paintings across 27 artistic styles.

## Feature Extraction

Extract CLIP ViT-B/16 features for all images

```bash
python scripts/extract_clip_features.py
```

Outputs (gitignored under `data/`):

- `data/features/clip_vitb16.npy`  — `(N, 512)` float16
- `data/features/index.parquet`    — `row_idx, path, style_name`


## Background and Motivation

Artistic style is hierarchical: broad traditions branch into fine-grained movements and substyles. Standard deep learning models represent images in Euclidean embedding spaces, which do not naturally encode tree-like structure. Hyperbolic geometry, with exponential volume growth and branch-like structure, is theoretically well-suited for representing hierarchical relationships. This project investigates whether hyperbolic embeddings better capture aesthetic hierarchy than Euclidean embeddings.

## Problem

Do hyperbolic embeddings improve hierarchical structure, clustering consistency, and style interpolation compared to Euclidean embeddings? Specifically, we test whether they:

- Better separate fine-grained artistic styles
- More accurately reflect known relationships between art movements
- Provide more stable behavior under feature averaging/interpolation

## Methods

We extract image features using a frozen CNN or CLIP encoder. On top of these features, we train:

- A **Euclidean embedding** baseline
- A **hyperbolic embedding** model in the Poincaré ball

Evaluation includes style classification accuracy, hierarchical clustering quality, tree reconstruction distortion, retrieval consistency, and comparison of Euclidean means vs. hyperbolic Fréchet means for style averaging.

As a 209b extension, we implement hyperbolic models using Riemannian optimization in the Poincaré ball, going beyond standard Euclidean methods covered in class.

## Concerns and Limitations

- Style labels may contain noise or inconsistencies
- Constructing a clean style hierarchy may require manual curation
- Hyperbolic training requires numerical stabilization
- The dataset consists of public artwork and poses no privacy concerns
