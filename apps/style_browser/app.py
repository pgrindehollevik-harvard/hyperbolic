"""
Style-browser demo: simulate Midjourney-style iterative style selection
using WikiArt images, comparing Euclidean and hyperbolic embeddings
side-by-side.

Workflow per geometry, independent in each column:
  1. Show 16 candidate images (initially random; later, k-NN to centroid).
  2. User clicks the "Like" checkbox under images they like.
  3. User clicks "Refine" → compute Fréchet mean of liked embeddings,
     retrieve 16 nearest neighbors in that geometry, display.
  4. Repeat. The two columns drift apart over rounds because the geometry
     determines both the centroid (Fréchet mean) and the neighborhood.

Run from the repo root:
  streamlit run apps/style_browser/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analysis import _embed_split  # noqa: E402
from eval import frechet_mean as _frechet_mean  # noqa: E402
from eval import load_split_metadata  # noqa: E402
from hierarchy import load_style_classes  # noqa: E402
from models import poincare_distance  # noqa: E402


SWEEP_CSV_CANDIDATES = [
    REPO_ROOT / "data" / "runs" / "sweep.csv",
    REPO_ROOT / "notebooks" / "sweep_results.csv",
]
WIKIART_DIR_CANDIDATES = [
    REPO_ROOT / "data" / "wikiart" / "wikiart",
    REPO_ROOT / "data" / "wikiart",
]


def _resolve(paths: list[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"none of {paths} exists")


@st.cache_resource(show_spinner="Loading sweep results…")
def _load_sweep() -> pd.DataFrame:
    return pd.read_csv(_resolve(SWEEP_CSV_CANDIDATES))


@st.cache_resource(show_spinner="Loading checkpoint and embeddings…")
def load_geometry(geometry: str, dim: int) -> dict:
    """Return embeddings, labels, prototypes, metadata, paths for the best
    Phase-1 checkpoint at (geometry, dim)."""
    sweep = _load_sweep()
    df = sweep[
        (sweep["geometry"] == geometry)
        & (sweep["dim"] == dim)
        & (sweep["tree_loss_weight"] == 0.0)
        & (sweep["tree_hierarchy"] == "default")
    ]
    if df.empty:
        raise RuntimeError(
            f"no checkpoint found for {geometry} d={dim}; run scripts/sweep.py --phase 1"
        )
    best = df.loc[df["top1_accuracy"].idxmax()]
    ckpt = REPO_ROOT / "data" / "runs" / "sweep" / best["config_hash"] / "ckpt.pt"
    if not ckpt.exists():
        raise RuntimeError(f"checkpoint missing on disk: {ckpt}")

    embeddings, labels, prototypes, meta = _embed_split(ckpt, max_samples=None)
    split_df = load_split_metadata("val")
    paths = split_df["path"].tolist()
    if len(paths) != len(embeddings):
        # _embed_split iterates FeatureDataset in order → same order as split_df
        # after dropna. If lengths disagree, we have a bug.
        raise RuntimeError(
            f"path/embedding misalignment: {len(paths)} vs {len(embeddings)}"
        )
    return {
        "embeddings": torch.from_numpy(embeddings),
        "labels": labels,
        "prototypes": prototypes,
        "meta": meta,
        "paths": paths,
    }


def nearest_neighbors(
    centroid: torch.Tensor,
    embeddings: torch.Tensor,
    geometry: str,
    curvature: float,
    k: int,
    exclude: set[int],
) -> list[int]:
    """Indices of the k closest val embeddings to `centroid`, skipping any in
    `exclude` (so already-liked images don't pollute the next batch)."""
    if geometry == "euclidean":
        dists = torch.cdist(centroid.unsqueeze(0), embeddings).squeeze(0)
    else:
        dists = poincare_distance(
            centroid.view(1, 1, -1),
            embeddings.unsqueeze(0),
            curvature=curvature,
        ).view(-1)
    if exclude:
        dists = dists.clone()
        dists[list(exclude)] = float("inf")
    return dists.topk(k, largest=False).indices.tolist()


def compute_centroid(
    indices: list[int],
    embeddings: torch.Tensor,
    geometry: str,
    curvature: float,
) -> torch.Tensor:
    points = embeddings[indices]
    if len(indices) == 1:
        return points[0]
    return _frechet_mean(points, geometry=geometry, curvature=curvature)


def centroid_distance(
    a: torch.Tensor,
    b: torch.Tensor,
    geometry: str,
    curvature: float,
) -> float:
    if geometry == "euclidean":
        return float((a - b).norm().item())
    d = poincare_distance(
        a.view(1, 1, -1), b.view(1, 1, -1), curvature=curvature
    )
    return float(d.item())


@st.cache_data(show_spinner=False)
def load_thumbnail(path_str: str) -> Image.Image:
    wikiart_dir = _resolve(WIKIART_DIR_CANDIDATES)
    img = Image.open(wikiart_dir / path_str).convert("RGB")
    img.thumbnail((220, 220))
    return img


def init_state(num_total: int, k: int, seed: int) -> None:
    if "initialized" in st.session_state and st.session_state["initialized"]:
        return
    rng = np.random.default_rng(seed)
    initial = rng.choice(num_total, size=k, replace=False).tolist()
    st.session_state.update(
        {
            "initialized": True,
            "initial_batch": initial,
            "candidates_eu": list(initial),
            "candidates_hy": list(initial),
            "liked_eu": set(),
            "liked_hy": set(),
            "history_eu": [],  # list of (round, centroid_tensor)
            "history_hy": [],
            "round_eu": 0,
            "round_hy": 0,
        }
    )


def reset_state() -> None:
    st.session_state["initialized"] = False


def render_grid(
    column,
    geo_name: str,
    geo_label: str,
    candidates: list[int],
    paths: list[str],
    labels: np.ndarray,
    style_names: list[str],
    state_liked_key: str,
) -> None:
    column.markdown(f"### {geo_label}")
    column.caption(
        f"round {st.session_state['round_' + geo_name]} · "
        f"{len(st.session_state[state_liked_key])} liked"
    )

    rows = 4
    cols_per_row = 4
    for r in range(rows):
        cells = column.columns(cols_per_row)
        for c, cell in enumerate(cells):
            i = r * cols_per_row + c
            if i >= len(candidates):
                continue
            idx = candidates[i]
            with cell:
                try:
                    st.image(
                        load_thumbnail(paths[idx]),
                        use_container_width=True,
                    )
                except Exception:
                    st.write(f"(missing: {paths[idx][:30]})")
                    continue
                cb_key = f"like_{geo_name}_{idx}_{st.session_state['round_' + geo_name]}"
                checked = idx in st.session_state[state_liked_key]
                new = st.checkbox(
                    style_names[int(labels[idx])][:14],
                    value=checked,
                    key=cb_key,
                )
                if new and idx not in st.session_state[state_liked_key]:
                    st.session_state[state_liked_key].add(idx)
                elif (not new) and idx in st.session_state[state_liked_key]:
                    st.session_state[state_liked_key].discard(idx)


def refine(
    geo_name: str,
    geometry: str,
    curvature: float,
    bundle: dict,
    k: int,
) -> None:
    liked_key = f"liked_{geo_name}"
    cand_key = f"candidates_{geo_name}"
    hist_key = f"history_{geo_name}"
    round_key = f"round_{geo_name}"

    liked = sorted(st.session_state[liked_key])
    if not liked:
        st.warning(f"no images liked yet in {geo_name}")
        return
    centroid = compute_centroid(liked, bundle["embeddings"], geometry, curvature)
    new_candidates = nearest_neighbors(
        centroid,
        bundle["embeddings"],
        geometry,
        curvature,
        k=k,
        exclude=set(liked),
    )
    st.session_state[round_key] += 1
    st.session_state[hist_key].append(
        {"round": st.session_state[round_key], "centroid": centroid.detach().clone()}
    )
    st.session_state[cand_key] = new_candidates


def render_drift_panel(
    bundle_eu: dict,
    bundle_hy: dict,
    curvature_hy: float,
) -> None:
    hist_eu = st.session_state.get("history_eu", [])
    hist_hy = st.session_state.get("history_hy", [])
    if not hist_eu and not hist_hy:
        return
    st.markdown("---")
    st.markdown("### Centroid drift per round")

    rows = []
    for hist, name, geo, c in [
        (hist_eu, "euclidean", "euclidean", 1.0),
        (hist_hy, "hyperbolic", "hyperbolic", curvature_hy),
    ]:
        for i, snap in enumerate(hist):
            if i == 0:
                continue
            d = centroid_distance(
                hist[i]["centroid"], hist[i - 1]["centroid"], geo, c
            )
            rows.append({"geometry": name, "round": snap["round"], "distance_moved": d})

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        try:
            import altair as alt

            chart = (
                alt.Chart(df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("round:O", title="refinement round"),
                    y=alt.Y("distance_moved:Q", title="distance moved (geometry-native)"),
                    color=alt.Color("geometry:N"),
                )
                .properties(height=240)
            )
            st.altair_chart(chart, use_container_width=True)
        except ImportError:
            pass


def main() -> None:
    st.set_page_config(
        page_title="Style Browser — Euclidean vs Hyperbolic",
        layout="wide",
    )
    st.markdown(
        "# Style browser: Euclidean vs hyperbolic embeddings\n"
        "An interactive simulation of iterative style selection (à la "
        "Midjourney's style-references workflow), using WikiArt paintings "
        "and our two trained prototype models. Independently like images "
        "in each column; click **Refine** to retrieve the 16 nearest "
        "neighbors of the geometry-native centroid of your liked set."
    )

    with st.sidebar:
        st.markdown("### Setup")
        dim = st.selectbox(
            "embedding dim",
            options=[8, 16, 32, 64],
            index=1,
            help="Higher d emphasizes the local-retrieval gap from Phase 1.",
        )
        seed = st.number_input("initial-batch seed", value=42, step=1)
        k = 16
        st.caption("Grid is 4×4 (k=16) per column.")

        if st.button("Reset (new random batch)", type="primary"):
            reset_state()
            st.rerun()

        st.markdown("---")
        st.markdown("### About")
        st.caption(
            "Hyperbolic uses the best curvature for the chosen d, picked from "
            "Phase-1 sweep top-1. Centroid is the Fréchet mean in each geometry."
        )

    bundle_eu = load_geometry("euclidean", dim)
    bundle_hy = load_geometry("hyperbolic", dim)
    num_total = len(bundle_eu["paths"])

    init_state(num_total=num_total, k=k, seed=int(seed))

    style_names = load_style_classes()
    cur_eu = 1.0
    cur_hy = float(bundle_hy["meta"]["curvature"])

    col_eu, col_hy = st.columns(2, gap="medium")

    render_grid(
        column=col_eu,
        geo_name="eu",
        geo_label=f"Euclidean · d={dim}",
        candidates=st.session_state["candidates_eu"],
        paths=bundle_eu["paths"],
        labels=bundle_eu["labels"],
        style_names=style_names,
        state_liked_key="liked_eu",
    )
    render_grid(
        column=col_hy,
        geo_name="hy",
        geo_label=f"Hyperbolic · d={dim}, c={cur_hy:g}",
        candidates=st.session_state["candidates_hy"],
        paths=bundle_hy["paths"],
        labels=bundle_hy["labels"],
        style_names=style_names,
        state_liked_key="liked_hy",
    )

    refine_eu_col, refine_hy_col = st.columns(2, gap="medium")
    if refine_eu_col.button("Refine euclidean", type="primary", use_container_width=True):
        refine("eu", "euclidean", cur_eu, bundle_eu, k=k)
        st.rerun()
    if refine_hy_col.button("Refine hyperbolic", type="primary", use_container_width=True):
        refine("hy", "hyperbolic", cur_hy, bundle_hy, k=k)
        st.rerun()

    render_drift_panel(bundle_eu, bundle_hy, cur_hy)


if __name__ == "__main__":
    main()
