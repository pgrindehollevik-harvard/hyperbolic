"""
Style-browser demo: simulate Midjourney-style iterative style selection
using WikiArt images, comparing Euclidean and hyperbolic embeddings
side-by-side.

Workflow per geometry, independent in each column:
  1. Show 16 candidate images (initially random; later, k-NN to centroid).
  2. User clicks images to like / un-like them.
  3. User clicks "Refine" → compute Fréchet mean of liked embeddings,
     retrieve 16 nearest neighbors in that geometry, display.
  4. Repeat. The two columns drift apart over rounds because the geometry
     determines both the centroid (Fréchet mean) and the neighborhood.

Live metrics per column (top of column):
  - round, liked count, centroid drift since last refine, batch entropy.
  - top-3 nearest prototypes to the centroid (what does the geometry
    think you want?).
  - boundary distance for the hyperbolic disk (how close is the centroid
    to the radius-1/sqrt(c) shell where Poincaré distances diverge?).

Run from the repo root:
  streamlit run apps/style_browser/app.py
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analysis import _embed_split  # noqa: E402
from eval import frechet_mean as _frechet_mean  # noqa: E402
from eval import load_split_metadata  # noqa: E402
from hierarchy import load_style_classes  # noqa: E402
from models import poincare_distance  # noqa: E402

try:
    from st_clickable_images import clickable_images
except ImportError as exc:
    st.error(
        "st-clickable-images is required for click-to-like. "
        "Install with: pip install st-clickable-images"
    )
    raise


SWEEP_CSV_CANDIDATES = [
    REPO_ROOT / "data" / "runs" / "sweep.csv",
    REPO_ROOT / "notebooks" / "sweep_results.csv",
]
WIKIART_DIR_CANDIDATES = [
    REPO_ROOT / "data" / "wikiart" / "wikiart",
    REPO_ROOT / "data" / "wikiart",
]

THUMB_PX = 160
LIKED_BORDER = "#ef4444"  # red ring on liked tiles
GRID_COLS = 4
GRID_ROWS = 4
K = GRID_COLS * GRID_ROWS  # 16


def _resolve(paths: list[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"none of {paths} exists")


# ----------------------------------------------------------------------------
# Caching: data + checkpoints + thumbnails
# ----------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading sweep results…")
def _load_sweep() -> pd.DataFrame:
    return pd.read_csv(_resolve(SWEEP_CSV_CANDIDATES))


@st.cache_resource(show_spinner="Loading checkpoint and embeddings…")
def load_geometry(geometry: str, dim: int) -> dict:
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
        raise RuntimeError(
            f"path/embedding misalignment: {len(paths)} vs {len(embeddings)}"
        )
    return {
        "embeddings": torch.from_numpy(embeddings),
        "labels": labels,
        "prototypes": torch.from_numpy(prototypes),
        "meta": meta,
        "paths": paths,
    }


@st.cache_data(show_spinner=False, max_entries=4096)
def _thumb_data_uri(path_str: str, liked: bool) -> str:
    """Render a thumbnail as a base64 data URI; draws a red ring if liked."""
    wikiart_dir = _resolve(WIKIART_DIR_CANDIDATES)
    img = Image.open(wikiart_dir / path_str).convert("RGB")
    img.thumbnail((THUMB_PX, THUMB_PX))
    if liked:
        draw = ImageDraw.Draw(img)
        w, h = img.size
        for i in range(6):
            draw.rectangle([i, i, w - 1 - i, h - 1 - i], outline=LIKED_BORDER)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


# ----------------------------------------------------------------------------
# Geometry primitives (thin wrappers around eval.py / models.py)
# ----------------------------------------------------------------------------

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


def nearest_neighbors(
    centroid: torch.Tensor,
    embeddings: torch.Tensor,
    geometry: str,
    curvature: float,
    k: int,
    exclude: set[int],
) -> list[int]:
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


def centroid_distance(
    a: torch.Tensor, b: torch.Tensor, geometry: str, curvature: float
) -> float:
    if geometry == "euclidean":
        return float((a - b).norm().item())
    d = poincare_distance(a.view(1, 1, -1), b.view(1, 1, -1), curvature=curvature)
    return float(d.item())


def nearest_prototypes(
    centroid: torch.Tensor,
    prototypes: torch.Tensor,
    geometry: str,
    curvature: float,
    k: int = 3,
) -> list[tuple[int, float]]:
    if geometry == "euclidean":
        dists = torch.cdist(centroid.unsqueeze(0), prototypes).squeeze(0)
    else:
        dists = poincare_distance(
            centroid.view(1, 1, -1),
            prototypes.unsqueeze(0),
            curvature=curvature,
        ).view(-1)
    top = dists.topk(k, largest=False)
    return list(zip(top.indices.tolist(), top.values.tolist()))


def boundary_distance(centroid: torch.Tensor, curvature: float) -> float:
    radius = 1.0 / float(curvature) ** 0.5
    return radius - float(centroid.norm().item())


def style_entropy(labels: np.ndarray, indices: list[int]) -> float:
    """Normalized Shannon entropy of style distribution in the batch (0–1)."""
    sub = labels[indices]
    _, counts = np.unique(sub, return_counts=True)
    p = counts / counts.sum()
    entropy = -np.sum(p * np.log2(p))
    max_e = np.log2(min(len(indices), 27))
    return float(entropy / max_e) if max_e > 0 else 0.0


# ----------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------

def init_state(num_total: int, k: int, seed: int) -> None:
    if st.session_state.get("initialized"):
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
            "history_eu": [],
            "history_hy": [],
            "round_eu": 0,
            "round_hy": 0,
            "drift_eu": 0.0,
            "drift_hy": 0.0,
        }
    )


def reset_state() -> None:
    for k in list(st.session_state.keys()):
        if k.startswith(
            (
                "initialized",
                "initial_batch",
                "candidates_",
                "liked_",
                "history_",
                "round_",
                "drift_",
                "last_click_",
            )
        ):
            del st.session_state[k]


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

def render_metrics_header(
    column,
    geo_name: str,
    geometry: str,
    curvature: float,
    bundle: dict,
    candidates: list[int],
    liked: set[int],
) -> None:
    round_num = st.session_state[f"round_{geo_name}"]
    drift = st.session_state.get(f"drift_{geo_name}", 0.0)
    diversity = style_entropy(bundle["labels"], candidates)

    m1, m2, m3, m4 = column.columns(4)
    m1.metric("round", round_num)
    m2.metric("liked", len(liked))
    drift_label = "centroid Δ" + ("" if geometry == "euclidean" else " (hyp.)")
    m3.metric(drift_label, f"{drift:.3f}" if round_num > 0 else "—")
    m4.metric("batch entropy", f"{diversity:.2f}")


def render_centroid_inspector(
    column,
    geo_name: str,
    geometry: str,
    curvature: float,
    bundle: dict,
    liked: set[int],
    style_names: list[str],
) -> None:
    if not liked:
        column.caption("Like a few images and click *Refine* to see what the centroid looks like.")
        return
    centroid = compute_centroid(sorted(liked), bundle["embeddings"], geometry, curvature)
    nearest = nearest_prototypes(
        centroid, bundle["prototypes"], geometry, curvature, k=3
    )
    rows = [
        {"style": style_names[i], "distance": f"{d:.3f}"} for i, d in nearest
    ]
    extras = []
    if geometry == "hyperbolic":
        bd = boundary_distance(centroid, curvature)
        extras.append(f"dist to disk boundary: **{bd:.3f}** (radius = {1/curvature**0.5:.3f})")
    column.markdown(
        "**centroid → nearest 3 prototypes**  \n"
        + ("  •  ".join(extras) if extras else "")
    )
    column.dataframe(
        pd.DataFrame(rows), use_container_width=True, hide_index=True
    )


def render_grid(
    column,
    geo_name: str,
    geo_label: str,
    candidates: list[int],
    paths: list[str],
    labels: np.ndarray,
    style_names: list[str],
    liked_key: str,
) -> bool:
    """Render the 4x4 image grid. Returns True if a click changed state
    (and the caller should st.rerun)."""
    column.markdown(f"### {geo_label}")
    liked = st.session_state[liked_key]

    changed = False
    round_num = st.session_state[f"round_{geo_name}"]

    for r in range(GRID_ROWS):
        row_idx = candidates[r * GRID_COLS : (r + 1) * GRID_COLS]
        if not row_idx:
            continue
        with column:
            row_imgs = [
                _thumb_data_uri(paths[idx], idx in liked) for idx in row_idx
            ]
            row_titles = [style_names[int(labels[idx])] for idx in row_idx]
            click_key = f"click_{geo_name}_r{r}_round{round_num}"
            clicked = clickable_images(
                row_imgs,
                titles=row_titles,
                div_style={
                    "display": "flex",
                    "justify-content": "space-between",
                    "gap": "6px",
                },
                img_style={
                    "height": f"{THUMB_PX}px",
                    "width": f"{THUMB_PX}px",
                    "object-fit": "cover",
                    "cursor": "pointer",
                    "border-radius": "8px",
                },
                key=click_key,
            )
            last_seen_key = f"last_seen_{click_key}"
            if clicked != -1 and clicked != st.session_state.get(last_seen_key, -2):
                st.session_state[last_seen_key] = clicked
                actual_idx = row_idx[clicked]
                if actual_idx in liked:
                    liked.discard(actual_idx)
                else:
                    liked.add(actual_idx)
                changed = True

            cap_cols = column.columns(GRID_COLS)
            for c, ccol in enumerate(cap_cols):
                if c < len(row_titles):
                    ccol.caption(row_titles[c])
    return changed


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
    drift_key = f"drift_{geo_name}"

    liked = sorted(st.session_state[liked_key])
    if not liked:
        st.warning(f"no images liked yet on the {geo_name} side")
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

    history = st.session_state[hist_key]
    if history:
        st.session_state[drift_key] = centroid_distance(
            centroid, history[-1]["centroid"], geometry, curvature
        )
    else:
        st.session_state[drift_key] = 0.0

    history.append(
        {
            "round": st.session_state[round_key] + 1,
            "centroid": centroid.detach().clone(),
        }
    )
    st.session_state[round_key] += 1
    st.session_state[cand_key] = new_candidates


def render_drift_panel(cur_eu: float, cur_hy: float) -> None:
    hist_eu = st.session_state.get("history_eu", [])
    hist_hy = st.session_state.get("history_hy", [])
    if not hist_eu and not hist_hy:
        return
    st.markdown("---")
    st.markdown("### Centroid drift trajectory")
    st.caption(
        "Distance between successive centroids per refine step. Note: "
        "Euclidean uses L2; hyperbolic uses Poincaré distance with "
        "curvature c — the units are not directly comparable across "
        "columns, but the trajectory shape within each column is."
    )

    rows = []
    for hist, name, geo, c in [
        (hist_eu, "euclidean", "euclidean", cur_eu),
        (hist_hy, "hyperbolic", "hyperbolic", cur_hy),
    ]:
        for i in range(1, len(hist)):
            d = centroid_distance(
                hist[i]["centroid"], hist[i - 1]["centroid"], geo, c
            )
            rows.append(
                {"geometry": name, "round": hist[i]["round"], "distance_moved": d}
            )

    if not rows:
        st.caption("Refine more than once to see drift.")
        return
    df = pd.DataFrame(rows)
    try:
        import altair as alt

        chart = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(
                x=alt.X("round:O", title="refinement round"),
                y=alt.Y("distance_moved:Q", title="distance moved"),
                color=alt.Color("geometry:N"),
            )
            .properties(height=240)
        )
        st.altair_chart(chart, use_container_width=True)
    except ImportError:
        st.dataframe(df, use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Style Browser — Euclidean vs Hyperbolic",
        layout="wide",
    )
    st.markdown(
        "# Style browser: Euclidean vs hyperbolic embeddings\n"
        "An interactive simulation of iterative style selection (à la "
        "Midjourney's `--sref` workflow), using WikiArt paintings and "
        "our two trained prototype models. Click images to like / un-like; "
        "click **Refine** to retrieve the 16 nearest neighbors of the "
        "geometry-native centroid of your liked set. The two columns drift "
        "apart over rounds — that's the artifact."
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

        if st.button("Reset (new random batch)", type="primary"):
            reset_state()
            st.rerun()

        st.markdown("---")
        st.markdown("### About")
        st.caption(
            "Hyperbolic uses the best curvature for the chosen d, picked "
            "from Phase-1 sweep top-1. Centroid is the Fréchet mean in "
            "each geometry."
        )
        st.caption(
            "Liked images get a red ring. Style names below each tile are "
            "for context; the model classifies on those labels but you "
            "click on visual style."
        )

    bundle_eu = load_geometry("euclidean", dim)
    bundle_hy = load_geometry("hyperbolic", dim)
    num_total = len(bundle_eu["paths"])

    init_state(num_total=num_total, k=K, seed=int(seed))

    style_names = load_style_classes()
    cur_eu = 1.0
    cur_hy = float(bundle_hy["meta"]["curvature"])

    col_eu, col_hy = st.columns(2, gap="medium")

    render_metrics_header(
        col_eu, "eu", "euclidean", cur_eu, bundle_eu,
        st.session_state["candidates_eu"], st.session_state["liked_eu"],
    )
    render_metrics_header(
        col_hy, "hy", "hyperbolic", cur_hy, bundle_hy,
        st.session_state["candidates_hy"], st.session_state["liked_hy"],
    )

    rerun_eu = render_grid(
        column=col_eu,
        geo_name="eu",
        geo_label=f"Euclidean · d={dim}",
        candidates=st.session_state["candidates_eu"],
        paths=bundle_eu["paths"],
        labels=bundle_eu["labels"],
        style_names=style_names,
        liked_key="liked_eu",
    )
    rerun_hy = render_grid(
        column=col_hy,
        geo_name="hy",
        geo_label=f"Hyperbolic · d={dim}, c={cur_hy:g}",
        candidates=st.session_state["candidates_hy"],
        paths=bundle_hy["paths"],
        labels=bundle_hy["labels"],
        style_names=style_names,
        liked_key="liked_hy",
    )

    refine_eu_col, refine_hy_col = st.columns(2, gap="medium")
    if refine_eu_col.button(
        "Refine euclidean", type="primary", use_container_width=True
    ):
        refine("eu", "euclidean", cur_eu, bundle_eu, k=K)
        rerun_eu = True
    if refine_hy_col.button(
        "Refine hyperbolic", type="primary", use_container_width=True
    ):
        refine("hy", "hyperbolic", cur_hy, bundle_hy, k=K)
        rerun_hy = True

    insp_eu, insp_hy = st.columns(2, gap="medium")
    render_centroid_inspector(
        insp_eu, "eu", "euclidean", cur_eu, bundle_eu,
        st.session_state["liked_eu"], style_names,
    )
    render_centroid_inspector(
        insp_hy, "hy", "hyperbolic", cur_hy, bundle_hy,
        st.session_state["liked_hy"], style_names,
    )

    render_drift_panel(cur_eu, cur_hy)

    if rerun_eu or rerun_hy:
        st.rerun()


if __name__ == "__main__":
    main()
