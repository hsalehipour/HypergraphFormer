#!/usr/bin/env python3
"""Visualize DiffPlanner floor plans (ground truth and/or predictions).

This is the benchmark-side counterpart to ``benchmark/utils/visualize_*``
helpers for other generators. It is **model-agnostic** in the sense that
it just reads any DiffPlanner-shaped record (the same JSON shape the
``DiffPlanner/output/*`` cascade emits) and renders it. Typical inputs:

* a monolithic ``data_test.json`` (a list of ground-truth records) -- as
  ``--gt-json`` and/or ``--predicted``;
* a benchmark cache pickle written by
  ``benchmark/src/evaluate_diffplanner.py`` (``--save_generations``) --
  as ``--predicted``;
* a directory of per-sample ``<name>.json`` files emitted by any other
  DiffPlanner-format pipeline -- as ``--predicted``.

For each sample the script writes ``vis/<name>.png`` (a side-by-side
panel of GT floor plan, predicted floor plan, GT bubble, predicted
bubble), an aggregate ``vis/summary.png`` and a multi-page
``vis/visualizations.pdf`` containing the summary followed by every
per-sample panel.

If you also pass ``--metrics-npz`` (the ``results.npz`` written by
:func:`utils.report_results.report_results`) the script overlays the
per-sample numbers in each panel header so the figure is self-contained
for inclusion in reports/papers.

Color palette and rendering primitives mirror
``DiffPlanner/output/visualization.py`` so panels are visually
consistent with the paper.

Examples
--------
    # Render the cache produced by the cascade benchmark, with metrics
    # overlaid from the run's NPZ:

    python -m benchmark.utils.visualize_diffplanner \\
        --predicted cache/dp_bnc_test_full.pkl \\
        --gt-json dataset/rplan/dataset_diffplanner/data_test.json \\
        --metrics-npz benchmark/results_dp_full/results.npz \\
        --out-dir benchmark/results_dp_full/vis \\
        --limit 50

    # Render only ground truth from data_test.json:
    python -m benchmark.utils.visualize_diffplanner \\
        --predicted dataset/rplan/dataset_diffplanner/data_test.json \\
        --out-dir /tmp/dp_gt_vis --limit 20
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("vis-dp")


# =====================================================================
# Color palette (matches DiffPlanner/output/visualization.py exactly)
# =====================================================================

COLOR_MAP = np.array([
    [244, 242, 229],     # 0 living room
    [251, 154, 153],     # 1 bedroom
    [178, 223, 128],     # 2 kitchen
    [166, 206, 227],     # 3 bathroom
    [253, 191, 111],     # 4 balcony
    [202, 178, 214],     # 5 storage
    [57,  67,  94],      # 6 exterior wall
    [217, 221, 229],     # 7 entrance
    [255, 255, 255],     # 8 interior wall
    [255, 255, 255],     # 9 external (background)
], dtype=np.int64)

CATEGORY_NAMES = {
    0: "LivingRoom",
    1: "Bedroom",
    2: "Kitchen",
    3: "Bathroom",
    4: "Balcony",
    5: "Storage",
}

CANVAS_SIZE_RENDER = 1024  # internal raster size; matplotlib downsamples

# Stroke widths in source-pixel units (BEFORE the canvas_size_k upscale).
# Decoupled from canvas size so partition walls stay readable on hi-dpi
# rasters without obscuring small post-processing gaps/overlaps.
INTERIOR_WALL_PX = 1
BOUNDARY_WALL_PX = 3
DOOR_PX = 4
BUBBLE_EDGE_PX = 1


# =====================================================================
# Raster drawing primitives
# =====================================================================

def _to_int_pairs(pairs: list) -> np.ndarray:
    return np.array([[int(p[0]), int(p[1])] for p in pairs], dtype=np.int32)


def _room_box(room: dict) -> Optional[tuple[int, int, int, int]]:
    """Return ``(x0, y0, x1, y1)`` from a DiffPlanner room dict.

    Prefers ``box_aligned`` (post-processed) and falls back to ``box``
    (raw cascade output / non-aligned LLM output).
    """
    box = room.get("box_aligned") or room.get("box")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x0, y0, x1, y1 = (int(round(float(v))) for v in box)
    except (TypeError, ValueError):
        return None
    if not (x0 < x1 and y0 < y1):
        return None
    return x0, y0, x1, y1


def _room_polygon(room: dict) -> Optional[np.ndarray]:
    """Return room outline as Nx2 int array.

    Prefers the precise post-processed polygon (``r_boundary_aligned``);
    falls back to the bounding box if that field is missing or
    degenerate (this happens for raw / LLM output and for rooms that
    post-processing dropped).
    """
    coords = room.get("r_boundary_aligned")
    if isinstance(coords, list) and len(coords) >= 3:
        try:
            return _to_int_pairs(coords)
        except (TypeError, ValueError, IndexError):
            pass
    box = _room_box(room)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32)


def draw_floorplan(
    boundary_pairs: list[list[int]],
    rooms: list[dict],
    *,
    canvas_size: int = CANVAS_SIZE_RENDER,
    door_segment: Optional[list[list[int]]] = None,
) -> np.ndarray:
    """Render a DiffPlanner floor plan to an RGB image.

    Mirrors :func:`DiffPlanner.output.visualization.draw_floorplan` but
    with thinner interior walls so post-processing gaps/overlaps stay
    visible on high-dpi rasters, and with optional precise polygons for
    rooms whose ``r_boundary_aligned`` is available.
    """
    canvas_size_k = max(1, canvas_size // 256)
    inwall_thick = max(1, INTERIOR_WALL_PX)
    boundary_thick = max(inwall_thick, BOUNDARY_WALL_PX)
    door_thick = max(boundary_thick, DOOR_PX)

    img = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
    img[:, :] = COLOR_MAP[9].tolist()

    if not boundary_pairs:
        return img

    boundary = _to_int_pairs(boundary_pairs) * canvas_size_k
    boundary_closed = np.concatenate([boundary, boundary[:1]])

    # Living-room base color fills the whole interior; rooms are painted
    # on top, then anything outside the boundary is wiped back to white.
    inside_mask_3c = np.zeros_like(img)
    cv2.fillPoly(inside_mask_3c, boundary_closed.reshape(1, -1, 2), [1, 1, 1])
    cv2.fillPoly(img, boundary_closed.reshape(1, -1, 2), COLOR_MAP[0].tolist())

    inwall_color = COLOR_MAP[8].tolist()
    for r in rooms:
        cat = int(r.get("category", -1))
        # Skip living rooms -- the base fill already paints them and the
        # post-processing convention is that living-room geometry "fills
        # whatever is left". Painting it again with box_aligned would
        # cover other rooms.
        if cat == 0:
            continue
        poly = _room_polygon(r)
        if poly is None:
            continue
        coords = poly * canvas_size_k
        room_color = COLOR_MAP[cat % len(COLOR_MAP)].tolist()
        cv2.fillPoly(img, coords.reshape(1, -1, 2), color=room_color)
        cv2.polylines(img, coords.reshape(1, -1, 2), isClosed=True,
                      color=inwall_color, thickness=inwall_thick,
                      lineType=cv2.LINE_AA)

    img[inside_mask_3c == 0] = 255

    boundary_color = COLOR_MAP[6].tolist()
    cv2.polylines(img, boundary_closed.reshape(1, -1, 2), isClosed=True,
                  color=boundary_color, thickness=boundary_thick,
                  lineType=cv2.LINE_AA)

    if door_segment is not None and len(door_segment) >= 2:
        seg = _to_int_pairs(door_segment[:2]) * canvas_size_k
        ent_color = COLOR_MAP[7].tolist()
        cv2.line(img, tuple(seg[0]), tuple(seg[1]),
                 ent_color, door_thick, lineType=cv2.LINE_AA)

    return img


def draw_bubble(
    boundary_pairs: list[list[int]],
    rooms: list[dict],
    adjacencies: list[list[int]],
    *,
    canvas_size: int = CANVAS_SIZE_RENDER,
) -> np.ndarray:
    """Render a DiffPlanner bubble diagram (``draw_bubble`` port)."""
    canvas_size_k = max(1, canvas_size // 256)
    boundary_thick = max(1, BOUNDARY_WALL_PX)
    edge_thick = max(1, BUBBLE_EDGE_PX)

    img = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
    img[:, :] = COLOR_MAP[9].tolist()

    if boundary_pairs:
        boundary = _to_int_pairs(boundary_pairs) * canvas_size_k
        boundary_closed = np.concatenate([boundary, boundary[:1]])
        cv2.polylines(img, boundary_closed.reshape(1, -1, 2), isClosed=True,
                      color=COLOR_MAP[6].tolist(), thickness=boundary_thick,
                      lineType=cv2.LINE_AA)

    if not rooms:
        return img

    # Adjacency edges first (under the bubbles).
    if adjacencies:
        id_to_loc = {}
        for r in rooms:
            loc = r.get("location_aligned") or r.get("location")
            box = _room_box(r)
            if isinstance(loc, (list, tuple)) and len(loc) == 2:
                cx, cy = int(loc[0]), int(loc[1])
            elif box is not None:
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)
            else:
                continue
            id_to_loc[int(r.get("id", -1))] = (cx * canvas_size_k, cy * canvas_size_k)

        line_color = COLOR_MAP[7].tolist()
        for con in adjacencies:
            if not isinstance(con, (list, tuple)) or len(con) < 2:
                continue
            a, b = int(con[0]), int(con[1])
            if a in id_to_loc and b in id_to_loc:
                cv2.line(img, id_to_loc[a], id_to_loc[b],
                         line_color, edge_thick, lineType=cv2.LINE_AA)

    # Bubbles. Radius = sqrt(area / 8pi) -- matches DiffPlanner exactly.
    for r in rooms:
        cat = int(r.get("category", -1))
        loc = r.get("location_aligned") or r.get("location")
        size = r.get("size_aligned") or r.get("size")
        box = _room_box(r)
        if isinstance(loc, (list, tuple)) and len(loc) == 2:
            cx, cy = int(loc[0]), int(loc[1])
        elif box is not None:
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
        else:
            continue
        if not isinstance(size, (int, float)) or size <= 0:
            if box is not None:
                size = max(1, (box[2] - box[0]) * (box[3] - box[1]))
            else:
                size = 256
        radius = int(math.sqrt(float(size) * canvas_size_k * canvas_size_k / (math.pi * 8)))
        radius = max(radius, max(2, boundary_thick))
        color = COLOR_MAP[cat % len(COLOR_MAP)].tolist()
        cv2.circle(img, (cx * canvas_size_k, cy * canvas_size_k),
                   radius, color, -1, lineType=cv2.LINE_AA)

    return img


# =====================================================================
# Loaders
# =====================================================================

def _index_records(records: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in records:
        if not isinstance(r, dict):
            continue
        name = str(r.get("name") or "")
        if not name:
            continue
        out[name] = r
    return out


def load_records(source: Path) -> dict[str, dict]:
    """Load DiffPlanner records keyed by ``name`` from any supported source.

    Supported:
      * ``.json`` -- either a monolithic list (``data_test.json``) or a
        single-record dict.
      * ``.pkl`` -- benchmark cache: ``{"metadata": ..., "samples": [...]}``
        OR a bare list of records. Records are cached cascade outputs and
        already have post-processed fields.
      * directory -- containing per-sample ``<name>.json`` files.
    """
    if source.is_dir():
        records: list[dict] = []
        for fp in sorted(source.glob("*.json")):
            try:
                with fp.open() as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Skipping %s: %s", fp.name, e)
                continue
            if isinstance(data, dict):
                # Make sure each per-file record has a name even if the
                # JSON itself doesn't set one.
                data.setdefault("name", fp.stem)
                records.append(data)
        if not records:
            raise FileNotFoundError(f"No per-sample JSONs in {source}")
        logger.info("Loaded %d records from directory %s", len(records), source)
        return _index_records(records)

    suffix = source.suffix.lower()
    if suffix == ".json":
        with source.open() as f:
            data = json.load(f)
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            records = [data]
        else:
            raise ValueError(f"Unsupported JSON shape in {source}: {type(data).__name__}")
        logger.info("Loaded %d records from %s", len(records), source)
        return _index_records(records)

    if suffix in {".pkl", ".pickle"}:
        with source.open("rb") as f:
            blob = pickle.load(f)
        if isinstance(blob, dict) and "samples" in blob:
            records = blob["samples"]
        elif isinstance(blob, list):
            records = blob
        else:
            raise ValueError(f"Unsupported pickle shape in {source}: {type(blob).__name__}")
        logger.info("Loaded %d records from cache %s", len(records), source)
        return _index_records(records)

    raise ValueError(f"Unsupported source extension: {source}")


# =====================================================================
# Optional metrics overlay (from the benchmark NPZ)
# =====================================================================

# Display title -> NPZ key (per-sample arrays only).
_METRIC_KEYS = [
    ("delta",     "similarity_scores"),
    ("area_err",  "area_proportion_errors"),
    ("gap",       "boundary_gap_ratios"),
    ("rooms_acc", "accuracies"),
    ("rcount",    "room_count_accuracies"),
    ("rtype",     "room_type_accuracies"),
]


def load_metrics(npz_path: Optional[Path], names: list[str]) -> dict[str, dict[str, float]]:
    """Pair the NPZ per-sample arrays with sample names by position.

    The benchmark NPZ does not store names alongside the metric arrays.
    By convention the arrays are written in evaluation order. This loader
    therefore needs the same ``names`` list (in order) to align rows --
    callers should pass the keys of the predicted records sorted exactly
    the way evaluation iterated over them.
    """
    if npz_path is None or not npz_path.is_file():
        if npz_path is not None:
            logger.warning("Metrics NPZ not found: %s", npz_path)
        return {}
    blob = np.load(npz_path, allow_pickle=True)

    arrays: dict[str, np.ndarray] = {}
    for _, key in _METRIC_KEYS:
        if key in blob.files:
            arr = np.asarray(blob[key], dtype=float).reshape(-1)
            arrays[key] = arr

    if not arrays:
        logger.warning("No per-sample metric arrays found in %s", npz_path)
        return {}

    n = min((len(a) for a in arrays.values()), default=0)
    if n == 0:
        return {}
    if n != len(names):
        logger.warning("Metrics NPZ has %d rows but %d predicted samples; "
                       "aligning by position to the first %d names. "
                       "If you used --include_ids/--shard at eval time the "
                       "alignment may be incorrect.",
                       n, len(names), min(n, len(names)))

    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(names[:n]):
        row: dict[str, float] = {}
        for label, key in _METRIC_KEYS:
            arr = arrays.get(key)
            if arr is not None and i < len(arr):
                row[label] = float(arr[i])
        out[name] = row
    return out


# =====================================================================
# Per-sample panel
# =====================================================================

LEGEND_PATCHES = [
    Patch(facecolor=np.array(COLOR_MAP[c]) / 255.0, edgecolor="black",
          label=f"{c}: {CATEGORY_NAMES[c]}")
    for c in CATEGORY_NAMES
]


def _pick_boundary(record: Optional[dict]) -> list[list[int]]:
    """Prefer ``boundary_expand`` (40-point variant) for crisper rendering."""
    if record is None:
        return []
    if isinstance(record.get("boundary_expand"), list) and record["boundary_expand"]:
        return [[int(p[0]), int(p[1])] for p in record["boundary_expand"]]
    raw = record.get("boundary") or []
    return [[int(p[0]), int(p[1])] for p in raw]


def _door_segment(record: Optional[dict]) -> Optional[list[list[int]]]:
    """Front door = first edge of raw ``boundary``. DiffPlanner convention.

    The separate ``entrance_expand`` field is a fattened rectangle offset
    OUTSIDE the wall and is NOT the door position; we deliberately ignore
    it for rendering, matching DiffPlanner's own visualizer.
    """
    if record is None:
        return None
    raw = record.get("boundary")
    if not isinstance(raw, list) or len(raw) < 2:
        return None
    try:
        return [[int(raw[0][0]), int(raw[0][1])], [int(raw[1][0]), int(raw[1][1])]]
    except (IndexError, TypeError, ValueError):
        return None


def _adjacencies(record: Optional[dict]) -> list[list[int]]:
    if record is None:
        return []
    return list(record.get("adjacencies_aligned")
                or record.get("adjacencies") or [])


def _format_metrics_header(
    name: str,
    pred: Optional[dict],
    gt: Optional[dict],
    metrics_row: Optional[dict[str, float]],
) -> str:
    parts = [f"sample={name}"]
    n_pred = len(pred.get("rooms", []) or []) if pred else 0
    n_gt = len(gt.get("rooms", []) or []) if gt else None
    if n_gt is not None and pred is not None:
        parts.append(f"|rooms| GT={n_gt} pred={n_pred}")
    elif n_gt is not None:
        parts.append(f"|rooms| GT={n_gt}")
    elif pred is not None:
        parts.append(f"|rooms| pred={n_pred}")
    if pred is not None and pred.get("__pp_status") == "raw_fallback":
        parts.append("PP=fallback")
    if metrics_row:
        for label, _ in _METRIC_KEYS:
            v = metrics_row.get(label)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            parts.append(f"{label}={v:.3f}")
    return "  |  ".join(parts)


def render_sample_panel(
    name: str,
    pred: Optional[dict],
    gt: Optional[dict],
    metrics_row: Optional[dict[str, float]],
    *,
    canvas_size: int = CANVAS_SIZE_RENDER,
) -> "plt.Figure":
    bnd_pred = _pick_boundary(pred)
    door_pred = _door_segment(pred)
    bnd_gt = _pick_boundary(gt)
    door_gt = _door_segment(gt)
    # Predicted records that came from the LLM (or any pipeline that
    # echoes the GT boundary back unchanged) sometimes lack a door of
    # their own. Borrow the GT's so the predicted panel still shows the
    # entrance.
    if door_pred is None and door_gt is not None:
        door_pred = door_gt

    pred_rooms = (pred or {}).get("rooms", []) or []
    gt_rooms = (gt or {}).get("rooms", []) or []
    pred_adj = _adjacencies(pred)
    gt_adj = _adjacencies(gt)

    img_gt_fp = draw_floorplan(bnd_gt, gt_rooms,
                               canvas_size=canvas_size, door_segment=door_gt) if gt else None
    img_pred_fp = draw_floorplan(bnd_pred, pred_rooms,
                                 canvas_size=canvas_size, door_segment=door_pred) if pred else None
    img_gt_bubble = draw_bubble(bnd_gt, gt_rooms, gt_adj,
                                canvas_size=canvas_size) if gt else None
    img_pred_bubble = draw_bubble(bnd_pred, pred_rooms, pred_adj,
                                  canvas_size=canvas_size) if pred else None

    panels = [
        ("GT floor plan", img_gt_fp),
        ("Predicted floor plan", img_pred_fp),
        ("GT bubble diagram", img_gt_bubble),
        ("Predicted bubble diagram", img_pred_bubble),
    ]
    panels = [(t, im) for t, im in panels if im is not None]

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.6))
    if n == 1:
        axes = [axes]
    for ax, (title, im) in zip(axes, panels):
        ax.imshow(im)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(_format_metrics_header(name, pred, gt, metrics_row),
                 fontsize=11, y=0.99)
    fig.legend(handles=LEGEND_PATCHES, loc="lower center",
               ncol=len(CATEGORY_NAMES), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    return fig


# =====================================================================
# Aggregate summary chart
# =====================================================================

def render_summary(
    names: list[str],
    metrics_by_name: dict[str, dict[str, float]],
    cache_meta: Optional[dict] = None,
) -> "plt.Figure":
    if not metrics_by_name:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5,
                "No metrics provided -- pass --metrics-npz to plot per-sample numbers.",
                ha="center", va="center")
        ax.set_axis_off()
        return fig

    # Bar plot of the three primary metrics (delta, area err, gap).
    primary_keys = ["delta", "area_err", "gap"]
    rows = [(n, metrics_by_name.get(n, {})) for n in names]
    rows = [(n, m) for n, m in rows if any(k in m for k in primary_keys)]
    if not rows:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No primary metrics to plot.", ha="center", va="center")
        ax.set_axis_off()
        return fig

    n = len(rows)
    fig, (ax_bars, ax_box) = plt.subplots(
        2, 1, figsize=(max(8, n * 0.7), 7.5),
        gridspec_kw=dict(height_ratios=[3, 1]),
    )

    width = 0.27
    idx = np.arange(n)
    delta = np.array([m.get("delta", np.nan) for _, m in rows], dtype=float)
    aerr = np.array([m.get("area_err", np.nan) for _, m in rows], dtype=float)
    gap = np.array([m.get("gap", np.nan) for _, m in rows], dtype=float)
    ax_bars.bar(idx - width, np.nan_to_num(delta), width,
                label="delta (lower=better)", color="#4c72b0")
    ax_bars.bar(idx, np.nan_to_num(aerr), width,
                label="area_proportion_error (lower=better)", color="#55a868")
    ax_bars.bar(idx + width, np.nan_to_num(gap), width,
                label="boundary_gap_ratio (lower=better)", color="#c44e52")
    ax_bars.set_xticks(idx)
    ax_bars.set_xticklabels([n for n, _ in rows], rotation=45, ha="right", fontsize=8)
    y_max = float(np.nanmax(np.concatenate([delta, aerr, gap, [0.1]])))
    ax_bars.set_ylim(0, max(0.2, y_max * 1.1))
    ax_bars.set_ylabel("metric value")
    ax_bars.set_title("Per-sample DiffPlanner metrics")
    ax_bars.legend(loc="upper left", fontsize=9)
    ax_bars.grid(axis="y", linestyle="--", alpha=0.4)

    # Headline aggregate text panel.
    ax_box.set_axis_off()
    headlines = [f"samples = {n}"]
    if cache_meta:
        for k in ("model", "data_path", "use_ddim", "batch_size", "timestamp"):
            v = cache_meta.get(k)
            if v is not None:
                headlines.append(f"{k:18}= {v}")

    def _agg(label: str, key: str) -> str:
        a = np.array([metrics_by_name.get(n, {}).get(key, np.nan) for n in names],
                     dtype=float)
        a = a[~np.isnan(a)]
        if a.size == 0:
            return f"{label:18}= n/a"
        return f"{label:18}= mean={a.mean():.4f}  std={a.std():.4f}  median={np.median(a):.4f}"

    headlines.append(_agg("delta",            "delta"))
    headlines.append(_agg("area_err",         "area_err"))
    headlines.append(_agg("boundary_gap",     "gap"))
    headlines.append(_agg("rooms_accuracy",   "rooms_acc"))
    headlines.append(_agg("rcount_accuracy",  "rcount"))
    headlines.append(_agg("rtype_accuracy",   "rtype"))
    ax_box.text(0.01, 0.95, "\n".join(headlines), va="top", ha="left",
                family="monospace", fontsize=10)

    fig.tight_layout()
    return fig


# =====================================================================
# Main
# =====================================================================

def fig_to_png(fig, path: Path, *, dpi: int = 220) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def _load_cache_metadata(path: Path) -> Optional[dict]:
    if path.suffix.lower() not in {".pkl", ".pickle"} or not path.is_file():
        return None
    try:
        with path.open("rb") as f:
            blob = pickle.load(f)
    except (OSError, pickle.UnpicklingError):
        return None
    if isinstance(blob, dict):
        return blob.get("metadata")
    return None


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Visualize DiffPlanner floor plans (GT and/or predicted)")
    p.add_argument("--predicted", type=Path, default=None,
                   help="Predicted records: a DiffPlanner JSON list, a "
                        "single-sample JSON, a benchmark cache .pkl, or a "
                        "directory of per-sample <name>.json files.")
    p.add_argument("--gt-json", type=Path,
                   default=Path("dataset/rplan/dataset_diffplanner/data_test.json"),
                   help="Monolithic DiffPlanner data_test.json for ground-truth panels. "
                        "Pass an empty path or a non-existent path to skip GT panels.")
    p.add_argument("--metrics-npz", type=Path, default=None,
                   help="results.npz emitted by benchmark.src.evaluate_diffplanner "
                        "(via utils.report_results). Per-sample arrays are aligned to "
                        "predicted records by sort order.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory (default: <predicted>_vis next to --predicted).")
    p.add_argument("--canvas-size", type=int, default=CANVAS_SIZE_RENDER,
                   help=f"Internal raster size (default: {CANVAS_SIZE_RENDER}).")
    p.add_argument("--dpi", type=int, default=220,
                   help="DPI for saved PNGs / PDF pages (default: 220).")
    p.add_argument("--limit", type=int, default=None,
                   help="Only render the first N samples (sorted by name).")
    p.add_argument("--names", type=str, default=None,
                   help="Comma-separated list of specific sample names to render.")
    args = p.parse_args(argv)

    # ----- Resolve sources ------------------------------------------------
    if args.predicted is None:
        # Predicted-less mode: just render GT.
        if not args.gt_json.is_file():
            logger.error("Neither --predicted nor a valid --gt-json provided.")
            return 1
        gt_by_name = load_records(args.gt_json)
        pred_by_name: dict[str, dict] = {}
        out_dir = args.out_dir or Path("dp_gt_vis")
        cache_meta = None
    else:
        if not args.predicted.exists():
            logger.error("predicted source not found: %s", args.predicted)
            return 1
        pred_by_name = load_records(args.predicted)
        gt_by_name = load_records(args.gt_json) if args.gt_json.is_file() else {}
        if not gt_by_name:
            logger.warning("Ground truth not loaded (%s); rendering predicted-only panels.",
                           args.gt_json)
        if args.predicted.is_dir():
            default_out = args.predicted.with_name(f"{args.predicted.name}_vis")
        else:
            default_out = args.predicted.with_suffix("").with_name(
                f"{args.predicted.stem}_vis")
        out_dir = args.out_dir or default_out
        cache_meta = _load_cache_metadata(args.predicted)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Choose names to render ----------------------------------------
    if args.names:
        wanted = [s.strip() for s in args.names.split(",") if s.strip()]
        names = [n for n in wanted
                 if (n in pred_by_name) or (n in gt_by_name and not pred_by_name)]
        missing = [n for n in wanted if n not in names]
        if missing:
            logger.warning("Skipping unknown names: %s", missing)
    else:
        source = pred_by_name or gt_by_name
        names = sorted(source.keys(), key=lambda s: (len(s), s))
        if args.limit:
            names = names[: args.limit]

    if not names:
        logger.error("Nothing to render -- no overlapping sample names.")
        return 1

    # ----- Metrics overlay -----------------------------------------------
    metrics_by_name = load_metrics(args.metrics_npz, names) if args.metrics_npz else {}

    # ----- Render --------------------------------------------------------
    pdf_path = out_dir / "visualizations.pdf"
    pdf = PdfPages(pdf_path)

    summary_fig = render_summary(names, metrics_by_name, cache_meta=cache_meta)
    fig_to_png(summary_fig, out_dir / "summary.png", dpi=args.dpi)
    pdf.savefig(summary_fig, bbox_inches="tight", dpi=args.dpi)
    plt.close(summary_fig)
    logger.info("Wrote %s", out_dir / "summary.png")

    n_done = 0
    for name in names:
        pred = pred_by_name.get(name)
        gt = gt_by_name.get(name)
        if pred is None and gt is None:
            continue
        try:
            fig = render_sample_panel(name, pred, gt,
                                      metrics_by_name.get(name),
                                      canvas_size=args.canvas_size)
        except Exception as e:  # never let one bad sample sink the whole job
            logger.warning("Render failed for %s: %s", name, e)
            continue
        png_path = out_dir / f"{name}.png"
        fig_to_png(fig, png_path, dpi=args.dpi)
        pdf.savefig(fig, bbox_inches="tight", dpi=args.dpi)
        plt.close(fig)
        n_done += 1
        logger.info("  %s.png", name)

    pdf.close()
    logger.info("Wrote %d per-sample PNGs to %s", n_done, out_dir)
    logger.info("Wrote combined PDF: %s", pdf_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
