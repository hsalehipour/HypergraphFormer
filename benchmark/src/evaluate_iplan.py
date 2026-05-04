#!/usr/bin/env python3
"""
Evaluate iPLAN (CVPR 2022) floor-plan generation in the **OurII** mode --
the user supplies the boundary AND the per-floorplan room types/count,
and iPLAN's three-stage cascade fills in the rest:

    Stage 1 (BCVAE for room types) ........... SKIPPED (rTypes := gt_rTypes)
    Stage 2 (room locations, ResNet+ASPP) .... runs
    Stage 3 (room partitions, GAN) ........... runs (with fine-tuning)

Inputs AND ground truth both come from iPLAN's native ``.mat`` files
(``dataset/rplan/dataset_iplan/``, mirroring ``dataset/rplan/dataset_diffplanner``).
Each .mat carries:
  * ``Boundary``    -- the boundary polygon (128-grid).
  * ``gt_rTypes``   -- ground-truth room types (iPLAN 13-class vocab).
  * ``gt_rBoxes``   -- ground-truth axis-aligned room boxes (128-grid).
  * ``rTypes`` / ``rBoxes`` -- filled in by inference.
The .mat is the iPLAN preprocessor's parse of the original RPLAN PNG, so
GT is the original RPLAN, in the same coordinate frame and same
13-class vocabulary as the predictions. No second category mapping table
is needed.

Metrics computed (per sample):

  1. ``delta``                   Room shape regularity deviation
                                  (``compute_room_similarity_score``).
  2. ``area_proportion_error``   Mean absolute area-proportion difference
                                  between paired rooms
                                  (``compute_room_area_proportion_error``).
  3. Boundary-aware geometry metrics (all normalized by the input
     boundary area F, so values are comparable across samples and
     methods; see :func:`compute_boundary_geometry_metrics`):

       * ``gap_ratio``     -- area(F − U) / area(F):
                              fraction of the boundary left uncovered.
                              Ideal = 0.
       * ``outside_ratio`` -- area(U − F) / area(F):
                              predicted room area spilling outside
                              the boundary, normalized by F. Ideal = 0.
       * ``overlap_ratio`` -- (S − area(U)) / area(F):
                              total area double-counted by overlapping
                              rooms, normalized by F. Ideal = 0.

     where U = union of predicted room polygons, S = sum of their areas.
  4. Room accuracy (combined / count-only / type-only)
                                 (``room_type_accuracy``).

Caching mirrors the DiffPlanner / HouseDiffusion evaluators:
``--save_generations`` writes a pickle of the iPlan-predicted records
(one per sample) so a subsequent ``--load_generations`` rerun recomputes
metrics without GPU work.

Examples
--------
End-to-end on the full test split, with caching::

    python -m benchmark.src.evaluate_iplan \
        --iplan_data_dir dataset/rplan/dataset_iplan \
        --iplan_root     /home/hesam/REPOs/iPlan \
        --save_generations cache/iplan_ourII_test.pkl \
        --out benchmark/eval_results_iplan_ourII

Re-run metrics from cache (no GPU, no iPlan repo needed)::

    python -m benchmark.src.evaluate_iplan \
        --load_generations cache/iplan_ourII_test.pkl \
        --out benchmark/eval_results_iplan_ourII

Smoke test on 10 samples::

    python -m benchmark.src.evaluate_iplan \
        --iplan_data_dir dataset/rplan/dataset_iplan \
        --iplan_root     /home/hesam/REPOs/iPlan \
        --max_samples 10 \
        --out benchmark/eval_results_iplan_smoke
"""

from __future__ import annotations

import argparse
import math
import pickle
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid
from tqdm import tqdm

# NOTE: ``utils/__init__.py`` re-exports HypergraphPlotter, which uses
# PEP 585 generics (``dict[str, str]``) and therefore requires Python
# >= 3.9. iPLAN's conda env is Python 3.8, and we want to run the cascade
# in that env. Side-load the two leaf modules we actually need
# (`utils.metrics` and `utils.report_results`) by file path so the
# package's __init__ never executes. ``hypergraph`` is only referenced
# from helpers we don't call (compute_ged_from_hypergraph,
# compute_room_type_accuracy), so we stub it out before exec.
def _load_utils_modules() -> tuple[Any, Any]:
    import importlib.util
    import types

    here = Path(__file__).resolve()
    repo_root = here.parents[2]                           # HypergraphFormer/
    utils_dir = repo_root / "utils"

    # Stub ``utils`` and ``utils.hypergraph`` before loading metrics.py so
    # its ``from .hypergraph import Hypergraph`` resolves without pulling
    # in HypergraphPlotter (which needs Python >= 3.9).
    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = [str(utils_dir)]
    sys.modules.setdefault("utils", utils_pkg)
    hg_stub = types.ModuleType("utils.hypergraph")
    class _HypergraphStub:                                # noqa: D401
        """Placeholder; never instantiated by this evaluator."""
        def __init__(self, *_, **__):
            raise RuntimeError(
                "utils.hypergraph.Hypergraph is unavailable in this "
                "evaluator (Python 3.8 iPLAN env). Load the cache in a "
                "Python 3.9+ env if you need GED / hypergraph metrics."
            )
    hg_stub.Hypergraph = _HypergraphStub
    sys.modules.setdefault("utils.hypergraph", hg_stub)

    spec_m = importlib.util.spec_from_file_location(
        "utils.metrics", utils_dir / "metrics.py"
    )
    metrics_mod = importlib.util.module_from_spec(spec_m)
    sys.modules["utils.metrics"] = metrics_mod
    spec_m.loader.exec_module(metrics_mod)

    spec_r = importlib.util.spec_from_file_location(
        "utils.report_results", utils_dir / "report_results.py"
    )
    report_mod = importlib.util.module_from_spec(spec_r)
    sys.modules["utils.report_results"] = report_mod
    spec_r.loader.exec_module(report_mod)

    return metrics_mod, report_mod


_metrics_mod, _report_mod = _load_utils_modules()
compute_room_similarity_score = _metrics_mod.compute_room_similarity_score
compute_room_area_proportion_error = _metrics_mod.compute_room_area_proportion_error
room_type_accuracy = _metrics_mod.room_type_accuracy
report_results = _report_mod.report_results


# =====================================================================
# iPLAN constants
# =====================================================================
#
# iPLAN's native room categories (13 classes; see room_partition/utils.py
# `room_label`). Mapped here to canonical ``ROOM_ORDER`` keys consumed by
# ``utils.metrics._canonical_room_type_key``: the metrics module groups
# rooms by these strings when pairing predictions with ground truth.
#
# Both predictions and ground truth come from iPLAN .mat files, so this
# is the only category->merge_id table we need.
#
#   iPLAN id  | name        | merge_id (ROOM_ORDER bucket)
#   0         | LivingRoom  | living
#   1         | MasterRoom  | bed
#   2         | Kitchen     | kitchen
#   3         | Bathroom    | bathroom
#   4         | DiningRoom  | dining
#   5         | ChildRoom   | bed
#   6         | StudyRoom   | bed
#   7         | SecondRoom  | bed
#   8         | GuestRoom   | bed
#   9         | Balcony     | extra
#   10        | Entrance    | foyer
#   11        | Storage     | storage
#   12        | Wall-in     | storage
IPLAN_ID_TO_MERGE_ID = {
    0: "living",
    1: "bed",
    2: "kitchen",
    3: "bathroom",
    4: "dining",
    5: "bed",
    6: "bed",
    7: "bed",
    8: "bed",
    9: "extra",
    10: "foyer",
    11: "storage",
    12: "storage",
}

# iPLAN works in a 128-pixel canvas. Both predictions and GT live in this
# frame, so no inter-frame scaling is needed.
IPLAN_CANVAS_SIZE = 128


# =====================================================================
# iPLAN cascade plumbing -- modeled after tools/run_ourII_and_visualize.py
# =====================================================================

def _add_iplan_to_path(iplan_root: Path) -> None:
    """Make iPLAN's modules importable. iPLAN's own scripts use ``import
    utils`` / ``from synth.x import y`` etc. that only resolve when its repo
    root is on sys.path before any other ``utils`` package."""
    sys.path.insert(0, str(iplan_root))


def _load_iplan_models(iplan_root: Path, device_str: str) -> dict:
    """Instantiate Stage-2 (living + location) and Stage-3 (G_net) models
    once and return them as a single bundle dict.

    Mirrors :func:`tools/run_ourII_and_visualize._load_*` from the iPlan
    repo. Pulled inline here so the evaluator doesn't depend on the
    visualisation script being on PYTHONPATH.
    """
    import torch
    from room_location.Living import models as living_mod
    from room_location.Location import models as location_mod
    from room_partition import models as part_models
    from room_partition.models.loss_layer import LossFun

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    weights_loc_dir = iplan_root / "room_location" / "weights"
    pretrained_resnet = (
        iplan_root / "room_location" / "pretrained_model" / "resnet18-5c106cde.pth"
    )
    weights_part_g = iplan_root / "room_partition" / "weights" / "G_net_210.pth"

    living_model = living_mod.model(
        module_name="living", model_name="resnet18_fc1",
        input_channel=3, output_channel=2, pretrained=False,
    )
    living_connect = living_mod.connect(
        module_name="living", model_name="resnet18_fc1",
        input_channel=512, output_channel=2, reshape=True,
    )
    living_model.load_model(str(weights_loc_dir / "living_resnet18_300.pth"))
    living_connect.load_model(str(weights_loc_dir / "living_fc1_300.pth"))
    living_model = living_model.to(device).eval()
    living_connect = living_connect.to(device).eval()

    location_model = location_mod.model(
        module_name="location", model_name="resnet18_up1",
        input_channel=13 + 4, output_channel=13 + 3,
        pretrained=True, pretrained_path=str(pretrained_resnet),
    )
    location_connect = location_mod.connect(
        module_name="location", model_name="resnet18_up1",
        input_channel=512, output_channel=13 + 3, reshape=False,
    )
    location_embedding = location_mod.embedding(
        module_name="location", model_name="resnet18_up1",
        input_channel=13, output_channel=256, reshape=False,
    )
    location_model.load_model(str(weights_loc_dir / "location_resnet18_100.pth"))
    location_connect.load_model(str(weights_loc_dir / "location_up1_100.pth"))
    location_embedding.load_model(str(weights_loc_dir / "location_embed_100.pth"))
    location_model = location_model.to(device).eval()
    location_connect = location_connect.to(device).eval()
    location_embedding = location_embedding.to(device).eval()

    # FloorPlanRNNTest is the inference variant of the Stage-3 GAN generator.
    rnn_opt = SimpleNamespace(
        gpu_ids=[int(device_str.split(":")[-1])] if device_str.startswith("cuda") else None,
        load_netG_path=str(weights_part_g),
    )
    fp_rnn = part_models.FloorPlanRNNTest(rnn_opt)
    fp_rnn.load_networks(rnn_opt)

    criterion = LossFun(device=fp_rnn.device)

    return {
        "device": device,
        "living_model": living_model,
        "living_connect": living_connect,
        "location_model": location_model,
        "location_connect": location_connect,
        "location_embedding": location_embedding,
        "fp_rnn": fp_rnn,
        "criterion": criterion,
    }


# ---------------------------------------------------------------------
# Stage-2 / Stage-3 wrappers (per-sample). These are intentionally
# functional wrappers around iPLAN's own per-stage scripts so that we
# stay byte-for-byte compatible with the official `synth/test_*.py`
# inference paths.
# ---------------------------------------------------------------------

def _seed_gt_room_types(mat_path: str) -> int:
    """OurII trick: copy ``gt_rTypes`` -> ``rTypes`` so Stage 1 (BCVAE) is
    skipped. Also clears any preexisting predicted boxes/centers in the
    .mat (they are re-derived by Stages 2/3).

    Returns the room count.
    """
    import scipy.io as sio

    data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)["data"]
    rtypes = np.atleast_1d(data.gt_rTypes).astype(int)
    out = {
        "name": str(data.name),
        "gt_rTypes": np.atleast_1d(data.gt_rTypes).astype(int),
        "gt_rBoxes": np.atleast_2d(data.gt_rBoxes).astype(int),
        "Boundary": np.atleast_2d(data.Boundary).astype(int),
        "rTypes": rtypes,
        "rBoxes": np.array([], dtype=int),
        "rCenters": np.array([], dtype=int),
    }
    sio.savemat(mat_path, {"data": out})
    return int(rtypes.shape[0])


def _run_stage2(mat_path: str, models: dict, max_iters: int = 50) -> tuple[bool, int, int]:
    """Predict room centers (Stage 2). Writes back ``rCenters`` to the
    .mat. Returns ``(fully_converged, n_placed, n_target)``.

    Mirrors :func:`tools/run_ourII_and_visualize._run_stage2`: re-rolls up
    to ``max_iters`` times (because ``get_rcenters`` shuffles room order
    internally) and falls back to the most-complete partial placement if
    none of the attempts converges.
    """
    import scipy.io as sio
    import torch
    from synth.floorplan.roomlocation_fp import FloorPlan as LocFloorPlan
    from synth.test_roomlocation import get_rcenters

    fp = LocFloorPlan(mat_path)
    inp = fp.get_composite_living().unsqueeze(0).to(models["device"])
    with torch.no_grad():
        feat = models["living_model"](inp)
        living_xy = torch.round(models["living_connect"](feat)).squeeze().cpu().numpy()

    h, w = int(living_xy[0]), int(living_xy[1])
    fp.add_room({"category": 0, "centroid": (h, w)})
    fp.living_node = {"category": 0, "centroid": (h, w)}

    target = int(np.atleast_1d(sio.loadmat(
        mat_path, squeeze_me=True, struct_as_record=False
    )["data"].rTypes).shape[0])

    best_centers, best_types, best_n = None, None, -1
    for attempt in range(max_iters):
        np.random.seed(attempt + 1)
        pred_centers, pred_types, ok = get_rcenters(
            fp, models["location_model"], models["location_connect"],
            models["location_embedding"],
        )
        if ok:
            fp.update_rCenters(pred_centers, pred_types)
            sio.savemat(mat_path, {"data": fp.to_dict()})
            return True, target, target
        if len(pred_centers) > best_n:
            best_n = len(pred_centers)
            best_centers, best_types = pred_centers, pred_types

    if best_centers is not None and best_n > 0:
        fp.update_rCenters(best_centers, best_types)
        sio.savemat(mat_path, {"data": fp.to_dict()})
        return False, best_n, target

    return False, 0, target


def _run_stage3(mat_path: str, models: dict, max_iter: int = 200, lr: float = 10.0,
                coverage: float = 1.0, inside: float = 0.5,
                mutex: float = 0.0) -> np.ndarray:
    """Predict room partitions (Stage 3). Returns the final 128x128
    integer layout (room codes 0..12 inside the boundary; 13 outside;
    16 unallocated interior).

    Mirrors :func:`synth/test_roompartition` with a configurable fine-
    tuning loop; the loop continues until the residual gap ratio drops
    below 5e-4 or ``max_iter`` is reached.
    """
    from synth.floorplan.roompartition_fp import FloorPlan as PartFloorPlan
    from synth.test_roompartition import (
        fine_turning, get_image, get_ratio, obtain_living,
    )

    fp_rnn = models["fp_rnn"]
    criterion = models["criterion"]

    fp = PartFloorPlan(mat_path)
    img, rCenters, rTypes, inside_mask, boundary = fp.get_input()
    fp_rnn.evaluate(img, rCenters, rTypes, inside_mask)

    boundary = boundary[:, [1, 0, 2, 3]]
    pred_rBoxes = fp_rnn.pred_rBoxes
    pred_rBoxes = pred_rBoxes[:, [1, 0, 3, 2]]

    layout = get_image(img, pred_rBoxes, rTypes, inside_mask)
    # obtain_living relabels unallocated interior pixels (== 16) that connect
    # to the boundary as Living (== 0), so the largest connected room is
    # encoded into the layout itself (NOT into pred_rBoxes).
    living_box, layout, living_mask = obtain_living(layout, boundary)

    criterion.get_initial_mutex(pred_rBoxes, boundary)

    opt = SimpleNamespace(lr=lr, coverage=coverage, inside=inside, mutex=mutex)
    it = 0
    while get_ratio(layout) > 5e-4 and it < max_iter:
        it += 1
        pred_rBoxes = fine_turning(pred_rBoxes, boundary, living_mask, criterion, opt)
        layout = get_image(img, pred_rBoxes, rTypes, inside_mask, living_mask)

    return layout.cpu().numpy().astype(np.int16)


# ---------------------------------------------------------------------
# Layout -> per-room polygons
# ---------------------------------------------------------------------

def _layout_to_room_polygons(layout: np.ndarray) -> list[dict]:
    """Extract one polygon per *connected component* of each room code in
    the final 128x128 layout, in iPLAN's native 128-pixel frame.

    Why connected components rather than one polygon per code? The same
    code can appear in two physically separate regions (e.g. two
    bedrooms both labelled with iPLAN id 1, or two bathrooms with id 3).
    We need each visible room as its own polygon so room counts and
    area-proportion pairings match what the renderer would draw.
    """
    from scipy import ndimage as _ndi
    from shapely.geometry import box as _shapely_box

    layout = np.asarray(layout)
    rooms: list[dict] = []
    # iPLAN room codes are 0..12; 13=external, 14=ext wall, 15=front door,
    # 16=unallocated interior, 17=wall-in. Anything > 12 is non-room.
    for code in range(13):
        mask = layout == code
        if not mask.any():
            continue
        labels, n = _ndi.label(mask)
        for cc in range(1, n + 1):
            cc_mask = labels == cc
            ys, xs = np.where(cc_mask)
            if ys.size < 2 or xs.size < 2:
                continue
            # Drop tiny components (likely sub-pixel artifacts of the
            # rasterized layout, not actual rooms).
            if cc_mask.sum() < 4:
                continue
            y0, y1 = float(ys.min()), float(ys.max())
            x0, x1 = float(xs.min()), float(xs.max())
            poly = _shapely_box(x0, y0, x1 + 1, y1 + 1)
            if poly.is_empty or poly.area < 1.0:
                continue
            rooms.append({
                "category": int(code),
                "polygon": poly,
                "area": float(poly.area),
                "perimeter": float(poly.length),
            })
    return rooms


# =====================================================================
# Per-sample record extractor (the only thing we cache + reuse downstream)
# =====================================================================

def _read_iplan_mat_meta(mat_path: str) -> dict:
    """Read the metadata + GT fields from an iPLAN .mat into a plain dict
    (we deliberately leave out the heavy mask arrays).
    """
    import scipy.io as sio

    data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)["data"]
    rtypes = np.atleast_1d(getattr(data, "rTypes", np.array([])))
    gt_types = np.atleast_1d(data.gt_rTypes)
    gt_boxes = np.atleast_2d(data.gt_rBoxes)
    boundary = np.atleast_2d(data.Boundary)
    return {
        "name": str(data.name),
        "rTypes": rtypes.astype(int).tolist(),
        "gt_rTypes": gt_types.astype(int).tolist(),
        "gt_rBoxes": gt_boxes.astype(int).tolist(),
        "Boundary_iplan": boundary.astype(int).tolist(),
    }


def run_iplan_one(mat_path_in: Path, models: dict,
                  workdir: Path) -> tuple[dict, str, dict]:
    """Run iPLAN OurII inference on one .mat (Stage 2 + Stage 3).

    Returns ``(record, status, stage_info)`` where:
      * ``record``     -- dict ready to be cached / pickled. Contains the
                          final 128x128 ``layout`` array (room codes
                          0..12 inside the boundary) plus metadata.
      * ``status``     -- 'ok' (Stage 2 placed every room) or 'partial'
                          (Stage 2 only placed *n_placed* < *n_rooms*; we
                          still ran Stage 3 over the partial set).
      * ``stage_info`` -- per-stage timings/counters useful for the report.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    mat_path = workdir / mat_path_in.name
    shutil.copyfile(mat_path_in, mat_path)

    n_rooms = _seed_gt_room_types(str(mat_path))

    t1 = time.time()
    fully_ok, n_placed, n_target = _run_stage2(str(mat_path), models)
    t_stage2 = time.time() - t1

    if n_placed == 0:
        # Stage 2 produced no centers; Stage 3 cannot run. Return a record
        # marked 'failed' so downstream metrics see an empty prediction
        # (no pred polygons -> gap_ratio = 1.0, no matched rooms).
        rec = _read_iplan_mat_meta(str(mat_path))
        rec["layout"] = np.full((IPLAN_CANVAS_SIZE, IPLAN_CANVAS_SIZE), 13,
                                dtype=np.int16).tolist()
        rec["__pp_status"] = "failed"
        rec["__n_placed"] = 0
        rec["__n_target"] = n_target
        return rec, "failed", {"stage2_s": t_stage2, "stage3_s": 0.0}

    t2 = time.time()
    layout = _run_stage3(str(mat_path), models)
    t_stage3 = time.time() - t2

    rec = _read_iplan_mat_meta(str(mat_path))
    rec["layout"] = layout.tolist()           # 128x128 list-of-lists; pickles small
    rec["__pp_status"] = "aligned" if fully_ok else "partial"
    rec["__n_placed"] = n_placed
    rec["__n_target"] = n_target
    return rec, ("ok" if fully_ok else "partial"), {
        "stage2_s": t_stage2, "stage3_s": t_stage3,
    }


# =====================================================================
# Polygon helpers + metric inputs
# =====================================================================

def _shapely_polygon_safe(coords) -> Optional[ShapelyPolygon]:
    if coords is None or len(coords) < 3:
        return None
    try:
        poly = ShapelyPolygon(coords)
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.is_empty or poly.area < 1e-6:
            return None
        return poly
    except Exception:
        return None


def _polygons_from_iplan_record(record: dict) -> list[dict]:
    """Build ``{category, polygon, area, perimeter}`` dicts from an iPLAN
    prediction record. Uses the rasterized 128x128 ``layout`` and extracts
    one bounding-box polygon per *connected component* of each room code,
    scaled to the GT 256-pixel frame.

    Going through the layout (rather than ``pred_rBoxes`` directly) is
    necessary because iPLAN's Stage 3 represents the Living room as the
    leftover-area component computed by ``obtain_living`` rather than as
    a row in ``pred_rBoxes``, so reading boxes alone would silently drop
    the largest room from every prediction.
    """
    layout = record.get("layout")
    if layout is None:
        return []
    return _layout_to_room_polygons(np.asarray(layout, dtype=np.int16))


def _polygons_from_gt_record(record: dict) -> list[dict]:
    """Build ``{category, polygon, area, perimeter}`` dicts from the iPLAN
    .mat ground truth (``gt_rTypes`` + ``gt_rBoxes``).

    ``gt_rBoxes`` are stored as ``(y0, x0, y1, x1)`` in iPLAN's 128-pixel
    frame (see ``data/preprocess/floorplan.py``); we materialize them as
    axis-aligned rectangle polygons so they live in the same coordinate
    frame as the predicted polygons extracted from the layout.
    """
    rooms = []
    gt_types = record.get("gt_rTypes", []) or []
    gt_boxes = record.get("gt_rBoxes", []) or []
    for cat, box in zip(gt_types, gt_boxes):
        if box is None or len(box) != 4:
            continue
        y0, x0, y1, x1 = (float(v) for v in box)
        if x1 <= x0 or y1 <= y0:
            continue
        coords = [[x0, y0], [x1 + 1, y0], [x1 + 1, y1 + 1], [x0, y1 + 1]]
        poly = _shapely_polygon_safe(coords)
        if poly is None:
            continue
        rooms.append({
            "category": int(cat),
            "polygon": poly,
            "area": float(poly.area),
            "perimeter": float(poly.length),
        })
    return rooms


def _to_similarity_list(rooms: list[dict]) -> list[tuple[str, float, float]]:
    """``[(merge_id, perimeter, square_perimeter), ...]`` for
    ``utils.metrics.compute_room_similarity_score``.
    """
    out = []
    for r in rooms:
        merge_id = IPLAN_ID_TO_MERGE_ID.get(r["category"], str(r["category"]))
        perim = max(r["perimeter"], 1e-6)
        square_perim = 4 * math.sqrt(max(r["area"], 1e-6))
        out.append((merge_id, perim, square_perim))
    return out


def _to_area_list(rooms: list[dict]) -> list[tuple[str, float]]:
    """``[(merge_id, area), ...]`` for
    ``utils.metrics.compute_room_area_proportion_error``.
    """
    return [(IPLAN_ID_TO_MERGE_ID.get(r["category"], str(r["category"])), r["area"])
            for r in rooms]


def _boundary_polygon_from_iplan(record: dict) -> Optional[ShapelyPolygon]:
    """Building outline as a Shapely polygon, in iPLAN's 128 frame.

    iPLAN's ``Boundary`` is an ``N x 4`` array of ``[row, col, dir, isNew]``
    rows produced by contour-walking the boundary mask in
    ``data/preprocess/floorplan.py`` (so col 0 holds image *row* = y, and
    col 1 holds image *col* = x). We swap to ``(x, y)`` for Shapely.

    This matches the same indexing used elsewhere in the iPLAN cascade,
    e.g. ``synth/test_roompartition.py`` permutes ``boundary[:, [1, 0, 2, 3]]``
    before passing it to the partition network.
    """
    coords_yx = record.get("Boundary_iplan")
    if not coords_yx:
        return None
    coords = [[float(p[1]), float(p[0])] for p in coords_yx if len(p) >= 2]
    return _shapely_polygon_safe(coords)


def compute_boundary_geometry_metrics(record: dict,
                                      pred_rooms: list[dict]) -> dict[str, float]:
    """Compute the three boundary-normalized geometry metrics for one
    sample. All denominators are ``area(F)`` (the input boundary area),
    so values are directly comparable across samples and methods.

    Returns a dict with keys ``gap_ratio``, ``outside_ratio``,
    ``overlap_ratio``. Each is NaN if F is degenerate; ``gap_ratio`` is
    1.0 (and the others 0.0) if there are no valid room polygons.

    Identities (per inclusion-exclusion):
        S / area(F) = coverage_ratio + outside_ratio + overlap_ratio
        gap_ratio   = 1 - coverage_ratio
    where coverage_ratio = area(U ∩ F) / area(F).

    Note on iPLAN's representation: the predicted polygons are the
    axis-aligned bounding boxes of connected components in iPLAN's
    one-hot layout. So a non-rectangular room (e.g. L-shaped) will have
    a bbox that *overlaps* with a neighbour (-> contributes to
    ``overlap_ratio``) AND *spills outside* the boundary in the L-cutout
    region (-> contributes to ``outside_ratio``). The two signals
    together let you read off how much of the geometry tax is due to
    the bbox representation vs. genuine misalignment.
    """
    boundary = _boundary_polygon_from_iplan(record)
    if boundary is None or boundary.area <= 0:
        return {"gap_ratio": float("nan"),
                "outside_ratio": float("nan"),
                "overlap_ratio": float("nan")}

    F_area = boundary.area
    if not pred_rooms:
        return {"gap_ratio": 1.0, "outside_ratio": 0.0, "overlap_ratio": 0.0}

    polys = [r["polygon"] for r in pred_rooms]
    union = unary_union(polys)
    sum_areas = sum(p.area for p in polys)
    inter_area = union.intersection(boundary).area
    union_area = union.area

    gap_ratio = max(0.0, 1.0 - inter_area / F_area)
    outside_ratio = max(0.0, (union_area - inter_area) / F_area)
    overlap_ratio = max(0.0, (sum_areas - union_area) / F_area)
    return {"gap_ratio": float(gap_ratio),
            "outside_ratio": float(outside_ratio),
            "overlap_ratio": float(overlap_ratio)}


# =====================================================================
# Per-sample metric driver
# =====================================================================

def evaluate_sample(record: dict, *, pp_status: str, results: dict) -> None:
    """Run all metrics for one sample and append into ``results``.

    Pred and GT both come from the same iPLAN record (predictions from
    ``layout``; GT from ``gt_rTypes`` / ``gt_rBoxes`` / ``Boundary_iplan``),
    so they share the iPLAN 13-class vocabulary and the 128-pixel frame.
    """
    pred_rooms = _polygons_from_iplan_record(record)
    gt_rooms = _polygons_from_gt_record(record)

    delta = compute_room_similarity_score(
        _to_similarity_list(pred_rooms), _to_similarity_list(gt_rooms)
    )

    area_metric = compute_room_area_proportion_error(
        _to_area_list(pred_rooms), _to_area_list(gt_rooms)
    )

    geom = compute_boundary_geometry_metrics(record, pred_rooms)

    pred_labels = [IPLAN_ID_TO_MERGE_ID.get(r["category"], str(r["category"]))
                   for r in pred_rooms]
    gt_labels = [IPLAN_ID_TO_MERGE_ID.get(r["category"], str(r["category"]))
                 for r in gt_rooms]
    rt_metric = room_type_accuracy(gt_labels, pred_labels, strict=True)

    n_target = int(record.get("__n_target", 0))
    visible = len(pred_rooms)
    lost = max(0, n_target - visible) if n_target > 0 else 0

    results["similarity_scores"].append(
        float(delta) if delta is not None else float("nan"))
    results["area_proportion_errors"].append(
        float(area_metric["area_proportion_error"]))
    results["gap_ratios"].append(geom["gap_ratio"])
    results["outside_ratios"].append(geom["outside_ratio"])
    results["overlap_ratios"].append(geom["overlap_ratio"])
    results["accuracies"].append(float(rt_metric["accuracy"]))
    results["room_count_accuracies"].append(float(rt_metric["room_count_accuracy"]))
    results["room_type_accuracies"].append(float(rt_metric["room_type_accuracy"]))
    results["per_sample_room_counts"].append(len(gt_rooms))
    results["pp_status"].append(pp_status)
    results["lost_rooms_per_sample"].append(int(lost))
    results["lost_room_count"] += int(lost)
    if lost > 0:
        results["samples_with_lost_rooms"] += 1
    if pp_status == "partial":
        results["partial_count"] += 1
    if pp_status == "failed":
        results["failed_count"] += 1


def _make_results_dict() -> dict:
    return {
        "similarity_scores": [],
        "area_proportion_errors": [],
        "gap_ratios": [],
        "outside_ratios": [],
        "overlap_ratios": [],
        "accuracies": [],
        "room_count_accuracies": [],
        "room_type_accuracies": [],
        "per_sample_room_counts": [],
        "pp_status": [],
        "lost_rooms_per_sample": [],
        "lost_room_count": 0,
        "samples_with_lost_rooms": 0,
        "partial_count": 0,
        "failed_count": 0,
    }


# =====================================================================
# Cache I/O (parallel to evaluate_diffplanner / evaluate_housediffusion)
# =====================================================================

def save_generations_iplan(filepath: Path, metadata: dict, samples: list[dict]) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "samples": samples}
    with filepath.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"Saved {len(samples)} samples to {filepath} ({size_mb:.1f} MB)")


def load_generations_iplan(filepath: Path) -> dict:
    filepath = Path(filepath)
    with filepath.open("rb") as f:
        payload = pickle.load(f)
    if "metadata" not in payload or "samples" not in payload:
        raise ValueError(f"Invalid generation cache: {filepath}")
    print(f"Loaded {len(payload['samples'])} cached samples from {filepath}")
    return payload


# =====================================================================
# CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="iPLAN (CVPR 2022) OurII benchmark on the RPLAN test split",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--iplan_data_dir", type=Path,
                   default=Path("dataset/rplan/dataset_iplan"),
                   help="Folder of iPLAN-format .mat files (one per test sample). "
                        "Each .mat carries both the inputs (Boundary, gt_rTypes) "
                        "and the ground truth (gt_rTypes, gt_rBoxes) used for "
                        "scoring -- mirrors the iPLAN preprocessor's parse of "
                        "the original RPLAN PNGs.")
    p.add_argument("--iplan_root", type=Path,
                   default=Path("/home/hesam/REPOs/iPlan"),
                   help="Path to the iPLAN repo root (where room_partition/, "
                        "synth/, room_location/ live).")
    p.add_argument("--device", type=str, default="cuda:0",
                   help='Torch device (e.g. "cuda:0" / "cpu").')

    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap evaluated samples (default: every .mat in iplan_data_dir).")
    p.add_argument("--shard", type=str, default=None,
                   help='Shard spec "K/N" -- evaluate every Nth sample starting at K.')
    p.add_argument("--include_ids", type=Path, default=None,
                   help="Optional file with one sample id per line; only those ids "
                        "are evaluated.")

    p.add_argument("--save_generations", type=Path, default=None,
                   help="If set, write a pickle of iPlan prediction records "
                        "(one per sample) for later metric reruns.")
    p.add_argument("--load_generations", type=Path, default=None,
                   help="If set, skip GPU work entirely and compute metrics from "
                        "a previously saved cache.")

    p.add_argument("--out", type=Path,
                   default=Path("benchmark/eval_results_iplan_ourII"),
                   help="Output folder for results.npz / results.txt.")
    p.add_argument("--workdir", type=Path, default=None,
                   help="Scratch folder for in-place .mat updates "
                        "(default: a fresh temp dir).")
    return p.parse_args()


def _select_sample_ids(all_ids: list[str], args: argparse.Namespace) -> list[str]:
    """Apply --include_ids / --shard / --max_samples filters in that order."""
    ids = list(all_ids)
    if args.include_ids:
        with args.include_ids.open() as f:
            wanted = {line.strip() for line in f if line.strip()}
        ids = [i for i in ids if i in wanted]
        print(f"--include_ids filter kept {len(ids)} samples")
    if args.shard:
        try:
            k, n = (int(x) for x in args.shard.split("/"))
        except ValueError:
            raise SystemExit(f"--shard must be 'K/N', got {args.shard!r}")
        ids = [i for idx, i in enumerate(ids) if idx % n == k]
        print(f"--shard {args.shard} kept {len(ids)} samples")
    if args.max_samples is not None:
        ids = ids[: args.max_samples]
    return ids


def _build_report_kwargs(results: dict) -> dict:
    n = len(results["similarity_scores"])
    return {
        "metrics": {
            "similarity_scores": ("Room Similarity (delta)",
                                  np.array(results["similarity_scores"])),
            "area_proportion_errors": ("Area Proportion Error",
                                       np.array(results["area_proportion_errors"])),
            "gap_ratios": ("Gap Ratio (uncovered boundary)",
                           np.array(results["gap_ratios"])),
            "outside_ratios": ("Outside Ratio (rooms outside boundary)",
                               np.array(results["outside_ratios"])),
            "overlap_ratios": ("Overlap Ratio (inter-room overlap)",
                               np.array(results["overlap_ratios"])),
            "accuracies": ("Room Accuracy (combined)",
                           np.array(results["accuracies"])),
            "room_count_accuracies": ("Room Count Accuracy",
                                      np.array(results["room_count_accuracies"])),
            "room_type_accuracies": ("Room Type Accuracy",
                                     np.array(results["room_type_accuracies"])),
        },
        "room_counts": np.array(results["per_sample_room_counts"]),
        "extra": {
            "partial_stage2_count": int(results["partial_count"]),
            "failed_stage2_count": int(results["failed_count"]),
            "lost_room_count": int(results["lost_room_count"]),
            "hidden_count": int(results["samples_with_lost_rooms"]),
            "total_samples": n,
        },
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    if args.save_generations and args.load_generations:
        raise SystemExit("Cannot use --save_generations and --load_generations together.")

    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    results = _make_results_dict()

    # ----- LOAD MODE: skip cascade, compute metrics from cache ------------
    if args.load_generations:
        cache = load_generations_iplan(args.load_generations)
        cached_records = {str(s["name"]): s for s in cache["samples"]}
        ids = sorted(cached_records.keys())
        ids = _select_sample_ids(ids, args)
        print(f"Will score {len(ids)} cached samples")
        for name in tqdm(ids, desc="Metrics (cached)"):
            rec = cached_records.get(name)
            if rec is None:
                continue
            evaluate_sample(
                rec,
                pp_status=rec.get("__pp_status", "aligned"),
                results=results,
            )

    # ----- GENERATE MODE: run the iPLAN OurII cascade ---------------------
    else:
        if not args.iplan_data_dir.is_dir():
            raise SystemExit(f"--iplan_data_dir not found: {args.iplan_data_dir}")
        if not args.iplan_root.is_dir():
            raise SystemExit(f"--iplan_root not found: {args.iplan_root}")

        all_ids = sorted(p.stem for p in args.iplan_data_dir.glob("*.mat"))
        ids = _select_sample_ids(all_ids, args)
        print(f"Will evaluate {len(ids)} samples")

        # iPlan modules use ``import utils`` etc. -- put the iPlan repo on
        # sys.path BEFORE we touch anything in this evaluator that also
        # imports from `utils.*`. Both happen to coexist because we use
        # absolute imports here (`from utils.metrics import ...`) and they
        # were resolved at module import time, before this point.
        _add_iplan_to_path(args.iplan_root)

        print(f"Loading iPLAN models on {args.device}...")
        models = _load_iplan_models(args.iplan_root, args.device)
        print("iPLAN models loaded.")

        # Scratch dir for in-place .mat updates so we don't mutate the
        # canonical dataset_iplan/ files.
        if args.workdir is None:
            workdir = Path(tempfile.mkdtemp(prefix="iplan_eval_"))
            print(f"Using temp workdir: {workdir}")
        else:
            workdir = args.workdir
            workdir.mkdir(parents=True, exist_ok=True)

        cache_samples: list[dict] = []
        try:
            for name in tqdm(ids, desc="iPLAN OurII"):
                src_mat = args.iplan_data_dir / f"{name}.mat"
                try:
                    rec, status, _ = run_iplan_one(src_mat, models, workdir)
                except Exception as exc:  # noqa: BLE001 - per-sample non-fatal
                    print(f"\n  [{name}] inference failed: {exc}")
                    # Keep GT fields even on failure so per-sample metrics
                    # still see something to compare against (predictions
                    # will be empty -> gap_ratio = 1.0, etc.).
                    try:
                        rec = _read_iplan_mat_meta(str(src_mat))
                    except Exception:
                        rec = {
                            "name": name, "rTypes": [],
                            "gt_rTypes": [], "gt_rBoxes": [],
                            "Boundary_iplan": [],
                        }
                    rec["layout"] = np.full(
                        (IPLAN_CANVAS_SIZE, IPLAN_CANVAS_SIZE), 13,
                        dtype=np.int16,
                    ).tolist()
                    rec["__pp_status"] = "failed"
                    rec["__n_placed"] = 0
                    rec["__n_target"] = len(rec.get("gt_rTypes", []))
                    status = "failed"

                if args.save_generations is not None:
                    cache_samples.append(rec)

                evaluate_sample(rec, pp_status=rec.get("__pp_status", status),
                                results=results)

        except KeyboardInterrupt:
            print("\nInterrupted by user; saving partial results...")
        finally:
            # Best-effort cleanup of scratch dir if we created it.
            if args.workdir is None:
                try:
                    shutil.rmtree(workdir, ignore_errors=True)
                except Exception:
                    pass

        if args.save_generations is not None:
            metadata = {
                "model": "iPLAN (CVPR 2022) -- OurII",
                "iplan_root": str(args.iplan_root),
                "iplan_data_dir": str(args.iplan_data_dir),
                "num_samples": len(cache_samples),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "version": "1.0",
            }
            save_generations_iplan(args.save_generations, metadata, cache_samples)

    # ----- Report ---------------------------------------------------------
    if not results["similarity_scores"]:
        print("No samples were scored -- nothing to report.")
        return 1

    n = len(results["similarity_scores"])
    print(f"\nScored {n} samples.")
    print(f"  Stage 2 PARTIAL (some rooms unplaced): {results['partial_count']}")
    print(f"  Stage 2 FAILED (no rooms placed):      {results['failed_count']}")
    print(f"  Lost rooms total: {results['lost_room_count']:,} across "
          f"{results['samples_with_lost_rooms']:,} samples "
          f"({100.0 * results['samples_with_lost_rooms'] / n:.1f}% have at "
          f"least one room dropped).")

    report_results(
        str(args.out),
        num_variations=1,
        checkpoint_path=str(args.iplan_root) if not args.load_generations else None,
        data_path=str(args.iplan_data_dir) if not args.load_generations
            else str(args.load_generations),
        **_build_report_kwargs(results),
    )
    print(f"\nWrote {args.out / 'results.txt'} and {args.out / 'results.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
