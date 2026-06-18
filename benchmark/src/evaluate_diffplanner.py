#!/usr/bin/env python3
"""
Evaluate DiffPlanner floor-plan generation in the boundary-constrained ``nc``
mode (``DiffPlanner^b_III`` in the paper).

For each test sample we run the full 3-stage cascade --

    NodeDiff  (b + nc)         predicts (size, location)            per room
        |
        v
    AdjacencyDiff (b + ncsl)   predicts the room adjacency matrix
        |
        v
    PartitioningDiff (b + ncsla) predicts axis-aligned room boxes
        |
        v
    output/post_processing.py  snaps boxes to walls, removes overlaps,
                               cuts precise polygons, fills the living-room
                               with leftover gaps.

-- and compute these metrics against ground truth:

  1. ``delta``                        Room shape regularity deviation
                                       (``compute_room_similarity_score``).
  2. ``area_proportion_error``        Mean absolute area-proportion difference
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

Caching mirrors ``benchmark/src/evaluate_housediffusion.py`` -- ``--save_generations``
writes a pickle of the post-processed DiffPlanner JSON records so a re-run of
the metric pass can skip GPU work entirely (``--load_generations``).

Examples
--------
Run end-to-end over the entire test split with caching::

    python benchmark/src/evaluate_diffplanner.py \\
        --data_path  dataset/rplan/dataset_diffplanner/data_test.json \\
        --diffplanner_root /home/hesam/REPOs/DiffPlanner \\
        --use_ddim \\
        --batch_size 256 \\
        --save_generations cache/dp_bnc_test.pkl \\
        --out  benchmark/eval_results_diffplanner

Re-run metrics on the cached generations only::

    python benchmark/src/evaluate_diffplanner.py \\
        --data_path  dataset/rplan/dataset_diffplanner/data_test.json \\
        --load_generations cache/dp_bnc_test.pkl \\
        --out  benchmark/eval_results_diffplanner

"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import torch as th
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid
from tqdm import tqdm

from utils.metrics import (
    compute_room_similarity_score,
    compute_room_area_proportion_error,
    ged_between_graphs,
    room_type_accuracy,
)
from utils.report_results import report_results


# =====================================================================
# DiffPlanner constants
# =====================================================================

DP_CATEGORY_TO_NAME = {
    0: "LivingRoom",
    1: "Bedroom",
    2: "Kitchen",
    3: "Bathroom",
    4: "Balcony",
    5: "Storage",
}

# Map DiffPlanner's 6-class category id -> canonical room-type key understood
# by ``utils.metrics._canonical_room_type_key`` (lowercase prefix match against
# ``ROOM_ORDER`` -- see utils/metrics.py for the matching rules).
DP_CATEGORY_TO_MERGE_ID = {
    0: "living",
    1: "bed",
    2: "kitchen",
    3: "bathroom",
    4: "extra",     # balcony -> 'extra' (last bucket in ROOM_ORDER)
    5: "storage",
}

CANVAS_SIZE = 256
MAX_NUM_ROOMS = 8


# =====================================================================
# Cascade plumbing -- all three sub-models loaded once
# =====================================================================

def _add_diffplanner_to_path(diffplanner_root: Path) -> None:
    """Make the three DiffPlanner sub-packages importable.

    Each sub-model lives in its own python package one level under
    ``DiffPlanner/{node_diff,adjacency_diff,partitioning_diff}/`` *and* a
    second time as ``DiffPlanner/{...}/{node_diff,adjacency_diff,
    partitioning_diff}/`` (the original repo's layout); we expose the outer
    folder so that ``import node_diff`` / etc. resolves.
    """
    for sub in ("node_diff", "adjacency_diff", "partitioning_diff"):
        sub_dir = diffplanner_root / sub
        if not sub_dir.is_dir():
            raise FileNotFoundError(f"DiffPlanner sub-package not found: {sub_dir}")
        sys.path.insert(0, str(sub_dir))
    sys.path.insert(0, str(diffplanner_root / "output"))


def _make_stage_args(
    *,
    support_conditions: str,
    use_ddim: bool,
    batch_size: int,
    model_path: Path,
    set_name: str,
) -> SimpleNamespace:
    """Construct the argparse-style namespace each sub-model's
    ``model_and_diffusion_defaults`` expects, so we can reuse
    ``create_model_and_diffusion`` and ``update_arg_parser`` without
    spawning a subprocess.
    """
    args = SimpleNamespace(
        dataset="rplan",
        clip_denoised=True,
        batch_size=batch_size,
        use_ddim=use_ddim,
        model_path=str(model_path),
        set_name=set_name,
        support_boundary=True,
        support_conditions=support_conditions,
        support_partial=False,
    )
    return args


def _load_one_stage(
    *,
    sub_pkg: str,                 # 'node_diff' | 'adjacency_diff' | 'partitioning_diff'
    support_conditions: str,
    use_ddim: bool,
    batch_size: int,
    model_path: Path,
    set_name: str,
    device: th.device,
):
    """Instantiate one diffusion sub-model and load its checkpoint.

    Returns the (model, diffusion, args) triple.  We import the package's own
    ``script_util`` so that ``input_channels`` / condition embeddings are wired
    up correctly for the requested ``support_conditions`` mode.
    """
    if sub_pkg == "node_diff":
        from node_diff.script_util import (        # noqa: WPS433  (delayed import is intentional)
            model_and_diffusion_defaults,
            create_model_and_diffusion,
            args_to_dict,
            update_arg_parser,
        )
    elif sub_pkg == "adjacency_diff":
        from adjacency_diff.script_util import (
            model_and_diffusion_defaults,
            create_model_and_diffusion,
            args_to_dict,
            update_arg_parser,
        )
    elif sub_pkg == "partitioning_diff":
        from partitioning_diff.script_util import (
            model_and_diffusion_defaults,
            create_model_and_diffusion,
            args_to_dict,
            update_arg_parser,
        )
    else:
        raise ValueError(f"Unknown DiffPlanner sub-package: {sub_pkg}")

    args = _make_stage_args(
        support_conditions=support_conditions,
        use_ddim=use_ddim,
        batch_size=batch_size,
        model_path=model_path,
        set_name=set_name,
    )

    # Inject defaults required by create_model_and_diffusion.
    for key, value in model_and_diffusion_defaults().items():
        if not hasattr(args, key):
            setattr(args, key, value)
    update_arg_parser(args)

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    state_dict = th.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, diffusion, args


# =====================================================================
# In-memory tensor builders (no on-disk JSON shuffle between stages)
# =====================================================================

_get_one_hot = lambda idx, sz: np.eye(sz)[idx]   # noqa: E731


def _build_node_inputs(sample: dict) -> dict:
    """Build the kwargs NodeDiff (``support_conditions='nc'``) needs for
    sampling. We only feed the few-shot inputs the user controls: boundary,
    door, room count, room categories.

    Mirrors ``node_diff/node_diff/rplan_datasets.get_datatensor`` for ``nc``
    mode.
    """
    canvas_size = CANVAS_SIZE
    max_num_rooms = MAX_NUM_ROOMS

    boundary_rplan = sample["boundary_expand"]
    cond_boundary = (np.array(boundary_rplan).flatten().reshape(1, -1) / canvas_size * 2) - 1
    cond_boundary = np.repeat(cond_boundary, max_num_rooms, axis=0)

    door_rplan = sample["entrance_expand"]
    cond_door = (np.array(door_rplan).flatten().reshape(1, -1) / canvas_size * 2) - 1
    cond_door = np.repeat(cond_door, max_num_rooms, axis=0)

    rooms_info = sample["rooms"]
    num_rooms = len(rooms_info)
    if num_rooms > max_num_rooms:
        # DiffPlanner's max is hard-coded to 8; safer to reject early than
        # silently truncate.
        raise ValueError(
            f"Sample {sample.get('name')} has {num_rooms} rooms but "
            f"DiffPlanner supports only {max_num_rooms}."
        )

    cond_number = np.zeros((max_num_rooms, max_num_rooms), dtype=np.int8)
    for r_i in range(num_rooms):
        cond_number[r_i] = _get_one_hot(r_i, max_num_rooms)

    cond_category = np.zeros((max_num_rooms, 6), dtype=np.int8)
    for r_i in range(num_rooms):
        cond_category[r_i] = _get_one_hot(int(rooms_info[r_i]["category"]), 6)

    atten_mask = np.ones((max_num_rooms, max_num_rooms))
    atten_mask[:num_rooms, :num_rooms] = 0
    padding_mask = np.zeros(max_num_rooms)
    padding_mask[num_rooms:] = 1                   # 1 = padded in DP convention

    cond_partial = np.full((max_num_rooms, 3), -1.0)   # nc -> 3 channels (sxy)

    return {
        "name_fp": sample["name"],
        "num_rooms_gt": num_rooms,
        "atten_mask": atten_mask,
        "padding_mask": padding_mask,
        "cond_boundary": cond_boundary,
        "cond_door": cond_door,
        "cond_number": cond_number,
        "cond_category": cond_category,
        "cond_partial": cond_partial,
        # 'syn_*' aliases used by the model's transformer:
        "syn_atten_mask": atten_mask,
        "syn_padding_mask": padding_mask,
        "syn_cond_boundary": cond_boundary,
        "syn_cond_door": cond_door,
        "syn_cond_number": cond_number,
        "syn_cond_category": cond_category,
        "syn_cond_partial": cond_partial,
    }


def _decode_node_output(node_tensor: np.ndarray, node_kwargs: dict) -> dict:
    """Convert NodeDiff's raw output tensor (max_num_rooms x 3) into the
    DiffPlanner JSON shape after ``support_conditions=nc``.

    Mirrors ``node_diff/scripts/sample.get_datajson_from_tensor`` for ``nc``.
    """
    cond_category = node_kwargs["cond_category"]
    rooms_category_list = np.argmax(cond_category, axis=1).tolist()
    num_rooms_gt = node_kwargs["num_rooms_gt"]

    canvas_area = CANVAS_SIZE * CANVAS_SIZE

    rooms_info = []
    for i in range(num_rooms_gt):
        rooms_info.append({
            "id": i,
            "category": int(rooms_category_list[i]),
            "size": int(round(((node_tensor[i, 0] + 1) / 2) * canvas_area)),
            "location": np.round(((node_tensor[i, 1:3] + 1) / 2) * CANVAS_SIZE)
                          .astype(int).tolist(),
        })
    return {"rooms": rooms_info}


def _build_adjacency_inputs(sample: dict, node_pred: dict) -> dict:
    """Build kwargs for AdjacencyDiff (``support_conditions='ncsl'``).

    The "syn" variant is what gets fed at inference (NodeDiff-predicted sizes
    and locations rather than GT).
    """
    max_num_rooms = MAX_NUM_ROOMS
    canvas_size = CANVAS_SIZE
    canvas_area = canvas_size * canvas_size

    boundary_rplan = sample["boundary_expand"]
    cond_boundary = (np.array(boundary_rplan).flatten().reshape(1, -1) / canvas_size * 2) - 1
    cond_boundary = np.repeat(cond_boundary, max_num_rooms, axis=0)

    door_rplan = sample["entrance_expand"]
    cond_door = (np.array(door_rplan).flatten().reshape(1, -1) / canvas_size * 2) - 1
    cond_door = np.repeat(cond_door, max_num_rooms, axis=0)

    rooms_info = node_pred["rooms"]
    num_rooms = len(rooms_info)

    cond_number = np.zeros((max_num_rooms, max_num_rooms))
    for r_i in range(num_rooms):
        cond_number[r_i] = _get_one_hot(r_i, max_num_rooms)

    cond_node = np.full((max_num_rooms, 4), -1.0)
    for r_i in range(num_rooms):
        cond_node[r_i] = [
            (int(rooms_info[r_i]["category"]) + 1) / 3 - 1,
            (rooms_info[r_i]["size"] / canvas_area * 2) - 1,
            (rooms_info[r_i]["location"][0] / canvas_size * 2) - 1,
            (rooms_info[r_i]["location"][1] / canvas_size * 2) - 1,
        ]

    atten_mask = np.ones((max_num_rooms, max_num_rooms))
    atten_mask[:num_rooms, :num_rooms] = 0
    padding_mask = np.zeros(max_num_rooms)
    padding_mask[num_rooms:] = 1

    cond_partial = np.full((max_num_rooms, max_num_rooms), -1.0)

    return {
        "name_fp": sample["name"],
        "num_rooms": num_rooms,
        "atten_mask": atten_mask,
        "padding_mask": padding_mask,
        "cond_boundary": cond_boundary,
        "cond_door": cond_door,
        "cond_number": cond_number,
        "cond_node": cond_node,
        "cond_partial": cond_partial,
        "syn_atten_mask": atten_mask,
        "syn_padding_mask": padding_mask,
        "syn_cond_boundary": cond_boundary,
        "syn_cond_door": cond_door,
        "syn_cond_number": cond_number,
        "syn_cond_node": cond_node,
        "syn_cond_partial": cond_partial,
    }


def _decode_adjacency_output(adj_tensor: np.ndarray, adj_kwargs: dict, node_pred: dict) -> dict:
    """Decode AdjacencyDiff's raw output (max_num_rooms x max_num_rooms) into
    a list of ``[a, b]`` adjacency edges.

    Mirrors ``adjacency_diff/scripts/sample.get_datajson_from_tensor``.
    """
    num_rooms = adj_kwargs["num_rooms"]
    block = adj_tensor[:num_rooms, :num_rooms]
    adj = np.zeros((num_rooms, num_rooms), dtype=np.uint8)
    adj[block > 0.5] = 1
    adj = np.triu(adj, k=0)
    edges = [[i, j] for i in range(num_rooms) for j in range(num_rooms) if adj[i, j] == 1]

    out = copy.deepcopy(node_pred)
    out["adjacencies"] = edges
    return out


def _build_partition_inputs(sample: dict, na_pred: dict) -> dict:
    """Build kwargs for PartitioningDiff (``support_conditions='ncsla'``)."""
    max_num_rooms = MAX_NUM_ROOMS
    canvas_size = CANVAS_SIZE
    canvas_area = canvas_size * canvas_size

    boundary_rplan = sample["boundary_expand"]
    cond_boundary = (np.array(boundary_rplan).flatten().reshape(1, -1) / canvas_size * 2) - 1
    cond_boundary = np.repeat(cond_boundary, max_num_rooms, axis=0)

    door_rplan = sample["entrance_expand"]
    cond_door = (np.array(door_rplan).flatten().reshape(1, -1) / canvas_size * 2) - 1
    cond_door = np.repeat(cond_door, max_num_rooms, axis=0)

    rooms_info = na_pred["rooms"]
    num_rooms = len(rooms_info)

    cond_number = np.zeros((max_num_rooms, max_num_rooms))
    for r_i in range(num_rooms):
        cond_number[r_i] = _get_one_hot(r_i, max_num_rooms)

    cond_node = np.full((max_num_rooms, 4), -1.0)
    for r_i in range(num_rooms):
        cond_node[r_i] = [
            (int(rooms_info[r_i]["category"]) + 1) / 3 - 1,
            (rooms_info[r_i]["size"] / canvas_area * 2) - 1,
            (rooms_info[r_i]["location"][0] / canvas_size * 2) - 1,
            (rooms_info[r_i]["location"][1] / canvas_size * 2) - 1,
        ]

    cond_adjacency = np.full((max_num_rooms, max_num_rooms), -1.0)
    for adja in na_pred.get("adjacencies", []):
        a, b = int(adja[0]), int(adja[1])
        cond_adjacency[a, b] = 1
        cond_adjacency[b, a] = 1

    atten_mask = np.ones((max_num_rooms, max_num_rooms))
    atten_mask[:num_rooms, :num_rooms] = 0
    padding_mask = np.zeros(max_num_rooms)
    padding_mask[num_rooms:] = 1

    cond_partial = np.full((max_num_rooms, 4), -1.0)

    return {
        "name_fp": sample["name"],
        "num_rooms": num_rooms,
        "atten_mask": atten_mask,
        "padding_mask": padding_mask,
        "cond_boundary": cond_boundary,
        "cond_door": cond_door,
        "cond_number": cond_number,
        "cond_node": cond_node,
        "cond_adjacency": cond_adjacency,
        "cond_partial": cond_partial,
        "syn_atten_mask": atten_mask,
        "syn_padding_mask": padding_mask,
        "syn_cond_boundary": cond_boundary,
        "syn_cond_door": cond_door,
        "syn_cond_number": cond_number,
        "syn_cond_node": cond_node,
        "syn_cond_adjacency": cond_adjacency,
        "syn_cond_partial": cond_partial,
    }


def _decode_partition_output(part_tensor: np.ndarray, part_kwargs: dict, na_pred: dict) -> dict:
    """Decode PartitioningDiff's raw output (max_num_rooms x 4) into the full
    DiffPlanner JSON record (with rooms[*].box and adjacencies).
    """
    num_rooms = part_kwargs["num_rooms"]
    boxes = np.round(((part_tensor[:num_rooms] + 1) / 2) * CANVAS_SIZE).astype(int)

    out = copy.deepcopy(na_pred)
    for i in range(num_rooms):
        out["rooms"][i]["box"] = boxes[i].tolist()
    return out


# =====================================================================
# Batched cascade runner
# =====================================================================

def _stack_kwargs(per_sample_kwargs: list[dict], device: th.device) -> dict:
    """Stack a list of per-sample condition dicts into a single batch dict
    of torch tensors (matching what each sub-model's transformer expects).
    """
    batch = {}
    keys = [k for k in per_sample_kwargs[0] if k not in ("name_fp", "num_rooms",
                                                         "num_rooms_gt")]
    for k in keys:
        arrs = [np.asarray(d[k]) for d in per_sample_kwargs]
        stacked = np.stack(arrs, axis=0)
        batch[k] = th.tensor(stacked, dtype=th.float32, device=device)
    return batch


def _sample_stage(model, diffusion, kwargs: dict, *, batch_size: int,
                  num_channels: int, max_num_rooms: int, use_ddim: bool):
    """Run one diffusion stage and return its final-step sample
    (shape: ``[batch, max_num_rooms, num_channels]`` after permute).
    """
    sample_fn = diffusion.ddim_sample_loop if use_ddim else diffusion.p_sample_loop
    # DP transformers expect the data tensor as (B, C, T) where T is room count;
    # see how each sample.py calls sample_fn with data_sample.shape.
    data_shape = (batch_size, num_channels, max_num_rooms)
    with th.no_grad():
        sample = sample_fn(
            model,
            data_shape,
            clip_denoised=True,
            model_kwargs=kwargs,
        )
    # sample shape: [time_steps, batch, C, T]; we keep only the last step
    # then permute to [batch, T, C].
    return sample[-1].permute(0, 2, 1).cpu().numpy()


def run_cascade_batch(
    samples: list[dict],
    *,
    node_bundle, adj_bundle, part_bundle,
    use_ddim: bool,
    device: th.device,
) -> list[dict]:
    """End-to-end ``b + nc`` cascade for a list of samples (one batch).

    Returns one full DiffPlanner JSON record per sample (with
    ``rooms[*].box`` and ``adjacencies`` filled in).  Post-processing is
    intentionally done one sample at a time downstream because
    ``output/post_processing.py`` is per-sample only.
    """
    node_model, node_diffusion, _ = node_bundle
    adj_model, adj_diffusion, _ = adj_bundle
    part_model, part_diffusion, _ = part_bundle

    # ---- Stage 1: NodeDiff ----
    node_kwargs_list = [_build_node_inputs(s) for s in samples]
    node_batch_kwargs = _stack_kwargs(node_kwargs_list, device)
    node_sample = _sample_stage(
        node_model, node_diffusion, node_batch_kwargs,
        batch_size=len(samples), num_channels=3,
        max_num_rooms=MAX_NUM_ROOMS, use_ddim=use_ddim,
    )
    node_preds = [
        _decode_node_output(node_sample[i], node_kwargs_list[i])
        for i in range(len(samples))
    ]

    # ---- Stage 2: AdjacencyDiff ----
    adj_kwargs_list = [
        _build_adjacency_inputs(samples[i], node_preds[i])
        for i in range(len(samples))
    ]
    adj_batch_kwargs = _stack_kwargs(adj_kwargs_list, device)
    adj_sample = _sample_stage(
        adj_model, adj_diffusion, adj_batch_kwargs,
        batch_size=len(samples), num_channels=MAX_NUM_ROOMS,
        max_num_rooms=MAX_NUM_ROOMS, use_ddim=use_ddim,
    )
    na_preds = [
        _decode_adjacency_output(adj_sample[i], adj_kwargs_list[i], node_preds[i])
        for i in range(len(samples))
    ]

    # ---- Stage 3: PartitioningDiff ----
    part_kwargs_list = [
        _build_partition_inputs(samples[i], na_preds[i])
        for i in range(len(samples))
    ]
    part_batch_kwargs = _stack_kwargs(part_kwargs_list, device)
    part_sample = _sample_stage(
        part_model, part_diffusion, part_batch_kwargs,
        batch_size=len(samples), num_channels=4,
        max_num_rooms=MAX_NUM_ROOMS, use_ddim=use_ddim,
    )

    # Re-attach the immutable boundary fields from the input sample so the
    # final record matches DiffPlanner's per-sample JSON shape.
    full_records = []
    for i, sample in enumerate(samples):
        rec = _decode_partition_output(part_sample[i], part_kwargs_list[i], na_preds[i])
        rec["name"] = sample["name"]
        rec["boundary"] = sample["boundary"]
        rec["boundary_expand"] = sample["boundary_expand"]
        rec["entrance_expand"] = sample["entrance_expand"]
        full_records.append(rec)
    return full_records


# =====================================================================
# Post-processing wrapper (with raw-fallback)
# =====================================================================

def _post_process_record(record: dict) -> tuple[dict, str]:
    """Run DiffPlanner's ``output/post_processing.main(...)`` on one record.

    Returns ``(aligned_record_or_record, status)`` where ``status`` is one of
    ``"aligned"`` (normal) or ``"raw_fallback"`` (alignment failed; raw
    PartitioningDiff boxes are kept and a synthetic ``r_boundary_aligned`` is
    derived from those raw rectangles so downstream metrics still work).
    """
    import post_processing as pp                # imported here so sys.path is set up
    try:
        aligned = pp.main(copy.deepcopy(record), support_boundary=True)
        return aligned, "aligned"
    except Exception:
        fallback = copy.deepcopy(record)
        for r in fallback["rooms"]:
            box = r.get("box")
            if box and len(box) == 4:
                x0, y0, x1, y1 = box
                r["box_aligned"] = [int(x0), int(y0), int(x1), int(y1)]
                r["r_boundary_aligned"] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                r["order_aligned"] = int(r.get("id", 0)) + 1
            else:
                r["box_aligned"] = None
                r["r_boundary_aligned"] = None
                r["order_aligned"] = 0
        fallback["adjacencies_aligned"] = list(fallback.get("adjacencies", []))
        return fallback, "raw_fallback"


# =====================================================================
# Polygon helpers + metric inputs
# =====================================================================

def _shapely_polygon_safe(coords: list[list[float]]) -> Optional[ShapelyPolygon]:
    if not coords or len(coords) < 3:
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


def _aligned_polygons_from_record(record: dict) -> list[dict]:
    """Extract one polygon (+ category) per room from a post-processed record.

    Prefers ``r_boundary_aligned`` (precise, possibly non-rectangular region
    cut by ``post_processing.get_room_boundary``) and falls back to
    ``box_aligned`` / ``box``.

    NOTE: this is the *forgiving* extractor used by ``delta`` and
    ``area_proportion_error`` -- it keeps a room in the list even when
    post-processing dropped its precise polygon (we substitute the raw
    bounding box). To count only the rooms that would actually appear in the
    rendered floor plan, use :func:`_visible_room_categories` below.
    """
    rooms = []
    for r in record.get("rooms", []):
        coords = r.get("r_boundary_aligned")
        if not coords or len(coords) < 3:
            box = r.get("box_aligned") or r.get("box")
            if box and len(box) == 4:
                x0, y0, x1, y1 = box
                coords = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        poly = _shapely_polygon_safe(coords) if coords else None
        if poly is None:
            continue
        rooms.append({
            "category": int(r.get("category", -1)),
            "polygon": poly,
            "area": poly.area,
            "perimeter": poly.length,
        })
    return rooms


# =====================================================================
# Adjacency-graph (shared-wall) construction + GED
#
# DiffPlanner's *adjacency graph* is the spatial-topological graph where
# two rooms are connected iff they share a wall segment. This is fundamentally
# different from House-Diffusion's *access graph*, which is door-mediated
# (two rooms connect only if an interior-door entry pairs them via ``ed_rm``).
#
# In the DP record this lives in ``adjacencies_aligned`` (post-processing
# output) -- a list of ``[i, j]`` pairs of room indices into ``record["rooms"]``.
# We label each node by its DiffPlanner category id (0-5), matching how
# ``_visible_room_categories`` already labels rooms, so GED's ``node_match``
# requires both an isomorphic structure AND matching room-type sequence to
# yield 0 -- the same semantics as the House-Diffusion access-graph GED.
# =====================================================================

def _build_adjacency_graph_from_record(record: dict, *, source: str = "auto") -> "Any":
    """Build the spatial wall-sharing graph from one DP record.

    Args:
        record: A DP per-sample dict with ``rooms`` and ``adjacencies_aligned``
            (or ``adjacencies`` as fallback).
        source: ``"aligned"`` to use ``adjacencies_aligned`` only,
            ``"raw"`` to use ``adjacencies`` only, or ``"auto"`` (default)
            to prefer ``adjacencies_aligned`` and fall back to ``adjacencies``.

    Returns:
        A ``networkx.Graph`` with one labeled node per VISIBLE room
        (rooms whose ``r_boundary_aligned`` polygon survives) and one
        undirected edge per adjacency that connects two visible rooms.
        Lost rooms (and edges that touch them) are dropped, mirroring
        :func:`_visible_room_categories`.
    """
    import networkx as nx

    G = nx.Graph()
    rooms = record.get("rooms", []) or []

    # Decide which adjacency list to use.
    if source == "aligned":
        adj_list = record.get("adjacencies_aligned", []) or []
    elif source == "raw":
        adj_list = record.get("adjacencies", []) or []
    else:
        adj_list = record.get("adjacencies_aligned")
        if not adj_list:
            adj_list = record.get("adjacencies", []) or []

    # Compute the visible-room mask (same notion as _visible_room_categories
    # / _aligned_polygons_from_record): a room is visible iff its post-aligned
    # polygon is a non-degenerate Shapely polygon. For raw_fallback records,
    # ``r_boundary_aligned`` is a synthesized rectangle from the raw box, so
    # all rooms are "visible" -- consistent with the rest of the pipeline.
    visible: dict[int, int] = {}
    for i, r in enumerate(rooms):
        coords = r.get("r_boundary_aligned")
        poly = _shapely_polygon_safe(coords) if coords else None
        if poly is None:
            continue
        visible[i] = int(r.get("category", -1))

    for i, cat in visible.items():
        G.add_node(i, label=cat)

    for edge in adj_list:
        if not (isinstance(edge, (list, tuple)) and len(edge) >= 2):
            continue
        try:
            a, b = int(edge[0]), int(edge[1])
        except (TypeError, ValueError):
            continue
        if a == b:
            continue
        if a in visible and b in visible:
            G.add_edge(a, b)
    return G


def _build_adjacency_graph_from_gt(gt_record: dict) -> "Any":
    """Build the spatial wall-sharing graph from a GT sample dict.

    GT samples in ``data_test.json`` use ``adjacencies`` (no _aligned suffix)
    and have valid ``box`` / ``r_boundary`` for every room, so we treat all
    rooms as visible and label by their declared category.
    """
    import networkx as nx

    G = nx.Graph()
    rooms = gt_record.get("rooms", []) or []
    for i, r in enumerate(rooms):
        G.add_node(i, label=int(r.get("category", -1)))
    for edge in gt_record.get("adjacencies", []) or []:
        if not (isinstance(edge, (list, tuple)) and len(edge) >= 2):
            continue
        try:
            a, b = int(edge[0]), int(edge[1])
        except (TypeError, ValueError):
            continue
        if a == b or a not in G or b not in G:
            continue
        G.add_edge(a, b)
    return G


def _adj_ged_worker(pair) -> float:
    """Compute GED for a single (G_pred, G_true, idx, upper_bound) tuple.

    Same shape as ``benchmark.src.evaluate_housediffusion._ged_worker`` so a
    parallel runner can be reused unchanged.
    """
    G_pred, G_true, sample_idx, upper_bound = pair
    ged, _ = ged_between_graphs(
        G_pred, G_true, sample_idx,
        upper_bound=upper_bound,
    )
    return ged if ged is not None else float("nan")


def _parallel_adj_ged(ged_pairs: list, desc: str = "Computing adjacency-graph GED") -> list[float]:
    """Resolve queued GED pairs in parallel via joblib (one core per pair)."""
    import multiprocessing
    from joblib import Parallel, delayed

    if not ged_pairs:
        return []
    num_cores = multiprocessing.cpu_count()
    print(f"\nComputing adjacency-graph GED for {len(ged_pairs)} pairs "
          f"using {num_cores} cores...")
    return Parallel(n_jobs=num_cores)(
        delayed(_adj_ged_worker)(pair)
        for pair in tqdm(ged_pairs, desc=desc)
    )


def _visible_room_categories(record: dict) -> tuple[list[int], int]:
    """Strict counterpart to :func:`_aligned_polygons_from_record`.

    Returns ``(categories, lost)`` where:
      * ``categories`` -- the room categories whose ``r_boundary_aligned`` is a
        valid (non-degenerate) polygon, i.e. the rooms that ``vis_floorplan``
        would actually paint.
      * ``lost`` -- the number of input rooms whose post-processed polygon is
        missing/degenerate. Increments on:
          - ``Empty box i!!!`` warnings (regularize_fp dropped the room),
          - ``Warning: Room i has an empty intersection ...`` warnings
            (a larger sibling room covered it during get_room_boundary),
          - any other case where Shapely refused to build a polygon.

    For raw-fallback records (post-processing aborted), ``r_boundary_aligned``
    is the raw bounding box -- always valid -- so ``lost`` will be 0 for those
    samples (they are tracked separately via ``raw_fallback_count``).
    """
    cats: list[int] = []
    lost = 0
    for r in record.get("rooms", []):
        coords = r.get("r_boundary_aligned")
        poly = _shapely_polygon_safe(coords) if coords else None
        if poly is None:
            lost += 1
            continue
        cats.append(int(r.get("category", -1)))
    return cats, lost


def _polygons_from_gt_record(record: dict) -> list[dict]:
    """Same as :func:`_aligned_polygons_from_record` but for GT samples in
    ``data_test.json``.  GT records use ``r_boundary`` as the precise polygon
    and fall back to ``box``.
    """
    rooms = []
    for r in record.get("rooms", []):
        coords = r.get("r_boundary")
        if not coords or len(coords) < 3:
            box = r.get("box")
            if box and len(box) == 4:
                x0, y0, x1, y1 = box
                coords = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        poly = _shapely_polygon_safe(coords) if coords else None
        if poly is None:
            continue
        rooms.append({
            "category": int(r.get("category", -1)),
            "polygon": poly,
            "area": poly.area,
            "perimeter": poly.length,
        })
    return rooms


def _to_similarity_list(rooms: list[dict]) -> list[tuple[str, float, float]]:
    """``[(merge_id, perimeter, square_perimeter), ...]`` for
    ``utils.metrics.compute_room_similarity_score``.
    """
    out = []
    for r in rooms:
        merge_id = DP_CATEGORY_TO_MERGE_ID.get(r["category"], str(r["category"]))
        perim = max(r["perimeter"], 1e-6)
        square_perim = 4 * math.sqrt(max(r["area"], 1e-6))
        out.append((merge_id, perim, square_perim))
    return out


def _to_area_list(rooms: list[dict]) -> list[tuple[str, float]]:
    """``[(merge_id, area), ...]`` for
    ``utils.metrics.compute_room_area_proportion_error``.
    """
    return [
        (DP_CATEGORY_TO_MERGE_ID.get(r["category"], str(r["category"])), r["area"])
        for r in rooms
    ]


def _boundary_polygon(record: dict) -> Optional[ShapelyPolygon]:
    """Build the building outline as a Shapely polygon (defensive)."""
    coords = record.get("boundary_expand") or [
        [p[0], p[1]] for p in record.get("boundary", []) if len(p) >= 2
    ]
    return _shapely_polygon_safe(coords)


def compute_boundary_geometry_metrics(record: dict) -> dict[str, float]:
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
    """
    boundary = _boundary_polygon(record)
    if boundary is None or boundary.area <= 0:
        return {"gap_ratio": float("nan"),
                "outside_ratio": float("nan"),
                "overlap_ratio": float("nan")}

    rooms = _aligned_polygons_from_record(record)
    F_area = boundary.area
    if not rooms:
        return {"gap_ratio": 1.0, "outside_ratio": 0.0, "overlap_ratio": 0.0}

    polys = [r["polygon"] for r in rooms]
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

def evaluate_sample(
    pred_record: dict,
    gt_record: dict,
    *,
    pp_status: str,
    results: dict,
    ged_upper_bound: int = 40,
) -> None:
    """Compute every metric for one sample and append into ``results``.

    Metrics computed (per sample):
      - delta (room similarity score)
      - area_proportion_error
      - gap_ratio, outside_ratio, overlap_ratio (boundary-normalized)
      - room accuracy (combined / count-only / type-only)
      - adjacency-graph GED (queued via ``adj_ged_pairs``; resolved in bulk
        after all samples are scored, see :func:`_parallel_adj_ged`)

    Side-effect counters: raw_fallback_count, lost_room_count,
    samples_with_lost_rooms (== ``hidden_count`` in HD nomenclature).
    """
    pred_rooms = _aligned_polygons_from_record(pred_record)
    gt_rooms = _polygons_from_gt_record(gt_record)

    sim_pred = _to_similarity_list(pred_rooms)
    sim_gt = _to_similarity_list(gt_rooms)
    delta = compute_room_similarity_score(sim_pred, sim_gt)

    area_pred = _to_area_list(pred_rooms)
    area_gt = _to_area_list(gt_rooms)
    area_metric = compute_room_area_proportion_error(area_pred, area_gt)

    geom = compute_boundary_geometry_metrics(pred_record)

    # Room accuracy: only count rooms whose post-aligned polygon survives
    # (the ones the renderer would actually paint). Lost rooms are tracked
    # explicitly via lost_room_count below.
    pred_cats_visible, lost = _visible_room_categories(pred_record)
    gt_cats = [int(r.get("category", -1)) for r in gt_record.get("rooms", [])]

    pred_labels = [DP_CATEGORY_TO_MERGE_ID.get(c, str(c)) for c in pred_cats_visible]
    gt_labels = [DP_CATEGORY_TO_MERGE_ID.get(c, str(c)) for c in gt_cats]
    rt_metric = room_type_accuracy(gt_labels, pred_labels, strict=True)

    results["similarity_scores"].append(float(delta) if delta is not None else float("nan"))
    results["area_proportion_errors"].append(float(area_metric["area_proportion_error"]))
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
    if pp_status == "raw_fallback":
        results["raw_fallback_count"] += 1

    # Queue an adjacency-graph GED pair for batched parallel resolution.
    G_pred_adj = _build_adjacency_graph_from_record(pred_record, source="auto")
    G_true_adj = _build_adjacency_graph_from_gt(gt_record)
    sample_idx = len(results["similarity_scores"]) - 1
    results["adj_ged_pairs"].append(
        (G_pred_adj, G_true_adj, sample_idx, ged_upper_bound)
    )


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
        "raw_fallback_count": 0,
        # Adjacency-graph (shared-wall) GED. Pairs are queued per-sample and
        # resolved in parallel after all samples are scored to avoid
        # spawning joblib workers inside the main per-sample loop.
        "adj_ged_pairs": [],
        "adj_geds": [],
    }


# =====================================================================
# Cache I/O (same shape as evaluate_housediffusion's pickle cache)
# =====================================================================

def save_generations_diffplanner(filepath: Path, metadata: dict, samples: list[dict]) -> None:
    """Persist the post-processed DiffPlanner records for later metric reruns."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "samples": samples}
    with filepath.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"Saved {len(samples)} samples to {filepath} ({size_mb:.1f} MB)")


def load_generations_diffplanner(filepath: Path) -> dict:
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
        description="DiffPlanner b+nc cascade benchmark on RPLAN test split",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data_path", type=Path, required=True,
                   help="Path to DiffPlanner data_test.json (full RPLAN test split).")
    p.add_argument("--diffplanner_root", type=Path,
                   default=Path("/home/hesam/REPOs/DiffPlanner"),
                   help="Path to the DiffPlanner repository root (where node_diff/ etc. live).")

    p.add_argument("--node_checkpoint", type=Path, default=None,
                   help="NodeDiff (b+nc) checkpoint. Defaults to "
                        "<diffplanner_root>/node_diff/scripts/trained_model/bnc_model300000.pt")
    p.add_argument("--adjacency_checkpoint", type=Path, default=None,
                   help="AdjacencyDiff (b+ncsl) checkpoint. Defaults to "
                        "<diffplanner_root>/adjacency_diff/scripts/trained_model/bncsl_model300000.pt")
    p.add_argument("--partition_checkpoint", type=Path, default=None,
                   help="PartitioningDiff (b+ncsla) checkpoint. Defaults to "
                        "<diffplanner_root>/partitioning_diff/scripts/trained_model/bncsla_model300000.pt")

    p.add_argument("--batch_size", type=int, default=128,
                   help="Cascade batch size. All three stages share the same batch.")
    p.add_argument("--use_ddim", action="store_true",
                   help="Use DDIM sampling (much faster, lower fidelity) instead of "
                        "DiffPlanner's default p_sample_loop.")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap the number of samples evaluated (default: all).")
    p.add_argument("--shard", type=str, default=None,
                   help='Shard spec "K/N" -- evaluate only every Nth sample starting at K.')
    p.add_argument("--include_ids", type=Path, default=None,
                   help="Optional file with one sample name per line; only those IDs "
                        "are evaluated.")

    p.add_argument("--save_generations", type=Path, default=None,
                   help="If set, write a pickle of post-processed DiffPlanner "
                        "records (one per sample) for later metric reruns.")
    p.add_argument("--load_generations", type=Path, default=None,
                   help="If set, skip GPU work entirely and compute metrics from a "
                        "previously saved cache.")

    p.add_argument("--out", type=Path, default=Path("benchmark/eval_results_diffplanner"),
                   help="Output folder for results.npz / results.txt.")
    p.add_argument("--device", type=str, default=None,
                   help='Force "cuda" / "cpu". Defaults to cuda if available.')
    return p.parse_args()


def _resolve_default_checkpoints(args: argparse.Namespace) -> None:
    root = args.diffplanner_root
    if args.node_checkpoint is None:
        args.node_checkpoint = root / "node_diff/scripts/trained_model/bnc_model300000.pt"
    if args.adjacency_checkpoint is None:
        args.adjacency_checkpoint = root / "adjacency_diff/scripts/trained_model/bncsl_model300000.pt"
    if args.partition_checkpoint is None:
        args.partition_checkpoint = root / "partitioning_diff/scripts/trained_model/bncsla_model300000.pt"


def _select_samples(samples: list[dict], args) -> list[dict]:
    """Apply --include_ids / --shard / --max_samples filters in that order."""
    if args.include_ids:
        with args.include_ids.open() as f:
            wanted = {line.strip() for line in f if line.strip()}
        samples = [s for s in samples if str(s.get("name")) in wanted]
        print(f"--include_ids filter kept {len(samples)} samples")
    if args.shard:
        try:
            k, n = (int(x) for x in args.shard.split("/"))
        except ValueError:
            raise SystemExit(f"--shard must be 'K/N', got {args.shard!r}")
        samples = [s for i, s in enumerate(samples) if i % n == k]
        print(f"--shard {args.shard} kept {len(samples)} samples")
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    return samples


def _build_report_kwargs(results: dict) -> dict:
    """Package per-sample arrays for ``utils.report_results.report_results``.

    The keys ``accuracies`` / ``room_count_accuracies`` / ``room_type_accuracies``
    are well-known to ``report_results`` and trigger the dedicated room-accuracy
    block (which also consumes the ``hidden_count`` extra to print the
    "Hidden rooms" rate).
    """
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
            "adj_geds": ("GED (adjacency graph, shared walls)",
                         np.array(results["adj_geds"], dtype=float)
                         if results["adj_geds"] else np.array([], dtype=float)),
        },
        "room_counts": np.array(results["per_sample_room_counts"]),
        "extra": {
            "raw_fallback_count": int(results["raw_fallback_count"]),
            "lost_room_count": int(results["lost_room_count"]),
            # Re-use the HD nomenclature so report_results prints the
            # "Hidden rooms: X (Y.Y%)" line under Room Accuracy.
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

    # ----- Load GT JSON ---------------------------------------------------
    if not args.data_path.is_file():
        raise SystemExit(f"--data_path not found: {args.data_path}")
    print(f"Loading dataset from {args.data_path}")
    with args.data_path.open() as f:
        full_gt = json.load(f)
    if not isinstance(full_gt, list):
        raise SystemExit("data_test.json must be a JSON list.")
    gt_by_name = {str(s["name"]): s for s in full_gt}
    samples = _select_samples(full_gt, args)
    print(f"Will evaluate {len(samples)} samples")

    results = _make_results_dict()

    # ----- LOAD MODE: skip cascade, compute metrics from cache ------------
    if args.load_generations:
        cache = load_generations_diffplanner(args.load_generations)
        cached_records = {str(s["name"]): s for s in cache["samples"]}
        for sample in tqdm(samples, desc="Metrics (cached)"):
            name = str(sample["name"])
            rec = cached_records.get(name)
            if rec is None:
                continue
            evaluate_sample(
                rec, gt_by_name[name],
                pp_status=rec.get("__pp_status", "aligned"),
                results=results,
            )

    # ----- GENERATE MODE: run the 3-stage cascade -------------------------
    else:
        _resolve_default_checkpoints(args)
        for ckpt in (args.node_checkpoint, args.adjacency_checkpoint, args.partition_checkpoint):
            if not ckpt.is_file():
                raise SystemExit(f"Checkpoint not found: {ckpt}")

        device = th.device(args.device) if args.device else th.device(
            "cuda" if th.cuda.is_available() else "cpu"
        )
        print(f"Using device: {device}")

        _add_diffplanner_to_path(args.diffplanner_root)
        # post_processing.py loads ../../dataset/dataset_json/data_test.json
        # at import time; make sure that path exists.
        dp_data = args.diffplanner_root / "dataset/dataset_json/data_test.json"
        if not dp_data.is_file():
            print(f"NOTE: {dp_data} missing -- symlinking your --data_path there "
                  "so DiffPlanner's helpers can read it.")
            dp_data.parent.mkdir(parents=True, exist_ok=True)
            try:
                dp_data.symlink_to(args.data_path)
            except OSError as exc:
                raise SystemExit(
                    f"Failed to symlink {dp_data} -> {args.data_path}: {exc}. "
                    "Please copy/symlink data_test.json manually."
                )

        print(f"Loading NodeDiff       checkpoint: {args.node_checkpoint}")
        node_bundle = _load_one_stage(
            sub_pkg="node_diff", support_conditions="nc",
            use_ddim=args.use_ddim, batch_size=args.batch_size,
            model_path=args.node_checkpoint, set_name="test", device=device,
        )
        print(f"Loading AdjacencyDiff  checkpoint: {args.adjacency_checkpoint}")
        adj_bundle = _load_one_stage(
            sub_pkg="adjacency_diff", support_conditions="ncsl",
            use_ddim=args.use_ddim, batch_size=args.batch_size,
            model_path=args.adjacency_checkpoint, set_name="test", device=device,
        )
        print(f"Loading PartitioningDiff checkpoint: {args.partition_checkpoint}")
        part_bundle = _load_one_stage(
            sub_pkg="partitioning_diff", support_conditions="ncsla",
            use_ddim=args.use_ddim, batch_size=args.batch_size,
            model_path=args.partition_checkpoint, set_name="test", device=device,
        )

        cache_samples: list[dict] = []
        n_done = 0
        try:
            for batch_start in tqdm(range(0, len(samples), args.batch_size),
                                    desc="Cascade"):
                batch = samples[batch_start: batch_start + args.batch_size]
                t0 = time.time()
                full_records = run_cascade_batch(
                    batch,
                    node_bundle=node_bundle,
                    adj_bundle=adj_bundle,
                    part_bundle=part_bundle,
                    use_ddim=args.use_ddim,
                    device=device,
                )
                t_cascade = time.time() - t0

                for rec, sample in zip(full_records, batch):
                    aligned, status = _post_process_record(rec)
                    aligned["__pp_status"] = status
                    if args.save_generations is not None:
                        cache_samples.append(aligned)
                    evaluate_sample(
                        aligned, gt_by_name[str(sample["name"])],
                        pp_status=status, results=results,
                    )
                    n_done += 1

                if batch_start == 0:
                    print(f"  first batch ({len(batch)} samples) "
                          f"cascade={t_cascade:.1f}s  "
                          f"raw_fallback so far={results['raw_fallback_count']}")
        except KeyboardInterrupt:
            print("\nInterrupted by user; saving partial results...")

        if args.save_generations is not None:
            metadata = {
                "model": "DiffPlanner b+nc cascade",
                "node_checkpoint": str(args.node_checkpoint),
                "adjacency_checkpoint": str(args.adjacency_checkpoint),
                "partition_checkpoint": str(args.partition_checkpoint),
                "data_path": str(args.data_path),
                "use_ddim": args.use_ddim,
                "batch_size": args.batch_size,
                "num_samples": len(cache_samples),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "version": "1.0",
            }
            save_generations_diffplanner(args.save_generations, metadata, cache_samples)

    # ----- Report ---------------------------------------------------------
    if not results["similarity_scores"]:
        print("No samples were scored -- nothing to report.")
        return 1

    n = len(results["similarity_scores"])
    print(f"\nScored {n} samples; raw_fallback (post-processing failed) "
          f"on {results['raw_fallback_count']} of them.")
    print(f"Lost rooms: {results['lost_room_count']:,} total across "
          f"{results['samples_with_lost_rooms']:,} of {n:,} samples "
          f"({100.0 * results['samples_with_lost_rooms'] / n:.1f}% of samples "
          f"have at least one room dropped by post-processing).")

    # Resolve queued adjacency-graph GED pairs in parallel. Each per-sample
    # entry was appended in scoring order, so positional alignment with
    # similarity_scores / area_proportion_errors / etc. is preserved.
    if results["adj_ged_pairs"] and not results["adj_geds"]:
        try:
            results["adj_geds"] = _parallel_adj_ged(results["adj_ged_pairs"])
        except Exception as exc:  # pragma: no cover -- defensive
            print(f"WARNING: Parallel adjacency-graph GED failed ({exc}); "
                  f"leaving GED column empty.")
            results["adj_geds"] = []

    report_results(
        str(args.out),
        num_variations=1,
        checkpoint_path=str(args.node_checkpoint) if not args.load_generations else None,
        data_path=str(args.data_path),
        **_build_report_kwargs(results),
    )
    print(f"\nWrote {args.out / 'results.txt'} and {args.out / 'results.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
