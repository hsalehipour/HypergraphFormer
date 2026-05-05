#!/usr/bin/env python3
"""
Few-shot evaluator for DiffPlanner-format LLM generation.

DiffPlanner (Wang & Pajarola, IEEE TVCG 2025) frames boundary-constrained floor
plan generation as a vector-to-vector task: given a vector boundary (with
entrance) and optional coarse conditions (room number, categories, sizes,
locations, adjacencies), the model emits a set of axis-aligned room boxes plus
metadata (category, size, location, optional adjacencies). This evaluator asks
an LLM to play the role of DiffPlanner under the
``DiffPlanner^b_III`` setting (boundary + room **count** + room **categories**
as input; the model must invent the boxes and adjacencies).

Why the choices below match the DiffPlanner paper
-------------------------------------------------
* ``support_boundary = True``  → the boundary polygon (with entrance) is in
  the prompt verbatim, on the same 256x256 canvas DiffPlanner uses.
* ``support_conditions = 'nc'`` → we tell the LLM how many rooms (`Rn`) and
  which categories (`Rc`) are required. In DiffPlanner this is enforced
  structurally; here we ask the LLM to honor it and *measure* whether it
  does.
* ``support_partial = False`` → no per-element user-fixed sub-inputs.

Output schema (the LLM's contract)
---------------------------------
The LLM emits a JSON object whose only required key is ``rooms``, a list of
length equal to the requested room count. Each entry is::

    {
      "id": int,                # 0-based, in the order rooms were requested
      "category": int,          # 0=Living, 1=Bedroom, 2=Kitchen, 3=Bathroom,
                                # 4=Balcony, 5=Storage  (DiffPlanner's scheme)
      "box":     [x0,y0,x1,y1], # integer pixels in [0,256], x0<x1, y0<y1
      "location":[cx,cy],       # integer center; if missing it's derived from box
      "size":    int            # area in px²; if missing it's derived from box
    }

Adjacencies are reconstructed from the boxes (touching/overlapping pairs) -
the LLM does NOT have to emit them. Boundary, ``boundary_expand`` and
``entrance_expand`` are echoed back from the input, so the saved per-sample
JSON is directly compatible with ``DiffPlanner/output/post_processing.py``
and ``DiffPlanner/output/visualization.py``.

Inputs
------
``--dataset-json`` (required) is the monolithic 12K-sample DiffPlanner
``data_test.json`` (a JSON list of sample dicts with keys
``name / boundary / boundary_expand / entrance_expand / rooms /
adjacencies``). Few-shot examples are drawn from the same file (with the
target sample excluded).

Outputs (same pattern as ``benchmark/src/fewshot_housediffusion.py``)
--------------------------------------------------------------------
``results/run.json``           Aggregate report (metadata, statistics, every
                               per-sample record incl. ``generated_dp``).
``results/run_partial.json``   Incremental snapshot for crash-resume via
                               ``--checkpoint``.
``results/run_dp/<name>.json`` ONE DiffPlanner-shaped JSON per success,
                               written incrementally (overridable with
                               ``--per-sample-dir``).

Usage
-----
    export OPENAI_API_KEY=...
    python benchmark/src/fewshot_diffplanner.py \\
        --dataset-json dataset/rplan/dataset_diffplanner/data_test.json \\
        --num-examples 3 --max-samples 100 \\
        --output results_dp_gpt4_3shot.json
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import re
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from tqdm import tqdm

from scripts.generators import GPTGenerator

# The few-shot script reuses the metric pipeline from the main DiffPlanner
# benchmark so the per-sample numbers it reports (delta, area_proportion_error,
# the boundary geometry triplet, the room-accuracy triplet, and the lost-room
# counters) match ``evaluate_diffplanner.py`` byte-for-byte.
from benchmark.src.evaluate_diffplanner import (
    DP_CATEGORY_TO_MERGE_ID,
    _add_diffplanner_to_path,
    _adj_ged_worker,
    _aligned_polygons_from_record,
    _build_report_kwargs,
    _make_results_dict,
    _parallel_adj_ged,
    _post_process_record as _dp_post_process_record,
    _visible_room_categories,
    compute_boundary_geometry_metrics,
)
from benchmark.src.evaluate_diffplanner import evaluate_sample as _dp_evaluate_sample
from utils.report_results import report_results

logger = logging.getLogger(__name__)


# ===================================================================
# DiffPlanner constants & system / format-rules text
# ===================================================================

CANVAS_SIZE = 256
MAX_ROOMS = 8

DP_CATEGORY_TO_NAME: dict[int, str] = {
    0: "LivingRoom",
    1: "Bedroom",
    2: "Kitchen",
    3: "Bathroom",
    4: "Balcony",
    5: "Storage",
}
DP_NAME_TO_CATEGORY: dict[str, int] = {v: k for k, v in DP_CATEGORY_TO_NAME.items()}

DP_SYSTEM_MESSAGE = (
    "You are an expert architect generating apartment floor plans in the "
    "DiffPlanner vector format. You always respond with a single JSON object "
    "containing the predicted rooms - no commentary, no markdown."
)

DP_FORMAT_RULES_TEXT = (
    "## DiffPlanner OUTPUT FORMAT (boundary-constrained, conditions = boundary + Rn + Rc)\n"
    f"- Canvas is {CANVAS_SIZE}x{CANVAS_SIZE} integer pixels. All coordinates lie in [0, {CANVAS_SIZE}].\n"
    "- The boundary polygon (and entrance rectangle) are GIVEN inputs; do NOT change them.\n"
    "- The room COUNT and the ORDERED LIST of categories are also GIVEN - your output MUST contain "
    "exactly that many rooms, in the same order, with the same categories.\n"
    "- Room categories use DiffPlanner's 6-class scheme:\n"
    "    0=LivingRoom, 1=Bedroom, 2=Kitchen, 3=Bathroom, 4=Balcony, 5=Storage.\n"
    "- Each room is an AXIS-ALIGNED bounding box [x0, y0, x1, y1] (top-left, bottom-right) "
    "with integer pixel coordinates and x0 < x1, y0 < y1.\n"
    "- Room boxes must lie inside the boundary polygon and SHOULD tile it without large gaps "
    "or large overlaps (small overlaps are tolerated; DiffPlanner's post-processing aligns them later).\n"
    "- Living rooms touch the entrance and tend to be central / largest.\n"
    "- Bathrooms / kitchens / storage are usually small and adjacent to the living room or to a bedroom.\n"
    "- Balconies typically sit on a protruding edge of the boundary.\n"
    "\n"
    "## RESPONSE SCHEMA (return ONLY this JSON object):\n"
    "```\n"
    "{\n"
    '  "rooms": [\n'
    '    {"id": 0, "category": <int 0-5>, "box": [x0, y0, x1, y1], '
    '"location": [cx, cy], "size": <int area>},\n'
    "    ...\n"
    "  ]\n"
    "}\n"
    "```\n"
    "- ``location`` is the integer (cx, cy) center of the box. If you omit it, the evaluator "
    "will derive it from ``box``. Same for ``size`` (= (x1-x0) * (y1-y0)).\n"
    "- DO NOT include the boundary, entrance or adjacencies - the evaluator re-uses the input "
    "boundary and reconstructs adjacencies from the boxes."
)

DP_REQUIREMENTS_TMPL = (
    "## REQUIREMENTS FOR THIS SAMPLE:\n"
    "- Generate exactly {num_rooms} rooms.\n"
    "- Categories (in this exact order): {categories}.\n"
    "- All boxes must be inside the boundary polygon shown above.\n"
    "- Output ONLY the JSON object described in the schema; no prose, no markdown fences."
)


# ===================================================================
# Sample / few-shot example construction from the monolithic JSON
# ===================================================================

def _polygon_from_pairs(pairs: list[list[int]]) -> "Any":
    """Build a shapely Polygon from a list of [x,y] pairs (defensive)."""
    from shapely.geometry import Polygon

    poly = Polygon(pairs)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def _boundary_pairs(sample: dict) -> list[list[int]]:
    """Return [[x,y], ...] boundary corner list, preferring ``boundary_expand``."""
    if isinstance(sample.get("boundary_expand"), list) and sample["boundary_expand"]:
        return [[int(p[0]), int(p[1])] for p in sample["boundary_expand"]]
    raw = sample.get("boundary", [])
    return [[int(p[0]), int(p[1])] for p in raw]


def _entrance_pairs(sample: dict) -> list[list[int]]:
    raw = sample.get("entrance_expand", [])
    return [[int(p[0]), int(p[1])] for p in raw]


def build_input_view(sample: dict) -> dict:
    """Build the LLM-visible 'input' dict for a sample.

    This is what the model sees as the 'task' \u2014 i.e. boundary + entrance +
    requested room count + ordered room categories. The original sample's
    rooms / adjacencies are NOT included here.
    """
    rooms = sample.get("rooms", []) or []
    categories = [int(r["category"]) for r in rooms]
    return {
        "name": sample.get("name"),
        "boundary": _boundary_pairs(sample),
        "entrance": _entrance_pairs(sample),
        "num_rooms": len(rooms),
        "room_categories": categories,
        "room_categories_named": [
            DP_CATEGORY_TO_NAME.get(c, f"category_{c}") for c in categories
        ],
    }


def build_target_view(sample: dict) -> dict:
    """Build the LLM-visible 'target' dict for a few-shot example.

    Strict subset of DiffPlanner's room schema \u2014 just what the LLM is
    expected to produce.
    """
    rooms_out: list[dict] = []
    for i, r in enumerate(sample.get("rooms", []) or []):
        box = [int(v) for v in r["box"]]
        loc = r.get("location")
        if isinstance(loc, list) and len(loc) == 2:
            loc = [int(loc[0]), int(loc[1])]
        else:
            loc = [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]
        size = int(r.get("size", (box[2] - box[0]) * (box[3] - box[1])))
        rooms_out.append({
            "id": i,
            "category": int(r["category"]),
            "box": box,
            "location": loc,
            "size": size,
        })
    return {"rooms": rooms_out}


# ===================================================================
# DiffPlanner JSON extraction (from raw LLM output)
# ===================================================================

_DP_REQUIRED_FIELDS = ("rooms",)


def _looks_like_dp_json(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    if not all(k in obj for k in _DP_REQUIRED_FIELDS):
        return False
    rooms = obj.get("rooms")
    return isinstance(rooms, list)


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


def _largest_balanced_objects(s: str) -> list[str]:
    out: list[str] = []
    stack: list[str] = []
    start: Optional[int] = None
    for i, ch in enumerate(s):
        if ch == "{":
            if not stack:
                start = i
            stack.append(ch)
        elif ch == "}" and stack:
            stack.pop()
            if not stack and start is not None:
                out.append(s[start:i + 1])
                start = None
    out.sort(key=len, reverse=True)
    return out


def extract_dp_json_from_gpt(response: str) -> Optional[dict]:
    """Extract a DiffPlanner-shaped JSON dict from a GPT response."""
    text = response or ""

    json_block_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    for match in re.findall(json_block_pattern, text, re.DOTALL):
        try:
            obj = json.loads(match.strip())
            if _looks_like_dp_json(obj):
                return obj
        except json.JSONDecodeError:
            continue

    for blob in _largest_balanced_objects(text):
        try:
            obj = json.loads(blob)
            if _looks_like_dp_json(obj):
                return obj
        except json.JSONDecodeError:
            continue

    try:
        obj = json.loads(_strip_code_fences(text))
        if _looks_like_dp_json(obj):
            return obj
    except json.JSONDecodeError:
        pass

    return None


# ===================================================================
# Few-shot prompt builder
# ===================================================================

def create_dp_fewshot_prompt(target_input: dict, examples: list[dict]) -> str:
    """Build the user-content prompt.

    ``examples`` is a list of dicts shaped like
    ``{"filename": str, "input": <input view>, "target": <target view>}``.
    With an empty ``examples`` list this collapses to a clean zero-shot
    instruction.
    """
    if not examples:
        return _zero_shot_user_content(target_input)

    intro = (
        "You are an expert architect generating apartment floor plan layouts in "
        "the DiffPlanner vector format.\n\n"
        "You will be shown examples that pair an INPUT (boundary polygon, entrance "
        "rectangle, requested room count, and an ordered list of room categories) "
        "with the matching OUTPUT (a JSON object containing the predicted rooms "
        "as axis-aligned bounding boxes). After the examples, generate the OUTPUT "
        "for a new INPUT in the same format."
    )

    parts: list[str] = [intro, "", DP_FORMAT_RULES_TEXT, "",
                        "# TRAINING EXAMPLES (Input -> DiffPlanner Rooms JSON):", ""]
    for i, ex in enumerate(examples, 1):
        parts.append(f"## Example {i} ({ex.get('filename', f'example_{i}')}):")
        parts.append("")
        parts.append("### Input:")
        parts.append("```json")
        parts.append(json.dumps(ex["input"], indent=2))
        parts.append("```")
        parts.append("")
        parts.append("### Output:")
        parts.append("```json")
        parts.append(json.dumps(ex["target"], indent=2))
        parts.append("```")
        parts.append("")

    parts.extend([
        "# YOUR TASK:",
        "",
        "Generate the DiffPlanner Rooms JSON for the following input.",
        "",
        "### Target Input:",
        "```json",
        json.dumps(target_input, indent=2),
        "```",
        "",
        DP_REQUIREMENTS_TMPL.format(
            num_rooms=target_input.get("num_rooms", 0),
            categories=target_input.get("room_categories", []),
        ),
        "",
        "Generate ONLY the DiffPlanner Rooms JSON object, no additional text or explanation:",
    ])
    return "\n".join(parts)


def _zero_shot_user_content(target_input: dict) -> str:
    return (
        f"{DP_FORMAT_RULES_TEXT}\n\n"
        f"### Target Input:\n```json\n{json.dumps(target_input, indent=2)}\n```\n\n"
        f"{DP_REQUIREMENTS_TMPL.format(num_rooms=target_input.get('num_rooms', 0), categories=target_input.get('room_categories', []))}\n\n"
        "Generate ONLY the DiffPlanner Rooms JSON object, no additional text or explanation:"
    )


# ===================================================================
# Structural validation
# ===================================================================

@dataclass
class DpValidation:
    is_valid: bool
    passed: list[str]
    failed: list[str]


def _is_int_box(box: Any) -> bool:
    if not isinstance(box, list) or len(box) != 4:
        return False
    if not all(isinstance(v, int) for v in box):
        return False
    x0, y0, x1, y1 = box
    if not all(0 <= v <= CANVAS_SIZE for v in (x0, y0, x1, y1)):
        return False
    return x0 < x1 and y0 < y1


def _normalize_room(room: dict, fallback_id: int, fallback_category: Optional[int]) -> dict:
    """Coerce a model-emitted room into the canonical schema, filling defaults.

    Defensive: integers are accepted as floats too (rounded), and ``location``
    / ``size`` are derived from ``box`` if missing.
    """
    out: dict = {}
    out["id"] = int(room.get("id", fallback_id))
    cat = room.get("category", fallback_category)
    out["category"] = int(cat) if cat is not None else fallback_category

    box = room.get("box")
    if isinstance(box, list) and len(box) == 4:
        try:
            box = [int(round(float(v))) for v in box]
        except (TypeError, ValueError):
            box = None
    if not isinstance(box, list) or len(box) != 4:
        box = [0, 0, 0, 0]
    out["box"] = box

    loc = room.get("location")
    if isinstance(loc, list) and len(loc) == 2:
        try:
            loc = [int(round(float(loc[0]))), int(round(float(loc[1])))]
        except (TypeError, ValueError):
            loc = None
    if not (isinstance(loc, list) and len(loc) == 2):
        loc = [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]
    out["location"] = loc

    size = room.get("size")
    try:
        size = int(round(float(size)))
    except (TypeError, ValueError):
        size = max(0, (box[2] - box[0]) * (box[3] - box[1]))
    out["size"] = size

    return out


def normalize_generated_dp(dp: dict, requested_categories: list[int]) -> dict:
    """Coerce the LLM output into a canonical rooms list, padding/truncating to the request length."""
    rooms_in = dp.get("rooms")
    if not isinstance(rooms_in, list):
        rooms_in = []

    rooms_out: list[dict] = []
    for i, requested_cat in enumerate(requested_categories):
        if i < len(rooms_in) and isinstance(rooms_in[i], dict):
            rooms_out.append(_normalize_room(rooms_in[i], fallback_id=i, fallback_category=requested_cat))
        else:
            rooms_out.append({
                "id": i,
                "category": int(requested_cat),
                "box": [0, 0, 0, 0],
                "location": [0, 0],
                "size": 0,
            })
    return {"rooms": rooms_out}


def validate_dp_json(
    dp: dict,
    *,
    requested_categories: list[int],
    boundary_pairs: list[list[int]],
) -> DpValidation:
    """Structural validation against the DiffPlanner contract for this sample."""
    from shapely.geometry import Polygon, box as shapely_box

    passed: list[str] = []
    failed: list[str] = []

    rooms = dp.get("rooms")
    if not isinstance(rooms, list):
        return DpValidation(False, passed, ["rooms missing or not a list"])

    n_req = len(requested_categories)
    n_got = len(rooms)
    if n_got == n_req:
        passed.append(f"room count == requested ({n_req})")
    else:
        failed.append(f"room count {n_got} != requested {n_req}")

    if all(isinstance(r, dict) for r in rooms):
        passed.append("rooms entries are dicts")
    else:
        failed.append("rooms list contains non-dict entries")

    boxes_ok = all(_is_int_box(r.get("box")) for r in rooms)
    (passed if boxes_ok else failed).append(f"all room boxes are integer pixel rects in [0, {CANVAS_SIZE}] with x0<x1, y0<y1")

    got_cats = [int(r.get("category", -1)) for r in rooms[:n_req]]
    if got_cats == [int(c) for c in requested_categories]:
        passed.append("room categories match requested order")
    elif sorted(got_cats) == sorted(int(c) for c in requested_categories):
        passed.append("room categories match requested multiset (out of order)")
    else:
        failed.append(
            f"room categories {got_cats} do not match requested {requested_categories}"
        )

    if boundary_pairs and boxes_ok and rooms:
        try:
            boundary_poly = _polygon_from_pairs(boundary_pairs)
            boundary_area = float(boundary_poly.area) or 1.0
            n_inside = 0
            for r in rooms:
                bx = r["box"]
                if shapely_box(bx[0], bx[1], bx[2], bx[3]).difference(boundary_poly).area < 1.0:
                    n_inside += 1
            if n_inside == len(rooms):
                passed.append("all boxes lie (strictly) inside the boundary polygon")
            else:
                failed.append(f"only {n_inside}/{len(rooms)} boxes lie inside the boundary polygon")

            from shapely.ops import unary_union
            union_poly = unary_union(
                [shapely_box(r["box"][0], r["box"][1], r["box"][2], r["box"][3]) for r in rooms]
            )
            cover_inside = union_poly.intersection(boundary_poly).area
            coverage = cover_inside / boundary_area
            if coverage >= 0.85:
                passed.append(f"box union covers >=85% of boundary (got {coverage:.2%})")
            else:
                failed.append(f"box union covers only {coverage:.2%} of boundary")
        except Exception as e:  # pragma: no cover - defensive
            failed.append(f"boundary geometry check raised: {e}")

    return DpValidation(is_valid=not failed, passed=passed, failed=failed)


# ===================================================================
# Per-sample DiffPlanner scoring
#
# We deliberately reuse the *exact* metric pipeline from
# ``benchmark/src/evaluate_diffplanner.py`` so the few-shot LLM run produces
# the same per-sample arrays (and the same ``results.npz`` / ``results.txt``)
# as the real DiffPlanner cascade evaluation. The only difference is the
# source of the predicted record: instead of running the cascade, we take the
# LLM's generated rooms and feed them through the same post-processing /
# metric path the cascade output goes through.
#
# When ``--diffplanner_root`` is provided, ``post_processing.main`` is run on
# each generated record (high-fidelity, identical to ``evaluate_diffplanner``).
# When it isn't, we synthesize the ``r_boundary_aligned`` / ``box_aligned``
# fields directly from the LLM's raw boxes and mark the record as
# ``raw_fallback`` -- the same shape ``evaluate_diffplanner`` already uses
# whenever post-processing fails.
# ===================================================================

def _raw_fallback_record(record: dict) -> dict:
    """Synthesize ``r_boundary_aligned`` / ``box_aligned`` from raw boxes.

    Mirrors the ``raw_fallback`` branch of
    :func:`evaluate_diffplanner._post_process_record` so that downstream metric
    code (which expects ``r_boundary_aligned``) keeps working even when
    DiffPlanner's ``output/post_processing.py`` is not on the import path.
    """
    fallback = copy.deepcopy(record)
    for r in fallback.get("rooms", []):
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
    return fallback


def post_process_or_fallback(record: dict, *, dp_available: bool) -> tuple[dict, str]:
    """Run DiffPlanner post-processing if available, else raw_fallback.

    Returns ``(aligned_record, status)`` where ``status`` is one of
    ``"aligned"`` / ``"raw_fallback"`` -- the same vocabulary
    ``evaluate_diffplanner.py`` uses, so the ``raw_fallback_count`` in
    ``run_results`` and ``results.txt`` is computed identically.
    """
    if dp_available:
        try:
            return _dp_post_process_record(record)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("post_processing.main raised %s; falling back to raw boxes", exc)
    return _raw_fallback_record(record), "raw_fallback"


def score_dp_sample(
    gt_sample: dict,
    generated_dp: dict,
    *,
    dp_available: bool,
    run_results: dict,
) -> dict:
    """Score one LLM generation with the evaluate_diffplanner metric pipeline.

    Side-effects: appends per-sample arrays into ``run_results`` (the
    aggregate dict returned by :func:`evaluate_diffplanner._make_results_dict`)
    and returns a flat per-sample dict for the JSON log / progress logger.
    """
    full_record = assemble_diffplanner_sample(gt_sample, generated_dp)
    aligned, pp_status = post_process_or_fallback(full_record, dp_available=dp_available)

    pre_idx = len(run_results["similarity_scores"])
    _dp_evaluate_sample(aligned, gt_sample, pp_status=pp_status, results=run_results)

    if len(run_results["similarity_scores"]) != pre_idx + 1:
        # _dp_evaluate_sample failed to append (e.g. shapely raised); surface a stub.
        return {
            "delta": float("nan"),
            "area_proportion_error": float("nan"),
            "gap_ratio": float("nan"),
            "outside_ratio": float("nan"),
            "overlap_ratio": float("nan"),
            "accuracy": 0.0,
            "room_count_accuracy": 0.0,
            "room_type_accuracy": 0.0,
            "adj_ged": float("nan"),
            "lost_rooms": 0,
            "pp_status": pp_status,
            "num_predicted_rooms_visible": 0,
            "num_gt_rooms": len(gt_sample.get("rooms", []) or []),
            "post_processed_dp": aligned,
        }

    # Resolve the adjacency-graph GED inline. ``_dp_evaluate_sample`` queues a
    # (G_pred, G_true, idx, upper_bound) tuple in ``adj_ged_pairs`` per call;
    # because the few-shot pipeline is per-sample (not batched like the cascade
    # in ``evaluate_diffplanner.main``), the cleanest place to score it is here.
    # The graphs are tiny (<= MAX_NUM_ROOMS = 8 nodes), so this is sub-ms in
    # the common case and well under the 20s timeout in
    # ``ged_between_graphs``. We pop the queue and append into ``adj_geds`` so
    # ``_build_report_kwargs`` finds positional alignment with the other
    # per-sample arrays at finalize time.
    adj_pair = run_results["adj_ged_pairs"].pop()
    try:
        adj_ged_val = _adj_ged_worker(adj_pair)
    except Exception as exc:  # pragma: no cover -- defensive
        logger.warning("adj-graph GED failed for sample %s: %s",
                       gt_sample.get("name", "?"), exc)
        adj_ged_val = float("nan")
    run_results["adj_geds"].append(adj_ged_val)

    pred_visible_cats, _ = _visible_room_categories(aligned)
    return {
        "delta": run_results["similarity_scores"][-1],
        "area_proportion_error": run_results["area_proportion_errors"][-1],
        "gap_ratio": run_results["gap_ratios"][-1],
        "outside_ratio": run_results["outside_ratios"][-1],
        "overlap_ratio": run_results["overlap_ratios"][-1],
        "accuracy": run_results["accuracies"][-1],
        "room_count_accuracy": run_results["room_count_accuracies"][-1],
        "room_type_accuracy": run_results["room_type_accuracies"][-1],
        "adj_ged": float(adj_ged_val) if adj_ged_val is not None else float("nan"),
        "lost_rooms": run_results["lost_rooms_per_sample"][-1],
        "pp_status": pp_status,
        "num_predicted_rooms_visible": len(pred_visible_cats),
        "num_gt_rooms": len(gt_sample.get("rooms", []) or []),
        "post_processed_dp": aligned,
    }


# ===================================================================
# Per-sample DiffPlanner JSON (shape compatible with DiffPlanner/output)
# ===================================================================

def _adjacencies_from_boxes(rooms: list[dict], touch_threshold: int = 4) -> list[list[int]]:
    """Reconstruct an adjacency edge list from predicted boxes.

    Two rooms are 'adjacent' iff their (slightly inflated) bounding boxes
    intersect. The threshold matches DiffPlanner's ``get_adjancenies_from_boxes``
    convention (small slack so near-touching boxes count).
    """
    from shapely.geometry import box as shapely_box

    out: list[list[int]] = []
    polys = []
    for r in rooms:
        if _is_int_box(r.get("box")):
            x0, y0, x1, y1 = r["box"]
            polys.append(shapely_box(
                x0 - touch_threshold, y0 - touch_threshold,
                x1 + touch_threshold, y1 + touch_threshold,
            ))
        else:
            polys.append(None)

    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polys[i] is None or polys[j] is None:
                continue
            if polys[i].intersection(polys[j]).area > 0:
                out.append([int(rooms[i].get("id", i)), int(rooms[j].get("id", j))])
    return out


def assemble_diffplanner_sample(
    input_sample: dict, generated_dp: dict,
) -> dict:
    """Combine the input boundary/entrance with the generated rooms into the full
    DiffPlanner per-sample shape (compatible with ``output/post_processing.py``
    and ``output/visualization.py``)."""
    rooms = generated_dp.get("rooms", []) or []
    return {
        "name": input_sample.get("name"),
        "boundary": input_sample.get("boundary", []),
        "boundary_expand": input_sample.get("boundary_expand", []),
        "entrance_expand": input_sample.get("entrance_expand", []),
        "rooms": rooms,
        "adjacencies": _adjacencies_from_boxes(rooms),
    }


# ===================================================================
# Dataset / few-shot loaders (monolithic JSON)
# ===================================================================

def load_dataset(json_path: Path) -> list[dict]:
    """Load the monolithic DiffPlanner data_test.json as a list of sample dicts."""
    with json_path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list at {json_path}, got {type(data).__name__}")
    return data


def filter_by_max_rooms(samples: list[dict], max_rooms: Optional[int]) -> list[dict]:
    if not max_rooms:
        return samples
    kept = [s for s in samples if len(s.get("rooms", []) or []) <= max_rooms]
    logger.info("Room filter (max %d): kept %d/%d", max_rooms, len(kept), len(samples))
    return kept


def select_examples(
    pool: list[dict],
    num_examples: int,
    *,
    exclude_name: Optional[str],
    seed: Optional[int],
) -> list[dict]:
    """Pick ``num_examples`` few-shot examples from the pool, excluding the target."""
    if num_examples <= 0:
        return []
    candidates = [s for s in pool if str(s.get("name")) != str(exclude_name)]
    if not candidates:
        return []
    rng = random.Random(seed) if seed is not None else random
    chosen = rng.sample(candidates, min(num_examples, len(candidates)))

    examples: list[dict] = []
    for s in chosen:
        examples.append({
            "filename": str(s.get("name")),
            "input": build_input_view(s),
            "target": build_target_view(s),
        })
    return examples


# ===================================================================
# Evaluator base + GPT backend
# ===================================================================

class DpFewShotEvaluator(ABC):
    """Single-process few-shot DiffPlanner evaluator template."""

    def __init__(self, args, dataset: list[dict]):
        self.args = args
        self.dataset = dataset
        self.generator = None  # set in load_model
        # Aggregate metric arrays in the same shape ``evaluate_diffplanner.py``
        # uses, so we can pipe them through ``utils.report_results`` at the end
        # of the run and produce identical ``results.npz`` / ``results.txt``.
        self.run_results: dict = _make_results_dict()
        # When --diffplanner_root is provided, expose DiffPlanner's
        # ``output/post_processing.py`` on sys.path so we can run the same
        # post-processing the cascade evaluator does.
        self.dp_available: bool = False
        dp_root = getattr(args, "diffplanner_root", None)
        if dp_root is not None:
            try:
                _add_diffplanner_to_path(Path(dp_root))
                self.dp_available = True
                logger.info("DiffPlanner post-processing enabled (root=%s)", dp_root)
            except Exception as exc:
                logger.warning(
                    "Could not enable DiffPlanner post-processing (%s); "
                    "falling back to raw LLM boxes.", exc,
                )

    # -- subclass hooks ----------------------------------------------------

    @abstractmethod
    def load_model(self) -> None: ...

    @abstractmethod
    def extract_dp(self, response: str) -> Optional[dict]: ...

    # -- prompt construction ----------------------------------------------

    def build_prompt(self, sample: dict) -> str:
        n = self.args.num_examples
        examples = select_examples(
            self.dataset, num_examples=n,
            exclude_name=str(sample.get("name")),
            seed=self.args.seed,
        )
        target_input = build_input_view(sample)
        return create_dp_fewshot_prompt(target_input, examples)

    # -- per-sample evaluation --------------------------------------------

    def build_result_template(self, sample: dict) -> dict:
        return {
            "filename": f"{sample.get('name')}.json",
            "name": str(sample.get("name")),
            "success": False,
            "error": None,
            "num_rooms_requested": len(sample.get("rooms", []) or []),
            "num_input_tokens": None,
            "generation_time": None,
            "validation": {"is_valid": False, "passed_checks": [], "failed_checks": []},
            "json_extracted": False,
            "retried": False,
            "timed_out": False,
        }

    def evaluate_sample(self, sample: dict) -> dict:
        result = self.build_result_template(sample)
        try:
            prompt = self.build_prompt(sample)
        except Exception as e:
            result["error"] = f"prompt build failed: {e}"
            return result

        timeout = getattr(self.args, "timeout", None)
        try:
            res = self.generator.generate(
                prompt,
                max_tokens=self.args.max_tokens,
                temperature=self.args.temperature,
                timeout=timeout,
            )
        except Exception as e:
            result["error"] = f"generation failed: {e}"
            return result

        result["num_input_tokens"] = res.input_tokens
        result["generation_time"] = res.generation_time
        if res.timed_out:
            result["error"] = f"Generation timed out after {timeout}s"
            result["timed_out"] = True
            return result

        dp_raw = self.extract_dp(res.text or "")
        if dp_raw is None:
            logger.info("JSON extraction failed for %s, retrying once...", sample.get("name"))
            try:
                res2 = self.generator.generate(
                    prompt,
                    max_tokens=self.args.max_tokens,
                    temperature=self.args.temperature,
                    timeout=timeout,
                )
            except Exception as e:
                result["error"] = f"generation failed on retry: {e}"
                return result
            result["generation_time"] += res2.generation_time
            if res2.timed_out:
                result["error"] = f"Generation timed out after {timeout}s (retry)"
                result["timed_out"] = True
                return result
            dp_raw = self.extract_dp(res2.text or "")
            result["retried"] = True
            if dp_raw is None:
                result["error"] = "Failed to extract DiffPlanner JSON from response (after retry)"
                return result

        result["json_extracted"] = True

        requested_cats = [int(r["category"]) for r in sample.get("rooms", []) or []]
        dp = normalize_generated_dp(dp_raw, requested_cats)
        result["generated_dp"] = dp

        validation = validate_dp_json(
            dp,
            requested_categories=requested_cats,
            boundary_pairs=_boundary_pairs(sample),
        )
        result["validation"] = {
            "is_valid": validation.is_valid,
            "passed_checks": validation.passed,
            "failed_checks": validation.failed,
        }

        # Score with the *exact* evaluate_diffplanner pipeline (post-process
        # then compute delta / area-prop / boundary geometry / room accuracy),
        # piping per-sample arrays into ``self.run_results`` so they can be
        # passed to ``report_results`` at the end of the run.
        scores = score_dp_sample(
            sample, dp,
            dp_available=self.dp_available,
            run_results=self.run_results,
        )
        # Save the post-processed DP record on disk (this is the canonical
        # shape ``benchmark/utils/visualize_diffplanner.py`` expects) and keep
        # a flat copy of the metric values for the JSON log.
        result["full_diffplanner_sample"] = scores.pop("post_processed_dp")
        result.update(scores)

        # Coarse "success": passes structural validation AND keeps the
        # requested room count after post-processing (i.e. no rooms lost).
        result["success"] = bool(
            validation.is_valid
            and result.get("num_predicted_rooms_visible", 0) == result["num_rooms_requested"],
        )
        return result

    # -- run loop ----------------------------------------------------------

    def run(self) -> int:
        args = self.args
        random.seed(args.seed)

        samples = filter_by_max_rooms(self.dataset, args.max_rooms)
        if not samples:
            logger.error("No samples after filtering")
            return 1

        # Random-subset selection BEFORE checkpoint filtering / first-N
        # truncation, so the chosen pool is fully determined by
        # (--subset-size, --subset-seed) and is independent of which
        # samples have already been completed in a prior run. The pool is
        # re-sorted afterwards to keep the per-sample ordering deterministic
        # (important for checkpoint resume).
        subset_size = getattr(args, "subset_size", None)
        if subset_size:
            if subset_size > len(samples):
                logger.warning(
                    "--subset-size %d > pool size %d; using full pool.",
                    subset_size, len(samples),
                )
            else:
                subset_seed = getattr(args, "subset_seed", None)
                if subset_seed is None:
                    subset_seed = args.seed
                rng = random.Random(subset_seed)
                samples = rng.sample(samples, subset_size)
                logger.info(
                    "Random subset: kept %d/%d samples (subset_seed=%d).",
                    len(samples), len(self.dataset), subset_seed,
                )

        existing, completed = self._load_checkpoint(args)
        # Sort for deterministic order so checkpoint resume is sensible.
        samples_sorted = sorted(samples, key=lambda s: str(s.get("name")))
        to_process = [s for s in samples_sorted if f"{s.get('name')}.json" not in completed]

        if args.max_samples:
            to_process = to_process[: args.max_samples]
            logger.info("Limited to %d samples", args.max_samples)

        if not to_process:
            logger.info("All samples already evaluated.")
            return 0

        logger.info("Processing %d samples (already done: %d)", len(to_process), len(completed))
        self.load_model()

        results = list(existing)
        start = time.time()
        n_total = len(to_process)

        for i, sample in enumerate(
            tqdm(to_process, desc="Evaluating", unit="sample", mininterval=1.0, file=sys.stdout),
            1,
        ):
            result = self.evaluate_sample(sample)
            results.append(result)
            self._save_partial(args, results)
            self._save_one_dp(args, result)

            if result["success"]:
                msg = (
                    f"OK  | delta={result.get('delta', float('nan')):.2f} "
                    f"area_err={result.get('area_proportion_error', float('nan')):.3f} "
                    f"gap={result.get('gap_ratio', float('nan')):.2f} "
                    f"adj_ged={result.get('adj_ged', float('nan')):.1f}"
                )
            elif result.get("json_extracted"):
                fails = result["validation"]["failed_checks"][:2]
                msg = f"INVALID-DP | failed: {', '.join(fails)}"
            else:
                err = (result.get("error") or "")[:80]
                msg = f"FAIL | {err}"
            logger.info("  [%d/%d] %s -- %s", i, n_total, sample.get("name"), msg)

            if i % 5 == 0:
                self._save_checkpoint(args, results)

            interval = max(1, n_total // 10)
            if i == 1 or i == n_total or i % interval == 0:
                elapsed = time.time() - start
                rate = i / elapsed if elapsed > 0 else 0.0
                eta = (n_total - i) / rate if rate > 0 else 0.0
                logger.info(
                    "Progress: sample %d/%d | elapsed %.1f min | ETA ~%.1f min",
                    i, n_total, elapsed / 60.0, eta / 60.0,
                )

        elapsed = time.time() - start
        logger.info("Total time: %.1f min", elapsed / 60.0)

        self._save_final(args, results)
        return 0

    # -- aggregate stats ---------------------------------------------------

    def compute_statistics(self, results: list[dict]) -> dict:
        """Aggregate the per-sample DiffPlanner metrics for the JSON log.

        Mirrors the numbers ``evaluate_diffplanner.py`` writes to
        ``results.txt`` so the JSON log + the NPZ + the TXT report all tell
        the same story.
        """
        n = len(results)
        json_ok = [r for r in results if r.get("json_extracted")]
        valid = [r for r in results if r.get("validation", {}).get("is_valid")]
        ok = [r for r in results if r.get("success")]
        timed_out = [r for r in results if r.get("timed_out")]
        times = [r["generation_time"] for r in results if r.get("generation_time") is not None]

        # Pull per-sample metric arrays from the run aggregate (preferred --
        # built by score_dp_sample) and fall back to the per-sample dicts in
        # ``results`` so a JSON-checkpoint resume still shows the same fields.
        rr = self.run_results
        if rr["similarity_scores"]:
            deltas = list(rr["similarity_scores"])
            area_errs = list(rr["area_proportion_errors"])
            gaps = list(rr["gap_ratios"])
            outsides = list(rr["outside_ratios"])
            overlaps = list(rr["overlap_ratios"])
            accs = list(rr["accuracies"])
            rcas = list(rr["room_count_accuracies"])
            rtas = list(rr["room_type_accuracies"])
            adj_geds = list(rr["adj_geds"])
            lost_total = int(rr["lost_room_count"])
            samples_with_lost = int(rr["samples_with_lost_rooms"])
            raw_fb = int(rr["raw_fallback_count"])
        else:
            deltas = [r["delta"] for r in results if "delta" in r]
            area_errs = [r["area_proportion_error"] for r in results if "area_proportion_error" in r]
            gaps = [r["gap_ratio"] for r in results if "gap_ratio" in r]
            outsides = [r["outside_ratio"] for r in results if "outside_ratio" in r]
            overlaps = [r["overlap_ratio"] for r in results if "overlap_ratio" in r]
            accs = [r["accuracy"] for r in results if "accuracy" in r]
            rcas = [r["room_count_accuracy"] for r in results if "room_count_accuracy" in r]
            rtas = [r["room_type_accuracy"] for r in results if "room_type_accuracy" in r]
            adj_geds = [r["adj_ged"] for r in results if "adj_ged" in r]
            lost_total = sum(int(r.get("lost_rooms", 0)) for r in results)
            samples_with_lost = sum(1 for r in results if int(r.get("lost_rooms", 0)) > 0)
            raw_fb = sum(1 for r in results if r.get("pp_status") == "raw_fallback")

        def _nan_avg(xs: list[float]) -> Optional[float]:
            arr = np.array([x for x in xs if not (isinstance(x, float) and np.isnan(x))], dtype=float)
            return float(arr.mean()) if arr.size else None

        return {
            "total": n,
            "json_extracted": len(json_ok),
            "valid_structures": len(valid),
            "successful": len(ok),
            "timed_out": len(timed_out),
            "success_rate": (len(ok) / n) if n else 0.0,
            "delta_avg": _nan_avg(deltas),
            "area_proportion_error_avg": _nan_avg(area_errs),
            "gap_ratio_avg": _nan_avg(gaps),
            "outside_ratio_avg": _nan_avg(outsides),
            "overlap_ratio_avg": _nan_avg(overlaps),
            "room_accuracy_avg": _nan_avg(accs),
            "room_count_accuracy_avg": _nan_avg(rcas),
            "room_type_accuracy_avg": _nan_avg(rtas),
            "adj_ged_avg": _nan_avg(adj_geds),
            "lost_room_count": lost_total,
            "samples_with_lost_rooms": samples_with_lost,
            "raw_fallback_count": raw_fb,
            "inference_time": {
                "average_seconds": (sum(times) / len(times)) if times else None,
                "min_seconds": min(times) if times else None,
                "max_seconds": max(times) if times else None,
                "total_minutes": (sum(times) / 60.0) if times else None,
            },
        }

    # -- I/O helpers -------------------------------------------------------

    def _load_checkpoint(self, args) -> tuple[list[dict], set[str]]:
        """Load the per-sample JSON checkpoint AND replay its metric arrays
        into ``self.run_results`` so a resumed run can still write a complete
        ``results.npz`` / ``results.txt`` that covers every completed sample.
        """
        existing: list[dict] = []
        completed: set[str] = set()
        cp = getattr(args, "checkpoint", None)
        if cp and Path(cp).exists():
            with open(cp) as f:
                data = json.load(f)
            existing = data.get("results", []) or []
            completed = {r["filename"] for r in existing}
            logger.info("Loaded checkpoint: %d samples", len(completed))
            for r in existing:
                if "delta" not in r:
                    continue  # incomplete record (json failed to parse, etc.)
                rr = self.run_results
                rr["similarity_scores"].append(float(r.get("delta", float("nan"))))
                rr["area_proportion_errors"].append(float(r.get("area_proportion_error", float("nan"))))
                rr["gap_ratios"].append(float(r.get("gap_ratio", float("nan"))))
                rr["outside_ratios"].append(float(r.get("outside_ratio", float("nan"))))
                rr["overlap_ratios"].append(float(r.get("overlap_ratio", float("nan"))))
                rr["accuracies"].append(float(r.get("accuracy", 0.0)))
                rr["room_count_accuracies"].append(float(r.get("room_count_accuracy", 0.0)))
                rr["room_type_accuracies"].append(float(r.get("room_type_accuracy", 0.0)))
                # ``adj_ged`` is serialized into the checkpoint per sample, so a
                # resume can reconstitute the GED column without re-running the
                # adjacency-graph optimizer. Older checkpoints (pre-GED) won't
                # have the field; fall back to NaN so the array stays aligned.
                rr["adj_geds"].append(float(r.get("adj_ged", float("nan"))))
                rr["per_sample_room_counts"].append(int(r.get("num_gt_rooms", 0)))
                pp_status = str(r.get("pp_status", "raw_fallback"))
                rr["pp_status"].append(pp_status)
                lost = int(r.get("lost_rooms", 0))
                rr["lost_rooms_per_sample"].append(lost)
                rr["lost_room_count"] += lost
                if lost > 0:
                    rr["samples_with_lost_rooms"] += 1
                if pp_status == "raw_fallback":
                    rr["raw_fallback_count"] += 1
        return existing, completed

    @staticmethod
    def _per_sample_dir(args) -> Path:
        explicit = getattr(args, "per_sample_dir", None)
        if explicit:
            return Path(explicit)
        out = Path(args.output)
        return out.with_name(f"{out.stem}_dp")

    @staticmethod
    def _save_one_dp(args, result: dict) -> Optional[Path]:
        full = result.get("full_diffplanner_sample")
        if full is None:
            return None
        out_dir = DpFewShotEvaluator._per_sample_dir(args)
        out_dir.mkdir(parents=True, exist_ok=True)
        sid = result.get("name") or result["filename"].replace(".json", "")
        target = out_dir / f"{sid}.json"
        tmp = target.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(full, f, indent=2)
        os.replace(tmp, target)
        return target

    @staticmethod
    def _save_partial(args, results: list[dict]) -> None:
        out_path = Path(args.output.replace(".json", "_partial.json"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {"timestamp": datetime.now().isoformat(),
                 "completed": len(results), "results": results},
                f, indent=2,
            )

    @staticmethod
    def _save_checkpoint(args, results: list[dict]) -> None:
        with open(args.output, "w") as f:
            json.dump(
                {"timestamp": datetime.now().isoformat(),
                 "completed": len(results), "results": results},
                f, indent=2,
            )

    def _save_final(self, args, results: list[dict]) -> None:
        stats = self.compute_statistics(results)
        meta = {
            "timestamp": datetime.now().isoformat(),
            "mode": getattr(args, "mode", "unknown"),
            "model": getattr(args, "model", None),
            "dataset_json": args.dataset_json,
            "diffplanner_root": str(getattr(args, "diffplanner_root", None) or ""),
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "subset_size": getattr(args, "subset_size", None),
            "subset_seed": (
                getattr(args, "subset_seed", None)
                if getattr(args, "subset_seed", None) is not None
                else (args.seed if getattr(args, "subset_size", None) else None)
            ),
            "max_samples": getattr(args, "max_samples", None),
            "max_rooms": getattr(args, "max_rooms", None),
            "num_examples": args.num_examples,
            "total_samples": len(results),
        }
        output_data = {"metadata": meta, "statistics": stats, "results": results}
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info("Results saved to: %s", args.output)

        # Defensive sweep: ensure every result with a generated payload has a
        # per-sample file on disk (catches checkpoint-resume into a new
        # --per-sample-dir).
        per_sample_dir = self._per_sample_dir(args)
        n_written = 0
        for r in results:
            if r.get("full_diffplanner_sample") is None:
                continue
            sid = r.get("name") or r["filename"].replace(".json", "")
            if not (per_sample_dir / f"{sid}.json").exists():
                self._save_one_dp(args, r)
                n_written += 1
        n_total_dp = sum(1 for r in results if r.get("full_diffplanner_sample") is not None)
        logger.info(
            "Per-sample DiffPlanner JSONs in %s (%d total; %d written this finalize pass)",
            per_sample_dir, n_total_dp, n_written,
        )

        # Drop a ``results.npz`` + ``results.txt`` next to ``--output`` using
        # the *exact* aggregate format that ``evaluate_diffplanner.py``
        # produces. This is the canonical benchmark report.
        out_dir = self._results_dir(args)
        if not self.run_results["similarity_scores"]:
            logger.warning(
                "No samples were scored -- skipping report_results NPZ/TXT.",
            )
            return
        n = len(self.run_results["similarity_scores"])
        rr = self.run_results
        # Defensive: drain any leftover queued GED pairs into ``adj_geds`` so
        # the array is positionally aligned with the other per-sample arrays.
        # Under the normal flow ``score_dp_sample`` already pops + scores each
        # pair inline, so this is a no-op; this guards future codepaths that
        # might enqueue without scoring (e.g. a batched evaluator variant).
        if rr["adj_ged_pairs"]:
            try:
                rr["adj_geds"].extend(_parallel_adj_ged(rr["adj_ged_pairs"]))
                rr["adj_ged_pairs"].clear()
            except Exception as exc:  # pragma: no cover -- defensive
                logger.warning("Final adjacency-graph GED drain failed (%s); "
                               "leaving GED column partial.", exc)
        print(
            f"\nScored {n} samples; raw_fallback (post-processing failed) "
            f"on {rr['raw_fallback_count']} of them."
        )
        print(
            f"Lost rooms: {rr['lost_room_count']:,} total across "
            f"{rr['samples_with_lost_rooms']:,} of {n:,} samples "
            f"({100.0 * rr['samples_with_lost_rooms'] / n:.1f}% of samples "
            f"have at least one room dropped by post-processing)."
        )
        report_results(
            str(out_dir),
            num_variations=1,
            checkpoint_path=None,
            data_path=str(args.dataset_json),
            **_build_report_kwargs(self.run_results),
        )
        print(f"\nWrote {out_dir / 'results.txt'} and {out_dir / 'results.npz'}")

    @staticmethod
    def _results_dir(args) -> Path:
        """Where to drop ``results.npz`` / ``results.txt``."""
        explicit = getattr(args, "out", None)
        if explicit:
            out_dir = Path(explicit)
        else:
            out = Path(args.output)
            out_dir = out.with_name(f"{out.stem}_metrics")
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir


# -------- GPT backend ----------------------------------------------------

class DpFewShotGPTEvaluator(DpFewShotEvaluator):
    """Few-shot DiffPlanner generation via the OpenAI API."""

    def load_model(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.generator = GPTGenerator(model_name=self.args.model, api_key=api_key)
        # Override the BSP system prompt baked into scripts.generators with the
        # DiffPlanner-specific one (single-process eval, mirrors what the HD
        # evaluator does).
        import scripts.generators as _gen
        _gen.SYSTEM_PROMPT = DP_SYSTEM_MESSAGE
        logger.info("GPT generator initialised (%s)", self.args.model)

    def extract_dp(self, response: str) -> Optional[dict]:
        return extract_dp_json_from_gpt(response)


# ===================================================================
# CLI
# ===================================================================

MODE_MAP: dict[str, type[DpFewShotEvaluator]] = {
    "fewshot-gpt": DpFewShotGPTEvaluator,
}


def cli() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Few-shot DiffPlanner-format LLM evaluator")
    parser.add_argument("--mode", choices=list(MODE_MAP), default="fewshot-gpt")

    parser.add_argument(
        "--dataset-json", type=str, required=True,
        help="Monolithic DiffPlanner data_test.json (a JSON list of sample dicts).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Aggregate JSON eval log (default: results_dp_<mode>_<model>_<seed>.json).",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help=(
            "Directory for ``results.npz`` and ``results.txt`` "
            "(written via ``utils.report_results.report_results`` -- the same "
            "shape ``benchmark/src/evaluate_diffplanner.py`` produces). "
            "Default: ``<output stem>_metrics/`` next to --output."
        ),
    )
    parser.add_argument(
        "--diffplanner_root", type=Path, default=None,
        help=(
            "Path to the DiffPlanner repo. When provided, generated rooms are "
            "fed through ``output/post_processing.py`` (high-fidelity, identical "
            "to evaluate_diffplanner.py). When omitted, raw LLM boxes are used "
            "and every sample is marked ``raw_fallback``."
        ),
    )
    parser.add_argument(
        "--per-sample-dir", type=str, default=None,
        help=(
            "Directory for one DiffPlanner-shaped JSON per generated sample "
            "(default: <output stem>_dp/ next to --output). Files are written "
            "incrementally and named <name>.json (compatible with "
            "DiffPlanner/output/post_processing.py and visualization.py)."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=5120)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for in-context example sampling and "
                             "(by default) for the --subset-size selection.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap the number of samples actually evaluated, "
                             "taken in deterministic sort order from the "
                             "(possibly subsetted) pool. Useful for short "
                             "smoke tests; for a STABLE evaluation pool use "
                             "--subset-size instead.")
    parser.add_argument("--subset-size", type=int, default=None,
                        help="Evaluate on a random subset of N samples drawn "
                             "uniformly without replacement from the full "
                             "test pool. Reproducible: the same "
                             "(--subset-size, --subset-seed) always yields "
                             "the same N samples. Applied AFTER --max-rooms "
                             "filtering and BEFORE --max-samples truncation, "
                             "so resuming from a checkpoint stays consistent.")
    parser.add_argument("--subset-seed", type=int, default=None,
                        help="Seed used for --subset-size selection. Defaults "
                             "to --seed; set independently to vary the random "
                             "subset without changing in-context example "
                             "sampling.")
    parser.add_argument("--max-rooms", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--num-examples", type=int, default=3,
        help="In-context examples for fewshot-* modes. Use 0 for zero-shot.",
    )
    parser.add_argument("--model", type=str, default=None)

    args = parser.parse_args()

    if args.model is None and args.mode == "fewshot-gpt":
        args.model = "gpt-4"

    if args.output is None:
        slug = Path(args.model).name
        args.output = f"results_dp_{args.mode.replace('-', '_')}_{slug}_{args.seed}.json"
        logger.info("Auto-generated output: %s", args.output)

    dataset_path = Path(args.dataset_json)
    if not dataset_path.is_file():
        logger.error("dataset-json not found: %s", dataset_path)
        return 1
    logger.info("Loading dataset from %s ...", dataset_path)
    dataset = load_dataset(dataset_path)
    logger.info("Loaded %d samples", len(dataset))

    cls = MODE_MAP[args.mode]
    evaluator = cls(args, dataset)
    return evaluator.run()


if __name__ == "__main__":
    sys.exit(cli())
