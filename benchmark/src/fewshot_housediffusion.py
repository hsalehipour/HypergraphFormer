#!/usr/bin/env python3
"""
Few-shot evaluator for House-Diffusion-LLM models.

Mode:
    fewshot-gpt -- GPT via the OpenAI API, single process

The model is asked to produce a House-Diffusion JSON (``room_type`` / ``boxes``
/ ``edges`` / ``ed_rm``) from an access graph. Eval samples are the canonical
House-Diffusion ``<id>.json`` files (e.g. those under
``dataset/rplan/dataset_housediffusion/test/``); the access graph the LLM
sees as input is derived on the fly from each HD JSON via the inlined
``extract_access_graph_from_hd`` helper. The same HD JSON is kept around as
``ground_truth_hd`` and used post-generation for geometric scoring.

Scoring is the SAME pipeline as ``benchmark/src/evaluate_housediffusion.py``
(delta / room similarity, area-proportion error, spatial overlap, room-count
and room-type accuracies, GED, and a hidden-room counter), reusing those
helpers directly so the few-shot ``results.npz`` / ``results.txt`` are byte-
for-byte the same shape as the diffusion benchmark's. The lighter-weight
access-graph reconstruction scores (edge F1, outside-match, etc.) are kept
around as diagnostic fields in the JSON log.

Outputs
-------
For ``--output results/run.json``:

* ``results/run.json``               Aggregate eval JSON log: metadata,
                                     statistics summary, and per-sample result
                                     records (each with its ``generated_hd``
                                     payload).
* ``results/run_partial.json``       Same shape, written incrementally after
                                     every sample so a crashed run can be
                                     resumed via ``--checkpoint``.
* ``results/run_hd/<id>.json``       ONE file per successful generation, holding
                                     just the HD JSON dict (``room_type`` /
                                     ``boxes`` / ``edges`` / ``ed_rm``). Written
                                     incrementally as each sample completes; this
                                     is the shape ``benchmark/utils/visualize_rplan_json.py``
                                     and ``benchmark/utils/convert_json_to_pickle.py``
                                     expect. Override the directory with
                                     ``--per-sample-dir``.
* ``results/run_metrics/results.{npz,txt}``
                                     Aggregate geometric-metrics report,
                                     produced via ``utils.report_results`` --
                                     same layout as
                                     ``benchmark/src/evaluate_housediffusion.py``.
                                     Override the directory with ``--out``.

Usage
-----
    export OPENAI_API_KEY=...

    # Few-shot GPT
    python benchmark/src/fewshot_housediffusion.py --mode fewshot-gpt \\
        --model gpt-4-turbo \\
        --dataset-dir dataset/rplan/dataset_housediffusion/test

    # Zero-shot variant (no in-context examples)
    python benchmark/src/fewshot_housediffusion.py --mode fewshot-gpt \\
        --model gpt-4-turbo \\
        --dataset-dir dataset/rplan/dataset_housediffusion/test --num-examples 0
"""

from __future__ import annotations

import argparse
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

# ===================================================================
# House-Diffusion JSON contract: category ids, label names, and the
# textual rules we put in front of the LLM. Inlined here so this script
# is self-contained (no dependency on a separate data-preparation module).
# Mirrors the layout used by ``benchmark/src/fewshot_diffplanner.py``.
# ===================================================================

CANVAS_SIZE = 256

# Door category ids in the House-Diffusion / RPLAN class scheme.
FRONT_DOOR_CAT = 15
INTERIOR_DOOR_CAT = 17
_DOOR_CATS = (FRONT_DOOR_CAT, INTERIOR_DOOR_CAT)

# Sentinel name for the synthetic "outside" reference appended to edges
# leaving the apartment via the front door. It shows up as the second
# endpoint of edges like ``["3", "outside"]`` but is NOT listed as a key
# in the access graph's ``nodes`` dict.
OUTSIDE_LABEL = "outside"

# House-Diffusion category id -> short access-graph room-type label,
# matching the labels used by the existing access graphs
# (``bed`` / ``bath`` / ``living`` / ``kitchen`` / ``balcony`` / ``storage``).
_HD_CATEGORY_TO_NAME: dict[int, str] = {
    1:  "living",
    2:  "kitchen",
    3:  "bed",
    4:  "bath",
    5:  "balcony",
    6:  "entrance",
    7:  "dining",
    8:  "study",
    10: "storage",
    16: "extra",   # rare in the dataset; safest catch-all label
}


HD_SYSTEM_MESSAGE = (
    "You are an expert at generating apartment floor plan layouts in "
    "House-Diffusion JSON format from access graphs. Each layout is a single "
    "JSON object describing room types, axis-aligned bounding boxes, and "
    "wall/door edges in pixel coordinates."
)


HD_FORMAT_RULES_TEXT = """\
# CRITICAL STRUCTURE RULES:

## 1. House-Diffusion JSON Schema (MUST FOLLOW):
The output is a single JSON object with EXACTLY these four fields:

### 1.1 `room_type` -- list[int]
- One integer category id per layout entry, in canonical order: rooms first, then door entries (interior doors, then a single front door at the end).
- Standard RPLAN-style category IDs:
    1  = Living Room       2  = Kitchen        3  = Bedroom (bed)
    4  = Bathroom (bath)   5  = Balcony        6  = Entrance / Hall
    7  = Dining Room       8  = Study Room    10  = Storage
   15  = Front Door       17  = Interior Door

### 1.2 `boxes` -- list[[x1, y1, x2, y2]]
- One axis-aligned bounding box per entry in `room_type`, SAME ORDER and SAME LENGTH.
- Coordinates are integer-valued pixels in `[0, 256]`; (x1, y1) is the top-left corner, (x2, y2) is the bottom-right corner (so x1 < x2 and y1 < y2).

### 1.3 `edges` -- list[[x1, y1, x2, y2, ra, rb]]
- One entry per wall / door segment that bounds a node (a room OR a door entry).
- (x1, y1, x2, y2) are the integer pixel endpoints of the segment.
- `ra` is the **category id** (a value taken from `room_type`) of the node that emits this segment.
- `rb` is **0** when the segment lies on the apartment exterior, otherwise the **category id** of the node on the other side of the wall. So interior doors appear as `rb = 17` for the rooms they connect, and the front door appears as `rb = 15`.

### 1.4 `ed_rm` -- list[list[int]]
- One entry per row of `edges`, in the SAME ORDER as `edges`.
- Each entry is a list of **node indices** (0-based positions into `room_type` / `boxes`) that the segment touches. The indices can refer to ROOMS or to DOOR entries (categories 15 / 17), not just to rooms.
- A length-1 entry `[i]` means the segment lies on the apartment exterior, owned by node `i`; the corresponding `edges` row has `rb == 0`.
- A length-2 entry `[i, j]` means the segment is shared between nodes `i` and `j`; the corresponding `edges` row has `rb != 0`, and the invariant `room_type[i] == ra` and `room_type[j] == rb` MUST hold.
- `ed_rm` is what disambiguates `edges` whenever multiple rooms share a category (e.g. two bedrooms): `ra` / `rb` only carry types, but `ed_rm` carries the exact two entries.

## 2. Geometric Consistency:
- For every room, the union of its segments in `edges` must form a closed axis-aligned polygon contained inside that room's `boxes` rectangle.
- Walls are emitted segment-by-segment; do not rely on implicit closures.
- All coordinates are integers in `[0, 256]` (the canonical House-Diffusion grid).
- `len(boxes) == len(room_type)` and `len(ed_rm) == len(edges)`.

## 3. Connectivity (Access Graph -> JSON):
- The access-graph `edges` list contains TWO kinds of entries:
    * room-to-room edges `[a, b]`, where both endpoints are room ids that appear as keys in `nodes`, and
    * outside edges `[r, "outside"]`, where the literal string `"outside"` indicates that room `r` carries the apartment's front door. `"outside"` is NOT a real room and is NOT a key in `nodes`; therefore room `r`'s box should sit on the apartment boundary so its front door faces the exterior.
- The number of ROOM entries in `room_type` (entries with category id NOT 15 and NOT 17) MUST equal the number of nodes in the access graph (i.e., the count of keys in `nodes`; the `"outside"` reference does NOT count).
- Each room entry's category id must match the corresponding access-graph node's room type, using this canonical mapping:
    living=1, kitchen=2, bed=3, bath=4, balcony=5, entrance=6, dining=7, study=8, storage=10.
- For every room-to-room edge `[a, b]` in the access graph, the JSON must realize that connection as an INTERIOR DOOR: an entry with category id 17 in `room_type` whose segment rows have `ed_rm = [door_index, room_index]` for each of the two connected rooms (so room `a` and room `b` end up sharing exactly one interior door).
- For every outside edge `[r, "outside"]` in the access graph, the JSON must include exactly one Front-Door entry (category 15) in `room_type` whose `ed_rm` rows pair it with room `r`; room `r` must therefore have at least one row in `edges` with `rb == 15`."""


HD_REQUIREMENTS_TMPL = """\
## Requirements:
1. The number of ROOM entries in `room_type` (entries with category id != 15 and != 17) MUST equal {num_nodes} (number of access-graph nodes; the `"outside"` reference does NOT count).
2. Each room's category id must match the corresponding access-graph room type.
3. Every room-to-room edge `[a, b]` in the access graph must be realized by an Interior-Door entry (category 17) whose `ed_rm` rows pair it with both rooms.
4. Every outside edge `[r, "outside"]` in the access graph must be realized by a Front-Door entry (category 15) whose `ed_rm` rows pair it with room `r`; that room's box must sit on the apartment boundary.
5. All coordinates are integers in `[0, 256]`.
6. `len(boxes) == len(room_type)` and `len(ed_rm) == len(edges)`.
7. Each room's `boxes` rectangle must enclose all of its segments in `edges`.
8. Output ONLY the JSON object, no markdown fences and no commentary."""


def create_hd_instruction_prompt() -> str:
    """Long-form instruction explaining the HD JSON format. No examples."""
    return f"""You are an expert at generating apartment floor plan layouts in the House-Diffusion JSON format that represent floor plans.

Your task is to generate a House-Diffusion JSON object from an access graph. An access graph is a simple representation of room connectivity:
- `nodes`: Dictionary mapping room IDs to room types (e.g., {{"1": "bed", "2": "kitchen"}}).
- `edges`: List of connections. Each entry is either a room-to-room pair (e.g. `["1", "2"]`) or an outside reference (e.g. `["1", "outside"]`). The literal string `"outside"` is NOT a room and is NOT listed in `nodes`; it marks the room that owns the apartment's front door (i.e., the entry-door room), and that room's box should therefore lie on the apartment boundary so its front door can face the exterior.

{HD_FORMAT_RULES_TEXT}

# YOUR TASK:

Generate a House-Diffusion JSON object for the following access graph."""


def extract_access_graph_from_hd(hd_json: dict) -> dict:
    """Build a ``{nodes, edges}`` access graph from a House-Diffusion JSON.

    Door-mediated convention:

    - Rooms (categories 1-10, 16) are nodes; door entries (categories 15
      and 17) are NOT in ``nodes``.
    - Two rooms ``a, b`` get a single edge ``[a, b]`` iff there is an
      interior-door entry (category 17) at index ``k`` such that ``ed_rm``
      pairs ``k`` with both ``a`` and ``b`` (i.e. some rows of ``ed_rm``
      look like ``[k, a]`` and ``[k, b]``).
    - The front-door entry (category 15) at index ``f`` produces the
      synthetic outside reference: every room ``r`` that ``ed_rm`` pairs
      with ``f`` gets an extra edge ``[r, "outside"]``. The literal string
      ``"outside"`` is intentionally NOT a key in ``nodes``.
    - Output uses 1-indexed string ids in ``nodes`` and pairs of those ids
      (or the literal string ``"outside"``) in ``edges``.
    """
    rt: list[int] = hd_json.get("room_type", [])
    ed_rm: list[list[int]] = hd_json.get("ed_rm", [])

    room_orig_indices = [i for i, c in enumerate(rt) if c not in _DOOR_CATS]
    orig_to_new_id = {orig: str(new_idx + 1) for new_idx, orig in enumerate(room_orig_indices)}
    room_orig_set = set(room_orig_indices)

    nodes: dict[str, str] = {
        orig_to_new_id[orig]: _HD_CATEGORY_TO_NAME.get(rt[orig], "extra")
        for orig in room_orig_indices
    }

    interior_door_rooms: dict[int, set[int]] = {
        i: set() for i, c in enumerate(rt) if c == INTERIOR_DOOR_CAT
    }
    front_door_indices: set[int] = {i for i, c in enumerate(rt) if c == FRONT_DOOR_CAT}
    front_door_rooms: set[int] = set()

    for row in ed_rm:
        for k in row:
            if k in interior_door_rooms:
                for r in row:
                    if r in room_orig_set:
                        interior_door_rooms[k].add(r)
            elif k in front_door_indices:
                for r in row:
                    if r in room_orig_set:
                        front_door_rooms.add(r)

    seen: set = set()
    edges: list = []
    for door_rooms in interior_door_rooms.values():
        rooms = sorted(door_rooms)
        for ai in range(len(rooms)):
            for bi in range(ai + 1, len(rooms)):
                a = orig_to_new_id[rooms[ai]]
                b = orig_to_new_id[rooms[bi]]
                key = tuple(sorted([a, b], key=lambda x: int(x)))
                if key not in seen:
                    seen.add(key)
                    edges.append([key[0], key[1]])

    for r_orig in sorted(front_door_rooms):
        edges.append([orig_to_new_id[r_orig], OUTSIDE_LABEL])

    return {"nodes": nodes, "edges": edges}

# We delegate metric computation to the same helpers used by
# benchmark.src.evaluate_housediffusion so the few-shot results pipeline emits
# byte-for-byte the same statistics (delta, area-prop error, spatial overlap,
# room-type/count accuracies, GED, hidden-room count) and the same
# results.npz / results.txt layout.
from benchmark.src.evaluate_housediffusion import (
    _build_report_kwargs,
    _make_results_dict,
    _parallel_ged,
    _room_geometries_to_area_tuples,
    build_access_graph_from_polygons,
    compute_room_similarity_score_polygons,
    compute_spatial_overlap_ratio_from_polygons,
    count_actual_rooms_from_polygons,
    extract_room_geometries_from_polygons,
)
from utils.metrics import compute_room_area_proportion_error as compute_area_prop_metric
from utils.metrics import room_type_accuracy
from utils.report_results import report_results

logger = logging.getLogger(__name__)


# ===================================================================
# HD JSON extraction (from raw model output)
# ===================================================================

_HD_REQUIRED_FIELDS = ("room_type", "boxes", "edges", "ed_rm")


def _looks_like_hd_json(obj: Any) -> bool:
    return isinstance(obj, dict) and all(k in obj for k in _HD_REQUIRED_FIELDS)


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


def _try_parse_first_object(s: str) -> Optional[dict]:
    """Parse the first top-level JSON object from *s*, ignoring trailing text."""
    s = s.strip()
    i = s.find("{")
    if i == -1:
        return None
    dec = json.JSONDecoder()
    try:
        obj, _end = dec.raw_decode(s[i:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _largest_balanced_objects(s: str) -> list[str]:
    """Return all top-level ``{...}`` substrings, longest first."""
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
                out.append(s[start : i + 1])
                start = None
    out.sort(key=len, reverse=True)
    return out


def extract_hd_json_from_gpt(response: str) -> Optional[dict]:
    """Extract an HD JSON dict from a GPT response (markdown fences or plain)."""
    text = response or ""

    json_block_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    for match in re.findall(json_block_pattern, text, re.DOTALL):
        try:
            obj = json.loads(match.strip())
            if _looks_like_hd_json(obj):
                return obj
        except json.JSONDecodeError:
            continue

    for blob in _largest_balanced_objects(text):
        try:
            obj = json.loads(blob)
            if _looks_like_hd_json(obj):
                return obj
        except json.JSONDecodeError:
            continue

    try:
        obj = json.loads(text.strip())
        if _looks_like_hd_json(obj):
            return obj
    except json.JSONDecodeError:
        pass

    return None


# ===================================================================
# Few-shot HD prompt builder
# ===================================================================

def create_hd_fewshot_prompt(target_accessgraph: dict, examples: list[dict]) -> str:
    """Build a few-shot user-content prompt for HD generation.

    ``examples`` is a list of dicts with ``filename`` / ``accessgraph`` / ``hypergraph``
    keys (same shape as :func:`utils.data_loading.load_example_hypergraphs` returns).
    Each example's ``hypergraph`` is expected to be an HD JSON dict.

    When ``examples`` is empty the prompt collapses to a clean zero-shot
    instruction (essentially equivalent to
    :func:`create_hd_instruction_prompt` plus the per-target requirements).
    """
    if not examples:
        return _zero_shot_user_content(target_accessgraph)

    intro = (
        "You are an expert at generating apartment floor plan layouts in the "
        "House-Diffusion JSON format that represent floor plans.\n\n"
        "You will be shown examples of access graphs paired with their corresponding "
        "House-Diffusion JSON layouts, then generate a new layout following the same "
        "structure and rules.\n\n"
        "An access graph is a simple representation of room connectivity:\n"
        "- `nodes`: Dictionary mapping room IDs to room types "
        '(e.g., {"1": "bed", "2": "kitchen"}).\n'
        "- `edges`: List of connections. Each entry is either a room-to-room pair "
        '(e.g. `["1", "2"]`) or an outside reference (e.g. `["1", "outside"]`). '
        'The literal string `"outside"` is NOT a room and is NOT listed in `nodes`; '
        "it marks the room that owns the apartment's front door (i.e., the entry-door "
        "room), and that room's box should therefore lie on the apartment boundary so "
        "its front door can face the exterior."
    )

    parts: list[str] = [intro, "", HD_FORMAT_RULES_TEXT, "",
                        "# TRAINING EXAMPLES (Access Graph -> House-Diffusion JSON):", ""]
    for i, ex in enumerate(examples, 1):
        parts.append(f"## Example {i} ({ex.get('filename', f'example_{i}')}):")
        parts.append("")
        parts.append("### Input - Access Graph:")
        parts.append("```json")
        parts.append(json.dumps(ex["accessgraph"], indent=2))
        parts.append("```")
        parts.append("")
        parts.append("### Output - House-Diffusion JSON:")
        parts.append("```json")
        parts.append(json.dumps(ex["hypergraph"], indent=2))
        parts.append("```")
        parts.append("")

    parts.extend([
        "# YOUR TASK:",
        "",
        "Generate a House-Diffusion JSON object for the following access graph.",
        "",
        "### Target Access Graph:",
        "```json",
        json.dumps(target_accessgraph, indent=2),
        "```",
        "",
        HD_REQUIREMENTS_TMPL.format(
            num_nodes=len(target_accessgraph.get("nodes", {})),
        ),
        "",
        "Generate ONLY the House-Diffusion JSON object, no additional text or explanation:",
    ])
    return "\n".join(parts)


def _zero_shot_user_content(target_accessgraph: dict) -> str:
    """Zero-shot user-content prompt: instruction + access graph + requirements."""
    instruction = create_hd_instruction_prompt()
    return (
        f"{instruction}\n\n"
        f"### Access Graph:\n```json\n{json.dumps(target_accessgraph, indent=2)}\n```\n\n"
        f"{HD_REQUIREMENTS_TMPL.format(num_nodes=len(target_accessgraph.get('nodes', {})))}\n\n"
        "Generate ONLY the House-Diffusion JSON object, no additional text or explanation:"
    )


# ===================================================================
# HD structural validation + access-graph reconstruction scoring
# ===================================================================

@dataclass
class HdValidation:
    is_valid: bool
    passed: list[str]
    failed: list[str]


def _is_int_box(box: Any) -> bool:
    if not isinstance(box, list) or len(box) != 4:
        return False
    if not all(isinstance(v, int) for v in box):
        return False
    x1, y1, x2, y2 = box
    if not (0 <= x1 <= 256 and 0 <= y1 <= 256 and 0 <= x2 <= 256 and 0 <= y2 <= 256):
        return False
    return x1 < x2 and y1 < y2


def _is_int_segment(seg: Any) -> bool:
    if not isinstance(seg, list) or len(seg) != 6:
        return False
    if not all(isinstance(v, int) for v in seg):
        return False
    x1, y1, x2, y2, _ra, _rb = seg
    return all(0 <= v <= 256 for v in (x1, y1, x2, y2))


def validate_hd_json(hd: dict) -> HdValidation:
    """Lightweight structural checks on a generated HD JSON.

    These mirror the rules in ``HD_FORMAT_RULES_TEXT``: shape / lengths,
    integer pixel ranges, ``ra``/``rb`` consistency, and door-pairing.
    Geometric closure (segments forming closed polygons) is intentionally
    NOT checked here -- it requires more work and is not needed for the
    coarse few-shot eval signal.
    """
    passed: list[str] = []
    failed: list[str] = []

    rt = hd.get("room_type")
    boxes = hd.get("boxes")
    edges = hd.get("edges")
    ed_rm = hd.get("ed_rm")

    if not isinstance(rt, list) or not rt:
        failed.append("room_type missing or not a non-empty list")
    else:
        if all(isinstance(c, int) for c in rt):
            passed.append("room_type entries are ints")
        else:
            failed.append("room_type contains non-int entries")

    if isinstance(boxes, list) and isinstance(rt, list):
        if len(boxes) == len(rt):
            passed.append("len(boxes) == len(room_type)")
        else:
            failed.append(f"len(boxes)={len(boxes)} != len(room_type)={len(rt)}")
        if all(_is_int_box(b) for b in boxes):
            passed.append("boxes are integer-pixel rects in [0, 256]")
        else:
            failed.append("one or more boxes are not integer-pixel rects in [0, 256]")
    else:
        failed.append("boxes missing or not a list")

    if isinstance(edges, list) and isinstance(ed_rm, list):
        if len(edges) == len(ed_rm):
            passed.append("len(edges) == len(ed_rm)")
        else:
            failed.append(f"len(edges)={len(edges)} != len(ed_rm)={len(ed_rm)}")
        if all(_is_int_segment(seg) for seg in edges):
            passed.append("edges are integer pixel segments in [0, 256]")
        else:
            failed.append("one or more edge segments are not integer pixel rows of length 6")
    else:
        failed.append("edges or ed_rm missing or not a list")

    if isinstance(rt, list) and isinstance(edges, list) and isinstance(ed_rm, list):
        n = len(rt)
        ra_rb_ok = True
        ed_rm_ok = True
        for seg, owners in zip(edges, ed_rm):
            if not (isinstance(seg, list) and len(seg) == 6):
                ra_rb_ok = False
                break
            ra, rb = seg[4], seg[5]
            if not isinstance(owners, list) or len(owners) not in (1, 2):
                ed_rm_ok = False
                continue
            if any((not isinstance(i, int)) or i < 0 or i >= n for i in owners):
                ed_rm_ok = False
                continue
            if len(owners) == 1:
                if rb != 0:
                    ra_rb_ok = False
                if rt[owners[0]] != ra:
                    ra_rb_ok = False
            else:
                i, j = owners
                if rb == 0:
                    ra_rb_ok = False
                if not (rt[i] == ra and rt[j] == rb):
                    ra_rb_ok = False
        (passed if ra_rb_ok else failed).append("ra/rb consistent with ed_rm and room_type")
        (passed if ed_rm_ok else failed).append("ed_rm rows reference valid node indices")

    if isinstance(rt, list):
        n_front = sum(1 for c in rt if c == FRONT_DOOR_CAT)
        if n_front == 1:
            passed.append("exactly one front-door entry (category 15)")
        else:
            failed.append(f"expected 1 front-door entry, got {n_front}")

    return HdValidation(is_valid=not failed, passed=passed, failed=failed)


# ===================================================================
# HD JSON -> polygon-list (the contract evaluate_housediffusion expects)
# ===================================================================

def hd_json_to_polygons(hd: dict) -> list[dict]:
    """Convert an HD JSON ({room_type, boxes, edges, ed_rm}) into the polygon
    list shape consumed by ``evaluate_housediffusion``'s metric helpers
    (``count_actual_rooms_from_polygons``, ``extract_room_geometries_from_polygons``,
    ``build_access_graph_from_polygons``, ``compute_room_similarity_score_polygons``).

    Each polygon dict has: ``coords`` (list of ``(x, y)`` corner tuples),
    ``room_type`` (RPLAN category id), ``room_idx`` (entry index in the HD JSON).

    Coordinates are integer pixels on the same 256x256 canvas that
    ``corners_to_polygons_and_types`` produces, so polygons coming from
    real House-Diffusion samples and polygons coming from LLM HD JSONs are
    directly comparable without any rescaling.

    For each room/door entry ``i`` we use ``boxes[i]`` (the axis-aligned
    bounding rectangle the LLM is required to provide) as the polygon
    geometry. ``boxes[i]`` is the LLM's effective room shape: the HD format
    forces every wall segment in ``edges`` to lie inside it, and downstream
    visualisers (``benchmark/utils/visualize_rplan_json.py``) likewise
    render rooms from the box. Using boxes here keeps the few-shot eval
    aligned with what the user actually sees and what gets exported as a
    per-sample HD JSON.
    """
    if not isinstance(hd, dict):
        return []
    rt = hd.get("room_type") or []
    boxes = hd.get("boxes") or []
    if not isinstance(rt, list) or not isinstance(boxes, list):
        return []

    polygons: list[dict] = []
    for i, (cat, box) in enumerate(zip(rt, boxes)):
        if not (isinstance(box, list) and len(box) == 4):
            continue
        try:
            x1, y1, x2, y2 = (int(v) for v in box)
        except (TypeError, ValueError):
            continue
        if not (x1 < x2 and y1 < y2):
            continue
        polygons.append({
            "coords": [(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
            "room_type": int(cat),
            "room_idx": i,
        })
    return polygons


# ===================================================================
# Per-sample evaluator: runs the same pipeline as
# benchmark.src.evaluate_housediffusion.postprocess_sample, but on
# (gt_hd, pred_hd) pairs rather than on diffusion tensors.
# ===================================================================

def score_hd_sample(
    gt_hd: Optional[dict],
    pred_hd: dict,
    *,
    sample_idx: int,
    run_results: dict,
    ged_upper_bound: int = 40,
) -> dict:
    """Run the full evaluate_housediffusion metric pipeline on one sample.

    Updates ``run_results`` (the aggregate dict produced by
    ``_make_results_dict``) in place AND returns a flat dict of the
    per-sample metric values, ready to merge into the result record.

    GED is *not* computed here: a (G_pred, G_true, sample_idx, upper_bound)
    pair is appended to ``run_results['ged_pairs']`` and resolved in
    parallel later via ``_parallel_ged``.
    """
    out: dict = {}

    if not isinstance(pred_hd, dict):
        out["scoring_error"] = "pred_hd is not a dict"
        return out
    if gt_hd is None:
        out["scoring_error"] = "ground_truth_hd missing for this sample"
        return out

    pred_polygons = hd_json_to_polygons(pred_hd)
    gt_polygons = hd_json_to_polygons(gt_hd)

    if not pred_polygons:
        out["scoring_error"] = "predicted HD JSON has no usable boxes"
        return out
    if not gt_polygons:
        out["scoring_error"] = "ground-truth HD JSON has no usable boxes"
        return out

    G_true = build_access_graph_from_polygons(gt_polygons)
    G_pred = build_access_graph_from_polygons(pred_polygons)

    gt_room_type_list = [
        int(p["room_type"]) for p in gt_polygons
        if p["room_type"] not in (FRONT_DOOR_CAT, INTERIOR_DOOR_CAT)
    ]
    run_results["per_sample_room_counts"].append(len(gt_room_type_list))

    room_count_info = count_actual_rooms_from_polygons(pred_polygons)
    if len(room_count_info["hidden_rooms"]) > 0:
        run_results["hidden_room_count"] += 1

    # NOTE: This mirrors evaluate_housediffusion exactly: 'accuracy' is the
    # combined room-count + room-type score, 'room_count_accuracy' is the
    # count-only score, and 'room_type_accuracy' is the multiset-overlap
    # score. The aggregate-bucket names are kept identical to the
    # evaluate_housediffusion ones so _build_report_kwargs Just Works.
    rta = room_type_accuracy(gt_room_type_list, room_count_info["actual_room_types"])
    run_results["room_count_accuracies"].append(rta["accuracy"])
    run_results["room_count_accuracy_scores"].append(rta["room_count_accuracy"])
    run_results["room_type_accuracy_scores"].append(rta["room_type_accuracy"])

    sim = compute_room_similarity_score_polygons(pred_polygons, gt_polygons)
    run_results["similarity_scores"].append(sim["similarity_score"])
    run_results["unmatched_pred_rooms"].append(sim["unmatched_pred"])
    run_results["unmatched_gt_rooms"].append(sim["unmatched_gt"])
    run_results["matched_room_pairs"].append(sim["matched_pairs"])

    pred_geoms = extract_room_geometries_from_polygons(pred_polygons)
    gt_geoms = extract_room_geometries_from_polygons(gt_polygons)
    area = compute_area_prop_metric(
        _room_geometries_to_area_tuples(pred_geoms),
        _room_geometries_to_area_tuples(gt_geoms),
    )
    run_results["area_proportion_errors"].append(area["area_proportion_error"])

    overlap = compute_spatial_overlap_ratio_from_polygons(pred_polygons)
    run_results["spatial_overlap_ratios"].append(overlap)

    run_results["ged_pairs"].append((G_pred, G_true, sample_idx, ged_upper_bound))

    out.update({
        # Combined accuracy (matches evaluate_housediffusion's
        # 'room_count_accuracies' bucket / report label "Room Accuracy").
        "accuracy": rta["accuracy"],
        "room_count_accuracy": rta["room_count_accuracy"],
        "room_type_accuracy": rta["room_type_accuracy"],
        "similarity_score": sim["similarity_score"],
        "area_proportion_error": area["area_proportion_error"],
        "spatial_overlap_ratio": overlap,
        "num_visible_pred_rooms": room_count_info["total_rooms"],
        "num_expected_pred_rooms": room_count_info["expected_rooms"],
        "num_hidden_pred_rooms": len(room_count_info["hidden_rooms"]),
        "num_gt_rooms": len(gt_room_type_list),
        "unmatched_pred_rooms": sim["unmatched_pred"],
        "unmatched_gt_rooms": sim["unmatched_gt"],
        "matched_room_pairs": sim["matched_pairs"],
    })
    return out


def score_against_accessgraph(
    target_ag: dict, hd: dict,
) -> dict:
    """Reconstruct the access graph from *hd* and compare to *target_ag*."""
    target_nodes: dict = target_ag.get("nodes", {}) or {}
    target_edges = target_ag.get("edges", []) or []

    try:
        gen_ag = extract_access_graph_from_hd(hd)
    except Exception as e:  # pragma: no cover -- defensive
        return {
            "reconstruction_error": str(e),
            "room_count_accuracy": 0.0,
            "room_type_accuracy": 0.0,
            "edge_f1": 0.0,
            "edge_precision": 0.0,
            "edge_recall": 0.0,
            "outside_match": False,
        }

    gen_nodes: dict = gen_ag.get("nodes", {})
    gen_edges = gen_ag.get("edges", [])

    n_target = len(target_nodes)
    n_gen = len(gen_nodes)
    room_count_accuracy = 1.0 if n_target == n_gen else 0.0

    target_types = sorted(str(v) for v in target_nodes.values())
    gen_types = sorted(str(v) for v in gen_nodes.values())
    if target_types == gen_types:
        room_type_accuracy = 1.0
    else:
        from collections import Counter

        ct, cg = Counter(target_types), Counter(gen_types)
        inter = sum((ct & cg).values())
        denom = max(sum(ct.values()), 1)
        room_type_accuracy = inter / denom

    def _edge_keys(edges) -> tuple[set[tuple[str, str]], set[str]]:
        room_pairs: set[tuple[str, str]] = set()
        outside_rooms: set[str] = set()
        for e in edges:
            if not isinstance(e, list) or len(e) != 2:
                continue
            a, b = str(e[0]), str(e[1])
            if a == OUTSIDE_LABEL or b == OUTSIDE_LABEL:
                room = b if a == OUTSIDE_LABEL else a
                outside_rooms.add(room)
            else:
                key = tuple(sorted([a, b]))
                room_pairs.add(key)
        return room_pairs, outside_rooms

    t_pairs, t_outside = _edge_keys(target_edges)
    g_pairs, g_outside = _edge_keys(gen_edges)

    tp = len(t_pairs & g_pairs)
    fp = len(g_pairs - t_pairs)
    fn = len(t_pairs - g_pairs)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "room_count_accuracy": room_count_accuracy,
        "room_type_accuracy": room_type_accuracy,
        "edge_f1": f1,
        "edge_precision": precision,
        "edge_recall": recall,
        "outside_match": t_outside == g_outside,
        "num_target_nodes": n_target,
        "num_generated_nodes": n_gen,
        "num_target_edges": len(target_edges),
        "num_generated_edges": len(gen_edges),
    }


# ===================================================================
# Sample loading & few-shot example loading (HD-aware)
# ===================================================================

# Glob pattern for the canonical House-Diffusion test dataset.
# Each file is one HD JSON (``room_type`` / ``boxes`` / ``edges`` / ``ed_rm``);
# the access graph used as the LLM-visible input is derived on the fly via
# ``extract_access_graph_from_hd``, so no separate data-prep step is needed.
_HD_GLOB = "*.json"


def _load_hd_json_file(fpath: Path) -> Optional[dict]:
    """Read one ``<id>.json`` HD file and return the parsed dict, or None on failure."""
    try:
        with fpath.open() as f:
            hd = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Skipping %s: %s", fpath.name, e)
        return None
    if not _looks_like_hd_json(hd):
        logger.warning("Skipping %s: not a House-Diffusion JSON (need %s)",
                       fpath.name, ", ".join(_HD_REQUIRED_FIELDS))
        return None
    return hd


def load_sample(fpath: Path) -> dict:
    """Load one canonical House-Diffusion ``<id>.json`` file.

    The access graph (the LLM's input) is derived on the fly from the HD
    JSON itself via :func:`extract_access_graph_from_hd`. The HD JSON is
    also kept around as ``ground_truth_hd`` for post-generation scoring.
    """
    hd = _load_hd_json_file(fpath)
    if hd is None:
        # Surface a stub so the caller can still see the filename in errors.
        return {
            "filename": fpath.name,
            "accessgraph": {"nodes": {}, "edges": []},
            "ground_truth_hd": None,
            "num_nodes": 0,
            "num_edges": 0,
            "load_error": True,
        }
    ag = extract_access_graph_from_hd(hd)
    return {
        "filename": fpath.name,
        "accessgraph": ag,
        "ground_truth_hd": hd,
        "num_nodes": len(ag.get("nodes", {}) or {}),
        "num_edges": len(ag.get("edges", []) or []),
    }


def load_hd_examples(
    examples_dir: Path,
    num_examples: int,
    *,
    exclude_file: Optional[str] = None,
    seed: Optional[int] = None,
) -> list[dict]:
    """Load few-shot examples from the dataset dir.

    Each example is one canonical ``<id>.json`` House-Diffusion file; the
    access graph the LLM sees as the example INPUT is derived from the HD
    JSON itself via :func:`extract_access_graph_from_hd` so no separate
    pre-built ``{accessgraph, hypergraph}`` records are needed.
    """
    if num_examples is None or num_examples <= 0:
        return []

    files = sorted(examples_dir.glob(_HD_GLOB), key=lambda p: p.name)
    if exclude_file:
        files = [f for f in files if f.name != exclude_file]
    if not files:
        raise FileNotFoundError(
            f"No example {_HD_GLOB} files in {examples_dir}"
        )

    if seed is not None:
        rng = random.Random(seed)
        chosen = rng.sample(files, min(num_examples, len(files)))
    else:
        chosen = random.sample(files, min(num_examples, len(files)))

    examples: list[dict] = []
    for fp in chosen:
        hd = _load_hd_json_file(fp)
        if hd is None:
            continue
        ag = extract_access_graph_from_hd(hd)
        if not ag.get("nodes"):
            logger.warning("Skipping example %s: derived access graph has no nodes",
                           fp.name)
            continue
        examples.append({"filename": fp.name, "accessgraph": ag, "hypergraph": hd})

    if not examples:
        raise ValueError(
            f"Could not load any HD examples from {examples_dir}; "
            "expected House-Diffusion JSON files (room_type / boxes / "
            "edges / ed_rm)."
        )
    return examples


# ===================================================================
# Evaluator base + two backends
# ===================================================================

class HdFewShotEvaluator(ABC):
    """Single-process few-shot HD evaluator template."""

    def __init__(self, args):
        self.args = args
        self.generator = None  # set in load_model
        # Aggregate metric arrays, identical layout to
        # benchmark.src.evaluate_housediffusion. Per-sample evals append into
        # this dict; _save_final flushes it to results.npz/results.txt via
        # report_results(). The counter for raw/scoring failures is local;
        # everything else is tracked through the standard buckets.
        self.run_results: dict = _make_results_dict()
        self.scoring_failure_count: int = 0
        # Index used as ``sample_idx`` for the GED pairs queue. Must be
        # unique per sample; we increment regardless of success so resumed
        # runs stay aligned with the original ordering.
        self._next_sample_idx: int = 0

    # -- subclass hooks ----------------------------------------------------

    @abstractmethod
    def load_model(self) -> None: ...

    @abstractmethod
    def extract_hd(self, response: str) -> Optional[dict]: ...

    # -- prompt construction ----------------------------------------------

    def build_prompt(self, sample: dict) -> str:
        n = self.args.num_examples
        if n and n > 0:
            examples = load_hd_examples(
                Path(self.args.examples_dir or self.args.dataset_dir),
                num_examples=n,
                exclude_file=sample["filename"],
                seed=self.args.seed,
            )
        else:
            examples = []
        return create_hd_fewshot_prompt(sample["accessgraph"], examples)

    # -- per-sample evaluation --------------------------------------------

    def build_result_template(self, sample: dict) -> dict:
        return {
            "filename": sample["filename"],
            "success": False,
            "error": None,
            "num_nodes": sample.get("num_nodes"),
            "num_edges": sample.get("num_edges"),
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

        hd = self.extract_hd(res.text or "")
        if hd is None:
            logger.info("JSON extraction failed for %s, retrying once...", sample["filename"])
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
            hd = self.extract_hd(res2.text or "")
            result["retried"] = True
            if hd is None:
                result["error"] = "Failed to extract HD JSON from response (after retry)"
                return result

        result["json_extracted"] = True
        result["generated_hd"] = hd

        validation = validate_hd_json(hd)
        result["validation"] = {
            "is_valid": validation.is_valid,
            "passed_checks": validation.passed,
            "failed_checks": validation.failed,
        }

        # Keep the lightweight access-graph reconstruction scores around as
        # diagnostic fields in the JSON log (edge F1, outside-match, etc.).
        # They are NOT what gets aggregated into results.txt -- that pipeline
        # is the evaluate_housediffusion-equivalent metrics computed below.
        ag_scores = score_against_accessgraph(sample["accessgraph"], hd)
        result.update(ag_scores)

        # Geometric metrics (delta, area-prop, overlap, room accuracies, GED
        # pair) -- the same pipeline as benchmark.src.evaluate_housediffusion.
        # The GT HD JSON comes straight from the input file (``load_sample``
        # parses each <id>.json HD file and exposes it as ``ground_truth_hd``).
        sample_idx = self._next_sample_idx
        self._next_sample_idx += 1
        try:
            hd_scores = score_hd_sample(
                sample.get("ground_truth_hd"),
                hd,
                sample_idx=sample_idx,
                run_results=self.run_results,
            )
        except Exception as exc:  # pragma: no cover -- defensive
            hd_scores = {"scoring_error": f"score_hd_sample crashed: {exc}"}
        if "scoring_error" in hd_scores:
            self.scoring_failure_count += 1
        result.update(hd_scores)

        # "success" is a coarse summary: structural validity + the visible
        # predicted room count matches the GT room count. This is the
        # closest analogue to evaluate_diffplanner's success criterion and
        # uses the *post-hidden-room* count (rooms that survive shapely
        # subset checks), so it stays consistent with the metrics we report.
        n_visible = result.get("num_visible_pred_rooms")
        n_gt = result.get("num_gt_rooms")
        result["success"] = bool(
            validation.is_valid
            and n_visible is not None
            and n_gt is not None
            and n_visible == n_gt,
        )
        return result

    # -- run loop ----------------------------------------------------------

    def run(self) -> int:
        args = self.args

        random.seed(args.seed)

        dataset_dir = Path(args.dataset_dir)
        all_files = sorted(dataset_dir.glob(_HD_GLOB), key=lambda p: p.name)
        if not all_files:
            logger.error("No %s files found in %s", _HD_GLOB, dataset_dir)
            return 1

        if getattr(args, "max_rooms", None):
            # Count rooms by reading the HD JSON's room_type list and
            # excluding door entries (categories 15 / 17). This matches
            # how the access graph is derived (door entries are NOT nodes),
            # so the cap stays consistent with the LLM's view.
            kept: list[Path] = []
            for fp in all_files:
                hd = _load_hd_json_file(fp)
                if hd is None:
                    continue
                rt = hd.get("room_type") or []
                n_rooms = sum(1 for c in rt if c not in _DOOR_CATS)
                if n_rooms <= args.max_rooms:
                    kept.append(fp)
            logger.info("Room filter (max %d): kept %d/%d", args.max_rooms, len(kept), len(all_files))
            all_files = kept

        # Random-subset selection BEFORE checkpoint filtering / first-N
        # truncation, so the chosen pool is fully determined by
        # (--subset-size, --subset-seed) and is independent of which
        # samples have already been completed in a prior run. The pool is
        # re-sorted afterwards to keep per-sample ordering deterministic
        # (important for checkpoint resume).
        subset_size = getattr(args, "subset_size", None)
        if subset_size:
            if subset_size > len(all_files):
                logger.warning(
                    "--subset-size %d > pool size %d; using full pool.",
                    subset_size, len(all_files),
                )
            else:
                subset_seed = getattr(args, "subset_seed", None)
                if subset_seed is None:
                    subset_seed = args.seed
                rng = random.Random(subset_seed)
                all_files = sorted(rng.sample(all_files, subset_size),
                                   key=lambda p: p.name)
                logger.info(
                    "Random subset: kept %d files (subset_seed=%d).",
                    len(all_files), subset_seed,
                )

        existing, completed = self._load_checkpoint(args)
        files_to_process = [f for f in all_files if f.name not in completed]
        if args.max_samples:
            files_to_process = files_to_process[: args.max_samples]
            logger.info("Limited to %d samples", args.max_samples)

        if not files_to_process:
            logger.info("All files already evaluated.")
            return 0

        logger.info("Processing %d files (already done: %d)", len(files_to_process), len(completed))
        self.load_model()

        results = list(existing)
        start = time.time()
        n_total = len(files_to_process)

        for i, fp in enumerate(
            tqdm(
                files_to_process,
                desc="Evaluating",
                unit="sample",
                mininterval=1.0,
                file=sys.stdout,
            ),
            1,
        ):
            sample = load_sample(fp)
            result = self.evaluate_sample(sample)
            results.append(result)
            self._save_partial(args, results)
            self._save_one_hd(args, result)

            if result["success"]:
                msg = (
                    f"OK  | delta={result.get('similarity_score', float('nan')):.3f} "
                    f"area-prop={result.get('area_proportion_error', float('nan')):.3f} "
                    f"overlap={result.get('spatial_overlap_ratio', float('nan')):.3f} "
                    f"room-acc={result.get('accuracy', 0):.2f}"
                )
            elif result.get("json_extracted"):
                msg = (
                    f"INVALID-HD | failed: "
                    f"{', '.join(result['validation']['failed_checks'][:2])}"
                )
            else:
                err = (result.get("error") or "")[:80]
                msg = f"FAIL | {err}"
            logger.info("  [%d/%d] %s -- %s", i, n_total, fp.name, msg)

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
        """Aggregate statistics for the JSON log.

        Exposes both the geometric metrics that are reported through
        ``report_results`` (delta, area-prop error, room accuracies, GED,
        spatial overlap) AND the access-graph diagnostics (edge F1) we
        keep around for backwards compatibility with older notebooks.
        """
        n = len(results)
        json_ok = [r for r in results if r.get("json_extracted")]
        valid = [r for r in results if r.get("validation", {}).get("is_valid")]
        ok = [r for r in results if r.get("success")]
        timed_out = [r for r in results if r.get("timed_out")]
        outside_ok = [r for r in results if r.get("outside_match")]
        scored = [r for r in results if r.get("scoring_error") is None
                  and r.get("similarity_score") is not None]

        def _vals(key):
            return [r[key] for r in scored if r.get(key) is not None
                    and not (isinstance(r[key], float) and np.isnan(r[key]))]

        # Use np.nanmean so a few NaNs (e.g. spatial overlap on degenerate
        # samples) don't poison the average; if every sample is NaN the mean
        # silently returns NaN, which is what report_results does too.
        def _nanmean(xs):
            if not xs:
                return None
            arr = np.asarray(xs, dtype=float)
            if not np.isfinite(arr).any():
                return None
            return float(np.nanmean(arr))

        deltas = _vals("similarity_score")
        area_props = _vals("area_proportion_error")
        overlaps = _vals("spatial_overlap_ratio")
        accs = _vals("accuracy")
        rcs = _vals("room_count_accuracy")
        rts = _vals("room_type_accuracy")

        # Hidden-room counter -- prefer the live aggregate (true while the
        # run is in progress); fall back to a per-result scan so resumed
        # runs (where self.run_results was rebuilt from a checkpoint) still
        # report a consistent number.
        hidden_count = self.run_results.get("hidden_room_count", 0)
        if hidden_count == 0 and any(r.get("num_hidden_pred_rooms") for r in results):
            hidden_count = sum(1 for r in results
                               if (r.get("num_hidden_pred_rooms") or 0) > 0)

        edge_f1 = [r["edge_f1"] for r in results if "edge_f1" in r]
        edge_p = [r["edge_precision"] for r in results if "edge_precision" in r]
        edge_r = [r["edge_recall"] for r in results if "edge_recall" in r]
        times = [r["generation_time"] for r in results if r.get("generation_time") is not None]

        def _avg(xs):
            return sum(xs) / len(xs) if xs else None

        return {
            "total": n,
            "json_extracted": len(json_ok),
            "valid_structures": len(valid),
            "successful": len(ok),
            "timed_out": len(timed_out),
            "success_rate": (len(ok) / n) if n else 0.0,
            "scored_samples": len(scored),
            "scoring_failures": self.scoring_failure_count
                or sum(1 for r in results if r.get("scoring_error")),
            "hidden_room_count": hidden_count,
            "outside_match": len(outside_ok),
            # The same metrics that get persisted to results.txt by
            # report_results, surfaced here for at-a-glance reading of
            # the JSON log.
            "geometric_metrics": {
                "similarity_avg": _nanmean(deltas),
                "area_proportion_error_avg": _nanmean(area_props),
                "spatial_overlap_ratio_avg": _nanmean(overlaps),
                "accuracy_avg": _nanmean(accs),
                "room_count_accuracy_avg": _nanmean(rcs),
                "room_type_accuracy_avg": _nanmean(rts),
            },
            "edge_metrics": {
                "f1_avg": _avg(edge_f1),
                "precision_avg": _avg(edge_p),
                "recall_avg": _avg(edge_r),
            },
            "inference_time": {
                "average_seconds": _avg(times),
                "min_seconds": min(times) if times else None,
                "max_seconds": max(times) if times else None,
                "total_minutes": (sum(times) / 60.0) if times else None,
            },
        }

    # -- I/O helpers -------------------------------------------------------

    def _load_checkpoint(self, args) -> tuple[list[dict], set[str]]:
        """Reload a partially-completed run.

        Replays the per-sample geometric metrics back into
        ``self.run_results`` so that the final ``results.txt`` /
        ``results.npz`` reflect ALL samples, not just the new ones. GED
        cannot be replayed (the access graphs are not in the JSON log),
        so for resumed runs the GED column in ``results.txt`` is computed
        only from the newly-evaluated samples; the warning logged below
        makes that explicit.
        """
        existing: list[dict] = []
        completed: set[str] = set()
        cp = getattr(args, "checkpoint", None)
        if not (cp and Path(cp).exists()):
            return existing, completed

        with open(cp) as f:
            data = json.load(f)
        existing = data.get("results", []) or []
        completed = {r["filename"] for r in existing}
        logger.info("Loaded checkpoint: %d samples", len(completed))

        replayed = 0
        for r in existing:
            if r.get("scoring_error") is not None:
                self.scoring_failure_count += 1
                continue
            if r.get("similarity_score") is None:
                continue

            self.run_results["similarity_scores"].append(r["similarity_score"])
            self.run_results["area_proportion_errors"].append(
                r.get("area_proportion_error", float("nan"))
            )
            self.run_results["spatial_overlap_ratios"].append(
                r.get("spatial_overlap_ratio", float("nan"))
            )
            self.run_results["room_count_accuracies"].append(r.get("accuracy", 0.0))
            self.run_results["room_count_accuracy_scores"].append(
                r.get("room_count_accuracy", 0.0)
            )
            self.run_results["room_type_accuracy_scores"].append(
                r.get("room_type_accuracy", 0.0)
            )
            self.run_results["per_sample_room_counts"].append(r.get("num_gt_rooms", 0))
            self.run_results["unmatched_pred_rooms"].append(
                r.get("unmatched_pred_rooms", 0)
            )
            self.run_results["unmatched_gt_rooms"].append(
                r.get("unmatched_gt_rooms", 0)
            )
            self.run_results["matched_room_pairs"].append(
                r.get("matched_room_pairs", [])
            )
            if (r.get("num_hidden_pred_rooms") or 0) > 0:
                self.run_results["hidden_room_count"] += 1
            replayed += 1

        # Advance the GED-pair sample index past the replayed records.
        self._next_sample_idx = len(existing)
        if replayed and replayed < len(existing):
            logger.info(
                "Replayed %d/%d checkpoint samples into the metric aggregate "
                "(%d had no scores).",
                replayed, len(existing), len(existing) - replayed,
            )
        elif replayed:
            logger.info("Replayed %d checkpoint samples into the metric aggregate.",
                        replayed)
        if replayed:
            logger.info(
                "Note: GED cannot be replayed from a JSON checkpoint; the "
                "GED column in results.txt reflects only the newly-evaluated "
                "samples in this resume.",
            )
        return existing, completed

    @staticmethod
    def _per_sample_dir(args) -> Path:
        """Directory for one HD JSON file per generated sample.

        Defaults to ``<output stem>_hd/`` next to ``--output`` so a user who runs
        ``--output results/run.json`` finds their per-id HD JSONs in
        ``results/run_hd/<id>.json``. Override with ``--per-sample-dir``.
        """
        explicit = getattr(args, "per_sample_dir", None)
        if explicit:
            return Path(explicit)
        out = Path(args.output)
        return out.with_name(f"{out.stem}_hd")

    @staticmethod
    def _save_one_hd(args, result: dict) -> Optional[Path]:
        """Write one ``<id>.json`` HD payload from a single result; no-op otherwise.

        Atomic-ish: write to a ``.tmp`` sibling and rename, so a crash mid-write
        cannot leave a half-written ``<id>.json`` that downstream tools would
        choke on.
        """
        hd = result.get("generated_hd")
        if hd is None:
            return None
        out_dir = HdFewShotEvaluator._per_sample_dir(args)
        out_dir.mkdir(parents=True, exist_ok=True)
        sid = Path(result["filename"]).stem
        target = out_dir / f"{sid}.json"
        tmp = target.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(hd, f, indent=2)
        os.replace(tmp, target)
        return target

    @staticmethod
    def _save_partial(args, results: list[dict]) -> None:
        """Write the in-progress aggregate results report (per-sample HDs are written separately)."""
        out_path = Path(args.output.replace(".json", "_partial.json"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {"timestamp": datetime.now().isoformat(),
                 "completed": len(results), "results": results},
                f,
                indent=2,
            )

    @staticmethod
    def _save_checkpoint(args, results: list[dict]) -> None:
        with open(args.output, "w") as f:
            json.dump(
                {"timestamp": datetime.now().isoformat(),
                 "completed": len(results), "results": results},
                f,
                indent=2,
            )

    def _resolve_out_dir(self, args) -> Path:
        """Where to write results.npz / results.txt.

        Defaults to ``<output stem>_metrics/`` next to ``--output`` so a
        user who runs ``--output results/run.json`` finds their aggregate
        report under ``results/run_metrics/``. Override with ``--out``.
        """
        explicit = getattr(args, "out", None)
        if explicit:
            return Path(explicit)
        out = Path(args.output)
        return out.with_name(f"{out.stem}_metrics")

    def _save_final(self, args, results: list[dict]) -> None:
        # Resolve any pending GED pairs IN PLACE on self.run_results so the
        # JSON-log statistics and results.txt see the same all_geds list.
        if self.run_results.get("ged_pairs") and not self.run_results.get("all_geds"):
            try:
                self.run_results["all_geds"] = _parallel_ged(
                    self.run_results["ged_pairs"]
                )
            except Exception as exc:  # pragma: no cover -- defensive
                logger.warning("Parallel GED failed (%s); leaving GED column empty.",
                               exc)
                self.run_results["all_geds"] = []

        # Also mirror the per-sample GED back onto the result records so a
        # user reading the JSON log sees the same number that ends up in
        # results.txt. We pair by sample_idx (preserved when each pair was
        # appended in evaluate_sample).
        ged_by_idx: dict[int, float] = {}
        for pair, ged in zip(self.run_results.get("ged_pairs", []),
                             self.run_results.get("all_geds", [])):
            if len(pair) >= 3:
                ged_by_idx[int(pair[2])] = ged
        # Position i in `results` matches sample_idx i (replays sit at the
        # front of `results` in the same order they were originally
        # evaluated, and new samples are appended afterwards with
        # self._next_sample_idx starting at len(existing)).
        for i, r in enumerate(results):
            if i in ged_by_idx and "ged" not in r:
                ged_val = ged_by_idx[i]
                if isinstance(ged_val, float) and np.isnan(ged_val):
                    r["ged"] = None
                else:
                    r["ged"] = float(ged_val)

        stats = self.compute_statistics(results)
        meta = {
            "timestamp": datetime.now().isoformat(),
            "mode": getattr(args, "mode", "unknown"),
            "model": getattr(args, "model", None),
            "dataset_dir": args.dataset_dir,
            "examples_dir": args.examples_dir,
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

        # Defensive: ensure every result with an HD payload has a per-sample file
        # on disk. The run loop already writes these incrementally; this catches
        # the case where the user resumed from a checkpoint into a new
        # --per-sample-dir, so older successes also get materialised there.
        per_sample_dir = self._per_sample_dir(args)
        n_written = 0
        for r in results:
            if r.get("generated_hd") is None:
                continue
            sid = Path(r["filename"]).stem
            if not (per_sample_dir / f"{sid}.json").exists():
                self._save_one_hd(args, r)
                n_written += 1
        n_total_hd = sum(1 for r in results if r.get("generated_hd") is not None)
        logger.info(
            "Per-sample HD JSONs in %s (%d total; %d written this finalize pass)",
            per_sample_dir, n_total_hd, n_written,
        )

        # Final report (results.npz + results.txt) -- byte-for-byte the same
        # layout produced by benchmark.src.evaluate_housediffusion.
        if not self.run_results.get("room_count_accuracies"):
            logger.warning(
                "No samples were geometrically scored; skipping results.npz/results.txt. "
                "Likely causes: every sample failed JSON extraction, or none of the "
                "input HD JSON files were readable.",
            )
            return

        out_dir = self._resolve_out_dir(args)
        out_dir.mkdir(parents=True, exist_ok=True)
        report_results(
            str(out_dir),
            num_variations=1,
            checkpoint_path=getattr(args, "model", None) or "n/a",
            data_path=args.dataset_dir,
            **_build_report_kwargs(self.run_results),
        )
        logger.info("Aggregate report (results.npz + results.txt) written to: %s",
                    out_dir)
        if self.run_results.get("hidden_room_count"):
            logger.info("Hidden rooms: %d sample(s) had >=1 fully-occluded predicted room.",
                        self.run_results["hidden_room_count"])
        if self.scoring_failure_count:
            logger.info("Scoring failures: %d sample(s) had no usable HD geometry.",
                        self.scoring_failure_count)


def _patch_system_prompt(new_system_prompt: str) -> None:
    """Override the default BSP system prompt baked into ``scripts.generators``.

    ``GPTGenerator`` builds its chat messages using the module-level
    ``SYSTEM_PROMPT`` constant. This replaces it for the lifetime of the
    process so HD evaluation gets the right system role.
    """
    import scripts.generators as _gen
    _gen.SYSTEM_PROMPT = new_system_prompt


# -------- GPT backend ----------------------------------------------------

class HdFewShotGPTEvaluator(HdFewShotEvaluator):
    """Few-shot HD generation via the OpenAI API."""

    def load_model(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.generator = GPTGenerator(model_name=self.args.model, api_key=api_key)
        _patch_system_prompt(HD_SYSTEM_MESSAGE)
        logger.info("GPT generator initialised (%s)", self.args.model)

    def extract_hd(self, response: str) -> Optional[dict]:
        return extract_hd_json_from_gpt(response)


# ===================================================================
# CLI
# ===================================================================

MODE_MAP: dict[str, type[HdFewShotEvaluator]] = {
    "fewshot-gpt": HdFewShotGPTEvaluator,
}


def cli() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Few-shot House-Diffusion-LLM evaluator")
    parser.add_argument("--mode", choices=list(MODE_MAP), required=True)

    parser.add_argument(
        "--dataset-dir", type=str, required=True,
        help="Directory of canonical House-Diffusion <id>.json files (each "
             "containing room_type / boxes / edges / ed_rm), e.g. "
             "dataset/rplan/dataset_housediffusion/test. The access graph "
             "shown to the LLM is derived on the fly from each HD JSON.",
    )
    parser.add_argument(
        "--examples-dir", type=str, default=None,
        help="Directory to draw few-shot examples from (defaults to --dataset-dir)",
    )
    parser.add_argument("--output", type=str, default=None,
                        help="Aggregate eval JSON log path (default: results_hd_<mode>_<model>_<seed>.json). "
                             "Holds metadata, the statistics summary, and every per-sample result record. "
                             "The geometric metrics report (results.npz + results.txt) is written to a "
                             "separate directory; see --out.")
    parser.add_argument(
        "--out", type=str, default=None,
        help=(
            "Directory for the aggregate metrics report (results.npz + results.txt), "
            "produced via the same utils.report_results pipeline as "
            "benchmark/src/evaluate_housediffusion.py. Defaults to "
            "<output stem>_metrics/ next to --output."
        ),
    )
    parser.add_argument(
        "--per-sample-dir", type=str, default=None,
        help=(
            "Directory for one HD JSON file per generated sample "
            "(default: <output stem>_hd/ next to --output). "
            "Files are written incrementally as each sample completes, "
            "and named <id>.json (matching the HD-JSON shape that "
            "benchmark/utils/visualize_rplan_json.py expects)."
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

    if args.model is None:
        args.model = "gpt-4"

    if args.output is None:
        slug = Path(args.model).name
        args.output = f"results_hd_{args.mode.replace('-', '_')}_{slug}_{args.seed}.json"
        logger.info("Auto-generated output: %s", args.output)

    cls = MODE_MAP[args.mode]
    evaluator = cls(args)
    return evaluator.run()


if __name__ == "__main__":
    sys.exit(cli())
