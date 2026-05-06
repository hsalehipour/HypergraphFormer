#!/usr/bin/env python3
"""
Convert a hypergraph dataset JSON (e.g. wmr24.json) to individual DiffPlanner
JSON files compatible with ``benchmark/src/evaluate_diffplanner.py`` and
DiffPlanner's own ``RPlanDataset`` / ``post_processing.py``.

Target format per file (one per entry) -- mirrors the schema documented at
``DiffPlanner/dataset/dataset_json/readme.txt`` and validated against
``dataset/rplan/dataset_diffplanner/data_test.json``::

    {
      "name": "<sample_id>",
      "boundary":        [[x, y, dir, isNew], ...],   # closed polygon, ints in [1,254]
                                                      # first two corners delimit the entrance
                                                      # dir: 0=right, 1=down, 2=left, 3=up
                                                      # isNew=1 for entrance points (not real corners)
      "boundary_expand": [[x, y], ...],               # boundary expanded to exactly 40 points
      "entrance_expand": [[x, y], ...],               # 4-corner rectangle around entrance, 8 px wide
      "rooms": [
          {
            "id": 0, "category": 0,
            "size": <int>, "location": [cx, cy],
            "box": [x0, y0, x1, y1],
            "r_boundary": [[x, y], ...],
            "order": <int>,
          }, ...
      ],
      "adjacencies": [[id_a, id_b], ...]
    }

Room type mapping (hypergraph mergeid -> DiffPlanner 0..5):
    living  -> 0 (LivingRoom)
    foyer   -> 0 (LivingRoom)   [merged with living]
    bed     -> 1 (Bedroom)
    kitchen -> 2 (Kitchen)
    bath    -> 3 (Bathroom)
    extra   -> 5 (Storage)
    (DiffPlanner category 4 = Balcony has no equivalent in WMR24.)

Validity gates (samples that fail any of these are skipped, like
``data_preparation.py`` does for the RPLAN .mat sources):
  * boundary needs <= 40 corners after closing
  * every coordinate must lie in (0, 256) -- we use (0, 255) post-rounding
  * each room polygon must have >= 4 vertices and area >= 150
  * sum of room areas must equal the boundary area (perfect tiling)

Usage::

    python benchmark/utils/convert_hypergraph_to_diffplanner.py \\
        -i dataset/wmr24/wmr24.json \\
        -o dataset/wmr24/diffplanner_json

Requires RGL (run with .venv).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from copy import deepcopy
from pathlib import Path

from shapely.geometry import Polygon
from shapely.ops import unary_union

from utils.hypergraph import Hypergraph

logger = logging.getLogger(__name__)


# ── Room type mapping (mergeid -> DiffPlanner 0..5 category id) ────────

MERGEID_TO_DP_CATEGORY: dict[str, int] = {
    "living":  0,
    "foyer":   0,   # merged with LivingRoom
    "bed":     1,
    "kitchen": 2,
    "bath":    3,
    "extra":   5,   # Storage
    # NOTE: DiffPlanner category 4 = Balcony; WMR24 has no balcony class.
}

# DiffPlanner uses a 256x256 integer canvas. The original
# data_preparation.py validates every coordinate with ``> 0 and < 256``,
# AND the entrance rectangle is padded ±ENTRANCE_HALF_WIDTH pixels to
# the OUTSIDE of the wall -- so the boundary itself must keep at least
# ``ENTRANCE_HALF_WIDTH + 1`` pixels of margin on every side, otherwise
# the entrance rectangle would spill out of the canvas.
#
# We therefore normalize the apartment polygon to fit in
# ``[COORD_MIN, COORD_MAX]`` with COORD_MIN/MAX picked so that the
# entrance pad always stays inside ``(0, CANVAS_SIZE)``. The same range
# also bounds the room polygons by construction (they live inside the
# boundary).
CANVAS_SIZE = 256
ENTRANCE_HALF_WIDTH = 8
COORD_MIN = ENTRANCE_HALF_WIDTH + 1            # 9
COORD_MAX = CANVAS_SIZE - ENTRANCE_HALF_WIDTH - 1   # 246

# Boundary corner cap is fixed by DiffPlanner's transformer
# (boundary_expand is always 40 points).
MAX_BOUNDARY_CORNERS = 40

# DiffPlanner requires every room polygon to have area >= 150 px^2 in the
# 256x256 canvas (matches data_preparation.py).
MIN_ROOM_AREA = 150

# DiffPlanner's b+nc transformer tensorises rooms as ``8 x 5`` (see
# ``MAX_NUM_ROOMS`` in ``benchmark/src/evaluate_diffplanner.py``). Samples
# with more rooms cannot be evaluated, so the converter caps them by
# default. Override with ``--max-rooms`` to lift / tighten the cap.
DEFAULT_MAX_ROOMS = 8

# Polygon comparisons can wobble by a fraction of a pixel after rounding;
# we use these loose tolerances when checking the perfect-tiling gate.
AREA_TOL_FRAC = 0.02  # |sum(rooms.area) - boundary.area| / boundary.area
DIR_RIGHT, DIR_DOWN, DIR_LEFT, DIR_UP = 0, 1, 2, 3


# ── Geometry helpers (shared with HouseGAN++ converter in spirit) ─────

def _face_polygon(face) -> list[tuple[float, float]]:
    """Extract (x, y) vertex list from an RGL NMesh face."""
    pts = face.getPoints()
    n = int(getattr(pts, "Count", 0) or 0)
    return [(float(pts[i].X), float(pts[i].Y)) for i in range(n)]


def _normalize_polygons(
    polygons: list[list[tuple[float, float]]],
    extra: list[list[tuple[float, float]]] | None = None,
) -> tuple[list[list[tuple[int, int]]],
           list[list[tuple[int, int]]],
           float, float, float]:
    """Zero-center + scale polygons so they fit tightly in [COORD_MIN, COORD_MAX].

    Both the room polygons and any *extra* polygons (e.g. the boundary,
    facade segments) are normalized with the SAME affine transform so they
    stay registered. Coordinates are rounded to int.

    Returns ``(rooms_int, extra_int, scale, offset_x, offset_y)``.
    """
    extra = extra or []
    xs: list[float] = []
    ys: list[float] = []
    for verts in polygons + extra:
        for x, y in verts:
            xs.append(x)
            ys.append(y)
    if not xs:
        raise ValueError("no vertices to normalise")
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    span = max(max_x - min_x, max_y - min_y)
    target_span = (COORD_MAX - COORD_MIN)
    s = target_span / span if span > 0 else 1.0
    half = (COORD_MAX + COORD_MIN) / 2.0

    def _xform(verts):
        return [
            (
                int(round((x - cx) * s + half)),
                int(round((y - cy) * s + half)),
            )
            for x, y in verts
        ]

    return [_xform(p) for p in polygons], [_xform(p) for p in extra], s, cx, cy


def _dedupe_consecutive(verts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Drop consecutive duplicate vertices (incl. the closing one)."""
    out: list[tuple[int, int]] = []
    for p in verts:
        if not out or out[-1] != p:
            out.append(p)
    if len(out) > 1 and out[-1] == out[0]:
        out.pop()
    return out


def _polygon_area_int(verts: list[tuple[int, int]]) -> float:
    if len(verts) < 3:
        return 0.0
    poly = Polygon(verts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return float(poly.area)


def _orient_ccw(verts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Make sure the polygon is oriented counter-clockwise (DP convention)."""
    if len(verts) < 3:
        return verts
    # Signed shoelace area: positive = CCW in standard math coords. The
    # DiffPlanner boundary in data_test.json is CCW, so we match.
    s = 0.0
    n = len(verts)
    for i in range(n):
        x0, y0 = verts[i]
        x1, y1 = verts[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    if s < 0:
        return list(reversed(verts))
    return verts


def _edge_dir(p0: tuple[int, int], p1: tuple[int, int]) -> int | None:
    """Classify an axis-aligned edge into DiffPlanner's dir code.

    Returns 0/1/2/3 for right/down/left/up, or None for a non-axis-aligned
    or zero-length edge.
    """
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    if dx == 0 and dy == 0:
        return None
    if abs(dx) >= abs(dy):
        return DIR_RIGHT if dx > 0 else DIR_LEFT
    return DIR_DOWN if dy > 0 else DIR_UP


# ── boundary_expand & entrance_expand (ports of data_preparation.py) ──

def _expand_polygon(polygon: list[list[int]], target_size: int = 40) -> list[list[int]]:
    """Insert midpoints at the longest edge until the polygon has *target_size*
    vertices. Direct port of :func:`expand_polygon` in DiffPlanner's
    ``dataset/data_preparation.py``.
    """
    poly = deepcopy(polygon)

    def _longest_edge_idx(p):
        best, best_len = -1, -1.0
        n = len(p)
        for i in range(n):
            x0, y0 = p[i]
            x1, y1 = p[(i + 1) % n]
            length = (x1 - x0) ** 2 + (y1 - y0) ** 2
            if length > best_len:
                best_len = length
                best = i
        return best

    while len(poly) < target_size:
        i = _longest_edge_idx(poly)
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % len(poly)]
        mid = [int(round((x0 + x1) / 2)), int(round((y0 + y1) / 2))]
        poly.insert(i + 1, mid)
    return poly


def _entrance_rectangle(p0: tuple[int, int], p1: tuple[int, int]
                        ) -> list[list[int]] | None:
    """4-corner rectangle around the entrance edge ``(p0, p1)``.

    Direct port of the rectangle-construction block in
    ``data_preparation.py``: assumes axis-aligned entrance, 8 px half-width
    on either side of the wall. Returns ``None`` for non-axis-aligned
    entrances.
    """
    if p0[0] == p1[0]:
        # Vertical wall -> rectangle expands in x.
        min_x = int(p0[0] - ENTRANCE_HALF_WIDTH)
        max_x = int(p0[0] + ENTRANCE_HALF_WIDTH)
        min_y = int(min(p0[1], p1[1]))
        max_y = int(max(p0[1], p1[1]))
    elif p0[1] == p1[1]:
        # Horizontal wall -> rectangle expands in y.
        min_x = int(min(p0[0], p1[0]))
        max_x = int(max(p0[0], p1[0]))
        min_y = int(p0[1] - ENTRANCE_HALF_WIDTH)
        max_y = int(p0[1] + ENTRANCE_HALF_WIDTH)
    else:
        return None

    rect = [
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y],
    ]
    # Validity check matches DiffPlanner's strict (>0 & <256) gate.
    for x, y in rect:
        if not (0 < x < CANVAS_SIZE and 0 < y < CANVAS_SIZE):
            return None
    return rect


# ── Entrance edge detection from facade + circulation ─────────────────

def _front_door_edge_index(
    boundary_int: list[tuple[int, int]],
    circulation_segs: list[tuple[tuple[float, float], tuple[float, float]]],
    cx: float, cy: float, scale: float,
) -> int | None:
    """Return the index ``i`` of the boundary edge ``(b[i], b[i+1])`` that is
    closest to a circulation segment (i.e. the front door's wall).

    Falls back to the longest axis-aligned edge if no circulation overlap
    exists. Returns None when the boundary has no axis-aligned edge.
    """
    half = (COORD_MAX + COORD_MIN) / 2.0

    def _to_canvas(pt):
        return (
            int(round((pt[0] - cx) * scale + half)),
            int(round((pt[1] - cy) * scale + half)),
        )

    circ_int = [(_to_canvas(s), _to_canvas(e)) for s, e in circulation_segs]

    n = len(boundary_int)
    best_idx, best_overlap = None, -1.0

    for i in range(n):
        p0 = boundary_int[i]
        p1 = boundary_int[(i + 1) % n]
        d = _edge_dir(p0, p1)
        if d is None:
            continue
        for c0, c1 in circ_int:
            cd = _edge_dir(c0, c1)
            if cd is None:
                continue
            # Both must run along the same axis.
            if (d in (DIR_RIGHT, DIR_LEFT)) != (cd in (DIR_RIGHT, DIR_LEFT)):
                continue
            if d in (DIR_RIGHT, DIR_LEFT):
                if abs(p0[1] - c0[1]) > 2:  # within 2 px of the same y line
                    continue
                lo = max(min(p0[0], p1[0]), min(c0[0], c1[0]))
                hi = min(max(p0[0], p1[0]), max(c0[0], c1[0]))
            else:
                if abs(p0[0] - c0[0]) > 2:
                    continue
                lo = max(min(p0[1], p1[1]), min(c0[1], c1[1]))
                hi = min(max(p0[1], p1[1]), max(c0[1], c1[1]))
            overlap = hi - lo
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i

    if best_idx is not None and best_overlap > 0:
        return best_idx

    # Fallback: longest axis-aligned edge.
    best_idx, best_len = None, -1
    for i in range(n):
        p0 = boundary_int[i]
        p1 = boundary_int[(i + 1) % n]
        if _edge_dir(p0, p1) is None:
            continue
        seg_len = abs(p1[0] - p0[0]) + abs(p1[1] - p0[1])
        if seg_len > best_len:
            best_len = seg_len
            best_idx = i
    return best_idx


def _build_boundary(
    boundary_int: list[tuple[int, int]],
    entrance_edge_idx: int,
) -> list[list[int]] | None:
    """Build DiffPlanner's ``boundary`` field.

    Steps:
      1. Rotate the polygon so that the entrance edge starts at index 0.
      2. Insert two new vertices on the entrance edge so the entrance is
         exactly the segment from index 0 -> index 1 (matches DP's "first
         two corners delimit the entrance" convention).
      3. For each vertex emit ``[x, y, dir, isNew]`` where dir is the
         orientation of the edge that LEAVES that vertex and isNew is 1
         iff the vertex is a synthetic entrance corner.

    Returns None if the entrance edge is too short to host an 8 px door
    or the resulting polygon would exceed MAX_BOUNDARY_CORNERS.
    """
    n = len(boundary_int)
    rotated = boundary_int[entrance_edge_idx:] + boundary_int[:entrance_edge_idx]

    p0 = rotated[0]
    p1 = rotated[1]
    d = _edge_dir(p0, p1)
    if d is None:
        return None

    # Place the door so it sits in the middle of the entrance edge,
    # exactly 2*ENTRANCE_HALF_WIDTH pixels long.
    ex0, ey0 = p0
    ex1, ey1 = p1
    edge_dx = ex1 - ex0
    edge_dy = ey1 - ey0
    edge_len = abs(edge_dx) + abs(edge_dy)  # axis-aligned, so L1 == L2
    if edge_len < 2 * ENTRANCE_HALF_WIDTH + 2:
        return None

    door_len = 2 * ENTRANCE_HALF_WIDTH
    sign_x = (edge_dx > 0) - (edge_dx < 0)
    sign_y = (edge_dy > 0) - (edge_dy < 0)
    midx = (ex0 + ex1) / 2.0
    midy = (ey0 + ey1) / 2.0

    door_a = (
        int(round(midx - sign_x * door_len / 2.0)),
        int(round(midy - sign_y * door_len / 2.0)),
    )
    door_b = (
        int(round(midx + sign_x * door_len / 2.0)),
        int(round(midy + sign_y * door_len / 2.0)),
    )

    # Build the new vertex list: [p0, door_a, door_b, p1, p2, ...].
    new_verts: list[tuple[int, int, bool]] = []
    new_verts.append((p0[0], p0[1], False))
    new_verts.append((door_a[0], door_a[1], True))
    new_verts.append((door_b[0], door_b[1], True))
    for v in rotated[1:]:
        new_verts.append((v[0], v[1], False))

    # The DiffPlanner convention is "first two corners delimit the entrance",
    # i.e. the entrance segment is index 0 -> 1. Make that hold by rotating
    # so door_a (the first synthetic point) is at index 0.
    rot = 1
    new_verts = new_verts[rot:] + new_verts[:rot]

    if len(new_verts) > MAX_BOUNDARY_CORNERS:
        return None

    out: list[list[int]] = []
    n_new = len(new_verts)
    for i in range(n_new):
        x, y, is_new = new_verts[i]
        nx, ny, _ = new_verts[(i + 1) % n_new]
        d_out = _edge_dir((x, y), (nx, ny))
        if d_out is None:
            return None
        out.append([int(x), int(y), int(d_out), int(is_new)])
    return out


# ── Core conversion ──────────────────────────────────────────────────

def convert_entry(entry: dict, *, max_rooms: int = DEFAULT_MAX_ROOMS) -> dict | None:
    """Convert a single hypergraph entry to a DiffPlanner JSON dict.

    Returns None on failure (the caller logs and counts). Failure modes:
    missing/invalid geometry, mergeid we cannot map, ``> max_rooms`` rooms,
    boundary > 40 corners, coordinates out of [COORD_MIN, COORD_MAX], room
    area < 150 px^2, or the rooms do not tile the boundary.

    Args:
        entry: one record from the WMR24 list.
        max_rooms: cap on number of rooms (default :data:`DEFAULT_MAX_ROOMS`,
            i.e. 8 to match DiffPlanner's b+nc transformer). Pass ``0`` to
            disable.
    """
    split_str = entry.get("split", "{}")
    hg_dict = json.loads(split_str) if isinstance(split_str, str) else split_str

    bounds_raw = entry.get("bounds", "")
    bounds = json.loads(bounds_raw) if isinstance(bounds_raw, str) and bounds_raw else bounds_raw
    if not bounds or not isinstance(bounds, dict):
        return None

    corners = bounds.get("corners") or []
    if len(corners) < 3:
        return None
    boundary_world = [(float(c["X"]), float(c["Y"])) for c in corners]

    hg = Hypergraph(hg_dict)
    leaves = hg.leaves
    n_rooms = len(leaves)
    if n_rooms == 0:
        return None
    if max_rooms and n_rooms > max_rooms:
        return None

    # Reject samples whose mergeid we have no DiffPlanner mapping for.
    # Better to skip than to silently coerce 'storage' -> 'extra' etc.
    for leaf in leaves:
        mid = leaf.get("mergeid", "extra")
        if mid not in MERGEID_TO_DP_CATEGORY:
            return None

    mesh = hg.to_mesh(bounds=bounds)
    n_faces = int(getattr(getattr(mesh, "faceList", None), "Count", 0) or 0)
    if n_faces == 0:
        return None

    name_to_leaf_idx = {leaf["name"]: i for i, leaf in enumerate(leaves)}

    room_verts: list[list[tuple[float, float]]] = [[] for _ in range(n_rooms)]
    for fi in range(n_faces):
        f = mesh.faceList[fi]
        uid = str(getattr(f, "unique_id", ""))
        li = name_to_leaf_idx.get(uid, -1)
        if li < 0:
            continue
        verts = _face_polygon(f)
        if len(verts) < 3:
            continue
        poly = Polygon(verts)
        if poly.is_empty or poly.area <= 1e-6:
            continue
        if room_verts[li]:
            existing = Polygon(room_verts[li])
            merged = unary_union([existing, poly])
            if merged.geom_type == "Polygon":
                room_verts[li] = list(merged.exterior.coords[:-1])
            else:
                room_verts[li] = list(merged.convex_hull.exterior.coords[:-1])
        else:
            room_verts[li] = verts

    if any(not v for v in room_verts):
        return None

    # ── Normalize boundary + rooms together so they stay registered ────
    rooms_int_lists, extras_int, scale, off_cx, off_cy = _normalize_polygons(
        room_verts, extra=[boundary_world],
    )
    boundary_int = _dedupe_consecutive(extras_int[0])
    boundary_int = _orient_ccw(boundary_int)

    if len(boundary_int) < 3 or len(boundary_int) > MAX_BOUNDARY_CORNERS - 3:
        # Need room for two synthetic entrance corners (and a small margin).
        return None

    for x, y in boundary_int:
        if not (COORD_MIN <= x <= COORD_MAX and COORD_MIN <= y <= COORD_MAX):
            return None

    # ── Find entrance edge from the circulation segments ───────────────
    circ_raw = entry.get("circulation", "[]")
    circ_segs_parsed = json.loads(circ_raw) if isinstance(circ_raw, str) else (circ_raw or [])
    circ_world = [
        ((float(seg["start"]["X"]), float(seg["start"]["Y"])),
         (float(seg["end"]["X"]),   float(seg["end"]["Y"])))
        for seg in circ_segs_parsed
        if isinstance(seg, dict) and "start" in seg and "end" in seg
    ]
    entrance_idx = _front_door_edge_index(
        boundary_int, circ_world, off_cx, off_cy, scale,
    )
    if entrance_idx is None:
        return None

    boundary_field = _build_boundary(boundary_int, entrance_idx)
    if boundary_field is None:
        return None

    # boundary_expand: drop dir/isNew, close polygon, expand to 40 pts.
    expand_seed = [[p[0], p[1]] for p in boundary_field]
    expand_seed.append([boundary_field[0][0], boundary_field[0][1]])
    if len(expand_seed) > MAX_BOUNDARY_CORNERS:
        return None
    boundary_expand = _expand_polygon(expand_seed, MAX_BOUNDARY_CORNERS)

    # entrance_expand: 4-corner rectangle around the first edge.
    entrance_expand = _entrance_rectangle(
        (boundary_field[0][0], boundary_field[0][1]),
        (boundary_field[1][0], boundary_field[1][1]),
    )
    if entrance_expand is None:
        return None

    # ── Rooms: validate, integerize, build the per-room records ────────
    boundary_polygon_area = _polygon_area_int(boundary_int)
    if boundary_polygon_area <= 0:
        return None

    rooms: list[dict] = []
    rooms_total_area = 0.0
    for li, verts_int in enumerate(rooms_int_lists):
        verts = _dedupe_consecutive(verts_int)
        if len(verts) < 4:
            return None
        for x, y in verts:
            if not (COORD_MIN <= x <= COORD_MAX and COORD_MIN <= y <= COORD_MAX):
                return None
        area = _polygon_area_int(verts)
        if area < MIN_ROOM_AREA:
            return None

        xs = [p[0] for p in verts]
        ys = [p[1] for p in verts]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        if x1 <= x0 or y1 <= y0:
            return None

        cx = int(round((x0 + x1) / 2))
        cy = int(round((y0 + y1) / 2))
        size = int((x1 - x0) * (y1 - y0))   # box-area, NOT polygon area:
                                            # matches data_preparation.py.

        category = MERGEID_TO_DP_CATEGORY[leaves[li].get("mergeid", "extra")]

        rooms.append({
            "id": int(li),
            "category": int(category),
            "size": int(size),
            "location": [cx, cy],
            "box": [int(x0), int(y0), int(x1), int(y1)],
            "r_boundary": [[int(x), int(y)] for x, y in verts],
            "order": int(li),
        })
        rooms_total_area += area

    # Perfect-tiling gate (area equality is the cheapest proxy for "rooms
    # fill the boundary exactly", which DiffPlanner's data_preparation.py
    # also enforces strictly).
    rel_err = abs(rooms_total_area - boundary_polygon_area) / boundary_polygon_area
    if rel_err > AREA_TOL_FRAC:
        return None

    # ── Adjacencies (room-room, from the access graph) ────────────────
    adjacencies: list[list[int]] = []
    seen: set[tuple[int, int]] = set()
    ag = hg.access_graph
    for e in ag.get("edges", []):
        a, b = int(e[0]) - 1, int(e[1]) - 1
        if a == b or a < 0 or b < 0 or a >= n_rooms or b >= n_rooms:
            continue
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        adjacencies.append([key[0], key[1]])
    adjacencies.sort()

    return {
        "name": str(entry.get("id", "")),
        "boundary": boundary_field,
        "boundary_expand": boundary_expand,
        "entrance_expand": entrance_expand,
        "rooms": rooms,
        "adjacencies": adjacencies,
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Convert hypergraph JSON to DiffPlanner per-sample JSON files.",
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True,
        help="Path to input JSON (e.g. dataset/wmr24/wmr24.json)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output directory for per-sample JSON files "
             "(default: alongside input, named <stem>_diffplanner_json/)",
    )
    parser.add_argument(
        "--max-rooms", type=int, default=DEFAULT_MAX_ROOMS,
        help=f"Skip samples with more than this many rooms "
             f"(default: {DEFAULT_MAX_ROOMS}, matches DiffPlanner's b+nc "
             "transformer; pass 0 to disable).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output is None:
        output_dir = input_path.parent / f"{input_path.stem}_diffplanner_json"
    else:
        output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path) as f:
        data = json.load(f)

    logger.info("Loaded %d entries from %s", len(data), input_path)
    logger.info("Output directory: %s", output_dir)
    logger.info("max_rooms cap: %s",
                f"{args.max_rooms}" if args.max_rooms else "disabled")

    success = 0
    failed = 0          # generic failure (geometry, validity gate, etc.)
    skipped_rooms = 0   # too many rooms -- broken out so the user sees it
    failed_ids: list[str] = []
    skipped_ids: list[str] = []
    all_results: list[dict] = []  # accumulated for combined data_test.json

    for i, entry in enumerate(data):
        eid = entry.get("id", f"sample_{i}")

        # Cheap pre-check so we can distinguish "too many rooms" from
        # genuine conversion failures in the summary.
        too_many = False
        if args.max_rooms:
            split_str = entry.get("split", "{}")
            try:
                hg_dict = json.loads(split_str) if isinstance(split_str, str) else split_str
                hg = Hypergraph(hg_dict)
                if len(hg.leaves) > args.max_rooms:
                    too_many = True
            except Exception:
                pass  # fall through to convert_entry, which will fail cleanly

        if too_many:
            skipped_rooms += 1
            skipped_ids.append(eid)
            continue

        try:
            result = convert_entry(entry, max_rooms=args.max_rooms)
        except Exception as exc:
            logger.debug("Entry %s failed: %s", eid, exc)
            result = None

        if result is None:
            failed += 1
            failed_ids.append(eid)
            continue

        out_path = output_dir / f"{eid}.json"
        with open(out_path, "w") as f:
            json.dump(result, f)
        all_results.append(result)
        success += 1

        if (i + 1) % 100 == 0:
            logger.info("  processed %d/%d ...", i + 1, len(data))

    excluded = set(failed_ids) | set(skipped_ids)
    list_path = output_dir / "list.txt"
    with open(list_path, "w") as f:
        for entry in data:
            eid = entry.get("id", "")
            if eid and eid not in excluded:
                f.write(f"{eid}.json\n")

    # Combined dataset file matching DiffPlanner's data_test.json schema
    # (a single JSON array of all converted samples). Useful for tools
    # that consume the dataset as one file rather than per-sample shards.
    combined_path = output_dir / "data_test.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Converted:        %d / %d", success, len(data))
    logger.info("Skipped (>rooms): %d", skipped_rooms)
    logger.info("Failed:           %d", failed)
    if skipped_ids:
        logger.info("Skipped IDs (first 20): %s", skipped_ids[:20])
    if failed_ids:
        logger.info("Failed IDs (first 20):  %s", failed_ids[:20])
    logger.info("Output:        %s", output_dir)
    logger.info("List file:     %s", list_path)
    logger.info("Combined file: %s", combined_path)
    logger.info("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
