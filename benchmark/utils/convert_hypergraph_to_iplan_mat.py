#!/usr/bin/env python3
"""Convert WMR24 hypergraph JSON to iPLAN ``.mat`` files.

This is the iPLAN counterpart of :mod:`convert_hypergraph_to_diffplanner`
and :mod:`convert_png_to_iplan_mat`. It lets ``evaluate_iplan.py`` run
on WMR24 (where we don't have RPLAN-style PNGs to feed iPLAN's own
:class:`floorplan.FloorPlan` parser) by producing ``.mat`` files in
exactly the schema iPLAN's preprocessor would emit.

Source format
-------------
One entry of ``dataset/wmr24/wmr24.json`` (see
:mod:`convert_hypergraph_to_diffplanner` for the full schema): an
RGL/hypergraph record with ``bounds`` (boundary polygon),
``circulation`` (front-door edge candidates), and ``split`` (the BSP
hypergraph that decomposes the floorplan into rooms).

Target format
-------------
One ``.mat`` per sample. The MATLAB file holds a single struct ``data``
with the fields iPLAN's :class:`floorplan.FloorPlan.to_dict` produces:

  * ``name``      -- scalar string, sample id (no extension)
  * ``Boundary``  -- ``(N_v, 4)`` int, ``(y, x, dir, isNew)`` rows;
                     polygon is closed; first two vertices delimit the
                     front door (the second has ``isNew == 1`` and is
                     the entrance start). ``dir ∈ {0,1,2,3}`` =
                     right / down / left / up of the edge that *leaves*
                     this vertex.
  * ``gt_rTypes`` -- ``(1, R)`` int, room categories in iPLAN's 13-class
                     vocabulary (0..12, see :data:`IPLAN_LABEL_NAME`),
                     SORTED ASCENDING (matches floorplan.py:55).
  * ``gt_rBoxes`` -- ``(R, 4)`` int, axis-aligned bboxes
                     ``(y0, x0, y1, x1)`` in 128-grid pixels, in the
                     same order as ``gt_rTypes``.
  * ``rTypes``, ``rCenters``, ``rBoxes`` -- empty arrays
                     (filled in by inference; iPLAN initialises these
                     to ``np.array([])`` in :func:`FloorPlan.to_dict`).

Coordinate frame: iPLAN works on a 128×128 integer grid with rows = y,
cols = x. We honour both the (y, x) ordering AND the 128-grid scale.

Room type mapping (WMR24 mergeid -> iPLAN 0..12 category id)
------------------------------------------------------------
WMR24's ``mergeid`` taxonomy is coarser than iPLAN's, so the mapping is
many-to-one for bedrooms (iPLAN has Master / Child / Second / Guest;
WMR24 has only ``bed``). We assign bedroom subtypes deterministically:

    living  -> 0  (LivingRoom)
    foyer   -> 10 (Entrance)
    kitchen -> 2  (Kitchen)
    bath    -> 3  (Bathroom)
    extra   -> 11 (Storage)
    bed     -> 1  (MasterRoom) for the first bedroom in a sample
               -> 7  (SecondRoom) for the second
               -> 8  (GuestRoom) for the third
               -> 5  (ChildRoom) for the fourth+

The bedroom-subtype tie-breaker order matches the empirical frequency
of these classes in RPLAN's iPLAN-format test split (Master >> Second
>> Guest > Child), so the resulting type histograms look like real
RPLAN data.

Validity gates (sample dropped if ANY fails)
--------------------------------------------
  * boundary polygon has at least 3 corners after dedup
  * boundary fits inside the 128x128 canvas with a 1-px margin
  * every room has >= 4 vertices
  * every room bbox has positive width and height in 128-grid pixels
  * every mergeid is in the table above
  * a circulation segment overlaps an axis-aligned boundary edge
    (otherwise we cannot identify the entrance edge)
  * the entrance edge is at least 6 px long after rasterisation
    (so the door can fit in the middle of it)

Usage
-----
    python benchmark/utils/convert_hypergraph_to_iplan_mat.py \\
        -i dataset/wmr24/wmr24.json \\
        -o dataset/wmr24/dataset_iplan

Run with the ``.venv`` (needs RGL via :mod:`utils.hypergraph`); the
output ``.mat`` files are then consumable by ``evaluate_iplan.py``
running in the ``iplan`` conda env.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from shapely.geometry import Polygon
from shapely.ops import unary_union

from utils.hypergraph import Hypergraph

# Reuse geometry helpers from the DiffPlanner converter so the two
# scripts can never drift apart on shared concerns (BSP face -> polygon,
# vertex dedup, CCW orientation, edge classification, front-door edge
# detection). Anything iPLAN-specific (canvas size, (y,x) ordering,
# door-vertex emission) lives below in this module.
from benchmark.utils.convert_hypergraph_to_diffplanner import (  # noqa: E402
    _dedupe_consecutive,
    _edge_dir,
    _face_polygon,
    _front_door_edge_index,
    _orient_ccw,
    _polygon_area_int,
    DIR_DOWN,
    DIR_LEFT,
    DIR_RIGHT,
    DIR_UP,
)

logger = logging.getLogger(__name__)


# ── iPLAN canvas + room-vocabulary constants ──────────────────────────

IPLAN_CANVAS_SIZE = 128

# iPLAN's own preprocessing (see ``floorplan.get_exterior_boundary``)
# pads the inside-mask scan window by +/- 8 px before tracing the
# boundary, so its preprocessed RPLAN .mat files NEVER use coordinates
# above 121 (verified empirically across all 10k test mats:
# max(y) = max(x) = 121, min = 6). Several downstream iPLAN
# post-processing steps assume that margin without bounds-checking,
# notably ``synth/test_roompartition.obtain_living`` which does
# ``living_mask[h+2, w]`` with ``h`` ranging up to ``max_x + 4`` and
# crashes with "index 128 is out of bounds for dimension 0 with size
# 128" when the boundary touches anything tighter than ~ [5, 122].
#
# We therefore pack the floorplan into [6, 121] -- the exact extreme
# range iPLAN's own RPLAN .mat files use -- so our converted samples
# are byte-shape AND value-range compatible with the rest of the iPLAN
# inference pipeline.
COORD_MIN = 6
COORD_MAX = 121

# Door endpoints are inserted into the boundary as TWO new vertices on
# the entrance edge (matches ``floorplan.get_exterior_boundary``'s
# ``new_p`` pair). We size the door at ~5 px (ratio of the boundary edge
# length, capped) -- iPLAN's data has door spans of 4-12 px on the
# 128-grid.
DOOR_LEN_PX = 5
MIN_ENTRANCE_EDGE_LEN_PX = DOOR_LEN_PX + 2  # +1 px on each side of the door

# WMR24 mergeid -> iPLAN room-id. Bedrooms are resolved per-sample (see
# :func:`_assign_iplan_categories`).
SIMPLE_MERGEID_TO_IPLAN: dict[str, int] = {
    "living":  0,   # LivingRoom
    "foyer":   10,  # Entrance
    "kitchen": 2,   # Kitchen
    "bath":    3,   # Bathroom
    "extra":   11,  # Storage
}

# Order in which we hand out iPLAN bedroom subtypes when a sample has
# multiple ``mergeid == "bed"`` rooms. Matches the empirical RPLAN
# frequency (Master >> Second >> Guest > Child).
BEDROOM_SUBTYPE_ORDER: list[int] = [
    1,   # MasterRoom
    7,   # SecondRoom
    8,   # GuestRoom
    5,   # ChildRoom
]

# iPLAN's full 13-class label table (kept as a quick reference; only
# used in error messages).
IPLAN_LABEL_NAME: dict[int, str] = {
    0:  "LivingRoom",
    1:  "MasterRoom",
    2:  "Kitchen",
    3:  "Bathroom",
    4:  "DiningRoom",
    5:  "ChildRoom",
    6:  "StudyRoom",
    7:  "SecondRoom",
    8:  "GuestRoom",
    9:  "Balcony",
    10: "Entrance",
    11: "Storage",
    12: "Wall-in",
}


# ── iPLAN-specific helpers ────────────────────────────────────────────

def _assign_iplan_categories(mergeids: list[str]) -> list[int] | None:
    """Map a sample's per-room WMR24 mergeids onto iPLAN category IDs.

    Returns the per-room iPLAN ID list, in the SAME order as
    ``mergeids``. Returns None if any mergeid is unmappable, or if the
    sample has more bedrooms than we have iPLAN bedroom subtypes
    (4) -- iPLAN has no "generic bedroom" bucket so we cannot
    represent floorplans with > 4 bedrooms.
    """
    out: list[int] = [-1] * len(mergeids)
    bed_indices = [i for i, m in enumerate(mergeids) if m == "bed"]
    if len(bed_indices) > len(BEDROOM_SUBTYPE_ORDER):
        return None
    for slot, idx in enumerate(bed_indices):
        out[idx] = BEDROOM_SUBTYPE_ORDER[slot]
    for i, m in enumerate(mergeids):
        if out[i] != -1:
            continue
        if m not in SIMPLE_MERGEID_TO_IPLAN:
            return None
        out[i] = SIMPLE_MERGEID_TO_IPLAN[m]
    return out


def _normalize_polygons_iplan(
    polygons: list[list[tuple[float, float]]],
    extra: list[list[tuple[float, float]]] | None = None,
) -> tuple[list[list[tuple[int, int]]],
           list[list[tuple[int, int]]],
           float, float, float]:
    """Centre-fit + scale polygons into ``[COORD_MIN, COORD_MAX]``.

    Same affine for ``polygons`` and ``extra`` so they stay registered
    on the 128-grid. We deliberately reimplement this rather than
    importing the DiffPlanner version because the target span /
    rounding tolerances are different.

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


def _build_iplan_boundary(
    boundary_xy: list[tuple[int, int]],
    entrance_edge_idx: int,
) -> np.ndarray | None:
    """Build iPLAN's ``Boundary`` array.

    iPLAN's :func:`get_exterior_boundary` rotates the polygon so that
    the first vertex is the *start* of the front door, then inserts up
    to two synthetic vertices (with ``isNew == 1``) for the door
    endpoints. We mimic the same shape:

      * Rotate so the entrance edge starts at index 0.
      * Insert two new vertices on the entrance edge to bracket a
        DOOR_LEN_PX-long door at the centre of the edge.
      * Drop the original vertex at index 0 and rotate again so the
        first synthetic door vertex sits at index 0 -- this gives a
        polygon whose first two vertices delimit the door (the iPLAN
        convention used by ``test_roomlocation.py`` and friends).
      * Emit ``(y, x, dir, isNew)`` rows where ``dir`` is the direction
        of the edge that LEAVES this vertex.

    Returns the ``(N, 4)`` int array, or ``None`` if the entrance edge
    is too short or has a non-axis-aligned section that we cannot
    classify.
    """
    n = len(boundary_xy)
    rotated = boundary_xy[entrance_edge_idx:] + boundary_xy[:entrance_edge_idx]

    p0 = rotated[0]
    p1 = rotated[1]
    d = _edge_dir(p0, p1)
    if d is None:
        return None

    edge_dx = p1[0] - p0[0]
    edge_dy = p1[1] - p0[1]
    edge_len = abs(edge_dx) + abs(edge_dy)  # axis-aligned -> L1 == L2
    if edge_len < MIN_ENTRANCE_EDGE_LEN_PX:
        return None

    # Door endpoints, centred on the edge.
    sign_x = (edge_dx > 0) - (edge_dx < 0)
    sign_y = (edge_dy > 0) - (edge_dy < 0)
    midx = (p0[0] + p1[0]) / 2.0
    midy = (p0[1] + p1[1]) / 2.0
    door_a = (
        int(round(midx - sign_x * DOOR_LEN_PX / 2.0)),
        int(round(midy - sign_y * DOOR_LEN_PX / 2.0)),
    )
    door_b = (
        int(round(midx + sign_x * DOOR_LEN_PX / 2.0)),
        int(round(midy + sign_y * DOOR_LEN_PX / 2.0)),
    )

    # Build the new vertex list: [p0, door_a, door_b, p1, p2, ...].
    # Each entry is (x, y, isNew).
    new_verts: list[tuple[int, int, bool]] = []
    new_verts.append((p0[0], p0[1], False))
    new_verts.append((door_a[0], door_a[1], True))
    new_verts.append((door_b[0], door_b[1], True))
    for v in rotated[1:]:
        new_verts.append((v[0], v[1], False))

    # iPLAN's convention: the FIRST vertex is the start of the door,
    # i.e. the polygon list begins with door_a (the synthetic vertex we
    # just inserted right after p0). Rotate to make that hold.
    rot = 1
    new_verts = new_verts[rot:] + new_verts[:rot]

    # Emit rows in iPLAN's (y, x, dir, isNew) order.
    out_rows: list[tuple[int, int, int, int]] = []
    n_new = len(new_verts)
    for i in range(n_new):
        x, y, is_new = new_verts[i]
        nx, ny, _ = new_verts[(i + 1) % n_new]
        d_out = _edge_dir((x, y), (nx, ny))
        if d_out is None:
            return None
        out_rows.append((int(y), int(x), int(d_out), int(is_new)))

    return np.array(out_rows, dtype=int)


# ── Core conversion ──────────────────────────────────────────────────

def convert_entry(entry: dict) -> dict | None:
    """Convert a single hypergraph entry to an iPLAN ``.mat`` dict.

    Returns the dict to pass to ``scipy.io.savemat`` (already wrapped
    as ``{'data': <fields>}``-equivalent), or None if the entry fails
    any validity gate.
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

    mergeids = [leaf.get("mergeid", "extra") for leaf in leaves]
    iplan_cats = _assign_iplan_categories(mergeids)
    if iplan_cats is None:
        return None

    # Build per-room polygons by unioning all BSP faces that share a
    # leaf id. Mirrors the diffplanner converter logic verbatim.
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
                # Multipolygon: fall back to convex hull. Rare in WMR24
                # but worth handling so we don't drop the sample.
                room_verts[li] = list(merged.convex_hull.exterior.coords[:-1])
        else:
            room_verts[li] = verts

    if any(not v for v in room_verts):
        return None

    # ── Normalise into the 128-grid (rooms + boundary together) ────
    rooms_int_lists, extras_int, scale, off_cx, off_cy = _normalize_polygons_iplan(
        room_verts, extra=[boundary_world],
    )
    boundary_xy = _dedupe_consecutive(extras_int[0])
    boundary_xy = _orient_ccw(boundary_xy)

    if len(boundary_xy) < 3:
        return None
    for x, y in boundary_xy:
        if not (COORD_MIN <= x <= COORD_MAX and COORD_MIN <= y <= COORD_MAX):
            return None

    # ── Find entrance edge from circulation segments ───────────────
    circ_raw = entry.get("circulation", "[]")
    circ_segs_parsed = json.loads(circ_raw) if isinstance(circ_raw, str) else (circ_raw or [])
    circ_world = [
        ((float(seg["start"]["X"]), float(seg["start"]["Y"])),
         (float(seg["end"]["X"]),   float(seg["end"]["Y"])))
        for seg in circ_segs_parsed
        if isinstance(seg, dict) and "start" in seg and "end" in seg
    ]
    entrance_idx = _front_door_edge_index(
        boundary_xy, circ_world, off_cx, off_cy, scale,
    )
    if entrance_idx is None:
        return None

    boundary_arr = _build_iplan_boundary(boundary_xy, entrance_idx)
    if boundary_arr is None:
        return None

    # ── Build per-room bboxes in iPLAN's (y0, x0, y1, x1) order ────
    rBoxes_yx: list[tuple[int, int, int, int]] = []
    for verts_int in rooms_int_lists:
        verts = _dedupe_consecutive(verts_int)
        if len(verts) < 4:
            return None
        for x, y in verts:
            if not (COORD_MIN <= x <= COORD_MAX and COORD_MIN <= y <= COORD_MAX):
                return None
        if _polygon_area_int(verts) <= 0:
            return None
        xs = [p[0] for p in verts]
        ys = [p[1] for p in verts]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        if x1 <= x0 or y1 <= y0:
            return None
        rBoxes_yx.append((int(y0), int(x0), int(y1), int(x1)))

    # iPLAN sorts rBoxes/rTypes by ascending rType (see floorplan.py:55).
    # Replicate that exactly so downstream code can't tell the
    # difference between our output and a real iPLAN-converted .mat.
    rTypes_arr = np.array(iplan_cats, dtype=int)
    rBoxes_arr = np.array(rBoxes_yx, dtype=int)
    order = np.argsort(rTypes_arr, kind="stable")
    gt_rTypes = rTypes_arr[order].reshape(1, -1)   # (1, R)
    gt_rBoxes = rBoxes_arr[order, :]               # (R, 4)

    # Field order matches iPLAN's :meth:`FloorPlan.to_dict` exactly
    # (name, gt_rTypes, gt_rBoxes, Boundary, rTypes, rBoxes, rCenters)
    # so the resulting struct array is byte-shape-identical to a real
    # iPLAN-converted .mat and downstream code that iterates
    # ``dtype.names`` in order can't tell the difference.
    name = str(entry.get("id", ""))
    return {
        "name":      name,
        "gt_rTypes": gt_rTypes.astype(int),
        "gt_rBoxes": gt_rBoxes.astype(int),
        "Boundary":  boundary_arr.astype(int),
        "rTypes":    np.array([]),
        "rBoxes":    np.array([]),
        "rCenters":  np.array([]),
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Convert hypergraph JSON to per-sample iPLAN .mat files.",
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True,
        help="Path to input JSON (e.g. dataset/wmr24/wmr24.json)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output directory for .mat files "
             "(default: alongside input, named <stem>_iplan/)",
    )
    args = parser.parse_args()

    # Sanitise paths -- Rule 1 (Secure Python Development): user input
    # must not be used in file ops without validation.
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        return 2

    if args.output is None:
        output_dir = input_path.parent / f"{input_path.stem}_iplan"
    else:
        output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path) as f:
        data = json.load(f)

    logger.info("Loaded %d entries from %s", len(data), input_path)
    logger.info("Output directory: %s", output_dir)

    success = 0
    failed = 0
    failed_ids: list[str] = []

    for i, entry in enumerate(data):
        eid = entry.get("id", f"sample_{i}")
        try:
            result = convert_entry(entry)
        except Exception as exc:
            logger.debug("Entry %s failed: %s", eid, exc)
            result = None

        if result is None:
            failed += 1
            failed_ids.append(eid)
            continue

        out_path = output_dir / f"{eid}.mat"
        # iPLAN writes the whole record under the top-level key 'data',
        # as a struct -- exactly what scipy.io.savemat does when given
        # ``{'data': dict}`` and the dict's values are scalars/arrays.
        sio.savemat(out_path, {"data": result})
        success += 1

        if (i + 1) % 100 == 0:
            logger.info("  processed %d/%d ...", i + 1, len(data))

    excluded = set(failed_ids)
    list_path = output_dir / "list.txt"
    with open(list_path, "w") as f:
        for entry in data:
            eid = entry.get("id", "")
            if eid and eid not in excluded:
                f.write(f"{eid}.mat\n")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Converted: %d / %d", success, len(data))
    logger.info("Failed:    %d", failed)
    if failed_ids:
        logger.info("Failed IDs (first 20): %s", failed_ids[:20])
    logger.info("Output:    %s", output_dir)
    logger.info("List file: %s", list_path)
    logger.info("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
