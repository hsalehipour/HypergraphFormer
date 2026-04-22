#!/usr/bin/env python3
"""
Convert a hypergraph dataset JSON (e.g. wmr24.json) to individual HouseGAN++
JSON files compatible with ``convert_json_to_npz.py``.

Target format per file (one per entry)::

    {
      "room_type": [1, 3, 2, 4, ...],          # RPLAN room-type IDs
      "boxes":     [[x0,y0,x1,y1], ...],        # bounding boxes on 256×256 grid
      "edges":     [[x0,y0,x1,y1,type,aux],...], # wall segments on 256×256 grid
      "ed_rm":     [[i], [i,j], ...]             # room indices per wall segment
    }

Room type mapping (hypergraph -> RPLAN):
    living  -> 1   bed    -> 3   kitchen -> 2
    bath    -> 4   foyer  -> 6   extra   -> 10
    door    -> 17  front  -> 15

Usage:
    python benchmark/utils/convert_hypergraph_to_housegan_json.py \\
        -i dataset/wmr24/wmr24.json \\
        -o dataset/wmr24/housegan_json

Requires RGL (run with .venv).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import scale as _scale

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.hypergraph import Hypergraph

logger = logging.getLogger(__name__)

# ── Room type mappings ─────────────────────────────────────────────────

MERGEID_TO_RPLAN: dict[str, int] = {
    "living": 1,
    "bed": 3,
    "kitchen": 2,
    "bath": 4,
    "foyer": 6,
    "extra": 10,
}

DOOR_TYPE = 17
FRONT_DOOR_TYPE = 15

SCALE_FACTOR_ADJ = 1.05
IMG_SIZE = 256
TOLERANCE = 0.05


# ── Geometry helpers ───────────────────────────────────────────────────

def _face_polygon(face) -> list[tuple[float, float]]:
    """Extract (x, y) vertex list from an RGL NMesh face."""
    pts = face.getPoints()
    n = int(getattr(pts, "Count", 0) or 0)
    return [(float(pts[i].X), float(pts[i].Y)) for i in range(n)]


def _bbox(vertices: list[tuple[float, float]]) -> list[float]:
    """Axis-aligned bounding box [min_x, min_y, max_x, max_y]."""
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    return [min(xs), min(ys), max(xs), max(ys)]


def _edges_from_vertices(vertices: list[tuple[float, float]]) -> list[tuple]:
    """Ordered edge segments [(x0,y0,x1,y1), ...] forming a closed polygon."""
    segs = []
    n = len(vertices)
    for i in range(n):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % n]
        segs.append((x0, y0, x1, y1))
    return segs


def _find_shared_wall(verts_a, verts_b, tolerance=TOLERANCE):
    """Find the longest collinear overlap between two rooms' edge sets.

    Returns (x0, y0, x1, y1) of the overlap or None.
    """
    edges_a = _edges_from_vertices(verts_a)
    edges_b = _edges_from_vertices(verts_b)

    best = None
    best_len = 0

    for ax0, ay0, ax1, ay1 in edges_a:
        adx, ady = ax1 - ax0, ay1 - ay0
        alen = (adx ** 2 + ady ** 2) ** 0.5
        if alen < 1e-6:
            continue
        for bx0, by0, bx1, by1 in edges_b:
            bdx, bdy = bx1 - bx0, by1 - by0
            blen = (bdx ** 2 + bdy ** 2) ** 0.5
            if blen < 1e-6:
                continue
            dot = (adx * bdx + ady * bdy) / (alen * blen)
            if abs(abs(dot) - 1.0) > 0.01:
                continue
            if abs(adx) > abs(ady):
                if abs(ay0 - by0) > tolerance:
                    continue
                overlap_min = max(min(ax0, ax1), min(bx0, bx1))
                overlap_max = min(max(ax0, ax1), max(bx0, bx1))
                if overlap_max - overlap_min > 0.01:
                    seg_len = overlap_max - overlap_min
                    if seg_len > best_len:
                        best_len = seg_len
                        best = (overlap_min, ay0, overlap_max, ay0)
            else:
                if abs(ax0 - bx0) > tolerance:
                    continue
                overlap_min = max(min(ay0, ay1), min(by0, by1))
                overlap_max = min(max(ay0, ay1), max(by0, by1))
                if overlap_max - overlap_min > 0.01:
                    seg_len = overlap_max - overlap_min
                    if seg_len > best_len:
                        best_len = seg_len
                        best = (ax0, overlap_min, ax0, overlap_max)
    return best


def _place_door_polygon(wall, door_width=0.8, door_thickness=0.1):
    """Create a thin rectangle centered on a shared wall segment."""
    x0, y0, x1, y1 = wall
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0
    wlen = (dx ** 2 + dy ** 2) ** 0.5
    if wlen < 1e-6:
        return None
    actual_w = min(door_width, wlen * 0.8)
    hw = actual_w / 2
    ux, uy = dx / wlen, dy / wlen
    px, py = -uy * door_thickness / 2, ux * door_thickness / 2
    return [
        (cx - ux * hw + px, cy - uy * hw + py),
        (cx - ux * hw - px, cy - uy * hw - py),
        (cx + ux * hw - px, cy + uy * hw - py),
        (cx + ux * hw + px, cy + uy * hw + py),
    ]


def _find_front_door_edge(room_verts, circ_segs, tolerance=0.1):
    """Find the room boundary edge that overlaps a circulation segment.

    Returns (x0, y0, x1, y1) of the best overlapping room edge,
    or the longest boundary edge as fallback.
    """
    room_edges = _edges_from_vertices(room_verts)

    best_edge = None
    best_overlap = 0

    for rx0, ry0, rx1, ry1 in room_edges:
        rdx, rdy = rx1 - rx0, ry1 - ry0
        rlen = (rdx ** 2 + rdy ** 2) ** 0.5
        if rlen < 1e-6:
            continue
        for (cx0, cy0), (cx1, cy1) in circ_segs:
            cdx, cdy = cx1 - cx0, cy1 - cy0
            clen = (cdx ** 2 + cdy ** 2) ** 0.5
            if clen < 1e-6:
                continue
            dot = (rdx * cdx + rdy * cdy) / (rlen * clen)
            if abs(abs(dot) - 1.0) > 0.01:
                continue
            if abs(rdx) > abs(rdy):
                if abs(ry0 - cy0) > tolerance:
                    continue
                o_min = max(min(rx0, rx1), min(cx0, cx1))
                o_max = min(max(rx0, rx1), max(cx0, cx1))
                overlap = o_max - o_min
            else:
                if abs(rx0 - cx0) > tolerance:
                    continue
                o_min = max(min(ry0, ry1), min(cy0, cy1))
                o_max = min(max(ry0, ry1), max(cy0, cy1))
                overlap = o_max - o_min
            if overlap > best_overlap:
                best_overlap = overlap
                best_edge = (rx0, ry0, rx1, ry1)

    if best_edge is not None:
        return best_edge

    longest_edge = None
    longest_len = 0
    for rx0, ry0, rx1, ry1 in room_edges:
        elen = ((rx1 - rx0) ** 2 + (ry1 - ry0) ** 2) ** 0.5
        if elen > longest_len:
            longest_len = elen
            longest_edge = (rx0, ry0, rx1, ry1)
    return longest_edge


# ── Normalization ──────────────────────────────────────────────────────

def _normalize_to_256(all_vertices_list: list[list[tuple[float, float]]]):
    """Zero-center + scale all polygons so they fit tightly in [0, 256].

    Returns (normalized_vertices_list, scale, offset_x, offset_y).
    """
    xs, ys = [], []
    for verts in all_vertices_list:
        for x, y in verts:
            xs.append(x)
            ys.append(y)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    w = max_x - min_x
    h = max_y - min_y
    s = IMG_SIZE / max(w, h) if max(w, h) > 0 else 1.0
    half = IMG_SIZE / 2.0

    normed = []
    for verts in all_vertices_list:
        normed.append(
            [(round((x - cx) * s + half, 1), round((y - cy) * s + half, 1))
             for x, y in verts]
        )
    return normed, s, cx, cy


# ── Core conversion ───────────────────────────────────────────────────

def convert_entry(entry: dict) -> dict | None:
    """Convert a single hypergraph entry to HouseGAN++ JSON dict.

    Returns None on failure.
    """
    split_str = entry.get("split", "{}")
    hg_dict = json.loads(split_str) if isinstance(split_str, str) else split_str
    bounds_str = entry.get("bounds", "")
    bounds = json.loads(bounds_str) if isinstance(bounds_str, str) and bounds_str else bounds_str
    if not bounds or not isinstance(bounds, dict):
        return None

    hg = Hypergraph(hg_dict)
    leaves = hg.leaves
    n_rooms = len(leaves)
    if n_rooms == 0:
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
        if len(verts) >= 3:
            poly = Polygon(verts)
            if not poly.is_empty and poly.area > 1e-6:
                if room_verts[li]:
                    from shapely.ops import unary_union
                    existing = Polygon(room_verts[li])
                    merged = unary_union([existing, poly])
                    if merged.geom_type == "Polygon":
                        room_verts[li] = list(merged.exterior.coords[:-1])
                    else:
                        room_verts[li] = list(merged.convex_hull.exterior.coords[:-1])
                else:
                    room_verts[li] = verts

    empty = [i for i in range(n_rooms) if not room_verts[i]]
    if empty:
        return None

    room_types_rplan = []
    for leaf in leaves:
        mid = leaf.get("mergeid", "extra")
        room_types_rplan.append(MERGEID_TO_RPLAN.get(mid, 10))

    ag = hg.access_graph
    ag_edges = ag.get("edges", [])
    connections = set()
    for e in ag_edges:
        a, b = int(e[0]) - 1, int(e[1]) - 1
        connections.add((min(a, b), max(a, b)))

    door_verts_list = []
    door_connections = []
    for ri, rj in sorted(connections):
        wall = _find_shared_wall(room_verts[ri], room_verts[rj])
        if wall is None:
            continue
        dpoly = _place_door_polygon(wall)
        if dpoly is not None:
            door_verts_list.append(dpoly)
            door_connections.append((ri, rj))

    # ── Front door (type 15) for outside-connected room ──────────────
    front_door_verts = None
    front_door_room_idx = None

    for li, leaf in enumerate(leaves):
        if "outside" in (leaf.get("connected") or []):
            front_door_room_idx = li
            break

    if front_door_room_idx is not None:
        circ_raw = entry.get("circulation", "[]")
        circ_segs_parsed = json.loads(circ_raw) if isinstance(circ_raw, str) else circ_raw
        circ_tuples = [
            ((seg["start"]["X"], seg["start"]["Y"]),
             (seg["end"]["X"], seg["end"]["Y"]))
            for seg in circ_segs_parsed
        ]
        fd_edge = _find_front_door_edge(
            room_verts[front_door_room_idx], circ_tuples
        )
        if fd_edge is not None:
            front_door_verts = _place_door_polygon(fd_edge)

    all_verts = room_verts + door_verts_list
    if front_door_verts is not None:
        all_verts.append(front_door_verts)
    normed, s, cx, cy = _normalize_to_256(all_verts)

    n_doors = len(door_verts_list)
    has_front_door = front_door_verts is not None
    total_nodes = n_rooms + n_doors + (1 if has_front_door else 0)

    all_types = room_types_rplan + [DOOR_TYPE] * n_doors
    if has_front_door:
        all_types.append(FRONT_DOOR_TYPE)

    boxes = []
    for verts in normed:
        boxes.append(_bbox(verts))

    edges_list = []
    ed_rm = []

    for node_idx in range(total_nodes):
        verts = normed[node_idx]
        rtype = all_types[node_idx]
        segs = _edges_from_vertices(verts)
        for seg in segs:
            x0, y0, x1, y1 = seg
            edges_list.append([x0, y0, x1, y1, rtype, 0])
            ed_rm.append([node_idx])

    for door_idx, (ri, rj) in enumerate(door_connections):
        door_node_idx = n_rooms + door_idx
        door_segs_start = sum(
            len(_edges_from_vertices(normed[ni])) for ni in range(door_node_idx)
        )
        n_door_segs = len(_edges_from_vertices(normed[door_node_idx]))
        for si in range(door_segs_start, door_segs_start + n_door_segs):
            if si < len(ed_rm):
                ed_rm[si] = [door_node_idx]

    for ri, rj in sorted(connections):
        ri_segs = _edges_from_vertices(normed[ri])
        seg_offset_ri = sum(
            len(_edges_from_vertices(normed[ni])) for ni in range(ri)
        )
        rj_segs = _edges_from_vertices(normed[rj])
        seg_offset_rj = sum(
            len(_edges_from_vertices(normed[ni])) for ni in range(rj)
        )
        for si, (ax0, ay0, ax1, ay1) in enumerate(ri_segs):
            adx, ady = ax1 - ax0, ay1 - ay0
            alen = (adx ** 2 + ady ** 2) ** 0.5
            if alen < 1e-6:
                continue
            for sj, (bx0, by0, bx1, by1) in enumerate(rj_segs):
                bdx, bdy = bx1 - bx0, by1 - by0
                blen = (bdx ** 2 + bdy ** 2) ** 0.5
                if blen < 1e-6:
                    continue
                dot = (adx * bdx + ady * bdy) / (alen * blen)
                if abs(abs(dot) - 1.0) > 0.02:
                    continue
                if abs(adx) > abs(ady):
                    if abs(ay0 - by0) > 1.0:
                        continue
                    o_min = max(min(ax0, ax1), min(bx0, bx1))
                    o_max = min(max(ax0, ax1), max(bx0, bx1))
                    if o_max - o_min > 0.5:
                        idx_i = seg_offset_ri + si
                        idx_j = seg_offset_rj + sj
                        if idx_i < len(ed_rm) and len(ed_rm[idx_i]) == 1:
                            ed_rm[idx_i] = [ri, rj]
                        if idx_j < len(ed_rm) and len(ed_rm[idx_j]) == 1:
                            ed_rm[idx_j] = [rj, ri]
                else:
                    if abs(ax0 - bx0) > 1.0:
                        continue
                    o_min = max(min(ay0, ay1), min(by0, by1))
                    o_max = min(max(ay0, ay1), max(by0, by1))
                    if o_max - o_min > 0.5:
                        idx_i = seg_offset_ri + si
                        idx_j = seg_offset_rj + sj
                        if idx_i < len(ed_rm) and len(ed_rm[idx_i]) == 1:
                            ed_rm[idx_i] = [ri, rj]
                        if idx_j < len(ed_rm) and len(ed_rm[idx_j]) == 1:
                            ed_rm[idx_j] = [rj, ri]

    for door_idx, (ri, rj) in enumerate(door_connections):
        door_node_idx = n_rooms + door_idx
        door_segs_start = sum(
            len(_edges_from_vertices(normed[ni])) for ni in range(door_node_idx)
        )
        n_door_segs = len(_edges_from_vertices(normed[door_node_idx]))
        targets = [ri, rj]
        for k, si in enumerate(range(door_segs_start, door_segs_start + n_door_segs)):
            if si < len(ed_rm):
                ed_rm[si] = [door_node_idx, targets[k % len(targets)]]

    if has_front_door:
        fd_node_idx = total_nodes - 1
        fd_segs_start = sum(
            len(_edges_from_vertices(normed[ni])) for ni in range(fd_node_idx)
        )
        n_fd_segs = len(_edges_from_vertices(normed[fd_node_idx]))
        for si in range(fd_segs_start, fd_segs_start + n_fd_segs):
            if si < len(ed_rm):
                ed_rm[si] = [fd_node_idx, front_door_room_idx]

    return {
        "room_type": all_types,
        "boxes": boxes,
        "edges": edges_list,
        "ed_rm": ed_rm,
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Convert hypergraph JSON to HouseGAN++ individual JSON files.",
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True,
        help="Path to input JSON (e.g. dataset/wmr24/wmr24.json)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output directory for per-sample JSON files "
             "(default: alongside input, named <stem>_housegan_json/)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output is None:
        output_dir = input_path.parent / f"{input_path.stem}_housegan_json"
    else:
        output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path) as f:
        data = json.load(f)

    logger.info("Loaded %d entries from %s", len(data), input_path)
    logger.info("Output directory: %s", output_dir)

    success = 0
    failed = 0
    failed_ids = []

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

        out_path = output_dir / f"{eid}.json"
        with open(out_path, "w") as f:
            json.dump(result, f)
        success += 1

        if (i + 1) % 100 == 0:
            logger.info("  processed %d/%d ...", i + 1, len(data))

    list_path = output_dir / "list.txt"
    with open(list_path, "w") as f:
        for entry in data:
            eid = entry.get("id", "")
            if eid not in failed_ids:
                f.write(f"{eid}.json\n")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Converted: %d / %d", success, len(data))
    logger.info("Failed:    %d", failed)
    if failed_ids:
        logger.info("Failed IDs: %s", failed_ids[:20])
    logger.info("Output:    %s", output_dir)
    logger.info("List file: %s", list_path)
    logger.info("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
