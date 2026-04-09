#!/usr/bin/env python3
"""
Pure-Python BSP conversion for RPLAN floor plans.

Reads RPLAN JSON samples (same as rplan_to_hypergraph.py) and builds a BSP
tree using shapely geometry instead of the C# RGL library.

Split-line priorities:
  1. Find axis-aligned lines that cleanly separate rooms without cutting any.
  2. If no clean line exists, cut through LivingRoom along a line whose
     endpoints lie on the outer boundary (not on interior-only edges).
     Among such lines prefer the most balanced area division.
  3. Last resort: cut a non-LivingRoom room along a boundary-touching line.
     Discard any sub-fragment whose area < 5 % of the original room.
  4. All children of a split LivingRoom are connected in the access graph
     according to their geometric adjacency.

Modes:
  Debug (default) — renders PNG visualisations:
    python -m rplan.rplan_to_hypergraph_native --sample 3951
    python -m rplan.rplan_to_hypergraph_native --sample 3951 57395
    python -m rplan.rplan_to_hypergraph_native --max-samples 100

  Export — writes hypergraph JSON files (same format as dataset/hypergraph):
    python -m rplan.rplan_to_hypergraph_native --export
    python -m rplan.rplan_to_hypergraph_native --sample 3951 --export
"""

import argparse
import json
import logging
import math
import random
from pathlib import Path
from statistics import mean as _mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import (
    LineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.ops import split as shapely_split, unary_union
from tqdm import tqdm

logger = logging.getLogger(__name__)

RPLAN_ROOM_NAMES = {
    0: "LivingRoom",
    1: "Bedroom",
    2: "Kitchen",
    3: "Bathroom",
    4: "Balcony",
    5: "Storage",
}

CATEGORY_TO_MERGEID = {
    0: "living",
    1: "bed",
    2: "kitchen",
    3: "bath",
    4: "balcony",
    5: "storage",
}

CATEGORY_COLORS = {
    0: "#FFEEAD",
    1: "#96CEB4",
    2: "#FF6F69",
    3: "#88D8B0",
    4: "#C5E1A5",
    5: "#B0BEC5",
}

SMALL_FRAGMENT_THRESHOLD = 0.05
LOOKAHEAD_WEIGHT = 0.5
LOOKAHEAD_NO_SPLIT_PENALTY = -5000
EVO_MAX_NODES = 30


def get_room_name(category_index: int) -> str:
    return RPLAN_ROOM_NAMES.get(category_index, f"Unknown_{category_index}")


def draw_rplan_on_axis(ax, sample: dict, title: str = ""):
    """Draw one RPLAN sample on the given *ax*.

    Draws room polygons coloured by category, the outer boundary, adjacency
    edges, and centroid labels.
    """
    rooms = sample["rooms"]
    adjacencies = sample.get("access_adjacencies", sample["adjacencies"])

    patches = []
    face_colors = []
    centroids = {}

    for room in rooms:
        coords = np.array(room["r_boundary"], dtype=float)
        poly = MplPolygon(coords, closed=True)
        patches.append(poly)
        face_colors.append(CATEGORY_COLORS.get(room["category"], "#DDDDDD"))

        shapely_poly = Polygon(coords)
        cx, cy = shapely_poly.centroid.x, shapely_poly.centroid.y
        centroids[room["id"]] = (cx, cy)

    pc = PatchCollection(patches, match_original=False, edgecolors="black",
                         linewidths=1.2, zorder=2)
    pc.set_facecolor(face_colors)
    ax.add_collection(pc)

    for u, v in adjacencies:
        if u in centroids and v in centroids:
            xu, yu = centroids[u]
            xv, yv = centroids[v]
            ax.plot([xu, xv], [yu, yv], color="#555555", linewidth=0.8,
                    linestyle="--", zorder=3)

    for room in rooms:
        rid = room["id"]
        if rid in centroids:
            cx, cy = centroids[rid]
            label = f"{get_room_name(room['category'])}\n#{rid}"
            ax.text(cx, cy, label, ha="center", va="center", fontsize=6,
                    fontweight="bold", zorder=4)

    boundary_pts = np.array(
        [[pt[0], pt[1]] for pt in sample["boundary"]], dtype=float
    )
    closed = np.vstack([boundary_pts, boundary_pts[0]])
    ax.plot(closed[:, 0], closed[:, 1], color="black", linewidth=2.0, zorder=5)

    categories_present = sorted({r["category"] for r in rooms})
    legend_patches = [
        mpatches.Patch(
            color=CATEGORY_COLORS.get(cat, "#DDDDDD"),
            label=get_room_name(cat),
        )
        for cat in categories_present
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=8,
              framealpha=0.8)

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.invert_yaxis()
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if title:
        ax.set_title(title, fontsize=10)


# ---------------------------------------------------------------------------
# Data loading helpers (mirrors rplan_to_hypergraph.RPLANGraph)
# ---------------------------------------------------------------------------

def sample_to_rooms(sample: dict) -> dict:
    """Parse rooms into a dict: room_id -> {polygon, room_type, category, name}."""
    rooms = {}
    for r in sample["rooms"]:
        coords = r["r_boundary"]
        if len(coords) < 3:
            continue
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < 1e-6:
            continue
        cat = r["category"]
        rooms[r["id"]] = {
            "polygon": poly,
            "room_type": cat,
            "name": get_room_name(cat),
            "original_id": r["id"],
        }
    return rooms


def sample_to_access_edges(sample: dict) -> list:
    adj_key = "access_adjacencies" if "access_adjacencies" in sample else "adjacencies"
    return [tuple(pair) for pair in sample[adj_key]]


def has_point_only_edges(sample: dict) -> bool:
    """True if the sample has any access edge where the rooms share only a
    point (not a proper edge).  These are dataset artifacts."""
    rooms = sample_to_rooms(sample)
    access = sample_to_access_edges(sample)
    for u, v in access:
        if u not in rooms or v not in rooms:
            continue
        if not rooms_share_edge(rooms[u]["polygon"], rooms[v]["polygon"]):
            return True
    return False


def compute_boundary(rooms: dict) -> Polygon:
    polys = [r["polygon"] for r in rooms.values()]
    if not polys:
        return None
    union = unary_union(polys)
    if union.is_empty:
        return None
    if union.geom_type == "MultiPolygon":
        union = max(union.geoms, key=lambda g: g.area)
    return union


def find_entrance_room(sample: dict, leaves: dict) -> int | None:
    """Return the leaf room_id whose polygon best overlaps the entrance.

    *leaves* maps ``room_id -> {"polygon": Polygon, ...}`` for every BSP leaf.
    The entrance rectangle is taken from ``sample["entrance_expand"]``.
    """
    entrance = sample.get("entrance_expand")
    if not entrance or len(entrance) < 3:
        return None
    entrance_poly = Polygon(entrance)
    if not entrance_poly.is_valid or entrance_poly.is_empty:
        return None

    best_id = None
    best_overlap = 0.0
    for rid, rinfo in leaves.items():
        poly = rinfo.get("polygon")
        if poly is None:
            continue
        overlap = poly.intersection(entrance_poly).area
        if overlap > best_overlap:
            best_overlap = overlap
            best_id = rid

    if best_id is not None:
        return best_id

    # Fallback: nearest leaf centroid to entrance centroid
    ec = entrance_poly.centroid
    best_dist = float("inf")
    for rid, rinfo in leaves.items():
        poly = rinfo.get("polygon")
        if poly is None:
            continue
        d = poly.centroid.distance(ec)
        if d < best_dist:
            best_dist = d
            best_id = rid
    return best_id


OUTSIDE_ROOM_ID = -1


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def polygon_touches_boundary(room_poly: Polygon, boundary: Polygon, tol=0.5) -> bool:
    """True if at least one edge of *room_poly* lies on the outer boundary."""
    coords = list(room_poly.exterior.coords)
    for i in range(len(coords) - 1):
        mid = Point((coords[i][0] + coords[i + 1][0]) / 2,
                     (coords[i][1] + coords[i + 1][1]) / 2)
        if boundary.exterior.distance(mid) < tol:
            return True
    return False


def line_endpoints_on_boundary(line: LineString, boundary: Polygon, tol=1.0) -> tuple:
    """Return (start_on_boundary, end_on_boundary)."""
    p0, p1 = Point(line.coords[0]), Point(line.coords[-1])
    return (boundary.exterior.distance(p0) < tol,
            boundary.exterior.distance(p1) < tol)


def rooms_share_edge(poly_a: Polygon, poly_b: Polygon, tol=0.5) -> bool:
    """True if two room polygons share an edge of non-trivial length."""
    shared = poly_a.intersection(poly_b)
    if shared.is_empty:
        return False
    return shared.length > tol


def collect_unique_coords(rooms: dict):
    """Collect all unique X and Y coordinates from room vertices."""
    xs, ys = set(), set()
    for r in rooms.values():
        for x, y in r["polygon"].exterior.coords:
            xs.add(round(x, 2))
            ys.add(round(y, 2))
    return sorted(xs), sorted(ys)


# ---------------------------------------------------------------------------
# Split candidate evaluation
# ---------------------------------------------------------------------------

def classify_rooms_by_line(rooms, direction, position, tol=0.3):
    """Classify rooms as left/right (or top/bottom) of a split line.

    direction: 'h' for horizontal (split at y=position) or 'v' for vertical
    Returns (side_a, side_b, straddling) as dicts of room_id -> room_info.
    side_a: rooms with centroids on the low side (left / top in image coords).
    """
    side_a, side_b, straddling = {}, {}, {}
    for rid, rinfo in rooms.items():
        poly = rinfo["polygon"]
        minx, miny, maxx, maxy = poly.bounds
        if direction == "h":
            if maxy <= position + tol:
                side_a[rid] = rinfo
            elif miny >= position - tol:
                side_b[rid] = rinfo
            else:
                straddling[rid] = rinfo
        else:
            if maxx <= position + tol:
                side_a[rid] = rinfo
            elif minx >= position - tol:
                side_b[rid] = rinfo
            else:
                straddling[rid] = rinfo
    return side_a, side_b, straddling


def build_split_line(direction, position, boundary: Polygon, extend=200):
    """Build a LineString for the split, clipped to the boundary."""
    minx, miny, maxx, maxy = boundary.bounds
    if direction == "h":
        raw = LineString([(minx - extend, position), (maxx + extend, position)])
    else:
        raw = LineString([(position, miny - extend), (position, maxy + extend)])
    clipped = raw.intersection(boundary)
    if clipped.is_empty:
        return None
    if clipped.geom_type == "MultiLineString":
        clipped = max(clipped.geoms, key=lambda g: g.length)
    if clipped.geom_type != "LineString":
        return None
    return clipped


def split_polygon_by_line(poly: Polygon, direction, position):
    """Split a polygon by a horizontal or vertical line.

    Returns ``(low_parts, high_parts)`` where each element is a **list** of
    Polygon objects (may be empty).  A non-convex polygon can produce
    multiple disconnected pieces on one side of the cut.
    """
    minx, miny, maxx, maxy = poly.bounds
    big = max(maxx - minx, maxy - miny) + 100

    if direction == "h":
        low_box = box(minx - big, miny - big, maxx + big, position)
        high_box = box(minx - big, position, maxx + big, maxy + big)
    else:
        low_box = box(minx - big, miny - big, position, maxy + big)
        high_box = box(position, miny - big, maxx + big, maxy + big)

    low_part = poly.intersection(low_box)
    high_part = poly.intersection(high_box)

    def _collect(geom):
        if geom.is_empty:
            return []
        if geom.geom_type == "MultiPolygon":
            return [g for g in geom.geoms if g.area > 1e-6]
        if geom.geom_type == "GeometryCollection":
            return [g for g in geom.geoms
                    if g.geom_type == "Polygon" and g.area > 1e-6]
        if geom.geom_type == "Polygon" and geom.area > 1e-6:
            return [geom]
        return []

    return _collect(low_part), _collect(high_part)


# ---------------------------------------------------------------------------
# Evaluate a candidate split
# ---------------------------------------------------------------------------

class SplitCandidate:
    """Encapsulates all information about a potential BSP split."""

    def __init__(self, direction, position, boundary):
        self.direction = direction
        self.position = position
        self.boundary = boundary
        self.side_a = {}        # rooms fully on low side
        self.side_b = {}        # rooms fully on high side
        self.straddling = {}    # rooms cut by the line
        self.cut_rooms = {}     # room_id -> (low_piece, high_piece)
        self.line = None
        self.score = -1e9
        self.valid = False
        self.cuts_living = False
        self.cuts_nonliving = False
        self.both_endpoints_on_boundary = False

    def evaluate(self, rooms, access_edges):
        self.side_a, self.side_b, self.straddling = classify_rooms_by_line(
            rooms, self.direction, self.position
        )

        self.line = build_split_line(self.direction, self.position, self.boundary)
        if self.line is None:
            return

        start_on, end_on = line_endpoints_on_boundary(self.line, self.boundary)
        self.both_endpoints_on_boundary = start_on and end_on

        for rid, rinfo in self.straddling.items():
            low_list, high_list = split_polygon_by_line(
                rinfo["polygon"], self.direction, self.position
            )
            self.cut_rooms[rid] = (low_list, high_list)

        for rid in self.straddling:
            if self.straddling[rid]["name"] == "LivingRoom":
                self.cuts_living = True
            else:
                self.cuts_nonliving = True

        if not self.side_a and not any(
            len(self.cut_rooms[rid][0]) > 0 for rid in self.cut_rooms
        ):
            return
        if not self.side_b and not any(
            len(self.cut_rooms[rid][1]) > 0 for rid in self.cut_rooms
        ):
            return

        self.valid = True
        self._compute_score(rooms)

    def _compute_score(self, rooms):
        total_area = sum(r["polygon"].area for r in rooms.values())
        if total_area < 1e-6:
            self.score = -1e9
            return

        area_a = sum(r["polygon"].area for r in self.side_a.values())
        area_b = sum(r["polygon"].area for r in self.side_b.values())
        for rid, (low_list, high_list) in self.cut_rooms.items():
            area_a += sum(p.area for p in low_list)
            area_b += sum(p.area for p in high_list)

        global_balance = 1.0 - abs(area_a - area_b) / total_area

        n_cut = len(self.straddling)
        n_rooms_a = len(self.side_a) + sum(
            1 for rid, (low_list, _) in self.cut_rooms.items() if low_list
        )
        n_rooms_b = len(self.side_b) + sum(
            1 for rid, (_, high_list) in self.cut_rooms.items() if high_list
        )

        if n_rooms_a == 0 or n_rooms_b == 0:
            self.valid = False
            self.score = -1e9
            return

        # Tier system — lower tier is strictly better.
        #   0: clean split (no rooms cut)
        #   1: only LivingRoom(s) cut, both endpoints on boundary
        #   2: only LivingRoom(s) cut, endpoints NOT both on boundary
        #   3: non-LivingRoom cut, both endpoints on boundary
        #   4: non-LivingRoom cut, endpoints NOT both on boundary
        if n_cut == 0:
            tier = 0
        elif not self.cuts_nonliving and self.both_endpoints_on_boundary:
            tier = 1
        elif not self.cuts_nonliving:
            tier = 2
        elif self.both_endpoints_on_boundary:
            tier = 3
        else:
            tier = 4

        # Fragment balance: how evenly each cut room is divided (0..1).
        fragment_balance = 0.0
        n_balanced = 0
        for rid, (low_list, high_list) in self.cut_rooms.items():
            low_area = sum(p.area for p in low_list)
            high_area = sum(p.area for p in high_list)
            if low_area > 1e-6 and high_area > 1e-6:
                fragment_balance += min(low_area, high_area) / max(low_area, high_area)
                n_balanced += 1
        if n_balanced > 0:
            fragment_balance /= n_balanced

        self.score = (
            -tier * 1000
            - n_cut * 50
            + fragment_balance * 30
            + global_balance * 5
        )


def find_best_split(rooms, boundary, access_edges, lookahead=0):
    """Find the best BSP split for the given set of rooms.

    When *lookahead* > 0 each valid candidate is additionally scored by
    simulating the best greedy split on each resulting child partition.
    This lets the algorithm prefer splits that lead to cleaner future
    partitions (fewer room cuts downstream).
    """
    xs, ys = collect_unique_coords(rooms)

    candidates = []
    for y in ys:
        c = SplitCandidate("h", y, boundary)
        c.evaluate(rooms, access_edges)
        if c.valid:
            candidates.append(c)
    for x in xs:
        c = SplitCandidate("v", x, boundary)
        c.evaluate(rooms, access_edges)
        if c.valid:
            candidates.append(c)

    if not candidates:
        return None

    if lookahead > 0:
        for c in candidates:
            rooms_a, rooms_b = _build_lookahead_rooms(c)
            child_score = 0.0
            for child_rooms in (rooms_a, rooms_b):
                if len(child_rooms) <= 1:
                    continue
                child_best = find_best_split(
                    child_rooms, boundary, [], lookahead=lookahead - 1,
                )
                if child_best is not None:
                    child_score += child_best.score
                else:
                    child_score += LOOKAHEAD_NO_SPLIT_PENALTY
            c.score += LOOKAHEAD_WEIGHT * child_score

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[0]


def _build_lookahead_rooms(candidate):
    """Build lightweight child-room dicts from an already-evaluated candidate.

    Returns ``(rooms_a, rooms_b)`` using temporary negative IDs for
    straddling-room fragments so they don't collide with real room IDs.
    """
    rooms_a = dict(candidate.side_a)
    rooms_b = dict(candidate.side_b)
    tmp_id = -1
    for orig_id, (low_list, high_list) in candidate.cut_rooms.items():
        rinfo = candidate.straddling[orig_id]
        for poly in low_list:
            if poly.area > 1e-6:
                rooms_a[tmp_id] = {
                    "polygon": poly,
                    "room_type": rinfo["room_type"],
                    "name": rinfo["name"],
                }
                tmp_id -= 1
        for poly in high_list:
            if poly.area > 1e-6:
                rooms_b[tmp_id] = {
                    "polygon": poly,
                    "room_type": rinfo["room_type"],
                    "name": rinfo["name"],
                }
                tmp_id -= 1
    return rooms_a, rooms_b


# ---------------------------------------------------------------------------
# Execute a split: produce two room groups + updated access edges
# ---------------------------------------------------------------------------

def execute_split(split: SplitCandidate, rooms, access_edges, boundary,
                   next_id_ref=None):
    """Apply the chosen split, returning rooms for each side and updated access.

    *next_id_ref* is a single-element list ``[int]`` used as a shared counter
    across recursive calls so that fragment IDs are globally unique.

    Returns (rooms_a, rooms_b, new_access_edges, split_info).
    """
    rooms_a = dict(split.side_a)
    rooms_b = dict(split.side_b)

    if next_id_ref is None:
        next_id_ref = [max(rooms.keys()) + 1]
    id_map = {}   # original_id -> list of (new_id, side)
    split_details = []

    for orig_id, rinfo in split.straddling.items():
        low_list, high_list = split.cut_rooms[orig_id]
        orig_area = rinfo["polygon"].area
        fragments = []

        for poly in low_list:
            if poly.area > 1e-6:
                fragments.append(("a", poly))
        for poly in high_list:
            if poly.area > 1e-6:
                fragments.append(("b", poly))

        # Discard tiny fragments of non-LivingRoom rooms
        if rinfo["name"] != "LivingRoom" and len(fragments) >= 2:
            kept = []
            for side, poly in fragments:
                if poly.area < SMALL_FRAGMENT_THRESHOLD * orig_area:
                    logger.debug(
                        "Discarding tiny fragment of %s_%d (%.1f%% of original)",
                        rinfo["name"], orig_id, poly.area / orig_area * 100,
                    )
                else:
                    kept.append((side, poly))
            fragments = kept

        id_map[orig_id] = []
        for side, poly in fragments:
            new_id = next_id_ref[0]
            next_id_ref[0] += 1
            true_original = rinfo.get("original_id", orig_id)
            new_room = {
                "polygon": poly,
                "room_type": rinfo["room_type"],
                "name": rinfo["name"],
                "original_id": true_original,
                "split_from": orig_id,
            }
            id_map[orig_id].append((new_id, side))
            if side == "a":
                rooms_a[new_id] = new_room
            else:
                rooms_b[new_id] = new_room

        split_details.append({
            "original_id": orig_id,
            "room_type": rinfo["name"],
            "fragments": [(nid, s) for nid, s in id_map[orig_id]],
        })

    all_split_rooms = {**rooms_a, **rooms_b}

    def _get_poly(rid):
        info = all_split_rooms.get(rid)
        return info["polygon"] if info else None

    new_access = []

    # 1. Propagate each original access edge to exactly ONE pair of
    #    fragments — the pair with the longest shared boundary.
    for u, v in access_edges:
        u_targets = id_map.get(u, [(u, None)])
        v_targets = id_map.get(v, [(v, None)])

        best_pair = None
        best_length = 0.0

        for uid, _ in u_targets:
            for vid, _ in v_targets:
                if uid == vid:
                    continue
                u_poly = _get_poly(uid)
                v_poly = _get_poly(vid)
                if u_poly is None or v_poly is None:
                    continue
                shared = u_poly.intersection(v_poly)
                if shared.is_empty:
                    continue
                slen = shared.length
                if slen > 0.5 and slen > best_length:
                    best_length = slen
                    best_pair = (uid, vid)

        if best_pair is not None:
            new_access.append(best_pair)

    # 2. Connect sibling fragments of the same split room that are
    #    geometrically adjacent.
    for orig_id, new_ids in id_map.items():
        if len(new_ids) < 2:
            continue
        for i in range(len(new_ids)):
            for j in range(i + 1, len(new_ids)):
                nid_i, _ = new_ids[i]
                nid_j, _ = new_ids[j]
                poly_i = _get_poly(nid_i)
                poly_j = _get_poly(nid_j)
                if poly_i is not None and poly_j is not None:
                    if rooms_share_edge(poly_i, poly_j):
                        new_access.append((nid_i, nid_j))

    # Deduplicate
    edge_set = set()
    for u, v in new_access:
        edge_set.add((min(u, v), max(u, v)))
    new_access = list(edge_set)

    split_info = {
        "direction": split.direction,
        "position": split.position,
        "angle": 0.0 if split.direction == "h" else math.pi / 2,
        "details": split_details,
    }

    return rooms_a, rooms_b, new_access, split_info


# ---------------------------------------------------------------------------
# Recursive BSP construction
# ---------------------------------------------------------------------------

def build_bsp_tree(rooms, boundary, access_edges, depth=0, name="root",
                   all_rooms_ref=None, next_id_ref=None, lookahead=1):
    """Recursively build a BSP tree from room polygons.

    *all_rooms_ref* is a mutable dict that accumulates every room (including
    split fragments) created during the recursion so that the caller can
    look up polygons for any room_id that appears in the final tree.

    *next_id_ref* is a single-element list ``[int]`` used as a shared counter
    to guarantee globally unique fragment IDs across all recursive calls.

    *lookahead* controls how many levels ahead ``find_best_split`` simulates
    when scoring candidate splits (0 = greedy, 1 = one step ahead, etc.).

    Returns a dict representing the BSP tree node.
    """
    if all_rooms_ref is None:
        all_rooms_ref = {}
    if next_id_ref is None:
        next_id_ref = [max(rooms.keys()) + 1 if rooms else 0]
    for rid, rinfo in rooms.items():
        all_rooms_ref[rid] = rinfo

    total_area = sum(r["polygon"].area for r in rooms.values())

    if len(rooms) == 0:
        return None

    if len(rooms) == 1:
        rid = next(iter(rooms))
        rinfo = rooms[rid]
        return {
            "name": rinfo["name"],
            "area": rinfo["polygon"].area,
            "room_id": rid,
            "original_id": rinfo.get("original_id", rid),
            "polygon": rinfo["polygon"],
            "room_type": rinfo["room_type"],
            "final": True,
        }

    split = find_best_split(rooms, boundary, access_edges, lookahead=lookahead)

    if split is None:
        logger.warning(
            "depth=%d: No valid split found for %d rooms, returning flat",
            depth, len(rooms),
        )
        children = []
        for rid, rinfo in rooms.items():
            children.append({
                "name": rinfo["name"],
                "area": rinfo["polygon"].area,
                "room_id": rid,
                "original_id": rinfo.get("original_id", rid),
                "polygon": rinfo["polygon"],
                "room_type": rinfo["room_type"],
                "final": True,
            })
        return {
            "name": name,
            "area": total_area,
            "children": children,
            "unsplit": True,
        }

    rooms_a, rooms_b, new_access, split_info = execute_split(
        split, rooms, access_edges, boundary, next_id_ref=next_id_ref
    )

    area_a = sum(r["polygon"].area for r in rooms_a.values())
    area_b = sum(r["polygon"].area for r in rooms_b.values())

    logger.debug(
        "depth=%d split=%s@%.1f rooms=%d->(%d,%d) area_ratio=%.2f cut=%s",
        depth,
        split_info["direction"],
        split_info["position"],
        len(rooms),
        len(rooms_a),
        len(rooms_b),
        area_a / (area_a + area_b) if (area_a + area_b) > 0 else 0,
        [d["room_type"] for d in split_info["details"]] or "none",
    )

    child_a = build_bsp_tree(
        rooms_a, boundary, new_access, depth + 1, name + "L",
        all_rooms_ref, next_id_ref, lookahead=lookahead,
    )
    child_b = build_bsp_tree(
        rooms_b, boundary, new_access, depth + 1, name + "R",
        all_rooms_ref, next_id_ref, lookahead=lookahead,
    )

    children = [c for c in (child_a, child_b) if c is not None]
    if len(children) == 1:
        return children[0]

    return {
        "name": name,
        "angle": split_info["angle"],
        "area": total_area,
        "split_direction": split_info["direction"],
        "split_position": split_info["position"],
        "children": children,
    }


# ---------------------------------------------------------------------------
# Evolutionary BSP search
# ---------------------------------------------------------------------------

def _evaluate_candidates(rooms, boundary):
    """Enumerate, evaluate, and tier-sort all valid split candidates."""
    xs, ys = collect_unique_coords(rooms)
    candidates = []
    for y in ys:
        c = SplitCandidate("h", y, boundary)
        c.evaluate(rooms, [])
        if c.valid:
            candidates.append(c)
    for x in xs:
        c = SplitCandidate("v", x, boundary)
        c.evaluate(rooms, [])
        if c.valid:
            candidates.append(c)
    if not candidates:
        return []
    for c in candidates:
        n_cut = len(c.straddling)
        if n_cut == 0:
            c._tier = 0
        elif not c.cuts_nonliving and c.both_endpoints_on_boundary:
            c._tier = 1
        elif not c.cuts_nonliving:
            c._tier = 2
        elif c.both_endpoints_on_boundary:
            c._tier = 3
        else:
            c._tier = 4
    candidates.sort(key=lambda c: (c._tier, -c.score))
    return candidates


def build_bsp_tree_with_genome(rooms, boundary, access_edges, genome,
                               depth=0, name="root",
                               all_rooms_ref=None, next_id_ref=None,
                               node_counter=None):
    """Build a BSP tree using *genome* to select among same-tier candidates.

    Structurally identical to ``build_bsp_tree`` but the split choice at each
    internal node is driven by ``genome[node_counter[0]]`` instead of always
    picking the highest-scoring candidate.
    """
    if all_rooms_ref is None:
        all_rooms_ref = {}
    if next_id_ref is None:
        next_id_ref = [max(rooms.keys()) + 1 if rooms else 0]
    if node_counter is None:
        node_counter = [0]
    for rid, rinfo in rooms.items():
        all_rooms_ref[rid] = rinfo

    if len(rooms) == 0:
        return None

    if len(rooms) == 1:
        rid = next(iter(rooms))
        rinfo = rooms[rid]
        return {
            "name": rinfo["name"],
            "area": rinfo["polygon"].area,
            "room_id": rid,
            "original_id": rinfo.get("original_id", rid),
            "polygon": rinfo["polygon"],
            "room_type": rinfo["room_type"],
            "final": True,
        }

    candidates = _evaluate_candidates(rooms, boundary)

    if not candidates:
        children = []
        for rid, rinfo in rooms.items():
            children.append({
                "name": rinfo["name"],
                "area": rinfo["polygon"].area,
                "room_id": rid,
                "original_id": rinfo.get("original_id", rid),
                "polygon": rinfo["polygon"],
                "room_type": rinfo["room_type"],
                "final": True,
            })
        return {
            "name": name,
            "area": sum(r["polygon"].area for r in rooms.values()),
            "children": children,
            "unsplit": True,
        }

    best_tier = candidates[0]._tier
    same_tier = [c for c in candidates if c._tier == best_tier]

    gi = node_counter[0] % len(genome)
    node_counter[0] += 1
    idx = int(genome[gi] * len(same_tier)) % len(same_tier)
    split = same_tier[idx]

    rooms_a, rooms_b, new_access, split_info = execute_split(
        split, rooms, access_edges, boundary, next_id_ref=next_id_ref,
    )

    total_area = sum(r["polygon"].area for r in rooms.values())
    n_cut = len(split.straddling)

    child_a = build_bsp_tree_with_genome(
        rooms_a, boundary, new_access, genome, depth + 1, name + "L",
        all_rooms_ref, next_id_ref, node_counter,
    )
    child_b = build_bsp_tree_with_genome(
        rooms_b, boundary, new_access, genome, depth + 1, name + "R",
        all_rooms_ref, next_id_ref, node_counter,
    )

    children = [c for c in (child_a, child_b) if c is not None]
    if len(children) == 1:
        return children[0]

    return {
        "name": name,
        "angle": split_info["angle"],
        "area": total_area,
        "split_direction": split_info["direction"],
        "split_position": split_info["position"],
        "children": children,
        "_evo_tier": best_tier,
        "_evo_n_cut": n_cut,
    }


def evaluate_fitness(tree):
    """Compute a multi-objective fitness score for a complete BSP tree.

    Higher is better.  The tier penalty dominates so that the existing
    priority system is preserved as a hard constraint.
    """
    if tree is None:
        return -1e18

    leaves = []
    internals = []

    def _walk(node):
        if node is None:
            return
        if node.get("final"):
            leaves.append(node)
            return
        internals.append(node)
        for ch in node.get("children", []):
            _walk(ch)

    _walk(tree)

    if not leaves:
        return -1e18

    tier_penalty = sum(n.get("_evo_tier", 0) for n in internals)
    total_cuts = sum(n.get("_evo_n_cut", 0) for n in internals)

    aspect_scores = []
    for leaf in leaves:
        poly = leaf.get("polygon")
        if poly is None:
            continue
        minx, miny, maxx, maxy = poly.bounds
        w, h = maxx - minx, maxy - miny
        if max(w, h) > 1e-6:
            aspect_scores.append(min(w, h) / max(w, h))
    avg_aspect = _mean(aspect_scores) if aspect_scores else 0.0

    def _depth(node):
        if node is None or node.get("final"):
            return 0
        children = node.get("children", [])
        if not children:
            return 0
        return 1 + max(_depth(ch) for ch in children)

    def _leaf_count(node):
        if node is None:
            return 0
        if node.get("final"):
            return 1
        return sum(_leaf_count(ch) for ch in node.get("children", []))

    children = tree.get("children", [])
    if len(children) == 2:
        lc_a = _leaf_count(children[0])
        lc_b = _leaf_count(children[1])
        balance = min(lc_a, lc_b) / max(lc_a, lc_b) if max(lc_a, lc_b) > 0 else 0
    else:
        balance = 0.0

    return (
        -tier_penalty * 10000
        - total_cuts * 200
        + avg_aspect * 50
        + balance * 30
    )


def evolve_bsp_tree(rooms, boundary, access_edges,
                    pop_size=100, n_gens=30, seed=None):
    """Find a high-quality BSP tree via evolutionary search.

    Returns ``(tree, all_rooms_ref)`` in the same format as
    ``build_bsp_tree``.
    """
    rng = random.Random(seed)

    population = [[0.0] * EVO_MAX_NODES]
    for _ in range(pop_size - 1):
        population.append([rng.random() for _ in range(EVO_MAX_NODES)])

    def _build_and_score(genome):
        arm = {}
        nid = [max(rooms.keys()) + 1 if rooms else 0]
        for rid, rinfo in rooms.items():
            arm[rid] = rinfo
        tree = build_bsp_tree_with_genome(
            dict(rooms), boundary, access_edges, genome,
            all_rooms_ref=arm, next_id_ref=nid,
        )
        return tree, arm, evaluate_fitness(tree)

    best_tree, best_arm, best_fitness = None, None, -1e18

    for gen in range(n_gens):
        scored = []
        for genome in population:
            tree, arm, fit = _build_and_score(genome)
            scored.append((genome, tree, arm, fit))
            if fit > best_fitness:
                best_fitness = fit
                best_tree = tree
                best_arm = arm

        scored.sort(key=lambda x: x[3], reverse=True)

        new_pop = [scored[0][0][:]]

        while len(new_pop) < pop_size:
            t_size = min(3, len(scored))
            competitors = rng.sample(scored, t_size)
            parent_a = max(competitors, key=lambda x: x[3])[0]
            competitors = rng.sample(scored, t_size)
            parent_b = max(competitors, key=lambda x: x[3])[0]

            child = []
            for a, b in zip(parent_a, parent_b):
                child.append(a if rng.random() < 0.5 else b)

            n_mutate = rng.randint(1, 3)
            for _ in range(n_mutate):
                idx = rng.randrange(len(child))
                child[idx] = (child[idx] + rng.gauss(0, 0.15)) % 1.0

            new_pop.append(child)

        population = new_pop

    if best_tree is None:
        best_arm = {}
        best_tree = build_bsp_tree(rooms, boundary, access_edges, depth=0,
                                   all_rooms_ref=best_arm, lookahead=0)

    return best_tree, best_arm


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_bsp_step(rooms, boundary, split_info, step_num, ax, title=""):
    """Draw one BSP step on the given axis."""
    for rid, rinfo in rooms.items():
        poly = rinfo["polygon"]
        coords = np.array(list(poly.exterior.coords))
        color = CATEGORY_COLORS.get(rinfo["room_type"], "#DDDDDD")
        p = MplPolygon(coords, closed=True, facecolor=color,
                       edgecolor="black", linewidth=0.8, alpha=0.7)
        ax.add_patch(p)
        cx, cy = poly.centroid.x, poly.centroid.y
        label = f"{rinfo['name']}\n#{rid}"
        ax.text(cx, cy, label, ha="center", va="center", fontsize=5,
                fontweight="bold", zorder=4)

    if boundary is not None:
        bcoords = np.array(list(boundary.exterior.coords))
        ax.plot(bcoords[:, 0], bcoords[:, 1], "k-", linewidth=2, zorder=5)

    if split_info is not None:
        d = split_info["direction"]
        p = split_info["position"]
        minx, miny, maxx, maxy = boundary.bounds
        if d == "h":
            ax.axhline(p, color="red", linewidth=2, linestyle="--", zorder=6)
        else:
            ax.axvline(p, color="red", linewidth=2, linestyle="--", zorder=6)

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.invert_yaxis()
    if title:
        ax.set_title(title, fontsize=8)


def visualise_bsp_tree(tree, rooms, boundary, output_path):
    """Create a multi-panel figure showing the BSP tree structure."""
    steps = []

    def _collect_steps(node, room_set, depth=0):
        if node is None or node.get("final"):
            return
        if "split_direction" in node:
            steps.append((
                depth,
                dict(room_set),
                {
                    "direction": node["split_direction"],
                    "position": node["split_position"],
                },
            ))
        for child in node.get("children", []):
            child_rooms = {}
            for rid, rinfo in room_set.items():
                if _room_in_subtree(rid, child):
                    child_rooms[rid] = rinfo
            _collect_steps(child, child_rooms, depth + 1)

    def _room_in_subtree(rid, node):
        if node.get("final") and node.get("room_id") == rid:
            return True
        for child in node.get("children", []):
            if _room_in_subtree(rid, child):
                return True
        return False

    _collect_steps(tree, rooms)

    n_steps = len(steps)
    if n_steps == 0:
        return

    cols = min(4, n_steps)
    rows = math.ceil(n_steps / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if n_steps == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, (depth, step_rooms, split_info) in enumerate(steps):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        title = f"Step {idx + 1} (d={depth}): {split_info['direction']}@{split_info['position']:.0f}"
        plot_bsp_step(step_rooms, boundary, split_info, idx, ax, title)

    for idx in range(n_steps, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].set_visible(False)

    fig.subplots_adjust(top=0.93, wspace=0.15, hspace=0.25)
    fig.suptitle(f"BSP Construction ({n_steps} splits)", fontsize=12)
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_bsp_result_panel(ax, room_dict, boundary, access_edges, title,
                           entrance_info=None):
    """Draw BSP result rooms with access edges on *ax*.

    *entrance_info*, when provided, is ``(entrance_room_id, entrance_poly)``
    and draws the Outside node and its edge to the entrance room.
    """
    for rid, rinfo in room_dict.items():
        poly = rinfo["polygon"]
        coords = np.array(list(poly.exterior.coords))
        color = CATEGORY_COLORS.get(rinfo["room_type"], "#DDDDDD")
        p = MplPolygon(coords, closed=True, facecolor=color,
                       edgecolor="black", linewidth=1.0, alpha=0.8)
        ax.add_patch(p)
        cx, cy = poly.centroid.x, poly.centroid.y
        label = f"{rinfo['name']}\n#{rid}"
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=5, fontweight="bold", zorder=4)

    if boundary is not None:
        bcoords = np.array(list(boundary.exterior.coords))
        ax.plot(bcoords[:, 0], bcoords[:, 1], "k-", linewidth=2, zorder=5)

    centroids = {}
    for rid, rinfo in room_dict.items():
        poly = rinfo["polygon"]
        centroids[rid] = (poly.centroid.x, poly.centroid.y)

    for u, v in access_edges:
        if u in centroids and v in centroids:
            xu, yu = centroids[u]
            xv, yv = centroids[v]
            ax.plot([xu, xv], [yu, yv], color="#555555", linewidth=0.8,
                    linestyle="--", zorder=3, alpha=0.6)

    if entrance_info is not None:
        ent_rid, ent_poly = entrance_info
        ecx, ecy = ent_poly.centroid.x, ent_poly.centroid.y
        ax.plot(ecx, ecy, marker="s", color="#D32F2F", markersize=7, zorder=7)
        ax.text(ecx, ecy - 6, "Outside", ha="center", va="bottom",
                fontsize=6, fontweight="bold", color="#D32F2F", zorder=7)
        if ent_rid in centroids:
            rx, ry = centroids[ent_rid]
            ax.plot([ecx, rx], [ecy, ry], color="#D32F2F", linewidth=1.2,
                    linestyle="--", zorder=6, alpha=0.8)

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10)


def _draw_entrance_on_axis(ax, sample: dict):
    """Draw the Outside marker and access edge on an axis showing original rooms."""
    entrance = sample.get("entrance_expand")
    if not entrance or len(entrance) < 3:
        return
    ent_poly = Polygon(entrance)
    if not ent_poly.is_valid or ent_poly.is_empty:
        return

    rooms_dict = sample_to_rooms(sample)
    ent_rid = find_entrance_room(sample, rooms_dict)
    if ent_rid is None:
        return

    ecx, ecy = ent_poly.centroid.x, ent_poly.centroid.y
    ax.plot(ecx, ecy, marker="s", color="#D32F2F", markersize=7, zorder=7)
    ax.text(ecx, ecy - 6, "Outside", ha="center", va="bottom",
            fontsize=6, fontweight="bold", color="#D32F2F", zorder=7)

    room_poly = rooms_dict[ent_rid]["polygon"]
    rx, ry = room_poly.centroid.x, room_poly.centroid.y
    ax.plot([ecx, rx], [ecy, ry], color="#D32F2F", linewidth=1.2,
            linestyle="--", zorder=6, alpha=0.8)


def plot_final_bsp(tree, rooms, boundary, access_edges, output_path,
                   sample_name="", sample=None, entrance_info=None):
    """Plot original RPLAN (left), BSP result (centre), and tree text (right)."""
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 7),
                                         gridspec_kw={"width_ratios": [1, 1, 0.7]})

    # -- Left panel: original RPLAN (identical to process_unsuccessful.py) --
    if sample is not None:
        draw_rplan_on_axis(ax0, sample,
                           title=f"Original RPLAN — {sample_name}")
        _draw_entrance_on_axis(ax0, sample)
    else:
        ax0.set_visible(False)

    # -- Centre panel: BSP result --
    all_leaf_rooms = {}
    _collect_leaves(tree, all_leaf_rooms)

    _plot_bsp_result_panel(ax1, all_leaf_rooms, boundary, access_edges,
                           f"BSP Result — {sample_name}",
                           entrance_info=entrance_info)

    categories_bsp = sorted({r["room_type"] for r in all_leaf_rooms.values()})
    legend_patches_bsp = [
        mpatches.Patch(color=CATEGORY_COLORS.get(cat, "#DDD"),
                       label=get_room_name(cat))
        for cat in categories_bsp
    ]
    ax1.legend(handles=legend_patches_bsp, loc="upper left",
               fontsize=7, framealpha=0.8)

    # -- Right panel: BSP tree text --
    tree_text = format_tree(tree, indent=0)
    ax2.text(0.02, 0.98, tree_text, transform=ax2.transAxes,
             fontsize=7, verticalalignment="top", fontfamily="monospace")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.set_title("BSP Tree", fontsize=10)

    fig.subplots_adjust(top=0.93, wspace=0.08)
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _collect_leaves(node, result):
    if node is None:
        return
    if node.get("final"):
        rid = node.get("room_id", id(node))
        poly = node.get("polygon")
        if poly is not None:
            result[rid] = {
                "polygon": poly,
                "room_type": node.get("room_type", 0),
                "name": node.get("name", "?"),
            }
        return
    for child in node.get("children", []):
        _collect_leaves(child, result)


def format_tree(node, indent=0):
    """Format the BSP tree as an indented text string."""
    if node is None:
        return ""
    prefix = "  " * indent
    if node.get("final"):
        return f"{prefix}[{node['name']} #{node.get('room_id', '?')} area={node.get('area', 0):.0f}]\n"

    angle = node.get("angle", 0)
    direction = "H" if abs(angle) < 0.01 else "V"
    pos = node.get("split_position", 0)
    lines = f"{prefix}{node.get('name', '?')} ({direction}@{pos:.0f}, area={node.get('area', 0):.0f})\n"
    for child in node.get("children", []):
        lines += format_tree(child, indent + 1)
    return lines


# ---------------------------------------------------------------------------
# Enrichment: attach polygon data to BSP tree leaves for plotting
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Entrance / "Outside" node
# ---------------------------------------------------------------------------

def _resolve_entrance(sample: dict, leaves: dict):
    """Return ``(entrance_room_id, entrance_polygon)`` or *None*."""
    entrance = sample.get("entrance_expand")
    if not entrance or len(entrance) < 3:
        return None
    ent_poly = Polygon(entrance)
    if not ent_poly.is_valid or ent_poly.is_empty:
        return None
    ent_rid = find_entrance_room(sample, leaves)
    if ent_rid is None:
        return None
    return (ent_rid, ent_poly)


# ---------------------------------------------------------------------------
# Process a single sample
# ---------------------------------------------------------------------------

def process_sample(sample, output_dir: Path, label: str = "",
                   lookahead: int = 1, evolve: bool = False,
                   evo_pop: int = 100, evo_gens: int = 30):
    """Run the full BSP pipeline on one RPLAN sample."""
    name = str(sample.get("name", "unknown"))
    rooms = sample_to_rooms(sample)
    if len(rooms) < 2:
        return False, "too few rooms"

    access_edges = sample_to_access_edges(sample)
    boundary = compute_boundary(rooms)
    if boundary is None:
        return False, "no boundary"

    if evolve:
        tree, all_rooms_ref = evolve_bsp_tree(
            rooms, boundary, access_edges,
            pop_size=evo_pop, n_gens=evo_gens,
        )
    else:
        all_rooms_ref = {}
        tree = build_bsp_tree(rooms, boundary, access_edges, depth=0,
                              all_rooms_ref=all_rooms_ref, lookahead=lookahead)
    if tree is None:
        return False, "BSP returned None"

    final_access = _collect_final_access(tree, all_rooms_ref, access_edges)

    leaves = {}
    _collect_leaves(tree, leaves)
    entrance_info = _resolve_entrance(sample, leaves)
    if entrance_info is not None:
        ent_rid, _ = entrance_info
        final_access.append((OUTSIDE_ROOM_ID, ent_rid))

    out_path = output_dir / f"{label}_{name}_bsp.png"
    plot_final_bsp(tree, all_rooms_ref, boundary, final_access, out_path, name,
                   sample=sample, entrance_info=entrance_info)

    tree_text = format_tree(tree)
    n_leaves = tree_text.count("[")
    n_rooms = len(rooms)
    return True, f"ok ({n_rooms} rooms -> {n_leaves} leaves)"




def _collect_final_access(tree, all_rooms_ref, original_access_edges):
    """Resolve original access edges to leaf-level rooms.

    For each original edge ``(u, v)``:
      - Find all leaf-level descendants of ``u`` and ``v``.
      - Pick the single pair ``(u_leaf, v_leaf)`` with the longest shared
        boundary.

    Additionally, connect sibling fragments (leaves sharing the same
    ``original_id``) that are geometrically adjacent.
    """
    leaves = {}
    _collect_leaves(tree, leaves)

    # Build mapping: original_room_id -> [leaf_ids that descend from it]
    orig_to_leaves = {}
    for lid, linfo in leaves.items():
        orig = all_rooms_ref[lid].get("original_id", lid)
        orig_to_leaves.setdefault(orig, []).append(lid)

    edges = set()

    # 1. Resolve each original access edge to exactly one leaf-pair.
    for u, v in original_access_edges:
        u_leaves = orig_to_leaves.get(u, [])
        v_leaves = orig_to_leaves.get(v, [])
        if not u_leaves or not v_leaves:
            continue

        best_pair = None
        best_length = 0.0
        for ul in u_leaves:
            for vl in v_leaves:
                pu = leaves[ul].get("polygon")
                pv = leaves[vl].get("polygon")
                if pu is None or pv is None:
                    continue
                shared = pu.intersection(pv)
                if shared.is_empty:
                    continue
                slen = shared.length
                if slen > 0.5 and slen > best_length:
                    best_length = slen
                    best_pair = (ul, vl)

        if best_pair is not None:
            edges.add((min(best_pair), max(best_pair)))

    # 2. Connect sibling fragments (same original_id) that are adjacent.
    for orig, lids in orig_to_leaves.items():
        if len(lids) < 2:
            continue
        for i in range(len(lids)):
            for j in range(i + 1, len(lids)):
                pi = leaves[lids[i]].get("polygon")
                pj = leaves[lids[j]].get("polygon")
                if pi is not None and pj is not None and rooms_share_edge(pi, pj):
                    edges.add((min(lids[i], lids[j]), max(lids[i], lids[j])))

    return list(edges)


# ---------------------------------------------------------------------------
# Hypergraph JSON export
# ---------------------------------------------------------------------------

def _point_dict(x, y):
    """Return a 3-D point dict matching the hypergraph schema."""
    return {"X": float(x), "Y": float(y), "Z": 0,
            "Mag": float(math.hypot(x, y))}


def _boundary_to_bounds_json(boundary: Polygon) -> str:
    coords = list(boundary.exterior.coords)[:-1]
    corners = [_point_dict(x, y) for x, y in coords]
    return json.dumps({"corners": corners})


def _boundary_to_facade_json(boundary: Polygon) -> str:
    coords = list(boundary.exterior.coords)
    segments = []
    for i in range(len(coords) - 1):
        segments.append({
            "start": _point_dict(*coords[i]),
            "end": _point_dict(*coords[i + 1]),
        })
    return json.dumps(segments)


def _assign_path_names(node, path="root"):
    """Walk the BSP tree and assign unique path-based ``_hg_name`` to each node.

    Leaf nodes in the internal tree carry room-type names ("Bedroom") which
    are not unique.  The hypergraph format requires unique path names
    ("rootLR", "rootLLL", etc.) so we assign them in a separate pass to avoid
    touching the visualisation code.
    """
    if node is None:
        return
    node["_hg_name"] = path
    children = node.get("children", [])
    if len(children) >= 1:
        _assign_path_names(children[0], path + "L")
    if len(children) >= 2:
        _assign_path_names(children[1], path + "R")


def _build_leaf_name_map(node, result=None):
    """Walk the BSP tree and return ``{room_id: _hg_name}`` for every leaf."""
    if result is None:
        result = {}
    if node is None:
        return result
    if node.get("final"):
        rid = node.get("room_id")
        if rid is not None:
            result[rid] = node["_hg_name"]
        return result
    for child in node.get("children", []):
        _build_leaf_name_map(child, result)
    return result


def _bsp_tree_to_split(node, access_edges, leaf_name_map,
                       parent_angle=0.0):
    """Convert internal BSP tree dict to the hypergraph ``split`` format.

    *access_edges* is a list of ``(room_id_a, room_id_b)`` tuples.
    *leaf_name_map* maps ``room_id -> _hg_name`` for every leaf.
    Edges involving ``OUTSIDE_ROOM_ID`` are resolved to ``"outside"``.

    RGL reads the **children's** angle fields to determine how to split the
    parent's space (see ``DataNode.SubdivideTree`` in C#).  Therefore each
    node must carry its *parent's* split angle, not its own.
    *parent_angle* is threaded through the recursion for this purpose.
    """
    if node is None:
        return None
    hg_name = node.get("_hg_name", node.get("name", "?"))

    if node.get("final"):
        rid = node.get("room_id")
        room_type = node.get("room_type", 0)
        connected_names = []
        for u, v in access_edges:
            if u == rid:
                if v == OUTSIDE_ROOM_ID:
                    connected_names.append("outside")
                elif v in leaf_name_map:
                    connected_names.append(leaf_name_map[v])
            elif v == rid:
                if u == OUTSIDE_ROOM_ID:
                    connected_names.append("outside")
                elif u in leaf_name_map:
                    connected_names.append(leaf_name_map[u])
        return {
            "name": hg_name,
            "area": float(node.get("area", 0)),
            "angle": parent_angle,
            "mergeid": CATEGORY_TO_MERGEID.get(room_type, "extra"),
            "final": True,
            "children": [],
            "connected": connected_names,
        }

    my_split_angle = float(node.get("angle", 0))

    bsp_children = list(node.get("children", []))
    # RGL (CW faces) places the first child on the BOTTOM for H splits but
    # on the LEFT for V splits.  Our BSP puts the "above" child first for H
    # splits, so we must reverse children for horizontal splits to match RGL.
    if node.get("split_direction") == "h":
        bsp_children = list(reversed(bsp_children))

    children_out = []
    for child in bsp_children:
        converted = _bsp_tree_to_split(child, access_edges, leaf_name_map,
                                       parent_angle=my_split_angle)
        if converted is not None:
            children_out.append(converted)

    return {
        "name": hg_name,
        "area": float(node.get("area", 0)),
        "angle": parent_angle,
        "mergeid": hg_name,
        "final": False,
        "children": children_out,
        "connected": [],
    }


def sample_to_hypergraph_entry(sample: dict, lookahead: int = 1,
                               evolve: bool = False,
                               evo_pop: int = 100,
                               evo_gens: int = 30) -> dict | None:
    """Run BSP on one RPLAN sample and return a hypergraph dict (or *None*)."""
    name = str(sample.get("name", "unknown"))
    rooms = sample_to_rooms(sample)
    if len(rooms) < 2:
        return None

    access_edges_orig = sample_to_access_edges(sample)
    boundary = compute_boundary(rooms)
    if boundary is None:
        return None

    if evolve:
        tree, all_rooms_ref = evolve_bsp_tree(
            rooms, boundary, access_edges_orig,
            pop_size=evo_pop, n_gens=evo_gens,
        )
    else:
        all_rooms_ref = {}
        tree = build_bsp_tree(rooms, boundary, access_edges_orig, depth=0,
                              all_rooms_ref=all_rooms_ref, lookahead=lookahead)
    if tree is None:
        return None

    final_access = _collect_final_access(tree, all_rooms_ref, access_edges_orig)

    leaves = {}
    _collect_leaves(tree, leaves)
    entrance_info = _resolve_entrance(sample, leaves)
    if entrance_info is not None:
        ent_rid, _ = entrance_info
        final_access.append((OUTSIDE_ROOM_ID, ent_rid))

    _assign_path_names(tree)
    leaf_name_map = _build_leaf_name_map(tree)
    split_dict = _bsp_tree_to_split(tree, final_access, leaf_name_map)

    bedroom_count = sum(1 for r in sample["rooms"] if r["category"] == 1)
    bathroom_count = sum(1 for r in sample["rooms"] if r["category"] == 3)

    return {
        "database": "rplan",
        "id": name,
        "area": float(boundary.area),
        "bedrooms": bedroom_count,
        "bathrooms": bathroom_count,
        "bounds": _boundary_to_bounds_json(boundary),
        "facade": _boundary_to_facade_json(boundary),
        "circulation": "[]",
        "split": json.dumps(split_dict),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _select_samples(all_samples, args):
    """Return the list of samples to process based on CLI flags."""
    if args.sample is not None:
        name_to_sample = {str(s.get("name", "")): s for s in all_samples}
        selected = []
        for name in args.sample:
            if name in name_to_sample:
                selected.append(name_to_sample[name])
            else:
                logger.error("Sample name '%s' not found in dataset", name)
        return selected

    if args.max_samples is not None:
        return all_samples[:min(args.max_samples, len(all_samples))]

    if args.export:
        return all_samples

    return [all_samples[0]]


def _passes_filter(sample, point_edges_mode):
    if point_edges_mode == "include":
        return True
    has = has_point_only_edges(sample)
    if point_edges_mode == "exclude":
        return not has
    return has  # "only"


def _export_worker(args):
    """Top-level worker for multiprocessing export."""
    sample, point_edges_mode, lookahead, evolve, evo_pop, evo_gens = args
    if not _passes_filter(sample, point_edges_mode):
        return None, "skipped"
    try:
        entry = sample_to_hypergraph_entry(
            sample, lookahead=lookahead,
            evolve=evolve, evo_pop=evo_pop, evo_gens=evo_gens,
        )
    except Exception:
        return None, "fail"
    if entry is not None:
        return entry, "ok"
    return None, "fail"


def main():
    parser = argparse.ArgumentParser(
        description="BSP conversion for RPLAN data (pure Python / shapely)",
    )
    parser.add_argument(
        "--data-dir", type=str, default="dataset/rplan/dataset_access",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default="debug_bsp_output",
        help="Output directory for debug PNGs",
    )
    parser.add_argument("--split", type=str, default=None,
                        choices=["train", "validation", "test"])
    parser.add_argument("--sample", type=str, nargs="+", default=None,
                        help="One or more RPLAN sample names (e.g. 3951 57395)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples to process")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--point-edges", type=str, default="include",
        choices=["include", "exclude", "only"],
        help="Filter samples by point-only access edges: "
             "'include' (default) processes all, "
             "'exclude' skips samples that have them, "
             "'only' processes only samples that have them",
    )
    parser.add_argument(
        "--export", action="store_true",
        help="Export hypergraph JSON instead of debug PNGs. "
             "Respects --sample, --max-samples, --point-edges. "
             "Without --sample or --max-samples, exports ALL samples.",
    )
    parser.add_argument(
        "--export-dir", type=str, default="dataset/rplan/dataset_hypergraph",
        help="Output directory for --export (default: dataset/rplan/dataset_hypergraph)",
    )
    parser.add_argument(
        "--workers", "-j", type=int, default=0,
        help="Number of parallel workers for --export (0 = single process)",
    )
    parser.add_argument(
        "--lookahead", type=int, default=1,
        help="Look-ahead depth for BSP split scoring (0 = greedy, 1 = default)",
    )
    parser.add_argument(
        "--evolve", action="store_true",
        help="Use evolutionary search for BSP split selection (slower, higher quality)",
    )
    parser.add_argument(
        "--evo-pop", type=int, default=100,
        help="Evolutionary search population size (default: 100)",
    )
    parser.add_argument(
        "--evo-gens", type=int, default=30,
        help="Evolutionary search generations (default: 30)",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    data_dir = Path(args.data_dir)
    splits = [args.split] if args.split else (
        ["train", "validation", "test"] if args.export else ["train"]
    )

    for split_name in splits:
        json_path = data_dir / f"data_{split_name}.json"
        if not json_path.exists():
            logger.error("File not found: %s", json_path)
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            all_samples = json.load(f)
        logger.info("Loaded %d samples from %s", len(all_samples), json_path)

        candidates = _select_samples(all_samples, args)
        if not candidates:
            continue

        if args.export:
            export_dir = Path(args.export_dir)
            export_dir.mkdir(parents=True, exist_ok=True)

            entries = []
            ok, fail, skipped = 0, 0, 0
            work_items = [
                (s, args.point_edges, args.lookahead,
                 args.evolve, args.evo_pop, args.evo_gens)
                for s in candidates
            ]

            if args.workers > 0:
                from multiprocessing import Pool
                with Pool(processes=args.workers) as pool:
                    for entry, status in tqdm(
                        pool.imap(_export_worker, work_items, chunksize=64),
                        total=len(work_items),
                        desc=f"export-{split_name}",
                    ):
                        if status == "ok":
                            entries.append(entry)
                            ok += 1
                        elif status == "skipped":
                            skipped += 1
                        else:
                            fail += 1
            else:
                for item in tqdm(work_items, desc=f"export-{split_name}"):
                    entry, status = _export_worker(item)
                    if status == "ok":
                        entries.append(entry)
                        ok += 1
                    elif status == "skipped":
                        skipped += 1
                    else:
                        fail += 1

            out_path = export_dir / f"{split_name}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(entries, f)
            logger.info("Wrote %d entries to %s (%d failed, %d skipped)",
                        ok, out_path, fail, skipped)
        else:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            ok, fail, skipped = 0, 0, 0
            for sample in tqdm(candidates, desc="BSP",
                               disable=len(candidates) == 1):
                if not _passes_filter(sample, args.point_edges):
                    skipped += 1
                    logger.info("Sample #%s skipped by --point-edges=%s",
                                sample.get("name", "?"), args.point_edges)
                    continue
                success, info = process_sample(
                    sample, output_dir, split_name,
                    lookahead=args.lookahead,
                    evolve=args.evolve,
                    evo_pop=args.evo_pop,
                    evo_gens=args.evo_gens,
                )
                if success:
                    ok += 1
                else:
                    fail += 1
                    logger.debug("Sample %s: %s", sample.get("name", "?"), info)
                if len(candidates) == 1:
                    logger.info("Sample #%s: %s",
                                sample.get("name", "?"), info)
            if len(candidates) > 1:
                logger.info("Done: %d ok, %d failed, %d skipped out of %d",
                            ok, fail, skipped, len(candidates))


if __name__ == "__main__":
    main()
