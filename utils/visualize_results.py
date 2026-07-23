#!/usr/bin/env python3
"""
Render a single floor-plan sample as (1) a blueprint-style mesh and (2) its
access graph, with three data modes.

Self-contained: includes the low-level blueprint mesh renderer (formerly
utils/visualization_helpers.py)
(walls, interior door gaps, entry door, apartment boundary) with the access-graph
renderer.

Modes (mutually exclusive CLI flags)
------------------------------------
* ``--generated_data`` (default): Qwen-generated apartments (RPLAN or WMR24).
  Needs ``--sample-id``, ``--hypergraph`` (a ``{id: hypergraph}`` results JSON),
  and ``--test-dir`` (folder of ``<id>.txt`` read for **bounds only**). Mesh and
  **corrected** access graph are built from the generated hypergraph.

* ``--gt_rplan``: RPLAN ground truth. Needs ``--sample-id`` and ``--test-dir``
  (folder of ``<id>.txt`` with ``room_polygons`` + ``accessgraph`` + ``bounds``,
  e.g. ``dataset/rplan/eval_access_mid/test``). The access graph is taken from the
  **stored ``accessgraph`` key**; the mesh is built directly from
  **``room_polygons``** (NOT from a hypergraph).

* ``--gt_wmr24``: WMR24 ground truth. Needs ``--sample-id`` and ``--test-dir``
  (folder of ``<id>.txt`` with an embedded ``hypergraph`` + ``bounds``). Mesh and
  **corrected** access graph are built from the embedded hypergraph, same as the
  generated path.

Library use
-----------
Every render function accepts ``save_path``: a PNG path writes+closes the figure
and returns the path; ``None`` returns the live Figure for embedding.

CLI examples
------------
    # Generated (default)
    python visualize_results.py --sample-id 19764 \\
        --hypergraph results_hypergraphs.json \\
        --test-dir dataset/rplan/dataset_hypergraph_la0_fixed/test \\
        --output-dir /tmp/viz

    # GT RPLAN (mesh from room_polygons)
    python visualize_results.py --gt_rplan --sample-id 19764 \\
        --test-dir dataset/rplan/eval_access_mid/test --output-dir /tmp/viz

    # GT WMR24 (mesh from embedded hypergraph)
    python visualize_results.py --gt_wmr24 --sample-id b-0001 \\
        --test-dir dataset/wmr24/test --output-dir /tmp/viz
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString
from shapely.geometry import Polygon as SPolygon

# This module lives in utils/. Two import concerns when run as a script:
#   1. The repo root (one level up) must be importable so `utils.*` / `rplan.*`
#      resolve.
#   2. Running `python utils/visualize_results.py` puts this file's own dir
#      (utils/) on sys.path, where utils/hypergraph.py would SHADOW the
#      top-level `hypergraph` package that RGL needs. Drop that entry.
# Prefer `python -m utils.visualize_results` from the repo root, but keep direct
# invocation working too.
_SELF_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _SELF_DIR)]
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.hypergraph import Hypergraph, HypergraphPlotter, repair_bsp_tree_names
from utils.metrics import accessgraph_dict_to_nx
from rplan.visualize_rplan_hypergraph import _inject_outside_node


# ===========================================================================
# Low-level blueprint mesh renderer (merged from the former
# utils/visualization_helpers.py). Owns wall/door geometry, open-zone merging,
# and the plot_mesh_on_axis renderer used by render_mesh below. The adjacency-
# graph utilities from that module were dropped (unused here).
# ===========================================================================

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _plot_geom_lines(ax, geom, **kwargs):
    """Plot any Shapely line-like geometry onto ax."""
    if geom is None:
        return
    try:
        if geom.is_empty:
            return
    except Exception:
        return
    gt = getattr(geom, 'geom_type', None)
    if gt in ('LineString', 'LinearRing'):
        try:
            ax.plot(*geom.xy, **kwargs)
        except Exception:
            pass
    elif gt in ('MultiLineString', 'GeometryCollection'):
        for g in geom.geoms:
            if getattr(g, 'geom_type', None) in ('LineString', 'LinearRing', 'MultiLineString'):
                _plot_geom_lines(ax, g, **kwargs)


def _shrink_polygon(poly, inset):
    """Inset poly inward; returns original on degenerate result."""
    if not inset or inset <= 0:
        return poly
    try:
        sh = poly.buffer(-float(inset), join_style=2, mitre_limit=3.0)
    except Exception:
        return poly
    if sh.is_empty or getattr(sh, 'area', 0.0) <= 1e-12:
        return poly
    if sh.geom_type == 'MultiPolygon':
        sh = max(sh.geoms, key=lambda g: g.area)
    return sh


def _shared_wall_line(poly_a, poly_b, *, inset):
    """Return the shared-wall LineString between two polygons, or None.

    Tries (1) exact boundary intersection, (2) boundary-proximity buffer,
    (3) MRR of buffered overlap.  Attempt 3 is skipped when poly_a is mostly
    inside poly_b (entry-door case) to avoid mis-oriented lines.
    """
    min_len = max(2.0 * float(inset), 1e-6)
    eps = max(0.05 * float(inset), 1e-6)

    def _best_line(geom):
        if geom is None or geom.is_empty:
            return None
        gt = geom.geom_type
        candidates = ([geom] if gt == 'LineString'
                      else list(geom.geoms) if gt == 'MultiLineString'
                      else [g for g in geom.geoms
                            if g.geom_type in ('LineString', 'MultiLineString')]
                      if gt == 'GeometryCollection' else [])
        if not candidates:
            return None
        best = max(candidates, key=lambda l: float(l.length))
        return best if float(best.length) >= min_len else None

    try:
        r = _best_line(poly_a.boundary.intersection(poly_b.boundary))
        if r: return r
    except Exception:
        pass

    try:
        near = poly_b.boundary.buffer(eps, join_style=2, mitre_limit=4.0)
        r = _best_line(poly_a.boundary.intersection(near))
        if r: return r
    except Exception:
        pass

    try:
        inside_frac = float(poly_a.intersection(poly_b).area) / max(float(poly_a.area), 1e-12)
    except Exception:
        inside_frac = 0.0
    if inside_frac > 0.5:
        return None

    try:
        overlap = (poly_a.buffer(eps, join_style=2, mitre_limit=4.0)
                   .intersection(poly_b.buffer(eps, join_style=2, mitre_limit=4.0)))
    except Exception:
        return None
    if overlap.is_empty:
        return None
    if overlap.geom_type == 'MultiPolygon':
        overlap = max(overlap.geoms, key=lambda g: g.area)
    elif overlap.geom_type == 'GeometryCollection':
        sub = [g for g in overlap.geoms if g.geom_type == 'Polygon']
        overlap = max(sub, key=lambda g: g.area) if sub else None
    if overlap is None or getattr(overlap, 'geom_type', '') != 'Polygon' or overlap.area <= 0:
        return None
    try:
        mrr = overlap.minimum_rotated_rectangle
    except Exception:
        return None
    coords = list(getattr(mrr, 'exterior', mrr).coords)
    if len(coords) < 5:
        return None
    edges = [(coords[i], coords[i + 1]) for i in range(4)]
    lens = [((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2)**0.5 for p0, p1 in edges]
    idx = max(range(4), key=lambda i: lens[i])
    if lens[idx] < min_len:
        return None
    p0, p1 = edges[idx]; opp0, opp1 = edges[(idx + 2) % 4]
    return LineString([((p0[0]+opp1[0])/2, (p0[1]+opp1[1])/2),
                       ((p1[0]+opp0[0])/2, (p1[1]+opp0[1])/2)])


# ---------------------------------------------------------------------------
# Door drawing
# ---------------------------------------------------------------------------

def _draw_door_rect(ax, A, B, n_out, *, t_inner_w, t_outer_w, t_inner_c, t_outer_c,
                    bg_zorder, fg_zorder, jamb_lw):
    """Draw white door fill + jambs along A-B; return cut SPolygon."""
    def _c(end, t): return end + t * n_out
    pts_w = [_c(A, t_inner_w), _c(B, t_inner_w), _c(B, t_outer_w), _c(A, t_outer_w)]
    ax.fill([p[0] for p in pts_w] + [pts_w[0][0]],
            [p[1] for p in pts_w] + [pts_w[0][1]],
            color='white', edgecolor='none', zorder=bg_zorder, antialiased=False)
    for end in (A, B):
        ja, jb = _c(end, t_inner_w), _c(end, t_outer_w)
        ax.plot([ja[0], jb[0]], [ja[1], jb[1]], '-', color='black',
                linewidth=jamb_lw, zorder=fg_zorder)
    return SPolygon([_c(A, t_inner_c), _c(B, t_inner_c), _c(B, t_outer_c), _c(A, t_outer_c)])


def _door_AB(line, inset, door_w):
    """Return (A, B, d, n) for a door centered on line, or None."""
    L = float(line.length)
    if L < max(2.0 * float(inset), 1e-6):
        return None
    half_w = min(0.5 * float(door_w), 0.45 * L)
    if half_w <= 0:
        return None
    p_a = line.interpolate(0.5 * L - half_w)
    p_b = line.interpolate(0.5 * L + half_w)
    A, B = np.array([p_a.x, p_a.y]), np.array([p_b.x, p_b.y])
    d = B - A
    norm = float(np.linalg.norm(d))
    if norm < 1e-9:
        return None
    d /= norm
    return A, B, d, np.array([-d[1], d[0]])


def _draw_door_between(ax, poly_a, poly_b, *, inset, door_w,
                       bg_zorder=3, fg_zorder=4, jamb_lw=1.2):
    """Draw door through shared wall; return cut SPolygon or None."""
    line = _shared_wall_line(poly_a, poly_b, inset=float(inset))
    if line is None:
        return None
    r = _door_AB(line, inset, door_w)
    if r is None:
        return None
    A, B, _, n = r
    t = float(inset)
    return _draw_door_rect(ax, A, B, n,
                           t_inner_w=-t, t_outer_w=t,
                           t_inner_c=-1.2*t, t_outer_c=1.2*t,
                           bg_zorder=bg_zorder, fg_zorder=fg_zorder, jamb_lw=jamb_lw)


def _draw_entry_door(ax, room_poly, apt_polygon_pts, *, inset, outer_pad, door_w,
                     bg_zorder=3, fg_zorder=4, jamb_lw=1.2):
    """Punch entry door through apartment exterior wall; return cut SPolygon or None."""
    from shapely.geometry import Point as SPoint
    try:
        apt = SPolygon(apt_polygon_pts)
        if not apt.is_valid: apt = apt.buffer(0)
    except Exception:
        return None
    if apt.is_empty:
        return None
    line = _shared_wall_line(room_poly, apt, inset=float(inset))
    if line is None:
        return None
    r = _door_AB(line, inset, door_w)
    if r is None:
        return None
    A, B, d, n_cand = r
    probe = 0.5 * (A + B) + 0.25 * float(inset) * n_cand
    try:
        inside = room_poly.contains(SPoint(probe[0], probe[1]))
    except Exception:
        inside = False
    n_out = -n_cand if inside else n_cand
    t_w, t_p = float(inset), float(outer_pad)
    return _draw_door_rect(ax, A, B, n_out,
                           t_inner_w=-t_w, t_outer_w=t_p,
                           t_inner_c=-1.2*t_w, t_outer_c=t_p+0.2*t_w,
                           bg_zorder=bg_zorder, fg_zorder=fg_zorder, jamb_lw=jamb_lw)


def _draw_entry_door_at_segment(ax, apt_polygon_pts, door_segment, *,
                                inset, outer_pad, bg_zorder=3, fg_zorder=4,
                                jamb_lw=1.2, door_w=None, corner_margin=None):
    """Punch entry door at an explicit boundary segment; return cut SPolygon or None."""
    from shapely.geometry import Point as SPoint
    try:
        apt = SPolygon(apt_polygon_pts)
        if not apt.is_valid: apt = apt.buffer(0)
    except Exception:
        return None
    if apt.is_empty:
        return None
    try:
        p0, p1 = door_segment
        seg_a = np.array([float(p0[0]), float(p0[1])])
        seg_b = np.array([float(p1[0]), float(p1[1])])
    except Exception:
        return None
    seg_d = seg_b - seg_a
    seg_L = float(np.linalg.norm(seg_d))
    if seg_L < 1e-9:
        return None
    seg_center = 0.5 * (seg_a + seg_b)

    boundary = [np.array([float(p[0]), float(p[1])]) for p in apt_polygon_pts]
    best = None
    tol = max(2.0 * float(inset), 1e-3)
    for i in range(len(boundary)):
        E0, E1 = boundary[i], boundary[(i + 1) % len(boundary)]
        ed = E1 - E0
        edge_L = float(np.linalg.norm(ed))
        if edge_L < 1e-9:
            continue
        u = ed / edge_L
        t = float(np.dot(seg_center - E0, u))
        proj = E0 + min(max(t, 0.0), edge_L) * u
        dist = float(np.linalg.norm(seg_center - proj))
        align = abs(float(np.dot(seg_d / seg_L, u)))
        if dist <= tol and align > 0.80:
            score = (dist, -align)
            if best is None or score < best[0]:
                best = (score, E0, u, edge_L, t)

    if best is not None:
        _, edge_start, d, L, center_t = best
        center_t = min(max(float(center_t), 0.0), L)
    else:
        edge_start, d, L, center_t = seg_a, seg_d / seg_L, seg_L, 0.5 * seg_L

    if corner_margin is None:
        corner_margin = 2.0 * float(inset)
    desired_half_w = 0.5 * float(door_w if door_w is not None else L)
    max_half_w = max(0.0, 0.5 * L - float(corner_margin))
    half_w = min(desired_half_w, max_half_w) if max_half_w > 0 else min(desired_half_w, 0.5 * L)
    lo, hi = float(corner_margin) + half_w, L - float(corner_margin) - half_w
    if lo <= hi:
        center_t = min(max(center_t, lo), hi)
    center = edge_start + center_t * d
    A, B = center - half_w * d, center + half_w * d

    n_cand = np.array([-d[1], d[0]])
    probe = 0.5 * (A + B) + max(float(inset), 1e-3) * n_cand
    try:
        n_out = -n_cand if apt.contains(SPoint(probe[0], probe[1])) else n_cand
    except Exception:
        n_out = n_cand
    t_w, t_p = float(inset), float(outer_pad)
    return _draw_door_rect(ax, A, B, n_out,
                           t_inner_w=-t_w, t_outer_w=t_p,
                           t_inner_c=-1.2*t_w, t_outer_c=t_p+0.2*t_w,
                           bg_zorder=bg_zorder, fg_zorder=fg_zorder, jamb_lw=jamb_lw)


# ---------------------------------------------------------------------------

# Open-zone pairs: rooms that render as one continuous space (no wall corridor).
OPEN_ZONE_TYPE_PAIRS = frozenset({
    frozenset({'foyer', 'kitchen'}),
    frozenset({'foyer', 'living'}),
    frozenset({'living', 'kitchen'}),
    frozenset({'living'}),
    frozenset({'foyer'}),
})

def mesh_entry_door_face_idx(hg, mesh):
    """Return face index of the entrance room (connected to 'outside'), or None."""
    leaves = hg.leaves
    if not leaves:
        return None
    candidates = [l for l in leaves
                  if any(str(c).strip().lower() == "outside" for c in (l.get("connected") or []))]
    if not candidates:
        return None
    entry_name = max(candidates, key=lambda l: float(l.get("area", 0.0))).get("name", "")
    if not entry_name:
        return None
    n = int(getattr(getattr(mesh, "faceList", None), "Count", 0) or 0)
    return next((i for i in range(n)
                 if (getattr(mesh.faceList[i], "unique_id", "") or "") == entry_name), None)


def mesh_door_face_pairs(hg, mesh, *, access_graph=None, bounds=None) -> list:
    """Return (face_idx_a, face_idx_b) for each access-graph edge (for door drawing)."""
    leaves = hg.leaves
    if not leaves:
        return []
    leaf_name_to_id = {leaf["name"]: str(i + 1) for i, leaf in enumerate(leaves)}
    id_to_name = {v: k for k, v in leaf_name_to_id.items()}
    n = int(getattr(getattr(mesh, "faceList", None), "Count", 0) or 0)
    name_to_face = {}
    for i in range(n):
        uid = getattr(mesh.faceList[i], "unique_id", "") or ""
        if uid and uid not in name_to_face:
            name_to_face[uid] = i

    if access_graph is None:
        if bounds is not None:
            try:
                access_graph = hg.correct_accessgraph(bounds=bounds)
            except Exception:
                access_graph = hg.access_graph or {}
        else:
            access_graph = hg.access_graph or {}

    pairs, seen = [], set()
    for edge in (access_graph or {}).get("edges", []) or []:
        if not edge or len(edge) < 2:
            continue
        u, v = edge[0], edge[1]
        if str(u).strip().lower() == "outside" or str(v).strip().lower() == "outside":
            continue
        i = name_to_face.get(id_to_name.get(str(u)))
        j = name_to_face.get(id_to_name.get(str(v)))
        if i is None or j is None or i == j:
            continue
        key = (min(i, j), max(i, j))
        if key not in seen:
            seen.add(key); pairs.append((i, j))
    return pairs

# ---------------------------------------------------------------------------
# Blueprint mesh renderer
# ---------------------------------------------------------------------------

def plot_mesh_on_axis(plotter, mesh, ax, boundary_polygon=None,
                      title="Floor Plan", face_labels=None,
                      show_title=True, show_area=True, show_frame=True,
                      wall_inset=None, door_pairs=None,
                      entry_door_face_idx=None, entry_door_segment=None):
    """Render mesh onto ax with blueprint-style walls and doors."""
    from shapely.geometry import Polygon
    min_area = plotter._mesh_min_area(mesh.faceList)

    polys, total_area = [], 0.0
    for i in range(mesh.faceList.Count):
        face = mesh.faceList[i]
        area_val = getattr(face, 'Area', 0)
        if area_val < min_area:
            polys.append(None); continue
        pts = face.getPoints()
        corners = [(pts[j].X, pts[j].Y) for j in range(pts.Count)]
        if len(corners) < 3:
            polys.append(None); continue
        polys.append(Polygon(corners))
        total_area += float(area_val)

    kept = [p for p in polys if p is not None]

    if wall_inset is None:
        if kept:
            all_pts = [(x, y) for p in kept for x, y in zip(*p.exterior.xy)]
            if boundary_polygon:
                all_pts += list(boundary_polygon)
            xs, ys = zip(*all_pts)
            wall_inset = 0.009 * max(max(xs)-min(xs), max(ys)-min(ys), 0.0)
        else:
            wall_inset = 0.0

    use_walls = bool(wall_inset and wall_inset > 0)
    shrunk = [_shrink_polygon(p, wall_inset) if (p is not None and use_walls) else p
              for p in polys]
    if use_walls:
        ax.set_facecolor('white')

    # Union-find for open-zone merging (e.g. kitchen + living share no wall corridor)
    n_faces = len(polys)
    uf = list(range(n_faces))

    def _find(x):
        while uf[x] != x: uf[x] = uf[uf[x]]; x = uf[x]
        return x

    def _union(x, y):
        rx, ry = _find(x), _find(y)
        if rx != ry: uf[rx] = ry

    face_types = [None if polys[i] is None else getattr(mesh.faceList[i], 'merge_id', None)
                  for i in range(n_faces)]

    if use_walls:
        for i in range(n_faces):
            if polys[i] is None or face_types[i] is None: continue
            for j in range(i + 1, n_faces):
                if polys[j] is None or face_types[j] is None: continue
                if frozenset({face_types[i], face_types[j]}) not in OPEN_ZONE_TYPE_PAIRS:
                    continue
                line = _shared_wall_line(polys[i], polys[j], inset=float(wall_inset))
                if line is not None and float(line.length) >= max(2.0*float(wall_inset), 1e-6):
                    _union(i, j)

    zone_members: dict = {}
    for i in range(n_faces):
        if polys[i] is not None:
            zone_members.setdefault(_find(i), []).append(i)

    zone_shrunk: dict = {}
    if use_walls:
        from shapely.ops import unary_union
        for root, members in zone_members.items():
            if len(members) <= 1:
                zone_shrunk[root] = shrunk[members[0]] if members else None
                continue
            eps = max(1e-3 * float(wall_inset), 1e-6)
            try:
                u_poly = unary_union([polys[k].buffer(eps, join_style=2, mitre_limit=4.0)
                                      for k in members])
                if not u_poly.is_empty:
                    u_poly = u_poly.buffer(-eps, join_style=2, mitre_limit=4.0)
            except Exception:
                u_poly = None
            if u_poly is None or u_poly.is_empty:
                zone_shrunk[root] = None; continue
            if u_poly.geom_type == 'MultiPolygon':
                u_poly = max(u_poly.geoms, key=lambda g: g.area)
            zone_shrunk[root] = _shrink_polygon(u_poly, wall_inset)

    outer_pad = 1.5 * float(wall_inset) if use_walls else 0.0
    outer_pts = list(boundary_polygon) if boundary_polygon else None
    if boundary_polygon and outer_pad > 0:
        try:
            bp = Polygon(boundary_polygon)
            if not bp.is_valid: bp = bp.buffer(0)
            bp_exp = bp.buffer(outer_pad, join_style=2, mitre_limit=4.0)
            if not bp_exp.is_empty and bp_exp.geom_type == 'Polygon':
                outer_pts = list(bp_exp.exterior.coords)
        except Exception:
            outer_pts = list(boundary_polygon)

    if outer_pts and not use_walls:
        bx = [p[0] for p in outer_pts] + [outer_pts[0][0]]
        by = [p[1] for p in outer_pts] + [outer_pts[0][1]]
        ax.plot(bx, by, 'k-', linewidth=2, zorder=1.5)

    for i in range(mesh.faceList.Count):
        if polys[i] is None: continue
        room_type = getattr(mesh.faceList[i], 'merge_id', 'unknown')
        color = plotter.color_map.get(room_type, '#D3D3D3')

        fill_geom = None
        if use_walls:
            root_i = _find(i)
            if len(zone_members.get(root_i, [])) > 1:
                zs = zone_shrunk.get(root_i)
                if zs is not None and not zs.is_empty:
                    try: fill_geom = polys[i].intersection(zs)
                    except Exception: fill_geom = None
        if fill_geom is None or fill_geom.is_empty:
            fill_geom = shrunk[i] if shrunk[i] is not None else polys[i]

        for g in (fill_geom.geoms if getattr(fill_geom, 'geom_type', None) == 'MultiPolygon'
                  else [fill_geom]):
            try:
                xs_g, ys_g = g.exterior.xy
                ec = 'none' if use_walls else 'black'
                lw = 0 if use_walls else 0.5
                ax.fill(xs_g, ys_g, color=color, edgecolor=ec, linewidth=lw, alpha=1, zorder=2)
            except Exception:
                continue

        try:
            cx_, cy_ = float((shrunk[i] or polys[i]).centroid.x), float((shrunk[i] or polys[i]).centroid.y)
        except Exception:
            try:
                xs0, ys0 = polys[i].exterior.xy
                cx_, cy_ = float(np.mean(list(xs0))), float(np.mean(list(ys0)))
            except Exception:
                cx_, cy_ = 0.0, 0.0
        label = face_labels[i] if face_labels is not None and i < len(face_labels) else room_type
        ax.text(cx_, cy_, label, ha='center', va='center', fontsize=16, fontweight='bold', zorder=5)

    door_w = 8.0 * float(wall_inset) if use_walls else 0.0
    door_cuts: list = []
    kw = dict(bg_zorder=3, fg_zorder=4, jamb_lw=1.5)

    if use_walls and door_pairs:
        for pair in door_pairs:
            if not pair or len(pair) < 2: continue
            a, b = int(pair[0]), int(pair[1])
            if not (0 <= a < len(polys) and 0 <= b < len(polys)): continue
            if polys[a] is None or polys[b] is None: continue
            if _find(a) == _find(b): continue  # same open zone, no door needed
            cut = _draw_door_between(ax, polys[a], polys[b],
                                     inset=float(wall_inset), door_w=door_w, **kw)
            if cut is not None: door_cuts.append(cut)

    if use_walls and boundary_polygon and outer_pad > 0:
        if entry_door_segment is not None:
            cut = _draw_entry_door_at_segment(
                ax, boundary_polygon, entry_door_segment,
                inset=float(wall_inset), outer_pad=float(outer_pad),
                door_w=door_w, corner_margin=2.0*float(wall_inset), **kw)
            if cut is not None: door_cuts.append(cut)
        elif entry_door_face_idx is not None:
            ei = int(entry_door_face_idx)
            if 0 <= ei < len(polys) and polys[ei] is not None:
                cut = _draw_entry_door(ax, polys[ei], boundary_polygon,
                                       inset=float(wall_inset), outer_pad=float(outer_pad),
                                       door_w=door_w, **kw)
                if cut is not None: door_cuts.append(cut)

    if use_walls:
        def _apply_cuts(geom):
            for cut in door_cuts:
                try:
                    if cut.intersects(geom): geom = geom.difference(cut)
                except Exception:
                    pass
            return geom

        for root, members in zone_members.items():
            if not members: continue
            sh = zone_shrunk.get(root) or shrunk[members[0]] or polys[members[0]]
            if sh is None: continue
            try:
                contour = _apply_cuts(sh.boundary)
            except Exception:
                continue
            _plot_geom_lines(ax, contour, color='black', linewidth=2.0, zorder=4)
            for ii in range(len(members)):
                for jj in range(ii + 1, len(members)):
                    line = _shared_wall_line(polys[members[ii]], polys[members[jj]],
                                            inset=float(wall_inset))
                    if line is None: continue
                    try: inside = line.intersection(sh)
                    except Exception: inside = None
                    if inside and not inside.is_empty:
                        _plot_geom_lines(ax, inside, color='black', linewidth=0.8, zorder=4)

        if outer_pts:
            from shapely.geometry import LineString as _SLine
            try:
                outer_geom = _apply_cuts(_SLine(list(outer_pts) + [outer_pts[0]]))
                _plot_geom_lines(ax, outer_geom, color='black', linewidth=3.0, zorder=4)
            except Exception:
                pass

    ax.set_aspect('equal')
    if show_title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    if show_area:
        ax.text(0.02, 0.98, f"Area: {total_area:.1f}m\u00b2",
                transform=ax.transAxes, verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor='lightyellow'))
    if not show_frame:
        ax.set_axis_off()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _parse_jsonish(val):
    """Return a dict whether *val* is already a dict or a JSON string."""
    if isinstance(val, str):
        val = json.loads(val)
    return val if isinstance(val, dict) else {}


def load_hypergraph(path: Path, sample_id: str | None):
    """Load a hypergraph dict from *path*, selecting *sample_id* if a results map."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, str):
        data = json.loads(data)

    looks_like_hg = isinstance(data, dict) and any(
        k in data for k in ("root", "children", "split", "name", "area")
    )
    if looks_like_hg:
        return data, sample_id

    if not isinstance(data, dict) or not data:
        raise ValueError(f"No hypergraph found in {path}")

    if sample_id is None:
        sample_id = next(iter(data))
        print(f"No sample-id given; using first key: {sample_id}")
    if sample_id not in data:
        raise KeyError(f"sample-id {sample_id!r} not in {path}")
    return _parse_jsonish(data[sample_id]), sample_id


def load_txt(test_dir: Path, sample_id: str) -> dict:
    """Load and JSON-normalize a dataset ``<id>.txt``.

    Returns a dict with whatever keys the file has, each parsed from JSON strings
    where needed: ``bounds`` (dict), ``hypergraph`` (dict|None),
    ``accessgraph`` (dict), ``room_polygons`` (list).
    """
    path = test_dir / f"{sample_id}.txt"
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, str):
        raw = json.loads(raw)

    out: dict = {}
    out["bounds"] = _parse_jsonish(raw.get("bounds", "{}"))

    hg = raw.get("hypergraph")
    if isinstance(hg, str):
        try:
            hg = json.loads(hg)
        except Exception:
            hg = None
    out["hypergraph"] = hg if isinstance(hg, dict) else None

    out["accessgraph"] = _parse_jsonish(raw.get("accessgraph", "{}"))

    rp = raw.get("room_polygons")
    if isinstance(rp, str):
        try:
            rp = json.loads(rp)
        except Exception:
            rp = None
    out["room_polygons"] = rp if isinstance(rp, list) else []
    return out


# ---------------------------------------------------------------------------
# Labels (room types only, no per-room numbers)
# ---------------------------------------------------------------------------

# Friendly display names for the short dataset room types.
ROOM_DISPLAY_NAMES = {
    "bed": "bedroom",
    "bath": "bath",
    "living": "living",
    "kitchen": "kitchen",
    "balcony": "balcony",
    "foyer": "foyer",
    "storage": "storage",
    "outside": "outside",
}

_TRAILING_DIGITS = re.compile(r"\d+$")


def _room_type_label(raw: str) -> str:
    """Map a raw type or numbered label ('bed', 'bed1') to a display name."""
    rtype = _TRAILING_DIGITS.sub("", str(raw)).strip()
    return ROOM_DISPLAY_NAMES.get(rtype.lower(), rtype)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

ENTRY_PREFERRED_TYPES = ("living", "foyer", "corridor")


def _bounds_polygon(bounds: dict) -> list[tuple[float, float]]:
    corners = (bounds or {}).get("corners", [])
    return [(c["X"], c["Y"]) for c in corners]


def _corrected_accessgraph(hg: Hypergraph, hg_dict: dict, bounds: dict) -> dict:
    """Mesh-corrected access graph with the 'outside' node injected.

    Falls back to the leaf-connectivity access graph if correction fails or the
    boundary is unusable.
    """
    ag: dict
    if bounds and isinstance(bounds, dict) and bounds.get("corners"):
        try:
            ag = hg.correct_accessgraph(bounds)
        except Exception:
            ag = hg.access_graph
    else:
        ag = hg.access_graph
    base = {"nodes": dict(ag.get("nodes") or {}), "edges": list(ag.get("edges") or [])}
    return _inject_outside_node(base, hg_dict)


def _save_or_return(fig, save_path):
    """Save *fig* to *save_path* (PNG) and close it, or return the live figure."""
    if save_path is None:
        return fig
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return save_path


def _build_mesh_from_room_polygons(rooms: list[dict]):
    """Build an ``RGL.NMesh`` from RPLAN ``room_polygons``.

    Each room becomes an ``NFace`` with ``merge_id`` and ``unique_id`` set.
    Returns ``(mesh, rid_to_face_idx)`` so access-graph edges (which use room ids)
    can be mapped to face indices for door drawing.
    """
    RGL, Collections = Hypergraph._init_rgl()
    Vec3d, NFace, NMesh = RGL.Vec3d, RGL.NFace, RGL.NMesh

    nfaces = []
    rid_to_idx: dict[str, int] = {}
    for i, r in enumerate(rooms):
        coords = r.get("coords") or []
        if len(coords) < 3:
            continue
        pts = Collections.Generic.List[Vec3d]()
        for x, y in coords:
            pts.Add(Vec3d(x=float(x), y=float(y), z=0.0))
        nf = NFace(pts)
        nf.merge_id = str(r.get("mergeid", "unknown"))
        rid = str(r.get("id", i + 1))
        try:
            nf.unique_id = rid
        except Exception:
            pass
        try:
            nf.updateEdgeConnectivity()
        except Exception:
            pass
        nfaces.append(nf)
        rid_to_idx[rid] = len(nfaces) - 1

    nfl = Collections.Generic.List[NFace]()
    for nf in nfaces:
        nfl.Add(nf)
    return NMesh(nfl), rid_to_idx


def _door_pairs_from_accessgraph(ag: dict, rid_to_idx: dict[str, int]) -> list[tuple[int, int]]:
    """Map access-graph edges (room ids) to (face_idx, face_idx) pairs."""
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for edge in (ag.get("edges") or []):
        if len(edge) < 2:
            continue
        u, v = str(edge[0]), str(edge[1])
        if u == "outside" or v == "outside" or u not in rid_to_idx or v not in rid_to_idx:
            continue
        a, b = rid_to_idx[u], rid_to_idx[v]
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    return pairs


def _pick_entry_face_from_rooms(rooms: list[dict]) -> int | None:
    """Entry face for room_polygons: lowest-id living, then foyer, then corridor."""
    if not rooms:
        return None

    def _key(idx_room):
        i, r = idx_room
        rid = r.get("id", i + 1)
        try:
            return (0, str(int(rid)).zfill(8))
        except (ValueError, TypeError):
            return (1, str(rid))

    indexed = sorted(enumerate(rooms), key=_key)
    for room_type in ENTRY_PREFERRED_TYPES:
        for i, r in indexed:
            if str(r.get("mergeid", "")).lower() == room_type:
                return i
    return None


# ---------------------------------------------------------------------------
# Mesh rendering
#
# All meshes render through the single _render_mesh() core, which owns the only
# call into plot_mesh_on_axis (always clean-blueprint
# style: no frame, no title, no area badge). The two prep functions below differ
# only in how they derive (mesh, door_pairs, entry_idx, face_labels).
# ---------------------------------------------------------------------------

def _render_mesh(mesh, polygon, *, door_pairs, entry_idx, face_labels,
                 plotter, save_path, title):
    """Blueprint-render an already-built mesh. Returns a Figure or save path."""
    if not polygon:
        raise ValueError("bounds has no 'corners' — cannot draw boundary/walls")
    plotter = plotter or HypergraphPlotter()
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_mesh_on_axis(
        plotter, mesh, ax,
        boundary_polygon=polygon,
        title=title or "Floor Plan",
        face_labels=[_room_type_label(lbl) for lbl in face_labels],
        door_pairs=door_pairs,
        entry_door_face_idx=entry_idx,
        show_title=True, show_area=False, show_frame=False,
    )
    ax.invert_yaxis()
    return _save_or_return(fig, save_path)


def render_mesh(hg_dict: dict, bounds: dict, *, save_path=None, title=None,
                plotter: HypergraphPlotter | None = None):
    """Render a hypergraph mesh (walls + doors). Returns a Figure or save path.

    Used for generated data and GT WMR24 (both hypergraph-based).
    """
    hg = Hypergraph(repair_bsp_tree_names(hg_dict))
    mesh = hg.to_mesh(bounds=bounds)
    return _render_mesh(
        mesh, _bounds_polygon(bounds),
        door_pairs=mesh_door_face_pairs(hg, mesh, bounds=bounds),
        entry_idx=mesh_entry_door_face_idx(hg, mesh),
        face_labels=hg.mesh_face_labels(mesh),
        plotter=plotter, save_path=save_path, title=title)


def render_mesh_from_polygons(rooms: list[dict], bounds: dict, accessgraph: dict, *,
                              save_path=None, title=None,
                              plotter: HypergraphPlotter | None = None):
    """Render a mesh built directly from ``room_polygons`` (GT RPLAN).

    Doors come from the stored access graph; entry door from the preferred-type
    heuristic. Returns a Figure or save path.
    """
    mesh, rid_to_idx = _build_mesh_from_room_polygons(rooms)
    return _render_mesh(
        mesh, _bounds_polygon(bounds),
        door_pairs=_door_pairs_from_accessgraph(accessgraph or {}, rid_to_idx),
        entry_idx=_pick_entry_face_from_rooms(rooms),
        face_labels=[r.get("mergeid", "") for r in rooms],
        plotter=plotter, save_path=save_path, title=title)


# ---------------------------------------------------------------------------
# Access-graph rendering
# ---------------------------------------------------------------------------

def _draw_accessgraph(ag: dict, *, save_path, title, plotter,
                      node_size, layout_compression, label_offset):
    """Draw an access-graph dict in the reference style (labels above nodes,
    types only, nodes pulled together). Shared by both AG sources."""
    import networkx as nx

    plotter = plotter or HypergraphPlotter()
    g = accessgraph_dict_to_nx(ag)
    fig, ax = plt.subplots(figsize=(8, 8))
    if g.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No nodes", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        ax.axis("off")
        return _save_or_return(fig, save_path)

    components = list(nx.connected_components(g))
    pos = plotter._layout_by_component(g, components)

    if pos and layout_compression and layout_compression != 1.0:
        cx = sum(p[0] for p in pos.values()) / len(pos)
        cy = sum(p[1] for p in pos.values()) / len(pos)
        pos = {n: (cx + (x - cx) * layout_compression,
                   cy + (y - cy) * layout_compression)
               for n, (x, y) in pos.items()}

    node_fill = [plotter.ROOM_COLORS.get(g.nodes[n].get("label", ""), "#CCCCCC")
                 for n in g.nodes()]

    nx.draw_networkx_edges(g, pos, edge_color="#b8b5b4", width=4.0,
                           style="dotted", ax=ax)
    nx.draw_networkx_nodes(g, pos, node_color=node_fill,
                           edgecolors="black", linewidths=1.0,
                           node_size=node_size, ax=ax)
    for n, (x, y) in pos.items():
        label = _room_type_label(g.nodes[n].get("label", n))
        ax.text(x, y + label_offset, label, ha="center", va="bottom",
                fontsize=12, fontweight="bold", zorder=5)

    ax.margins(0.18)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    return _save_or_return(fig, save_path)


def render_accessgraph(hg_dict: dict, bounds: dict, *, save_path=None, title=None,
                       plotter: HypergraphPlotter | None = None,
                       node_size=1200, layout_compression=0.55, label_offset=0.16):
    """Render the **corrected** access graph from a hypergraph + bounds.

    Used for generated data and GT WMR24. Returns a Figure or save path.
    """
    hg_dict = repair_bsp_tree_names(hg_dict)
    hg = Hypergraph(hg_dict)
    ag = _corrected_accessgraph(hg, hg_dict, bounds)
    return _draw_accessgraph(
        ag, save_path=save_path, title=title or "Corrected Access Graph",
        plotter=plotter, node_size=node_size,
        layout_compression=layout_compression, label_offset=label_offset)


def render_accessgraph_from_dict(ag: dict, *, save_path=None, title=None,
                                 plotter: HypergraphPlotter | None = None,
                                 node_size=1200, layout_compression=0.55,
                                 label_offset=0.16):
    """Render a **stored** access-graph dict as-is (GT RPLAN ``accessgraph`` key).

    Returns a Figure or save path.
    """
    return _draw_accessgraph(
        ag or {"nodes": {}, "edges": []},
        save_path=save_path, title=title or "Access Graph",
        plotter=plotter, node_size=node_size,
        layout_compression=layout_compression, label_offset=label_offset)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def visualize_generated(sample_id: str, hypergraph_json: Path, test_dir: Path, *,
                        mesh_path=None, accessgraph_path=None,
                        use_generated_bounds: bool = False):
    """Generated data: mesh + corrected AG from the generated hypergraph.

    Bounds source:
    - default: GT bounds from ``<test_dir>/<id>.txt``
    - ``use_generated_bounds=True``: the ``bounds`` key embedded in the
      generated hypergraph root (model-predicted footprint).
    """
    hg_dict, sid = load_hypergraph(hypergraph_json, sample_id)
    if use_generated_bounds:
        bounds = _parse_jsonish(hg_dict.get("bounds", "{}"))
        if not bounds.get("corners"):
            raise ValueError(
                f"{sid}: generated hypergraph has no 'bounds' key — "
                "was the model run with --include-bounds?"
            )
    else:
        bounds = load_txt(test_dir, sid)["bounds"]
    plotter = HypergraphPlotter()
    pfx = f"{sid} " if sid else ""
    mesh = render_mesh(hg_dict, bounds, save_path=mesh_path,
                       title=f"{pfx}Mesh".strip(), plotter=plotter)
    ag = render_accessgraph(hg_dict, bounds, save_path=accessgraph_path,
                            title=f"{pfx}Access Graph".strip(), plotter=plotter)
    return mesh, ag


def visualize_gt_rplan(sample_id: str, test_dir: Path, *,
                       mesh_path=None, accessgraph_path=None):
    """GT RPLAN: mesh from ``room_polygons``, AG from stored ``accessgraph`` key."""
    txt = load_txt(test_dir, sample_id)
    bounds, ag, rooms = txt["bounds"], txt["accessgraph"], txt["room_polygons"]
    if not rooms:
        raise ValueError(f"{sample_id}: no 'room_polygons' in {test_dir} (use eval_access_mid?)")
    plotter = HypergraphPlotter()
    pfx = f"{sample_id} "
    mesh = render_mesh_from_polygons(rooms, bounds, ag, save_path=mesh_path,
                                     title=f"{pfx}Mesh".strip(), plotter=plotter)
    ag_res = render_accessgraph_from_dict(ag, save_path=accessgraph_path,
                                          title=f"{pfx}Access Graph".strip(),
                                          plotter=plotter)
    return mesh, ag_res


def visualize_gt_wmr24(sample_id: str, test_dir: Path, *,
                       mesh_path=None, accessgraph_path=None):
    """GT WMR24: mesh + corrected AG from the embedded hypergraph."""
    txt = load_txt(test_dir, sample_id)
    bounds, hg_dict = txt["bounds"], txt["hypergraph"]
    if not hg_dict:
        raise ValueError(f"{sample_id}: no embedded 'hypergraph' in {test_dir}")
    plotter = HypergraphPlotter()
    pfx = f"{sample_id} "
    mesh = render_mesh(hg_dict, bounds, save_path=mesh_path,
                       title=f"{pfx}Mesh".strip(), plotter=plotter)
    ag = render_accessgraph(hg_dict, bounds, save_path=accessgraph_path,
                            title=f"{pfx}Access Graph".strip(), plotter=plotter)
    return mesh, ag


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--generated_data", action="store_true",
                      help="(default) Qwen-generated apartments; needs --hypergraph.")
    mode.add_argument("--gt_rplan", action="store_true",
                      help="RPLAN GT; mesh from room_polygons, AG from stored accessgraph.")
    mode.add_argument("--gt_wmr24", action="store_true",
                      help="WMR24 GT; mesh + corrected AG from embedded hypergraph.")

    p.add_argument("--sample-id", required=True, help="Sample id (file stem / results key).")
    p.add_argument("--test-dir", type=Path, required=True,
                   help="Folder of <id>.txt files.")
    p.add_argument("--hypergraph", type=Path, default=None,
                   help="Results JSON {id: hypergraph} (generated mode only).")
    p.add_argument("--output-dir", type=Path, default=Path("."),
                   help="Directory for the two output PNGs (default: cwd).")
    p.add_argument("--use-generated-bounds", action="store_true",
                   help="Use the model-predicted 'bounds' from the hypergraph JSON "
                        "instead of GT bounds from --test-dir (generated mode only).")
    args = p.parse_args()

    tag = str(args.sample_id)
    mesh_png = args.output_dir / f"{tag}_mesh.png"
    ag_png = args.output_dir / f"{tag}_accessgraph.png"

    if args.gt_rplan:
        mesh_res, ag_res = visualize_gt_rplan(
            args.sample_id, args.test_dir,
            mesh_path=mesh_png, accessgraph_path=ag_png)
    elif args.gt_wmr24:
        mesh_res, ag_res = visualize_gt_wmr24(
            args.sample_id, args.test_dir,
            mesh_path=mesh_png, accessgraph_path=ag_png)
    else:  # generated (default)
        if args.hypergraph is None:
            p.error("--generated_data requires --hypergraph")
        mesh_res, ag_res = visualize_generated(
            args.sample_id, args.hypergraph, args.test_dir,
            mesh_path=mesh_png, accessgraph_path=ag_png,
            use_generated_bounds=args.use_generated_bounds)

    print(f"Saved mesh:        {Path(mesh_res).resolve()}")
    print(f"Saved accessgraph: {Path(ag_res).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
