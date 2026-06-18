"""Floor-plan mesh visualization helpers for HypergraphPlotter.plot_mesh_on_axis.

All functions here deal with geometry / drawing only; they have no dependency on
``Hypergraph`` or ``HypergraphPlotter`` and may be imported independently.
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import LineString
from shapely.geometry import Polygon as SPolygon


# ---------------------------------------------------------------------------
# Shapely line rendering
# ---------------------------------------------------------------------------

def _plot_geom_lines(ax, geom, **kwargs):
    """Plot any Shapely line-like geometry onto *ax* via ``ax.plot``."""
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
            xs, ys = geom.xy
        except Exception:
            return
        ax.plot(list(xs), list(ys), **kwargs)
    elif gt in ('MultiLineString', 'GeometryCollection'):
        for g in geom.geoms:
            if getattr(g, 'geom_type', None) in ('LineString', 'LinearRing', 'MultiLineString'):
                _plot_geom_lines(ax, g, **kwargs)


# ---------------------------------------------------------------------------
# Polygon wall helpers
# ---------------------------------------------------------------------------

def _shrink_polygon(poly, inset):
    """Inset *poly* inward by *inset*; returns original on degenerate result."""
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
    """Return the shared-wall LineString between two adjacent polygons, or None.

    Tries exact boundary intersection, then boundary-proximity, then MRR
    fallback (the MRR fallback is skipped when poly_a is mostly inside poly_b,
    i.e. the entry-door case, to avoid mis-oriented wall lines).
    """
    def _extract_lines(geom):
        if geom is None or geom.is_empty:
            return []
        gt = geom.geom_type
        if gt == 'LineString':
            return [geom]
        if gt == 'MultiLineString':
            return list(geom.geoms)
        if gt == 'GeometryCollection':
            out = []
            for g in geom.geoms:
                out.extend(_extract_lines(g))
            return out
        return []

    min_len = max(2.0 * float(inset), 1e-6)
    eps = max(0.05 * float(inset), 1e-6)

    # Attempt 1: exact boundary intersection
    try:
        inter = poly_a.boundary.intersection(poly_b.boundary)
    except Exception:
        inter = None
    flat = _extract_lines(inter)
    if flat:
        best = max(flat, key=lambda l: float(l.length))
        if float(best.length) >= min_len:
            return best

    # Attempt 2: boundary-proximity (handles FP misalignment and entry-door case)
    try:
        near_b = poly_b.boundary.buffer(eps, join_style=2, mitre_limit=4.0)
        segs = poly_a.boundary.intersection(near_b)
    except Exception:
        segs = None
    flat_b = _extract_lines(segs)
    if flat_b:
        best_b = max(flat_b, key=lambda l: float(l.length))
        if float(best_b.length) >= min_len:
            return best_b

    # Attempt 3: MRR of buffered overlap (last resort; skip for entry-door case)
    try:
        inside_frac = float(poly_a.intersection(poly_b).area) / max(float(poly_a.area), 1e-12)
    except Exception:
        inside_frac = 0.0
    if inside_frac > 0.5:
        return None

    try:
        overlap = poly_a.buffer(eps, join_style=2, mitre_limit=4.0).intersection(
            poly_b.buffer(eps, join_style=2, mitre_limit=4.0)
        )
    except Exception:
        return None
    if overlap.is_empty:
        return None
    if overlap.geom_type == 'MultiPolygon':
        overlap = max(overlap.geoms, key=lambda g: g.area)
    elif overlap.geom_type == 'GeometryCollection':
        sub = [g for g in overlap.geoms if g.geom_type == 'Polygon']
        overlap = max(sub, key=lambda g: g.area) if sub else None
    if overlap is None or overlap.geom_type != 'Polygon' or overlap.area <= 0:
        return None
    try:
        mrr = overlap.minimum_rotated_rectangle
    except Exception:
        return None
    coords = list(getattr(mrr, 'exterior', mrr).coords)
    if len(coords) < 5:
        return None
    edges = [(coords[i], coords[i + 1]) for i in range(4)]
    lens = [((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5 for p0, p1 in edges]
    idx = max(range(4), key=lambda i: lens[i])
    if lens[idx] < min_len:
        return None
    p0, p1 = edges[idx]
    opp0, opp1 = edges[(idx + 2) % 4]
    c0 = ((p0[0] + opp1[0]) / 2, (p0[1] + opp1[1]) / 2)
    c1 = ((p1[0] + opp0[0]) / 2, (p1[1] + opp0[1]) / 2)
    return LineString([c0, c1])


# ---------------------------------------------------------------------------
# Door drawing
# ---------------------------------------------------------------------------

def _draw_door_rect(ax, A, B, n_out, *, t_inner_w, t_outer_w, t_inner_c, t_outer_c,
                    bg_zorder, fg_zorder, jamb_lw):
    """Draw white door fill + perpendicular jambs along A-B; return cut SPolygon.

    ``t_inner_w`` / ``t_outer_w`` are signed distances along ``n_out`` for the
    white fill (and jambs).  ``t_inner_c`` / ``t_outer_c`` are slightly wider
    for the returned cut polygon so that room contour lines break cleanly.
    """
    def _corner(end, t):
        return end + t * n_out

    xs_f = [_corner(A, t_inner_w)[0], _corner(B, t_inner_w)[0],
            _corner(B, t_outer_w)[0], _corner(A, t_outer_w)[0],
            _corner(A, t_inner_w)[0]]
    ys_f = [_corner(A, t_inner_w)[1], _corner(B, t_inner_w)[1],
            _corner(B, t_outer_w)[1], _corner(A, t_outer_w)[1],
            _corner(A, t_inner_w)[1]]
    ax.fill(xs_f, ys_f, color='white', edgecolor='none',
            zorder=bg_zorder, antialiased=False)
    for end in (A, B):
        ja, jb = _corner(end, t_inner_w), _corner(end, t_outer_w)
        ax.plot([ja[0], jb[0]], [ja[1], jb[1]], '-', color='black',
                linewidth=jamb_lw, zorder=fg_zorder)
    cut_pts = [_corner(A, t_inner_c), _corner(B, t_inner_c),
               _corner(B, t_outer_c), _corner(A, t_outer_c)]
    return SPolygon(cut_pts)


def _draw_door_between(ax, poly_a, poly_b, *, inset, door_w,
                       bg_zorder=3, fg_zorder=4, jamb_lw=1.2):
    """Draw a door symbol through the shared wall between two rooms.

    Returns a cut SPolygon that callers should subtract from room contour
    LineStrings, or None if no shared wall could be found.
    """
    line = _shared_wall_line(poly_a, poly_b, inset=float(inset))
    if line is None:
        return None
    L = float(line.length)
    if L < max(2.0 * float(inset), 1e-6):
        return None
    half_w = min(0.5 * float(door_w), 0.45 * L)
    if half_w <= 0:
        return None
    p_a = line.interpolate(0.5 * L - half_w)
    p_b = line.interpolate(0.5 * L + half_w)
    A = np.array([p_a.x, p_a.y])
    B = np.array([p_b.x, p_b.y])
    d = B - A
    norm = float(np.linalg.norm(d))
    if norm < 1e-9:
        return None
    n = np.array([-d[1], d[0]]) / norm  # unit normal (either side)
    t = float(inset)
    return _draw_door_rect(ax, A, B, n,
                           t_inner_w=-t, t_outer_w=t,
                           t_inner_c=-1.20 * t, t_outer_c=1.20 * t,
                           bg_zorder=bg_zorder, fg_zorder=fg_zorder, jamb_lw=jamb_lw)


def _draw_entry_door(ax, room_poly, apt_polygon_pts, *, inset, outer_pad, door_w,
                     bg_zorder=3, fg_zorder=4, jamb_lw=1.2):
    """Punch an entry door through the apartment exterior wall of *room_poly*.

    Returns a cut SPolygon, or None on failure.
    """
    from shapely.geometry import Point as SPoint
    try:
        apt = SPolygon(apt_polygon_pts)
        if not apt.is_valid:
            apt = apt.buffer(0)
    except Exception:
        return None
    if apt.is_empty:
        return None
    line = _shared_wall_line(room_poly, apt, inset=float(inset))
    if line is None:
        return None
    L = float(line.length)
    if L < max(2.0 * float(inset), 1e-6):
        return None
    half_w = min(0.5 * float(door_w), 0.45 * L)
    if half_w <= 0:
        return None
    p_a = line.interpolate(0.5 * L - half_w)
    p_b = line.interpolate(0.5 * L + half_w)
    A = np.array([p_a.x, p_a.y])
    B = np.array([p_b.x, p_b.y])
    d = B - A
    norm = float(np.linalg.norm(d))
    if norm < 1e-9:
        return None
    d /= norm
    n_cand = np.array([-d[1], d[0]])
    probe = 0.5 * (A + B) + 0.25 * float(inset) * n_cand
    try:
        inside = room_poly.contains(SPoint(probe[0], probe[1]))
    except Exception:
        inside = False
    n_out = -n_cand if inside else n_cand
    t_w, t_p = float(inset), float(outer_pad)
    return _draw_door_rect(ax, A, B, n_out,
                           t_inner_w=-t_w, t_outer_w=t_p,
                           t_inner_c=-1.20 * t_w, t_outer_c=t_p + 0.20 * t_w,
                           bg_zorder=bg_zorder, fg_zorder=fg_zorder, jamb_lw=jamb_lw)


def _draw_entry_door_at_segment(ax, apt_polygon_pts, door_segment, *,
                                inset, outer_pad, bg_zorder=3, fg_zorder=4,
                                jamb_lw=1.2, door_w=None, corner_margin=None):
    """Punch an entry door at an explicit apartment-boundary segment.

    Used when the benchmark source (e.g. DiffPlanner/RPLAN) already declares
    the entrance as a boundary segment, rather than inferring it from the
    hypergraph's ``outside`` edge.  Returns a cut SPolygon or None.
    """
    from shapely.geometry import Point as SPoint
    try:
        apt = SPolygon(apt_polygon_pts)
        if not apt.is_valid:
            apt = apt.buffer(0)
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

    # Find the containing apartment boundary edge by collinearity + proximity.
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
    half_w = (min(desired_half_w, max_half_w) if max_half_w > 0
              else min(desired_half_w, 0.5 * L))
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
                           t_inner_c=-1.20 * t_w, t_outer_c=t_p + 0.20 * t_w,
                           bg_zorder=bg_zorder, fg_zorder=fg_zorder, jamb_lw=jamb_lw)


# ---------------------------------------------------------------------------
# Hypergraph viz-data helpers (moved from Hypergraph class; not in dev)
# ---------------------------------------------------------------------------

def mesh_entry_door_face_idx(hg, mesh):
    """Return the mesh face index of the entrance room (connected to 'outside').

    Picks the largest leaf with ``outside`` in its ``connected`` list and
    returns its corresponding face index in ``mesh.faceList``.  Returns
    ``None`` if no such leaf or face exists.
    """
    leaves = hg.leaves
    if not leaves:
        return None
    candidates = [
        leaf for leaf in leaves
        if any(str(c).strip().lower() == "outside" for c in (leaf.get("connected") or []))
    ]
    if not candidates:
        return None
    entry_name = max(candidates, key=lambda L: float(L.get("area", 0.0))).get("name", "")
    if not entry_name:
        return None
    n = int(getattr(getattr(mesh, "faceList", None), "Count", 0) or 0)
    for i in range(n):
        if (getattr(mesh.faceList[i], "unique_id", "") or "") == entry_name:
            return i
    return None


def mesh_door_face_pairs(hg, mesh, *, access_graph=None, bounds=None) -> list:
    """Return ``(face_idx_a, face_idx_b)`` pairs for each access-graph edge.

    Used by :meth:`HypergraphPlotter.plot_mesh_on_axis` to draw door symbols
    at the shared wall between adjacent rooms.
    """
    leaves = hg.leaves
    if not leaves:
        return []
    leaf_name_to_id = {leaf["name"]: str(i + 1) for i, leaf in enumerate(leaves)}
    id_to_leaf_name = {v: k for k, v in leaf_name_to_id.items()}

    n = int(getattr(getattr(mesh, "faceList", None), "Count", 0) or 0)
    leaf_to_face_idx: dict = {}
    for i in range(n):
        uid = getattr(mesh.faceList[i], "unique_id", "") or ""
        if uid and uid not in leaf_to_face_idx:
            leaf_to_face_idx[uid] = i

    if access_graph is None:
        if bounds is not None:
            try:
                access_graph = hg.correct_accessgraph(bounds=bounds)
            except Exception:
                access_graph = hg.access_graph or {}
        else:
            access_graph = hg.access_graph or {}
    ag = access_graph or {}
    pairs: list = []
    seen: set = set()
    for edge in ag.get("edges", []) or []:
        if not edge or len(edge) < 2:
            continue
        u, v = edge[0], edge[1]
        if str(u).strip().lower() == "outside" or str(v).strip().lower() == "outside":
            continue
        ln_u = id_to_leaf_name.get(str(u))
        ln_v = id_to_leaf_name.get(str(v))
        if not ln_u or not ln_v:
            continue
        i = leaf_to_face_idx.get(ln_u)
        j = leaf_to_face_idx.get(ln_v)
        if i is None or j is None or i == j:
            continue
        key = (min(i, j), max(i, j))
        if key not in seen:
            seen.add(key)
            pairs.append((i, j))
    return pairs


def adjacency_graph_from_polygons(
    room_polygons: list, *, scale_factor: float = 1.05,
) -> dict:
    """Build a mesh-adjacency graph from a ``room_polygons`` dataset list.

    Each item must have ``id`` (1-based), ``mergeid``, and ``coords``
    (format stored in dataset ``.txt`` files).  Uses the same adjacency rules
    as :func:`adjacency_graph`.

    Returns ``{"nodes": {id: mergeid, ...}, "edges": [[u, v], ...]}``.
    """
    from shapely.affinity import scale as _scale
    from shapely.geometry import Polygon
    from utils.hypergraph import _polygon_adjacency_pairs_by_id

    polys: list = []
    ids: list = []
    nodes: dict = {}
    for rp in room_polygons or []:
        rid = rp.get("id")
        if rid is None:
            continue
        rid_str = str(rid)
        coords = rp.get("coords") or []
        poly = Polygon(coords) if len(coords) >= 3 else Polygon()
        if not poly.is_empty and abs(float(scale_factor) - 1.0) > 1e-9:
            try:
                poly = _scale(poly, xfact=float(scale_factor), yfact=float(scale_factor),
                              origin="centroid")
            except Exception:
                pass
        polys.append(poly)
        ids.append(rid_str)
        nodes[rid_str] = str(rp.get("mergeid", "Unknown"))

    pairs = _polygon_adjacency_pairs_by_id(polys, ids)
    return {
        "nodes": {k: nodes[k] for k in sorted(nodes, key=int)},
        "edges": [[u, v] for (u, v) in sorted(pairs, key=lambda e: (int(e[0]), int(e[1])))],
    }


def adjacency_graph(hg, bounds: dict, *, scale_factor: float = 1.05) -> dict:
    """Pure mesh-adjacency graph between BSP leaves (no edge filtering).

    Returns every pair of leaves whose face polygons share a non-sliver wall,
    without filtering by the predicted access graph or adding bridge edges.
    Complement of :meth:`Hypergraph.correct_accessgraph`.

    Returns ``{"nodes": {id: mergeid, ...}, "edges": [[u, v], ...]}``.
    """
    out = hg._compute_mesh_adjacency(bounds, scale_factor=scale_factor)
    if out is None:
        return {"nodes": {}, "edges": []}
    mesh_adj, id_to_mergeid, node_ids = out
    return {
        "nodes": {nid: id_to_mergeid[nid] for nid in sorted(node_ids, key=int)},
        "edges": [[u, v] for (u, v) in sorted(mesh_adj, key=lambda e: (int(e[0]), int(e[1])))],
    }
