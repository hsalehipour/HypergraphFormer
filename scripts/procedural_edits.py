#!/usr/bin/env python3
"""Procedural BSP edits: add/delete/resize/swap, ``reorient``."""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.zero_shot_edits import (
    load_bounds,
    load_hypergraph,
    render_comparison,
    render_input_output_pair,
    render_single_mesh,
)
from utils.hypergraph import Hypergraph

logger = logging.getLogger(__name__)

ROOM_ALIASES = {
    "bath": "bathroom",
    "bathroom": "bathroom",
    "bed": "bedroom",
    "bedroom": "bedroom",
    "living": "living",
    "livingroom": "living",  # "living room" -> canonical "livingroom"
    "living_room": "living",
    "kitchen": "kitchen",
    "dining": "dining",
    "foyer": "foyer",
    "entry": "foyer",
    "entrance": "foyer",
    "storage": "storage",
    "extra": "extra",
    "outside": "outside",
}


def canonical_room_name(name: str) -> str:
    cleaned = re.sub(r"[^a-z]", "", str(name).strip().lower())
    return ROOM_ALIASES.get(cleaned, cleaned)


def slugify_command(command: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", command.strip().lower())
    slug = slug.strip("_")
    return slug or "command"


def iter_nodes(node: dict):
    yield node
    for child in node.get("children", []):
        yield from iter_nodes(child)


def collect_matching_leaves(root: dict, room_name: str) -> list[dict]:
    target = canonical_room_name(room_name)
    matches = []
    for node in iter_nodes(root):
        if not node.get("final", False):
            continue
        mid = str(node.get("mergeid", "")).strip()
        if mid:
            m = canonical_room_name(mid) == target
        else:
            m = canonical_room_name(str(node.get("name", ""))) == target
        if m:
            matches.append(node)
    return matches


def collect_matching_leaf_paths(root: dict, room_name: str) -> list[list[dict]]:
    """
    One path per **final** leaf whose room type (``mergeid``) matches the command,
    after :func:`canonical_room_name` (e.g. ``living`` / ``living room`` / ``livingroom`` all
    map to the same target). The command phrase must not be matched to BSP ``name``
    with exact-string equality, which can mix unrelated rooms with the same string.
    If ``mergeid`` is empty, the leaf is matched by canonical ``name`` only.
    """
    target = canonical_room_name(room_name)
    matches: list[list[dict]] = []

    def _walk(node: dict, path: list[dict]) -> None:
        current_path = path + [node]
        if node.get("final", False):
            mid = str(node.get("mergeid", "")).strip()
            if mid:
                m = canonical_room_name(mid) == target
            else:
                m = canonical_room_name(str(node.get("name", ""))) == target
            if m:
                matches.append(current_path)
            return
        for child in node.get("children", []):
            _walk(child, current_path)

    _walk(root, [])
    return matches


def choose_leaf(matches: list[dict]) -> dict:
    return max(matches, key=lambda node: float(node.get("area", 0.0)))


def choose_leaf_path(matches: list[list[dict]]) -> list[dict]:
    return max(matches, key=lambda path: float(path[-1].get("area", 0.0)))


def _leaf_room_neighbor_count(leaf: dict) -> int:
    """Distinct adjacent leaf names on *leaf*'s ``connected`` list (not counting self or *outside*)."""
    self_name = leaf.get("name", "")
    seen: set[str] = set()
    for c in leaf.get("connected", []):
        if not c or c == self_name or str(c).strip().lower() == "outside":
            continue
        seen.add(str(c))
    return len(seen)


def recompute_areas(node: dict) -> float:
    if node.get("final", False):
        node["area"] = float(node.get("area", 0.0))
        return node["area"]
    total = sum(recompute_areas(child) for child in node.get("children", []))
    node["area"] = total
    return total


def rename_subtree(node: dict, new_name: str, mapping: dict[str, str]) -> None:
    old_name = node.get("name", "")
    if old_name:
        mapping[old_name] = new_name
    node["name"] = new_name
    if not node.get("final", False):
        node["mergeid"] = new_name
        children = node.get("children", [])
        if len(children) >= 1:
            rename_subtree(children[0], f"{new_name}L", mapping)
        if len(children) >= 2:
            rename_subtree(children[1], f"{new_name}R", mapping)


def remap_connectivity(root: dict, *, deleted_name: str, rename_map: dict[str, str]) -> None:
    for node in iter_nodes(root):
        if not node.get("final", False):
            continue
        remapped = []
        seen = set()
        for conn in node.get("connected", []):
            if conn == deleted_name:
                continue
            mapped = rename_map.get(conn, conn)
            if mapped == node.get("name", "") or mapped in seen:
                continue
            seen.add(mapped)
            remapped.append(mapped)
        node["connected"] = remapped


def dedupe_names(names: list[str]) -> list[str]:
    result = []
    seen = set()
    for name in names:
        if not name or name in seen:
            continue
        seen.add(name)
        result.append(name)
    return result


def replace_connected_room(root: dict, old_name: str, new_names: list[str]) -> None:
    for node in iter_nodes(root):
        if not node.get("final", False):
            continue
        updated = []
        for conn in node.get("connected", []):
            if conn == old_name:
                updated.extend(new_names)
            else:
                updated.append(conn)
        node["connected"] = dedupe_names([name for name in updated if name != node.get("name", "")])


def _compute_split_areas(old_area: float, frac_new: float) -> tuple[float, float]:
    """Return (new_room_area, kept_area) for *frac_new* share of *old_area* for the new leaf."""
    new_room_area = max(old_area * frac_new, 1.0)
    kept_area = old_area - new_room_area
    if kept_area <= 1.0:
        new_room_area = old_area * max(frac_new, 0.4)
        kept_area = old_area - new_room_area
    if kept_area <= 1.0 or new_room_area <= 1.0:
        raise ValueError("Target leaf does not have enough area to add another room")
    return new_room_area, kept_area


def _normalize_cardinal_angle(rad: float) -> float:
    """Map *rad* to ``{0, π/2, π, 3π/2}`` (axis-aligned BSP)."""
    two_pi = 2 * math.pi
    x = float(rad) % two_pi
    if x < 0:
        x += two_pi
    k = int(round(x / (math.pi / 2))) % 4
    return k * (math.pi / 2)


def parse_reflect_mode_to_code(reflect_mode: str) -> str | None:
    """
    Map CLI-style reflect labels to ``None`` / ``"x"`` / ``"y"``.
    ``None`` means no mirror; ``"x"`` and ``"y"`` are flips in the centroid frame used by
    :func:`transform_bounds_polygon`.
    """
    mode = (reflect_mode or "normal").strip().lower()
    if mode in ("normal", "none", "no_flip", "no-mirror", "nomirror"):
        return None
    if mode in ("mirrored", "mirror", "mirrored_x", "flip_x", "x"):
        return "x"
    if mode in ("mirrored_y", "flip_y", "y"):
        return "y"
    raise ValueError(
        f"Unknown reflect mode {reflect_mode!r}; use 'normal' / 'no_flip', "
        "'x' / 'flip_x', or 'y' / 'flip_y'"
    )


def _apply_bsp_rotation_inplace(node: dict, delta_rad: float) -> None:
    if "angle" in node:
        node["angle"] = _normalize_cardinal_angle(float(node["angle"]) + delta_rad)
    for c in node.get("children", []):
        _apply_bsp_rotation_inplace(c, delta_rad)


def _apply_bsp_reflect_x_inplace(node: dict) -> None:
    if not node.get("final", False):
        ch = node.get("children") or []
        if len(ch) == 2:
            ch[0], ch[1] = ch[1], ch[0]
    if "angle" in node:
        a = float(node.get("angle", 0.0))
        node["angle"] = _normalize_cardinal_angle(math.pi - a)
    for c in node.get("children", []):
        _apply_bsp_reflect_x_inplace(c)


def _apply_bsp_reflect_y_inplace(node: dict) -> None:
    if not node.get("final", False):
        ch = node.get("children") or []
        if len(ch) == 2:
            ch[0], ch[1] = ch[1], ch[0]
    if "angle" in node:
        a = float(node.get("angle", 0.0))
        node["angle"] = _normalize_cardinal_angle(-a)
    for c in node.get("children", []):
        _apply_bsp_reflect_y_inplace(c)


def _apply_bsp_orientation_inplace(root: dict, degrees_ccw: float, reflect: str | None) -> None:
    """Reflect (optional) about x or y, then CCW rotation for all internal split *angle* s."""
    if reflect == "x":
        _apply_bsp_reflect_x_inplace(root)
    elif reflect == "y":
        _apply_bsp_reflect_y_inplace(root)
    _apply_bsp_rotation_inplace(root, math.radians(float(degrees_ccw)))



def reorient_hypergraph_bsp(hypergraph: dict, deg_ccw: float, reflect_mode: str) -> dict:
    """
    Apply a **rigid** orientation of the **BSP** (child swaps + split *angle* updates) in the
    same *reflect-then-rotate* order as :func:`transform_bounds_polygon` uses for the
    footprint polygon.     Returns a **deep copy**; *hypergraph*
    is not modified.

    *reflect_mode* uses the same spellings as :func:`parse_reflect_mode_to_code` (e.g. ``normal``,
    ``x``, ``flip_x``, ``y``). *deg_ccw* is the same CCW angle in degrees (typically ``0, 90, 180, 270``).
    """
    out = copy.deepcopy(hypergraph)
    code = parse_reflect_mode_to_code(reflect_mode)
    _apply_bsp_orientation_inplace(out, float(deg_ccw), code)
    recompute_areas(out)
    return out


def swap_l_r_letters_in_string(s: str) -> str:
    """
    Map every ``L`` to ``R`` and every ``R`` to ``L`` in the string (ASCII, uppercase
    only). Used for structural BSP identifiers, not for natural-language room ``mergeid``\ s.
    """
    return str(s).translate(str.maketrans("LR", "RL"))


def apply_hypergraph_path_lr_name_swap_in_place(root: dict) -> None:
    """
    In-place: on each node, every ``L``/``R`` in ``name`` is swapped; ``connected``\ s
    are remapped the same way; internal ``mergeid`` follows the new ``name`` (structural
    ids), same idea as :func:`repair_bsp_tree_names`. Leaf room-type ``mergeid`` is
    **not** character-swapped. ``angle``\ s and ``children`` order are unchanged. The
    special neighbor ``outside`` in ``connected`` is case-insensitive and left as-is.
    """
    all_nodes = list(iter_nodes(root))
    old_to_new: dict[str, str] = {}
    for n in all_nodes:
        old = str(n.get("name", ""))
        if old not in old_to_new:
            old_to_new[old] = swap_l_r_letters_in_string(old)

    for n in all_nodes:
        new_name = old_to_new[str(n.get("name", ""))]
        n["name"] = new_name
        if not n.get("final", False):
            n["mergeid"] = new_name

    def _remap_ref(x: object) -> str:
        xs = str(x)
        if xs.strip().lower() == "outside":
            return xs
        return old_to_new.get(xs, xs)

    for n in all_nodes:
        if not n.get("final", False):
            continue
        conn = n.get("connected")
        if not conn:
            continue
        n["connected"] = [_remap_ref(c) for c in conn]


def _polygon_signed_area_xy(pts: list[tuple[float, float]]) -> float:
    """Shoelace signed area of a simple polygon (positive → CCW in a y-up system)."""
    n = len(pts)
    if n < 3:
        return 0.0
    t = 0.0
    for i in range(n):
        j = (i + 1) % n
        t += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return 0.5 * t


def transform_bounds_polygon(
    bounds: dict,
    reflect_code: str | None,
    deg_ccw: float,
) -> dict:
    """
    Apply a **2D** transform to *only* ``bounds['corners']``: translate to centroid,
    optional reflect in x and/or y (in that frame), then CCW rotation by *deg_ccw*,
    then translate back. The hypergraph is **not** modified.

    A reflection in x or y **reverses** polygon orientation (CCW ↔ CW) while
    **preserving the same vertex list order**; that can confuse meshers that assume
    a fixed exterior winding. If the **signed** area flips sign vs the input, the
    corner list is **reversed** so the loop keeps the same handedness as the
    original bounds.

    ``reflect_code`` is ``None`` (no flip), ``\"x\"`` (negate x about centroid), or
    ``\"y\"`` (negate y about centroid).
    """
    if not (isinstance(bounds, dict) and bounds.get("corners")):
        raise ValueError("transform_bounds_polygon: bounds with non-empty 'corners' is required")
    corners = bounds["corners"]
    pts = [(float(c["X"]), float(c["Y"])) for c in corners]
    n = len(pts)
    if n == 0:
        raise ValueError("transform_bounds_polygon: empty corners")
    s_in = _polygon_signed_area_xy(pts)
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n
    th = math.radians(float(deg_ccw))
    cos_t, sin_t = math.cos(th), math.sin(th)
    out_pts: list[tuple[float, float]] = []
    for x, y in pts:
        x -= cx
        y -= cy
        if reflect_code == "x":
            x = -x
        elif reflect_code == "y":
            y = -y
        xr = x * cos_t - y * sin_t
        yr = x * sin_t + y * cos_t
        out_pts.append((xr + cx, yr + cy))
    s_out = _polygon_signed_area_xy(out_pts)
    eps = 1e-9
    if abs(s_in) > eps and abs(s_out) > eps and s_in * s_out < 0:
        out_pts = list(reversed(out_pts))
    out = copy.deepcopy(bounds)
    out["corners"] = [{"X": px, "Y": py} for px, py in out_pts]
    return out



def _split_leaf_for_add_impl(
    root: dict,
    path: list[dict],
    anchor_room: str,
    new_room: str,
    *,
    new_room_mergeid: str | None,
    new_room_on_right: bool,
    split_angle: float,
    new_room_area: float,
    kept_area: float,
) -> tuple[str, str, float, float]:
    """Splice a binary split at *path*[-1]; anchor half keeps *old_mergeid*, other half is *new_room*."""
    leaf = path[-1]
    old_name = leaf.get("name", "")
    old_mergeid = leaf.get("mergeid", canonical_room_name(anchor_room))
    inherited_connections = list(leaf.get("connected", []))
    old_area = float(leaf.get("area", 0.0))

    left_name = f"{old_name}L"
    right_name = f"{old_name}R"
    new_mergeid = (
        str(new_room_mergeid).strip()
        if new_room_mergeid is not None
        else canonical_room_name(new_room)
    )
    ang = float(split_angle)
    if new_room_on_right:
        child_l = {
            "name": left_name,
            "area": kept_area,
            "angle": ang,
            "mergeid": old_mergeid,
            "final": True,
            "children": [],
            "connected": dedupe_names([right_name, *inherited_connections]),
        }
        child_r = {
            "name": right_name,
            "area": new_room_area,
            "angle": ang,
            "mergeid": new_mergeid,
            "final": True,
            "children": [],
            "connected": dedupe_names([left_name]),
        }
        anchor_leaf_name = left_name
        new_leaf_name = right_name
    else:
        child_l = {
            "name": left_name,
            "area": new_room_area,
            "angle": ang,
            "mergeid": new_mergeid,
            "final": True,
            "children": [],
            "connected": dedupe_names([right_name]),
        }
        child_r = {
            "name": right_name,
            "area": kept_area,
            "angle": ang,
            "mergeid": old_mergeid,
            "final": True,
            "children": [],
            "connected": dedupe_names([left_name, *inherited_connections]),
        }
        anchor_leaf_name = right_name
        new_leaf_name = left_name

    split_node = {
        "name": old_name,
        "area": old_area,
        "angle": ang,
        "mergeid": old_name,
        "final": False,
        "children": [child_l, child_r],
        "connected": [],
    }

    if len(path) == 1:
        root.clear()
        root.update(split_node)
    else:
        parent = path[-2]
        children = parent.get("children", [])
        for idx, child in enumerate(children):
            if child.get("name") == old_name:
                children[idx] = split_node
                break
        else:
            raise ValueError("Could not locate target room in parent children")

    replace_connected_room(root, old_name, [anchor_leaf_name])
    recompute_areas(root)
    return old_name, new_leaf_name, kept_area, new_room_area


def split_leaf_for_add(
    root: dict,
    anchor_room: str,
    new_room: str,
    *,
    new_room_mergeid: str | None = None,
    bounds: dict | None = None,
) -> tuple[str, str, float, float, str]:
    """
    Split the first matching anchor leaf in preorder (see
    :func:`collect_matching_leaf_paths`) to add *new_room*, with a default area fraction
    and the leaf's own split axis.

    *bounds* is accepted for call compatibility; mesh-scored add exploration is not used.
    """
    _ = bounds
    paths = collect_matching_leaf_paths(root, anchor_room)
    if not paths:
        raise ValueError(f"No leaf found with room identifier {anchor_room!r}")

    path = paths[0]
    leaf = path[-1]
    old_area = float(leaf.get("area", 0.0))
    if old_area <= 1.0:
        raise ValueError(f"Target room {leaf.get('name', '')!r} is too small to split")

    old_name = leaf.get("name", "")
    leaf_angle = float(leaf.get("angle", 0.0))
    new_room_area, kept_area = _compute_split_areas(old_area, 0.35)
    _on, _nl, kept_area, new_room_area = _split_leaf_for_add_impl(
        root,
        path,
        anchor_room,
        new_room,
        new_room_mergeid=new_room_mergeid,
        new_room_on_right=True,
        split_angle=leaf_angle,
        new_room_area=new_room_area,
        kept_area=kept_area,
    )
    return _on, _nl, kept_area, new_room_area, ""


def delete_room(root: dict, room_name: str) -> tuple[str, str]:
    paths = collect_matching_leaf_paths(root, room_name)
    if not paths:
        raise ValueError(f"No leaf found with room identifier {room_name!r}")

    eligible = [p for p in paths if len(p) >= 2]
    if not eligible:
        raise ValueError("Cannot delete the root node")

    path = min(
        eligible,
        key=lambda p: (_leaf_room_neighbor_count(p[-1]), p[-1].get("name", "")),
    )
    return delete_room_at_path(root, path)


def _resolve_bsp_path_by_names(root: dict, path: list[dict]) -> list[dict]:
    """
    Re-walk *root* with the same ``name`` list as *path* so all dict references are
    nodes in *path*.  Paths from :func:`collect_matching_leaf_paths` on one graph
    must not be passed to :func:`delete_room_at_path` together with
    :func:`copy.deepcopy` of that graph: parent/grandparent would still point at the
    original tree, the splice can update the source while *root* (the copy) stays
    unchanged and connectivity is remapped on the copy.
    """
    names: list[str] = [n.get("name", "") for n in path]
    if not names or names[0] != root.get("name", ""):
        raise ValueError("delete path must start at the root of the tree passed in")
    out: list[dict] = [root]
    node: dict = root
    for nname in names[1:]:
        nxt: dict | None = None
        for ch in node.get("children", []):
            if ch.get("name") == nname:
                nxt = ch
                break
        if nxt is None:
            raise ValueError(
                f"resolve path: no child {nname!r} under {node.get('name')!r}"
            )
        out.append(nxt)
        node = nxt
    return out


def delete_room_at_path(root: dict, path: list[dict]) -> tuple[str, str]:
    """Delete a specific final leaf given a path from root (same as :func:`delete_room` for that path)."""
    if not path or len(path) < 2:
        raise ValueError("delete_room_at_path: need path ending at a final node with a parent")
    path = _resolve_bsp_path_by_names(root, path)
    if not path[-1].get("final", False):
        raise ValueError("delete_room_at_path: need path ending at a final node with a parent")
    leaf = path[-1]
    parent = path[-2]
    grandparent = path[-3] if len(path) >= 3 else None
    deleted_name = leaf.get("name", "")

    children = parent.get("children", [])
    if len(children) != 2:
        raise ValueError(f"Parent {parent.get('name', '')!r} is not binary")

    sibling = copy.deepcopy(children[1] if children[0].get("name") == deleted_name else children[0])
    rename_map: dict[str, str] = {}

    if grandparent is None:
        rename_subtree(sibling, "root", rename_map)
        root.clear()
        root.update(sibling)
    else:
        promoted_name = parent.get("name", "")
        rename_subtree(sibling, promoted_name, rename_map)
        gp_children = grandparent.get("children", [])
        for idx, child in enumerate(gp_children):
            if child.get("name") == parent.get("name"):
                gp_children[idx] = sibling
                break
        else:
            raise ValueError("Could not locate parent in grandparent children")

    remap_connectivity(root, deleted_name=deleted_name, rename_map=rename_map)
    recompute_areas(root)
    return deleted_name, sibling.get("name", "")


def eligible_delete_paths(root: dict, room_name: str) -> list[list[dict]]:
    """Paths to final leaves for *room_name* that are valid delete targets (non-root)."""
    paths = collect_matching_leaf_paths(root, room_name)
    eligible = [p for p in paths if len(p) >= 2]
    if not paths:
        raise ValueError(f"No leaf found with room identifier {room_name!r}")
    if not eligible:
        raise ValueError("Cannot delete the root node")
    return eligible


def split_leaf_for_add_on_path(
    root: dict,
    path: list[dict],
    new_room: str,
    *,
    new_room_mergeid: str | None = None,
    new_room_on_right: bool = True,
    split_angle: float = 0.0,
    area_fraction: float = 0.35,
) -> str:
    """
    Binary-split a **specific** final leaf (after :func:`_resolve_bsp_path_by_names`), placing
    the new room on the left or right. *split_angle* is typically a multiple of ``π/2`` for
    axis-aligned BSP cuts. Mutates *root* in place.
    """
    path = _resolve_bsp_path_by_names(root, path)
    leaf = path[-1]
    if not leaf.get("final", False):
        raise ValueError("split_leaf_for_add_on_path: target must be a final room")
    old_area = float(leaf.get("area", 0.0))
    if old_area <= 1.0:
        raise ValueError("split target leaf is too small")
    new_room_area, kept_area = _compute_split_areas(old_area, float(area_fraction))
    anchor_room = str(leaf.get("mergeid", "")).strip() or new_room
    _split_leaf_for_add_impl(
        root,
        path,
        anchor_room,
        new_room,
        new_room_mergeid=new_room_mergeid,
        new_room_on_right=bool(new_room_on_right),
        split_angle=float(split_angle),
        new_room_area=new_room_area,
        kept_area=kept_area,
    )
    return (
        f"Y={leaf.get('name', '')!r} new_on_R={new_room_on_right} "
        f"angle={float(split_angle):.4f} frac_new≈{new_room_area / old_area:.2f}"
    )


def resize_room_area(root: dict, room_name: str, factor: float) -> tuple[str, float, float]:
    matches = collect_matching_leaves(root, room_name)
    if not matches:
        raise ValueError(f"No leaf found with room type {room_name!r}")

    leaf = choose_leaf(matches)
    old_area = float(leaf.get("area", 0.0))
    if old_area <= 0:
        raise ValueError(
            f"Target room {leaf.get('name', '')!r} has non-positive area {old_area}"
        )
    if factor <= 0.0:
        raise ValueError("Resize factor must be greater than 0")

    leaf["area"] = old_area * factor
    recompute_areas(root)
    return leaf.get("name", ""), old_area, float(leaf["area"])


def resize_leaf_to_target_area(root: dict, leaf: dict, target_area: float) -> tuple[float, float, float]:
    old_area = float(leaf.get("area", 0.0))
    if old_area <= 0.0:
        raise ValueError(f"Target room {leaf.get('name', '')!r} has non-positive area {old_area}")
    if target_area <= 0.0:
        raise ValueError(f"Target area must be positive, got {target_area}")

    factor = target_area / old_area
    leaf["area"] = old_area * factor
    recompute_areas(root)
    return old_area, float(leaf["area"]), factor


def swap_rooms(root: dict, room1: str, room2: str) -> tuple[str, str, float, float]:
    matches1 = collect_matching_leaf_paths(root, room1)
    if not matches1:
        raise ValueError(f"No leaf found with room identifier {room1!r}")
    matches2 = collect_matching_leaf_paths(root, room2)
    if not matches2:
        raise ValueError(f"No leaf found with room identifier {room2!r}")

    leaf1 = choose_leaf_path(matches1)[-1]
    leaf2 = choose_leaf_path(matches2)[-1]
    if leaf1.get("name") == leaf2.get("name"):
        raise ValueError("Swap targets resolved to the same room node")

    area1 = float(leaf1.get("area", 0.0))
    area2 = float(leaf2.get("area", 0.0))
    if area1 <= 0.0 or area2 <= 0.0:
        raise ValueError("Swap targets must both have positive area")

    # Preserve each leaf's original mergeid string verbatim (e.g. RPLAN uses
    # 'bed' / 'bath' lowercase, which the renderer's ROOM_COLORS map keys
    # against). Re-canonicalising here would rewrite 'bed' -> 'bedroom',
    # which falls outside the lowercase color palette and renders as a gray
    # 'bedroom' instead of the expected purple 'bed'. Only fall back to the
    # canonical form if the source leaf has no mergeid at all.
    mergeid1 = leaf1.get("mergeid") or canonical_room_name(room1)
    mergeid2 = leaf2.get("mergeid") or canonical_room_name(room2)
    leaf1["mergeid"] = mergeid2
    leaf2["mergeid"] = mergeid1

    _, new_area1, factor1 = resize_leaf_to_target_area(root, leaf1, area2)
    _, new_area2, factor2 = resize_leaf_to_target_area(root, leaf2, area1)
    return leaf1.get("name", ""), leaf2.get("name", ""), new_area1, new_area2, factor1, factor2




def apply_command(
    hypergraph: dict,
    command: str,
    increase_factor: float,
    *,
    new_room_mergeid: str | None = None,
    bounds: dict | None = None,
) -> tuple[dict, str]:
    command = command.strip()
    match = re.fullmatch(r"(increase|reduce)\s+([a-zA-Z_ ]+)", command, flags=re.IGNORECASE)
    if match:
        action = match.group(1).strip().lower()
        room_name = match.group(2).strip()
        edited = copy.deepcopy(hypergraph)

        if action == "increase":
            factor = increase_factor
        else:
            if increase_factor <= 1.0:
                raise ValueError("--increase-factor must be greater than 1 for reduce mode")
            factor = 1.0 / increase_factor

        leaf_name, old_area, new_area = resize_room_area(edited, room_name, factor)
        verb = "Increased" if action == "increase" else "Reduced"
        summary = (
            f"{verb} {room_name} ({leaf_name}) area from {old_area:.2f} "
            f"to {new_area:.2f} using factor {factor:.2f}"
        )
        return edited, summary

    swap_match = re.fullmatch(r"swap\s+([a-zA-Z_ ]+?)\s+and\s+([a-zA-Z_ ]+)", command, flags=re.IGNORECASE)
    if swap_match:
        room1 = swap_match.group(1).strip()
        room2 = swap_match.group(2).strip()
        edited = copy.deepcopy(hypergraph)
        leaf1_name, leaf2_name, new_area1, new_area2, factor1, factor2 = swap_rooms(edited, room1, room2)
        summary = (
            f"Swapped {room1} and {room2}; relabeled {leaf1_name} and {leaf2_name}, then resized them "
            f"with factors {factor1:.2f} and {factor2:.2f} to areas {new_area1:.2f} and {new_area2:.2f}"
        )
        return edited, summary

    add_match = re.fullmatch(r"add\s+([a-zA-Z_ ]+?)\s+by\s+([a-zA-Z_ ]+)", command, flags=re.IGNORECASE)
    if add_match:
        new_room = add_match.group(1).strip()
        anchor_room = add_match.group(2).strip()
        edited = copy.deepcopy(hypergraph)
        anchor_leaf_name, new_leaf_name, kept_area, new_area, add_detail = split_leaf_for_add(
            edited,
            anchor_room,
            new_room,
            new_room_mergeid=new_room_mergeid,
            bounds=bounds,
        )
        mid = (
            str(new_room_mergeid).strip()
            if new_room_mergeid is not None
            else canonical_room_name(new_room)
        )
        summary = (
            f"Added {new_room} by {anchor_room} by splitting {anchor_leaf_name} into "
            f"{anchor_room} ({kept_area:.2f}) and {new_room} ({new_area:.2f}); "
            f"new leaf mergeid={mid}; new room node is {new_leaf_name}"
        )
        if add_detail:
            summary += f";{add_detail}"
        return edited, summary

    delete_match = re.fullmatch(r"delete\s+([a-zA-Z_ ]+)", command, flags=re.IGNORECASE)
    if delete_match:
        room_name = delete_match.group(1).strip()
        edited = copy.deepcopy(hypergraph)
        deleted_name, promoted_name = delete_room(edited, room_name)
        summary = (
            f"Deleted room {room_name} ({deleted_name}) and promoted sibling subtree "
            f"to occupy the partition ({promoted_name})"
        )
        return edited, summary

    reorient_match = re.fullmatch(
        r"reorient\s+([+-]?\d+(?:\.\d+)?)\s+([a-zA-Z0-9_ -]+)\s*",
        command,
        flags=re.IGNORECASE,
    )
    if reorient_match:
        deg = float(reorient_match.group(1).strip())
        mode = reorient_match.group(2).strip()
        try:
            edited = reorient_hypergraph_bsp(hypergraph, deg, mode)
        except ValueError as e:
            raise ValueError(f"reorient: {e}") from e
        summary = f"Reoriented BSP: CCW {deg}°, reflect mode {mode!r} (reflect, then rotate; same order as bounds transform_bounds_polygon)"
        return edited, summary

    raise ValueError(
        f"Unsupported command {command!r}. Supported: "
        f"'increase <room>', 'reduce <room>', 'delete <room>', 'add <room2> by <room1>', "
        f"'swap <room1> and <room2>', "
        f"'reorient <deg_ccw> <reflect_mode>' (e.g. reorient 90 normal, reorient 0 flip_x)"
    )


def _run_one_cli(
    sample_id: str,
    input_hg: dict,
    bounds: dict,
    command: str,
    increase_factor: float,
    tag: str,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_hg, summary = apply_command(
        input_hg, command, increase_factor, bounds=bounds
    )
    is_valid, _, failed, _ = Hypergraph(output_hg).validate(quick=True)
    if not is_valid:
        raise ValueError("Edited hypergraph failed validation: " + "; ".join(failed))
    # 'rotate <deg>' keeps the footprint identical and only rotates BSP cuts; pass
    # the original bounds straight through. Renderers still accept output_bounds
    # so future commands that want to transform the footprint can plumb it.
    output_bounds = bounds
    command_slug = slugify_command(command)
    run_dir = output_dir / f"{command_slug}_{sample_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{sample_id}_{command_slug}_{tag}"
    json_path = run_dir / f"{stem}.json"
    png_path = run_dir / f"{stem}.png"
    summary_path = run_dir / f"{stem}_summary.txt"
    with open(json_path, "w") as f:
        json.dump(output_hg, f, indent=2)
    render_comparison(
        sample_id, command, bounds, input_hg, output_hg, png_path,
        output_bounds=output_bounds,
    )

    input_png = run_dir / f"{stem}_input.png"
    output_png = run_dir / f"{stem}_output.png"
    pair_png = run_dir / f"{stem}_input_output.png"
    try:
        render_single_mesh(Hypergraph(input_hg), bounds, input_png)
        render_single_mesh(Hypergraph(output_hg), output_bounds, output_png)
        render_input_output_pair(
            input_hg, output_hg, bounds, pair_png,
            output_bounds=output_bounds,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Standalone mesh PNGs failed for %s: %s", sample_id, exc)

    summary_path.write_text(summary + "\n")
    return json_path, png_path, summary_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a single procedural command on one or all samples."
    )
    parser.add_argument("--hypergraphs-file", type=Path, required=True)
    parser.add_argument(
        "--sample-id",
        default=None,
        help="One id from the hypergraphs JSON (omit if --all-samples).",
    )
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Run --command for every key in --hypergraphs-file; write outputs per id under --output-dir.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("/raid/home/nikita/HypergraphFormer/dataset/rplan/dataset_hypergraph_la0_fixed/test"),
    )
    parser.add_argument("--command", required=True)
    parser.add_argument("--tag", default="procedural")
    parser.add_argument("--increase-factor", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / ".procedural_edit_outputs")
    args = parser.parse_args()

    if args.all_samples:
        with open(args.hypergraphs_file, encoding="utf-8") as f:
            all_hg: dict = json.load(f)
        ids = sorted(all_hg.keys(), key=lambda x: (len(str(x)), str(x)))
        n_ok = 0
        n_err = 0
        err_path = args.output_dir / f"_batch_errors_{slugify_command(args.command)}_{args.tag}.txt"
        lines: list[str] = []
        for sid in ids:
            sid_s = str(sid)
            try:
                input_hg = all_hg[sid]
                if not isinstance(input_hg, dict):
                    raise TypeError("hypergraph must be a dict")
                bounds = load_bounds(args.test_dir, sid_s)
                jp, pp, sp = _run_one_cli(
                    sid_s,
                    input_hg,
                    bounds,
                    args.command,
                    args.increase_factor,
                    args.tag,
                    args.output_dir,
                )
                n_ok += 1
                print(jp)
                print(pp)
                print(sp)
            except Exception as exc:  # noqa: BLE001
                n_err += 1
                lines.append(f"{sid_s}\t{exc}\n")
                logger.warning("Sample %s failed: %s", sid_s, exc)
        if lines:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            err_path.write_text("".join(lines), encoding="utf-8")
        logger.info("Batch done: %d ok, %d errors (see %s)", n_ok, n_err, err_path if lines else "—")
        return 0 if n_err == 0 else 1

    if not args.sample_id:
        parser.error("Provide --sample-id or use --all-samples")

    input_hg = load_hypergraph(args.hypergraphs_file, args.sample_id)
    bounds = load_bounds(args.test_dir, args.sample_id)
    json_path, png_path, summary_path = _run_one_cli(
        args.sample_id,
        input_hg,
        bounds,
        args.command,
        args.increase_factor,
        args.tag,
        args.output_dir,
    )
    print(json_path)
    print(png_path)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
