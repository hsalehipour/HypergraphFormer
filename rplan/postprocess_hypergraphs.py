#!/usr/bin/env python3
"""
Post-process exported hypergraphs: fix disconnected access graphs.

For each sample with a disconnected access graph, attempts to
reconnect components by adding edges to the leaf ``connected`` lists:

  1. **Living-Living bridge**: If two components each contain a living
     room and those rooms share a polygon edge, connect them.
  2. **Node-Living bridge (edge)**: Connect any node in one component
     to an edge-adjacent living room in another component.
  3. **Node-Living bridge (point)**: Same as 2, but for rooms that
     share only a point with a living room (no shared edge).
  4. **Any adjacent bridge**: As a last resort, connect any pair of
     physically adjacent rooms across components.

"Physically adjacent" means the two rooms' mesh polygons overlap when
slightly scaled (same heuristic as ``Hypergraph.correct_accessgraph``).

Clipper numerical noise (tiny sliver faces in the RGL mesh) is
filtered at the C# level using a relative area threshold
(``totalArea * 1e-3``) inside ``SubdivideMultiFaceDirectionClipper``.
The mesh-level sliver filter in ``_build_physical_adjacency`` handles
any residual artifacts so face indices align with BSP leaves.

Writes repaired splits to new JSON files in an output directory.

Usage:
    python rplan/postprocess_hypergraphs.py \\
        --dataset-dir dataset/rplan/dataset_hypergraph \\
        --output-dir  dataset/rplan/dataset_hypergraph_fixed \\
        -j 64
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

from shapely.affinity import scale as _scale
from shapely.geometry import Polygon

from utils.hypergraph import Hypergraph

logger = logging.getLogger(__name__)

SCALE_FACTOR = 1.05
SLIVER_MESH_FRAC = 1e-3            # mesh faces < 0.1 % of total mesh area


# ── Physical adjacency (mesh-based) ───────────────────────────────────

def _build_physical_adjacency(hg: Hypergraph, bounds: dict, n_leaves: int):
    """Return edge-adjacent and point-adjacent room pairs.

    Builds face polygons from the RGL mesh and filters out any Clipper
    sliver artifacts (tiny faces) so that the remaining face indices
    correspond 1-to-1 with BSP leaf indices.

    Returns
    -------
    (edge_adj, point_adj, ok)
        *ok* is True when the (filtered) face count matches *n_leaves*.
    """
    mesh = hg.to_mesh(bounds=bounds)
    raw_count = int(getattr(getattr(mesh, "faceList", None), "Count", 0) or 0)
    if raw_count <= 0:
        return set(), set(), False

    face_data = []
    for fi in range(raw_count):
        f = mesh.faceList[fi]
        pts = f.getPoints()
        corners = [
            (float(pts[j].X), float(pts[j].Y))
            for j in range(int(getattr(pts, "Count", 0) or 0))
        ]
        poly = Polygon(corners) if len(corners) >= 3 else Polygon()
        face_data.append((fi, poly, poly.area if not poly.is_empty else 0.0))

    total_mesh_area = sum(a for _, _, a in face_data)
    min_face_area = total_mesh_area * SLIVER_MESH_FRAC
    kept = [(fi, poly) for fi, poly, area in face_data
            if area > min_face_area]

    if len(kept) != n_leaves:
        logger.debug(
            "Face count after sliver filter (%d) != leaves (%d) "
            "(raw mesh had %d faces).",
            len(kept), n_leaves, raw_count,
        )
        if len(kept) > n_leaves:
            face_data_sorted = sorted(face_data, key=lambda x: x[2])
            excess = len(kept) - n_leaves
            drop_indices = {fd[0] for fd in face_data_sorted[:excess]}
            kept = [(fi, poly) for fi, poly, area in face_data
                    if fi not in drop_indices and area > 0]

        if len(kept) != n_leaves:
            return set(), set(), False

    raw_polys = [poly for _, poly in kept]
    scaled_polys = [
        _scale(p, xfact=SCALE_FACTOR, yfact=SCALE_FACTOR, origin="centroid")
        if not p.is_empty else p
        for p in raw_polys
    ]

    edge_adj: set[tuple[int, int]] = set()
    point_adj: set[tuple[int, int]] = set()
    n = len(raw_polys)
    for i in range(n):
        if raw_polys[i].is_empty:
            continue
        for j in range(i + 1, n):
            if raw_polys[j].is_empty:
                continue
            try:
                if scaled_polys[i].intersection(scaled_polys[j]).area > 1e-12:
                    ix = raw_polys[i].intersection(raw_polys[j])
                    if ix.is_empty:
                        point_adj.add((i, j))
                    elif ix.geom_type in ("Point", "MultiPoint"):
                        point_adj.add((i, j))
                    else:
                        edge_adj.add((i, j))
            except Exception:
                pass

    return edge_adj, point_adj, True


# ── Bridge helpers ────────────────────────────────────────────────────

def _component_of(node_id: str, components: list[set[str]]) -> int:
    for ci, comp in enumerate(components):
        if node_id in comp:
            return ci
    return -1


def _try_bridge(
    components: list[set[str]],
    node_types: dict[str, str],
    phys_adj: set[tuple[int, int]],
    filter_fn,
) -> tuple[int, int] | None:
    for i, j in sorted(phys_adj):
        ni, nj = str(i + 1), str(j + 1)
        ci, cj = _component_of(ni, components), _component_of(nj, components)
        if ci == cj or ci < 0 or cj < 0:
            continue
        ti = node_types.get(ni, "unknown").lower()
        tj = node_types.get(nj, "unknown").lower()
        if filter_fn(ti, tj):
            return i, j
    return None


# ── Core fix ──────────────────────────────────────────────────────────

def fix_access_graph(hg_dict: dict, bounds: dict) -> tuple[dict, int]:
    """Attempt to make the access graph connected by adding edges.

    Returns ``(fixed_dict, edges_added)``.
    """
    fixed = copy.deepcopy(hg_dict)
    hg = Hypergraph(fixed)

    if hg.is_access_graph_connected():
        return fixed, 0

    leaves = hg.leaves
    n_leaves = len(leaves)
    edge_adj, point_adj, ok = _build_physical_adjacency(hg, bounds, n_leaves)

    if not ok:
        logger.warning(
            "Could not align mesh faces with %d leaves; skipping fix.",
            n_leaves,
        )
        return fixed, 0

    all_adj = edge_adj | point_adj
    ag = hg.access_graph
    node_types = ag.get("nodes", {})
    edges_added = 0

    max_iters = n_leaves * 2
    for _ in range(max_iters):
        components = hg.access_graph_components()
        if len(components) <= 1:
            break

        bridge = None

        bridge = _try_bridge(
            components, node_types, edge_adj,
            lambda ti, tj: ti == "living" and tj == "living",
        )

        if bridge is None:
            bridge = _try_bridge(
                components, node_types, edge_adj,
                lambda ti, tj: ti == "living" or tj == "living",
            )

        if bridge is None:
            bridge = _try_bridge(
                components, node_types, point_adj,
                lambda ti, tj: ti == "living" or tj == "living",
            )

        if bridge is None:
            bridge = _try_bridge(
                components, node_types, all_adj,
                lambda ti, tj: True,
            )

        if bridge is None:
            logger.debug("No physically adjacent bridge found; giving up.")
            break

        li, lj = bridge
        leaf_a, leaf_b = leaves[li], leaves[lj]
        name_a, name_b = leaf_a["name"], leaf_b["name"]

        if "connected" not in leaf_a:
            leaf_a["connected"] = []
        if "connected" not in leaf_b:
            leaf_b["connected"] = []

        if name_b not in leaf_a["connected"]:
            leaf_a["connected"].append(name_b)
        if name_a not in leaf_b["connected"]:
            leaf_b["connected"].append(name_a)

        edges_added += 1

    return fixed, edges_added


# ── Worker ────────────────────────────────────────────────────────────

def _fix_worker(work_item):
    """Top-level worker for multiprocessing."""
    idx, item = work_item
    split_str = item.get("split", "")
    if not split_str:
        return idx, item, "skip"

    hg_dict = json.loads(split_str) if isinstance(split_str, str) else split_str
    hg = Hypergraph(hg_dict)

    if hg.is_access_graph_connected():
        return idx, item, "connected"

    bounds_str = item.get("bounds", "")
    bounds = (
        json.loads(bounds_str)
        if isinstance(bounds_str, str) and bounds_str
        else bounds_str
    )

    if not bounds or not isinstance(bounds, dict):
        return idx, item, "no_bounds"

    try:
        fixed_dict, n_added = fix_access_graph(hg_dict, bounds)
    except Exception:
        return idx, item, "error"

    hg_after = Hypergraph(fixed_dict)
    now_connected = hg_after.is_access_graph_connected()

    patched = dict(item)
    patched["split"] = json.dumps(fixed_dict)
    status = f"fixed:{n_added}" if now_connected else f"partial:{n_added}"
    return idx, patched, status


# ── I/O ───────────────────────────────────────────────────────────────

def load_split(dataset_dir: Path, split: str) -> list[dict]:
    filepath = dataset_dir / f"{split}.json"
    if not filepath.exists():
        logger.warning("File not found: %s", filepath)
        return []
    with open(filepath, "r") as f:
        return json.load(f)


def save_split(items: list[dict], output_dir: Path, split: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{split}.json"
    with open(out_path, "w") as f:
        json.dump(items, f)
    logger.info("  Saved %d items -> %s", len(items), out_path)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Post-process hypergraphs: fix disconnected access graphs.",
    )
    parser.add_argument(
        "--dataset-dir", type=str, default="dataset/rplan/dataset_hypergraph",
        help="Directory containing train.json / val.json / test.json",
    )
    parser.add_argument(
        "--output-dir", type=str, default="dataset/rplan/dataset_hypergraph_fixed",
        help="Directory for repaired JSON files",
    )
    parser.add_argument(
        "--split", type=str, nargs="*",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Which split(s) to process (default: all three)",
    )
    parser.add_argument(
        "--workers", "-j", type=int, default=0,
        help="Number of parallel workers (0 = single process)",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    grand_total = 0
    grand_disconnected = 0
    grand_fixed = 0
    grand_edges = 0

    for split in args.split:
        logger.info("Processing %s ...", split)
        items = load_split(dataset_dir, split)
        if not items:
            continue

        disconnected = 0
        fixed = 0
        total_edges = 0

        work_items = [(i, item) for i, item in enumerate(items)]

        if args.workers > 0:
            from multiprocessing import Pool
            from tqdm import tqdm
            with Pool(processes=args.workers) as pool:
                results = list(tqdm(
                    pool.imap(_fix_worker, work_items, chunksize=32),
                    total=len(work_items),
                    desc=f"postprocess-{split}",
                ))
        else:
            results = [_fix_worker(wi) for wi in work_items]

        for idx, patched_item, status in results:
            items[idx] = patched_item
            if status.startswith("fixed:"):
                n_added = int(status.split(":")[1])
                disconnected += 1
                fixed += 1
                total_edges += n_added
            elif status.startswith("partial:"):
                n_added = int(status.split(":")[1])
                disconnected += 1
                total_edges += n_added
            elif status == "no_bounds":
                disconnected += 1
            elif status == "error":
                disconnected += 1

        save_split(items, output_dir, split)

        pct_disc = 100 * disconnected / len(items) if items else 0
        pct_fixed = 100 * fixed / disconnected if disconnected else 0
        logger.info(
            "  %s: %d disconnected / %d total (%.1f%%), "
            "%d fixed (%.1f%% of disconnected), %d edges added",
            split, disconnected, len(items), pct_disc,
            fixed, pct_fixed, total_edges,
        )

        grand_total += len(items)
        grand_disconnected += disconnected
        grand_fixed += fixed
        grand_edges += total_edges

    logger.info("")
    logger.info("=" * 60)
    logger.info(
        "Overall: %d disconnected / %d total (%.1f%%)",
        grand_disconnected, grand_total,
        100 * grand_disconnected / grand_total if grand_total else 0,
    )
    logger.info(
        "Fixed:   %d / %d (%.1f%%)   |   Edges added: %d",
        grand_fixed, grand_disconnected,
        100 * grand_fixed / grand_disconnected if grand_disconnected else 0,
        grand_edges,
    )
    remaining = grand_disconnected - grand_fixed
    logger.info(
        "Still disconnected: %d (%.1f%% of total)",
        remaining,
        100 * remaining / grand_total if grand_total else 0,
    )
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
