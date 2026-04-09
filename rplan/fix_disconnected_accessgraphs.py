#!/usr/bin/env python3
"""
Fix disconnected access graphs in the RPLAN dataset.

For each sample with a disconnected access graph, attempts to reconnect
components by adding edges to the leaf ``connected`` lists:

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

Writes repaired splits to new JSON files in an output directory.

Usage:
    python rplan/fix_disconnected_accessgraphs.py
    python rplan/fix_disconnected_accessgraphs.py --output-dir dataset/rplan/dataset_hypergraph_fixed
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import networkx as nx
from shapely.affinity import scale as _scale
from shapely.geometry import Polygon

from utils.hypergraph import Hypergraph

logger = logging.getLogger(__name__)

SCALE_FACTOR = 1.05


# ── Helpers ────────────────────────────────────────────────────────────

def _build_physical_adjacency(hg: Hypergraph, bounds: dict):
    """Return edge-adjacent and point-adjacent room pairs.

    Uses the same scaled-polygon heuristic as ``correct_accessgraph``
    for edge adjacency, and raw polygon intersection for point adjacency.
    ``i`` and ``j`` are 0-based face/leaf indices.

    Returns
    -------
    (edge_adj, point_adj, n)
        *edge_adj*: pairs sharing a polygon edge (scaled overlap > 0).
        *point_adj*: pairs sharing only a point (touch but no edge).
        *n*: number of faces.
    """
    mesh = hg.to_mesh(bounds=bounds)
    n = int(getattr(getattr(mesh, "faceList", None), "Count", 0) or 0)
    if n <= 0:
        return set(), set(), n

    raw_polys = []
    scaled_polys = []
    for i in range(n):
        f = mesh.faceList[i]
        pts = f.getPoints()
        corners = [
            (float(pts[j].X), float(pts[j].Y))
            for j in range(int(getattr(pts, "Count", 0) or 0))
        ]
        raw = Polygon(corners) if len(corners) >= 3 else Polygon()
        raw_polys.append(raw)
        scaled_polys.append(
            _scale(raw, xfact=SCALE_FACTOR, yfact=SCALE_FACTOR, origin="centroid")
            if not raw.is_empty else raw
        )

    edge_adj: set[tuple[int, int]] = set()
    point_adj: set[tuple[int, int]] = set()
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
                    elif ix.geom_type == "Point" or (
                        ix.geom_type == "MultiPoint"
                    ):
                        point_adj.add((i, j))
                    else:
                        edge_adj.add((i, j))
            except Exception:
                pass

    return edge_adj, point_adj, n


def _component_of(node_id: str, components: list[set[str]]) -> int:
    """Return the index of the component containing *node_id*, or -1."""
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
    """Find a pair (leaf_i, leaf_j) in different components satisfying *filter_fn*.

    *filter_fn(type_i, type_j)* should return True if the pair is an
    acceptable bridge.  Physical adjacency is always required.
    Returns the first valid (leaf_i, leaf_j) or None.
    """
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


def fix_access_graph(hg_dict: dict, bounds: dict) -> tuple[dict, int]:
    """Attempt to make the access graph connected by adding edges.

    Parameters
    ----------
    hg_dict : dict
        Raw BSP-tree dict (the ``split`` field content, already parsed).
    bounds : dict
        Parsed ``bounds`` dict with ``corners``.

    Returns
    -------
    (fixed_dict, edges_added)
        A deep-copied BSP-tree dict with extra ``connected`` entries,
        and the count of edges that were added.
    """
    fixed = copy.deepcopy(hg_dict)
    hg = Hypergraph(fixed)

    if hg.is_access_graph_connected():
        return fixed, 0

    edge_adj, point_adj, n_faces = _build_physical_adjacency(hg, bounds)
    all_adj = edge_adj | point_adj
    leaves = hg.leaves
    n_leaves = len(leaves)

    if n_faces != n_leaves:
        logger.warning(
            "Face count (%d) != leaf count (%d); skipping fix.", n_faces, n_leaves
        )
        return fixed, 0

    ag = hg.access_graph
    node_types = ag.get("nodes", {})
    edges_added = 0

    max_iters = n_leaves * 2
    for _ in range(max_iters):
        components = hg.access_graph_components()
        if len(components) <= 1:
            break

        bridge = None

        # Strategy 1: living ↔ living (edge-adjacent)
        bridge = _try_bridge(
            components, node_types, edge_adj,
            lambda ti, tj: ti == "living" and tj == "living",
        )

        # Strategy 2: any node ↔ living (edge-adjacent)
        if bridge is None:
            bridge = _try_bridge(
                components, node_types, edge_adj,
                lambda ti, tj: ti == "living" or tj == "living",
            )

        # Strategy 3: any node ↔ living (point-only adjacent)
        if bridge is None:
            bridge = _try_bridge(
                components, node_types, point_adj,
                lambda ti, tj: ti == "living" or tj == "living",
            )

        # Strategy 4: any physically adjacent pair across components
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
        logger.debug(
            "  Added edge: %s(%s) <-> %s(%s)",
            name_a, leaf_a.get("mergeid"), name_b, leaf_b.get("mergeid"),
        )

    return fixed, edges_added


def _fix_worker(item_with_idx):
    """Top-level worker for multiprocessing."""
    idx, item = item_with_idx
    split_str = item.get("split", "")
    if not split_str:
        return idx, item, "skip"

    hg_dict = json.loads(split_str) if isinstance(split_str, str) else split_str
    hg = Hypergraph(hg_dict)

    if hg.is_access_graph_connected():
        return idx, item, "connected"

    item_id = item.get("id", f"item_{idx}")
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


# ── I/O ────────────────────────────────────────────────────────────────

def load_split(dataset_dir: Path, split: str) -> list[dict]:
    """Load a full split JSON, returning the raw item list."""
    filepath = dataset_dir / f"{split}.json"
    if not filepath.exists():
        logger.warning("File not found: %s", filepath)
        return []
    with open(filepath, "r") as f:
        return json.load(f)


def save_split(items: list[dict], output_dir: Path, split: str):
    """Write items to ``output_dir/<split>.json``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{split}.json"
    with open(out_path, "w") as f:
        json.dump(items, f)
    logger.info("  Saved %d items → %s", len(items), out_path)


# ── Main ───────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Fix disconnected access graphs in the RPLAN dataset.",
    )
    parser.add_argument(
        "--dataset-dir", type=str, default="dataset/rplan/dataset_hypergraph",
        help="Directory containing train.json / validation.json / test.json",
    )
    parser.add_argument(
        "--output-dir", type=str, default="dataset/rplan/dataset_hypergraph_fixed",
        help="Directory for repaired JSON files",
    )
    parser.add_argument(
        "--split", type=str, nargs="*",
        default=["train", "validation", "test"],
        choices=["train", "validation", "test"],
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

        work_items = list(enumerate(items))

        if args.workers > 0:
            from multiprocessing import Pool
            from tqdm import tqdm
            with Pool(processes=args.workers) as pool:
                results = list(tqdm(
                    pool.imap(_fix_worker, work_items, chunksize=32),
                    total=len(work_items),
                    desc=f"fix-{split}",
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
