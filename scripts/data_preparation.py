#!/usr/bin/env python3
"""
Data preparation for hypergraph model training and evaluation.

Modes:
    train        -- Convert pre-split hypergraph JSON to instruction JSONL
    eval         -- Export per-sample .txt eval files from a JSON split

Usage:
    # Convert to instruction JSONL (SFT)
    python scripts/data_preparation.py --mode train \\
        --dataset-dir dataset/hypergraph

    # Export eval .txt from dataset_access (room polygons + door adjacencies)
    python scripts/data_preparation.py --mode eval --data-format access \\
        --input-file dataset/rplan/dataset_access/data_test.json \\
        --output-dir dataset/rplan/eval_test

    # Export eval .txt from hypergraph JSON (BSP tree)
    python scripts/data_preparation.py --mode eval --data-format hypergraph \\
        --input-file dataset/hypergraph/test.json \\
        --output-dir dataset/hypergraph/test

    # Export eval .txt from hypergraph JSON with aggregated access graphs
    python scripts/data_preparation.py --mode eval --data-format hypergraph \\
        --aggregate \\
        --input-file dataset/hypergraph/test.json \\
        --output-dir dataset/hypergraph/test_aggregated
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from utils.hypergraph import Hypergraph

logger = logging.getLogger(__name__)


# ===================================================================
# Shared helper
# ===================================================================

def load_json_split(file_path):
    """Load a hypergraph dataset split from a JSON file.

    Each item is expected to have a ``split`` field containing the
    hypergraph (as a JSON string or dict).  The access graph is
    extracted automatically via :class:`Hypergraph`.

    Returns a list of tuples::

        (item_id, accessgraph, hypergraph)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return []

    with open(file_path, "r") as f:
        data = json.load(f)

    samples = []
    skipped = 0

    for i, item in enumerate(data):
        try:
            hypergraph_str = item.get("split", "")
            if not hypergraph_str:
                logger.warning(f"Skipping item {i}: missing 'split' field")
                skipped += 1
                continue

            hypergraph = (
                json.loads(hypergraph_str)
                if isinstance(hypergraph_str, str)
                else hypergraph_str
            )

            accessgraph = Hypergraph(hypergraph).access_graph

            if not accessgraph or not accessgraph.get("nodes"):
                logger.warning(f"Skipping item {i}: could not extract access graph")
                skipped += 1
                continue

            item_id = item.get("id", f"item_{i}")
            samples.append((item_id, accessgraph, hypergraph))

        except Exception as e:
            logger.warning(f"Skipping item {i}: {e}")
            skipped += 1

    logger.info(
        f"Loaded {len(samples)} samples from {file_path.name} ({skipped} skipped)"
    )
    return samples


CATEGORY_TO_MERGEID = {
    0: "living",
    1: "bed",
    2: "kitchen",
    3: "bath",
    4: "balcony",
    5: "storage",
}


# ===================================================================
# Access-format eval helpers
# ===================================================================

def access_sample_to_eval_entry(sample: dict) -> dict | None:
    """Convert one dataset_access item into an eval ``.txt`` dict.

    Returns ``None`` when the sample has no valid rooms or access adjacencies.

    The ``bounds`` field is built by reusing the same helpers that
    :mod:`rplan.rplan_to_hypergraph` uses for hypergraph-mode exports
    (``compute_boundary`` -> ``_boundary_to_bounds_json``), so the JSON
    string is byte-identical to what hypergraph mode would emit for the
    same RPLAN sample.
    """
    from rplan.rplan_to_hypergraph import (
        _boundary_to_bounds_json,
        compute_boundary,
        sample_to_rooms,
    )

    rooms_raw = sample.get("rooms", [])
    adj_list = sample.get("access_adjacencies")
    if not rooms_raw or adj_list is None:
        return None

    valid_rooms = sample_to_rooms(sample)
    if not valid_rooms:
        return None

    valid_ids = set(valid_rooms.keys())

    nodes: dict[str, str] = {}
    room_polygons: list[dict] = []
    for room in rooms_raw:
        rid_0 = room["id"]
        if rid_0 not in valid_ids:
            continue
        rid_1 = str(rid_0 + 1)
        category = room.get("category", -1)
        mergeid = CATEGORY_TO_MERGEID.get(category, f"unknown_{category}")
        nodes[rid_1] = mergeid
        room_polygons.append({
            "id": rid_0 + 1,
            "category": category,
            "mergeid": mergeid,
            "coords": room["r_boundary"],
        })

    if not nodes:
        return None

    edges: list[list[str]] = []
    for pair in adj_list:
        a, b = str(pair[0] + 1), str(pair[1] + 1)
        if a in nodes and b in nodes:
            edges.append([a, b])

    boundary_poly = compute_boundary(valid_rooms)
    if boundary_poly is None or boundary_poly.is_empty:
        return None
    bounds_json = _boundary_to_bounds_json(boundary_poly)

    return {
        "accessgraph": {"nodes": nodes, "edges": edges},
        "room_polygons": room_polygons,
        "bounds": bounds_json,
    }


def export_eval_access(args) -> int:
    """Export per-sample .txt eval files from a dataset_access JSON."""
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir or "dataset/rplan/eval_test")

    logger.info("=" * 80)
    logger.info("Export Eval-Format .txt Files  (data-format: access)")
    logger.info("=" * 80)
    logger.info(f"Input file:       {input_file}")
    logger.info(f"Output directory:  {output_dir}")
    logger.info("=" * 80)

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r") as f:
        raw_data = json.load(f)

    successful = 0
    skipped = 0

    for i, sample in enumerate(raw_data):
        sample_name = str(sample.get("name", f"item_{i}"))
        try:
            entry = access_sample_to_eval_entry(sample)
            if entry is None:
                logger.warning(f"Skipping {sample_name}: invalid rooms or access")
                skipped += 1
                continue

            with open(output_dir / f"{sample_name}.txt", "w") as f:
                json.dump(entry, f, indent=2)

            successful += 1

            if (i + 1) % 500 == 0:
                logger.info(f"  Processed {i + 1}/{len(raw_data)} samples...")

        except Exception as e:
            logger.warning(f"Error processing {sample_name}: {e}")
            skipped += 1

    logger.info(f"\nDone! {successful} files written, {skipped} skipped.")
    logger.info(f"Output directory: {output_dir}")
    return 0


# ===================================================================
# convert_to_jsonl  (--mode train)
# ===================================================================

def convert_to_jsonl(args) -> int:
    """Convert pre-split hypergraph JSON files to instruction JSONL."""
    from utils.data_loading import save_training_data

    dataset_dir = Path(args.dataset_dir)
    output_dir = dataset_dir / "sft"

    logger.info("=" * 80)
    logger.info("Hypergraph JSON -> Instruction JSONL")
    logger.info("=" * 80)
    logger.info(f"Dataset directory: {dataset_dir}")
    logger.info(f"Output directory:  {output_dir}")
    logger.info("=" * 80)

    splits = ["train", "validation", "test"]
    total = 0

    for split in splits:
        logger.info(f"\nProcessing {split} split...")
        samples = load_json_split(dataset_dir / f"{split}.json")

        if not samples:
            logger.warning(f"No samples found for {split}")
            continue

        save_training_data(samples, output_dir / f"{split}.jsonl", split)
        total += len(samples)

    logger.info(f"\nDone! {total} samples across {len(splits)} splits.")
    for split in splits:
        p = output_dir / f"{split}.jsonl"
        if p.exists():
            logger.info(f"  {p}")
    return 0


# ===================================================================
# Access-graph aggregation
# ===================================================================

def aggregate_accessgraph(ag: dict, hg_obj: "Hypergraph", bounds: dict,
                          *, scale_factor: float = 1.05) -> dict:
    """Merge access-graph nodes that share room type, spatial adjacency, and access.

    Two nodes are candidates for merging when ALL three hold:
      1. Same room type (``nodes[a] == nodes[b]``)
      2. Spatially adjacent (mesh polygons intersect after slight scaling)
      3. Connected by an edge in the original access graph

    Connected components of the resulting "mergeable" graph are collapsed
    into single representative nodes.  The returned graph preserves all
    cross-component edges (deduped, no self-loops).
    """
    import networkx as nx
    from shapely.geometry import Polygon
    from shapely.affinity import scale as _scale

    nodes = ag.get("nodes", {})
    edges = ag.get("edges", [])

    if len(nodes) <= 1:
        return ag

    mesh = hg_obj.to_mesh(bounds=bounds)
    n_faces = int(getattr(getattr(mesh, "faceList", None), "Count", 0) or 0)
    if n_faces <= 0:
        return ag

    leaves = hg_obj.leaves
    leaf_name_to_id = {leaf["name"]: str(i + 1) for i, leaf in enumerate(leaves)}

    face_to_node_id: dict[int, str] = {}
    polys: dict[str, Polygon] = {}
    for i in range(n_faces):
        face = mesh.faceList[i]
        uid = getattr(face, "unique_id", "")
        node_id = leaf_name_to_id.get(uid)
        if node_id is None or node_id not in nodes:
            continue
        face_to_node_id[i] = node_id
        pts = face.getPoints()
        corners = [
            (float(pts[j].X), float(pts[j].Y))
            for j in range(int(getattr(pts, "Count", 0) or 0))
        ]
        poly = Polygon(corners) if len(corners) >= 3 else Polygon()
        if not poly.is_empty:
            poly = _scale(poly, xfact=scale_factor, yfact=scale_factor, origin="centroid")
        polys[node_id] = poly

    spatial_adj: set[tuple[str, str]] = set()
    node_ids = list(polys.keys())
    for i_idx in range(len(node_ids)):
        nid_a = node_ids[i_idx]
        if polys[nid_a].is_empty:
            continue
        for j_idx in range(i_idx + 1, len(node_ids)):
            nid_b = node_ids[j_idx]
            if polys[nid_b].is_empty:
                continue
            try:
                if polys[nid_a].intersection(polys[nid_b]).area > 1e-12:
                    spatial_adj.add(tuple(sorted([nid_a, nid_b], key=int)))
            except Exception:
                pass

    access_edges: set[tuple[str, str]] = set()
    for e in edges:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            access_edges.add(tuple(sorted([str(e[0]), str(e[1])], key=int)))

    merge_graph = nx.Graph()
    merge_graph.add_nodes_from(nodes.keys())
    for a, b in access_edges:
        if a in nodes and b in nodes:
            same_type = (
                str(nodes[a]).strip().lower() == str(nodes[b]).strip().lower()
            )
            spatially_near = (a, b) in spatial_adj or (b, a) in spatial_adj
            if same_type and spatially_near:
                merge_graph.add_edge(a, b)

    components = list(nx.connected_components(merge_graph))

    node_to_rep: dict[str, str] = {}
    for comp in components:
        rep = min(comp, key=int)
        for nid in comp:
            node_to_rep[nid] = rep

    new_nodes: dict[str, str] = {}
    for comp in components:
        rep = min(comp, key=int)
        new_nodes[rep] = nodes[rep]

    new_edges_set: set[tuple[str, str]] = set()
    for e in edges:
        if not isinstance(e, (list, tuple)) or len(e) != 2:
            continue
        u, v = node_to_rep.get(str(e[0]), str(e[0])), node_to_rep.get(str(e[1]), str(e[1]))
        if u == v:
            continue
        new_edges_set.add(tuple(sorted([u, v], key=int)))

    new_edges = [[a, b] for a, b in sorted(new_edges_set, key=lambda e: (int(e[0]), int(e[1])))]

    return {"nodes": new_nodes, "edges": new_edges}


# ===================================================================
# export_eval_format  (--mode eval)
# ===================================================================

def export_eval_format(args) -> int:
    """Dispatch eval export by ``--data-format``."""
    data_format = getattr(args, "data_format", "hypergraph")

    if data_format == "access":
        return export_eval_access(args)
    return _export_eval_hypergraph(args)


def _export_eval_hypergraph(args) -> int:
    """Export per-sample .txt eval files from a hypergraph JSON split."""
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir or "dataset/hypergraph/test")
    aggregate = getattr(args, "aggregate", False)

    logger.info("=" * 80)
    logger.info("Export Eval-Format .txt Files  (data-format: hypergraph)")
    logger.info("=" * 80)
    logger.info(f"Input file:       {input_file}")
    logger.info(f"Output directory:  {output_dir}")
    if aggregate:
        logger.info("Aggregation:       ON (merge same-type adjacent connected rooms)")
    logger.info("=" * 80)

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r") as f:
        raw_data = json.load(f)

    successful = 0
    skipped = 0
    total_nodes_before = 0
    total_nodes_after = 0

    for i, item in enumerate(raw_data):
        try:
            item_id = item.get("id", f"item_{i}")

            hypergraph_str = item.get("split", "")
            if not hypergraph_str:
                logger.warning(f"Skipping {item_id}: missing 'split' field")
                skipped += 1
                continue

            hypergraph = (
                json.loads(hypergraph_str)
                if isinstance(hypergraph_str, str)
                else hypergraph_str
            )

            hg_obj = Hypergraph(hypergraph)
            accessgraph = hg_obj.access_graph

            if not accessgraph or not accessgraph.get("nodes"):
                logger.warning(f"Skipping {item_id}: empty access graph")
                skipped += 1
                continue

            if aggregate:
                bounds_str = item.get("bounds", "{}")
                bounds = (
                    json.loads(bounds_str)
                    if isinstance(bounds_str, str)
                    else bounds_str
                )
                n_before = len(accessgraph["nodes"])
                accessgraph = aggregate_accessgraph(accessgraph, hg_obj, bounds)
                total_nodes_before += n_before
                total_nodes_after += len(accessgraph["nodes"])

            output_data = {
                "accessgraph": accessgraph,
                "hypergraph": (
                    json.dumps(hypergraph)
                    if isinstance(hypergraph, dict)
                    else hypergraph
                ),
                "bounds": item.get("bounds", ""),
            }

            with open(output_dir / f"{item_id}.txt", "w") as f:
                json.dump(output_data, f, indent=2)

            successful += 1

            if (i + 1) % 50 == 0:
                logger.info(f"  Processed {i + 1}/{len(raw_data)} samples...")

        except Exception as e:
            logger.warning(f"Error processing {item.get('id', f'item_{i}')}: {e}")
            skipped += 1

    logger.info(f"\nDone! {successful} files written, {skipped} skipped.")
    if aggregate and total_nodes_before > 0:
        logger.info(
            f"Aggregation: {total_nodes_before} nodes -> {total_nodes_after} nodes "
            f"({total_nodes_before - total_nodes_after} merged)"
        )
    logger.info(f"Output directory: {output_dir}")
    return 0


# ===================================================================
# CLI
# ===================================================================

MODE_MAP = {
    "train": convert_to_jsonl,
    "eval": export_eval_format,
}


def cli():
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Hypergraph data preparation")
    parser.add_argument("--mode", choices=list(MODE_MAP), required=True)

    # Shared
    parser.add_argument("--dataset-dir", type=str, default="dataset/hypergraph")
    parser.add_argument("--seed", type=int, default=42)

    # eval mode
    parser.add_argument(
        "--data-format",
        type=str,
        choices=["access", "hypergraph"],
        default=None,
        help="(eval mode) Input data format: 'access' reads dataset_access JSON "
             "(room polygons + door adjacencies); 'hypergraph' reads BSP hypergraph JSON",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="dataset/hypergraph/test.json",
        help="Path to JSON file to export (eval mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for eval .txt files",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="(eval mode, hypergraph only) Merge same-type, spatially adjacent, "
             "access-connected rooms",
    )

    args = parser.parse_args()

    if args.mode == "eval":
        if args.data_format is None:
            parser.error("--data-format is required for --mode eval")
        if args.aggregate and args.data_format != "hypergraph":
            parser.error("--aggregate is only supported with --data-format hypergraph")

    return MODE_MAP[args.mode](args)


if __name__ == "__main__":
    sys.exit(cli())
