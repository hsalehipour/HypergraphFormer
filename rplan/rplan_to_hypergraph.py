#!/usr/bin/env python3
"""
RPLAN to Hypergraph Conversion

Converts RPLAN floor plan data (JSON format from dataset_access or
dataset_diffplanner) to hypergraph representation using the RGL library.
Supports both single-sample interactive demo and batch processing.

Single-sample mode:
    python -m rplan.rplan_to_hypergraph --sample-index 0

Batch mode (processes all splits):
    python -m rplan.rplan_to_hypergraph --batch
    python -m rplan.rplan_to_hypergraph --batch --min-leaves 4 --verbose
"""

import argparse
import json
import logging
from pathlib import Path

import clr
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from shapely.geometry import MultiPoint, Polygon
from tqdm import tqdm

from utils import HypergraphPlotter, remove_mergeid_suffixes

from hypergraph.api.lib.tools import RGL

clr.AddReference("System.Collections")
from System import Collections
from RGeoLib import DataNodeUtils


logger = logging.getLogger(__name__)

RPLAN_ROOM_NAMES = {
    0: "LivingRoom",
    1: "Bedroom",
    2: "Kitchen",
    3: "Bathroom",
    4: "Balcony",
    5: "Storage",
}


def get_room_name(category_index: int) -> str:
    return RPLAN_ROOM_NAMES.get(category_index, f"Unknown_{category_index}")


# ---------------------------------------------------------------------------
# RPLANGraph — graph construction and geometry helpers
# ---------------------------------------------------------------------------

class RPLANGraph:
    """Static helper methods for building graphs and RGL objects from RPLAN data."""

    @staticmethod
    def sample_to_graph(sample: dict) -> nx.Graph:
        """Build a NetworkX graph from a single RPLAN JSON sample.

        If the sample contains ``access_adjacencies`` (door-based pairs
        produced by :mod:`rplan.build_dataset_access`), those are used for
        edges.  Otherwise falls back to the spatial ``adjacencies`` field.

        Each node stores:
          - geometry: list of [x, y] polygon vertices
          - room_type: integer category (0-5)
          - size: area value from the dataset
          - order: ordering index from the dataset
        """
        G = nx.Graph()

        for room in sample["rooms"]:
            G.add_node(
                room["id"],
                geometry=room["r_boundary"],
                room_type=room["category"],
                size=room.get("size", 0),
                order=room.get("order", 0),
            )

        adj_key = (
            "access_adjacencies"
            if "access_adjacencies" in sample
            else "adjacencies"
        )
        for u, v in sample[adj_key]:
            if G.has_node(u) and G.has_node(v):
                G.add_edge(u, v)

        return G

    @staticmethod
    def get_boundary_polygon(sample: dict) -> Polygon:
        """Derive the outer boundary as the union of all room polygons.

        Using the union of rooms is more robust than the raw boundary field,
        because the rooms tile the interior exactly.
        """
        room_polygons = []
        for room in sample["rooms"]:
            coords = room["r_boundary"]
            if len(coords) >= 3:
                p = Polygon(coords)
                if p.is_valid:
                    room_polygons.append(p)
                else:
                    room_polygons.append(p.buffer(0))

        if not room_polygons:
            return None

        from shapely.ops import unary_union
        union = unary_union(room_polygons)
        if union.is_empty:
            return None
        if union.geom_type == "MultiPolygon":
            union = max(union.geoms, key=lambda g: g.area)
        return union

    @staticmethod
    def get_convex_hull(G: nx.Graph):
        """Compute the convex hull of all room polygon vertices."""
        all_points = []
        for _, data in G.nodes(data=True):
            geom = data.get("geometry", [])
            for pt in geom:
                all_points.append(pt[:2])

        if len(all_points) < 3:
            return None

        hull = MultiPoint(all_points).convex_hull
        if hull.geom_type != "Polygon":
            return None
        return hull

    @staticmethod
    def convex_hull_to_face_wrapper(hull_polygon):
        """Convert a Shapely Polygon (convex hull) to a Face wrapper for DataNode."""
        from hypergraph.api.lib.tools import Face, Pt

        if not hull_polygon:
            return None

        coords = list(hull_polygon.exterior.coords)[:-1]
        face_points = [Pt(loc=(float(x), float(y), 0.0)) for x, y in coords]
        hull_face = Face(corners=face_points)
        return hull_face.obj

    @staticmethod
    def create_nmesh(G: nx.Graph):
        """Create an RGL NMesh from room polygons in the graph.

        Returns (nmesh, room_faces) where room_faces is a dict mapping
        node-id -> NFace.
        """
        face_list = Collections.Generic.List[RGL.NFace]()
        room_faces = {}

        for node_id in sorted(G.nodes()):
            geom = G.nodes[node_id].get("geometry", [])
            if len(geom) < 3:
                continue

            pts = Collections.Generic.List[RGL.Vec3d]()
            for x, y in geom:
                pts.Add(RGL.Vec3d(x=float(x), y=float(y), z=0.0))

            face = RGL.NFace(pts)
            room_type = G.nodes[node_id].get("room_type", 0)
            face.merge_id = f"{get_room_name(room_type)}_{node_id}"

            face.updateEdgeConnectivity()
            face_list.Add(face)
            room_faces[node_id] = face

        nmesh = RGL.NMesh(face_list)
        return nmesh, room_faces

    @staticmethod
    def build_connectivity_dict(G: nx.Graph):
        """Build an RGL connectivity dictionary from graph edges.

        Returns (connectivity_dict, node_to_merge_id).
        """
        from System.Collections.Generic import Dictionary, List as CSharpList

        node_to_merge_id = {}
        for node_id in sorted(G.nodes()):
            room_type = G.nodes[node_id].get("room_type", 0)
            node_to_merge_id[node_id] = f"{get_room_name(room_type)}_{node_id}"

        connectivity = Dictionary[str, CSharpList[str]]()

        for node_id in G.nodes():
            merge_id = node_to_merge_id[node_id]
            neighbors = list(G.neighbors(node_id))
            connected_ids = CSharpList[str]()
            for neighbor in neighbors:
                connected_ids.Add(node_to_merge_id[neighbor])
            connectivity[merge_id] = connected_ids

        return connectivity, node_to_merge_id

    @staticmethod
    def generate_access_graph_json(G: nx.Graph) -> dict:
        """Serialise the graph to a JSON-ready dict for the access graph."""
        nodes = []
        for node_id in sorted(G.nodes()):
            data = G.nodes[node_id]
            nodes.append({
                "id": node_id,
                "room_type": get_room_name(data.get("room_type", 0)),
                "room_type_id": data.get("room_type", 0),
            })

        edges = []
        for u, v in G.edges():
            edges.append({"source": u, "target": v})

        return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_and_save_sample(sample, split_name, output_dir, min_leaves=4):
    """Process one RPLAN sample, validate, and save if successful.

    Returns (success, info, saved_path).
    """
    name = str(sample.get("name", "unknown"))
    try:
        G = RPLANGraph.sample_to_graph(sample)

        if G.number_of_nodes() < 2:
            return False, "too few nodes", None

        access_graph_dict = RPLANGraph.generate_access_graph_json(G)

        hull = RPLANGraph.get_convex_hull(G)
        if hull is None:
            return False, "no convex hull", None

        face = RPLANGraph.convex_hull_to_face_wrapper(hull)
        if face is None:
            return False, "face wrapper failed", None

        nmesh, _ = RPLANGraph.create_nmesh(G)
        if nmesh is None:
            return False, "nmesh creation failed", None

        connectivity, _ = RPLANGraph.build_connectivity_dict(G)

        data_node = DataNodeUtils.CreateDataNodeWithConnectivity(
            nmesh, face, connectivity
        )
        if data_node is None:
            return False, "DataNode creation failed", None

        face_count = len(nmesh.faceList)
        try:
            all_children = RGL.DataNode.GetAllChildren(data_node)
            leaf_count = sum(
                1 for n in all_children if hasattr(n, "final") and n.final
            )
        except Exception as exc:
            return False, f"leaf counting failed: {exc}", None

        if face_count != leaf_count:
            return False, (
                f"face/leaf mismatch ({face_count} faces, {leaf_count} leaves)"
            ), None

        if leaf_count < min_leaves:
            return False, (
                f"too few leaves ({leaf_count} < {min_leaves})"
            ), None

        serialized_text = RGL.DataNode.serializeDataNode(data_node)
        serialized_text = remove_mergeid_suffixes(serialized_text)

        output_filename = f"{split_name}_{name}.txt"
        output_path = output_dir / output_filename

        combined = {
            "accessgraph": access_graph_dict,
            "hypergraph": serialized_text,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)

        return True, {
            "face_count": face_count,
            "leaf_count": leaf_count,
            "nodes_count": G.number_of_nodes(),
            "edges_count": G.number_of_edges(),
        }, output_path

    except Exception as exc:
        return False, f"error: {exc}", None


def process_all_splits(data_dir, output_dir, splits, min_leaves=4):
    """Iterate over requested splits and save valid hypergraphs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_samples": 0,
        "saved": 0,
        "validation_failures": 0,
        "insufficient_leaves": 0,
    }

    for split in splits:
        json_path = data_dir / f"data_{split}.json"
        if not json_path.exists():
            logger.warning("Split not found: %s", json_path)
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        logger.info("Processing %s: %d samples", split, len(samples))

        pbar = tqdm(samples, desc=split, unit="sample")
        for sample in pbar:
            stats["total_samples"] += 1
            success, info, _ = process_and_save_sample(
                sample, split, output_dir, min_leaves
            )
            if success:
                stats["saved"] += 1
            else:
                if isinstance(info, str) and "too few leaves" in info:
                    stats["insufficient_leaves"] += 1
                else:
                    stats["validation_failures"] += 1
            pbar.set_postfix_str(
                f"saved {stats['saved']}/{stats['total_samples']}"
            )
        pbar.close()

    return stats


# ---------------------------------------------------------------------------
# Single-sample demo
# ---------------------------------------------------------------------------

def run_single_sample(data_dir, split, sample_index):
    """Load one sample, convert to hypergraph, and visualise."""
    json_path = data_dir / f"data_{split}.json"
    if not json_path.exists():
        logger.error("File not found: %s", json_path)
        return

    with open(json_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if sample_index >= len(samples):
        logger.error("Index %d out of range (split has %d samples)",
                      sample_index, len(samples))
        return

    sample = samples[sample_index]
    name = sample.get("name", sample_index)
    logger.info("Sample #%s  (%d rooms)", name, len(sample["rooms"]))

    G = RPLANGraph.sample_to_graph(sample)
    logger.info("Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    hull = RPLANGraph.get_convex_hull(G)
    face = RPLANGraph.convex_hull_to_face_wrapper(hull)
    nmesh, _ = RPLANGraph.create_nmesh(G)
    connectivity, _ = RPLANGraph.build_connectivity_dict(G)

    data_node = DataNodeUtils.CreateDataNodeWithConnectivity(
        nmesh, face, connectivity
    )

    if data_node is None:
        logger.error("DataNode creation failed")
        return

    serialized = RGL.DataNode.serializeDataNode(data_node)
    serialized = remove_mergeid_suffixes(serialized)

    plotter = HypergraphPlotter()
    fig = plotter.plot_hypergraph(serialized)
    out_path = f"rplan_{name}_hypergraph.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved hypergraph plot -> %s", out_path)

    boundary = RPLANGraph.get_boundary_polygon(sample)
    if boundary is not None:
        coords = list(boundary.exterior.coords)[:-1]
        pts = Collections.Generic.List[RGL.Vec3d]()
        for x, y in coords:
            pts.Add(RGL.Vec3d(x=float(x), y=float(y), z=0.0))
        face_bounds = RGL.NFace(pts)
        nface_list = Collections.Generic.List[RGL.NFace]()
        nface_list.Add(face_bounds)
        boundary_mesh = RGL.NMesh(nface_list)
        generated = RGL.DataNode.subdivideWithMeshRatio(data_node, boundary_mesh)

        fig2 = plotter.simple_plot_generated_mesh(generated)
        out2 = f"rplan_{name}_floor_plan.png"
        plt.savefig(out2, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved floor-plan plot -> %s", out2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RPLAN to Hypergraph conversion (single-sample or batch)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m rplan.rplan_to_hypergraph --sample-index 0                    # single demo
  python -m rplan.rplan_to_hypergraph --batch                              # batch all
  python -m rplan.rplan_to_hypergraph --batch --min-leaves 4 --verbose     # batch + options
        """,
    )
    parser.add_argument(
        "--data-dir", type=str,
        default="dataset/rplan/dataset_access",
        help="Directory containing data_train.json / data_val.json / data_test.json",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str,
        default="training_data/extracted_hypergraph_rplan",
        help="Output directory for saved hypergraphs",
    )
    parser.add_argument(
        "--split", type=str, default=None,
        choices=["train", "val", "test"],
        help="Restrict to a single split (default: process all in batch mode)",
    )
    parser.add_argument(
        "--sample-index", type=int, default=None,
        help="Index of a single sample to visualise (triggers demo mode)",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Run batch processing over all splits",
    )
    parser.add_argument(
        "--min-leaves", type=int, default=4,
        help="Minimum leaf count to save a hypergraph (default: 4)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    data_dir = Path(args.data_dir)

    if args.batch:
        splits = [args.split] if args.split else ["train", "val", "test"]
        output_dir = Path(args.output_dir)
        stats = process_all_splits(data_dir, output_dir, splits, args.min_leaves)

        logger.info("=" * 60)
        logger.info("RESULTS")
        for k, v in stats.items():
            logger.info("  %s: %s", k, f"{v:,}")
        total = stats["total_samples"]
        rate = (stats["saved"] / total * 100) if total > 0 else 0
        logger.info("  save_rate: %.2f%%", rate)
        logger.info("Output: %s", output_dir)
    else:
        split = args.split or "train"
        index = args.sample_index if args.sample_index is not None else 0
        run_single_sample(data_dir, split, index)


if __name__ == "__main__":
    main()
