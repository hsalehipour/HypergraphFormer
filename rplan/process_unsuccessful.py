#!/usr/bin/env python3
"""
Visualise RPLAN samples that failed hypergraph conversion.

Identifies samples whose name does NOT appear in the successful-output
directory, picks N at random, and saves a PNG for each showing:
  - room polygons coloured by category
  - access graph edges between room centroids (falls back to spatial
    adjacencies when access_adjacencies is absent)
  - outer boundary
  - room-type labels

Usage:
    python -m rplan.process_unsuccessful
    python -m rplan.process_unsuccessful --count 20 --split train --seed 42
    python -m rplan.process_unsuccessful --output-dir my_plots/
    python -m rplan.process_unsuccessful --diagnose   # re-run conversion to get failure reasons
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon

# Room category index -> display name (RPLAN / DiffPlanner convention)
RPLAN_ROOM_NAMES = {
    0: "LivingRoom",
    1: "Bedroom",
    2: "Kitchen",
    3: "Bathroom",
    4: "Balcony",
    5: "Storage",
}

# Distinct colour per category
CATEGORY_COLORS = {
    0: "#FFEEAD",  # LivingRoom  – warm cream
    1: "#96CEB4",  # Bedroom     – sage green
    2: "#FF6F69",  # Kitchen     – salmon
    3: "#88D8B0",  # Bathroom    – mint
    4: "#C5E1A5",  # Balcony     – light green
    5: "#B0BEC5",  # Storage     – blue-grey
}


def get_room_name(category_index: int) -> str:
    return RPLAN_ROOM_NAMES.get(category_index, f"Unknown_{category_index}")


def plot_rplan_sample(sample: dict, output_path: Path, failure_reason: str = ""):
    """Render one RPLAN sample to a PNG.

    Draws room polygons coloured by category, the outer boundary, adjacency
    edges, and centroid labels.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    rooms = sample["rooms"]
    adjacencies = sample.get("access_adjacencies", sample["adjacencies"])

    # -- room polygons --
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

    # -- adjacency edges --
    for u, v in adjacencies:
        if u in centroids and v in centroids:
            xu, yu = centroids[u]
            xv, yv = centroids[v]
            ax.plot([xu, xv], [yu, yv], color="#555555", linewidth=0.8,
                    linestyle="--", zorder=3)

    # -- centroid labels (room type + id) --
    for room in rooms:
        rid = room["id"]
        if rid in centroids:
            cx, cy = centroids[rid]
            label = f"{get_room_name(room['category'])}\n#{rid}"
            ax.text(cx, cy, label, ha="center", va="center", fontsize=6,
                    fontweight="bold", zorder=4)

    # -- outer boundary --
    boundary_pts = np.array(
        [[pt[0], pt[1]] for pt in sample["boundary"]], dtype=float
    )
    closed = np.vstack([boundary_pts, boundary_pts[0]])
    ax.plot(closed[:, 0], closed[:, 1], color="black", linewidth=2.0, zorder=5)

    # -- legend --
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

    # -- title --
    name = sample.get("name", "?")
    n_rooms = len(rooms)
    n_adj = len(adjacencies)
    adj_label = "access" if "access_adjacencies" in sample else "adj"
    title = f"RPLAN #{name}  |  {n_rooms} rooms, {n_adj} {adj_label}"
    if failure_reason:
        title += f"\nFailed: {failure_reason}"
    ax.set_title(title, fontsize=10)

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.invert_yaxis()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()

    plt.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def get_successful_names(successful_dir: Path) -> set:
    """Collect sample names that were successfully saved."""
    names = set()
    for txt_file in successful_dir.glob("*.txt"):
        # Filenames are like "train_57395.txt" -> extract "57395"
        stem = txt_file.stem
        parts = stem.split("_", 1)
        if len(parts) >= 2:
            names.add(parts[1])
    return names


def load_unsuccessful_samples(json_path: Path, successful_names: set):
    """Return samples whose name is NOT in the successful set."""
    with open(json_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    return [
        s for s in samples if str(s.get("name", "")) not in successful_names
    ]


def try_convert_and_get_reason(sample: dict) -> str:
    """Attempt conversion and return a short failure reason (empty on success).

    This lazily imports the RGL-dependent modules so that basic plotting works
    without the hypergraph library installed.
    """
    try:
        from .rplan_to_hypergraph import RPLANGraph
        from hypergraph.api.lib.tools import RGL
        from RGeoLib import DataNodeUtils

        graph = RPLANGraph.sample_to_graph(sample)
        hull = RPLANGraph.get_convex_hull(graph)
        if hull is None:
            return "no convex hull"

        face = RPLANGraph.convex_hull_to_face_wrapper(hull)
        if face is None:
            return "face wrapper failed"

        nmesh, _ = RPLANGraph.create_nmesh(graph)
        connectivity, _ = RPLANGraph.build_connectivity_dict(graph)

        data_node = DataNodeUtils.CreateDataNodeWithConnectivity(
            nmesh, face, connectivity
        )
        if data_node is None:
            return "DataNode creation failed"

        face_count = len(nmesh.faceList)
        all_children = RGL.DataNode.GetAllChildren(data_node)
        leaf_count = sum(
            1 for n in all_children if hasattr(n, "final") and n.final
        )

        if face_count != leaf_count:
            return f"face/leaf mismatch ({face_count} faces, {leaf_count} leaves)"
        if leaf_count < 4:
            return f"too few leaves ({leaf_count})"
        return ""
    except Exception as exc:
        return str(exc)[:120]


def main():
    parser = argparse.ArgumentParser(
        description="Visualise unsuccessful RPLAN-to-hypergraph conversions",
    )
    parser.add_argument(
        "--data-dir", type=str,
        default="dataset/rplan/dataset_access",
        help="Directory with data_train.json / data_val.json / data_test.json",
    )
    parser.add_argument(
        "--successful-dir", type=str,
        default="training_data/extracted_hypergraph_rplan",
        help="Directory containing successfully saved .txt hypergraphs",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str,
        default="unsuccessful_rplan_plots",
        help="Output directory for PNGs",
    )
    parser.add_argument(
        "--split", type=str, default="train",
        choices=["train", "val", "test"],
        help="Which split to inspect (default: train)",
    )
    parser.add_argument(
        "--count", "-c", type=int, default=10,
        help="Number of samples to visualise (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible selection",
    )
    parser.add_argument(
        "--diagnose", action="store_true",
        help="Re-run conversion to capture the specific failure reason per sample "
             "(requires the hypergraph/RGL library)",
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    data_dir = Path(args.data_dir)
    successful_dir = Path(args.successful_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    successful_names = get_successful_names(successful_dir)
    print(f"Found {len(successful_names)} successful hypergraphs in {successful_dir}")

    json_path = data_dir / f"data_{args.split}.json"
    if not json_path.exists():
        print(f"File not found: {json_path}")
        return

    unsuccessful = load_unsuccessful_samples(json_path, successful_names)
    print(f"Split '{args.split}': {len(unsuccessful)} unsuccessful samples")

    if not unsuccessful:
        print("Nothing to visualise.")
        return

    selected = random.sample(unsuccessful, min(args.count, len(unsuccessful)))
    print(f"Selected {len(selected)} samples to visualise")
    print("=" * 60)

    for i, sample in enumerate(selected, 1):
        name = sample.get("name", "unknown")
        n_edges = len(sample.get("access_adjacencies", sample["adjacencies"]))
        edge_label = "access" if "access_adjacencies" in sample else "adj"
        print(f"[{i}/{len(selected)}] Sample #{name}  "
              f"({len(sample['rooms'])} rooms, {n_edges} {edge_label})")

        reason = ""
        if args.diagnose:
            reason = try_convert_and_get_reason(sample)
            if reason:
                print(f"  Failure: {reason}")
            else:
                print("  (succeeded — may have been filtered by min_leaves)")

        out_path = output_dir / f"{args.split}_{name}.png"
        plot_rplan_sample(sample, out_path, failure_reason=reason)
        print(f"  Saved -> {out_path}")

    print("=" * 60)
    print(f"Done. {len(selected)} plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
