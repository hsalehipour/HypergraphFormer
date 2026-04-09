#!/usr/bin/env python3
"""
Visualise RPLAN samples exported in hypergraph format.

Loads the JSON files produced by ``rplan_to_hypergraph_native.py --export``
and renders each sample using the same RGL-based ``Hypergraph`` /
``HypergraphPlotter`` pipeline used for the Hypergraph1k benchmark.

For every sample two panels are produced:

    Left  -- Access graph (nodes = rooms, edges = adjacencies).
    Right -- RGL mesh floor plan with boundary (requires the C# RGL library).

Usage:
    python dataset/rplan/visualize_rplan_hypergraph.py
    python dataset/rplan/visualize_rplan_hypergraph.py --max 20
    python dataset/rplan/visualize_rplan_hypergraph.py --dataset-dir dataset/rplan/dataset_hypergraph --split train
    python dataset/rplan/visualize_rplan_hypergraph.py --output-dir rplan_hg_vis/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.hypergraph import Hypergraph, HypergraphPlotter

logger = logging.getLogger(__name__)


# ── Data loading ───────────────────────────────────────────────────────

def load_split(dataset_dir: Path, split: str):
    filepath = dataset_dir / f"{split}.json"
    if not filepath.exists():
        logger.warning("File not found: %s", filepath)
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = []
    for i, item in enumerate(data):
        split_str = item.get("split", "")
        if not split_str:
            continue
        hg_dict = json.loads(split_str) if isinstance(split_str, str) else split_str

        bounds_str = item.get("bounds", "")
        bounds = (
            json.loads(bounds_str)
            if isinstance(bounds_str, str) and bounds_str
            else bounds_str
        )

        items.append({
            "index": i,
            "id": item.get("id", f"item_{i}"),
            "database": item.get("database", "?"),
            "area": item.get("area", 0),
            "hg_dict": hg_dict,
            "bounds": bounds,
        })

    return items


# ── Visualisation ──────────────────────────────────────────────────────

def _inject_outside_node(ag: dict, hg_dict: dict) -> dict:
    """Add an 'outside' node + edges to *ag* if any leaf references it."""
    leaves = []

    def _collect(node):
        if node.get("final"):
            leaves.append(node)
        for child in node.get("children", []):
            _collect(child)

    _collect(hg_dict)

    leaf_name_to_id = {leaf["name"]: str(i + 1) for i, leaf in enumerate(leaves)}

    outside_edges = []
    for leaf in leaves:
        if "outside" in leaf.get("connected", []):
            lid = leaf_name_to_id[leaf["name"]]
            outside_edges.append(lid)

    if not outside_edges:
        return ag

    outside_id = "outside"
    nodes = dict(ag.get("nodes", {}))
    nodes[outside_id] = "outside"
    edges = list(ag.get("edges", []))
    for lid in outside_edges:
        edges.append([outside_id, lid])
    return {"nodes": nodes, "edges": edges}


def visualize_item(item: dict, plotter: HypergraphPlotter, output_dir: Path | None):
    item_id = item["id"]

    hg = Hypergraph(item["hg_dict"])
    ag = hg.access_graph
    ag = _inject_outside_node(ag, item["hg_dict"])
    n_nodes = len(ag.get("nodes", {}))
    n_edges = len(ag.get("edges", []))
    import networkx as nx
    from utils.metrics import accessgraph_dict_to_nx
    full_g = accessgraph_dict_to_nx(ag)
    components = list(nx.connected_components(full_g))
    n_comp = len(components)
    comp_tag = f"{n_comp} component{'s' if n_comp != 1 else ''}"

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f"RPLAN #{item_id}  |  db={item['database']}  |  "
        f"{n_nodes} rooms, {n_edges} edges  |  {comp_tag}",
        fontsize=11,
        fontweight="bold",
    )

    ag_rooms_only = {
        "nodes": {k: v for k, v in ag.get("nodes", {}).items() if k != "outside"},
        "edges": [e for e in ag.get("edges", []) if "outside" not in e],
    }
    distinct = plotter.make_distinct_labels(ag_rooms_only)
    if "outside" in ag.get("nodes", {}):
        distinct["outside"] = "Outside"
    face_labels = [
        distinct.get(str(i + 1), f"room{i + 1}") for i in range(n_nodes)
        if str(i + 1) in ag.get("nodes", {})
    ]

    plotter.plot_access_graph_on_axis(
        ag,
        ax_left,
        title="Access Graph",
        components=components,
        node_labels=distinct,
    )

    bounds = item.get("bounds")
    if bounds and isinstance(bounds, dict):
        corners = bounds.get("corners", [])
        polygon = [(c["X"], c["Y"]) for c in corners]
        try:
            mesh = hg.to_mesh(bounds=bounds)
            plotter.plot_mesh_on_axis(
                mesh,
                ax_right,
                boundary_polygon=polygon,
                title="Floor Plan (RGL mesh)",
                face_labels=face_labels,
            )
            ax_right.invert_yaxis()
        except Exception as exc:
            logger.warning("  Mesh failed for %s: %s", item_id, exc)
            ax_right.text(
                0.5, 0.5,
                f"Mesh render failed:\n{exc}",
                ha="center", va="center",
                transform=ax_right.transAxes,
                fontsize=10, wrap=True,
            )
            ax_right.axis("off")
    else:
        ax_right.text(
            0.5, 0.5, "No boundary data available",
            ha="center", va="center",
            transform=ax_right.transAxes, fontsize=12,
        )
        ax_right.axis("off")

    fig.subplots_adjust(top=0.90, wspace=0.05)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"rplan_hg_{item_id}.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)
    else:
        plt.show()
        plt.close(fig)
        return None


# ── Main ───────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Visualise RPLAN hypergraph samples using the RGL pipeline.",
    )
    parser.add_argument(
        "--dataset-dir", type=str, default="dataset/rplan/dataset_hypergraph",
        help="Directory containing train.json / val.json / test.json",
    )
    parser.add_argument(
        "--split", type=str, nargs="*", default=["train"],
        choices=["train", "val", "test"],
        help="Which split(s) to visualise (default: train)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="debug_bsp_output/rplan_hg",
        help="Save figures here (default: debug_bsp_output/rplan_hg)",
    )
    parser.add_argument(
        "--max", type=int, default=None,
        help="Maximum number of samples to visualise per split",
    )
    parser.add_argument(
        "--sample", type=str, nargs="+", default=None,
        help="One or more RPLAN sample names to visualise (e.g. 3951 57395)",
    )
    parser.add_argument(
        "--disconnected-only", action="store_true",
        help="Only visualise samples with disconnected access graphs",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None
    plotter = HypergraphPlotter()

    requested_names = set(args.sample) if args.sample else None

    for split in args.split:
        logger.info("Loading %s ...", split)
        items = load_split(dataset_dir, split)
        if not items:
            continue
        logger.info("  %d samples in %s", len(items), split)

        if requested_names is not None:
            items = [it for it in items if str(it["id"]) in requested_names]
            if not items:
                logger.info("  No matching samples found for --sample %s",
                            " ".join(args.sample))
                continue

        rendered = 0
        disconnected = 0
        for item in items:
            hg = Hypergraph(item["hg_dict"])
            n_comp = len(hg.access_graph_components())
            if n_comp >= 2:
                disconnected += 1

            if args.disconnected_only and n_comp < 2:
                continue

            saved = visualize_item(item, plotter, output_dir)
            rendered += 1
            status = f"  {item['id']}: {n_comp} component(s)"
            if saved:
                status += f"  -> {saved}"
            logger.info(status)

            if args.max and rendered >= args.max:
                logger.info("  Reached --max %d, stopping.", args.max)
                break

        logger.info(
            "  %s: rendered %d, disconnected %d / %d total",
            split, rendered, disconnected, len(items),
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
