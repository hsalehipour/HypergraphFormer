#!/usr/bin/env python3
"""
Visualize the fixed access-graph cases.

Compares each original disconnected sample against its fixed counterpart,
producing a side-by-side figure identical in style to
``detect_disconnected_accessgraphs.py`` but using the **fixed** JSON data:

    Top-left    -- Fixed access graph (now connected).
    Bottom-left -- Original access graph (disconnected).
    Right       -- RGL mesh floor plan with boundary.

Only samples that were actually repaired (were disconnected in the original
data and are now connected in the fixed data) are visualised.

Usage:
    python rplan/visualize_fixed_accessgraphs.py
    python rplan/visualize_fixed_accessgraphs.py --output-dir dataset/disconnected_fixed_vis
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from utils.hypergraph import Hypergraph, HypergraphPlotter

logger = logging.getLogger(__name__)


# ── Data loading ───────────────────────────────────────────────────────

def load_split_raw(dataset_dir: Path, split: str):
    """Load raw items from a dataset split JSON, preserving all metadata."""
    filepath = dataset_dir / f"{split}.json"
    if not filepath.exists():
        logger.warning("File not found: %s", filepath)
        return []

    with open(filepath, "r") as f:
        data = json.load(f)

    items = []
    for i, item in enumerate(data):
        split_str = item.get("split", "")
        if not split_str:
            continue
        hg_dict = json.loads(split_str) if isinstance(split_str, str) else split_str

        bounds_str = item.get("bounds", "")
        bounds = json.loads(bounds_str) if isinstance(bounds_str, str) and bounds_str else bounds_str

        items.append({
            "index": i,
            "id": item.get("id", f"item_{i}"),
            "database": item.get("database", "?"),
            "area": item.get("area", 0),
            "hg_dict": hg_dict,
            "bounds": bounds,
        })

    return items


# ── Main ───────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Visualize fixed access-graph cases from the RPLAN dataset.",
    )
    parser.add_argument(
        "--original-dir", type=str, default="dataset/rplan/dataset_hypergraph",
        help="Directory with the original train.json / validation.json / test.json",
    )
    parser.add_argument(
        "--fixed-dir", type=str, default="dataset/rplan/dataset_hypergraph_fixed",
        help="Directory with the fixed train.json / validation.json / test.json",
    )
    parser.add_argument(
        "--split", type=str, nargs="*",
        default=["train", "validation", "test"],
        choices=["train", "validation", "test"],
        help="Which split(s) to process (default: all three)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="dataset/rplan/disconnected_fixed_vis",
        help="Save figures to this directory (per-split sub-folders)",
    )
    args = parser.parse_args()

    original_dir = Path(args.original_dir)
    fixed_dir = Path(args.fixed_dir)
    plotter = HypergraphPlotter()
    grand_total = 0

    for split in args.split:
        logger.info("Processing %s ...", split)
        orig_items = load_split_raw(original_dir, split)
        fixed_items = load_split_raw(fixed_dir, split)

        if not orig_items or not fixed_items:
            continue

        fixed_by_id = {it["id"]: it for it in fixed_items}
        found = 0

        for orig in orig_items:
            hg_orig = Hypergraph(orig["hg_dict"])
            if hg_orig.is_access_graph_connected():
                continue

            fixed = fixed_by_id.get(orig["id"])
            if fixed is None:
                continue

            hg_fixed = Hypergraph(fixed["hg_dict"])
            if not hg_fixed.is_access_graph_connected():
                continue

            found += 1
            grand_total += 1

            ag = hg_fixed.access_graph
            n_nodes = len(ag.get("nodes", {}))
            n_edges = len(ag.get("edges", []))

            ag_orig = hg_orig.access_graph
            n_edges_orig = len(ag_orig.get("edges", []))
            added = n_edges - n_edges_orig

            logger.info(
                "  FIXED  %s  (index=%d, %d rooms, %d→%d edges, +%d)",
                orig["id"], orig["index"], n_nodes,
                n_edges_orig, n_edges, added,
            )

            import matplotlib.gridspec as gridspec

            fig = plt.figure(figsize=(18, 9))
            gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], hspace=0.30)
            ax_fixed = fig.add_subplot(gs[0, 0])
            ax_orig = fig.add_subplot(gs[1, 0])
            ax_right = fig.add_subplot(gs[:, 1])

            fig.suptitle(
                f"{orig['id']}  |  db={orig['database']}  |  "
                f"{n_nodes} rooms, {n_edges} edges (+{added} added)  |  "
                f"FIXED (was disconnected)",
                fontsize=11, fontweight="bold",
            )

            distinct = plotter.make_distinct_labels(ag)
            distinct_orig = plotter.make_distinct_labels(ag_orig)

            components_orig = hg_orig.access_graph_components()
            plotter.plot_access_graph_on_axis(
                ag_orig, ax_orig, title="Original (disconnected)",
                components=components_orig,
                node_labels=distinct_orig,
            )

            plotter.plot_access_graph_on_axis(
                ag, ax_fixed, title="Fixed (connected)",
                node_labels=distinct,
            )

            bounds = fixed.get("bounds")
            if bounds and isinstance(bounds, dict):
                corners = bounds.get("corners", [])
                polygon = [(c["X"], c["Y"]) for c in corners]
                try:
                    mesh = hg_fixed.to_mesh(bounds=bounds)
                    face_labels = hg_fixed.mesh_face_labels(
                        mesh, distinct=distinct)
                    plotter.plot_mesh_on_axis(
                        mesh, ax_right,
                        boundary_polygon=polygon,
                        title="Floor Plan",
                        face_labels=face_labels,
                    )
                except Exception as exc:
                    logger.warning("  Could not render mesh for %s: %s",
                                   orig["id"], exc)
                    ax_right.text(
                        0.5, 0.5, f"Mesh render failed:\n{exc}",
                        ha="center", va="center", transform=ax_right.transAxes,
                        fontsize=10, wrap=True,
                    )
                    ax_right.axis("off")
            else:
                ax_right.text(
                    0.5, 0.5, "No boundary data available",
                    ha="center", va="center", transform=ax_right.transAxes,
                    fontsize=12,
                )
                ax_right.axis("off")

            fig.subplots_adjust(top=0.90, wspace=0.05)

            out = Path(args.output_dir) / split
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                out / f"{orig['id']}.png",
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig)

        logger.info("  %s: %d fixed cases visualised", split, found)

    logger.info("Overall: %d fixed cases visualised", grand_total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
