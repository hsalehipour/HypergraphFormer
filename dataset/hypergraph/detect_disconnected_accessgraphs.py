#!/usr/bin/env python3
"""
Detect and visualize floor plans with disconnected access graphs.

Scans the Hypergraph1k dataset for samples whose access graph consists of
two or more isolated connected components.  For each disconnected sample,
produces a side-by-side figure:

    Left  -- Access graph (styled per benchmark conventions).
    Right -- RGL mesh floor plan with boundary (via subdivideWithMeshRatio).

Usage:
    python scripts/detect_disconnected_accessgraphs.py
    python scripts/detect_disconnected_accessgraphs.py --split test --max 10
    python scripts/detect_disconnected_accessgraphs.py --output-dir docs/disconnected
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
        description="Detect and visualise disconnected access graphs "
                    "in the Hypergraph1k dataset.",
    )
    parser.add_argument(
        "--dataset-dir", type=str, default="dataset/hypergraph",
        help="Directory containing train.json / validation.json / test.json",
    )
    parser.add_argument(
        "--split", type=str, nargs="*",
        default=["train", "validation", "test"],
        choices=["train", "validation", "test"],
        help="Which split(s) to scan (default: all three)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Save figures to this directory instead of displaying them",
    )
    parser.add_argument(
        "--max", type=int, default=None,
        help="Stop after this many disconnected samples (per split)",
    )
    parser.add_argument(
        "--min-components", type=int, default=2,
        help="Minimum number of connected components to flag (default: 2)",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    plotter = HypergraphPlotter()
    grand_total = 0
    grand_disconnected = 0

    for split in args.split:
        logger.info("Scanning %s ...", split)
        items = load_split_raw(dataset_dir, split)
        if not items:
            continue

        found = 0
        for item in items:
            hg = Hypergraph(item["hg_dict"])
            components = hg.access_graph_components()

            if len(components) < args.min_components:
                continue

            found += 1
            grand_disconnected += 1

            ag = hg.access_graph
            n_comp = len(components)
            comp_summary = ", ".join(
                f"C{i+1}({len(c)})" for i, c in enumerate(components)
            )
            n_nodes = len(ag.get("nodes", {}))
            n_edges = len(ag.get("edges", []))

            logger.info(
                "  DISCONNECTED  %s  (index=%d, %d components, "
                "%d rooms, %d edges)",
                item["id"], item["index"], n_comp, n_nodes, n_edges,
            )

            fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 7))
            fig.suptitle(
                f"{item['id']}  |  db={item['database']}  |  "
                f"{n_nodes} rooms, {n_edges} edges  |  "
                f"{n_comp} components: {comp_summary}",
                fontsize=11, fontweight="bold",
            )

            plotter.plot_access_graph_on_axis(
                ag, ax_left, title="Access Graph",
                components=components,
            )

            bounds = item.get("bounds")
            if bounds and isinstance(bounds, dict):
                corners = bounds.get("corners", [])
                polygon = [(c["X"], c["Y"]) for c in corners]
                try:
                    mesh = hg.to_mesh(bounds=bounds)
                    plotter.plot_mesh_on_axis(
                        mesh, ax_right,
                        boundary_polygon=polygon,
                        title="Floor Plan",
                    )
                except Exception as exc:
                    logger.warning("  Could not render mesh for %s: %s",
                                   item["id"], exc)
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

            fig.tight_layout(rect=[0, 0, 1, 0.93])

            if args.output_dir:
                out = Path(args.output_dir) / split
                out.mkdir(parents=True, exist_ok=True)
                fig.savefig(
                    out / f"{item['id']}.png",
                    dpi=150, bbox_inches="tight",
                )
                plt.close(fig)
            else:
                plt.show()

            if args.max and found >= args.max:
                logger.info("  Reached --max %d, moving on.", args.max)
                break

        grand_total += len(items)
        logger.info(
            "  %s: %d disconnected / %d total (%.1f%%)",
            split, found, len(items),
            100 * found / len(items) if items else 0,
        )

    logger.info(
        "Overall: %d disconnected / %d total (%.1f%%)",
        grand_disconnected, grand_total,
        100 * grand_disconnected / grand_total if grand_total else 0,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
