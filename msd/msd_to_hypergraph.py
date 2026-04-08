#!/usr/bin/env python3
"""
MSD to Hypergraph Conversion

Converts Modified Swiss Dwellings (MSD) graph data to hypergraph representation
using the RGL library. Supports both single-sample interactive demo and batch
processing of the full dataset.

Single-sample mode:
    python -m msd.msd_to_hypergraph --file-id 7498 --subgraph-index 1

Batch mode (processes all train + test data):
    python -m msd.msd_to_hypergraph --batch
    python -m msd.msd_to_hypergraph --batch --min-leaves 5 --verbose
    python -m msd.msd_to_hypergraph --batch --output-dir my_output/
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import clr
import matplotlib.pyplot as plt
from tqdm import tqdm

from .graph import MSDGraph
from .plotter import MSDPlotter
from utils import HypergraphPlotter, remove_mergeid_suffixes

from hypergraph.api.lib.tools import RGL

clr.AddReference("System.Collections")
from System import Collections
from RGeoLib import DataNodeUtils


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MSDHypergraph — single-sample conversion and visualisation
# ---------------------------------------------------------------------------

class MSDHypergraph(MSDGraph):
    """
    MSD to Hypergraph conversion workflow.

    Encapsulates the complete pipeline from loading MSD data, converting to
    hypergraph, and generating floor plans.
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.msd_plotter = MSDPlotter()
        self.hypergraph_plotter = HypergraphPlotter()
        logger.info("MSDHypergraph initialised  |  data_dir=%s", self.data_dir)

    def convert_to_hypergraph(self, graph_data):
        """Convert a subgraph to its hypergraph DataNode representation."""
        assert graph_data is not None, "No graph data to convert."

        hull_data = self.get_geometry_based_boundaries(graph_data)
        face = self.convex_hull_to_face_wrapper(hull_data['convex_hull_geometry'])

        nmesh, room_faces = self.create_nmesh_from_subgraph(graph_data)
        connectivity_dict, _ = self.build_connectivity_dict(graph_data)

        data_node = DataNodeUtils.CreateDataNodeWithConnectivity(
            nmesh, face, connectivity_dict
        )
        return data_node

    def plot_hypergraph(self, hypergraph, prefix):
        """Serialize and plot the hypergraph tree."""
        assert hypergraph is not None, "No hypergraph data to plot."
        serialized_data = RGL.DataNode.serializeDataNode(hypergraph)
        fig = self.hypergraph_plotter.plot_hypergraph(serialized_data, node_scale=2000)
        out = f"{prefix}_hypergraph.png"
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved hypergraph plot -> %s", out)
        return fig

    def hypergraph_to_floor_plan(self, hypergraph, custom_boundary_points):
        """Generate a floor plan mesh from hypergraph + boundary polygon."""
        assert hypergraph is not None, "No hypergraph data."
        assert custom_boundary_points is not None, "No boundary points."

        points_list = Collections.Generic.List[RGL.Vec3d]()
        for x, y in custom_boundary_points:
            points_list.Add(RGL.Vec3d(x=x, y=y, z=0))

        face_bounds = RGL.NFace(points_list)
        nface_list = Collections.Generic.List[RGL.NFace]()
        nface_list.Add(face_bounds)
        boundary_mesh = RGL.NMesh(nface_list)

        generated_mesh = RGL.DataNode.subdivideWithMeshRatio(hypergraph, boundary_mesh)
        return generated_mesh

    def plot_hypergraph_floor_plan(self, hypergraph_floor_plan, prefix):
        """Plot the generated floor plan."""
        assert hypergraph_floor_plan is not None, "No floor plan data."
        fig = self.hypergraph_plotter.simple_plot_generated_mesh(hypergraph_floor_plan)
        out = f"{prefix}_hypergraph_floor_plan.png"
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved floor-plan plot -> %s", out)
        return fig

    def run_complete_workflow(self, file_id=7498, subgraph_index=1):
        """Run the full single-sample demo: load -> subgraph -> hypergraph -> floor plan."""
        logger.info("Starting single-sample workflow  file_id=%d  subgraph=%d",
                     file_id, subgraph_index)

        graph_data = self.load_graph_data(file_id, str(self.data_dir))
        self.plot_graph(graph_data, f"{file_id}_msd_graph.png", "Original Floor Plan Graph")

        subgraphs = self.extract_subgraphs_without_entrance(graph_data, min_nodes=2)
        logger.info("Found %d subgraphs", len(subgraphs))
        for i, sg in enumerate(subgraphs):
            logger.info("  Subgraph %d: %d nodes, %d edges",
                        i, sg.number_of_nodes(), sg.number_of_edges())

        assert 0 <= subgraph_index < len(subgraphs), (
            f"Invalid subgraph index {subgraph_index}. Available: 0-{len(subgraphs)-1}"
        )
        target_subgraph = subgraphs[subgraph_index]
        self.plot_graph(target_subgraph,
                        f"{file_id}_{subgraph_index}_subgraph.png", "Selected Subgraph")

        rotated_subgraph = self.rotate_x_aligned(target_subgraph)
        self.plot_graph(rotated_subgraph,
                        f"{file_id}_{subgraph_index}_rotated.png", "Rotated Floor Plan")

        hypergraph = self.convert_to_hypergraph(rotated_subgraph)
        self.plot_hypergraph(hypergraph, f"{file_id}_{subgraph_index}")

        custom_boundary_points = [
            (0, 0), (0, 8), (10, 8), (10, 3), (12, 3), (12, 0)
        ]
        floor_plan = self.hypergraph_to_floor_plan(hypergraph, custom_boundary_points)
        self.plot_hypergraph_floor_plan(floor_plan, f"{file_id}_{subgraph_index}")

        logger.info("Single-sample workflow completed successfully")
        return graph_data, subgraphs, rotated_subgraph, hypergraph, floor_plan

    def get_info(self, graph_data, subgraphs, target_subgraph, hypergraph, floor_plan):
        """Return a summary dict of the current workflow state."""
        info = {
            'graph_loaded': graph_data is not None,
            'subgraphs_extracted': subgraphs is not None,
            'subgraph_selected': target_subgraph is not None,
            'hypergraph_created': hypergraph is not None,
            'floor_plan_generated': floor_plan is not None,
        }
        if graph_data:
            info['graph_nodes'] = graph_data.number_of_nodes()
            info['graph_edges'] = graph_data.number_of_edges()
        if subgraphs:
            info['num_subgraphs'] = len(subgraphs)
        if target_subgraph:
            info['subgraph_nodes'] = target_subgraph.number_of_nodes()
            info['subgraph_edges'] = target_subgraph.number_of_edges()
        return info


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_and_save_subgraph(subgraph, file_name, subgraph_idx,
                              output_dir, data_type, min_leaves=4):
    """Process one MSD subgraph, validate, and save if successful.

    Returns (success, info, saved_path).
    """
    try:
        rotated_subgraph = MSDGraph.rotate_x_aligned(subgraph)
        access_graph_dict = MSDGraph.generate_access_graph_json(rotated_subgraph)

        hull_data = MSDGraph.get_geometry_based_boundaries(rotated_subgraph)
        if not hull_data or not hull_data.get('convex_hull_geometry'):
            return False, "No convex hull geometry", None

        face = MSDGraph.convex_hull_to_face_wrapper(hull_data['convex_hull_geometry'])
        if not face:
            return False, "Failed to create face wrapper", None

        nmesh, _ = MSDGraph.create_nmesh_from_subgraph(rotated_subgraph)
        if not nmesh:
            return False, "Failed to create nmesh", None

        connectivity_dict, _ = MSDGraph.build_connectivity_dict(rotated_subgraph)

        data_node = DataNodeUtils.CreateDataNodeWithConnectivity(
            nmesh, face, connectivity_dict
        )
        if not data_node:
            return False, "Failed to create data_node", None

        face_count = len(nmesh.faceList)

        try:
            all_children = RGL.DataNode.GetAllChildren(data_node)
            leaf_count = sum(
                1 for node in all_children if hasattr(node, 'final') and node.final
            )
        except Exception as exc:
            return False, f"Failed to count leaves: {exc}", None

        if face_count != leaf_count:
            return False, {
                'face_count': face_count,
                'leaf_count': leaf_count,
                'reason': 'face_count != leaf_count',
            }, None

        if leaf_count < min_leaves:
            return False, {
                'face_count': face_count,
                'leaf_count': leaf_count,
                'reason': f'leaf_count ({leaf_count}) < min_leaves ({min_leaves})',
            }, None

        serialized_text = RGL.DataNode.serializeDataNode(data_node)
        serialized_text = remove_mergeid_suffixes(serialized_text)

        base_name = Path(file_name).stem
        output_filename = f"{data_type}_{base_name}_subgraph_{subgraph_idx}.txt"
        output_path = output_dir / output_filename

        combined = {
            "accessgraph": access_graph_dict,
            "hypergraph": serialized_text,
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(combined, f, indent=2)

        return True, {
            'face_count': face_count,
            'leaf_count': leaf_count,
            'nodes_count': subgraph.number_of_nodes(),
            'edges_count': subgraph.number_of_edges(),
            'output_file': output_filename,
        }, output_path

    except Exception as exc:
        return False, f"Processing error: {exc}", None


def process_all_data(output_dir, min_leaves=4):
    """Iterate over all MSD train/test pickle files and save valid hypergraphs."""
    train_dir = Path("dataset/msd/modified-swiss-dwellings-v2/train/graph_out")
    test_dir = Path("dataset/msd/modified-swiss-dwellings-v2/test/graph_out")

    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        'total_files': 0,
        'total_subgraphs': 0,
        'saved': 0,
        'insufficient_leaves': 0,
        'validation_failures': 0,
        'failed_files': 0,
    }
    results = {'successful': [], 'failed': []}

    for data_type, data_dir in [("train", train_dir), ("test", test_dir)]:
        if not data_dir.exists():
            logger.warning("%s directory not found: %s", data_type, data_dir)
            continue

        files = list(data_dir.glob("*.pickle"))
        logger.info("Processing %s: %d files", data_type, len(files))

        pbar = tqdm(files, desc=f"{data_type}", unit="file", leave=True)
        for file_path in pbar:
            stats['total_files'] += 1
            pbar.set_postfix_str(file_path.name)

            try:
                with open(file_path, 'rb') as f:
                    graph_data = pickle.load(f)

                subgraphs = MSDGraph.extract_subgraphs_without_entrance(graph_data)
                if not subgraphs:
                    continue

                for sub_idx, subgraph in enumerate(subgraphs):
                    stats['total_subgraphs'] += 1
                    if subgraph.number_of_nodes() < 2:
                        continue

                    success, info, saved_path = process_and_save_subgraph(
                        subgraph, file_path.name, sub_idx,
                        output_dir, data_type, min_leaves,
                    )

                    if success:
                        stats['saved'] += 1
                        results['successful'].append({
                            'file': file_path.name,
                            'split': data_type,
                            'subgraph_idx': sub_idx,
                            'saved_to': str(saved_path),
                            **info,
                        })
                    else:
                        if isinstance(info, dict):
                            reason = info.get('reason', '')
                            if 'leaf_count' in reason and 'min_leaves' in reason:
                                stats['insufficient_leaves'] += 1
                            else:
                                stats['validation_failures'] += 1
                        else:
                            stats['validation_failures'] += 1
                        results['failed'].append({
                            'file': file_path.name,
                            'split': data_type,
                            'subgraph_idx': sub_idx,
                            'info': info,
                        })

                pbar.set_postfix_str(
                    f"saved {stats['saved']}/{stats['total_subgraphs']}"
                )

            except Exception as exc:
                stats['failed_files'] += 1
                logger.error("Failed to process %s: %s", file_path.name, exc)

        pbar.close()

    return results, stats


def save_summary_report(results, stats, output_file):
    """Write a human-readable report of the batch run."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MSD HYPERGRAPH PROCESSING REPORT\n")
        f.write("=" * 80 + "\n\n")

        for key, val in stats.items():
            f.write(f"  {key}: {val:,}\n")

        total = stats['total_subgraphs']
        rate = (stats['saved'] / total * 100) if total > 0 else 0
        f.write(f"\n  Save rate: {rate:.2f}%\n\n")

        f.write("-" * 80 + "\n")
        f.write(f"Saved hypergraphs: {len(results['successful'])}\n\n")

        for split_name in ("train", "test"):
            items = [r for r in results['successful'] if r['split'] == split_name]
            if not items:
                continue
            f.write(f"\n{'=' * 80}\n")
            f.write(f"{split_name.upper()} DATA - {len(items)} saved\n")
            f.write(f"{'=' * 80}\n\n")
            for i, r in enumerate(items, 1):
                f.write(
                    f"  {i}. {r['file']}  subgraph={r['subgraph_idx']}  "
                    f"faces={r['face_count']}  leaves={r['leaf_count']}  "
                    f"nodes={r['nodes_count']}  edges={r['edges_count']}\n"
                )

        f.write("\n" + "=" * 80 + "\nEND OF REPORT\n" + "=" * 80 + "\n")

    logger.info("Summary report saved -> %s", output_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def setup_logging(level=logging.INFO, log_file=None):
    fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=level, format=fmt, datefmt='%Y-%m-%d %H:%M:%S')
    global logger
    logger = logging.getLogger(__name__)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(fmt))
        logger.addHandler(fh)
    return logger


def parse_args():
    parser = argparse.ArgumentParser(
        description='MSD to Hypergraph conversion (single-sample or batch)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m msd.msd_to_hypergraph --file-id 7498 --subgraph-index 1    # single demo
  python -m msd.msd_to_hypergraph --batch                               # batch all
  python -m msd.msd_to_hypergraph --batch --min-leaves 5 --verbose      # batch + options
        """,
    )

    # Single-sample options
    parser.add_argument('--file-id', type=int, default=7498,
                        help='MSD file ID for single-sample mode (default: 7498)')
    parser.add_argument('--subgraph-index', type=int, default=1,
                        help='Subgraph index for single-sample mode (default: 1)')
    parser.add_argument('--data-dir', type=str,
                        default="dataset/msd/modified-swiss-dwellings-v2/train/graph_out",
                        help='MSD data directory')

    # Batch options
    parser.add_argument('--batch', action='store_true',
                        help='Run batch processing over all train/test data')
    parser.add_argument('--output-dir', '-o', type=str,
                        default='training_data/extracted_hypergraph_msd',
                        help='Output directory for batch mode')
    parser.add_argument('--min-leaves', '-m', type=int, default=4,
                        help='Minimum leaves to save (batch mode, default: 4)')
    parser.add_argument('--summary-file', '-s', type=str,
                        default='msd_hypergraph_summary.txt',
                        help='Summary report output file')

    # Logging
    parser.add_argument('--log-file', type=str, default='msd_to_hypergraph.log')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--quiet', '-q', action='store_true')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    else:
        level = logging.INFO

    setup_logging(level=level, log_file=args.log_file)

    if args.batch:
        # ---- batch mode ----
        logger.info("MSD -> Hypergraph batch conversion started")
        logger.info("  output_dir = %s", args.output_dir)
        logger.info("  min_leaves = %d", args.min_leaves)

        output_dir = Path(args.output_dir)
        results, stats = process_all_data(output_dir, min_leaves=args.min_leaves)

        logger.info("=" * 60)
        logger.info("RESULTS")
        for k, v in stats.items():
            logger.info("  %s: %s", k, f"{v:,}")
        total = stats['total_subgraphs']
        rate = (stats['saved'] / total * 100) if total > 0 else 0
        logger.info("  save_rate: %.2f%%", rate)
        logger.info("=" * 60)

        save_summary_report(results, stats, Path(args.summary_file))
    else:
        # ---- single-sample demo mode ----
        converter = MSDHypergraph(data_dir=args.data_dir)

        graph_data, subgraphs, target_subgraph, hypergraph, floor_plan = (
            converter.run_complete_workflow(
                file_id=args.file_id,
                subgraph_index=args.subgraph_index,
            )
        )

        info = converter.get_info(
            graph_data, subgraphs, target_subgraph, hypergraph, floor_plan
        )
        logger.info("Workflow info:")
        for key, value in info.items():
            logger.info("  %s: %s", key, value)


if __name__ == "__main__":
    main()
