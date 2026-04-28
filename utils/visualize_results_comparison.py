#!/usr/bin/env python3
"""Render per-sample comparison figures for generated hypergraphs.

Run the CLI via ``python3 scripts/visualize_results_comparison.py ...`` (same pattern as
``scripts/procedural_edits.py``) so the same venv / RGL install as meshing jobs is used.

Creates one PNG per sample (2×3 grid):
1. Ground-truth access graph
2. Ground-truth mesh
3. Generated access graph (leaf connectivity)
4. Generated mesh
5. Corrected access graph (:meth:`~utils.hypergraph.Hypergraph.correct_accessgraph`)
6. Same generated mesh (aligned under corrected graph)

Generated hypergraphs are passed through ``repair_bsp_tree_names`` before
plotting (in-memory only; the results JSON file is not modified).

By default this script targets the checkpoint-5256 RPLAN results discussed in
the current workflow and writes images into ``checkpoint-5256-viz/`` at the
repository root.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt

from utils.hypergraph import Hypergraph, HypergraphPlotter, repair_bsp_tree_names
from utils.metrics import accessgraph_dict_to_nx


LOGGER = logging.getLogger(__name__)

DEFAULT_RESULTS = REPO_ROOT / "results_trained_checkpoint-5256_42_hypergraphs.json"
DEFAULT_TEST_DIR = REPO_ROOT / "dataset" / "rplan" / "test"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "checkpoint-5256-viz"


def _parse_jsonish(value):
    return json.loads(value) if isinstance(value, str) else value


def load_ground_truth(test_dir: Path) -> dict[str, dict]:
    ground_truth: dict[str, dict] = {}
    for fp in sorted(test_dir.glob("*.txt")):
        with open(fp) as f:
            raw = json.load(f)
        ground_truth[fp.stem] = {
            "accessgraph": raw.get("accessgraph", {}),
            "hypergraph": _parse_jsonish(raw.get("hypergraph", "{}")),
            "bounds": _parse_jsonish(raw.get("bounds", "{}")),
        }
    return ground_truth


def boundary_polygon(bounds: dict) -> list[tuple[float, float]]:
    corners = (bounds or {}).get("corners", [])
    return [(float(c["X"]), float(c["Y"])) for c in corners]


def access_graph_with_outside(hg: Hypergraph, base_ag: dict) -> dict:
    """Augment a room-only access graph with an explicit outside node."""
    ag = {
        "nodes": dict((base_ag or {}).get("nodes", {}) or {}),
        "edges": [list(e) for e in ((base_ag or {}).get("edges", []) or [])],
    }

    leaves = hg.leaves
    if not leaves:
        return ag

    leaf_name_to_id = {
        leaf["name"]: str(i)
        for i, leaf in enumerate(leaves, start=1)
    }
    outside_edges = []
    for leaf in leaves:
        if any(str(conn).strip().lower() == "outside" for conn in (leaf.get("connected") or [])):
            lid = leaf_name_to_id.get(leaf["name"])
            if lid is not None:
                outside_edges.append(lid)

    if not outside_edges:
        return ag

    numeric_ids = [int(k) for k in ag["nodes"].keys() if str(k).isdigit()]
    outside_id = str(max(numeric_ids, default=0) + 1)
    ag["nodes"][outside_id] = "outside"

    existing = {
        tuple(sorted((str(e[0]), str(e[1])), key=lambda x: (not x.isdigit(), x)))
        for e in ag["edges"]
        if isinstance(e, (list, tuple)) and len(e) == 2
    }
    for lid in sorted(set(outside_edges), key=int):
        edge = tuple(sorted((lid, outside_id), key=lambda x: int(x)))
        if edge not in existing:
            ag["edges"].append([edge[0], edge[1]])
            existing.add(edge)
    return ag


def _corrected_access_graph_for_viz(gen_hg: Hypergraph, bounds: dict) -> dict | None:
    """``correct_accessgraph`` plus outside-node augmentation for plotting; ``None`` on failure."""
    if not (isinstance(bounds, dict) and bounds.get("corners")):
        return None
    try:
        corr = gen_hg.correct_accessgraph(bounds)
    except Exception:
        return None
    if not corr or not corr.get("nodes"):
        return None
    return access_graph_with_outside(gen_hg, corr)


def _mesh_failure_caption(exc: BaseException, *, label: str = "Mesh") -> str:
    """Short text for PNG panels when :meth:`~Hypergraph.Hypergraph.to_mesh` fails."""
    msg = str(exc)
    base = f"{label} failed:\n{msg}"
    if "hypergraph" in msg.lower() and (
        "not a package" in msg
        or "hypergraph.api" in msg
        or "no module named 'hypergraph" in msg.lower()
    ):
        return (
            f"{label} failed (RGL Python package):\n"
            "Install the RGL interop that provides `from hypergraph.api.lib.tools import RGL` "
            "(this is unrelated to `utils/hypergraph.py`).\n"
            "If only `hypergraph-*.dist-info` exists without a `hypergraph/` folder in "
            "site-packages, reinstall that wheel. Or run this script with --no-mesh.\n\n"
            f"Detail: {msg}"
        )
    if "Failed to create a default .NET runtime" in msg or "libmono" in msg.lower():
        return (
            f"{label} failed (.NET / Mono):\n"
            "pythonnet needs Mono or .NET; set PYTHONNET_* or install Mono.\n\n"
            f"Detail: {msg}"
        )
    return base


def render_sample(
    sample_id: str,
    gt: dict,
    generated_hg_data: dict,
    output_path: Path,
    *,
    no_mesh: bool = False,
) -> None:
    """Render comparison PNG.

    For **generated** room meshes, use the *same* pipeline as
    :func:`scripts.score_results.score_delta` and
    :func:`scripts.score_results.score_area_proportion_error`: an unmodified
    ``Hypergraph(generated)`` and ``.to_mesh(bounds=gt['bounds'])`` (no
    :func:`repair_bsp_tree_names` before meshing). Access-graph panels still use
    ``repair_bsp_tree_names`` so node names match a consistent eval-friendly tree.

    If *no_mesh* is true, only the access-graph row is drawn (no RGL / ``hypergraph.api``),
    and the corrected-access column is skipped because it requires meshing.
    """
    import networkx as nx

    plotter = HypergraphPlotter()

    gt_hg = Hypergraph(gt["hypergraph"])
    # Match score_results / "delta" and "area_proportion_error" — raw dict for mesh
    gen_hg_mesh = Hypergraph(copy.deepcopy(generated_hg_data))
    # Repaired tree for 2D access-graph plots and for correct_accessgraph
    gen_hg = Hypergraph(repair_bsp_tree_names(generated_hg_data))
    gt_ag = access_graph_with_outside(gt_hg, gt.get("accessgraph", {}) or {})
    gen_ag = access_graph_with_outside(gen_hg, gen_hg.access_graph)
    bounds = gt.get("bounds", {}) or {}
    polygon = boundary_polygon(bounds)
    if no_mesh:
        gen_ag_corr = None
    else:
        gen_ag_corr = _corrected_access_graph_for_viz(gen_hg, bounds)

    if no_mesh:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
        ax_gt_ag, ax_gen_ag, ax_gen_corr = axes
    else:
        fig, axes = plt.subplots(2, 3, figsize=(24, 12))
        ax_gt_ag, ax_gen_ag, ax_gen_corr = axes[0]
        ax_gt_mesh, ax_gen_mesh, ax_gen_mesh_b = axes[1]

    fig.suptitle(sample_id, fontsize=14, fontweight="bold")

    plotter.plot_access_graph_on_axis(
        gt_ag,
        ax_gt_ag,
        title="Ground Truth Access Graph",
        components=list(nx.connected_components(accessgraph_dict_to_nx(gt_ag))),
        node_labels=plotter.make_distinct_labels(gt_ag),
    )

    plotter.plot_access_graph_on_axis(
        gen_ag,
        ax_gen_ag,
        title="Generated Access Graph",
        components=list(nx.connected_components(accessgraph_dict_to_nx(gen_ag))),
        node_labels=plotter.make_distinct_labels(gen_ag),
    )

    if gen_ag_corr is not None:
        plotter.plot_access_graph_on_axis(
            gen_ag_corr,
            ax_gen_corr,
            title="Generated (correct_accessgraph)",
            components=list(
                nx.connected_components(accessgraph_dict_to_nx(gen_ag_corr))
            ),
            node_labels=plotter.make_distinct_labels(gen_ag_corr),
        )
    else:
        corr_msg = (
            "correct_accessgraph\nskipped\n(--no-mesh: needs RGL)"
            if no_mesh
            else "correct_accessgraph\nunavailable\n(no bounds or mesh failed)"
        )
        ax_gen_corr.text(
            0.5,
            0.5,
            corr_msg,
            ha="center",
            va="center",
            transform=ax_gen_corr.transAxes,
            fontsize=10,
        )
        ax_gen_corr.set_title("Generated (correct_accessgraph)")
        ax_gen_corr.axis("off")

    if no_mesh:
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    try:
        gt_mesh = gt_hg.to_mesh(bounds=bounds)
        plotter.plot_mesh_on_axis(
            gt_mesh,
            ax_gt_mesh,
            boundary_polygon=polygon,
            title="Ground Truth Mesh",
            face_labels=gt_hg.mesh_face_labels(
                gt_mesh, plotter.make_distinct_labels(gt_ag)
            ),
        )
        ax_gt_mesh.invert_yaxis()
    except Exception as exc:
        ax_gt_mesh.text(
            0.5,
            0.5,
            _mesh_failure_caption(exc, label="GT mesh"),
            ha="center",
            va="center",
            transform=ax_gt_mesh.transAxes,
            wrap=True,
        )
        ax_gt_mesh.set_title("Ground Truth Mesh")
        ax_gt_mesh.axis("off")

    try:
        gen_mesh = gen_hg_mesh.to_mesh(bounds=bounds)
        ag_for_mesh = access_graph_with_outside(
            gen_hg_mesh, gen_hg_mesh.access_graph
        )
        distinct_gen = plotter.make_distinct_labels(ag_for_mesh)
        mesh_lbls = gen_hg_mesh.mesh_face_labels(gen_mesh, distinct_gen)
        plotter.plot_mesh_on_axis(
            gen_mesh,
            ax_gen_mesh,
            boundary_polygon=polygon,
            title="Generated Mesh",
            face_labels=mesh_lbls,
        )
        plotter.plot_mesh_on_axis(
            gen_mesh,
            ax_gen_mesh_b,
            boundary_polygon=polygon,
            title="Generated Mesh (same)",
            face_labels=mesh_lbls,
        )
        ax_gen_mesh.invert_yaxis()
        ax_gen_mesh_b.invert_yaxis()
    except Exception as exc:
        cap = _mesh_failure_caption(exc, label="Generated mesh")
        ax_gen_mesh.text(
            0.5,
            0.5,
            cap,
            ha="center",
            va="center",
            transform=ax_gen_mesh.transAxes,
            wrap=True,
        )
        ax_gen_mesh.set_title("Generated Mesh")
        ax_gen_mesh.axis("off")
        ax_gen_mesh_b.text(
            0.5,
            0.5,
            cap,
            ha="center",
            va="center",
            transform=ax_gen_mesh_b.transAxes,
            wrap=True,
        )
        ax_gen_mesh_b.set_title("Generated Mesh (same)")
        ax_gen_mesh_b.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _results_key_for_id(results: dict, sid: str):
    """Return the key present in *results* for string id *sid* (str or int JSON keys)."""
    if sid in results:
        return sid
    try:
        n = int(sid, 10)
        if n in results:
            return n
    except ValueError:
        pass
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Render GT vs generated hypergraph comparison PNGs")
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="ID",
        help="Render only these sample IDs, in this order (e.g. --only 0 1 103)",
    )
    parser.add_argument(
        "--no-mesh",
        action="store_true",
        help="Only draw access graphs (1×3). Skips RGL and correct_accessgraph (both need "
        "hypergraph.api). Use when the RGL interop is not installed in this environment.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    with open(args.results_file) as f:
        results = json.load(f)

    ground_truth = load_ground_truth(args.test_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.only is not None:
        id_and_key: list[tuple[str, object]] = []
        for raw in args.only:
            sid = str(raw)
            rk = _results_key_for_id(results, sid)
            if rk is None:
                LOGGER.warning("Sample %s not in results file, skipping", sid)
                continue
            id_and_key.append((sid, rk))
        sample_ids = id_and_key
        if args.max_samples is not None:
            LOGGER.warning("--max-samples ignored when --only is set")
    else:
        sample_ids = sorted(results.keys())
        if args.max_samples is not None:
            sample_ids = sample_ids[: args.max_samples]
        sample_ids = [(str(s), s) for s in sample_ids]

    if args.no_mesh:
        LOGGER.info(
            "RGL/mesh disabled (--no-mesh): access graphs only, no correct_accessgraph column data"
        )
    LOGGER.info("Rendering %d samples into %s", len(sample_ids), args.output_dir)

    rendered = 0
    skipped = 0
    for sample_id, result_key in sample_ids:
        gt = ground_truth.get(sample_id)
        if gt is None:
            LOGGER.warning("Skipping %s: missing ground truth", sample_id)
            skipped += 1
            continue

        try:
            render_sample(
                sample_id,
                gt,
                results[result_key],
                args.output_dir / f"{sample_id}.png",
                no_mesh=args.no_mesh,
            )
            rendered += 1
        except Exception as exc:
            LOGGER.warning("Skipping %s: %s", sample_id, exc)
            skipped += 1

    LOGGER.info("Done. Rendered %d, skipped %d", rendered, skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
