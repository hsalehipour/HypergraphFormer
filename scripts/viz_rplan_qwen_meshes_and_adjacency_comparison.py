#!/usr/bin/env python3
"""4-panel comparison PNGs for the qwen-accessgraphs rotated RPLAN run.

For each selected sample id we render a single composite figure:

    +-----------------------------+-----------------------------+
    |   Predicted mesh            |   GT mesh                   |
    |   (rotated hypergraph       |   (room_polygons + bounds)  |
    |    meshed against bounds)   |                             |
    +-----------------------------+-----------------------------+
    |   Predicted adjacency graph |   GT adjacency graph        |
    |   (Hypergraph.adjacency_    |   (adjacency_graph_from_    |
    |    graph(bounds))           |    polygons(room_polygons)) |
    +-----------------------------+-----------------------------+

Defaults are pinned to the same run as
``scripts/viz_rplan_qwen_accessgraphs_meshes.py``:

  --hypergraphs-json:
      inference_qwen_diffplanner/inference_8799_on_eval_test_qwen_accessgraphs_vllm_64_run_1/
        rotate/results_hypergraphs_orientation.json
  --bounds-dir:
      dataset/rplan/eval_test
  --output-dir:
      <hypergraphs-json's parent>/viz_meshes_comparison

By default we sample 20 random ids that exist in BOTH the hypergraphs JSON
and the eval_test dir. Pass ``--seed`` for reproducibility, ``--num`` to
change the count, or ``--sample-ids`` to pin specific ids.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.hypergraph import Hypergraph, HypergraphPlotter, repair_bsp_tree_names
from utils.visualization_helpers import adjacency_graph as _hg_adjacency_graph, adjacency_graph_from_polygons

# Reuse the GT-mesh build + access-graph drawing primitives so the look
# matches the rest of the tooling exactly.
from scripts.viz_rplan_gt_meshes import (
    _build_gt_mesh,
    _door_pairs_from_accessgraph,
    _pick_entry_face_idx,
    _load_gt_sample,
)
from scripts.viz_accessgraph_three_column_grid import _draw_ag_on_axis

logger = logging.getLogger(__name__)

DEFAULT_HYPERGRAPHS = (
    REPO_ROOT
    / "inference_qwen_diffplanner"
    / "inference_8799_on_eval_test_qwen_accessgraphs_vllm_64_run_1"
    / "rotate"
    / "results_hypergraphs_orientation.json"
)
DEFAULT_BOUNDS_DIR = REPO_ROOT / "dataset" / "rplan" / "eval_test"


def _natural_id_key(sid: str) -> tuple[int, int, str]:
    """Sort by trailing integer when possible (so '10' comes after '2')."""
    m = re.search(r"(\d+)\s*$", sid)
    if m:
        return (0, int(m.group(1)), sid)
    return (1, 0, sid)


def _draw_pred_mesh(ax, hg_dict: dict, bounds: dict) -> None:
    """Render the predicted mesh (with doors / entry-door) on *ax*."""
    plotter = HypergraphPlotter()
    polygon = [(c["X"], c["Y"]) for c in (bounds or {}).get("corners", [])]
    try:
        hg_obj = Hypergraph(hg_dict)
        mesh = hg_obj.to_mesh(bounds=bounds)
        door_pairs = hg_obj.mesh_door_face_pairs(mesh, bounds=bounds)
        entry_idx = hg_obj.mesh_entry_door_face_idx(mesh)
        plotter.plot_mesh_on_axis(
            mesh, ax,
            boundary_polygon=polygon,
            show_title=False, show_area=False, show_frame=False,
            door_pairs=door_pairs, entry_door_face_idx=entry_idx,
        )
        ax.invert_yaxis()
    except Exception as exc:  # noqa: BLE001
        ax.text(0.5, 0.5, f"pred mesh failed:\n{exc}",
                ha="center", va="center", transform=ax.transAxes,
                wrap=True, fontsize=9, color="#a00")
        ax.set_axis_off()


def _draw_gt_mesh(ax, gt_sample: dict) -> None:
    """Render the GT mesh built from ``room_polygons`` on *ax*."""
    plotter = HypergraphPlotter()
    polygon = [(c["X"], c["Y"]) for c in (gt_sample["bounds"] or {}).get("corners", [])]
    try:
        mesh, rid_to_idx = _build_gt_mesh(gt_sample["rooms"])
        door_pairs = _door_pairs_from_accessgraph(
            gt_sample.get("accessgraph") or {}, rid_to_idx,
        )
        entry_idx = _pick_entry_face_idx(gt_sample["rooms"])
        plotter.plot_mesh_on_axis(
            mesh, ax,
            boundary_polygon=polygon,
            show_title=False, show_area=False, show_frame=False,
            door_pairs=door_pairs, entry_door_face_idx=entry_idx,
        )
        ax.invert_yaxis()
    except Exception as exc:  # noqa: BLE001
        ax.text(0.5, 0.5, f"gt mesh failed:\n{exc}",
                ha="center", va="center", transform=ax.transAxes,
                wrap=True, fontsize=9, color="#a00")
        ax.set_axis_off()


def _compute_one(
    hg_dict: dict, bounds: dict, gt_polygons: list,
) -> tuple[dict, dict]:
    """Build (pred_adj_graph, gt_adj_graph) for one sample."""
    try:
        pred_adj = _hg_adjacency_graph(Hypergraph(hg_dict), bounds)
    except Exception as exc:  # noqa: BLE001
        logger.warning("pred adjacency_graph failed: %s", exc)
        pred_adj = {"nodes": {}, "edges": []}

    try:
        gt_adj = adjacency_graph_from_polygons(gt_polygons)
    except Exception as exc:  # noqa: BLE001
        logger.warning("gt adjacency_graph_from_polygons failed: %s", exc)
        gt_adj = {"nodes": {}, "edges": []}

    return pred_adj, gt_adj


def render_one_composite(
    sid: str, hg_dict: dict, gt_sample: dict, output_path: Path,
) -> None:
    """Render one 4-panel comparison."""
    bounds = gt_sample["bounds"]

    pred_adj, gt_adj = _compute_one(
        hg_dict, bounds, gt_sample["rooms"],
    )

    plotter = HypergraphPlotter()

    # 2x2 layout. Use constrained_layout=False + a reserved suptitle band so
    # the matplotlib bbox heuristics don't try to shrink the meshes when the
    # title text is long.
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 12),
        gridspec_kw=dict(left=0.02, right=0.98, top=0.92, bottom=0.02,
                         wspace=0.05, hspace=0.10),
        squeeze=False,
    )

    _draw_pred_mesh(axes[0, 0], hg_dict, bounds)
    _draw_gt_mesh(axes[0, 1], gt_sample)

    _draw_ag_on_axis(
        pred_adj, axes[1, 0], plotter,
        placeholder=None if pred_adj.get("nodes") else "(no pred adjacency)",
    )
    _draw_ag_on_axis(
        gt_adj, axes[1, 1], plotter,
        placeholder=None if gt_adj.get("nodes") else "(no gt adjacency)",
    )

    # Per-panel titles via figure-relative text (set_title on axis-off panels
    # can interact badly with bbox math; we already saw that explode the
    # canvas in the 3-column figure). 4 column-panel labels + the global
    # title.
    panel_titles = [
        ("Pred mesh", 0.27, 0.945),
        ("GT mesh",   0.77, 0.945),
        ("Pred adjacency graph", 0.27, 0.485),
        ("GT adjacency graph",   0.77, 0.485),
    ]
    for label, x, y in panel_titles:
        fig.text(x, y, label, ha="center", va="bottom",
                 fontsize=12, fontweight="bold")

    fig.suptitle(f"sample {sid}",
                 fontsize=14, fontweight="bold", y=0.985)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _shared_ids(hg_keys: set[str], bounds_dir: Path) -> list[str]:
    """Ids that exist in BOTH the hypergraphs JSON and the eval_test dir."""
    have = {fp.stem for fp in bounds_dir.glob("*.txt")}
    return sorted(hg_keys & have, key=_natural_id_key)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hypergraphs-json", type=Path, default=DEFAULT_HYPERGRAPHS,
                   help=f"Rotated hypergraphs JSON (default: {DEFAULT_HYPERGRAPHS}).")
    p.add_argument("--bounds-dir", type=Path, default=DEFAULT_BOUNDS_DIR,
                   help=f"Directory of <id>.txt with bounds + room_polygons "
                        f"(default: {DEFAULT_BOUNDS_DIR}).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output dir for composite PNGs "
                        "(default: <hypergraphs-json parent>/viz_meshes_comparison).")
    p.add_argument("--num", type=int, default=20,
                   help="Number of random samples (default: 20).")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed (default: 0).")
    p.add_argument("--sample-ids", nargs="*", default=None,
                   help="Explicit list of ids; bypasses random sampling.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-render existing PNGs (default: skip).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    hg_path = args.hypergraphs_json.expanduser().resolve()
    bounds_dir = args.bounds_dir.expanduser().resolve()
    out_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (hg_path.parent / "viz_meshes_comparison").resolve()
    )
    if not hg_path.is_file():
        p.error(f"hypergraphs JSON not found: {hg_path}")
    if not bounds_dir.is_dir():
        p.error(f"bounds dir not found: {bounds_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(hg_path, encoding="utf-8") as f:
        all_hg: dict = json.load(f)
    logger.info("Loaded %d hypergraphs from %s", len(all_hg), hg_path.name)
    logger.info("Output dir: %s", out_dir)

    common = _shared_ids({str(k) for k in all_hg.keys()}, bounds_dir)
    if args.sample_ids:
        ids = [str(s) for s in args.sample_ids]
        missing = [s for s in ids if s not in set(common)]
        if missing:
            logger.warning("ids not in intersection (still attempted): %s", missing[:10])
    else:
        if not common:
            p.error("no overlap between --hypergraphs-json keys and --bounds-dir ids")
        rng = random.Random(args.seed)
        n = max(1, min(int(args.num), len(common)))
        ids = rng.sample(common, n)
        ids.sort(key=_natural_id_key)

    logger.info("Rendering %d composite(s): %s", len(ids), ids)

    rendered = 0
    failed = 0
    skipped_existing = 0
    started = time.time()
    summary_lines: list[str] = []
    for sid in ids:
        out_path = out_dir / f"{sid}.png"
        if out_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue
        if sid not in all_hg:
            logger.warning("[%s] not in hypergraphs JSON; skipping", sid)
            failed += 1
            continue
        gt_sample = _load_gt_sample(bounds_dir / f"{sid}.txt")
        if gt_sample is None:
            logger.warning("[%s] missing/invalid GT in %s; skipping", sid, bounds_dir)
            failed += 1
            continue
        try:
            hg_dict = repair_bsp_tree_names(json.loads(json.dumps(all_hg[sid])))
            render_one_composite(sid, hg_dict, gt_sample, out_path)
            summary_lines.append(sid)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] composite render failed: %s", sid, exc)
            failed += 1
            continue
        rendered += 1
        if rendered % 5 == 0:
            logger.info(
                "  rendered %d/%d (skipped_existing=%d, failed=%d) in %.1fs",
                rendered, len(ids), skipped_existing, failed, time.time() - started,
            )

    if summary_lines:
        summary_path = out_dir / "summary.txt"
        summary_path.write_text("\n".join(summary_lines) + "\n")
        logger.info("Wrote rendered sample list: %s", summary_path)

    logger.info(
        "Done. rendered=%d, skipped_existing=%d, failed=%d. Output: %s",
        rendered, skipped_existing, failed, out_dir,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
