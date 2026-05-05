#!/usr/bin/env python3
"""
For one sample, **rotate/reflect the footprint** (``bounds['corners']`` in 2D about
the polygon centroid; see :func:`scripts.procedural_edits.transform_bounds_polygon`)
for each reflect mode × {0,90,180,270}. The **hypergraph** is **not** reoriented—only
the boundary passed to ``to_mesh`` changes—so the floor appears in a different
on-page orientation in each panel.

**Modes** (``--mode``): **bounds** = rotate/reflect the footprint, prediction BSP
unchanged. **bsp** = reorient the prediction BSP per panel, **canonical** footprint.
**bounds2** = the **full 3×4 grid of BSP reorients** (same as **bsp**: pred
``reorient_hypergraph_bsp`` per (mirror,°)), but the footprint is **one fixed** extra
rotation: ``transform_bounds_polygon`` with the **same** CCW **+90°** (and no extra flip) for
**every** cell, so the floor outline is always at +90° from the dataset while pred varies.
**GT** in each cell uses that same +90° bounds. The first column is still **GT in the original**
(dataset) frame. The
prediction grid is **3 rows × 4 columns**: rows are mirror mode (no mirror, x flip,
y flip) and columns are CCW rotation (0°, 90°, 180°, 270°).

In ``--mode bounds``, APE/δ compare the prediction meshed in each **rotated** footprint
to the GT mesh in the **original** bounds (same convention as ``score_results``; the
left column GT is already drawn in the original frame). In ``bsp`` / ``bounds2``,
``to_mesh`` for both branches uses the same per-cell ``mesh_bounds``.

Metrics match ``scripts/score_results.py`` (area proportion error + room-shape
delta). The **lowest** δ (room similarity) is highlighted in green on pred panels.

With ``--no-gt``, **no GT hypergraph** is loaded (only ``bounds.corners`` is required—the
usual case for eval splits like ``eval_test`` that omit ``hypergraph``). Each pred cell is
then scored by **δ_sq** (square compactness, same as ``postprocessing --pp-mode orientation``),
and green marks the lowest δ_sq.

The **first** left column stacks **three** panels: **access_graph** (BSP leaf connectivity),
**correct_accessgraph** (mesh adjacency; may be blank if it fails), then the mesh. Each **grid
cell** has the same **three** panels left-to-right: generated **access_graph**, **correct_accessgraph**,
then footprint mesh. **Pink** highlights cells where the variant **mesh** does not satisfy
every edge of the left-column **input** ``correct_accessgraph`` (:func:`~scripts.utils_parametric.check_accessgraph_edges_against_mesh`, same as parametric optimization).

Use ``--mode bounds`` | ``bsp`` | ``bounds2`` (default: bounds). Legacy ``--bsp-reorient`` is
the same as ``--mode bsp``.

Requires RGL / meshing and ``bounds.corners`` in the test ``.txt`` file.

Default output (no ``--output``)::

  <repo>/inference_8700_on_wmr24_inference_vllm_64/explore_rotation/<sample_id>_explore_rotation.png

Example:
  /path/to/.venv/bin/python scripts/explore_rotation.py \\
    --mode bounds \\
    --sample-id b-0120 \\
    --results-hypergraphs inference_8700/.../results_hypergraphs.json \\
    --test-dir dataset/wmr24/test
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.gridspec import GridSpecFromSubplotSpec

from scripts.postprocessing import orientation_variant_passes_ref_accessgraph
from scripts.procedural_edits import reorient_hypergraph_bsp, transform_bounds_polygon
from scripts.utils_parametric import _delta_metric_square_from_room_list
from scripts.visualize_procedural_edited import load_eval_access_gt
from utils.hypergraph import Hypergraph, HypergraphPlotter, repair_bsp_tree_names
from utils.metrics import (
    accessgraph_dict_to_nx,
    compute_room_area_proportion_error,
    compute_room_similarity_score,
    produce_room_area_list,
    produce_room_list,
)

logger = logging.getLogger(__name__)

_DEFAULT_EXPLORE_ROTATION_DIR = (
    REPO_ROOT
    / "inference_8700_on_wmr24_inference_vllm_64"
    / "explore_rotation"
)

_REFLECT_MODES: tuple[tuple[str | None, str], ...] = (
    (None, "no_flip"),
    ("x", "flip_x"),
    ("y", "flip_y"),
)
_ROT_DEGS = (0, 90, 180, 270)

# ``transform_bounds_polygon`` reflect code -> :func:`reorient_hypergraph_bsp` *reflect_mode* string
_REFLECT_CODE_TO_REORIENT_MODE: dict[str | None, str] = {
    None: "normal",
    "x": "x",
    "y": "y",
}

# bounds2: fixed extra CCW rotation on the footprint (same for all 12 panels; no mirror)
_BOUNDS2_FOOTPRINT_EXTRA_DEG = 90.0

# Row label for the 3×4 pred grid (left annotation)
_REFLECT_ROW_TITLE: dict[str, str] = {
    "no_flip": "no mirror",
    "flip_x": "x flip",
    "flip_y": "y flip",
}


def _plot_mesh(
    ax,
    plotter: HypergraphPlotter,
    hg: dict,
    bounds: dict,
    title: str,
    sample_id: str,
) -> None:
    if not (bounds and bounds.get("corners")):
        ax.text(0.5, 0.5, "No bounds", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return
    try:
        hg_m = Hypergraph(hg)
        mesh = hg_m.to_mesh(bounds=bounds)
        polygon = [(c["X"], c["Y"]) for c in bounds.get("corners", [])]
        ag = hg_m.access_graph
        ag_rooms = {
            "nodes": {k: v for k, v in ag.get("nodes", {}).items() if k != "outside"},
            "edges": [e for e in ag.get("edges", []) if "outside" not in e],
        }
        distinct = plotter.make_distinct_labels(ag_rooms)
        if "outside" in ag.get("nodes", {}):
            distinct["outside"] = "Outside"
        face_labels = hg_m.mesh_face_labels(mesh, distinct=distinct)
        plotter.plot_mesh_on_axis(
            mesh,
            ax,
            boundary_polygon=polygon,
            title=title,
            face_labels=face_labels,
        )
        ax.set_title(title, fontsize=7, fontweight="normal")
        ax.invert_yaxis()
    except Exception as exc:  # noqa: BLE001
        logger.warning("mesh failed %s: %s", title, exc)
        ax.text(
            0.5,
            0.5,
            f"Mesh failed:\n{exc}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=7,
        )
        ax.axis("off")


def _plot_access_graph_panel(
    ax,
    plotter: HypergraphPlotter,
    ag: dict | None,
    title: str,
    *,
    empty_stub: str = "access graph\n(empty)",
) -> None:
    """Draw ``{nodes,edges}`` access graph (BSP ``access_graph`` or ``correct_accessgraph``)."""
    import networkx as nx

    if not isinstance(ag, dict):
        ag = {}
    nodes = ag.get("nodes") or {}
    edges = ag.get("edges") or []
    if not nodes and not edges:
        ax.text(
            0.5,
            0.5,
            empty_stub,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.axis("off")
        return
    ag_dict = {
        "nodes": {str(k): v for k, v in nodes.items()},
        "edges": [
            [str(e[0]), str(e[1])]
            for e in edges
            if isinstance(e, (list, tuple)) and len(e) == 2
        ],
    }
    try:
        g = accessgraph_dict_to_nx(ag_dict)
        plotter.plot_access_graph_on_axis(
            ag_dict,
            ax,
            title=title,
            components=list(nx.connected_components(g)),
            node_labels=plotter.make_distinct_labels(ag_dict),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("access graph plot failed: %s", exc)
        ax.text(
            0.5,
            0.5,
            f"Access graph plot:\n{exc}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=7,
        )
        ax.axis("off")



def run() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample-id", required=True)
    p.add_argument("--results-hypergraphs", type=Path, required=True)
    p.add_argument("--test-dir", type=Path, required=True)
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG. Default: "
        f"{_DEFAULT_EXPLORE_ROTATION_DIR}/<sample_id>_explore_rotation.png (parent dirs created).",
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Log warnings/errors only",
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=("bounds", "bsp", "bounds2"),
        default="bounds",
        help=(
            "bounds: transform footprint per (reflect,°); same pred BSP. "
            "bsp: reorient pred with reorient_hypergraph_bsp; canonical footprint. "
            "bounds2: full bsp 3×4 reorients, but all panels use the same +90° footprint (fixed)."
        ),
    )
    p.add_argument(
        "--bsp-reorient",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--no-gt",
        action="store_true",
        help=(
            "Do not load GT hypergraph from test .txt; only bounds.corners are used. "
            "Scores δ_sq like postprocessing orientation; left column = input prediction mesh."
        ),
    )
    a = p.parse_args()
    if a.bsp_reorient:
        a.mode = "bsp"
    if a.output is None:
        a.output = _DEFAULT_EXPLORE_ROTATION_DIR / f"{a.sample_id}_explore_rotation.png"

    logging.basicConfig(
        level=logging.WARNING if a.quiet else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    with open(a.results_hypergraphs) as f:
        results: dict = json.load(f)
    if a.sample_id not in results:
        print(f"Unknown sample_id {a.sample_id!r}", file=sys.stderr)
        return 2

    b2, gt_hg, _gt_ag = load_eval_access_gt(a.test_dir, a.sample_id)
    if not (b2 and isinstance(b2, dict) and b2.get("corners")):
        print("Test record missing bounds.corners; cannot build meshes.", file=sys.stderr)
        return 2

    use_gt = not a.no_gt
    if use_gt:
        if not (gt_hg and isinstance(gt_hg, dict) and gt_hg.get("name")):
            print("Test record missing hypergraph; cannot compare.", file=sys.stderr)
            return 2
        try:
            _ = Hypergraph(gt_hg).to_mesh(bounds=b2)
        except Exception as exc:  # noqa: BLE001
            print(f"GT mesh (canonical bounds) failed: {exc}", file=sys.stderr)
            return 2

    bounds = b2
    base = repair_bsp_tree_names(copy.deepcopy(results[a.sample_id]))

    ref_generated_ag: dict | None = None
    try:
        ref_generated_ag = Hypergraph(copy.deepcopy(base)).access_graph
    except Exception:
        ref_generated_ag = None

    ref_ag: dict | None = None
    try:
        ref_ag = Hypergraph(copy.deepcopy(base)).correct_accessgraph(bounds)
    except Exception:
        ref_ag = None

    reflect_modes = _REFLECT_MODES
    # bounds2: one footprint for every cell (90° CCW, no reflect)
    if a.mode == "bounds2":
        _bounds2_mesh_bounds = transform_bounds_polygon(
            bounds, None, _BOUNDS2_FOOTPRINT_EXTRA_DEG
        )
    rows: list[dict] = []
    for reflect_code, reflect_label in reflect_modes:
        for deg in _ROT_DEGS:
            try:
                bvar = transform_bounds_polygon(bounds, reflect_code, float(deg))
                if a.mode == "bsp":
                    pred = reorient_hypergraph_bsp(
                        base,
                        float(deg),
                        _REFLECT_CODE_TO_REORIENT_MODE[reflect_code],
                    )
                    mesh_bounds = bounds
                elif a.mode == "bounds2":
                    pred = reorient_hypergraph_bsp(
                        base,
                        float(deg),
                        _REFLECT_CODE_TO_REORIENT_MODE[reflect_code],
                    )
                    mesh_bounds = _bounds2_mesh_bounds
                else:
                    pred = base
                    mesh_bounds = bvar
                pred_mesh = Hypergraph(copy.deepcopy(pred)).to_mesh(bounds=mesh_bounds)
                if use_gt:
                    # In bounds mode, GT ref for APE/δ uses canonical (dataset) bounds only;
                    # the pred panel still uses the rotated footprint (mesh_bounds / bvar).
                    gt_mesh_v = Hypergraph(gt_hg).to_mesh(  # type: ignore[arg-type]
                        bounds=bounds if a.mode == "bounds" else mesh_bounds
                    )
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "variant failed %s %s°: %s",
                    reflect_label,
                    deg,
                    exc,
                )
                rows.append(
                    {
                        "reflect": reflect_code,
                        "reflect_label": reflect_label,
                        "deg": deg,
                        "mode": a.mode,
                        "area_proportion_error": None,
                        "delta": None,
                        "error": str(exc),
                        "bounds_variant": None,
                        "variant_generated_ag": None,
                        "variant_corrected_ag": None,
                    }
                )
                continue

            pred_room = produce_room_list(pred_mesh)
            if use_gt:
                pred_area = produce_room_area_list(pred_mesh)
                gt_area_v = produce_room_area_list(gt_mesh_v)  # type: ignore[arg-type]
                gt_room_v = produce_room_list(gt_mesh_v)  # type: ignore[arg-type]
                print(
                    f"[explore_rotation] {a.sample_id} {reflect_label} {deg}°\n"
                    f"  pred_room={pred_room!r}\n  gt_room_v={gt_room_v!r}",
                    file=sys.stderr,
                    flush=True,
                )
                ape_d = compute_room_area_proportion_error(
                    pred_area, gt_area_v, aggregate=False
                )
                ape = float(ape_d["area_proportion_error"])
                dlt = float(compute_room_similarity_score(pred_room, gt_room_v))
            else:
                print(
                    f"[explore_rotation] {a.sample_id} {reflect_label} {deg}°  pred_room={pred_room!r}",
                    file=sys.stderr,
                    flush=True,
                )
                ape = None
                dlt = float(_delta_metric_square_from_room_list(pred_room))

            variant_generated_ag: dict | None = None
            try:
                variant_generated_ag = Hypergraph(copy.deepcopy(pred)).access_graph
            except Exception:
                variant_generated_ag = None

            variant_corrected_ag: dict | None = None
            try:
                variant_corrected_ag = Hypergraph(copy.deepcopy(pred)).correct_accessgraph(bounds)
            except Exception:
                variant_corrected_ag = None

            ag_matches_input_corr: bool | None = None
            if ref_ag is not None:
                try:
                    ag_matches_input_corr = orientation_variant_passes_ref_accessgraph(
                        pred_mesh, ref_ag
                    )
                except Exception:
                    ag_matches_input_corr = False

            row_out: dict = {
                "reflect": reflect_code,
                "reflect_label": reflect_label,
                "deg": deg,
                "mode": a.mode,
                "area_proportion_error": ape,
                "delta": dlt,
                "bounds_variant": mesh_bounds,
                "variant_generated_ag": variant_generated_ag,
                "variant_corrected_ag": variant_corrected_ag,
            }
            if ag_matches_input_corr is not None:
                row_out["ag_matches_input_corr"] = ag_matches_input_corr
            rows.append(row_out)

    finite_apes = [r["area_proportion_error"] for r in rows if r.get("area_proportion_error") is not None and math.isfinite(r["area_proportion_error"])]
    best_ape: float | None = min(finite_apes) if finite_apes else None
    for r in rows:
        v = r.get("area_proportion_error")
        if v is not None and best_ape is not None and isinstance(v, (int, float)) and math.isfinite(v):
            r["ape_minus_best_ape"] = float(v) - float(best_ape)
        else:
            r["ape_minus_best_ape"] = None
    # Green border: lowest δ vs GT (unless --no-gt: δ_sq vs square; same as orientation PP)
    finite_deltas = [
        r["delta"]
        for r in rows
        if r.get("delta") is not None and math.isfinite(r["delta"])
    ]
    best_delta: float | None = min(finite_deltas) if finite_deltas else None
    best_key: set[tuple[str, int]] = set()
    if best_delta is not None:
        for r in rows:
            v = r.get("delta")
            if v is not None and isinstance(v, (int, float)) and math.isfinite(v):
                if math.isclose(
                    float(v), float(best_delta), rel_tol=0.0, abs_tol=1e-6
                ):
                    best_key.add(
                        (
                            str(r["reflect_label"]),
                            int(r["deg"]),
                        )
                    )

    by_reflect_deg: dict[tuple[str, int], dict] = {
        (str(r["reflect_label"]), int(r["deg"])): r for r in rows
    }

    fig = plt.figure(figsize=(38, 16))
    gs = gridspec.GridSpec(
        3,
        5,
        figure=fig,
        wspace=0.15,
        hspace=0.36,
        width_ratios=(1.0, 1, 1, 1, 1),
    )
    n_pred = len([r for r in rows if not r.get("error")]) or len(rows)
    bsp_mode = {
        "bsp": "mode=bsp (reorient pred BSP, canonical bounds)",
        "bounds2": f"mode=bounds2 (BSP 3×4 + fixed footprint +{_BOUNDS2_FOOTPRINT_EXTRA_DEG:.0f}° all cells)",
        "bounds": "mode=bounds (rotating footprint, pred tree fixed)",
    }[a.mode]
    if best_delta is not None:
        if use_gt:
            st_extra = (
                f"lowest δ = {best_delta:.6f} (green border); lowest APE = {best_ape:.6f}"
                if best_ape is not None
                else f"lowest δ = {best_delta:.6f} (green border)"
            )
        else:
            st_extra = f"lowest δ_sq = {best_delta:.6f} (green border)"
    elif best_ape is not None:
        st_extra = f"lowest APE = {best_ape:.10f} (no valid δ; no border)"
    else:
        st_extra = "no valid metrics" if use_gt else "no valid δ_sq"
    if ref_ag is not None:
        st_extra += " | pink = mesh ≠ ref corr.AG edges (parametric-style check)"
    fig.suptitle(
        f"{a.sample_id}  |  {bsp_mode}  |  {n_pred} pred panels  |  {st_extra}",
        fontsize=12,
    )

    plotter = HypergraphPlotter()
    left_gs = GridSpecFromSubplotSpec(
        3,
        1,
        subplot_spec=gs[:, 0],
        height_ratios=[1.0, 1.0, 1.02],
        hspace=0.38,
    )
    ax_ref_gen = fig.add_subplot(left_gs[0, 0])
    ax_ref_corr = fig.add_subplot(left_gs[1, 0])
    ax_left_bottom = fig.add_subplot(left_gs[2, 0])
    _plot_access_graph_panel(
        ax_ref_gen,
        plotter,
        ref_generated_ag if isinstance(ref_generated_ag, dict) else {},
        "Input: access_graph (BSP leaf connectivity)\nreference — unreoriented pred",
        empty_stub="access_graph\n(empty)",
    )
    if ref_ag is not None:
        _plot_access_graph_panel(
            ax_ref_corr,
            plotter,
            ref_ag,
            "Input: correct_accessgraph — reference\n(mesh edges used for parametric-style filter vs each cell mesh)",
            empty_stub="correct_accessgraph\n(empty)",
        )
    else:
        ax_ref_corr.text(
            0.5,
            0.5,
            "Input: correct_accessgraph\n(unavailable)",
            ha="center",
            va="center",
            transform=ax_ref_corr.transAxes,
            fontsize=8,
        )
        ax_ref_corr.axis("off")
    if use_gt:
        assert gt_hg is not None
        _plot_mesh(
            ax_left_bottom,
            plotter,
            gt_hg,
            bounds,
            "Ground truth mesh (eval_access)\n(BSP graphs above = input pred, not GT)",
            a.sample_id,
        )
    else:
        _plot_mesh(
            ax_left_bottom,
            plotter,
            base,
            bounds,
            "Input prediction mesh (canonical bounds;\n"
            "BSP not mirrored/rotated here)\n"
                "Pink = edges failed check vs ref corr.AG (each cell: access_graph | corr.AG | mesh)",
            a.sample_id,
        )

    d_metric = "δ_sq" if not use_gt else "δ"
    d_min_label = "min δ_sq" if not use_gt else "min δ"

    first_row_axes: list = []
    row_start_axes: list = []
    for ri, (_reflect_code, reflect_label) in enumerate(_REFLECT_MODES):
        for ci, deg in enumerate(_ROT_DEGS):
            r = by_reflect_deg.get((reflect_label, deg), {})
            inner_gs = GridSpecFromSubplotSpec(
                1,
                3,
                subplot_spec=gs[ri, 1 + ci],
                width_ratios=(0.95, 0.95, 1.06),
                wspace=0.03,
            )
            ax_gen = fig.add_subplot(inner_gs[0, 0])
            ax_corr = fig.add_subplot(inner_gs[0, 1])
            ax_mesh = fig.add_subplot(inner_gs[0, 2])
            if ri == 0:
                first_row_axes.append(ax_mesh)
            if ci == 0:
                row_start_axes.append(ax_mesh)
            lab = r.get("reflect_label", reflect_label) if r else reflect_label
            deg = r.get("deg", deg) if r else deg
            apev = r.get("area_proportion_error") if r else None
            if (
                apev is None
                or not isinstance(apev, (int, float))
                or not math.isfinite(float(apev))
            ):
                ape_lines = "APE=—"
            else:
                af = float(apev)
                ape_lines = f"APE={af:.10f}"
                if best_ape is not None:
                    d_ape = af - float(best_ape)
                    if d_ape > 1e-20:
                        ape_lines += f"\n(Δ+{d_ape:.1e} vs min)"
            dlt_s = (
                "—"
                if not r
                or r.get("delta") is None
                or (isinstance(r.get("delta"), float) and not math.isfinite(r["delta"]))
                else f"{r['delta']:.4f}"
            )
            if r and r.get("delta") is not None and isinstance(
                r["delta"], (int, float)
            ) and math.isfinite(r["delta"]) and best_delta is not None:
                dlt_v = float(r["delta"])
                if dlt_v > best_delta + 1e-9:
                    dlt_s = f"{dlt_s} (Δ+{dlt_v - best_delta:.2e} vs {d_min_label})"
            row_name = _REFLECT_ROW_TITLE.get(str(lab), str(lab))
            ag_line = ""
            if ref_ag is not None and r and not r.get("error"):
                ac = r.get("ag_matches_input_corr")
                if ac is False:
                    ag_line = "\nvs ref corr.AG: mesh edge fail"
            mesh_title = f"{ape_lines}{ag_line}\n{d_metric}={dlt_s}"
            gen_title_base = f"access_graph (BSP)\n{row_name} {int(deg)}°"
            corr_title_base = f"correct_accessgraph\n{row_name} {int(deg)}°"
            if r.get("error"):
                err_t = f"{row_name}  {int(deg)}°"
                et = str(r.get("error", ""))
                err_full = f"{err_t}\n{et[:80]}…" if len(et) > 80 else f"{err_t}\n{et}"
                for ax, msg in (
                    (ax_gen, "access_graph\n(mesh error)"),
                    (ax_corr, "correct_accessgraph\n(mesh error)"),
                ):
                    ax.text(
                        0.5,
                        0.5,
                        msg,
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=7,
                    )
                    ax.axis("off")
                ax_mesh.text(
                    0.5,
                    0.5,
                    err_full,
                    ha="center",
                    va="center",
                    transform=ax_mesh.transAxes,
                    fontsize=8,
                    wrap=True,
                )
                ax_mesh.axis("off")
            elif r:
                bvar = r.get("bounds_variant")
                if not isinstance(bvar, dict) or not bvar.get("corners"):
                    ax_gen.axis("off")
                    ax_corr.axis("off")
                    ax_mesh.text(
                        0.5,
                        0.5,
                        "missing bounds",
                        transform=ax_mesh.transAxes,
                    )
                    ax_mesh.axis("off")
                else:
                    pink = r.get("ag_matches_input_corr") is False
                    face = "#fce4ec" if pink else "white"
                    for ax_k in (ax_gen, ax_corr, ax_mesh):
                        ax_k.set_facecolor(face)
                    vgen = r.get("variant_generated_ag")
                    gen_title = gen_title_base
                    vcag = r.get("variant_corrected_ag")
                    corr_title = corr_title_base
                    if ref_ag is not None and pink:
                        corr_title += "\n(≠ ref)"
                    if isinstance(vgen, dict):
                        _plot_access_graph_panel(
                            ax_gen,
                            plotter,
                            vgen,
                            gen_title,
                            empty_stub="access_graph\n(empty)",
                        )
                    else:
                        _plot_access_graph_panel(
                            ax_gen,
                            plotter,
                            {},
                            gen_title + "\n(unavailable)",
                            empty_stub="access_graph\n(empty)",
                        )
                    if isinstance(vcag, dict):
                        _plot_access_graph_panel(
                            ax_corr,
                            plotter,
                            vcag,
                            corr_title,
                            empty_stub="correct_accessgraph\n(empty)",
                        )
                    else:
                        _plot_access_graph_panel(
                            ax_corr,
                            plotter,
                            {},
                            corr_title + "\n(build failed)",
                            empty_stub="correct_accessgraph\n(empty)",
                        )
                    if a.mode in ("bsp", "bounds2"):
                        rc = r.get("reflect")
                        hg_p = reorient_hypergraph_bsp(
                            base,
                            float(r.get("deg", 0)),
                            _REFLECT_CODE_TO_REORIENT_MODE[rc],  # type: ignore[index]
                        )
                    else:
                        hg_p = base
                    _plot_mesh(ax_mesh, plotter, hg_p, bvar, mesh_title, a.sample_id)
            else:
                for ax_bad in (ax_gen, ax_corr):
                    ax_bad.text(
                        0.5,
                        0.5,
                        "missing variant",
                        transform=ax_bad.transAxes,
                    )
                    ax_bad.axis("off")
                ax_mesh.axis("off")
            if r and (str(r["reflect_label"]), int(r["deg"])) in best_key:
                for ax in (ax_gen, ax_corr, ax_mesh):
                    for spine in ax.spines.values():
                        spine.set_edgecolor("green")
                        spine.set_linewidth(4)

    fig.tight_layout(rect=(0.12, 0.05, 0.99, 0.88))
    # One y for all four labels so they line up, well above 1st-row APE/δ ax titles
    if first_row_axes:
        _top_y1 = max(ax.get_position().y1 for ax in first_row_axes)
        _deg_label_y = min(0.99, _top_y1 + 0.07)
    else:
        _deg_label_y = 0.95
    for ax, deg in zip(first_row_axes, _ROT_DEGS):
        pos = ax.get_position()
        fig.text(
            pos.x0 + 0.5 * pos.width,
            _deg_label_y,
            f"{deg}°",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="semibold",
            transform=fig.transFigure,
        )
    for ri, (_rc, rlab) in enumerate(_REFLECT_MODES):
        pos = row_start_axes[ri].get_position()
        fig.text(
            0.07,
            pos.y0 + 0.5 * pos.height,
            _REFLECT_ROW_TITLE.get(rlab, rlab),
            ha="right",
            va="center",
            rotation=90,
            fontsize=10,
            fontweight="semibold",
            transform=fig.transFigure,
        )

    leg = [
        "Each variant row: access_graph (BSP leaf connectivity) | correct_accessgraph (mesh adjacency) | mesh.",
        "mode=bsp: reorient pred BSP; canonical bounds.",
        "mode=bounds: pred in rotated bvar; GT for APE/δ = canonical bounds (plot pred still uses bvar).",
        f"mode=bounds2: same 12 BSP reorients as bsp; all cells share footprint +{_BOUNDS2_FOOTPRINT_EXTRA_DEG:.0f}° only.",
        "Switch: --mode bounds | bsp | bounds2  (legacy --bsp-reorient = bsp).",
    ]
    if ref_ag is not None:
        leg.append(
            "Pink cell: variant mesh fails edge-vs-ref check (same as parametric accessgraph_penalty)."
        )
    fig.text(0.5, 0.02, "\n".join(leg), ha="center", fontsize=9, style="italic")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(a.output), dpi=150, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)

    def _json_num(x: object) -> object:
        if isinstance(x, float) and (math.isnan(x) or not math.isfinite(x)):
            return None
        return x

    def _row_for_json(r: dict) -> dict:
        out: dict = {
            k: v
            for k, v in r.items()
            if k not in ("bounds_variant", "hypergraph", "variant_corrected_ag", "variant_generated_ag")
        }
        for k in ("area_proportion_error", "delta", "ape_minus_best_ape"):
            if k in out and isinstance(out[k], (int, float)):
                out[k] = _json_num(out[k])
        return out

    sidecar = a.output.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "sample_id": a.sample_id,
                "mode": a.mode,
                "metric": "delta_vs_gt" if use_gt else "delta_sq",
                "reference_correct_accessgraph_ok": ref_ag is not None,
                "reference_access_graph_ok": ref_generated_ag is not None,
                "best_area_proportion_error": _json_num(best_ape) if best_ape is not None else None,
                "best_delta": _json_num(best_delta) if best_delta is not None else None,
                "best_variants": [
                    {
                        "reflect_label": lab,
                        "deg": d,
                    }
                    for (lab, d) in sorted(best_key)
                ],
                "variants": [_row_for_json(r) for r in rows],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if not a.quiet:
        print(f"Wrote {a.output} and {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
