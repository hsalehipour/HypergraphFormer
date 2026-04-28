#!/usr/bin/env python3
"""Hypergraph optimization via continuous perturbations of binary area splits.

Ground truth defaults to WMR24 ``dataset/wmr24/test``; predictions default to
``inference_8700_on_wmr24_inference_vllm_64/results_hypergraphs.json``.
Override with ``--test-dir`` and ``--results-hypergraphs``.

Run from repo root::

  python -m scripts.run_parametric b-0001 --objective delta

  python -m scripts.run_parametric b-0001 --objective epsilon  # min ε vs ref; reported ε in logs is vs GT

  python -m scripts.run_parametric b-0001 --objective delta_epsilon

  python -m scripts.run_parametric b-0001 --objective delta --optimizer gd --maxiter 80

  python -m scripts.run_parametric --all --limit 5 --objective area

  # Parallel batch (one process per apartment; 128 ≈ 128-way throughput if machine can feed it)
  python -m scripts.run_parametric --all --batch-workers 128
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# utils_parametric lives next to this file, not at repo root
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from utils.hypergraph import repair_bsp_tree_names
from utils.metrics import (
    compute_room_area_proportion_error,
    produce_room_area_list,
    produce_room_area_list_from_polygons,
    produce_room_list_from_polygons,
)

from utils_parametric import (
    AREA_REF_TARGETS_ABS,
    DEFAULT_WMR24_RESULTS,
    OptimConfig,
    _bedroom_group_from_accessgraph,
    _validate_apt_id,
    parse_args,
    _expected_counts_from_accessgraph,
    _leaf_type_counts_from_hypergraph,
    _delta_metric_square_from_room_list,
    _area_metric_gt,
    _area_ref_metric,
    _save_metric_progress_plot,
    _render_hypergraph_frame,
    access_graph_from_prediction,
    check_accessgraph_edges_against_mesh,
    compute_delta_metric_soft_from_room_lists,
    compute_epsilon_metric_vs_gt,
    compute_epsilon_metric_vs_ref,
    _area_metric_gt_from_type_maps,
    _mesh_areas_by_type_from_polygons,
    hypergraph_bounds_to_mesh,
    init_rgl,
    load_ground_truth_txt,
    produce_room_list,
    _resolve_under_repo,
    annotate_area_splits,
    collect_area_splits,
    apply_area_splits,
    print_area_split_list,
    validate_leaf_area_floor,
    _print_room_areas,
    save_hypergraph_json,
    save_metrics_report,
    save_accessgraph_penalty_report,
    save_metric_progress_json,
    save_single_apt_artifacts,
    save_batch_results,
    render_from_json,
    _to_jsonable,
)


try:
    import imageio.v2 as imageio  # type: ignore

    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False

try:
    from scipy.optimize import minimize

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


def _fd_grad_forward(
    fun,
    x: np.ndarray,
    eps: float,
    lo: float,
    hi: float,
) -> np.ndarray:
    """One-sided finite-difference gradient; uses +h at i if in bounds, else −h."""
    x = np.asarray(x, dtype=float)
    fx = float(fun(x))
    h = max(float(eps), 1e-10)
    g = np.zeros_like(x, dtype=float)
    for i in range(int(x.size)):
        if x[i] + h <= hi - 1e-15:
            xp = x.copy()
            xp[i] = x[i] + h
            g[i] = (float(fun(xp)) - fx) / h
        elif x[i] - h >= lo + 1e-15:
            xm = x.copy()
            xm[i] = x[i] - h
            g[i] = (fx - float(fun(xm))) / h
        else:
            g[i] = 0.0
    return g


def optimize_hypergraph(
    apt_id: str,
    config: OptimConfig,
    *,
    artifacts_dir: Path | None = None,
    grpo_file_path: Path | None = None,
    preloaded_hg: dict | None = None,
):
    """Run optimization for one apartment. Returns (optimized_hg, best_metric, original_metric, gt_hg, metrics_original, metrics_optimized, bedroom_group, expected_counts, ag_report)."""
    apt_id = _validate_apt_id(apt_id)
    root = REPO_ROOT

    if grpo_file_path is not None:
        grpo_file = grpo_file_path
    elif config.grpo_file:
        grpo_file = _resolve_under_repo(root, str(config.grpo_file))
    else:
        grpo_file = DEFAULT_WMR24_RESULTS
    objective = str(config.objective).strip().lower()
    if objective not in ("area", "delta", "area_delta", "delta_epsilon", "epsilon"):
        raise ValueError("objective must be 'area', 'delta', 'area_delta', 'delta_epsilon', or 'epsilon'")
    if preloaded_hg is not None:
        original_hg = repair_bsp_tree_names(copy.deepcopy(preloaded_hg))
    else:
        if not Path(grpo_file).exists():
            raise FileNotFoundError(f"Results hypergraphs file not found: {grpo_file}")
        with open(grpo_file, encoding="utf-8") as f:
            grpo_results = json.load(f)
        if apt_id not in grpo_results:
            raise KeyError(f"{apt_id} not found in {grpo_file}")
        original_hg = repair_bsp_tree_names(copy.deepcopy(grpo_results[apt_id]))

    test_dir = _resolve_under_repo(root, str(getattr(config, "test_dir", "dataset/wmr24/test")))
    bounds, gt_hg, accessgraph, room_polygons = load_ground_truth_txt(test_dir, apt_id)
    gt_hg = copy.deepcopy(gt_hg)

    # corrected→raw→file(GT) for *edge compliance*; objective scores never use soft δ/ε vs GT.
    accessgraph_penalty_target = access_graph_from_prediction(
        original_hg, bounds=bounds, file_fallback=accessgraph if isinstance(accessgraph, dict) else {}
    )
    _nag = len(accessgraph_penalty_target.get("edges") or [])
    print(f"Accessgraph edge penalty: {_nag} edges (corrected→raw→file(GT) fallback for compliance)")

    # For area-based objectives, evaluate only room types that actually exist in the
    # hypergraph JSON leaves (e.g. if there is no 'foyer' in the hypergraph, don't
    # include it in the area objective).
    hg_leaf_counts = _leaf_type_counts_from_hypergraph(original_hg)
    if isinstance(accessgraph_penalty_target, dict) and (accessgraph_penalty_target.get("nodes") or accessgraph_penalty_target.get("edges")):
        bedroom_group = _bedroom_group_from_accessgraph(accessgraph_penalty_target)
    else:
        bedroom_group = _bedroom_group_from_accessgraph(accessgraph if isinstance(accessgraph, dict) else {})
    expected_counts = hg_leaf_counts if hg_leaf_counts else _expected_counts_from_accessgraph(accessgraph if isinstance(accessgraph, dict) else {})
    print(f"Bedroom group: {bedroom_group}, leaf counts: {expected_counts}")

    RGL, Collections = init_rgl()

    # GT reference: WMR24-style .txt has a BSP ``hypergraph``; RPLAN eval_test has
    # ``room_polygons`` only (no tree). The latter cannot go through RGL until meshed
    # from a tree — use polygon lists for GT room geometry instead.
    gt_mesh = None
    if gt_hg and isinstance(gt_hg, dict) and gt_hg.get("name"):
        gt_mesh = hypergraph_bounds_to_mesh(
            copy.deepcopy(gt_hg), bounds, RGL=RGL, Collections=Collections
        )
        gt_room_list = produce_room_list(gt_mesh)
    elif room_polygons:
        gt_room_list = produce_room_list_from_polygons(room_polygons)
    else:
        raise ValueError(
            f"{apt_id}: {test_dir}/{apt_id}.txt has no GT 'hypergraph' tree and no "
            "'room_polygons'. run_parametric needs at least one (use eval_test with "
            "polygons, or a split that includes hypergraph in each .txt)."
        )
    gt_room_names = set(room[0] for room in gt_room_list)

    if float(config.area_weight) < 0.0:
        raise ValueError("area_weight must be >= 0")

    targets_by_group = AREA_REF_TARGETS_ABS
    original_mesh = hypergraph_bounds_to_mesh(copy.deepcopy(original_hg), bounds, RGL=RGL, Collections=Collections)
    original_room_list = produce_room_list(original_mesh)
    original_delta_gt = float(compute_delta_metric_soft_from_room_lists(original_room_list, gt_room_list, area_sorting=True))
    original_delta_sq = float(_delta_metric_square_from_room_list(original_room_list))
    original_epsilon_ref = float(
        compute_epsilon_metric_vs_ref(
            original_mesh,
            group=bedroom_group,
            expected_counts=expected_counts,
            targets_by_group=targets_by_group,
        )
    )
    if gt_mesh is not None:
        original_epsilon_gt = float(compute_epsilon_metric_vs_gt(original_mesh, gt_mesh))
    else:
        _pred_a = produce_room_area_list(original_mesh)
        _gt_a = produce_room_area_list_from_polygons(room_polygons)
        _ape = compute_room_area_proportion_error(_pred_a, _gt_a, aggregate=False)
        original_epsilon_gt = float(_ape.get("area_proportion_error", float("nan")))

    def _area_gt_metric(gen_m):
        if gt_mesh is not None:
            return float(_area_metric_gt(gen_m, gt_mesh))
        return float(
            _area_metric_gt_from_type_maps(
                gen_m, _mesh_areas_by_type_from_polygons(room_polygons)
            )
        )
    print(
        f"Original  soft_δ={original_delta_gt:.6f}  δ_sq={original_delta_sq:.6f}  "
        f"ε(ref targets)={original_epsilon_ref:.6f}  ε(GT)={original_epsilon_gt:.6f}"
    )

    delta_scale = 1.0
    if objective in ("area", "area_delta"):
        area_ref = float(
            _area_ref_metric(
                original_mesh,
                group=bedroom_group,
                targets_by_group=targets_by_group,
                expected_counts=expected_counts,
            )
        )
        print(f"Original area ref (group={bedroom_group}): {area_ref:.6f}")

    if objective == "delta":
        original_metric = float(original_delta_sq)
    elif objective == "area":
        original_metric = float(area_ref)
    elif objective == "area_delta":
        # Same δ term as --objective delta: perimeter–square (δ_sq), not soft δ vs GT.
        denom = max(1e-9, float(original_delta_sq))
        delta_scale = float(area_ref) / float(denom)
        original_metric = 0.5 * (float(delta_scale) * float(original_delta_sq)) + 0.5 * float(
            area_ref
        )
        print(f"area_delta: delta_scale={delta_scale:.3f}  combo={original_metric:.6f}  (δ=δ_sq)")
    elif objective == "delta_epsilon":
        # Loss uses δ_sq and ε_ref only; soft_δ vs GT is for reporting at the end.
        denom = max(1e-9, float(original_delta_sq))
        delta_scale = float(original_epsilon_ref) / float(denom)
        original_metric = 0.5 * (float(delta_scale) * float(original_delta_sq)) + 0.5 * float(
            original_epsilon_ref
        )
        print(
            f"delta_epsilon: delta_scale={delta_scale:.3f}  combo={original_metric:.6f}  (obj: δ_sq+ε_ref)"
        )
    elif objective == "epsilon":
        original_metric = float(original_epsilon_ref)
    else:
        raise ValueError(f"unhandled objective: {objective!r}")

    annotate_area_splits(original_hg, min_split=config.min_split, max_split=config.max_split)
    _split_paths, split_vals = collect_area_splits(original_hg)
    print(f"\nCollected {len(split_vals)} area_split params. Range: [{min(split_vals):.3f}, {max(split_vals):.3f}]" if split_vals else "\nCollected 0 area_split params.")

    _print_room_areas(original_hg, "Original")

    print("\nOriginal mesh room areas (from mesh faces):")
    for i in range(original_mesh.faceList.Count):
        face = original_mesh.faceList[i]
        room_name = getattr(face, "merge_id", "unknown")
        area = face.Area
        marker = "🚽" if "bath" in str(room_name).lower() else ""
        print(f"  {marker} {room_name}: area={area:.6f}")

    if not (0.0 < config.min_split < config.max_split < 1.0):
        raise ValueError("min_split/max_split must satisfy 0 < min_split < max_split < 1")
    if not (0.0 < config.min_leaf_frac < 1.0):
        raise ValueError("min_leaf_frac must be in (0, 1)")

    rng = np.random.default_rng(config.seed)
    artifacts_dir = artifacts_dir or (Path(__file__).parent / "output_parametric")

    ag_eval_counter = 0
    ag_fail_count = 0
    ag_first_fail_eval: int | None = None
    ag_fail_evals_sample: list[int] = []

    def _eval_metric_for_splits(splits_vec: np.ndarray) -> tuple[float, dict, dict | None]:
        nonlocal ag_eval_counter, ag_fail_count, ag_first_fail_eval, ag_fail_evals_sample
        ag_eval_counter += 1
        hg = copy.deepcopy(original_hg)
        apply_area_splits(hg, splits_vec, min_split=config.min_split, max_split=config.max_split)
        _ok_leaf, leaf_stats = validate_leaf_area_floor(hg, min_leaf_frac=config.min_leaf_frac)

        accessgraph_penalty = 0.0

        if objective in ("delta", "area", "area_delta", "delta_epsilon", "epsilon"):
            try:
                mesh = hypergraph_bounds_to_mesh(copy.deepcopy(hg), bounds, RGL=RGL, Collections=Collections)
                room_list = produce_room_list(mesh)
                generated_room_names = set(room[0] for room in room_list)
                if generated_room_names != gt_room_names:
                    return 1.0, hg, leaf_stats
                if isinstance(accessgraph_penalty_target, dict) and accessgraph_penalty_target.get("edges"):
                    checks = check_accessgraph_edges_against_mesh(mesh, accessgraph_penalty_target)
                    if checks and not all(bool(c.get("ok")) for c in checks):
                        ag_fail_count += 1
                        if ag_first_fail_eval is None:
                            ag_first_fail_eval = int(ag_eval_counter)
                            print(f"Accessgraph check failed at eval#{ag_eval_counter}", flush=True)
                        if len(ag_fail_evals_sample) < 50:
                            ag_fail_evals_sample.append(int(ag_eval_counter))
                        accessgraph_penalty = 1.0
                if objective == "epsilon":
                    eps = float(
                        compute_epsilon_metric_vs_ref(
                            mesh,
                            group=bedroom_group,
                            expected_counts=expected_counts,
                            targets_by_group=targets_by_group,
                        )
                    )
                    return eps + accessgraph_penalty, hg, leaf_stats
                delta_sq = float(_delta_metric_square_from_room_list(room_list))
                if objective == "delta_epsilon":
                    eps = float(
                        compute_epsilon_metric_vs_ref(
                            mesh,
                            group=bedroom_group,
                            expected_counts=expected_counts,
                            targets_by_group=targets_by_group,
                        )
                    )
                    return (0.5 * (float(delta_scale) * delta_sq) + 0.5 * eps) + accessgraph_penalty, hg, leaf_stats
                if objective == "delta":
                    return delta_sq + accessgraph_penalty, hg, leaf_stats
                area_m = float(
                    _area_ref_metric(
                        mesh,
                        group=bedroom_group,
                        targets_by_group=targets_by_group,
                        expected_counts=expected_counts,
                    )
                )
                if objective == "area":
                    return area_m + accessgraph_penalty, hg, leaf_stats
                # area_delta: δ term is δ_sq (not soft δ), matching --objective delta
                return (0.5 * (float(delta_scale) * delta_sq) + 0.5 * area_m) + accessgraph_penalty, hg, leaf_stats
            except Exception:
                return 1.0, hg, leaf_stats

    splits_new: list[float] | None = None
    best_hg: dict | None = None
    last_stats: dict | None = None
    x_history: list[np.ndarray] = []
    metric_trace: list[float] = []
    best_trace: list[float] = []
    # Baseline objective after annotate + initial split vector (after optional init search);
    # if final opt objective is worse, we restore this hypergraph.
    revert_baseline_f: float | None = None
    hg_at_annotate: dict | None = None
    splits_at_annotate: np.ndarray | None = None

    if len(split_vals) == 0:
        splits_new = []
        best_hg = copy.deepcopy(original_hg)
        last_stats = validate_leaf_area_floor(best_hg, min_leaf_frac=config.min_leaf_frac)[1]
    else:
        x0 = np.asarray(split_vals, dtype=float)
        hg_at_annotate = copy.deepcopy(original_hg)
        splits_at_annotate = x0.copy()
        bounds_box = [(float(config.min_split), float(config.max_split))] * len(split_vals)
        x_history.append(x0.copy())
        scipy_eps = config.scipy_eps
        if scipy_eps is None:
            width = float(config.max_split) - float(config.min_split)
            base = 0.10 * width
            scipy_eps = base * (2.0 if bool(config.brave) else 1.0)
            scipy_eps = min(float(width), float(scipy_eps))
            scipy_eps = max(1e-4, float(scipy_eps))
        init_samples = config.init_samples
        init_mix = config.init_mix
        if init_samples is None:
            init_samples = 500 if bool(config.brave) else 300
        if init_mix is None:
            init_mix = 1.0 if bool(config.brave) else 0.8
        init_samples = int(init_samples)
        init_mix = float(init_mix)
        if init_samples < 0:
            raise ValueError("init_samples must be >= 0")
        if not (0.0 <= init_mix <= 1.0):
            raise ValueError("init_mix must be in [0, 1]")
        if init_samples > 0:
            f_base, _, _ = _eval_metric_for_splits(x0)
            best0_f = float(f_base)
            best0_x = np.asarray(x0, dtype=float).copy()
            t_init0 = time.time()
            for k in range(init_samples):
                xr = rng.uniform(float(config.min_split), float(config.max_split), size=len(split_vals)).astype(float)
                x_try = (1.0 - init_mix) * x0 + init_mix * xr
                f_try, _, _ = _eval_metric_for_splits(x_try)
                if f_try < best0_f or best0_x is None:
                    best0_f = float(f_try)
                    best0_x = x_try.copy()
                if (k + 1) % 10 == 0:
                    dt = time.time() - t_init0
                    print(f"Init {k+1}/{init_samples} ({dt:.1f}s, best f={best0_f:.6f})", flush=True)
                if (k + 1) % 10 == 0:
                    metric_trace.append(float(best0_f))
                    best_trace.append(float(best0_f))
            print(f"Init best f={best0_f:.6f} from {init_samples} samples")
            if best0_x is not None:
                x0 = best0_x
            x_history[-1] = x0.copy()

        def fun(x: np.ndarray) -> float:
            f, _, _ = _eval_metric_for_splits(x)
            return float(f)

        f0, _, leaf0 = _eval_metric_for_splits(x0)
        revert_baseline_f = float(f0)
        best_seen = {"f": float(f0), "x": np.asarray(x0, dtype=float).copy(), "leaf": leaf0}
        metric_trace.append(float(f0))
        best_trace.append(float(f0))

        def record_step(xk: np.ndarray) -> None:
            x_history.append(np.asarray(xk, dtype=float).copy())
            f, _, leaf = _eval_metric_for_splits(xk)
            if f < best_seen["f"]:
                best_seen["f"] = float(f)
                best_seen["x"] = np.asarray(xk, dtype=float).copy()
                best_seen["leaf"] = leaf
            metric_trace.append(float(f))
            best_trace.append(float(best_seen["f"]))

        opt = str(getattr(config, "optimizer", "lbfgs") or "lbfgs").lower()
        if opt == "gd":
            lo, hi = float(config.min_split), float(config.max_split)
            width = max(hi - lo, 1e-9)
            lr = config.gd_lr if config.gd_lr is not None else 0.1 * width
            if not (lr > 0.0):
                raise ValueError("gd learning rate (gd_lr or default) must be > 0")
            n_steps = int(config.maxiter)
            if n_steps < 0:
                raise ValueError("maxiter must be >= 0 for gd")
            d = int(len(x0))
            print(
                f"Vanilla GD: steps={n_steps}  lr={lr:.6f}  fd_eps={scipy_eps:.6f}  "
                f"(each step: ~{1 + d} evals for FD + 1 for trace; forward difference)"
            )
            xk = np.asarray(x0, dtype=float).copy()
            for it in range(n_steps):
                g = _fd_grad_forward(fun, xk, float(scipy_eps), lo, hi)
                xk = np.clip(xk - float(lr) * g, lo, hi)
                record_step(xk)
            x_best = best_seen["x"]
            f_best, hg_best, leaf_best = _eval_metric_for_splits(x_best)
            splits_new = np.asarray(x_best, dtype=float).tolist()
            best_hg = hg_best
            last_stats = leaf_best
            print(f"GD done: iters={n_steps}  best f={float(f_best):.6f}")
        else:
            if not HAS_SCIPY:
                raise RuntimeError("SciPy is required for L-BFGS-B. Use --optimizer gd to avoid SciPy, or install scipy.")
            def callback(xk: np.ndarray) -> None:
                record_step(xk)

            print(f"SciPy L-BFGS-B (maxiter={config.maxiter})...")
            res = minimize(
                fun,
                x0,
                method="L-BFGS-B",
                bounds=bounds_box,
                options={
                    "maxiter": int(config.maxiter),
                    "eps": float(scipy_eps),
                    **({} if config.scipy_ftol is None else {"ftol": float(config.scipy_ftol)}),
                    **({} if config.scipy_gtol is None else {"gtol": float(config.scipy_gtol)}),
                },
                callback=callback,
            )
            x_best = best_seen["x"] if best_seen["x"] is not None else np.asarray(res.x, dtype=float)
            f_best, hg_best, leaf_best = _eval_metric_for_splits(x_best)
            splits_new = np.asarray(x_best, dtype=float).tolist()
            best_hg = hg_best
            last_stats = leaf_best
            print(f"SciPy done: success={res.success}, nit={getattr(res, 'nit', None)}, fun={float(f_best):.6f}")
        if metric_trace:
            out_png = Path(artifacts_dir) / f"{apt_id}_metric_progress.png"
            _opt_label = "gd" if opt == "gd" else "lbfgs"
            try:
                _save_metric_progress_plot(
                    metric_trace,
                    best_trace,
                    out_png,
                    title=f"{apt_id}  opt={_opt_label}  objective={objective}  steps={len(metric_trace)}",
                )
                save_metric_progress_json(Path(artifacts_dir), apt_id, objective, metric_trace, best_trace, verbose=False)
            except OSError as e:
                print(f"⚠️  Could not save metric progress plot: {e}")

    if best_hg is None or splits_new is None:
        raise RuntimeError("Optimization failed to produce a candidate.")

    if last_stats:
        print(f"Leaf-area floor: min_observed={last_stats.get('min_observed_leaf_frac', 0):.4f}")

    best_mesh = hypergraph_bounds_to_mesh(copy.deepcopy(best_hg), bounds, RGL=RGL, Collections=Collections)
    best_room_list = produce_room_list(best_mesh)

    best_room_names = set(room[0] for room in best_room_list)
    ag_report = {
        "n_evals": int(ag_eval_counter),
        "n_fails": int(ag_fail_count),
        "first_fail_eval": int(ag_first_fail_eval) if ag_first_fail_eval is not None else None,
        "fail_evals_sample": list(ag_fail_evals_sample),
    }
    save_accessgraph_penalty_report(Path(artifacts_dir), ag_report)

    if best_room_names != gt_room_names:
        metrics_original = {
            "delta_metric": float(compute_delta_metric_soft_from_room_lists(original_room_list, gt_room_list, area_sorting=True)),
            "delta_metric_square": float(_delta_metric_square_from_room_list(original_room_list)),
            "area_metric": float(
                _area_ref_metric(
                    original_mesh,
                    group=bedroom_group,
                    targets_by_group=AREA_REF_TARGETS_ABS,
                    expected_counts=expected_counts,
                )
            ),
            "area_metric_gt": float(_area_gt_metric(original_mesh)),
            "epsilon_metric": float(original_epsilon_gt),
            "epsilon_metric_ref": float(original_epsilon_ref),
        }
        metrics_optimized = dict(metrics_original)
        save_metrics_report(Path(artifacts_dir), apt_id, metrics_original, metrics_optimized)
        return original_hg, float(original_metric), float(original_metric), gt_hg, metrics_original, metrics_optimized, bedroom_group, expected_counts, ag_report

    best_delta_gt = float(compute_delta_metric_soft_from_room_lists(best_room_list, gt_room_list, area_sorting=True))
    best_delta_sq = float(_delta_metric_square_from_room_list(best_room_list))
    best_epsilon_ref = float(
        compute_epsilon_metric_vs_ref(
            best_mesh,
            group=bedroom_group,
            expected_counts=expected_counts,
            targets_by_group=AREA_REF_TARGETS_ABS,
        )
    )
    if gt_mesh is not None:
        best_epsilon_gt = float(compute_epsilon_metric_vs_gt(best_mesh, gt_mesh))
    else:
        _bpa = produce_room_area_list(best_mesh)
        _bga = produce_room_area_list_from_polygons(room_polygons)
        _bape = compute_room_area_proportion_error(_bpa, _bga, aggregate=False)
        best_epsilon_gt = float(_bape.get("area_proportion_error", float("nan")))
    original_area_ref = float(_area_ref_metric(original_mesh, group=bedroom_group, targets_by_group=AREA_REF_TARGETS_ABS, expected_counts=expected_counts))
    best_area_ref = float(_area_ref_metric(best_mesh, group=bedroom_group, targets_by_group=AREA_REF_TARGETS_ABS, expected_counts=expected_counts))
    original_area_gt = float(_area_gt_metric(original_mesh))
    best_area_gt = float(_area_gt_metric(best_mesh))

    if objective == "delta":
        cand_metric = float(best_delta_sq)
    elif objective == "area":
        cand_metric = float(best_area_ref)
    elif objective == "area_delta":
        cand_metric = 0.5 * (float(delta_scale) * float(best_delta_sq)) + 0.5 * float(best_area_ref)
    elif objective == "delta_epsilon":
        cand_metric = 0.5 * (float(delta_scale) * float(best_delta_sq)) + 0.5 * float(best_epsilon_ref)
    elif objective == "epsilon":
        cand_metric = float(best_epsilon_ref)
    else:
        cand_metric = float(best_delta_gt)

    if (
        len(split_vals) > 0
        and hg_at_annotate is not None
        and splits_at_annotate is not None
        and objective in ("delta", "area", "area_delta", "delta_epsilon", "epsilon")
    ):
        hg_ent = copy.deepcopy(hg_at_annotate)
        apply_area_splits(
            hg_ent,
            np.asarray(splits_at_annotate, dtype=float).tolist(),
            min_split=config.min_split,
            max_split=config.max_split,
        )
        entry_mesh = hypergraph_bounds_to_mesh(copy.deepcopy(hg_ent), bounds, RGL=RGL, Collections=Collections)
        ent_room_list = produce_room_list(entry_mesh)
        if set(room[0] for room in ent_room_list) == gt_room_names:
            ent_delta_gt = float(
                compute_delta_metric_soft_from_room_lists(ent_room_list, gt_room_list, area_sorting=True)
            )
            ent_delta_sq = float(_delta_metric_square_from_room_list(ent_room_list))
            ent_epsilon_ref = float(
                compute_epsilon_metric_vs_ref(
                    entry_mesh,
                    group=bedroom_group,
                    expected_counts=expected_counts,
                    targets_by_group=AREA_REF_TARGETS_ABS,
                )
            )
            if gt_mesh is not None:
                ent_epsilon_gt = float(compute_epsilon_metric_vs_gt(entry_mesh, gt_mesh))
            else:
                _epa = produce_room_area_list(entry_mesh)
                _ega = produce_room_area_list_from_polygons(room_polygons)
                _eape = compute_room_area_proportion_error(_epa, _ega, aggregate=False)
                ent_epsilon_gt = float(_eape.get("area_proportion_error", float("nan")))
            ent_area_ref = float(
                _area_ref_metric(
                    entry_mesh,
                    group=bedroom_group,
                    targets_by_group=AREA_REF_TARGETS_ABS,
                    expected_counts=expected_counts,
                )
            )
            ent_area_gt = float(_area_gt_metric(entry_mesh))
            if objective == "delta":
                cand_entry = float(ent_delta_sq)
            elif objective == "area":
                cand_entry = float(ent_area_ref)
            elif objective == "area_delta":
                cand_entry = 0.5 * (float(delta_scale) * float(ent_delta_sq)) + 0.5 * float(ent_area_ref)
            elif objective == "delta_epsilon":
                cand_entry = 0.5 * (float(delta_scale) * float(ent_delta_sq)) + 0.5 * float(ent_epsilon_ref)
            else:
                cand_entry = float(ent_epsilon_ref)
            if (
                math.isfinite(cand_metric)
                and math.isfinite(cand_entry)
                and float(cand_metric) > float(cand_entry) + 1e-9
            ):
                print(
                    f"Reverting to entry splits: objective {float(cand_metric):.6f} "
                    f"> entry-split baseline {float(cand_entry):.6f} (same scaling as --objective).",
                    flush=True,
                )
                best_hg = hg_ent
                splits_new = np.asarray(splits_at_annotate, dtype=float).tolist()
                last_stats = validate_leaf_area_floor(best_hg, min_leaf_frac=config.min_leaf_frac)[1]
                best_delta_gt = ent_delta_gt
                best_delta_sq = ent_delta_sq
                best_epsilon_ref = ent_epsilon_ref
                best_epsilon_gt = ent_epsilon_gt
                best_epsilon_gt = ent_epsilon_gt
                best_area_ref = ent_area_ref
                best_area_gt = ent_area_gt
                cand_metric = float(cand_entry)

    presentation_mesh = hypergraph_bounds_to_mesh(copy.deepcopy(best_hg), bounds, RGL=RGL, Collections=Collections)
    print_area_split_list(original_hg, best_hg, min_split=config.min_split, max_split=config.max_split)
    _print_room_areas(best_hg, "Optimized")
    print("\nOptimized mesh room areas (from mesh faces):")
    for i in range(presentation_mesh.faceList.Count):
        face = presentation_mesh.faceList[i]
        room_name = getattr(face, "merge_id", "unknown")
        marker = "🚽" if "bath" in str(room_name).lower() else ""
        print(f"  {marker} {room_name}: area={face.Area:.6f}")

    metrics_original = {
        "delta_metric": float(original_delta_gt),
        "delta_metric_square": float(original_delta_sq),
        "area_metric": float(original_area_ref),
        "area_metric_gt": float(original_area_gt),
        "epsilon_metric": float(original_epsilon_gt),
        "epsilon_metric_ref": float(original_epsilon_ref),
    }
    metrics_optimized = {
        "delta_metric": float(best_delta_gt),
        "delta_metric_square": float(best_delta_sq),
        "area_metric": float(best_area_ref),
        "area_metric_gt": float(best_area_gt),
        "epsilon_metric": float(best_epsilon_gt),
        "epsilon_metric_ref": float(best_epsilon_ref),
    }
    save_metrics_report(Path(artifacts_dir), apt_id, metrics_original, metrics_optimized)

    print("\n=== Metrics (original vs optimized; lower is better) ===")
    def _sig2(v: float) -> str:
        return format(float(v), ".2g") if math.isfinite(float(v)) else "nan"
    _mkeys = [
        ("delta_metric", "delta"),
        ("epsilon_metric", "epsilon (GT)"),
        ("delta_metric_square", "delta_square"),
        ("area_metric", "area_metric"),
        ("area_metric_gt", "area_metric_gt"),
    ]
    for k, label in _mkeys:
        print(
            f"  {label:<20}  orig={_sig2(metrics_original[k])}  opt={_sig2(metrics_optimized[k])}  "
            f"Δ={_sig2(metrics_optimized[k] - metrics_original[k])}"
        )

    best_metric = float(cand_metric)
    _om = float(original_metric)
    print(f"\nObjective {objective}: best={best_metric:.6f}  (Δ vs start {best_metric - _om:+.6f})")

    if config.make_gif:
        if not HAS_IMAGEIO:
            print("\n⚠️  imageio not available; skipping GIF export.")
        elif len(x_history) < 2:
            print("\n⚠️  Not enough optimizer history to build GIF; skipping.")
        else:
            n = max(2, int(config.gif_frames))
            idxs = np.linspace(0, len(x_history) - 1, num=n).round().astype(int).tolist()
            seen = set()
            idxs = [i for i in idxs if not (i in seen or seen.add(i))]
            frames = []
            for k, i in enumerate(idxs):
                xk = x_history[i]
                fk, hgk, _ = _eval_metric_for_splits(xk)
                title = f"{apt_id}  {k+1}/{len(idxs)}  f={float(fk):.4f}"
                frames.append(_render_hypergraph_frame(hgk, bounds, RGL=RGL, Collections=Collections, title=title))
            Path(artifacts_dir).mkdir(exist_ok=True, parents=True)
            gif_path = Path(artifacts_dir) / f"{apt_id}_opt.gif"
            try:
                imageio.mimsave(str(gif_path), frames, duration=float(config.gif_duration), loop=0)
                print(f"Saved optimization GIF to: {gif_path}")
            except OSError as e:
                print(f"⚠️  Could not save GIF to {gif_path}: {type(e).__name__}: {e}")

    return best_hg, best_metric, original_metric, gt_hg, metrics_original, metrics_optimized, bedroom_group, expected_counts, ag_report


# Populated in ProcessPool child processes via _mp_init_pool_batch (one RGL per worker).
_POOL_RGL: tuple[object, object] = (None, None)


def _mp_init_pool_batch() -> None:
    global _POOL_RGL
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    _POOL_RGL = (None, None)
    try:
        _POOL_RGL = init_rgl()
    except Exception:
        _POOL_RGL = (None, None)


def _run_one_batch_apt(
    apt_id: str,
    config: OptimConfig,
    preloaded_hg: dict,
    grpo_file_path: Path,
    batch_root: Path,
    repo_root: Path,
    rgl_b,
    col_b,
) -> tuple[dict, dict | None, dict | None]:
    """One batch apartment: row dict, merged hypergraph (or None), optional ag row for CSV."""
    _id = apt_id
    row: dict = {
        "id": _id,
        "objective": config.objective,
        "status": "ok",
        "error": "",
    }
    merged_hg: dict | None = None
    ag_row_extra: dict | None = None
    try:
        original_hg0 = repair_bsp_tree_names(copy.deepcopy(preloaded_hg))
        test_path0 = _resolve_under_repo(repo_root, str(config.test_dir))
        bounds0, gt_hg0, _ag0, rp0 = load_ground_truth_txt(test_path0, _id)
        if rgl_b is not None:
            try:
                mesh_gen0 = hypergraph_bounds_to_mesh(copy.deepcopy(original_hg0), bounds0, RGL=rgl_b, Collections=col_b)
                rl_gen0 = produce_room_list(mesh_gen0)
                if gt_hg0 and isinstance(gt_hg0, dict) and gt_hg0.get("name"):
                    mesh_gt0 = hypergraph_bounds_to_mesh(
                        copy.deepcopy(gt_hg0), bounds0, RGL=rgl_b, Collections=col_b
                    )
                    rl_gt0 = produce_room_list(mesh_gt0)
                elif rp0:
                    rl_gt0 = produce_room_list_from_polygons(rp0)
                else:
                    raise ValueError("no GT hypergraph or room_polygons")
                row["delta_metric_orig"] = float(compute_delta_metric_soft_from_room_lists(rl_gen0, rl_gt0, area_sorting=True))
                row["delta_metric_square_orig"] = float(_delta_metric_square_from_room_list(rl_gen0))
            except Exception:
                pass

        (
            optimized_hg,
            _optimized_metric,
            _original_metric,
            gt_hg,
            metrics_original,
            metrics_optimized,
            _bedroom_group,
            _expected_counts,
            ag_report,
        ) = optimize_hypergraph(
            _id,
            config,
            grpo_file_path=grpo_file_path,
            artifacts_dir=batch_root / _id,
            preloaded_hg=preloaded_hg,
        )
        output_dir = batch_root / _id
        test_path = _resolve_under_repo(repo_root, str(config.test_dir))
        bounds, _gt_unused, accessgraph0, _rp_unused = load_ground_truth_txt(test_path, _id)
        save_single_apt_artifacts(
            output_dir,
            original_hg0,
            optimized_hg,
            gt_hg,
            bounds,
            _id,
            metrics_original,
            metrics_optimized,
            min_split=config.min_split,
            max_split=config.max_split,
        )
        row.update(
            {
                "original_objective": float(_original_metric),
                "optimized_objective": float(_optimized_metric),
                "delta_metric_orig": float(metrics_original.get("delta_metric", row.get("delta_metric_orig", float("nan")))),
                "delta_metric_opt": float(metrics_optimized.get("delta_metric", float("nan"))),
                "delta_metric_square_orig": float(metrics_original.get("delta_metric_square", row.get("delta_metric_square_orig", float("nan")))),
                "delta_metric_square_opt": float(metrics_optimized.get("delta_metric_square", float("nan"))),
                "area_metric_orig": float(metrics_original.get("area_metric", float("nan"))),
                "area_metric_opt": float(metrics_optimized.get("area_metric", float("nan"))),
                "area_metric_gt_orig": float(metrics_original.get("area_metric_gt", float("nan"))),
                "area_metric_gt_opt": float(metrics_optimized.get("area_metric_gt", float("nan"))),
                "epsilon_metric_orig": float(metrics_original.get("epsilon_metric", float("nan"))),
                "epsilon_metric_opt": float(metrics_optimized.get("epsilon_metric", float("nan"))),
                "ag_check_evals": int((ag_report or {}).get("n_evals", 0) or 0),
                "ag_check_fails": int((ag_report or {}).get("n_fails", 0) or 0),
                "ag_check_first_fail_eval": (ag_report or {}).get("first_fail_eval", None),
            }
        )
        if int((ag_report or {}).get("n_fails", 0) or 0) > 0:
            ag_row_extra = {
                "id": _id,
                "ag_check_fails": int(ag_report.get("n_fails", 0) or 0),
                "ag_check_first_fail_eval": ag_report.get("first_fail_eval", None),
            }

        if isinstance(accessgraph0, dict) and accessgraph0.get("edges") and rgl_b is not None:
            try:
                mesh_orig = hypergraph_bounds_to_mesh(copy.deepcopy(original_hg0), bounds, RGL=rgl_b, Collections=col_b)
                mesh_opt = hypergraph_bounds_to_mesh(copy.deepcopy(optimized_hg), bounds, RGL=rgl_b, Collections=col_b)
                c0 = check_accessgraph_edges_against_mesh(mesh_orig, accessgraph0)
                c1 = check_accessgraph_edges_against_mesh(mesh_opt, accessgraph0)
                row.update(
                    {
                        "ag_edges_total": int(len(c0)),
                        "ag_edges_ok_orig": int(sum(1 for c in c0 if c.get("ok"))),
                        "ag_edges_ok_opt": int(sum(1 for c in c1 if c.get("ok"))),
                    }
                )
            except Exception:
                pass
        merged_hg = copy.deepcopy(optimized_hg)
    except Exception as e:
        print(f"ERROR {_id}: {type(e).__name__}: {e}", flush=True)
        row["status"] = "error"
        row["error"] = f"{type(e).__name__}: {e}"
    return row, merged_hg, ag_row_extra


def _pool_batch_entry(
    pack: tuple[str, OptimConfig, dict, str, str, str],
) -> tuple[dict, dict | None, dict | None]:
    """Picklable top-level: unpack paths and use pool-local RGL."""
    apt_id, config, preloaded_hg, grpo_s, batch_root_s, repo_root_s = pack
    rgl_b, col_b = _POOL_RGL
    return _run_one_batch_apt(
        apt_id,
        config,
        preloaded_hg,
        Path(grpo_s),
        Path(batch_root_s),
        Path(repo_root_s),
        rgl_b,
        col_b,
    )


def main():
    config, apt_id, run_all = parse_args()
    if apt_id is None and not run_all:
        render_from_json(script_dir=Path(__file__).parent)
        return

    repo_root = REPO_ROOT
    if config.grpo_file:
        grpo_file_path = _resolve_under_repo(repo_root, str(config.grpo_file))
    else:
        grpo_file_path = DEFAULT_WMR24_RESULTS

    if run_all:
        # Batch runs go under a single objective-scoped folder, e.g.:
        #   parametric_optimization/output_parametric/wmr24_epsilon/<id>/
        # Override with --batch-output-dir.
        if getattr(config, "batch_output_dir", None):
            batch_root = Path(str(config.batch_output_dir)).expanduser().resolve()
        else:
            batch_root = (
                REPO_ROOT
                / "parametric_optimization"
                / "output_parametric"
                / f"wmr24_{str(config.objective).strip().lower()}"
            )
        batch_root.mkdir(parents=True, exist_ok=True)
        batch_csv = batch_root / f"batch_{Path(grpo_file_path).stem}_{config.objective}.csv"
        batch_ag_csv = batch_root / f"batch_{Path(grpo_file_path).stem}_{config.objective}_accessgraph_penalties.csv"
        rows: list[dict] = []
        ag_rows: list[dict] = []

        results = json.load(open(grpo_file_path, "r", encoding="utf-8"))
        ids = sorted(list(results.keys()))
        excluded_ids: set[str] = set()
        if config.skip_ids_file:
            try:
                skip_path = _resolve_under_repo(repo_root, str(config.skip_ids_file))
                excluded_ids = set(
                    line.strip()
                    for line in skip_path.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.strip().startswith("#")
                )
            except Exception as e:
                print(f"⚠️  Could not read --skip-ids-file: {type(e).__name__}: {e}")
                excluded_ids = set()
        if excluded_ids:
            ids = [i for i in ids if i not in excluded_ids]
        if config.limit is not None and int(config.limit) > 0:
            ids = ids[: int(config.limit)]
        batch_workers = max(1, int(getattr(config, "batch_workers", 1) or 1))
        RGL_B, Collections_B = None, None
        if batch_workers == 1:
            try:
                RGL_B, Collections_B = init_rgl()
            except Exception:
                RGL_B, Collections_B = None, None
        grpo_s = str(grpo_file_path)
        br_s = str(batch_root)
        rp_s = str(repo_root)
        print(
            f"Batch optimize: n={len(ids)} objective={config.objective} grpo_file={grpo_file_path} "
            f"batch_workers={batch_workers}"
        )
        merged_out: dict[str, dict] = {}
        if batch_workers == 1:
            for k, _id in enumerate(ids, start=1):
                print(f"\n[{k}/{len(ids)}] {_id}")
                row, mhg, aextra = _run_one_batch_apt(
                    _id,
                    config,
                    results[_id],
                    grpo_file_path,
                    batch_root,
                    repo_root,
                    RGL_B,
                    Collections_B,
                )
                rows.append(row)
                if mhg is not None:
                    merged_out[_id] = mhg
                if aextra is not None:
                    ag_rows.append(aextra)
        else:
            tasks: list[tuple[str, OptimConfig, dict, str, str, str]] = [
                (_id, config, results[_id], grpo_s, br_s, rp_s) for _id in ids
            ]
            with ProcessPoolExecutor(max_workers=batch_workers, initializer=_mp_init_pool_batch) as ex:
                for k, (row, mhg, aextra) in enumerate(ex.map(_pool_batch_entry, tasks), start=1):
                    _id = str(row.get("id", ""))
                    print(f"\n[{k}/{len(ids)}] done {_id} status={row.get('status', '')}", flush=True)
                    rows.append(row)
                    if mhg is not None and _id:
                        merged_out[_id] = mhg
                    if aextra is not None:
                        ag_rows.append(aextra)

        try:
            fieldnames = [
                "id",
                "objective",
                "status",
                "error",
                "original_objective",
                "optimized_objective",
                "delta_metric_orig",
                "delta_metric_opt",
                "delta_metric_square_orig",
                "delta_metric_square_opt",
                "area_metric_orig",
                "area_metric_opt",
                "area_metric_gt_orig",
                "area_metric_gt_opt",
                "epsilon_metric_orig",
                "epsilon_metric_opt",
                "ag_check_evals",
                "ag_check_fails",
                "ag_check_first_fail_eval",
                "ag_edges_total",
                "ag_edges_ok_orig",
                "ag_edges_ok_opt",
            ]
            save_batch_results(batch_root, batch_csv, batch_ag_csv, rows, ag_rows, config.objective, grpo_file_path, fieldnames=fieldnames)
        except OSError as e:
            print(f"\nCould not write batch CSV: {type(e).__name__}: {e}")
        mp = grpo_file_path.parent / f"{grpo_file_path.stem}_optimized.json"
        try:
            with open(mp, "w", encoding="utf-8") as f:
                json.dump(_to_jsonable(merged_out), f, indent=2, sort_keys=True)
            print(f"Saved merged hypergraphs ({len(merged_out)} ids) → {mp}")
        except OSError as e:
            print(f"⚠️  Could not write merged hypergraphs {mp}: {type(e).__name__}: {e}")
        return

    (
        optimized_hg,
        _optimized_metric,
        _original_metric,
        gt_hg,
        metrics_original,
        metrics_optimized,
        _bedroom_group,
        _expected_counts,
        _ag_report,
    ) = optimize_hypergraph(
        apt_id,
        config,
        grpo_file_path=grpo_file_path,
        artifacts_dir=Path(__file__).parent / "output_parametric" / apt_id,
    )

    output_dir = Path(__file__).parent / "output_parametric" / apt_id
    with open(grpo_file_path, encoding="utf-8") as f:
        grpo_results = json.load(f)
    original_hg = repair_bsp_tree_names(copy.deepcopy(grpo_results[apt_id]))
    test_path = _resolve_under_repo(REPO_ROOT, str(config.test_dir))
    bounds, _, _, _ = load_ground_truth_txt(test_path, apt_id)
    save_single_apt_artifacts(
        output_dir,
        original_hg,
        optimized_hg,
        gt_hg,
        bounds,
        apt_id,
        metrics_original,
        metrics_optimized,
        min_split=config.min_split,
        max_split=config.max_split,
    )


if __name__ == "__main__":
    main()

