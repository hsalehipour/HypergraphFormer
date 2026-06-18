#!/usr/bin/env python3
"""Hypergraph BSP-tree optimization via projected gradient descent on area splits.

Three objectives (see :data:`OBJECTIVES`):

* ``delta``         — δ_sq (perimeter–square per leaf).
* ``epsilon``       — ε vs shared per-room reference fractions.
* ``delta_epsilon`` — equally weighted blend (with auto-rescaled δ term).

Splits live in the hard-coded interval ``[0.15, 0.85]``; learning rate, max
iterations, dataset, batch size etc. are CLI flags.

Examples (run from repo root)::

  python -m scripts.run_parametric b-0001 --objective delta
  python -m scripts.run_parametric b-0001 --objective delta_epsilon --maxiter 80
  python -m scripts.run_parametric --all --limit 5 --objective epsilon
  python -m scripts.run_parametric --all --batch-workers 64 \\
      --objective delta_epsilon --gd-lr 0.15 --maxiter 200

The merged optimized JSON defaults to
``<batch-output-dir>/<input_stem>_optimized.json`` and can be overridden with
``--merged-hypergraphs-json``.
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (REPO_ROOT, SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from utils.hypergraph import repair_bsp_tree_names  # noqa: E402
from utils.metrics import (  # noqa: E402
    compute_room_area_proportion_error,
    compute_room_similarity_score,
    produce_room_area_list,
    produce_room_area_list_from_polygons,
    produce_room_list,
    produce_room_list_from_polygons,
)

MIN_SPLIT, MAX_SPLIT = 0.15, 0.85
ACCESSGRAPH_VIOLATION_WEIGHT = 0.05
OBJECTIVES = ("delta", "delta_epsilon", "epsilon")
DEFAULT_WMR24_RESULTS = REPO_ROOT / "inference_8700_on_wmr24_inference_vllm_64" / "results_hypergraphs.json"

from utils_parametric import (  # noqa: E402
    OptimConfig,
    _delta_metric_square_from_room_list,
    _bedroom_group_from_accessgraph,
    _expected_counts_from_accessgraph,
    _leaf_type_counts_from_hypergraph,
    _resolve_under_repo,
    _validate_apt_id,
    access_graph_from_prediction,
    annotate_area_splits,
    apply_area_splits,
    check_accessgraph_edges_against_mesh,
    collect_area_splits,
    compute_epsilon_metric_vs_ref,
    hypergraph_bounds_to_mesh,
    init_rgl,
    load_ground_truth_txt,
    save_batch_results,
    save_metric_progress_csv,
    validate_leaf_area_floor,
    _to_jsonable,
)


# ---------------------------------------------------------------------------
# Gradient descent helpers
# ---------------------------------------------------------------------------


def _fd_grad(
    fun,
    x: np.ndarray,
    eps: float,
    lo: float,
    hi: float,
) -> np.ndarray:
    """Finite-difference gradient with centered differences when possible.

    For each coordinate, prefer centered ``(f(x+h) - f(x-h)) / (2h)`` since it
    has ``O(h^2)`` truncation error vs ``O(h)`` for forward; fall back to a
    one-sided estimate when the box ``[lo, hi]`` clips one side, and to zero
    when neither perturbation fits.
    """
    x = np.asarray(x, dtype=float)
    h = max(float(eps), 1e-10)
    g = np.zeros_like(x, dtype=float)
    fx_cached: float | None = None
    for i in range(int(x.size)):
        up_ok = x[i] + h <= hi - 1e-15
        dn_ok = x[i] - h >= lo + 1e-15
        if up_ok and dn_ok:
            xp = x.copy(); xp[i] = x[i] + h
            xm = x.copy(); xm[i] = x[i] - h
            g[i] = (float(fun(xp)) - float(fun(xm))) / (2.0 * h)
        elif up_ok:
            if fx_cached is None:
                fx_cached = float(fun(x))
            xp = x.copy(); xp[i] = x[i] + h
            g[i] = (float(fun(xp)) - fx_cached) / h
        elif dn_ok:
            if fx_cached is None:
                fx_cached = float(fun(x))
            xm = x.copy(); xm[i] = x[i] - h
            g[i] = (fx_cached - float(fun(xm))) / h
    return g


# ---------------------------------------------------------------------------
# Per-apartment optimization
# ---------------------------------------------------------------------------


def _epsilon_vs_gt_or_polygons(mesh, gt_mesh, room_polygons) -> float:
    """ε vs GT mesh when available, else fall back to polygon-only GT."""
    pred_a = produce_room_area_list(mesh)
    gt_a = produce_room_area_list(gt_mesh) if gt_mesh is not None else produce_room_area_list_from_polygons(room_polygons or [])
    return float(compute_room_area_proportion_error(pred_a, gt_a, aggregate=False).get("area_proportion_error", float("nan")))


def optimize_hypergraph(
    apt_id: str,
    config: OptimConfig,
    *,
    grpo_file_path: Path | None = None,
    preloaded_hg: dict | None = None,
):
    """Run GD on one apartment.

    Returns ``(optimized_hg, best_metric, original_metric, gt_hg,
    metrics_original, metrics_optimized, bedroom_group, expected_counts,
    ag_report, metric_trace, best_trace, bounds, accessgraph)``.
    """
    apt_id = _validate_apt_id(apt_id)
    objective = str(config.objective).strip().lower()
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")

    # ── Load original hypergraph ──────────────────────────────────────────
    if preloaded_hg is not None:
        original_hg = repair_bsp_tree_names(copy.deepcopy(preloaded_hg))
    else:
        grpo_file = grpo_file_path or (
            _resolve_under_repo(REPO_ROOT, str(config.grpo_file))
            if config.grpo_file else DEFAULT_WMR24_RESULTS
        )
        if not Path(grpo_file).exists():
            raise FileNotFoundError(f"Results hypergraphs file not found: {grpo_file}")
        with open(grpo_file, encoding="utf-8") as f:
            grpo_results = json.load(f)
        if apt_id not in grpo_results:
            raise KeyError(f"{apt_id} not found in {grpo_file}")
        original_hg = repair_bsp_tree_names(copy.deepcopy(grpo_results[apt_id]))

    # ── Load GT bounds + access graph + reference geometry ────────────────
    test_dir = _resolve_under_repo(REPO_ROOT, str(config.test_dir))
    bounds, gt_hg, accessgraph, room_polygons = load_ground_truth_txt(test_dir, apt_id)
    gt_hg = copy.deepcopy(gt_hg)

    ag_target = access_graph_from_prediction(
        original_hg,
        bounds=bounds,
        file_fallback=accessgraph if isinstance(accessgraph, dict) else {},
    )
    print(f"Accessgraph edge penalty: {len(ag_target.get('edges') or [])} edges")

    if isinstance(ag_target, dict) and (ag_target.get("nodes") or ag_target.get("edges")):
        bedroom_group = _bedroom_group_from_accessgraph(ag_target)
    else:
        bedroom_group = _bedroom_group_from_accessgraph(accessgraph if isinstance(accessgraph, dict) else {})

    leaf_counts = _leaf_type_counts_from_hypergraph(original_hg)
    expected_counts = leaf_counts or _expected_counts_from_accessgraph(
        accessgraph if isinstance(accessgraph, dict) else {}
    )
    print(f"Bedroom group: {bedroom_group}, leaf counts: {expected_counts}")

    # ── RGL meshes for GT + initial ──────────────────────────────────────
    RGL, Collections = init_rgl()

    if gt_hg and isinstance(gt_hg, dict) and gt_hg.get("name"):
        gt_mesh = hypergraph_bounds_to_mesh(copy.deepcopy(gt_hg), bounds, RGL=RGL, Collections=Collections)
        gt_room_list = produce_room_list(gt_mesh)
    elif room_polygons:
        gt_mesh = None
        gt_room_list = produce_room_list_from_polygons(room_polygons)
    else:
        raise ValueError(
            f"{apt_id}: {test_dir}/{apt_id}.txt has no GT 'hypergraph' tree and no 'room_polygons'."
        )
    gt_room_names = {r[0] for r in gt_room_list}

    original_mesh = hypergraph_bounds_to_mesh(
        copy.deepcopy(original_hg), bounds, RGL=RGL, Collections=Collections
    )
    original_room_list = produce_room_list(original_mesh)
    orig_delta_gt = float(compute_room_similarity_score(
        original_room_list, gt_room_list, strict=False, area_sorting=True
    ))
    orig_delta_sq = _delta_metric_square_from_room_list(original_room_list)
    orig_eps_ref = compute_epsilon_metric_vs_ref(
        original_mesh, group=bedroom_group, expected_counts=expected_counts,
    )
    orig_eps_gt = _epsilon_vs_gt_or_polygons(original_mesh, gt_mesh, room_polygons)
    print(
        f"Original  soft_δ={orig_delta_gt:.6f}  δ_sq={orig_delta_sq:.6f}  "
        f"ε(ref)={orig_eps_ref:.6f}  ε(GT)={orig_eps_gt:.6f}"
    )

    delta_scale: float = 1.0
    if objective == "delta":
        original_metric = float(orig_delta_sq)
    elif objective == "epsilon":
        original_metric = float(orig_eps_ref)
    else:  # delta_epsilon
        delta_scale = float(orig_eps_ref) / max(float(orig_delta_sq), 1e-9)
        original_metric = 0.5 * (delta_scale * orig_delta_sq) + 0.5 * orig_eps_ref
        print(f"delta_epsilon: delta_scale={delta_scale:.3f}  initial={original_metric:.6f}")

    # ── Collect splits ────────────────────────────────────────────────────
    annotate_area_splits(original_hg)
    _, split_vals = collect_area_splits(original_hg)
    if split_vals:
        print(f"\n{len(split_vals)} area_split params, range=[{min(split_vals):.3f}, {max(split_vals):.3f}]")
    else:
        print("\nNo area_split params; nothing to optimize.")

    # ── Eval closure (objective + accessgraph penalty) ───────────────────
    ag_eval_counter = 0
    ag_fail_count = 0
    ag_first_fail_eval: int | None = None
    ag_fail_evals_sample: list[int] = []

    def _eval(splits_vec: np.ndarray) -> tuple[float, dict]:
        nonlocal ag_eval_counter, ag_fail_count, ag_first_fail_eval
        ag_eval_counter += 1
        hg = copy.deepcopy(original_hg)
        try:
            apply_area_splits(hg, splits_vec)
            mesh = hypergraph_bounds_to_mesh(
                copy.deepcopy(hg), bounds, RGL=RGL, Collections=Collections,
            )
            room_list = produce_room_list(mesh)
            if {r[0] for r in room_list} != gt_room_names:
                return 1.0, hg

            penalty = 0.0
            if (
                config.with_accessgraph_penalty
                and isinstance(ag_target, dict)
                and ag_target.get("edges")
            ):
                edges_check = check_accessgraph_edges_against_mesh(mesh, ag_target)
                n_failed = sum(1 for e in edges_check if not e.get("ok", True))
                penalty = ACCESSGRAPH_VIOLATION_WEIGHT * float(n_failed)
                if n_failed > 0:
                    ag_fail_count += 1
                    if ag_first_fail_eval is None:
                        ag_first_fail_eval = ag_eval_counter
                        print(
                            f"Accessgraph violation at eval#{ag_eval_counter}: "
                            f"penalty=+{penalty:.3f} ({n_failed} edge(s) failing)",
                            flush=True,
                        )
                    if len(ag_fail_evals_sample) < 50:
                        ag_fail_evals_sample.append(ag_eval_counter)

            if objective == "delta":
                return _delta_metric_square_from_room_list(room_list) + penalty, hg
            eps = compute_epsilon_metric_vs_ref(
                mesh, group=bedroom_group, expected_counts=expected_counts,
            )
            if objective == "epsilon":
                return float(eps) + penalty, hg
            delta_sq = _delta_metric_square_from_room_list(room_list)
            return 0.5 * (delta_scale * delta_sq) + 0.5 * float(eps) + penalty, hg
        except Exception:
            return 1.0, hg

    # ── Run GD ────────────────────────────────────────────────────────────
    width = MAX_SPLIT - MIN_SPLIT
    lr = float(config.gd_lr) if config.gd_lr is not None else 0.1 * width
    if lr <= 0:
        raise ValueError("gd_lr must be > 0")
    fd_eps = max(1e-4, 0.02 * width)
    n_steps = int(config.maxiter)
    if n_steps < 0:
        raise ValueError("maxiter must be >= 0")
    ftol = float(config.ftol)
    patience = (
        max(10, n_steps // 5) if config.patience is None else max(0, int(config.patience))
    )

    if not split_vals:
        best_hg = copy.deepcopy(original_hg)
        leaf_stats = validate_leaf_area_floor(best_hg)[1]
        metric_trace: list[float] = []
        best_trace: list[float] = []
    else:
        x0 = np.asarray(split_vals, dtype=float)
        f0, _ = _eval(x0)
        best = {"f": float(f0), "x": x0.copy()}
        metric_trace = [float(f0)]
        best_trace = [float(f0)]

        print(
            f"\nGD: steps={n_steps} lr={lr:.4f} fd_eps={fd_eps:.4f} "
            f"patience={patience} ftol={ftol:g}",
            flush=True,
        )
        xk = x0.copy()
        since_improved = 0
        steps_taken = 0
        for steps_taken in range(1, n_steps + 1):
            g = _fd_grad(lambda x: _eval(x)[0], xk, fd_eps, MIN_SPLIT, MAX_SPLIT)
            xk = np.clip(xk - lr * g, MIN_SPLIT, MAX_SPLIT)
            f, _ = _eval(xk)
            if f < best["f"] - ftol:
                best["f"] = float(f)
                best["x"] = xk.copy()
                since_improved = 0
            else:
                since_improved += 1
            metric_trace.append(float(f))
            best_trace.append(float(best["f"]))
            if patience > 0 and since_improved >= patience:
                print(
                    f"GD early-stop at step {steps_taken}: "
                    f"best f={best['f']:.6f} stable for {patience} steps "
                    f"(saved {n_steps - steps_taken} steps).",
                    flush=True,
                )
                break
        else:
            print(f"GD done at maxiter={n_steps}: best f={best['f']:.6f}")

        _, best_hg = _eval(best["x"])
        leaf_stats = validate_leaf_area_floor(best_hg)[1]

    print(f"Leaf-area floor: min_observed={leaf_stats.get('min_observed_leaf_frac', 0):.4f}")

    # ── Final metrics on the best hypergraph ─────────────────────────────
    best_mesh = hypergraph_bounds_to_mesh(
        copy.deepcopy(best_hg), bounds, RGL=RGL, Collections=Collections,
    )
    best_room_list = produce_room_list(best_mesh)
    best_room_names = {r[0] for r in best_room_list}

    ag_report = {"n_evals": ag_eval_counter, "n_fails": ag_fail_count,
                 "first_fail_eval": ag_first_fail_eval, "fail_evals_sample": ag_fail_evals_sample}
    metrics_original = {"delta_metric": orig_delta_gt, "delta_metric_square": orig_delta_sq,
                        "epsilon_metric": orig_eps_gt, "epsilon_metric_ref": orig_eps_ref}

    if best_room_names != gt_room_names:
        return (
            original_hg, float(original_metric), float(original_metric), gt_hg,
            metrics_original, dict(metrics_original),
            bedroom_group, expected_counts, ag_report,
            metric_trace, best_trace, bounds, accessgraph,
        )

    best_delta_gt = float(compute_room_similarity_score(
        best_room_list, gt_room_list, strict=False, area_sorting=True
    ))
    best_delta_sq = _delta_metric_square_from_room_list(best_room_list)
    best_eps_ref = compute_epsilon_metric_vs_ref(
        best_mesh, group=bedroom_group, expected_counts=expected_counts,
    )
    best_eps_gt = _epsilon_vs_gt_or_polygons(best_mesh, gt_mesh, room_polygons)

    if objective == "delta":
        best_metric = float(best_delta_sq)
    elif objective == "epsilon":
        best_metric = float(best_eps_ref)
    else:
        best_metric = 0.5 * (delta_scale * best_delta_sq) + 0.5 * best_eps_ref

    metrics_optimized = {
        "delta_metric": float(best_delta_gt),
        "delta_metric_square": float(best_delta_sq),
        "epsilon_metric": float(best_eps_gt),
        "epsilon_metric_ref": float(best_eps_ref),
    }

    fmt = lambda v: format(v, ".2g") if math.isfinite(v) else "nan"  # noqa: E731
    print("\n=== Metrics (lower is better) ===")
    for k, lbl in (("delta_metric", "soft_δ(GT)"), ("delta_metric_square", "δ_sq"),
                   ("epsilon_metric", "ε(GT)"), ("epsilon_metric_ref", "ε(ref)")):
        o, n = metrics_original[k], metrics_optimized[k]
        print(f"  {lbl:<14}  orig={fmt(o)}  opt={fmt(n)}  Δ={fmt(n - o)}")
    print(f"Objective {objective}: best={best_metric:.6f} (Δ {best_metric - original_metric:+.6f})")

    return (
        best_hg, float(best_metric), float(original_metric), gt_hg,
        metrics_original, metrics_optimized,
        bedroom_group, expected_counts, ag_report,
        metric_trace, best_trace, bounds, accessgraph,
    )


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------


_POOL_RGL: tuple[object, object] = (None, None)


def _pool_init() -> None:
    """ProcessPool initializer: one RGL per worker, single-threaded BLAS."""
    global _POOL_RGL
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    try:
        _POOL_RGL = init_rgl()
    except Exception:
        _POOL_RGL = (None, None)


def _process_one_apt(
    apt_id: str,
    config: OptimConfig,
    preloaded_hg: dict,
    grpo_file_path: Path,
    rgl,
    collections,
) -> tuple[dict, dict | None, list[float], list[float]]:
    """Optimize a single apartment for batch mode.

    Returns ``(csv_row, optimized_hg|None, metric_trace, best_trace)``.
    """
    row: dict = {
        "id": apt_id,
        "objective": config.objective,
        "status": "ok",
        "error": "",
    }
    try:
        (
            optimized_hg,
            optimized_metric,
            original_metric,
            _gt_hg,
            metrics_original,
            metrics_optimized,
            _bedroom_group,
            _expected_counts,
            ag_report,
            metric_trace,
            best_trace,
            bounds,
            accessgraph0,
        ) = optimize_hypergraph(
            apt_id,
            config,
            grpo_file_path=grpo_file_path,
            preloaded_hg=preloaded_hg,
        )

        nan = float("nan")
        row.update({"original_objective": float(original_metric), "optimized_objective": float(optimized_metric),
                    "ag_check_evals": int(ag_report.get("n_evals", 0) or 0),
                    "ag_check_fails": int(ag_report.get("n_fails", 0) or 0),
                    "ag_check_first_fail_eval": ag_report.get("first_fail_eval")})
        for k in ("delta_metric", "delta_metric_square", "epsilon_metric", "epsilon_metric_ref"):
            row[f"{k}_orig"] = float(metrics_original.get(k, nan))
            row[f"{k}_opt"] = float(metrics_optimized.get(k, nan))

        if isinstance(accessgraph0, dict) and accessgraph0.get("edges") and rgl is not None:
            try:
                hg0 = repair_bsp_tree_names(copy.deepcopy(preloaded_hg))
                m0 = hypergraph_bounds_to_mesh(copy.deepcopy(hg0), bounds, RGL=rgl, Collections=collections)
                m1 = hypergraph_bounds_to_mesh(copy.deepcopy(optimized_hg), bounds, RGL=rgl, Collections=collections)
                c0, c1 = (check_accessgraph_edges_against_mesh(m, accessgraph0) for m in (m0, m1))
                row.update({"ag_edges_total": len(c0), "ag_edges_ok_orig": sum(1 for c in c0 if c.get("ok")),
                            "ag_edges_ok_opt": sum(1 for c in c1 if c.get("ok"))})
            except Exception:
                pass

        return row, copy.deepcopy(optimized_hg), metric_trace, best_trace
    except Exception as e:
        print(f"ERROR {apt_id}: {type(e).__name__}: {e}", flush=True)
        row["status"] = "error"
        row["error"] = f"{type(e).__name__}: {e}"
        return row, None, [], []


def _pool_entry(pack):
    apt_id, config, preloaded_hg, grpo_s, _batch_s = pack
    rgl, col = _POOL_RGL
    return _process_one_apt(apt_id, config, preloaded_hg, Path(grpo_s), rgl, col)


def _run_batch(config: OptimConfig) -> None:
    grpo_file_path = (
        _resolve_under_repo(REPO_ROOT, str(config.grpo_file)) if config.grpo_file
        else DEFAULT_WMR24_RESULTS
    )
    batch_root = (
        Path(config.batch_output_dir).expanduser().resolve() if config.batch_output_dir
        else REPO_ROOT / "parametric_optimization" / "output_parametric" / f"{config.dataset}_{config.objective}"
    )
    batch_root.mkdir(parents=True, exist_ok=True)

    with open(grpo_file_path, encoding="utf-8") as f:
        results = json.load(f)
    ids = sorted(results.keys())
    if config.skip_ids_file:
        try:
            skip = {l.strip() for l in _resolve_under_repo(REPO_ROOT, str(config.skip_ids_file))
                    .read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")}
            ids = [i for i in ids if i not in skip]
        except Exception as e:
            print(f"⚠️  Could not read --skip-ids-file: {e}")
    if config.limit and config.limit > 0:
        ids = ids[: config.limit]

    workers = max(1, int(config.batch_workers))
    print(f"Batch optimize: n={len(ids)} objective={config.objective} grpo_file={grpo_file_path} workers={workers}")

    rows: list[dict] = []
    merged: dict[str, dict] = {}
    all_traces: list[tuple[str, list[float], list[float]]] = []

    def _accumulate(row, mhg, mt, bt):
        _id = str(row.get("id", ""))
        rows.append(row)
        if mhg is not None:
            merged[_id] = mhg
        if mt:
            all_traces.append((_id, mt, bt))

    if workers == 1:
        rgl, col = init_rgl() if True else (None, None)
        for k, _id in enumerate(ids, start=1):
            print(f"\n[{k}/{len(ids)}] {_id}")
            _accumulate(*_process_one_apt(_id, config, results[_id], grpo_file_path, rgl, col))
    else:
        tasks = [(_id, config, results[_id], str(grpo_file_path), str(batch_root)) for _id in ids]
        with ProcessPoolExecutor(max_workers=workers, initializer=_pool_init) as ex:
            for k, result in enumerate(ex.map(_pool_entry, tasks), start=1):
                print(f"\n[{k}/{len(ids)}] done {result[0].get('id','')} status={result[0].get('status','')}", flush=True)
                _accumulate(*result)

    save_batch_results(batch_root, batch_root / f"batch_{Path(grpo_file_path).stem}_{config.objective}.csv",
                       rows, config.objective, grpo_file_path)
    if all_traces:
        progress_csv = batch_root / "metric_progress.csv"
        save_metric_progress_csv(progress_csv, all_traces)
        print(f"Saved metric progress CSV ({len(all_traces)} apts) → {progress_csv}")

    merged_path = (
        _resolve_under_repo(REPO_ROOT, str(config.merged_hypergraphs_json)) if config.merged_hypergraphs_json
        else batch_root / f"{Path(grpo_file_path).stem}_optimized.json"
    )
    try:
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        with open(merged_path, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(merged), f, indent=2, sort_keys=True)
        print(f"Saved merged hypergraphs ({len(merged)} ids) → {merged_path}")
    except OSError as e:
        print(f"⚠️  Could not write merged hypergraphs: {e}")


def _run_single(config: OptimConfig, apt_id: str) -> None:
    grpo_file_path = (
        _resolve_under_repo(REPO_ROOT, str(config.grpo_file)) if config.grpo_file
        else DEFAULT_WMR24_RESULTS
    )
    *_, metric_trace, best_trace, _bounds, _ag = optimize_hypergraph(
        apt_id, config, grpo_file_path=grpo_file_path
    )

    if metric_trace:
        out_dir = SCRIPTS_DIR / "output_parametric"
        trace_csv = out_dir / f"{apt_id}_metric_trace.csv"
        save_metric_progress_csv(trace_csv, [(apt_id, metric_trace, best_trace)])
        print(f"Saved metric trace → {trace_csv}")


def main() -> None:
    config, apt_id, run_all = OptimConfig.from_args()
    if run_all:
        _run_batch(config)
    elif apt_id:
        _run_single(config, apt_id)
    else:
        raise SystemExit("Provide an apt_id or use --all.")


if __name__ == "__main__":
    main()
