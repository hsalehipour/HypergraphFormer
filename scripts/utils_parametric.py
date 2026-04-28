import argparse
import copy
import csv
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.hypergraph import Hypergraph, HypergraphPlotter
from utils.metrics import (
    compute_room_area_proportion_error,
    compute_room_similarity_score,
    produce_room_area_list,
    produce_room_list,
)

PLOTTER = HypergraphPlotter()

# Default inference JSON for WMR24 (overridable via --grpo-file / --results-hypergraphs)
DEFAULT_WMR24_RESULTS = (
    REPO_ROOT / "inference_8700_on_wmr24_inference_vllm_64" / "results_hypergraphs.json"
)
DEFAULT_WMR24_TEST_DIR = REPO_ROOT / "dataset" / "wmr24" / "test"


def init_rgl():
    """Pre-initialise RGL; same entry point as :meth:`Hypergraph.init_rgl`."""
    return Hypergraph.init_rgl()


def hypergraph_bounds_to_mesh(hg: dict, bounds: dict, *, RGL=None, Collections=None):
    """Build an RGL room mesh from a hypergraph dict and dataset ``bounds``."""
    return Hypergraph(copy.deepcopy(hg)).to_mesh(
        bounds=bounds, RGL=RGL, Collections=Collections
    )


def compute_delta_metric_soft_from_room_lists(
    room_list_a, room_list_b, *, area_sorting: bool = True
) -> float:
    """Same soft δ as :func:`utils.metrics.compute_room_similarity_score` (non-strict)."""
    return float(
        compute_room_similarity_score(
            room_list_a, room_list_b, strict=False, area_sorting=area_sorting
        )
    )


def compute_epsilon_metric_vs_gt(pred_mesh, gt_mesh) -> float:
    """Mean area-proportion error vs GT; same as :mod:`scripts.explore_rotation` (``aggregate=False``)."""
    pred_area = produce_room_area_list(pred_mesh)
    gt_area = produce_room_area_list(gt_mesh)
    d = compute_room_area_proportion_error(pred_area, gt_area, aggregate=False)
    return float(d["area_proportion_error"])


def ref_room_area_list_from_targets(
    group: str,
    expected_counts: dict[str, int] | None,
    targets_by_group: dict[str, dict[str, float]] | None = None,
) -> list[tuple[str, float]]:
    """
    Build a synthetic ``(merge_id, area)`` list from :data:`AREA_REF_TARGETS_ABS` and
    per-type counts, for :func:`compute_room_area_proportion_error` (``aggregate=False``).
    """
    tb = targets_by_group if targets_by_group is not None else AREA_REF_TARGETS_ABS
    targets = tb.get(str(group).strip().lower())
    if not targets:
        return []
    exp = expected_counts or {}
    out: list[tuple[str, float]] = []
    for k, a_tgt in targets.items():
        n = int(exp.get(k, 0) or 0)
        if n < 1:
            continue
        a = float(a_tgt)
        if not math.isfinite(a) or a < 0.0:
            continue
        for _ in range(n):
            out.append((str(k), a))
    return out


def compute_epsilon_metric_vs_ref(
    pred_mesh,
    *,
    group: str,
    expected_counts: dict[str, int] | None,
    targets_by_group: dict[str, dict[str, float]] | None = None,
) -> float:
    """
    Same structure as :func:`compute_epsilon_metric_vs_gt` (mean |prop_pred - prop_ref|
    per room, ``aggregate=False``), but the reference apartment uses absolute areas from
    :data:`AREA_REF_TARGETS_ABS` and *expected_counts* (repeated per room instance) instead
    of the GT mesh.
    """
    from utils.metrics import _room_sort_key

    pred_area = produce_room_area_list(pred_mesh)
    ref_area = ref_room_area_list_from_targets(
        group, expected_counts, targets_by_group=targets_by_group
    )
    if not ref_area:
        return 1.0
    ref_area = sorted(ref_area, key=_room_sort_key)
    tot = sum(a for _merge_id, a in ref_area)
    if not math.isfinite(tot) or tot <= 0.0:
        return 1.0
    d = compute_room_area_proportion_error(pred_area, ref_area, aggregate=False)
    v = float(d.get("area_proportion_error", float("nan")))
    if not math.isfinite(v):
        return 1.0
    return v


def access_graph_from_prediction(
    hg: dict,
    *,
    bounds: dict | None = None,
    file_fallback: dict | None = None,
) -> dict:
    """Model access graph. With *bounds*, use :meth:`Hypergraph.correct_accessgraph` when it yields edges, else raw graph, else *file_fallback* (e.g. dataset GT)."""
    fb = file_fallback if isinstance(file_fallback, dict) else {}
    if bounds is not None:
        try:
            c = Hypergraph(copy.deepcopy(hg)).correct_accessgraph(bounds)
            if isinstance(c, dict) and c.get("edges"):
                return c
        except Exception:
            pass
    try:
        raw = Hypergraph(copy.deepcopy(hg)).access_graph
    except Exception:
        raw = None
    raw = raw if isinstance(raw, dict) else {}
    return raw if raw.get("edges") else (fb if fb.get("edges") else {})


def _parse_jsonish(val) -> object:
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return {}
        return json.loads(s)
    return val if val is not None else {}


def load_ground_truth_txt(
    test_dir: Path, sample_id: str
) -> tuple[dict, dict, dict, list | None]:
    """
    Load ``<test_dir>/<id>.txt``: ``bounds``, ``hypergraph``, ``accessgraph`` (RPLAN/WMR24 style),
    and optional ``room_polygons`` (RPLAN eval_test: polygons only, no BSP tree).
    """
    path = test_dir / f"{sample_id}.txt"
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    bounds = _parse_jsonish(raw.get("bounds", "{}"))
    if not isinstance(bounds, dict):
        bounds = {}
    hg = raw.get("hypergraph")
    if isinstance(hg, str):
        hg = json.loads(hg)
    if not isinstance(hg, dict):
        hg = {}
    ag = raw.get("accessgraph") or {}
    if isinstance(ag, str):
        ag = json.loads(ag) if ag else {}
    if not isinstance(ag, dict):
        ag = {}
    rp = raw.get("room_polygons")
    if not isinstance(rp, list):
        rp = None
    return bounds, hg, ag, rp

# AREA_REF_TARGETS_ABS: dict[str, dict[str, float]] = {
#     "studio": {"bath": 4.5, "kitchen": 7.5, "living": 24.0},
#     "one_bed": {"bath": 5.0, "bed": 13.0, "kitchen": 9.0, "living": 18.0},
#     "two_plus": {"bath": 5.5, "bed": 14.0, "kitchen": 12.0, "living": 22.0},
# }


AREA_REF_TARGETS_ABS: dict[str, dict[str, float]] = {
    "studio": {},
    "one_bed": {"bath": 1145.5417, "bed": 3762.2961, "kitchen": 1626.7943, "living": 6273.9704, "balcony": 1463.5588, "storage": 1200.5263},
    "two_plus": {"bath": 882.8436, "bed": 2591.3825, "kitchen": 1246.7811, "living": 7044.4455, "balcony": 1021.3723, "storage": 611.3082},
}


@dataclass
class OptimConfig:
    """Optimization parameters with defaults."""

    seed: int | None = None
    min_split: float = 0.25
    max_split: float = 0.75
    min_leaf_frac: float = 0.03
    maxiter: int = 50
    make_gif: bool = False
    gif_frames: int = 10
    gif_duration: float = 0.8
    objective: str = "delta"
    area_weight: float = 1.0
    scipy_eps: float | None = None
    scipy_ftol: float | None = None
    scipy_gtol: float | None = None
    init_samples: int | None = None
    init_mix: float | None = None
    brave: bool = False
    grpo_file: str | None = None
    test_dir: str = "dataset/wmr24/test"
    limit: int | None = None
    skip_ids_file: str | None = None
    artifacts_dir: Path | None = None
    # "lbfgs" = SciPy L-BFGS-B (default). "gd" = projected gradient descent, FD gradients.
    optimizer: str = "lbfgs"
    gd_lr: float | None = None  # if None, 0.1 * (max_split - min_split)
    # ProcessPool workers for --all (1 = current sequential loop).
    batch_workers: int = 1
    # If set, --all writes per-id dirs and batch CSV/JSON under this path (else default wmr24_<objective>).
    batch_output_dir: str | None = None

    @classmethod
    def from_args(cls, args: list[str] | None = None) -> tuple["OptimConfig", str | None, bool]:
        """Parse CLI. Returns (config, apt_id, run_all). apt_id is None when run_all."""
        p = argparse.ArgumentParser(description="Hypergraph parametric optimization")
        p.add_argument("apt_id", nargs="?", help="Apartment ID (or use --all for batch)")
        p.add_argument("--all", action="store_true", help="Batch mode over GRPO results")
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--min-split", type=float, default=0.25)
        p.add_argument("--max-split", type=float, default=0.75)
        p.add_argument("--min-leaf-frac", type=float, default=0.03)
        p.add_argument("--maxiter", type=int, default=50)
        p.add_argument("--gif", action="store_true")
        p.add_argument("--gif-frames", type=int, default=10)
        p.add_argument("--gif-duration", type=float, default=0.8)
        p.add_argument(
            "--objective",
            type=str,
            default="delta",
            choices=["delta", "area", "area_delta", "delta_epsilon", "epsilon"],
        )
        p.add_argument("--area-weight", type=float, default=1.0)
        p.add_argument("--scipy-eps", type=float, default=None)
        p.add_argument("--scipy-ftol", type=float, default=None)
        p.add_argument("--scipy-gtol", type=float, default=None)
        p.add_argument("--init-samples", type=int, default=None)
        p.add_argument("--init-mix", type=float, default=None)
        p.add_argument("--brave", action="store_true")
        p.add_argument(
            "--grpo-file",
            "--results-hypergraphs",
            type=str,
            default=None,
            dest="grpo_file",
            help="JSON dict sample_id -> hypergraph (e.g. inference results). "
            f"Default: {DEFAULT_WMR24_RESULTS.name} under repo root when omitted.",
        )
        p.add_argument(
            "--test-dir",
            type=str,
            default="dataset/wmr24/test",
            help="Ground truth folder with <id>.txt (default: WMR24 test split).",
        )
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--skip-ids-file", type=str, default=None)
        p.add_argument("--area", action="store_true", help="Objective: area")
        p.add_argument("--delta", action="store_true", help="Objective: delta")
        p.add_argument(
            "--epsilon",
            action="store_true",
            help="Objective: epsilon (area proportion error vs ref: AREA_REF_TARGETS_ABS + counts; not vs GT mesh)",
        )
        p.add_argument(
            "--area-delta",
            "--area_delta",
            dest="area_delta",
            action="store_true",
            help="Objective: area_delta (50/50: δ_sq + area ref; δ_sq same as --objective delta)",
        )
        p.add_argument(
            "--delta-epsilon",
            "--delta_epsilon",
            dest="delta_epsilon",
            action="store_true",
            help="Objective: delta_epsilon (50/50 soft δ vs GT + ref ε; area_delta uses δ_sq instead)",
        )
        p.add_argument(
            "--optimizer",
            type=str,
            default="lbfgs",
            choices=["lbfgs", "gd"],
            help="lbfgs: SciPy L-BFGS-B (default). gd: vanilla projected gradient descent with finite-difference gradients.",
        )
        p.add_argument(
            "--gd-lr",
            "--gd_lr",
            dest="gd_lr",
            type=float,
            default=None,
            help="Step size for --optimizer gd; default: 0.1 * (max_split - min_split).",
        )
        p.add_argument(
            "--batch-workers",
            type=int,
            default=1,
            help="--all only: run N processes in parallel (1 = sequential). Set OMP/BLAS to 1 thread per process.",
        )
        p.add_argument(
            "--batch-output-dir",
            type=str,
            default=None,
            help="--all only: root directory for per-id subfolders, batch CSV, and merged *_optimized.json "
            "(default: parametric_optimization/output_parametric/wmr24_<objective> under repo root).",
        )
        parsed = p.parse_args(args)
        obj = parsed.objective
        if parsed.area:
            obj = "area"
        elif parsed.delta:
            obj = "delta"
        elif parsed.epsilon:
            obj = "epsilon"
        elif parsed.area_delta:
            obj = "area_delta"
        elif parsed.delta_epsilon:
            obj = "delta_epsilon"
        w = int(parsed.batch_workers)
        if w < 1:
            p.error("--batch-workers must be >= 1")
        cfg = cls(
            seed=parsed.seed,
            min_split=parsed.min_split,
            max_split=parsed.max_split,
            min_leaf_frac=parsed.min_leaf_frac,
            maxiter=parsed.maxiter,
            make_gif=parsed.gif,
            gif_frames=parsed.gif_frames,
            gif_duration=parsed.gif_duration,
            objective=obj,
            area_weight=parsed.area_weight,
            scipy_eps=parsed.scipy_eps,
            scipy_ftol=parsed.scipy_ftol,
            scipy_gtol=parsed.scipy_gtol,
            init_samples=parsed.init_samples,
            init_mix=parsed.init_mix,
            brave=parsed.brave,
            grpo_file=parsed.grpo_file,
            test_dir=parsed.test_dir,
            limit=parsed.limit,
            skip_ids_file=parsed.skip_ids_file,
            optimizer=str(parsed.optimizer).lower(),
            gd_lr=parsed.gd_lr,
            batch_workers=w,
            batch_output_dir=parsed.batch_output_dir,
        )
        apt_id = getattr(parsed, "apt_id", None) if not parsed.all else None
        if apt_id:
            apt_id = _validate_apt_id(str(apt_id))
        return cfg, apt_id, parsed.all


def parse_args(args: list[str] | None = None) -> tuple[OptimConfig, str | None, bool]:
    """Parse CLI. Returns (config, apt_id, run_all)."""
    return OptimConfig.from_args(args)


def check_accessgraph_edges_against_mesh(
    mesh,
    accessgraph: dict,
    *,
    scale_factor: float = 1.05,
    min_overlap_area: float = 1e-12,
) -> list[dict]:
    from shapely.affinity import scale as _scale

    n = int(getattr(getattr(mesh, "faceList", None), "Count", 0) or 0)
    if n <= 0:
        return []

    face_types: list[str] = []
    polys: list[Polygon] = []
    for i in range(n):
        face = mesh.faceList[i]
        face_types.append(str(getattr(face, "merge_id", "unknown")).strip().lower())
        pts = face.getPoints()
        corners = [(float(pts[j].X), float(pts[j].Y)) for j in range(int(getattr(pts, "Count", 0) or 0))]
        polys.append(Polygon(corners) if len(corners) >= 3 else Polygon())

    polys_scaled: list[Polygon] = []
    for p in polys:
        if p.is_empty:
            polys_scaled.append(p)
            continue
        try:
            polys_scaled.append(_scale(p, xfact=float(scale_factor), yfact=float(scale_factor), origin="centroid"))
        except Exception:
            polys_scaled.append(p)

    adj_type_pairs: set[tuple[str, str]] = set()
    for i in range(n):
        if polys_scaled[i].is_empty:
            continue
        ti = face_types[i]
        for j in range(i + 1, n):
            if polys_scaled[j].is_empty:
                continue
            try:
                inter_area = float(polys_scaled[i].intersection(polys_scaled[j]).area or 0.0)
            except Exception:
                inter_area = 0.0
            if inter_area > float(min_overlap_area):
                tj = face_types[j]
                a, b = (ti, tj) if ti <= tj else (tj, ti)
                adj_type_pairs.add((a, b))

    nodes = (accessgraph or {}).get("nodes", {}) or {}
    out: list[dict] = []
    for e in (accessgraph or {}).get("edges", []) or []:
        if not isinstance(e, (list, tuple)) or len(e) != 2:
            continue
        u, v = str(e[0]), str(e[1])
        tu = str(nodes.get(u, "unknown")).strip().lower()
        tv = str(nodes.get(v, "unknown")).strip().lower()
        a, b = (tu, tv) if tu <= tv else (tv, tu)
        ok = (a, b) in adj_type_pairs
        out.append({"u": u, "v": v, "u_type": tu, "v_type": tv, "ok": bool(ok)})
    return out


def get_mesh_adjacency_type_pairs(
    mesh,
    *,
    scale_factor: float = 1.05,
    min_overlap_area: float = 1e-12,
) -> set[tuple[str, str]]:
    """Return set of (type_a, type_b) pairs for adjacent room types in mesh (sorted)."""
    from shapely.affinity import scale as _scale

    n = int(getattr(getattr(mesh, "faceList", None), "Count", 0) or 0)
    if n <= 0:
        return set()

    face_types: list[str] = []
    polys: list[Polygon] = []
    for i in range(n):
        face = mesh.faceList[i]
        face_types.append(str(getattr(face, "merge_id", "unknown")).strip().lower())
        pts = face.getPoints()
        corners = [(float(pts[j].X), float(pts[j].Y)) for j in range(int(getattr(pts, "Count", 0) or 0))]
        polys.append(Polygon(corners) if len(corners) >= 3 else Polygon())

    polys_scaled: list[Polygon] = []
    for p in polys:
        if p.is_empty:
            polys_scaled.append(p)
            continue
        try:
            polys_scaled.append(_scale(p, xfact=float(scale_factor), yfact=float(scale_factor), origin="centroid"))
        except Exception:
            polys_scaled.append(p)

    adj_type_pairs: set[tuple[str, str]] = set()
    for i in range(n):
        if polys_scaled[i].is_empty:
            continue
        ti = face_types[i]
        for j in range(i + 1, n):
            if polys_scaled[j].is_empty:
                continue
            try:
                inter_area = float(polys_scaled[i].intersection(polys_scaled[j]).area or 0.0)
            except Exception:
                inter_area = 0.0
            if inter_area > float(min_overlap_area):
                tj = face_types[j]
                a, b = (ti, tj) if ti <= tj else (tj, ti)
                adj_type_pairs.add((a, b))
    return adj_type_pairs


def _validate_apt_id(apt_id: str) -> str:
    if not isinstance(apt_id, str):
        raise ValueError("apt_id must be a string")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", apt_id):
        raise ValueError(f"Invalid apt_id '{apt_id}'. Allowed: letters, digits, '_' and '-'.")
    return apt_id


def _resolve_under_repo(repo_root: Path, p: str) -> Path:
    rp = Path(p).expanduser().resolve()
    try:
        rp.relative_to(repo_root)
    except Exception as e:
        raise ValueError(f"Refusing to use path outside repo root: {rp}\nRepo root: {repo_root}") from e
    return rp


def _canon_room_type(s: str) -> str:
    t = str(s).strip().lower()
    if "bath" in t:
        return "bath"
    if t in ("bedroom", "bed"):
        return "bed"
    return t


def _bedroom_group_from_accessgraph(accessgraph: dict | None) -> str:
    nodes = (accessgraph or {}).get("nodes", {}) or {}
    n_bed = 0
    for _k, v in nodes.items():
        t = str(v).strip().lower()
        if t in ("bed", "bedroom"):
            n_bed += 1
    if n_bed <= 0:
        return "studio"
    if n_bed == 1:
        return "one_bed"
    return "two_plus"


def _leaf_type_counts_from_hypergraph(hg: dict) -> dict[str, int]:
    """Count room types in hypergraph leaves (preorder)."""
    counts: dict[str, int] = {}
    stack = [hg]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        children = node.get("children") or []
        if isinstance(children, list) and len(children) > 0:
            for c in children:
                stack.append(c)
            continue
        t = _canon_room_type(str(node.get("mergeid", node.get("name", "unknown"))))
        counts[t] = int(counts.get(t, 0)) + 1
    return counts


def _expected_counts_from_accessgraph(accessgraph: dict | None) -> dict[str, int]:
    nodes = (accessgraph or {}).get("nodes", {}) or {}
    counts: dict[str, int] = {}
    for _k, v in nodes.items():
        ck = _canon_room_type(str(v))
        counts[ck] = int(counts.get(ck, 0)) + 1
    return counts


def _mesh_areas_by_type_abs(mesh) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for i in range(mesh.faceList.Count):
        face = mesh.faceList[i]
        k = _canon_room_type(str(getattr(face, "merge_id", "unknown")))
        a = float(getattr(face, "Area", 0.0) or 0.0)
        if a <= 0.0:
            continue
        out.setdefault(k, []).append(float(a))
    return out


def _mesh_areas_by_type_from_polygons(room_polygons: list | None) -> dict[str, list[float]]:
    """Same keying as :func:`_mesh_areas_by_type_abs` for GT from RPLAN ``room_polygons``."""
    from shapely.geometry import Polygon

    out: dict[str, list[float]] = {}
    for rp in room_polygons or []:
        if not isinstance(rp, dict):
            continue
        coords = rp.get("coords", [])
        if len(coords) < 3:
            continue
        poly = Polygon(coords)
        if poly.is_empty or poly.area < 1e-6:
            continue
        k = _canon_room_type(str(rp.get("mergeid", "unknown")))
        out.setdefault(k, []).append(float(poly.area))
    for k in list(out):
        out[k] = sorted(
            [float(x) for x in out[k] if math.isfinite(float(x))], reverse=True
        )
    return out


def _delta_metric_square_from_room_list(room_list: list[tuple[str, float, float]]) -> float:
    vals: list[float] = []
    for _name, L, L_sq in room_list:
        L = float(L)
        L_sq = float(L_sq)
        if not math.isfinite(L) or L <= 0.0:
            continue
        if not math.isfinite(L_sq) or L_sq <= 0.0:
            continue
        vals.append(abs(1.0 - (L_sq / L)))
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _area_metric_gt_from_type_maps(
    gen_mesh, gt_by: dict[str, list[float]]
) -> float:
    """Pair per-type room areas (gen mesh vs pre-built GT area lists, e.g. from polygons)."""
    gen_by = _mesh_areas_by_type_abs(gen_mesh)
    keys = set(gen_by) & set(gt_by)
    total = 0.0
    n = 0
    for k in keys:
        g = sorted([float(x) for x in gen_by.get(k, []) if math.isfinite(float(x))], reverse=True)
        t = sorted([float(x) for x in gt_by.get(k, []) if math.isfinite(float(x))], reverse=True)
        m = min(len(g), len(t))
        if m <= 0:
            continue
        for i in range(m):
            a_gen = float(g[i])
            a_gt = float(t[i])
            denom = abs(a_gt)
            if not math.isfinite(denom) or denom <= 0.0:
                return float("nan")
            rel = abs(a_gen - a_gt) / denom
            if rel > 10.0:
                return float("nan")
            total += float(rel)
            n += 1
    return float(total / max(1, n))


def _area_metric_gt(gen_mesh, gt_mesh) -> float:
    return _area_metric_gt_from_type_maps(gen_mesh, _mesh_areas_by_type_abs(gt_mesh))


def _area_ref_metric(
    gen_mesh,
    *,
    group: str,
    targets_by_group: dict[str, dict[str, float]],
    expected_counts: dict[str, int] | None,
    eps: float = 1e-6,
) -> float:
    grp = str(group).strip().lower()
    targets = targets_by_group.get(grp)
    if targets is None:
        raise ValueError(f"Unknown bedroom group '{group}'. Expected one of: {sorted(targets_by_group)}")
    gen_by = _mesh_areas_by_type_abs(gen_mesh)
    exp = expected_counts or {}
    total = 0.0
    n = 0
    for k, a_tgt0 in targets.items():
        a_tgt = float(a_tgt0)
        if not math.isfinite(a_tgt) or a_tgt < 0.0:
            continue
        denom = max(float(eps), abs(a_tgt))
        # If expected_counts is provided, treat it as an explicit set of room types
        # to evaluate. Missing keys (or non-positive counts) are skipped.
        if expected_counts is not None:
            want = int(exp.get(k, 0) or 0)
            if want < 1:
                continue
        else:
            want = int(exp.get(k, 1))
            if want < 1:
                want = 1
        areas = sorted([float(x) for x in gen_by.get(k, []) if math.isfinite(float(x))], reverse=True)
        for i in range(want):
            a_gen = float(areas[i]) if i < len(areas) else 0.0
            total += abs(a_gen - a_tgt) / denom
            n += 1
        if len(areas) > want:
            for a_gen in areas[want:]:
                total += abs(float(a_gen) - 0.0) / denom
                n += 1
    return float(total / max(1, n))


def _save_metric_progress_plot(metric_trace: list[float], best_trace: list[float], out_path: Path, *, title: str) -> None:
    if not metric_trace:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xs = np.arange(len(metric_trace), dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(xs, metric_trace, lw=1.4, label="objective")
    if best_trace and len(best_trace) == len(metric_trace):
        ax.plot(xs, best_trace, lw=1.8, label="best-so-far")
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("objective")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    return obj


def _render_hypergraph_frame(hg: dict, bounds: dict, *, RGL, Collections, title: str) -> np.ndarray:
    mesh = hypergraph_bounds_to_mesh(copy.deepcopy(hg), bounds, RGL=RGL, Collections=Collections)
    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(mesh.faceList.Count):
        face = mesh.faceList[i]
        room_type = getattr(face, "merge_id", "unknown")
        pts = face.getPoints()
        corners = [(pts[j].X, pts[j].Y) for j in range(pts.Count)]
        if len(corners) >= 3:
            x, y = Polygon(corners).exterior.xy
            ax.fill(x, y, color=PLOTTER.color_map.get(room_type, "#D3D3D3"), edgecolor="black", linewidth=0.5)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)
    return img


def _collect_nodes(node: dict, path=()):
    out = [{"path": path, "name": node.get("name", "?"), "mergeid": node.get("mergeid", "?"), "area": float(node.get("area", 0.0)), "angle": float(node.get("angle", 0.0))}]
    for i, child in enumerate(node.get("children", []) or []):
        out.extend(_collect_nodes(child, path + (i,)))
    return out


def _get_node_by_path(root: dict, path):
    n = root
    for i in path:
        n = n["children"][i]
    return n


def _as_pos_float(x, default: float) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if not math.isfinite(v):
        return float(default)
    return v


def _clip(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def annotate_area_splits(hg: dict, *, min_split: float = 0.1, max_split: float = 0.9, min_area: float = 1e-9) -> dict:
    def _rec(n: dict) -> None:
        children = n.get("children", []) or []
        if len(children) == 2 and not bool(n.get("final", False)):
            left_a = max(0.0, _as_pos_float(children[0].get("area", 0.0), 0.0))
            right_a = max(0.0, _as_pos_float(children[1].get("area", 0.0), 0.0))
            denom = left_a + right_a
            split = 0.5 if denom <= min_area else left_a / denom
            n["area_split"] = float(_clip(float(split), float(min_split), float(max_split)))
        for ch in children:
            _rec(ch)
    _rec(hg)
    return hg


def collect_area_splits(hg: dict) -> tuple[list[tuple[int, ...]], list[float]]:
    paths: list[tuple[int, ...]] = []
    splits: list[float] = []

    def _rec(n: dict, path: tuple[int, ...]) -> None:
        children = n.get("children", []) or []
        if len(children) == 2 and not bool(n.get("final", False)):
            paths.append(path)
            splits.append(float(_as_pos_float(n.get("area_split", 0.5), 0.5)))
            _rec(children[0], path + (0,))
            _rec(children[1], path + (1,))
        else:
            for i, ch in enumerate(children):
                _rec(ch, path + (i,))
    _rec(hg, ())
    return paths, splits


def apply_area_splits(hg: dict, splits: list[float] | np.ndarray, *, min_split: float = 0.1, max_split: float = 0.9, min_area: float = 1e-9) -> dict:
    splits_arr = np.asarray(splits, dtype=float).tolist()
    idx = 0

    def _rec(n: dict) -> None:
        nonlocal idx
        children = n.get("children", []) or []
        if len(children) != 2 or bool(n.get("final", False)):
            for ch in children:
                _rec(ch)
            return
        if idx >= len(splits_arr):
            raise ValueError(f"Not enough split values: expected > {idx}, got {len(splits_arr)}")
        parent_area = max(float(min_area), _as_pos_float(n.get("area", min_area), min_area))
        s = float(_clip(_as_pos_float(splits_arr[idx], 0.5), float(min_split), float(max_split)))
        n["area_split"] = s
        idx += 1
        left_area = max(float(min_area), parent_area * s)
        children[0]["area"] = float(left_area)
        children[1]["area"] = float(parent_area - left_area)
        _rec(children[0])
        _rec(children[1])
    _rec(hg)
    if idx != len(splits_arr):
        raise ValueError(f"Unused split values: used {idx}, provided {len(splits_arr)}")
    return hg


def print_area_split_list(hg_before: dict, hg_after: dict, *, min_split: float, max_split: float) -> None:
    paths, splits_before = collect_area_splits(hg_before)
    paths2, splits_after = collect_area_splits(hg_after)
    if paths != paths2:
        raise RuntimeError("area_split traversal mismatch between before/after graphs")
    if len(splits_before) != len(splits_after):
        raise RuntimeError("area_split length mismatch between before/after graphs")
    print("\narea_split parameters (preorder root->left->right):")
    print(f"  constraint: [{min_split}, {max_split}]")
    for i, (p, sb, sa) in enumerate(zip(paths, splits_before, splits_after)):
        n_before = _get_node_by_path(hg_before, p)
        print(f"  [{i:03d}] path={p}  {n_before.get('name', '?')} ({n_before.get('mergeid', '?')})  before={sb:.6f}  after={sa:.6f}")


def export_area_split_csv(hg_before: dict, hg_after: dict, out_csv: Path, *, min_split: float, max_split: float) -> None:
    paths, splits_before = collect_area_splits(hg_before)
    paths2, splits_after = collect_area_splits(hg_after)
    if paths != paths2:
        raise RuntimeError("area_split traversal mismatch between before/after graphs")
    if len(splits_before) != len(splits_after):
        raise RuntimeError("area_split length mismatch between before/after graphs")
    out_csv.parent.mkdir(exist_ok=True, parents=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["before", *[float(v) for v in splits_before]])
        w.writerow(["after", *[float(v) for v in splits_after]])


def validate_leaf_area_floor(hg: dict, *, min_leaf_frac: float = 0.03, min_area: float = 1e-9):
    root_area = float(hg.get("area", 0.0) or 0.0)
    root_area = max(float(min_area), root_area)
    floor_area = float(min_leaf_frac) * root_area
    leaves = [n for n in _collect_nodes(hg) if not _get_node_by_path(hg, n["path"]).get("children")]
    offenders = []
    min_frac = float("inf")
    for lf in leaves:
        a = float(lf.get("area", 0.0) or 0.0)
        frac = a / root_area
        if frac < min_frac:
            min_frac = frac
        if a < floor_area:
            offenders.append({**lf, "frac": frac})
    return (len(offenders) == 0), {
        "root_area": root_area,
        "floor_area": floor_area,
        "min_leaf_frac": float(min_leaf_frac),
        "min_observed_leaf_frac": float(min_frac if math.isfinite(min_frac) else 0.0),
        "n_leaves": len(leaves),
        "offenders": offenders,
    }


def _print_room_areas(hg: dict, label: str):
    print(f"\n{label} hypergraph node areas and angles:")
    nodes = _collect_nodes(hg)
    for n in nodes:
        mergeid = n.get("mergeid", "?")
        name = n.get("name", "?")
        area = n.get("area", 0.0)
        angle = n.get("angle", 0.0)
        angle_deg = angle * 180.0 / math.pi
        final = "FINAL" if n.get("final", False) else ""
        marker = "🚽" if "bath" in str(mergeid).lower() else ""
        print(f"  {marker} {name} ({mergeid}): area={area:.6f}, angle={angle_deg:.1f}° {final}")


# ---------------------------------------------------------------------------
# Documentation and record-keeping
# ---------------------------------------------------------------------------


def save_hypergraph_json(path: Path, hg: dict) -> None:
    """Save hypergraph to JSON file (JSON-serializable format)."""
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(hg), f, indent=2, sort_keys=True)


def save_metrics_report(artifacts_dir: Path, apt_id: str, metrics_original: dict, metrics_optimized: dict) -> None:
    """Write metrics_report.json with original vs optimized metrics."""
    try:
        Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(artifacts_dir) / "metrics_report.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "id": str(apt_id),
                    "original": dict(metrics_original),
                    "optimized": dict(metrics_optimized),
                },
                f,
                indent=2,
                sort_keys=True,
            )
    except OSError:
        pass


def save_accessgraph_penalty_report(artifacts_dir: Path, ag_report: dict) -> None:
    """Write accessgraph_penalty_report.json with eval/fail counts."""
    try:
        Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(artifacts_dir) / "accessgraph_penalty_report.json", "w", encoding="utf-8") as f:
            json.dump(ag_report, f, indent=2)
    except OSError:
        pass


def save_metric_progress_json(
    artifacts_dir: Path,
    apt_id: str,
    objective: str,
    metric_trace: list[float],
    best_trace: list[float],
    *,
    verbose: bool = False,
) -> None:
    """Write metric progress trace to JSON for later analysis."""
    out_json = Path(artifacts_dir) / f"{apt_id}_metric_progress.json"
    try:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "id": str(apt_id),
                    "objective": str(objective),
                    "metric_trace": [float(x) for x in metric_trace],
                    "best_trace": [float(x) for x in best_trace],
                },
                f,
                indent=2,
                sort_keys=True,
            )
        if verbose:
            print(f"Saved metric progress JSON to: {out_json}")
    except OSError as e:
        if verbose:
            print(f"⚠️  Could not save metric progress JSON to {out_json}: {type(e).__name__}: {e}")


def visualize_results(
    original_hg,
    optimized_hg,
    gt_hg,
    bounds,
    apt_id: str,
    *,
    metrics_original: dict[str, float],
    metrics_optimized: dict[str, float],
    output_dir: Path | None = None,
    script_dir: Path | None = None,
) -> None:
    """Render and save side-by-side visualization (GT, original, optimized) with metrics."""
    RGL, Collections = init_rgl()

    def _fmt_metrics_block(label: str, m: dict[str, float]) -> str:
        def f(k: str) -> str:
            v = float(m.get(k, float("nan")))
            return format(v, ".2g") if math.isfinite(v) else "nan"

        return (
            f"{label}\n"
            f"delta (GT, soft): {f('delta_metric')}\n"
            f"delta (ref, δ_sq): {f('delta_metric_square')}\n"
            f"epsilon (GT): {f('epsilon_metric')}\n"
            f"epsilon (ref): {f('epsilon_metric_ref')}"
        )

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 6.2))
    titles = [
        "Ground Truth",
        _fmt_metrics_block("Original", metrics_original),
        _fmt_metrics_block("Optimized", metrics_optimized),
    ]
    for ax, hg, title in zip(axes, [gt_hg, original_hg, optimized_hg], titles):
        try:
            mesh = hypergraph_bounds_to_mesh(copy.deepcopy(hg), bounds, RGL=RGL, Collections=Collections)
            for i in range(mesh.faceList.Count):
                face = mesh.faceList[i]
                room_type = str(getattr(face, "merge_id", "unknown"))
                pts = face.getPoints()
                corners = [(pts[j].X, pts[j].Y) for j in range(pts.Count)]
                if len(corners) >= 3:
                    x, y = Polygon(corners).exterior.xy
                    ax.fill(x, y, color=PLOTTER.color_map.get(room_type, "#D3D3D3"), edgecolor="black", linewidth=0.5)
        except Exception:
            corners = bounds.get("corners", [])
            poly = Polygon([(c["X"], c["Y"]) for c in corners]) if corners else Polygon()
            if not poly.is_empty:
                x, y = poly.exterior.xy
                ax.plot(x, y, "k-", lw=0.8)
        ax.set_title(f"{title}")
        ax.set_aspect("equal")
        ax.axis("off")

    base = script_dir or Path(__file__).parent
    output_dir = output_dir or (base / "output_parametric")
    output_dir.mkdir(exist_ok=True, parents=True)
    out_path = output_dir / f"optimized_{apt_id}.png"
    try:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    except OSError as e:
        print(f"⚠️  Could not save visualization to {out_path}: {type(e).__name__}: {e}")
    plt.close(fig)
    if out_path.exists():
        print(f"Saved visualization to: {out_path}")


def render_from_json(json_path: Path | None = None, output_path: Path | None = None, script_dir: Path | None = None) -> None:
    """Load hypergraph from parametric.json and render to parametric.png."""
    base = script_dir or Path(__file__).parent
    json_file = json_path or (base / "parametric.json")
    if not json_file.exists():
        raise FileNotFoundError(f"parametric.json not found: {json_file}")
    data = json.load(open(json_file))
    apt_id = data["apt_id"]
    bounds = data["bounds"]
    hg = data["hypergraph"]
    RGL, Collections = init_rgl()
    mesh = hypergraph_bounds_to_mesh(copy.deepcopy(hg), bounds, RGL=RGL, Collections=Collections)
    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(mesh.faceList.Count):
        face = mesh.faceList[i]
        room_type = str(getattr(face, "merge_id", "unknown"))
        pts = face.getPoints()
        corners = [(pts[j].X, pts[j].Y) for j in range(pts.Count)]
        if len(corners) >= 3:
            x, y = Polygon(corners).exterior.xy
            ax.fill(x, y, color=PLOTTER.color_map.get(room_type, "#D3D3D3"), edgecolor="black", linewidth=0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    out = output_path or (base / "parametric.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_single_apt_artifacts(
    output_dir: Path,
    original_hg: dict,
    optimized_hg: dict,
    gt_hg: dict,
    bounds: dict,
    apt_id: str,
    metrics_original: dict,
    metrics_optimized: dict,
    *,
    min_split: float = 0.25,
    max_split: float = 0.75,
) -> None:
    """Save before/after JSON, area_splits CSV, and visualization for a single apt."""
    output_dir.mkdir(exist_ok=True, parents=True)
    annotate_area_splits(original_hg, min_split=min_split, max_split=max_split)
    save_hypergraph_json(output_dir / "before.json", original_hg)
    save_hypergraph_json(output_dir / "after.json", optimized_hg)
    save_hypergraph_json(output_dir / "optimized.json", optimized_hg)
    export_area_split_csv(original_hg, optimized_hg, output_dir / "area_splits.csv", min_split=min_split, max_split=max_split)
    visualize_results(
        original_hg, optimized_hg, gt_hg, bounds, apt_id,
        metrics_original=metrics_original, metrics_optimized=metrics_optimized, output_dir=output_dir,
    )


def save_batch_results(
    batch_root: Path,
    batch_csv: Path,
    batch_ag_csv: Path,
    rows: list[dict],
    ag_rows: list[dict],
    objective: str,
    grpo_file_path: Path,
    *,
    fieldnames: list[str] | None = None,
) -> None:
    """Write batch CSV, accessgraph penalty CSV, and bulk_metrics_summary.json."""

    def _f(x):
        try:
            v = float(x)
            return v if math.isfinite(v) else None
        except Exception:
            return None

    def _as_int(x):
        try:
            if x is None:
                return None
            if isinstance(x, bool):
                return int(x)
            if isinstance(x, int):
                return int(x)
            s = str(x).strip()
            if s == "" or s.lower() == "none" or s.lower() == "null":
                return None
            return int(float(s))
        except Exception:
            return None

    def _mean(rows_in: list[dict], key: str) -> float | None:
        vs = [_f(r.get(key)) for r in rows_in]
        vs = [v for v in vs if v is not None]
        if not vs:
            return None
        return float(sum(vs) / len(vs))

    default_fieldnames = [
        "id", "objective", "status", "error",
        "original_objective", "optimized_objective",
        "delta_metric_orig", "delta_metric_opt",
        "delta_metric_square_orig", "delta_metric_square_opt",
        "area_metric_orig", "area_metric_opt",
        "area_metric_gt_orig", "area_metric_gt_opt",
        "epsilon_metric_orig", "epsilon_metric_opt",
        "ag_check_evals", "ag_check_fails", "ag_check_first_fail_eval",
        "ag_edges_total", "ag_edges_ok_orig", "ag_edges_ok_opt",
    ]
    names = fieldnames or default_fieldnames

    try:
        with open(batch_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=names)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nSaved batch CSV summary to: {batch_csv}")

        if ag_rows:
            with open(batch_ag_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["id", "ag_check_fails", "ag_check_first_fail_eval"])
                w.writeheader()
                for r in ag_rows:
                    w.writerow(r)
            print(f"Saved accessgraph penalty list to: {batch_ag_csv}")

        ok_rows = [r for r in rows if str(r.get("status", "")).strip().lower() == "ok"]
        all_rows = list(rows)
        ok_rows_no_first_iter_fail = [
            r
            for r in ok_rows
            if (
                int(_as_int(r.get("ag_check_evals")) or 0) > 0
                and (_as_int(r.get("ag_check_first_fail_eval")) is None or _as_int(r.get("ag_check_first_fail_eval")) != 1)
            )
        ]

        bulk = {
            "objective": str(objective),
            "grpo_file": str(grpo_file_path),
            "batch_root": str(batch_root),
            "n_total": int(len(rows)),
            "n_ok": int(len(ok_rows)),
            "n_error": int(sum(1 for r in rows if str(r.get("status", "")).strip().lower() != "ok")),
            "subsets": {
                "all_ok": {
                    "n": int(len(ok_rows)),
                    "means": {
                        "delta_metric": {"original": _mean(ok_rows, "delta_metric_orig"), "optimized": _mean(ok_rows, "delta_metric_opt")},
                        "delta_metric_square": {
                            "original": _mean(ok_rows, "delta_metric_square_orig"),
                            "optimized": _mean(ok_rows, "delta_metric_square_opt"),
                        },
                        "area_metric": {"original": _mean(ok_rows, "area_metric_orig"), "optimized": _mean(ok_rows, "area_metric_opt")},
                        "area_metric_gt": {"original": _mean(ok_rows, "area_metric_gt_orig"), "optimized": _mean(ok_rows, "area_metric_gt_opt")},
                        "objective_value": {"original": _mean(ok_rows, "original_objective"), "optimized": _mean(ok_rows, "optimized_objective")},
                    },
                },
                "original_metric_all": {
                    "n": int(len(all_rows)),
                    "definition": "all attempted ids (including status=error); means ignore missing/non-finite values",
                    "means": {
                        "delta_metric": {"original": _mean(all_rows, "delta_metric_orig"), "optimized": _mean(all_rows, "delta_metric_opt")},
                        "delta_metric_square": {
                            "original": _mean(all_rows, "delta_metric_square_orig"),
                            "optimized": _mean(all_rows, "delta_metric_square_opt"),
                        },
                    },
                },
                "ag_ok_at_first_eval": {
                    "n": int(len(ok_rows_no_first_iter_fail)),
                    "definition": "status==ok and ag_check_evals>0 and ag_check_first_fail_eval is None or != 1",
                    "means": {
                        "delta_metric": {
                            "original": _mean(ok_rows_no_first_iter_fail, "delta_metric_orig"),
                            "optimized": _mean(ok_rows_no_first_iter_fail, "delta_metric_opt"),
                        },
                        "delta_metric_square": {
                            "original": _mean(ok_rows_no_first_iter_fail, "delta_metric_square_orig"),
                            "optimized": _mean(ok_rows_no_first_iter_fail, "delta_metric_square_opt"),
                        },
                        "area_metric": {
                            "original": _mean(ok_rows_no_first_iter_fail, "area_metric_orig"),
                            "optimized": _mean(ok_rows_no_first_iter_fail, "area_metric_opt"),
                        },
                        "area_metric_gt": {
                            "original": _mean(ok_rows_no_first_iter_fail, "area_metric_gt_orig"),
                            "optimized": _mean(ok_rows_no_first_iter_fail, "area_metric_gt_opt"),
                        },
                        "objective_value": {
                            "original": _mean(ok_rows_no_first_iter_fail, "original_objective"),
                            "optimized": _mean(ok_rows_no_first_iter_fail, "optimized_objective"),
                        },
                    },
                },
            },
        }
        out_bulk = batch_root / "bulk_metrics_summary.json"
        with open(out_bulk, "w", encoding="utf-8") as f:
            json.dump(bulk, f, indent=2, sort_keys=True)
        print(f"Saved bulk metrics JSON summary to: {out_bulk}")
    except OSError as e:
        print(f"\nCould not write batch CSV: {type(e).__name__}: {e}")

