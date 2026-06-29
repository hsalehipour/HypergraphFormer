"""Helpers for parametric optimization of BSP-tree area splits.

Three objectives are supported (see :data:`OBJECTIVES`):

* ``delta``         — minimise δ_sq (perimeter–square ratio per leaf).
* ``epsilon``       — minimise ε vs the shared per-room reference fractions.
* ``delta_epsilon`` — equally weighted blend of the two (with auto-rescale).

Splits are clipped to ``[MIN_SPLIT, MAX_SPLIT] = [0.15, 0.85]``; this is fixed
on purpose (no flag).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.hypergraph import Hypergraph  # noqa: E402
from utils.metrics import (  # noqa: E402
    compute_room_area_proportion_error,
    produce_room_area_list,
    produce_room_list,
)

# ---------------------------------------------------------------------------
# Hard-coded knobs (intentionally not configurable via CLI).
# ---------------------------------------------------------------------------
MIN_SPLIT: float = 0.15
MAX_SPLIT: float = 0.85
MIN_LEAF_FRAC: float = 0.03
MIN_AREA: float = 1e-9
# Per-failed-edge weight added to the GD objective for accessgraph violations.
ACCESSGRAPH_VIOLATION_WEIGHT: float = 0.05

OBJECTIVES: tuple[str, ...] = ("delta", "delta_epsilon", "epsilon")
DEFAULT_OBJECTIVE: str = "delta"

DEFAULT_WMR24_RESULTS = (
    REPO_ROOT / "inference_8700_on_wmr24_inference_vllm_64" / "results_hypergraphs.json"
)
# Per-room ε reference fractions (shared between WMR24 and RPLAN).
AREA_REF_FRACTIONS: dict[str, dict[str, float]] = {
    "studio": {
        "balcony": 0.2001, "bath": 0.0578, "kitchen": 0.079, "living": 0.5579,
    },
    "one_bed": {
        "balcony": 0.0838, "bath": 0.0762, "bed": 0.2477, "extra": 0.0429,
        "foyer": 0.0784, "kitchen": 0.1045, "living": 0.4208, "storage": 0.0497,
    },
    "two_bed": {
        "balcony": 0.063, "bath": 0.0571, "bed": 0.1746, "extra": 0.0328,
        "foyer": 0.0491, "kitchen": 0.0804, "living": 0.422, "storage": 0.0377,
    },
    "three_plus": {
        "balcony": 0.0558, "bath": 0.0464, "bed": 0.135, "extra": 0.0261,
        "foyer": 0.0402, "kitchen": 0.0665, "living": 0.4006, "storage": 0.0332,
    },
}


# ---------------------------------------------------------------------------
# Config + CLI
# ---------------------------------------------------------------------------


@dataclass
class OptimConfig:
    """Optimization parameters parsed from the CLI."""

    seed: int | None = None
    maxiter: int = 50
    objective: str = DEFAULT_OBJECTIVE
    grpo_file: str | None = None
    test_dir: str = "dataset/wmr24/test"
    dataset: str = "wmr24"
    limit: int | None = None
    skip_ids_file: str | None = None
    gd_lr: float | None = None  # default: 0.1 * (MAX_SPLIT - MIN_SPLIT)
    patience: int | None = None
    ftol: float = 1e-5
    with_accessgraph_penalty: bool = True
    batch_workers: int = 1
    batch_output_dir: str | None = None
    merged_hypergraphs_json: str | None = None
    quiet: bool = False

    @classmethod
    def from_args(cls, argv: list[str] | None = None) -> tuple["OptimConfig", str | None, bool]:
        """Parse CLI. Returns (config, apt_id, run_all). apt_id is None when run_all."""
        p = argparse.ArgumentParser(description="Hypergraph parametric optimization (GD)")
        p.add_argument("apt_id", nargs="?", help="Apartment id (or use --all for batch)")
        p.add_argument("--all", action="store_true", help="Batch mode over all ids in --results-hypergraphs")
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--maxiter", type=int, default=50)
        p.add_argument(
            "--objective",
            choices=OBJECTIVES,
            default=DEFAULT_OBJECTIVE,
            help=f"One of {OBJECTIVES}",
        )
        p.add_argument(
            "--grpo-file",
            "--results-hypergraphs",
            dest="grpo_file",
            default=None,
            help=f"sample_id→hypergraph JSON (default: {DEFAULT_WMR24_RESULTS.name}).",
        )
        p.add_argument(
            "--test-dir",
            default="dataset/wmr24/test",
            help="Ground-truth folder with <id>.txt (default: WMR24 test split).",
        )
        p.add_argument(
            "--dataset",
            choices=("auto", "wmr24", "rplan"),
            default="auto",
            help="Dataset label (only affects logging). 'auto' infers from --test-dir.",
        )
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--skip-ids-file", default=None)
        p.add_argument(
            "--gd-lr", "--gd_lr", dest="gd_lr", type=float, default=None,
            help=f"GD step size (default: 0.1*(MAX_SPLIT-MIN_SPLIT)={0.1*(MAX_SPLIT-MIN_SPLIT):.3f}).",
        )
        p.add_argument(
            "--patience", type=int, default=None,
            help="Stop GD when best_seen hasn't improved by >ftol for N consecutive steps "
            "(default: max(10, maxiter//5)). Use 0 to disable.",
        )
        p.add_argument("--ftol", type=float, default=1e-5,
                       help="Improvement threshold for early-stop (default: 1e-5).")
        p.add_argument(
            "--no-accessgraph", dest="no_accessgraph", action="store_true",
            help="Disable the binary +1-per-failed-edge accessgraph penalty in the GD objective.",
        )
        p.add_argument("--batch-workers", type=int, default=1,
                       help="Parallel processes for --all (1 = sequential).")
        p.add_argument("--batch-output-dir", default=None)
        p.add_argument(
            "--merged-hypergraphs-json", dest="merged_hypergraphs_json", default=None,
            help="--all only: path for merged optimized hypergraphs.",
        )
        p.add_argument("--quiet", action="store_true",
                       help="Suppress per-sample verbose output; only show batch progress.")
        ns = p.parse_args(argv)

        if int(ns.batch_workers) < 1:
            p.error("--batch-workers must be >= 1")

        ds = str(ns.dataset).strip().lower()
        if ds == "auto":
            ds = _infer_dataset(ns.test_dir)

        cfg = cls(
            seed=ns.seed,
            maxiter=int(ns.maxiter),
            objective=ns.objective,
            grpo_file=ns.grpo_file,
            test_dir=ns.test_dir,
            dataset=ds,
            limit=ns.limit,
            skip_ids_file=ns.skip_ids_file,
            gd_lr=ns.gd_lr,
            patience=ns.patience,
            ftol=float(ns.ftol),
            with_accessgraph_penalty=not bool(ns.no_accessgraph),
            batch_workers=int(ns.batch_workers),
            batch_output_dir=ns.batch_output_dir,
            merged_hypergraphs_json=ns.merged_hypergraphs_json,
            quiet=bool(ns.quiet),
        )

        apt_id = _validate_apt_id(str(ns.apt_id)) if ns.apt_id and not ns.all else None
        return cfg, apt_id, bool(ns.all)


def _infer_dataset(test_dir: str) -> str:
    s = str(test_dir).replace("\\", "/").lower()
    if "wmr24" in s:
        return "wmr24"
    if "rplan" in s or "eval_test" in s:
        return "rplan"
    return "wmr24"


# ---------------------------------------------------------------------------
# Path / id validation
# ---------------------------------------------------------------------------


def _validate_apt_id(apt_id: str) -> str:
    # Strict whitelist guards against directory-traversal / shell metachars
    # because apt_id is interpolated into output paths and CSV keys.
    if not isinstance(apt_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", apt_id):
        raise ValueError(f"Invalid apt_id {apt_id!r}. Allowed: letters, digits, '_' and '-'.")
    return apt_id


def _resolve_under_repo(repo_root: Path, p: str) -> Path:
    rp = Path(p).expanduser().resolve()
    try:
        rp.relative_to(repo_root)
    except ValueError as e:
        raise ValueError(f"Refusing to use path outside repo root: {rp} (root={repo_root})") from e
    return rp


# ---------------------------------------------------------------------------
# Hypergraph + GT loading
# ---------------------------------------------------------------------------


def init_rgl():
    """Pre-initialise RGL (idempotent)."""
    return Hypergraph.init_rgl()


def hypergraph_bounds_to_mesh(hg: dict, bounds: dict, *, RGL=None, Collections=None):
    """Build an RGL room mesh from a hypergraph and dataset ``bounds``."""
    return Hypergraph(copy.deepcopy(hg)).to_mesh(bounds=bounds, RGL=RGL, Collections=Collections)


def load_ground_truth_txt(test_dir: Path, sample_id: str) -> tuple[dict, dict, dict, list | None]:
    """Load ``<test_dir>/<id>.txt`` → ``(bounds, hypergraph, accessgraph, room_polygons|None)``."""
    def _dict(val) -> dict:
        """Parse a JSON-or-dict field, returning {} for anything non-dict."""
        if isinstance(val, str):
            s = val.strip()
            val = json.loads(s) if s else {}
        return val if isinstance(val, dict) else {}

    with open(test_dir / f"{sample_id}.txt", encoding="utf-8") as f:
        raw = json.load(f)

    bounds = _dict(raw.get("bounds", "{}"))
    hg = _dict(raw.get("hypergraph"))
    ag = _dict(raw.get("accessgraph") or {})
    rp = raw.get("room_polygons")
    return bounds, hg, ag, (rp if isinstance(rp, list) else None)


def access_graph_from_prediction(
    hg: dict,
    *,
    bounds: dict | None = None,
    file_fallback: dict | None = None,
) -> dict:
    """Pick best access graph for compliance checks: corrected → raw → file fallback."""
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


# ---------------------------------------------------------------------------
# Type bucketing + bedroom group
# ---------------------------------------------------------------------------


def _canon_room_type(s: str) -> str:
    t = str(s).strip().lower()
    if "bath" in t:
        return "bath"
    if t in ("bedroom", "bed"):
        return "bed"
    return t


def _bedroom_group_from_accessgraph(accessgraph: dict | None) -> str:
    n_bed = sum(
        1 for v in (accessgraph or {}).get("nodes", {}).values()
        if str(v).strip().lower() in ("bed", "bedroom")
    )
    if n_bed <= 0:
        return "studio"
    if n_bed == 1:
        return "one_bed"
    if n_bed == 2:
        return "two_bed"
    return "three_plus"


def _expected_counts_from_accessgraph(accessgraph: dict | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in (accessgraph or {}).get("nodes", {}).values():
        ck = _canon_room_type(str(v))
        counts[ck] = counts.get(ck, 0) + 1
    return counts


def _leaf_type_counts_from_hypergraph(hg: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    stack = [hg]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        children = node.get("children") or []
        if children:
            stack.extend(children)
            continue
        t = _canon_room_type(str(node.get("mergeid", node.get("name", "unknown"))))
        counts[t] = counts.get(t, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _delta_metric_square_from_room_list(room_list) -> float:
    """Mean ``|1 - L_sq/L|`` across leaves (perimeter–square ratio)."""
    vals: list[float] = []
    for _name, L, L_sq in room_list:
        L = float(L); L_sq = float(L_sq)
        if math.isfinite(L) and L > 0 and math.isfinite(L_sq) and L_sq > 0:
            vals.append(abs(1.0 - L_sq / L))
    return float(sum(vals) / len(vals)) if vals else 0.0


def compute_epsilon_metric_vs_ref(
    pred_mesh,
    *,
    group: str,
    expected_counts: dict[str, int] | None,
) -> float:
    """Mean ``|prop_pred - prop_ref|`` per room vs :data:`AREA_REF_FRACTIONS`.

    Returns 1.0 when no usable reference / prediction is available.
    """
    targets = AREA_REF_FRACTIONS.get(str(group).strip().lower())
    if not targets:
        return 1.0
    exp = expected_counts or {}

    pred_area = produce_room_area_list(pred_mesh)
    pred_total = sum(float(a) for _, a in pred_area)
    if not (math.isfinite(pred_total) and pred_total > 0):
        return 1.0

    pred_by_type: dict[str, list[float]] = {}
    for mid, a in pred_area:
        pred_by_type.setdefault(_canon_room_type(str(mid)), []).append(float(a) / pred_total)

    ref_by_type: dict[str, list[float]] = {}
    for k, frac in targets.items():
        n = int(exp.get(k, 0) or 0)
        f = float(frac)
        if n < 1 or not math.isfinite(f) or f < 0:
            continue
        ref_by_type.setdefault(_canon_room_type(str(k)), []).extend([f] * n)

    errors: list[float] = []
    for room_type in sorted(set(pred_by_type) | set(ref_by_type)):
        pv = sorted(pred_by_type.get(room_type, []), reverse=True)
        rv = sorted(ref_by_type.get(room_type, []), reverse=True)
        for i in range(max(len(pv), len(rv))):
            errors.append(abs(
                (pv[i] if i < len(pv) else 0.0) - (rv[i] if i < len(rv) else 0.0)
            ))

    if not errors:
        return float("nan")
    v = float(sum(errors) / len(errors))
    return v if math.isfinite(v) else 1.0


def check_accessgraph_edges_against_mesh(
    mesh,
    accessgraph: dict,
    *,
    scale_factor: float = 1.05,
    min_overlap_area: float = 1e-12,
) -> list[dict]:
    """Per-edge accessgraph compliance: do the room polygons (scaled) actually touch?"""
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
            polys_scaled.append(_scale(p, xfact=scale_factor, yfact=scale_factor, origin="centroid"))
        except Exception:
            polys_scaled.append(p)

    adj_pairs: set[tuple[str, str]] = set()
    for i in range(n):
        if polys_scaled[i].is_empty:
            continue
        for j in range(i + 1, n):
            if polys_scaled[j].is_empty:
                continue
            try:
                area = float(polys_scaled[i].intersection(polys_scaled[j]).area or 0.0)
            except Exception:
                area = 0.0
            if area > min_overlap_area:
                a, b = sorted([face_types[i], face_types[j]])
                adj_pairs.add((a, b))

    nodes = (accessgraph or {}).get("nodes", {}) or {}
    out: list[dict] = []
    for e in (accessgraph or {}).get("edges") or []:
        if not isinstance(e, (list, tuple)) or len(e) != 2:
            continue
        u, v = str(e[0]), str(e[1])
        tu = str(nodes.get(u, "unknown")).strip().lower()
        tv = str(nodes.get(v, "unknown")).strip().lower()
        a, b = sorted([tu, tv])
        out.append({"u": u, "v": v, "u_type": tu, "v_type": tv, "ok": (a, b) in adj_pairs})
    return out


# ---------------------------------------------------------------------------
# Splits: annotate / collect / apply
# ---------------------------------------------------------------------------


def _as_pos_float(x, default: float) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    return v if math.isfinite(v) else float(default)


def _clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def annotate_area_splits(hg: dict) -> dict:
    """Add ``area_split`` to every binary, non-final node based on child areas."""

    def _rec(n: dict) -> None:
        children = n.get("children", []) or []
        if len(children) == 2 and not bool(n.get("final", False)):
            la = max(0.0, _as_pos_float(children[0].get("area", 0.0), 0.0))
            ra = max(0.0, _as_pos_float(children[1].get("area", 0.0), 0.0))
            denom = la + ra
            split = 0.5 if denom <= MIN_AREA else la / denom
            n["area_split"] = float(_clip(split, MIN_SPLIT, MAX_SPLIT))
        for ch in children:
            _rec(ch)

    _rec(hg)
    return hg


def collect_area_splits(hg: dict) -> tuple[list[tuple[int, ...]], list[float]]:
    """Preorder list of (path, split_value) for every binary, non-final node."""
    paths: list[tuple[int, ...]] = []
    splits: list[float] = []

    def _rec(n: dict, path: tuple[int, ...]) -> None:
        children = n.get("children", []) or []
        if len(children) == 2 and not bool(n.get("final", False)):
            paths.append(path)
            splits.append(_as_pos_float(n.get("area_split", 0.5), 0.5))
            _rec(children[0], path + (0,))
            _rec(children[1], path + (1,))
        else:
            for i, ch in enumerate(children):
                _rec(ch, path + (i,))

    _rec(hg, ())
    return paths, splits


def apply_area_splits(hg: dict, splits) -> dict:
    """Overwrite ``area_split`` and propagate areas from a flat split vector."""
    arr = np.asarray(splits, dtype=float).tolist()
    idx = 0

    def _rec(n: dict) -> None:
        nonlocal idx
        children = n.get("children", []) or []
        if len(children) != 2 or bool(n.get("final", False)):
            for ch in children:
                _rec(ch)
            return
        if idx >= len(arr):
            raise ValueError(f"Not enough split values: idx={idx} arr_len={len(arr)}")
        parent_area = max(MIN_AREA, _as_pos_float(n.get("area", MIN_AREA), MIN_AREA))
        s = float(_clip(_as_pos_float(arr[idx], 0.5), MIN_SPLIT, MAX_SPLIT))
        n["area_split"] = s
        idx += 1
        left_area = max(MIN_AREA, parent_area * s)
        children[0]["area"] = float(left_area)
        children[1]["area"] = float(parent_area - left_area)
        _rec(children[0])
        _rec(children[1])

    _rec(hg)
    if idx != len(arr):
        raise ValueError(f"Unused split values: used {idx}, provided {len(arr)}")
    return hg


def validate_leaf_area_floor(hg: dict) -> tuple[bool, dict]:
    """All leaves ≥ ``MIN_LEAF_FRAC`` of root area? Returns (ok, stats)."""
    root_area = max(MIN_AREA, float(hg.get("area", 0.0) or 0.0))
    floor_area = MIN_LEAF_FRAC * root_area
    stack, leaves = [hg], []
    while stack:
        n = stack.pop()
        if not isinstance(n, dict):
            continue
        ch = n.get("children") or []
        if ch:
            stack.extend(ch)
        else:
            leaves.append(n)
    min_frac = float("inf")
    n_offenders = 0
    for lf in leaves:
        a = float(lf.get("area", 0.0) or 0.0)
        frac = a / root_area
        if frac < min_frac:
            min_frac = frac
        if a < floor_area:
            n_offenders += 1
    return n_offenders == 0, {
        "min_observed_leaf_frac": float(min_frac if math.isfinite(min_frac) else 0.0),
        "n_leaves": len(leaves),
        "n_offenders": int(n_offenders),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    return obj


def save_metric_progress_csv(
    out_csv: Path,
    traces: list[tuple[str, list[float], list[float]]],
) -> None:
    """Append per-step metric traces to a long-format CSV.

    Each entry in ``traces`` is ``(apt_id, objective_trace, best_so_far_trace)``.
    Columns: id, step, objective, best_so_far.
    File is created if it does not exist; existing rows are preserved.
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists()
    try:
        with open(out_csv, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["id", "step", "objective", "best_so_far"])
            for apt_id, obj_trace, best_trace in traces:
                for step, (obj, best) in enumerate(zip(obj_trace, best_trace)):
                    w.writerow([apt_id, step, obj, best])
    except OSError as e:
        print(f"⚠️  Could not write metric progress CSV {out_csv}: {type(e).__name__}: {e}")


# Columns produced by the per-apt batch row builder in run_parametric.py.
BATCH_CSV_FIELDS: list[str] = [
    "id", "objective", "status", "error",
    "original_objective", "optimized_objective",
    "delta_metric_orig", "delta_metric_opt",
    "delta_metric_square_orig", "delta_metric_square_opt",
    "epsilon_metric_orig", "epsilon_metric_opt",
    "epsilon_metric_ref_orig", "epsilon_metric_ref_opt",
    "ag_check_evals", "ag_check_fails", "ag_check_first_fail_eval",
    "ag_edges_total", "ag_edges_ok_orig", "ag_edges_ok_opt",
]


def _safe_float(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _mean(rows: list[dict], key: str):
    vs = [_safe_float(r.get(key)) for r in rows]
    vs = [v for v in vs if v is not None]
    return float(sum(vs) / len(vs)) if vs else None


def save_batch_results(
    batch_root: Path,
    batch_csv: Path,
    rows: list[dict],
    objective: str,
    grpo_file_path: Path,
) -> None:
    """Write per-id batch CSV and a small bulk_metrics_summary.json."""
    try:
        with open(batch_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=BATCH_CSV_FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in BATCH_CSV_FIELDS})
        print(f"\nSaved batch CSV to: {batch_csv}")
    except OSError as e:
        print(f"\n⚠️  Could not write batch CSV: {type(e).__name__}: {e}")
        return

    ok_rows = [r for r in rows if str(r.get("status", "")).strip().lower() == "ok"]
    metric_keys = (
        "delta_metric", "delta_metric_square",
        "epsilon_metric", "epsilon_metric_ref",
    )
    summary = {
        "objective": str(objective),
        "grpo_file": str(grpo_file_path),
        "batch_root": str(batch_root),
        "n_total": len(rows),
        "n_ok": len(ok_rows),
        "n_error": len(rows) - len(ok_rows),
        "means_ok": {
            k: {
                "original": _mean(ok_rows, f"{k}_orig"),
                "optimized": _mean(ok_rows, f"{k}_opt"),
            }
            for k in metric_keys
        }
        | {
            "objective_value": {
                "original": _mean(ok_rows, "original_objective"),
                "optimized": _mean(ok_rows, "optimized_objective"),
            }
        },
    }
    out = batch_root / "bulk_metrics_summary.json"
    try:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"Saved bulk metrics JSON summary to: {out}")
    except OSError as e:
        print(f"⚠️  Could not write {out}: {type(e).__name__}: {e}")
