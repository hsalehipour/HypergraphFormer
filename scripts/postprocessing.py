#!/usr/bin/env python3
"""
Post-processing for predicted hypergraphs: three modes via ``--pp-mode``.

* ``parametric`` (default) — greedy add/remove multiset planning + optional procedural edits
  (``--apply-procedural-step1``). Default ``--procedural-steps 5``.
* ``orientation`` — BSP reorientation search (3 mirrors × 4 rotations), picks min δ_sq.
* ``optimization`` — gradient-descent split optimization via ``run_parametric.optimize_hypergraph``.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import multiprocessing
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.hypergraph import Hypergraph, repair_bsp_tree_names  # noqa: E402

logger = logging.getLogger(__name__)

# Greedy multiset planning/apply: max iterations to avoid infinite loops if logic regresses.
SAFETY_MAX_MULTISET_STEPS = 512

_SKIP_NODE_TYPES = frozenset({"outside"})


def _parse_jsonish(val: Any) -> Any:
    if isinstance(val, str):
        return json.loads(val)
    return val


def load_test_record(test_dir: Path, sample_id: str) -> dict[str, Any]:
    path = test_dir / f"{sample_id}.txt"
    with open(path) as f:
        raw = json.load(f)
    return {
        "accessgraph": raw.get("accessgraph") or {},
        "bounds": _parse_jsonish(raw.get("bounds", "{}")),
    }


def room_type_multiset(accessgraph: dict | None) -> Counter[str]:
    """Multiset of room types from ``nodes`` values (lowercased), excluding *outside*."""
    nodes = (accessgraph or {}).get("nodes") or {}
    c: Counter[str] = Counter()
    for v in nodes.values():
        t = str(v).strip().lower()
        if not t or t in _SKIP_NODE_TYPES:
            continue
        c[t] += 1
    return c


def _gt_preferred_mergeid_for_room_hint(
    gt_accessgraph: dict | None,
    room_hint: str,
) -> str | None:
    """
    Pick the most common raw ``nodes`` label in GT whose canonical room type matches
    *room_hint* (same normalization as procedural_edits). Used so procedural ``add`` writes
    leaf ``mergeid`` spellings that match eval GT (e.g. ``bed`` vs ``bedroom``).
    """
    from scripts.procedural_edits import canonical_room_name

    hint = str(room_hint).strip()
    if not hint:
        return None
    hint_canon = canonical_room_name(hint)
    nodes = (gt_accessgraph or {}).get("nodes") or {}
    raw_counts: Counter[str] = Counter()
    for v in nodes.values():
        s = str(v).strip()
        if not s or str(s).strip().lower() in _SKIP_NODE_TYPES:
            continue
        if canonical_room_name(s) == hint_canon:
            raw_counts[s] += 1
    if not raw_counts:
        return None
    return sorted(raw_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _multiset_diff_detail(cnt_pred: Counter[str], cnt_gt: Counter[str]) -> dict[str, Any]:
    """Shared ``missing`` / ``extra`` table for *cnt_pred* vs *cnt_gt* (multiset diffs)."""
    missing = cnt_gt - cnt_pred
    extra = cnt_pred - cnt_gt
    return {
        "missing": dict(sorted(missing.items())),
        "extra": dict(sorted(extra.items())),
    }


def multiset_command(cnt_pred: Counter[str], cnt_gt: Counter[str]) -> tuple[str, dict[str, Any]]:
    """
    One greedy step: match *cnt_pred* to *cnt_gt* using the same rule as before.

    Returns:
        ``(command_str, step_detail)`` with keys ``missing`` / ``extra`` (multiset diffs).
    """
    step_detail = _multiset_diff_detail(cnt_pred, cnt_gt)
    missing = cnt_gt - cnt_pred
    extra = cnt_pred - cnt_gt
    if not missing and not extra:
        return "NOTHING", step_detail
    if missing:
        room = max(missing.keys(), key=lambda k: (missing[k], k))
        return f"add {room}", step_detail
    room = max(extra.keys(), key=lambda k: (extra[k], k))
    return f"remove {room}", step_detail



def simulate_multiset_command(cnt: Counter[str], command: str) -> Counter[str]:
    """
    Apply one *add* / *remove* / ``NOTHING`` to a room-type multiset (copy).

    This approximates one procedural edit at the **room-count** level only.
    """
    out = copy.copy(cnt)
    cmd = (command or "").strip()
    if cmd == "NOTHING" or not cmd:
        return out
    if cmd.startswith("add "):
        room = cmd[4:].strip().lower()
        if room:
            out[room] += 1
        return out
    if cmd.startswith("remove "):
        room = cmd[7:].strip().lower()
        if room and out[room] > 0:
            out[room] -= 1
            if out[room] <= 0:
                del out[room]
        return out
    return out


def plan_multiset_command_chain(
    cnt_corr: Counter[str],
    cnt_gt: Counter[str],
    max_steps: int,
) -> list[str]:
    """Greedy multiset plan: repeat ``multiset_command`` + simulate until ``NOTHING`` or *max_steps*."""
    ms = int(max_steps)
    max_iter = SAFETY_MAX_MULTISET_STEPS if ms <= 0 else min(ms, SAFETY_MAX_MULTISET_STEPS)
    chain: list[str] = []
    cnt = copy.copy(cnt_corr)
    for _ in range(max_iter):
        cmd, _ = multiset_command(cnt, cnt_gt)
        if cmd == "NOTHING":
            break
        chain.append(cmd)
        cnt = simulate_multiset_command(cnt, cmd)
    return chain


def compare_accessgraphs(
    input_accessgraph: dict,
    corrected_accessgraph: dict,
    gt_accessgraph: dict,
    *,
    max_plan_steps: int = 0,
) -> tuple[str, dict[str, Any]]:
    """
    Compare room-type multisets and plan greedy add/remove commands.

    Returns:
        ``(command_step1, detail)`` — *detail* includes ``multiset_command_chain``.
    """
    cnt_in = room_type_multiset(input_accessgraph)
    cnt_corr = room_type_multiset(corrected_accessgraph)
    cnt_gt = room_type_multiset(gt_accessgraph)

    chain = plan_multiset_command_chain(cnt_corr, cnt_gt, int(max_plan_steps))
    diff = _multiset_diff_detail(cnt_corr, cnt_gt)
    detail: dict[str, Any] = {
        "input_room_counts": dict(sorted(cnt_in.items())),
        "corrected_room_counts": dict(sorted(cnt_corr.items())),
        "gt_room_counts": dict(sorted(cnt_gt.items())),
        "multiset_diff": diff,
        "multiset_command_chain": chain,
    }
    return (chain[0] if chain else "NOTHING"), detail


def _fallback_anchor_largest_non_target(hypergraph: dict, room_to_add: str) -> str:
    """Largest-area pred leaf whose ``mergeid`` is not *room_to_add* (lowercased match)."""
    hg = Hypergraph(hypergraph)
    t_new = str(room_to_add).strip().lower()
    best_area = -1.0
    anchor_mergeid: str | None = None
    for leaf in hg.leaves:
        mid = str(leaf.get("mergeid", "")).strip().lower()
        if mid == t_new:
            continue
        area = float(leaf.get("area", 0.0))
        if area > best_area:
            best_area = area
            anchor_mergeid = str(leaf.get("mergeid", "")).strip()

    if anchor_mergeid is None:
        raise ValueError(
            f"No anchor room to split when adding {room_to_add!r} "
            "(all leaves appear to be that type)."
        )
    return anchor_mergeid


def _representative_mergeid_for_type(hg: Hypergraph, room_type_lower: str) -> str | None:
    """A leaf ``mergeid`` string in *hg* matching *room_type_lower*, or ``None``."""
    rtl = room_type_lower.strip().lower()
    for leaf in hg.leaves:
        mid = str(leaf.get("mergeid", "")).strip()
        if mid.lower() == rtl:
            return mid
    return None


def complete_add_command_for_procedural(
    hypergraph: dict,
    room_to_add: str,
    gt_accessgraph: dict,
) -> tuple[str, str]:
    """
    Build ``add <new> by <anchor>`` for :func:`scripts.procedural_edits.apply_command`.

    Picks the **lexicographically first** anchor type that is a **neighbor in the GT access
    graph** of some node typed *room_to_add*, and that exists in the predicted hypergraph.
    Otherwise falls back to :func:`_fallback_anchor_largest_non_target`.

    Returns:
        ``(command_string, anchor_source)`` where *anchor_source* is ``gt_neighbor`` or
        ``fallback_largest_leaf``.
    """
    hg = Hypergraph(hypergraph)
    t_new = str(room_to_add).strip().lower()
    new_room = str(room_to_add).strip()

    nodes = (gt_accessgraph or {}).get("nodes") or {}
    edges = (gt_accessgraph or {}).get("edges") or []

    adj: dict[str, set[str]] = {}
    for e in edges:
        if not isinstance(e, (list, tuple)) or len(e) != 2:
            continue
        a, b = str(e[0]), str(e[1])
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    gt_nodes_new = [
        str(nid)
        for nid, typ in nodes.items()
        if str(typ).strip().lower() == t_new
    ]

    neighbor_types: set[str] = set()
    for nid in gt_nodes_new:
        for nb in adj.get(nid, ()):
            typ = nodes.get(nb)
            if typ is None:
                continue
            tnb = str(typ).strip().lower()
            if tnb and tnb not in _SKIP_NODE_TYPES and tnb != t_new:
                neighbor_types.add(tnb)

    pred_types = {str(leaf.get("mergeid", "")).strip().lower() for leaf in hg.leaves}
    candidates = sorted(neighbor_types & pred_types)

    if candidates:
        anchor_type = candidates[0]
        anchor_mid = _representative_mergeid_for_type(hg, anchor_type)
        if anchor_mid is not None:
            return f"add {new_room} by {anchor_mid}", "gt_neighbor"

    anchor_mergeid = _fallback_anchor_largest_non_target(hypergraph, room_to_add)
    return f"add {new_room} by {anchor_mergeid}", "fallback_largest_leaf"


def multiset_command_to_procedural(
    command: str,
    hypergraph: dict,
    gt_accessgraph: dict,
) -> tuple[str, dict[str, Any]]:
    """
    Map multiset hint (``add`` / ``remove``) to an ``apply_command`` string.

    Returns:
        ``(procedural_command, extra)`` — *extra* may include ``add_anchor_source``.
    """
    c = (command or "").strip()
    extra: dict[str, Any] = {}
    if c == "NOTHING" or not c:
        raise ValueError("no procedural command")
    if c.startswith("remove "):
        return f"delete {c[7:].strip()}", extra
    if c.startswith("add "):
        pcmd, src = complete_add_command_for_procedural(
            hypergraph, c[4:].strip(), gt_accessgraph
        )
        extra["add_anchor_source"] = src
        return pcmd, extra
    raise ValueError(
        f"Command {command!r} is not add/remove; cannot map to procedural_edits."
    )


def _multiset_commands_from_row(row: dict[str, Any]) -> list[str]:
    """Ordered add/remove strings for procedural apply (skip ``NOTHING``)."""
    mc = row.get("multiset_commands")
    if isinstance(mc, list) and mc:
        return [
            str(c).strip()
            for c in mc
            if c is not None
            and str(c).strip()
            and str(c).strip().upper() != "NOTHING"
        ]
    out: list[str] = []
    c1 = row.get("command")
    if c1 not in (None, "NOTHING") and str(c1).strip():
        out.append(str(c1).strip())
    c2 = row.get("command_step2")
    if c2 not in (None, "NOTHING") and str(c2).strip():
        out.append(str(c2).strip())
    return out



def _apply_procedural_chain_for_sample(
    sid: str,
    multiset_commands: list[str],
    row_error: Any,
    base_hg: dict,
    gt_ag: dict,
    bounds_b: dict,
    increase_factor: float,
    max_procedural_steps: int,
) -> tuple[dict, dict[str, Any]]:
    """Apply procedural add/delete chain for one sample. Returns ``(hypergraph, meta)``."""
    from scripts.procedural_edits import apply_command

    base = copy.deepcopy(base_hg)
    if row_error:
        return base, {"procedural_steps_applied": 0}

    raw = [c for c in multiset_commands if c]
    mps = int(max_procedural_steps)
    cmds = raw if mps <= 0 else raw[:min(len(raw), mps)]
    if not cmds:
        return base, {"procedural_steps_applied": 0}

    bounds_kw = bounds_b if isinstance(bounds_b, dict) and bounds_b.get("corners") else None
    current = base
    for step_i, cmd in enumerate(cmds, start=1):
        try:
            new_room_mergeid = (
                _gt_preferred_mergeid_for_room_hint(gt_ag, cmd[4:].strip())
                if cmd.lower().startswith("add ")
                else None
            )
            pcmd, _ = multiset_command_to_procedural(cmd, current, gt_ag)
            edited, _ = apply_command(current, pcmd, increase_factor, new_room_mergeid=new_room_mergeid, bounds=bounds_kw)
            is_valid, _, failed, _ = Hypergraph(edited).validate(quick=True)
            if not is_valid:
                raise ValueError("validation failed: " + "; ".join(failed))
            current = edited
        except Exception as exc:
            logger.warning("Procedural failed for %s at step %d: %s", sid, step_i, exc)
            return base, {"procedural_steps_applied": step_i - 1, "procedural_error": str(exc)}

    return current, {"procedural_steps_applied": len(cmds)}


def _procedural_worker_pack(pack: tuple) -> tuple[str, dict, dict[str, Any]]:
    sid, cmds_tuple, row_error, base_hg, gt_ag, bounds_b, inc_f, mps = pack
    hg, meta = _apply_procedural_chain_for_sample(sid, list(cmds_tuple), row_error, base_hg, gt_ag, bounds_b, inc_f, mps)
    return sid, hg, meta


def apply_procedural_step1_to_hypergraphs(
    output: dict[str, Any],
    results: dict[str, dict],
    gt_accessgraph_by_id: dict[str, dict],
    bounds_by_id: dict[str, dict],
    *,
    increase_factor: float,
    procedural_steps: int = 5,
    workers: int = 16,
    show_progress: bool = True,
) -> dict[str, dict]:
    """Apply procedural chain for each sample in parallel (spawn). Returns ``id -> hypergraph``."""
    n_out = len(output)
    hyper_out: dict[str, dict] = {}

    def _finish(sid: str, row: dict[str, Any], hg: dict, meta: dict[str, Any]) -> None:
        row.update(meta)
        hyper_out[sid] = hg

    if workers <= 1:
        for sid, row in tqdm(output.items(), total=n_out, desc="Procedural edits", unit="sample", disable=not show_progress):
            hg, meta = _apply_procedural_chain_for_sample(
                sid, _multiset_commands_from_row(row), row.get("error"),
                repair_bsp_tree_names(results[sid]),
                gt_accessgraph_by_id.get(sid, {}), bounds_by_id.get(sid) or {},
                increase_factor, procedural_steps,
            )
            _finish(sid, row, hg, meta)
        return hyper_out

    packs = [
        (sid, tuple(_multiset_commands_from_row(row)), row.get("error"),
         repair_bsp_tree_names(results[sid]),
         copy.deepcopy(gt_accessgraph_by_id.get(sid, {})),
         copy.deepcopy(bounds_by_id.get(sid) or {}),
         increase_factor, procedural_steps)
        for sid, row in output.items()
    ]
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futs = [pool.submit(_procedural_worker_pack, p) for p in packs]
        with tqdm(total=n_out, desc="Procedural edits", unit="sample", disable=not show_progress) as pbar:
            for fut in as_completed(futs):
                sid, hg, meta = fut.result()
                _finish(sid, output[sid], hg, meta)
                pbar.update(1)
    return hyper_out


def process_sample(
    sample_id: str,
    hypergraph: dict,
    gt_accessgraph: dict,
    bounds: dict,
    *,
    multiset_plan_max: int = 0,
) -> dict[str, Any]:
    hypergraph = repair_bsp_tree_names(hypergraph)
    hg = Hypergraph(hypergraph)
    input_ag = hg.access_graph
    try:
        corrected_ag = hg.correct_accessgraph(bounds)
    except Exception as exc:
        logger.warning("correct_accessgraph failed for %s: %s", sample_id, exc)
        return {"command": None, "error": f"correct_accessgraph: {exc}"}

    cmd1, detail = compare_accessgraphs(input_ag, corrected_ag, gt_accessgraph, max_plan_steps=multiset_plan_max)
    chain = detail.get("multiset_command_chain") or []
    logger.info("%s -> %s | chain=%s", sample_id, cmd1, chain)
    return {"command": cmd1, "detail": detail, "multiset_commands": chain}


# --- Optional batch modes: orientation (bsp) & parametric optimization (GD) ---


def orientation_variant_passes_ref_accessgraph(
    mesh: Any,
    ref_ag: dict[str, Any] | None,
) -> bool:
    """
    Whether *mesh* complies with **ref_ag**, the unrepaired hypergraph's
    ``correct_accessgraph(bounds)``: each edge *(u,v)* in **ref_ag** must appear as adjacent
    room faces in the geometry, keyed by leaf ``mergeid`` types—the same semantics as
    :func:`scripts.utils_parametric.check_accessgraph_edges_against_mesh` used in parametric δ/ε runs.

    If **ref_ag** is missing or has no edges (or no checks are returned), passes ``True``, matching the
    optimization ``accessgraph_penalty`` logic.

    Called with the **variant mesh** (reoriented BSP, canonical bounds); **does not**
    recompute ``correct_accessgraph`` on the variant.
    """
    from scripts.utils_parametric import check_accessgraph_edges_against_mesh

    if not isinstance(ref_ag, dict) or not ref_ag.get("edges"):
        return True
    checks = check_accessgraph_edges_against_mesh(mesh, ref_ag)
    if not checks:
        return True
    return all(bool(c.get("ok")) for c in checks)


_REFLECT_MODES_OR: tuple[tuple[str | None, str], ...] = (
    (None, "no_flip"),
    ("x", "flip_x"),
    ("y", "flip_y"),
)
_ROT_DEGS_OR: tuple[int, ...] = (0, 90, 180, 270)
_REFLECT_CODE_TO_REORIENT_MODE_OR: dict[str | None, str] = {
    None: "normal",
    "x": "x",
    "y": "y",
}


def orient_one_sample(
    sid: str,
    base_hg: dict,
    bounds: dict,
) -> dict[str, Any]:
    """
    ``explore_rotation`` **bsp**-style search: reorient the pred BSP with canonical
    ``bounds``, pick the (mirror,°) with lowest **δ_sq** (room compactness vs a square
    of equal area; same as ``--objective delta`` in ``run_parametric`` / ``utils_parametric``).
    Does **not** use a ground-truth hypergraph. Requires ``bounds.corners`` for meshing.

    Variants excluded from the argmin unless the variant **mesh** satisfies every edge of
    the reference ``correct_accessgraph`` for the unreoriented repaired input (**ref_ag**),
    using :func:`~scripts.postprocessing.orientation_variant_passes_ref_accessgraph` (same edge
    check as ``run_parametric`` / ``check_accessgraph_edges_against_mesh`` — no recomputation of
    ``correct_accessgraph`` on the rotated BSP).
    If the reference ``correct_accessgraph`` itself fails (**ref_ag** is None), no such filter applies.
    """
    from scripts.procedural_edits import reorient_hypergraph_bsp
    from scripts.utils_parametric import _delta_metric_square_from_room_list
    from utils.metrics import produce_room_list

    if not (bounds and isinstance(bounds, dict) and bounds.get("corners")):
        return {
            "error": "missing bounds.corners",
            "hypergraph": copy.deepcopy(repair_bsp_tree_names(base_hg)),
        }

    base = repair_bsp_tree_names(copy.deepcopy(base_hg))
    ref_ag: dict[str, Any] | None = None
    try:
        ref_ag = Hypergraph(copy.deepcopy(base)).correct_accessgraph(bounds)
    except Exception:
        ref_ag = None

    rows: list[dict[str, Any]] = []
    for reflect_code, reflect_label in _REFLECT_MODES_OR:
        for deg in _ROT_DEGS_OR:
            try:
                pred = reorient_hypergraph_bsp(
                    base,
                    float(deg),
                    _REFLECT_CODE_TO_REORIENT_MODE_OR[reflect_code],
                )
                mesh_bounds = bounds
                pred_mesh = Hypergraph(copy.deepcopy(pred)).to_mesh(bounds=mesh_bounds)
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "reflect": reflect_code,
                        "reflect_label": reflect_label,
                        "deg": deg,
                        "delta": None,
                        "delta_sq": None,
                        "error": str(exc),
                    }
                )
                continue
            accessgraph_mismatch = False
            if ref_ag is not None and not orientation_variant_passes_ref_accessgraph(
                pred_mesh, ref_ag
            ):
                accessgraph_mismatch = True
            pred_room = produce_room_list(pred_mesh)
            dlt = float(_delta_metric_square_from_room_list(pred_room))
            row: dict[str, Any] = {
                "reflect": reflect_code,
                "reflect_label": reflect_label,
                "deg": deg,
                "delta": dlt,
                "delta_sq": dlt,
                "out_hypergraph": copy.deepcopy(pred),
                "mesh_bounds": copy.deepcopy(mesh_bounds),
            }
            if accessgraph_mismatch:
                row["accessgraph_mismatch"] = True
            rows.append(row)

    valid = [
        r
        for r in rows
        if "error" not in r
        and r.get("delta") is not None
        and math.isfinite(float(r["delta"]))
        and not r.get("accessgraph_mismatch")
    ]
    if not valid:
        return {
            "error": "no valid orientation cells",
            "hypergraph": copy.deepcopy(base),
            "meta": {
                "style": "bsp",
                "metric": "delta_sq",
                "rows": rows,
                "n_tried": len(rows),
                "accessgraph_reference_ok": ref_ag is not None,
            },
        }

    best = min(valid, key=lambda r: float(r["delta"]))

    return {
        "hypergraph": copy.deepcopy(best["out_hypergraph"]),
        "mesh_bounds": copy.deepcopy(best.get("mesh_bounds") or {}),
        "dataset_bounds": copy.deepcopy(bounds),
        "meta": {
            "style": "bsp",
            "metric": "delta_sq",
            "reflect": best.get("reflect"),
            "reflect_label": best.get("reflect_label"),
            "deg": int(best.get("deg", 0)),
            "delta_sq": best.get("delta"),
            "delta": best.get("delta"),
            "n_variants_tried": len(rows),
            "accessgraph_reference_ok": ref_ag is not None,
        },
    }


def _orientation_pack(
    p: tuple[str, dict, dict],
) -> tuple[str, dict[str, Any]]:
    sid, base_hg, b2 = p
    return sid, orient_one_sample(sid, base_hg, b2)


def _hypergraph_from_results_value(v: Any) -> dict[str, Any]:
    """``results[id]`` may be a root tree or ``{\"hypergraph\": <tree>, ...}``."""
    if not isinstance(v, dict):
        return {}
    if "hypergraph" in v and isinstance(v["hypergraph"], dict) and v["hypergraph"]:
        return v["hypergraph"]
    return v


def run_orientation_mode(
    results: dict[str, dict],
    test_dir: Path,
    *,
    out_path: Path,
    max_samples: int | None,
    workers: int,
    show_progress: bool,
) -> None:
    from scripts.visualize_procedural_edited import load_eval_access_gt

    ids = sorted(results.keys(), key=lambda x: (len(str(x)), str(x)))
    if max_samples is not None:
        ids = ids[: int(max_samples)]
    n_out = len(ids)
    out: dict[str, Any] = {}

    def _record(sid: str, rec: dict[str, Any]) -> None:
        if rec.get("error"):
            logger.warning("Orientation %s: %s", sid, rec["error"])
        hg = rec.get("hypergraph")
        if not isinstance(hg, dict) or not hg:
            base = _hypergraph_from_results_value(results.get(sid, {}))
            hg = copy.deepcopy(repair_bsp_tree_names(base))
        else:
            hg = copy.deepcopy(hg)
        out[sid] = hg

    if workers <= 1:
        it = ids
        if show_progress:
            it = tqdm(ids, desc="Orientation", total=n_out, unit="sample")
        for sid in it:
            b2, _gth, _ag = load_eval_access_gt(test_dir, sid)
            rec = orient_one_sample(sid, _hypergraph_from_results_value(results[sid]), b2)
            _record(sid, rec)
    else:
        packs: list[tuple[str, dict, dict]] = []
        for sid in ids:
            b2, _gth, _ag = load_eval_access_gt(test_dir, sid)
            packs.append((sid, _hypergraph_from_results_value(results[sid]), b2))
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futs = [pool.submit(_orientation_pack, p) for p in packs]
            it = as_completed(futs)
            if show_progress:
                it = tqdm(it, total=n_out, desc="Orientation", unit="sample")
            for fut in it:
                sid, rec = fut.result()
                _record(sid, rec)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    logger.info("Wrote hypergraphs only (%d ids, same format as --results-hypergraphs) to %s", len(out), out_path)


def _optim_worker_pack(t: tuple) -> tuple[str, dict, dict[str, Any] | None, str | None]:
    import json as _json
    from pathlib import Path as _Path

    from scripts.run_parametric import optimize_hypergraph
    from utils_parametric import OptimConfig

    sid, grpo_path_s, cfg_json, _unused = t
    grpo = _Path(grpo_path_s)
    with open(grpo, encoding="utf-8") as f:
        allh: dict = _json.load(f)
    if sid not in allh:
        return sid, repair_bsp_tree_names({}), None, f"missing id {sid}"
    cfg = OptimConfig(**_json.loads(cfg_json))
    try:
        best_hg, _bm, _om, _gt, mo, mp, *_ = optimize_hypergraph(sid, cfg, grpo_file_path=grpo)
        return sid, best_hg, {"metrics_original": mo, "metrics_optimized": mp}, None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Optimization failed for %s: %s", sid, exc)
        return sid, repair_bsp_tree_names(allh[sid]), None, f"{type(exc).__name__}: {exc}"


def run_optimization_mode(
    results: dict[str, dict],
    test_dir: Path,
    grpo: Path,
    *,
    out_hg: Path,
    out_meta: Path,
    max_samples: int | None,
    workers: int,
    show_progress: bool,
    objective: str = "delta",
) -> None:
    from dataclasses import asdict

    from utils_parametric import OptimConfig

    try:
        td_rel = str(test_dir.resolve().relative_to(REPO_ROOT))
    except ValueError:
        td_rel = str(test_dir)
    cfg_json = json.dumps(asdict(OptimConfig(objective=objective, test_dir=td_rel, grpo_file=str(grpo.resolve()))), default=str)

    ids = sorted(results.keys(), key=lambda x: (len(str(x)), str(x)))
    if max_samples is not None:
        ids = ids[:int(max_samples)]
    n_out = len(ids)
    pack_args = [(sid, str(grpo.resolve()), cfg_json, td_rel) for sid in ids]
    out_h: dict[str, dict] = {}
    meta: dict[str, Any] = {}

    def _collect(sid: str, hg: dict, m: dict | None, err: str | None) -> None:
        out_h[sid] = hg
        if m or err:
            meta[sid] = {**(m or {}), **({} if not err else {"error": err})}

    if workers <= 1:
        for p in tqdm(pack_args, total=n_out, desc="Optimization", unit="sample", disable=not show_progress):
            _collect(*_optim_worker_pack(p))
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futs = [pool.submit(_optim_worker_pack, p) for p in pack_args]
            with tqdm(total=n_out, desc="Optimization", unit="sample", disable=not show_progress) as pbar:
                for fut in as_completed(futs):
                    _collect(*fut.result())
                    pbar.update(1)

    out_hg.parent.mkdir(parents=True, exist_ok=True)
    with open(out_hg, "w", encoding="utf-8") as f:
        json.dump(out_h, f, indent=2, sort_keys=True)
    logger.info("Wrote %d optimized hypergraphs to %s", len(out_h), out_hg)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-hypergraphs", type=Path,
        default=REPO_ROOT / "inference_8700_on_eval_access_vllm_64" / "results_hypergraphs.json")
    parser.add_argument("--test-dir", type=Path,
        default=REPO_ROOT / "dataset" / "rplan" / "eval_access" / "test")
    parser.add_argument("--output-json", type=Path, default=None,
        help="Output for parametric commands (default: parametric_commands.json beside --results-hypergraphs).")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--apply-procedural-step1", action="store_true",
        help="Apply multiset add/remove plan via procedural_edits; writes hypergraph JSON.")
    parser.add_argument("--procedural-steps", type=int, default=5, metavar="N",
        help="Max add/remove steps to plan and apply (default: %(default)s; 0 = greedy until converged).")
    parser.add_argument("--output-step1-hypergraphs", type=Path, default=None,
        help="Default: results_hypergraphs_step1_procedural.json beside --results-hypergraphs.")
    parser.add_argument("--increase-factor", type=float, default=2.0,
        help="Scaling factor for increase/reduce commands.")
    parser.add_argument("--workers", type=int, default=16, metavar="N",
        help="Parallel spawn processes for procedural/orientation/optimization (default: %(default)s).")
    parser.add_argument("--pp-mode", type=str, default="parametric",
        choices=("parametric", "orientation", "optimization"),
        help="parametric (default), orientation (BSP 3×4 search), or optimization (GD).")
    parser.add_argument("--output-orientation", type=Path, default=None,
        help="Orientation output (default: results_hypergraphs_orientation.json).")
    parser.add_argument("--output-optimization", type=Path, default=None,
        help="Optimization output hypergraphs (default: results_hypergraphs_optimized.json).")
    parser.add_argument("--optimization-objective", type=str, default="delta",
        choices=("delta", "area", "area_delta", "delta_epsilon", "epsilon"))
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(levelname)s: %(message)s")

    with open(args.results_hypergraphs, encoding="utf-8") as f:
        results: dict[str, dict] = json.load(f)

    if args.pp_mode == "orientation":
        out_o = args.output_orientation or (args.results_hypergraphs.parent / "results_hypergraphs_orientation.json")
        run_orientation_mode(results, args.test_dir, out_path=out_o,
            max_samples=args.max_samples, workers=int(args.workers), show_progress=not args.quiet)
        return 0

    if args.pp_mode == "optimization":
        out_hg = args.output_optimization or (args.results_hypergraphs.parent / "results_hypergraphs_optimized.json")
        out_meta = out_hg.parent / f"{out_hg.stem}_meta.json"
        run_optimization_mode(results, args.test_dir, args.results_hypergraphs,
            out_hg=out_hg, out_meta=out_meta, max_samples=args.max_samples,
            workers=int(args.workers), show_progress=not args.quiet,
            objective=str(args.optimization_objective))
        return 0

    # --- parametric mode ---
    out_path = args.output_json or (args.results_hypergraphs.parent / "parametric_commands.json")
    ids = sorted(results.keys(), key=lambda x: (len(str(x)), str(x)))
    if args.max_samples is not None:
        ids = ids[:args.max_samples]

    output: dict[str, Any] = {}
    gt_accessgraph_by_id: dict[str, dict] = {}
    bounds_by_id: dict[str, dict] = {}

    for sid in tqdm(ids, desc="Parametric commands", unit="sample", disable=args.quiet):
        try:
            rec = load_test_record(args.test_dir, sid)
        except FileNotFoundError:
            output[sid] = {"command": None, "error": "missing_test_txt"}
            continue
        except json.JSONDecodeError as exc:
            output[sid] = {"command": None, "error": f"test_json: {exc}"}
            continue
        gt_accessgraph_by_id[sid] = rec["accessgraph"]
        bounds_by_id[sid] = rec.get("bounds") or {}
        try:
            output[sid] = process_sample(sid, results[sid], rec["accessgraph"], rec["bounds"],
                multiset_plan_max=args.procedural_steps)
        except Exception as exc:
            logger.warning("Failed sample %s: %s", sid, exc)
            output[sid] = {"command": None, "error": str(exc)}

    if args.apply_procedural_step1:
        hg_path = args.output_step1_hypergraphs or (
            args.results_hypergraphs.parent / "results_hypergraphs_step1_procedural.json"
        )
        hyper_step1 = apply_procedural_step1_to_hypergraphs(
            output, results, gt_accessgraph_by_id, bounds_by_id,
            increase_factor=args.increase_factor, procedural_steps=args.procedural_steps,
            workers=args.workers, show_progress=not args.quiet,
        )
        hg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(hg_path, "w") as f:
            json.dump(hyper_step1, f, indent=2, sort_keys=True)
        n_applied = sum(1 for r in output.values() if (r.get("procedural_steps_applied") or 0) >= 1)
        logger.info("Wrote procedural hypergraphs for %d ids to %s (%d with >=1 step)", len(hyper_step1), hg_path, n_applied)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)
    logger.info("Wrote %d entries to %s", len(output), out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
