#!/usr/bin/env python3
"""
Derive simple parametric edit hints from predicted vs ground-truth room multisets.

For each predicted hypergraph, first applies ``repair_bsp_tree_names`` (in-memory
copy only; the JSON file passed via ``--results-hypergraphs`` is never modified). That
fixes duplicate BSP ``name`` strings so access graphs and mesh ``unique_id`` mapping are
well-defined. For each repaired copy, the script then builds:
  - *input* access graph from leaf connectivity (``Hypergraph.access_graph``)
  - *corrected* access graph via mesh adjacency (``Hypergraph.correct_accessgraph``)

Then compares **room-type multisets** (values of ``accessgraph["nodes"]``) between
corrected prediction and ground-truth access graph from the eval test ``.txt``.

Output command strings (one per sample, written to ``parametric_commands.json``):
  - ``NOTHING`` — multisets match
  - ``add <room_type>`` — at least one room type appears fewer times in corrected than in GT
  - ``remove <room_type>`` — at least one room type appears more in corrected than in GT

When both missing and extra types exist in one step, **add** is preferred (largest deficit first).

**Two-step mode** (default): after choosing the first command, the script **simulates** that
command on the corrected room multiset (``add`` / ``remove`` one count) and runs the same
check again vs GT. This surfaces cases that need **both** an add and a remove (or two ops
in either order at the multiset level). Fields ``command_step2`` and
``needs_second_modification`` record the outcome.

**Procedural edits** (optional, ``--apply-procedural-step1``): maps multiset **plans**
(``add`` / ``remove``) to ``scripts.procedural_edits.apply_command`` (``remove`` → ``delete``;
``add X`` → ``add X by <anchor>``).  With ``--procedural-steps N`` (1–4), plans and applies up
to **N** greedy multiset steps (see ``multiset_commands`` / ``detail.multiset_command_chain``).
The anchor
room type is chosen from the **ground-truth access graph**: neighbor types of a GT node whose
label is *X* (room to add), restricted to types that exist in the prediction; tie-break by
largest total leaf area in the pred hypergraph.  If that fails, falls back to splitting the
largest non-*X* leaf in the pred.  Runs on the **input** hypergraph from ``--results-hypergraphs``.
For ``add``, the new leaf's ``mergeid`` is the most common raw GT label matching the multiset
room type (via the same canonicalization as procedural_edits), when present, so strict multiset
scoring aligns with eval access-graph spelling (e.g. ``bed`` vs ``bedroom``). Writes only a JSON
dict ``sample_id -> hypergraph`` — no images or extra folders. **Orientation search** runs last
by default (BSP reflect/rotate; bounds unchanged); use ``--skip-orientation-opt`` for add/remove
only. Procedural + orientation stages are
**CPU-heavy** (RGL meshing via .NET); use ``--workers N`` to process samples in parallel
(``spawn`` processes; default ``N=16``).
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import multiprocessing
import re
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


def multiset_command_add_only(cnt_pred: Counter[str], cnt_gt: Counter[str]) -> tuple[str, dict[str, Any]]:
    """
    One greedy *add* step, or ``NOTHING`` if there is no missing count (excess types are ignored).
    """
    step_detail = _multiset_diff_detail(cnt_pred, cnt_gt)
    missing = cnt_gt - cnt_pred
    if not missing:
        return "NOTHING", step_detail
    room = max(missing.keys(), key=lambda k: (missing[k], k))
    return f"add {room}", step_detail


def multiset_command_remove_only(
    cnt_pred: Counter[str], cnt_gt: Counter[str],
) -> tuple[str, dict[str, Any]]:
    """
    One greedy *remove* step, or ``NOTHING`` if there is no surplus count (deficits are ignored).
    """
    step_detail = _multiset_diff_detail(cnt_pred, cnt_gt)
    extra = cnt_pred - cnt_gt
    if not extra:
        return "NOTHING", step_detail
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
    add_only: bool = False,
    remove_only: bool = False,
) -> list[str]:
    """
    Greedy multiset plan: repeat ``multiset_command`` + simulate until match or *max_steps* (cap 4).
    If *add_only* is true, use only ``add`` steps; if *remove_only*, only ``remove`` steps.
    """
    cap = max(0, min(int(max_steps), 4))
    if cap == 0:
        return []
    chain: list[str] = []
    cnt = copy.copy(cnt_corr)
    if add_only and remove_only:
        raise ValueError("add_only and remove_only are mutually exclusive")
    if add_only:
        mcmd = multiset_command_add_only
    elif remove_only:
        mcmd = multiset_command_remove_only
    else:
        mcmd = multiset_command
    for _ in range(cap):
        cmd, _st = mcmd(cnt, cnt_gt)
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
    two_step: bool = True,
    max_plan_steps: int = 2,
    add_only: bool = False,
    remove_only: bool = False,
) -> tuple[str, dict[str, Any], str | None, dict[str, Any]]:
    """
    Compare room-type multisets; plan up to *max_plan_steps* greedy add/remove commands (cap 4).

    Returns:
        ``command_step1, detail, command_step2, two_step_info``

        *command_step2* is ``None`` when ``two_step`` is False.
        *detail* includes ``multiset_command_chain`` and simulated counts after each planned step.
    """
    cnt_in = room_type_multiset(input_accessgraph)
    cnt_corr = room_type_multiset(corrected_accessgraph)
    cnt_gt = room_type_multiset(gt_accessgraph)

    if add_only and remove_only:
        raise ValueError("add_only and remove_only are mutually exclusive")
    _mode = "full"
    if add_only:
        _mode = "add_only"
    elif remove_only:
        _mode = "remove_only"

    detail: dict[str, Any] = {
        "input_room_counts": dict(sorted(cnt_in.items())),
        "corrected_room_counts": dict(sorted(cnt_corr.items())),
        "gt_room_counts": dict(sorted(cnt_gt.items())),
        "multiset_mode": _mode,
    }

    step1 = _multiset_diff_detail(cnt_corr, cnt_gt)
    detail["step1"] = step1
    missing1 = Counter(step1["missing"])
    extra1 = Counter(step1["extra"])
    detail["step1_both_add_and_remove_needed"] = bool(missing1 and extra1)

    cap = max(1, min(int(max_plan_steps), 4))
    chain = plan_multiset_command_chain(
        cnt_corr, cnt_gt, cap, add_only=add_only, remove_only=remove_only
    )
    detail["multiset_command_chain"] = chain
    cnt_sim = copy.copy(cnt_corr)
    after_each: list[dict[str, Any]] = []
    for cmd in chain:
        cnt_sim = simulate_multiset_command(cnt_sim, cmd)
        after_each.append(dict(sorted(cnt_sim.items())))
    detail["room_counts_after_each_multiset_step"] = after_each

    cmd1 = chain[0] if chain else "NOTHING"
    cmd2: str | None
    if not two_step:
        cmd2 = None
    else:
        cmd2 = chain[1] if len(chain) > 1 else "NOTHING"

    two_step_info: dict[str, Any] = {
        "needs_second_modification": len(chain) >= 2,
        "room_counts_after_step1_simulation": {},
        "step2": {},
    }
    if chain:
        cnt_after_one = simulate_multiset_command(cnt_corr, chain[0])
        two_step_info["room_counts_after_step1_simulation"] = dict(
            sorted(cnt_after_one.items())
        )
        if add_only:
            mnext = multiset_command_add_only
        elif remove_only:
            mnext = multiset_command_remove_only
        else:
            mnext = multiset_command
        _c2, step2 = mnext(cnt_after_one, cnt_gt)
        two_step_info["step2"] = step2
    else:
        two_step_info["step2"] = {"missing": {}, "extra": {}}

    detail["multiset_diff"] = {"missing": step1["missing"], "extra": step1["extra"]}
    if len(chain) > 1:
        detail["step2_if_step1_simulated"] = two_step_info["step2"]

    return cmd1, detail, cmd2, two_step_info


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


def _total_leaf_area_for_type(hg: Hypergraph, room_type_lower: str) -> float:
    rtl = room_type_lower.strip().lower()
    return sum(
        float(leaf.get("area", 0.0))
        for leaf in hg.leaves
        if str(leaf.get("mergeid", "")).strip().lower() == rtl
    )


def complete_add_command_for_procedural(
    hypergraph: dict,
    room_to_add: str,
    gt_accessgraph: dict,
) -> tuple[str, str]:
    """
    Build ``add <new> by <anchor>`` for :func:`scripts.procedural_edits.apply_command`.

    Prefers an **anchor** room type that is a **neighbor in the GT access graph** of some
    node typed *room_to_add*, and that exists in the predicted hypergraph.  Otherwise
    falls back to :func:`_fallback_anchor_largest_non_target`.

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
    candidates = sorted(
        neighbor_types & pred_types,
        key=lambda t: (-_total_leaf_area_for_type(hg, t), t),
    )

    if candidates:
        anchor_type = candidates[0]
        anchor_mid = _representative_mergeid_for_type(hg, anchor_type)
        if anchor_mid is not None:
            return f"add {new_room} by {anchor_mid}", "gt_neighbor"

    anchor_mergeid = _fallback_anchor_largest_non_target(hypergraph, room_to_add)
    return f"add {new_room} by {anchor_mergeid}", "fallback_largest_leaf"


def complete_add_command_candidates(
    hypergraph: dict,
    room_to_add: str,
    gt_accessgraph: dict,
    max_cand: int = 4,
) -> list[tuple[str, str]]:
    max_cand = max(1, min(int(max_cand), 4))
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
    candidates = sorted(
        neighbor_types & pred_types,
        key=lambda t: (-_total_leaf_area_for_type(hg, t), t),
    )
    out: list[tuple[str, str]] = []
    if candidates:
        for anchor_type in candidates:
            if len(out) >= max_cand:
                break
            anchor_mid = _representative_mergeid_for_type(hg, anchor_type)
            if anchor_mid is not None:
                out.append((f"add {new_room} by {anchor_mid}", "gt_neighbor"))
    if not out:
        anchor_mergeid = _fallback_anchor_largest_non_target(hypergraph, room_to_add)
        return [(f"add {new_room} by {anchor_mergeid}", "fallback_largest_leaf")]
    return out[:max_cand]


def access_graph_light_distance(ag_pred: dict, ag_gt: dict) -> int:
    """
    Fast surrogate for GED: L1 on room-type node multisets plus L1 on the multiset
    of undirected ``(type_i, type_j)`` pairs (``i <= j`` lexicographically) for edges.
    O(|nodes| + |edges|). Lower is better.
    """
    from collections import Counter

    def _node_counter(ag: dict) -> Counter:
        c: Counter = Counter()
        for t in (ag or {}).get("nodes", {}).values():
            c[str(t).strip().lower()] += 1
        return c

    def _edge_type_counter(ag: dict) -> Counter:
        c: Counter = Counter()
        nodes = {k: str(v).strip().lower() for k, v in (ag or {}).get("nodes", {}).items()}
        for e in (ag or {}).get("edges") or []:
            if not isinstance(e, (list, tuple)) or len(e) < 2:
                continue
            u, v = str(e[0]), str(e[1])
            a, b = nodes.get(u, ""), nodes.get(v, "")
            if a > b:
                a, b = b, a
            c[(a, b)] += 1
        return c

    np, ng = _node_counter(ag_pred), _node_counter(ag_gt)
    ep, eg = _edge_type_counter(ag_pred), _edge_type_counter(ag_gt)
    keys_n = set(np) | set(ng)
    keys_e = set(ep) | set(eg)
    node_loss = sum(abs(np.get(k, 0) - ng.get(k, 0)) for k in keys_n)
    edge_loss = sum(abs(ep.get(k, 0) - eg.get(k, 0)) for k in keys_e)
    return int(node_loss + edge_loss)


def _ged_corrected_vs_gt(
    hypergraph: dict,
    gt_accessgraph: dict,
    bounds: dict | None,
    sample_id: str,
    timeout_ms: int,
) -> tuple[float | None, int | None]:
    from utils import compute_ged_from_hypergraph

    if not (isinstance(bounds, dict) and bounds.get("corners")):
        return None, None
    g, n = compute_ged_from_hypergraph(
        copy.deepcopy(gt_accessgraph),
        copy.deepcopy(hypergraph),
        str(sample_id),
        timeout=int(timeout_ms),
        correct_accessgraph=True,
        bounds=bounds,
        upper_bound=40,
        timeout_capture_best=False,
    )
    if g is None:
        return None, n
    return float(g), n if n is not None else None


def _candidate_score_vs_gt(
    hypergraph: dict,
    gt_accessgraph: dict,
    bounds: dict | None,
    sample_id: str,
    timeout_ms: int,
    candidates_metric: str,
) -> tuple[float | None, int | None]:
    """
    Score a candidate *hypergraph* against GT. ``candidates_metric`` / ``"ged"`` uses
    :func:`_ged_corrected_vs_gt` (NetworkX; may time out). ``"light"`` uses
    :func:`access_graph_light_distance` on **mesh-corrected** access graphs
    (same ``correct_accessgraph`` as GED) — no GED, typically cheap.
    """
    if candidates_metric == "light":
        if not (isinstance(bounds, dict) and bounds.get("corners")):
            return None, None
        try:
            pred_ag = Hypergraph(hypergraph).correct_accessgraph(bounds)
        except Exception as exc:
            logger.debug("correct_accessgraph (candidates) failed %s: %s", sample_id, exc)
            return None, None
        d = float(access_graph_light_distance(pred_ag, gt_accessgraph))
        return d, 0
    g, n = _ged_corrected_vs_gt(
        hypergraph, gt_accessgraph, bounds, sample_id, timeout_ms
    )
    return g, n


def _mesh_connectivity_tie_key(hg: dict, bounds: dict) -> tuple[int, int] | None:
    from scripts.procedural_edits import _mesh_corrected_connectivity_score

    if not (isinstance(bounds, dict) and bounds.get("corners")):
        return None
    s = _mesh_corrected_connectivity_score(hg, bounds)
    if s is None:
        return None
    return int(s[0]), -int(s[1])


def _pick_delete_by_ged_candidates(
    sid: str,
    current: dict,
    room_name: str,
    gt_ag: dict,
    bounds: dict,
    increase_factor: float,
    ged_timeout_ms: int,
    *,
    candidates_metric: str = "ged",
) -> tuple[dict, str, str, dict[str, Any], dict[str, Any]]:
    from scripts.procedural_edits import apply_command, eligible_delete_paths

    pcmd = f"delete {room_name}"
    cm = candidates_metric if candidates_metric in ("ged", "light") else "ged"
    try:
        paths = eligible_delete_paths(current, room_name)
    except ValueError as exc:
        empty: dict[str, Any] = {}
        edited, summary = apply_command(
            copy.deepcopy(current), pcmd, increase_factor, bounds=None
        )
        return edited, summary, pcmd, empty, {"candidates_ged": "not_applicable", "error": str(exc)}

    n_paths_raw = len(paths)
    ded = collect_delete_candidate_variants(
        current, room_name, increase_factor
    )
    scored: list[
        tuple[
            float | None,
            str,
            int,
            list[dict] | None,
            dict,
            tuple[int, int] | str,
        ]
    ] = []
    for t, pnm, path in ded:
        g, _gn = _candidate_score_vs_gt(
            t, gt_ag, bounds, sid, ged_timeout_ms, cm
        )
        path_name = str(path[-1].get("name", "")) if path else str(pnm)
        neighb = (
            _leaf_room_neighbor_count(path[-1])
            if path and len(path) >= 1
            else 0
        )
        mk = _mesh_connectivity_tie_key(t, bounds) or (999, 0)
        scored.append((g, path_name, neighb, path, t, mk))
    if not scored:
        t0 = copy.deepcopy(current)
        ed, sm = apply_command(t0, pcmd, increase_factor, bounds=None)
        return ed, sm, pcmd, {}, {"candidates_ged": "all_paths_failed", "pick": "default_apply_command"}

    ok = [x for x in scored if x[0] is not None]
    if ok:
        ok.sort(
            key=lambda it: (it[0] if it[0] is not None else 1e9, it[5], it[2], it[1])
        )
        g, pnm, _nb, _path, best_h, _mk = (
            ok[0][0],
            ok[0][1],
            ok[0][2],
            ok[0][3],
            ok[0][4],
            ok[0][5],
        )
        mtag = "min_ged" if cm == "ged" else "min_access_light"
        sym = "GED" if cm == "ged" else "access-light"
        return (
            best_h,
            f"Deleted {room_name} ({sym} pick among {n_paths_raw} target leaves); leaf={pnm!r} score={g!r}",
            pcmd,
            {},
            {
                "candidates_metric": cm,
                "candidates_ged": mtag,
                "candidates_ged_value": g,
                "candidates_tried": len(scored),
                "candidates_paths": n_paths_raw,
            },
        )

    best_tie = min(
        scored,
        key=lambda it: (it[5], it[2], it[1]),
    )
    g0, pnm, _nb, _path, best_h, _mk = (
        best_tie[0],
        best_tie[1],
        best_tie[2],
        best_tie[3],
        best_tie[4],
        best_tie[5],
    )
    ttag = (
        "all_timeout_tiebreak_mesh" if cm == "ged" else "all_no_score_tiebreak_mesh"
    )
    tsym = "GED timeout" if cm == "ged" else "no score; mesh+neighbor"
    return (
        best_h,
        f"Deleted {room_name} ({tsym} tie) leaf={pnm!r}",
        pcmd,
        {},
        {
            "candidates_metric": cm,
            "candidates_ged": ttag,
            "candidates_tried": len(scored),
            "candidates_paths": n_paths_raw,
        },
    )


def _leaf_room_neighbor_count(leaf: dict) -> int:
    from scripts.procedural_edits import _leaf_room_neighbor_count as _ln

    return _ln(leaf)


def _add_candidate_budget_per_anchor(n_anchors: int, total: int = 4) -> list[int]:
    """
    Split *total* GED slots across *n_anchors* anchor choices (e.g. 1→4, 2→2+2, 3→2+1+1, 4→1+1+1+1).
    """
    n_anchors = max(1, min(int(n_anchors), total))
    base, rem = total // n_anchors, total % n_anchors
    return [base + (1 if i < rem else 0) for i in range(n_anchors)]


def collect_add_candidate_variants(
    current: dict,
    room_phrase: str,
    gt_ag: dict,
    bounds: dict,
    increase_factor: float,
    new_room_mergeid: str | None,
    *,
    add_explore: bool = False,
) -> tuple[list[tuple[dict, str, str, tuple | None]], list[int], int]:
    """
    Enumerate add split / anchor candidates before GED/light scoring.

    * **Default** (``add_explore=False``): same as :func:`_pick_add_by_ged_candidates`
      mesh top-k path (up to 4 candidates across anchors).
    * **Explore** (``add_explore=True``): build ``add <new> by <anchor>`` via
      :func:`complete_add_command_for_procedural`, list every final leaf matching
      anchor *Y* via :func:`collect_matching_leaf_paths`, then for each leaf try the
      four axis-aligned split angles in ``_ADD_MESH_SEARCH_ANGLES`` (new room on
      the “right” half; “four directions” = four rotation choices for the cut).

    Returns:
        ``(unique, k_list, n_pc)`` — *unique* is ``(hypergraph, pcmd, tag, mesh_key)`` tuples.
    """
    from scripts.procedural_edits import (
        _ADD_MESH_SEARCH_ANGLES,
        apply_command,
        collect_matching_leaf_paths,
        split_leaf_for_add_on_path,
        split_leaf_for_add_topk,
    )

    _add_re = re.compile(
        r"add\s+([a-zA-Z_ ]+?)\s+by\s+([a-zA-Z_ ]+)", flags=re.IGNORECASE
    )

    if add_explore:
        cands_ex: list[tuple[dict, str, str, tuple | None]] = []
        try:
            pcmd_def, src = complete_add_command_for_procedural(
                current, room_phrase, gt_ag
            )
        except Exception as exc:
            logger.debug("complete_add_command_for_procedural: %s", exc)
            pcmd_def, src = "", ""
        m0 = _add_re.fullmatch(pcmd_def.strip()) if pcmd_def else None
        if m0:
            new_r, anchor = m0.group(1).strip(), m0.group(2).strip()
            try:
                paths_y = collect_matching_leaf_paths(current, anchor)
            except Exception as exc:
                logger.debug("collect_matching_leaf_paths %s: %s", anchor, exc)
                paths_y = []
            for path in paths_y:
                if len(path) < 2:
                    continue
                for ang in _ADD_MESH_SEARCH_ANGLES:
                    t = copy.deepcopy(current)
                    try:
                        det = split_leaf_for_add_on_path(
                            t,
                            path,
                            new_r,
                            new_room_mergeid=new_room_mergeid,
                            new_room_on_right=True,
                            split_angle=float(ang),
                        )
                    except Exception as exc:
                        logger.debug(
                            "split_leaf_for_add_on_path %s angle=%s: %s",
                            path[-1].get("name"),
                            ang,
                            exc,
                        )
                        continue
                    try:
                        is_val, _, _, _ = Hypergraph(t).validate(quick=True)
                    except Exception:
                        continue
                    if not is_val:
                        continue
                    pcmd_row = f"add {new_r} by {anchor}"
                    leaf_nm = str(path[-1].get("name", ""))
                    tag = f"anchor:{src};leaf:{leaf_nm};{det}"
                    cands_ex.append(
                        (
                            t,
                            pcmd_row,
                            tag,
                            _mesh_connectivity_tie_key(t, bounds),
                        )
                    )
        seen_e: set[int] = set()
        unique_e: list[tuple[dict, str, str, tuple | None]] = []
        for it in cands_ex:
            if id(it[0]) in seen_e:
                continue
            seen_e.add(id(it[0]))
            unique_e.append(it)
        if unique_e:
            return unique_e, [], len(unique_e)

    pcmds = complete_add_command_candidates(current, room_phrase, gt_ag, 4)
    cands: list[tuple[dict, str, str, tuple | None]] = []
    n_pc = len(pcmds)
    k_list: list[int] = []
    if pcmds:
        k_list = _add_candidate_budget_per_anchor(n_pc, 4)
        for (pcmd, src), ki in zip(pcmds, k_list):
            m = _add_re.fullmatch(pcmd.strip())
            if m is None:
                continue
            new_r, anchor = m.group(1).strip(), m.group(2).strip()
            try:
                pieces = split_leaf_for_add_topk(
                    copy.deepcopy(current),
                    anchor,
                    new_r,
                    new_room_mergeid=new_room_mergeid,
                    bounds=bounds,
                    max_k=max(1, int(ki)),
                )
            except Exception as exc:
                logger.debug("add topk failed %s: %s", pcmd, exc)
                continue
            for piece in pieces[: int(ki)]:
                tdict, det, mkey = piece
                tag = f"anchor:{src};{det.strip() if det else 'split'}"
                cands.append((tdict, pcmd, tag, mkey))
    if not cands and pcmds:
        t = copy.deepcopy(current)
        p0, s0 = pcmds[0]
        try:
            ed, sm = apply_command(
                t,
                p0,
                increase_factor,
                new_room_mergeid=new_room_mergeid,
                bounds=bounds,
            )
            cands.append(
                (ed, p0, f"anchor:{s0},apply", _mesh_connectivity_tie_key(ed, bounds))
            )
        except Exception as exc:
            logger.debug("add single fallback failed %s: %s", p0, exc)
    seen: set[int] = set()
    unique: list[tuple[dict, str, str, tuple | None]] = []
    for it in cands:
        if id(it[0]) in seen:
            continue
        seen.add(id(it[0]))
        unique.append(it)
    unique = unique[:4]
    return unique, k_list, n_pc


def collect_delete_candidate_variants(
    current: dict,
    room_name: str,
    increase_factor: float,
) -> list[tuple[dict, str, list[dict] | None]]:
    """
    One candidate per final leaf of the requested type: for each path from
    :func:`eligible_delete_paths`, apply :func:`delete_room_at_path` (exactly one leaf
    removed). There is no maximum count (unlike add-candidate enumeration). Optional
    fallbacks use :func:`apply_command` only when there are no valid paths or all path
    deletes fail validation.
    """
    from scripts.procedural_edits import apply_command, delete_room_at_path, eligible_delete_paths

    pcmd = f"delete {room_name}"
    try:
        paths = eligible_delete_paths(current, room_name)
    except ValueError:
        t = copy.deepcopy(current)
        ed, _sm = apply_command(t, pcmd, increase_factor, bounds=None)
        return [(ed, "default_apply", None)]

    out: list[tuple[dict, str, list[dict] | None]] = []
    for path in paths:
        t = copy.deepcopy(current)
        try:
            delete_room_at_path(t, path)
            is_val, _, failed, _ = Hypergraph(t).validate(quick=True)
            if not is_val:
                continue
        except Exception:
            continue
        out.append((t, str(path[-1].get("name", "")), path))
    if not out:
        t0 = copy.deepcopy(current)
        ed, _sm = apply_command(t0, pcmd, increase_factor, bounds=None)
        return [(ed, "default_apply_command", None)]
    return out


def _pick_add_by_ged_candidates(
    sid: str,
    current: dict,
    room_phrase: str,
    gt_ag: dict,
    bounds: dict,
    increase_factor: float,
    new_room_mergeid: str | None,
    ged_timeout_ms: int,
    *,
    candidates_metric: str = "ged",
    add_explore: bool = False,
) -> tuple[dict, str, str, dict[str, Any], dict[str, Any]]:
    from scripts.procedural_edits import apply_command

    cm = candidates_metric if candidates_metric in ("ged", "light") else "ged"
    unique, k_list, n_pc = collect_add_candidate_variants(
        current,
        room_phrase,
        gt_ag,
        bounds,
        increase_factor,
        new_room_mergeid,
        add_explore=add_explore,
    )
    enum_base: dict[str, Any] = {
        "candidates_enumeration": (
            "add_explore_each_Y_times_4_angles"
            if add_explore
            else "split_leaf_for_add_topk_per_anchor"
        ),
    }
    if k_list:
        enum_base["candidates_n_anchors"] = n_pc
        enum_base["candidates_k_per_anchor"] = k_list
    if not unique:
        pcmd, src = complete_add_command_for_procedural(current, room_phrase, gt_ag)
        t = copy.deepcopy(current)
        ed, sm = apply_command(
            t,
            pcmd,
            increase_factor,
            new_room_mergeid=new_room_mergeid,
            bounds=bounds,
        )
        return ed, sm, pcmd, {"add_anchor_source": src}, {"candidates_ged": "no_candidates_fallback"}

    scored: list[tuple[float | None, tuple, dict, str, str]] = []
    for ed, pcmd, tag, mkey in unique:
        g, _n = _candidate_score_vs_gt(
            ed, gt_ag, bounds, sid, ged_timeout_ms, cm
        )
        mk = mkey if mkey is not None else (999, 0)
        scored.append((g, mk, ed, pcmd, tag))
    enum_base["candidates_tried"] = len(scored)
    enum_base["candidates_metric"] = cm
    ok = [x for x in scored if x[0] is not None]
    if ok:
        ok.sort(key=lambda it: (it[0] if it[0] is not None else 1e9, it[1]))
        g, _mk, best_h, pcmd, tag = ok[0]
        mtag = "min_ged" if cm == "ged" else "min_access_light"
        ex: dict[str, Any] = {
            "add_anchor_source": "candidates_ged_min"
            if cm == "ged"
            else "candidates_access_light_min"
        }
        sym = "GED" if cm == "ged" else "access-light"
        return (
            best_h,
            f"add {sym}-pick: {pcmd!r} ({tag}) score={g!r}",
            pcmd,
            ex,
            {
                **enum_base,
                "candidates_ged": mtag,
                "candidates_ged_value": g,
                "candidates_tag": tag,
            },
        )
    best_tie = min(
        scored,
        key=lambda it: (it[1], it[0] if it[0] is not None else 1e9),
    )
    g0, mkey0, best_h, pcmd, tag = best_tie
    ttag = (
        "all_timeout_tiebreak_mesh" if cm == "ged" else "all_no_score_tiebreak_mesh"
    )
    ex = {
        "add_anchor_source": "candidates_ged_tiebreak_mesh"
        if cm == "ged"
        else "candidates_access_light_tiebreak_mesh"
    }
    return (
        best_h,
        f"add (no {('GED' if cm == 'ged' else 'light')} score; mesh tie) {pcmd!r} ({tag})",
        pcmd,
        ex,
        {**enum_base, "candidates_ged": ttag, "candidates_tag": tag},
    )


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
    *,
    candidates_mode: bool = False,
    candidates_ged_timeout_ms: int = 3000,
    candidates_metric: str = "ged",
) -> tuple[dict, dict[str, Any]]:
    """
    Run procedural add/delete chain for one sample. Returns ``(hypergraph, row_updates)``.
    *row_updates* are keys to merge into the parametric row (procedural_* fields only).
    """
    from scripts.procedural_edits import apply_command

    base = copy.deepcopy(base_hg)
    cap = max(1, min(int(max_procedural_steps), 4))
    meta: dict[str, Any] = {
        "procedural_step1_applied": False,
        "procedural_step2_applied": False,
        "procedural_step3_applied": False,
        "procedural_step4_applied": False,
    }
    if candidates_mode:
        meta["procedural_ged_candidates_enabled"] = True
        meta["procedural_ged_candidates_timeout_ms"] = int(candidates_ged_timeout_ms)
        meta["candidates_metric"] = str(candidates_metric)

    if row_error:
        return base, meta

    cmds = [c for c in multiset_commands if c][:cap]
    if not cmds:
        return base, meta

    bounds_kw = (
        bounds_b if isinstance(bounds_b, dict) and bounds_b.get("corners") else None
    )
    current = base
    attempted_step = 0
    err_suffix = ("", "_step2", "_step3", "_step4")

    def _record_step(
        step_i: int,
        pcmd: str,
        summary: str,
        proc_extra: dict[str, Any],
        new_room_mergeid: str | None,
    ) -> None:
        meta[f"procedural_step{step_i}_applied"] = True
        if step_i == 1:
            meta["procedural_command"] = pcmd
            meta["procedural_apply_summary"] = summary
            if proc_extra.get("add_anchor_source"):
                meta["procedural_add_anchor_source"] = proc_extra["add_anchor_source"]
            if new_room_mergeid is not None:
                meta["procedural_add_mergeid_from_gt"] = new_room_mergeid
        else:
            suf = err_suffix[step_i - 1]
            meta[f"procedural_command{suf}"] = pcmd
            meta[f"procedural_apply_summary{suf}"] = summary
            if proc_extra.get("add_anchor_source"):
                meta[f"procedural_add_anchor_source{suf}"] = proc_extra[
                    "add_anchor_source"
                ]
            if new_room_mergeid is not None:
                meta[f"procedural_add_mergeid_from_gt{suf}"] = new_room_mergeid

    try:
        for step_i, cmd in enumerate(cmds, start=1):
            attempted_step = step_i
            new_room_mergeid = None
            if isinstance(cmd, str) and cmd.lower().startswith("add "):
                new_room_mergeid = _gt_preferred_mergeid_for_room_hint(
                    gt_ag, cmd[4:].strip()
                )
            use_cand = bool(candidates_mode and bounds_kw)
            cmd_l = (cmd or "").strip().lower()
            if use_cand and cmd_l.startswith("remove "):
                room = cmd[7:].strip()
                edited, summary, pcmd, proc_extra, cand_meta = _pick_delete_by_ged_candidates(
                    sid,
                    current,
                    room,
                    gt_ag,
                    bounds_b,
                    increase_factor,
                    candidates_ged_timeout_ms,
                    candidates_metric=candidates_metric,
                )
                meta[f"procedural_ged_candidates_step{step_i}"] = cand_meta
            elif use_cand and cmd_l.startswith("add "):
                room_phrase = cmd[4:].strip()
                edited, summary, pcmd, proc_extra, cand_meta = _pick_add_by_ged_candidates(
                    sid,
                    current,
                    room_phrase,
                    gt_ag,
                    bounds_b,
                    increase_factor,
                    new_room_mergeid,
                    candidates_ged_timeout_ms,
                    candidates_metric=candidates_metric,
                )
                meta[f"procedural_ged_candidates_step{step_i}"] = cand_meta
            else:
                pcmd, proc_extra = multiset_command_to_procedural(cmd, current, gt_ag)
                edited, summary = apply_command(
                    current,
                    pcmd,
                    increase_factor,
                    new_room_mergeid=new_room_mergeid,
                    bounds=bounds_kw,
                )
            is_valid, _, failed, _ = Hypergraph(edited).validate(quick=True)
            if not is_valid:
                raise ValueError("validation failed: " + "; ".join(failed))
            current = edited
            _record_step(step_i, pcmd, summary, proc_extra, new_room_mergeid)

        return current, meta
    except Exception as exc:
        logger.warning(
            "Procedural failed for %s (chain step %s): %s", sid, attempted_step, exc
        )
        err_key = (
            "procedural_apply_error"
            if attempted_step <= 1
            else f"procedural_apply_error{err_suffix[attempted_step - 1]}"
        )
        out_meta = {
            "procedural_step1_applied": False,
            "procedural_step2_applied": False,
            "procedural_step3_applied": False,
            "procedural_step4_applied": False,
            err_key: str(exc),
        }
        return base, out_meta


def _procedural_worker_pack(
    pack: tuple[
        str,
        tuple[str, ...],
        Any,
        dict,
        dict,
        dict,
        float,
        int,
        bool,
        int,
        str,
    ],
) -> tuple[str, dict, dict[str, Any]]:
    """Picklable entry for :class:`ProcessPoolExecutor` (``spawn`` context)."""
    (
        sid,
        multiset_commands_tuple,
        row_error,
        base_hg,
        gt_ag,
        bounds_b,
        increase_factor,
        max_procedural_steps,
        candidates_mode,
        candidates_ged_timeout_ms,
        candidates_metric,
    ) = pack
    hg, meta = _apply_procedural_chain_for_sample(
        sid,
        list(multiset_commands_tuple),
        row_error,
        base_hg,
        gt_ag,
        bounds_b,
        increase_factor,
        max_procedural_steps,
        candidates_mode=candidates_mode,
        candidates_ged_timeout_ms=candidates_ged_timeout_ms,
        candidates_metric=candidates_metric,
    )
    return sid, hg, meta


def _orientation_worker_pack(
    pack: tuple[str, dict, dict],
) -> tuple[str, dict, str | None, str]:
    """
    Returns ``(sid, hypergraph_out, orientation_summary_or_none, problem)``.
    *problem* is ``\"\"`` on success, else ``no_bounds`` / ``invalid`` / exception text.
    """
    sid, hg, bounds = pack
    from scripts.procedural_edits import optimize_rotation

    if not (isinstance(bounds, dict) and bounds.get("corners")):
        return sid, hg, None, "no_bounds"
    try:
        oh, omsg = optimize_rotation(hg, bounds)
        is_ok, _, failed, _ = Hypergraph(oh).validate(quick=True)
        if not is_ok:
            return sid, hg, None, "invalid:" + "; ".join(failed)
        return sid, oh, omsg, ""
    except Exception as exc:
        return sid, hg, None, str(exc)


def apply_procedural_step1_to_hypergraphs(
    output: dict[str, Any],
    results: dict[str, dict],
    gt_accessgraph_by_id: dict[str, dict],
    bounds_by_id: dict[str, dict],
    *,
    increase_factor: float,
    procedural_steps: int = 1,
    workers: int = 16,
    show_progress: bool = True,
    candidates_mode: bool = False,
    candidates_ged_timeout_ms: int = 3000,
    candidates_metric: str = "ged",
) -> dict[str, dict]:
    """
    For each sample, run up to *procedural_steps* (1–4) ``procedural_edits.apply_command``
    calls from ``row[\"multiset_commands\"]`` (greedy multiset plan in ``detail``).

    *bounds_by_id* is used for each ``add`` so procedural_edits can mesh-score BSP split
    variants (``correct_accessgraph`` connectivity).

    Mutates *output* rows with apply metadata. Returns ``id -> hypergraph`` (edited or
    original on skip/failure). Any failure in the chain reverts to the **input** hypergraph.

    With *workers* > 1, samples are processed in parallel using ``spawn`` processes (avoids
    forking after the CLR / RGL has been loaded in the parent). Use *workers* ``1`` for sequential.
    """
    n_out = len(output)
    hyper_out: dict[str, dict] = {}

    def _merge_row(sid: str, row: dict[str, Any], hg: dict, meta: dict[str, Any]) -> None:
        for k in (
            "procedural_apply_error",
            "procedural_apply_error_step2",
            "procedural_apply_error_step3",
            "procedural_apply_error_step4",
        ):
            row.pop(k, None)
        row.update(meta)
        hyper_out[sid] = hg

    if workers <= 1:
        row_iter = output.items()
        if show_progress:
            row_iter = tqdm(
                row_iter,
                total=n_out,
                desc="Procedural edits",
                unit="sample",
            )
        for sid, row in row_iter:
            base_hg = repair_bsp_tree_names(results[sid])
            mcmds = _multiset_commands_from_row(row)
            hg, meta = _apply_procedural_chain_for_sample(
                sid,
                mcmds,
                row.get("error"),
                base_hg,
                gt_accessgraph_by_id.get(sid, {}),
                bounds_by_id.get(sid) or {},
                increase_factor,
                procedural_steps,
                candidates_mode=candidates_mode,
                candidates_ged_timeout_ms=candidates_ged_timeout_ms,
                candidates_metric=candidates_metric,
            )
            _merge_row(sid, row, hg, meta)
        return hyper_out

    packs = [
        (
            sid,
            tuple(_multiset_commands_from_row(row)),
            row.get("error"),
            repair_bsp_tree_names(results[sid]),
            copy.deepcopy(gt_accessgraph_by_id.get(sid, {})),
            copy.deepcopy(bounds_by_id.get(sid) or {}),
            increase_factor,
            procedural_steps,
            candidates_mode,
            candidates_ged_timeout_ms,
            candidates_metric,
        )
        for sid, row in output.items()
    ]
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futs = [pool.submit(_procedural_worker_pack, p) for p in packs]
        with tqdm(
            total=n_out,
            desc="Procedural edits",
            unit="sample",
            disable=not show_progress,
        ) as pbar:
            for fut in as_completed(futs):
                sid, hg, meta = fut.result()
                _merge_row(sid, output[sid], hg, meta)
                pbar.update(1)

    return hyper_out


def process_sample(
    sample_id: str,
    hypergraph: dict,
    gt_accessgraph: dict,
    bounds: dict,
    *,
    include_accessgraphs: bool,
    two_step: bool,
    multiset_plan_max: int = 2,
    add_only: bool = False,
    remove_only: bool = False,
) -> dict[str, Any]:
    hypergraph = repair_bsp_tree_names(hypergraph)
    hg = Hypergraph(hypergraph)
    input_ag = hg.access_graph
    try:
        corrected_ag = hg.correct_accessgraph(bounds)
    except Exception as exc:
        logger.warning("correct_accessgraph failed for %s: %s", sample_id, exc)
        err: dict[str, Any] = {
            "command": None,
            "error": f"correct_accessgraph: {exc}",
        }
        if include_accessgraphs:
            err["input_accessgraph"] = input_ag
            err["corrected_accessgraph"] = None
            err["gt_accessgraph"] = gt_accessgraph
        return err

    cmd1, detail, cmd2, two_step_info = compare_accessgraphs(
        input_ag,
        corrected_ag,
        gt_accessgraph,
        two_step=two_step,
        max_plan_steps=multiset_plan_max,
        add_only=add_only,
        remove_only=remove_only,
    )
    chain = detail.get("multiset_command_chain") or []
    if two_step:
        logger.info("%s -> %s | step2=%s | chain=%s", sample_id, cmd1, cmd2, chain)
    else:
        logger.info("%s -> %s | chain=%s", sample_id, cmd1, chain)

    row: dict[str, Any] = {
        "command": cmd1,
        "detail": detail,
        "multiset_commands": chain,
    }
    if two_step:
        row["command_step2"] = cmd2
        row["needs_second_modification"] = two_step_info["needs_second_modification"]
        row["two_step"] = two_step_info
    if include_accessgraphs:
        row["input_accessgraph"] = input_ag
        row["corrected_accessgraph"] = corrected_ag
        row["gt_accessgraph"] = gt_accessgraph
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-hypergraphs",
        type=Path,
        default=REPO_ROOT
        / "inference_8700_on_eval_access_vllm_64"
        / "results_hypergraphs.json",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=REPO_ROOT / "dataset" / "rplan" / "eval_access" / "test",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Defaults to parametric_commands.json beside --results-hypergraphs",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--include-accessgraphs",
        action="store_true",
        help="Store full input/corrected/GT access graphs per id (large JSON).",
    )
    parser.add_argument(
        "--no-two-step",
        action="store_true",
        help="Disable second-step simulation (only command / step1).",
    )
    multiset_mode = parser.add_mutually_exclusive_group()
    multiset_mode.add_argument(
        "--add-only",
        action="store_true",
        help="Multiset plan: only 'add' steps (no 'remove' in the chain; surplus types stay).",
    )
    multiset_mode.add_argument(
        "--remove-only",
        action="store_true",
        help="Multiset plan: only 'remove' steps (no 'add' in the chain; missing types stay).",
    )
    parser.add_argument(
        "--apply-procedural-step1",
        action="store_true",
        help="Apply add/remove via procedural_edits (no PNGs; writes hypergraph JSON).",
    )
    parser.add_argument(
        "--procedural-steps",
        type=int,
        default=1,
        choices=(1, 2, 3, 4),
        help="Multiset plan depth and max procedural add/delete edits to apply (1–4). "
        "Uses detail['multiset_command_chain']. Steps 3–4 need two-step parametric mode "
        "for command_step2 fields in logs; avoid --no-two-step when using 3–4.",
    )
    parser.add_argument(
        "--skip-orientation-opt",
        action="store_true",
        help="With --apply-procedural-step1, skip BSP orientation search (optimize_rotation); "
        "only multiset-driven add/remove edits are applied.",
    )
    parser.add_argument(
        "--output-step1-hypergraphs",
        type=Path,
        default=None,
        help="Defaults to results_hypergraphs_step1_procedural.json beside --results-hypergraphs.",
    )
    parser.add_argument(
        "--increase-factor",
        type=float,
        default=2.0,
        help="Passed to procedural_edits.apply_command (increase/reduce scaling).",
    )
    parser.add_argument(
        "--candidates",
        action="store_true",
        help="With --apply-procedural-step1, choose among delete and add variants vs GT: "
        "one candidate per eligible leaf to delete (no cap) and up to 4 add variants; "
        "see --candidates-metric.",
    )
    parser.add_argument(
        "--candidates-metric",
        choices=("ged", "light"),
        default="ged",
        help="With --candidates: rank variants by ged (graph edit distance; may time out; "
        "uses --candidates-ged-timeout-ms per candidate) or light (L1 on room-type multiset + "
        "L1 on undirected (type,type) edge pairs after mesh-corrected access graph; fast).",
    )
    parser.add_argument(
        "--candidates-ged-timeout-ms",
        type=int,
        default=3000,
        metavar="MS",
        help="With --candidates-metric ged, timeout per GED call in ms (default 3000). Ignored for light.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        metavar="N",
        help="With --apply-procedural-step1, run procedural edits over samples in "
        "N parallel processes (spawn); also used for orientation unless "
        "--skip-orientation-opt. Default %(default)s. RGL mesh work is CPU-bound; "
        "memory grows with N. Use 1 for sequential.",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Warnings/errors only",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    two_step = not args.no_two_step
    if args.apply_procedural_step1 and args.procedural_steps >= 2 and not two_step:
        logger.warning(
            "--procedural-steps %d with --no-two-step: multiset chain is still planned up "
            "to that depth in detail, but command_step2 is omitted; legacy consumers may "
            "miss steps. Prefer leaving two-step mode on for multi-step procedural.",
            args.procedural_steps,
        )
    if args.apply_procedural_step1 and args.workers < 1:
        logger.error("--workers must be >= 1")
        return 2
    out_path = args.output_json or (args.results_hypergraphs.parent / "parametric_commands.json")

    with open(args.results_hypergraphs) as f:
        results: dict[str, dict] = json.load(f)

    ids = sorted(results.keys(), key=lambda x: (len(str(x)), str(x)))
    if args.max_samples is not None:
        ids = ids[: args.max_samples]

    if args.apply_procedural_step1:
        multiset_plan_max = args.procedural_steps
    else:
        multiset_plan_max = max(args.procedural_steps, 2 if two_step else 1)

    output: dict[str, Any] = {}
    gt_accessgraph_by_id: dict[str, dict] = {}
    bounds_by_id: dict[str, dict] = {}
    n_second = 0
    n_both_step1 = 0
    id_iter: Any = ids
    if not args.quiet:
        id_iter = tqdm(ids, desc="Parametric commands", unit="sample")
    for sid in id_iter:
        try:
            rec = load_test_record(args.test_dir, sid)
        except FileNotFoundError:
            logger.warning("Missing test file for sample %s", sid)
            output[sid] = {"command": None, "error": "missing_test_txt"}
            continue
        except json.JSONDecodeError as exc:
            logger.warning("Bad JSON in test file for %s: %s", sid, exc)
            output[sid] = {"command": None, "error": f"test_json: {exc}"}
            continue

        gt_accessgraph_by_id[sid] = rec["accessgraph"]
        bounds_by_id[sid] = rec.get("bounds") or {}

        try:
            row = process_sample(
                sid,
                results[sid],
                rec["accessgraph"],
                rec["bounds"],
                include_accessgraphs=args.include_accessgraphs,
                two_step=two_step,
                multiset_plan_max=multiset_plan_max,
                add_only=args.add_only,
                remove_only=args.remove_only,
            )
            output[sid] = row
            if two_step and row.get("needs_second_modification"):
                n_second += 1
            if two_step and row.get("detail", {}).get("step1_both_add_and_remove_needed"):
                n_both_step1 += 1
        except Exception as exc:
            logger.warning("Failed sample %s: %s", sid, exc)
            output[sid] = {"command": None, "error": str(exc)}

    if args.apply_procedural_step1:
        if args.skip_orientation_opt:
            for row in output.values():
                if isinstance(row, dict):
                    row.pop("orientation_opt_summary", None)

        hg_path = (
            args.output_step1_hypergraphs
            or (args.results_hypergraphs.parent / "results_hypergraphs_step1_procedural.json")
        )
        hyper_step1 = apply_procedural_step1_to_hypergraphs(
            output,
            results,
            gt_accessgraph_by_id,
            bounds_by_id,
            increase_factor=args.increase_factor,
            procedural_steps=args.procedural_steps,
            workers=args.workers,
            show_progress=not args.quiet,
            candidates_mode=args.candidates,
            candidates_ged_timeout_ms=args.candidates_ged_timeout_ms,
            candidates_metric=args.candidates_metric if args.candidates else "ged",
        )

        from scripts.procedural_edits import optimize_rotation

        n_hg = len(hyper_step1)
        n_orientation = 0
        n_orientation_skip = 0
        w = args.workers

        if args.skip_orientation_opt:
            logger.info("Skipping orientation optimization (--skip-orientation-opt).")
        elif w <= 1:
            orient_iter = hyper_step1.items()
            if not args.quiet:
                orient_iter = tqdm(
                    orient_iter,
                    total=n_hg,
                    desc="Orientation",
                    unit="sample",
                )
            for sid, hg in orient_iter:
                b = bounds_by_id.get(sid) or {}
                if not (isinstance(b, dict) and b.get("corners")):
                    n_orientation_skip += 1
                    continue
                try:
                    oh, omsg = optimize_rotation(hg, b)
                    is_ok, _, failed, _ = Hypergraph(oh).validate(quick=True)
                    if not is_ok:
                        logger.warning(
                            "orientation_opt invalid hypergraph %s: %s",
                            sid,
                            "; ".join(failed),
                        )
                        n_orientation_skip += 1
                        continue
                    hyper_step1[sid] = oh
                    row = output.get(sid)
                    if isinstance(row, dict):
                        row["orientation_opt_summary"] = omsg
                    n_orientation += 1
                except Exception as exc:
                    logger.warning("orientation_opt failed for %s: %s", sid, exc)
                    n_orientation_skip += 1
        else:
            opacks = [
                (sid, copy.deepcopy(hg), copy.deepcopy(bounds_by_id.get(sid) or {}))
                for sid, hg in hyper_step1.items()
            ]
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=w, mp_context=ctx) as pool:
                ofuts = [pool.submit(_orientation_worker_pack, p) for p in opacks]
                with tqdm(
                    total=n_hg,
                    desc="Orientation",
                    unit="sample",
                    disable=args.quiet,
                ) as opbar:
                    for fut in as_completed(ofuts):
                        sid, oh, omsg, problem = fut.result()
                        if omsg is not None:
                            hyper_step1[sid] = oh
                            row = output.get(sid)
                            if isinstance(row, dict):
                                row["orientation_opt_summary"] = omsg
                            n_orientation += 1
                        else:
                            n_orientation_skip += 1
                            if problem == "no_bounds":
                                pass
                            elif problem.startswith("invalid:"):
                                logger.warning(
                                    "orientation_opt invalid hypergraph %s: %s",
                                    sid,
                                    problem[len("invalid:") :],
                                )
                            else:
                                logger.warning(
                                    "orientation_opt failed for %s: %s", sid, problem
                                )
                        opbar.update(1)

        hg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(hg_path, "w") as f:
            json.dump(hyper_step1, f, indent=2, sort_keys=True)
        n_applied = sum(1 for r in output.values() if r.get("procedural_step1_applied"))
        n_applied2 = sum(1 for r in output.values() if r.get("procedural_step2_applied"))
        n_applied3 = sum(1 for r in output.values() if r.get("procedural_step3_applied"))
        n_applied4 = sum(1 for r in output.values() if r.get("procedural_step4_applied"))
        logger.info(
            "Wrote procedural hypergraphs for %d ids to %s "
            "(procedural steps applied: %d / %d / %d / %d; orientation-optimized: %d)",
            len(hyper_step1),
            hg_path,
            n_applied,
            n_applied2,
            n_applied3,
            n_applied4,
            n_orientation,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)

    logger.info("Wrote %d entries to %s", len(output), out_path)
    if two_step:
        logger.info(
            "Two-step: %d samples need a second modification (after simulating step1); "
            "%d samples had both missing and extra room types already at step1",
            n_second,
            n_both_step1,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
