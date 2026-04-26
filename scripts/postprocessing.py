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
room type is the **first** in sorted order among GT-neighbor types of *X* that also appear in
the prediction.  If that fails, falls back to splitting the
largest non-*X* leaf in the pred.  Runs on the **input** hypergraph from ``--results-hypergraphs``.
For ``add``, the new leaf's ``mergeid`` is the most common raw GT label matching the multiset
room type (via the same canonicalization as procedural_edits), when present, so strict multiset
scoring aligns with eval access-graph spelling (e.g. ``bed`` vs ``bedroom``). Writes only a JSON
dict ``sample_id -> hypergraph`` — no images or extra folders. Procedural edits are
**CPU-heavy** (RGL meshing via .NET); use ``--workers N`` to process samples in parallel
(``spawn`` processes; default ``N=16``).

**Modes** (``--pp-mode``; default: ``parametric`` — behavior above).

* ``orientation`` — For each id, a **bsp** 3×4 search (mirrors × CCW°) as in
  ``explore_rotation --mode bsp``; the best cell is the lowest room-shape **δ** (not APE
  in this short path). Output JSON matches ``--results-hypergraphs`` shape: each id maps to
  a **hypergraph object only** (no ``mesh_bounds`` / ``orientation`` wrappers).

* ``optimization`` — Runs :func:`scripts.run_parametric.optimize_hypergraph` per id with
  **vanilla GD** and :class:`utils_parametric.OptimConfig` defaults (objective **delta**).
  See ``--output-optimization*`` and ``--no-optimization-meta``.
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
    )
    return sid, hg, meta


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
) -> dict[str, dict]:
    """
    For each sample, run up to *procedural_steps* (1–4) ``procedural_edits.apply_command``
    calls from ``row[\"multiset_commands\"]`` (greedy multiset plan in ``detail``).

    *bounds_by_id* is passed to ``apply_command`` for ``add`` when corners exist (for mesh
    adjacency in procedural splits).

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


# --- Optional batch modes: orientation (bsp) & parametric optimization (GD) ---

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
    gt_hg: dict,
) -> dict[str, Any]:
    """
    ``explore_rotation`` **bsp** only: reorient the pred BSP, canonical ``bounds``, pick
    the (mirror,°) with lowest room-shape **δ** vs GT. Returns ``hypergraph``, ``mesh_bounds``, etc.
    """
    from scripts.procedural_edits import reorient_hypergraph_bsp
    from utils.metrics import compute_room_similarity_score, produce_room_list

    if not (bounds and isinstance(bounds, dict) and bounds.get("corners")):
        return {
            "error": "missing bounds.corners",
            "hypergraph": copy.deepcopy(repair_bsp_tree_names(base_hg)),
        }
    if not (gt_hg and isinstance(gt_hg, dict) and gt_hg.get("name")):
        return {
            "error": "missing gt hypergraph",
            "hypergraph": copy.deepcopy(repair_bsp_tree_names(base_hg)),
        }

    base = repair_bsp_tree_names(copy.deepcopy(base_hg))
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
                pred_mesh = Hypergraph(pred).to_mesh(bounds=mesh_bounds)
                gt_mesh_v = Hypergraph(gt_hg).to_mesh(bounds=mesh_bounds)
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "reflect": reflect_code,
                        "reflect_label": reflect_label,
                        "deg": deg,
                        "delta": None,
                        "error": str(exc),
                    }
                )
                continue
            pred_room = produce_room_list(pred_mesh)
            gt_room_v = produce_room_list(gt_mesh_v)
            dlt = float(compute_room_similarity_score(pred_room, gt_room_v))
            row: dict[str, Any] = {
                "reflect": reflect_code,
                "reflect_label": reflect_label,
                "deg": deg,
                "delta": dlt,
                "out_hypergraph": copy.deepcopy(pred),
                "mesh_bounds": copy.deepcopy(mesh_bounds),
            }
            rows.append(row)

    valid = [
        r
        for r in rows
        if "error" not in r
        and r.get("delta") is not None
        and math.isfinite(float(r["delta"]))
    ]
    if not valid:
        return {
            "error": "no valid orientation cells",
            "hypergraph": copy.deepcopy(base),
            "meta": {"style": "bsp", "rows": rows, "n_tried": len(rows)},
        }

    best = min(valid, key=lambda r: float(r["delta"]))

    return {
        "hypergraph": copy.deepcopy(best["out_hypergraph"]),
        "mesh_bounds": copy.deepcopy(best.get("mesh_bounds") or {}),
        "dataset_bounds": copy.deepcopy(bounds),
        "meta": {
            "style": "bsp",
            "metric": "delta",
            "reflect": best.get("reflect"),
            "reflect_label": best.get("reflect_label"),
            "deg": int(best.get("deg", 0)),
            "delta": best.get("delta"),
            "n_variants_tried": len(rows),
        },
    }


def _orientation_pack(
    p: tuple[str, dict, dict, dict],
) -> tuple[str, dict[str, Any]]:
    sid, base_hg, b2, gth = p
    return sid, orient_one_sample(sid, base_hg, b2, gth)


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
            b2, gth, _ag = load_eval_access_gt(test_dir, sid)
            rec = orient_one_sample(sid, _hypergraph_from_results_value(results[sid]), b2, gth)
            _record(sid, rec)
    else:
        packs: list[tuple[str, dict, dict, dict]] = []
        for sid in ids:
            b2, gth, _ag = load_eval_access_gt(test_dir, sid)
            packs.append((sid, _hypergraph_from_results_value(results[sid]), b2, gth))
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


def _optim_worker_pack(
    t: tuple[str, str, str, str, str],
) -> tuple[str, dict, dict[str, Any] | None, str | None]:
    import json
    from pathlib import Path

    from scripts.run_parametric import optimize_hypergraph
    from utils_parametric import OptimConfig

    sid, grpo_path_s, cfg_json, art_base_s, _td = t
    grpo = Path(grpo_path_s)
    with open(grpo, encoding="utf-8") as f:
        allh: dict = json.load(f)
    if sid not in allh:
        return sid, repair_bsp_tree_names({}), None, f"missing id {sid}"
    cfg = OptimConfig(**json.loads(cfg_json))
    art = Path(art_base_s) / sid
    try:
        best_hg, _bm, _om, _gt, mo, mp, _bg, _ec, _ag = optimize_hypergraph(
            sid,
            cfg,
            grpo_file_path=grpo,
            artifacts_dir=art,
        )
        msummary = {"metrics_original": mo, "metrics_optimized": mp}
        return sid, best_hg, msummary, None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Optimization failed for %s: %s", sid, exc)
        return sid, repair_bsp_tree_names(allh[sid]), None, f"{type(exc).__name__}: {exc}"


def run_optimization_mode(
    results: dict[str, dict],
    test_dir: Path,
    grpo: Path,
    *,
    out_hg: Path,
    out_meta: Path | None,
    max_samples: int | None,
    workers: int,
    show_progress: bool,
) -> None:
    import json
    from dataclasses import asdict

    from utils_parametric import OptimConfig

    try:
        td_rel = str(test_dir.resolve().relative_to(REPO_ROOT))
    except ValueError:
        td_rel = str(test_dir)
    wcfg = asdict(OptimConfig(optimizer="gd"))
    wcfg["test_dir"] = td_rel
    wcfg["grpo_file"] = str(grpo.resolve())
    wcfg["artifacts_dir"] = None
    for k, v in list(wcfg.items()):
        if v is not None and isinstance(v, Path):
            wcfg[k] = str(v)
    cfg_json = json.dumps(wcfg, default=str)

    ids = sorted(results.keys(), key=lambda x: (len(str(x)), str(x)))
    if max_samples is not None:
        ids = ids[: int(max_samples)]
    n_out = len(ids)
    art_base = (grpo.parent / "postprocessing_optimization_artifacts").resolve()
    out_h: dict[str, dict] = {}
    meta: dict[str, Any] = {}

    if workers <= 1:
        it = ids
        if show_progress:
            it = tqdm(ids, desc="Optimization", total=n_out, unit="sample")
        for sid in it:
            sid, hg, m, err = _optim_worker_pack(
                (sid, str(grpo.resolve()), cfg_json, str(art_base), td_rel)
            )
            out_h[sid] = hg
            if m:
                meta[sid] = m
            if err:
                meta[sid] = {**(meta.get(sid) or {}), "error": err}
    else:
        packs = [
            (sid, str(grpo.resolve()), cfg_json, str(art_base), td_rel) for sid in ids
        ]
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futs = [pool.submit(_optim_worker_pack, p) for p in packs]
            with tqdm(
                total=n_out, desc="Optimization", unit="sample", disable=not show_progress
            ) as pbar:
                for fut in as_completed(futs):
                    sid, hg, m, err = fut.result()
                    out_h[sid] = hg
                    if m:
                        meta[sid] = m
                    if err:
                        meta[sid] = {**(meta.get(sid) or {}), "error": err}
                    pbar.update(1)

    out_hg.parent.mkdir(parents=True, exist_ok=True)
    with open(out_hg, "w", encoding="utf-8") as f:
        json.dump(out_h, f, indent=2, sort_keys=True)
    logger.info("Wrote optimized hypergraphs (%d ids) to %s", len(out_h), out_hg)
    if out_meta is not None:
        out_meta.parent.mkdir(parents=True, exist_ok=True)
        with open(out_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        logger.info("Wrote optimization meta to %s", out_meta)


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
        "--workers",
        type=int,
        default=16,
        metavar="N",
        help="Procedural, orientation, or optimization: parallel processes (spawn). "
        "Default %(default)s. RGL is CPU-bound; use 1 for sequential.",
    )
    parser.add_argument(
        "--pp-mode",
        type=str,
        default="parametric",
        choices=("parametric", "orientation", "optimization"),
        help="parametric: multiset commands + optional procedural (default). "
        "orientation: bsp 3×4 search, min δ. "
        "optimization: gradient descent with OptimConfig defaults (objective delta).",
    )
    ogrp = parser.add_argument_group("orientation mode (--pp-mode orientation)")
    ogrp.add_argument(
        "--output-orientation",
        type=Path,
        default=None,
        help="Default: results_hypergraphs_orientation.json beside --results-hypergraphs. "
        "Each id -> hypergraph tree only (same format as the input file).",
    )
    popt = parser.add_argument_group("optimization mode (--pp-mode optimization)")
    popt.add_argument(
        "--output-optimization",
        type=Path,
        default=None,
        help="Default: results_hypergraphs_optimized.json beside --results-hypergraphs.",
    )
    popt.add_argument(
        "--output-optimization-meta",
        type=Path,
        default=None,
        help="Per-id metrics and errors. Default: <output-optimization base>_meta.json. Pass a path; omit = default.",
    )
    popt.add_argument(
        "--no-optimization-meta",
        action="store_true",
        help="With optimization mode, do not write the metrics sidecar JSON.",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Warnings/errors only",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    with open(args.results_hypergraphs, encoding="utf-8") as f:
        results: dict[str, dict] = json.load(f)

    if args.pp_mode == "orientation":
        if args.workers < 1:
            logger.error("--workers must be >= 1")
            return 2
        out_o = args.output_orientation or (
            args.results_hypergraphs.parent / "results_hypergraphs_orientation.json"
        )
        run_orientation_mode(
            results,
            args.test_dir,
            out_path=out_o,
            max_samples=args.max_samples,
            workers=int(args.workers),
            show_progress=not args.quiet,
        )
        return 0

    if args.pp_mode == "optimization":
        if args.workers < 1:
            logger.error("--workers must be >= 1")
            return 2
        out_hg = args.output_optimization or (
            args.results_hypergraphs.parent / "results_hypergraphs_optimized.json"
        )
        if args.no_optimization_meta:
            meta_path = None
        else:
            meta_path = args.output_optimization_meta
            if meta_path is None and out_hg is not None:
                meta_path = out_hg.parent / f"{out_hg.stem}_meta.json"
        run_optimization_mode(
            results,
            args.test_dir,
            args.results_hypergraphs,
            out_hg=out_hg,
            out_meta=meta_path,
            max_samples=args.max_samples,
            workers=int(args.workers),
            show_progress=not args.quiet,
        )
        return 0

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
        )

        hg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(hg_path, "w") as f:
            json.dump(hyper_step1, f, indent=2, sort_keys=True)
        n_applied = sum(1 for r in output.values() if r.get("procedural_step1_applied"))
        n_applied2 = sum(1 for r in output.values() if r.get("procedural_step2_applied"))
        n_applied3 = sum(1 for r in output.values() if r.get("procedural_step3_applied"))
        n_applied4 = sum(1 for r in output.values() if r.get("procedural_step4_applied"))
        logger.info(
            "Wrote procedural hypergraphs for %d ids to %s "
            "(procedural steps applied: %d / %d / %d / %d)",
            len(hyper_step1),
            hg_path,
            n_applied,
            n_applied2,
            n_applied3,
            n_applied4,
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
