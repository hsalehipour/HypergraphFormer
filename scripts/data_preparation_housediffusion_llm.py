#!/usr/bin/env python3
"""
Data preparation for House-Diffusion-LLM training.

Builds SFT chat data and per-id eval files where the assistant target is a
House-Diffusion JSON object (room_type / boxes / edges / ed_rm). The
user-side conditioning (the access graph) is **derived from the same
House-Diffusion JSON itself**, using the same door-mediated convention as
``benchmark/utils/convert_hypergraph_to_housegan_json.py``: two rooms are
connected iff an interior-door (category 17) entry pairs them via ``ed_rm``,
and a room is connected to ``"outside"`` iff a front-door (category 15) entry
pairs with that room. No external hypergraph dataset is required.

INPUT
-----
--hd-dir   House-Diffusion JSONs root with ``train/``, ``val/``, ``test/``
           subdirs (each containing ``<id>.json`` files).
           Default: ``dataset/rplan/dataset_housediffusion``

OUTPUT (under ``--output-dir``, default = ``--hd-dir``)
-------------------------------------------------------
``train.jsonl``           SFT chat data, one JSON object per line.
``validation.jsonl``      SFT chat data, one JSON object per line.
``<test-dirname>/<id>.txt``
                          Per-sample eval files containing
                          ``{"accessgraph": ..., "hypergraph": <hd_json_str>}``
                          (the existing eval pipeline reads exactly these
                          two keys, so it keeps working unchanged; the
                          ``hypergraph`` field is now a serialized HD JSON
                          rather than a BSP tree).

Usage
-----
    cd /raid/home/nikita/HypergraphFormer
    python scripts/data_preparation_housediffusion_llm.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Make ``utils`` importable when running this script directly from anywhere
# (e.g. ``python scripts/data_preparation_housediffusion_llm.py``).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.prompts import create_hd_training_sample  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("hd-llm-prep")


# ---------------------------------------------------------------------------
# 1. Access-graph extraction directly from a House-Diffusion JSON.
#
#    Mirrors the door-mediated convention used by the inverse converter
#    ``benchmark/utils/convert_hypergraph_to_housegan_json.py``:
#    - room-to-room edges come from interior doors (category 17): each
#      interior-door entry contributes edges between every pair of rooms it
#      touches via ``ed_rm`` (one door connects exactly two rooms in
#      practice, but the code is general).
#    - the synthetic ``"outside"`` reference comes from the front door
#      (category 15): every room paired with the front door in ``ed_rm``
#      gets an extra edge ``[room_id, "outside"]``.
#    Door entries themselves are excluded from the access graph's ``nodes``.
# ---------------------------------------------------------------------------

# Door category ids in the House-Diffusion / RPLAN class scheme.
FRONT_DOOR_CAT = 15
INTERIOR_DOOR_CAT = 17
DOOR_CATS = (FRONT_DOOR_CAT, INTERIOR_DOOR_CAT)

# Sentinel name for the synthetic "outside" reference appended to edges
# leaving the apartment via the front door. Mirrors the BSP convention:
# it shows up as the second endpoint of edges like ["3", "outside"] but is
# NOT listed in the access graph's ``nodes`` dict.
OUTSIDE_LABEL = "outside"

# House-Diffusion category id -> short access-graph room-type label.
# Matches the labels used by the existing BSP-derived access graphs
# (``bed`` / ``bath`` / ``living`` / ``kitchen`` / ``balcony`` / ``storage``).
HD_CATEGORY_TO_NAME: dict[int, str] = {
    1:  "living",
    2:  "kitchen",
    3:  "bed",
    4:  "bath",
    5:  "balcony",
    6:  "entrance",
    7:  "dining",
    8:  "study",
    10: "storage",
    16: "extra",   # rare in the dataset; safest catch-all label
}


def extract_access_graph_from_hd(hd_json: dict) -> dict:
    """Build a ``{nodes, edges}`` access graph from a House-Diffusion JSON.

    Door-mediated, mirroring ``convert_hypergraph_to_housegan_json.py``:

    - Rooms (categories 1-10, 16) are nodes; door entries (categories 15
      and 17) are NOT in ``nodes``.
    - Two rooms ``a, b`` get a single edge ``[a, b]`` iff there is an
      interior-door entry (category 17) at index ``k`` such that ``ed_rm``
      pairs ``k`` with both ``a`` and ``b`` (i.e. some rows of ``ed_rm``
      look like ``[k, a]`` and ``[k, b]``).
    - The front-door entry (category 15) at index ``f`` produces the
      synthetic outside reference: every room ``r`` that ``ed_rm`` pairs
      with ``f`` gets an extra edge ``[r, "outside"]``. The literal string
      ``"outside"`` is intentionally NOT a key in ``nodes``.
    - Output uses 1-indexed string ids in ``nodes`` and pairs of those ids
      (or the literal string ``"outside"``) in ``edges``.
    """
    rt: list[int] = hd_json.get("room_type", [])
    ed_rm: list[list[int]] = hd_json.get("ed_rm", [])

    # Original-index -> 1-indexed string id, for ROOM entries only.
    room_orig_indices = [i for i, c in enumerate(rt) if c not in DOOR_CATS]
    orig_to_new_id = {orig: str(new_idx + 1) for new_idx, orig in enumerate(room_orig_indices)}
    room_orig_set = set(room_orig_indices)

    nodes: dict[str, str] = {
        orig_to_new_id[orig]: HD_CATEGORY_TO_NAME.get(rt[orig], "extra")
        for orig in room_orig_indices
    }

    # For each interior-door entry, collect the rooms it touches.
    interior_door_rooms: dict[int, set[int]] = {
        i: set() for i, c in enumerate(rt) if c == INTERIOR_DOOR_CAT
    }
    front_door_indices: set[int] = {i for i, c in enumerate(rt) if c == FRONT_DOOR_CAT}
    front_door_rooms: set[int] = set()

    for row in ed_rm:
        for k in row:
            if k in interior_door_rooms:
                for r in row:
                    if r in room_orig_set:
                        interior_door_rooms[k].add(r)
            elif k in front_door_indices:
                for r in row:
                    if r in room_orig_set:
                        front_door_rooms.add(r)

    # Room-to-room edges: each interior door yields edges between every
    # pair of rooms it touches.
    seen: set = set()
    edges: list = []
    for door_rooms in interior_door_rooms.values():
        rooms = sorted(door_rooms)
        for ai in range(len(rooms)):
            for bi in range(ai + 1, len(rooms)):
                a = orig_to_new_id[rooms[ai]]
                b = orig_to_new_id[rooms[bi]]
                key = tuple(sorted([a, b], key=lambda x: int(x)))
                if key not in seen:
                    seen.add(key)
                    edges.append([key[0], key[1]])

    # Outside edges: every room touched by the front-door entry.
    for r_orig in sorted(front_door_rooms):
        edges.append([orig_to_new_id[r_orig], OUTSIDE_LABEL])

    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# 2. Pipeline: stream every HD JSON, derive its access graph, write outputs.
#    Prompt templates (system message, format rules, etc.) live in
#    ``utils.prompts`` so the evaluator can reuse the exact training prompt.
# ---------------------------------------------------------------------------
SPLIT_MAP = [
    # (HD subdir, output kind)
    ("train", "jsonl"),
    ("val",   "jsonl"),
    ("test",  "perid"),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build SFT data with House-Diffusion JSON targets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--hd-dir", type=Path,
        default=Path("dataset/rplan/dataset_housediffusion"),
        help="House-Diffusion JSONs root (with train/, val/, test/ subdirs).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output root (defaults to --hd-dir).",
    )
    parser.add_argument(
        "--test-dirname", type=str, default="test",
        help="Subdir under --output-dir for per-id eval .txt files (default: test).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap number of HD samples processed per split (debug).",
    )
    args = parser.parse_args()

    out_root: Path = args.output_dir or args.hd_dir
    out_root.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("HD JSONs:        %s", args.hd_dir)
    logger.info("Output root:     %s", out_root)
    logger.info("Test sub-dir:    %s", args.test_dirname)
    if args.limit:
        logger.info("Limit per split: %d", args.limit)
    logger.info("=" * 70)

    summary: list[tuple[str, int, int, str]] = []

    for hd_split, kind in SPLIT_MAP:
        hd_dir = args.hd_dir / hd_split

        if not hd_dir.is_dir():
            logger.warning("Skipping %s: HD subdir missing (%s)", hd_split, hd_dir)
            continue

        logger.info("")
        logger.info("Split %-5s : HD=%s", hd_split, hd_dir)

        hd_files = sorted(hd_dir.glob("*.json"))
        if args.limit:
            hd_files = hd_files[: args.limit]

        n_total = len(hd_files)
        n_paired = 0
        n_empty_ag = 0
        n_bad = 0

        if kind == "jsonl":
            out_path = out_root / ("train.jsonl" if hd_split == "train" else "validation.jsonl")
            with out_path.open("w") as fout:
                for fp in hd_files:
                    try:
                        with fp.open() as f:
                            hd = json.load(f)
                    except (OSError, json.JSONDecodeError):
                        n_bad += 1
                        continue
                    ag = extract_access_graph_from_hd(hd)
                    if not ag.get("nodes"):
                        n_empty_ag += 1
                        continue
                    sample = create_hd_training_sample(ag, hd)
                    fout.write(json.dumps(sample) + "\n")
                    n_paired += 1
                    if n_paired % 5000 == 0:
                        logger.info("    ... wrote %d / %d", n_paired, n_total)
            logger.info(
                "  -> %s   (%d / %d written; %d empty-ag; %d unreadable)",
                out_path, n_paired, n_total, n_empty_ag, n_bad,
            )
            summary.append((hd_split, n_total, n_paired, str(out_path)))

        else:  # per-id .txt eval files
            test_dir = out_root / args.test_dirname
            test_dir.mkdir(parents=True, exist_ok=True)
            if test_dir.resolve() == hd_dir.resolve():
                logger.warning(
                    "  NOTE: writing .txt next to existing .json files in %s "
                    "(different extensions, no overwrite).", test_dir,
                )
            for fp in hd_files:
                sid = fp.stem
                try:
                    with fp.open() as f:
                        hd = json.load(f)
                except (OSError, json.JSONDecodeError):
                    n_bad += 1
                    continue
                ag = extract_access_graph_from_hd(hd)
                if not ag.get("nodes"):
                    n_empty_ag += 1
                    continue
                rec = {
                    "accessgraph": ag,
                    "hypergraph": json.dumps(hd),
                }
                with (test_dir / f"{sid}.txt").open("w") as f:
                    json.dump(rec, f, indent=2)
                n_paired += 1
                if n_paired % 2000 == 0:
                    logger.info("    ... wrote %d / %d", n_paired, n_total)
            logger.info(
                "  -> %s   (%d / %d written; %d empty-ag; %d unreadable)",
                test_dir, n_paired, n_total, n_empty_ag, n_bad,
            )
            summary.append((hd_split, n_total, n_paired, str(test_dir)))

    logger.info("")
    logger.info("=" * 70)
    logger.info("Summary:")
    for split, total, paired, path in summary:
        logger.info("  %-5s : %d / %d written -> %s", split, paired, total, path)
    logger.info("=" * 70)
    logger.info("Output root: %s", out_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
