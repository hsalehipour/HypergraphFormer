#!/usr/bin/env python3
"""
Data preparation for hypergraph model training and evaluation.

Modes:
    train        -- Convert pre-split hypergraph JSON to instruction JSONL
    eval         -- Export per-sample .txt eval files from a JSON split
    room-lists   -- Convert hypergraph JSON splits to pre-computed room-list
                    JSON files (requires RGL)

Usage:
    # Convert to instruction JSONL (SFT)
    python scripts/data_preparation.py --mode train \\
        --dataset-dir dataset/hypergraph

    # Convert to instruction JSONL (RL with footprint)
    python scripts/data_preparation.py --mode train \\
        --dataset-dir dataset/hypergraph --footprint

    # Export eval-format .txt files
    python scripts/data_preparation.py --mode eval \\
        --input-file dataset/hypergraph/test.json \\
        --output-dir dataset/hypergraph/test

    # Generate room-list JSONs from a hypergraph JSON file
    python scripts/data_preparation.py --mode room-lists \\
        --input-file dataset/hypergraph/train.json \\
        --output-dir dataset/hypergraph/rl/train_room_lists
"""

from __future__ import annotations

import copy
import json
import logging
import sys
from pathlib import Path

from utils.hypergraph import Hypergraph

logger = logging.getLogger(__name__)


# ===================================================================
# Shared helper
# ===================================================================

def load_json_split(file_path, *, footprint=False):
    """Load a hypergraph dataset split from a JSON file.

    Each item is expected to have a ``split`` field containing the
    hypergraph (as a JSON string or dict).  The access graph is
    extracted automatically via :class:`Hypergraph`.

    Returns a list of tuples::

        (item_id, accessgraph, hypergraph)

    or, when *footprint* is True::

        (item_id, accessgraph, hypergraph, bounds)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return []

    with open(file_path, "r") as f:
        data = json.load(f)

    samples = []
    skipped = 0

    for i, item in enumerate(data):
        try:
            hypergraph_str = item.get("split", "")
            if not hypergraph_str:
                logger.warning(f"Skipping item {i}: missing 'split' field")
                skipped += 1
                continue

            hypergraph = (
                json.loads(hypergraph_str)
                if isinstance(hypergraph_str, str)
                else hypergraph_str
            )

            accessgraph = Hypergraph(hypergraph).access_graph

            if not accessgraph or not accessgraph.get("nodes"):
                logger.warning(f"Skipping item {i}: could not extract access graph")
                skipped += 1
                continue

            item_id = item.get("id", f"item_{i}")

            if footprint:
                bounds_str = item.get("bounds", "{}")
                bounds = (
                    json.loads(bounds_str)
                    if isinstance(bounds_str, str)
                    else bounds_str
                )
                samples.append((item_id, accessgraph, hypergraph, bounds))
            else:
                samples.append((item_id, accessgraph, hypergraph))

        except Exception as e:
            logger.warning(f"Skipping item {i}: {e}")
            skipped += 1

    logger.info(
        f"Loaded {len(samples)} samples from {file_path.name} ({skipped} skipped)"
    )
    return samples


# ===================================================================
# convert_to_jsonl  (--mode train)
# ===================================================================

def convert_to_jsonl(args) -> int:
    """Convert pre-split hypergraph JSON files to instruction JSONL."""
    from utils.data_loading import save_training_data

    dataset_dir = Path(args.dataset_dir)
    footprint = getattr(args, "footprint", False)
    output_dir = dataset_dir / ("rl" if footprint else "sft")

    logger.info("=" * 80)
    logger.info("Hypergraph JSON -> Instruction JSONL")
    logger.info("=" * 80)
    logger.info(f"Dataset directory: {dataset_dir}")
    logger.info(f"Output directory:  {output_dir}")
    if footprint:
        logger.info("Footprint mode:    ON (includes bounds in prompts)")
    logger.info("=" * 80)

    splits = ["train", "validation", "test"]
    total = 0

    for split in splits:
        logger.info(f"\nProcessing {split} split...")
        samples = load_json_split(
            dataset_dir / f"{split}.json", footprint=footprint
        )

        if not samples:
            logger.warning(f"No samples found for {split}")
            continue

        save_training_data(samples, output_dir / f"{split}.jsonl", split)
        total += len(samples)

    logger.info(f"\nDone! {total} samples across {len(splits)} splits.")
    for split in splits:
        p = output_dir / f"{split}.jsonl"
        if p.exists():
            logger.info(f"  {p}")
    return 0


# ===================================================================
# export_eval_format  (--mode eval)
# ===================================================================

def export_eval_format(args) -> int:
    """Export per-sample .txt eval files from a JSON split."""
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir or "dataset/hypergraph/test")

    logger.info("=" * 80)
    logger.info("Export Eval-Format .txt Files")
    logger.info("=" * 80)
    logger.info(f"Input file:       {input_file}")
    logger.info(f"Output directory:  {output_dir}")
    logger.info("=" * 80)

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r") as f:
        raw_data = json.load(f)

    successful = 0
    skipped = 0

    for i, item in enumerate(raw_data):
        try:
            item_id = item.get("id", f"item_{i}")

            hypergraph_str = item.get("split", "")
            if not hypergraph_str:
                logger.warning(f"Skipping {item_id}: missing 'split' field")
                skipped += 1
                continue

            hypergraph = (
                json.loads(hypergraph_str)
                if isinstance(hypergraph_str, str)
                else hypergraph_str
            )

            accessgraph = Hypergraph(hypergraph).access_graph

            if not accessgraph or not accessgraph.get("nodes"):
                logger.warning(f"Skipping {item_id}: could not extract access graph")
                skipped += 1
                continue

            output_data = {
                "accessgraph": accessgraph,
                "hypergraph": (
                    json.dumps(hypergraph)
                    if isinstance(hypergraph, dict)
                    else hypergraph
                ),
                "bounds": item.get("bounds", ""),
            }

            with open(output_dir / f"{item_id}.txt", "w") as f:
                json.dump(output_data, f, indent=2)

            successful += 1

            if (i + 1) % 50 == 0:
                logger.info(f"  Processed {i + 1}/{len(raw_data)} samples...")

        except Exception as e:
            logger.warning(f"Error processing {item.get('id', f'item_{i}')}: {e}")
            skipped += 1

    logger.info(f"\nDone! {successful} files written, {skipped} skipped.")
    logger.info(f"Output directory: {output_dir}")
    return 0


# ===================================================================
# generate_room_lists  (--mode room-lists)
# ===================================================================

def generate_room_lists(args) -> int:
    """Convert a hypergraph JSON file into per-sample room-list JSONs via RGL.

    Formerly the standalone ``delta_metric.py`` CLI.
    """
    from utils.metrics import produce_room_list

    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir or input_file.with_suffix("") / "room_lists")

    logger.info("=" * 80)
    logger.info("Hypergraph JSON -> Room-List JSONs (via RGL)")
    logger.info("=" * 80)
    logger.info(f"Input file:        {input_file}")
    logger.info(f"Output directory:  {output_dir}")
    logger.info("=" * 80)

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_file) as f:
        data = json.load(f)

    logger.info(f"Processing {len(data)} entries ...")
    results = []
    for idx, entry in enumerate(data):
        apt_id = entry.get("id", f"idx_{idx}")
        try:
            hg_raw = entry.get("split", entry.get("hypergraph", "{}"))
            hg = json.loads(hg_raw) if isinstance(hg_raw, str) else hg_raw
            hg = copy.deepcopy(hg)

            bounds_raw = entry.get("bounds", "{}")
            bounds = json.loads(bounds_raw) if isinstance(bounds_raw, str) else bounds_raw
            bounds = copy.deepcopy(bounds)

            mesh = Hypergraph(hg).to_mesh(bounds=bounds)
            room_list = produce_room_list(mesh)
            output_data = {"id": apt_id, "room_list": room_list}
        except Exception as e:
            logger.warning(f"  Error processing {apt_id}: {e}")
            output_data = {"id": apt_id, "room_list": None, "error": str(e)}

        with open(output_dir / f"{apt_id}.json", "w") as f:
            json.dump(output_data, f, indent=2)
        results.append(output_data)

    combined = output_dir / "all_room_lists.json"
    with open(combined, "w") as f:
        json.dump(results, f, indent=2)

    ok = sum(1 for r in results if r.get("room_list") is not None)
    logger.info(f"\nDone! {ok}/{len(results)} succeeded -> {combined}")
    return 0


# ===================================================================
# CLI
# ===================================================================

MODE_MAP = {
    "train": convert_to_jsonl,
    "eval": export_eval_format,
    "room-lists": generate_room_lists,
}


def cli():
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Hypergraph data preparation")
    parser.add_argument("--mode", choices=list(MODE_MAP), required=True)

    # Shared
    parser.add_argument("--dataset-dir", type=str, default="dataset/hypergraph")
    parser.add_argument("--seed", type=int, default=42)

    # train mode
    parser.add_argument(
        "--footprint",
        action="store_true",
        help="Include apartment footprint (bounds) in prompts; output to rl/ subdir",
    )

    # eval mode
    parser.add_argument(
        "--input-file",
        type=str,
        default="dataset/hypergraph/test.json",
        help="Path to JSON file to export (eval mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (eval: .txt files; room-lists: room-list JSONs)",
    )

    args = parser.parse_args()
    return MODE_MAP[args.mode](args)


if __name__ == "__main__":
    sys.exit(cli())
