#!/usr/bin/env python3
"""
Extract input access graphs from the original RPLAN data for evaluation.

The dataset_access format (produced by rplan/build_dataset_access.py) stores
door-based room adjacency information extracted from RPLAN PNGs.  This script
converts those records into per-sample .txt eval files expected by
scripts/evaluator.py.

Because dataset_access contains the original RPLAN access graphs (not the
hypergraph BSP split tree), the output .txt files have an empty ``hypergraph``
field.

Usage:
    # Convert test split
    python scripts/data_preparation_access.py \\
        --input-file dataset/rplan/dataset_access/data_test.json \\
        --output-dir training_data/extracted_access/test

    # Convert all splits
    python scripts/data_preparation_access.py \\
        --input-dir dataset/rplan/dataset_access \\
        --output-dir training_data/extracted_access
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

CATEGORY_TO_MERGEID = {
    0: "living",
    1: "bed",
    2: "kitchen",
    3: "bath",
    4: "balcony",
    5: "storage",
}


def convert_sample(item: dict) -> dict | None:
    """Convert one dataset_access record to the evaluator .txt format.

    Returns the output dict, or None if the sample should be skipped.
    """
    rooms = item.get("rooms", [])
    if not rooms:
        return None

    adj_list = item.get("access_adjacencies")
    if adj_list is None:
        return None

    nodes = {}
    for room in rooms:
        rid = str(room["id"] + 1)
        category = room.get("category", -1)
        room_type = CATEGORY_TO_MERGEID.get(category, f"unknown_{category}")
        nodes[rid] = room_type

    edges = []
    for pair in adj_list:
        a, b = str(pair[0] + 1), str(pair[1] + 1)
        if a in nodes and b in nodes:
            edges.append([a, b])

    if not nodes:
        return None

    accessgraph = {"nodes": nodes, "edges": edges}

    return {
        "accessgraph": accessgraph,
        "hypergraph": "",
        "bounds": "",
    }


def convert_file(input_file: Path, output_dir: Path) -> tuple[int, int]:
    """Convert a single JSON file. Returns (successful, skipped) counts."""
    with open(input_file, "r") as f:
        data = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    successful = 0
    skipped = 0

    for i, item in enumerate(data):
        item_id = str(item.get("name", f"item_{i}"))

        result = convert_sample(item)
        if result is None:
            logger.warning("Skipping %s: missing rooms or adjacencies", item_id)
            skipped += 1
            continue

        with open(output_dir / f"{item_id}.txt", "w") as f:
            json.dump(result, f, indent=2)

        successful += 1

        if (i + 1) % 500 == 0:
            logger.info("  Processed %d/%d samples...", i + 1, len(data))

    return successful, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Prepare OOD eval files from dataset_access JSONs",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input-file", type=str,
        help="Single JSON file to convert (e.g. data_test.json)",
    )
    group.add_argument(
        "--input-dir", type=str,
        help="Directory containing data_*.json files (converts all splits)",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, required=True,
        help="Output directory for .txt files",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    output_dir = Path(args.output_dir)
    total_ok = 0
    total_skip = 0

    if args.input_file:
        input_file = Path(args.input_file)
        if not input_file.exists():
            logger.error("Input file not found: %s", input_file)
            return 1

        logger.info("Converting %s -> %s", input_file, output_dir)
        ok, skip = convert_file(input_file, output_dir)
        total_ok += ok
        total_skip += skip

    else:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            logger.error("Input directory not found: %s", input_dir)
            return 1

        split_map = {"data_train": "train", "data_val": "val", "data_test": "test"}

        for json_file in sorted(input_dir.glob("data_*.json")):
            split_name = split_map.get(json_file.stem, json_file.stem)
            split_output = output_dir / split_name
            logger.info("Converting %s -> %s", json_file.name, split_output)
            ok, skip = convert_file(json_file, split_output)
            total_ok += ok
            total_skip += skip

    logger.info("Done! %d files written, %d skipped.", total_ok, total_skip)
    logger.info("Output: %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
