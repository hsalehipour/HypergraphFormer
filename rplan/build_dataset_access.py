#!/usr/bin/env python3
"""
Build dataset_access: augment DiffPlanner JSONs with door-based access adjacencies.

For each sample in dataset_diffplanner, loads the matching RPLAN PNG from
dataset_original, extracts interior-door (label 17) and front-door (label 15)
pixel clusters, determines which rooms each door connects, and writes
a new JSON with the original fields plus an ``access_adjacencies`` field.

When a door cluster touches 3+ rooms (e.g. at a junction), all pairwise
connections are generated.  Window pixels (label 16) are additionally
used to detect balcony access, since balconies in RPLAN often lack
labelled doors.

Usage:
    python rplan/build_dataset_access.py
    python rplan/build_dataset_access.py --splits train
    python rplan/build_dataset_access.py --dilation 4 --workers 8
"""

import argparse
import json
import logging
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

logger = logging.getLogger(__name__)

INTERIOR_DOOR_LABEL = 17
FRONT_DOOR_LABEL = 15
WINDOW_LABEL = 16
BALCONY_CATEGORY = 4


def _pairs_from_instances(instance_ids: np.ndarray) -> list[tuple[int, int]]:
    """Return all pairwise (0-indexed room) combos from 1-indexed instance IDs."""
    from itertools import combinations
    rooms = sorted(set(int(v) - 1 for v in instance_ids))
    return [(min(a, b), max(a, b)) for a, b in combinations(rooms, 2)]


def extract_access_adjacencies(
    png_path: str,
    n_rooms: int,
    dilation_iterations: int = 3,
    room_categories: list[int] | None = None,
) -> list[list[int]]:
    """Return door-based room-room pairs (0-indexed) from an RPLAN PNG.

    Parameters
    ----------
    png_path : str
        Path to the 4-channel RPLAN PNG.
    n_rooms : int
        Number of rooms in the sample (used for sanity checks).
    dilation_iterations : int
        How many pixels to dilate each door cluster before checking for
        nearby room instances.  3 covers typical wall thickness (~2-4 px).
    room_categories : list[int] | None
        DiffPlanner category for each room (0-indexed).  When provided,
        window pixels (label 16) are treated as doors for balconies
        (category 4).
    """
    img = np.asarray(Image.open(png_path))
    ch1 = img[:, :, 1]  # semantic labels
    ch2 = img[:, :, 2]  # room instance IDs (1-indexed)

    access_pairs: set[tuple[int, int]] = set()

    # --- Door-based connections (interior + front doors) ---
    door_mask = (ch1 == INTERIOR_DOOR_LABEL) | (ch1 == FRONT_DOOR_LABEL)
    if door_mask.any():
        door_labels, n_clusters = ndimage.label(door_mask)
        for cluster_id in range(1, n_clusters + 1):
            cluster_mask = door_labels == cluster_id
            dilated = ndimage.binary_dilation(
                cluster_mask, iterations=dilation_iterations,
            )
            nearby_instances = ch2[dilated & (ch2 > 0)]
            if nearby_instances.size == 0:
                continue
            vals = np.unique(nearby_instances)
            if len(vals) >= 2:
                for pair in _pairs_from_instances(vals):
                    access_pairs.add(pair)

    # --- Window-based connections for balconies ---
    if room_categories is not None:
        window_mask = ch1 == WINDOW_LABEL
        if window_mask.any():
            win_labels, n_win = ndimage.label(window_mask)
            balcony_instances = {
                i + 1
                for i, cat in enumerate(room_categories)
                if cat == BALCONY_CATEGORY
            }
            for cluster_id in range(1, n_win + 1):
                cluster_mask = win_labels == cluster_id
                dilated = ndimage.binary_dilation(
                    cluster_mask, iterations=dilation_iterations,
                )
                nearby_instances = ch2[dilated & (ch2 > 0)]
                if nearby_instances.size == 0:
                    continue
                vals = np.unique(nearby_instances)
                has_balcony = bool(balcony_instances & set(vals.tolist()))
                if len(vals) >= 2 and has_balcony:
                    for pair in _pairs_from_instances(vals):
                        if any(
                            (inst + 1) in balcony_instances
                            for inst in pair
                        ):
                            access_pairs.add(pair)

    return sorted([list(p) for p in access_pairs])


def _process_sample(args):
    """Worker function for multiprocessing (must be top-level for pickling)."""
    sample, png_dir, dilation = args
    name = str(sample["name"])
    png_path = png_dir / f"{name}.png"
    if not png_path.exists():
        return sample, "missing_png"

    rooms = sample.get("rooms", [])
    n_rooms = len(rooms)
    room_categories = [r.get("category", -1) for r in rooms]
    try:
        access_adj = extract_access_adjacencies(
            str(png_path), n_rooms,
            dilation_iterations=dilation,
            room_categories=room_categories,
        )
    except Exception as exc:
        logger.warning("Failed on %s: %s", name, exc)
        return sample, "error"

    augmented = dict(sample)
    augmented["access_adjacencies"] = access_adj
    return augmented, "ok"


def process_split(
    diffplanner_json: Path,
    png_dir: Path,
    output_json: Path,
    dilation: int = 3,
    workers: int = 0,
):
    """Process one split file and write the augmented version."""
    with open(diffplanner_json, "r", encoding="utf-8") as f:
        samples = json.load(f)

    logger.info(
        "Processing %s: %d samples (dilation=%d)",
        diffplanner_json.name, len(samples), dilation,
    )

    stats = {"ok": 0, "missing_png": 0, "error": 0, "total_doors": 0}
    results = []

    work_items = [(s, png_dir, dilation) for s in samples]

    if workers > 1:
        with Pool(processes=workers) as pool:
            for augmented, status in tqdm(
                pool.imap(_process_sample, work_items, chunksize=64),
                total=len(samples),
                desc=diffplanner_json.stem,
                unit="sample",
            ):
                stats[status] += 1
                if status == "ok":
                    stats["total_doors"] += len(
                        augmented.get("access_adjacencies", [])
                    )
                    results.append(augmented)
                else:
                    results.append(augmented if isinstance(augmented, dict) else augmented)
    else:
        for item in tqdm(work_items, desc=diffplanner_json.stem, unit="sample"):
            augmented, status = _process_sample(item)
            stats[status] += 1
            if status == "ok":
                stats["total_doors"] += len(
                    augmented.get("access_adjacencies", [])
                )
                results.append(augmented)
            else:
                results.append(augmented if isinstance(augmented, dict) else augmented)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, separators=(",", ":"))

    logger.info(
        "  %s: ok=%d  missing=%d  errors=%d  total_access_pairs=%d  -> %s",
        diffplanner_json.stem,
        stats["ok"], stats["missing_png"], stats["error"],
        stats["total_doors"], output_json,
    )
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Build dataset_access with door-based access adjacencies",
    )
    parser.add_argument(
        "--diffplanner-dir", type=str,
        default="dataset/rplan/dataset_diffplanner",
        help="Directory containing DiffPlanner data_*.json files",
    )
    parser.add_argument(
        "--png-dir", type=str,
        default="dataset/rplan/dataset_original/floorplan_dataset",
        help="Directory containing RPLAN {name}.png files",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str,
        default="dataset/rplan/dataset_access",
        help="Output directory for augmented JSON files",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        help="Which splits to process (default: train val test)",
    )
    parser.add_argument(
        "--dilation", type=int, default=3,
        help="Dilation iterations for door-to-room matching (default: 3)",
    )
    parser.add_argument(
        "--workers", "-j", type=int, default=0,
        help="Number of parallel workers (0 = single process, default: 0)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    diffplanner_dir = Path(args.diffplanner_dir)
    png_dir = Path(args.png_dir)
    output_dir = Path(args.output_dir)

    grand_stats = {"ok": 0, "missing_png": 0, "error": 0, "total_doors": 0}

    for split in args.splits:
        src = diffplanner_dir / f"data_{split}.json"
        if not src.exists():
            logger.warning("Split file not found, skipping: %s", src)
            continue
        dst = output_dir / f"data_{split}.json"
        split_stats = process_split(
            src, png_dir, dst,
            dilation=args.dilation,
            workers=args.workers,
        )
        for k in grand_stats:
            grand_stats[k] += split_stats[k]

    logger.info("=" * 60)
    logger.info("DONE  ok=%d  missing=%d  errors=%d  total_access_pairs=%d",
                grand_stats["ok"], grand_stats["missing_png"],
                grand_stats["error"], grand_stats["total_doors"])
    logger.info("Output directory: %s", output_dir)


if __name__ == "__main__":
    main()
