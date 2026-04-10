"""
Convert hypergraph data to DiffPlanner JSON format.

Room Type Mapping (hypergraph → DiffPlanner):
    hypergraph 0  (living)  → DiffPlanner 0 (LivingRoom)
    hypergraph 1  (bed)     → DiffPlanner 1 (Bedroom)
    hypergraph 2  (kitchen) → DiffPlanner 2 (Kitchen)
    hypergraph 3  (bath)    → DiffPlanner 3 (Bathroom)
    hypergraph 10 (foyer)   → DiffPlanner 0 (LivingRoom)  [merged with living]
    hypergraph 11 (extra)   → DiffPlanner 5 (Storage)

Output JSON schema per sample (boundary-unconstrained mode):
    {
        "name": "<sample_id>",
        "rooms": [
            {"id": 0, "category": 0, "r_boundary": [[x,y],...],
             "size": 1234.0, "location": [cx,cy], "box": [x0,y0,x1,y1], "order": 0},
            ...
        ],
        "adjacencies": [[0,1], [1,2], ...],
        "access_adjacencies": [[0,1], ...]
    }

Usage:
    python benchmark/utils/convert_hypergraph_to_diffplanner.py
    python benchmark/utils/convert_hypergraph_to_diffplanner.py --splits train test
    python benchmark/utils/convert_hypergraph_to_diffplanner.py --canvas-size 256 --max-rooms 8
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from shapely.geometry import Polygon

_UTILS_DIR = Path(__file__).resolve().parent
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from convert_hypergraph_to_rplan import (
    extract_adjacency_triples,
    extract_polygons_and_room_types,
    normalize_polygons,
)

from hypergraph.api.lib.tools import RGL

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# RPLAN room-type IDs → DiffPlanner 0-5 categories
RPLAN_TO_DIFFPLANNER: Dict[int, int] = {
    1: 0,   # living_room → LivingRoom
    2: 2,   # kitchen     → Kitchen
    3: 1,   # bedroom     → Bedroom
    4: 3,   # bathroom    → Bathroom
    6: 0,   # entrance    → LivingRoom (merged)
    10: 5,  # storage     → Storage
}

DIFFPLANNER_CATEGORY_NAMES = {
    0: "LivingRoom",
    1: "Bedroom",
    2: "Kitchen",
    3: "Bathroom",
    4: "Balcony",
    5: "Storage",
}

SPLIT_NAME_MAP = {
    "train": "train",
    "validation": "val",
    "test": "test",
}


def extract_access_adjacencies(data: dict) -> List[List[int]]:
    """Extract door-based room-room pairs from the DataNode ``connected`` field.

    Returns sorted list of ``[room_a, room_b]`` pairs (0-indexed, ``a < b``).
    """
    hypergraph_json_str = data["split"]
    data_node = RGL.DataNode.deserializeDataNode(hypergraph_json_str)
    all_children = RGL.DataNode.GetAllChildren(data_node)

    name_to_idx: Dict[str, int] = {}
    idx = 0
    for i in range(all_children.Count):
        child = all_children[i]
        if child.final:
            name_to_idx[child.name] = idx
            idx += 1

    pairs: set[Tuple[int, int]] = set()
    for i in range(all_children.Count):
        child = all_children[i]
        if not (child.final and hasattr(child, "connected") and child.connected):
            continue
        room_idx = name_to_idx.get(child.name)
        if room_idx is None:
            continue
        for neighbor_name in list(child.connected):
            neighbor_idx = name_to_idx.get(str(neighbor_name))
            if neighbor_idx is not None and neighbor_idx != room_idx:
                a, b = min(room_idx, neighbor_idx), max(room_idx, neighbor_idx)
                pairs.add((a, b))

    return sorted([list(p) for p in pairs])


def convert_sample(
    data: dict,
    canvas_size: int = 256,
) -> Optional[dict]:
    """Convert one hypergraph sample to a DiffPlanner-compatible dict.

    Returns ``None`` if the sample cannot be converted.
    """
    try:
        polygons, rplan_types = extract_polygons_and_room_types(data)
    except Exception:
        return None

    if len(polygons) < 2:
        return None

    # Filter out door polygons (type 17) — DiffPlanner uses room-only graphs
    room_indices = [i for i, rt in enumerate(rplan_types) if rt != 17]
    polygons = [polygons[i] for i in room_indices]
    rplan_types = [rplan_types[i] for i in room_indices]

    if len(polygons) < 2:
        return None

    norm_polys = normalize_polygons(polygons, target_size=canvas_size)

    categories = [RPLAN_TO_DIFFPLANNER.get(rt, 5) for rt in rplan_types]

    # Spatial adjacency (geometric room-room neighbours)
    try:
        triples = extract_adjacency_triples(data)
        adjacencies = sorted([[t[0], t[2]] for t in triples if t[1] == 1])
    except Exception:
        adjacencies = []

    # Door-based access adjacency
    try:
        access_adj = extract_access_adjacencies(data)
    except Exception:
        access_adj = []

    rooms = []
    for i, (poly, cat) in enumerate(zip(norm_polys, categories)):
        sp = Polygon(poly)
        if not sp.is_valid:
            sp = sp.buffer(0)
        cx, cy = sp.centroid.x, sp.centroid.y
        area = sp.area
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]

        rooms.append({
            "id": i,
            "category": cat,
            "r_boundary": [[float(x), float(y)] for x, y in poly],
            "size": float(area),
            "location": [float(cx), float(cy)],
            "box": [float(min(xs)), float(min(ys)),
                    float(max(xs)), float(max(ys))],
            "order": i,
        })

    result = {
        "name": str(data.get("id", "")),
        "rooms": rooms,
        "adjacencies": adjacencies,
    }
    if access_adj:
        result["access_adjacencies"] = access_adj

    return result


def convert_split(
    input_file: Path,
    output_file: Path,
    canvas_size: int = 256,
    max_rooms: int = 0,
    verbose: bool = False,
) -> Tuple[int, int, int]:
    """Process one split file and write a DiffPlanner JSON.

    Returns (success, skipped, failed) counts.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"\nProcessing {input_file.name}: {len(dataset)} samples")

    success, skipped, failed = 0, 0, 0
    failed_ids = []
    results = []

    for i, data in enumerate(dataset):
        sample_id = data.get("id", f"sample_{i}")

        if (i + 1) % 100 == 0 or verbose:
            print(f"  {i + 1}/{len(dataset)}  ({sample_id})", end="\r")

        sample = convert_sample(data, canvas_size=canvas_size)

        if sample is None:
            failed += 1
            failed_ids.append(sample_id)
            if verbose:
                print(f"  FAILED: {sample_id}")
            continue

        n_rooms = len(sample["rooms"])
        if max_rooms > 0 and n_rooms > max_rooms:
            skipped += 1
            if verbose:
                print(f"  SKIPPED: {sample_id} ({n_rooms} rooms > {max_rooms})")
            continue

        results.append(sample)
        success += 1

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, separators=(",", ":"))

    n_total = len(dataset)
    print(f"\n  {input_file.stem}: ok={success}  skipped={skipped}  "
          f"failed={failed}  total={n_total}  -> {output_file}")

    # Category distribution
    cat_counts: Dict[int, int] = {}
    for s in results:
        for r in s["rooms"]:
            cat_counts[r["category"]] = cat_counts.get(r["category"], 0) + 1
    for cat_id in sorted(cat_counts):
        name = DIFFPLANNER_CATEGORY_NAMES.get(cat_id, f"Unknown_{cat_id}")
        print(f"    {name}: {cat_counts[cat_id]}")

    avg_rooms = (sum(len(s["rooms"]) for s in results) / len(results)
                 if results else 0)
    avg_adj = (sum(len(s["adjacencies"]) for s in results) / len(results)
               if results else 0)
    avg_access = (sum(len(s.get("access_adjacencies", [])) for s in results)
                  / len(results) if results else 0)
    print(f"    avg_rooms={avg_rooms:.1f}  avg_spatial_adj={avg_adj:.1f}  "
          f"avg_access_adj={avg_access:.1f}")

    return success, skipped, failed, failed_ids


def main():
    parser = argparse.ArgumentParser(
        description="Convert Hypergraph1K dataset to DiffPlanner JSON format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python benchmark/utils/convert_hypergraph_to_diffplanner.py
    python benchmark/utils/convert_hypergraph_to_diffplanner.py --splits train test
    python benchmark/utils/convert_hypergraph_to_diffplanner.py --max-rooms 8
        """,
    )
    parser.add_argument(
        "--input", "-i", type=Path,
        default=PROJECT_ROOT / "dataset/hypergraph",
        help="Directory containing {train,validation,test}.json "
             "(default: dataset/hypergraph)",
    )
    parser.add_argument(
        "--output", "-o", type=Path,
        default=PROJECT_ROOT / "benchmark/dataset/Hypergraph_to_DiffPlanner",
        help="Output directory for DiffPlanner JSON files",
    )
    parser.add_argument(
        "--splits", nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val", "test"],
        help="Splits to process (default: all)",
    )
    parser.add_argument(
        "--canvas-size", type=int, default=256,
        help="Canvas size for coordinate normalisation (default: 256)",
    )
    parser.add_argument(
        "--max-rooms", type=int, default=0,
        help="Skip samples with more rooms (0 = no limit, DiffPlanner uses 8)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input directory not found: {args.input}")
        return 1

    print("=" * 60)
    print("Hypergraph1K -> DiffPlanner JSON Converter")
    print("=" * 60)
    print(f"Input:       {args.input}")
    print(f"Output:      {args.output}")
    print(f"Canvas size: {args.canvas_size}")
    print(f"Max rooms:   {args.max_rooms or 'unlimited'}")
    print(f"Splits:      {', '.join(args.splits)}")

    totals = {"ok": 0, "skipped": 0, "failed": 0}
    all_failed_ids = {}

    for split in args.splits:
        src = args.input / f"{split}.json"
        if not src.exists():
            print(f"\nWarning: {src} not found, skipping")
            continue
        dst_name = f"data_{SPLIT_NAME_MAP.get(split, split)}.json"
        dst = args.output / dst_name

        ok, sk, fa, fids = convert_split(
            src, dst,
            canvas_size=args.canvas_size,
            max_rooms=args.max_rooms,
            verbose=args.verbose,
        )
        totals["ok"] += ok
        totals["skipped"] += sk
        totals["failed"] += fa
        if fids:
            all_failed_ids[split] = fids

    print("\n" + "=" * 60)
    print(f"DONE  ok={totals['ok']}  skipped={totals['skipped']}  "
          f"failed={totals['failed']}")
    print(f"Output: {args.output}")

    if all_failed_ids:
        failed_log = args.output / "failed_conversions.json"
        with open(failed_log, "w", encoding="utf-8") as f:
            json.dump(all_failed_ids, f, indent=2)
        print(f"\nFailed sample IDs saved to: {failed_log}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
