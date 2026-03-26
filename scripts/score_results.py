#!/usr/bin/env python3
"""
Score saved inference results against ground truth using metrics from
``utils/metrics.py``.

Available metrics:
    ged       -- Graph Edit Distance between access graphs
    accuracy  -- Room-type multiset matching (bucketed by room count)
    delta     -- Shape regularity deviation via RGL mesh comparison

Usage:
    # All metrics
    python scripts/score_results.py \\
        --results-file checkpoints/model/test_hypergraphs.json \\
        --test-dir dataset/hypergraph/test \\
        --metrics ged accuracy delta

    # Just GED
    python scripts/score_results.py \\
        --results-file results.json \\
        --test-dir dataset/hypergraph/test \\
        --metrics ged

    # GED with corrected accessgraph (mesh-adjacency corrected)
    python scripts/score_results.py \\
        --results-file results.json \\
        --test-dir dataset/hypergraph/test \\
        --metrics ged \\
        --correct-accessgraph

    # Accuracy with custom buckets
    python scripts/score_results.py \\
        --results-file results.json \\
        --test-dir dataset/hypergraph/test \\
        --metrics accuracy \\
        --buckets '<=4' 5 6 7 8 '>=9'

    # Delta with CSV output
    python scripts/score_results.py \\
        --results-file results.json \\
        --test-dir dataset/hypergraph/test \\
        --metrics delta \\
        --output-csv delta_scores.csv

    # Summary-only (machine-readable, for scripting)
    python scripts/score_results.py \\
        --results-file results.json \\
        --test-dir dataset/hypergraph/test \\
        --metrics ged accuracy \\
        --summary-only
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from utils import (
    bucketed_room_accuracy,
    compute_room_similarity_score,
    compute_ged_from_hypergraph,
    produce_room_list,
)
from utils.hypergraph import Hypergraph

logger = logging.getLogger(__name__)


AVAILABLE_METRICS = ("ged", "accuracy", "delta")


# ===================================================================
# Shared data loading
# ===================================================================

def load_skip_ids(path):
    """Load a newline-delimited skip-IDs file, ignoring comments."""
    skip_ids: set[str] = set()
    if path and Path(path).exists():
        with open(path) as f:
            skip_ids = {
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            }
    return skip_ids


def load_ground_truth(test_dir):
    """Load all ``.txt`` ground-truth files from *test_dir*.

    Returns ``{sample_id: {"accessgraph": ..., "hypergraph": ..., "bounds": ...}}``.
    """
    test_dir = Path(test_dir)
    ground_truth = {}
    for fp in sorted(test_dir.glob("*.txt")):
        with open(fp) as f:
            raw = json.load(f)

        def _parse(val):
            return json.loads(val) if isinstance(val, str) else val

        ground_truth[fp.stem] = {
            "accessgraph": raw.get("accessgraph", {}),
            "hypergraph": _parse(raw.get("hypergraph", "{}")),
            "bounds": _parse(raw.get("bounds", "{}")),
        }
    return ground_truth


# ===================================================================
# Metric: GED
# ===================================================================

def score_ged(results, ground_truth, skip_ids, correct_accessgraph=False):
    """Compute GED for every sample in *results*.

    If *correct_accessgraph* is True, applies correct_accessgraph before GED.
    Returns ``{"per_sample": [...], "summary": {...}}``.
    """
    import copy as _copy

    per_sample = []
    skipped = 0

    for sample_id in tqdm(results, desc="Computing GED" + (" (corrected)" if correct_accessgraph else ""), unit="sample"):
        if sample_id in skip_ids:
            skipped += 1
            continue
        gt = ground_truth.get(sample_id)
        if gt is None:
            continue

        bounds = None
        if correct_accessgraph:
            bounds = gt.get("bounds") or {}
            if isinstance(bounds, str):
                bounds = json.loads(bounds) if bounds else {}

        ged, graph_len = compute_ged_from_hypergraph(
            gt["accessgraph"],
            _copy.deepcopy(results[sample_id]) if correct_accessgraph else results[sample_id],
            sample_id,
            correct_accessgraph=correct_accessgraph,
            bounds=bounds,
        )
        per_sample.append({
            "sample_id": sample_id,
            "ged": ged,
            "graph_len": graph_len,
        })

    values = [r["ged"] for r in per_sample if r["ged"] is not None]
    summary = {
        "mean": sum(values) / len(values) if values else float("nan"),
        "min": min(values) if values else float("nan"),
        "max": max(values) if values else float("nan"),
        "successful": len(values),
        "total": len(results),
        "skipped": skipped,
    }
    return {"per_sample": per_sample, "summary": summary}


# ===================================================================
# Metric: Accuracy
# ===================================================================

def score_accuracy(results, ground_truth, skip_ids, buckets):
    """Compute bucketed room-type accuracy.

    Delegates to :func:`utils.metrics.bucketed_room_accuracy`.
    Returns ``{"per_sample": None, "summary": {...}}``.
    """
    accessgraphs = {
        sid: gt["accessgraph"]
        for sid, gt in ground_truth.items()
    }
    stats = bucketed_room_accuracy(
        accessgraphs, results, buckets, skip_ids=skip_ids,
    )
    return {"per_sample": None, "summary": stats}


# ===================================================================
# Metric: Delta
# ===================================================================

def score_delta(results, ground_truth, skip_ids):
    """Compute delta (shape regularity) metric for every sample.

    Returns ``{"per_sample": [...], "summary": {...}}``.
    """
    per_sample = []
    skipped = 0
    errors = 0

    for sample_id in tqdm(results, desc="Computing delta", unit="sample"):
        if sample_id in skip_ids:
            skipped += 1
            continue
        gt = ground_truth.get(sample_id)
        if gt is None:
            continue

        try:
            bounds = gt["bounds"]
            gen_mesh = Hypergraph(results[sample_id]).to_mesh(bounds=bounds)
            gt_mesh = Hypergraph(gt["hypergraph"]).to_mesh(bounds=bounds)
            delta = compute_room_similarity_score(
                produce_room_list(gen_mesh),
                produce_room_list(gt_mesh),
            )
            per_sample.append({"sample_id": sample_id, "delta": delta})
        except Exception as e:
            logger.warning(f"Delta failed for {sample_id}: {e}")
            per_sample.append({"sample_id": sample_id, "delta": None, "error": str(e)})
            errors += 1

    values = [r["delta"] for r in per_sample if r["delta"] is not None]
    summary = {
        "mean": sum(values) / len(values) if values else float("nan"),
        "successful": len(values),
        "total": len(results),
        "skipped": skipped,
        "errors": errors,
    }
    return {"per_sample": per_sample, "summary": summary}


# ===================================================================
# Reporting
# ===================================================================

def _print_ged_report(result, summary_only):
    s = result["summary"]
    print(f"\n{'=' * 70}")
    print("GED (Graph Edit Distance)")
    print(f"{'=' * 70}")
    print(f"  Mean:       {s['mean']:.2f}")
    print(f"  Min:        {s['min']:.2f}")
    print(f"  Max:        {s['max']:.2f}")
    print(f"  Successful: {s['successful']}/{s['total']}")
    print(f"  Excluded:   {s['skipped']}")


def _print_accuracy_report(result, summary_only):
    stats = result["summary"]
    print(f"\n{'=' * 70}")
    print("ACCURACY BY NUMBER OF ROOMS")
    print(f"{'=' * 70}\n")
    print(f"{'Category':<12} {'Matches':<10} {'Total':<10} {'Accuracy'}")
    print(f"{'-' * 70}")

    for label in stats["category_order"]:
        cat = stats["categories"][label]
        if cat["total"] > 0:
            acc = cat["matches"] / cat["total"] * 100
            print(f"{label:<12} {cat['matches']:<10} {cat['total']:<10} {acc:>6.1f}%")
        else:
            print(f"{label:<12} {cat['matches']:<10} {cat['total']:<10} {'N/A':>6}")

    print(f"{'-' * 70}")
    if stats["overall_total"] > 0:
        overall = stats["overall_matches"] / stats["overall_total"] * 100
        print(
            f"{'Overall':<12} {stats['overall_matches']:<10} "
            f"{stats['overall_total']:<10} {overall:>6.1f}%"
        )
    print(f"  Excluded:   {stats['skipped']}")


def _print_delta_report(result, summary_only):
    s = result["summary"]
    print(f"\n{'=' * 70}")
    print("DELTA (Shape Regularity)")
    print(f"{'=' * 70}")
    print(f"  Mean:       {s['mean']:.4f}")
    print(f"  Successful: {s['successful']}/{s['total']}")
    print(f"  Excluded:   {s['skipped']}")
    if s["errors"]:
        print(f"  Errors:     {s['errors']}")


_REPORT_PRINTERS = {
    "ged": _print_ged_report,
    "accuracy": _print_accuracy_report,
    "delta": _print_delta_report,
}


def print_report(metric_name, result, summary_only=False):
    _REPORT_PRINTERS[metric_name](result, summary_only)


def write_csv(path, metric_results, results_keys):
    """Write a combined CSV with one row per sample.

    Columns: ``sample_id`` + one column per metric that has per-sample data.
    """
    col_map = {}
    for name in ("ged", "delta"):
        if name in metric_results and metric_results[name]["per_sample"]:
            col_map[name] = {
                r["sample_id"]: r.get(name)
                for r in metric_results[name]["per_sample"]
            }

    if not col_map:
        logger.info("No per-sample data to write to CSV.")
        return

    fieldnames = ["sample_id"] + list(col_map.keys())
    path = Path(path)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sid in results_keys:
            row = {"sample_id": sid}
            for metric_name, lookup in col_map.items():
                row[metric_name] = lookup.get(sid)
            writer.writerow(row)

    logger.info(f"CSV written to {path}")


# ===================================================================
# CLI
# ===================================================================

def cli():
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Score saved inference results against ground truth"
    )
    parser.add_argument(
        "--results-file", required=True,
        help="Path to {sample_id: hypergraph} JSON file",
    )
    parser.add_argument(
        "--test-dir", required=True,
        help="Directory with ground-truth .txt files",
    )
    parser.add_argument(
        "--metrics", nargs="+", required=True,
        choices=AVAILABLE_METRICS,
        help="Metrics to compute (ged, accuracy, delta)",
    )
    parser.add_argument(
        "--correct-accessgraph",
        action="store_true",
        help="Apply correct_accessgraph before GED (only affects ged metric).",
    )
    parser.add_argument(
        "--skip-ids-file", default=None,
        help="Newline-delimited file of sample IDs to exclude "
             "(default: utils/skipped_ids)",
    )
    parser.add_argument(
        "--buckets", nargs="+",
        default=["<=4", "5", "6", "7", "8", ">=9"],
        help="Accuracy bucket boundaries (default: <=4 5 6 7 8 >=9)",
    )
    parser.add_argument(
        "--output-csv", default=None,
        help="Write per-sample results to this CSV file",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Print only aggregate statistics",
    )
    args = parser.parse_args()

    results_path = Path(args.results_file)
    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        return 1

    skip_ids_path = (
        args.skip_ids_file
        or str(Path(__file__).parent.parent / "utils" / "skipped_ids")
    )
    skip_ids = load_skip_ids(skip_ids_path)
    if skip_ids:
        logger.info(f"Loaded {len(skip_ids)} skip IDs from {skip_ids_path}")

    logger.info(f"Loading results from {results_path} ...")
    with open(results_path) as f:
        results = json.load(f)
    logger.info(f"  {len(results)} samples")

    logger.info(f"Loading ground truth from {args.test_dir} ...")
    ground_truth = load_ground_truth(args.test_dir)
    logger.info(f"  {len(ground_truth)} ground-truth files")

    metric_results: dict[str, dict] = {}

    for metric in args.metrics:
        if metric == "ged":
            metric_results["ged"] = score_ged(
                results, ground_truth, skip_ids,
                correct_accessgraph=args.correct_accessgraph,
            )
        elif metric == "accuracy":
            metric_results["accuracy"] = score_accuracy(
                results, ground_truth, skip_ids, args.buckets,
            )
        elif metric == "delta":
            metric_results["delta"] = score_delta(results, ground_truth, skip_ids)

    for metric in args.metrics:
        print_report(metric, metric_results[metric], args.summary_only)

    if args.output_csv:
        write_csv(args.output_csv, metric_results, results.keys())

    return 0


if __name__ == "__main__":
    sys.exit(cli())
