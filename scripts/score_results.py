#!/usr/bin/env python3
"""
Score saved inference results against ground truth using metrics from
``utils/metrics.py``, then write unified NPZ + TXT reports via
``utils.report_results``.

Available metrics:
    ged       -- Graph Edit Distance between access graphs
    accuracy  -- Room-type multiset matching (bucketed by room count)
    delta     -- Shape regularity deviation via RGL mesh comparison
    area_proportion_error -- Area proportion error per room type (see metrics)

Usage:
    # All metrics
    python scripts/score_results.py \\
        --results-file checkpoints/model/test_hypergraphs.json \\
        --test-dir dataset/hypergraph/test \\
        --output-dir checkpoints/model/eval \\
        --metrics ged accuracy delta area_proportion_error

    # Just GED
    python scripts/score_results.py \\
        --results-file results.json \\
        --test-dir dataset/hypergraph/test \\
        --output-dir eval_output \\
        --metrics ged

    # GED with corrected accessgraph (mesh-adjacency corrected)
    python scripts/score_results.py \\
        --results-file results.json \\
        --test-dir dataset/hypergraph/test \\
        --output-dir eval_output \\
        --metrics ged \\
        --correct-accessgraph

    # Accuracy only
    python scripts/score_results.py \\
        --results-file results.json \\
        --test-dir dataset/hypergraph/test \\
        --output-dir eval_output \\
        --metrics accuracy
"""

from __future__ import annotations

import copy
import json
import logging
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils import (
    compute_room_area_proportion_error,
    compute_room_similarity_score,
    compute_room_type_accuracy,
    produce_room_area_list,
    produce_room_list,
)
from utils.hypergraph import Hypergraph
from utils.report_results import report_results

logger = logging.getLogger(__name__)

AVAILABLE_METRICS = ("ged", "accuracy", "delta", "area_proportion_error")


# ===================================================================
# GED worker mode  (re-invoked as a subprocess for each sample)
# ===================================================================

def _ged_worker_main() -> int:
    """Compute GED for a single sample and print results to stdout."""
    import argparse
    from utils import compute_ged_from_hypergraph

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--_ged-worker", action="store_true")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--results-file", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--correct-accessgraph", action="store_true")
    args = parser.parse_args()

    def _parse_jsonish(val):
        return json.loads(val) if isinstance(val, str) else val

    with open(args.results_file) as f:
        results = json.load(f)

    sample_id = str(args.sample_id)
    if sample_id not in results:
        raise KeyError(f"Sample {sample_id} not found in {args.results_file}")

    gt_path = args.test_dir / f"{sample_id}.txt"
    with open(gt_path) as f:
        raw = json.load(f)
    gt = {
        "accessgraph": raw.get("accessgraph", {}),
        "bounds": _parse_jsonish(raw.get("bounds", "{}")),
    }

    bounds = gt["bounds"] if args.correct_accessgraph else None

    start = time.perf_counter()
    ged, graph_len = compute_ged_from_hypergraph(
        gt["accessgraph"],
        copy.deepcopy(results[sample_id]) if args.correct_accessgraph else results[sample_id],
        sample_id,
        timeout=int(args.timeout_seconds * 1000),
        correct_accessgraph=args.correct_accessgraph,
        bounds=bounds,
    )
    elapsed = time.perf_counter() - start

    print(f"ged: {ged}")
    print(f"graph_len: {graph_len}")
    print(f"elapsed_seconds: {elapsed:.2f}")
    return 0


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

def _parse_output_field(text: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}:\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _run_ged_subprocess(
    sample_id: str,
    results_file: Path,
    test_dir: Path,
    timeout_seconds: float,
    correct_accessgraph: bool,
) -> dict:
    """Re-invoke this script in worker mode for a single sample."""
    wall_timeout = timeout_seconds + 30.0
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--_ged-worker",
        "--sample-id", sample_id,
        "--results-file", str(results_file),
        "--test-dir", str(test_dir),
        "--timeout-seconds", str(timeout_seconds),
    ]
    if correct_accessgraph:
        cmd.append("--correct-accessgraph")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=wall_timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"GED timed out for sample {sample_id} after {wall_timeout:.0f}s")
        return {"sample_id": sample_id, "ged": None, "graph_len": None}

    text = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    raw_ged = _parse_output_field(text, "ged")
    raw_len = _parse_output_field(text, "graph_len")

    ged = None
    graph_len = None
    try:
        if raw_ged and raw_ged.lower() not in ("none", ""):
            ged = float(raw_ged)
    except ValueError:
        pass
    try:
        if raw_len and raw_len.lower() not in ("none", ""):
            graph_len = int(float(raw_len))
    except ValueError:
        pass

    return {"sample_id": sample_id, "ged": ged, "graph_len": graph_len}


def score_ged(results, ground_truth, skip_ids, correct_accessgraph=False,
              results_file: Path = None, test_dir: Path = None,
              timeout_seconds: float = 20.0, workers: int = 64):
    """Compute GED for every sample in *results*.

    Each sample runs in an isolated subprocess so timed-out GED threads never
    accumulate and throttle subsequent computations.

    If *correct_accessgraph* is True, applies correct_accessgraph before GED.
    Returns ``{"per_sample": [...], "summary": {...}}``.
    """
    sample_ids = [
        sid for sid in results
        if sid not in skip_ids and sid in ground_truth
    ]

    per_sample: list[dict] = []
    desc = "Computing GED" + (" (corrected)" if correct_accessgraph else "")

    if workers <= 1:
        for sid in tqdm(sample_ids, desc=desc, unit="sample"):
            per_sample.append(_run_ged_subprocess(
                sid, results_file, test_dir, timeout_seconds, correct_accessgraph,
            ))
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for sid in sample_ids:
                f = pool.submit(
                    _run_ged_subprocess,
                    sid, results_file, test_dir, timeout_seconds, correct_accessgraph,
                )
                futures[f] = sid

            pbar = tqdm(total=len(futures), desc=desc, unit="sample")
            for f in as_completed(futures):
                per_sample.append(f.result())
                pbar.update(1)
            pbar.close()

    values = [r["ged"] for r in per_sample if r["ged"] is not None]
    summary = {
        "mean": sum(values) / len(values) if values else float("nan"),
        "min": min(values) if values else float("nan"),
        "max": max(values) if values else float("nan"),
        "successful": len(values),
        "total": len(results),
        "skipped": len(results) - len(sample_ids),
    }
    return {"per_sample": per_sample, "summary": summary}


# ===================================================================
# Metric: Accuracy
# ===================================================================

def score_accuracy(results, ground_truth, skip_ids):
    """Compute per-sample room-type accuracy (combined, count, type).

    Returns ``{"per_sample": [{"accuracy": ..., "room_count_accuracy": ...,
    "room_type_accuracy": ...}, ...]}``.
    """
    per_sample = []
    skipped = 0

    for sample_id in tqdm(results, desc="Computing accuracy", unit="sample"):
        if sample_id in skip_ids:
            skipped += 1
            continue
        gt = ground_truth.get(sample_id)
        if gt is None:
            continue

        accessgraph = gt["accessgraph"]
        try:
            result = compute_room_type_accuracy(accessgraph, results[sample_id])
            per_sample.append(result)
        except Exception as exc:
            logger.warning("Accuracy failed for %s: %s", sample_id, exc)
            per_sample.append({
                "accuracy": None,
                "room_count_accuracy": None,
                "room_type_accuracy": None,
            })

    if skipped:
        logger.info("Accuracy: skipped %d samples", skipped)
    return {"per_sample": per_sample}


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
# Metric: Area proportion error
# ===================================================================

def score_area_proportion_error(results, ground_truth, skip_ids):
    """Compute area-proportion error for every sample.

    Uses :func:`utils.metrics.compute_room_area_proportion_error`: each room's
    area is converted to a proportion of total apartment area, rooms are paired
    1-to-1 within each type, and the mean absolute proportion difference is
    returned.
    """
    per_sample = []
    skipped = 0
    errors = 0

    for sample_id in tqdm(results, desc="Computing area_proportion_error", unit="sample"):
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
            result = compute_room_area_proportion_error(
                produce_room_area_list(gen_mesh),
                produce_room_area_list(gt_mesh),
            )
            per_sample.append({
                "sample_id": sample_id,
                "area_proportion_error": result['area_proportion_error'],
                "area_proportion_error_detail": result,
            })
        except Exception as e:
            logger.warning(f"Area proportion error failed for {sample_id}: {e}")
            per_sample.append({"sample_id": sample_id, "area_proportion_error": None, "error": str(e)})
            errors += 1

    values = [r["area_proportion_error"] for r in per_sample if r["area_proportion_error"] is not None]
    summary = {
        "mean": sum(values) / len(values) if values else float("nan"),
        "successful": len(values),
        "total": len(results),
        "skipped": skipped,
        "errors": errors,
    }
    return {"per_sample": per_sample, "summary": summary}


# ===================================================================
# CLI
# ===================================================================

def cli():
    import argparse

    if "--_ged-worker" in sys.argv:
        sys.exit(_ged_worker_main())

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
        "--output-dir", required=True,
        help="Directory for results.npz and results.txt output",
    )
    parser.add_argument(
        "--metrics", nargs="+", required=True,
        choices=AVAILABLE_METRICS,
        help="Metrics to compute (ged, accuracy, delta, area_proportion_error)",
    )
    parser.add_argument(
        "--correct-accessgraph",
        action="store_true",
        help="Apply correct_accessgraph before GED (only affects ged metric).",
    )
    parser.add_argument(
        "--ged-timeout", type=float, default=20.0,
        help="Per-sample GED timeout in seconds (default: 20)",
    )
    parser.add_argument(
        "--ged-workers", type=int, default=64,
        help="Number of parallel GED subprocesses (default: 64)",
    )
    parser.add_argument(
        "--skip-ids-file", default=None,
        help="Newline-delimited file of sample IDs to exclude "
             "(default: no skipping)",
    )
    args = parser.parse_args()

    results_path = Path(args.results_file)
    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        return 1

    skip_ids = set()
    if args.skip_ids_file:
        skip_ids = load_skip_ids(args.skip_ids_file)
        if skip_ids:
            logger.info(f"Loaded {len(skip_ids)} skip IDs from {args.skip_ids_file}")

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
                results_file=results_path,
                test_dir=Path(args.test_dir),
                timeout_seconds=args.ged_timeout,
                workers=args.ged_workers,
            )
        elif metric == "accuracy":
            metric_results["accuracy"] = score_accuracy(
                results, ground_truth, skip_ids,
            )
        elif metric == "delta":
            metric_results["delta"] = score_delta(results, ground_truth, skip_ids)
        elif metric == "area_proportion_error":
            metric_results["area_proportion_error"] = score_area_proportion_error(
                results, ground_truth, skip_ids)

    # ── Collect per-sample values into flat arrays for the reporter ──
    all_metrics: dict[str, tuple[str, np.ndarray]] = {}

    ged_result = metric_results.get("ged")
    if ged_result and ged_result["per_sample"]:
        all_metrics["geds"] = (
            "GED",
            np.array([
                r["ged"] if r["ged"] is not None else np.nan
                for r in ged_result["per_sample"]
            ]),
        )

    acc_result = metric_results.get("accuracy")
    if acc_result and acc_result["per_sample"]:
        for key, npz_key, title in [
            ("accuracy", "accuracies", "Room Accuracy"),
            ("room_count_accuracy", "room_count_accuracies", "Room Count Accuracy"),
            ("room_type_accuracy", "room_type_accuracies", "Room Type Accuracy"),
        ]:
            all_metrics[npz_key] = (
                title,
                np.array([
                    r[key] if r[key] is not None else np.nan
                    for r in acc_result["per_sample"]
                ]),
            )

    delta_result = metric_results.get("delta")
    if delta_result and delta_result["per_sample"]:
        all_metrics["similarity_scores"] = (
            "Room Similarity (delta)",
            np.array([
                r["delta"] if r["delta"] is not None else np.nan
                for r in delta_result["per_sample"]
            ]),
        )

    ape_result = metric_results.get("area_proportion_error")
    if ape_result and ape_result["per_sample"]:
        all_metrics["area_proportion_errors"] = (
            "Area Proportion Error",
            np.array([
                r["area_proportion_error"] if r["area_proportion_error"] is not None else np.nan
                for r in ape_result["per_sample"]
            ]),
        )

    report_results(args.output_dir, metrics=all_metrics)

    return 0


if __name__ == "__main__":
    sys.exit(cli())
