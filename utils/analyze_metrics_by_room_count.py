#!/usr/bin/env python3
"""
Analyze benchmark metrics binned by room count.

Reads ``results.npz`` produced by ``utils.report_results``.  The NPZ must
contain a ``room_counts`` array (one entry per unique sample) and
``num_variations``.  Every per-sample metric array found in the NPZ is
grouped into room-count bins and reported.

Room-count bins: <=4, 5, 6, 7, 8, >8

Usage:
    python utils/analyze_metrics_by_room_count.py \\
        --results_npz  eval_results_houseganpp/results.npz

    python utils/analyze_metrics_by_room_count.py \\
        --results_npz  eval_results_housediffusion/results.npz \\
        --output_json  eval_results_housediffusion/scores_per_room.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

BIN_ORDER = ["<=4", "5", "6", "7", "8", ">8"]


# ── Helpers ──────────────────────────────────────────────────────────


def _room_bin(n: int) -> str:
    if n <= 4:
        return "<=4"
    if n <= 8:
        return str(n)
    return ">8"


def _stats(values: np.ndarray) -> dict:
    """Compute summary stats for a 1-D array (may contain NaN)."""
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return {"mean": None, "std": None, "median": None,
                "min": None, "max": None, "n_valid": 0}
    return {
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
        "median": float(np.median(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "n_valid": int(len(valid)),
    }


# ── Extract per-sample score arrays from the NPZ ────────────────────

_SKIP_NPZ_KEYS = {
    "num_variations", "summary", "room_counts",
    "matched_room_pairs", "unmatched_pred_rooms", "unmatched_gt_rooms",
    "total_samples", "hidden_count", "disconnected_count", "target_set",
}


def _extract_metric_arrays(npz) -> Dict[str, np.ndarray]:
    """Return ``{npz_key: 1-D array}`` for every binnable metric in the NPZ.

    Any top-level key whose value is a 1-D numeric array is treated as a
    metric.  Non-metric keys are skipped automatically.
    """
    arrays: Dict[str, np.ndarray] = {}
    for key in npz.files:
        if key in _SKIP_NPZ_KEYS:
            continue
        try:
            arr = np.array(npz[key], dtype=float)
        except (ValueError, TypeError):
            continue
        if arr.ndim == 1 and arr.size > 0:
            arrays[key] = arr
    return arrays


# ── Core binning logic ───────────────────────────────────────────────


def bin_metrics(room_counts: np.ndarray,
                metric_arrays: Dict[str, np.ndarray],
                num_variations: int) -> Dict[str, dict]:
    """Group every metric array into room-count bins and compute stats.

    ``room_counts`` has one entry per unique sample.  Metric arrays have
    ``len(room_counts) * num_variations`` entries.
    """
    from collections import defaultdict

    num_samples = len(room_counts)

    bins: Dict[str, Dict[str, list]] = defaultdict(
        lambda: {k: [] for k in metric_arrays}
    )

    for idx in range(num_samples):
        label = _room_bin(int(room_counts[idx]))
        lo = idx * num_variations
        hi = lo + num_variations
        for key, arr in metric_arrays.items():
            if hi <= len(arr):
                bins[label][key].extend(arr[lo:hi].tolist())

    result: Dict[str, dict] = {}
    for label in BIN_ORDER:
        if label not in bins:
            result[label] = {"n": 0, "metrics": {}}
            continue
        metrics = {}
        n = 0
        for key, values in bins[label].items():
            a = np.array(values, dtype=float)
            n = max(n, len(a))
            metrics[key] = _stats(a)
        result[label] = {"n": n, "metrics": metrics}

    return result


# ── Printing ─────────────────────────────────────────────────────────


def _fmt(val, width=10, prec=3):
    if val is None:
        return "N/A".rjust(width)
    return f"{val:.{prec}f}".rjust(width)


def print_table(bin_stats: Dict[str, dict], metric_keys: list,
                title: str = ""):
    header_metrics = [k.replace("_", " ") for k in metric_keys]
    col_w = 12

    print(f"\n{'=' * 80}")
    if title:
        print(f"  METRICS BY ROOM COUNT — {title}")
    else:
        print(f"  METRICS BY ROOM COUNT")
    print(f"{'=' * 80}\n")

    # ── Summary table ────────────────────────────────────────────────
    header = f"{'Bin':<6} {'N':>6}"
    for name in header_metrics:
        header += f"  {name + ' mean':>{col_w}}  {name + ' std':>{col_w}}"
    print(header)
    print("-" * len(header))

    for label in BIN_ORDER:
        bs = bin_stats[label]
        n = bs["n"]
        row = f"{label:<6} {n:>6}"
        for key in metric_keys:
            m = bs["metrics"].get(key)
            if m:
                row += f"  {_fmt(m['mean'], col_w)}  {_fmt(m['std'], col_w)}"
            else:
                row += f"  {'N/A':>{col_w}}  {'N/A':>{col_w}}"
        print(row)

    print()

    # ── Detailed per-bin ─────────────────────────────────────────────
    for label in BIN_ORDER:
        bs = bin_stats[label]
        if bs["n"] == 0:
            continue
        print(f"{'=' * 50}")
        print(f"  Room Bin: {label}  (N = {bs['n']})")
        print(f"{'=' * 50}")
        for key in metric_keys:
            m = bs["metrics"].get(key)
            if not m or m["mean"] is None:
                print(f"\n  {key}: (no data)")
                continue
            print(f"\n  {key}:")
            print(f"    Mean:   {m['mean']:.4f}")
            print(f"    Std:    {m['std']:.4f}")
            print(f"    Median: {m['median']:.4f}")
            print(f"    Range:  [{m['min']:.4f}, {m['max']:.4f}]")
            print(f"    Valid:  {m['n_valid']}")
        print()


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Analyze benchmark metrics binned by room count.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results_npz", type=str, required=True,
                        help="Path to results.npz")
    parser.add_argument("--title", type=str, default="",
                        help="Display title for the report header")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Save binned statistics as JSON")
    args = parser.parse_args()

    npz_path = Path(args.results_npz)
    if not npz_path.exists():
        print(f"Error: NPZ file not found: {npz_path}")
        return 1

    npz = np.load(str(npz_path), allow_pickle=True)

    if "room_counts" not in npz:
        print("Error: NPZ file does not contain 'room_counts'. "
              "Re-run the evaluation to generate an updated results.npz.")
        return 1

    room_counts = np.array(npz["room_counts"], dtype=int)
    metric_arrays = _extract_metric_arrays(npz)

    nv = npz["num_variations"].item() if "num_variations" in npz else 1
    num_variations = int(nv) if nv else 1

    num_samples = len(room_counts)

    print(f"{'=' * 80}")
    print(f"  ANALYZING METRICS BY ROOM COUNT")
    print(f"{'=' * 80}")
    print(f"  NPZ:        {npz_path}")
    print(f"  Variations:  {num_variations}")
    print(f"  Metrics:     {', '.join(metric_arrays.keys())}")
    print(f"  Samples:     {num_samples}")

    bin_stats = bin_metrics(room_counts, metric_arrays, num_variations)
    metric_keys = list(metric_arrays.keys())
    print_table(bin_stats, metric_keys, title=args.title)

    if args.output_json:
        out = Path(args.output_json)
        serialisable = {}
        for label, bs in bin_stats.items():
            serialisable[label] = {"n": bs["n"], "metrics": bs["metrics"]}
        with open(out, "w") as f:
            json.dump({
                "results_npz": str(npz_path),
                "num_variations": num_variations,
                "num_samples": num_samples,
                "metric_keys": metric_keys,
                "bin_statistics": serialisable,
            }, f, indent=2)
        print(f"  Statistics saved to: {out}")

    print(f"\n{'=' * 80}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'=' * 80}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
