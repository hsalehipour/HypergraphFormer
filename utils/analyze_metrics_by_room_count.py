#!/usr/bin/env python3
"""
Analyze benchmark metrics binned by room count.

Reads a cached-generations pickle (for ground-truth room types) and the
``results.npz`` produced by ``utils.report_results``.
Every per-sample array found in the NPZ (GED, accuracy, similarity,
area proportion error, …) is grouped into room-count bins and reported.

Room-count bins: <=4, 5, 6, 7, 8, >8

Usage:
    python utils/analyze_metrics_by_room_count.py \\
        --model houseganpp \\
        --cache_pickle eval_results_houseganpp/generations.pkl \\
        --results_npz  eval_results_houseganpp/results.npz \\
        --num_variations 5

    python utils/analyze_metrics_by_room_count.py \\
        --model housediffusion \\
        --cache_pickle eval_results_housediffusion/generations.pkl \\
        --results_npz  eval_results_housediffusion/results.npz \\
        --num_variations 5
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np

BIN_ORDER = ["<=4", "5", "6", "7", "8", ">8"]
DOOR_TYPES = {11, 12, 13, 15, 17}


# ── Helpers ──────────────────────────────────────────────────────────


def _count_rooms(room_types: list) -> int:
    return sum(1 for rt in room_types if rt not in DOOR_TYPES)


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
    "num_variations", "summary",
    "matched_room_pairs", "unmatched_pred_rooms", "unmatched_gt_rooms",
    "total_samples", "hidden_count", "disconnected_count", "target_set",
}


def _extract_metric_arrays(npz) -> Dict[str, np.ndarray]:
    """Return ``{npz_key: 1-D array}`` for every binnable metric in the NPZ.

    Any top-level key whose value is a 1-D numeric array is treated as a
    metric.  Non-metric keys (``num_variations``, ``summary``, and
    anything stored via *extra*) are skipped automatically.
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


# ── Room-type extraction from cached generations ─────────────────────


def _room_types_houseganpp(sample: dict) -> list:
    if "raw_sample" in sample:
        return sample["raw_sample"].get("room_types", [])
    return sample.get("room_types", [])


def _room_types_housediffusion(sample: dict) -> list:
    if "gt_room_types_oh" in sample:
        rt_oh = sample["gt_room_types_oh"]
        pad = sample["gt_padding_mask"]
        full = np.argmax(rt_oh, axis=1)
        valid = full[pad == 1]
        types: list = []
        prev = None
        for rt in valid:
            if rt != prev:
                types.append(int(rt))
                prev = rt
        return types
    if "raw_sample" in sample:
        return sample["raw_sample"].get("room_types", [])
    return sample.get("room_types", [])


_ROOM_TYPE_FN = {
    "houseganpp": _room_types_houseganpp,
    "housediffusion": _room_types_housediffusion,
}


# ── Core binning logic ───────────────────────────────────────────────


def bin_metrics(gt_samples: list, metric_arrays: Dict[str, np.ndarray],
                num_variations: int, model: str) -> Dict[str, dict]:
    """Group every metric array into room-count bins and compute stats."""
    room_type_fn = _ROOM_TYPE_FN[model]
    num_samples = len(gt_samples)

    bins: Dict[str, Dict[str, list]] = defaultdict(
        lambda: {k: [] for k in metric_arrays}
    )

    for idx in range(num_samples):
        room_types = room_type_fn(gt_samples[idx])
        label = _room_bin(_count_rooms(room_types))

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
                model_name: str):
    header_metrics = [k.replace("_", " ") for k in metric_keys]
    col_w = 12

    print(f"\n{'=' * 80}")
    print(f"  METRICS BY ROOM COUNT — {model_name}")
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
    parser.add_argument("--model", type=str, required=True,
                        choices=["houseganpp", "housediffusion"])
    parser.add_argument("--cache_pickle", type=str, required=True,
                        help="Path to cached generations pickle")
    parser.add_argument("--results_npz", type=str, required=True,
                        help="Path to results.npz")
    parser.add_argument("--num_variations", type=int, default=None,
                        help="Variations per sample (auto-detected from NPZ"
                             " if omitted)")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Save binned statistics as JSON")
    args = parser.parse_args()

    cache_path = Path(args.cache_pickle)
    npz_path = Path(args.results_npz)

    if not cache_path.exists():
        print(f"Error: cache file not found: {cache_path}")
        return 1
    if not npz_path.exists():
        print(f"Error: NPZ file not found: {npz_path}")
        return 1

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    gt_samples = cache.get("samples") or cache.get("ground_truth")
    if gt_samples is None:
        print(f"Error: unknown cache format (keys: {list(cache.keys())})")
        return 1

    npz = np.load(str(npz_path), allow_pickle=True)
    metric_arrays = _extract_metric_arrays(npz)

    num_variations = args.num_variations
    if num_variations is None:
        nv = npz["num_variations"].item() if "num_variations" in npz else None
        if nv:
            num_variations = int(nv)
        else:
            num_variations = len(metric_arrays.get("ged", [])) // len(gt_samples)
            if num_variations < 1:
                num_variations = 1

    model_name = "HouseGAN++" if args.model == "houseganpp" else "House-Diffusion"

    print(f"{'=' * 80}")
    print(f"  ANALYZING METRICS BY ROOM COUNT")
    print(f"{'=' * 80}")
    print(f"  Model:      {model_name}")
    print(f"  Cache:      {cache_path}")
    print(f"  NPZ:        {npz_path}")
    print(f"  Variations: {num_variations}")
    print(f"  Metrics:    {', '.join(metric_arrays.keys())}")
    print(f"  Samples:    {len(gt_samples)}")

    bin_stats = bin_metrics(gt_samples, metric_arrays, num_variations,
                           args.model)
    metric_keys = list(metric_arrays.keys())
    print_table(bin_stats, metric_keys, model_name)

    if args.output_json:
        out = Path(args.output_json)
        serialisable = {}
        for label, bs in bin_stats.items():
            serialisable[label] = {"n": bs["n"], "metrics": bs["metrics"]}
        with open(out, "w") as f:
            json.dump({
                "model": args.model,
                "model_name": model_name,
                "cache_pickle": str(cache_path),
                "results_npz": str(npz_path),
                "num_variations": num_variations,
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
