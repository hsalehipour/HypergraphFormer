#!/usr/bin/env python3
"""
Shared utilities for scoring and saving benchmark results.

This module provides consistent post-processing of evaluation results
(GED, room accuracy, similarity, area difference) across different model
evaluation scripts (HouseGAN++, House-Diffusion, etc.).

All console and text-file reporting is centralised here so that the
individual evaluation scripts only need to build the metric dicts and
pass them through ``additional_info``.
"""

import os
import numpy as np


# ── Helpers ──────────────────────────────────────────────────────────


def _safe_float(val, fmt=".4f"):
    """Format a value that may be ``None``."""
    if val is None:
        return "N/A"
    return f"{val:{fmt}}"


def _print_metric_block(title, d, extra_lines=None):
    """Print a standard metric block from a summary dict.

    Expected keys: mean, median, std, min, max, valid_count, valid_rate,
    std_of_variation_means (all optional).
    """
    if d is None:
        return
    mean = d.get("mean")
    if mean is None:
        print(f"\n  (no valid {title} data)")
        return

    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print("=" * 80)
    print(f"  Mean:            {_safe_float(mean)}")
    if d.get("median") is not None:
        print(f"  Median:          {_safe_float(d['median'])}")
    if d.get("std") is not None:
        print(f"  STD:             {_safe_float(d['std'])}")
    if d.get("min") is not None and d.get("max") is not None:
        print(f"  Range:           [{_safe_float(d['min'])}, {_safe_float(d['max'])}]")
    vc = d.get("valid_count")
    vr = d.get("valid_rate")
    if vc is not None:
        rate_str = f" ({vr:.1f}%)" if vr is not None else ""
        print(f"  Valid:           {vc:,}{rate_str}")
    svm = d.get("std_of_variation_means")
    if svm is not None:
        print(f"  Std (var means): {_safe_float(svm)}")
    if extra_lines:
        for line in extra_lines:
            print(f"  {line}")


def _write_metric_block(f, title, d, extra_lines=None):
    """Write a standard metric block to a text file."""
    if d is None:
        return
    mean = d.get("mean")
    if mean is None:
        return

    f.write(f"\n--- {title} ---\n")
    f.write(f"Mean: {_safe_float(mean)}\n")
    if d.get("median") is not None:
        f.write(f"Median: {_safe_float(d['median'])}\n")
    if d.get("std") is not None:
        f.write(f"STD: {_safe_float(d['std'])}\n")
    if d.get("min") is not None and d.get("max") is not None:
        f.write(f"Range: [{_safe_float(d['min'])}, {_safe_float(d['max'])}]\n")
    vc = d.get("valid_count")
    vr = d.get("valid_rate")
    if vc is not None:
        rate_str = f" ({vr:.1f}%)" if vr is not None else ""
        f.write(f"Valid: {vc}{rate_str}\n")
    svm = d.get("std_of_variation_means")
    if svm is not None:
        f.write(f"Std (var means): {_safe_float(svm)}\n")
    if extra_lines:
        for line in extra_lines:
            f.write(f"{line}\n")


# ── Room Count Accuracy ─────────────────────────────────────────────


def _print_room_count_accuracy(rc):
    if rc is None or not isinstance(rc, dict):
        return

    print(f"\n{'=' * 80}")
    print("  Room Count Accuracy")
    print("=" * 80)
    print(f"  Exact match:     {rc.get('exact_match_rate', 0):.1f}%"
          f"  (std: {_safe_float(rc.get('accuracy_std', 0))})")
    print(f"  Count error:     mean={rc.get('mean_count_error', 0):.3f}"
          f"  median={rc.get('median_count_error', 0):.1f}"
          f"  max={rc.get('max_count_error', 0):.0f}"
          f"  (std: {_safe_float(rc.get('count_error_std', 0))})")
    print(f"  Type error:      mean={rc.get('mean_type_error', 0):.3f}"
          f"  median={rc.get('median_type_error', 0):.1f}"
          f"  max={rc.get('max_type_error', 0):.0f}"
          f"  (std: {_safe_float(rc.get('type_error_std', 0))})")

    disc = rc.get("disconnected_samples")
    disc_rate = rc.get("disconnected_rate")
    hidden = rc.get("hidden_samples")
    hidden_rate = rc.get("hidden_rate")

    if disc is not None:
        print(f"  Disconnected:    {disc:,} ({disc_rate:.1f}%)")
    if hidden is not None:
        print(f"  Hidden rooms:    {hidden:,} ({hidden_rate:.1f}%)")


def _write_room_count_accuracy(f, rc):
    if rc is None or not isinstance(rc, dict):
        return

    f.write(f"\n--- Room Count Accuracy ---\n")
    f.write(f"Exact match: {rc.get('exact_match_rate', 0):.1f}%"
            f"  (std: {_safe_float(rc.get('accuracy_std', 0))})\n")
    f.write(f"Count error: mean={rc.get('mean_count_error', 0):.3f}"
            f"  median={rc.get('median_count_error', 0):.1f}"
            f"  max={rc.get('max_count_error', 0):.0f}\n")
    f.write(f"Type error:  mean={rc.get('mean_type_error', 0):.3f}"
            f"  median={rc.get('median_type_error', 0):.1f}"
            f"  max={rc.get('max_type_error', 0):.0f}\n")


# ── Room Similarity ─────────────────────────────────────────────────


def _room_similarity_extra(rs):
    """Build extra lines for room-matching statistics."""
    lines = []
    mp = rs.get("total_matched_pairs")
    if mp is not None:
        lines.append(f"Matched pairs:   {mp:,}")
    up = rs.get("avg_unmatched_pred")
    ug = rs.get("avg_unmatched_gt")
    if up is not None and ug is not None:
        lines.append(f"Avg unmatched:   pred={up:.2f}  gt={ug:.2f}")
    return lines or None


# ── Main entry point ────────────────────────────────────────────────


def score_benchmark_result(all_geds, num_variations, output_dir,
                           checkpoint_path=None, data_path=None,
                           additional_info=None):
    """Compute GED statistics, print all metrics, and save to NPZ + text.

    This function provides consistent processing across all evaluation
    scripts:

    - Handles NaN values from timeouts/failures
    - Computes std of variation means (consistency metric)
    - Prints comprehensive statistics for **all** metrics passed through
      *additional_info* (room count accuracy, room similarity, area
      difference)
    - Saves results in standardised NPZ format

    Args:
        all_geds: numpy array of all GED values (may contain np.nan).
        num_variations: number of generation variations per sample.
        output_dir: directory to save results.
        checkpoint_path: optional path to model checkpoint (for text file).
        data_path: optional path to data file (for text file).
        additional_info: optional dict with extra metric dicts to save.
            Recognised keys: ``room_count_accuracy``,
            ``room_similarity``, ``area_difference``.

    Returns:
        dict -- statistics dictionary saved to the NPZ file.
    """
    additional_info = additional_info or {}

    # ── GED ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  GED RESULTS")
    print("=" * 80)

    if len(all_geds) == 0:
        print("  No valid GED computations")
        return None

    all_geds = np.array(all_geds)

    num_valid = int(np.sum(~np.isnan(all_geds)))
    num_failed = int(np.sum(np.isnan(all_geds)))

    print(f"  Samples:         {len(all_geds) // num_variations:,}")
    if num_variations > 1:
        print(f"  Variations:      {num_variations} per sample ({len(all_geds):,} total)")
    print(f"  Valid:           {num_valid:,}")
    if num_failed > 0:
        print(f"  Failed:          {num_failed:,} ({num_failed / len(all_geds) * 100:.1f}%)")

    mean_ged = float(np.nanmean(all_geds))
    median_ged = float(np.nanmedian(all_geds))
    min_ged = float(np.nanmin(all_geds))
    max_ged = float(np.nanmax(all_geds))

    print(f"\n  Mean GED:        {mean_ged:.4f}")
    print(f"  Median GED:      {median_ged:.4f}")
    print(f"  GED range:       [{min_ged:.0f}, {max_ged:.0f}]")

    std_of_variation_means = 0.0
    if num_variations > 1:
        num_samples = len(all_geds) // num_variations
        geds_reshaped = all_geds.reshape(num_samples, num_variations)
        means_per_variation = np.nanmean(geds_reshaped, axis=0)
        std_of_variation_means = float(np.nanstd(means_per_variation, ddof=1))
        cv = (std_of_variation_means / mean_ged * 100) if mean_ged else 0
        print(f"  Std (var means): {std_of_variation_means:.4f}  (CV: {cv:.2f}%)")

    # ── Additional metrics (console) ─────────────────────────────────
    rc = additional_info.get("room_count_accuracy")
    _print_room_count_accuracy(rc)

    rs = additional_info.get("room_similarity")
    _print_metric_block("Room Similarity (delta)", rs,
                        extra_lines=_room_similarity_extra(rs) if rs else None)

    ad = additional_info.get("area_difference")
    _print_metric_block("Area Difference (relative room-area error)", ad)

    # ── Save NPZ ─────────────────────────────────────────────────────
    results_file = os.path.join(output_dir, "benchmark_results.npz")
    save_dict = {
        "geds": all_geds,
        "mean": mean_ged,
        "std": std_of_variation_means,
        "median": median_ged,
        "num_valid": num_valid,
        "num_failed": num_failed,
    }
    save_dict.update(additional_info)

    np.savez(results_file, **save_dict)
    print(f"\n  Results saved to: {results_file}")

    # ── Save text summary ────────────────────────────────────────────
    if checkpoint_path or data_path:
        text_file = os.path.join(output_dir, "benchmark_results.txt")
        with open(text_file, "w") as f:
            f.write("Benchmark Results\n")
            f.write("=" * 80 + "\n\n")

            if checkpoint_path:
                f.write(f"Checkpoint: {checkpoint_path}\n")
            if data_path:
                f.write(f"Data path: {data_path}\n")

            f.write(f"\n--- GED ---\n")
            f.write(f"Samples: {len(all_geds) // num_variations}\n")
            f.write(f"Total computations: {len(all_geds)}"
                    f" ({len(all_geds) // num_variations} x {num_variations})\n")
            f.write(f"Valid: {num_valid}\n")
            if num_failed > 0:
                f.write(f"Failed: {num_failed}"
                        f" ({num_failed / len(all_geds) * 100:.1f}%)\n")
            f.write(f"\nMean GED: {mean_ged:.4f}\n")
            f.write(f"Median GED: {median_ged:.4f}\n")
            f.write(f"GED range: [{min_ged:.0f}, {max_ged:.0f}]\n")
            if num_variations > 1:
                f.write(f"Std (var means): {std_of_variation_means:.4f}\n")

            _write_room_count_accuracy(f, rc)
            _write_metric_block(f, "Room Similarity (delta)", rs)
            _write_metric_block(f, "Area Difference", ad)

        print(f"  Summary saved to: {text_file}")

    print("=" * 80)
    print("  Evaluation complete!")

    return save_dict
