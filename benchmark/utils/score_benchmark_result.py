#!/usr/bin/env python3
"""
Compute, report, and persist benchmark evaluation metrics.

Accepts raw per-sample score arrays from evaluation scripts and handles
all statistics (mean, median, std, variation consistency, etc.),
console output, text-file summaries, and NPZ serialisation.

Metrics handled:

- **GED** (Graph Edit Distance) -- passed as a top-level argument.
- **Room Count Accuracy** -- per-sample accuracy, count-error, and
  type-error arrays plus a disconnected/hidden count.
- **Room Similarity (delta)** -- per-sample scores with optional
  room-matching metadata (unmatched counts, matched pairs).
- **Area Difference** -- per-sample relative room-area error scores.
"""

import os
import numpy as np


# ── Internal helpers ─────────────────────────────────────────────────


def _safe(val, fmt=".4f"):
    if val is None:
        return "N/A"
    return f"{val:{fmt}}"


def _summarise_scores(scores, num_variations):
    """Return a summary dict from a 1-D score array (may contain NaN)."""
    scores = np.asarray(scores, dtype=float)
    valid = scores[~np.isnan(scores)]
    total = len(scores)

    if len(valid) == 0:
        return None

    d = {
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "std": float(np.std(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "valid_count": int(len(valid)),
        "valid_rate": float(len(valid) / total * 100) if total else 0.0,
        "all_scores": scores.tolist(),
    }

    if num_variations > 1:
        n = len(valid) // num_variations
        if n > 0 and len(valid) % num_variations == 0:
            reshaped = valid.reshape(n, num_variations)
            d["std_of_variation_means"] = float(
                np.std(np.mean(reshaped, axis=0), ddof=1)
            )

    return d


def _build_room_count_dict(accuracies, count_errors, type_errors,
                           num_variations, disconnected_count=None,
                           hidden_count=None, total_samples=None):
    """Build the room-count-accuracy summary dict from raw arrays."""
    accuracies = np.asarray(accuracies, dtype=float)
    count_errors = np.asarray(count_errors, dtype=float)
    type_errors = np.asarray(type_errors, dtype=float)
    total = total_samples or len(accuracies)

    d = {
        "exact_match_rate": float(np.mean(accuracies) * 100),
        "mean_count_error": float(np.mean(count_errors)),
        "median_count_error": float(np.median(count_errors)),
        "max_count_error": float(np.max(count_errors)),
        "mean_type_error": float(np.mean(type_errors)),
        "median_type_error": float(np.median(type_errors)),
        "max_type_error": float(np.max(type_errors)),
        "all_accuracies": accuracies.tolist(),
        "all_count_errors": count_errors.tolist(),
        "all_type_errors": type_errors.tolist(),
    }

    acc_std = cnt_std = typ_std = 0.0
    if num_variations > 1:
        n = len(accuracies) // num_variations
        if n > 0 and len(accuracies) % num_variations == 0:
            acc_std = float(np.std(np.mean(accuracies.reshape(n, num_variations), axis=0), ddof=1))
            cnt_std = float(np.std(np.mean(count_errors.reshape(n, num_variations), axis=0), ddof=1))
            typ_std = float(np.std(np.mean(type_errors.reshape(n, num_variations), axis=0), ddof=1))

    d["accuracy_std"] = acc_std
    d["count_error_std"] = cnt_std
    d["type_error_std"] = typ_std

    if disconnected_count is not None:
        d["disconnected_samples"] = int(disconnected_count)
        d["disconnected_rate"] = float(disconnected_count / total * 100) if total else 0.0
    if hidden_count is not None:
        d["hidden_samples"] = int(hidden_count)
        d["hidden_rate"] = float(hidden_count / total * 100) if total else 0.0

    return d


def _build_similarity_dict(scores, num_variations,
                           matched_room_pairs=None,
                           unmatched_pred_rooms=None,
                           unmatched_gt_rooms=None):
    """Build the room-similarity summary dict from raw arrays."""
    d = _summarise_scores(scores, num_variations)
    if d is None:
        return None

    if matched_room_pairs is not None:
        d["total_matched_pairs"] = int(sum(matched_room_pairs))
    if unmatched_pred_rooms is not None:
        d["total_unmatched_pred"] = int(sum(unmatched_pred_rooms))
        d["avg_unmatched_pred"] = float(np.mean(unmatched_pred_rooms))
    if unmatched_gt_rooms is not None:
        d["total_unmatched_gt"] = int(sum(unmatched_gt_rooms))
        d["avg_unmatched_gt"] = float(np.mean(unmatched_gt_rooms))

    return d


# ── Console / text-file printers ────────────────────────────────────


def _print_metric(title, d, extra_lines=None):
    if d is None:
        return
    if d.get("mean") is None:
        print(f"\n  (no valid {title} data)")
        return

    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print("=" * 80)
    print(f"  Mean:            {_safe(d['mean'])}")
    if d.get("median") is not None:
        print(f"  Median:          {_safe(d['median'])}")
    if d.get("std") is not None:
        print(f"  STD:             {_safe(d['std'])}")
    if d.get("min") is not None and d.get("max") is not None:
        print(f"  Range:           [{_safe(d['min'])}, {_safe(d['max'])}]")
    vc = d.get("valid_count")
    vr = d.get("valid_rate")
    if vc is not None:
        rate_str = f" ({vr:.1f}%)" if vr is not None else ""
        print(f"  Valid:           {vc:,}{rate_str}")
    svm = d.get("std_of_variation_means")
    if svm is not None:
        print(f"  Std (var means): {_safe(svm)}")
    for line in (extra_lines or []):
        print(f"  {line}")


def _print_room_count(rc):
    if not rc:
        return
    print(f"\n{'=' * 80}")
    print("  Room Count Accuracy")
    print("=" * 80)
    print(f"  Exact match:     {rc['exact_match_rate']:.1f}%"
          f"  (std: {_safe(rc.get('accuracy_std', 0))})")
    print(f"  Count error:     mean={rc['mean_count_error']:.3f}"
          f"  median={rc['median_count_error']:.1f}"
          f"  max={rc['max_count_error']:.0f}"
          f"  (std: {_safe(rc.get('count_error_std', 0))})")
    print(f"  Type error:      mean={rc['mean_type_error']:.3f}"
          f"  median={rc['median_type_error']:.1f}"
          f"  max={rc['max_type_error']:.0f}"
          f"  (std: {_safe(rc.get('type_error_std', 0))})")
    if rc.get("disconnected_samples") is not None:
        print(f"  Disconnected:    {rc['disconnected_samples']:,}"
              f" ({rc['disconnected_rate']:.1f}%)")
    if rc.get("hidden_samples") is not None:
        print(f"  Hidden rooms:    {rc['hidden_samples']:,}"
              f" ({rc['hidden_rate']:.1f}%)")


def _write_metric(f, title, d):
    if not d or d.get("mean") is None:
        return
    f.write(f"\n--- {title} ---\n")
    f.write(f"Mean: {_safe(d['mean'])}\n")
    if d.get("median") is not None:
        f.write(f"Median: {_safe(d['median'])}\n")
    if d.get("std") is not None:
        f.write(f"STD: {_safe(d['std'])}\n")
    if d.get("min") is not None and d.get("max") is not None:
        f.write(f"Range: [{_safe(d['min'])}, {_safe(d['max'])}]\n")
    vc = d.get("valid_count")
    vr = d.get("valid_rate")
    if vc is not None:
        rate_str = f" ({vr:.1f}%)" if vr is not None else ""
        f.write(f"Valid: {vc}{rate_str}\n")
    svm = d.get("std_of_variation_means")
    if svm is not None:
        f.write(f"Std (var means): {_safe(svm)}\n")


def _write_room_count(f, rc):
    if not rc:
        return
    f.write(f"\n--- Room Count Accuracy ---\n")
    f.write(f"Exact match: {rc['exact_match_rate']:.1f}%"
            f"  (std: {_safe(rc.get('accuracy_std', 0))})\n")
    f.write(f"Count error: mean={rc['mean_count_error']:.3f}"
            f"  median={rc['median_count_error']:.1f}"
            f"  max={rc['max_count_error']:.0f}\n")
    f.write(f"Type error:  mean={rc['mean_type_error']:.3f}"
            f"  median={rc['median_type_error']:.1f}"
            f"  max={rc['max_type_error']:.0f}\n")


# ── Similarity extra lines ──────────────────────────────────────────


def _similarity_extras(rs):
    lines = []
    if rs.get("total_matched_pairs") is not None:
        lines.append(f"Matched pairs:   {rs['total_matched_pairs']:,}")
    up = rs.get("avg_unmatched_pred")
    ug = rs.get("avg_unmatched_gt")
    if up is not None and ug is not None:
        lines.append(f"Avg unmatched:   pred={up:.2f}  gt={ug:.2f}")
    return lines or None


# ── Public API ───────────────────────────────────────────────────────


def score_benchmark_result(all_geds, num_variations, output_dir,
                           checkpoint_path=None, data_path=None,
                           additional_info=None):
    """Compute statistics for all metrics, print results, and save to disk.

    Evaluation scripts collect raw per-sample arrays and pass them here
    via *additional_info*.  This function computes every summary
    statistic, prints a formatted report to the console, writes an
    optional text summary, and persists everything to
    ``benchmark_results.npz``.

    Args:
        all_geds: 1-D array of GED values (may contain ``np.nan``).
        num_variations: number of generation variations per sample.
        output_dir: directory for output files.
        checkpoint_path: model checkpoint path (written to text summary).
        data_path: evaluation data path (written to text summary).
        additional_info: dict with raw metric data.  Recognised keys:

            ``room_count_accuracy``
                Dict with lists ``accuracies``, ``count_errors``,
                ``type_errors`` and an int ``disconnected_count`` or
                ``hidden_count`` plus ``total_samples``.
                **Or** a pre-computed summary dict (backwards compat).

            ``room_similarity``
                Dict with list ``scores`` and optional lists
                ``matched_room_pairs``, ``unmatched_pred_rooms``,
                ``unmatched_gt_rooms``.
                **Or** a pre-computed summary dict.

            ``area_difference``
                Dict with list ``scores``.
                **Or** a pre-computed summary dict.

    Returns:
        dict saved to the NPZ file.
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
        print(f"  Variations:      {num_variations} per sample"
              f" ({len(all_geds):,} total)")
    print(f"  Valid:           {num_valid:,}")
    if num_failed > 0:
        print(f"  Failed:          {num_failed:,}"
              f" ({num_failed / len(all_geds) * 100:.1f}%)")

    mean_ged = float(np.nanmean(all_geds))
    median_ged = float(np.nanmedian(all_geds))
    min_ged = float(np.nanmin(all_geds))
    max_ged = float(np.nanmax(all_geds))

    print(f"\n  Mean GED:        {mean_ged:.4f}")
    print(f"  Median GED:      {median_ged:.4f}")
    print(f"  GED range:       [{min_ged:.0f}, {max_ged:.0f}]")

    std_of_variation_means = 0.0
    if num_variations > 1:
        n = len(all_geds) // num_variations
        reshaped = all_geds.reshape(n, num_variations)
        means_pv = np.nanmean(reshaped, axis=0)
        std_of_variation_means = float(np.nanstd(means_pv, ddof=1))
        cv = (std_of_variation_means / mean_ged * 100) if mean_ged else 0
        print(f"  Std (var means): {std_of_variation_means:.4f}  (CV: {cv:.2f}%)")

    # ── Build summary dicts from raw arrays ──────────────────────────
    raw_rc = additional_info.get("room_count_accuracy")
    if raw_rc and "accuracies" in raw_rc:
        rc = _build_room_count_dict(
            raw_rc["accuracies"], raw_rc["count_errors"],
            raw_rc["type_errors"], num_variations,
            disconnected_count=raw_rc.get("disconnected_count"),
            hidden_count=raw_rc.get("hidden_count"),
            total_samples=raw_rc.get("total_samples"),
        )
    else:
        rc = raw_rc

    raw_rs = additional_info.get("room_similarity")
    if raw_rs and "scores" in raw_rs:
        rs = _build_similarity_dict(
            raw_rs["scores"], num_variations,
            matched_room_pairs=raw_rs.get("matched_room_pairs"),
            unmatched_pred_rooms=raw_rs.get("unmatched_pred_rooms"),
            unmatched_gt_rooms=raw_rs.get("unmatched_gt_rooms"),
        )
    else:
        rs = raw_rs

    raw_ad = additional_info.get("area_difference")
    if raw_ad and "scores" in raw_ad:
        ad = _summarise_scores(raw_ad["scores"], num_variations)
    else:
        ad = raw_ad

    # ── Console output ───────────────────────────────────────────────
    _print_room_count(rc)
    _print_metric("Room Similarity (delta)", rs,
                  extra_lines=_similarity_extras(rs) if rs else None)
    _print_metric("Area Difference (relative room-area error)", ad)

    # ── Save NPZ ─────────────────────────────────────────────────────
    results_file = os.path.join(output_dir, "benchmark_results.npz")
    save_dict = {
        "geds": all_geds,
        "mean": mean_ged,
        "std": std_of_variation_means,
        "median": median_ged,
        "num_valid": num_valid,
        "num_failed": num_failed,
        "num_variations": num_variations,
    }
    if rc is not None:
        save_dict["room_count_accuracy"] = rc
    if rs is not None:
        save_dict["room_similarity"] = rs
    if ad is not None:
        save_dict["area_difference"] = ad
    for k, v in additional_info.items():
        if k not in ("room_count_accuracy", "room_similarity", "area_difference"):
            save_dict[k] = v

    np.savez(results_file, **save_dict)
    print(f"\n  Results saved to: {results_file}")

    # ── Text summary ─────────────────────────────────────────────────
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

            _write_room_count(f, rc)
            _write_metric(f, "Room Similarity (delta)", rs)
            _write_metric(f, "Area Difference", ad)

        print(f"  Summary saved to: {text_file}")

    print("=" * 80)
    print("  Evaluation complete!")

    return save_dict
