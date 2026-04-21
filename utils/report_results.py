#!/usr/bin/env python3
"""
Unified reporting: compute summary statistics, print to console, and persist
results as NPZ + TXT.

All metric arrays are accepted as optional flat 1-D NumPy arrays (may contain
NaN).  NaN values are excluded consistently from every summary statistic.

Outputs
-------
- ``{output_dir}/results.npz`` -- raw arrays + summary dict.
- ``{output_dir}/results.txt`` -- human-readable summary.
"""

from __future__ import annotations

import os

import numpy as np


# ── Stats helper ──────────────────────────────────────────────────────


def _stats(arr: np.ndarray) -> dict:
    """Summary statistics for a 1-D float array, excluding NaN."""
    valid = arr[~np.isnan(arr)]
    n_valid = int(len(valid))
    n_nan = int(len(arr)) - n_valid
    if n_valid == 0:
        return {"mean": None, "std": None, "median": None,
                "min": None, "max": None, "n_valid": 0, "n_nan": n_nan}
    return {
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid, ddof=1)) if n_valid > 1 else 0.0,
        "median": float(np.median(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "n_valid": n_valid,
        "n_nan": n_nan,
    }


def _variation_std(arr: np.ndarray, num_variations: int) -> float | None:
    """Std of per-variation column means (cross-variation consistency).

    Returns *None* when ``num_variations <= 1`` or the array length is not
    evenly divisible by *num_variations*.
    """
    if num_variations <= 1:
        return None
    valid = arr[~np.isnan(arr)]
    n = len(valid)
    if n == 0 or n % num_variations != 0:
        return None
    reshaped = valid.reshape(n // num_variations, num_variations)
    return float(np.std(np.mean(reshaped, axis=0), ddof=1))


# ── Formatting helpers ────────────────────────────────────────────────


def _safe(val, fmt=".4f"):
    if val is None:
        return "N/A"
    return f"{val:{fmt}}"


# ── Room-count accuracy helpers ───────────────────────────────────────


def _room_accuracy_summary(accuracies, room_count_accuracies,
                           room_type_accuracies, num_variations, extra):
    """Build a rich summary dict for the three room accuracy metrics."""
    acc = np.asarray(accuracies, dtype=float)
    rca = np.asarray(room_count_accuracies, dtype=float)
    rta = np.asarray(room_type_accuracies, dtype=float)
    total = extra.get("total_samples", len(acc)) if extra else len(acc)

    d = {
        "accuracy": _stats(acc),
        "room_count_accuracy": _stats(rca),
        "room_type_accuracy": _stats(rta),
    }

    if num_variations > 1:
        for key, arr in [("accuracy", acc), ("room_count_accuracy", rca),
                         ("room_type_accuracy", rta)]:
            vs = _variation_std(arr, num_variations)
            if vs is not None:
                d[key]["std_of_variation_means"] = vs

    if extra:
        dc = extra.get("disconnected_count")
        if dc is not None:
            d["disconnected_samples"] = int(dc)
            d["disconnected_rate"] = float(dc / total * 100) if total else 0.0
        hc = extra.get("hidden_count")
        if hc is not None:
            d["hidden_samples"] = int(hc)
            d["hidden_rate"] = float(hc / total * 100) if total else 0.0

    return d


# ── Console / TXT printers ───────────────────────────────────────────


def _emit_metric(write_fn, title, st, var_std=None, extra_lines=None):
    """Print a metric block to *write_fn* (``print`` or ``f.write``)."""
    if st is None or st.get("mean") is None:
        return
    write_fn(f"\n--- {title} ---")
    write_fn(f"  Mean:   {_safe(st['mean'])}")
    write_fn(f"  Std:    {_safe(st['std'])}")
    write_fn(f"  Median: {_safe(st['median'])}")
    if st.get("min") is not None and st.get("max") is not None:
        write_fn(f"  Range:  [{_safe(st['min'])}, {_safe(st['max'])}]")
    write_fn(f"  Valid:  {st['n_valid']} / {st['n_valid'] + st['n_nan']}")
    if var_std is not None:
        write_fn(f"  Std (variation means): {_safe(var_std)}")
    for line in (extra_lines or []):
        write_fn(f"  {line}")


def _emit_room_accuracies(write_fn, rc):
    if not rc:
        return
    for key, title in [("accuracy", "Room Accuracy (combined)"),
                       ("room_count_accuracy", "Room Count Accuracy"),
                       ("room_type_accuracy", "Room Type Accuracy")]:
        st = rc.get(key)
        if st and st.get("mean") is not None:
            _emit_metric(write_fn, title, st,
                         var_std=st.get("std_of_variation_means"))
    if rc.get("disconnected_samples") is not None:
        write_fn(f"  Disconnected: {rc['disconnected_samples']:,}"
                 f" ({rc['disconnected_rate']:.1f}%)")
    if rc.get("hidden_samples") is not None:
        write_fn(f"  Hidden rooms: {rc['hidden_samples']:,}"
                 f" ({rc['hidden_rate']:.1f}%)")


def _similarity_extras(extra):
    if not extra:
        return None
    lines = []
    mp = extra.get("matched_room_pairs")
    if mp is not None:
        lines.append(f"Matched pairs:  {int(sum(mp)):,}")
    up = extra.get("unmatched_pred_rooms")
    ug = extra.get("unmatched_gt_rooms")
    if up is not None and ug is not None:
        lines.append(f"Avg unmatched:  pred={float(np.mean(up)):.2f}"
                     f"  gt={float(np.mean(ug)):.2f}")
    return lines or None


# ── Public API ────────────────────────────────────────────────────────


def report_results(
    output_dir: str,
    *,
    geds: np.ndarray | None = None,
    accuracies: np.ndarray | None = None,
    room_count_accuracies: np.ndarray | None = None,
    room_type_accuracies: np.ndarray | None = None,
    similarity_scores: np.ndarray | None = None,
    area_proportion_errors: np.ndarray | None = None,
    spatial_overlap_ratios: np.ndarray | None = None,
    num_variations: int = 1,
    checkpoint_path: str | None = None,
    data_path: str | None = None,
    extra: dict | None = None,
) -> dict:
    """Compute statistics, print report, and write NPZ + TXT.

    Every metric parameter is an optional flat 1-D array that may contain
    NaN.  NaN values are excluded from all summary statistics.

    Args:
        output_dir: Directory for ``results.npz`` and ``results.txt``.
        geds: Per-sample GED values.
        accuracies: Per-sample combined room accuracy (count and types).
        room_count_accuracies: Per-sample room-count accuracy.
        room_type_accuracies: Per-sample room-type-set accuracy.
        similarity_scores: Per-sample delta / room-similarity scores.
        area_proportion_errors: Per-sample area-proportion error.
        spatial_overlap_ratios: Per-sample spatial overlap ratio (pred only).
        num_variations: Generation variations per sample (default 1).
        checkpoint_path: Included in TXT header when provided.
        data_path: Included in TXT header when provided.
        extra: Passthrough dict saved into the NPZ.  Also used for
            ``disconnected_count``, ``hidden_count``, ``total_samples``,
            ``matched_room_pairs``, ``unmatched_pred_rooms``,
            ``unmatched_gt_rooms``, ``target_set``.

    Returns:
        The dict that was saved to the NPZ file.
    """
    extra = extra or {}
    os.makedirs(output_dir, exist_ok=True)

    # ── Compute per-metric stats ──────────────────────────────────────
    summary: dict = {}
    metric_entries: list[tuple[str, str, np.ndarray]] = []

    ged_accuracies = None
    if geds is not None:
        geds = np.asarray(geds, dtype=float)
        metric_entries.append(("geds", "GED", geds))
        ged_accuracies = np.where(np.isnan(geds), np.nan,
                                  (geds == 0).astype(float))
    if accuracies is not None:
        accuracies = np.asarray(accuracies, dtype=float)
    if room_count_accuracies is not None:
        room_count_accuracies = np.asarray(room_count_accuracies, dtype=float)
    if room_type_accuracies is not None:
        room_type_accuracies = np.asarray(room_type_accuracies, dtype=float)
    if similarity_scores is not None:
        similarity_scores = np.asarray(similarity_scores, dtype=float)
        metric_entries.append(
            ("similarity_scores", "Room Similarity (delta)", similarity_scores))
    if area_proportion_errors is not None:
        area_proportion_errors = np.asarray(area_proportion_errors, dtype=float)
        metric_entries.append(
            ("area_proportion_errors", "Area Proportion Error",
             area_proportion_errors))
    if spatial_overlap_ratios is not None:
        spatial_overlap_ratios = np.asarray(spatial_overlap_ratios, dtype=float)
        metric_entries.append(
            ("spatial_overlap_ratios", "Spatial Overlap Ratio",
             spatial_overlap_ratios))

    for npz_key, _, arr in metric_entries:
        summary[npz_key] = _stats(arr)
        vs = _variation_std(arr, num_variations)
        if vs is not None:
            summary[npz_key]["std_of_variation_means"] = vs

    if ged_accuracies is not None:
        summary["ged_accuracies"] = _stats(ged_accuracies)

    rc_summary = None
    if (accuracies is not None and room_count_accuracies is not None
            and room_type_accuracies is not None):
        rc_summary = _room_accuracy_summary(
            accuracies, room_count_accuracies, room_type_accuracies,
            num_variations, extra)
        summary["room_accuracies"] = rc_summary

    # ── Build output lines (shared by console + TXT) ──────────────────
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("  EVALUATION RESULTS")
    lines.append("=" * 80)
    if checkpoint_path:
        lines.append(f"  Checkpoint: {checkpoint_path}")
    if data_path:
        lines.append(f"  Data path:  {data_path}")
    if num_variations > 1:
        lines.append(f"  Variations: {num_variations} per sample")

    def _line(text):
        lines.append(text)

    # GED gets a custom block because of its special formatting
    if geds is not None:
        st = summary["geds"]
        n_total = len(geds)
        n_samples = n_total // num_variations if num_variations else n_total
        _line(f"\n--- GED ---")
        _line(f"  Samples:  {n_samples:,}")
        if num_variations > 1:
            _line(f"  Total:    {n_total:,} ({n_samples} x {num_variations})")
        _line(f"  Valid:    {st['n_valid']:,}")
        if st["n_nan"] > 0:
            _line(f"  Failed:   {st['n_nan']:,}"
                  f" ({st['n_nan'] / n_total * 100:.1f}%)")
        _line(f"  Mean:     {_safe(st['mean'])}")
        _line(f"  Std:      {_safe(st['std'])}")
        _line(f"  Median:   {_safe(st['median'])}")
        if st["min"] is not None and st["max"] is not None:
            _line(f"  Range:    [{_safe(st['min'], '.0f')}, "
                  f"{_safe(st['max'], '.0f')}]")
        vs = summary["geds"].get("std_of_variation_means")
        if vs is not None:
            cv = (vs / st["mean"] * 100) if st["mean"] else 0
            _line(f"  Std (variation means): {_safe(vs)}  (CV: {cv:.2f}%)")
        ga_st = summary.get("ged_accuracies")
        if ga_st and ga_st.get("mean") is not None:
            _line(f"  GED Accuracy (GED=0 rate): {ga_st['mean'] * 100:.1f}%"
                  f"  ({ga_st['n_valid']} valid samples)")

    # Room accuracies
    if rc_summary:
        _emit_room_accuracies(_line, rc_summary)

    # Generic metrics (similarity, area proportion error)
    for npz_key, title, _arr in metric_entries:
        if npz_key == "geds":
            continue
        st = summary[npz_key]
        extra_lines = None
        if npz_key == "similarity_scores":
            extra_lines = _similarity_extras(extra)
        _emit_metric(_line, title, st,
                     var_std=st.get("std_of_variation_means"),
                     extra_lines=extra_lines)

    # ── Console ───────────────────────────────────────────────────────
    for line in lines:
        print(line)

    # ── Save NPZ ──────────────────────────────────────────────────────
    save_dict: dict = {"num_variations": num_variations, "summary": summary}
    if geds is not None:
        save_dict["geds"] = geds
    if ged_accuracies is not None:
        save_dict["ged_accuracies"] = ged_accuracies
    if accuracies is not None:
        save_dict["accuracies"] = accuracies
    if room_count_accuracies is not None:
        save_dict["room_count_accuracies"] = room_count_accuracies
    if room_type_accuracies is not None:
        save_dict["room_type_accuracies"] = room_type_accuracies
    if similarity_scores is not None:
        save_dict["similarity_scores"] = similarity_scores
    if area_proportion_errors is not None:
        save_dict["area_proportion_errors"] = area_proportion_errors
    if spatial_overlap_ratios is not None:
        save_dict["spatial_overlap_ratios"] = spatial_overlap_ratios
    for k, v in extra.items():
        if k not in save_dict:
            save_dict[k] = v

    npz_path = os.path.join(output_dir, "results.npz")
    np.savez(npz_path, **save_dict)
    print(f"\n  Results saved to: {npz_path}")

    # ── Save TXT ──────────────────────────────────────────────────────
    txt_path = os.path.join(output_dir, "results.txt")
    with open(txt_path, "w") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"  Summary saved to: {txt_path}")

    print("=" * 80)
    print("  Evaluation complete!")

    return save_dict
