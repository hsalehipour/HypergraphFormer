#!/usr/bin/env python3
"""
Unified reporting: compute summary statistics, print to console, and persist
results as NPZ + TXT.

All metrics are passed via a single ``metrics`` dict mapping NPZ keys to
``(display_title, 1-D array)`` tuples.  Adding a new metric requires **no
changes** to this module -- just add an entry to the dict at the call site.

Special formatting is applied automatically for well-known keys:

- ``"geds"`` -- custom GED block with failure rate, plus a derived
  ``ged_accuracies`` (GED == 0) array.
- ``"accuracies"`` + ``"room_count_accuracies"`` + ``"room_type_accuracies"``
  -- grouped room-accuracy summary with disconnected/hidden counts.
- ``"similarity_scores"`` -- extra lines for matched/unmatched room pairs.

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

    Returns *None* when ``num_variations < 2`` or the total array length
    is not evenly divisible by *num_variations*.  NaN values are handled
    via ``nanmean`` so that a few failed samples don't break the reshape.
    """
    if num_variations < 2:
        return None
    n = len(arr)
    if n == 0 or n % num_variations != 0:
        return None
    reshaped = arr.reshape(n // num_variations, num_variations)
    col_means = np.nanmean(reshaped, axis=0)
    if np.any(np.isnan(col_means)):
        return None
    return float(np.std(col_means, ddof=1))


# ── Formatting helpers ────────────────────────────────────────────────


def _safe(val, fmt=".4f"):
    if val is None:
        return "N/A"
    return f"{val:{fmt}}"


# ── Room-accuracy helpers ────────────────────────────────────────────

_ROOM_ACC_KEYS = ("accuracies", "room_count_accuracies", "room_type_accuracies")


def _room_accuracy_summary(stats_by_key, num_variations, extra):
    """Build a rich summary dict for the three room accuracy metrics.

    ``stats_by_key`` maps NPZ key -> pre-computed ``_stats()`` dict (with
    an optional ``std_of_variation_means`` entry already attached).
    """
    total = extra.get("total_samples") if extra else None
    if total is None:
        st = stats_by_key.get("accuracies")
        total = (st["n_valid"] + st["n_nan"]) if st else 0

    d = {
        "accuracy": stats_by_key["accuracies"],
        "room_count_accuracy": stats_by_key["room_count_accuracies"],
        "room_type_accuracy": stats_by_key["room_type_accuracies"],
    }

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
        total = int(np.nansum(np.asarray(mp, dtype=float)))
        lines.append(f"Matched pairs:  {total:,}")
    up = extra.get("unmatched_pred_rooms")
    ug = extra.get("unmatched_gt_rooms")
    if up is not None and ug is not None:
        lines.append(f"Avg unmatched:  pred={float(np.nanmean(up)):.2f}"
                     f"  gt={float(np.nanmean(ug)):.2f}")
    return lines or None


# ── Public API ────────────────────────────────────────────────────────


def report_results(
    output_dir: str,
    *,
    metrics: dict[str, tuple[str, np.ndarray]] | None = None,
    room_counts: np.ndarray | None = None,
    num_variations: int = 1,
    checkpoint_path: str | None = None,
    data_path: str | None = None,
    extra: dict | None = None,
) -> dict:
    """Compute statistics, print report, and write NPZ + TXT.

    Args:
        output_dir: Directory for ``results.npz`` and ``results.txt``.
        metrics: All metrics.  Each entry maps an NPZ key to a
            ``(display_title, array)`` tuple.  Well-known keys trigger
            special formatting (see module docstring).  Example::

                metrics={
                    "geds": ("GED", ged_array),
                    "accuracies": ("Room Accuracy", acc_array),
                    "room_count_accuracies": ("Room Count Accuracy", rca_array),
                    "room_type_accuracies": ("Room Type Accuracy", rta_array),
                    "similarity_scores": ("Room Similarity (delta)", sim_array),
                }
        room_counts: Per-sample GT room count (1-D, one entry per unique
            sample, **not** expanded by *num_variations*).  Saved as
            ``room_counts`` in the NPZ for downstream per-room-count
            analysis.
        num_variations: Generation variations per sample (default 1).
            Must be >= 1.
        checkpoint_path: Included in TXT header when provided.
        data_path: Included in TXT header when provided.
        extra: Passthrough dict saved into the NPZ.  Also used for
            ``disconnected_count``, ``hidden_count``, ``total_samples``,
            ``matched_room_pairs``, ``unmatched_pred_rooms``,
            ``unmatched_gt_rooms``, ``target_set``.

    Returns:
        The dict that was saved to the NPZ file.

    Raises:
        ValueError: If *num_variations* < 1 or if an *extra* key
            collides with a metric key or reserved key.
    """
    if num_variations < 1:
        raise ValueError(f"num_variations must be >= 1, got {num_variations}")

    extra = extra or {}
    metrics = metrics or {}
    os.makedirs(output_dir, exist_ok=True)

    # ── Resolve all arrays ────────────────────────────────────────────
    resolved: dict[str, np.ndarray] = {}
    titles: dict[str, str] = {}
    for npz_key, (title, arr) in metrics.items():
        resolved[npz_key] = np.asarray(arr, dtype=float)
        titles[npz_key] = title

    # ── Derived metric: ged_accuracies ────────────────────────────────
    if "geds" in resolved:
        geds = resolved["geds"]
        ged_acc = np.where(np.isnan(geds), np.nan, (geds == 0).astype(float))
        resolved["ged_accuracies"] = ged_acc
        titles["ged_accuracies"] = "GED Accuracy"

    # ── Compute per-metric stats (single pass) ────────────────────────
    summary: dict = {}
    for npz_key, arr in resolved.items():
        summary[npz_key] = _stats(arr)
        vs = _variation_std(arr, num_variations)
        if vs is not None:
            summary[npz_key]["std_of_variation_means"] = vs

    # ── Room accuracy grouped summary (reuses pre-computed stats) ─────
    rc_summary = None
    if all(k in resolved for k in _ROOM_ACC_KEYS):
        rc_summary = _room_accuracy_summary(summary, num_variations, extra)
        summary["room_accuracies"] = rc_summary

    # ── Ordered list for generic display (excludes special-cased keys) ─
    _SPECIAL_KEYS = {"geds", "ged_accuracies", *_ROOM_ACC_KEYS}
    generic_keys = [k for k in resolved if k not in _SPECIAL_KEYS]

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

    # GED gets a custom block
    if "geds" in resolved:
        geds = resolved["geds"]
        st = summary["geds"]
        n_total = len(geds)
        n_samples = n_total // num_variations
        _line(f"\n--- {titles['geds']} ---")
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
        vs = st.get("std_of_variation_means")
        if vs is not None:
            mean = st["mean"]
            cv = (vs / mean * 100) if mean is not None and mean != 0 else None
            cv_str = f"  (CV: {cv:.2f}%)" if cv is not None else ""
            _line(f"  Std (variation means): {_safe(vs)}{cv_str}")
        ga_st = summary.get("ged_accuracies")
        if ga_st and ga_st.get("mean") is not None:
            _line(f"  GED Accuracy (GED=0 rate): {ga_st['mean'] * 100:.1f}%"
                  f"  ({ga_st['n_valid']} valid samples)")

    # Room accuracies (grouped)
    if rc_summary:
        _emit_room_accuracies(_line, rc_summary)

    # All other metrics (generic)
    for npz_key in generic_keys:
        st = summary[npz_key]
        extra_lines = None
        if npz_key == "similarity_scores":
            extra_lines = _similarity_extras(extra)
        _emit_metric(_line, titles[npz_key], st,
                     var_std=st.get("std_of_variation_means"),
                     extra_lines=extra_lines)

    # ── Console ───────────────────────────────────────────────────────
    for line in lines:
        print(line)

    # ── Save NPZ ──────────────────────────────────────────────────────
    reserved_keys = {"num_variations", "summary", "room_counts"}
    save_dict: dict = {"num_variations": num_variations, "summary": summary}
    if room_counts is not None:
        save_dict["room_counts"] = np.asarray(room_counts, dtype=int)
    for npz_key, arr in resolved.items():
        save_dict[npz_key] = arr

    collisions = set(extra) & (set(save_dict) | reserved_keys)
    if collisions:
        raise ValueError(
            f"extra keys collide with metric/reserved keys: {collisions}")
    for k, v in extra.items():
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
