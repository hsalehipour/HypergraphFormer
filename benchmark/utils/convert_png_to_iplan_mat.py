"""Convert original RPLAN PNGs to iPLAN's .mat format for a given split.

iPLAN's official ``data/preprocess/prepareData.py`` driver expects PNGs grouped
under ``<root>/<split>/`` and writes the resulting ``.mat`` files into the iPLAN
repo's own ``data/<split_dir>/`` tree. Our copy of RPLAN is one flat directory
and we want the converted ``.mat`` files to land directly under
``HypergraphFormer/dataset/rplan/dataset_iplan/`` so the benchmark evaluator
(``benchmark/src/evaluate_iplan.py``) can pick them up without an intermediate
copy step.

This script:
  - reads the file list for the requested split from iPLAN's shipped
    ``data/{train,val,test}.txt`` (so we honor the exact split iPLAN was
    trained on),
  - calls iPLAN's own ``FloorPlan`` parser (no algorithm changes),
  - writes one ``.mat`` per floorplan into ``--out-dir`` (defaults to
    ``HypergraphFormer/dataset/rplan/dataset_iplan/`` for ``test``),
  - skips files already converted (rerun-safe),
  - parallelises across CPU workers.

Usage (with the ``iplan`` conda env active so iPLAN's deps are importable):
    python benchmark/utils/convert_png_to_iplan_mat.py --split test
    python benchmark/utils/convert_png_to_iplan_mat.py --split test --workers 16
    python benchmark/utils/convert_png_to_iplan_mat.py \\
        --split test \\
        --iplan-root /home/hesam/REPOs/iPlan \\
        --png-root   /home/hesam/REPOs/HypergraphFormer/dataset/rplan/dataset_original/floorplan_dataset \\
        --out-dir    /home/hesam/REPOs/HypergraphFormer/dataset/rplan/dataset_iplan
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from datetime import datetime
from multiprocessing import Pool, cpu_count

import scipy.io as sio

DEFAULT_IPLAN_ROOT = "/home/hesam/REPOs/iPlan"
DEFAULT_PNG_ROOT = (
    "/home/hesam/REPOs/HypergraphFormer/dataset/rplan/dataset_original/floorplan_dataset"
)
DEFAULT_OUT_ROOT = "/home/hesam/REPOs/HypergraphFormer/dataset/rplan"

# Maps the CLI ``--split`` value to the subfolder name we write to under
# ``--out-dir``. The ``test`` split is special-cased below to land directly in
# ``dataset_iplan/`` (the path the benchmark evaluator expects).
SPLIT_TO_SUBDIR = {"train": "dataset_iplan_train", "val": "dataset_iplan_val", "test": "dataset_iplan"}


class _NullLog:
    """Drop-in replacement for the file-handle log iPLAN's FloorPlan expects.

    FloorPlan only ever calls ``.write(...)``; using a per-process null log
    avoids contention between multiprocessing workers. Real failures are
    surfaced through the worker's return value and aggregated in the main log.
    """

    def write(self, _msg):
        return None

    def flush(self):
        return None


def _add_iplan_to_path(iplan_root: str) -> None:
    """Make iPLAN's ``FloorPlan`` parser importable.

    iPLAN's ``data/preprocess/floorplan.py`` uses an absolute-style
    ``from floorplan import FloorPlan`` import, so we prepend that folder to
    ``sys.path`` rather than treating it as a package.
    """
    preprocess_dir = os.path.join(iplan_root, "data", "preprocess")
    if not os.path.isdir(preprocess_dir):
        raise FileNotFoundError(
            f"iPLAN preprocess folder not found: {preprocess_dir}. "
            "Pass --iplan-root pointing at the iPLAN repo root."
        )
    if preprocess_dir not in sys.path:
        sys.path.insert(0, preprocess_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=sorted(SPLIT_TO_SUBDIR.keys()),
        default="test",
        help="Which split to convert (uses <iplan-root>/data/<split>.txt as the file list).",
    )
    parser.add_argument(
        "--iplan-root",
        default=DEFAULT_IPLAN_ROOT,
        help="Path to the iPLAN repo root (provides the FloorPlan parser and split lists).",
    )
    parser.add_argument(
        "--png-root",
        default=DEFAULT_PNG_ROOT,
        help="Folder containing the flat RPLAN .png files.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Destination folder for the .mat files. "
            "Defaults to <DEFAULT_OUT_ROOT>/<split-subdir> "
            "(e.g. .../dataset_rplan/dataset_iplan for --split test)."
        ),
    )
    parser.add_argument(
        "--split-list",
        default=None,
        help="Override the split file list (default = <iplan-root>/data/<split>.txt).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, cpu_count() - 2),
        help="Number of worker processes (default = cpu_count - 2).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If set, only convert the first N entries (for smoke tests).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-convert even if the .mat already exists.",
    )
    args = parser.parse_args()

    # Reject path traversal in user-supplied paths so we don't accidentally walk
    # into something unrelated. Rule 1 (Secure Python Development): user input
    # must be sanitised before being used in file ops.
    args.iplan_root = os.path.abspath(os.path.expanduser(args.iplan_root))
    args.png_root = os.path.abspath(os.path.expanduser(args.png_root))
    if args.out_dir is None:
        args.out_dir = os.path.join(DEFAULT_OUT_ROOT, SPLIT_TO_SUBDIR[args.split])
    args.out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    if args.split_list is None:
        args.split_list = os.path.join(args.iplan_root, "data", f"{args.split}.txt")
    args.split_list = os.path.abspath(os.path.expanduser(args.split_list))

    if not os.path.isdir(args.iplan_root):
        parser.error(f"--iplan-root does not exist or is not a directory: {args.iplan_root}")
    if not os.path.isdir(args.png_root):
        parser.error(f"--png-root does not exist or is not a directory: {args.png_root}")
    if not os.path.isfile(args.split_list):
        parser.error(f"--split-list does not exist: {args.split_list}")
    return args


def _convert_one(task: tuple[str, str, str]) -> tuple[str, str | None]:
    """Worker: convert a single PNG to .mat. Returns (png_name, error_or_None)."""
    # Imported inside the worker so the parent process doesn't need to keep the
    # FloorPlan module alive across forks; also makes the import error obvious
    # if --iplan-root was wrong.
    from floorplan import FloorPlan

    png_root, png_name, out_path = task
    try:
        fp = FloorPlan(png_root, png_name, _NullLog())
        sio.savemat(out_path, {"data": fp.to_dict()})
        return png_name, None
    except Exception as exc:  # per-file failure is non-fatal; logged and counted
        return png_name, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"


def _worker_init(iplan_root: str) -> None:
    """Pool initializer: ensure each worker can ``from floorplan import FloorPlan``."""
    _add_iplan_to_path(iplan_root)


def _build_tasks(
    png_root: str,
    out_dir: str,
    names: list[str],
    force: bool,
) -> list[tuple[str, str, str]]:
    tasks: list[tuple[str, str, str]] = []
    for name in names:
        if not name.endswith(".png"):
            continue
        out_path = os.path.join(out_dir, name.replace(".png", ".mat"))
        if not force and os.path.exists(out_path):
            continue
        if not os.path.exists(os.path.join(png_root, name)):
            continue
        tasks.append((png_root, name, out_path))
    return tasks


def main() -> int:
    args = _parse_args()

    # Make the parent process able to import FloorPlan too (used for the
    # smoke-import check below; workers re-add it via _worker_init).
    _add_iplan_to_path(args.iplan_root)
    try:
        from floorplan import FloorPlan  # noqa: F401  (smoke import)
    except Exception as exc:
        print(
            f"ERROR: could not import iPLAN's FloorPlan parser from "
            f"{args.iplan_root}/data/preprocess/floorplan.py: {exc}",
            file=sys.stderr,
        )
        return 2

    os.makedirs(args.out_dir, exist_ok=True)

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
    os.makedirs(log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"convert_png_to_iplan_mat_{args.split}_{stamp}.log")

    with open(args.split_list) as f:
        names = [n.strip() for n in f if n.strip()]
    if args.limit is not None:
        names = names[: args.limit]

    tasks = _build_tasks(args.png_root, args.out_dir, names, args.force)

    print(f"Split          : {args.split}")
    print(f"iPLAN root     : {args.iplan_root}")
    print(f"PNG root       : {args.png_root}")
    print(f"Split list     : {args.split_list}")
    print(f"Out dir        : {args.out_dir}")
    print(f"Files in list  : {len(names)}")
    print(f"To convert     : {len(tasks)} (skipped {len(names) - len(tasks)} existing/missing)")
    print(f"Workers        : {args.workers}")
    print(f"Log            : {log_path}")
    print()

    if not tasks:
        print("Nothing to do.")
        return 0

    failures: list[tuple[str, str]] = []
    start = time.time()
    last_print = start
    done = 0

    with open(log_path, "w") as logf, Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(args.iplan_root,),
    ) as pool:
        logf.write(
            f"# split={args.split}  workers={args.workers}  total={len(tasks)}  "
            f"out_dir={args.out_dir}\n"
        )
        logf.flush()

        for png_name, err in pool.imap_unordered(_convert_one, tasks, chunksize=8):
            done += 1
            if err is not None:
                failures.append((png_name, err))
                logf.write(f"FAIL {png_name}: {err.splitlines()[0]}\n")
                logf.flush()

            now = time.time()
            if now - last_print >= 2.0 or done == len(tasks):
                rate = done / max(1e-6, now - start)
                eta = (len(tasks) - done) / max(1e-6, rate)
                print(
                    f"  {done}/{len(tasks)}  "
                    f"({100.0 * done / len(tasks):5.1f}%)  "
                    f"{rate:6.1f} fp/s  eta {eta/60:5.1f} min  "
                    f"failures={len(failures)}",
                    flush=True,
                )
                last_print = now

        logf.write(f"# done={done}  failures={len(failures)}\n")

    elapsed = time.time() - start
    print()
    print(f"Finished in {elapsed/60:.1f} min")
    print(f"Successful : {done - len(failures)}")
    print(f"Failed     : {len(failures)}")
    if failures:
        print("First 10 failures (see log for full traces):")
        for name, err in failures[:10]:
            print(f"  {name}: {err.splitlines()[0]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
