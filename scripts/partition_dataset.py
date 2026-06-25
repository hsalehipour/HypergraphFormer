#!/usr/bin/env python3
"""
Build a downsampled SFT folder: copy validation/test JSONL unchanged, subsample train.jsonl.

Example:
    python scripts/partition_dataset.py --num-train 1000
    python scripts/partition_dataset.py -n 5000 --input-dir dataset/rplan/dataset_hypergraph_la0_fixed/sft
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SFT = REPO_ROOT / "dataset" / "rplan" / "dataset_hypergraph_la0_fixed" / "sft"


def main() -> int:
    p = argparse.ArgumentParser(
        description="Subsample train.jsonl; copy validation and test to sft_<N>."
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_SFT,
        help=f"Directory with train/validation/test.jsonl (default: {DEFAULT_SFT})",
    )
    p.add_argument(
        "-n",
        "--num-train",
        type=int,
        required=True,
        help="Number of training lines to keep (sampled uniformly without replacement).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <parent of input>/sft_<N>).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for shuffling (default: 42).",
    )
    args = p.parse_args()

    input_dir = args.input_dir.resolve()
    n = int(args.num_train)
    if n < 1:
        p.error("--num-train must be >= 1")

    train_p = input_dir / "train.jsonl"
    val_p = input_dir / "validation.jsonl"
    test_p = input_dir / "test.jsonl"
    for name, path in (("train", train_p), ("validation", val_p), ("test", test_p)):
        if not path.is_file():
            print(f"error: missing {name}: {path}", file=sys.stderr)
            return 1

    if args.output_dir is not None:
        out_dir = args.output_dir.resolve()
    else:
        out_dir = (input_dir.parent / f"sft_{n}").resolve()

    lines = train_p.read_text(encoding="utf-8").splitlines()
    m = len([ln for ln in lines if ln.strip()])
    if m < n:
        print(
            f"error: train.jsonl has only {m} non-empty lines, cannot take {n}",
            file=sys.stderr,
        )
        return 1

    rng = random.Random(int(args.seed))
    nonempty = [ln for ln in lines if ln.strip()]
    rng.shuffle(nonempty)
    selected = nonempty[:n]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.jsonl").write_text("\n".join(selected) + "\n", encoding="utf-8")
    shutil.copy2(val_p, out_dir / "validation.jsonl")
    shutil.copy2(test_p, out_dir / "test.jsonl")

    print(f"Wrote {n} train lines to {out_dir / 'train.jsonl'}")
    print(f"Copied validation and test to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
