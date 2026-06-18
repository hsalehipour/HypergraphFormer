#!/bin/bash

# ============================================
# DiffPlanner Setup Script
# ============================================
# Prepares everything benchmark/src/evaluate_diffplanner.py needs to run
# the full b+nc -> b+ncsl -> b+ncsla cascade end-to-end:
#
#   1. Clone the DiffPlanner repository (skip if it already exists).
#   2. Download trained_model.zip from the official GitHub release and
#      lay the three required checkpoints out at the exact paths the
#      evaluator's --node_checkpoint / --adjacency_checkpoint /
#      --partition_checkpoint defaults expect:
#
#          <DP>/node_diff/scripts/trained_model/bnc_model300000.pt
#          <DP>/adjacency_diff/scripts/trained_model/bncsl_model300000.pt
#          <DP>/partitioning_diff/scripts/trained_model/bncsla_model300000.pt
#
#   3. Symlink dataset_diffplanner/data_test.json into
#      <DP>/dataset/dataset_json/data_test.json so DiffPlanner's
#      RPlanDataset / post_processing helpers (which import that file
#      relative to their own package) just work. The dataset itself is
#      downloaded by ./prepare_rplan_hypergraph.sh -- this script will
#      remind you to run it if the JSONs are not there yet.
#
# DiffPlanner has no setup.py / requirements.txt: the evaluator imports
# its sub-packages by adding them to sys.path at runtime, so there is
# no pip install step.
#
# Usage:
#   ./benchmark/setup_diffplanner.sh
#   ./benchmark/setup_diffplanner.sh --force-checkpoints   # re-download even if present
# ============================================

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BENCHMARK_DIR="$SCRIPT_DIR"
HF_ROOT="$(dirname "$BENCHMARK_DIR")"
REPOS_DIR="$(dirname "$HF_ROOT")"
DIFFPLANNER_REPO="$REPOS_DIR/DiffPlanner"

DIFFPLANNER_GIT_URL="https://github.com/shidong-wang/DiffPlanner.git"
TRAINED_MODEL_URL="https://github.com/shidong-wang/DiffPlanner/releases/download/trained_model/trained_model.zip"

# These are the EXACT paths benchmark/src/evaluate_diffplanner.py uses
# when --node_checkpoint / --adjacency_checkpoint / --partition_checkpoint
# are not given on the command line. Keep in sync with
# _resolve_default_checkpoints() in that file.
NODE_CKPT_DIR="$DIFFPLANNER_REPO/node_diff/scripts/trained_model"
ADJ_CKPT_DIR="$DIFFPLANNER_REPO/adjacency_diff/scripts/trained_model"
PART_CKPT_DIR="$DIFFPLANNER_REPO/partitioning_diff/scripts/trained_model"

NODE_CKPT="$NODE_CKPT_DIR/bnc_model300000.pt"
ADJ_CKPT="$ADJ_CKPT_DIR/bncsl_model300000.pt"
PART_CKPT="$PART_CKPT_DIR/bncsla_model300000.pt"

# Dataset paths (provided by ./prepare_rplan_hypergraph.sh).
RPLAN_DATASET_ROOT="$HF_ROOT/dataset/rplan/dataset_diffplanner"
RPLAN_TEST_JSON="$RPLAN_DATASET_ROOT/data_test.json"

# DiffPlanner's RPlanDataset / post_processing.py loads
# `../../dataset/dataset_json/data_test.json` relative to each sub-package
# at import time. We symlink the HypergraphFormer copy there so a single
# source of truth is used across both repos.
DP_DATA_DIR="$DIFFPLANNER_REPO/dataset/dataset_json"
DP_DATA_TEST="$DP_DATA_DIR/data_test.json"

# ── CLI ───────────────────────────────────────────────────────────────
FORCE_CKPT=false
for arg in "$@"; do
    case "$arg" in
        --force-checkpoints) FORCE_CKPT=true ;;
        -h|--help)
            sed -n '3,40p' "$0"
            exit 0
            ;;
        *)  echo "Unknown option: $arg"; exit 1 ;;
    esac
done

echo "============================================"
echo "DiffPlanner Setup"
echo "============================================"
echo "Repos directory:       $REPOS_DIR"
echo "DiffPlanner repo:      $DIFFPLANNER_REPO"
echo "RPLAN dataset root:    $RPLAN_DATASET_ROOT"
echo ""

# ============================================
# Step 1: Clone DiffPlanner repo
# ============================================
if [ ! -d "$DIFFPLANNER_REPO" ]; then
    echo "[1/3] Cloning DiffPlanner repository..."
    git clone --depth 1 "$DIFFPLANNER_GIT_URL" "$DIFFPLANNER_REPO"
    echo "  ✓ Cloned to $DIFFPLANNER_REPO"
else
    echo "[1/3] DiffPlanner repository already present"
    echo "  ✓ Path: $DIFFPLANNER_REPO"
fi
echo ""

# ============================================
# Step 2: Download + lay out checkpoints
# ============================================
need_download=false
if $FORCE_CKPT; then
    need_download=true
elif [ ! -f "$NODE_CKPT" ] || [ ! -f "$ADJ_CKPT" ] || [ ! -f "$PART_CKPT" ]; then
    need_download=true
fi

if $need_download; then
    echo "[2/3] Downloading trained_model.zip (~665 MB)..."

    TMP_ZIP=$(mktemp /tmp/diffplanner_trained_XXXXXX.zip)
    TMP_EXTRACT=$(mktemp -d /tmp/diffplanner_trained_extract_XXXXXX)
    trap "rm -rf '$TMP_ZIP' '$TMP_EXTRACT'" EXIT

    # -L follows redirects (GitHub release URLs redirect to a CDN);
    # --fail makes curl exit non-zero on HTTP errors.
    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --retry 3 --progress-bar \
             -o "$TMP_ZIP" "$TRAINED_MODEL_URL"
    elif command -v wget >/dev/null 2>&1; then
        wget --show-progress -O "$TMP_ZIP" "$TRAINED_MODEL_URL"
    else
        echo "  ✗ Need either curl or wget to download the checkpoints."
        exit 1
    fi

    echo "  Extracting to a staging directory..."
    unzip -q "$TMP_ZIP" -d "$TMP_EXTRACT"

    # The release zip's internal layout has changed before; rather than
    # hard-code it we look for the three target filenames anywhere
    # underneath the staging directory and route each to the right
    # `<sub>/scripts/trained_model/` folder.
    mkdir -p "$NODE_CKPT_DIR" "$ADJ_CKPT_DIR" "$PART_CKPT_DIR"

    place_ckpt() {
        local fname="$1"   # e.g. bnc_model300000.pt
        local dest="$2"    # absolute destination path
        local found
        found=$(find "$TMP_EXTRACT" -type f -name "$fname" -print -quit || true)
        if [ -z "$found" ]; then
            echo "  ✗ $fname not found inside trained_model.zip"
            return 1
        fi
        if [ "$found" != "$dest" ]; then
            cp -f "$found" "$dest"
        fi
        echo "  ✓ $(basename "$dest")  ->  $dest"
    }

    place_ckpt "bnc_model300000.pt"    "$NODE_CKPT"
    place_ckpt "bncsl_model300000.pt"  "$ADJ_CKPT"
    place_ckpt "bncsla_model300000.pt" "$PART_CKPT"

    # Optional convenience: copy ALL <sub>/*_model300000.pt checkpoints
    # too, so other DiffPlanner condition modes still work if you switch
    # them on later. Safe no-op when the zip is bnc-only.
    for sub in node_diff adjacency_diff partitioning_diff; do
        sub_dir="$DIFFPLANNER_REPO/$sub/scripts/trained_model"
        mkdir -p "$sub_dir"
        # Copy any other checkpoints found under a folder of the same
        # sub-package name (case-insensitive match on the leading
        # directory inside the zip).
        find "$TMP_EXTRACT" -type d \( -iname "$sub" -o -iname "${sub//_/-}" \) \
             -exec sh -c '
                 src="$1"; dst="$2"
                 find "$src" -type f -name "*_model*.pt" -print0 |
                     while IFS= read -r -d "" f; do
                         cp -n "$f" "$dst/$(basename "$f")"
                     done
             ' _ {} "$sub_dir" \; 2>/dev/null || true
    done

    rm -rf "$TMP_ZIP" "$TMP_EXTRACT"
    trap - EXIT
    echo "  ✓ Checkpoints installed"
else
    echo "[2/3] Checkpoints already present, skipping download"
    echo "  ✓ $NODE_CKPT"
    echo "  ✓ $ADJ_CKPT"
    echo "  ✓ $PART_CKPT"
    echo "  (re-run with --force-checkpoints to re-download)"
fi
echo ""

# ============================================
# Step 3: Wire DiffPlanner -> HypergraphFormer dataset
# ============================================
echo "[3/3] Wiring DiffPlanner repo to the prepared RPLAN dataset..."

if [ ! -f "$RPLAN_TEST_JSON" ]; then
    echo "  ⚠ Expected dataset not found: $RPLAN_TEST_JSON"
    echo "    Run ./prepare_rplan_hypergraph.sh from $HF_ROOT first."
    echo "    (That script downloads dataset_diffplanner/data_*.json from the"
    echo "     official DiffPlanner release into the right place.)"
else
    mkdir -p "$DP_DATA_DIR"
    if [ -L "$DP_DATA_TEST" ] || [ -f "$DP_DATA_TEST" ]; then
        # Refresh the symlink in case --data_path points somewhere new.
        rm -f "$DP_DATA_TEST"
    fi
    ln -s "$RPLAN_TEST_JSON" "$DP_DATA_TEST"
    echo "  ✓ Symlinked: $DP_DATA_TEST -> $RPLAN_TEST_JSON"

    # data_train.json is needed if you ever want to retrain inside the
    # DiffPlanner repo; symlink it too when present.
    RPLAN_TRAIN_JSON="$RPLAN_DATASET_ROOT/data_train.json"
    if [ -f "$RPLAN_TRAIN_JSON" ]; then
        DP_DATA_TRAIN="$DP_DATA_DIR/data_train.json"
        rm -f "$DP_DATA_TRAIN"
        ln -s "$RPLAN_TRAIN_JSON" "$DP_DATA_TRAIN"
        echo "  ✓ Symlinked: $DP_DATA_TRAIN -> $RPLAN_TRAIN_JSON"
    fi
fi
echo ""

# ============================================
# Verify
# ============================================
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "Summary:"
echo "  DiffPlanner repo:      $DIFFPLANNER_REPO"
[ -f "$NODE_CKPT" ] && echo "  NodeDiff ckpt:         $NODE_CKPT"      || echo "  ✗ NodeDiff ckpt MISSING:        $NODE_CKPT"
[ -f "$ADJ_CKPT"  ] && echo "  AdjacencyDiff ckpt:    $ADJ_CKPT"       || echo "  ✗ AdjacencyDiff ckpt MISSING:   $ADJ_CKPT"
[ -f "$PART_CKPT" ] && echo "  PartitioningDiff ckpt: $PART_CKPT"      || echo "  ✗ PartitioningDiff ckpt MISSING:$PART_CKPT"
[ -L "$DP_DATA_TEST" ] && echo "  data_test.json symlink: $DP_DATA_TEST" || echo "  ⚠ data_test.json symlink missing (run ./prepare_rplan_hypergraph.sh first)"
echo ""
echo "To run the cascade benchmark on the full RPLAN test split:"
echo "  cd $HF_ROOT"
echo "  python benchmark/src/evaluate_diffplanner.py \\"
echo "      --data_path $RPLAN_TEST_JSON \\"
echo "      --diffplanner_root $DIFFPLANNER_REPO \\"
echo "      --use_ddim \\"
echo "      --save_generations cache/dp_bnc_test_full.pkl \\"
echo "      --out benchmark/results_dp_full"
echo ""
echo "To re-score from a saved cache (no GPU needed):"
echo "  python benchmark/src/evaluate_diffplanner.py \\"
echo "      --load_generations cache/dp_bnc_test_full.pkl \\"
echo "      --data_path $RPLAN_TEST_JSON \\"
echo "      --out benchmark/results_dp_full"
echo ""
