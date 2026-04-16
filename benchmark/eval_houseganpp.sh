#!/bin/bash

# ============================================
# HouseGAN++ Evaluation Script
# ============================================
# Generates floorplans and computes metrics (GED, room accuracy, delta)
# from a directory of RPLAN JSON files (produced by Housegan-data-reader).
#
# Automatically converts JSON -> pickle (if not already cached) using
# benchmark/utils/convert_json_to_pickle.py, then runs evaluation.
#
# Usage:
#   ./eval_houseganpp.sh /path/to/json_dir
# ============================================

set -e

# Get the directory where this script is located (benchmark folder)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BENCHMARK_DIR="$SCRIPT_DIR"
HF_ROOT="$(dirname "$BENCHMARK_DIR")"

# Repository paths
REPOS_DIR="$(dirname "$HF_ROOT")"
HOUSEGANPP_REPO="$REPOS_DIR/houseganpp"

# Virtual environment
VENV_PATH="$HF_ROOT/.venv"

# Data path -- directory of RPLAN JSON files
JSON_DIR="${1:-}"

# Model checkpoint (modify this to use your trained model)
CHECKPOINT="$HOUSEGANPP_REPO/checkpoints/ft_housegan_final.pth"

# Evaluation parameters
NUM_SAMPLES=""          # Number of samples (empty = all)
NUM_VARIATIONS=1        # Number of generation variations per sample
BATCH_SIZE=1
TARGET_SET=""           # Filter by room count (empty = no filter)

# Output directory
OUTPUT_DIR="$BENCHMARK_DIR/eval_results_houseganpp"

# Generation caching (leave empty to skip)
SAVE_GENERATIONS=""     # Path to save generated outputs pickle
LOAD_GENERATIONS=""     # Path to load previously generated outputs pickle

# ============================================
# House-GAN++ Evaluation
# ============================================
echo "============================================"
echo "HouseGAN++ Evaluation"
echo "============================================"
echo "  JSON dir:       $JSON_DIR"
echo "  Checkpoint:     $CHECKPOINT"
echo "  Variations:     $NUM_VARIATIONS"
echo "  Output:         $OUTPUT_DIR"
echo ""

# ============================================
# Verify Prerequisites
# ============================================

if [ -z "$JSON_DIR" ]; then
    echo "Usage: $0 <json_dir>"
    echo ""
    echo "  json_dir  Directory containing RPLAN JSON files"
    echo "            (produced by Housegan-data-reader's batch_raster_to_json.py)"
    exit 1
fi

if [ ! -d "$JSON_DIR" ]; then
    echo "✗ JSON directory not found: $JSON_DIR"
    exit 1
fi

if [ -n "$CHECKPOINT" ] && [ -z "$LOAD_GENERATIONS" ] && [ ! -f "$CHECKPOINT" ]; then
    echo "✗ Checkpoint not found: $CHECKPOINT"
    echo ""
    echo "Available checkpoints:"
    ls -1 "$HOUSEGANPP_REPO"/checkpoints/*.pth 2>/dev/null | head -10 || echo "  None found"
    exit 1
fi

# ============================================
# Setup Environment
# ============================================

if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
    echo "✓ Virtual environment activated"
else
    echo "✗ Virtual environment not found at $VENV_PATH"
    echo "  Please run ./setup_houseganpp.sh first"
    exit 1
fi

export PYTHONPATH="$HOUSEGANPP_REPO:$HF_ROOT:$PYTHONPATH"
echo "✓ PYTHONPATH configured"

mkdir -p "$OUTPUT_DIR"
echo "✓ Output directory: $OUTPUT_DIR"
echo ""

# ============================================
# JSON -> Pickle Conversion (if needed)
# ============================================

# Resolve JSON_DIR to absolute path for consistent pickle placement
JSON_DIR="$(cd "$JSON_DIR" && pwd)"
DATASET_ROOT="$(dirname "$JSON_DIR")"
PKL_DIR="$DATASET_ROOT/processed_rplan_pickles"
# Derive pickle name from the JSON directory name (e.g. test -> test_data.pkl)
SPLIT_NAME="$(basename "$JSON_DIR")"
TEST_DATA="$PKL_DIR/${SPLIT_NAME}_data.pkl"

if [ -f "$TEST_DATA" ]; then
    echo "✓ Using cached pickle: $TEST_DATA"
else
    echo "Converting JSON -> pickle ..."
    mkdir -p "$PKL_DIR"
    python3 "$BENCHMARK_DIR/utils/convert_json_to_pickle.py" \
        --json_dir "$JSON_DIR" \
        --output "$TEST_DATA"
    echo "✓ Pickle conversion complete: $TEST_DATA"
fi
echo ""

# ============================================
# Run Evaluation
# ============================================

CMD="python3 \"$BENCHMARK_DIR/src/evaluate_houseganpp.py\" \
    --data_path \"$TEST_DATA\" \
    --checkpoint \"$CHECKPOINT\" \
    --out \"$OUTPUT_DIR\" \
    --batch_size $BATCH_SIZE \
    --num_variations $NUM_VARIATIONS \
    --save_graphs"

if [ -n "$NUM_SAMPLES" ]; then
    CMD="$CMD --num_samples $NUM_SAMPLES"
fi

if [ -n "$TARGET_SET" ]; then
    CMD="$CMD --target_set $TARGET_SET"
fi

if [ -n "$SAVE_GENERATIONS" ]; then
    CMD="$CMD --save_generations \"$SAVE_GENERATIONS\""
fi

if [ -n "$LOAD_GENERATIONS" ]; then
    CMD="$CMD --load_generations \"$LOAD_GENERATIONS\""
fi

echo "Running: $CMD"
echo ""

eval $CMD

echo ""
echo "✓ Evaluation complete!"
echo "  Results saved to: $OUTPUT_DIR"
