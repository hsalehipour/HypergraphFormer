#!/bin/bash

# ============================================
# House-Diffusion Evaluation Script
# ============================================
# Generates floorplans and computes metrics (GED, room accuracy, delta)
# from a directory of RPLAN JSON files (produced by Housegan-data-reader).
#
# Automatically converts JSON -> NPZ (if not already cached) using
# benchmark/utils/convert_json_to_npz.py, then runs evaluation.
#
# Usage:
#   ./eval_housediffusion.sh /path/to/json_dir
# ============================================

set -e

# Get the directory where this script is located (benchmark folder)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$SCRIPT_DIR"
HF_ROOT="$(dirname "$BENCHMARK_DIR")"

# Repository paths
REPOS_DIR="$(dirname "$HF_ROOT")"
HOUSEDIFFUSION_REPO="$REPOS_DIR/house_diffusion"

# Virtual environment
VENV_PATH="$HF_ROOT/.venv"

# Data path -- directory of RPLAN JSON files
JSON_DIR="${1:-}"

# Model checkpoint
CHECKPOINT="$HOUSEDIFFUSION_REPO/checkpoints/pretrained_model250000.pt"

# Evaluation parameters
NUM_SAMPLES=""          # Number of samples (empty = all)
NUM_VARIATIONS=1        # Number of generation variations per sample

# Output directory
OUTPUT_DIR="$BENCHMARK_DIR/eval_results_housediffusion"

# Generation caching (leave empty to skip)
SAVE_GENERATIONS=""     # Path to save generated outputs pickle
LOAD_GENERATIONS=""     # Path to load previously generated outputs pickle

# ============================================
# Display Configuration
# ============================================

echo "============================================"
echo "House-Diffusion Evaluation"
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
    ls -1 "$HOUSEDIFFUSION_REPO"/checkpoints/*.pt 2>/dev/null | head -10 || echo "  None found"
    exit 1
fi

if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "✗ Virtual environment not found at $VENV_PATH"
    echo "  Please run ./setup_housediffusion.sh first."
    exit 1
fi

echo "✓ All prerequisites verified"

# ============================================
# Setup Environment
# ============================================

source "$VENV_PATH/bin/activate"
echo "✓ Virtual environment activated"

export PYTHONPATH="$HOUSEDIFFUSION_REPO:$HF_ROOT:$PYTHONPATH"
echo "✓ PYTHONPATH configured"

mkdir -p "$OUTPUT_DIR"
echo "✓ Output directory: $OUTPUT_DIR"
echo ""

# ============================================
# JSON -> NPZ Conversion (if needed)
# ============================================

# Resolve JSON_DIR to absolute path for consistent NPZ placement
JSON_DIR="$(cd "$JSON_DIR" && pwd)"
DATASET_ROOT="$(dirname "$JSON_DIR")"
NPZ_DIR="$DATASET_ROOT/processed_rplan"
EVAL_NPZ="$NPZ_DIR/rplan_eval_all.npz"
CNDIST_NPZ="$NPZ_DIR/rplan_train_all_cndist.npz"
SYN_NPZ="$NPZ_DIR/rplan_eval_all_syn.npz"

# Train JSON dir: sibling "train/" directory (for building cndist from training data)
TRAIN_JSON_DIR="$DATASET_ROOT/train"

if [ -f "$EVAL_NPZ" ] && [ -f "$CNDIST_NPZ" ] && [ -f "$SYN_NPZ" ]; then
    echo "✓ Using cached NPZ files in $NPZ_DIR"
else
    CONVERT_CMD="python3 \"$BENCHMARK_DIR/utils/convert_json_to_npz.py\" \
        --json_dir \"$JSON_DIR\" \
        --output_dir \"$NPZ_DIR\""
    if [ -d "$TRAIN_JSON_DIR" ]; then
        echo "Using train data from $TRAIN_JSON_DIR for corner distribution"
        CONVERT_CMD="$CONVERT_CMD --train_json_dir \"$TRAIN_JSON_DIR\""
    else
        echo "Warning: Train directory not found at $TRAIN_JSON_DIR"
        echo "  Corner distribution will be built from eval data (less accurate)"
    fi
    echo "Converting JSON -> NPZ ..."
    eval $CONVERT_CMD
    echo "✓ NPZ conversion complete"
fi
echo ""

# ============================================
# Run Evaluation
# ============================================

CMD="python3 \"$BENCHMARK_DIR/src/evaluate_housediffusion.py\" \
    --data_path \"$EVAL_NPZ\" \
    --checkpoint \"$CHECKPOINT\" \
    --num_variations $NUM_VARIATIONS \
    --save_graphs \
    --out \"$OUTPUT_DIR\""

if [ -n "$NUM_SAMPLES" ]; then
    CMD="$CMD --num_samples $NUM_SAMPLES"
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
