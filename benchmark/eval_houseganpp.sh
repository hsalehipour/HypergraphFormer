#!/bin/bash

# ============================================
# HouseGAN++ Evaluation Script
# ============================================
# Generates floorplans and computes metrics (GED, room accuracy, delta)
# against the Hypergraph1K test set.
#
# Usage:
#   ./eval_houseganpp.sh
# ============================================

# Get the directory where this script is located (benchmark folder)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BENCHMARK_DIR="$SCRIPT_DIR"
HF_ROOT="$(dirname "$BENCHMARK_DIR")"

# Repository paths
REPOS_DIR="$(dirname "$HF_ROOT")"
HOUSEGANPP_REPO="$REPOS_DIR/houseganpp"

# Virtual environment
VENV_PATH="$HF_ROOT/.venv"

# Data paths
TEST_DATA="$BENCHMARK_DIR/dataset/Hypergraph_to_RPLAN/test/test_data.pkl"

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
# Run Evaluation
# ============================================

echo "============================================"
echo "HouseGAN++ Evaluation"
echo "============================================"
echo "  Data:           $TEST_DATA"
echo "  Checkpoint:     $CHECKPOINT"
echo "  Variations:     $NUM_VARIATIONS"
echo "  Output:         $OUTPUT_DIR"
echo ""

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

eval $CMD

echo ""
echo "✓ Evaluation complete!"
echo "  Results saved to: $OUTPUT_DIR"
