#!/bin/bash

# ============================================
# House-Diffusion Evaluation Script
# ============================================
# Generates floorplans and computes metrics (GED, room accuracy, delta)
# against the Hypergraph1K eval set.
#
# Usage:
#   ./eval_housediffusion.sh
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

# Data path (eval NPZ from house_diffusion)
EVAL_NPZ="$HOUSEDIFFUSION_REPO/processed_rplan/rplan_eval_all.npz"

# Model checkpoint
CHECKPOINT="$HOUSEDIFFUSION_REPO/runs/ft_housediffusion_v1/model795000.pt"

# Evaluation parameters
NUM_SAMPLES=""          # Number of samples (empty = all from NPZ)
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
echo "  Data:           $EVAL_NPZ"
echo "  Checkpoint:     $CHECKPOINT"
echo "  Variations:     $NUM_VARIATIONS"
echo "  Output:         $OUTPUT_DIR"
echo ""

# ============================================
# Verify Prerequisites
# ============================================

if [ -n "$CHECKPOINT" ] && [ -z "$LOAD_GENERATIONS" ] && [ ! -f "$CHECKPOINT" ]; then
    echo "✗ Checkpoint not found: $CHECKPOINT"
    echo ""
    echo "Available checkpoints:"
    ls -1 "$HOUSEDIFFUSION_REPO"/runs/*/model*.pt 2>/dev/null | head -10 || echo "  None found"
    exit 1
fi

if [ ! -f "$EVAL_NPZ" ]; then
    echo "✗ Evaluation NPZ not found: $EVAL_NPZ"
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
