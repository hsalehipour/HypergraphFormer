#!/bin/bash

# ============================================
# House-Diffusion Fine-tuning Script
# ============================================
# This script launches fine-tuning of House-Diffusion
# on the Hypergrph1K Dataset using the converted RPLAN format data.
# ============================================

set -e  # Exit on error

# Get the directory where this script is located (benchmark folder)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BENCHMARK_DIR="$SCRIPT_DIR"

# Project paths
HF_ROOT="$(dirname "$BENCHMARK_DIR")"
REPOS_DIR="$(dirname "$HF_ROOT")"
HOUSEDIFFUSION_REPO="$REPOS_DIR/house_diffusion"

# Model checkpoint
# Note: Training data is loaded from house_diffusion/processed_rplan/rplan_train_all.npz
# The NPZ files are created during setup (setup_housediffusion.sh)
CHECKPOINT="$HOUSEDIFFUSION_REPO/checkpoints/pretrained_model250000.pt"


# Training configuration (matching house_diffusion/scripts/script.sh)
EXP_NAME="ft_housediffusion_v1"
BATCH_SIZE=16
LEARNING_RATE=1e-5  # Lower LR for safer finetuning
WEIGHT_DECAY=0.0    # Disable weight decay to avoid over-regularization

# Intervals
LOG_INTERVAL=20
SAVE_INTERVAL=50    # Save more frequently to catch early checkpoints

# Training duration: 723 samples / batch_size 16 = ~45 steps/epoch
# Target: 5-10 epochs = 225-450 steps
# We'll save every 50 steps to evaluate checkpoints


echo "============================================"
echo "House-Diffusion Fine-tuning"
echo "============================================"
echo "Benchmark directory: $BENCHMARK_DIR"
echo "House-Diffusion repo: $HOUSEDIFFUSION_REPO"

# Check if training data exists
cd "$HOUSEDIFFUSION_REPO"
TRAIN_DATA="$HOUSEDIFFUSION_REPO/processed_rplan/rplan_train_all.npz"
if [ ! -f "$TRAIN_DATA" ]; then
    echo ""
    echo "✗ Error: Training NPZ file not found"
    echo "  Expected: $TRAIN_DATA"
    echo "  Please run setup_housediffusion.sh first to convert the dataset"
    exit 1
fi
echo "Training data: $TRAIN_DATA"
echo "Checkpoint: $CHECKPOINT"
echo ""

# ============================================
# Clean existing experiment folder (if any)
# ============================================
# House-Diffusion writes logs/checkpoints under runs/$EXP_NAME
EXP_DIR="$HOUSEDIFFUSION_REPO/runs/$EXP_NAME"
if [ -d "$EXP_DIR" ]; then
    echo "Removing existing experiment folder: $EXP_DIR"
    rm -rf "$EXP_DIR"
    echo "✓ Removed $EXP_DIR"
    echo ""
fi


# ============================================
# Launch training
# ============================================
echo "============================================"
echo "Launching fine-tuning..."
echo "============================================"
echo ""

MAX_STEPS=250500  # Start from 250000, train for 500 steps (~11 epochs)

python "$BENCHMARK_DIR/src/finetune_housediffusion.py" \
    --checkpoint "$CHECKPOINT" \
    --exp_name "$EXP_NAME" \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --log_interval $LOG_INTERVAL \
    --save_interval $SAVE_INTERVAL \
    --max_steps $MAX_STEPS
