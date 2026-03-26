#!/bin/bash

# ============================================
# Configuration Parameters
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
TRAIN_DATA="$BENCHMARK_DIR/dataset/Hypergraph_to_RPLAN/train/train_data.pkl"
VAL_DATA="$BENCHMARK_DIR/dataset/Hypergraph_to_RPLAN/validation/validation_data.pkl"

# Model checkpoint
CHECKPOINT="$HOUSEGANPP_REPO/checkpoints/pretrained.pth"

# Training parameters
EXP_FOLDER="ft_houseganpp_v1"
N_EPOCHS=1000
BATCH_SIZE=4

# ============================================
# Setup Environment
# ============================================

# Activate virtual environment
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
    echo "✓ Virtual environment activated"
else
    echo "⚠ Warning: Virtual environment not found at $VENV_PATH"
    echo "  Please run ./setup_houseganpp.sh first"
    exit 1
fi

# Change to houseganpp directory so relative imports work
cd "$HOUSEGANPP_REPO"
echo "✓ Changed to HouseGAN++ directory"
echo ""

# ============================================
# Run Fine-tuning
# ============================================

echo "Starting fine-tuning..."
echo "  Train data: $TRAIN_DATA"
echo "  Val data: $VAL_DATA"
echo "  Checkpoint: $CHECKPOINT"
echo "  Output folder: $EXP_FOLDER"
echo ""

# ============================================
# Clean existing experiment outputs (if any)
# ============================================
# HouseGAN++ writes:
# - images to: ./exps/$EXP_FOLDER/
# - tensorboard to: ./runs/$EXP_FOLDER/
EXP_EXPS_DIR="$HOUSEGANPP_REPO/exps/$EXP_FOLDER"
EXP_RUNS_DIR="$HOUSEGANPP_REPO/runs/$EXP_FOLDER"
if [ -n "$EXP_FOLDER" ]; then
    if [ -d "$EXP_EXPS_DIR" ]; then
        echo "Removing existing experiment images folder: $EXP_EXPS_DIR"
        rm -rf "$EXP_EXPS_DIR"
        echo "✓ Removed $EXP_EXPS_DIR"
    fi
    if [ -d "$EXP_RUNS_DIR" ]; then
        echo "Removing existing tensorboard logs folder: $EXP_RUNS_DIR"
        rm -rf "$EXP_RUNS_DIR"
        echo "✓ Removed $EXP_RUNS_DIR"
    fi
    echo ""
fi

python "$BENCHMARK_DIR/src/finetune_houseganpp.py" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --checkpoint "$CHECKPOINT" \
    --exp_folder "$EXP_FOLDER" \
    --n_epochs "$N_EPOCHS" \
    --batch_size "$BATCH_SIZE"

