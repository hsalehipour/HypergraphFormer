#!/bin/bash

# ============================================
# House-Diffusion Setup Script
# ============================================
# This script:
# 1. Verifies house_diffusion repo exists
# 2. Creates/uses virtual environment
# 3. Installs dependencies
# 4. Install house_diffusion package
# 5. Reuses Hypergraph_to_RPLAN format data if available
# ============================================

set -e  # Exit on error

# Get the directory where this script is located (benchmark folder)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BENCHMARK_DIR="$SCRIPT_DIR"

# Repository paths
REPOS_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
HOUSEDIFFUSION_REPO="$REPOS_DIR/house_diffusion"

# Data paths
HF_ROOT="$(dirname "$BENCHMARK_DIR")"
Hypergraph_to_RPLAN_INPUT="$HF_ROOT/dataset/hypergraph"
Hypergraph_to_RPLAN_OUTPUT="$BENCHMARK_DIR/dataset/Hypergraph_to_RPLAN"

# House-Diffusion expects data at ../datasets/rplan relative to its root
HOUSEDIFFUSION_DATASETS_DIR="$HOUSEDIFFUSION_REPO/datasets"
HOUSEDIFFUSION_RPLAN_DIR="$HOUSEDIFFUSION_DATASETS_DIR/rplan"

# Virtual environment path (in HypergraphFormer root)
VENV_PATH="$HF_ROOT/.venv"

# Conversion script parameters
TARGET_SIZE=64

echo "============================================"
echo "House-Diffusion Setup"
echo "============================================"
echo "Benchmark directory: $BENCHMARK_DIR"
echo "Repos directory: $REPOS_DIR"
echo "House-Diffusion repo: $HOUSEDIFFUSION_REPO"
echo ""

# ============================================
# Step 1: Verify house_diffusion repo exists
# ============================================
if [ ! -d "$HOUSEDIFFUSION_REPO" ]; then
    echo "✗ Error: House-Diffusion repository not found at $HOUSEDIFFUSION_REPO"
    echo "  Please clone it first:"
    echo "  cd $REPOS_DIR"
    echo "  git clone https://github.com/hsalehipour/house_diffusion.git"
    exit 1
else
    echo "[1/6] House-Diffusion repository found"
    echo "✓ Path: $HOUSEDIFFUSION_REPO"
fi
echo ""

# ============================================
# Step 2: Create virtual environment if needed
# ============================================
if [ ! -d "$VENV_PATH" ]; then
    echo "[2/6] Creating virtual environment..."
    python3 -m venv "$VENV_PATH"
    echo "✓ Virtual environment created"
else
    echo "[2/6] Virtual environment already exists"
    echo "✓ Using existing environment"
fi
echo ""

# ============================================
# Step 3: Install requirements
# ============================================
echo "[3/6] Installing requirements from house_diffusion..."
source "$VENV_PATH/bin/activate"

# Upgrade pip first
pip install --upgrade pip

# Install requirements from house_diffusion
if [ -f "$HOUSEDIFFUSION_REPO/requirements.txt" ]; then
    pip install -r "$HOUSEDIFFUSION_REPO/requirements.txt"
    echo "✓ Requirements installed"
else
    echo "⚠ Warning: requirements.txt not found in house_diffusion repo"
fi
echo ""

# ============================================
# Step 4: Install house_diffusion package
# ============================================
echo "[4/6] Installing house_diffusion package..."
if [ -f "$HOUSEDIFFUSION_REPO/setup.py" ]; then
    pip install -e "$HOUSEDIFFUSION_REPO"
    echo "✓ House-Diffusion package installed"
else
    echo "⚠ Warning: setup.py not found in house_diffusion repo"
fi
echo ""

# ============================================
# Step 5: Check/Convert Hypergrph1K Dataset
# ============================================
echo "[5/6] Setting up Hypergrph1K Dataset..."

# Check if already converted data exists
if [ -d "$Hypergraph_to_RPLAN_OUTPUT" ] && [ -f "$Hypergraph_to_RPLAN_OUTPUT/train/train_data.pkl" ]; then
    echo "✓ Hypergraph_to_RPLAN format data already exists"
    echo "  Skipping conversion (reusing existing data)"
else
    echo "Converting Hypergrph1K Dataset to RPLAN format..."
    
    # Check if conversion script exists
    if [ ! -f "$BENCHMARK_DIR/utils/convert_hypergraph_to_rplan.py" ]; then
        echo "✗ Error: Conversion script not found at $BENCHMARK_DIR/utils/convert_hypergraph_to_rplan.py"
        exit 1
    fi
    
    # Check if input directory exists
    if [ ! -d "$Hypergraph_to_RPLAN_INPUT" ]; then
        echo "✗ Error: Hypergrph1K Dataset directory not found at $Hypergraph_to_RPLAN_INPUT"
        echo "  Please ensure the Hypergrph1K Dataset is available."
        exit 1
    fi
    
    # Create output directory structure
    mkdir -p "$BENCHMARK_DIR/dataset"
    mkdir -p "$Hypergraph_to_RPLAN_OUTPUT"
    
    # Set PYTHONPATH to include HypergraphFormer root for imports
    export PYTHONPATH="$HF_ROOT:$PYTHONPATH"
    
    # Run conversion
    python "$BENCHMARK_DIR/utils/convert_hypergraph_to_rplan.py" \
        --input "$Hypergraph_to_RPLAN_INPUT" \
        --output "$Hypergraph_to_RPLAN_OUTPUT" \
        --target-size "$TARGET_SIZE"
    
    echo "✓ Conversion completed"
fi
echo ""

# ============================================
# Step 5 (continued): Convert pickle to NPZ format
# ============================================
echo "[6/6] Converting pickle data to house-diffusion NPZ format..."

# Create processed_rplan directory
mkdir -p "$HOUSEDIFFUSION_REPO/processed_rplan"

# Check if NPZ files already exist
if [ -f "$HOUSEDIFFUSION_REPO/processed_rplan/rplan_train_all.npz" ] && \
   [ -f "$HOUSEDIFFUSION_REPO/processed_rplan/rplan_eval_all.npz" ]; then
    echo "✓ NPZ files already exist, skipping conversion"
else
    echo "Converting Hypergraph_to_RPLAN pickle to NPZ format..."
    echo "(No target_set filtering - using all samples)"
    
    # Set PYTHONPATH
    export PYTHONPATH="$HF_ROOT:$PYTHONPATH"
    
    python "$BENCHMARK_DIR/utils/convert_pickle_to_npz.py" \
        --input-dir "$Hypergraph_to_RPLAN_OUTPUT" \
        --output-dir "$HOUSEDIFFUSION_REPO/processed_rplan"
    
    echo "✓ Conversion completed"
    
    # Create synthetic eval file (required by House-Diffusion's eval mode)
    EVAL_NPZ="$HOUSEDIFFUSION_REPO/processed_rplan/rplan_eval_all.npz"
    EVAL_SYN_NPZ="$HOUSEDIFFUSION_REPO/processed_rplan/rplan_eval_all_syn.npz"
    if [ -f "$EVAL_NPZ" ] && [ ! -f "$EVAL_SYN_NPZ" ]; then
        echo "Creating synthetic eval file..."
        cp "$EVAL_NPZ" "$EVAL_SYN_NPZ"
        echo "✓ Synthetic eval file created"
    fi
fi
echo ""

# ============================================
# Setup Complete
# ============================================
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "Summary:"
echo "  - House-Diffusion repo: $HOUSEDIFFUSION_REPO"
echo "  - Virtual environment: $VENV_PATH"
echo "  - Input dataset: $Hypergraph_to_RPLAN_INPUT"
echo "  - Converted data: $Hypergraph_to_RPLAN_OUTPUT"
echo ""
echo "To activate the virtual environment:"
echo "  source $VENV_PATH/bin/activate"
echo ""
echo "To run fine-tuning (after activating venv):"
echo "  cd $BENCHMARK_DIR"
echo "  ./finetune_housediffusion.sh"
echo ""

