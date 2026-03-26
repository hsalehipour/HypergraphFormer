#!/bin/bash

# ============================================
# House-GAN++ Setup Script
# ============================================
# This script:
# 1. Clones houseganpp repo if needed
# 2. Creates virtual environment
# 3. Installs dependencies
# 4. Installs houseganpp package (editable mode)
# 5. Converts Hypergrph1K Dataset to HouseGAN++ format
# ============================================

set -e  # Exit on error

# Get the directory where this script is located (benchmark folder)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BENCHMARK_DIR="$SCRIPT_DIR"

# Repository paths
REPOS_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
HOUSEGANPP_REPO="$REPOS_DIR/houseganpp"
HOUSEGANPP_REMOTE="https://github.com/hsalehipour/houseganpp"

# Data paths
# /dataset/hypergraph is at the root of HypergraphFormer
# Hypergraph_to_RPLAN will be in benchmark/dataset/
HF_ROOT="$(dirname "$BENCHMARK_DIR")"
Hypergraph_to_RPLAN_INPUT="$HF_ROOT/dataset/hypergraph"
Hypergraph_to_RPLAN_OUTPUT="$BENCHMARK_DIR/dataset/Hypergraph_to_RPLAN"

# Virtual environment path (in HypergraphFormer root)
VENV_PATH="$HF_ROOT/.venv"

# Checkpoint paths
HOUSEGANPP_CHECKPOINTS_DIR="$BENCHMARK_DIR/houseganpp_checkpoints"

# Conversion script parameters
TARGET_SIZE=64

echo "============================================"
echo "House-GAN++ Setup"
echo "============================================"
echo "Benchmark directory: $BENCHMARK_DIR"
echo "Repos directory: $REPOS_DIR"
echo "HouseGAN++ repo: $HOUSEGANPP_REPO"
echo ""

# ============================================
# Step 1: Clone houseganpp if it doesn't exist
# ============================================
if [ ! -d "$HOUSEGANPP_REPO" ]; then
    echo "[1/6] Cloning houseganpp repository..."
    git clone "$HOUSEGANPP_REMOTE" "$HOUSEGANPP_REPO"
    echo "✓ Repository cloned successfully"
else
    echo "[1/6] HouseGAN++ repository already exists"
    echo "✓ Skipping clone"
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
    echo "✓ Skipping creation"
fi
echo ""

# ============================================
# Step 3: Install requirements
# ============================================
echo "[3/6] Installing requirements from houseganpp..."
source "$VENV_PATH/bin/activate"

# Upgrade pip first
pip install --upgrade pip

# Install requirements from houseganpp
if [ -f "$HOUSEGANPP_REPO/requirements.txt" ]; then
    pip install -r "$HOUSEGANPP_REPO/requirements.txt"
    echo "✓ Requirements installed"
else
    echo "⚠ Warning: requirements.txt not found in houseganpp repo"
fi
echo ""

# ============================================
# Step 4: Install houseganpp package
# ============================================
echo "[4/6] Installing houseganpp package..."
if [ -f "$HOUSEGANPP_REPO/pyproject.toml" ] || [ -f "$HOUSEGANPP_REPO/setup.py" ]; then
    pip install -e "$HOUSEGANPP_REPO"
    echo "✓ HouseGAN++ package installed"
else
    echo "⚠ Warning: pyproject.toml or setup.py not found in houseganpp repo"
fi
echo ""

# ============================================
# Step 5: Run conversion script
# ============================================
echo "[5/6] Converting Hypergrph1K Dataset to HouseGAN++ format..."

# Check if conversion script exists
if [ ! -f "$BENCHMARK_DIR/utils/convert_hypergraph_to_rplan.py" ]; then
    echo "✗ Error: Conversion script not found at $BENCHMARK_DIR/utils/convert_hypergraph_to_rplan.py"
    exit 1
fi

# Check if input directory exists
if [ ! -d "$Hypergraph_to_RPLAN_INPUT" ]; then
    echo "⚠ Warning: Hypergrph1K Dataset directory not found at $Hypergraph_to_RPLAN_INPUT"
    echo "  Please ensure the Hypergrph1K Dataset is available at the root of HypergraphFormer."
    echo "  Skipping conversion step..."
else
    # Create output directory structure if it doesn't exist
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
    echo "  Output saved to: $Hypergraph_to_RPLAN_OUTPUT"
fi
echo ""


# ============================================
# Step 6: Setup Complete
# ============================================
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "Summary:"
echo "  - HouseGAN++ repo: $HOUSEGANPP_REPO"
echo "  - Virtual environment: $VENV_PATH"
echo "  - Input dataset: $Hypergraph_to_RPLAN_INPUT"
echo "  - Output dataset: $Hypergraph_to_RPLAN_OUTPUT"
echo ""
echo "To activate the virtual environment:"
echo "  source $VENV_PATH/bin/activate"
echo ""
echo "To run fine-tuning (after activating venv):"
echo "  cd $BENCHMARK_DIR"
echo "  ./finetune_houseganpp.sh"
echo ""
