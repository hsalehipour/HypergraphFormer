#!/bin/bash

# ============================================
# Hypergraph Repository Setup Script
# ============================================
# This script:
# 1. Clones the hypergraph repository from your fork if not present
# 2. Uses the existing .venv virtual environment
# 3. Installs Python dependencies for hypergraph
# 4. Compiles custom C# DLLs (RGeoLib, GeometryUtils)
# ============================================

set -e  # Exit on error

# Determine paths
HF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOS_DIR="$(dirname "$HF_ROOT")"
HYPERGRAPH_REPO="$REPOS_DIR/hypergraph"
VENV_PATH="$HF_ROOT/.venv"

# Hypergraph configuration
HYPERGRAPH_REMOTE="https://github.com/hsalehipour/hypergraph.git"  # Update this to your fork URL

echo "============================================"
echo "Hypergraph Repository Setup"
echo "============================================"
echo "HypergraphFormer root: $HF_ROOT"
echo "Repositories directory: $REPOS_DIR"
echo "Hypergraph repository: $HYPERGRAPH_REPO"
echo "Virtual environment: $VENV_PATH"
echo ""

# ============================================
# Step 1: Clone hypergraph repository
# ============================================
echo "[1/5] Checking hypergraph repository..."

if [ ! -d "$HYPERGRAPH_REPO" ]; then
    echo "Hypergraph repository not found, cloning from $HYPERGRAPH_REMOTE..."
    cd "$REPOS_DIR"
    git clone "$HYPERGRAPH_REMOTE" hypergraph
    echo "✓ Hypergraph repository cloned"
else
    echo "✓ Hypergraph repository already exists"
fi

# Verify it's the correct repository
if [ ! -d "$HYPERGRAPH_REPO/ResearchGeometryLibrary" ]; then
    echo "✗ Error: Expected directory structure not found in $HYPERGRAPH_REPO"
    echo "  Please ensure the repository contains ResearchGeometryLibrary/"
    exit 1
fi

echo ""

# ============================================
# Step 2: Check/Create virtual environment
# ============================================
echo "[2/5] Checking Python virtual environment..."

if [ ! -d "$VENV_PATH" ]; then
    echo "Virtual environment not found, creating at $VENV_PATH..."
    python3 -m venv "$VENV_PATH"
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"
echo "✓ Virtual environment activated"

# Upgrade pip
pip install --upgrade pip > /dev/null 2>&1

echo ""

# ============================================
# Step 3: Install Python dependencies
# ============================================
echo "[3/5] Installing Python dependencies..."

# Check for hypergraph Python requirements
if [ -f "$HYPERGRAPH_REPO/requirements.txt" ]; then
    echo "Installing dependencies from hypergraph/requirements.txt..."
    pip install -r "$HYPERGRAPH_REPO/requirements.txt"
    echo "✓ Python dependencies installed"
else
    echo "⚠ Warning: No requirements.txt found in hypergraph repository"
fi

# Ensure pythonnet is installed for C# interop
if ! pip list | grep -q pythonnet; then
    echo "Installing pythonnet for C# DLL interop..."
    pip install pythonnet
    echo "✓ pythonnet installed"
else
    echo "✓ pythonnet already installed"
fi

echo ""

# ============================================
# Step 4: Compile C# DLLs
# ============================================
echo "[4/5] Compiling custom C# DLLs..."

# Check if build_dlls.sh exists in hypergraph repo
BUILD_DLLS_SCRIPT="$HYPERGRAPH_REPO/build_dlls.sh"

if [ ! -f "$BUILD_DLLS_SCRIPT" ]; then
    echo "⚠ Warning: build_dlls.sh not found in hypergraph repository"
    echo "  Expected at: $BUILD_DLLS_SCRIPT"
    echo "  Please copy the build_dlls.sh script to the hypergraph repository"
    echo ""
    echo "You can manually run DLL compilation later using:"
    echo "  cd $HYPERGRAPH_REPO && ./build_dlls.sh"
else
    # Make it executable if not already
    chmod +x "$BUILD_DLLS_SCRIPT"
    
    # Run the DLL compilation script
    cd "$HYPERGRAPH_REPO"
    ./build_dlls.sh --clean
    
    echo "✓ C# DLLs compiled successfully"
fi

echo ""

# ============================================
# Step 5: Install hypergraph package in editable mode
# ============================================
echo "[5/5] Installing hypergraph package..."

# Make sure we're in the venv and hypergraph directory
source "$VENV_PATH/bin/activate"
cd "$HYPERGRAPH_REPO"

# Check if setup.py exists
if [ -f "$HYPERGRAPH_REPO/setup.py" ]; then
    echo "Installing hypergraph in editable mode (pip install -e)..."
    if pip install -e . > /dev/null 2>&1; then
        echo "✓ Hypergraph package installed"
        echo "  Now you can import with: from hypergraph.api.lib.tools import RGL"
        echo "  No need to manually add sys.path in your scripts!"
    else
        echo "⚠ Warning: Installation failed, but you can still use manual imports"
    fi
else
    echo "⚠ Warning: setup.py not found in hypergraph repository"
    echo "  You'll need to manually add sys.path in your scripts"
fi

echo ""

# ============================================
# Summary and Next Steps
# ============================================
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "Repository locations:"
echo "  • HypergraphFormer: $HF_ROOT"
echo "  • hypergraph:      $HYPERGRAPH_REPO"
echo ""
echo "Virtual environment: $VENV_PATH"
echo ""
echo "To use the hypergraph library in Python:"
echo "  1. Activate the virtual environment:"
echo "     source $VENV_PATH/bin/activate"
echo ""
echo "  2. Import with hypergraph namespace (no path setup needed!):"
echo "     from hypergraph.api.lib.tools import RGL"
echo "     from RGeoLib import Vec3d, NLine, GeometryUtils, DataNodeUtils"
echo ""
echo "  The hypergraph package is installed in editable mode, so:"
echo "    ✓ No need to set PYTHONPATH"
echo "    ✓ No need to manually add sys.path"
echo "    ✓ Works from any directory"
echo "    ✓ Maintains clean 'hypergraph' namespace"
echo ""
echo "Next steps:"
echo "  • Update HYPERGRAPH_REMOTE in this script if using your own fork"
echo "  • Ensure build_dlls.sh is in the hypergraph repository"
echo "  • Run your benchmark scripts from the benchmark/ directory"
echo ""
