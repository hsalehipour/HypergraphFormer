#!/bin/bash

# ============================================
# House-GAN++ Setup Script
# ============================================
# This script:
# 1. Clones houseganpp and Housegan-data-reader repos if needed
# 2. Creates/uses virtual environment
# 3. Installs dependencies
# 4. Installs houseganpp package (editable mode)
# 5. Prepares RPLAN dataset (JSON -> pickle for HouseGAN++)
#
# Supports two dataset sources:
#   A) RPLAN PNGs  -> batch_raster_to_json.py -> JSONs -> convert_json_to_pickle.py -> pickle
#   B) Hypergraph1K -> convert_hypergraph_to_rplan.py -> pickle  (legacy)
# ============================================

set -e

# Get the directory where this script is located (benchmark folder)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BENCHMARK_DIR="$SCRIPT_DIR"

# Repository paths
HF_ROOT="$(dirname "$BENCHMARK_DIR")"
REPOS_DIR="$(dirname "$HF_ROOT")"
HOUSEGANPP_REPO="$REPOS_DIR/houseganpp"
HOUSEGANPP_REMOTE="https://github.com/hsalehipour/houseganpp"
HOUSEGAN_DATA_READER="$REPOS_DIR/Housegan-data-reader"

# Virtual environment path (in HypergraphFormer root)
VENV_PATH="$HF_ROOT/.venv"

# RPLAN dataset paths
RPLAN_PNG_DIR="$HF_ROOT/dataset/rplan/dataset_original/floorplan_dataset"
RPLAN_ACCESS_DIR="$HF_ROOT/dataset/rplan/dataset_access"
RPLAN_DATASET_ROOT="$HF_ROOT/dataset/rplan/dataset_housediffusion"
RPLAN_TEST_JSON_DIR="$RPLAN_DATASET_ROOT/test"
RPLAN_TRAIN_JSON_DIR="$RPLAN_DATASET_ROOT/train"

# Split ID lists (plain text, one PNG ID per line)
RPLAN_TEST_IDS="$RPLAN_DATASET_ROOT/ids_test.txt"
RPLAN_TRAIN_IDS="$RPLAN_DATASET_ROOT/ids_train.txt"

# HouseGAN++ pickle output paths
RPLAN_PKL_DIR="$RPLAN_DATASET_ROOT/processed_rplan_pickles"
HOUSEGANPP_TEST_PKL="$RPLAN_PKL_DIR/test_data.pkl"
HOUSEGANPP_TRAIN_PKL="$RPLAN_PKL_DIR/train_data.pkl"

# Checkpoint paths
HOUSEGANPP_CHECKPOINTS_DIR="$BENCHMARK_DIR/houseganpp_checkpoints"

echo "============================================"
echo "House-GAN++ Setup"
echo "============================================"
echo "Benchmark directory:   $BENCHMARK_DIR"
echo "HouseGAN++ repo:       $HOUSEGANPP_REPO"
echo "RPLAN dataset root:    $RPLAN_DATASET_ROOT"
echo ""

# ============================================
# Step 1: Clone repos if needed
# ============================================
if [ ! -d "$HOUSEGANPP_REPO" ]; then
    echo "[1/5] Cloning houseganpp repository..."
    git clone "$HOUSEGANPP_REMOTE" "$HOUSEGANPP_REPO"
    echo "  ✓ Repository cloned successfully"
else
    echo "[1/5] HouseGAN++ repository already exists"
    echo "  ✓ Path: $HOUSEGANPP_REPO"
fi

if [ ! -d "$HOUSEGAN_DATA_READER" ]; then
    echo "  Housegan-data-reader not found, cloning..."
    git clone https://github.com/hsalehipour/Housegan-data-reader.git "$HOUSEGAN_DATA_READER"
    echo "  ✓ Cloned Housegan-data-reader to $HOUSEGAN_DATA_READER"
else
    echo "  ✓ Housegan-data-reader already exists at $HOUSEGAN_DATA_READER"
fi
echo ""

# ============================================
# Step 2: Create virtual environment if needed
# ============================================
if [ ! -d "$VENV_PATH" ]; then
    echo "[2/5] Creating virtual environment..."
    python3 -m venv "$VENV_PATH"
    echo "  ✓ Virtual environment created at $VENV_PATH"
else
    echo "[2/5] Virtual environment already exists"
    echo "  ✓ Using existing environment at $VENV_PATH"
fi
echo ""

# ============================================
# Step 3: Install requirements
# ============================================
echo "[3/5] Installing requirements..."
source "$VENV_PATH/bin/activate"

pip install --upgrade pip

if [ -f "$HOUSEGANPP_REPO/requirements.txt" ]; then
    pip install -r "$HOUSEGANPP_REPO/requirements.txt"
    echo "  ✓ HouseGAN++ requirements installed"
else
    echo "  ⚠ Warning: requirements.txt not found in houseganpp repo"
fi
echo ""

# ============================================
# Step 4: Install houseganpp package
# ============================================
echo "[4/5] Installing houseganpp package..."
if [ -f "$HOUSEGANPP_REPO/pyproject.toml" ] || [ -f "$HOUSEGANPP_REPO/setup.py" ]; then
    pip install -e "$HOUSEGANPP_REPO"
    echo "  ✓ HouseGAN++ package installed"
else
    echo "  ⚠ Warning: pyproject.toml or setup.py not found in houseganpp repo"
fi
echo ""

# ============================================
# Step 5: Prepare dataset (pickle files)
# ============================================
echo "[5/5] Preparing dataset for HouseGAN++..."

# --- 5a: Check if pickle files already exist ---
if [ -f "$HOUSEGANPP_TEST_PKL" ]; then
    echo "  ✓ Test pickle already exists: $HOUSEGANPP_TEST_PKL"
    echo "    Skipping dataset preparation"
else
    # --- 5b: Extract split ID lists from dataset_access JSONs ---
    mkdir -p "$RPLAN_DATASET_ROOT"

    if [ ! -f "$RPLAN_TEST_IDS" ]; then
        if [ -f "$RPLAN_ACCESS_DIR/data_test.json" ]; then
            echo "  Extracting test IDs from data_test.json..."
            python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
with open(sys.argv[2], 'w') as out:
    for s in data:
        out.write(str(s['name']) + '\n')
" "$RPLAN_ACCESS_DIR/data_test.json" "$RPLAN_TEST_IDS"
            echo "  ✓ $(wc -l < "$RPLAN_TEST_IDS") test IDs -> $RPLAN_TEST_IDS"
        else
            echo "  ⚠ Cannot extract test IDs: $RPLAN_ACCESS_DIR/data_test.json not found"
        fi
    else
        echo "  ✓ Test ID list already exists: $RPLAN_TEST_IDS"
    fi

    if [ ! -f "$RPLAN_TRAIN_IDS" ]; then
        if [ -f "$RPLAN_ACCESS_DIR/data_train.json" ]; then
            echo "  Extracting train IDs from data_train.json..."
            python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
with open(sys.argv[2], 'w') as out:
    for s in data:
        out.write(str(s['name']) + '\n')
" "$RPLAN_ACCESS_DIR/data_train.json" "$RPLAN_TRAIN_IDS"
            echo "  ✓ $(wc -l < "$RPLAN_TRAIN_IDS") train IDs -> $RPLAN_TRAIN_IDS"
        else
            echo "  ⚠ Cannot extract train IDs: $RPLAN_ACCESS_DIR/data_train.json not found"
        fi
    else
        echo "  ✓ Train ID list already exists: $RPLAN_TRAIN_IDS"
    fi

    # --- 5c: Generate JSONs from RPLAN PNGs ---
    BATCH_SCRIPT="$HOUSEGAN_DATA_READER/batch_raster_to_json.py"

    # Test split
    if [ -d "$RPLAN_TEST_JSON_DIR" ] && [ -f "$RPLAN_TEST_JSON_DIR/list.txt" ]; then
        echo "  ✓ Test JSONs already exist in $RPLAN_TEST_JSON_DIR"
    elif [ -f "$BATCH_SCRIPT" ] && [ -d "$RPLAN_PNG_DIR" ] && [ -f "$RPLAN_TEST_IDS" ]; then
        echo "  Converting RPLAN PNGs to JSON (test split)..."
        export PYTHONPATH="$HOUSEGAN_DATA_READER:$PYTHONPATH"
        python3 "$BATCH_SCRIPT" \
            --png-dir "$RPLAN_PNG_DIR" \
            --id-list "$RPLAN_TEST_IDS" \
            --output-dir "$RPLAN_TEST_JSON_DIR"
        echo "  ✓ Test JSONs generated"
    else
        echo "  ⚠ No test JSONs found and cannot generate them."
        echo "    Either place JSONs in $RPLAN_TEST_JSON_DIR"
        echo "    or ensure $BATCH_SCRIPT, $RPLAN_PNG_DIR, and $RPLAN_TEST_IDS exist."
    fi

    # Train split
    if [ -d "$RPLAN_TRAIN_JSON_DIR" ] && [ -f "$RPLAN_TRAIN_JSON_DIR/list.txt" ]; then
        echo "  ✓ Train JSONs already exist in $RPLAN_TRAIN_JSON_DIR"
    elif [ -f "$BATCH_SCRIPT" ] && [ -d "$RPLAN_PNG_DIR" ] && [ -f "$RPLAN_TRAIN_IDS" ]; then
        echo "  Converting RPLAN PNGs to JSON (train split)..."
        export PYTHONPATH="$HOUSEGAN_DATA_READER:$PYTHONPATH"
        python3 "$BATCH_SCRIPT" \
            --png-dir "$RPLAN_PNG_DIR" \
            --id-list "$RPLAN_TRAIN_IDS" \
            --output-dir "$RPLAN_TRAIN_JSON_DIR"
        echo "  ✓ Train JSONs generated"
    else
        echo "  ⚠ No train JSONs found and cannot generate them."
    fi

    # --- 5d: Convert JSON -> pickle ---
    CONVERT_PKL_SCRIPT="$BENCHMARK_DIR/utils/convert_json_to_pickle.py"

    if [ -d "$RPLAN_TEST_JSON_DIR" ] && [ -f "$RPLAN_TEST_JSON_DIR/list.txt" ]; then
        echo "  Converting test JSONs -> pickle..."
        export PYTHONPATH="$HF_ROOT:$PYTHONPATH"
        mkdir -p "$(dirname "$HOUSEGANPP_TEST_PKL")"
        python3 "$CONVERT_PKL_SCRIPT" \
            --json_dir "$RPLAN_TEST_JSON_DIR" \
            --output "$HOUSEGANPP_TEST_PKL"
        echo "  ✓ Test pickle saved to $HOUSEGANPP_TEST_PKL"
    else
        echo "  ⚠ Cannot generate test pickle: no test JSONs available"
    fi

    if [ -d "$RPLAN_TRAIN_JSON_DIR" ] && [ -f "$RPLAN_TRAIN_JSON_DIR/list.txt" ]; then
        echo "  Converting train JSONs -> pickle..."
        export PYTHONPATH="$HF_ROOT:$PYTHONPATH"
        mkdir -p "$(dirname "$HOUSEGANPP_TRAIN_PKL")"
        python3 "$CONVERT_PKL_SCRIPT" \
            --json_dir "$RPLAN_TRAIN_JSON_DIR" \
            --output "$HOUSEGANPP_TRAIN_PKL"
        echo "  ✓ Train pickle saved to $HOUSEGANPP_TRAIN_PKL"
    else
        echo "  ⚠ Cannot generate train pickle: no train JSONs available"
    fi
fi
echo ""

# ============================================
# Verify checkpoint
# ============================================
CHECKPOINT="$HOUSEGANPP_REPO/checkpoints/ft_housegan_final.pth"
if [ -f "$CHECKPOINT" ]; then
    echo "✓ Model checkpoint found: $CHECKPOINT"
else
    echo "⚠ Model checkpoint not found: $CHECKPOINT"
    echo "  Available checkpoints:"
    ls -1 "$HOUSEGANPP_REPO"/checkpoints/*.pth 2>/dev/null | head -10 || echo "    None found"
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
echo "  HouseGAN++ repo:      $HOUSEGANPP_REPO"
echo "  Virtual environment:   $VENV_PATH"
echo "  RPLAN dataset root:    $RPLAN_DATASET_ROOT"
echo "  Test pickle:           $HOUSEGANPP_TEST_PKL"
echo "  Checkpoint:            $CHECKPOINT"
echo ""
echo "To run evaluation:"
echo "  cd $BENCHMARK_DIR"
echo "  ./eval_houseganpp.sh"
echo ""
