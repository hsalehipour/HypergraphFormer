#!/bin/bash

# ============================================
# House-Diffusion Setup Script
# ============================================
# This script:
# 1. Verifies house_diffusion repo exists
# 2. Creates/uses virtual environment
# 3. Installs dependencies
# 4. Installs house_diffusion package
# 5. (Optional) Converts Hypergraph1K data to RPLAN format
# 6. Prepares RPLAN dataset JSONs (if raw PNGs available)
# 7. Converts JSON -> NPZ for evaluation
#
# Supports two dataset sources:
#   A) RPLAN PNGs  -> batch_raster_to_json.py -> JSONs -> convert_json_to_npz.py -> NPZ
#   B) Hypergraph1K pickles -> convert_pickle_to_npz.py -> NPZ  (legacy)
# ============================================

set -e

# Get the directory where this script is located (benchmark folder)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BENCHMARK_DIR="$SCRIPT_DIR"

# Repository paths
HF_ROOT="$(dirname "$BENCHMARK_DIR")"
REPOS_DIR="$(dirname "$HF_ROOT")"
HOUSEDIFFUSION_REPO="$REPOS_DIR/house_diffusion"
HOUSEGAN_DATA_READER="$REPOS_DIR/Housegan-data-reader"

# Virtual environment path (in HypergraphFormer root)
VENV_PATH="$HF_ROOT/.venv"

# RPLAN dataset paths
RPLAN_PNG_DIR="$HF_ROOT/dataset/rplan/dataset_original/floorplan_dataset"
RPLAN_ACCESS_DIR="$HF_ROOT/dataset/rplan/dataset_access"
RPLAN_DATASET_ROOT="$HF_ROOT/dataset/rplan/dataset_housediffusion"
RPLAN_TEST_JSON_DIR="$RPLAN_DATASET_ROOT/test"
RPLAN_TRAIN_JSON_DIR="$RPLAN_DATASET_ROOT/train"
RPLAN_NPZ_DIR="$RPLAN_DATASET_ROOT/processed_rplan"

# Split ID lists (plain text, one PNG ID per line)
RPLAN_TEST_IDS="$RPLAN_DATASET_ROOT/ids_test.txt"
RPLAN_TRAIN_IDS="$RPLAN_DATASET_ROOT/ids_train.txt"

# Model checkpoint
CHECKPOINT="$HOUSEDIFFUSION_REPO/checkpoints/pretrained_model250000.pt"

echo "============================================"
echo "House-Diffusion Setup"
echo "============================================"
echo "Benchmark directory:   $BENCHMARK_DIR"
echo "House-Diffusion repo:  $HOUSEDIFFUSION_REPO"
echo "RPLAN dataset root:    $RPLAN_DATASET_ROOT"
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
    echo "[1/5] House-Diffusion repository found"
    echo "  ✓ Path: $HOUSEDIFFUSION_REPO"
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

if [ -f "$HOUSEDIFFUSION_REPO/requirements.txt" ]; then
    pip install -r "$HOUSEDIFFUSION_REPO/requirements.txt"
    echo "  ✓ House-Diffusion requirements installed"
else
    echo "  ⚠ Warning: requirements.txt not found in house_diffusion repo"
fi
echo ""

# ============================================
# Step 4: Install house_diffusion package
# ============================================
echo "[4/5] Installing house_diffusion package..."
if [ -f "$HOUSEDIFFUSION_REPO/setup.py" ]; then
    pip install -e "$HOUSEDIFFUSION_REPO"
    echo "  ✓ House-Diffusion package installed"
else
    echo "  ⚠ Warning: setup.py not found in house_diffusion repo"
fi
echo ""

# ============================================
# Step 5: Prepare RPLAN dataset
# ============================================
echo "[5/5] Preparing RPLAN dataset for evaluation..."

# --- 5a: Check if NPZ files already exist ---
if [ -f "$RPLAN_NPZ_DIR/rplan_eval_all.npz" ] && \
   [ -f "$RPLAN_NPZ_DIR/rplan_train_all_cndist.npz" ] && \
   [ -f "$RPLAN_NPZ_DIR/rplan_eval_all_syn.npz" ]; then
    echo "  ✓ NPZ files already exist in $RPLAN_NPZ_DIR"
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
        echo "    Train JSONs are needed for corner-number distribution (cndist)."
    fi

    # --- 5d: Convert JSON -> NPZ ---
    if [ -d "$RPLAN_TEST_JSON_DIR" ] && [ -f "$RPLAN_TEST_JSON_DIR/list.txt" ]; then
        echo "  Converting JSON -> NPZ..."
        export PYTHONPATH="$HOUSEDIFFUSION_REPO:$HF_ROOT:$PYTHONPATH"

        CONVERT_CMD="python3 \"$BENCHMARK_DIR/utils/convert_json_to_npz.py\" \
            --json_dir \"$RPLAN_TEST_JSON_DIR\" \
            --output_dir \"$RPLAN_NPZ_DIR\""

        if [ -d "$RPLAN_TRAIN_JSON_DIR" ] && [ -f "$RPLAN_TRAIN_JSON_DIR/list.txt" ]; then
            echo "    Using train data from $RPLAN_TRAIN_JSON_DIR for corner distribution"
            CONVERT_CMD="$CONVERT_CMD --train_json_dir \"$RPLAN_TRAIN_JSON_DIR\""
        else
            echo "    ⚠ Train JSONs not available; cndist will be built from eval data"
        fi

        eval $CONVERT_CMD
        echo "  ✓ NPZ conversion complete"
    else
        echo "  ⚠ Cannot generate NPZ: no test JSONs available"
    fi
fi
echo ""

# ============================================
# Verify checkpoint
# ============================================
if [ -f "$CHECKPOINT" ]; then
    echo "✓ Model checkpoint found: $CHECKPOINT"
else
    echo "⚠ Model checkpoint not found: $CHECKPOINT"
    echo "  Available checkpoints:"
    ls -1 "$HOUSEDIFFUSION_REPO"/checkpoints/*.pt 2>/dev/null | head -10 || echo "    None found"
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
echo "  House-Diffusion repo:  $HOUSEDIFFUSION_REPO"
echo "  Virtual environment:   $VENV_PATH"
echo "  RPLAN dataset root:    $RPLAN_DATASET_ROOT"
echo "  NPZ directory:         $RPLAN_NPZ_DIR"
echo "  Checkpoint:            $CHECKPOINT"
echo ""
echo "To run evaluation:"
echo "  cd $BENCHMARK_DIR"
echo "  ./eval_housediffusion.sh $RPLAN_TEST_JSON_DIR"
echo ""
