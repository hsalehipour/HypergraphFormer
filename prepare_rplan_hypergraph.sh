#!/usr/bin/env bash
#
# Prepare the RPLAN hypergraph dataset end-to-end.
#
#   1. Download RPLAN original floorplan PNGs  (required, from Box)
#   2. Download DiffPlanner data  (required, from GitHub)
#   3. Build dataset_access from DiffPlanner + original PNGs  (optional, off by default)
#   4. Export BSP hypergraphs  (dataset_hypergraph_la0)
#   5. Post-process: fix disconnected access graphs  (dataset_hypergraph_la0_fixed)
#   6. Visualise any remaining disconnections  (debug/)  [--debug only]
#
# Usage:
#   ./prepare_rplan_hypergraph.sh              # steps 1-2, 4-5 (reuse existing dataset_access)
#   ./prepare_rplan_hypergraph.sh --access     # steps 1-5 (rebuild dataset_access from scratch)
#   ./prepare_rplan_hypergraph.sh --clean      # wipe hypergraph/debug data, then rebuild steps 4-5
#   ./prepare_rplan_hypergraph.sh --debug      # also run step 6 (visualise disconnections)
#   ./prepare_rplan_hypergraph.sh -j 32        # override worker count

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────
WORKERS=$(nproc)
LOOKAHEAD=0
BUILD_ACCESS=false
CLEAN=false
DEBUG=false

DATASET_ROOT="dataset/rplan"
ORIGINAL_DIR="${DATASET_ROOT}/dataset_original"
DIFFPLANNER_DIR="${DATASET_ROOT}/dataset_diffplanner"
ACCESS_DIR="${DATASET_ROOT}/dataset_access"
HYPERGRAPH_DIR="${DATASET_ROOT}/dataset_hypergraph_la0"
FIXED_DIR="${DATASET_ROOT}/dataset_hypergraph_la0_fixed"
DEBUG_DIR="debug"

RPLAN_DOWNLOAD_URL="https://berkeley.app.box.com/index.php?rm=box_download_shared_file&shared_name=m8369bxmaixananf4otajydraxa7y92v&file_id=f_2068745315403"
DIFFPLANNER_URL="https://github.com/shidong-wang/DiffPlanner/releases/download/dataset/dataset.zip"

# ── Parse arguments ───────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --access)   BUILD_ACCESS=true; shift ;;
        --clean)    CLEAN=true;        shift ;;
        --debug)    DEBUG=true;        shift ;;
        -j)         WORKERS="$2";      shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--access] [--clean] [-j WORKERS]"
            echo ""
            echo "  --access   Rebuild dataset_access (cleans and rebuilds it)"
            echo "  --clean    Remove hypergraph and debug data, then rebuild (does NOT touch dataset_access)"
            echo "  --debug    Visualise remaining disconnections after postprocessing"
            echo "  -j N       Number of parallel workers (default: nproc = $(nproc))"
            exit 0
            ;;
        *)  echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== RPLAN Hypergraph Preparation ==="
echo "  workers:        ${WORKERS}"
echo "  lookahead:      ${LOOKAHEAD}"
echo "  build access:   ${BUILD_ACCESS}"
echo "  clean first:    ${CLEAN}"
echo "  debug vis:      ${DEBUG}"
echo ""

# ── Step 0: Clean (optional) ─────────────────────────────────────────
if $CLEAN; then
    echo ">> Cleaning hypergraph and debug data..."
    rm -rf "${HYPERGRAPH_DIR}" "${FIXED_DIR}"
    rm -rf "${DEBUG_DIR}" debug_*
    echo ""
fi

# ── Step 1: Download RPLAN original dataset ───────────────────────────
if [ ! -d "${ORIGINAL_DIR}" ]; then
    echo ">> Downloading RPLAN original dataset..."
    mkdir -p "${DATASET_ROOT}"
    TMP_ZIP=$(mktemp /tmp/rplan_original_XXXXXX.zip)
    wget -q --show-progress -O "${TMP_ZIP}" "${RPLAN_DOWNLOAD_URL}"

    echo ">> Extracting to ${ORIGINAL_DIR}..."
    TMP_EXTRACT=$(mktemp -d /tmp/rplan_extract_XXXXXX)
    unzip -q "${TMP_ZIP}" -d "${TMP_EXTRACT}"
    rm -f "${TMP_ZIP}"

    # The zip nests as RPLAN/dataset/floorplan_dataset/; flatten to match
    # the expected layout: dataset_original/floorplan_dataset/*.png
    if [ -d "${TMP_EXTRACT}/RPLAN/dataset" ]; then
        mv "${TMP_EXTRACT}/RPLAN/dataset" "${ORIGINAL_DIR}"
    else
        mv "${TMP_EXTRACT}" "${ORIGINAL_DIR}"
    fi
    rm -rf "${TMP_EXTRACT}"
    echo "   Done."
    echo ""
else
    echo ">> RPLAN original data already present at ${ORIGINAL_DIR}, skipping download."
    echo ""
fi

# ── Step 2: Download DiffPlanner data ─────────────────────────────────
if [ ! -d "${DIFFPLANNER_DIR}" ]; then
    echo ">> Downloading DiffPlanner dataset..."
    mkdir -p "${DATASET_ROOT}"
    TMP_ZIP=$(mktemp /tmp/diffplanner_dataset_XXXXXX.zip)
    wget -q --show-progress -O "${TMP_ZIP}" "${DIFFPLANNER_URL}"

    echo ">> Extracting to ${DIFFPLANNER_DIR}..."
    TMP_EXTRACT=$(mktemp -d /tmp/diffplanner_extract_XXXXXX)
    unzip -q "${TMP_ZIP}" -d "${TMP_EXTRACT}"
    rm -f "${TMP_ZIP}"

    # The zip nests as dataset/dataset_json/data_*.json; flatten so
    # data_*.json lives directly under DIFFPLANNER_DIR.
    if [ -d "${TMP_EXTRACT}/dataset/dataset_json" ]; then
        mv "${TMP_EXTRACT}/dataset/dataset_json" "${DIFFPLANNER_DIR}"
    elif [ -d "${TMP_EXTRACT}/dataset" ]; then
        mv "${TMP_EXTRACT}/dataset" "${DIFFPLANNER_DIR}"
    else
        mv "${TMP_EXTRACT}" "${DIFFPLANNER_DIR}"
    fi
    rm -rf "${TMP_EXTRACT}"
    echo "   Done."
    echo ""
else
    echo ">> DiffPlanner data already present at ${DIFFPLANNER_DIR}, skipping download."
    echo ""
fi

# ── Step 3: Build dataset_access (optional) ───────────────────────────
if $BUILD_ACCESS; then
    echo ">> Building dataset_access..."
    rm -rf "${ACCESS_DIR}"
    python rplan/build_dataset_access.py \
        --diffplanner-dir "${DIFFPLANNER_DIR}" \
        --output-dir "${ACCESS_DIR}" \
        -j "${WORKERS}"
    echo ""
else
    if [ ! -d "${ACCESS_DIR}" ]; then
        echo "ERROR: ${ACCESS_DIR} does not exist."
        echo "       Run with --access to build it, or create it manually."
        exit 1
    fi
    echo ">> Skipping dataset_access build (use --access to rebuild)."
    echo ""
fi

# ── Step 4: Export BSP hypergraphs ────────────────────────────────────
echo ">> Exporting BSP hypergraphs (lookahead=${LOOKAHEAD})..."
rm -rf "${HYPERGRAPH_DIR}"
python rplan/rplan_to_hypergraph.py \
    --export \
    --export-dir "${HYPERGRAPH_DIR}" \
    --lookahead "${LOOKAHEAD}" \
    -j "${WORKERS}"
echo ""

# ── Step 5: Post-process (fix disconnected access graphs) ────────────
echo ">> Post-processing: fixing disconnected access graphs..."
rm -rf "${FIXED_DIR}"
python rplan/postprocess_hypergraphs.py \
    --dataset-dir "${HYPERGRAPH_DIR}" \
    --output-dir "${FIXED_DIR}" \
    -j "${WORKERS}"
echo ""

# ── Step 6: Visualise remaining disconnections (optional) ────────────
if $DEBUG; then
    echo ">> Visualising remaining disconnections..."
    rm -rf "${DEBUG_DIR}"
    python rplan/visualize_disconnected_accessgraphs.py \
        --dataset-dir "${FIXED_DIR}" \
        --output-dir "${DEBUG_DIR}"
    echo ""
fi

echo "=== All done ==="
echo "  Hypergraph data: ${FIXED_DIR}"
if $DEBUG; then
    echo "  Debug images:    ${DEBUG_DIR}"
fi
