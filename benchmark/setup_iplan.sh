#!/bin/bash

# ============================================
# iPLAN Setup Script
# ============================================
# Prepares everything benchmark/src/evaluate_iplan.py needs to run the
# OurII cascade end-to-end (Stage 2 room locations + Stage 3 room
# partitions, with rTypes := gt_rTypes; Stage 1 is skipped):
#
#   1. Clone the iPLAN repository (skip if it already exists). Uses
#      hsalehipour's fork by default.
#
#   2. Apply 3 small inference-time patches to the upstream code so it
#      can be imported and run from outside its own repo. These are not
#      changes to the algorithm; they fix three pre-existing
#      import/path bugs that only matter when iPLAN is used as a
#      library by an external caller (us). Skipped if already applied.
#
#         room_partition/models/__init__.py    expose the inference
#                                              variant of the GAN
#                                              generator; the upstream
#                                              __init__ wires up the
#                                              training variant which
#                                              transitively does
#                                              ``import utils`` and only
#                                              works from inside its
#                                              folder.
#         room_partition/models/loss_layer.py  load box_utils by file
#                                              path; the upstream import
#                                              path
#                                              ``room_partition.models.utils.box_utils``
#                                              points at a folder that
#                                              does not exist.
#         room_partition/models/net.py         resolve renderer.pkl
#                                              relative to the file
#                                              instead of cwd.
#
#   3. Ensure the `iplan` conda env exists with the exact Python 3.8 +
#      torch/cv2 stack iPLAN needs. If the env is missing, we create it
#      and pip-install a version-pinned dependency list (frozen from a
#      known-working env, including the cu113 torch wheel from the
#      official PyTorch index). If the env already exists, we leave it
#      alone unless --force-env is passed -- bumping versions of a
#      working env is the most common way to break GPU stacks. gdown
#      (used by Step 4) is part of the pinned deps, so it lands in the
#      env automatically.
#
#   4. Download the iPLAN trained-model bundle from the official Google
#      Drive folder using the env's gdown, and place each .pth/.pkl at
#      the EXACT path the evaluator's _load_iplan_models() expects:
#
#         <iPlan>/room_location/pretrained_model/resnet18-5c106cde.pth
#         <iPlan>/room_location/weights/living_resnet18_300.pth
#         <iPlan>/room_location/weights/living_fc1_300.pth
#         <iPlan>/room_location/weights/location_resnet18_100.pth
#         <iPlan>/room_location/weights/location_up1_100.pth
#         <iPlan>/room_location/weights/location_embed_100.pth
#         <iPlan>/room_partition/weights/G_net_210.pth
#         <iPlan>/room_partition/weights/renderer.pkl
#
#   5. Build dataset/rplan/dataset_iplan/ by running
#      benchmark/utils/convert_png_to_iplan_mat.py on the test split.
#      Depends on dataset_original/floorplan_dataset/*.png being there
#      (downloaded by ./prepare_rplan_hypergraph.sh from the HF root).
#
# iPLAN has no setup.py / requirements.txt. The evaluator imports its
# sub-packages by adding the repo root to sys.path at runtime, so there
# is no pip install of iPLAN itself.
#
# Usage:
#   ./benchmark/setup_iplan.sh
#   ./benchmark/setup_iplan.sh --force-checkpoints   # re-download weights
#   ./benchmark/setup_iplan.sh --force-dataset       # rebuild dataset_iplan/
#   ./benchmark/setup_iplan.sh --force-patches       # re-apply iPLAN patches
#   ./benchmark/setup_iplan.sh --force-env           # re-install pinned deps into existing env
#   ./benchmark/setup_iplan.sh --skip-dataset        # skip Step 5 (e.g. rerun later)
# ============================================

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BENCHMARK_DIR="$SCRIPT_DIR"
HF_ROOT="$(dirname "$BENCHMARK_DIR")"
REPOS_DIR="$(dirname "$HF_ROOT")"
IPLAN_REPO="$REPOS_DIR/iPlan"

IPLAN_GIT_URL="https://github.com/hsalehipour/iPLAN-Interactive-and-Procedural-Layout-Planning.git"

# Google Drive folder advertised in iPLAN's upstream README under
# "The trained model can be download here for test."
IPLAN_GDRIVE_FOLDER_ID="1TRMKu6zw-pgEpGja2zTCixA2WhhU5KXr"

CONDA_ENV_NAME="iplan"

# Weight destinations -- keep in sync with _load_iplan_models() in
# benchmark/src/evaluate_iplan.py (lines 232-264 at time of writing).
LOC_PRETRAINED_DIR="$IPLAN_REPO/room_location/pretrained_model"
LOC_WEIGHTS_DIR="$IPLAN_REPO/room_location/weights"
PART_WEIGHTS_DIR="$IPLAN_REPO/room_partition/weights"

REQUIRED_WEIGHTS=(
    "$LOC_PRETRAINED_DIR/resnet18-5c106cde.pth"
    "$LOC_WEIGHTS_DIR/living_resnet18_300.pth"
    "$LOC_WEIGHTS_DIR/living_fc1_300.pth"
    "$LOC_WEIGHTS_DIR/location_resnet18_100.pth"
    "$LOC_WEIGHTS_DIR/location_up1_100.pth"
    "$LOC_WEIGHTS_DIR/location_embed_100.pth"
    "$PART_WEIGHTS_DIR/G_net_210.pth"
    "$PART_WEIGHTS_DIR/renderer.pkl"
)

# Dataset paths.
RPLAN_PNG_DIR="$HF_ROOT/dataset/rplan/dataset_original/floorplan_dataset"
DATASET_IPLAN_DIR="$HF_ROOT/dataset/rplan/dataset_iplan"
CONVERT_SCRIPT="$BENCHMARK_DIR/utils/convert_png_to_iplan_mat.py"

# ── CLI ───────────────────────────────────────────────────────────────
FORCE_CKPT=false
FORCE_DATASET=false
FORCE_PATCHES=false
FORCE_ENV=false
SKIP_DATASET=false
for arg in "$@"; do
    case "$arg" in
        --force-checkpoints) FORCE_CKPT=true ;;
        --force-dataset)     FORCE_DATASET=true ;;
        --force-patches)     FORCE_PATCHES=true ;;
        --force-env)         FORCE_ENV=true ;;
        --skip-dataset)      SKIP_DATASET=true ;;
        -h|--help)
            sed -n '3,80p' "$0"
            exit 0
            ;;
        *)  echo "Unknown option: $arg"; exit 1 ;;
    esac
done

echo "============================================"
echo "iPLAN Setup"
echo "============================================"
echo "Repos directory:       $REPOS_DIR"
echo "iPLAN repo:            $IPLAN_REPO"
echo "Conda env:             $CONDA_ENV_NAME"
echo "Dataset (input PNGs):  $RPLAN_PNG_DIR"
echo "Dataset (iPLAN .mat):  $DATASET_IPLAN_DIR"
echo ""

# ============================================
# Step 1: Clone iPLAN repo
# ============================================
if [ ! -d "$IPLAN_REPO" ]; then
    echo "[1/5] Cloning iPLAN repository..."
    git clone --depth 1 "$IPLAN_GIT_URL" "$IPLAN_REPO"
    echo "  ✓ Cloned to $IPLAN_REPO"
else
    echo "[1/5] iPLAN repository already present"
    echo "  ✓ Path: $IPLAN_REPO"
fi
echo ""

# ============================================
# Step 2: Apply inference-time patches
# ============================================
echo "[2/5] Applying inference patches to iPLAN..."

# Marker file: presence means all 3 patches have been applied. We use a
# file inside the iPLAN repo so re-cloning resets the marker correctly.
PATCH_MARKER="$IPLAN_REPO/.hf_inference_patches_applied"

if [ -f "$PATCH_MARKER" ] && ! $FORCE_PATCHES; then
    echo "  ✓ Patches already applied (marker: $PATCH_MARKER)"
    echo "    (re-run with --force-patches to re-apply)"
else
    # ---- Patch A: room_partition/models/__init__.py ------------------
    # Look for the marker that's only in the patched version. If absent,
    # rewrite the file; if present, leave it alone.
    INIT_PY="$IPLAN_REPO/room_partition/models/__init__.py"
    if ! grep -qF "FloorPlanRNNTest" "$INIT_PY" 2>/dev/null; then
        cat > "$INIT_PY" <<'PATCH_A_EOF'
# `from .floorplan_rnn import FloorPlanRNN` is intentionally NOT done here:
# floorplan_rnn.py uses `import utils`, which only resolves when the working
# directory happens to be `room_partition/`. Inference paths use the test
# wrapper below, which imports `from room_partition import utils` correctly.
from .floorplan_rnn_test import FloorPlanRNN as FloorPlanRNNTest  # noqa: F401
PATCH_A_EOF
        echo "  ✓ Patched room_partition/models/__init__.py"
    else
        echo "  ✓ room_partition/models/__init__.py already patched"
    fi

    # ---- Patch B: room_partition/models/loss_layer.py ---------------
    # Replace the broken first line with a path-based importlib loader
    # for box_utils. Detection looks for the unpatched first line
    # specifically, so re-running on an already-patched file is a no-op
    # (we don't want to prepend the same block twice).
    LOSS_LAYER_PY="$IPLAN_REPO/room_partition/models/loss_layer.py"
    UNPATCHED_FIRST_LINE='import room_partition.models.utils.box_utils as box_utils'
    if grep -qxF "$UNPATCHED_FIRST_LINE" "$LOSS_LAYER_PY"; then
        TMP_LOSS=$(mktemp)
        {
            cat <<'PATCH_B_EOF'
# `room_partition/utils.py` (a module) shadows `room_partition/utils/` (a package),
# so `from room_partition.utils import box_utils` cannot resolve. Load box_utils
# directly from its file path to avoid renaming either side.
import importlib.util as _ilu
import os as _os

_BOX_UTILS_PATH = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "utils", "box_utils.py"
)
_spec = _ilu.spec_from_file_location("room_partition_box_utils", _BOX_UTILS_PATH)
box_utils = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(box_utils)
PATCH_B_EOF
            # Print everything *after* the unpatched first line, so we
            # drop the broken import and prepend our replacement.
            grep -vxF "$UNPATCHED_FIRST_LINE" "$LOSS_LAYER_PY"
        } > "$TMP_LOSS"
        mv "$TMP_LOSS" "$LOSS_LAYER_PY"
        echo "  ✓ Patched room_partition/models/loss_layer.py"
    else
        echo "  ✓ loss_layer.py box_utils import already patched"
    fi

    # ---- Patch C: room_partition/models/net.py ----------------------
    # Make the renderer.pkl lookup cwd-independent. We do this with two
    # small in-place sed substitutions so we don't have to rewrite the
    # whole file (it's long and would be brittle).
    NET_PY="$IPLAN_REPO/room_partition/models/net.py"
    if ! grep -q "^import os$" "$NET_PY"; then
        # Add `import os` before `import torch` (only on first match).
        sed -i '0,/^import torch$/{s|^import torch$|import os\nimport torch|}' "$NET_PY"
    fi
    if grep -q '^        pretrain_model = "\./weights/renderer\.pkl"$' "$NET_PY"; then
        # Replace the single-line cwd-relative load with a candidate-list
        # that falls back to a path resolved relative to net.py.
        python3 - "$NET_PY" <<'PATCH_C_EOF'
import sys
path = sys.argv[1]
with open(path) as f:
    src = f.read()
old = '        pretrain_model = "./weights/renderer.pkl"\n'
new = (
    '        # Resolve renderer.pkl relative to this file rather than cwd, so the\n'
    '        # network loads correctly no matter where inference is launched from.\n'
    '        candidates = [\n'
    '            "./weights/renderer.pkl",\n'
    '            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights", "renderer.pkl"),\n'
    '        ]\n'
    '        pretrain_model = next((p for p in candidates if os.path.exists(p)), candidates[0])\n'
)
if old not in src:
    sys.exit("expected line not found; aborting Patch C")
with open(path, "w") as f:
    f.write(src.replace(old, new, 1))
PATCH_C_EOF
        echo "  ✓ Patched room_partition/models/net.py"
    else
        echo "  ✓ net.py renderer.pkl path already resolved (or upstream changed)"
    fi

    touch "$PATCH_MARKER"
    echo "  ✓ All inference patches applied; marker written"
fi
echo ""

# ============================================
# Step 3: Conda env (create + install pinned deps if missing)
# ============================================
echo "[3/5] Setting up conda env '$CONDA_ENV_NAME' ..."

# Pinned versions captured from a known-working iplan env. torch is the
# CUDA-11.3 build, only available from PyTorch's own wheel index.
# Bumping these is the most common way to break the GAN stack, so we
# keep them frozen and only re-install on --force-env.
IPLAN_PY_VERSION="3.8"
IPLAN_TORCH_INDEX="https://download.pytorch.org/whl/cu113"
IPLAN_DEPS=(
    "torch==1.10.1+cu113"
    "torchvision==0.11.2+cu113"
    "numpy==1.23.5"
    "scipy==1.10.1"
    "scikit-image==0.21.0"
    "opencv-python==4.13.0.92"
    "pillow==10.4.0"
    "matplotlib==3.7.5"
    "shapely==2.0.7"
    "networkx==3.1"
    "tqdm==4.67.3"
    # gdown is used by Step 4 (Google-Drive weight download). Loosely
    # pinned so we always get a working Drive client.
    "gdown"
)

# `conda env list` works whether or not conda is currently activated.
if ! command -v conda >/dev/null 2>&1; then
    echo "  ✗ 'conda' not found on PATH."
    echo "    Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

env_exists=false
if conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
    env_exists=true
fi

CONDA_ENV_PIP=""
CONDA_ENV_PY=""

if ! $env_exists; then
    echo "  Env '$CONDA_ENV_NAME' not found; creating with Python $IPLAN_PY_VERSION ..."
    # `conda create -y` is non-interactive. Use `--quiet` to keep output
    # readable but not silent.
    conda create -n "$CONDA_ENV_NAME" python="$IPLAN_PY_VERSION" --quiet -y
    echo "  ✓ Env created"
    install_deps=true
elif $FORCE_ENV; then
    echo "  Env '$CONDA_ENV_NAME' present; --force-env set, will re-install pinned deps"
    install_deps=true
else
    echo "  ✓ Env '$CONDA_ENV_NAME' already present (use --force-env to re-install deps)"
    install_deps=false
fi

# Resolve the env's python/pip binaries. We do this AFTER potentially
# creating the env so the prefix actually exists. Use `conda run` if the
# canonical miniconda layout isn't where we expect.
CONDA_PREFIX_GUESS="$(conda info --base 2>/dev/null)/envs/$CONDA_ENV_NAME"
if [ -x "$CONDA_PREFIX_GUESS/bin/python" ]; then
    CONDA_ENV_PY="$CONDA_PREFIX_GUESS/bin/python"
    CONDA_ENV_PIP="$CONDA_PREFIX_GUESS/bin/pip"
else
    echo "  ✗ Could not locate python at $CONDA_PREFIX_GUESS/bin/python"
    echo "    (conda env may be in a non-standard prefix; try 'conda info --envs')"
    exit 1
fi

PY_VER=$("$CONDA_ENV_PY" -c "import sys; print('.'.join(map(str, sys.version_info[:3])))")
echo "  ✓ Python $PY_VER at $CONDA_ENV_PY"

if $install_deps; then
    echo "  Installing pinned dependencies (torch from $IPLAN_TORCH_INDEX) ..."
    "$CONDA_ENV_PIP" install --upgrade pip --quiet
    # --extra-index-url lets the cu113 torch wheel resolve while still
    # pulling everything else from PyPI.
    "$CONDA_ENV_PIP" install \
        --extra-index-url "$IPLAN_TORCH_INDEX" \
        "${IPLAN_DEPS[@]}"
    echo "  ✓ Dependencies installed"
fi

echo "  Activate with:  conda activate $CONDA_ENV_NAME"
echo ""

# ============================================
# Step 4: Download trained-model bundle from Google Drive
# ============================================
need_download=false
if $FORCE_CKPT; then
    need_download=true
else
    for w in "${REQUIRED_WEIGHTS[@]}"; do
        if [ ! -f "$w" ]; then
            need_download=true
            break
        fi
    done
fi

if $need_download; then
    echo "[4/5] Downloading iPLAN trained models from Google Drive..."

    # Step 3 already installed gdown into the env. If it's somehow not
    # there (env was created out-of-band without --force-env), reinstall.
    GDOWN_BIN="$CONDA_PREFIX_GUESS/bin/gdown"
    if [ ! -x "$GDOWN_BIN" ]; then
        echo "  gdown not found in env; installing ..."
        "$CONDA_ENV_PIP" install --quiet gdown
    fi

    TMP_DIR=$(mktemp -d /tmp/iplan_weights_XXXXXX)
    trap "rm -rf '$TMP_DIR'" EXIT

    echo "  Downloading folder $IPLAN_GDRIVE_FOLDER_ID into $TMP_DIR ..."
    "$GDOWN_BIN" --folder "$IPLAN_GDRIVE_FOLDER_ID" -O "$TMP_DIR"

    mkdir -p "$LOC_PRETRAINED_DIR" "$LOC_WEIGHTS_DIR" "$PART_WEIGHTS_DIR"

    place_weight() {
        local fname="$1"   # e.g. G_net_210.pth
        local dest="$2"    # absolute destination path
        local found
        found=$(find "$TMP_DIR" -type f -name "$fname" -print -quit || true)
        if [ -z "$found" ]; then
            echo "  ✗ $fname not found in downloaded bundle"
            echo "    (the Google Drive folder layout may have changed; download manually from"
            echo "     https://drive.google.com/drive/folders/$IPLAN_GDRIVE_FOLDER_ID )"
            return 1
        fi
        cp -f "$found" "$dest"
        echo "  ✓ $(basename "$dest")  ->  $dest"
    }

    place_weight "resnet18-5c106cde.pth"   "$LOC_PRETRAINED_DIR/resnet18-5c106cde.pth"
    place_weight "living_resnet18_300.pth" "$LOC_WEIGHTS_DIR/living_resnet18_300.pth"
    place_weight "living_fc1_300.pth"      "$LOC_WEIGHTS_DIR/living_fc1_300.pth"
    place_weight "location_resnet18_100.pth" "$LOC_WEIGHTS_DIR/location_resnet18_100.pth"
    place_weight "location_up1_100.pth"    "$LOC_WEIGHTS_DIR/location_up1_100.pth"
    place_weight "location_embed_100.pth"  "$LOC_WEIGHTS_DIR/location_embed_100.pth"
    place_weight "G_net_210.pth"           "$PART_WEIGHTS_DIR/G_net_210.pth"
    place_weight "renderer.pkl"            "$PART_WEIGHTS_DIR/renderer.pkl"

    rm -rf "$TMP_DIR"
    trap - EXIT
    echo "  ✓ Weights installed"
else
    echo "[4/5] iPLAN weights already present, skipping download"
    for w in "${REQUIRED_WEIGHTS[@]}"; do
        echo "  ✓ $w"
    done
    echo "  (re-run with --force-checkpoints to re-download)"
fi
echo ""

# ============================================
# Step 5: Build dataset_iplan/ from RPLAN PNGs
# ============================================
if $SKIP_DATASET; then
    echo "[5/5] --skip-dataset set; not (re)building $DATASET_IPLAN_DIR"
    echo ""
elif [ -d "$DATASET_IPLAN_DIR" ] && [ "$(ls -A "$DATASET_IPLAN_DIR" 2>/dev/null | wc -l)" -gt 0 ] && ! $FORCE_DATASET; then
    n_mat=$(find "$DATASET_IPLAN_DIR" -maxdepth 1 -name '*.mat' | wc -l)
    echo "[5/5] $DATASET_IPLAN_DIR already contains $n_mat .mat files"
    echo "  ✓ Skipping; re-run with --force-dataset to rebuild from PNGs"
    echo ""
else
    echo "[5/5] Building $DATASET_IPLAN_DIR from RPLAN PNGs..."

    if [ ! -d "$RPLAN_PNG_DIR" ]; then
        echo "  ✗ Source PNGs not found: $RPLAN_PNG_DIR"
        echo "    Run ./prepare_rplan_hypergraph.sh from $HF_ROOT first to"
        echo "    download the original RPLAN dataset."
        exit 1
    fi

    if [ ! -f "$CONVERT_SCRIPT" ]; then
        echo "  ✗ Conversion script not found: $CONVERT_SCRIPT"
        exit 1
    fi

    if $FORCE_DATASET && [ -d "$DATASET_IPLAN_DIR" ]; then
        echo "  --force-dataset: deleting existing $DATASET_IPLAN_DIR"
        rm -rf "$DATASET_IPLAN_DIR"
    fi

    # CONDA_ENV_PY was resolved in Step 3 (env setup).
    "$CONDA_ENV_PY" "$CONVERT_SCRIPT" \
        --split test \
        --iplan-root "$IPLAN_REPO" \
        --png-root "$RPLAN_PNG_DIR" \
        --out-dir "$DATASET_IPLAN_DIR"

    n_mat=$(find "$DATASET_IPLAN_DIR" -maxdepth 1 -name '*.mat' | wc -l)
    echo "  ✓ Wrote $n_mat .mat files to $DATASET_IPLAN_DIR"
    echo ""
fi

# ============================================
# Verify
# ============================================
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "Summary:"
echo "  iPLAN repo:            $IPLAN_REPO"
echo "  Inference patches:     $([ -f "$PATCH_MARKER" ] && echo "applied" || echo "NOT applied")"
all_weights_ok=true
for w in "${REQUIRED_WEIGHTS[@]}"; do
    if [ -f "$w" ]; then
        :
    else
        all_weights_ok=false
        echo "  ✗ Missing weight: $w"
    fi
done
$all_weights_ok && echo "  Weights:               all present (${#REQUIRED_WEIGHTS[@]} files)"
if [ -d "$DATASET_IPLAN_DIR" ]; then
    n_mat=$(find "$DATASET_IPLAN_DIR" -maxdepth 1 -name '*.mat' | wc -l)
    echo "  dataset_iplan/:        $n_mat .mat files"
else
    echo "  ⚠ dataset_iplan/ not built yet (run without --skip-dataset to build)"
fi
echo ""
echo "To run the OurII benchmark on the full RPLAN test split:"
echo "  conda activate $CONDA_ENV_NAME"
echo "  cd $HF_ROOT"
echo "  python -m benchmark.src.evaluate_iplan \\"
echo "      --iplan_data_dir $DATASET_IPLAN_DIR \\"
echo "      --iplan_root     $IPLAN_REPO \\"
echo "      --save_generations cache/iplan_ourII_test.pkl \\"
echo "      --out benchmark/eval_results_iplan_ourII"
echo ""
echo "To re-score from a saved cache (no GPU needed):"
echo "  python -m benchmark.src.evaluate_iplan \\"
echo "      --load_generations cache/iplan_ourII_test.pkl \\"
echo "      --out benchmark/eval_results_iplan_ourII"
echo ""
