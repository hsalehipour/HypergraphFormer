# Benchmark: HouseGAN++ & House-Diffusion on the Hypergraph1K Dataset

This benchmark provides a complete workflow for training and evaluating both **HouseGAN++** and **House-Diffusion** on the Hypergraph1K dataset. The implementation includes automated setup scripts, data conversion utilities, evaluation pipelines with three metrics (GED, room accuracy, delta), and visualization tools.

## 📋 Table of Contents

- [Quick Start: HouseGAN++](#quick-start-houseganpp)
- [Quick Start: House-Diffusion](#quick-start-house-diffusion)
- [Shell Scripts](#shell-scripts)
- [Core Scripts](#core-scripts)
- [Utility Scripts](#utility-scripts)
- [Dataset Format](#dataset-format)
- [Project Structure](#project-structure)

---

## 🚀 Quick Start: HouseGAN++

### 1. One-Command Setup

```bash
cd benchmark
./setup_houseganpp.sh
```

This will:
- Clone the HouseGAN++ repository (if needed)
- Create a Python virtual environment
- Install all dependencies
- Convert the Hypergraph1K dataset to HouseGAN++ format

### 2. Train the Model

```bash
cd benchmark
./finetune_houseganpp.sh
```

Monitor training with TensorBoard:
```bash
tensorboard --logdir=../houseganpp/runs/ft_houseganpp_v1
```

### 3. Evaluate the Model

```bash
cd benchmark
./eval_houseganpp.sh
```

Or run directly:
```bash
cd benchmark
source ../.venv/bin/activate

python src/evaluate_houseganpp.py \
    --data_path dataset/Hypergraph_to_RPLAN/test/test_data.pkl \
    --checkpoint ../houseganpp/checkpoints/ft_housegan_final.pth \
    --out test_results \
    --num_samples 20 \
    --save_graphs
```

---

## 🚀 Quick Start: House-Diffusion

### 1. One-Command Setup

```bash
cd benchmark
./setup_housediffusion.sh
```

This will:
- Verify the `house_diffusion` repository exists
- Create/reuse the shared virtual environment
- Install House-Diffusion dependencies and package
- Reuse Hypergraph-to-RPLAN data (from HouseGAN++ setup) or convert fresh
- Convert pickle data to NPZ format for House-Diffusion

### 2. Train the Model

```bash
cd benchmark
./finetune_housediffusion.sh
```

### 3. Evaluate the Model

```bash
cd benchmark
./eval_housediffusion.sh
```

---

## 🔧 Shell Scripts

### `setup_houseganpp.sh`

**Purpose:** Complete automated setup for HouseGAN++ training on the Hypergraph1K dataset.

**What it does:**

1. **Clones HouseGAN++ Repository** from `https://github.com/hsalehipour/houseganpp`
2. **Creates Virtual Environment** with all Python dependencies
3. **Converts Hypergraph1K Dataset** to HouseGAN++ pickle format (64×64 normalized polygons)

**Output:**
```
benchmark/dataset/Hypergraph_to_RPLAN/
├── train/train_data.pkl
├── validation/validation_data.pkl
└── test/test_data.pkl
```

---

### `setup_housediffusion.sh`

**Purpose:** Complete automated setup for House-Diffusion training on the Hypergraph1K dataset.

**What it does:**

1. **Verifies** the `house_diffusion` repo exists alongside the project
2. **Creates/reuses** the shared virtual environment
3. **Installs** House-Diffusion as a package
4. **Converts** Hypergraph1K data to RPLAN pickle format (reuses existing if available)
5. **Converts** pickle files to NPZ format expected by House-Diffusion's `RPlanhgDataset`

---

### `finetune_houseganpp.sh`

**Purpose:** Launch fine-tuning of HouseGAN++ on the Hypergraph1K dataset.

**Configuration:**
```bash
TRAIN_DATA="dataset/Hypergraph_to_RPLAN/train/train_data.pkl"
VAL_DATA="dataset/Hypergraph_to_RPLAN/validation/validation_data.pkl"
CHECKPOINT="houseganpp/checkpoints/pretrained.pth"
EXP_FOLDER="ft_houseganpp_v1"
N_EPOCHS=1000
BATCH_SIZE=4
```

**Outputs:**
- **Checkpoints:** `houseganpp/checkpoints/ft_houseganpp_v1_{step}.pth`
- **Visualizations:** `houseganpp/exps/ft_houseganpp_v1/`
- **TensorBoard Logs:** `houseganpp/runs/ft_houseganpp_v1/`
- **Final Model:** `houseganpp/checkpoints/ft_houseganpp_final.pth`

---

### `finetune_housediffusion.sh`

**Purpose:** Launch fine-tuning of House-Diffusion on the Hypergraph1K dataset using converted NPZ data.

---

### `eval_houseganpp.sh`

**Purpose:** Evaluate a trained HouseGAN++ model on the Hypergraph1K test dataset.

**Configuration:**
```bash
TEST_DATA="dataset/Hypergraph_to_RPLAN/test/test_data.pkl"
CHECKPOINT="../houseganpp/checkpoints/ft_housegan_final.pth"
OUTPUT_DIR="test_results"
MAX_SAMPLES=20
```

**Outputs:**
- **Comparison plots:** `test_results/comparison_{i:04d}.png`
- **Statistics JSON:** `test_results/ged_statistics.json` (GED, room accuracy, delta metrics)

---

### `eval_housediffusion.sh`

**Purpose:** Evaluate a trained House-Diffusion model on the Hypergraph1K test dataset.

Generates floorplans and computes the same three metrics (GED, room accuracy, delta) as the HouseGAN++ evaluation.

---

## 📝 Core Scripts

### `src/dataset_loader.py`

**Purpose:** Shared PyTorch dataset and collation functions for loading Hypergraph1K data in HouseGAN++ format.

**Key Components:**

1. **`Hypergraph1kDataset` Class** — Loads pickle files, builds room masks from polygons, one-hot encodes room types (RPLAN format), creates adjacency edge lists. Supports train/val/test splits with room-count filtering.

2. **`collate_fn` Function** — Batches samples with variable numbers of rooms. Concatenates masks, nodes, and edges with proper index adjustment.

**Usage:**
```python
from dataset_loader import Hypergraph1kDataset, collate_fn

dataset = Hypergraph1kDataset('train_data.pkl', target_set=None, split='train')
loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
```

---

### `src/finetune_houseganpp.py`

**Purpose:** Fine-tune HouseGAN++ generator and discriminator on the Hypergraph1K dataset.

**Key Features:**
- WGAN-GP training (Wasserstein GAN with gradient penalty)
- Uses shared `Hypergraph1kDataset` from `dataset_loader.py`
- TensorBoard logging (losses, images)
- Periodic validation visualization and checkpoint saving

---

### `src/finetune_housediffusion.py`

**Purpose:** Fine-tune House-Diffusion on the Hypergraph1K dataset using pre-converted NPZ data.

**Key Features:**
- Loads data from `processed_rplan/rplan_{set_name}_{target_set}.npz`
- Resumes from a pretrained House-Diffusion checkpoint
- Configurable model architecture (base channels, attention resolutions)

---

### `src/evaluate_houseganpp.py`

**Purpose:** Evaluate HouseGAN++ on the Hypergraph1K test dataset with three metrics.

**Key Features:**

1. **Incremental Generation** — Generates floorplans room-by-room using HouseGAN++'s iterative refinement. Supports multiple variations per sample.

2. **Three Metrics:**
   - **GED**: Graph Edit Distance between predicted and GT access graphs
   - **Room Accuracy**: Room count and type distribution matching
   - **Delta**: Geometric shape similarity between matched room pairs

3. **Generation Caching** — `--save_generations` / `--load_generations` for reproducible re-evaluation without re-running inference.

4. **Visualization** — `--save_graphs` produces side-by-side GT vs predicted comparison images.

---

### `src/evaluate_housediffusion.py`

**Purpose:** Evaluate House-Diffusion on the Hypergraph1K test dataset with the same three metrics (GED, room accuracy, delta).

**Key Features:**
- Generates floorplans using House-Diffusion's diffusion sampling process
- Extracts room polygons from generated outputs
- Builds predicted access graphs from spatial adjacency
- Computes GED, room accuracy, and delta against ground truth
- Supports generation caching (`--save_generations` / `--load_generations`)

---

## 🔍 Utility Scripts

### `utils/convert_hypergraph_to_rplan.py`

**Purpose:** Convert Hypergraph1K JSON format to HouseGAN++ pickle format.

**Key Functions:**
- `extract_polygons_and_room_types` — Extracts room boundaries, maps Hypergraph1K room types to RPLAN IDs
- `extract_doors` — Extracts door polygons, computes positions and orientations
- `normalize_polygons` — Centers and scales polygons to target grid (64×64)
- `build_adjacency_triples` — Constructs adjacency matrix with adjacent (1) and non-adjacent (-1) relationships

**Usage:**
```bash
python utils/convert_hypergraph_to_rplan.py \
    --input /path/to/dataset/hypergraph \
    --output /path/to/Hypergraph_to_RPLAN \
    --splits train validation test \
    --target-size 64 \
    --verbose
```

---

### `utils/convert_pickle_to_npz.py`

**Purpose:** Convert HouseGAN++ pickle files to House-Diffusion's NPZ format.

Remaps room types to match House-Diffusion's encoding (e.g., front_door type 15 → 11) and structures the data as expected by House-Diffusion's `RPlanhgDataset`.

---

### `utils/ged_statistics.py`

**Purpose:** Shared utilities for computing and saving GED statistics consistently across evaluation scripts.

Provides `compute_and_save_ged_statistics` which post-processes raw GED arrays (handling NaN/timeout values, computing mean/std/median), and saves results to both NPZ and JSON formats.

---

### `utils/analyze_metrics_by_room_count.py`

**Purpose:** Post-evaluation analysis that bins GED and room accuracy metrics by room count.

Takes cached generation pickle files and result NPZ files from `evaluate_houseganpp.py` or `evaluate_housediffusion.py`, groups samples into bins (`<=4`, `5`, `6`, `7`, `8`, `>8` rooms), and computes per-bin statistics (mean, std, median, min, max).

**Usage:**
```bash
python utils/analyze_metrics_by_room_count.py \
    --model houseganpp \
    --cache_pickle cached_gen/cache.pkl \
    --results_npz ged_results/ged_results.npz \
    --num_variations 5 \
    --output_json binned_stats.json
```

---

### `utils/visualize_hypergraph_pickle.py`

**Purpose:** Visualize Hypergraph1K dataset samples from pickle files.

Renders floor plans with colored room polygons, doors, and room boundaries. Provides `draw_floorplan_from_pickle` for programmatic use and a CLI for quick inspection.

---

### `utils/visualize_rplan_json.py`

**Purpose:** Visualize original RPLAN JSON format samples.

Reconstructs polygons from edge segments and renders color-coded floor plans.

---

## 📊 Dataset Format

### Input: Hypergraph1K Format
```json
{
    "id": "sample_001",
    "bounds": [...],
    "split": [...],
    "room_types": [0, 1, 2, ...]
}
```

### Output: HouseGAN++ Pickle Format
```python
{
    'id': 'sample_001',
    'polygons': [
        [(x1, y1), (x2, y2), ...],
        [(x1, y1), (x2, y2), ...],
        ...
    ],
    'room_types': [1, 2, 3, ...],
    'triples': [
        [0, 1, 1],
        [0, -1, 2],
        ...
    ]
}
```

### Room Type Mapping

| Hypergraph1K Type | RPLAN ID | Room Type      | Color Code |
|-------------------|----------|----------------|------------|
| 0                 | 1        | Living Room    | #EE4D4D    |
| 1                 | 3        | Bedroom        | #C67C7B    |
| 2                 | 2        | Kitchen        | #FFD274    |
| 3                 | 4        | Bathroom       | #BEBEBE    |
| 10                | 6        | Foyer/Entrance | #E397D9    |
| 11                | 10       | Extra/Storage  | #9D9D9D    |
| 17                | 17       | Interior Door  | #5B5B5B    |

---

## 📁 Project Structure

```
benchmark/
├── README.md                            # This file
├── setup_houseganpp.sh                  # HouseGAN++ automated setup
├── setup_housediffusion.sh              # House-Diffusion automated setup
├── finetune_houseganpp.sh               # HouseGAN++ training launcher
├── finetune_housediffusion.sh           # House-Diffusion training launcher
├── eval_houseganpp.sh                   # HouseGAN++ evaluation launcher
├── eval_housediffusion.sh               # House-Diffusion evaluation launcher
│
├── src/                                 # Core Python scripts
│   ├── dataset_loader.py                # Shared PyTorch dataset module
│   ├── evaluate_houseganpp.py           # HouseGAN++ evaluation (GED, accuracy, delta)
│   ├── evaluate_housediffusion.py       # House-Diffusion evaluation (GED, accuracy, delta)
│   ├── finetune_houseganpp.py           # HouseGAN++ fine-tuning
│   └── finetune_housediffusion.py       # House-Diffusion fine-tuning
│
├── utils/                               # Utility and conversion scripts
│   ├── analyze_metrics_by_room_count.py # Post-evaluation metric binning by room count
│   ├── convert_hypergraph_to_rplan.py   # Hypergraph1K JSON → HouseGAN++ pickle
│   ├── convert_pickle_to_npz.py         # HouseGAN++ pickle → House-Diffusion NPZ
│   ├── ged_statistics.py                # Shared GED statistics computation
│   ├── visualize_hypergraph_pickle.py   # Pickle visualization
│   └── visualize_rplan_json.py          # RPLAN JSON visualization
│
└── dataset/                             # Generated datasets
    └── Hypergraph_to_RPLAN/
        ├── train/train_data.pkl
        ├── validation/validation_data.pkl
        └── test/test_data.pkl
```

---
