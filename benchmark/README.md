# Benchmark: HouseGAN++ & House-Diffusion on the WMR24 Dataset

This benchmark evaluates **HouseGAN++** and **House-Diffusion** on the WMR24
dataset.  It includes automated setup scripts, data conversion utilities, and
evaluation pipelines with four metrics: GED, room accuracy, delta (geometric
similarity), and area proportion error.

## Table of Contents

- [Quick Start: HouseGAN++](#quick-start-houseganpp)
- [Quick Start: House-Diffusion](#quick-start-house-diffusion)
- [Evaluation Metrics](#evaluation-metrics)
- [Shell Scripts](#shell-scripts)
- [Core Scripts](#core-scripts)
- [Utility Scripts](#utility-scripts)
- [Dataset Format](#dataset-format)
- [Project Structure](#project-structure)

---

## Quick Start: HouseGAN++

### 1. One-Command Setup

```bash
cd benchmark
./setup_houseganpp.sh
```

This will:
- Clone the HouseGAN++ repository (if needed)
- Create a Python virtual environment
- Install all dependencies
- Convert the WMR24 dataset to HouseGAN++ format (JSON → pickle)

### 2. Evaluate

```bash
cd benchmark
./eval_houseganpp.sh
```

Or run directly:
```bash
cd benchmark
source ../.venv/bin/activate

python src/evaluate_houseganpp.py \
    --data_path /path/to/test_data.pkl \
    --checkpoint ../houseganpp/checkpoints/pretrained.pth \
    --out eval_results_houseganpp \
    --num_variations 5 \
    --save_graphs
```

Results are saved to `eval_results_houseganpp/results.npz` and
`results.txt`.  Comparison images go into `eval_results_houseganpp/fig/`.

---

## Quick Start: House-Diffusion

### 1. One-Command Setup

```bash
cd benchmark
./setup_housediffusion.sh
```

This will:
- Verify the `house_diffusion` repository exists
- Create/reuse the shared virtual environment
- Install House-Diffusion dependencies and package
- Reuse HouseGAN++ JSON data (or convert fresh) and build NPZ files

### 2. Evaluate

```bash
cd benchmark
./eval_housediffusion.sh
```

Results are saved to `eval_results_housediffusion/results.npz`.

---

## Evaluation Metrics

Both evaluation scripts compute four metrics and delegate all statistics
computation, printing, and file output to `utils/report_results.py`:

| Metric | Description |
|---|---|
| **GED** | Graph Edit Distance between predicted and ground-truth door-based access graphs. Computed with a 10 s timeout per pair; timed-out cases are recorded as NaN. |
| **Room Count Accuracy** | Exact match rate, mean count error, and mean type error between predicted and expected room sets. |
| **Delta (Room Similarity)** | Geometric shape similarity between matched predicted and ground-truth room pairs, based on area and perimeter ratios. |
| **Area Proportion Error** | Mean absolute difference between room-area proportions of paired predicted and ground-truth rooms of the same type. |

All metrics are saved to `results.npz` and a human-readable
`results.txt` summary.

### Inspecting results

```bash
# Bin metrics by room count (auto-detects all available metrics in the NPZ)
python utils/analyze_metrics_by_room_count.py \
    --results_npz eval_results_houseganpp/results.npz

# With display title + JSON export
python utils/analyze_metrics_by_room_count.py \
    --results_npz eval_results_housediffusion/results.npz \
    --title "House-Diffusion" \
    --output_json binned_stats.json
```

---

## Shell Scripts

### `setup_houseganpp.sh`

Complete automated setup for HouseGAN++ evaluation on the WMR24 dataset.

1. Clones HouseGAN++ from `https://github.com/hsalehipour/houseganpp`
2. Creates a virtual environment with all Python dependencies
3. Converts WMR24 hypergraph JSON → HouseGAN++ JSON → pickle format

### `setup_housediffusion.sh`

Complete automated setup for House-Diffusion evaluation on the WMR24 dataset.

1. Verifies the `house_diffusion` repo exists alongside the project
2. Creates/reuses the shared virtual environment
3. Installs House-Diffusion as a package
4. Converts WMR24 data to HouseGAN++ JSON (reuses existing if available)
5. Converts JSON data to NPZ format expected by House-Diffusion

### `eval_houseganpp.sh`

Evaluate a trained HouseGAN++ model.  Supports generation caching via
`--save_generations` / `--load_generations` for reproducible re-evaluation.

### `eval_housediffusion.sh`

Evaluate a trained House-Diffusion model.  Same caching support as above.

---

## Core Scripts

### `src/dataset_loader.py`

Shared PyTorch dataset and collation functions for loading WMR24 data in
HouseGAN++ format.

- **`BenchmarkDataset`** — Loads pickle files, builds room masks from
  polygons, one-hot encodes room types (RPLAN format), creates adjacency edge
  lists. Supports room-count filtering via `target_set`.
- **`collate_fn`** — Batches samples with variable numbers of rooms,
  concatenating masks, nodes, and edges with proper index adjustment.

### `src/evaluate_houseganpp.py`

Evaluate HouseGAN++ with four metrics (GED, room accuracy, delta, area
proportion error).

- Incremental generation — generates floorplans room-by-room using
  HouseGAN++'s iterative refinement
- Builds door-based access graphs from generated masks (IoU-based door
  detection) and from ground-truth structure (graph triples)
- Multiple variations per sample for consistency measurement
- `--save_graphs` produces side-by-side GT vs predicted comparison images
- Generation caching via `--save_generations` / `--load_generations`

### `src/evaluate_housediffusion.py`

Evaluate House-Diffusion with the same four metrics.

- Generates floorplans using diffusion sampling
- Extracts room polygons from generated outputs
- Builds predicted access graphs from polygon adjacency
- Generation caching for reproducible re-evaluation

---

## Utility Scripts

### `utils/convert_hypergraph_to_housegan_json.py`

Convert WMR24 hypergraph JSON format to HouseGAN++-compatible JSON files.
Extracts room polygons, door positions, and adjacency triples from the
BSP-tree representation using the RGL geometry library.

### `utils/convert_json_to_pickle.py`

Convert HouseGAN++ JSON files to pickle format used by `BenchmarkDataset`.
Normalizes polygons to a target grid (64x64), maps room types to RPLAN IDs,
and builds adjacency triples.

### `utils/convert_json_to_npz.py`

Convert HouseGAN++ JSON files to NPZ format expected by House-Diffusion's
`RPlanhgDataset`.  Remaps room types and structures the data as `houses`,
`graphs`, `door_masks`, `self_masks`, `gen_masks`, and `ids` arrays.

**Note:** Reporting and per-room-count analysis have been moved to the shared
`utils/` package:

- **`utils/report_results.py`** -- unified NPZ + TXT reporting used by both
  benchmark eval scripts and `scripts/score_results.py`.
- **`utils/analyze_metrics_by_room_count.py`** -- bins NPZ metrics by room
  count (read from ``room_counts`` in the NPZ) and produces per-bin statistics.

---

## Dataset Format

### Input: WMR24 Hypergraph Format

Each entry in `dataset/wmr24/wmr24.json`:

```json
{
    "database": "zurich",
    "id": "ch-z-0020",
    "area": 32.73,
    "bedrooms": 1,
    "bathrooms": 1,
    "bounds": { "corners": [...], ... },
    "facade": [...],
    "circulation": [...],
    "split": { "name": "root", "children": [...], ... }
}
```

### Intermediate: HouseGAN++ JSON (per sample)

Produced by `convert_hypergraph_to_housegan_json.py`:

```json
{
    "id": "ch-z-0020",
    "room_type": [1, 6, 2, 4, 17, 17, 17],
    "boxes": [...],
    "edges": [[0, 1, 1], [0, 1, 2], [1, -1, 3], ...]
}
```

### HouseGAN++ Pickle Format

Produced by `convert_json_to_pickle.py`:

```python
{
    'id': 'ch-z-0020',
    'polygons': [[(x1, y1), (x2, y2), ...], ...],
    'room_types': [1, 6, 2, 4, 17, 17, 17],
    'triples': [[0, 1, 1], [0, -1, 2], ...]
}
```

### House-Diffusion NPZ Format

Produced by `convert_json_to_npz.py`:

| Key | Shape | Description |
|---|---|---|
| `houses` | `(N, 100, 94)` | Packed room geometry (corners + one-hot types) |
| `graphs` | `(N,)` object | Adjacency triples per sample |
| `door_masks` | `(N, 100, 100)` | Door adjacency matrices |
| `self_masks` | `(N, 100, 100)` | Self-connection masks |
| `gen_masks` | `(N, 100, 100)` | Generation masks |
| `ids` | `(N,)` object | Sample ID strings |

### Room Type Mapping

| WMR24 mergeid | RPLAN ID | Room Type | HouseGAN++ Color |
|---|---|---|---|
| living | 1 | Living Room | #ED6D87 |
| bed | 3 | Bedroom | #5B448C |
| kitchen | 2 | Kitchen | #F0E34F |
| bath | 4 | Bathroom | #5D9AD4 |
| foyer | 6 | Entrance | #FFA17A |
| extra | 10 | Storage | #B0B0B0 |
| — | 17 | Interior Door | #888888 |

---

## Project Structure

```
benchmark/
├── README.md                            # This file
├── setup_houseganpp.sh                  # HouseGAN++ automated setup
├── setup_housediffusion.sh              # House-Diffusion automated setup
├── eval_houseganpp.sh                   # HouseGAN++ evaluation launcher
├── eval_housediffusion.sh               # House-Diffusion evaluation launcher
│
├── src/                                 # Core Python scripts
│   ├── dataset_loader.py                # Shared PyTorch dataset module
│   ├── evaluate_houseganpp.py           # HouseGAN++ evaluation (4 metrics)
│   └── evaluate_housediffusion.py       # House-Diffusion evaluation (4 metrics)
│
├── utils/                               # Utility and conversion scripts
│   ├── convert_hypergraph_to_housegan_json.py  # WMR24 JSON → HouseGAN++ JSON
│   ├── convert_json_to_pickle.py               # HouseGAN++ JSON → pickle
│   ├── convert_json_to_npz.py                  # HouseGAN++ JSON → NPZ
│   └── visualize_rplan_json.py                 # RPLAN JSON visualization
│
└── eval_results_*/                      # Output directories (per experiment)
    ├── results.npz                          # All metric arrays + summary dicts
    ├── results.txt                          # Human-readable summary
    ├── fig/                                 # Comparison images (if --save_graphs)
    └── generations*.pkl                     # Cached generations (if --save_generations)
```

---
