# Benchmark: HouseGAN++, House-Diffusion & DiffPlanner

This benchmark evaluates three published floor-plan generators against
HypergraphFormer:

| Model | Dataset | Inputs |
|---|---|---|
| **HouseGAN++** | WMR24 | room set + adjacency graph |
| **House-Diffusion** | WMR24 | room set + adjacency graph |
| **DiffPlanner** | RPLAN | input boundary + room count + room categories (`b+nc` mode) |

It ships automated setup scripts, data conversion utilities, and a
shared evaluation pipeline. Available metrics are: GED, room accuracy
(combined / count-only / type-only), delta (geometric similarity), area
proportion error, and -- for DiffPlanner -- boundary gap ratio.

## Table of Contents

- [Quick Start: HouseGAN++](#quick-start-houseganpp)
- [Quick Start: House-Diffusion](#quick-start-house-diffusion)
- [Quick Start: DiffPlanner](#quick-start-diffplanner)
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

## Quick Start: DiffPlanner

DiffPlanner is evaluated on the **RPLAN** test split (12k samples) in
its `b+nc` cascade mode: the model receives the input boundary, the
desired room count, and the desired room categories, then runs three
sub-models in sequence (NodeDiff -> AdjacencyDiff -> PartitioningDiff)
followed by post-processing.

### 1. One-Time Dataset Prep

```bash
# from the repo root
./prepare_rplan_hypergraph.sh
```

This downloads the RPLAN floorplan PNGs and the official DiffPlanner
JSON dataset and lays them out at
`dataset/rplan/dataset_diffplanner/data_{train,val,test}.json`.

### 2. One-Command DiffPlanner Setup

```bash
cd benchmark
./setup_diffplanner.sh
```

This will:

- Clone the `DiffPlanner` repository alongside the project (skipped if
  already present)
- Download `trained_model.zip` (~665 MB) from the official release and
  place the three required checkpoints at the exact paths the
  evaluator's `--node_checkpoint` / `--adjacency_checkpoint` /
  `--partition_checkpoint` defaults expect
- Symlink the prepared `data_test.json` (and `data_train.json` if
  present) into `<DiffPlanner>/dataset/dataset_json/` so DiffPlanner's
  internal helpers (`RPlanDataset`, `post_processing.py`) load the same
  source-of-truth data

There is no pip-install step: DiffPlanner ships no `setup.py` /
`requirements.txt`, and the evaluator imports its sub-packages by
adding them to `sys.path` at runtime.

### 3. Evaluate

Run the full cascade once and cache the post-processed records (a few
hours on one GPU):

```bash
cd <repo root>
python benchmark/src/evaluate_diffplanner.py \
    --data_path dataset/rplan/dataset_diffplanner/data_test.json \
    --diffplanner_root ../DiffPlanner \
    --use_ddim \
    --save_generations cache/dp_bnc_test_full.pkl \
    --out benchmark/eval_results_diffplanner_bnc
```

Then re-score from the cache as many times as you like (no GPU
required):

```bash
python benchmark/src/evaluate_diffplanner.py \
    --load_generations cache/dp_bnc_test_full.pkl \
    --data_path dataset/rplan/dataset_diffplanner/data_test.json \
    --out benchmark/eval_results_diffplanner_bnc
```

Results are saved to
`benchmark/eval_results_diffplanner_bnc/results.npz` and
`results.txt`. To render side-by-side GT vs predicted floor plans
(plus bubble diagrams) with metrics overlaid, use the visualizer:

```bash
python -m benchmark.utils.visualize_diffplanner \
    --predicted cache/dp_bnc_test_full.pkl \
    --gt-json dataset/rplan/dataset_diffplanner/data_test.json \
    --metrics-npz benchmark/eval_results_diffplanner_bnc/results.npz \
    --out-dir benchmark/eval_results_diffplanner_bnc/vis \
    --limit 50
```

---

## Evaluation Metrics

All evaluation scripts delegate statistics computation, printing, and
file output to `utils/report_results.py`. Per-script metric coverage:

| Metric | HouseGAN++ | House-Diffusion | DiffPlanner |
|---|:---:|:---:|:---:|
| **GED** | yes | yes | -- |
| **Room Accuracy** (combined / count / type) | yes | yes | yes |
| **Delta (Room Similarity)** | yes | yes | yes |
| **Area Proportion Error** | yes | yes | yes |
| **Boundary Gap Ratio** | -- | -- | yes |

Definitions:

- **GED** -- Graph Edit Distance between predicted and ground-truth
  door-based access graphs. Computed with a 10 s timeout per pair;
  timed-out cases are recorded as NaN.
- **Room Accuracy** -- Three sub-scores from
  `utils.metrics.room_type_accuracy`: combined `accuracy` (matches
  count *and* types), `room_count_accuracy` (count-only), and
  `room_type_accuracy` (type multiset only). For DiffPlanner, "predicted
  rooms" means the rooms that survive post-processing (rooms whose
  `r_boundary_aligned` was dropped by `regularize_fp` or covered by a
  sibling in `get_room_boundary` are counted as lost). The lost-room
  rate is also reported as `Hidden rooms` in `results.txt`.
- **Delta (Room Similarity)** -- Geometric shape similarity between
  matched predicted and ground-truth room pairs, based on area and
  perimeter ratios.
- **Area Proportion Error** -- Mean absolute difference between
  room-area proportions of paired predicted and ground-truth rooms of
  the same type.
- **Boundary Gap Ratio** *(DiffPlanner only)* -- Fraction of the input
  boundary polygon NOT covered by the union of post-aligned predicted
  rooms. Quantifies the visible "gap" artifacts produced by the cascade
  + post-processing pipeline. 0.0 = perfect tiling, 1.0 = no rooms.

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

### `setup_diffplanner.sh`

Complete automated setup for DiffPlanner evaluation on the RPLAN test
split. Idempotent; pass `--force-checkpoints` to re-download the
`trained_model.zip`.

1. Clones `DiffPlanner` from `https://github.com/shidong-wang/DiffPlanner`
   alongside the project (skipped if already present)
2. Downloads `trained_model.zip` and lays out the three required
   checkpoints at the exact paths the evaluator's CLI defaults expect:
   - `<DP>/node_diff/scripts/trained_model/bnc_model300000.pt`
   - `<DP>/adjacency_diff/scripts/trained_model/bncsl_model300000.pt`
   - `<DP>/partitioning_diff/scripts/trained_model/bncsla_model300000.pt`
3. Symlinks the prepared `data_test.json` (and `data_train.json` if
   present) into `<DP>/dataset/dataset_json/`

The script assumes the dataset JSONs themselves were already produced
by `./prepare_rplan_hypergraph.sh` from the repo root.

### `eval_houseganpp.sh`

Evaluate a trained HouseGAN++ model.  Supports generation caching via
`--save_generations` / `--load_generations` for reproducible re-evaluation.

### `eval_housediffusion.sh`

Evaluate a trained House-Diffusion model.  Same caching support as above.

DiffPlanner has no `eval_*.sh` wrapper today; see the
[Quick Start: DiffPlanner](#quick-start-diffplanner) section for the
recommended invocation.

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

### `src/evaluate_diffplanner.py`

Evaluate DiffPlanner's `b+nc` cascade end-to-end on the RPLAN test
split. Computes delta, area proportion error, boundary gap ratio, and
the three room-accuracy sub-scores (combined / count / type).

- Loads all three sub-model checkpoints (NodeDiff, AdjacencyDiff,
  PartitioningDiff) and runs the cascade in-memory in batches
- Wraps DiffPlanner's `output/post_processing.py`; falls back to raw
  cascade boxes when alignment aborts (recorded as `raw_fallback_count`)
- Tracks rooms dropped by post-processing (`Empty box i!!!` /
  `empty intersection`) and reports the lost-room rate as `Hidden rooms`
- Generation caching via `--save_generations` / `--load_generations`
  (post-processed records are pickled, so re-scoring metrics is GPU-free)
- Supports `--use_ddim` for ~10x faster sampling than ancestral DDPM

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

### `utils/convert_hypergraph_to_diffplanner.py`

Convert hypergraph data to DiffPlanner JSON format. Maps the hypergraph
room taxonomy (`living`, `bed`, `kitchen`, `bath`, `foyer`, `extra`)
onto DiffPlanner's 6-class category id space (0..5) and emits per-sample
records with `rooms[*].{id, category, r_boundary, size, location, box,
order}`, `adjacencies`, and `access_adjacencies`.

### `utils/visualize_diffplanner.py`

Model-agnostic DiffPlanner visualizer. Reads any DiffPlanner-shaped
input (a `data_test.json`-style list, a single-record JSON, a benchmark
cache `.pkl`, or a directory of per-sample `<name>.json` files) and
writes a side-by-side panel per sample (GT floor plan, predicted floor
plan, GT bubble diagram, predicted bubble diagram), an aggregate
`summary.png`, and a multi-page `visualizations.pdf`. Pass
`--metrics-npz` to overlay per-sample numbers from a `results.npz`.
Color palette and rendering primitives mirror
`DiffPlanner/output/visualization.py` exactly. Run as
`python -m benchmark.utils.visualize_diffplanner --help` for the full
flag list.

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
├── setup_diffplanner.sh                 # DiffPlanner automated setup
├── eval_houseganpp.sh                   # HouseGAN++ evaluation launcher
├── eval_housediffusion.sh               # House-Diffusion evaluation launcher
│
├── src/                                 # Core Python scripts
│   ├── dataset_loader.py                # Shared PyTorch dataset module
│   ├── evaluate_houseganpp.py           # HouseGAN++ evaluation (WMR24)
│   ├── evaluate_housediffusion.py       # House-Diffusion evaluation (WMR24)
│   └── evaluate_diffplanner.py          # DiffPlanner b+nc cascade evaluation (RPLAN)
│
├── utils/                               # Utility and conversion scripts
│   ├── convert_hypergraph_to_housegan_json.py  # WMR24 JSON → HouseGAN++ JSON
│   ├── convert_hypergraph_to_diffplanner.py    # WMR24 JSON → DiffPlanner JSON
│   ├── convert_json_to_pickle.py               # HouseGAN++ JSON → pickle
│   ├── convert_json_to_npz.py                  # HouseGAN++ JSON → NPZ
│   ├── visualize_rplan_json.py                 # RPLAN JSON visualization
│   └── visualize_diffplanner.py                # DiffPlanner panel renderer (GT/pred)
│
└── eval_results_*/                      # Output directories (per experiment)
    ├── results.npz                          # All metric arrays + summary dicts
    ├── results.txt                          # Human-readable summary
    ├── fig/                                 # Comparison images (if --save_graphs)
    └── generations*.pkl                     # Cached generations (if --save_generations)
```

---
