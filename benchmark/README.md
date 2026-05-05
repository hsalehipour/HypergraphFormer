# Benchmark: HouseGAN++, House-Diffusion, DiffPlanner & iPLAN

This benchmark evaluates four published floor-plan generators against
HypergraphFormer:

| Model | Dataset | Inputs |
|---|---|---|
| **HouseGAN++** | WMR24 | room set + adjacency graph |
| **House-Diffusion** | WMR24 | room set + adjacency graph |
| **DiffPlanner** | RPLAN | input boundary + room count + room categories (`b+nc` mode) |
| **iPLAN** (CVPR 2022) | RPLAN | input boundary + room count + room categories (`OurII` mode; Stage 1 skipped) |

It ships automated setup scripts, data conversion utilities, and a
shared evaluation pipeline. Available metrics are: GED, room accuracy
(combined / count-only / type-only), delta (geometric similarity), area
proportion error, plus boundary-aware coverage / outside-spill /
inter-room overlap ratios for the boundary-conditioned models.

## Table of Contents

- [Quick Start: HouseGAN++](#quick-start-houseganpp)
- [Quick Start: House-Diffusion](#quick-start-house-diffusion)
- [Quick Start: DiffPlanner](#quick-start-diffplanner)
- [Quick Start: iPLAN](#quick-start-iplan)
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

## Quick Start: iPLAN

iPLAN is evaluated on the **RPLAN** test split (12,110 samples) in its
`OurII` mode: the user supplies the boundary AND the per-floorplan
ground-truth room types/count, and iPLAN's three-stage cascade fills
in the rest:

- Stage 1 (BCVAE for room types) -- **skipped** (`rTypes := gt_rTypes`)
- Stage 2 (room locations, ResNet18 + ASPP) -- runs
- Stage 3 (room partitions, step-wise GAN) -- runs (with iterative refinement)

iPLAN ships its own pinned Python 3.8 / `torch==1.10.1+cu113` stack
that is incompatible with the shared `.venv` used by HouseGAN++ /
House-Diffusion / DiffPlanner, so it lives in a dedicated `iplan` conda
env. The setup script bootstraps it on first run.

### 1. One-Time Dataset Prep

If you have not already, download the RPLAN floorplan PNGs:

```bash
# from the repo root
./prepare_rplan_hypergraph.sh
```

This places the PNGs at
`dataset/rplan/dataset_original/floorplan_dataset/*.png`. iPLAN's
`.mat` files are derived from these in step 2.

### 2. One-Command iPLAN Setup

```bash
cd benchmark
./setup_iplan.sh
```

This will (idempotently):

- Clone the iPLAN repository
  (`https://github.com/hsalehipour/iPLAN-Interactive-and-Procedural-Layout-Planning.git`)
  alongside the project
- Apply 3 small inference-time patches to upstream iPLAN so its modules
  can be imported and run from outside the iPLAN repo (these fix
  pre-existing import-shadowing and cwd-relative path bugs in the
  upstream code; not algorithm changes)
- Create the `iplan` conda env if missing and `pip install` a
  version-pinned dependency list (Python 3.8 + the cu113 torch wheel
  from PyTorch's own index)
- Download the iPLAN trained-model bundle from the official Google
  Drive folder using `gdown` and lay each `.pth` / `.pkl` at the exact
  path the evaluator's `_load_iplan_models()` expects
- Build `dataset/rplan/dataset_iplan/` by running
  `benchmark/utils/convert_png_to_iplan_mat.py` on RPLAN's test split
  (~12,110 `.mat` files; ~4 minutes on 64 cores)

Useful flags: `--force-checkpoints`, `--force-dataset`,
`--force-patches`, `--force-env`, `--skip-dataset`.

### 3. Evaluate

Run the full cascade once and cache the post-processed records:

```bash
conda activate iplan
cd <repo root>
python -m benchmark.src.evaluate_iplan \
    --iplan_data_dir dataset/rplan/dataset_iplan \
    --iplan_root     ../iPlan \
    --save_generations cache/iplan_ourII_test.pkl \
    --out benchmark/eval_results_iplan_ourII
```

Re-score from the cache as many times as you like (no GPU required):

```bash
python -m benchmark.src.evaluate_iplan \
    --load_generations cache/iplan_ourII_test.pkl \
    --out benchmark/eval_results_iplan_ourII
```

Both inputs and ground truth are read from the same iPLAN `.mat` file
(`Boundary`, `gt_rTypes`, `gt_rBoxes` for GT; `rTypes` is seeded from
`gt_rTypes` and Stage 2/3 fill in `rCenters` / `rBoxes`). This
sidesteps any cross-format coordinate-frame or category-vocabulary
mapping and keeps the evaluation strictly inside iPLAN's native 128px
13-class space. There is no `eval_iplan.sh` wrapper today; the two
commands above are the recommended invocations.

---

## Evaluation Metrics

All evaluation scripts delegate statistics computation, printing, and
file output to `utils/report_results.py`. Per-script metric coverage:

| Metric | HouseGAN++ | House-Diffusion | DiffPlanner | iPLAN |
|---|:---:|:---:|:---:|:---:|
| **GED** | yes | yes | -- | -- |
| **Room Accuracy** (combined / count / type) | yes | yes | yes | yes |
| **Delta (Room Similarity)** | yes | yes | yes | yes |
| **Area Proportion Error** | yes | yes | yes | yes |
| **Spatial Overlap Ratio** (no boundary) | yes | yes | -- | -- |
| **Boundary geometry triplet** (gap / outside / overlap) | -- | -- | yes | yes |

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
  rate is also reported as `Hidden rooms` in `results.txt`. For iPLAN,
  predicted rooms are read from the rasterized 128px Stage-3 layout (one
  bbox per connected component, including the Living room which iPLAN
  recovers as the leftover interior).
- **Delta (Room Similarity)** -- Geometric shape similarity between
  matched predicted and ground-truth room pairs, based on area and
  perimeter ratios.
- **Area Proportion Error** -- Mean absolute difference between
  room-area proportions of paired predicted and ground-truth rooms of
  the same type.
- **Spatial Overlap Ratio** *(HouseGAN++ / House-Diffusion)* --
  `1 - area(union(pred)) / sum(area(pred_i))`. Captures inter-room
  overlap only; not normalized by an input boundary because the
  WMR24-conditioned models are not given one.
- **Boundary geometry triplet** *(DiffPlanner / iPLAN)* -- Three
  complementary ratios, all normalized by the input boundary area F so
  values are directly comparable across samples and methods. Let
  `U = union of predicted room polygons` and
  `S = sum of their individual areas`:

  | Metric | Formula | Ideal | Failure mode it catches |
  |---|---|---|---|
  | **`gap_ratio`** | `area(F − U) / area(F)` | 0 | predicted rooms leave the boundary uncovered |
  | **`outside_ratio`** | `area(U − F) / area(F)` | 0 | predicted rooms spill outside the boundary |
  | **`overlap_ratio`** | `(S − area(U)) / area(F)` | 0 | rooms double-count area by overlapping each other |

  See `compute_boundary_geometry_metrics()` in
  `evaluate_diffplanner.py` and `evaluate_iplan.py` for the exact
  Shapely implementation.

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

### `setup_iplan.sh`

Complete automated setup for iPLAN evaluation on the RPLAN test split
in `OurII` mode. Five idempotent steps; pass `--force-checkpoints`,
`--force-dataset`, `--force-patches`, `--force-env`, or
`--skip-dataset` to override individual stages.

1. Clones the iPLAN repo (`hsalehipour/iPLAN-Interactive-and-Procedural-Layout-Planning`)
   alongside the project (skipped if already present)
2. Applies 3 small inference-time patches to upstream iPLAN
   (`room_partition/models/__init__.py`, `loss_layer.py`, `net.py`)
   that fix pre-existing import-shadowing and cwd-relative path bugs.
   Marker file `<iPlan>/.hf_inference_patches_applied` makes re-runs a
   no-op.
3. Creates the `iplan` conda env with Python 3.8 and `pip install`s a
   version-pinned dependency list (including the cu113 torch wheel
   from PyTorch's own index). Existing envs are left alone unless
   `--force-env` is passed.
4. Downloads the iPLAN trained-model bundle from the official Google
   Drive folder using `gdown` and lays out 8 weight files
   (`<iPlan>/room_location/{pretrained_model,weights}/*.pth` and
   `<iPlan>/room_partition/weights/{G_net_210.pth,renderer.pkl}`).
5. Builds `dataset/rplan/dataset_iplan/` from
   `dataset_original/floorplan_dataset/*.png` via
   `benchmark/utils/convert_png_to_iplan_mat.py` (~12,110 `.mat` files).

The script assumes `./prepare_rplan_hypergraph.sh` has already
populated the original RPLAN PNGs in `dataset_original/`.

### `eval_houseganpp.sh`

Evaluate a trained HouseGAN++ model.  Supports generation caching via
`--save_generations` / `--load_generations` for reproducible re-evaluation.

### `eval_housediffusion.sh`

Evaluate a trained House-Diffusion model.  Same caching support as above.

DiffPlanner and iPLAN have no `eval_*.sh` wrappers today; see the
[Quick Start: DiffPlanner](#quick-start-diffplanner) and
[Quick Start: iPLAN](#quick-start-iplan) sections for the recommended
invocations.

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
split. Computes delta, area proportion error, the boundary geometry
triplet (gap / outside / overlap ratios), and the three
room-accuracy sub-scores (combined / count / type).

- Loads all three sub-model checkpoints (NodeDiff, AdjacencyDiff,
  PartitioningDiff) and runs the cascade in-memory in batches
- Wraps DiffPlanner's `output/post_processing.py`; falls back to raw
  cascade boxes when alignment aborts (recorded as `raw_fallback_count`)
- Tracks rooms dropped by post-processing (`Empty box i!!!` /
  `empty intersection`) and reports the lost-room rate as `Hidden rooms`
- Generation caching via `--save_generations` / `--load_generations`
  (post-processed records are pickled, so re-scoring metrics is GPU-free)
- Supports `--use_ddim` for ~10x faster sampling than ancestral DDPM

### `src/evaluate_iplan.py`

Evaluate iPLAN's `OurII` cascade on the RPLAN test split (Stage 1
skipped; Stage 2 + Stage 3 run with ground-truth room types as input).
Computes delta, area proportion error, the boundary geometry triplet,
and the three room-accuracy sub-scores. **Must run inside the `iplan`
conda env** -- iPLAN's pinned torch / cv2 stack is not in the shared
`.venv`.

- Loads Stage 2 (Living + Location ResNet18+ASPP) and Stage 3 (G_net
  GAN + renderer) checkpoints from the iPLAN repo, then runs both
  stages per sample
- Stage 2 is iteratively re-rolled (default 50 attempts) when
  `get_rcenters` fails to place all rooms; falls back to the
  most-complete partial placement so Stage 3 still has centers to
  partition over
- Reads BOTH inputs (`Boundary`, `gt_rTypes`) and ground truth
  (`gt_rTypes`, `gt_rBoxes`) from the same iPLAN `.mat` file --
  no cross-format coordinate or category mapping is needed
- Predicted rooms come from connected-component analysis of the
  rasterized 128px Stage-3 layout, so the Living room (which iPLAN
  recovers as the leftover interior, not a predicted bbox) is included
- Generation caching via `--save_generations` / `--load_generations`
  (cached records are pickled `.mat`-style dicts, so re-scoring is
  GPU-free and doesn't even need the iPLAN repo)
- Python-3.8-friendly: imports `utils.metrics` and
  `utils.report_results` by absolute file path so the evaluator runs
  inside iPLAN's pinned env without dragging in the rest of the
  HypergraphFormer Python-3.10 stack

### `src/fewshot_diffplanner.py`

Few-shot LLM evaluator for the DiffPlanner JSON format. Asks an LLM
(GPT via OpenAI API) to play the role of DiffPlanner under the
`b+nc` setting (boundary + room **count** + room **categories** as
input; the LLM invents room boxes and adjacencies are reconstructed
from the resulting boxes).

- Reads the monolithic `data_test.json` (DiffPlanner's 12K-sample
  RPLAN test split); few-shot examples are drawn from the same file
  with the target sample excluded
- Validates the LLM's JSON against DiffPlanner's schema (room count,
  category id range, box bounds, x0&lt;x1 / y0&lt;y1) and reports
  parse/structure failure rates
- Echoes `boundary` / `boundary_expand` / `entrance_expand` back into
  the saved per-sample JSON so the LLM output is directly compatible
  with `DiffPlanner/output/post_processing.py` and the
  `benchmark/utils/visualize_diffplanner.py` panel renderer
- Same I/O pattern as the diffusion evaluators: `results/run.json`
  (aggregate), `results/run_partial.json` (incremental snapshot for
  `--checkpoint`), and one `results/run_dp/&lt;name&gt;.json` per
  successful generation

### `src/fewshot_housediffusion.py`

Few-shot LLM evaluator for the House-Diffusion JSON format. Two
backends are supported:

- `--mode fewshot-qwen` -- base or SFT-fine-tuned Qwen via vLLM
  (default) or HuggingFace, single GPU
- `--mode fewshot-gpt` -- GPT via the OpenAI API, single process

The model is asked to produce a House-Diffusion JSON
(`room_type` / `boxes` / `edges` / `ed_rm`) from an access graph.
Scoring is HD-specific (no GED): structural validation on the
generated HD JSON, plus reconstruction of an access graph from the
generated HD via the same door-mediated convention used at data-prep
time and a comparison with the input access graph (room count,
room-type multiset, edge-set F1). Per-success outputs land in
`results/run_hd/&lt;id&gt;.json` -- the same shape that
`benchmark/utils/visualize_rplan_json.py` and
`benchmark/utils/convert_json_to_pickle.py` expect.

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

### `utils/convert_png_to_iplan_mat.py`

Convert original RPLAN PNGs to iPLAN's native `.mat` format for a given
split. Wraps iPLAN's own `FloorPlan` parser (no algorithm changes) and
parallelises across CPU workers. The output `.mat` carries `name`,
`Boundary`, `gt_rTypes`, `gt_rBoxes`, `rTypes`, `rBoxes`, `rCenters`
fields and is the canonical input for `evaluate_iplan.py`. Runs inside
the `iplan` conda env (needs iPLAN's pinned `scipy` and `cv2`); the
default `--out-dir` points directly at
`dataset/rplan/dataset_iplan/` so re-running `setup_iplan.sh` is
sufficient to refresh the dataset. Honors iPLAN's shipped split lists
(`<iplan-root>/data/{train,val,test}.txt`) so we evaluate on the same
test split iPLAN was trained against. Default `test` split has 12,110
entries; ~4 minutes on 64 cores.

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

### iPLAN `.mat` Format

Produced by `convert_png_to_iplan_mat.py` (one file per RPLAN sample,
under `dataset/rplan/dataset_iplan/<id>.mat`). Each file has a top-level
`data` struct with the fields:

| Field | Shape | Description |
|---|---|---|
| `name` | scalar string | Source PNG filename |
| `Boundary` | `(N_v, 4)` int | Boundary polygon vertices `(y, x, dir, isNew)`; first two points indicate the front door |
| `gt_rTypes` | `(1, R_gt)` int | Ground-truth room categories (iPLAN 13-class vocab) |
| `gt_rBoxes` | `(R_gt, 4)` int | Ground-truth axis-aligned room boxes `(y0, x0, y1, x1)` in 128-grid pixels |
| `rTypes` | `(1, R_pred)` int | Inference-filled predicted room types (Stage 1 output, or seeded from `gt_rTypes` in OurII mode) |
| `rCenters` | `(R_pred, 2)` int | Stage 2 predicted room centers |
| `rBoxes` | `(R_pred, 4)` int | Stage 3 predicted axis-aligned room boxes |

Coordinate conventions in `.mat` files come straight from iPLAN's
upstream parser, so `(y, x)` ordering is preserved; `evaluate_iplan.py`
swaps to `(x, y)` for Shapely internally.

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
├── setup_iplan.sh                       # iPLAN automated setup (clone+patch+env+weights+data)
├── eval_houseganpp.sh                   # HouseGAN++ evaluation launcher
├── eval_housediffusion.sh               # House-Diffusion evaluation launcher
│
├── src/                                 # Core Python scripts
│   ├── dataset_loader.py                # Shared PyTorch dataset module
│   ├── evaluate_houseganpp.py           # HouseGAN++ evaluation (WMR24)
│   ├── evaluate_housediffusion.py       # House-Diffusion evaluation (WMR24)
│   ├── evaluate_diffplanner.py          # DiffPlanner b+nc cascade evaluation (RPLAN)
│   ├── evaluate_iplan.py                # iPLAN OurII cascade evaluation (RPLAN, runs in iplan conda env)
│   ├── fewshot_diffplanner.py           # LLM few-shot eval in DiffPlanner JSON format (b+nc)
│   └── fewshot_housediffusion.py        # LLM few-shot eval in House-Diffusion JSON format (Qwen / GPT)
│
├── utils/                               # Utility and conversion scripts
│   ├── convert_hypergraph_to_housegan_json.py  # WMR24 JSON → HouseGAN++ JSON
│   ├── convert_hypergraph_to_diffplanner.py    # WMR24 JSON → DiffPlanner JSON
│   ├── convert_json_to_pickle.py               # HouseGAN++ JSON → pickle
│   ├── convert_json_to_npz.py                  # HouseGAN++ JSON → NPZ
│   ├── convert_png_to_iplan_mat.py             # RPLAN PNG → iPLAN .mat (parallel)
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
