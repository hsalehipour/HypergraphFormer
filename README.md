# HypergraphFormer

**Learning Hypergraph Representation with Large Language Models for Controllable Floor-Plan Generation**

This repository contains the code for *HypergraphFormer*, a novel approach to
automatic floor-plan generation based on learning hypergraph representations
with large language models.  An instruction-fine-tuned **Qwen 3-4B** model
(via LoRA) is trained to translate *access graphs* (room connectivity) into
hierarchical **BSP-tree hypergraphs** that encode spatial relationships and
connectivity within floor plans.  Unlike rasterized or vectorized methods, the
hypergraph formulation decouples the apartment footprint from its functional
and geometric subdivisions, enabling generation of floor plans for arbitrary,
user-specified boundaries.  Reinforcement learning fine-tuning with GRPO
further improves structural consistency and semantic accuracy.

Evaluated on the **WMR24** dataset (~1,000 architect-designed floor
plans from Zurich, New York, and Singapore), HypergraphFormer achieves a GED
of **1.45** and room-set accuracy of **94.2%**, outperforming HouseDiffusion
and HouseGAN++ while requiring significantly less training data.

> **Paper:** Submitted to ICML 2026.  Preprint available on
> [arXiv](https://arxiv.org/) (link will be added upon publication).

---

## Quick Setup

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.9+ | Tested with 3.11 |
| CUDA GPU(s) | Training and evaluation require GPU |
| .NET Runtime (Mono 6.12+) | Only for the C# hypergraph geometry library |
| Git | For cloning external dependencies |

### Install

```bash
# Clone
git clone https://github.com/hsalehipour/HypergraphFormer.git
cd HypergraphFormer

# Virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Python packages
pip install -r requirements.txt

# Editable install (makes utils, scripts, benchmark, msd, rplan importable as packages)
pip install -e .

# External repos (geometry library + HouseGAN++ baseline)
./setup_hypergraph.sh
cd benchmark && ./setup_houseganpp.sh && cd ..
```

### Download the base model (one-time)

The training script runs in offline mode.  Cache the model first:

```bash
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
m = 'Qwen/Qwen3-4B-Instruct-2507'
AutoTokenizer.from_pretrained(m)
AutoModelForCausalLM.from_pretrained(m, torch_dtype='auto')
"
```

The model is stored in `~/.cache/huggingface/hub/`.

---

## Repository Structure

```
HypergraphFormer/
│
├── scripts/                        # ── Pipelines & CLI tools ──────────
│   │
│   │  # Data preparation (--mode train | eval | room-lists)
│   ├── data_preparation.py          JSON → JSONL, eval .txt, or room-list JSONs
│   │
│   │  # Training
│   ├── sft_pipeline.py              LoRA fine-tuning of Qwen (SFT)
│   ├── rlft_pipeline.py             GRPO reinforcement learning fine-tuning (RLFT)
│   │
│   │  # Generator classes (library, no CLI)
│   ├── generators.py                HypergraphGenerator, QwenGenerator, GPTGenerator
│   │
│   │  # Evaluation (all modes via --mode flag)
│   ├── evaluator.py                 Unified evaluator (trained, rooms-only,
│   │                                      fewshot-qwen, fewshot-gpt)
│   │
│   │  # Post-inference scoring (--metrics ged accuracy delta area_proportion_error)
│   └── score_results.py             Score saved results against ground truth
│
├── rplan/                          # ── RPLAN dataset processing ───────
│   ├── build_dataset_access.py      Build access graphs from DiffPlanner + original PNGs
│   ├── rplan_to_hypergraph.py       Native Python BSP conversion (with look-ahead
│   │                                      and evolutionary search)
│   ├── postprocess_hypergraphs.py   Fix disconnected access graphs
│   ├── visualize_rplan_hypergraph.py            Visualize BSP + RGL mesh
│   ├── visualize_disconnected_accessgraphs.py   Visualize disconnected samples
│   └── visualize_fixed_accessgraphs.py          Before/after comparison
│
├── utils/                          # ── Shared utility library ─────────
│   ├── __init__.py                     Public API re-exports
│   ├── utils.py                        Validation, JSON extraction, timeout
│   ├── metrics.py                      GED, delta, area proportion error, room-type accuracy,
│   │                                        produce_room_list
│   ├── report_results.py              Unified reporting: NPZ + TXT output
│   ├── analyze_metrics_by_room_count.py  Bin metrics by room count from NPZ
│   ├── hypergraph.py                   Hypergraph class, HypergraphPlotter
│   ├── prompts.py                      BSP rules, prompt templates
│   ├── data_loading.py                 Dataset helpers, example loading
│   └── benchmark_qwen_inference.py     Inference timing benchmark
│
├── dataset/                        # ── Raw datasets ───────────────────
│   ├── wmr24/                           WMR24: wmr24.json + HouseGAN++ conversions
│   │   ├── wmr24.json                       Full dataset (hypergraph format)
│   │   └── wmr24_housegan_json/             Converted for baseline evaluation
│   │       ├── processed_rplan/                 NPZ files (House-Diffusion)
│   │       └── processed_rplan_pickles/         Pickle files (HouseGAN++)
│   ├── rplan/                           RPLAN data (see "Preparing the RPLAN Dataset")
│   │   ├── dataset_original/               RPLAN floorplan PNGs (downloaded)
│   │   ├── dataset_diffplanner/            DiffPlanner JSON splits (downloaded)
│   │   ├── dataset_access/                 Access graphs (built from above)
│   │   ├── dataset_hypergraph_la0/         BSP hypergraphs (exported)
│   │   └── dataset_hypergraph_la0_fixed/   Post-processed (access graphs fixed)
│   └── msd/                             Modified Swiss Dwellings CSV
│
├── eval_format/                    # ── Prepared evaluation data ───────
│   ├── train/                          779 .txt files (accessgraph + hypergraph)
│   └── test/                           167 .txt files
│
├── checkpoints/                    # ── Trained model checkpoints ──────
│
├── msd/                            # MSD dataset processing
│   ├── graph.py                        MSDGraph class
│   ├── plotter.py                      MSDPlotter class
│   ├── msd_to_hypergraph.py            Conversion utilities
│   └── notebooks/                      MSD exploration notebooks
│
├── benchmark/                      # HouseGAN++ & House-Diffusion baselines
│   ├── eval_houseganpp.sh              Evaluate HouseGAN++
│   ├── eval_housediffusion.sh          Evaluate House-Diffusion
│   ├── setup_houseganpp.sh             Setup HouseGAN++ environment + data conversion
│   ├── setup_housediffusion.sh         Setup House-Diffusion environment + data conversion
│   ├── utils/
│   │   ├── convert_hypergraph_to_housegan_json.py   WMR24 JSON → HouseGAN++ JSON
│   │   ├── convert_json_to_pickle.py                HouseGAN++ JSON → pickle
│   │   └── convert_json_to_npz.py                   HouseGAN++ JSON → NPZ
│   └── src/
│       ├── dataset_loader.py            Shared PyTorch dataset module
│       ├── evaluate_houseganpp.py       HouseGAN++ evaluation
│       └── evaluate_housediffusion.py   House-Diffusion evaluation
│
├── notebooks/                      # Jupyter notebooks
│   └── generate_and_visualize_hypergraphs.ipynb
│
├── prepare_rplan_hypergraph.sh     # End-to-end RPLAN pipeline (download, build, export)
├── run_fewshot_experiments.sh      # Batch few-shot experiments + scoring
├── setup_hypergraph.sh             # Clone & build C# geometry library
├── pyproject.toml                  # Editable install config (pip install -e .)
├── requirements.txt
└── README.md
```

---

## Preparing the RPLAN Dataset

The RPLAN pipeline converts the raw [RPLAN](https://wutomwu.github.io/particulars.html?id=1)
floor-plan images into BSP-tree hypergraphs.  The automated script handles
downloading, building, and exporting:

```bash
# Full pipeline (downloads data, exports hypergraphs, fixes access graphs)
./prepare_rplan_hypergraph.sh --access

# Reuse existing dataset_access (faster, skips step 3)
./prepare_rplan_hypergraph.sh

# Clean rebuild of hypergraph data only
./prepare_rplan_hypergraph.sh --clean

# Also generate debug visualizations of remaining disconnections
./prepare_rplan_hypergraph.sh --debug

# Override worker count (default: nproc)
./prepare_rplan_hypergraph.sh -j 32
```

The pipeline runs the following steps:

1. **Download RPLAN original dataset** — Floorplan PNGs from
   [Box](https://berkeley.app.box.com/s/m8369bxmaixananf4otajydraxa7y92v)
   → `dataset/rplan/dataset_original/`
2. **Download DiffPlanner data** — Room annotations from
   [GitHub](https://github.com/shidong-wang/DiffPlanner/releases) →
   `dataset/rplan/dataset_diffplanner/`
3. **Build access graphs** (`--access`) — Extracts door positions from the
   original PNGs and matches them to DiffPlanner room annotations →
   `dataset/rplan/dataset_access/`
4. **Export BSP hypergraphs** — Pure-Python BSP conversion with configurable
   look-ahead scoring → `dataset/rplan/dataset_hypergraph_la0/`
5. **Post-process** — Fixes disconnected access graphs by adding edges between
   physically adjacent rooms → `dataset/rplan/dataset_hypergraph_la0_fixed/`
6. **Debug visualization** (`--debug`) — Renders remaining disconnected samples
   for manual inspection → `debug/`

### BSP Conversion Details

The native Python BSP converter (`rplan/rplan_to_hypergraph.py`) supports
several split-selection strategies:

```bash
# Greedy (fastest)
python rplan/rplan_to_hypergraph.py --export --lookahead 0 -j 64

# 1-step look-ahead (default)
python rplan/rplan_to_hypergraph.py --export --lookahead 1 -j 64

# Evolutionary search (slowest, highest quality)
python rplan/rplan_to_hypergraph.py --export --evolve --evo-pop 100 --evo-gens 30 -j 64
```

The split selection optimizes for: (1) minimizing the number of cuts through
adjacent rooms, (2) balanced area splits, and (3) reasonable aspect ratios.

### Visualization

```bash
# Visualize specific samples (BSP + RGL mesh side by side)
python rplan/visualize_rplan_hypergraph.py --sample 3951 57395

# Visualize disconnected access graphs in a dataset
python rplan/visualize_disconnected_accessgraphs.py \
    --dataset-dir dataset/rplan/dataset_hypergraph_la0_fixed --output-dir debug

# Compare before/after access graph fixes
python rplan/visualize_fixed_accessgraphs.py \
    --before-dir dataset/rplan/dataset_hypergraph_la0 \
    --after-dir dataset/rplan/dataset_hypergraph_la0_fixed --output-dir debug_fixed
```

### RPLAN Dataset Statistics

| Split | Samples |
|---|---|
| Train | 56,053 |
| Val | 12,018 |
| Test | 12,002 |
| **Total** | **80,073** |

---

## Workflow

### 1. Prepare training data

The dataset ships pre-split (`train.json`, `validation.json`,
`test.json`).  Convert it to the JSONL instruction format that
`sft_pipeline.py` expects:

```bash
python scripts/data_preparation.py --mode train \
    --dataset-dir dataset/hypergraph
```

This produces `train.jsonl`, `validation.jsonl`, and `test.jsonl` in
`dataset/hypergraph/sft/`.

### 2. Train with SFT (LoRA fine-tuning)

```bash
# Single GPU (default: 3 epochs, LoRA r=64, alpha=128)
python scripts/sft_pipeline.py --gpu 0

# Multi-GPU with Accelerate
accelerate launch --num_processes=2 scripts/sft_pipeline.py --multi-gpu

# Custom settings
python scripts/sft_pipeline.py \
    --epochs 5 --lora-r 128 --lora-alpha 256 --gpu 0
```

Checkpoints are written to
`checkpoints/qwen_hypergraph_filtered/epochs{N}_lora_r{R}_alpha{A}/`.

### 3. Train with RL (GRPO fine-tuning)

Prepare RL training JSONL (footprint prompt) into `dataset/hypergraph/rl/`:

```bash
python scripts/data_preparation.py --mode train \
    --dataset-dir dataset/hypergraph --footprint
```

Reduce RL training set by assistant token length (writes `*_reduced.jsonl`):

```bash
python scripts/rlft_pipeline.py --reduce-tokens --max-tokens 2000
```

Export room lists (used by the RL reward):

```bash
python scripts/data_preparation.py --mode room-lists \
    --input-file dataset/hypergraph/train.json \
    --output-dir dataset/hypergraph/rl/train_reduced_room_lists

python scripts/data_preparation.py --mode room-lists \
    --input-file dataset/hypergraph/validation.json \
    --output-dir dataset/hypergraph/rl/validation_reduced_room_lists
```

Run GRPO fine-tuning:

```bash
# Single GPU
python scripts/rlft_pipeline.py --gpu 0

# Multi-GPU
accelerate launch --num_processes=8 scripts/rlft_pipeline.py --multi-gpu
```

### 4. Evaluate the fine-tuned model

Prepare the eval-format directory (one `{id}.txt` per sample):

```bash
python scripts/data_preparation.py --mode eval \
    --input-file dataset/hypergraph/test.json \
    --output-dir dataset/hypergraph/test
```

Optionally pass `--aggregate` to merge same-type, spatially adjacent,
access-connected rooms into single nodes in the output access graphs:

```bash
python scripts/data_preparation.py --mode eval --aggregate \
    --input-file dataset/hypergraph/test.json \
    --output-dir dataset/hypergraph/test_aggregated
```

Run evaluation (generates hypergraphs + inline scoring):

```bash
# Single GPU (use --footprint if model was trained with footprint prompts)
python scripts/evaluator.py --mode trained \
    --model checkpoints/grpo_hypergraph/checkpoint-500 \
    --dataset-dir dataset/hypergraph/test \
    --footprint --noged --shuffle \
    --output results_eval.json

# Multi-GPU
accelerate launch --num_processes=8 scripts/evaluator.py --mode trained \
    --model checkpoints/grpo_hypergraph/checkpoint-500 \
    --dataset-dir dataset/hypergraph/test \
    --footprint --noged --shuffle
```

### 5. Evaluate the base model (few-shot, no fine-tuning)

```bash
# Qwen few-shot (single GPU)
python scripts/evaluator.py --mode fewshot-qwen \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --dataset-dir eval_format/test

# GPT few-shot (OpenAI API)
python scripts/evaluator.py --mode fewshot-gpt \
    --model gpt-4-turbo --dataset-dir eval_format/test
```

### 6. Score saved results (post-inference)

Compute any combination of GED, accuracy, delta, and area proportion error
metrics on saved `*_hypergraphs.json` files:

```bash
# All metrics at once
python scripts/score_results.py \
    --results-file path/to/results_test_hypergraphs.json \
    --test-dir dataset/hypergraph/test \
    --output-dir eval_output \
    --metrics ged accuracy delta area_proportion_error

# Just GED
python scripts/score_results.py \
    --results-file path/to/results_test_hypergraphs.json \
    --test-dir dataset/hypergraph/test \
    --output-dir eval_output \
    --metrics ged

# GED with corrected access graph (mesh-adjacency corrected)
python scripts/score_results.py \
    --results-file path/to/results_test_hypergraphs.json \
    --test-dir dataset/hypergraph/test \
    --output-dir eval_output \
    --metrics ged \
    --correct-accessgraph
```

### 7. Benchmark against baselines (HouseGAN++, House-Diffusion)

See [`benchmark/README.md`](benchmark/README.md) for setting up and evaluating
the HouseGAN++ and House-Diffusion baselines on the WMR24 dataset.

### 8. Evaluate a single hypergraph (quick test)

```bash
# Few-shot Qwen on one sample
python scripts/evaluator.py --mode fewshot-qwen \
    --dataset-dir eval_format/train --max-samples 1

# Few-shot GPT on one sample (requires OPENAI_API_KEY in .env)
python scripts/evaluator.py --mode fewshot-gpt \
    --dataset-dir eval_format/train --max-samples 1
```

### 9. Run batch few-shot experiments

`run_fewshot_experiments.sh` automates running `evaluator.py` across
multiple shot counts and rounds, with optional post-run scoring:

```bash
# GPT: 1/2/3-shot, 5 rounds each, then score with GED
bash run_fewshot_experiments.sh --backend gpt \
    --examples-dir eval_format/train \
    --score --test-dir dataset/hypergraph/test --metrics ged

# Qwen: custom shots and rounds
bash run_fewshot_experiments.sh --backend qwen \
    --shots 1 3 5 --rounds 3 \
    --examples-dir eval_format/train

# Score existing results only (no generation)
bash run_fewshot_experiments.sh --score-only \
    --output-dir fewshot_results \
    --test-dir dataset/hypergraph/test --metrics ged accuracy
```

---

## Key Concepts

| Concept | Description |
|---|---|
| **Hypergraph** | A reduced-order representation of a floor plan consisting of a BSP tree (spatial hierarchy) combined with an access graph (room connectivity). Each intermediate node encodes area and splitting angle; leaf nodes correspond to rooms and store their type and connections. |
| **Access Graph** | A graph where nodes represent rooms (labelled by type) and edges represent door-based connections. Serves as the input condition for generation. |
| **BSP Tree** | Binary Space Partition — the root represents the full apartment; each split recursively divides a region into two sub-regions along an axis-aligned angle until individual rooms are obtained. |
| **GED** | Graph Edit Distance — minimum-cost sequence of vertex and edge edits to transform one access graph into another, used as the primary structural similarity metric. |
| **Room-Set Accuracy** | Fraction of test samples where the generated floor plan exactly matches the ground-truth room composition (count and types). |
| **Geometric Compactness Deviation (\(\delta\))** | Per-room shape deviation measuring how much a generated room polygon departs from the compactness of its ground-truth counterpart. |
| **Area Proportion Error** | Mean absolute difference between room-area proportions of paired predicted and ground-truth rooms, averaged per sample. |
| **LoRA** | Low-Rank Adaptation — parameter-efficient fine-tuning; default configuration: rank 64, alpha 128. |
| **GRPO** | Group Relative Policy Optimization — RL algorithm used in the second training stage to optimize metric-based rewards (accuracy, GED, compactness). |

---

## Shared Modules

The `utils/` package provides shared helpers used by the pipeline scripts:

| Module | Contents |
|---|---|
| `utils/hypergraph.py` | `Hypergraph` class (validation, transformers, RGL integration, parsing, access-graph component analysis), `HypergraphPlotter` (mesh sliver filtering, `unique_id`-based face labelling) |
| `utils/utils.py` | `validate_hypergraph`, `TimeoutException`, `time_limit`, JSON extraction helpers |
| `utils/metrics.py` | `accessgraph_dict_to_nx`, `ged_between_graphs`, `compute_ged_from_hypergraph`, `compute_room_similarity_score`, `compute_room_area_proportion_error`, `room_type_accuracy`, `compute_room_type_accuracy`, `produce_room_list` |
| `utils/report_results.py` | `report_results` -- unified NPZ + TXT reporting for all evaluation pipelines |
| `utils/analyze_metrics_by_room_count.py` | CLI script to bin NPZ metrics by room count and produce per-bin statistics |
| `utils/prompts.py` | BSP rules text, prompt templates (`create_structured_prompt`, `create_simple_prompt`, etc.) |
| `utils/data_loading.py` | `load_example_hypergraphs`, `load_target_from_dataset`, `save_training_data`, `log_slow_generation`, `HypergraphDataset`, `collate_fn` |
| `utils/benchmark_qwen_inference.py` | Inference timing benchmark for Qwen models |

---

## External Dependencies

### hypergraph (C# Geometry Library)

- **Location**: `~/REPOs/hypergraph`
- **Purpose**: Hypergraph construction and geometry operations via pythonnet
- **Setup**: `./setup_hypergraph.sh` (clones repo, builds DLLs, installs as editable package)
- **Import**: `from hypergraph.api.lib.tools import RGL`

### houseganpp (Baseline GAN)

- **Location**: `~/REPOs/houseganpp`
- **Purpose**: Baseline generative model for comparison
- **Setup**: `cd benchmark && ./setup_houseganpp.sh`
- **Details**: See `benchmark/README.md`

Both repositories remain **unmodified** and are cloned as external dependencies.

---

## Troubleshooting

### "No module named 'utils'" (or scripts, benchmark, msd, rplan)

```bash
pip install -e .   # editable install — makes project packages importable
```

### "No module named 'transformers'" (or accelerate, peft, etc.)

```bash
pip install -r requirements.txt
```

### Import Error: "No module named 'hypergraph'"

```bash
./setup_hypergraph.sh   # installs as editable package
```

### "Could not find libmono" Error

```bash
conda install -c conda-forge mono=6.12.0.199 -y
```

Required for `pythonnet` to interface with C# DLLs.

### HouseGAN++ Import Errors

```bash
cd benchmark && ./setup_houseganpp.sh
```

---

## Datasets

### WMR24 Dataset

The **WMR24** dataset (Weber, Mueller, Roth 2024) comprises ~1,100
architect-designed, real-world residential floor plans from three geographic
regions (Zurich, New York, Singapore), ranging from studios to 6-bedroom
apartments.

| Region | Total | Mean Area (m²) |
|---|---|---|
| New York | 472 | 65.1 |
| Zurich | 440 | 82.1 |
| Singapore | 202 | 77.4 |
| **Total** | **~1,100** | **74.0** |

- **Location**: `dataset/wmr24/wmr24.json`
- **Format**: Single JSON array of hypergraph records (database, id, area, bounds, facade, circulation, split)
- **Use**: Primary dataset for training, evaluation, and baseline benchmarking

### RPLAN Dataset

The [RPLAN](https://doi.org/10.1145/3355089.3356524) dataset contains ~80,000
real-world residential floor plans.  We convert each sample into a BSP-tree
hypergraph using the pipeline described in
[Preparing the RPLAN Dataset](#preparing-the-rplan-dataset).

| Split | Samples |
|---|---|
| Train | 56,053 |
| Val | 12,018 |
| Test | 12,002 |
| **Total** | **80,073** |

- **Location**: `dataset/rplan/dataset_hypergraph_la0_fixed/`
- **Format**: Pre-split JSON files (`train.json`, `val.json`, `test.json`)
- **Use**: Large-scale training and evaluation

### Modified Swiss Dwellings (MSD)

- **Location**: `dataset/msd/`
- **Format**: CSV + processed apartment data
- **Use**: Additional hypergraph generation experiments

### Evaluation Format

- **Location**: `eval_format/`
- **Format**: One `.txt` file per sample, each containing `accessgraph` and `hypergraph` as JSON
- **Use**: Direct input for evaluation and generation scripts

---

## Citation

If you use this code or the WMR24 dataset, please cite:

```bibtex
@inproceedings{hypergraphformer2026,
  title     = {Learning Hypergraph Representation with Large Language Models
               for Controllable Floor-Plan Generation},
  author    = {Anonymous},
  booktitle = {Submitted to the International Conference on Machine Learning (ICML)},
  year      = {2026},
  note      = {Under review. Preprint available on arXiv.}
}
```

---

**Project**: HypergraphFormer
**Last Updated**: 2026-04-16
**Python**: 3.9+
