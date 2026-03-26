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

Evaluated on the **Hypergraph1k** dataset (~1,000 architect-designed floor
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

# Editable install (makes utils, scripts, benchmark, msd importable as packages)
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
│   │  # Data preparation (--mode jsonl | eval-format | room-lists)
│   ├── data_preparation.py          JSON → JSONL, eval-format .txt, or room-list JSONs
│   │
│   │  # Training
│   ├── sft_pipeline.py              LoRA fine-tuning of Qwen (SFT)
│   ├── rlft_pipeline.py             GRPO reinforcement learning fine-tuning (RLFT)
│   │
│   │  # Few-shot generation (single + batch, Qwen + GPT)
│   ├── fewshot_generator.py         OOP generator (QwenGenerator, GPTGenerator)
│   │
│   │  # Evaluation (all modes via --mode flag)
│   ├── evaluator.py                 Unified evaluator (trained, rooms-only,
│   │                                      fewshot-qwen, fewshot-gpt)
│   │
│   │  # Post-inference scoring (--metrics ged accuracy delta)
│   └── score_results.py             Score saved results against ground truth
│
├── utils/                          # ── Shared utility library ─────────
│   ├── __init__.py                     Public API re-exports
│   ├── utils.py                        Validation, JSON extraction, timeout
│   ├── metrics.py                      GED, delta, room-type accuracy, produce_room_list
│   ├── hypergraph.py                   Hypergraph class, HypergraphPlotter
│   ├── prompts.py                      BSP rules, prompt templates
│   ├── data_loading.py                 Dataset helpers, example loading
│   └── benchmark_qwen_inference.py     Inference timing benchmark
│
├── dataset/                        # ── Raw datasets ───────────────────
│   ├── hypergraph/                      train.json, validation.json, test.json
│   ├── msd/                            Modified Swiss Dwellings CSV
│   └── furniture_json/                 Furniture placement data
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
│   ├── eval_houseganpp.sh
│   ├── eval_housediffusion.sh
│   ├── finetune_houseganpp.sh
│   ├── finetune_housediffusion.sh
│   └── src/
│
├── notebooks/                      # Jupyter notebooks
│   └── generate_and_visualize_hypergraphs.ipynb
│
├── run_fewshot_experiments.sh       # Batch few-shot experiments + scoring
├── pyproject.toml                  # Editable install config (pip install -e .)
├── requirements.txt
├── setup_hypergraph.sh
└── README.md
```

---

## Workflow

### 1. Prepare training data

The dataset ships pre-split (`train.json`, `validation.json`,
`test.json`).  Convert it to the JSONL instruction format that
`sft_pipeline.py` expects:

```bash
python scripts/data_preparation.py --mode jsonl \
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
python scripts/data_preparation.py --mode jsonl \
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
python scripts/data_preparation.py --mode eval-format \
    --input-file dataset/hypergraph/test.json \
    --output-dir dataset/hypergraph/test
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

Compute any combination of GED, accuracy, and delta metrics on saved
`*_hypergraphs.json` files:

```bash
# All metrics at once
python scripts/score_results.py \
    --results-file path/to/results_test_hypergraphs.json \
    --test-dir dataset/hypergraph/test \
    --metrics ged accuracy delta

# Just GED
python scripts/score_results.py \
    --results-file path/to/results_test_hypergraphs.json \
    --test-dir dataset/hypergraph/test \
    --metrics ged

# GED with corrected access graph (mesh-adjacency corrected)
python scripts/score_results.py \
    --results-file path/to/results_test_hypergraphs.json \
    --test-dir dataset/hypergraph/test \
    --metrics ged \
    --correct-accessgraph

# Delta with CSV output
python scripts/score_results.py \
    --results-file path/to/results_test_hypergraphs.json \
    --test-dir dataset/hypergraph/test \
    --metrics delta \
    --output-csv delta_metrics.csv

# Summary-only (machine-readable, for scripting)
python scripts/score_results.py \
    --results-file path/to/results_test_hypergraphs.json \
    --test-dir dataset/hypergraph/test \
    --metrics ged accuracy \
    --summary-only
```

### 7. Generate a single hypergraph (interactive)

```bash
# With Qwen (local, streaming progress bar)
python scripts/fewshot_generator.py --backend qwen \
    --from-dataset --examples-dir eval_format/train

# With GPT (OpenAI API — requires OPENAI_API_KEY in .env)
python scripts/fewshot_generator.py --backend gpt \
    --from-dataset --examples-dir eval_format/train
```

### 8. Run batch few-shot experiments

`run_fewshot_experiments.sh` automates running `fewshot_generator.py` across
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
| **Access Graph** | A graph where nodes represent rooms (labelled by type) and edges represent adjacency (doors / open connections). Serves as the input condition for generation. |
| **BSP Tree** | Binary Space Partition — the root represents the full apartment; each split recursively divides a region into two sub-regions along an axis-aligned angle until individual rooms are obtained. |
| **GED** | Graph Edit Distance — minimum-cost sequence of vertex and edge edits to transform one access graph into another, used as the primary structural similarity metric. |
| **Room-Set Accuracy** | Fraction of test samples where the generated floor plan exactly matches the ground-truth room composition (count and types). |
| **Geometric Compactness Deviation (\(\delta\))** | Per-room shape deviation measuring how much a generated room polygon departs from the compactness of its ground-truth counterpart. |
| **LoRA** | Low-Rank Adaptation — parameter-efficient fine-tuning; default configuration: rank 64, alpha 128. |
| **GRPO** | Group Relative Policy Optimization — RL algorithm used in the second training stage to optimize metric-based rewards (accuracy, GED, compactness). |

---

## Shared Modules

The `utils/` package provides shared helpers used by the pipeline scripts:

| Module | Contents |
|---|---|
| `utils/hypergraph.py` | `Hypergraph` class (validation, transformers, RGL integration, parsing), `HypergraphPlotter` |
| `utils/utils.py` | `validate_hypergraph`, `TimeoutException`, `time_limit`, JSON extraction helpers |
| `utils/metrics.py` | `accessgraph_dict_to_nx`, `ged_between_graphs`, `compute_ged_from_hypergraph`, `compute_room_similarity_score`, `room_type_accuracy`, `compute_room_type_accuracy`, `bucketed_room_accuracy`, `produce_room_list` |
| `utils/prompts.py` | BSP rules text, prompt templates (`create_structured_prompt`, `create_simple_prompt`, etc.) |
| `utils/data_loading.py` | `load_example_hypergraphs`, `load_target_from_dataset`, `save_training_data`, `log_slow_generation`, `HypergraphDataset`, `collate_fn` |
| `utils/benchmark_qwen_inference.py` | Inference timing benchmark for Qwen models |

---

## External Dependencies

### hypergraph (C# Geometry Library)

- **Location**: `~/REPOs/hypergraph`
- **Purpose**: Hypergraph construction and geometry operations via pythonnet
- **Setup**: `./setup_hypergraph.sh`
- **Import**: `from hypergraph.api.lib.tools import RGL`

### houseganpp (Baseline GAN)

- **Location**: `~/REPOs/houseganpp`
- **Purpose**: Baseline generative model for comparison
- **Setup**: `cd benchmark && ./setup_houseganpp.sh`
- **Details**: See `benchmark/README.md`

Both repositories remain **unmodified** and are cloned as external dependencies.

---

## Troubleshooting

### "No module named 'utils'" (or scripts, benchmark, msd)

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

### Hypergraph1k Dataset

The **Hypergraph1k** dataset, introduced in the paper, comprises ~1,100
architect-designed, real-world residential floor plans from three geographic
regions (Zurich, New York, Singapore), ranging from studios to 6-bedroom
apartments.  It is split 7:1:1 into train / validation / test sets.

| Region | Train | Val | Test | Mean Area (m^2) |
|---|---|---|---|---|
| New York | 326 | 70 | 76 | 65.1 |
| Zurich | 305 | 64 | 71 | 82.1 |
| Singapore | 149 | 33 | 20 | 77.4 |
| **Total** | **780** | **167** | **167** | **74.0** |

- **Location**: `dataset/hypergraph/`
- **Format**: Pre-split JSON files (`train.json`, `validation.json`, `test.json`)
- **Use**: Primary dataset for training and evaluation

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

If you use this code or the Hypergraph1k dataset, please cite:

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
**Last Updated**: 2026-03-16
**Python**: 3.9+
