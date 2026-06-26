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
user-specified boundaries.

Evaluated on the **WMR24** dataset (~1,000 architect-designed floor
plans from Zurich, New York, and Singapore), and **RPLAN** dataset (Converted to the Hyperrgaph format), HypergraphFormer outperforms HouseDiffusion
and HouseGAN++ while requiring significantly less training data and allowing controllable edits of generated floorplans.

> **Paper:** Submitted to NeurIPS 2026.  Preprint available on
> [arXiv](https://arxiv.org/abs/2605.18932).

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

# (Recommended) Start a Docker container with CUDA 12.4 + Python 3.11 and mount the repo.
# Run this from the parent directory of HypergraphFormer (e.g. your workspace root).

docker run -it --gpus all --name hypergraphformer \
    -v "$(pwd)":/workspace \
    -w /workspace/HypergraphFormer \
    nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 \
    bash

# Install Python 3.11
apt update && apt install -y python3.11 python3.11-venv python3-dev python3-pip



# System dependencies (required to build pygraphviz and the C# geometry library)
apt install -y --no-install-recommends gcc graphviz libgraphviz-dev git mono-complete nuget curl wget unzip

# Virtual environment
python3.11 -m venv .venv
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

The training script runs in offline mode.  Cache the model first.

`transformers` 5.x requires **PyTorch ≥ 2.4**; if you use CUDA 12.1 wheels (e.g. driver max **CUDA 12.2**), install PyTorch before the snippet:  
`pip install "torch>=2.4" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

```bash
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
m = 'Qwen/Qwen3-4B-Instruct-2507'
AutoTokenizer.from_pretrained(m)
AutoModelForCausalLM.from_pretrained(m, torch_dtype='auto')
"
```

The model is stored in `~/.cache/huggingface/hub/`.

### Load the fine-tuned adapters from the Hub

The trained LoRA adapters are published on the Hugging Face Hub at
[`NikitaKlimenko/HypergraphFormer`](https://huggingface.co/NikitaKlimenko/HypergraphFormer),
one adapter per training-set size. Each adapter lives in its own subfolder:

| Subfolder (`subfolder=`) | Training set |
|---|---|
| `qwen_hypergraphformer_1000_samples/checkpoint-240`   | 1,000 samples |
| `qwen_hypergraphformer_5000_samples/checkpoint-750`   | 5,000 samples |
| `qwen_hypergraphformer_10000_samples/checkpoint-1500` | 10,000 samples |
| `qwen_hypergraphformer_25000_samples/checkpoint-3900` | 25,000 samples |
| `qwen_hypergraphformer/checkpoint-8700`               | full dataset |

Load any checkpoint on top of the base model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "Qwen/Qwen3-4B-Instruct-2507"
repo_id = "NikitaKlimenko/HypergraphFormer"
subfolder = "qwen_hypergraphformer/checkpoint-8700"

tok = AutoTokenizer.from_pretrained(repo_id, subfolder=subfolder)
base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, repo_id, subfolder=subfolder)
```

Swap `subfolder` to use a different dataset size (e.g.
`qwen_hypergraphformer_1000_samples/checkpoint-240`).

---

## Repository Structure

> The tree below lists **version-controlled** files only.  Datasets, prepared
> evaluation data, checkpoints, and inference outputs are generated or downloaded
> locally (see [Datasets](#datasets) and
> [Preparing the RPLAN Dataset](#preparing-the-rplan-dataset)) and are not tracked.

```
HypergraphFormer/
│
├── scripts/                        # ── Pipelines & CLI tools ──────────
│   │
│   │  # Data preparation (--mode train | eval)
│   ├── data_preparation.py          JSON → JSONL or eval .txt
│   │
│   │  # Training
│   ├── sft_pipeline.py              LoRA fine-tuning of Qwen (SFT)
│   │
│   │  # Generator classes (library, no CLI)
│   ├── generators.py                HypergraphGenerator, QwenGenerator, GPTGenerator
│   │
│   │  # Evaluation (all modes via --mode flag)
│   ├── evaluator.py                 Unified evaluator (trained, rooms-only,
│   │                                      fewshot-qwen, fewshot-gpt)
│   │
│   │  # Post-inference scoring (--metrics ged accuracy delta area_proportion_error)
│   ├── score_results.py             Score saved results against ground truth
│   │
│   │  # Procedural editing & parametric optimization
│   ├── procedural_edits.py          Add-remove / rotate procedural edit operations
│   ├── postprocessing.py            Post-inference hypergraph cleanup
│   ├── run_parametric.py            Parametric optimization driver
│   ├── utils_parametric.py          Helpers for parametric optimization
│   ├── qwen_predict_accessgraph_edges.py   Predict access-graph edges from room sets
│   ├── partition_dataset.py         Subsample sft/ → sft_<N>/ for dataset-size ablations
│   └── __init__.py
│
├── rplan/                          # ── RPLAN dataset processing ───────
│   ├── build_dataset_access.py      Build access graphs from DiffPlanner + original PNGs
│   ├── rplan_to_hypergraph.py       Native Python BSP conversion (with look-ahead
│   │                                      and evolutionary search)
│   ├── fix_rplan_accessgraph_connectivity.py   Fix disconnected access graphs
│   ├── visualize_rplan_hypergraph.py            Visualize BSP + RGL mesh
│   ├── visualize_disconnected_accessgraphs.py   Visualize disconnected samples
│   ├── visualize_fixed_accessgraphs.py          Before/after comparison
│   └── __init__.py
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
│   ├── visualize_results.py            Floor-plan & access-graph renderer (generated, GT RPLAN, GT WMR24)
│   └── benchmark_qwen_inference.py     Inference timing benchmark
│
├── dataset/                        # ── Raw datasets (tracked: WMR24 only) ──
│   └── wmr24/                           WMR24 dataset
│       └── wmr24.json                       Full dataset (hypergraph format)
│
├── msd/                            # MSD dataset processing
│   ├── graph.py                        MSDGraph class
│   ├── plotter.py                      MSDPlotter class
│   ├── msd_to_hypergraph.py            Conversion utilities
│   ├── validate_faces_leaves.py        Validate mesh faces against leaf nodes
│   ├── process_unsuccessful.py         Retry / handle failed conversions
│   ├── notebooks/
│   │   └── MSD to Hypergraph and Reverse.ipynb
│   └── __init__.py
│
├── benchmark/                      # ── Baselines: HouseGAN++, House-Diffusion, DiffPlanner, iPlan ──
│   ├── README.md
│   ├── eval_houseganpp.sh              Evaluate HouseGAN++
│   ├── eval_housediffusion.sh          Evaluate House-Diffusion
│   ├── setup_houseganpp.sh             Setup HouseGAN++ environment + data conversion
│   ├── setup_housediffusion.sh         Setup House-Diffusion environment + data conversion
│   ├── setup_diffplanner.sh            Setup DiffPlanner baseline
│   ├── setup_iplan.sh                  Setup iPlan baseline
│   ├── utils/
│   │   ├── convert_hypergraph_to_housegan_json.py   WMR24 JSON → HouseGAN++ JSON
│   │   ├── convert_hypergraph_to_diffplanner.py     WMR24 JSON → DiffPlanner format
│   │   ├── convert_hypergraph_to_iplan_mat.py       WMR24 JSON → iPlan .mat
│   │   ├── convert_png_to_iplan_mat.py              RPLAN PNG → iPlan .mat
│   │   ├── convert_json_to_pickle.py                HouseGAN++ JSON → pickle
│   │   ├── convert_json_to_npz.py                   HouseGAN++ JSON → NPZ
│   │   ├── visualize_diffplanner.py                 Visualize DiffPlanner outputs
│   │   └── visualize_rplan_json.py                  Visualize RPLAN JSON samples
│   └── src/
│       ├── dataset_loader.py            Shared PyTorch dataset module
│       ├── evaluate_houseganpp.py       HouseGAN++ evaluation
│       ├── evaluate_housediffusion.py   House-Diffusion evaluation
│       ├── evaluate_diffplanner.py      DiffPlanner evaluation
│       └── evaluate_iplan.py            iPlan evaluation
│
├── prepare_rplan_hypergraph.sh     # End-to-end RPLAN pipeline (download, build, export)
├── setup_hypergraph.sh             # Clone & build C# geometry library
├── pyproject.toml                  # Editable install config (pip install -e .)
├── requirements.txt
├── .gitignore
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
# Output saved to rplan_rgl_vis/rplan_hg_{id}.png
python rplan/visualize_rplan_hypergraph.py \
    --dataset-dir dataset/rplan/dataset_hypergraph_la0_fixed \
    --sample 3951 57395

# Visualize disconnected access graphs in a dataset
python rplan/visualize_disconnected_accessgraphs.py \
    --dataset-dir dataset/rplan/dataset_hypergraph_la0_fixed --output-dir debug

# Compare before/after access graph fixes
python rplan/visualize_fixed_accessgraphs.py \
    --original-dir dataset/rplan/dataset_hypergraph_la0 \
    --fixed-dir dataset/rplan/dataset_hypergraph_la0_fixed --output-dir debug_fixed
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

The RPLAN dataset is pre-split into `train.json`, `validation.json`,
`test.json` by `prepare_rplan_hypergraph.sh`.  Convert it to the JSONL
instruction format that `sft_pipeline.py` expects:

```bash
python scripts/data_preparation.py --mode train \
    --dataset-dir dataset/rplan/dataset_hypergraph_la0_fixed
```

This produces `train.jsonl`, `validation.jsonl`, and `test.jsonl` in
`dataset/rplan/dataset_hypergraph_la0_fixed/sft/`.

### 2. Train with SFT (LoRA fine-tuning)

```bash
# Single GPU (default: 3 epochs, LoRA r=64, alpha=128)
python scripts/sft_pipeline.py --gpu 0 \
    --data-dir dataset/rplan/dataset_hypergraph_la0_fixed/sft

# Multi-GPU with Accelerate
accelerate launch --num_processes=2 scripts/sft_pipeline.py --multi-gpu \
    --data-dir dataset/rplan/dataset_hypergraph_la0_fixed/sft

# Custom settings
python scripts/sft_pipeline.py \
    --epochs 5 --lora-r 128 --lora-alpha 256 --gpu 0 \
    --data-dir dataset/rplan/dataset_hypergraph_la0_fixed/sft
```

Checkpoints are written to
`checkpoints/qwen_hypergraphformer/epochs{N}_lora_r{R}_alpha{A}/`.

### 3. Evaluate the fine-tuned model

Prepare the eval-format directory (one `{id}.txt` per sample):

```bash
python scripts/data_preparation.py --mode eval --data-format access \
    --input-file dataset/rplan/dataset_access/data_test.json \
    --output-dir dataset/rplan/eval_test
```
See scripts/data_preparation.py for other options to prepare test data directly from the hypergraph format.

Prepare WMR24 test data:

```bash
python scripts/data_preparation.py --mode eval --data-format hypergraph \
    --input-file dataset/wmr24/wmr24.json \
    --output-dir dataset/wmr24/test
```

Run evaluation (generates hypergraphs + inline scoring):

```bash
# From a local checkpoint (single GPU)
python scripts/evaluator.py --mode trained \
    --model checkpoints/qwen_hypergraphformer/epochs8_lora_r64_alpha128/checkpoint-8700 \
    --dataset-dir dataset/rplan/dataset_hypergraph_la0_fixed/test_aggregated \
    --noged --shuffle \
    --output results_eval.json

# From a local checkpoint (multi-GPU)
VLLM_USE_V1=0 torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/evaluator.py --mode trained \
    --model checkpoints/qwen_hypergraphformer/epochs8_lora_r64_alpha128/checkpoint-8700 \
    --dataset-dir dataset/rplan/dataset_hypergraph_la0_fixed/test_aggregated \
    --noged --shuffle \
    --vllm-prompt-batch-size 8

# From Hugging Face Hub (full-dataset adapter, default subfolder), with multi-GPU:
VLLM_USE_V1=0 torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/evaluator.py --mode trained --hub \
    --dataset-dir dataset/rplan/dataset_hypergraph_la0_fixed/test \
    --hub-subfolder qwen_hypergraphformer/checkpoint-8700 \
    --noged --shuffle \
    --vllm-prompt-batch-size 8
```


### 4. Procedural Editing and Optimization

After inference, generated hypergraphs can be (a) **procedurally edited** with
high-level room commands and (b) **parametrically optimized** to refine room
geometry. Both operate on a `*_hypergraphs.json` file produced by
`evaluator.py` (a `sample_id → hypergraph` mapping).

#### Procedural editing

`scripts/procedural_edits.py` applies a natural-language-style command to one
sample or all samples. Supported commands: `increase <room>`, `reduce <room>`,
`add <new_room> by <anchor_room>`, `delete <room>`, `swap <room1> and <room2>`,
`reorient <deg_ccw> <reflect_mode>`.

```bash
# Edit a single sample (writes s-r-0009_increase_kitchen.json beside the input)
python scripts/procedural_edits.py \
    --hypergraphs-file results_test_hypergraphs.json \
    --sample-id s-r-0009 \
    --command "increase kitchen" \
    --increase-factor 2.0 \
    --test-dir dataset/wmr24/test
```

To apply a command to all samples at once, use `--all-samples`:

```bash
python scripts/procedural_edits.py \
    --hypergraphs-file results_test_hypergraphs.json \
    --all-samples \
    --command "increase kitchen" \
    --increase-factor 2.0 \
    --test-dir dataset/wmr24/test
```

#### Automatic post-processing (bulk add/remove + rotate)

`scripts/procedural_edits.py` requires you to name each command. To fix up a
**whole results file automatically**, use `scripts/postprocessing.py`: it
inspects each predicted hypergraph, decides *which* rooms to add or remove (and
how to reorient), and applies the edits in bulk. It has two modes via
`--pp-mode`:

| `--pp-mode` | What it does |
|---|---|
| `addremove` (default) | Compares each prediction's room multiset against the GT access graph, greedily **plans an add/remove chain** (which room type is missing/extra), and — with `--apply-procedural-step1` — applies it through `procedural_edits`. The `add` anchor is chosen from GT-neighbor room types automatically. |
| `orientation` | Bulk **rotate/reflect** search: for every sample tries all 3 mirrors × 4 rotations (90° steps) and keeps the orientation with the lowest δ_sq compactness, subject to the access-graph edge check. |

For gradient-descent split optimization over all samples, use
`run_parametric.py --all` ([Parametric optimization](#parametric-optimization) below).

```bash
RESULTS=path/to/results_test_hypergraphs.json
TEST_DIR=dataset/rplan/eval_access/test

# (a) Plan add/remove for every sample WITHOUT editing — just writes the plan
python scripts/postprocessing.py \
    --results-hypergraphs $RESULTS --test-dir $TEST_DIR \
    --output-json parametric_commands.json
# parametric_commands.json holds, per id, the chosen command + multiset_command_chain
# (the missing/extra room diff that drove the decision).

# (b) Plan AND apply the add/remove chain in bulk (procedural edits)
python scripts/postprocessing.py \
    --results-hypergraphs $RESULTS --test-dir $TEST_DIR \
    --apply-procedural-step1 \
    --procedural-steps 5 \
    --increase-factor 2.0 \
    --workers 16 \
    --output-step1-hypergraphs results_hypergraphs_step1_procedural.json

# (c) Bulk rotate/reflect: pick the best orientation per sample
python scripts/postprocessing.py --pp-mode orientation \
    --results-hypergraphs $RESULTS --test-dir $TEST_DIR \
    --workers 16 \
    --output-orientation results_hypergraphs_orientation.json
```

`--procedural-steps N` caps how many add/remove edits to plan and apply per
sample (`0` = greedily continue until the room multiset matches GT). Use
`--max-samples` to dry-run on a handful of ids, and `-q/--quiet` to reduce
logging. Each mode writes a `sample_id → hypergraph` JSON in the same format as
the inference results, so the output can be fed straight into
`score_results.py` (next section) or back into another post-processing pass.
For gradient-descent split optimization over all samples, use
`run_parametric.py --all` (below).

#### Parametric optimization

`scripts/run_parametric.py` refines the BSP area-split fractions of a generated
hypergraph via projected gradient descent against ground-truth geometry. Three
objectives are available (lower is better):

| `--objective` | Optimizes |
|---|---|
| `delta` | δ_sq — per-leaf perimeter-vs-square compactness |
| `epsilon` | ε — room-area proportions vs reference fractions |
| `delta_epsilon` | equally-weighted blend of the two (default) |

Run it as a module from the repo root. Pass an apartment id for a single sample,
or `--all` for batch mode over every id in the results file:

```bash
# Single apartment
python -m scripts.run_parametric b-0001 \
    --results-hypergraphs path/to/results_test_hypergraphs.json \
    --test-dir dataset/wmr24/test \
    --objective delta_epsilon --maxiter 80

# Batch over all samples (parallel), writing a merged optimized JSON
python -m scripts.run_parametric --all \
    --results-hypergraphs path/to/results_test_hypergraphs.json \
    --test-dir dataset/wmr24/test \
    --objective delta_epsilon \
    --batch-workers 64 --gd-lr 0.15 --maxiter 200 \
    --batch-output-dir INFERENCE_RESULTS/<EXPERIMENT>/optimize_delta_epsilon
```

Key flags: `--maxiter` (GD steps), `--gd-lr` (step size), `--patience` /
`--ftol` (early-stop), `--limit` (cap number of samples), `--no-accessgraph`
(disable the access-graph edge penalty in the objective). In `--all` mode the
merged, optimized hypergraphs are written to
`<batch-output-dir>/<input_stem>_optimized.json` (override with
`--merged-hypergraphs-json`), alongside a per-sample metrics CSV and a
`metric_progress.csv` convergence trace. The optimized JSON can then be scored
with `score_results.py` (next section).

### 5. Score saved results (post-inference)

Compute any combination of GED, accuracy, delta, and area proportion error
metrics on saved `*_hypergraphs.json` files:

```bash
# All metrics at once
python scripts/score_results.py \
    --results-file path/to/results_test_hypergraphs.json \
    --test-dir dataset/hypergraph/test \
    --output-dir eval_output \
    --metrics ged accuracy delta area_proportion_error

# Just GED (access-graph correction applied by default)
python scripts/score_results.py \
    --results-file path/to/results_test_hypergraphs.json \
    --test-dir dataset/hypergraph/test \
    --output-dir eval_output \
    --metrics ged
```

### 6. Benchmark against baselines (HouseGAN++, House-Diffusion, DiffPlanner, iPlan)

See [`benchmark/README.md`](benchmark/README.md) for setting up and evaluating
the HouseGAN++, House-Diffusion, DiffPlanner, and iPlan baselines on the WMR24
dataset.

### 7. Visualize results

`utils/visualize_results.py` renders a single sample as two PNGs: a blueprint-style
floor-plan **mesh** (walls, interior door gaps, entry door, apartment boundary) and
its **access graph** (room types only, labels above nodes). It has three data modes —
pick one flag:

```bash
# Generated apartments (default): mesh + corrected access graph from the
# generated hypergraph. Bounds are read from <test-dir>/<id>.txt.
python -m utils.visualize_results --generated_data --sample-id 19764 \
    --hypergraph .../results_hypergraphs.json \
    --test-dir dataset/rplan/dataset_hypergraph_la0_fixed/test \
    --output-dir /tmp/viz

# RPLAN ground truth: mesh built from room_polygons, access graph from the
# stored "accessgraph" key (needs a test dir that has room_polygons).
python -m utils.visualize_results --gt_rplan --sample-id 19764 \
    --test-dir dataset/rplan/eval_access_mid/test --output-dir /tmp/viz

# WMR24 ground truth: mesh + corrected access graph from the embedded hypergraph.
python -m utils.visualize_results --gt_wmr24 --sample-id b-0001 \
    --test-dir dataset/wmr24/test --output-dir /tmp/viz
```

Outputs are written as `<id>_mesh.png` and `<id>_accessgraph.png` in `--output-dir`.
The module is also importable: `render_mesh`, `render_accessgraph`, and the
`visualize_*` mode functions accept a `save_path` — pass a path to write a PNG, or
`None` to get the live matplotlib `Figure` back for embedding in other scripts.

---

## Additional Workflows

Miscellaneous routines that complement the main workflow but are not part of the standard training/evaluation pipeline.

### Training on a smaller subset of RPLAN

To train on a smaller subset (e.g. ablations by dataset size), create a
downsampled partition from the full `sft/` directory:

```bash
python scripts/partition_dataset.py -n 1000 \
    --input-dir dataset/rplan/dataset_hypergraph_la0_fixed/sft
# → writes sft_1000/ alongside sft/ with subsampled train.jsonl
#   and unchanged validation.jsonl / test.jsonl
```



### Predict access-graph edges for an input room set

`scripts/qwen_predict_accessgraph_edges.py` uses the fine-tuned LoRA adapter to
predict `accessgraph.edges` given only the room-type node list in each
`<eval-test-dir>/<id>.txt`. It writes updated `.txt` files (identical to the
input, with only the edges replaced) to `--output-dir`, which can then be used
as a custom eval set.

```bash
VLLM_USE_V1=0 torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/qwen_predict_accessgraph_edges.py \
    --eval-test-dir dataset/wmr24/test \
    --output-dir dataset/wmr24/test_predicted_edges \
    --checkpoint checkpoints/qwen_hypergraphformer/checkpoint-8700
```

### Evaluate the base model in few-shot mode (no fine-tuning)

```bash
# Qwen few-shot (single GPU)
VLLM_USE_V1=0 python scripts/evaluator.py --mode fewshot-qwen \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --dataset-dir dataset/wmr24/test \
    --output outputs/results_fewshot_qwen_wmr24.json

# GPT few-shot (OpenAI API) — requires OPENAI_API_KEY to be set
export OPENAI_API_KEY=sk-...
python scripts/evaluator.py --mode fewshot-gpt \
    --model gpt-4-turbo \
    --dataset-dir dataset/wmr24/test \
    --output outputs/results_fewshot_gpt_wmr24.json
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

---

## Shared Modules

The `utils/` package provides shared helpers used by the pipeline scripts:

| Module | Contents |
|---|---|
| `utils/hypergraph.py` | `Hypergraph` class (validation, transformers, RGL integration, parsing, access-graph component analysis), `HypergraphPlotter` (mesh sliver filtering, `unique_id`-based face labelling) |
| `utils/utils.py` | `validate_hypergraph`, `TimeoutException`, `time_limit`, JSON extraction helpers |
| `utils/metrics.py` | `accessgraph_dict_to_nx`, `ged_between_graphs`, `compute_ged_from_hypergraph`, `compute_room_similarity_score`, `compute_room_area_proportion_error`, `room_type_accuracy`, `compute_room_type_accuracy`, `produce_room_list` |
| `utils/report_results.py` | `report_results` — unified NPZ + TXT reporting for all evaluation pipelines |
| `utils/analyze_metrics_by_room_count.py` | CLI script to bin NPZ metrics by room count and produce per-bin statistics |
| `utils/prompts.py` | BSP rules text, prompt templates (`create_structured_prompt`, `create_simple_prompt`, etc.) |
| `utils/data_loading.py` | `load_example_hypergraphs`, `load_target_from_dataset`, `save_training_data`, `log_slow_generation`, `HypergraphDataset`, `collate_fn` |
| `utils/visualize_results.py` | `render_mesh`, `render_accessgraph`, `render_mesh_from_polygons`, `render_accessgraph_from_dict` — blueprint-style floor-plan and access-graph renderer; CLI supports `--generated_data`, `--gt_rplan`, `--gt_wmr24` modes |
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

### Batched LoRA inference produces garbled / invalid JSON outputs

vLLM 0.8.x introduced a V1 engine that has a known bug with batched LoRA
generation — single-sample inference works correctly but batches of ≥16
prompts produce corrupt outputs. The fix is to force the stable V0 engine:

```bash
VLLM_USE_V1=0 torchrun ...
```

All evaluation commands in this README already include this flag. If you
observe high JSON extraction failure rates (>5%) with a LoRA checkpoint,
check that `VLLM_USE_V1=0` is set.

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
  title     = {HypergraphFormer: Learning Hypergraphs from LLMs for Editable Floor Plan Generation},
  author    = {Anonymous},
  booktitle = {Submitted to the Conference on Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  note      = {Under review. Preprint available on arXiv:2605.18932.}
}
```

---

**Project**: HypergraphFormer
**Last Updated**: 2026-06-25
**Python**: 3.9+
