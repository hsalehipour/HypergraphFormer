"""
Shared data loading and I/O utilities for hypergraph scripts.

This module centralizes functions that were previously duplicated across
generation, evaluation, and training-data preparation scripts:
- load_example_hypergraphs: load few-shot examples from a directory
- load_target_from_dataset: pick a target access graph
- save_training_data: write JSONL training files
- log_slow_generation: append slow-generation diagnostics to a log file
- HypergraphDataset / collate_fn: PyTorch Dataset for distributed evaluation
"""

import json
import random
import logging
from pathlib import Path
from datetime import datetime

from torch.utils.data import Dataset

from utils.prompts import create_training_sample

logger = logging.getLogger(__name__)


def load_example_hypergraphs(examples_dir, num_examples=3, exclude_file=None, seed=None):
    """
    Load example hypergraphs from training data.
    Files contain JSON with 'accessgraph' and 'hypergraph' fields.

    Args:
        examples_dir: Directory containing training examples
        num_examples: Number of examples to load
        exclude_file: Optional filename to exclude (e.g., when it's the target)
        seed: Optional random seed for deterministic selection. When provided,
              files are sorted by name and sampled with a seeded RNG.
              When None, uses the global random state.
    """
    examples_dir = Path(examples_dir)
    example_files = list(examples_dir.glob("*.txt"))

    if not example_files:
        raise FileNotFoundError(f"No example files found in {examples_dir}")

    if exclude_file:
        example_files = [f for f in example_files if f.name != exclude_file]
        logger.info(f"Excluding target file '{exclude_file}' from training examples")

    if not example_files:
        raise ValueError("No valid example files available after exclusion")

    if seed is not None:
        example_files.sort(key=lambda x: x.name)
        rng = random.Random(seed)
        selected_files = rng.sample(example_files, min(num_examples, len(example_files)))
    else:
        selected_files = random.sample(example_files, min(num_examples, len(example_files)))

    examples = []
    for file_path in selected_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            accessgraph = data.get('accessgraph', {})
            hypergraph_str = data.get('hypergraph', '')
            hypergraph = json.loads(hypergraph_str) if isinstance(hypergraph_str, str) else hypergraph_str

            examples.append({
                'filename': file_path.name,
                'accessgraph': accessgraph,
                'hypergraph': hypergraph
            })
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Skipping {file_path.name}: {e}")
            continue

    if not examples:
        raise ValueError("No valid examples could be loaded")

    return examples


def load_target_from_dataset(examples_dir, target_file=None):
    """
    Load a target accessgraph from the dataset.

    Args:
        examples_dir: Directory containing training examples
        target_file: Optional specific file to use as target. If None, selects random file.

    Returns:
        tuple: (target_accessgraph, target_filename, ground_truth_hypergraph)
    """
    examples_dir = Path(examples_dir)
    example_files = list(examples_dir.glob("*.txt"))

    if not example_files:
        raise FileNotFoundError(f"No example files found in {examples_dir}")

    if target_file:
        target_path = examples_dir / target_file
        if not target_path.exists():
            raise FileNotFoundError(f"Target file not found: {target_path}")
    else:
        target_path = random.choice(example_files)

    try:
        with open(target_path, 'r') as f:
            data = json.load(f)

        accessgraph = data.get('accessgraph', {})
        hypergraph_str = data.get('hypergraph', '')
        hypergraph = json.loads(hypergraph_str) if isinstance(hypergraph_str, str) else hypergraph_str

        logger.info(f"Selected target: {target_path.name}")
        logger.info(f"   Nodes: {len(accessgraph.get('nodes', {}))} rooms")
        logger.info(f"   Edges: {len(accessgraph.get('edges', []))} connections")

        return accessgraph, target_path.name, hypergraph

    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Failed to load target from {target_path.name}: {e}")


def save_training_data(samples, output_path, dataset_type="train"):
    """
    Save training data in JSONL format.
    Each line is a JSON object with 'messages' field.

    Args:
        samples: List of (identifier, accessgraph, hypergraph) tuples.
                 The first element (identifier) can be a filename or item id;
                 it is not written to the output.
        output_path: Path to output .jsonl file
        dataset_type: Type of dataset (train/validation/test) — used only for logging
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for sample in samples:
            if len(sample) != 3:
                raise ValueError(f"Unexpected sample tuple length: {len(sample)}")
            _identifier, accessgraph, hypergraph = sample
            training_sample = create_training_sample(accessgraph, hypergraph)
            f.write(json.dumps(training_sample) + '\n')

    logger.info(f"Saved {len(samples)} {dataset_type} samples to {output_path}")


class HypergraphDataset(Dataset):
    """
    PyTorch Dataset for loading hypergraph samples from JSON files.

    Args:
        file_paths: List of Path objects pointing to .txt JSON files
        include_ground_truth: If True, each item includes 'ground_truth' from the
                              file's 'hypergraph' field when present; otherwise
                              ``None`` (e.g. access-only eval files).
                              Set to False to omit the key (rooms-only).
        include_bounds: If True, each item includes 'bounds' parsed from the
                        file's 'bounds' field (apartment footprint).
    """

    def __init__(self, file_paths, include_ground_truth=True):
        self.file_paths = file_paths
        self.include_ground_truth = include_ground_truth

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        with open(file_path, 'r') as f:
            data = json.load(f)

        accessgraph = data.get('accessgraph', {})

        item = {
            'filename': file_path.name,
            'accessgraph': accessgraph,
            'num_nodes': len(accessgraph.get('nodes', {})),
            'num_edges': len(accessgraph.get('edges', [])),
        }

        if self.include_ground_truth:
            h = data.get("hypergraph")
            if h is None or (isinstance(h, str) and not h.strip()):
                item["ground_truth"] = None
            else:
                item["ground_truth"] = json.loads(h) if isinstance(h, str) else h

        return item


def collate_fn(batch):
    """Custom collate that returns batch items as lists without tensor stacking.

    Dynamically collates whatever keys are present, so it works for both
    standard (with ground_truth) and rooms-only (without) datasets.
    """
    keys = batch[0].keys()
    return {key: [item[key] for item in batch] for key in keys}


def log_slow_generation(filename, generation_time, num_nodes, num_edges,
                        num_input_tokens, threshold,
                        log_filename="slow_generations.log",
                        header_title="SLOW GENERATION LOG"):
    """
    Log information about slow generations to a dedicated log file.

    Args:
        filename: Name of the file being evaluated
        generation_time: Time taken to generate (seconds)
        num_nodes: Number of nodes in the access graph
        num_edges: Number of edges in the access graph
        num_input_tokens: Number of input tokens in the prompt
        threshold: Threshold used to determine "slow" (seconds)
        log_filename: Name of the log file (default: slow_generations.log)
        header_title: Title written in the log file header
    """
    log_file = Path(log_filename)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not log_file.exists():
        with open(log_file, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write(header_title + "\n")
            f.write(f"Threshold: {threshold}s\n")
            f.write("=" * 100 + "\n\n")

    with open(log_file, 'a') as f:
        f.write(f"[{timestamp}] SLOW GENERATION DETECTED\n")
        f.write(f"  Filename: {filename}\n")
        f.write(f"  Generation Time: {generation_time:.2f}s ({generation_time/threshold:.1f}x threshold)\n")
        f.write(f"  Graph Size: {num_nodes} nodes, {num_edges} edges\n")
        f.write(f"  Input Tokens: {num_input_tokens}\n")
        f.write("-" * 100 + "\n\n")
