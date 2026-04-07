"""
GRPO (Reinforcement Learning) Fine-tuning Pipeline for Hypergraph Generation.


Usage:
    # Single GPU training:
    python scripts/rlft_pipeline.py --gpu 0

    # Multi-GPU with accelerate (data parallelism):
    accelerate launch --num_processes=2 scripts/rlft_pipeline.py --multi-gpu

    # Token-length filtering only (no training):
    python scripts/rlft_pipeline.py --reduce-tokens --max-tokens 2000
"""

import os
import json
import argparse
import logging
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import TrainerCallback, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import GRPOTrainer, GRPOConfig

from utils import compute_ged_from_hypergraph, compute_room_similarity_score, produce_room_list
from utils.hypergraph import Hypergraph

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ===========================================================================
# Configuration
# ===========================================================================
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
LORA_ADAPTER_PATH = "checkpoints/qwen_hypergraph_filtered/checkpoint-300"

DATA_DIR = _PROJECT_ROOT / "dataset/hypergraph"
RL_DATA_DIR = DATA_DIR / "rl"
OUTPUT_DIR = _PROJECT_ROOT / "checkpoints/retrain_grpo_hypergraph_2"
DEBUG_DIR = _PROJECT_ROOT / "checkpoints/retrain_grpo_hypergraph_2/debug_rewards"


# ===========================================================================
# Boundary-mesh helper
# ===========================================================================

def sample_to_boundary_mesh(sample):
    """Convert a dataset sample dict to an RGL boundary mesh.

    Args:
        sample: dict with a ``'bounds'`` field (JSON string) containing
                a ``'corners'`` list of ``{X, Y}`` dicts.

    Returns:
        ``(boundary_mesh, boundary_face, polygon_array)``
    """
    import numpy as np

    RGL, Collections = Hypergraph.init_rgl()

    bounds = json.loads(sample["bounds"])
    corners = bounds["corners"]
    polygon = [(c["X"], c["Y"]) for c in corners]
    polygon_array = np.array(polygon)

    boundary_mesh = Hypergraph._polygon_to_boundary_mesh(polygon, RGL, Collections)

    return boundary_mesh, polygon_array


# ===========================================================================
# Reward module
# ===========================================================================

GROUND_TRUTH_TRAIN = None
GROUND_TRUTH_VALIDATION = None
_REWARD_DEBUG_DIR = None
_RGL = None

_debug_counter = 0
_current_iter = 0


def init_reward_module(ground_truth_train, ground_truth_validation, debug_dir, rgl_module):
    """Initialise the reward module with ground-truth data and RGL."""
    global GROUND_TRUTH_TRAIN, GROUND_TRUTH_VALIDATION, _REWARD_DEBUG_DIR, _RGL
    GROUND_TRUTH_TRAIN = ground_truth_train
    GROUND_TRUTH_VALIDATION = ground_truth_validation
    _REWARD_DEBUG_DIR = Path(debug_dir)
    _RGL = rgl_module
    logger.info(
        f"Reward module initialised with {len(ground_truth_train)} train, "
        f"{len(ground_truth_validation)} validation samples"
    )


def _save_reward_debug(apt_id, hypergraph, polygon, reward, scores,
                       error=None, is_validation=False, raw_completion=None):
    """Persist per-sample reward debug data to JSON."""
    global _debug_counter, _current_iter
    _debug_counter += 1

    rank = os.environ.get("LOCAL_RANK", "0")
    iter_dir = _REWARD_DEBUG_DIR / f"iter_{_current_iter:04d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    split = "val" if is_validation else "train"

    debug_data = {
        "debug_id": _debug_counter,
        "iter": _current_iter,
        "rank": int(rank),
        "split": split,
        "apt_id": apt_id,
        "reward": reward,
        "delta_metric": scores.get("delta_metric") if scores else None,
        "ged": scores.get("ged") if scores else None,
        "scores": scores,
        "error": error,
        "polygon": polygon,
        "hypergraph": hypergraph,
        "raw_completion": raw_completion,
    }

    filename = iter_dir / f"gpu{rank}_{split}_{_debug_counter:06d}_{apt_id}.json"
    with open(filename, "w") as f:
        json.dump(debug_data, f, indent=2)


def create_accuracy_reward(mode="room_reward_delta_soft"):
    """Factory that returns a GRPO-compatible reward function.

    Supported *mode* values: ``'room_reward_delta_soft'``, ``'room_reward_delta'``.
    """
    if mode not in {"room_reward_delta_soft", "room_reward_delta"}:
        raise ValueError(
            f"Only modes 'room_reward_delta_soft' and 'room_reward_delta' "
            f"are supported (got {mode!r})"
        )

    def accuracy_reward(completions, prompts=None, ground_truth_idx=None,
                        is_train=None, **kwargs):
        global GROUND_TRUTH_TRAIN, GROUND_TRUTH_VALIDATION, _current_iter
        _current_iter += 1
        logger.info(f"🎯 Reward [{mode}] (iter {_current_iter}) with {len(completions)} completions")

        def _penalty(kind: str) -> float:
            if mode == "room_reward_delta":
                return {
                    "json": -0.3, "struct": -0.25, "mismatch": -0.25,
                    "room_reward_fail": -0.3, "unexpected": -0.3,
                }.get(kind, -0.3)
            return {
                "json": -0.6, "struct": -0.5, "mismatch": -0.25,
                "room_reward_fail": -0.2, "unexpected": -0.6,
            }.get(kind, -0.6)

        rewards = []
        for i, completion in enumerate(completions):
            reward = 0
            apt_id = "unknown"
            hypergraph = None
            polygon = None
            scores = None
            error_msg = None
            is_validation = False

            try:
                if ground_truth_idx is None:
                    error_msg = "ground_truth_idx is None"
                    reward = -0.6
                    rewards.append(reward)
                    continue

                idx = ground_truth_idx[i]
                is_validation = is_train is not None and not is_train[i]
                gt_data = GROUND_TRUTH_VALIDATION if is_validation else GROUND_TRUTH_TRAIN
                if gt_data is None:
                    error_msg = "gt_data is None"
                    reward = -0.6
                    rewards.append(reward)
                    continue

                gt_entry = gt_data[idx]
                apt_id = gt_entry.get("id", f"idx_{idx}")

                boundary_mesh, polygon_array = sample_to_boundary_mesh(gt_entry)
                polygon = polygon_array.tolist()

                try:
                    hypergraph = json.loads(completion)
                except json.JSONDecodeError as e:
                    error_msg = f"JSON parse error: {e}"
                    logger.info(f"❌ {apt_id}: {error_msg}")
                    reward = _penalty("json")
                    _save_reward_debug(apt_id, hypergraph, polygon, reward, None,
                                       error_msg, is_validation, raw_completion=completion)
                    rewards.append(reward)
                    continue

                if not isinstance(hypergraph, dict):
                    error_msg = f"Not a dict: {type(hypergraph).__name__}"
                    logger.info(f"❌ {apt_id}: {error_msg}")
                    reward = _penalty("json")
                    _save_reward_debug(apt_id, hypergraph, polygon, reward, None,
                                       error_msg, is_validation, raw_completion=completion)
                    rewards.append(reward)
                    continue

                struct_valid, _, issues, _ = Hypergraph(hypergraph).validate(quick=True)
                if not struct_valid:
                    error_msg = f"Structure: {issues[:2]}"
                    logger.info(f"❌ {apt_id}: {error_msg}")
                    reward = _penalty("struct")
                    _save_reward_debug(apt_id, hypergraph, polygon, reward, None,
                                       error_msg, is_validation, raw_completion=completion)
                    rewards.append(reward)
                    continue

                hypergraph = Hypergraph(hypergraph).with_ratio_areas().data

                # ----- room-list computation --------------------------------
                def compute_room_list_and_gt():
                    data_dir = _PROJECT_ROOT / "dataset" / "hypergraph"
                    if is_validation:
                        room_lists_path = data_dir / "validation_filtered_room_lists" / "all_room_lists.json"
                    else:
                        room_lists_path = data_dir / "train_filtered_room_lists" / "all_room_lists.json"

                    room_list_gt = None
                    if room_lists_path.exists():
                        try:
                            with open(room_lists_path, "r") as f:
                                all_room_lists = json.load(f)
                            for entry in all_room_lists:
                                if entry.get("id") == apt_id:
                                    room_list_gt = entry.get("room_list")
                                    break
                            if room_list_gt is None and idx < len(all_room_lists):
                                room_list_gt = all_room_lists[idx].get("room_list")
                        except Exception as e:
                            logger.warning(f"Failed to load room list GT for {apt_id}: {e}")

                    try:
                        data_node = _RGL.DataNode.deserializeDataNode(json.dumps(hypergraph))
                    except Exception as e:
                        raise Exception(f"JSON structure error in hypergraph deserialization: {e}")

                    generated_mesh = _RGL.DataNode.subdivideWithMeshRatio(data_node, boundary_mesh)
                    room_list = produce_room_list(generated_mesh)
                    return room_list, room_list_gt

                # ----- mode dispatch ----------------------------------------
                def run_room_reward_delta_soft():
                    try:
                        room_list, room_list_gt = compute_room_list_and_gt()
                        delta_metric_soft = compute_room_similarity_score(room_list, room_list_gt)

                        try:
                            gt_hg_raw = gt_entry.get("split")
                            gt_hypergraph = json.loads(gt_hg_raw) if isinstance(gt_hg_raw, str) else gt_hg_raw
                            if gt_hypergraph is None:
                                raise Exception("GT hypergraph is None")
                            gt_access_graph = Hypergraph(gt_hypergraph).access_graph
                        except Exception as e:
                            raise Exception(f"JSON structure error in access graph extraction: {e}")

                        ged, _ = compute_ged_from_hypergraph(gt_access_graph, hypergraph, apt_id)
                        return delta_metric_soft, room_list, room_list_gt, ged
                    except Exception as e:
                        raise Exception(f"Error in room_reward_delta_soft: {e}")

                def run_room_reward_delta():
                    try:
                        room_list, room_list_gt = compute_room_list_and_gt()
                        delta_metric_val = compute_room_similarity_score(room_list, room_list_gt, strict=True)
                        return delta_metric_val, room_list, room_list_gt
                    except Exception as e:
                        raise Exception(f"Error in room_reward_delta: {e}")

                try:
                    if mode == "room_reward_delta_soft":
                        result = run_room_reward_delta_soft()
                    else:
                        result = run_room_reward_delta()
                    err = None
                except Exception as e:
                    result = None
                    err = f"{type(e).__name__}: {e}"

                if err:
                    error_msg = f"Room reward {mode}: {err}"
                    logger.info(f"❌ {apt_id}: {error_msg}")
                    reward = _penalty("room_reward_fail")
                    _save_reward_debug(apt_id, hypergraph, polygon, reward, None,
                                       error_msg, is_validation, raw_completion=completion)
                    rewards.append(reward)
                    continue

                if mode == "room_reward_delta_soft":
                    delta_metric_soft, room_list, room_list_gt, ged = result

                    if ged is None or delta_metric_soft is None:
                        error_msg = f"GED computation timed out or failed for {apt_id}"
                        logger.info(f"❌ {apt_id}: {error_msg}")
                        mode_reward = -0.6
                        scores = {
                            "room_list": room_list,
                            "room_list_gt": room_list_gt,
                            "delta_metric": delta_metric_soft,
                            "ged": None,
                            "ged_failed": True,
                        }
                    else:
                        if room_list_gt is not None:
                            rooms_generated = [t[0] for t in room_list] if room_list else []
                            rooms_gt = [t[0] for t in room_list_gt] if room_list_gt else []
                            if rooms_generated != rooms_gt:
                                error_msg = f"Room list mismatch: generated={rooms_generated}, gt={rooms_gt}"
                                logger.info(f"⚠️  {apt_id}: {error_msg}")
                        else:
                            error_msg = "Room list GT not found"
                            logger.info(f"⚠️  {apt_id}: {error_msg}")

                        mode_reward = -(delta_metric_soft + ged)
                        scores = {
                            "room_list": room_list,
                            "room_list_gt": room_list_gt,
                            "delta_metric": delta_metric_soft,
                            "ged": ged,
                            "ged_failed": False,
                        }
                else:
                    delta_metric_val, room_list, room_list_gt = result
                    if room_list_gt is not None:
                        rooms_generated = [t[0] for t in room_list] if room_list else []
                        rooms_gt = [t[0] for t in room_list_gt] if room_list_gt else []
                        if rooms_generated != rooms_gt:
                            error_msg = f"Room list mismatch: generated={rooms_generated}, gt={rooms_gt}"
                            logger.info(f"⚠️  {apt_id}: {error_msg}")
                            reward = _penalty("mismatch")
                            _save_reward_debug(apt_id, hypergraph, polygon, reward, None,
                                               error_msg, is_validation, raw_completion=completion)
                            rewards.append(reward)
                            continue
                    mode_reward = -delta_metric_val
                    scores = {
                        "room_list": room_list,
                        "room_list_gt": room_list_gt,
                        "delta_metric": delta_metric_val,
                        "ged": None,
                        "ged_failed": True,
                    }

                reward += mode_reward

                if scores is not None:
                    dm = scores.get("delta_metric")
                    gv = scores.get("ged")
                    if scores.get("ged_failed", False) or gv is None:
                        if dm is None:
                            logger.info(f"✅ {apt_id}: reward={reward:.4f}")
                        else:
                            logger.info(f"✅ {apt_id}: reward={reward:.4f} | delta_metric={dm:.4f}")
                    else:
                        logger.info(f"✅ {apt_id}: reward={reward:.4f} | delta_metric={dm:.4f} | ged={gv}")

            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                logger.info(f"❌ {apt_id}: {error_msg}")
                reward = _penalty("unexpected")

            _save_reward_debug(apt_id, hypergraph, polygon, reward, scores,
                               error_msg, is_validation, raw_completion=completion)
            rewards.append(reward)

        return rewards

    return accuracy_reward


# ===========================================================================
# Data helpers
# ===========================================================================

def load_ground_truth_data():
    """Load ground truth JSON files for reward computation."""
    train_json_path = DATA_DIR / "train.json"
    val_json_path = DATA_DIR / "validation.json"

    with open(train_json_path, "r") as f:
        train_gt = json.load(f)
    with open(val_json_path, "r") as f:
        val_gt = json.load(f)

    logger.info(f"Loaded ground truth: {len(train_gt)} train, {len(val_gt)} validation")
    return train_gt, val_gt


def convert_messages_to_prompt(example, tokenizer, idx, is_train=True):
    """Convert chat-messages format to a prompt string for GRPO."""
    messages = example["messages"]
    prompt_messages = [msg for msg in messages if msg["role"] in ["system", "user"]]

    prompt = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    gt_idx = example.get("orig_idx", idx)
    return {"prompt": prompt, "ground_truth_idx": gt_idx, "is_train": is_train}


def filter_jsonl_by_token_length(split_name, *, data_dir, tokenizer, max_tokens):
    """Filter a JSONL split, keeping only samples where the assistant
    response is shorter than *max_tokens* tokens.

    Writes ``{split_name}_reduced.jsonl`` next to the source file.
    Returns the number of samples kept.

    Formerly the standalone ``reduce_dataset.py``.
    """
    input_path = Path(data_dir) / f"{split_name}.jsonl"
    logger.info(f"Filtering {input_path} (max {max_tokens} tokens)...")

    with open(input_path) as f:
        jsonl_data = [json.loads(line) for line in f]

    logger.info(f"Loaded {len(jsonl_data)} samples from {split_name}.jsonl")

    filtered = []
    token_counts = []

    for i, item in enumerate(jsonl_data):
        for msg in item["messages"]:
            if msg["role"] == "assistant":
                n = len(tokenizer.encode(msg["content"]))
                if n < max_tokens:
                    entry = {**item, "orig_idx": i}
                    filtered.append(entry)
                    token_counts.append(n)
                break

    output_path = Path(data_dir) / f"{split_name}_reduced.jsonl"
    with open(output_path, "w") as f:
        for item in filtered:
            f.write(json.dumps(item) + "\n")

    logger.info(
        f"{split_name}: {len(filtered)}/{len(jsonl_data)} kept -> {output_path}"
    )
    if token_counts:
        logger.info(
            f"  tokens: min={min(token_counts)}, max={max(token_counts)}, "
            f"avg={sum(token_counts)/len(token_counts):.0f}"
        )
    return len(filtered)


def run_reduce_tokens(args):
    """Entry point for --reduce-tokens mode."""
    data_dir = Path(getattr(args, "data_dir", RL_DATA_DIR))
    max_tokens = getattr(args, "max_tokens", 2000)
    tokenizer_path = getattr(args, "tokenizer", LORA_ADAPTER_PATH)

    logger.info("Loading tokenizer for token counting...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    train_n = filter_jsonl_by_token_length(
        "train", data_dir=data_dir, tokenizer=tokenizer, max_tokens=max_tokens
    )
    val_n = filter_jsonl_by_token_length(
        "validation", data_dir=data_dir, tokenizer=tokenizer, max_tokens=max_tokens
    )
    logger.info(f"Total reduced: {train_n + val_n} (train={train_n}, val={val_n})")


# ===========================================================================
# Training callback
# ===========================================================================

class MetricsLoggingCallback(TrainerCallback):
    """Callback to log training metrics."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            metrics_str = ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in logs.items()
            )
            logger.info(f"Step {state.global_step}: {metrics_str}")

    def on_train_begin(self, args, state, control, **kwargs):
        logger.info("Training started...")

    def on_train_end(self, args, state, control, **kwargs):
        logger.info(f"Training finished. Total steps: {state.global_step}")


# ===========================================================================
# Main
# ===========================================================================

def main(args):
    lora_path = str((_PROJECT_ROOT / LORA_ADAPTER_PATH).resolve())

    logger.info("=" * 80)
    logger.info("🚀 GRPO Fine-tuning for Hypergraph Generation")
    logger.info("=" * 80)
    logger.info(f"Mode: {'Multi-GPU (accelerate)' if args.multi_gpu else f'Single-GPU ({args.gpu})'}")
    logger.info(f"Base model: {BASE_MODEL}")
    logger.info(f"LoRA adapter: {lora_path}")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # Load ground truth data for reward computation
    train_gt, val_gt = load_ground_truth_data()

    # Initialise RGL for geometry operations
    logger.info("Initialising RGL...")
    from hypergraph.api.lib.tools import RGL
    import clr
    clr.AddReference("System.Collections")
    logger.info("✅ RGL initialised")

    # Initialise the reward module
    init_reward_module(train_gt, val_gt, DEBUG_DIR, RGL)
    accuracy_reward = create_accuracy_reward(mode="room_reward_delta")
    logger.info("✅ Reward module initialised")

    # Load tokenizer
    logger.info(f"Loading tokenizer from {lora_path}...")
    tokenizer = AutoTokenizer.from_pretrained(lora_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("✅ Tokenizer loaded")

    # Load base model
    logger.info(f"Loading base model {BASE_MODEL}...")
    if args.multi_gpu:
        logger.info("Multi-GPU mode: Loading model without device_map (accelerate handles placement)")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16,
        )
    else:
        logger.info(f"Single-GPU mode: Loading model to GPU {args.gpu}")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map={"": 0},
        )
    logger.info("✅ Base model loaded")

    # Load LoRA adapter
    logger.info(f"Loading LoRA adapter from {lora_path}...")
    model = PeftModel.from_pretrained(base_model, lora_path, is_trainable=True, local_files_only=True)
    logger.info("✅ LoRA adapter loaded")
    model.print_trainable_parameters()

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset("json", data_files={
        "train": str(RL_DATA_DIR / "train_reduced.jsonl"),
        "validation": str(RL_DATA_DIR / "validation_reduced.jsonl"),
    })

    if "orig_idx" not in dataset["train"].column_names or "orig_idx" not in dataset["validation"].column_names:
        raise RuntimeError(
            "Reduced jsonl files must include 'orig_idx' for correct ground-truth alignment. "
            "Re-run: python scripts/rlft_pipeline.py --reduce-tokens"
        )

    logger.info(f"Loaded {len(dataset['train'])} train samples")
    logger.info(f"Loaded {len(dataset['validation'])} validation samples")

    # Convert to GRPO format
    logger.info("Converting dataset to GRPO format...")
    train_data = dataset["train"].map(
        lambda x, idx: convert_messages_to_prompt(x, tokenizer, idx, is_train=True),
        with_indices=True, desc="Converting train to prompts",
    )
    val_data = dataset["validation"].map(
        lambda x, idx: convert_messages_to_prompt(x, tokenizer, idx, is_train=False),
        with_indices=True, desc="Converting validation to prompts",
    )

    eval_dataset = val_data.select(range(min(4, len(val_data))))
    logger.info(f"Using {len(eval_dataset)} samples for evaluation")

    batch_size = 1

    config = GRPOConfig(
        output_dir=str(OUTPUT_DIR),
        logging_steps=1,
        logging_dir=str(OUTPUT_DIR / "logs"),
        report_to=["tensorboard"],
        log_completions=False,
        num_generations=4,
        max_completion_length=3600,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        epsilon=0.3,
        use_vllm=not args.no_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.25,
        vllm_max_model_length=6900,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        num_train_epochs=20,
        learning_rate=5e-6,
        gradient_checkpointing=True,
        bf16=True,
        eval_strategy="steps",
        eval_steps=10,
        per_device_eval_batch_size=batch_size,
        save_steps=100,
        save_total_limit=30,
    )

    logger.info("=" * 80)
    logger.info("📋 Training Configuration:")
    logger.info(f"   mode: {'multi-gpu' if args.multi_gpu else 'single-gpu'}")
    logger.info(f"   vLLM: {'enabled ⚡' if not args.no_vllm else 'disabled'}")
    logger.info(f"   learning_rate: {config.learning_rate}")
    logger.info(f"   num_generations: {config.num_generations}")
    logger.info(f"   per_device_train_batch_size: {config.per_device_train_batch_size}")
    logger.info(f"   max_completion_length: {config.max_completion_length}")
    logger.info(f"   temperature: {config.temperature}")
    if args.multi_gpu:
        logger.info(f"   global_batch_size: {batch_size} × num_gpus")
    logger.info("=" * 80)

    logger.info("Initialising GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=accuracy_reward,
        train_dataset=train_data,
        eval_dataset=eval_dataset,
        callbacks=[MetricsLoggingCallback()],
    )

    logger.info("Starting GRPO training...")
    trainer.train()
    logger.info("✅ Training complete!")

    logger.info("Saving final model...")
    trainer.save_model(str(OUTPUT_DIR / "final"))
    logger.info("✅ Model saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO Fine-tuning for Hypergraph Generation")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID for single-GPU training (default: 0)")
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Use multi-GPU training with accelerate (data parallelism)")
    parser.add_argument("--no-vllm", action="store_true",
                        help="Disable vLLM for generation (use standard HF generate)")

    # Token-length filtering (formerly reduce_dataset.py)
    parser.add_argument("--reduce-tokens", action="store_true",
                        help="Only run token-length filtering on JSONL (no training)")
    parser.add_argument("--max-tokens", type=int, default=2000,
                        help="Max assistant tokens threshold for --reduce-tokens (default: 2000)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="JSONL directory for --reduce-tokens (default: dataset/hypergraph/rl)")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer path for --reduce-tokens (default: LoRA checkpoint)")

    args = parser.parse_args()

    if args.reduce_tokens:
        if args.data_dir is None:
            args.data_dir = str(RL_DATA_DIR)
        if args.tokenizer is None:
            args.tokenizer = LORA_ADAPTER_PATH
        run_reduce_tokens(args)
    else:
        if not args.multi_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            logger.info(f"Single-GPU mode: Setting CUDA_VISIBLE_DEVICES={args.gpu}")
        main(args)
