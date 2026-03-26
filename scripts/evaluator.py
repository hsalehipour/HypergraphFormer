#!/usr/bin/env python3
"""
Unified evaluator for hypergraph generation models.

Modes:
    trained      -- Fine-tuned Qwen, zero-shot, GED + validation (distributed)
    rooms-only   -- Fine-tuned Qwen, rooms-only prompt, room matching (distributed)
    fewshot-qwen -- Base Qwen, few-shot prompting, GED (single GPU)
    fewshot-gpt  -- GPT via OpenAI API, few-shot prompting, GED (single process)

Usage:
    # Fine-tuned model (distributed)
    accelerate launch --num_processes=8 scripts/evaluator.py --mode trained \\
        --model checkpoints/lora_r64_alpha128

    # Rooms-only (distributed)
    accelerate launch --num_processes=8 scripts/evaluator.py --mode rooms-only \\
        --model checkpoints/lora_r64_alpha128

    # Few-shot Qwen (single GPU)
    python scripts/evaluator.py --mode fewshot-qwen --dataset-dir data/test

    # Few-shot GPT
    python scripts/evaluator.py --mode fewshot-gpt --model gpt-4-turbo \\
        --dataset-dir data/test
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path

import torch
from tqdm import tqdm

from utils.hypergraph import Hypergraph
from scripts.fewshot_generator import QwenGenerator, GPTGenerator

logger = logging.getLogger(__name__)


# ===================================================================
# Base class
# ===================================================================

class Evaluator(ABC):
    """Abstract base for all evaluation modes."""

    distributed: bool = False  # subclasses override to enable Accelerate

    def __init__(self, args):
        self.args = args
        self.generator = None  # set by subclass

    # -- hooks that subclasses override ------------------------------------

    @abstractmethod
    def create_prompt(self, sample: dict) -> str: ...

    @abstractmethod
    def extract_hypergraph(self, response: str) -> Hypergraph | None: ...

    @abstractmethod
    def score_result(self, result: dict, hg: Hypergraph, sample: dict) -> None:
        """Mutate *result* in-place with scoring fields (GED, room matching, etc.)."""

    @abstractmethod
    def compute_statistics(self, results: list[dict]) -> dict:
        """Return a dict with mode-specific aggregate statistics."""

    def load_model(self):
        """Load model / init client. Called once before the evaluation loop."""

    def build_result_template(self, sample: dict) -> dict:
        """Return the initial result dict for a sample."""
        return {
            "filename": sample["filename"],
            "success": False,
            "error": None,
            "num_nodes": sample.get("num_nodes"),
            "num_edges": sample.get("num_edges"),
            "num_input_tokens": None,
            "generation_time": None,
            "validation": {"is_valid": False, "passed_checks": [], "failed_checks": []},
        }

    # -- template method: evaluate one sample ------------------------------

    def evaluate_sample(self, sample: dict) -> dict:
        result = self.build_result_template(sample)
        try:
            prompt = self.create_prompt(sample)

            timeout = getattr(self.args, "timeout", None)
            res = self.generator.generate(
                prompt,
                max_tokens=self.args.max_tokens,
                temperature=self.args.temperature,
                timeout=timeout,
            )
            result["num_input_tokens"] = res.input_tokens
            result["generation_time"] = res.generation_time

            if res.timed_out:
                result["error"] = f"Generation timed out after {timeout}s"
                result["timed_out"] = True
                self._log_slow(sample, res)
                return result

            if res.generation_time > getattr(self.args, "slow_gen_threshold", 1e9):
                self._log_slow(sample, res)

            hg = self.extract_hypergraph(res.text)

            if hg is None:
                logger.warning(f"JSON extraction failed for {sample['filename']}, retrying...")
                res2 = self.generator.generate(
                    prompt,
                    max_tokens=self.args.max_tokens,
                    temperature=self.args.temperature,
                    timeout=timeout,
                )
                result["generation_time"] += res2.generation_time
                if res2.timed_out:
                    result["error"] = f"Generation timed out after {timeout}s (retry)"
                    result["timed_out"] = True
                    return result
                hg = self.extract_hypergraph(res2.text)
                if hg is None:
                    result["error"] = "Failed to extract JSON from response (after retry)"
                    return result
                result["retried"] = True

            is_valid, passed, failed, ged_info = hg.validate(accessgraph=None, debug=True)
            result["validation"] = {
                "is_valid": is_valid,
                "passed_checks": passed,
                "failed_checks": failed,
            }
            result["_ged_info"] = ged_info

            self.score_result(result, hg, sample)

        except Exception as e:
            result["error"] = str(e)
        return result

    # -- run loop ----------------------------------------------------------

    def run(self) -> int:
        args = self.args

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        # Dataset files
        dataset_dir = Path(args.dataset_dir)
        all_files = sorted(dataset_dir.glob("*.txt"), key=lambda p: p.name)
        if not all_files:
            logger.error(f"No files found in {dataset_dir}")
            return 1

        # Max-rooms filter
        if getattr(args, "max_rooms", None):
            all_files = self._filter_max_rooms(all_files, args.max_rooms)

        if args.max_samples:
            all_files = all_files[: args.max_samples]
            logger.info(f"Limited to {args.max_samples} samples")

        # Checkpoint resume
        existing_results, completed = self._load_checkpoint(args)
        files_to_process = [f for f in all_files if f.name not in completed]

        if not files_to_process:
            logger.info("All files already evaluated!")
            return 0

        logger.info(f"Processing {len(files_to_process)} remaining files")

        self.load_model()

        if self.distributed:
            return self._run_distributed(args, files_to_process, existing_results, all_files)
        return self._run_single(args, files_to_process, existing_results, all_files)

    # -- single-process loop -----------------------------------------------

    def _run_single(self, args, files_to_process, existing_results, all_files):
        results = list(existing_results)
        start = time.time()

        for i, fpath in enumerate(tqdm(files_to_process, desc="Evaluating"), 1):
            sample = self._load_sample(fpath)
            result = self.evaluate_sample(sample)
            results.append(result)

            if result["success"]:
                ged = result.get("ged")
                msg = f"GED: {ged:.2f}" if ged is not None else "OK"
                logger.info(f"  [{i}/{len(files_to_process)}] {fpath.name} -- {msg}")
            else:
                logger.info(f"  [{i}/{len(files_to_process)}] {fpath.name} -- FAILED: {result.get('error', '')[:80]}")

            if i % 5 == 0:
                self._save_checkpoint(args, results, all_files)

        elapsed = time.time() - start
        logger.info(f"Total time: {elapsed / 60:.1f} min")

        self._save_final(args, results)
        return 0

    # -- distributed loop (Accelerate) -------------------------------------

    def _run_distributed(self, args, files_to_process, existing_results, all_files):
        from accelerate import Accelerator, InitProcessGroupKwargs
        from torch.utils.data import DataLoader
        from utils.data_loading import HypergraphDataset, collate_fn

        nccl_timeout = getattr(args, "nccl_timeout", 7200)
        ddp_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=nccl_timeout))
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

        if accelerator.is_main_process:
            logger.info(f"Distributed evaluation with {accelerator.num_processes} GPUs")

        include_gt = not isinstance(self, RoomsOnlyEvaluator)
        include_bounds = getattr(args, "footprint", False)
        dataset = HypergraphDataset(
            files_to_process,
            include_ground_truth=include_gt,
            include_bounds=include_bounds,
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=getattr(args, "shuffle", False),
                                num_workers=0, collate_fn=collate_fn)

        model = accelerator.prepare(self._raw_model)
        self._update_generator_model(model)
        dataloader = accelerator.prepare(dataloader)

        start = time.time()
        local_results = []

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating",
                                               disable=not accelerator.is_main_process)):
            sample = self._batch_to_sample(batch, include_gt, include_bounds)
            result = self.evaluate_sample(sample)
            local_results.append(result)

        # Save per-rank results to temp files
        tmp = Path(tempfile.gettempdir())
        rank_file = tmp / f"eval_results_rank_{accelerator.process_index}.json"
        with open(rank_file, "w") as f:
            json.dump(local_results, f)
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            gathered = []
            for rank in range(accelerator.num_processes):
                rf = tmp / f"eval_results_rank_{rank}.json"
                if rf.exists():
                    with open(rf) as f:
                        gathered.extend(json.load(f))
                    rf.unlink()

            seen = set()
            unique = []
            for r in gathered:
                if r["filename"] not in seen:
                    seen.add(r["filename"])
                    unique.append(r)
            results = list(existing_results) + unique

            elapsed = time.time() - start
            logger.info(f"Gathered {len(unique)} unique results in {elapsed / 60:.1f} min")

            self._save_final(args, results, accelerator=accelerator)
        return 0

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _filter_max_rooms(files, max_rooms):
        kept = []
        for fp in files:
            with open(fp) as f:
                data = json.load(f)
            n = len(data.get("accessgraph", {}).get("nodes", {}))
            if n <= max_rooms:
                kept.append(fp)
        logger.info(f"Room filter (max {max_rooms}): kept {len(kept)}/{len(files)}")
        return kept

    @staticmethod
    def _load_checkpoint(args):
        existing = []
        completed = set()
        cp = getattr(args, "checkpoint", None)
        if cp and Path(cp).exists():
            with open(cp) as f:
                data = json.load(f)
            existing = data.get("results", [])
            completed = {r["filename"] for r in existing}
            logger.info(f"Loaded checkpoint: {len(completed)} samples")
        return existing, completed

    def _load_sample(self, fpath: Path) -> dict:
        with open(fpath) as f:
            data = json.load(f)
        ag = data.get("accessgraph", {})
        hg_str = data.get("hypergraph", "")
        hg = json.loads(hg_str) if isinstance(hg_str, str) else hg_str
        bounds_str = data.get("bounds", "{}")
        bounds = json.loads(bounds_str) if isinstance(bounds_str, str) else bounds_str
        return {
            "filename": fpath.name,
            "accessgraph": ag,
            "ground_truth": hg,
            "bounds": bounds or {},
            "num_nodes": len(ag.get("nodes", {})),
            "num_edges": len(ag.get("edges", [])),
        }

    @staticmethod
    def _batch_to_sample(batch, include_gt, include_bounds):
        sample = {
            "filename": batch["filename"][0],
            "accessgraph": batch["accessgraph"][0],
            "num_nodes": batch["num_nodes"][0].item() if torch.is_tensor(batch["num_nodes"][0]) else batch["num_nodes"][0],
            "num_edges": batch["num_edges"][0].item() if torch.is_tensor(batch["num_edges"][0]) else batch["num_edges"][0],
        }
        if include_gt:
            sample["ground_truth"] = batch["ground_truth"][0]
        if include_bounds and "bounds" in batch:
            sample["bounds"] = batch.get("bounds", [{}])[0]
        return sample

    def _log_slow(self, sample, res):
        from utils.data_loading import log_slow_generation
        threshold = getattr(self.args, "slow_gen_threshold", 300.0)
        log_slow_generation(
            sample["filename"], res.generation_time,
            sample.get("num_nodes", 0), sample.get("num_edges", 0),
            res.input_tokens, threshold,
        )

    def _save_checkpoint(self, args, results, all_files=None):
        out = {"timestamp": datetime.now().isoformat(),
               "completed": len(results), "results": results}
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)

    def _save_final(self, args, results, *, accelerator=None):
        stats = self.compute_statistics(results)
        meta = {
            "timestamp": datetime.now().isoformat(),
            "mode": getattr(args, "mode", "unknown"),
            "model": getattr(args, "model", None),
            "dataset_dir": args.dataset_dir,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "total_samples": len(results),
        }
        if accelerator:
            meta["num_processes"] = accelerator.num_processes

        output_data = {"metadata": meta, "statistics": stats, "results": results}
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to: {args.output}")

        self._save_hypergraphs(args, results)

    @staticmethod
    def _save_hypergraphs(args, results):
        hgs = {}
        for r in results:
            gh = r.pop("generated_hypergraph", None)
            if gh is not None:
                hgs[r["filename"].replace(".txt", "")] = gh
        if hgs:
            hp = args.output.replace(".json", "_hypergraphs.json")
            with open(hp, "w") as f:
                json.dump(hgs, f, indent=2)
            logger.info(f"Hypergraphs saved to: {hp} ({len(hgs)} entries)")


# ===================================================================
# TrainedModelEvaluator  (--mode trained)
# ===================================================================

class TrainedModelEvaluator(Evaluator):
    distributed = True

    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = Path(self.args.model)
        is_lora = (model_path / "adapter_config.json").exists()

        if is_lora:
            with open(model_path / "adapter_config.json") as f:
                cfg = json.load(f)
            base_name = cfg.get("base_model_name_or_path")
            if not base_name:
                raise ValueError("Missing base_model_name_or_path in adapter_config.json")
            logger.info(f"LoRA adapter detected, base model: {base_name}")

            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            base_model = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.float16)
            from peft import PeftModel
            model = PeftModel.from_pretrained(base_model, str(model_path))
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.args.model, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.args.model, torch_dtype=torch.float16, local_files_only=True,
            )

        self._raw_model = model
        self.generator = QwenGenerator(model=model, tokenizer=tokenizer)

    def _update_generator_model(self, prepared_model):
        self.generator._model = prepared_model

    def create_prompt(self, sample):
        from utils.prompts import create_simple_prompt, create_simple_prompt_footprint
        ag = sample["accessgraph"]
        if getattr(self.args, "footprint", False):
            bounds = sample.get("bounds", {}) or {}
            return create_simple_prompt_footprint(ag, bounds)
        return create_simple_prompt(ag)

    def extract_hypergraph(self, response):
        return Hypergraph.from_qwen_response(
            response,
            allow_raw_json_fallback=getattr(self.args, "footprint", False),
        )

    def score_result(self, result, hg, sample):
        result["generated_hypergraph"] = hg.data
        noged = getattr(self.args, "noged", False)

        if not noged:
            from utils import compute_ged_from_hypergraph
            ag = sample["accessgraph"]

            ged_info = result.pop("_ged_info", None)
            if ged_info:
                ged = ged_info["ged"]
            else:
                _, _, _, ged_info_v = hg.validate(accessgraph=ag, debug=True)
                ged = ged_info_v["ged"] if ged_info_v else None
                if ged is None:
                    ged, _ = compute_ged_from_hypergraph(ag, hg.data, sample["filename"])

            if ged is not None:
                result["success"] = True
                result["ged"] = ged
            else:
                result["error"] = result.get("error") or "GED computation failed"
        else:
            result.pop("_ged_info", None)
            if result["validation"]["is_valid"]:
                result["success"] = True

    def compute_statistics(self, results):
        noged = getattr(self.args, "noged", False)
        json_failed = [r for r in results if "Failed to extract JSON" in str(r.get("error", ""))]
        timed_out = [r for r in results if r.get("timed_out")]
        json_ok = [r for r in results if r not in json_failed and r not in timed_out]
        valid = [r for r in results if r.get("validation", {}).get("is_valid")]

        stats: dict = {
            "json_extracted": len(json_ok),
            "json_failed": len(json_failed),
            "timed_out": len(timed_out),
            "valid_structures": len(valid),
        }

        if not noged:
            ged_ok = [r for r in results if r.get("success")]
            geds = [r["ged"] for r in ged_ok if r.get("ged") is not None]
            stats["ged"] = {
                "computed": len(ged_ok),
                "average": sum(geds) / len(geds) if geds else None,
                "min": min(geds) if geds else None,
                "max": max(geds) if geds else None,
            }

        times = [r["generation_time"] for r in results if r.get("generation_time") is not None]
        if times:
            stats["inference_time"] = {
                "average_seconds": sum(times) / len(times),
                "min_seconds": min(times),
                "max_seconds": max(times),
                "total_minutes": sum(times) / 60,
            }
        return stats


# ===================================================================
# RoomsOnlyEvaluator  (--mode rooms-only)
# ===================================================================

class RoomsOnlyEvaluator(Evaluator):
    distributed = True

    def build_result_template(self, sample):
        r = super().build_result_template(sample)
        r.update({"room_count_match": False, "room_types_match": False, "room_matching": None})
        return r

    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.args.model, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.args.model, torch_dtype=torch.float16, local_files_only=True,
        )
        self._raw_model = model
        self.generator = QwenGenerator(model=model, tokenizer=tokenizer)

    def _update_generator_model(self, prepared_model):
        self.generator._model = prepared_model

    def create_prompt(self, sample):
        from utils.prompts import create_simple_prompt_rooms_only
        return create_simple_prompt_rooms_only(sample["accessgraph"])

    def extract_hypergraph(self, response):
        return Hypergraph.from_qwen_response(response)

    def score_result(self, result, hg, sample):
        result.pop("_ged_info", None)
        from utils import compute_room_type_accuracy
        rm = compute_room_type_accuracy(sample["accessgraph"], hg.data)
        result["room_matching"] = rm
        result["room_count_match"] = rm["room_count_match"]
        result["room_types_match"] = rm["room_types_match"]
        if rm["room_types_match"]:
            result["success"] = True

    def compute_statistics(self, results):
        json_failed = [r for r in results if "Failed to extract JSON" in str(r.get("error", ""))]
        timed_out = [r for r in results if r.get("timed_out")]
        json_ok = [r for r in results if r not in json_failed and r not in timed_out]
        valid = [r for r in results if r.get("validation", {}).get("is_valid")]
        count_match = [r for r in results if r.get("room_count_match")]
        types_match = [r for r in results if r.get("room_types_match")]

        stats: dict = {
            "json_extracted": len(json_ok),
            "json_failed": len(json_failed),
            "timed_out": len(timed_out),
            "valid_structures": len(valid),
            "room_matching": {
                "room_count_matches": len(count_match),
                "room_types_matches": len(types_match),
            },
        }
        times = [r["generation_time"] for r in results if r.get("generation_time") is not None]
        if times:
            stats["inference_time"] = {
                "average_seconds": sum(times) / len(times),
                "total_minutes": sum(times) / 60,
            }
        return stats


# ===================================================================
# FewShotQwenEvaluator  (--mode fewshot-qwen)
# ===================================================================

class FewShotQwenEvaluator(Evaluator):
    distributed = False

    def build_result_template(self, sample):
        r = super().build_result_template(sample)
        r.update({"json_extracted": False, "ged_computed": False, "ged": None, "error_type": None})
        return r

    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        gpu = getattr(self.args, "gpu", 0)
        logger.info(f"Loading model: {self.args.model} on GPU {gpu}")
        tokenizer = AutoTokenizer.from_pretrained(self.args.model)
        model = AutoModelForCausalLM.from_pretrained(
            self.args.model, torch_dtype=torch.float16, device_map={"": gpu},
        )
        self.generator = QwenGenerator(model=model, tokenizer=tokenizer)
        logger.info("Model loaded")

    def create_prompt(self, sample):
        from utils.prompts import create_structured_prompt
        from utils.data_loading import load_example_hypergraphs

        examples = load_example_hypergraphs(
            self.args.dataset_dir, self.args.num_examples,
            exclude_file=sample["filename"], seed=self.args.seed,
        )
        return create_structured_prompt(examples, sample["accessgraph"])

    def extract_hypergraph(self, response):
        return Hypergraph.from_qwen_response(response)

    def score_result(self, result, hg, sample):
        result.pop("_ged_info", None)
        result["json_extracted"] = True
        from utils import compute_ged_from_hypergraph
        ged, _ = compute_ged_from_hypergraph(sample["accessgraph"], hg.data, sample["filename"])
        if ged is not None:
            result["ged_computed"] = True
            result["success"] = True
            result["ged"] = ged
        else:
            result["error"] = "GED computation failed or timed out"
            result["error_type"] = "ged_computation"

    def compute_statistics(self, results):
        ok = [r for r in results if r.get("success")]
        geds = [r["ged"] for r in ok if r.get("ged") is not None]
        json_ok = [r for r in results if r.get("json_extracted")]

        stats: dict = {
            "successful": len(ok),
            "failed": len(results) - len(ok),
            "json_extracted": len(json_ok),
        }
        if geds:
            stats["ged"] = {
                "average": sum(geds) / len(geds),
                "min": min(geds),
                "max": max(geds),
            }

        error_types: dict[str, int] = {}
        for r in results:
            if not r.get("success"):
                et = r.get("error_type", "unknown")
                error_types[et] = error_types.get(et, 0) + 1
        if error_types:
            stats["error_type_distribution"] = error_types
        return stats


# ===================================================================
# FewShotGPTEvaluator  (--mode fewshot-gpt)
# ===================================================================

class FewShotGPTEvaluator(Evaluator):
    distributed = False

    def build_result_template(self, sample):
        r = super().build_result_template(sample)
        r.update({"json_extracted": False, "ged_computed": False, "ged": None,
                   "error_type": None, "timestamp": datetime.now().isoformat()})
        return r

    def load_model(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.generator = GPTGenerator(model_name=self.args.model, api_key=api_key)
        logger.info(f"GPT generator initialised ({self.args.model})")

    def create_prompt(self, sample):
        from utils.prompts import create_structured_prompt
        from utils.data_loading import load_example_hypergraphs

        examples = load_example_hypergraphs(
            self.args.dataset_dir, self.args.num_examples,
            exclude_file=sample["filename"], seed=self.args.seed,
        )
        return create_structured_prompt(examples, sample["accessgraph"])

    def extract_hypergraph(self, response):
        return Hypergraph.from_gpt_response(response)

    def score_result(self, result, hg, sample):
        result.pop("_ged_info", None)
        result["json_extracted"] = True
        from utils import compute_ged_from_hypergraph
        try:
            ged, _ = compute_ged_from_hypergraph(sample["accessgraph"], hg.data, sample["filename"])
            if ged is not None:
                result["ged"] = ged
                result["ged_computed"] = True
                result["success"] = True
            else:
                result["error"] = "GED computation failed or timed out"
                result["error_type"] = "ged_computation"
        except Exception as e:
            result["error"] = f"GED error: {e}"
            result["error_type"] = "ged_computation"

    def compute_statistics(self, results):
        ok = [r for r in results if r.get("success")]
        geds = [r["ged"] for r in ok if r.get("ged") is not None]
        json_ok = [r for r in results if r.get("json_extracted")]
        times = [r["generation_time"] for r in ok if r.get("generation_time") is not None]

        stats: dict = {
            "successful": len(ok),
            "failed": len(results) - len(ok),
            "json_extracted": len(json_ok),
            "success_rate": len(ok) / len(results) if results else 0,
        }
        if geds:
            stats["ged"] = {
                "average": sum(geds) / len(geds),
                "min": min(geds),
                "max": max(geds),
            }
        if times:
            stats["avg_generation_time"] = sum(times) / len(times)

        error_types: dict[str, int] = {}
        for r in results:
            if not r.get("success"):
                et = r.get("error_type", "unknown")
                error_types[et] = error_types.get(et, 0) + 1
        if error_types:
            stats["error_type_distribution"] = error_types
        return stats


# ===================================================================
# CLI
# ===================================================================

MODE_MAP = {
    "trained": TrainedModelEvaluator,
    "rooms-only": RoomsOnlyEvaluator,
    "fewshot-qwen": FewShotQwenEvaluator,
    "fewshot-gpt": FewShotGPTEvaluator,
}


def cli():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Unified hypergraph evaluator")
    parser.add_argument("--mode", choices=list(MODE_MAP), required=True)

    # Shared
    parser.add_argument("--dataset-dir", type=str, default="training_data/extracted_hypergraph_msd")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=3328)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max-rooms", type=int, default=None)

    # Trained / rooms-only
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--noged", action="store_true")
    parser.add_argument("--footprint", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--slow-gen-threshold", type=float, default=300.0)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--nccl-timeout", type=int, default=7200)

    # Few-shot
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()

    # Defaults
    if args.model is None:
        if args.mode == "fewshot-qwen":
            args.model = "Qwen/Qwen3-4B-Instruct-2507"
        elif args.mode == "fewshot-gpt":
            args.model = "gpt-4"
        elif args.mode in ("trained", "rooms-only"):
            parser.error(f"--model is required for --mode {args.mode}")

    if args.output is None:
        args.output = f"results_{args.mode.replace('-', '_')}_{Path(args.model).name}_{args.seed}.json"
        logger.info(f"Auto-generated output: {args.output}")

    if args.mode.startswith("fewshot"):
        args.max_tokens = args.max_tokens or 5120

    cls = MODE_MAP[args.mode]
    evaluator = cls(args)
    return evaluator.run()


if __name__ == "__main__":
    sys.exit(cli())
