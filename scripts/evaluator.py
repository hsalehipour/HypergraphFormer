#!/usr/bin/env python3
"""
Unified evaluator for hypergraph generation models.

Modes:
    trained      -- Fine-tuned Qwen, zero-shot, GED + validation (distributed)
    rooms-only   -- Fine-tuned Qwen, rooms-only prompt, room matching (distributed)
    trained-vllm -- Fine-tuned Qwen via vLLM, single process + tensor parallelism
    fewshot-qwen -- Base Qwen, few-shot prompting, GED (single GPU)
    fewshot-gpt  -- GPT via OpenAI API, few-shot prompting, GED (single process)

Usage:
    # Fine-tuned model (distributed, HuggingFace)
    accelerate launch --num_processes=8 scripts/evaluator.py --mode trained \\
        --model checkpoints/lora_r64_alpha128

    # Fine-tuned model (single process, vLLM — much faster)
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/evaluator.py \\
        --mode trained-vllm --model checkpoints/lora_r64_alpha128

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
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path

import torch
from tqdm import tqdm
from torch.distributed.elastic.multiprocessing.errors import record

from utils.hypergraph import Hypergraph
from scripts.fewshot_generator import QwenGenerator, GPTGenerator

logger = logging.getLogger(__name__)


def _gpu_count_via_nvidia_smi() -> int:
    """GPU count without importing CUDA in the current process (for vLLM parent)."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=30,
            stderr=subprocess.DEVNULL,
        )
        n = len([ln for ln in out.strip().splitlines() if ln.strip()])
        return n if n > 0 else 1
    except Exception:
        return 1


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
        # Do not touch CUDA in the parent before vLLM spawns worker processes: PyTorch
        # raises "Cannot re-initialize CUDA in forked subprocess" if the parent has
        # already initialized the driver and workers are forked.  Note:
        # ``torch.cuda.is_available()`` itself can initialize CUDA, so check the flag
        # *before* calling any torch.cuda API.
        if not getattr(self, "defer_cuda_seed_until_model_loaded", False):
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

        skip_hg_path = getattr(args, "skip_covered_hypergraphs", None)
        if skip_hg_path:
            with open(skip_hg_path) as f:
                covered_ids = set(json.load(f).keys())
            before = len(all_files)
            all_files = [f for f in all_files if f.stem not in covered_ids]
            logger.info(f"Skipped {before - len(all_files)} samples already covered in {skip_hg_path}")

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
            self._save_partial(args, results)

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
        import concurrent.futures
        import threading
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

        # How many hypergraphs to keep in-flight on this GPU simultaneously.
        n_parallel = getattr(args, "batch_size_per_gpu", 1)

        # Hard per-sample deadline: same as the soft generation timeout.
        # A future that exceeds this is abandoned (marked timed-out) so the slot
        # can be refilled immediately, regardless of whether the CUDA kernel finishes.
        soft_timeout = getattr(args, "timeout", None)
        hard_timeout: float | None = float(soft_timeout) if soft_timeout else None

        # Serialize GPU kernel calls so concurrent threads don't race on the same
        # device.  Non-GPU work (prompt creation, JSON parsing, GED) still overlaps.
        #
        # We use lock.acquire(timeout=hard_timeout) rather than a plain `with lock:`
        # so that if one thread is stuck inside a deadlocked CUDA kernel and holds
        # the lock indefinitely, other threads waiting to run inference give up after
        # hard_timeout seconds and return a synthetic timed-out result — keeping them
        # fully unblocked from the stuck thread.
        if n_parallel > 1:
            import types as _types
            _gen_lock = threading.Lock()
            _orig_generate = self.generator.generate

            def _locked_generate(*a, **kw):
                acquired = _gen_lock.acquire(timeout=hard_timeout if hard_timeout else -1)
                if not acquired:
                    return _types.SimpleNamespace(
                        timed_out=True,
                        text="",
                        input_tokens=0,
                        generation_time=hard_timeout or 0.0,
                    )
                try:
                    return _orig_generate(*a, **kw)
                finally:
                    _gen_lock.release()

            self.generator.generate = _locked_generate  # type: ignore[method-assign]
            logger.info(
                f"[rank {accelerator.process_index}] "
                f"Running {n_parallel} hypergraphs per GPU in parallel "
                f"(hard_timeout={hard_timeout}s)"
            )

        start = time.time()
        local_results: list[dict] = []
        rank = accelerator.process_index
        dataloader_iter = iter(tqdm(dataloader, desc="Evaluating",
                                    disable=not accelerator.is_main_process))

        def _safe_evaluate(sample: dict) -> dict:
            try:
                result = self.evaluate_sample(sample)
                # evaluate_sample swallows all exceptions into result["error"], so
                # detect CUDA OOM by inspecting the error string.
                if result.get("error") and "out of memory" in str(result["error"]).lower():
                    result["oom"] = True
                return result
            except Exception as exc:
                r = self.build_result_template(sample)
                r["error"] = f"Unexpected thread error: {exc}"
                if "out of memory" in str(exc).lower():
                    r["oom"] = True
                return r

        def _make_sample(batch: dict) -> dict:
            return self._batch_to_sample(batch, include_gt, include_bounds)

        # Sliding-window executor: always keep up to n_parallel futures in flight.
        # pending maps future -> (sample_dict, monotonic_submit_time)
        pending: dict[concurrent.futures.Future, tuple[dict, float]] = {}
        # Samples that OOM'd or timed out and are waiting for a serial retry after
        # the current in-flight batch drains.  Timed-out samples are worth retrying
        # because with n_parallel>1 each thread waits up to (n-1)×timeout for the
        # inference lock, so a sample that appears to time out at batch=4 may succeed
        # fine when run alone.
        retry_queue: list[dict] = []
        draining_for_retry = False  # when True, no new submissions until pending is empty

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel) as executor:

            def _try_submit() -> bool:
                """Pull the next batch from the dataloader and submit it. Return False when exhausted."""
                if draining_for_retry:
                    return False
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    return False
                sample = _make_sample(batch)
                future = executor.submit(_safe_evaluate, sample)
                pending[future] = (sample, time.monotonic())
                return True

            # Fill the initial window.
            for _ in range(n_parallel):
                if not _try_submit():
                    break

            while pending or retry_queue:

                # --- Serial retry phase ---
                # Entered once all in-flight samples have drained.  Both OOM and
                # timed-out samples land here.  Timed-out samples are worth retrying
                # because with n_parallel>1 each thread waits up to (n-1)×timeout for
                # the inference lock, so a sample that appears to time out at batch=4
                # may succeed fine when run alone.
                if not pending and retry_queue:
                    n_oom = sum(1 for s in retry_queue if s.get("_retry_reason") == "oom")
                    n_to  = sum(1 for s in retry_queue if s.get("_retry_reason") == "timeout")
                    logger.warning(
                        f"[rank {rank}] Serial retry: {len(retry_queue)} sample(s) "
                        f"({n_oom} OOM, {n_to} timeout) — clearing GPU cache first"
                    )
                    torch.cuda.empty_cache()
                    for retry_sample in retry_queue:
                        reason = retry_sample.pop("_retry_reason", "unknown")
                        logger.info(
                            f"[rank {rank}] Serial retry ({reason}): {retry_sample['filename']}"
                        )
                        retry_result = _safe_evaluate(retry_sample)
                        retry_result["serial_retried"] = True
                        retry_result["serial_retry_reason"] = reason
                        if retry_result.get("oom"):
                            retry_result["error"] = (
                                f"OOM persists on serial retry: {retry_result.get('error', '')}"
                            )
                        if retry_result.get("timed_out"):
                            retry_result["error"] = (
                                f"Timeout persists on serial retry: {retry_result.get('error', '')}"
                            )
                        local_results.append(retry_result)
                        self._save_partial(args, local_results, rank)
                    retry_queue.clear()
                    draining_for_retry = False
                    # Refill the window back to full batch size.
                    for _ in range(n_parallel):
                        if not _try_submit():
                            break
                    continue

                # --- Normal collection phase ---
                poll = min(5.0, hard_timeout / 4) if hard_timeout else 5.0

                done, _ = concurrent.futures.wait(
                    list(pending.keys()),
                    timeout=poll,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                now = time.monotonic()

                def _queue_for_retry(sample: dict, reason: str):
                    """Route a sample to the serial retry queue and start draining."""
                    nonlocal draining_for_retry
                    sample["_retry_reason"] = reason
                    retry_queue.append(sample)
                    draining_for_retry = True

                # Abandon futures that exceeded the hard deadline — queue for retry
                # instead of discarding, so they get a fair chance alone.
                if hard_timeout:
                    for fut, (sample, submit_t) in list(pending.items()):
                        if fut not in done and now - submit_t > hard_timeout:
                            logger.warning(
                                f"[rank {rank}] Hard timeout for {sample['filename']} "
                                f"after {now - submit_t:.0f}s — queuing for serial retry"
                            )
                            del pending[fut]
                            _queue_for_retry(sample, "timeout")
                            # Do not call _try_submit(): let pending drain.

                # Collect completed futures.
                for fut in done:
                    if fut not in pending:
                        continue  # already handled as hard-timeout above
                    sample, _ = pending.pop(fut)
                    try:
                        result = fut.result(timeout=0)
                    except concurrent.futures.TimeoutError:
                        result = self.build_result_template(sample)
                        result["error"] = "Future result unavailable after completion"
                        result["timed_out"] = True
                    except Exception as exc:
                        result = self.build_result_template(sample)
                        result["error"] = f"Thread raised: {exc}"

                    if result.get("oom"):
                        logger.warning(
                            f"[rank {rank}] OOM on {sample['filename']} — "
                            f"queuing for serial retry after {len(pending)} in-flight drain"
                        )
                        _queue_for_retry(sample, "oom")
                        # Do not call _try_submit(): let pending drain naturally.
                    elif result.get("timed_out") and n_parallel > 1:
                        logger.warning(
                            f"[rank {rank}] Timeout on {sample['filename']} — "
                            f"queuing for serial retry after {len(pending)} in-flight drain"
                        )
                        _queue_for_retry(sample, "timeout")
                        # Do not call _try_submit(): let pending drain naturally.
                    else:
                        local_results.append(result)
                        self._save_partial(args, local_results, rank)
                        _try_submit()

        # Save per-rank results to temp files
        tmp = Path(tempfile.gettempdir())
        rank_file = tmp / f"eval_results_rank_{accelerator.process_index}.json"
        with open(rank_file, "w") as f:
            json.dump(local_results, f)
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            gathered = []
            for r in range(accelerator.num_processes):
                rf = tmp / f"eval_results_rank_{r}.json"
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

    @staticmethod
    def _save_partial(args, results, rank=None):
        suffix = f"_rank_{rank}" if rank is not None else "_partial"
        out_path = Path(args.output.replace(".json", f"{suffix}.json"))
        hg_path = Path(args.output.replace(".json", f"{suffix}_hypergraphs.json"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {"timestamp": datetime.now().isoformat(), "completed": len(results), "results": results},
                f,
                indent=2,
            )
        hgs = {
            r["filename"].replace(".txt", ""): r["generated_hypergraph"]
            for r in results
            if r.get("generated_hypergraph") is not None
        }
        with open(hg_path, "w") as f:
            json.dump(hgs, f, indent=2)

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
# VLLMTrainedEvaluator  (--mode trained-vllm)
# ===================================================================

class VLLMTrainedEvaluator(TrainedModelEvaluator):
    """
    vLLM-based evaluator: single Python process, vLLM handles all GPU
    parallelism internally via tensor_parallel_size.

    Unlike TrainedModelEvaluator (which uses ``accelerate launch``), this
    runs as a plain ``python`` invocation:

        CUDA_VISIBLE_DEVICES=0,1,2,...  python scripts/evaluator.py \\
            --mode trained-vllm [--batch-size-per-gpu 4] ...

    vLLM's continuous batching + PagedAttention gives ~10-20x throughput
    versus the HuggingFace single-sample path.
    """

    # Single-process: vLLM owns all GPUs via tensor_parallel_size.
    distributed = False
    # See Evaluator.run(): skip torch.cuda.manual_seed_all before load_model so the
    # parent never initializes CUDA ahead of vLLM's worker processes (fork + CUDA
    # triggers "Cannot re-initialize CUDA in forked subprocess").
    defer_cuda_seed_until_model_loaded = True

    def load_model(self):
        """Parse adapter config and initialise the vLLM engine."""
        model_path = Path(self.args.model)
        is_lora = (model_path / "adapter_config.json").exists()

        if is_lora:
            with open(model_path / "adapter_config.json") as f:
                cfg = json.load(f)
            base_name = cfg.get("base_model_name_or_path")
            if not base_name:
                raise ValueError("Missing base_model_name_or_path in adapter_config.json")
            lora_rank = cfg.get("r", 64)
            lora_path = str(model_path)
            logger.info(f"LoRA adapter detected — base: {base_name}  rank: {lora_rank}")
        else:
            base_name = str(model_path)
            lora_rank = 64
            lora_path = None

        # Auto-detect GPU count from CUDA_VISIBLE_DEVICES or nvidia-smi — never
        # ``torch.cuda.device_count()`` here (initializes CUDA in the parent before vLLM workers).
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_devices:
            tensor_parallel_size = len([d for d in cuda_devices.split(",") if d.strip()])
        else:
            tensor_parallel_size = _gpu_count_via_nvidia_smi()

        batch_size = self.args.batch_size_per_gpu
        logger.info(
            f"vLLM single-process evaluation — {tensor_parallel_size} GPU(s)  "
            f"batch_size={batch_size}"
        )

        from scripts.fewshot_generator import VLLMQwenGenerator
        self.generator = VLLMQwenGenerator(
            base_model=base_name,
            lora_path=lora_path,
            lora_rank=lora_rank,
            gpu_memory_utilization=getattr(self.args, "gpu_memory_utilization", 0.90),
            max_model_len=self.args.max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=getattr(self.args, "enforce_eager", False),
        )
        self.generator.load()

    def _update_generator_model(self, _):
        pass  # not applicable for vLLM

    def extract_hypergraph(self, response: str):
        """
        vLLM returns only the completion string, not the full HF decode that still
        contains ``<|im_start|>assistant``.  Reconcile with ``from_qwen_response``.
        """
        if not response:
            return None
        if "<|im_start|>assistant" not in response:
            response = f"<|im_start|>assistant\n{response}"
        return Hypergraph.from_qwen_response(
            response,
            allow_raw_json_fallback=True,
        )

    # -- override single-process loop to use batched inference ---------------

    def _run_single(self, args, files_to_process, existing_results, all_files):
        from torch.utils.data import DataLoader
        from utils.data_loading import HypergraphDataset, collate_fn

        include_gt = not isinstance(self, RoomsOnlyEvaluator)
        include_bounds = getattr(args, "footprint", False)
        dataset = HypergraphDataset(
            files_to_process,
            include_ground_truth=include_gt,
            include_bounds=include_bounds,
        )
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=getattr(args, "shuffle", False),
            num_workers=0, collate_fn=collate_fn,
        )

        batch_size = args.batch_size_per_gpu
        timeout = getattr(args, "timeout", None)
        batch_timeout = (timeout * batch_size + 60) if timeout else None

        results = list(existing_results)
        batch_buffer: list[dict] = []
        start = time.time()

        for batch in tqdm(dataloader, desc="Evaluating"):
            sample = self._batch_to_sample(batch, include_gt, include_bounds)
            batch_buffer.append(sample)

            if len(batch_buffer) >= batch_size:
                self._flush_batch(args, batch_buffer, results, None,
                                  timeout, batch_timeout)
                self._save_checkpoint(args, results, all_files)
                batch_buffer = []

        if batch_buffer:
            self._flush_batch(args, batch_buffer, results, None,
                              timeout, batch_timeout)

        elapsed = time.time() - start
        logger.info(f"Total time: {elapsed / 60:.1f} min")
        self._save_final(args, results)
        return 0

    # -- batch helpers ----------------------------------------------------

    def _flush_batch(self, args, samples, local_results, rank, timeout, batch_timeout):
        """Run one batched vLLM call and process each result."""
        prompts = [self.create_prompt(s) for s in samples]

        try:
            gen_results = self.generator.generate_batch(
                prompts,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=batch_timeout,
            )
        except Exception as exc:
            err = str(exc)
            is_oom = "out of memory" in err.lower()
            logger.warning(
                f"[rank {rank}] Batch generation failed ({err[:120]}) — "
                f"retrying {len(samples)} sample(s) individually"
            )
            if is_oom:
                torch.cuda.empty_cache()
            for s in samples:
                result = self._generate_and_process_single(args, s, timeout)
                result["batch_retry"] = True
                result["batch_retry_reason"] = "oom" if is_oom else "error"
                local_results.append(result)
                self._save_partial(args, local_results, rank)
            return

        # If the whole batch timed out, retry each sample alone.
        if all(r.timed_out for r in gen_results):
            logger.warning(
                f"[rank {rank}] Entire batch timed out — "
                f"retrying {len(samples)} sample(s) individually"
            )
            for s in samples:
                result = self._generate_and_process_single(args, s, timeout)
                result["batch_retry"] = True
                result["batch_retry_reason"] = "timeout"
                local_results.append(result)
                self._save_partial(args, local_results, rank)
            return

        for sample, gen_res in zip(samples, gen_results):
            if gen_res.timed_out:
                result = self._generate_and_process_single(args, sample, timeout)
                result["batch_retry"] = True
                result["batch_retry_reason"] = "timeout"
            else:
                result = self._process_gen_result(args, sample, gen_res, timeout)
            local_results.append(result)
            self._save_partial(args, local_results, rank)

    def _generate_and_process_single(self, args, sample, timeout):
        """Generate and process one sample (used when a batch-level retry is needed)."""
        prompt = self.create_prompt(sample)
        gen_res = self.generator.generate(
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=timeout,
        )
        return self._process_gen_result(args, sample, gen_res, timeout)

    def _process_gen_result(self, args, sample, gen_res, timeout):
        """
        Turn a GenerationResult into a scored result dict.

        Mirrors evaluate_sample() from the base class but works on an already-
        generated GenerationResult rather than calling generate() itself.
        """
        result = self.build_result_template(sample)
        result["num_input_tokens"] = gen_res.input_tokens
        result["generation_time"] = gen_res.generation_time

        if gen_res.timed_out:
            result["error"] = f"Generation timed out"
            result["timed_out"] = True
            return result

        if not gen_res.text:
            result["error"] = "Empty generation output"
            return result

        hg = self.extract_hypergraph(gen_res.text)

        if hg is None:
            logger.warning(f"JSON extraction failed for {sample['filename']}, retrying...")
            retry_res = self.generator.generate(
                self.create_prompt(sample),
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=timeout,
            )
            result["generation_time"] += retry_res.generation_time
            if retry_res.timed_out:
                result["error"] = "Timed out on retry"
                result["timed_out"] = True
                return result
            hg = self.extract_hypergraph(retry_res.text or "")
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

        try:
            self.score_result(result, hg, sample)
        except Exception as exc:
            result["error"] = str(exc)

        return result


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
    "trained-vllm": VLLMTrainedEvaluator,
    "rooms-only": RoomsOnlyEvaluator,
    "fewshot-qwen": FewShotQwenEvaluator,
    "fewshot-gpt": FewShotGPTEvaluator,
}


@record
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
    parser.add_argument("--skip-covered-hypergraphs", type=str, default=None)
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
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.90,
        help="vLLM only: fraction of GPU VRAM to use for model + KV cache (default 0.90).",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=8192,
        help="vLLM only: maximum total sequence length (prompt + output tokens, default 8192).",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help=(
            "vLLM only: disable CUDA graph capture (lower peak VRAM during startup; slower decoding)."
        ),
    )
    parser.add_argument(
        "--batch-size-per-gpu", type=int, default=None,
        help=(
            "Hypergraphs per vLLM batch (trained-vllm) or concurrent slots per GPU (trained). "
            "Default: 4 for trained-vllm, 1 for other modes. Raise if you have VRAM headroom."
        ),
    )

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
        elif args.mode in ("trained", "trained-vllm", "rooms-only"):
            parser.error(f"--model is required for --mode {args.mode}")

    if args.output is None:
        args.output = f"results_{args.mode.replace('-', '_')}_{Path(args.model).name}_{args.seed}.json"
        logger.info(f"Auto-generated output: {args.output}")

    if args.mode.startswith("fewshot"):
        args.max_tokens = args.max_tokens or 5120

    if args.batch_size_per_gpu is None:
        args.batch_size_per_gpu = 4 if args.mode == "trained-vllm" else 1

    cls = MODE_MAP[args.mode]
    evaluator = cls(args)
    return evaluator.run()


if __name__ == "__main__":
    sys.exit(cli())