#!/usr/bin/env python3
"""
Unified evaluator for hypergraph generation models.

Modes:
    trained      -- Fine-tuned Qwen, zero-shot, GED + validation (distributed)
    rooms-only   -- Fine-tuned Qwen, rooms-only prompt, room matching (distributed)
    fewshot-qwen -- Base Qwen, few-shot prompting, GED (single GPU)
    fewshot-gpt  -- GPT via OpenAI API, few-shot prompting, GED (single process)

Usage:
    # Single GPU (one vLLM engine on that card)
    accelerate launch --num_processes=1 scripts/evaluator.py --mode trained \\
        --model checkpoints/lora_r64_alpha128

    # All GPUs — one process per GPU (each rank loads its own vLLM). Use --multi_gpu and set
    # num_processes to the GPU count. Do not export CUDA_VISIBLE_DEVICES to a single GPU for
    # the whole job (every rank would share that GPU). Do not use num_processes>1 on one card.
    accelerate launch --multi_gpu --num_processes=8 scripts/evaluator.py --mode trained \\
        --model checkpoints/lora_r64_alpha128

    # torchrun (sets LOCAL_RANK / WORLD_SIZE); evaluator pins each rank to one GPU via
    # CUDA_VISIBLE_DEVICES before importing torch/vLLM.
    torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/evaluator.py --mode trained \\
        --model checkpoints/lora_r64_alpha128

    # Micro-batch two prompts per vLLM generate() per GPU (may need lower --vllm-max-model-len if OOM)
    torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/evaluator.py --mode trained \\
        --model checkpoints/lora_r64_alpha128 --vllm-prompt-batch-size 2

    # Rooms-only (same rules)
    accelerate launch --multi_gpu --num_processes=8 scripts/evaluator.py --mode rooms-only \\
        --model checkpoints/lora_r64_alpha128

    # Few-shot Qwen (single GPU; vLLM by default)
    python scripts/evaluator.py --mode fewshot-qwen --dataset-dir data/test

    # Few-shot GPT
    python scripts/evaluator.py --mode fewshot-gpt --model gpt-4-turbo \\
        --dataset-dir data/test

    # Zero-shot (no in-context examples) — works with both fewshot-qwen and fewshot-gpt
    python scripts/evaluator.py --mode fewshot-gpt --model gpt-5.5 \\
        --dataset-dir data/test --num-examples 0
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path

# vLLM is imported by scripts.generators; set env before that so V1 engine sees it.
# V1 multiprocess backend + Accelerate often triggers "Engine core initialization failed"
# (see https://github.com/vllm-project/vllm/issues/25837). Override with
# VLLM_ENABLE_V1_MULTIPROCESSING=1 before launch if you need the default.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

# torchrun sets LOCAL_RANK / WORLD_SIZE. Pin each worker to ONE physical GPU *before* torch/cuda
# or vLLM — otherwise every rank defaults to the same GPU (nvidia-smi: 8 procs on GPU 0, OOM).
_lr = os.environ.get("LOCAL_RANK")
_ws = os.environ.get("WORLD_SIZE")
if _lr is not None and _ws is not None and int(_ws) > 1:
    lr_i = int(_lr)
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(lr_i)
    elif "," in cvd:
        _ids = [x.strip() for x in cvd.split(",") if x.strip()]
        if lr_i < len(_ids):
            os.environ["CUDA_VISIBLE_DEVICES"] = _ids[lr_i]

import torch
from tqdm import tqdm
from torch.distributed.elastic.multiprocessing.errors import record

from utils.hypergraph import Hypergraph, repair_bsp_tree_names
from scripts.generators import GPTGenerator, QwenGenerator, VLLM_AVAILABLE, VllmQwenGenerator

logger = logging.getLogger(__name__)


def _is_likely_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


def _recover_cuda_after_oom() -> None:
    """Best-effort allocator cleanup so the next sample may run after a caught OOM."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def _distributed_barrier_safe() -> None:
    """All-rank barrier compatible with per-rank ``CUDA_VISIBLE_DEVICES`` remapping.

    Hugging Face Accelerate's ``wait_for_everyone()`` uses
    ``torch.distributed.barrier(device_ids=[local_process_index])``, which assumes rank *k*
    uses physical GPU *k*. Here each process typically sees only one GPU as ``cuda:0`` (see
    top-of-file ``CUDA_VISIBLE_DEVICES`` logic), so ``device_ids=[7]`` on rank 7 raises
    *CUDA error: invalid device ordinal* after eval finishes.

    Calling ``barrier()`` without ``device_ids`` uses the current device (``cuda:0`` per process).
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    torch.distributed.barrier()


def _merge_distributed_hypergraph_partials(args) -> int:
    """Merge ``{stem}_rank_*_hypergraphs.json`` into ``{stem}_hypergraphs.json``.

    Starts from the combined file already written by :meth:`_save_hypergraphs` (if present),
    then unions all rank partials so the final file is the superset. Per-rank partials on disk
    are not removed.
    """
    out_path = Path(args.output)
    parent = out_path.parent
    stem = out_path.stem
    combined = out_path.with_name(f"{stem}_hypergraphs.json")
    merged: dict = {}
    if combined.exists():
        try:
            with open(combined) as f:
                m = json.load(f)
            if isinstance(m, dict):
                merged.update(m)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Could not read existing combined hypergraphs %s: %s", combined, e)
    paths = sorted(parent.glob(f"{stem}_rank_*_hypergraphs.json"))
    for p in paths:
        try:
            with open(p) as f:
                d = json.load(f)
            if isinstance(d, dict):
                merged.update(d)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Could not read hypergraph partial %s: %s", p, e)
    if not merged:
        return 0
    with open(combined, "w") as f:
        json.dump(merged, f, indent=2)
    logger.info(
        "Merged %d hypergraph entries (combined + %d rank partials) → %s",
        len(merged),
        len(paths),
        combined,
    )
    return len(merged)


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
            self._apply_generation_to_result(sample, result, res, prompt=prompt)
        except Exception as e:
            result["error"] = str(e)
            if _is_likely_cuda_oom(e):
                result["cuda_oom"] = True
                _recover_cuda_after_oom()
        return result

    def _apply_generation_to_result(self, sample: dict, result: dict, res, *, prompt: str) -> None:
        """Fill *result* from a ``GenerationResult`` (mutates in place)."""
        timeout = getattr(self.args, "timeout", None)
        result["num_input_tokens"] = res.input_tokens
        result["generation_time"] = res.generation_time

        if res.timed_out:
            result["error"] = f"Generation timed out after {timeout}s"
            result["timed_out"] = True
            self._log_slow(sample, res)
            return

        if res.generation_time > getattr(self.args, "slow_gen_threshold", 1e9):
            self._log_slow(sample, res)

        hg = self.extract_hypergraph(res.text)

        if hg is None:
            # Optional second LLM call to recover from a JSON parse failure.
            # OFF by default: at the high concurrencies we run these scripts at
            # the marginal recovery rate doesn't justify the ~2x cost on
            # retried samples (and retries tend to cluster, which amplifies
            # rate-limit pressure). Pass --retry-on-json-fail to opt in.
            # The error string still contains "Failed to extract JSON" so
            # downstream summary code that filters on that substring keeps
            # working unchanged in either branch.
            if not getattr(self.args, "retry_on_json_fail", False):
                result["error"] = "Failed to extract JSON from response"
                return
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
                return
            hg = self.extract_hypergraph(res2.text)
            if hg is None:
                result["error"] = "Failed to extract JSON from response (after retry)"
                return
            result["retried"] = True

        is_valid, passed, failed, ged_info = hg.validate(accessgraph=None, debug=False)
        result["validation"] = {
            "is_valid": is_valid,
            "passed_checks": passed,
            "failed_checks": failed,
        }
        result["_ged_info"] = ged_info

        self.score_result(result, hg, sample)

    def _evaluate_samples_vllm_batch(self, samples: list[dict]) -> list[dict]:
        """Run one vLLM batched ``generate``, then per-sample validation (retries stay single-shot).

        If the batched engine call fails (e.g. CUDA OOM on a large batch), fall back to
        one prompt at a time so other samples in the shard can still complete.
        """
        if not hasattr(self.generator, "generate_batch"):
            raise RuntimeError("vLLM prompt batching requires VllmQwenGenerator.generate_batch")
        prompts = [self.create_prompt(s) for s in samples]
        timeout = getattr(self.args, "timeout", None)
        try:
            gen_results = self.generator.generate_batch(
                prompts,
                max_tokens=self.args.max_tokens,
                temperature=self.args.temperature,
                timeout=timeout,
            )
        except Exception as e:
            logger.warning(
                "Batched vLLM generate failed (%s); falling back to per-sample generate for this batch.",
                e,
            )
            if _is_likely_cuda_oom(e):
                _recover_cuda_after_oom()
            return [self.evaluate_sample(s) for s in samples]

        out: list[dict] = []
        for sample, prompt, res in zip(samples, prompts, gen_results):
            result = self.build_result_template(sample)
            try:
                self._apply_generation_to_result(sample, result, res, prompt=prompt)
            except Exception as e:
                result["error"] = str(e)
                if _is_likely_cuda_oom(e):
                    result["cuda_oom"] = True
                    _recover_cuda_after_oom()
            out.append(result)
        return out

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

        # Bind CUDA *before* Accelerator(): otherwise Accelerate/torch may init on GPU 0 on every
        # rank and vLLM still stacks on physical GPU 0 (OOM). torchrun sets LOCAL_RANK/WORLD_SIZE.
        self._accelerator = None
        if self.distributed and torch.cuda.is_available():
            Evaluator._early_bind_cuda_before_accelerate()

        # Initialise Accelerate before vLLM (device already selected when env is set).
        if self.distributed:
            from accelerate import Accelerator, InitProcessGroupKwargs

            nccl_timeout = getattr(args, "nccl_timeout", 7200)
            ddp_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=nccl_timeout))
            self._accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
            if torch.cuda.is_available():
                self._bind_cuda_device_for_distributed_rank()

            if (
                torch.cuda.is_available()
                and not getattr(args, "no_vllm", False)
                and VLLM_AVAILABLE
            ):
                nproc = self._accelerator.num_processes
                ndev = torch.cuda.device_count()
                lr = int(os.environ.get("LOCAL_RANK", self._accelerator.local_process_index))
                logger.info(
                    f"vLLM distributed: num_processes={nproc}, CUDA devices visible to this rank={ndev}, "
                    f"local_rank={lr} (use --multi_gpu with num_processes = #GPUs, or torchrun "
                    f"--nproc_per_node=#GPUs)"
                )

        self.load_model()

        if self.distributed:
            return self._run_distributed(args, files_to_process, existing_results, all_files)
        return self._run_single(args, files_to_process, existing_results, all_files)

    # -- single-process loop -----------------------------------------------

    def _run_single(self, args, files_to_process, existing_results, all_files):
        results = list(existing_results)
        start = time.time()
        n_total = len(files_to_process)
        if n_total:
            logger.info(
                "Evaluation: %d samples (tqdm bar on stdout; logs on stderr).",
                n_total,
            )

        for i, fpath in enumerate(
            tqdm(
                files_to_process,
                desc="Evaluating",
                unit="sample",
                mininterval=1.0,
                file=sys.stdout,
            ),
            1,
        ):
            sample = self._load_sample(fpath)
            result = self.evaluate_sample(sample)
            results.append(result)
            self._save_partial(args, results)

            interval = max(1, n_total // 10) if n_total else 1
            if n_total and (i == 1 or i == n_total or i % interval == 0):
                elapsed = time.time() - start
                rate = i / elapsed if elapsed > 0 else 0.0
                eta_sec = (n_total - i) / rate if rate > 0 else 0.0
                logger.info(
                    "Progress: sample %d/%d | elapsed %.1f min | ETA ~%.1f min",
                    i,
                    n_total,
                    elapsed / 60.0,
                    eta_sec / 60.0,
                )

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
        from torch.utils.data import DataLoader
        from utils.data_loading import HypergraphDataset, collate_fn

        accelerator = self._accelerator
        if accelerator is None:
            from accelerate import Accelerator, InitProcessGroupKwargs

            nccl_timeout = getattr(args, "nccl_timeout", 7200)
            ddp_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=nccl_timeout))
            accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

        if accelerator.is_main_process:
            logger.info(f"Distributed evaluation with {accelerator.num_processes} GPUs")

        include_gt = not isinstance(self, RoomsOnlyEvaluator)
        dataset = HypergraphDataset(
            files_to_process,
            include_ground_truth=include_gt,
        )
        pb = max(1, int(getattr(args, "vllm_prompt_batch_size", 1)))
        use_vllm = not getattr(args, "no_vllm", False) and VLLM_AVAILABLE
        use_prompt_batch = use_vllm and pb > 1
        if use_prompt_batch:
            logger.info(f"vLLM prompt batch size: {pb} (batched generate per step)")
        dataloader = DataLoader(
            dataset,
            batch_size=pb if use_prompt_batch else 1,
            shuffle=getattr(args, "shuffle", False),
            num_workers=0,
            collate_fn=collate_fn,
        )

        if self._raw_model is not None:
            model = accelerator.prepare(self._raw_model)
            self._update_generator_model(model)
        dataloader = accelerator.prepare(dataloader)

        try:
            n_batches_total = len(dataloader)
        except TypeError:
            n_batches_total = None

        start = time.time()
        local_results = []

        if accelerator.is_main_process:
            if n_batches_total is not None:
                logger.info(
                    "Evaluation: %d batches on main process (tqdm bar on stdout; logs on stderr).",
                    n_batches_total,
                )
            else:
                logger.info(
                    "Evaluation: batch count unknown; tqdm may not show ETA; "
                    "watch for 'Progress:' log lines.",
                )

        for batch_idx, batch in enumerate(
            tqdm(
                dataloader,
                desc="Evaluating",
                unit="batch",
                disable=not accelerator.is_main_process,
                mininterval=1.0,
                file=sys.stdout,
            )
        ):
            samples = self._collated_batch_to_samples(batch, include_gt)
            if use_prompt_batch and len(samples) > 1:
                local_results.extend(self._evaluate_samples_vllm_batch(samples))
            else:
                for sample in samples:
                    local_results.append(self.evaluate_sample(sample))
            self._save_partial(args, local_results, accelerator.process_index)

            if accelerator.is_main_process and n_batches_total:
                done = batch_idx + 1
                interval = max(1, n_batches_total // 10)
                if done == 1 or done == n_batches_total or done % interval == 0:
                    elapsed = time.time() - start
                    rate = done / elapsed if elapsed > 0 else 0.0
                    eta_sec = (n_batches_total - done) / rate if rate > 0 else 0.0
                    logger.info(
                        "Progress: batch %d/%d | results this rank: %d | elapsed %.1f min | ETA ~%.1f min",
                        done,
                        n_batches_total,
                        len(local_results),
                        elapsed / 60.0,
                        eta_sec / 60.0,
                    )

        # Save per-rank results to temp files
        tmp = Path(tempfile.gettempdir())
        rank_file = tmp / f"eval_results_rank_{accelerator.process_index}.json"
        with open(rank_file, "w") as f:
            json.dump(local_results, f)
        _distributed_barrier_safe()

        out_dir = Path(args.output).parent
        out_dir.mkdir(parents=True, exist_ok=True)

        if accelerator.is_main_process:
            gathered = []
            for rank in range(accelerator.num_processes):
                rf = tmp / f"eval_results_rank_{rank}.json"
                if rf.exists():
                    dest_gather = out_dir / f"eval_gather_rank_{rank}.json"
                    shutil.copy2(rf, dest_gather)
                    with open(rf) as f:
                        gathered.extend(json.load(f))
                    rf.unlink()
            logger.info(
                "Copied per-rank gather snapshots next to output → %s/eval_gather_rank_*.json",
                out_dir,
            )

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
            _merge_distributed_hypergraph_partials(args)
            logger.info(
                "Distributed artifacts under %s: merged %s + %s_hypergraphs.json; "
                "per-rank %s_rank_*.json, %s_rank_*_hypergraphs.json, eval_gather_rank_*.json (copies from gather).",
                out_dir,
                Path(args.output).name,
                Path(args.output).stem,
                Path(args.output).stem,
                Path(args.output).stem,
            )
        return 0

    @staticmethod
    def _early_bind_cuda_before_accelerate() -> None:
        """Set default CUDA device from ``LOCAL_RANK`` before ``Accelerator()`` is constructed.

        Accelerate and PyTorch may touch CUDA while building the ``Accelerator``, which defaults
        to device 0; all ranks then reserve GPU 0 before vLLM loads.
        """
        if "LOCAL_RANK" not in os.environ:
            return
        lr = int(os.environ["LOCAL_RANK"])
        ws = int(os.environ.get("WORLD_SIZE", "1"))
        ndev = torch.cuda.device_count()
        if ws <= 1 or ndev <= 0:
            return
        if ndev >= ws:
            torch.cuda.set_device(lr)
        elif ndev == 1:
            torch.cuda.set_device(0)
        else:
            torch.cuda.set_device(lr % ndev)
        logger.info(
            f"CUDA (pre-Accelerate): set_device for rank — WORLD_SIZE={ws}, LOCAL_RANK={lr}, "
            f"visible_devices={ndev}"
        )

    def _bind_cuda_device_for_distributed_rank(self) -> None:
        """Bind this process to one GPU before vLLM init (one full model per rank).

        **Important:** Inside each worker, ``Accelerator.num_processes`` is often **1** even when
        the job launched N processes — if we only looked at that, every rank would take the
        single-process branch and call ``set_device(0)`` → all vLLM instances on physical GPU 0.

        We use ``max(WORLD_SIZE, num_processes)`` and ``LOCAL_RANK`` (from ``torchrun`` /
        ``accelerate launch --multi_gpu``). If each rank only *sees* one device (remapped
        ``cuda:0``), ``ndev==1`` and we still call ``set_device(0)`` (correct per process).
        """
        if not torch.cuda.is_available():
            return
        accel = self._accelerator
        ndev = torch.cuda.device_count()
        if ndev <= 0:
            return

        ws = int(os.environ.get("WORLD_SIZE", "1"))
        nproc = int(accel.num_processes)
        workers = max(ws, nproc)
        lr = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", accel.local_process_index)))

        if workers <= 1:
            dev = accel.device
            if dev.type == "cuda":
                idx = dev.index
                if idx is None:
                    idx = 0 if ndev <= 1 else int(accel.local_process_index)
                torch.cuda.set_device(idx)
            return

        if ndev >= workers:
            torch.cuda.set_device(lr)
        elif ndev == 1:
            torch.cuda.set_device(0)
        else:
            torch.cuda.set_device(lr % ndev)
            if workers > ndev:
                logger.warning(
                    f"More workers ({workers}) than visible GPUs ({ndev}); "
                    f"rank {lr} → device {lr % ndev} (multiple vLLM per GPU may OOM)."
                )

        logger.info(
            f"CUDA: bound rank for vLLM — workers={workers}, lr={lr}, visible_devices={ndev}, "
            f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}, LOCAL_RANK={os.environ.get('LOCAL_RANK')}"
        )

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
        h = data.get("hypergraph")
        if h is None or (isinstance(h, str) and not h.strip()):
            hg = None
        else:
            hg = json.loads(h) if isinstance(h, str) else h
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
    def _collated_batch_to_samples(batch, include_gt) -> list[dict]:
        n = len(batch["filename"])
        samples: list[dict] = []
        for i in range(n):
            sample = {
                "filename": batch["filename"][i],
                "accessgraph": batch["accessgraph"][i],
                "num_nodes": batch["num_nodes"][i].item() if torch.is_tensor(batch["num_nodes"][i]) else batch["num_nodes"][i],
                "num_edges": batch["num_edges"][i].item() if torch.is_tensor(batch["num_edges"][i]) else batch["num_edges"][i],
            }
            if include_gt:
                sample["ground_truth"] = batch["ground_truth"][i]
            samples.append(sample)
        return samples

    @staticmethod
    def _batch_to_sample(batch, include_gt):
        return Evaluator._collated_batch_to_samples(batch, include_gt)[0]

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
        mode = getattr(args, "mode", "")
        if mode in ("trained", "rooms-only", "fewshot-qwen"):
            meta["vllm"] = not getattr(args, "no_vllm", False) and VLLM_AVAILABLE
            if getattr(args, "vllm_prompt_batch_size", 1) > 1:
                meta["vllm_prompt_batch_size"] = args.vllm_prompt_batch_size

        output_data = {"metadata": meta, "statistics": stats, "results": results}
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to: {args.output}")

        self._save_hypergraphs(args, results)

    @staticmethod
    def _save_hypergraphs(args, results):
        hgs = {}
        for r in results:
            gh = r.get("generated_hypergraph")
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
        use_vllm = not getattr(self.args, "no_vllm", False) and VLLM_AVAILABLE
        if not getattr(self.args, "no_vllm", False) and not VLLM_AVAILABLE:
            logger.warning("vLLM not installed; using HuggingFace generate (pip install vllm)")

        if use_vllm:
            if is_lora:
                with open(model_path / "adapter_config.json") as f:
                    cfg = json.load(f)
                base_name = cfg.get("base_model_name_or_path")
                if not base_name:
                    raise ValueError("Missing base_model_name_or_path in adapter_config.json")
                logger.info(f"LoRA adapter detected, base model: {base_name}")
                self.generator = VllmQwenGenerator(
                    base_name,
                    lora_path=model_path,
                    max_tokens=self.args.max_tokens,
                    temperature=self.args.temperature,
                    gpu_memory_utilization=self.args.vllm_gpu_memory,
                    max_model_len=self.args.vllm_max_model_len,
                    tensor_parallel_size=self.args.tensor_parallel,
                    enforce_eager=not getattr(self.args, "vllm_no_enforce_eager", False),
                )
            else:
                self.generator = VllmQwenGenerator(
                    str(model_path),
                    lora_path=None,
                    max_tokens=self.args.max_tokens,
                    temperature=self.args.temperature,
                    gpu_memory_utilization=self.args.vllm_gpu_memory,
                    max_model_len=self.args.vllm_max_model_len,
                    tensor_parallel_size=self.args.tensor_parallel,
                    enforce_eager=not getattr(self.args, "vllm_no_enforce_eager", False),
                )
            self._raw_model = None
            logger.info("vLLM generator ready (one engine per process / GPU)")
            return

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
        from utils.prompts import create_simple_prompt
        return create_simple_prompt(sample["accessgraph"])

    def extract_hypergraph(self, response):
        return Hypergraph.from_qwen_response(response)

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
                # Avoid hg.validate(..., debug=True): it logs huge graph dumps every sample and
                # breaks tqdm; GED is computed here only.
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
        r.update({"room_count_accuracy": 0.0, "room_type_accuracy": 0.0, "room_matching": None})
        return r

    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        use_vllm = not getattr(self.args, "no_vllm", False) and VLLM_AVAILABLE
        if not getattr(self.args, "no_vllm", False) and not VLLM_AVAILABLE:
            logger.warning("vLLM not installed; using HuggingFace generate (pip install vllm)")

        if use_vllm:
            self.generator = VllmQwenGenerator(
                self.args.model,
                lora_path=None,
                max_tokens=self.args.max_tokens,
                temperature=self.args.temperature,
                gpu_memory_utilization=self.args.vllm_gpu_memory,
                max_model_len=self.args.vllm_max_model_len,
                tensor_parallel_size=self.args.tensor_parallel,
                enforce_eager=not getattr(self.args, "vllm_no_enforce_eager", False),
            )
            self._raw_model = None
            logger.info("vLLM generator ready (one engine per process / GPU)")
            return

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
        result["room_count_accuracy"] = rm["room_count_accuracy"]
        result["room_type_accuracy"] = rm["room_type_accuracy"]
        if rm["room_type_accuracy"] == 1.0:
            result["success"] = True

    def compute_statistics(self, results):
        json_failed = [r for r in results if "Failed to extract JSON" in str(r.get("error", ""))]
        timed_out = [r for r in results if r.get("timed_out")]
        json_ok = [r for r in results if r not in json_failed and r not in timed_out]
        valid = [r for r in results if r.get("validation", {}).get("is_valid")]
        count_match = [r for r in results if r.get("room_count_accuracy", 0) == 1.0]
        types_match = [r for r in results if r.get("room_type_accuracy", 0) == 1.0]

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
        r.update({"json_extracted": False, "ged_computed": False, "ged": None,
                   "error_type": None, "timestamp": datetime.now().isoformat()})
        return r

    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        gpu = getattr(self.args, "gpu", 0)
        use_vllm = not getattr(self.args, "no_vllm", False) and VLLM_AVAILABLE
        if not getattr(self.args, "no_vllm", False) and not VLLM_AVAILABLE:
            logger.warning("vLLM not installed; using HuggingFace generate (pip install vllm)")

        if use_vllm:
            tp = getattr(self.args, "tensor_parallel", 1)
            if tp > 1:
                gpu_list = ",".join(str(gpu + i) for i in range(tp))
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
                logger.info(f"vLLM tensor parallel: CUDA_VISIBLE_DEVICES={gpu_list}")
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            self.generator = VllmQwenGenerator(
                self.args.model,
                lora_path=None,
                max_tokens=self.args.max_tokens,
                temperature=self.args.temperature,
                gpu_memory_utilization=self.args.vllm_gpu_memory,
                max_model_len=self.args.vllm_max_model_len,
                tensor_parallel_size=tp,
                enforce_eager=not getattr(self.args, "vllm_no_enforce_eager", False),
            )
            logger.info("vLLM generator loaded")
            return

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

        if self.args.num_examples and self.args.num_examples > 0:
            examples = load_example_hypergraphs(
                self.args.dataset_dir, self.args.num_examples,
                exclude_file=sample["filename"], seed=self.args.seed,
            )
        else:
            examples = []
        return create_structured_prompt(examples, sample["accessgraph"])

    def extract_hypergraph(self, response):
        hg = Hypergraph.from_qwen_response(response)
        if hg is not None:
            hg = Hypergraph(repair_bsp_tree_names(hg.data))
        return hg

    def score_result(self, result, hg, sample):
        result.pop("_ged_info", None)
        result["json_extracted"] = True
        from utils import compute_ged_from_hypergraph
        bounds = sample.get("bounds") or {}
        try:
            ged, _ = compute_ged_from_hypergraph(
                sample["accessgraph"], hg.data, sample["filename"],
                correct_accessgraph=bool(bounds.get("corners")),
                bounds=bounds,
            )
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

        if self.args.num_examples and self.args.num_examples > 0:
            examples = load_example_hypergraphs(
                self.args.dataset_dir, self.args.num_examples,
                exclude_file=sample["filename"], seed=self.args.seed,
            )
        else:
            examples = []
        return create_structured_prompt(examples, sample["accessgraph"])

    def extract_hypergraph(self, response):
        hg = Hypergraph.from_gpt_response(response)
        if hg is not None:
            hg = Hypergraph(repair_bsp_tree_names(hg.data))
        return hg

    def score_result(self, result, hg, sample):
        result.pop("_ged_info", None)
        result["json_extracted"] = True
        from utils import compute_ged_from_hypergraph
        bounds = sample.get("bounds") or {}
        try:
            ged, _ = compute_ged_from_hypergraph(
                sample["accessgraph"], hg.data, sample["filename"],
                correct_accessgraph=bool(bounds.get("corners")),
                bounds=bounds,
            )
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
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--slow-gen-threshold", type=float, default=300.0)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--nccl-timeout", type=int, default=7200)
    parser.add_argument(
        "--retry-on-json-fail",
        action="store_true",
        help=("Issue a second LLM call when JSON extraction fails on the "
              "first response. OFF by default: at the concurrencies we run "
              "these scripts at the marginal recovery rate doesn't justify "
              "the ~2x cost on retried samples."),
    )

    # Few-shot (set --num-examples 0 for zero-shot prompting)
    parser.add_argument(
        "--num-examples",
        type=int,
        default=3,
        help="Number of in-context examples for fewshot-* modes. Use 0 for zero-shot.",
    )
    parser.add_argument("--gpu", type=int, default=0)

    vllm = parser.add_argument_group("vLLM (Qwen local inference; default on if installed)")
    vllm.add_argument(
        "--no-vllm",
        action="store_true",
        help="Use HuggingFace generate instead of vLLM",
    )
    vllm.add_argument(
        "--vllm-gpu-memory",
        type=float,
        default=0.85,
        help=(
            "vLLM gpu_memory_utilization (default: 0.85 for inference-only). "
            "Lower values leave more headroom for the KV cache; raise toward 0.9 if needed."
        ),
    )
    vllm.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=6900,
        help="vLLM max_model_len (default: 6900)",
    )
    vllm.add_argument(
        "--tensor-parallel",
        type=int,
        default=1,
        help="vLLM tensor_parallel_size (default: 1)",
    )
    vllm.add_argument(
        "--vllm-no-enforce-eager",
        action="store_true",
        help="Allow vLLM CUDA graphs (default is enforce_eager=True to avoid OOM during capture)",
    )
    vllm.add_argument(
        "--vllm-prompt-batch-size",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Distributed vLLM only: number of samples per GPU per vLLM generate() call (default: 1). "
            "Try 2 for higher throughput; may OOM on tight VRAM."
        ),
    )

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