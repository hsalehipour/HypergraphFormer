#!/usr/bin/env python3
"""
Few-Shot Hypergraph Generator -- OOP consolidation.

Provides ``QwenGenerator`` and ``GPTGenerator`` for generating floor-plan
hypergraphs from access graphs using few-shot prompting.

Usage:
    # Single-sample Qwen (interactive, streaming)
    python scripts/fewshot_generator.py --backend qwen --from-dataset

    # Single-sample GPT
    python scripts/fewshot_generator.py --backend gpt --from-dataset --model gpt-4

    # Batch generation
    python scripts/fewshot_generator.py --backend gpt --batch \
        --test-dir data/test --output results.json --resume

    # Qwen via vLLM (default if installed; use --no-vllm for HuggingFace)
    python scripts/fewshot_generator.py --backend qwen --batch --test-dir data/test \\
        --output results.json --vllm-gpu-memory 0.85 --vllm-max-model-len 6900
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# See scripts/evaluator.py — set before importing vllm below.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from utils.hypergraph import Hypergraph
from utils.prompts import create_structured_prompt
from utils.data_loading import load_example_hypergraphs, load_target_from_dataset

logger = logging.getLogger(__name__)

# vLLM (optional; used for fast local inference — see rlft_pipeline GRPOConfig / evaluate_rl_model.py)
VLLM_AVAILABLE = False
try:
    from vllm import LLM as _VLLM_LLM
    from vllm import SamplingParams as _VLLM_SamplingParams
    from vllm.lora.request import LoRARequest as _VLLM_LoRARequest

    VLLM_AVAILABLE = True
except ImportError:
    _VLLM_LLM = _VLLM_SamplingParams = _VLLM_LoRARequest = None  # type: ignore[misc, assignment]

SYSTEM_PROMPT = (
    "You are an expert at generating structured hierarchical BSP tree data "
    "for floor plans in JSON format."
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """Unified container for a single generation call."""
    text: Optional[str] = None
    input_tokens: int = 0
    output_tokens: Optional[int] = None
    generation_time: float = 0.0
    timed_out: bool = False


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class FewShotGenerator(ABC):
    """Abstract base for few-shot hypergraph generators."""

    def __init__(self, model_name: str, *, max_tokens: int = 5120,
                 temperature: float = 0.7):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    # -- abstract interface --------------------------------------------------

    @abstractmethod
    def generate(self, prompt: str, *, max_tokens: int | None = None,
                 temperature: float | None = None,
                 timeout: int | None = None) -> GenerationResult:
        """Non-streaming generation (used by evaluators)."""

    @abstractmethod
    def generate_streaming(self, prompt: str, *, max_tokens: int | None = None,
                           temperature: float | None = None) -> GenerationResult:
        """Streaming generation with progress display."""

    @abstractmethod
    def extract_hypergraph(self, response: str) -> Hypergraph | None:
        """Extract a ``Hypergraph`` from the raw model response."""

    # -- shared helpers ------------------------------------------------------

    def load_examples(self, examples_dir, num_examples=3, exclude_file=None,
                      seed=None):
        return load_example_hypergraphs(
            examples_dir, num_examples, exclude_file=exclude_file, seed=seed,
        )

    @staticmethod
    def build_prompt(examples, target_accessgraph=None):
        return create_structured_prompt(examples, target_accessgraph)

    def validate_and_log(self, hg, target_accessgraph, *, debug=False):
        is_valid, passed, failed, ged_info = hg.validate(
            target_accessgraph, debug=debug,
        )

        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"\nPASSED CHECKS ({len(passed)}):")
        for c in passed:
            logger.info(f"  {c}")
        if failed:
            logger.warning(f"\nFAILED CHECKS ({len(failed)}):")
            for c in failed:
                logger.warning(f"  {c}")
        else:
            logger.info("\nFAILED CHECKS (0): None")
        logger.info("=" * 80)
        if is_valid:
            logger.info("Overall: VALID")
        else:
            logger.warning(f"Overall: INVALID -- {len(failed)} check(s) failed")
        logger.info("=" * 80 + "\n")

        if ged_info:
            logger.info("=" * 80)
            logger.info("GRAPH EDIT DISTANCE ANALYSIS")
            logger.info("=" * 80)
            logger.info(f"Raw GED: {ged_info['ged']:.2f}")
            logger.info("=" * 80 + "\n")

        return is_valid, passed, failed, ged_info

    # -- CLI runners ---------------------------------------------------------

    def run_single(self, args) -> int:
        """Single-sample generation (interactive)."""
        target_accessgraph = None
        target_filename = None
        ground_truth_hypergraph = None

        if args.from_dataset:
            target_dir = getattr(args, "target_dir", None) or args.examples_dir
            logger.info(f"Loading target from dataset: {target_dir}")
            try:
                target_accessgraph, target_filename, ground_truth_hypergraph = (
                    load_target_from_dataset(target_dir, args.target_file)
                )
                nodes = len(target_accessgraph.get("nodes", {}))
                edges = len(target_accessgraph.get("edges", []))
                logger.info(f"Loaded target with {nodes} nodes, {edges} edges")
            except Exception as e:
                logger.error(f"Failed to load target from dataset: {e}")
                return 1
        elif args.accessgraph:
            logger.info(f"Loading target access graph from: {args.accessgraph}")
            try:
                with open(args.accessgraph, "r") as f:
                    target_accessgraph = json.load(f)
                nodes = len(target_accessgraph.get("nodes", {}))
                edges = len(target_accessgraph.get("edges", []))
                logger.info(f"Loaded access graph with {nodes} nodes, {edges} edges")
            except Exception as e:
                logger.error(f"Failed to load access graph: {e}")
                return 1
        else:
            logger.info("No target access graph -- will generate new floor plan")

        examples = self.load_examples(
            args.examples_dir, args.num_examples, exclude_file=target_filename,
        )
        logger.info(f"Loaded examples: {', '.join(e['filename'] for e in examples)}")

        prompt = self.build_prompt(examples, target_accessgraph)

        with open("last_prompt.txt", "w") as f:
            f.write(prompt)
        logger.info("Prompt saved to: last_prompt.txt")

        logger.info("Generating hypergraph...")
        result = self.generate_streaming(prompt)

        if result.text is None:
            logger.error("Generation returned no text")
            return 1

        with open("last_response.txt", "w") as f:
            f.write(result.text)
        logger.info("Full response saved to: last_response.txt")

        hg = self.extract_hypergraph(result.text)
        if hg is None:
            logger.error("Failed to extract valid JSON from response")
            return 1

        logger.info("Successfully extracted JSON")

        is_valid, _, _, ged_info = self.validate_and_log(
            hg, target_accessgraph, debug=args.debug,
        )

        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(hg.data, f, indent=2)
        logger.info(f"Generated hypergraph saved to: {output_path}")

        logger.info("=" * 80)
        logger.info("GENERATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Examples used: {args.num_examples}")
        logger.info(f"Input tokens: {result.input_tokens:,}")
        logger.info(f"Generation time: {result.generation_time:.1f}s")
        if target_accessgraph and target_filename:
            nodes = len(target_accessgraph.get("nodes", {}))
            edges = len(target_accessgraph.get("edges", []))
            logger.info(f"Target: {target_filename} ({nodes} nodes, {edges} edges)")
        logger.info(f"Output: {output_path}")
        logger.info(f"Valid: {is_valid}")
        if ged_info and target_accessgraph:
            logger.info(f"GED: {ged_info['ged']:.2f}")
        logger.info("=" * 80)

        return 0 if is_valid else 1

    def run_batch(self, args) -> int:
        """Batch generation over a test directory with resume support."""
        from tqdm import tqdm

        logger.info("=" * 80)
        logger.info("Batch Hypergraph Generation")
        logger.info("=" * 80)
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Test directory: {args.test_dir}")
        logger.info(f"Few-shot examples: {args.num_examples}")
        logger.info(f"Output: {args.output}")
        logger.info("=" * 80)

        test_dir = Path(args.test_dir)
        test_files = sorted(test_dir.glob("*.txt"))
        samples = []
        for fp in test_files:
            try:
                with open(fp, "r") as f:
                    data = json.load(f)
                ag = data.get("accessgraph", {})
                hg_str = data.get("hypergraph", "")
                hg = json.loads(hg_str) if isinstance(hg_str, str) else hg_str
                samples.append({
                    "filename": fp.name,
                    "accessgraph": ag,
                    "ground_truth": hg,
                })
            except Exception as e:
                logger.warning(f"Skipping {fp.name}: {e}")
        logger.info(f"Loaded {len(samples)} test samples")

        results: dict = {}
        if args.resume and Path(args.output).exists():
            with open(args.output, "r") as f:
                results = json.load(f)
            logger.info(f"Resumed {len(results)} existing results")

        failed_samples: list[str] = []
        start_time = time.time()

        with tqdm(total=len(samples), desc="Generating", unit="sample") as pbar:
            for sample in samples:
                filename = sample["filename"]
                sample_id = filename.replace(".txt", "")

                if sample_id in results:
                    pbar.update(1)
                    continue

                pbar.set_description(f"Processing {filename}")
                try:
                    examples = self.load_examples(
                        args.test_dir, args.num_examples,
                        exclude_file=filename,
                    )
                    prompt = self.build_prompt(examples, sample["accessgraph"])
                    gen = self.generate(prompt)

                    hg = self.extract_hypergraph(gen.text) if gen.text else None
                    if hg is None:
                        logger.warning(f"Failed to extract JSON for {filename}")
                        failed_samples.append(filename)
                        pbar.update(1)
                        continue

                    is_valid, _, _, _ = hg.validate(
                        sample["accessgraph"], debug=True,
                    )
                    results[sample_id] = hg.data

                    if len(results) % 10 == 0:
                        with open(args.output, "w") as f:
                            json.dump(results, f, indent=2)

                    status = "Valid" if is_valid else "Invalid"
                    pbar.set_postfix_str(f"{status} | Total: {len(results)}")

                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
                    failed_samples.append(filename)

                pbar.update(1)

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

        elapsed = time.time() - start_time
        logger.info("=" * 80)
        logger.info("BATCH SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total: {len(samples)}")
        logger.info(f"Generated: {len(results)}")
        logger.info(f"Failed: {len(failed_samples)}")
        if samples:
            logger.info(f"Success rate: {len(results)/len(samples)*100:.1f}%")
            logger.info(f"Time: {elapsed/60:.1f} min ({elapsed/len(samples):.1f}s/sample)")
        if failed_samples:
            logger.info(f"Failed samples: {failed_samples}")
        logger.info("=" * 80)

        return 0 if not failed_samples else 1


# ---------------------------------------------------------------------------
# Qwen backend
# ---------------------------------------------------------------------------

class QwenGenerator(FewShotGenerator):
    """Local Qwen (HuggingFace) generator."""

    def __init__(self, model_name: str = "Qwen/Qwen3-4B-Instruct-2507", *,
                 model=None, tokenizer=None,
                 max_tokens: int = 5120, temperature: float = 0.7):
        super().__init__(model_name, max_tokens=max_tokens, temperature=temperature)
        self._model = model
        self._tokenizer = tokenizer

    def _ensure_loaded(self):
        if self._model is not None and self._tokenizer is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading tokenizer: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        logger.info(f"Loading model: {self.model_name}")
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16, device_map={"": 0},
        )
        logger.info("Model loaded")

    @property
    def model(self):
        self._ensure_loaded()
        return self._model

    @property
    def tokenizer(self):
        self._ensure_loaded()
        return self._tokenizer

    def generate(self, prompt: str, *, max_tokens: int | None = None,
                 temperature: float | None = None,
                 timeout: int | None = None) -> GenerationResult:
        from utils import TimeoutException, time_limit

        mt = max_tokens or self.max_tokens
        temp = temperature or self.temperature

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        num_input = inputs.input_ids.shape[1]

        gen_kwargs = dict(
            **inputs, max_new_tokens=mt, temperature=temp,
            do_sample=True, top_p=0.95, top_k=50,
        )

        start = time.time()
        try:
            if timeout is not None:
                with time_limit(timeout):
                    outputs = self.model.generate(**gen_kwargs)
            else:
                outputs = self.model.generate(**gen_kwargs)
            elapsed = time.time() - start
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            return GenerationResult(
                text=decoded, input_tokens=num_input,
                generation_time=elapsed,
            )
        except TimeoutException:
            elapsed = time.time() - start
            logger.error(f"TIMEOUT: Generation exceeded {timeout}s limit")
            return GenerationResult(
                input_tokens=num_input, generation_time=elapsed, timed_out=True,
            )

    def generate_streaming(self, prompt: str, *, max_tokens: int | None = None,
                           temperature: float | None = None) -> GenerationResult:
        from threading import Thread
        from tqdm import tqdm
        from transformers import TextIteratorStreamer

        mt = max_tokens or self.max_tokens
        temp = temperature or self.temperature

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        num_input = inputs.input_ids.shape[1]
        logger.info(f"Input tokens: {num_input:,}")

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True,
        )
        gen_kwargs = dict(
            **inputs, max_new_tokens=mt, temperature=temp,
            do_sample=True, top_p=0.95, top_k=50, streamer=streamer,
        )

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        start = time.time()
        thread.start()

        generated = ""
        with tqdm(total=mt, desc="Generating tokens", unit="token",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
            for chunk in streamer:
                generated += chunk
                pbar.update(1)
        thread.join()
        elapsed = time.time() - start

        full_response = text + generated
        return GenerationResult(
            text=full_response, input_tokens=num_input,
            generation_time=elapsed,
        )

    def extract_hypergraph(self, response: str) -> Hypergraph | None:
        return Hypergraph.from_qwen_response(response)


def _wrap_vllm_completion_for_qwen_parse(generated_text: str) -> str:
    """vLLM returns only new tokens; ``Hypergraph.from_qwen_response`` looks for the assistant span."""
    g = (generated_text or "").strip()
    marker = "<|im_start|>assistant"
    if marker in g:
        return g
    return f"{marker}\n{g}\n<|redacted_im_end|>"


class VllmQwenGenerator(FewShotGenerator):
    """Local Qwen via vLLM (faster batched inference; aligns with rlft_pipeline / evaluate_rl_model)."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        *,
        lora_path: str | Path | None = None,
        max_tokens: int = 5120,
        temperature: float = 0.7,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 6900,
        tensor_parallel_size: int = 1,
        max_lora_rank: int | None = None,
        enforce_eager: bool = True,
    ):
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM is not installed. Install with: pip install vllm")
        super().__init__(model_name, max_tokens=max_tokens, temperature=temperature)
        self._lora_path: Path | None = Path(lora_path).resolve() if lora_path else None
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._tensor_parallel_size = tensor_parallel_size

        tokenizer_src = str(self._lora_path) if self._lora_path and (self._lora_path / "adapter_config.json").exists() else model_name
        from transformers import AutoTokenizer

        _local = bool(self._lora_path) or (Path(model_name).exists() and Path(model_name).is_dir())
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, local_files_only=_local)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        base_model = model_name
        enable_lora = False
        if self._lora_path and (self._lora_path / "adapter_config.json").exists():
            with open(self._lora_path / "adapter_config.json") as f:
                cfg = json.load(f)
            base_model = cfg.get("base_model_name_or_path") or base_model
            enable_lora = True
            if max_lora_rank is None:
                max_lora_rank = int(cfg.get("r", 64))

        logger.info(
            f"Loading vLLM: base={base_model}, lora={self._lora_path}, "
            f"gpu_mem={gpu_memory_utilization}, max_model_len={max_model_len}, tp={tensor_parallel_size}",
        )
        _llm_kw: dict = dict(
            model=base_model,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
        )
        if enable_lora:
            _llm_kw["enable_lora"] = True
            _llm_kw["max_lora_rank"] = max_lora_rank or 64
        # Default True: CUDA graph capture/warmup can OOM on 40GB cards with long max_model_len.
        _llm_kw["enforce_eager"] = enforce_eager
        self._llm = _VLLM_LLM(**_llm_kw)

    @property
    def tokenizer(self):
        return self._tokenizer

    def _lora_request(self):
        if self._lora_path is None or not (self._lora_path / "adapter_config.json").exists():
            return None
        return _VLLM_LoRARequest(
            lora_name=self._lora_path.name,
            lora_int_id=hash(str(self._lora_path)) % (2**31),
            lora_path=str(self._lora_path),
        )

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: int | None = None,
    ) -> GenerationResult:
        from utils import TimeoutException, time_limit

        mt = max_tokens or self.max_tokens
        temp = temperature or self.temperature

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        num_input = len(self._tokenizer.encode(text))

        sampling_params = _VLLM_SamplingParams(
            temperature=temp,
            top_k=50,
            top_p=0.95,
            max_tokens=mt,
        )
        lora_req = self._lora_request()

        def _run_generate():
            if lora_req is not None:
                return self._llm.generate(
                    [text], sampling_params, lora_request=lora_req, use_tqdm=False,
                )
            return self._llm.generate([text], sampling_params, use_tqdm=False)

        start = time.time()
        try:
            if timeout is not None:
                with time_limit(timeout):
                    outputs = _run_generate()
            else:
                outputs = _run_generate()
            elapsed = time.time() - start
            raw = outputs[0].outputs[0].text
            wrapped = _wrap_vllm_completion_for_qwen_parse(raw)
            out_tokens = len(self._tokenizer.encode(raw)) if raw else 0
            return GenerationResult(
                text=wrapped,
                input_tokens=num_input,
                output_tokens=out_tokens,
                generation_time=elapsed,
            )
        except TimeoutException:
            elapsed = time.time() - start
            logger.error(f"TIMEOUT: Generation exceeded {timeout}s limit")
            return GenerationResult(
                input_tokens=num_input, generation_time=elapsed, timed_out=True,
            )

    def generate_batch(
        self,
        prompts: list[str],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: int | None = None,
    ) -> list[GenerationResult]:
        """Batched vLLM inference (one engine; multiple prompts per ``generate`` call)."""
        from utils import TimeoutException, time_limit

        if not prompts:
            return []
        mt = max_tokens or self.max_tokens
        temp = temperature or self.temperature

        texts: list[str] = []
        input_counts: list[int] = []
        for prompt in prompts:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            texts.append(text)
            input_counts.append(len(self._tokenizer.encode(text)))

        sampling_params = _VLLM_SamplingParams(
            temperature=temp,
            top_k=50,
            top_p=0.95,
            max_tokens=mt,
        )
        lora_req = self._lora_request()

        def _run_batch():
            if lora_req is not None:
                return self._llm.generate(
                    texts, sampling_params, lora_request=lora_req, use_tqdm=False,
                )
            return self._llm.generate(texts, sampling_params, use_tqdm=False)

        start = time.time()
        try:
            if timeout is not None:
                with time_limit(timeout):
                    outputs = _run_batch()
            else:
                outputs = _run_batch()
        except TimeoutException:
            elapsed = time.time() - start
            logger.error(f"TIMEOUT: batched generation exceeded {timeout}s limit")
            return [
                GenerationResult(
                    input_tokens=input_counts[i],
                    generation_time=elapsed / max(len(prompts), 1),
                    timed_out=True,
                )
                for i in range(len(prompts))
            ]

        elapsed = time.time() - start
        n = len(prompts)
        per = elapsed / n if n else 0.0
        out: list[GenerationResult] = []
        for i in range(n):
            raw = outputs[i].outputs[0].text
            wrapped = _wrap_vllm_completion_for_qwen_parse(raw)
            out_tokens = len(self._tokenizer.encode(raw)) if raw else 0
            out.append(
                GenerationResult(
                    text=wrapped,
                    input_tokens=input_counts[i],
                    output_tokens=out_tokens,
                    generation_time=per,
                )
            )
        return out

    def generate_streaming(self, prompt: str, *, max_tokens: int | None = None,
                           temperature: float | None = None) -> GenerationResult:
        logger.info("vLLM: non-streaming generate (streaming not wired; single-shot decode)")
        return self.generate(prompt, max_tokens=max_tokens, temperature=temperature, timeout=None)

    def extract_hypergraph(self, response: str) -> Hypergraph | None:
        return Hypergraph.from_qwen_response(response)


# ---------------------------------------------------------------------------
# GPT backend
# ---------------------------------------------------------------------------

class GPTGenerator(FewShotGenerator):
    """OpenAI API generator."""

    def __init__(self, model_name: str = "gpt-4", *, api_key: str | None = None,
                 max_tokens: int = 4096, temperature: float = 0.7):
        super().__init__(model_name, max_tokens=max_tokens, temperature=temperature)
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            if not self._api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY not set. Export it or pass api_key= to GPTGenerator."
                )
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def _messages(self, prompt: str):
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

    def generate(self, prompt: str, *, max_tokens: int | None = None,
                 temperature: float | None = None,
                 timeout: int | None = None) -> GenerationResult:
        mt = max_tokens or self.max_tokens
        temp = temperature or self.temperature
        msgs = self._messages(prompt)
        estimated_input = len(prompt) // 4

        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model_name, messages=msgs,
            max_tokens=mt, temperature=temp,
        )
        elapsed = time.time() - start

        text = response.choices[0].message.content
        return GenerationResult(
            text=text, input_tokens=estimated_input,
            output_tokens=len(text) // 4 if text else 0,
            generation_time=elapsed,
        )

    def generate_streaming(self, prompt: str, *, max_tokens: int | None = None,
                           temperature: float | None = None) -> GenerationResult:
        from tqdm import tqdm

        mt = max_tokens or self.max_tokens
        temp = temperature or self.temperature
        msgs = self._messages(prompt)
        estimated_input = len(prompt) // 4

        logger.info(f"Estimated input tokens: ~{estimated_input:,}")

        start = time.time()
        generated = ""

        use_new_api = (
            self.model_name.startswith("gpt-5")
            or self.model_name.startswith("o1")
            or self.model_name.startswith("o3")
        )

        create_kwargs = dict(model=self.model_name, messages=msgs, stream=True)
        if use_new_api:
            create_kwargs["max_completion_tokens"] = mt
        else:
            create_kwargs["max_tokens"] = mt
            create_kwargs["temperature"] = temp

        with tqdm(total=mt, desc="Generating tokens", unit="token",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]") as pbar:
            stream = self.client.chat.completions.create(**create_kwargs)
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    generated += content
                    pbar.update(len(content) // 4)

        elapsed = time.time() - start
        logger.info(f"Generation completed in {elapsed:.2f}s")

        return GenerationResult(
            text=generated, input_tokens=estimated_input,
            output_tokens=len(generated) // 4,
            generation_time=elapsed,
        )

    def extract_hypergraph(self, response: str) -> Hypergraph | None:
        return Hypergraph.from_gpt_response(response)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cli():
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Few-shot hypergraph generation (Qwen / GPT)",
    )
    parser.add_argument("--backend", choices=["qwen", "gpt"], required=True)
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (default: backend-dependent)")
    parser.add_argument("--examples-dir", type=str,
                        default="training_data/extracted_hypergraph_msd")
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=5120)
    parser.add_argument("--output", type=str, default="generated_hypergraph.json")
    parser.add_argument("--debug", action="store_true")

    single = parser.add_argument_group("single-sample mode")
    single.add_argument("--from-dataset", action="store_true")
    single.add_argument("--target-file", type=str, default=None)
    single.add_argument("--target-dir", type=str, default=None,
                        help="Target directory (GPT: separate test dir)")
    single.add_argument("--accessgraph", type=str, default=None)

    batch = parser.add_argument_group("batch mode")
    batch.add_argument("--batch", action="store_true")
    batch.add_argument("--test-dir", type=str, default=None)
    batch.add_argument("--resume", action="store_true")

    vllm = parser.add_argument_group("vLLM (Qwen backend)")
    vllm.add_argument(
        "--no-vllm",
        action="store_true",
        help="Use HuggingFace generate instead of vLLM (slower)",
    )
    vllm.add_argument(
        "--vllm-gpu-memory",
        type=float,
        default=0.85,
        help=(
            "vLLM gpu_memory_utilization (default: 0.85 for inference). "
            "Use ~0.25 only when sharing the GPU (e.g. GRPO colocate); too low causes KV cache OOM."
        ),
    )
    vllm.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=6900,
        help="vLLM max_model_len (default: 6900, matches rlft_pipeline GRPOConfig)",
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

    args = parser.parse_args()

    model_name = args.model
    if args.backend == "qwen":
        model_name = model_name or "Qwen/Qwen3-4B-Instruct-2507"
        use_vllm = not args.no_vllm and VLLM_AVAILABLE
        if not args.no_vllm and not VLLM_AVAILABLE:
            logger.warning("vLLM not installed; falling back to HuggingFace generate (pip install vllm)")
        if use_vllm:
            gen = VllmQwenGenerator(
                model_name,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                gpu_memory_utilization=args.vllm_gpu_memory,
                max_model_len=args.vllm_max_model_len,
                tensor_parallel_size=args.tensor_parallel,
                enforce_eager=not args.vllm_no_enforce_eager,
            )
        else:
            gen = QwenGenerator(model_name, max_tokens=args.max_tokens,
                                temperature=args.temperature)
    else:
        model_name = model_name or "gpt-4"
        gen = GPTGenerator(model_name, max_tokens=args.max_tokens,
                           temperature=args.temperature)

    if args.batch:
        if not args.test_dir:
            parser.error("--batch requires --test-dir")
        return gen.run_batch(args)
    return gen.run_single(args)


if __name__ == "__main__":
    sys.exit(cli())
