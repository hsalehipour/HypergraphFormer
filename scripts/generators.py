#!/usr/bin/env python3
"""
Hypergraph generator classes (library module, no CLI).

Provides ``QwenGenerator``, ``VllmQwenGenerator``, and ``GPTGenerator`` --
prompt-agnostic inference engines for generating floor-plan hypergraphs.

Used by ``scripts/evaluator.py`` for evaluation::

    python scripts/evaluator.py --mode fewshot-qwen --dataset-dir data/test
    python scripts/evaluator.py --mode trained --model checkpoints/lora
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# See scripts/evaluator.py — set before importing vllm below.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from utils.hypergraph import Hypergraph

logger = logging.getLogger(__name__)

# vLLM (optional; used for fast local inference)
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

class HypergraphGenerator(ABC):
    """Abstract base for hypergraph generators."""

    def __init__(self, model_name: str, *, max_tokens: int = 5120,
                 temperature: float = 0.7):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

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


# ---------------------------------------------------------------------------
# Qwen backend
# ---------------------------------------------------------------------------

class QwenGenerator(HypergraphGenerator):
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


class VllmQwenGenerator(HypergraphGenerator):
    """Local Qwen via vLLM (faster batched inference)."""

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
        # use resolved absolute path to avoid huggingface_hub repo-id validation on relative paths
        if _local and not tokenizer_src.startswith("/"):
            tokenizer_src = str(Path(tokenizer_src).resolve())
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

class GPTGenerator(HypergraphGenerator):
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

    def _openai_uses_max_completion_tokens(self) -> bool:
        """``gpt-5*``, ``o1*``, ``o3*`` chat models reject ``max_tokens``; use ``max_completion_tokens``."""
        name = self.model_name
        return name.startswith("gpt-5") or name.startswith("o1") or name.startswith("o3")

    def generate(self, prompt: str, *, max_tokens: int | None = None,
                 temperature: float | None = None,
                 timeout: int | None = None) -> GenerationResult:
        mt = max_tokens or self.max_tokens
        temp = temperature or self.temperature
        msgs = self._messages(prompt)
        estimated_input = len(prompt) // 4

        create_kwargs: dict = dict(model=self.model_name, messages=msgs)
        if self._openai_uses_max_completion_tokens():
            create_kwargs["max_completion_tokens"] = mt
        else:
            create_kwargs["max_tokens"] = mt
            create_kwargs["temperature"] = temp

        start = time.time()
        response = self.client.chat.completions.create(**create_kwargs)
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

        create_kwargs = dict(model=self.model_name, messages=msgs, stream=True)
        if self._openai_uses_max_completion_tokens():
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

