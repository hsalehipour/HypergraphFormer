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
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import sysconfig
import time
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils.hypergraph import Hypergraph
from utils.prompts import create_structured_prompt
from utils.data_loading import load_example_hypergraphs, load_target_from_dataset

logger = logging.getLogger(__name__)

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
                time.sleep(1)

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
        import threading

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

        # Use a daemon thread so the timeout works from any thread (including
        # ThreadPoolExecutor workers).  signal.alarm() only works in the main
        # thread, so we can't use time_limit() here.
        _outputs: list = []
        _exc: list = []

        def _run():
            try:
                _outputs.append(self.model.generate(**gen_kwargs))
            except Exception as e:
                _exc.append(e)

        t = threading.Thread(target=_run, daemon=True)
        start = time.time()
        t.start()
        t.join(timeout=timeout)
        elapsed = time.time() - start

        if t.is_alive():
            logger.error(f"TIMEOUT: Generation exceeded {timeout}s limit")
            return GenerationResult(
                input_tokens=num_input, generation_time=elapsed, timed_out=True,
            )

        if _exc:
            raise _exc[0]

        decoded = self.tokenizer.decode(_outputs[0][0], skip_special_tokens=False)
        return GenerationResult(
            text=decoded, input_tokens=num_input,
            generation_time=elapsed,
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


# ---------------------------------------------------------------------------
# vLLM backend (high-throughput batched inference)
# ---------------------------------------------------------------------------

def _deb_arch() -> str:
    m = platform.machine().lower()
    return {"x86_64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(m, m)


def _ensure_python_include_for_vllm_triton() -> Path:
    """
    vLLM / Triton JIT-compile a small CUDA helper; gcc needs Python.h.

    On minimal or broken images ``python3.10-dev`` is marked installed but
    ``/usr/include/python3.X`` is empty.      We recover by extracting headers from Ubuntu's ``libpython3.10-dev`` .deb
    into ``.cache/`` (no root, no apt).  The ``python3.10-dev`` metapackage
    does not ship ``Python.h``.
    """
    inc_sys = Path(sysconfig.get_path("include"))
    if (inc_sys / "Python.h").is_file():
        return inc_sys

    py_mm = f"{sys.version_info.major}.{sys.version_info.minor}"
    proj_root = Path(__file__).resolve().parent.parent
    extract_root = proj_root / ".cache" / "python3-dev-extract"
    inc_extracted = extract_root / "usr" / "include" / f"python{py_mm}"
    if (inc_extracted / "Python.h").is_file():
        return inc_extracted

    arch = _deb_arch()
    # Headers live in libpythonX.Y-dev, not the pythonX.Y-dev metapackage (docs only).
    deb_pkg = f"libpython{py_mm}-dev"
    # Ubuntu rotates minor revisions; older .deb URLs 404.  Keep newest first.
    deb_versions = [
        "3.10.12-1~22.04.17",
        "3.10.12-1~22.04.16",
        "3.10.12-1~22.04.15",
        "3.10.12-1~22.04.14",
        "3.10.12-1~22.04.13",
        "3.10.12-1~22.04.11",
        "3.10.12-1~22.04.10",
        "3.10.12-1~22.04.9",
        "3.10.12-1~22.04.8",
        "3.10.12-1~22.04.7",
        "3.10.12-1~22.04.6",
        "3.10.12-1~22.04.5",
        "3.10.12-1~22.04.4",
        "3.10.12-1~22.04.3",
        "3.10.12-1~22.04.2",
        "3.10.12-1~22.04.1",
    ]
    deb_path = proj_root / ".cache" / f"{deb_pkg}_{arch}.deb"
    deb_path.parent.mkdir(parents=True, exist_ok=True)

    last_err: Exception | None = None
    for ver in deb_versions:
        fname = f"{deb_pkg}_{ver}_{arch}.deb"
        for base in (
            f"http://archive.ubuntu.com/ubuntu/pool/main/p/python{py_mm}/",
            f"http://security.ubuntu.com/ubuntu/pool/main/p/python{py_mm}/",
        ):
            url = base + fname
            try:
                logger.info("Fetching missing Python headers from %s", url)
                urllib.request.urlretrieve(url, deb_path)
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                continue
        if last_err is None:
            break
    if last_err is not None:
        raise RuntimeError(
            "Python development headers (Python.h) are missing and could not "
            "be downloaded automatically. Install on the host/container, e.g.:\n"
            "  apt-get update && apt-get install -y python3.10-dev\n"
            f"(sysconfig include path was: {inc_sys})"
        ) from last_err

    extract_root.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["dpkg-deb", "-x", str(deb_path), str(extract_root)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Extracting python3-dev .deb requires `dpkg-deb` (install dpkg)."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"dpkg-deb failed: {exc.stderr or exc.stdout or exc}"
        ) from exc

    if not (inc_extracted / "Python.h").is_file():
        raise RuntimeError(
            f"Extracted .deb but Python.h not at {inc_extracted}. "
            "Try apt-get install python3.10-dev after apt-get update."
        )
    logger.info("Using Python headers from extracted .deb at %s", inc_extracted)
    return inc_extracted


def _multiarch_lib_dir() -> str:
    m = platform.machine().lower()
    if m in ("x86_64", "amd64"):
        return "/usr/lib/x86_64-linux-gnu"
    if m in ("aarch64", "arm64"):
        return "/usr/lib/aarch64-linux-gnu"
    return "/usr/lib/x86_64-linux-gnu"


def _prepare_vllm_subprocess_compile_env(include_dir: Path) -> None:
    """Ensure gcc / Triton child processes can find Python and libcuda."""
    # include_dir is .../usr/include/python3.10; multiarch pyconfig lives under
    # .../usr/include/x86_64-linux-gnu/python3.10/ and is pulled in via
    # #include <x86_64-linux-gnu/python3.10/pyconfig.h> — parent must be on CPATH.
    usr_include = str(include_dir.parent.resolve())
    py_inc = str(include_dir.resolve())
    lib = _multiarch_lib_dir()
    combined_inc = f"{usr_include}:{py_inc}"
    for key in ("CPATH", "C_INCLUDE_PATH"):
        prev = os.environ.get(key, "")
        os.environ[key] = combined_inc if not prev else f"{combined_inc}:{prev}"
    prev_l = os.environ.get("LIBRARY_PATH", "")
    os.environ["LIBRARY_PATH"] = lib if not prev_l else f"{lib}:{prev_l}"


class VLLMQwenGenerator:
    """
    vLLM-based generator for high-throughput batched inference.

    Unlike QwenGenerator (one sequence at a time), this submits N prompts in a
    single llm.generate() call.  vLLM runs them in parallel on the GPU using
    continuous batching and PagedAttention, giving ~5-10× higher GPU utilisation.

    Usage::

        gen = VLLMQwenGenerator(base_model="Qwen/...", lora_path="/path/to/ckpt")
        gen.load()                          # call once per process, after CUDA_VISIBLE_DEVICES is set
        results = gen.generate_batch(prompts, max_tokens=3328, temperature=0.7)
    """

    def __init__(
        self,
        base_model: str,
        *,
        lora_path: str | None = None,
        lora_rank: int = 64,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 8192,
        tensor_parallel_size: int = 1,
        enforce_eager: bool = False,
    ):
        self.base_model = base_model
        self.lora_path = lora_path
        self.lora_rank = lora_rank
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.enforce_eager = enforce_eager
        self._llm = None
        self._tokenizer = None
        self._lora_request = None

    def load(self):
        """Initialise the vLLM engine.  Must be called after CUDA_VISIBLE_DEVICES is set."""
        include_dir = _ensure_python_include_for_vllm_triton()
        _prepare_vllm_subprocess_compile_env(include_dir)

        from vllm import LLM
        from transformers import AutoTokenizer

        engine_kwargs: dict = dict(
            model=self.base_model,
            dtype="float16",
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            trust_remote_code=True,
            tensor_parallel_size=self.tensor_parallel_size,
        )
        if self.enforce_eager:
            engine_kwargs["enforce_eager"] = True
        if self.lora_path:
            engine_kwargs["enable_lora"] = True
            engine_kwargs["max_lora_rank"] = self.lora_rank

        logger.info(
            f"Loading vLLM engine: {self.base_model}  "
            f"LoRA: {self.lora_path}  "
            f"gpu_mem_util={self.gpu_memory_utilization}"
        )
        self._llm = LLM(**engine_kwargs)

        tokenizer_src = self.lora_path or self.base_model
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)

        if self.lora_path:
            from vllm.lora.request import LoRARequest
            self._lora_request = LoRARequest("adapter", 1, self.lora_path)

        logger.info("vLLM engine ready")

    # -- prompt formatting ---------------------------------------------------

    def _format_prompts(self, prompts: list[str]) -> list[str]:
        out = []
        for p in prompts:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": p},
            ]
            out.append(
                self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            )
        return out

    # -- batched generation --------------------------------------------------

    def generate_batch(
        self,
        prompts: list[str],
        *,
        max_tokens: int = 3328,
        temperature: float = 0.7,
        timeout: int | None = None,
    ) -> list[GenerationResult]:
        """
        Generate for N prompts in one vLLM call (true GPU batching).

        If *timeout* is set and the call exceeds it, every result in the
        returned list has ``timed_out=True``.
        """
        import threading
        from vllm import SamplingParams

        formatted = self._format_prompts(prompts)
        input_lengths = [
            len(self._tokenizer.encode(fp, add_special_tokens=False))
            for fp in formatted
        ]

        sampling_params = SamplingParams(
            temperature=temperature, top_p=0.95, top_k=50, max_tokens=max_tokens,
        )

        _outputs: list = []
        _exc: list = []

        def _run():
            try:
                _outputs.extend(
                    self._llm.generate(
                        formatted,
                        sampling_params,
                        lora_request=self._lora_request,
                        use_tqdm=False,
                    )
                )
            except Exception as exc:
                _exc.append(exc)

        start = time.time()
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=timeout)
        total_elapsed = time.time() - start

        if t.is_alive():
            logger.error(
                f"vLLM batch of {len(prompts)} timed out after {total_elapsed:.0f}s"
            )
            return [
                GenerationResult(
                    input_tokens=n, generation_time=total_elapsed, timed_out=True,
                )
                for n in input_lengths
            ]

        if _exc:
            raise _exc[0]

        results = []
        for output, n_input in zip(_outputs, input_lengths):
            text = output.outputs[0].text if output.outputs else None
            try:
                gen_time = (
                    output.metrics.finished_time - output.metrics.first_scheduled_time
                )
            except Exception:
                gen_time = total_elapsed / max(len(prompts), 1)
            results.append(GenerationResult(
                text=text, input_tokens=n_input, generation_time=gen_time,
            ))
        return results

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 3328,
        temperature: float = 0.7,
        timeout: int | None = None,
    ) -> GenerationResult:
        """Single-prompt wrapper — API-compatible with QwenGenerator."""
        return self.generate_batch(
            [prompt], max_tokens=max_tokens, temperature=temperature, timeout=timeout,
        )[0]


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

    args = parser.parse_args()

    model_name = args.model
    if args.backend == "qwen":
        model_name = model_name or "Qwen/Qwen3-4B-Instruct-2507"
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
