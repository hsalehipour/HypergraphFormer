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
import sys
import time
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

    def generate_batch(self, prompts: list[str], *, max_tokens: int | None = None,
                       temperature: float | None = None,
                       timeout: int | None = None) -> list[GenerationResult]:
        """Fallback batch generation by looping one prompt at a time."""
        return [
            self.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
            for prompt in prompts
        ]

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
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"
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

    def generate_batch(self, prompts: list[str], *, max_tokens: int | None = None,
                       temperature: float | None = None,
                       timeout: int | None = None) -> list[GenerationResult]:
        from transformers import StoppingCriteria, StoppingCriteriaList

        mt = max_tokens or self.max_tokens
        temp = temperature or self.temperature
        batch_size = len(prompts)

        texts = []
        for prompt in prompts:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            texts.append(self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            ))

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.model.device)
        # Real (non-padding) prompt token counts per sequence, for reporting.
        input_lens = inputs.attention_mask.sum(dim=1).tolist()
        # Padded input length: same for every sequence in the batch.
        # outputs[i] = [padded_input_tokens | generated_tokens], so we must
        # slice from here (not from the non-padded length) to isolate new tokens.
        # Using non-padded length would include left-padding eos/pad tokens in
        # the scan, causing every sequence to look "finished".
        padded_input_len = inputs.input_ids.shape[1]

        gen_kwargs = dict(
            **inputs, max_new_tokens=mt, temperature=temp,
            do_sample=True, top_p=0.95, top_k=50,
        )

        # Per-sequence timeout: checked after every generated token.
        # Returns a BoolTensor of shape (batch_size,) so HuggingFace can stop
        # individual sequences independently.  Sequences that already produced
        # an EOS before the deadline are counted as successful in post-processing.
        criteria_obj: "_PerSampleTimeout | None" = None
        if timeout is not None:
            batch_start = time.time()

            class _PerSampleTimeout(StoppingCriteria):
                def __init__(self_):
                    self_.timed_out: bool = False

                def __call__(self_, input_ids, scores, **kwargs) -> bool:
                    if time.time() - batch_start >= timeout:
                        self_.timed_out = True
                    return self_.timed_out

            criteria_obj = _PerSampleTimeout()
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([criteria_obj])

        start = time.time()
        outputs = self.model.generate(**gen_kwargs)
        elapsed = time.time() - start

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        eos_id = self.tokenizer.eos_token_id

        results = []
        for i, (text, num_input) in enumerate(zip(decoded, input_lens)):
            # Slice only the newly generated tokens (after the padded input).
            generated_ids = outputs[i, padded_input_len:]
            finished = bool((generated_ids == eos_id).any().item())

            seq_timed_out = (
                criteria_obj is not None
                and criteria_obj.timed_out
                and not finished
            )
            if seq_timed_out:
                logger.warning(
                    f"Batch sample {i} timed out after {elapsed:.1f}s "
                    f"(no EOS in generated tokens)"
                )
                results.append(GenerationResult(
                    input_tokens=int(num_input),
                    generation_time=elapsed,
                    timed_out=True,
                ))
            else:
                results.append(GenerationResult(
                    text=text,
                    input_tokens=int(num_input),
                    generation_time=elapsed,
                ))
        return results

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
