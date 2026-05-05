#!/usr/bin/env python3
"""
Few-shot evaluator for House-Diffusion-LLM models.

Two modes:
    fewshot-qwen -- Base / fine-tuned Qwen via vLLM (default) or HuggingFace, single GPU
    fewshot-gpt  -- GPT via OpenAI API, single process

The model is asked to produce a House-Diffusion JSON (``room_type`` / ``boxes``
/ ``edges`` / ``ed_rm``) from an access graph. Eval samples are the per-id
``<id>.txt`` files written by ``scripts/data_preparation_housediffusion_llm.py``
(``{"accessgraph": ..., "hypergraph": <hd_json_str>}``).

Scoring is HD-specific (no GED): structural validation on the generated HD
JSON, plus reconstruction of an access graph from the generated HD via the
same door-mediated convention used at data-prep time and a comparison with
the input access graph (room count, room-type multiset, edge-set F1).

Outputs
-------
For ``--output results/run.json``:

* ``results/run.json``           Aggregate eval report: metadata, statistics,
                                 and per-sample result records (one entry per
                                 sample, including its ``generated_hd`` payload).
* ``results/run_partial.json``   Same shape, written incrementally after every
                                 sample so a crashed run can be resumed via
                                 ``--checkpoint``.
* ``results/run_hd/<id>.json``   ONE file per successful generation, holding
                                 just the HD JSON dict (``room_type`` /
                                 ``boxes`` / ``edges`` / ``ed_rm``). Written
                                 incrementally as each sample completes; this
                                 is the shape ``benchmark/utils/visualize_rplan_json.py``
                                 and ``benchmark/utils/convert_json_to_pickle.py``
                                 expect. Override the directory with
                                 ``--per-sample-dir``.

Usage
-----
    # Few-shot Qwen (single GPU; vLLM by default if installed)
    python scripts/evaluator_housediffusion.py --mode fewshot-qwen \\
        --dataset-dir dataset/rplan/dataset_housediffusion/test

    # Zero-shot variant (no in-context examples)
    python scripts/evaluator_housediffusion.py --mode fewshot-qwen \\
        --dataset-dir dataset/rplan/dataset_housediffusion/test --num-examples 0

    # Few-shot GPT
    python scripts/evaluator_housediffusion.py --mode fewshot-gpt \\
        --model gpt-4-turbo \\
        --dataset-dir dataset/rplan/dataset_housediffusion/test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Set BEFORE importing vllm (via scripts.generators) — see scripts/evaluator.py.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

# Make ``utils`` importable when the script is run from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tqdm import tqdm

from scripts.data_preparation_housediffusion_llm import (
    DOOR_CATS,
    FRONT_DOOR_CAT,
    HD_CATEGORY_TO_NAME,
    HD_FORMAT_RULES_TEXT,
    HD_REQUIREMENTS_TMPL,
    HD_SYSTEM_MESSAGE,
    INTERIOR_DOOR_CAT,
    OUTSIDE_LABEL,
    create_hd_instruction_prompt,
    extract_access_graph_from_hd,
)
from scripts.generators import (
    VLLM_AVAILABLE,
    GPTGenerator,
    QwenGenerator,
    VllmQwenGenerator,
)

logger = logging.getLogger(__name__)


# ===================================================================
# HD JSON extraction (from raw model output)
# ===================================================================

_HD_REQUIRED_FIELDS = ("room_type", "boxes", "edges", "ed_rm")


def _looks_like_hd_json(obj: Any) -> bool:
    return isinstance(obj, dict) and all(k in obj for k in _HD_REQUIRED_FIELDS)


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


def _try_parse_first_object(s: str) -> Optional[dict]:
    """Parse the first top-level JSON object from *s*, ignoring trailing text."""
    s = s.strip()
    i = s.find("{")
    if i == -1:
        return None
    dec = json.JSONDecoder()
    try:
        obj, _end = dec.raw_decode(s[i:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _largest_balanced_objects(s: str) -> list[str]:
    """Return all top-level ``{...}`` substrings, longest first."""
    out: list[str] = []
    stack: list[str] = []
    start: Optional[int] = None
    for i, ch in enumerate(s):
        if ch == "{":
            if not stack:
                start = i
            stack.append(ch)
        elif ch == "}" and stack:
            stack.pop()
            if not stack and start is not None:
                out.append(s[start : i + 1])
                start = None
    out.sort(key=len, reverse=True)
    return out


def extract_hd_json_from_qwen(response: str) -> Optional[dict]:
    """Extract an HD JSON dict from a Qwen-style response (with ``<|im_start|>assistant``).

    The vLLM wrapper in :mod:`scripts.generators` already prepends the assistant
    marker if missing, so this works both for HF and vLLM outputs. Falls back to
    parsing the raw response when the marker is absent.
    """
    text = response or ""

    marker = "<|im_start|>assistant"
    idx = text.find(marker)
    candidates: list[str] = []
    if idx != -1:
        section = text[idx + len(marker):]
        end = section.find("<|im_end|>")
        if end != -1:
            section = section[:end]
        candidates.append(_strip_code_fences(section))
    candidates.append(_strip_code_fences(text))

    for cand in candidates:
        try:
            obj = json.loads(cand)
            if _looks_like_hd_json(obj):
                return obj
        except json.JSONDecodeError:
            pass

        obj = _try_parse_first_object(cand)
        if _looks_like_hd_json(obj):
            return obj

        for blob in _largest_balanced_objects(cand):
            try:
                obj = json.loads(blob)
                if _looks_like_hd_json(obj):
                    return obj
            except json.JSONDecodeError:
                continue

    return None


def extract_hd_json_from_gpt(response: str) -> Optional[dict]:
    """Extract an HD JSON dict from a GPT response (markdown fences or plain)."""
    text = response or ""

    json_block_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    for match in re.findall(json_block_pattern, text, re.DOTALL):
        try:
            obj = json.loads(match.strip())
            if _looks_like_hd_json(obj):
                return obj
        except json.JSONDecodeError:
            continue

    for blob in _largest_balanced_objects(text):
        try:
            obj = json.loads(blob)
            if _looks_like_hd_json(obj):
                return obj
        except json.JSONDecodeError:
            continue

    try:
        obj = json.loads(text.strip())
        if _looks_like_hd_json(obj):
            return obj
    except json.JSONDecodeError:
        pass

    return None


# ===================================================================
# Few-shot HD prompt builder
# ===================================================================

def create_hd_fewshot_prompt(target_accessgraph: dict, examples: list[dict]) -> str:
    """Build a few-shot user-content prompt for HD generation.

    ``examples`` is a list of dicts with ``filename`` / ``accessgraph`` / ``hypergraph``
    keys (same shape as :func:`utils.data_loading.load_example_hypergraphs` returns).
    Each example's ``hypergraph`` is expected to be an HD JSON dict.

    When ``examples`` is empty the prompt collapses to a clean zero-shot
    instruction (essentially equivalent to
    :func:`create_hd_instruction_prompt` plus the per-target requirements).
    """
    if not examples:
        return _zero_shot_user_content(target_accessgraph)

    intro = (
        "You are an expert at generating apartment floor plan layouts in the "
        "House-Diffusion JSON format that represent floor plans.\n\n"
        "You will be shown examples of access graphs paired with their corresponding "
        "House-Diffusion JSON layouts, then generate a new layout following the same "
        "structure and rules.\n\n"
        "An access graph is a simple representation of room connectivity:\n"
        "- `nodes`: Dictionary mapping room IDs to room types "
        '(e.g., {"1": "bed", "2": "kitchen"}).\n'
        "- `edges`: List of connections. Each entry is either a room-to-room pair "
        '(e.g. `["1", "2"]`) or an outside reference (e.g. `["1", "outside"]`). '
        'The literal string `"outside"` is NOT a room and is NOT listed in `nodes`; '
        "it marks the room that owns the apartment's front door (i.e., the entry-door "
        "room), and that room's box should therefore lie on the apartment boundary so "
        "its front door can face the exterior."
    )

    parts: list[str] = [intro, "", HD_FORMAT_RULES_TEXT, "",
                        "# TRAINING EXAMPLES (Access Graph -> House-Diffusion JSON):", ""]
    for i, ex in enumerate(examples, 1):
        parts.append(f"## Example {i} ({ex.get('filename', f'example_{i}')}):")
        parts.append("")
        parts.append("### Input - Access Graph:")
        parts.append("```json")
        parts.append(json.dumps(ex["accessgraph"], indent=2))
        parts.append("```")
        parts.append("")
        parts.append("### Output - House-Diffusion JSON:")
        parts.append("```json")
        parts.append(json.dumps(ex["hypergraph"], indent=2))
        parts.append("```")
        parts.append("")

    parts.extend([
        "# YOUR TASK:",
        "",
        "Generate a House-Diffusion JSON object for the following access graph.",
        "",
        "### Target Access Graph:",
        "```json",
        json.dumps(target_accessgraph, indent=2),
        "```",
        "",
        HD_REQUIREMENTS_TMPL.format(
            num_nodes=len(target_accessgraph.get("nodes", {})),
        ),
        "",
        "Generate ONLY the House-Diffusion JSON object, no additional text or explanation:",
    ])
    return "\n".join(parts)


def _zero_shot_user_content(target_accessgraph: dict) -> str:
    """Same shape as ``data_preparation_housediffusion_llm.create_hd_training_sample`` user side."""
    instruction = create_hd_instruction_prompt()
    return (
        f"{instruction}\n\n"
        f"### Access Graph:\n```json\n{json.dumps(target_accessgraph, indent=2)}\n```\n\n"
        f"{HD_REQUIREMENTS_TMPL.format(num_nodes=len(target_accessgraph.get('nodes', {})))}\n\n"
        "Generate ONLY the House-Diffusion JSON object, no additional text or explanation:"
    )


# ===================================================================
# HD structural validation + access-graph reconstruction scoring
# ===================================================================

@dataclass
class HdValidation:
    is_valid: bool
    passed: list[str]
    failed: list[str]


def _is_int_box(box: Any) -> bool:
    if not isinstance(box, list) or len(box) != 4:
        return False
    if not all(isinstance(v, int) for v in box):
        return False
    x1, y1, x2, y2 = box
    if not (0 <= x1 <= 256 and 0 <= y1 <= 256 and 0 <= x2 <= 256 and 0 <= y2 <= 256):
        return False
    return x1 < x2 and y1 < y2


def _is_int_segment(seg: Any) -> bool:
    if not isinstance(seg, list) or len(seg) != 6:
        return False
    if not all(isinstance(v, int) for v in seg):
        return False
    x1, y1, x2, y2, _ra, _rb = seg
    return all(0 <= v <= 256 for v in (x1, y1, x2, y2))


def validate_hd_json(hd: dict) -> HdValidation:
    """Lightweight structural checks on a generated HD JSON.

    These mirror the rules in ``HD_FORMAT_RULES_TEXT``: shape / lengths,
    integer pixel ranges, ``ra``/``rb`` consistency, and door-pairing.
    Geometric closure (segments forming closed polygons) is intentionally
    NOT checked here -- it requires more work and is not needed for the
    coarse few-shot eval signal.
    """
    passed: list[str] = []
    failed: list[str] = []

    rt = hd.get("room_type")
    boxes = hd.get("boxes")
    edges = hd.get("edges")
    ed_rm = hd.get("ed_rm")

    if not isinstance(rt, list) or not rt:
        failed.append("room_type missing or not a non-empty list")
    else:
        if all(isinstance(c, int) for c in rt):
            passed.append("room_type entries are ints")
        else:
            failed.append("room_type contains non-int entries")

    if isinstance(boxes, list) and isinstance(rt, list):
        if len(boxes) == len(rt):
            passed.append("len(boxes) == len(room_type)")
        else:
            failed.append(f"len(boxes)={len(boxes)} != len(room_type)={len(rt)}")
        if all(_is_int_box(b) for b in boxes):
            passed.append("boxes are integer-pixel rects in [0, 256]")
        else:
            failed.append("one or more boxes are not integer-pixel rects in [0, 256]")
    else:
        failed.append("boxes missing or not a list")

    if isinstance(edges, list) and isinstance(ed_rm, list):
        if len(edges) == len(ed_rm):
            passed.append("len(edges) == len(ed_rm)")
        else:
            failed.append(f"len(edges)={len(edges)} != len(ed_rm)={len(ed_rm)}")
        if all(_is_int_segment(seg) for seg in edges):
            passed.append("edges are integer pixel segments in [0, 256]")
        else:
            failed.append("one or more edge segments are not integer pixel rows of length 6")
    else:
        failed.append("edges or ed_rm missing or not a list")

    if isinstance(rt, list) and isinstance(edges, list) and isinstance(ed_rm, list):
        n = len(rt)
        ra_rb_ok = True
        ed_rm_ok = True
        for seg, owners in zip(edges, ed_rm):
            if not (isinstance(seg, list) and len(seg) == 6):
                ra_rb_ok = False
                break
            ra, rb = seg[4], seg[5]
            if not isinstance(owners, list) or len(owners) not in (1, 2):
                ed_rm_ok = False
                continue
            if any((not isinstance(i, int)) or i < 0 or i >= n for i in owners):
                ed_rm_ok = False
                continue
            if len(owners) == 1:
                if rb != 0:
                    ra_rb_ok = False
                if rt[owners[0]] != ra:
                    ra_rb_ok = False
            else:
                i, j = owners
                if rb == 0:
                    ra_rb_ok = False
                if not (rt[i] == ra and rt[j] == rb):
                    ra_rb_ok = False
        (passed if ra_rb_ok else failed).append("ra/rb consistent with ed_rm and room_type")
        (passed if ed_rm_ok else failed).append("ed_rm rows reference valid node indices")

    if isinstance(rt, list):
        n_front = sum(1 for c in rt if c == FRONT_DOOR_CAT)
        if n_front == 1:
            passed.append("exactly one front-door entry (category 15)")
        else:
            failed.append(f"expected 1 front-door entry, got {n_front}")

    return HdValidation(is_valid=not failed, passed=passed, failed=failed)


def score_against_accessgraph(
    target_ag: dict, hd: dict,
) -> dict:
    """Reconstruct the access graph from *hd* and compare to *target_ag*."""
    target_nodes: dict = target_ag.get("nodes", {}) or {}
    target_edges = target_ag.get("edges", []) or []

    try:
        gen_ag = extract_access_graph_from_hd(hd)
    except Exception as e:  # pragma: no cover -- defensive
        return {
            "reconstruction_error": str(e),
            "room_count_accuracy": 0.0,
            "room_type_accuracy": 0.0,
            "edge_f1": 0.0,
            "edge_precision": 0.0,
            "edge_recall": 0.0,
            "outside_match": False,
        }

    gen_nodes: dict = gen_ag.get("nodes", {})
    gen_edges = gen_ag.get("edges", [])

    n_target = len(target_nodes)
    n_gen = len(gen_nodes)
    room_count_accuracy = 1.0 if n_target == n_gen else 0.0

    target_types = sorted(str(v) for v in target_nodes.values())
    gen_types = sorted(str(v) for v in gen_nodes.values())
    if target_types == gen_types:
        room_type_accuracy = 1.0
    else:
        from collections import Counter

        ct, cg = Counter(target_types), Counter(gen_types)
        inter = sum((ct & cg).values())
        denom = max(sum(ct.values()), 1)
        room_type_accuracy = inter / denom

    def _edge_keys(edges) -> tuple[set[tuple[str, str]], set[str]]:
        room_pairs: set[tuple[str, str]] = set()
        outside_rooms: set[str] = set()
        for e in edges:
            if not isinstance(e, list) or len(e) != 2:
                continue
            a, b = str(e[0]), str(e[1])
            if a == OUTSIDE_LABEL or b == OUTSIDE_LABEL:
                room = b if a == OUTSIDE_LABEL else a
                outside_rooms.add(room)
            else:
                key = tuple(sorted([a, b]))
                room_pairs.add(key)
        return room_pairs, outside_rooms

    t_pairs, t_outside = _edge_keys(target_edges)
    g_pairs, g_outside = _edge_keys(gen_edges)

    tp = len(t_pairs & g_pairs)
    fp = len(g_pairs - t_pairs)
    fn = len(t_pairs - g_pairs)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "room_count_accuracy": room_count_accuracy,
        "room_type_accuracy": room_type_accuracy,
        "edge_f1": f1,
        "edge_precision": precision,
        "edge_recall": recall,
        "outside_match": t_outside == g_outside,
        "num_target_nodes": n_target,
        "num_generated_nodes": n_gen,
        "num_target_edges": len(target_edges),
        "num_generated_edges": len(gen_edges),
    }


# ===================================================================
# Sample loading & few-shot example loading (HD-aware)
# ===================================================================

def load_sample(fpath: Path) -> dict:
    """Load one ``<id>.txt`` file written by ``data_preparation_housediffusion_llm.py``."""
    with fpath.open() as f:
        data = json.load(f)
    ag = data.get("accessgraph", {}) or {}
    hg_raw = data.get("hypergraph")
    if hg_raw is None or (isinstance(hg_raw, str) and not hg_raw.strip()):
        gt_hd: Optional[dict] = None
    else:
        gt_hd = json.loads(hg_raw) if isinstance(hg_raw, str) else hg_raw
    return {
        "filename": fpath.name,
        "accessgraph": ag,
        "ground_truth_hd": gt_hd,
        "num_nodes": len(ag.get("nodes", {}) or {}),
        "num_edges": len(ag.get("edges", []) or []),
    }


def load_hd_examples(
    examples_dir: Path,
    num_examples: int,
    *,
    exclude_file: Optional[str] = None,
    seed: Optional[int] = None,
) -> list[dict]:
    """Load few-shot examples from the dataset dir, parsing the HD JSON payload.

    Mirrors :func:`utils.data_loading.load_example_hypergraphs` but kept local
    so this script does not depend on the BSP-specific module.
    """
    if num_examples is None or num_examples <= 0:
        return []

    files = sorted(examples_dir.glob("*.txt"), key=lambda p: p.name)
    if exclude_file:
        files = [f for f in files if f.name != exclude_file]
    if not files:
        raise FileNotFoundError(f"No example .txt files in {examples_dir}")

    if seed is not None:
        rng = random.Random(seed)
        chosen = rng.sample(files, min(num_examples, len(files)))
    else:
        chosen = random.sample(files, min(num_examples, len(files)))

    examples: list[dict] = []
    for fp in chosen:
        try:
            with fp.open() as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Skipping example %s: %s", fp.name, e)
            continue
        ag = data.get("accessgraph", {}) or {}
        hg = data.get("hypergraph")
        if isinstance(hg, str):
            try:
                hg = json.loads(hg)
            except json.JSONDecodeError as e:
                logger.warning("Skipping example %s: bad hypergraph JSON (%s)", fp.name, e)
                continue
        if not _looks_like_hd_json(hg):
            logger.warning("Skipping example %s: hypergraph is not an HD JSON", fp.name)
            continue
        examples.append({"filename": fp.name, "accessgraph": ag, "hypergraph": hg})

    if not examples:
        raise ValueError(
            f"Could not load any HD examples from {examples_dir}; "
            "did you run data_preparation_housediffusion_llm.py?"
        )
    return examples


# ===================================================================
# Evaluator base + two backends
# ===================================================================

class HdFewShotEvaluator(ABC):
    """Single-process few-shot HD evaluator template."""

    def __init__(self, args):
        self.args = args
        self.generator = None  # set in load_model

    # -- subclass hooks ----------------------------------------------------

    @abstractmethod
    def load_model(self) -> None: ...

    @abstractmethod
    def extract_hd(self, response: str) -> Optional[dict]: ...

    # -- prompt construction ----------------------------------------------

    def build_prompt(self, sample: dict) -> str:
        n = self.args.num_examples
        if n and n > 0:
            examples = load_hd_examples(
                Path(self.args.examples_dir or self.args.dataset_dir),
                num_examples=n,
                exclude_file=sample["filename"],
                seed=self.args.seed,
            )
        else:
            examples = []
        return create_hd_fewshot_prompt(sample["accessgraph"], examples)

    # -- per-sample evaluation --------------------------------------------

    def build_result_template(self, sample: dict) -> dict:
        return {
            "filename": sample["filename"],
            "success": False,
            "error": None,
            "num_nodes": sample.get("num_nodes"),
            "num_edges": sample.get("num_edges"),
            "num_input_tokens": None,
            "generation_time": None,
            "validation": {"is_valid": False, "passed_checks": [], "failed_checks": []},
            "json_extracted": False,
            "retried": False,
            "timed_out": False,
        }

    def evaluate_sample(self, sample: dict) -> dict:
        result = self.build_result_template(sample)
        try:
            prompt = self.build_prompt(sample)
        except Exception as e:
            result["error"] = f"prompt build failed: {e}"
            return result

        timeout = getattr(self.args, "timeout", None)
        try:
            res = self.generator.generate(
                prompt,
                max_tokens=self.args.max_tokens,
                temperature=self.args.temperature,
                timeout=timeout,
            )
        except Exception as e:
            result["error"] = f"generation failed: {e}"
            return result

        result["num_input_tokens"] = res.input_tokens
        result["generation_time"] = res.generation_time
        if res.timed_out:
            result["error"] = f"Generation timed out after {timeout}s"
            result["timed_out"] = True
            return result

        hd = self.extract_hd(res.text or "")
        if hd is None:
            logger.info("JSON extraction failed for %s, retrying once...", sample["filename"])
            try:
                res2 = self.generator.generate(
                    prompt,
                    max_tokens=self.args.max_tokens,
                    temperature=self.args.temperature,
                    timeout=timeout,
                )
            except Exception as e:
                result["error"] = f"generation failed on retry: {e}"
                return result
            result["generation_time"] += res2.generation_time
            if res2.timed_out:
                result["error"] = f"Generation timed out after {timeout}s (retry)"
                result["timed_out"] = True
                return result
            hd = self.extract_hd(res2.text or "")
            result["retried"] = True
            if hd is None:
                result["error"] = "Failed to extract HD JSON from response (after retry)"
                return result

        result["json_extracted"] = True
        result["generated_hd"] = hd

        validation = validate_hd_json(hd)
        result["validation"] = {
            "is_valid": validation.is_valid,
            "passed_checks": validation.passed,
            "failed_checks": validation.failed,
        }

        scores = score_against_accessgraph(sample["accessgraph"], hd)
        result.update(scores)
        # "success" is a coarse summary: structural validity + room count match.
        result["success"] = bool(
            validation.is_valid and scores.get("room_count_accuracy", 0.0) == 1.0,
        )
        return result

    # -- run loop ----------------------------------------------------------

    def run(self) -> int:
        args = self.args

        random.seed(args.seed)

        dataset_dir = Path(args.dataset_dir)
        all_files = sorted(dataset_dir.glob("*.txt"), key=lambda p: p.name)
        if not all_files:
            logger.error("No .txt files found in %s", dataset_dir)
            return 1

        if getattr(args, "max_rooms", None):
            kept: list[Path] = []
            for fp in all_files:
                try:
                    with fp.open() as f:
                        n = len(json.load(f).get("accessgraph", {}).get("nodes", {}))
                except (OSError, json.JSONDecodeError):
                    continue
                if n <= args.max_rooms:
                    kept.append(fp)
            logger.info("Room filter (max %d): kept %d/%d", args.max_rooms, len(kept), len(all_files))
            all_files = kept

        existing, completed = self._load_checkpoint(args)
        files_to_process = [f for f in all_files if f.name not in completed]
        if args.max_samples:
            files_to_process = files_to_process[: args.max_samples]
            logger.info("Limited to %d samples", args.max_samples)

        if not files_to_process:
            logger.info("All files already evaluated.")
            return 0

        logger.info("Processing %d files (already done: %d)", len(files_to_process), len(completed))
        self.load_model()

        results = list(existing)
        start = time.time()
        n_total = len(files_to_process)

        for i, fp in enumerate(
            tqdm(
                files_to_process,
                desc="Evaluating",
                unit="sample",
                mininterval=1.0,
                file=sys.stdout,
            ),
            1,
        ):
            sample = load_sample(fp)
            result = self.evaluate_sample(sample)
            results.append(result)
            self._save_partial(args, results)
            self._save_one_hd(args, result)

            if result["success"]:
                msg = (
                    f"OK  | edge-F1={result.get('edge_f1', 0):.2f} "
                    f"room-types={result.get('room_type_accuracy', 0):.2f}"
                )
            elif result.get("json_extracted"):
                msg = (
                    f"INVALID-HD | failed: "
                    f"{', '.join(result['validation']['failed_checks'][:2])}"
                )
            else:
                err = (result.get("error") or "")[:80]
                msg = f"FAIL | {err}"
            logger.info("  [%d/%d] %s -- %s", i, n_total, fp.name, msg)

            if i % 5 == 0:
                self._save_checkpoint(args, results)

            interval = max(1, n_total // 10)
            if i == 1 or i == n_total or i % interval == 0:
                elapsed = time.time() - start
                rate = i / elapsed if elapsed > 0 else 0.0
                eta = (n_total - i) / rate if rate > 0 else 0.0
                logger.info(
                    "Progress: sample %d/%d | elapsed %.1f min | ETA ~%.1f min",
                    i, n_total, elapsed / 60.0, eta / 60.0,
                )

        elapsed = time.time() - start
        logger.info("Total time: %.1f min", elapsed / 60.0)

        self._save_final(args, results)
        return 0

    # -- aggregate stats ---------------------------------------------------

    def compute_statistics(self, results: list[dict]) -> dict:
        n = len(results)
        json_ok = [r for r in results if r.get("json_extracted")]
        valid = [r for r in results if r.get("validation", {}).get("is_valid")]
        ok = [r for r in results if r.get("success")]
        timed_out = [r for r in results if r.get("timed_out")]
        room_count_match = [r for r in results if r.get("room_count_accuracy", 0.0) == 1.0]
        room_types_exact = [r for r in results if r.get("room_type_accuracy", 0.0) == 1.0]
        outside_ok = [r for r in results if r.get("outside_match")]

        edge_f1 = [r["edge_f1"] for r in results if "edge_f1" in r]
        edge_p = [r["edge_precision"] for r in results if "edge_precision" in r]
        edge_r = [r["edge_recall"] for r in results if "edge_recall" in r]
        room_acc = [r["room_type_accuracy"] for r in results if "room_type_accuracy" in r]
        times = [r["generation_time"] for r in results if r.get("generation_time") is not None]

        def _avg(xs):
            return sum(xs) / len(xs) if xs else None

        return {
            "total": n,
            "json_extracted": len(json_ok),
            "valid_structures": len(valid),
            "successful": len(ok),
            "timed_out": len(timed_out),
            "success_rate": (len(ok) / n) if n else 0.0,
            "room_count_match": len(room_count_match),
            "room_types_exact_match": len(room_types_exact),
            "outside_match": len(outside_ok),
            "edge_metrics": {
                "f1_avg": _avg(edge_f1),
                "precision_avg": _avg(edge_p),
                "recall_avg": _avg(edge_r),
            },
            "room_type_accuracy_avg": _avg(room_acc),
            "inference_time": {
                "average_seconds": _avg(times),
                "min_seconds": min(times) if times else None,
                "max_seconds": max(times) if times else None,
                "total_minutes": (sum(times) / 60.0) if times else None,
            },
        }

    # -- I/O helpers -------------------------------------------------------

    @staticmethod
    def _load_checkpoint(args) -> tuple[list[dict], set[str]]:
        existing: list[dict] = []
        completed: set[str] = set()
        cp = getattr(args, "checkpoint", None)
        if cp and Path(cp).exists():
            with open(cp) as f:
                data = json.load(f)
            existing = data.get("results", []) or []
            completed = {r["filename"] for r in existing}
            logger.info("Loaded checkpoint: %d samples", len(completed))
        return existing, completed

    @staticmethod
    def _per_sample_dir(args) -> Path:
        """Directory for one HD JSON file per generated sample.

        Defaults to ``<output stem>_hd/`` next to ``--output`` so a user who runs
        ``--output results/run.json`` finds their per-id HD JSONs in
        ``results/run_hd/<id>.json``. Override with ``--per-sample-dir``.
        """
        explicit = getattr(args, "per_sample_dir", None)
        if explicit:
            return Path(explicit)
        out = Path(args.output)
        return out.with_name(f"{out.stem}_hd")

    @staticmethod
    def _save_one_hd(args, result: dict) -> Optional[Path]:
        """Write one ``<id>.json`` HD payload from a single result; no-op otherwise.

        Atomic-ish: write to a ``.tmp`` sibling and rename, so a crash mid-write
        cannot leave a half-written ``<id>.json`` that downstream tools would
        choke on.
        """
        hd = result.get("generated_hd")
        if hd is None:
            return None
        out_dir = HdFewShotEvaluator._per_sample_dir(args)
        out_dir.mkdir(parents=True, exist_ok=True)
        sid = result["filename"].replace(".txt", "")
        target = out_dir / f"{sid}.json"
        tmp = target.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(hd, f, indent=2)
        os.replace(tmp, target)
        return target

    @staticmethod
    def _save_partial(args, results: list[dict]) -> None:
        """Write the in-progress aggregate results report (per-sample HDs are written separately)."""
        out_path = Path(args.output.replace(".json", "_partial.json"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {"timestamp": datetime.now().isoformat(),
                 "completed": len(results), "results": results},
                f,
                indent=2,
            )

    @staticmethod
    def _save_checkpoint(args, results: list[dict]) -> None:
        with open(args.output, "w") as f:
            json.dump(
                {"timestamp": datetime.now().isoformat(),
                 "completed": len(results), "results": results},
                f,
                indent=2,
            )

    def _save_final(self, args, results: list[dict]) -> None:
        stats = self.compute_statistics(results)
        meta = {
            "timestamp": datetime.now().isoformat(),
            "mode": getattr(args, "mode", "unknown"),
            "model": getattr(args, "model", None),
            "dataset_dir": args.dataset_dir,
            "examples_dir": args.examples_dir,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "num_examples": args.num_examples,
            "total_samples": len(results),
        }
        if args.mode == "fewshot-qwen":
            meta["vllm"] = (not getattr(args, "no_vllm", False)) and VLLM_AVAILABLE

        output_data = {"metadata": meta, "statistics": stats, "results": results}
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info("Results saved to: %s", args.output)

        # Defensive: ensure every result with an HD payload has a per-sample file
        # on disk. The run loop already writes these incrementally; this catches
        # the case where the user resumed from a checkpoint into a new
        # --per-sample-dir, so older successes also get materialised there.
        per_sample_dir = self._per_sample_dir(args)
        n_written = 0
        for r in results:
            if r.get("generated_hd") is None:
                continue
            sid = r["filename"].replace(".txt", "")
            if not (per_sample_dir / f"{sid}.json").exists():
                self._save_one_hd(args, r)
                n_written += 1
        n_total_hd = sum(1 for r in results if r.get("generated_hd") is not None)
        logger.info(
            "Per-sample HD JSONs in %s (%d total; %d written this finalize pass)",
            per_sample_dir, n_total_hd, n_written,
        )


# -------- Qwen backend ---------------------------------------------------

class HdFewShotQwenEvaluator(HdFewShotEvaluator):
    """Few-shot HD generation with a local Qwen model (vLLM by default)."""

    def load_model(self) -> None:
        gpu = getattr(self.args, "gpu", 0)
        use_vllm = (not getattr(self.args, "no_vllm", False)) and VLLM_AVAILABLE
        if not getattr(self.args, "no_vllm", False) and not VLLM_AVAILABLE:
            logger.warning("vLLM not installed; falling back to HuggingFace generate (pip install vllm)")

        if use_vllm:
            tp = getattr(self.args, "tensor_parallel", 1)
            if tp > 1:
                gpu_list = ",".join(str(gpu + i) for i in range(tp))
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
                logger.info("vLLM tensor parallel: CUDA_VISIBLE_DEVICES=%s", gpu_list)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

            model_path = Path(self.args.model)
            is_lora = (model_path / "adapter_config.json").exists()
            self.generator = VllmQwenGenerator(
                self.args.model if not is_lora else str(_resolve_base_from_lora(model_path)),
                lora_path=model_path if is_lora else None,
                max_tokens=self.args.max_tokens,
                temperature=self.args.temperature,
                gpu_memory_utilization=self.args.vllm_gpu_memory,
                max_model_len=self.args.vllm_max_model_len,
                tensor_parallel_size=tp,
                enforce_eager=not getattr(self.args, "vllm_no_enforce_eager", False),
            )
            # Override BSP system prompt with HD-specific one (one-process eval).
            _patch_qwen_system_prompt(HD_SYSTEM_MESSAGE)
            logger.info("vLLM generator loaded (HD system prompt active).")
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        logger.info("Loading model: %s on GPU %s", self.args.model, gpu)
        tokenizer = AutoTokenizer.from_pretrained(self.args.model)
        model = AutoModelForCausalLM.from_pretrained(
            self.args.model, torch_dtype=torch.float16, device_map={"": gpu},
        )
        self.generator = QwenGenerator(model=model, tokenizer=tokenizer)
        _patch_qwen_system_prompt(HD_SYSTEM_MESSAGE)

    def extract_hd(self, response: str) -> Optional[dict]:
        return extract_hd_json_from_qwen(response)


def _resolve_base_from_lora(lora_path: Path) -> str:
    cfg_path = lora_path / "adapter_config.json"
    with cfg_path.open() as f:
        cfg = json.load(f)
    base = cfg.get("base_model_name_or_path")
    if not base:
        raise ValueError(f"Missing base_model_name_or_path in {cfg_path}")
    return base


def _patch_qwen_system_prompt(new_system_prompt: str) -> None:
    """Override the BSP system prompt baked into ``scripts.generators``.

    Both ``QwenGenerator`` and ``VllmQwenGenerator`` build their chat messages
    using the module-level ``SYSTEM_PROMPT`` constant. This replaces it for the
    lifetime of the process so HD evaluation gets the right system role.
    """
    import scripts.generators as _gen
    _gen.SYSTEM_PROMPT = new_system_prompt


# -------- GPT backend ----------------------------------------------------

class HdFewShotGPTEvaluator(HdFewShotEvaluator):
    """Few-shot HD generation via the OpenAI API."""

    def load_model(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.generator = GPTGenerator(model_name=self.args.model, api_key=api_key)
        _patch_qwen_system_prompt(HD_SYSTEM_MESSAGE)
        logger.info("GPT generator initialised (%s)", self.args.model)

    def extract_hd(self, response: str) -> Optional[dict]:
        return extract_hd_json_from_gpt(response)


# ===================================================================
# CLI
# ===================================================================

MODE_MAP: dict[str, type[HdFewShotEvaluator]] = {
    "fewshot-qwen": HdFewShotQwenEvaluator,
    "fewshot-gpt": HdFewShotGPTEvaluator,
}


def cli() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Few-shot House-Diffusion-LLM evaluator")
    parser.add_argument("--mode", choices=list(MODE_MAP), required=True)

    parser.add_argument(
        "--dataset-dir", type=str, required=True,
        help="Directory of <id>.txt files written by data_preparation_housediffusion_llm.py",
    )
    parser.add_argument(
        "--examples-dir", type=str, default=None,
        help="Directory to draw few-shot examples from (defaults to --dataset-dir)",
    )
    parser.add_argument("--output", type=str, default=None,
                        help="Aggregate eval report path (default: results_hd_<mode>_<model>_<seed>.json)")
    parser.add_argument(
        "--per-sample-dir", type=str, default=None,
        help=(
            "Directory for one HD JSON file per generated sample "
            "(default: <output stem>_hd/ next to --output). "
            "Files are written incrementally as each sample completes, "
            "and named <id>.json (matching the HD-JSON shape that "
            "benchmark/utils/visualize_rplan_json.py expects)."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=5120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-rooms", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--num-examples", type=int, default=3,
        help="In-context examples for fewshot-* modes. Use 0 for zero-shot.",
    )

    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)

    vllm = parser.add_argument_group("vLLM (Qwen local inference; default on if installed)")
    vllm.add_argument("--no-vllm", action="store_true",
                      help="Use HuggingFace generate instead of vLLM")
    vllm.add_argument("--vllm-gpu-memory", type=float, default=0.85)
    vllm.add_argument("--vllm-max-model-len", type=int, default=6900)
    vllm.add_argument("--tensor-parallel", type=int, default=1)
    vllm.add_argument("--vllm-no-enforce-eager", action="store_true",
                      help="Allow vLLM CUDA graphs (default: enforce_eager=True)")

    args = parser.parse_args()

    if args.model is None:
        if args.mode == "fewshot-qwen":
            args.model = "Qwen/Qwen3-4B-Instruct-2507"
        elif args.mode == "fewshot-gpt":
            args.model = "gpt-4"

    if args.output is None:
        slug = Path(args.model).name
        args.output = f"results_hd_{args.mode.replace('-', '_')}_{slug}_{args.seed}.json"
        logger.info("Auto-generated output: %s", args.output)

    cls = MODE_MAP[args.mode]
    evaluator = cls(args)
    return evaluator.run()


if __name__ == "__main__":
    sys.exit(cli())
