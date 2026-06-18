#!/usr/bin/env python3
"""Inference: ask the LoRA-finetuned Qwen3-4B-Instruct to predict the
``accessgraph.edges`` for each RPLAN ``eval_test/<id>.txt`` given just the
list of room types in ``accessgraph.nodes``.

Pipeline per sample id:

1. Read ``<eval_test>/<id>.txt`` (JSON with ``accessgraph.nodes``,
   ``accessgraph.edges``, ``bounds``, ``room_polygons``).
2. Build a chat prompt: a one-shot example carved out of sample ``0`` of
   the same eval_test set (its room set + GT edges), then a new query
   listing the current sample's room set and asking the model for the
   edges as a JSON array.
3. Run vLLM ``generate`` with the LoRA adapter applied on top of the base
   ``Qwen/Qwen3-4B-Instruct-2507``.
4. Parse the model's JSON edge list, validate each edge against the
   sample's node ids, then write ``<output_dir>/<id>.txt`` -- a verbatim
   copy of the input file with **only** ``accessgraph.edges`` replaced.

Distributed: torchrun-style. Each rank loads its own vLLM engine on its
local GPU (mirrors the bind-rank-to-device pattern in
``scripts/evaluator.py``) and processes ``files[rank::world]``. No
``torch.distributed`` collective calls are made -- the only "communication"
is each rank writing to disjoint output files in a shared dir, which is
filesystem-atomic per file.

Run::

    torchrun --standalone --nnodes=1 --nproc_per_node=8 \\
        scripts/qwen_predict_accessgraph_edges.py \\
        --eval-test-dir dataset/rplan/eval_test \\
        --output-dir dataset/rplan/eval_test_qwen_accessgraphs \\
        --checkpoint checkpoints/qwen_rplan_filtered_2/epochs8_lora_r64_alpha128/checkpoint-8700

For a single-GPU smoke run drop torchrun and add ``--world-override 1`` so
the script doesn't shard the work, e.g.::

    CUDA_VISIBLE_DEVICES=0 python scripts/qwen_predict_accessgraph_edges.py \\
        --eval-test-dir dataset/rplan/eval_test \\
        --output-dir /tmp/qwen_ag_smoke --limit 8

Token budget: with up to ~12 rooms the response is a JSON array of at most
~30 edges, which fits comfortably in 512 generated tokens. The default
``--max-new-tokens 768`` leaves headroom for the model rambling before the
JSON; the parser below tolerates leading/trailing prose either way.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Mirror evaluator.py: select V0 to avoid a vLLM V1 multi-process hang on
# certain image+CUDA combos (see https://github.com/vllm-project/vllm/issues/25837).
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [r%(rank)s] %(levelname)s %(message)s")


def _rank_logger(rank: int) -> logging.LoggerAdapter:
    """Tag every log line with the current rank for easier 8-GPU debugging."""
    base = logging.getLogger("qwen-ag-edges")
    return logging.LoggerAdapter(base, extra={"rank": rank})


# ----------------------------------------------------------------------
# Prompt construction
# ----------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert in residential floor-plan layouts. Given a list of "
    "rooms in an apartment, you predict which rooms are directly connected "
    "(share a doorway / opening) so the resulting access graph is realistic "
    "and navigable. You ALWAYS reply with a single JSON array of edges and "
    "nothing else."
)


def _build_prompt(example_nodes: dict[str, str],
                  example_edges: list[list[str]],
                  query_nodes: dict[str, str]) -> list[dict]:
    """One-shot messages list: example (sample 0 GT) then the new query."""
    user_intro = (
        "Based on your understanding of apartment layouts and typical room "
        "connections, build a valid and realistic access graph for the "
        "given set of rooms. The access graph describes which rooms are "
        "directly connected: each edge means two rooms share a doorway "
        "or open passage. The graph must be connected (no isolated room) "
        "and reflect typical residential layouts -- e.g. bedrooms typically "
        "connect to a hallway/living room, bathrooms to a hallway or "
        "bedroom, kitchens to a living/dining area, balconies to an "
        "interior room, etc.\n\n"
        "Nodes are given as ``{node_id: room_type}``. Reply with the edges "
        "ONLY -- a JSON array of two-element arrays, where each inner array "
        "is a pair of node_ids (strings). Do not include the nodes, do not "
        "wrap in another object, do not add commentary. Format exactly like "
        "the example below."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"{user_intro}\n\n"
            f"Example input nodes:\n```json\n{json.dumps(example_nodes, indent=2)}\n```\n\n"
            "Now respond with the edges for this example."
        )},
        {"role": "assistant", "content": f"```json\n{json.dumps(example_edges, indent=2)}\n```"},
        {"role": "user", "content": (
            f"Now produce the edges for these new nodes (same format, same rules):\n"
            f"```json\n{json.dumps(query_nodes, indent=2)}\n```"
        )},
    ]


# ----------------------------------------------------------------------
# Response parsing
# ----------------------------------------------------------------------

_FENCED_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_first_json_array(text: str) -> list | None:
    """Pull the first balanced ``[...]`` block out of *text* and ``json.loads``
    it, tolerating fenced code blocks, leading/trailing prose, and minor
    quoting variations. Returns ``None`` when nothing parseable is found.

    We look for fenced code first (the example primes the model to wrap its
    answer in ``\\`\\`\\`json ... \\`\\`\\``) and fall back to a brace-balance
    scan over the raw text if no fence is present.
    """
    candidates: list[str] = []
    for m in _FENCED_RE.finditer(text):
        candidates.append(m.group(1).strip())
    candidates.append(text)

    for cand in candidates:
        start = cand.find("[")
        if start < 0:
            continue
        depth = 0
        for i in range(start, len(cand)):
            ch = cand[i]
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    blob = cand[start:i + 1]
                    try:
                        parsed = json.loads(blob)
                    except json.JSONDecodeError:
                        # Tolerate single-quoted edges by swapping quote style.
                        try:
                            parsed = json.loads(blob.replace("'", '"'))
                        except json.JSONDecodeError:
                            break
                    if isinstance(parsed, list):
                        return parsed
                    break
    return None


def _normalise_edges(parsed: list, valid_ids: set[str]) -> tuple[list[list[str]], list[str]]:
    """Coerce parsed edge entries to ``[str, str]`` and drop:
    - non-pair entries
    - self-loops
    - edges referencing an unknown node id
    - duplicates (independent of orientation)

    Returns ``(edges, warnings)`` so the caller can log per-sample drops
    without failing the whole run.
    """
    out: list[list[str]] = []
    seen: set[tuple[str, str]] = set()
    warns: list[str] = []
    for raw in parsed:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            warns.append(f"dropping non-pair entry: {raw!r}")
            continue
        a, b = str(raw[0]), str(raw[1])
        if a == b:
            warns.append(f"dropping self-loop: {a}-{b}")
            continue
        if a not in valid_ids or b not in valid_ids:
            warns.append(f"dropping edge with unknown id: {a}-{b}")
            continue
        key = tuple(sorted([a, b]))
        if key in seen:
            continue
        seen.add(key)
        out.append([a, b])
    return out, warns


# ----------------------------------------------------------------------
# IO + sharding helpers
# ----------------------------------------------------------------------

def _emit_output(in_path: Path, out_path: Path, new_edges: list[list[str]]) -> None:
    """Re-load *in_path* verbatim, replace only ``accessgraph.edges``, and write to *out_path*."""
    with open(in_path, encoding="utf-8") as f:
        raw = json.load(f)
    ag = raw.get("accessgraph") or {}
    if not isinstance(ag, dict):
        raise ValueError(f"{in_path}: 'accessgraph' is not an object")
    ag["edges"] = new_edges
    raw["accessgraph"] = ag
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)


def _natural_id_key(name: str) -> tuple[int, int, str]:
    m = re.search(r"(\d+)\s*$", name)
    if m:
        return (0, int(m.group(1)), name)
    return (1, 0, name)


def _resolve_dist_env(world_override: int | None) -> tuple[int, int, int]:
    """Return ``(rank, world_size, local_rank)``.

    With torchrun launches both ``RANK``/``WORLD_SIZE`` and ``LOCAL_RANK`` are
    set; we read those. When ``--world-override 1`` is given we pretend to
    be rank 0 of a single-process world, regardless of any env (useful for
    smoke runs / Cursor-driven debugging).
    """
    if world_override is not None and world_override == 1:
        return 0, 1, int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world, local_rank


# Env vars torchrun (and friends) set on each worker. We snapshot the
# rank/world we need into Python variables FIRST, then strip these from
# os.environ before vLLM is imported -- otherwise vLLM 0.8.x sees
# WORLD_SIZE>1 with no torch.distributed group initialised yet and tries
# to do its own NCCL handshake, which deadlocks because every rank is
# running an independent engine (we want one engine per rank, not
# tensor-parallel across all 8). evaluator.py sidesteps this by going
# through Accelerate, which inits the dist group; we do the simpler thing.
_TORCHRUN_ENV_KEYS = (
    "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
    "GROUP_RANK", "GROUP_WORLD_SIZE",
    "MASTER_ADDR", "MASTER_PORT",
    "TORCHELASTIC_RUN_ID", "TORCHELASTIC_USE_AGENT_STORE",
    "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_ERROR_FILE",
    "ROLE_RANK", "ROLE_WORLD_SIZE", "ROLE_NAME",
)


def _strip_torchrun_env(log) -> None:
    """Remove torchrun-injected env vars so each rank looks single-process to
    vLLM. Must be called after rank/world have been read into Python vars
    and BEFORE vLLM is imported (inside ``main``).
    """
    removed = {k: os.environ.pop(k) for k in _TORCHRUN_ENV_KEYS if k in os.environ}
    if removed:
        log.info("stripped torchrun env (%d keys) so vLLM treats each rank "
                 "as standalone: %s", len(removed), sorted(removed.keys()))


def _bind_rank_to_gpu(local_rank: int, log) -> None:
    """Same idea as ``scripts/evaluator.py::_bind_rank_for_vllm``: set
    ``CUDA_VISIBLE_DEVICES`` to a single GPU index BEFORE we import vLLM/torch
    in this process so each rank gets its own card. If ``CUDA_VISIBLE_DEVICES``
    was already set externally (e.g. via Slurm), we leave it alone.
    """
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        log.info("CUDA_VISIBLE_DEVICES already set to %s; honoring it",
                 os.environ["CUDA_VISIBLE_DEVICES"])
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    log.info("bound rank to CUDA_VISIBLE_DEVICES=%s", local_rank)



# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-test-dir", type=Path,
                   default=REPO_ROOT / "dataset" / "rplan" / "eval_test",
                   help="Directory of <id>.txt eval_test files (default: "
                        "dataset/rplan/eval_test).")
    p.add_argument("--output-dir", type=Path,
                   default=REPO_ROOT / "dataset" / "rplan" / "eval_test_qwen_accessgraphs",
                   help="Directory to write predicted-edges JSON copies "
                        "(default: dataset/rplan/eval_test_qwen_accessgraphs).")
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "checkpoints" / "qwen_rplan_filtered_2"
                           / "epochs8_lora_r64_alpha128" / "checkpoint-8700",
                   help="LoRA adapter dir (must contain adapter_config.json + "
                        "adapter_model.safetensors).")
    p.add_argument("--base-model", type=str, default=None,
                   help="HF id of the base model. Default: read from "
                        "adapter_config.json (Qwen/Qwen3-4B-Instruct-2507).")
    p.add_argument("--example-id", type=str, default="0",
                   help="Sample id used as the one-shot example "
                        "(must exist in --eval-test-dir; default: '0').")
    p.add_argument("--limit", type=int, default=0,
                   help="Process at most N samples (after sharding). 0 = no "
                        "limit (default).")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Prompts per vLLM generate() batch on each rank "
                        "(default: 32).")
    p.add_argument("--max-new-tokens", type=int, default=768,
                   help="Generation cap; with up to ~12 rooms the JSON edge "
                        "list is well under 200 tokens, 768 leaves room for "
                        "any preamble (default: 768).")
    p.add_argument("--max-model-len", type=int, default=4096,
                   help="vLLM context window. Default: 4096 (the prompt "
                        "stays well under 1k tokens, but Qwen3 happily "
                        "supports up to 32k).")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                   help="vLLM gpu_memory_utilization (default: 0.90).")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature; 0.0 = greedy (default).")
    p.add_argument("--world-override", type=int, default=None,
                   help="Set to 1 to ignore RANK/WORLD_SIZE and run "
                        "single-process, useful for smoke tests outside "
                        "torchrun.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run + overwrite existing <output>/<id>.txt "
                        "(default: skip ids whose output already exists).")
    p.add_argument("--log-every", type=int, default=200,
                   help="Per-rank progress log cadence (default: 200).")
    args = p.parse_args(argv)

    rank, world, local_rank = _resolve_dist_env(args.world_override)
    log = _rank_logger(rank)
    _bind_rank_to_gpu(local_rank, log)
    _strip_torchrun_env(log)

    if not args.eval_test_dir.is_dir():
        sys.exit(f"--eval-test-dir not a directory: {args.eval_test_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # One-shot example loaded once, on every rank, from the SAME source dir
    # (so the example matches the inputs the model will see).
    example_path = args.eval_test_dir / f"{args.example_id}.txt"
    if not example_path.is_file():
        sys.exit(f"--example-id {args.example_id} not found at {example_path}")
    with open(example_path, encoding="utf-8") as f:
        example_entry = json.load(f)
    example_ag = example_entry.get("accessgraph") or {}
    example_nodes = example_ag.get("nodes") or {}
    example_edges = example_ag.get("edges") or []
    if not example_nodes or not example_edges:
        sys.exit(f"example {args.example_id} has empty accessgraph nodes/edges")
    log.info("example: %d nodes, %d edges", len(example_nodes), len(example_edges))

    all_files = sorted(args.eval_test_dir.glob("*.txt"),
                       key=lambda p: _natural_id_key(p.stem))
    rank_files = all_files[rank::world]
    if args.limit and args.limit > 0:
        rank_files = rank_files[: args.limit]
    log.info("eval_test total=%d, rank slice=%d", len(all_files), len(rank_files))

    # Skip already-done ids upfront so a re-launch resumes cheaply.
    if not args.overwrite:
        before = len(rank_files)
        rank_files = [
            p for p in rank_files
            if not (args.output_dir / p.name).exists()
        ]
        log.info("after dedup against existing outputs: %d (skipped %d)",
                 len(rank_files), before - len(rank_files))

    if not rank_files:
        log.info("nothing to do for this rank")
        return 0

    # Lazy import after GPU is bound and torchrun env is stripped, so each
    # rank initialises its own vLLM engine on its assigned card.
    from scripts.generators import VllmQwenGenerator
    from vllm import SamplingParams  # type: ignore[import-not-found]
    gen = VllmQwenGenerator(
        model_name=args.base_model or "Qwen/Qwen3-4B-Instruct-2507",
        lora_path=args.checkpoint,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,
    )
    tokenizer = gen._tokenizer
    lora_request = gen._lora_request()
    sampling_params = SamplingParams(
        temperature=float(args.temperature),
        top_p=1.0 if args.temperature == 0 else 0.95,
        max_tokens=int(args.max_new_tokens),
    )

    started = time.time()
    written = parsed_ok = parse_failed = 0
    drops_total = 0
    bs = max(1, int(args.batch_size))

    for i in range(0, len(rank_files), bs):
        batch_paths = rank_files[i:i + bs]
        prompts: list[str] = []
        node_sets: list[set[str]] = []
        nodes_per_sample: list[dict[str, str]] = []
        for fp in batch_paths:
            try:
                with open(fp, encoding="utf-8") as _f:
                    entry = json.load(_f)
            except Exception as exc:
                log.warning("[%s] failed to read: %s", fp.stem, exc)
                prompts.append(""); node_sets.append(set()); nodes_per_sample.append({})
                continue
            ag = entry.get("accessgraph") or {}
            nodes = ag.get("nodes") or {}
            if not isinstance(nodes, dict) or not nodes:
                log.warning("[%s] empty/invalid accessgraph.nodes; skipping", fp.stem)
                prompts.append(""); node_sets.append(set()); nodes_per_sample.append({})
                continue
            messages = _build_prompt(example_nodes, example_edges, nodes)
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            node_sets.append({str(k) for k in nodes})
            nodes_per_sample.append(nodes)

        # Drop blank prompts (failed reads) before batching into vLLM.
        live_indices = [j for j, t in enumerate(prompts) if t]
        if not live_indices:
            continue
        live_prompts = [prompts[j] for j in live_indices]
        outputs = gen._llm.generate(
            live_prompts, sampling_params,
            lora_request=lora_request, use_tqdm=False,
        )
        # vLLM preserves input order in its output list.
        for j, vllm_out in zip(live_indices, outputs):
            fp = batch_paths[j]
            text = vllm_out.outputs[0].text if vllm_out.outputs else ""
            parsed = _extract_first_json_array(text)
            if parsed is None:
                parse_failed += 1
                log.warning("[%s] could not parse edges from response (first 200 chars: %r)",
                            fp.stem, text[:200])
                edges, warns = [], []
            else:
                edges, warns = _normalise_edges(parsed, node_sets[j])
                parsed_ok += 1
                drops_total += len(warns)
                if warns:
                    log.info("[%s] %d edge(s) dropped: %s",
                             fp.stem, len(warns), warns[:3])
            try:
                _emit_output(fp, args.output_dir / fp.name, edges)
                written += 1
            except Exception as exc:
                log.warning("[%s] failed to write output: %s", fp.stem, exc)

        done = min(i + bs, len(rank_files))
        if done % max(args.log_every, bs) < bs:
            elapsed = time.time() - started
            rate = done / max(elapsed, 1e-6)
            log.info("progress: %d/%d (%.1f sample/s, %.0fs elapsed)",
                     done, len(rank_files), rate, elapsed)

    log.info(
        "done. written=%d parsed_ok=%d parse_failed=%d drops=%d in %.0fs",
        written, parsed_ok, parse_failed, drops_total, time.time() - started,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
