#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from rplan.visualize_rplan_hypergraph import _inject_outside_node
from utils.hypergraph import Hypergraph, HypergraphPlotter
from utils.metrics import accessgraph_dict_to_nx
from utils.prompts import (
    _ANGLE_RULES,
    _OUTSIDE_RULES_TEXT_1,
    _SIMPLE_BSP_RULES_TEXT,
    _SIMPLE_CONNECTIVITY_RULES,
)

INSTRUCTION_PROMPT = (
    "You are aware of hypergraph rules based on your previous training."
    "You have been trained to generate hypergraphs from access graphs and know their structure well."
    "A valid hypergraph is a binary BSP tree: every internal node has exactly 2 children, "
    "uses hierarchical names like root/rootL/rootR, has final=false, mergeid=name, and an empty connected list. "
    "Leaves are rooms with final=true, mergeid as the room type, no children, and connected listing only other leaf node names, "
    "with no self-loops and bidirectional room adjacencies. "
    "Parent and child areas should be approximately conserved, split angles must be axis-aligned "
    "(0, pi/2, pi, or 3pi/2), and the Outside node only denotes a boundary door and is not a normal room."
    "Also, each hypergraph is associated with an access graph that describes the connectivity between the rooms."
    "You task now is to edit an EXISTING hypergraph based on a command passed to you. "
    "You will be given a hypergraph, and a prompt that describes the edit you need to perform."
    "You need to edit the hypergraph to satisfy the prompt."
    "You need to return the edited hypergraph in JSON format."
    "If you are asked to remove a room, you need to remove one node of the BSP tree of your choice"
    "Then, you need to adjust partitions to account for reasonable area shifts. "
    "You must NOT return the original hypergraph, but the edited one."
)
INSTRUCTION_PROMPT_GPT = (
    "You are editing a floor-plan hypergraph represented as a hierarchical Binary Space Partition (BSP) tree in JSON format. "
    "A hypergraph represents rooms as leaf nodes of a binary tree and internal BSP splits as non-leaf nodes. "
    "Each non-leaf node must have exactly 2 children, use hierarchical names such as root, rootL, rootR, rootLL, and so on, "
    "have final=false, mergeid equal to name, and an empty connected list. "
    "Each leaf node is a room with final=true, mergeid equal to the room type, no children, and connected containing only the name values of other leaf nodes. "
    "Connected lists must never include internal nodes, mergeid strings, or the node's own name, and room adjacencies should be bidirectional. "
    "Areas should be approximately conserved so a parent area is close to the sum of its children. "
    "Angles describe BSP split orientation and must be axis-aligned only: 0, pi/2, pi, or 3pi/2. "
    "The Outside node, if relevant, only denotes that a room touches the apartment boundary through an outside door and is not a normal room. "
    "When editing, preserve a valid BSP tree and produce a modified hypergraph that follows these structural rules. "
    "Make the requested change concretely rather than copying the input unchanged. "
    "Return only the edited hypergraph JSON with no explanation."
    "You task now is to edit an EXISTING hypergraph based on a command passed to you. "
    "You will be given a hypergraph, and a prompt that describes the edit you need to perform."
    "You need to edit the hypergraph to satisfy the prompt."
    "You need to return the edited hypergraph in JSON format."
    "If you are asked to remove a room, you need to remove one node of the BSP tree of your choice"
    "Then, you need to adjust partitions to account for reasonable area shifts. "
    "You must NOT return the original hypergraph, but the edited one."
)
INSTRUCTION_PROMPT_GPT_RAG = (
    "You are editing a floor-plan hypergraph represented as a hierarchical Binary Space Partition (BSP) tree in JSON format. "
    "A hypergraph represents rooms as leaf nodes of a binary tree and BSP splits as internal nodes. "
    "You will receive retrieved rule snippets and a few similar valid example hypergraphs as reference context. "
    "Use them to preserve structural validity while carrying out the requested edit. "
    "Internal nodes must have exactly 2 children, use hierarchical names such as root/rootL/rootR, have final=false, mergeid=name, and empty connected lists. "
    "Leaf nodes are rooms with final=true, no children, mergeid equal to the room type, and connected containing only other leaf node names. "
    "Connected lists must exclude internal nodes, mergeid strings, and self-loops, and adjacency should be bidirectional. "
    "Areas should remain approximately conserved and split angles must stay axis-aligned only: 0, pi/2, pi, or 3pi/2. "
    "If the requested edit removes or changes rooms, update the BSP tree and connectivity coherently instead of copying the input unchanged. "
    "Return only the edited hypergraph JSON and no explanation."
)
EDIT_OPERATION_GUIDANCE = """\
### Editing Guidance
- You are editing an EXISTING hypergraph, not generating a fresh one from scratch.
- You will be given a current hypergraph, its access graph, retrieved rules, similar examples, and a command describing the requested operation.
- You must apply the requested operation to the current hypergraph and return the EDITED hypergraph JSON.
- Do not return the original hypergraph unchanged unless the command explicitly asks you to preserve it exactly.
- Preserve as much of the original valid structure as possible, except where the requested edit requires a change.

### If the command asks to remove a room
- If the room is specified, remove that room.
- If the room is not specified, choose one reasonable leaf room to remove.
- Delete exactly one leaf room unless the command says otherwise.
- Repair the BSP tree so it remains a valid binary tree after the removal.
- Adjust areas and nearby partitions so the edited apartment remains spatially plausible.
- Update all affected `connected` lists so connectivity is still valid, bidirectional, and free of self-loops.

### If the command asks to add a room
- Find an appropriate place in the tree to introduce a new split and create the new room as a valid leaf.
- The new room must have final=true, a room-type mergeid, no children, valid connected entries, and realistic area and angle values.
- Adjust neighboring partitions to account for reasonable area shifts while preserving a valid BSP structure.

### If the command asks to resize a room
- Find the target room and the surrounding BSP split that controls its allocated area.
- Revise partition areas to give that room more or less space while keeping the rest of the apartment plausible.
- If needed, propagate area changes upward or to nearby siblings, but keep the tree valid.

### If the command asks to swap rooms
- Identify the two rooms and swap their room-type assignments.
- Preserve the rest of the tree structure unless the command requires more changes.
- If needed, adapt local areas to keep the result plausible while retaining valid connectivity.

### Output Requirements
- Return only the edited hypergraph JSON.
- Do not add explanations, markdown, comments, or any surrounding text.
"""
OPENAI_API_KEY = ""
OPENAI_API_URL = "https://api.openai.com/v1/responses"
OPENAI_MODEL_NAME = "gpt-5.4"
API_KEY_PATH = REPO_ROOT / "api_key"


def load_hypergraph(path: Path, sample_id: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return data[str(sample_id)]


def load_openai_api_key() -> str:
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    if API_KEY_PATH.exists():
        return API_KEY_PATH.read_text().strip()
    return ""


def load_bounds(test_dir: Path, sample_id: str) -> dict:
    with open(test_dir / f"{sample_id}.txt") as f:
        raw = json.load(f)
    bounds = raw.get("bounds", {})
    return json.loads(bounds) if isinstance(bounds, str) else bounds


def load_model(model_path: Path, device: str):
    dtype = torch.float16 if device == "cuda" else torch.float32
    if (model_path / "adapter_config.json").exists():
        with open(model_path / "adapter_config.json") as f:
            base_name = json.load(f)["base_model_name_or_path"]
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        base_model = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype)
        model = PeftModel.from_pretrained(base_model, str(model_path))
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=dtype)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model.to(device)
    model.eval()
    return model, tokenizer


def build_prompt(
    accessgraph: dict,
    hypergraph: dict,
    sample_id: str,
    prompt_file: Path | None,
    extra_text: str,
    command_prompt: str,
) -> str:
    accessgraph_json = json.dumps(accessgraph, indent=2)
    hypergraph_json = json.dumps(hypergraph, indent=2)
    if prompt_file is not None:
        template = prompt_file.read_text()
        return (
            template
            .replace("__COMMAND_PROMPT__", command_prompt)
            .replace("__ACCESSGRAPH_JSON__", accessgraph_json)
            .replace("__HYPERGRAPH_JSON__", hypergraph_json)
            .replace("__EXTRA_TEXT__", extra_text)
            .replace("__SAMPLE_ID__", str(sample_id))
        )
    parts = [
        command_prompt,
        "### Current Hypergraph:",
        "```json",
        hypergraph_json,
        "```",
    ]
    return "\n\n".join(part for part in parts if part)


def build_rule_context() -> str:
    return "\n\n".join(
        [
            "### Retrieved Rule Snippets",
            EDIT_OPERATION_GUIDANCE.strip(),
            _OUTSIDE_RULES_TEXT_1.strip(),
            _SIMPLE_BSP_RULES_TEXT.strip(),
            _SIMPLE_CONNECTIVITY_RULES.strip(),
            _ANGLE_RULES.strip(),
        ]
    )


def graph_signature(hg_dict: dict) -> tuple[Counter, int, int]:
    hg = Hypergraph(hg_dict)
    room_counts = Counter(leaf.get("mergeid", "Unknown") for leaf in hg.leaves)
    ag = hg.access_graph
    return room_counts, len(hg.leaves), len(ag.get("edges", []))


def retrieve_examples(hypergraphs_file: Path, sample_id: str, target_hg: dict, k: int) -> list[tuple[str, dict]]:
    with open(hypergraphs_file) as f:
        all_hgs = json.load(f)
    target_rooms, target_leaves, target_edges = graph_signature(target_hg)
    scored = []
    for cand_id, cand_hg in all_hgs.items():
        if str(cand_id) == str(sample_id):
            continue
        cand_rooms, cand_leaves, cand_edges = graph_signature(cand_hg)
        overlap = sum((target_rooms & cand_rooms).values())
        room_l1 = sum(abs(target_rooms[r] - cand_rooms[r]) for r in set(target_rooms) | set(cand_rooms))
        score = (overlap, -room_l1, -abs(target_leaves - cand_leaves), -abs(target_edges - cand_edges))
        scored.append((score, str(cand_id), cand_hg))
    scored.sort(reverse=True)
    return [(cand_id, cand_hg) for _, cand_id, cand_hg in scored[:k]]


def build_examples_context(examples: list[tuple[str, dict]]) -> str:
    parts = ["### Retrieved Similar Examples"]
    for idx, (sample_id, hg_dict) in enumerate(examples, 1):
        hg = Hypergraph(hg_dict)
        room_counts = Counter(leaf.get("mergeid", "Unknown") for leaf in hg.leaves)
        parts.extend(
            [
                f"#### Example {idx} (sample {sample_id})",
                f"Leaf count: {len(hg.leaves)}",
                f"Room counts: {dict(sorted(room_counts.items()))}",
                "```json",
                json.dumps(hg_dict, indent=2),
                "```",
            ]
        )
    return "\n\n".join(parts)


def build_rag_prompt(command_prompt: str, current_hg: dict, accessgraph: dict, rule_context: str, examples_context: str) -> str:
    return "\n\n".join(
        [
            command_prompt,
            rule_context,
            examples_context,
            "### Current Access Graph",
            "```json",
            json.dumps(accessgraph, indent=2),
            "```",
            "### Current Hypergraph",
            "```json",
            json.dumps(current_hg, indent=2),
            "```",
        ]
    )


def build_messages(prompt: str, mode: str) -> list[dict[str, str]]:
    if mode == "local":
        return [
            {"role": "system", "content": INSTRUCTION_PROMPT},
            {"role": "user", "content": prompt},
        ]
    system_prompt = INSTRUCTION_PROMPT_GPT_RAG if mode == "rag" else INSTRUCTION_PROMPT_GPT
    return [
        {"role": "developer", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def generate_local(model, tokenizer, messages: list[dict[str, str]], max_new_tokens: int, temperature: float):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.95,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=False)


def generate_openai(messages: list[dict[str, str]], max_output_tokens: int, temperature: float):
    api_key = load_openai_api_key()
    if not api_key:
        raise ValueError(f"Set OPENAI_API_KEY in the environment or place the key in {API_KEY_PATH}.")
    payload = json.dumps(
        {
            "model": OPENAI_MODEL_NAME,
            "input": messages,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        OPENAI_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI request failed: HTTP {exc.code} {exc.reason}\n{detail}") from exc

    output_parts = []
    for item in body.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                output_parts.append(content.get("text", ""))
    if output_parts:
        return "\n".join(output_parts).strip()
    raise ValueError(f"No text output found in OpenAI response: {json.dumps(body)[:2000]}")


def render_comparison(sample_id: str, command_prompt: str, bounds: dict, input_hg: dict, output_hg: dict, output_path: Path) -> None:
    plotter = HypergraphPlotter()
    input_obj = Hypergraph(input_hg)
    output_obj = Hypergraph(output_hg)
    input_ag = _inject_outside_node(input_obj.access_graph, input_hg)
    output_ag = _inject_outside_node(output_obj.access_graph, output_hg)
    polygon = [(c["X"], c["Y"]) for c in (bounds or {}).get("corners", [])]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_in_ag, ax_in_mesh = axes[0]
    ax_out_ag, ax_out_mesh = axes[1]
    fig.suptitle(str(sample_id), fontsize=14, fontweight="bold")
    fig.text(0.5, 0.955, f"\"{command_prompt}\"", ha="center", va="top", fontsize=20, wrap=True)

    for ag, ax, title in [
        (input_ag, ax_in_ag, "Input Access Graph"),
        (output_ag, ax_out_ag, "Output Access Graph"),
    ]:
        labels_base = {
            "nodes": {k: v for k, v in ag.get("nodes", {}).items() if k != "outside"},
            "edges": [e for e in ag.get("edges", []) if "outside" not in e],
        }
        labels = plotter.make_distinct_labels(labels_base)
        if "outside" in ag.get("nodes", {}):
            labels["outside"] = "Outside"
        plotter.plot_access_graph_on_axis(
            ag,
            ax,
            title=title,
            components=list(nx.connected_components(accessgraph_dict_to_nx(ag))),
            node_labels=labels,
        )

    for hg_obj, ag, ax, title in [
        (input_obj, input_ag, ax_in_mesh, "Input Mesh"),
        (output_obj, output_ag, ax_out_mesh, "Output Mesh"),
    ]:
        try:
            labels_base = {
                "nodes": {k: v for k, v in ag.get("nodes", {}).items() if k != "outside"},
                "edges": [e for e in ag.get("edges", []) if "outside" not in e],
            }
            labels = plotter.make_distinct_labels(labels_base)
            mesh = hg_obj.to_mesh(bounds=bounds)
            plotter.plot_mesh_on_axis(
                mesh,
                ax,
                boundary_polygon=polygon,
                title=title,
                face_labels=hg_obj.mesh_face_labels(mesh, distinct=labels),
            )
            ax.invert_yaxis()
        except Exception as exc:
            ax.text(0.5, 0.5, f"{title} failed:\n{exc}", ha="center", va="center", transform=ax.transAxes, wrap=True)
            ax.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["local", "gpt54", "rag"], default="local")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--hypergraphs-file", type=Path, required=True)
    parser.add_argument("--examples-file", type=Path, default=None)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--test-dir", type=Path, default=Path("/raid/home/nikita/HypergraphFormer/dataset/rplan/dataset_hypergraph_la0_fixed/test"))
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--command-prompt", required=True)
    parser.add_argument("--extra-text", default="")
    parser.add_argument("--tag", default="prompt1")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / ".zero_shot_edit_outputs")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=3328)
    parser.add_argument("--rag-k", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    input_hg = load_hypergraph(args.hypergraphs_file, args.sample_id)
    accessgraph = Hypergraph(input_hg).access_graph
    bounds = load_bounds(args.test_dir, args.sample_id)

    rule_context = ""
    examples_context = ""
    if args.mode == "rag":
        examples_file = args.examples_file or args.hypergraphs_file
        rule_context = build_rule_context()
        examples = retrieve_examples(examples_file, args.sample_id, input_hg, args.rag_k)
        examples_context = build_examples_context(examples)
        prompt = build_rag_prompt(args.command_prompt, input_hg, accessgraph, rule_context, examples_context)
    else:
        prompt = build_prompt(
            accessgraph,
            input_hg,
            args.sample_id,
            args.prompt_file,
            args.extra_text,
            args.command_prompt,
        )

    messages = build_messages(prompt, args.mode)
    if args.mode == "local":
        if args.model is None:
            raise ValueError("--model is required when --mode local")
        model, tokenizer = load_model(args.model, device)
        response = generate_local(model, tokenizer, messages, args.max_new_tokens, args.temperature)
    else:
        response = generate_openai(messages, args.max_new_tokens, args.temperature)

    parsed = Hypergraph.from_qwen_response(response, allow_raw_json_fallback=True)
    if parsed is None:
        raise ValueError("Failed to parse hypergraph JSON from model response")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.sample_id}_{args.tag}"
    json_path = args.output_dir / f"{stem}.json"
    png_path = args.output_dir / f"{stem}.png"
    prompt_path = args.output_dir / f"{stem}_{'full_prompt' if args.mode == 'rag' else 'prompt'}.txt"

    with open(json_path, "w") as f:
        json.dump(parsed.data, f, indent=2)
    prompt_path.write_text(prompt)
    render_comparison(args.sample_id, args.command_prompt, bounds, input_hg, parsed.data, png_path)
    print(json_path)
    print(png_path)
    print(prompt_path)

    if args.mode == "rag":
        rules_path = args.output_dir / f"{stem}_rules.txt"
        examples_path = args.output_dir / f"{stem}_retrieved_examples.txt"
        rules_path.write_text(rule_context)
        examples_path.write_text(examples_context)
        print(rules_path)
        print(examples_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
