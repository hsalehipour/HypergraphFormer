"""
Shared prompt templates for hypergraph generation.

This module centralizes BSP rules text and prompt construction used across
Qwen and GPT generation/evaluation scripts.
"""

import json


_OUTSIDE_RULES_TEXT_1 ="""\
  The only exception is the Outside node. It is a special node that denotes that the room it is connected to has an outside door.
  Therefore, it should be located by the boundary of the apartment and not be surrounded by other rooms. 
  However, this node should not be considered part of the accessgraph or part of the list of rooms.
  """


_OUTSIDE_RULES_TEXT_2 ="""\
  Make sure that the accessgraph respects the Outside property of one of the rooms.
  """


BSP_RULES_TEXT = """\
# CRITICAL STRUCTURE RULES:

## 1. BSP Tree Structure (MUST FOLLOW):
- This is a BINARY tree structure where each non-leaf node has EXACTLY 2 children
- Each node splits into LEFT (L) and RIGHT (R) children
- Naming convention follows the path from root:
  * root → rootL and rootR
  * rootL → rootLL and rootLR
  * rootR → rootRL and rootRR
  * rootLL → rootLLL and rootLLR
  * And so on...

## 2. Node Types:
### Non-Leaf Nodes (Internal BSP splits):
- `final`: false
- `mergeid`: Same as `name` (e.g., "root", "rootL", "rootLL")
- `children`: Array of EXACTLY 2 child nodes [Left, Right]
- `connected`: Empty array []
- Contains: name, area, angle, mergeid, final, children, connected, treeNodeMesh

### Leaf Nodes (Actual rooms):
- `final`: true
- `mergeid`: Room type with ID number (e.g., "Bedroom_1", "Kitchen_2", "Corridor_3")
- `children`: Empty array []
- `connected`: Array containing the `name` values of OTHER LEAF NODES (rooms) this room connects to
  * CRITICAL: Only reference LEAF nodes (final=true), not internal BSP nodes (final=false)
  * Use the hierarchical node names from the `name` field (e.g., ["rootLLR", "rootR", "rootLLLLL"])
  * Example: If room "rootLLLR" connects to rooms "rootLLR", "rootLLLLL", and "rootLR"
    → `"connected": ["rootLLR", "rootLLLLL", "rootLR"]`
- Contains: name, area, angle, mergeid, final, children, connected, treeNodeMesh

## 3. Area Conservation:
- Parent node area = Sum of its children's areas (approximately)
- Areas should be realistic for room types:
  * Bedroom: 10-20 sq units
  * Kitchen: 5-10 sq units
  * Bathroom: 3-6 sq units
  * Corridor: 3-8 sq units
  * Livingroom: 15-30 sq units
  * Balcony: 2-5 sq units

## 4. TreeNodeMesh Properties:
- `Area`: Same as node's area
- `Centroid`: Contains X, Y, Z (always 0), and Mag (magnitude = sqrt(X² + Y²))
- Centroids should be spatially consistent with the tree structure

## 5. Connectivity (Access Graph):
- **CRITICAL**: The `connected` array must ONLY contain `name` values of LEAF NODES (final=true)
- Do NOT include internal BSP nodes (final=false) in connected arrays
- **NO SELF-LOOPS**: A node must NEVER include its own name in its `connected` array
  * ❌ WRONG: A node "rootLL" with `"connected": ["rootLL", "rootLR"]` (contains itself)
  * ✅ CORRECT: A node "rootLL" with `"connected": ["rootLR", "rootR"]` (only other nodes)
- Corridors typically connect to multiple rooms
- Adjacent rooms in the floor plan should list each other's node names in `connected`
- Example from training data:
  * "rootLLLR" (Corridor, leaf node) → `"connected": ["rootLLR", "rootLLLLL", "rootLR"]`
  * All three referenced nodes (rootLLR, rootLLLLL, rootLR) are also leaf nodes
  * Note: "rootLLLR" does NOT include itself in the connected array
- Each connection should be bidirectional (if A connects to B, B should connect to A)

## 6. Angles (MUST BE HORIZONTAL OR VERTICAL):
- Angles represent the orientation of the BSP split line
- **CRITICAL**: Angles must be ONLY horizontal or vertical
- Valid angle values:
  * 0 (or ~0): Horizontal split
  * π/2 (≈1.5708): Vertical split
  * π (≈3.1416): Horizontal split (opposite direction)
  * 3π/2 (≈4.7124): Vertical split (opposite direction)
- Do NOT use diagonal or arbitrary angles
- This ensures axis-aligned splits typical of architectural floor plans"""


def create_structured_prompt(examples, target_accessgraph=None):
    """
    Create a detailed prompt that explains the BSP tree structure.

    Args:
        examples: List of example hypergraphs with accessgraph and hypergraph
        target_accessgraph: Optional target access graph to generate hypergraph for.
            When None, produces a generic generation prompt instead.
    """

    prompt = f"""You are an expert at generating hierarchical Binary Space Partition (BSP) tree structures (hypergraphs) that represent floor plans.

Your task is to learn how to generate a hypergraph from an access graph. An access graph is a simple representation of room connectivity:
- `nodes`: Dictionary mapping room IDs to room types (e.g., {{"1": "Bedroom", "2": "Kitchen"}})
- `edges`: List of connections between rooms (e.g., [["1", "2"], ["2", "3"]])

{_OUTSIDE_RULES_TEXT_1}

You will be shown examples of access graphs paired with their corresponding hypergraphs, then generate a new hypergraph following the same structure and rules.

{BSP_RULES_TEXT}

# TRAINING EXAMPLES (Access Graph → Hypergraph):

Below are examples showing how access graphs map to hypergraphs. Learn from these input-output pairs:

"""

    for i, example in enumerate(examples, 1):
        prompt += f"\n## Example {i} ({example['filename']}):\n\n"
        prompt += "### Input - Access Graph:\n"
        prompt += "```json\n"
        prompt += json.dumps(example['accessgraph'], indent=2)
        prompt += "\n```\n\n"
        prompt += "### Output - Hypergraph:\n"
        prompt += "```json\n"
        prompt += json.dumps(example['hypergraph'], indent=2)
        prompt += "\n```\n"

    if target_accessgraph:
        prompt += f"""

# YOUR TASK:

Generate a hypergraph (BSP tree) for the following access graph. {_OUTSIDE_RULES_TEXT_2}

### Target Access Graph:
```json
{json.dumps(target_accessgraph, indent=2)}
```

## Requirements:
1. Number of leaf nodes MUST equal number of nodes in access graph ({len(target_accessgraph.get('nodes', {}))})
2. Each leaf's mergeid must match a room type from the access graph nodes
3. The connectivity between leaves must match the edges in the access graph
4. Follow the binary tree structure (each non-leaf has exactly 2 children)
5. Use proper hierarchical naming (root → rootL/rootR → rootLL/rootLR/rootRL/rootRR, etc.)
6. Ensure area conservation (parent ≈ sum of children)
7. Create realistic centroids for spatial consistency
8. **CRITICAL - ANGLES**: All angles must be horizontal or vertical (0, π/2, π, or 3π/2 only)
9. **CRITICAL - CONNECTIVITY**: The `connected` array in each LEAF node must contain ONLY the `name` values of OTHER LEAF NODES
   - ✅ CORRECT: `"connected": ["rootLLR", "rootLLLLL", "rootLR"]` (all are leaf node names)
   - ❌ WRONG: Including internal nodes like "rootLL" or "rootL" in connected
   - ❌ WRONG: Including mergeid values like "Bedroom_1" or "Corridor_3"
   - ❌ WRONG: Including the node's own name (NO SELF-LOOPS)
10. Use final=false for internal nodes, final=true for room leaves

Generate ONLY the hypergraph JSON structure, no additional text or explanation:

"""
    else:
        prompt += f"""

# YOUR TASK:

Generate a NEW floor plan hypergraph following the EXACT structure rules above.

## Requirements:
1. Follow the binary tree structure (each non-leaf has exactly 2 children)
2. Use proper hierarchical naming (root → rootL/rootR → rootLL/rootLR/rootRL/rootRR, etc.)
3. Ensure area conservation (parent ≈ sum of children)
4. Create realistic centroids for spatial consistency
5. **CRITICAL - ANGLES**: All angles must be horizontal or vertical (0, π/2, π, or 3π/2 only)
6. **CRITICAL - CONNECTIVITY**: The `connected` array in each LEAF node must contain ONLY the `name` values of OTHER LEAF NODES
   - ✅ CORRECT: `"connected": ["rootLLR", "rootLLLLL", "rootLR"]` (all are leaf node names)
   - ❌ WRONG: Including internal nodes like "rootLL" or "rootL" in connected
   - ❌ WRONG: Including mergeid values like "Bedroom_1" or "Corridor_3"
   - ❌ WRONG: Including the node's own name (NO SELF-LOOPS)
7. Use final=false for internal nodes, final=true for room leaves
8. Generate valid JSON that can be parsed

Generate ONLY the JSON structure, no additional text or explanation:

"""

    return prompt


# Simplified BSP rules (no area ranges, bare room type mergeids) used by
# the fine-tuned model evaluation prompts where examples are omitted.
_SIMPLE_BSP_RULES_TEXT = """\
# CRITICAL STRUCTURE RULES:

## 1. BSP Tree Structure (MUST FOLLOW):
- This is a BINARY tree structure where each non-leaf node has EXACTLY 2 children
- Each node splits into LEFT (L) and RIGHT (R) children
- Naming convention follows the path from root:
  * root → rootL and rootR
  * rootL → rootLL and rootLR
  * rootR → rootRL and rootRR
  * rootLL → rootLLL and rootLLR
  * And so on...

## 2. Node Types:
### Non-Leaf Nodes (Internal BSP splits):
- `final`: false
- `mergeid`: Same as `name` (e.g., "root", "rootL", "rootLL")
- `children`: Array of EXACTLY 2 child nodes [Left, Right]
- `connected`: Empty array []
- Contains: name, area, angle, mergeid, final, children, connected, treeNodeMesh

### Leaf Nodes (Actual rooms):
- `final`: true
- `mergeid`: Room type (e.g., "Bedroom", "Kitchen", "Corridor")
- `children`: Empty array []
- `connected`: Array containing the `name` values of OTHER LEAF NODES (rooms) this room connects to
  * CRITICAL: Only reference LEAF nodes (final=true), not internal BSP nodes (final=false)
  * Use the hierarchical node names from the `name` field (e.g., ["rootLLR", "rootR", "rootLLLLL"])
  * Example: If room "rootLLLR" connects to rooms "rootLLR", "rootLLLLL", and "rootLR"
    → `"connected": ["rootLLR", "rootLLLLL", "rootLR"]`
- Contains: name, area, angle, mergeid, final, children, connected, treeNodeMesh

## 3. Area Conservation:
- Parent node area = Sum of its children's areas (approximately)

## 4. TreeNodeMesh Properties:
- `Area`: Same as node's area
- `Centroid`: Contains X, Y, Z (always 0), and Mag (magnitude = sqrt(X² + Y²))
- Centroids should be spatially consistent with the tree structure"""


_SIMPLE_CONNECTIVITY_RULES = """\
## 5. Connectivity (Access Graph):
- **CRITICAL**: The `connected` array must ONLY contain `name` values of LEAF NODES (final=true)
- Do NOT include internal BSP nodes (final=false) in connected arrays
- **NO SELF-LOOPS**: A node must NEVER include its own name in its `connected` array
  * ❌ WRONG: A node "rootLL" with `"connected": ["rootLL", "rootLR"]` (contains itself)
  * ✅ CORRECT: A node "rootLL" with `"connected": ["rootLR", "rootR"]` (only other nodes)
- Corridors typically connect to multiple rooms
- Adjacent rooms in the floor plan should list each other's node names in `connected`
- Each connection should be bidirectional (if A connects to B, B should connect to A)"""


_ROOMS_ONLY_CONNECTIVITY_RULES = """\
## 5. Connectivity (Must Infer from Room Types):
- **CRITICAL**: The `connected` array must ONLY contain `name` values of LEAF NODES (final=true)
- Do NOT include internal BSP nodes (final=false) in connected arrays
- **NO SELF-LOOPS**: A node must NEVER include its own name in its `connected` array
  * ❌ WRONG: A node "rootLL" with `"connected": ["rootLL", "rootLR"]` (contains itself)
  * ✅ CORRECT: A node "rootLL" with `"connected": ["rootLR", "rootR"]` (only other nodes)
- **Infer reasonable connectivity based on architectural patterns**:
  * Corridors typically connect to multiple rooms (bedrooms, bathrooms, living areas)
  * Bedrooms usually connect to corridors or hallways
  * Bathrooms typically connect to corridors or adjacent bedrooms
  * Kitchen often connects to living/dining areas and corridors
  * Living rooms connect to entry corridors and adjacent spaces
- Each connection should be bidirectional (if A connects to B, B should connect to A)"""


_ANGLE_RULES = """\
## 6. Angles (MUST BE HORIZONTAL OR VERTICAL):
- Angles represent the orientation of the BSP split line
- **CRITICAL**: Angles must be ONLY horizontal or vertical
- Valid angle values:
  * 0 (or ~0): Horizontal split
  * π/2 (≈1.5708): Vertical split
  * π (≈3.1416): Horizontal split (opposite direction)
  * 3π/2 (≈4.7124): Vertical split (opposite direction)
- Do NOT use diagonal or arbitrary angles
- This ensures axis-aligned splits typical of architectural floor plans"""


_SIMPLE_REQUIREMENTS = """\
## Requirements:
1. Number of leaf nodes MUST equal number of nodes in access graph ({num_nodes})
2. Each leaf's mergeid must match a room type from the access graph nodes
3. The connectivity between leaves must match the edges in the access graph
4. Follow the binary tree structure (each non-leaf has exactly 2 children)
5. Use proper hierarchical naming (root → rootL/rootR → rootLL/rootLR/rootRL/rootRR, etc.)
6. Ensure area conservation (parent ≈ sum of children)
7. Create realistic centroids for spatial consistency
8. **CRITICAL - ANGLES**: All angles must be horizontal or vertical (0, π/2, π, or 3π/2 only)
9. **CRITICAL - CONNECTIVITY**: The `connected` array in each LEAF node must contain ONLY the `name` values of OTHER LEAF NODES
   - ✅ CORRECT: `"connected": ["rootLLR", "rootLLLLL", "rootLR"]` (all are leaf node names)
   - ❌ WRONG: Including internal nodes like "rootLL" or "rootL" in connected
   - ❌ WRONG: Including mergeid values like "Bedroom" or "Corridor"
   - ❌ WRONG: Including the node's own name (NO SELF-LOOPS)
10. Use final=false for internal nodes, final=true for room leaves"""


def create_simple_prompt(target_accessgraph):
    """
    Create a prompt with BSP rules and target access graph but no few-shot examples.
    Used by evaluator.py (--mode trained) for fine-tuned model evaluation.
    """
    prompt = f"""You are an expert at generating hierarchical Binary Space Partition (BSP) tree structures (hypergraphs) that represent floor plans.

Your task is to generate a hypergraph from an access graph. An access graph is a simple representation of room connectivity:
- `nodes`: Dictionary mapping room IDs to room types (e.g., {{"1": "Bedroom", "2": "Kitchen"}})
- `edges`: List of connections between rooms (e.g., [["1", "2"], ["2", "3"]])

{_OUTSIDE_RULES_TEXT_1}

{_SIMPLE_BSP_RULES_TEXT}

{_SIMPLE_CONNECTIVITY_RULES}

{_ANGLE_RULES}

# YOUR TASK:

Generate a hypergraph (BSP tree) for the following access graph. {_OUTSIDE_RULES_TEXT_2}

### Access Graph:
```json
{json.dumps(target_accessgraph, indent=2)}
```

{_SIMPLE_REQUIREMENTS.format(num_nodes=len(target_accessgraph.get('nodes', {})))}

Generate ONLY the hypergraph JSON structure, no additional text or explanation:"""

    return prompt


def create_simple_prompt_rooms_only(target_accessgraph):
    """
    Create a prompt using only room types (no connectivity edges).
    Tests if the model can infer reasonable connectivity from room types alone.
    Used by evaluator.py (--mode rooms-only).
    """
    room_types = target_accessgraph.get('nodes', {})

    prompt = f"""You are an expert at generating hierarchical Binary Space Partition (BSP) tree structures (hypergraphs) that represent floor plans.

Your task is to generate a hypergraph from a list of room types. You are given the types of rooms that should be in the floor plan, but NOT their connectivity. You must infer reasonable connectivity based on typical architectural patterns.

You are provided with:
- A list of room types (e.g., {{"1": "Bedroom", "2": "Kitchen", "3": "Bathroom"}})

{_SIMPLE_BSP_RULES_TEXT}

{_ROOMS_ONLY_CONNECTIVITY_RULES}

{_ANGLE_RULES}

# YOUR TASK:

Generate a hypergraph (BSP tree) for the following room types. You must infer reasonable connectivity between rooms based on typical architectural patterns.

### Room Types:
```json
{json.dumps(room_types, indent=2)}
```

## Requirements:
1. Number of leaf nodes MUST equal number of room types provided ({len(room_types)})
2. Each leaf's mergeid must match a room type from the list above
3. **Infer reasonable connectivity** between rooms based on architectural norms (corridors connect multiple rooms, bedrooms connect to corridors, etc.)
4. Follow the binary tree structure (each non-leaf has exactly 2 children)
5. Use proper hierarchical naming (root → rootL/rootR → rootLL/rootLR/rootRL/rootRR, etc.)
6. Ensure area conservation (parent ≈ sum of children)
7. Create realistic centroids for spatial consistency
8. **CRITICAL - ANGLES**: All angles must be horizontal or vertical (0, π/2, π, or 3π/2 only)
9. **CRITICAL - CONNECTIVITY**: The `connected` array in each LEAF node must contain ONLY the `name` values of OTHER LEAF NODES
   - ✅ CORRECT: `"connected": ["rootLLR", "rootLLLLL", "rootLR"]` (all are leaf node names)
   - ❌ WRONG: Including internal nodes like "rootLL" or "rootL" in connected
   - ❌ WRONG: Including mergeid values like "Bedroom" or "Corridor"
   - ❌ WRONG: Including the node's own name (NO SELF-LOOPS)
10. Use final=false for internal nodes, final=true for room leaves

Generate ONLY the hypergraph JSON structure, no additional text or explanation:"""

    return prompt


def create_instruction_prompt():
    """
    Create the instruction prompt used for training data preparation.
    Contains BSP rules and task description but no examples or target access graph.
    """
    prompt = f"""You are an expert at generating hierarchical Binary Space Partition (BSP) tree structures (hypergraphs) that represent floor plans.

Your task is to generate a hypergraph from an access graph. An access graph is a simple representation of room connectivity:
- `nodes`: Dictionary mapping room IDs to room types (e.g., {{"1": "Bedroom", "2": "Kitchen"}})
- `edges`: List of connections between rooms (e.g., [["1", "2"], ["2", "3"]])

{_OUTSIDE_RULES_TEXT_1}

{_SIMPLE_BSP_RULES_TEXT}

{_SIMPLE_CONNECTIVITY_RULES}

{_ANGLE_RULES}

# YOUR TASK:

Generate a hypergraph (BSP tree) for the following access graph. {_OUTSIDE_RULES_TEXT_2}"""

    return prompt


def create_training_sample(accessgraph, hypergraph):
    """
    Create a single training sample in chat format for fine-tuning.

    Args:
        accessgraph: Access graph dictionary
        hypergraph: Ground truth hypergraph dictionary

    Returns:
        dict: Training sample with 'messages' field in chat format
    """
    instruction = create_instruction_prompt()
    user_content = f"""{instruction}

### Access Graph:
```json
{json.dumps(accessgraph, indent=2)}
```

{_SIMPLE_REQUIREMENTS.format(num_nodes=len(accessgraph.get('nodes', {})))}

Generate ONLY the hypergraph JSON structure, no additional text or explanation:"""

    assistant_content = json.dumps(hypergraph, indent=2)

    messages = [
        {"role": "system", "content": BSP_SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]

    return {"messages": messages}


# System message paired with BSP user prompts during training.
# Exposed so callers (evaluator/generators) can mirror it at inference.
BSP_SYSTEM_MESSAGE = (
    "You are an expert at generating structured hierarchical BSP tree data "
    "for floor plans in JSON format."
)


# ---------------------------------------------------------------------------
# House-Diffusion JSON prompts
#
# These mirror the prompt format used by
# ``scripts/data_preparation_housediffusion_llm.py`` so that the evaluator can
# reproduce the EXACT user/assistant template the model was trained on.
# Single source of truth: data prep imports these helpers.
# ---------------------------------------------------------------------------

HD_SYSTEM_MESSAGE = (
    "You are an expert at generating apartment floor plan layouts in "
    "House-Diffusion JSON format from access graphs. Each layout is a single "
    "JSON object describing room types, axis-aligned bounding boxes, and "
    "wall/door edges in pixel coordinates."
)


HD_FORMAT_RULES_TEXT = """\
# CRITICAL STRUCTURE RULES:

## 1. House-Diffusion JSON Schema (MUST FOLLOW):
The output is a single JSON object with EXACTLY these four fields:

### 1.1 `room_type` -- list[int]
- One integer category id per layout entry, in canonical order: rooms first, then door entries (interior doors, then a single front door at the end).
- Standard RPLAN-style category IDs:
    1  = Living Room       2  = Kitchen        3  = Bedroom (bed)
    4  = Bathroom (bath)   5  = Balcony        6  = Entrance / Hall
    7  = Dining Room       8  = Study Room    10  = Storage
   15  = Front Door       17  = Interior Door

### 1.2 `boxes` -- list[[x1, y1, x2, y2]]
- One axis-aligned bounding box per entry in `room_type`, SAME ORDER and SAME LENGTH.
- Coordinates are integer-valued pixels in `[0, 256]`; (x1, y1) is the top-left corner, (x2, y2) is the bottom-right corner (so x1 < x2 and y1 < y2).

### 1.3 `edges` -- list[[x1, y1, x2, y2, ra, rb]]
- One entry per wall / door segment that bounds a node (a room OR a door entry).
- (x1, y1, x2, y2) are the integer pixel endpoints of the segment.
- `ra` is the **category id** (a value taken from `room_type`) of the node that emits this segment.
- `rb` is **0** when the segment lies on the apartment exterior, otherwise the **category id** of the node on the other side of the wall. So interior doors appear as `rb = 17` for the rooms they connect, and the front door appears as `rb = 15`.

### 1.4 `ed_rm` -- list[list[int]]
- One entry per row of `edges`, in the SAME ORDER as `edges`.
- Each entry is a list of **node indices** (0-based positions into `room_type` / `boxes`) that the segment touches. The indices can refer to ROOMS or to DOOR entries (categories 15 / 17), not just to rooms.
- A length-1 entry `[i]` means the segment lies on the apartment exterior, owned by node `i`; the corresponding `edges` row has `rb == 0`.
- A length-2 entry `[i, j]` means the segment is shared between nodes `i` and `j`; the corresponding `edges` row has `rb != 0`, and the invariant `room_type[i] == ra` and `room_type[j] == rb` MUST hold.
- `ed_rm` is what disambiguates `edges` whenever multiple rooms share a category (e.g. two bedrooms): `ra` / `rb` only carry types, but `ed_rm` carries the exact two entries.

## 2. Geometric Consistency:
- For every room, the union of its segments in `edges` must form a closed axis-aligned polygon contained inside that room's `boxes` rectangle.
- Walls are emitted segment-by-segment; do not rely on implicit closures.
- All coordinates are integers in `[0, 256]` (the canonical House-Diffusion grid).
- `len(boxes) == len(room_type)` and `len(ed_rm) == len(edges)`.

## 3. Connectivity (Access Graph -> JSON):
- The access-graph `edges` list contains TWO kinds of entries:
    * room-to-room edges `[a, b]`, where both endpoints are room ids that appear as keys in `nodes`, and
    * outside edges `[r, "outside"]`, where the literal string `"outside"` indicates that room `r` carries the apartment's front door. `"outside"` is NOT a real room and is NOT a key in `nodes`; therefore room `r`'s box should sit on the apartment boundary so its front door faces the exterior.
- The number of ROOM entries in `room_type` (entries with category id NOT 15 and NOT 17) MUST equal the number of nodes in the access graph (i.e., the count of keys in `nodes`; the `"outside"` reference does NOT count).
- Each room entry's category id must match the corresponding access-graph node's room type, using this canonical mapping:
    living=1, kitchen=2, bed=3, bath=4, balcony=5, entrance=6, dining=7, study=8, storage=10.
- For every room-to-room edge `[a, b]` in the access graph, the JSON must realize that connection as an INTERIOR DOOR: an entry with category id 17 in `room_type` whose segment rows have `ed_rm = [door_index, room_index]` for each of the two connected rooms (so room `a` and room `b` end up sharing exactly one interior door).
- For every outside edge `[r, "outside"]` in the access graph, the JSON must include exactly one Front-Door entry (category 15) in `room_type` whose `ed_rm` rows pair it with room `r`; room `r` must therefore have at least one row in `edges` with `rb == 15`."""


HD_REQUIREMENTS_TMPL = """\
## Requirements:
1. The number of ROOM entries in `room_type` (entries with category id != 15 and != 17) MUST equal {num_nodes} (number of access-graph nodes; the `"outside"` reference does NOT count).
2. Each room's category id must match the corresponding access-graph room type.
3. Every room-to-room edge `[a, b]` in the access graph must be realized by an Interior-Door entry (category 17) whose `ed_rm` rows pair it with both rooms.
4. Every outside edge `[r, "outside"]` in the access graph must be realized by a Front-Door entry (category 15) whose `ed_rm` rows pair it with room `r`; that room's box must sit on the apartment boundary.
5. All coordinates are integers in `[0, 256]`.
6. `len(boxes) == len(room_type)` and `len(ed_rm) == len(edges)`.
7. Each room's `boxes` rectangle must enclose all of its segments in `edges`.
8. Output ONLY the JSON object, no markdown fences and no commentary."""


def create_hd_instruction_prompt():
    """Long-form instruction explaining the HD JSON format. No examples."""
    return f"""You are an expert at generating apartment floor plan layouts in the House-Diffusion JSON format that represent floor plans.

Your task is to generate a House-Diffusion JSON object from an access graph. An access graph is a simple representation of room connectivity:
- `nodes`: Dictionary mapping room IDs to room types (e.g., {{"1": "bed", "2": "kitchen"}}).
- `edges`: List of connections. Each entry is either a room-to-room pair (e.g. `["1", "2"]`) or an outside reference (e.g. `["1", "outside"]`). The literal string `"outside"` is NOT a room and is NOT listed in `nodes`; it marks the room that owns the apartment's front door (i.e., the entry-door room), and that room's box should therefore lie on the apartment boundary so its front door can face the exterior.

{HD_FORMAT_RULES_TEXT}

# YOUR TASK:

Generate a House-Diffusion JSON object for the following access graph."""


def create_simple_prompt_hd(target_accessgraph):
    """User-side prompt for HD JSON generation (matches training).

    Returns the SAME ``user`` content emitted by
    ``create_hd_training_sample``. Used by ``evaluator.py`` (--mode trained,
    --prompt-format hd) so the fine-tuned model sees the exact template it
    was trained on.
    """
    instruction = create_hd_instruction_prompt()
    return f"""{instruction}

### Access Graph:
```json
{json.dumps(target_accessgraph, indent=2)}
```

{HD_REQUIREMENTS_TMPL.format(num_nodes=len(target_accessgraph.get('nodes', {})))}

Generate ONLY the House-Diffusion JSON object, no additional text or explanation:"""


def create_hd_training_sample(accessgraph, hd_json):
    """Build one chat-format SFT sample (system + user + assistant) for HD."""
    user_content = create_simple_prompt_hd(accessgraph)
    assistant_content = json.dumps(hd_json, indent=2)

    return {
        "messages": [
            {"role": "system", "content": HD_SYSTEM_MESSAGE},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }
