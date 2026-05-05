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
    Create a prompt with BSP rules and a target ROOM SET (no edges) for the
    fine-tuned model. The model must invent its own connectivity; exactly one
    room must own the outside door.

    The argument is still the full ``accessgraph`` dict (so existing call
    sites pass ``sample["accessgraph"]`` unchanged), but only ``nodes`` is
    shown to the model -- ``edges`` are intentionally hidden so the model
    cannot rely on supplied connectivity.

    Used by evaluator.py (--mode trained).
    """
    room_set = target_accessgraph.get('nodes', {}) or {}
    n_rooms = len(room_set)

    prompt = f"""You are an expert at generating hierarchical Binary Space Partition (BSP) tree structures (hypergraphs) that represent floor plans.

Your task is to generate a hypergraph from a SET OF ROOMS. A room set is a dictionary mapping room IDs to room types (e.g., {{"1": "Bedroom", "2": "Kitchen"}}). You are NOT given any connectivity information; you must invent a sensible spatial layout AND a valid adjacency graph between the rooms.

{_OUTSIDE_RULES_TEXT_1}

{_SIMPLE_BSP_RULES_TEXT}

{_ROOMS_ONLY_CONNECTIVITY_RULES}

{_ANGLE_RULES}

# YOUR TASK:

Generate a hypergraph (BSP tree) for the following room set. {_OUTSIDE_RULES_TEXT_2}

### Room Set:
```json
{json.dumps(room_set, indent=2)}
```

## Requirements:
1. Number of leaf nodes MUST equal the number of rooms in the input set ({n_rooms}); do not add, drop, or duplicate any room.
2. Each leaf's mergeid must match a room type from the input set, and the multiset of leaf mergeids must equal the multiset of values in the input set.
3. Infer reasonable connectivity between rooms based on typical architectural patterns. Adjacency must be bidirectional (if A's `connected` lists B, B's `connected` must list A).
4. Exactly ONE leaf must own the outside door -- mark this by including the literal string `"outside"` in that leaf's `connected` array; its box should sit on the apartment boundary so the front door faces the exterior. `"outside"` is NOT itself a leaf and must not appear as a `name`.
5. Follow the binary tree structure (each non-leaf has exactly 2 children).
6. Use proper hierarchical naming (root → rootL/rootR → rootLL/rootLR/rootRL/rootRR, etc.).
7. Ensure area conservation (parent ≈ sum of children).
8. Create realistic centroids for spatial consistency.
9. **CRITICAL - ANGLES**: All angles must be horizontal or vertical (0, π/2, π, or 3π/2 only).
10. **CRITICAL - CONNECTIVITY**: The `connected` array in each LEAF node must contain ONLY the `name` values of OTHER LEAF NODES (plus optionally the literal string `"outside"` for the single leaf that owns the outside door).
   - ✅ CORRECT: `"connected": ["rootLLR", "rootLLLLL", "outside"]`
   - ❌ WRONG: Including internal nodes like "rootLL" or "rootL" in connected
   - ❌ WRONG: Including mergeid values like "Bedroom" or "Corridor"
   - ❌ WRONG: Including the node's own name (NO SELF-LOOPS)
11. Use final=false for internal nodes, final=true for room leaves.

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
        {"role": "system", "content": "You are an expert at generating structured hierarchical BSP tree data for floor plans in JSON format."},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]

    return {"messages": messages}
