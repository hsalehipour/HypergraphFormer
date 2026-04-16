"""
Convert hypergraph data to RPLAN format compatible with HouseGAN++ and House-Diffusion.

Room Type Mapping (hypergraph → RPLAN for HouseGAN++/House-Diffusion compatibility):
    hypergraph 0 (living)  → RPLAN 1 (living_room)
    hypergraph 1 (bed)     → RPLAN 3 (bedroom)
    hypergraph 2 (kitchen) → RPLAN 2 (kitchen)
    hypergraph 3 (bath)    → RPLAN 4 (bathroom)
    hypergraph 10 (foyer)  → RPLAN 6 (entrance)
    hypergraph 11 (extra)  → RPLAN 10 (storage)
    hypergraph 17 (door)   → RPLAN 17 (interior_door)
    
This ensures hypergraph data is compatible with HouseGAN++'s/House-Diffusion's pretrained RPLAN model.
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from hypergraph.api.lib.tools import RGL
import clr
clr.AddReference("System.Collections")
from System.Collections.Generic import List as DotNetList


def _areas_to_ratios(node: dict) -> None:
    """Convert absolute areas to parent-relative ratios in-place (recursive)."""
    parent_area = node.get("area", 0)
    for child in node.get("children", []):
        _areas_to_ratios(child)
        if parent_area > 0:
            child["area"] = child["area"] / parent_area
    if node.get("name") == "root":
        node["area"] = 1.0


def _deserialize_as_ratio(hypergraph_json_str: str):
    """Deserialize a hypergraph JSON string, converting to ratio-based areas."""
    import copy
    tree = json.loads(hypergraph_json_str)
    _areas_to_ratios(tree)
    return RGL.DataNode.deserializeDataNode(json.dumps(tree))


def calculate_polygon_center(poly_vertices: List[List[Tuple[float, float]]]) -> Tuple[float, float]:
    """
    Calculate the center of the bounding box that wraps all polygons.
    
    Args:
        poly_vertices: List of polygons, where each polygon is a list of (x, y) tuples
        
    Returns:
        Tuple of (center_x, center_y)
    """
    all_x = []
    all_y = []
    
    for polygon in poly_vertices:
        for x, y in polygon:
            all_x.append(x)
            all_y.append(y)
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    return (center_x, center_y)


def normalize_polygons(poly_vertices: List[List[Tuple[float, float]]], target_size: int = 64) -> List[List[Tuple[float, float]]]:
    """
    Zero-center and scale polygons to fit within [0, target_size] range.
    The floorplan center will be placed at (target_size/2, target_size/2).
    
    Args:
        poly_vertices: List of polygons, where each polygon is a list of (x, y) tuples
        target_size: Target size for normalized coordinates (default: 64)
        
    Returns:
        List of normalized polygons with center at (target_size/2, target_size/2)
    """
    # Step 1: Calculate center
    center_x, center_y = calculate_polygon_center(poly_vertices)
    
    # Step 2: Zero-center all vertices (center at origin)
    centered_polygons = []
    for polygon in poly_vertices:
        centered_poly = [(x - center_x, y - center_y) for x, y in polygon]
        centered_polygons.append(centered_poly)
    
    # Step 3: Find bounds after centering
    all_x = []
    all_y = []
    for polygon in centered_polygons:
        for x, y in polygon:
            all_x.append(x)
            all_y.append(y)
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    # Step 4: Calculate scale factor to fit within target_size
    # Leave some margin so floorplan doesn't touch edges
    width = max_x - min_x
    height = max_y - min_y
    scale = target_size / max(width, height)  # Use max to maintain aspect ratio
    
    # Step 5: Scale and shift to center of image (target_size/2, target_size/2)
    image_center = target_size / 2.0
    
    normalized_polygons = []
    for polygon in centered_polygons:
        normalized_poly = [
            (round(x * scale + image_center), 
             round(y * scale + image_center))
            for x, y in polygon
        ]
        normalized_polygons.append(normalized_poly)
    
    return normalized_polygons


# Room type mapping: hypergraph string names → hypergraph IDs
hypergraph_ROOM_TYPE_MAP: Dict[str, int] = {
    'living': 0,
    'bed': 1,
    'kitchen': 2,
    'bath': 3,
    'foyer': 10,
    'extra': 11
}

# Room type mapping: hypergraph IDs → RPLAN IDs (for HouseGAN++/House-Diffusion compatibility)
hypergraph_TO_RPLAN_MAP: Dict[int, int] = {
    0: 1,   # hypergraph living → RPLAN living_room
    1: 3,   # hypergraph bed → RPLAN bedroom
    2: 2,   # hypergraph kitchen → RPLAN kitchen
    3: 4,   # hypergraph bath → RPLAN bathroom
    10: 6,  # hypergraph foyer → RPLAN entrance
    11: 10, # hypergraph extra → RPLAN storage
    17: 17  # hypergraph door → RPLAN interior_door (already aligned)
}


def extract_polygons_and_room_types(data: dict) -> Tuple[List[List[Tuple[float, float]]], List[int]]:
    """
    Extract polygon vertices and room types from a hypergraph dictionary.
    
    This extracts individual room polygons from the DataNode tree's leaf nodes.
    Each leaf node has a treeNodeMesh that contains the geometry for that specific room.
    
    Args:
        data: Dictionary containing hypergraph data with 'bounds' and 'split' fields
        
    Returns:
        Tuple of (polygons, room_types):
            - polygons: List of polygons, where each polygon is a list of (x, y) tuples
            - room_types: List of room type IDs (integers) corresponding to each polygon
    """
    
    # Step 1: Parse the boundary JSON string
    bounds_data = json.loads(data['bounds'])
    corners = bounds_data['corners']
    
    # Step 2: Create Vec3d points from the corners
    points_list = DotNetList[RGL.Vec3d]()
    for corner in corners:
        point = RGL.Vec3d(x=corner['X'], y=corner['Y'], z=corner['Z'])
        points_list.Add(point)
    
    # Step 3: Create NFace (polygon) from points
    face_bounds = RGL.NFace(points_list)
    
    # Step 4: Create NMesh from the face
    nface_list = DotNetList[RGL.NFace]()
    nface_list.Add(face_bounds)
    boundary_mesh = RGL.NMesh(nface_list)
    
    # Step 5: Load data_node from hypergraph split (ratio-based for numerical stability)
    hypergraph_json_str = data['split']
    data_node = _deserialize_as_ratio(hypergraph_json_str)
    
    # Step 6: Subdivide the boundary mesh using the hypergraph structure
    generated_mesh = RGL.DataNode.subdivideWithMeshRatio(data_node, boundary_mesh)
    
    # Step 7: Extract polygon vertices and room types from generated_mesh
    polygons = []
    room_types = []
    
    for i in range(generated_mesh.faceList.Count):
        face = generated_mesh.faceList[i]
        
        # Extract vertices from edges
        edges = RGL.RhConvert.NFaceToLineList(face)
        vertices = []
        
        # Extract start point of each edge (forming a closed polygon)
        for j in range(edges.Count):
            edge = edges[j]
            vertices.append((edge.start.X, edge.start.Y))
        
        # Only add if we got valid vertices
        if len(vertices) >= 3:
            polygons.append(vertices)
            
            # Extract room type and map to RPLAN ID
            room_type_str = getattr(face, 'merge_id', 'extra')
            hypergraph_id = hypergraph_ROOM_TYPE_MAP.get(room_type_str, 11)  # Default to 'extra' (11)
            rplan_id = hypergraph_TO_RPLAN_MAP.get(hypergraph_id, hypergraph_id)  # Map to RPLAN
            room_types.append(rplan_id)
    
    return polygons, room_types


def extract_adjacency_triples(data: dict) -> List[List[int]]:
    """
    Extract adjacency triples from hypergraph data (room-to-room only, no doors).
    
    Two rooms are ADJACENT if they share a geometric edge (not just connected by door).
    
    Triples format: [room_idx_1, relationship, room_idx_2]
    where relationship = 1 (adjacent/share edge) or -1 (not adjacent)
    
    Args:
        data: Dictionary containing hypergraph data with 'bounds' and 'split' fields
        
    Returns:
        List of triples [k, relationship, l] for all room pairs where k < l
    """
    # Step 1: Parse boundary and create mesh
    bounds_data = json.loads(data['bounds'])
    corners = bounds_data['corners']
    
    points_list = DotNetList[RGL.Vec3d]()
    for corner in corners:
        point = RGL.Vec3d(x=corner['X'], y=corner['Y'], z=corner['Z'])
        points_list.Add(point)
    
    face_bounds = RGL.NFace(points_list)
    nface_list = DotNetList[RGL.NFace]()
    nface_list.Add(face_bounds)
    boundary_mesh = RGL.NMesh(nface_list)
    
    # Step 2: Load and subdivide (ratio-based for numerical stability)
    hypergraph_json_str = data['split']
    data_node = _deserialize_as_ratio(hypergraph_json_str)
    generated_mesh = RGL.DataNode.subdivideWithMeshRatio(data_node, boundary_mesh)
    
    # Step 3: Update adjacency matrix to populate neighborNFace
    generated_mesh.UpdateAdjacencyMatrix()
    
    num_rooms = generated_mesh.faceList.Count
    
    # Step 4: Build adjacency set using geometric neighbors (neighborNFace)
    adjacency_set = set()
    face_to_idx = {}  # Map face object to its index
    
    for i in range(num_rooms):
        face = generated_mesh.faceList[i]
        face_to_idx[face] = i
    
    for i in range(num_rooms):
        face = generated_mesh.faceList[i]
        
        # Get geometric neighbors (faces that share an edge)
        if hasattr(face, 'neighborNFace') and face.neighborNFace:
            for j in range(face.neighborNFace.Count):
                neighbor_face = face.neighborNFace[j]
                if neighbor_face in face_to_idx:
                    neighbor_idx = face_to_idx[neighbor_face]
                    # Store as sorted tuple to avoid duplicates
                    edge = tuple(sorted([i, neighbor_idx]))
                    adjacency_set.add(edge)
    
    # Step 5: Generate triples for all room pairs
    triples = []
    for k in range(num_rooms):
        for l in range(k + 1, num_rooms):
            edge = (k, l)
            if edge in adjacency_set:
                triples.append([k, 1, l])  # Adjacent (share edge)
            else:
                triples.append([k, -1, l])  # Not adjacent
    
    return triples


def extract_adjacency_triples_with_doors(data: dict, num_doors: int) -> List[List[int]]:
    """
    Extract adjacency triples INCLUDING doors as nodes (RPLAN-compatible).
    
    This matches RPLAN's build_graph() function exactly:
    - Nodes include ALL rooms + ALL doors
    - Room indices: 0 to (num_rooms-1)
    - Door indices: num_rooms to (num_rooms+num_doors-1)
    - Two nodes are adjacent if they share a geometric edge OR are connected by a door
    
    Triples format: [node_idx_1, relationship, node_idx_2]
    where relationship = 1 (adjacent) or -1 (not adjacent)
    
    Args:
        data: Dictionary containing hypergraph data with 'bounds', 'split', 'connected' fields
        num_doors: Number of door nodes to include
        
    Returns:
        List of triples [k, relationship, l] for all node pairs where k < l
    """
    # Step 1: Parse boundary and create mesh
    bounds_data = json.loads(data['bounds'])
    corners = bounds_data['corners']
    
    points_list = DotNetList[RGL.Vec3d]()
    for corner in corners:
        point = RGL.Vec3d(x=corner['X'], y=corner['Y'], z=corner['Z'])
        points_list.Add(point)
    
    face_bounds = RGL.NFace(points_list)
    nface_list = DotNetList[RGL.NFace]()
    nface_list.Add(face_bounds)
    boundary_mesh = RGL.NMesh(nface_list)
    
    # Step 2: Load and subdivide (ratio-based for numerical stability)
    hypergraph_json_str = data['split']
    data_node = _deserialize_as_ratio(hypergraph_json_str)
    generated_mesh = RGL.DataNode.subdivideWithMeshRatio(data_node, boundary_mesh)
    
    # Step 3: Update adjacency matrix to populate neighborNFace
    generated_mesh.UpdateAdjacencyMatrix()
    
    num_rooms = generated_mesh.faceList.Count
    
    # Step 4: Build edge-to-nodes mapping (similar to RPLAN's eds_to_rms)
    # Each "edge" represents a relationship: geometric edge OR door connection
    edge_to_nodes = []
    
    # 4a. Add geometric adjacency edges (rooms that share walls)
    face_to_idx = {}
    for i in range(num_rooms):
        face = generated_mesh.faceList[i]
        face_to_idx[face] = i
    
    for i in range(num_rooms):
        face = generated_mesh.faceList[i]
        
        # Get geometric neighbors (faces that share an edge)
        if hasattr(face, 'neighborNFace') and face.neighborNFace:
            for j in range(face.neighborNFace.Count):
                neighbor_face = face.neighborNFace[j]
                if neighbor_face in face_to_idx:
                    neighbor_idx = face_to_idx[neighbor_face]
                    if i < neighbor_idx:  # Avoid duplicates
                        edge_to_nodes.append([i, neighbor_idx])
    
    # 4b. Add door connections (doors connect specific room pairs)
    # Extract connections from child nodes (where they're actually stored)
    all_children = RGL.DataNode.GetAllChildren(data_node)
    
    # Build room name to index mapping
    name_to_idx = {}
    idx = 0
    for i in range(all_children.Count):
        child = all_children[i]
        if child.final:  # Leaf nodes only
            name_to_idx[child.name] = idx
            idx += 1
    
    # Extract connections from the 'connected' field in each child node
    connections = set()
    for i in range(all_children.Count):
        child = all_children[i]
        if child.final and hasattr(child, 'connected'):
            room_name = child.name
            if room_name in name_to_idx:
                room_idx = name_to_idx[room_name]
                connected_names = list(child.connected)
                for neighbor_name in connected_names:
                    if neighbor_name in name_to_idx:
                        neighbor_idx = name_to_idx[neighbor_name]
                        # Create a tuple with smaller index first for consistency
                        edge = tuple(sorted([room_idx, neighbor_idx]))
                        connections.add(edge)
    
    # Convert set to list
    connections = list(connections)
    
    # Each door creates an edge connecting two rooms
    # Doors are indexed from num_rooms onwards
    # Take only the first num_doors connections (one connection per door)
    for door_idx, (room1, room2) in enumerate(list(connections)[:num_doors]):
        node_idx_door = num_rooms + door_idx
        # Door connects to both rooms
        edge_to_nodes.append([room1, node_idx_door])
        edge_to_nodes.append([room2, node_idx_door])
    
    # Step 5: Generate triples for ALL node pairs (rooms + doors)
    # This matches RPLAN's build_graph() exactly:
    # for k in range(len(nodes)):
    #     for l in range(len(nodes)):
    #         if l > k:
    #             is_adjacent = any([True for e_map in eds_to_rms if (l in e_map) and (k in e_map)])
    
    total_nodes = num_rooms + num_doors
    triples = []
    
    for k in range(total_nodes):
        for l in range(k + 1, total_nodes):
            # Check if k and l are adjacent (share an edge)
            is_adjacent = any((k in e_map and l in e_map) for e_map in edge_to_nodes)
            
            if is_adjacent:
                triples.append([k, 1, l])  # Adjacent
            else:
                triples.append([k, -1, l])  # Not adjacent
    
    return triples


def extract_doors(data: dict, door_width: float = 0.8, door_thickness: float = 0.1) -> List[List[Tuple[float, float]]]:
    """
    Extract door polygons from hypergraph based on 'connected' field.
    
    Uses a two-phase approach:
    1. Find ALL shared walls between room pairs
    2. Place doors on walls where hypergraph indicates connection
    
    Args:
        data: Dictionary containing hypergraph data with 'bounds' and 'split' fields
        door_width: Door width in meters (default: 0.8m, similar to real door width)
        door_thickness: Door thickness in meters (default: 0.1m for visibility)
        
    Returns:
        List of door polygons, where each polygon is a list of (x, y) tuples (4 corners)
    """
    # Step 1: Parse boundary and create mesh
    bounds_data = json.loads(data['bounds'])
    corners = bounds_data['corners']
    
    points_list = DotNetList[RGL.Vec3d]()
    for corner in corners:
        point = RGL.Vec3d(x=corner['X'], y=corner['Y'], z=corner['Z'])
        points_list.Add(point)
    
    face_bounds = RGL.NFace(points_list)
    nface_list = DotNetList[RGL.NFace]()
    nface_list.Add(face_bounds)
    boundary_mesh = RGL.NMesh(nface_list)
    
    # Step 2: Load and subdivide (ratio-based for numerical stability)
    hypergraph_json_str = data['split']
    data_node = _deserialize_as_ratio(hypergraph_json_str)
    generated_mesh = RGL.DataNode.subdivideWithMeshRatio(data_node, boundary_mesh)
    
    # Step 3: Build room name to index mapping
    all_children = RGL.DataNode.GetAllChildren(data_node)
    name_to_idx = {}
    idx = 0
    
    for i in range(all_children.Count):
        child = all_children[i]
        if child.final:  # Leaf nodes only
            name_to_idx[child.name] = idx
            idx += 1
    
    # Step 4: Extract connections from hypergraph 'connected' field
    connections = set()
    for i in range(all_children.Count):
        child = all_children[i]
        if child.final and hasattr(child, 'connected'):
            room_name = child.name
            if room_name in name_to_idx:
                room_idx = name_to_idx[room_name]
                connected_names = list(child.connected)
                for neighbor_name in connected_names:
                    if neighbor_name in name_to_idx:
                        neighbor_idx = name_to_idx[neighbor_name]
                        edge = tuple(sorted([room_idx, neighbor_idx]))
                        connections.add(edge)
    
    # Step 5: Find ALL shared walls between room pairs (similar to convert_hypergraph_to_rplan)
    # This creates a mapping: (room_idx1, room_idx2) -> shared_wall_segment
    shared_walls = {}
    tolerance = 0.05
    
    num_rooms = generated_mesh.faceList.Count
    for i in range(num_rooms):
        for j in range(i + 1, num_rooms):
            face_i = generated_mesh.faceList[i]
            face_j = generated_mesh.faceList[j]
            
            edges_i = RGL.RhConvert.NFaceToLineList(face_i)
            edges_j = RGL.RhConvert.NFaceToLineList(face_j)
            
            # Find overlapping collinear edges between these two rooms
            for edge_i in edges_i:
                for edge_j in edges_j:
                    # Calculate direction vectors
                    dx_i = edge_i.end.X - edge_i.start.X
                    dy_i = edge_i.end.Y - edge_i.start.Y
                    dx_j = edge_j.end.X - edge_j.start.X
                    dy_j = edge_j.end.Y - edge_j.start.Y
                    
                    len_i = (dx_i**2 + dy_i**2)**0.5
                    len_j = (dx_j**2 + dy_j**2)**0.5
                    
                    if len_i < 0.001 or len_j < 0.001:
                        continue
                    
                    # Normalize
                    dx_i /= len_i
                    dy_i /= len_i
                    dx_j /= len_j
                    dy_j /= len_j
                    
                    # Check if parallel
                    dot_product = dx_i * dx_j + dy_i * dy_j
                    if abs(abs(dot_product) - 1.0) > 0.01:  # Not parallel
                        continue
                    
                    # Find overlap region
                    if abs(dx_i) > abs(dy_i):  # More horizontal
                        # Check if on same horizontal line
                        if abs(edge_i.start.Y - edge_j.start.Y) > tolerance:
                            continue
                        
                        min_i = min(edge_i.start.X, edge_i.end.X)
                        max_i = max(edge_i.start.X, edge_i.end.X)
                        min_j = min(edge_j.start.X, edge_j.end.X)
                        max_j = max(edge_j.start.X, edge_j.end.X)
                        
                        overlap_min = max(min_i, min_j)
                        overlap_max = min(max_i, max_j)
                        
                        if overlap_max > overlap_min + 0.01:
                            y_coord = edge_i.start.Y
                            # Store the shared wall for this room pair
                            shared_walls[(i, j)] = {
                                'start': (overlap_min, y_coord),
                                'end': (overlap_max, y_coord)
                            }
                            break  # Found shared wall, move to next room pair
                    else:  # More vertical
                        # Check if on same vertical line
                        if abs(edge_i.start.X - edge_j.start.X) > tolerance:
                            continue
                        
                        min_i = min(edge_i.start.Y, edge_i.end.Y)
                        max_i = max(edge_i.start.Y, edge_i.end.Y)
                        min_j = min(edge_j.start.Y, edge_j.end.Y)
                        max_j = max(edge_j.start.Y, edge_j.end.Y)
                        
                        overlap_min = max(min_i, min_j)
                        overlap_max = min(max_i, max_j)
                        
                        if overlap_max > overlap_min + 0.01:
                            x_coord = edge_i.start.X
                            shared_walls[(i, j)] = {
                                'start': (x_coord, overlap_min),
                                'end': (x_coord, overlap_max)
                            }
                            break  # Found shared wall, move to next room pair
                
                if (i, j) in shared_walls:
                    break  # Found shared wall, move to next room pair
    
    # Step 6: Create door polygons for connections that have shared walls
    door_polygons = []
    
    for idx1, idx2 in connections:
        room_pair = tuple(sorted([idx1, idx2]))
        
        # Check if these rooms have a shared wall
        if room_pair not in shared_walls:
            continue  # No shared wall = no door (e.g., corner connection only)
        
        wall = shared_walls[room_pair]
        x1, y1 = wall['start']
        x2, y2 = wall['end']
        
        # Calculate wall properties
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        dx = x2 - x1
        dy = y2 - y1
        wall_length = (dx**2 + dy**2)**0.5
        
        if wall_length < 0.01:
            continue
        
        # Limit door width to wall length (use 80% to maximize door size)
        actual_door_width = min(door_width, wall_length * 0.8)
        
        # Normalize direction vector
        dx /= wall_length
        dy /= wall_length
        
        # Calculate door endpoints along the wall
        half_width = actual_door_width / 2
        door_x1 = center_x - dx * half_width
        door_y1 = center_y - dy * half_width
        door_x2 = center_x + dx * half_width
        door_y2 = center_y + dy * half_width
        
        # Create perpendicular thickness for door
        perp_dx = -dy * door_thickness / 2
        perp_dy = dx * door_thickness / 2
        
        # Calculate 4 corners of door rectangle
        c1_x = door_x1 + perp_dx
        c1_y = door_y1 + perp_dy
        c2_x = door_x1 - perp_dx
        c2_y = door_y1 - perp_dy
        c3_x = door_x2 - perp_dx
        c3_y = door_y2 - perp_dy
        c4_x = door_x2 + perp_dx
        c4_y = door_y2 + perp_dy
        
        # Create door polygon (4 corners)
        door_polygon = [(c1_x, c1_y), (c2_x, c2_y), (c3_x, c3_y), (c4_x, c4_y)]
        door_polygons.append(door_polygon)
    
    return door_polygons


def extract_complete_floorplan_data(data: dict, target_size: int = 64, door_size_pixels: int = 3) -> Dict:
    """
    Extract complete floorplan data from hypergraph with all components.
    
    This function extracts rooms, doors, adjacency triples, and normalizes all polygons.
    Node indices 0 to (num_rooms-1) are regular rooms.
    Node indices num_rooms to (num_rooms+num_doors-1) are doors (type 17).
    
    RPLAN-Compatible Door Treatment:
    - Doors are treated as NODES in the graph structure
    - Triples include ALL node pairs (rooms + doors)
    - Adjacency is based on geometric edges OR door connections
    - This matches RPLAN's build_graph() function exactly
    
    Args:
        data: Dictionary containing hypergraph data with 'bounds' and 'split' fields
        target_size: Target size for normalized coordinates (default: 64)
        door_size_pixels: Door size in normalized pixel coordinates (default: 3)
        
    Returns:
        Dictionary containing:
            - 'polygons': List of ALL polygons (rooms + doors), normalized to [0, target_size]
            - 'room_types': List of room type IDs (doors are type 17)
            - 'triples': List of adjacency triples [node_idx_1, relationship, node_idx_2]
                         where relationship = 1 (adjacent) or -1 (not adjacent)
                         Note: Triples include ALL nodes (rooms AND doors as graph nodes)
    """
    # Step 1: Extract room polygons and types
    room_polygons, room_types = extract_polygons_and_room_types(data)
    
    # Step 2: Normalize ONLY room polygons first
    normalized_room_polygons = normalize_polygons(room_polygons, target_size=target_size)
    
    # Step 3: Extract door polygons in original coordinates
    door_polygons_original = extract_doors(data)
    
    # Step 4: Get normalization parameters to transform door coordinates
    # Calculate the same transformation used for rooms
    center_x, center_y = calculate_polygon_center(room_polygons)
    
    # Zero-center all room vertices
    centered_rooms = []
    for polygon in room_polygons:
        centered_poly = [(x - center_x, y - center_y) for x, y in polygon]
        centered_rooms.append(centered_poly)
    
    # Find bounds after centering
    all_x = []
    all_y = []
    for polygon in centered_rooms:
        for x, y in polygon:
            all_x.append(x)
            all_y.append(y)
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    width = max_x - min_x
    height = max_y - min_y
    scale = target_size / max(width, height)
    image_center = target_size / 2.0
    
    # Step 5: Transform door polygons using same transformation and create thicker doors in pixel space
    normalized_door_polygons = []
    for door_poly in door_polygons_original:
        # Transform door vertices to normalized coordinates (same as rooms)
        centered_door = [(x - center_x, y - center_y) for x, y in door_poly]
        normalized_door = [(round(x * scale + image_center), 
                           round(y * scale + image_center)) 
                          for x, y in centered_door]
        
        # Calculate door center and direction in normalized space
        door_x = [p[0] for p in normalized_door]
        door_y = [p[1] for p in normalized_door]
        center_door_x = sum(door_x) / len(door_x)
        center_door_y = sum(door_y) / len(door_y)
        
        # Determine door orientation (horizontal or vertical)
        dx = max(door_x) - min(door_x)
        dy = max(door_y) - min(door_y)
        
        # Create thicker door in pixel space with guaranteed thickness
        # Use integer offsets to ensure doors are always visible after rounding
        half_length = max(1, door_size_pixels // 2)  # At least 1 pixel
        thickness = max(2, door_size_pixels // 2)    # At least 2 pixels thick
        
        if dx > dy:  # Horizontal door
            # Door extends along x-axis, centered on the wall
            x_min = int(center_door_x - half_length)
            x_max = int(center_door_x + half_length)
            y_min = int(center_door_y - thickness // 2)
            y_max = int(center_door_y + thickness // 2)
            thick_door = [
                (x_min, y_min),
                (x_min, y_max),
                (x_max, y_max),
                (x_max, y_min)
            ]
        else:  # Vertical door
            # Door extends along y-axis, centered on the wall
            x_min = int(center_door_x - thickness // 2)
            x_max = int(center_door_x + thickness // 2)
            y_min = int(center_door_y - half_length)
            y_max = int(center_door_y + half_length)
            thick_door = [
                (x_min, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (x_min, y_max)
            ]
        
        normalized_door_polygons.append(thick_door)
    
    # Step 6: Combine rooms and doors
    all_polygons = normalized_room_polygons + normalized_door_polygons
    all_room_types = room_types + [17] * len(normalized_door_polygons)  # Doors are type 17
    
    # Step 7: Extract adjacency triples INCLUDING doors as nodes (RPLAN-compatible)
    # This matches RPLAN's build_graph() function where doors are nodes in the graph
    num_doors = len(normalized_door_polygons)
    triples = extract_adjacency_triples_with_doors(data, num_doors)
    
    return {
        'polygons': all_polygons,
        'room_types': all_room_types,
        'triples': triples
    }


def convert_dataset(input_dir: Path, output_dir: Path, split: str, target_size: int = 64, verbose: bool = False):
    """
    Convert a single dataset split (train/validation/test) to RPLAN format compatible with HouseGAN++ and House-Diffusion.
    
    Args:
        input_dir: Path to /dataset/hypergraph directory
        output_dir: Path to output directory
        split: Split name ('train', 'validation', or 'test')
        target_size: Target size for normalized coordinates (default: 64)
        verbose: Print detailed progress
        
    Returns:
        tuple: (success_count, failed_count, failed_ids)
    """
    import pickle
    
    print(f"\n{'='*60}")
    print(f"Processing {split.upper()} split")
    print(f"{'='*60}")
    
    # Load input JSON
    input_file = input_dir / f"{split}.json"
    if not input_file.exists():
        print(f"❌ Error: {input_file} not found")
        return 0, 0, []
    
    with open(input_file, 'r') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} samples from {input_file.name}")
    
    # Create output directory
    split_output_dir = output_dir / split
    split_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each sample
    success_count = 0
    failed_count = 0
    failed_ids = []
    converted_data = []
    
    for i, data in enumerate(dataset):
        sample_id = data.get('id', f'sample_{i}')
        
        if verbose or (i + 1) % 100 == 0:
            print(f"Processing {i+1}/{len(dataset)}... ({sample_id})", end='\r')
        
        try:
            # Convert to RPLAN format compatible with HouseGAN++ and House-Diffusion
            floorplan_data = extract_complete_floorplan_data(data, target_size=target_size)
            
            # Store with ID
            converted_sample = {
                'id': sample_id,
                'polygons': floorplan_data['polygons'],
                'room_types': floorplan_data['room_types'],
                'triples': floorplan_data['triples']
            }
            converted_data.append(converted_sample)
            success_count += 1
            
            if verbose:
                num_rooms = sum(1 for rt in floorplan_data['room_types'] if rt != 17)
                num_doors = sum(1 for rt in floorplan_data['room_types'] if rt == 17)
                num_triples = len(floorplan_data['triples'])
                print(f"   ✅ {sample_id}: {num_rooms} rooms, {num_doors} doors, {num_triples} triples")
        
        except Exception as e:
            failed_count += 1
            failed_ids.append(sample_id)
            if verbose:
                print(f"   ❌ {sample_id}: FAILED - {str(e)}")
    
    # Save converted data as pickle
    output_file = split_output_dir / f"{split}_data.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(converted_data, f)
    
    print(f"\n✅ Successful: {success_count}/{len(dataset)}")
    if failed_count > 0:
        print(f"❌ Failed: {failed_count}")
        print(f"   Failed IDs: {failed_ids[:10]}{'...' if len(failed_ids) > 10 else ''}")
    print(f"📁 Saved to: {output_file}")
    
    return success_count, failed_count, failed_ids


def main():
    """Main entry point for dataset conversion script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert Hypergrph1K Dataset to RPLAN format compatible with HouseGAN++ and House-Diffusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all splits
  python convert_hypergraph_to_rplan.py
  
  # Convert specific splits
  python convert_hypergraph_to_rplan.py --splits train validation
  
  # Verbose output
  python convert_hypergraph_to_rplan.py --verbose
  
  # Custom paths and target size
  python convert_hypergraph_to_rplan.py --input /path/to/dataset/hypergraph --output /path/to/processed_rplan_pickles --target-size 128
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=Path,
        default=PROJECT_ROOT / 'dataset/hypergraph',
        help='Path to /dataset/hypergraph directory (default: PROJECT_ROOT/dataset/hypergraph)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=PROJECT_ROOT.parent / 'processed_rplan_pickles',
        help='Path to output directory (default: PROJECT_ROOT/../processed_rplan_pickles)'
    )
    
    parser.add_argument(
        '--splits', '-s',
        nargs='+',
        choices=['train', 'validation', 'test'],
        default=['train', 'validation', 'test'],
        help='Which splits to process (default: all)'
    )
    
    parser.add_argument(
        '--target-size', '-t',
        type=int,
        default=64,
        help='Target size for normalized coordinates (default: 64)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed progress for each sample'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.input.exists():
        print(f"❌ Error: Input directory not found: {args.input}")
        return 1
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Hypergrph1K Dataset → HOUSEGANPP FORMAT CONVERTER")
    print("="*60)
    print(f"Input:       {args.input}")
    print(f"Output:      {args.output}")
    print(f"Splits:      {', '.join(args.splits)}")
    print(f"Target size: {args.target_size}x{args.target_size}")
    
    # Process each split
    total_success = 0
    total_failed = 0
    all_failed_ids = {}
    
    for split in args.splits:
        success, failed, failed_ids = convert_dataset(
            args.input,
            args.output,
            split,
            target_size=args.target_size,
            verbose=args.verbose
        )
        
        total_success += success
        total_failed += failed
        if failed_ids:
            all_failed_ids[split] = failed_ids
    
    # Final summary
    print(f"\n{'='*60}")
    print("CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Total successful: {total_success}")
    print(f"❌ Total failed: {total_failed}")
    print(f"📁 Output directory: {args.output}")
    
    # Save failed IDs to file
    if all_failed_ids:
        failed_log = args.output / 'failed_conversions.json'
        with open(failed_log, 'w') as f:
            json.dump(all_failed_ids, f, indent=2)
        print(f"\n📝 Failed sample IDs saved to: {failed_log}")
    
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    import sys
    main()

