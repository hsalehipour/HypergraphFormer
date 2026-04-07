"""
Utility functions for hypergraph processing and manipulation.

This module provides common utility functions used across the project for:
- Text processing and cleaning
- Data manipulation
- String transformations
- Tree visualization and analysis

Functions can be imported and used in scripts, notebooks, and other modules.
"""

import re
import json
import signal
import logging
from contextlib import contextmanager
from typing import Dict, List, Union

logger = logging.getLogger(__name__)


def remove_mergeid_suffixes(serialized_text):
    """
    Remove numeric suffixes from mergeid values in serialized text.
    
    This function is useful for cleaning up mergeid values that have
    been automatically numbered (e.g., "Kitchen_1", "Bedroom_2") by 
    removing the numeric suffix to get the base room type.
    
    Args:
        serialized_text (str): The serialized data_node text containing
                              mergeid values with numeric suffixes
    
    Returns:
        str: Modified text with suffixes removed from mergeid values
        
    Examples:
        >>> text = '{"mergeid":"Kitchen_1","name":"root"}'
        >>> remove_mergeid_suffixes(text)
        '{"mergeid":"Kitchen","name":"root"}'
        
        >>> text = '{"mergeid":"Bedroom_2","area":15.5}'
        >>> remove_mergeid_suffixes(text)
        '{"mergeid":"Bedroom","area":15.5}'
    """
    # Pattern to match "mergeid":"value_number"
    # This will find patterns like "mergeid":"Kitchen_1", "mergeid":"Bedroom_2", etc.
    pattern = r'("mergeid":"[^"]+?)_\d+(")'
    
    # Replace with just the base name (without _number suffix)
    cleaned_text = re.sub(pattern, r'\1\2', serialized_text)
    
    return cleaned_text


def extract_room_type(mergeid):
    """
    Extract the room type from a mergeid string.
    
    Args:
        mergeid (str): The mergeid string (e.g., "Bedroom_1", "Kitchen_2")
    
    Returns:
        str: The room type without the numeric suffix
        
    Examples:
        >>> extract_room_type("Bedroom_1")
        'Bedroom'
        
        >>> extract_room_type("Kitchen")
        'Kitchen'
    """
    if not mergeid:
        return None
    
    # Split on underscore and take the first part
    return mergeid.split('_')[0] if '_' in mergeid else mergeid


def add_mergeid_suffix(room_type, index):
    """
    Add a numeric suffix to a room type to create a mergeid.
    
    Args:
        room_type (str): The base room type (e.g., "Bedroom", "Kitchen")
        index (int): The numeric index to append
    
    Returns:
        str: The mergeid with suffix (e.g., "Bedroom_1", "Kitchen_2")
        
    Examples:
        >>> add_mergeid_suffix("Bedroom", 1)
        'Bedroom_1'
        
        >>> add_mergeid_suffix("Kitchen", 3)
        'Kitchen_3'
    """
    return f"{room_type}_{index}"


def visualize_tree_structure(
    hypergraph_json: Union[str, Dict],
    show_area: bool = True,
    show_angle: bool = True,
    show_mergeid: bool = True,
    show_connected: bool = False,
    show_final: bool = True,
    compact: bool = False
) -> str:
    """
    Visualize a hypergraph tree structure in a path-based format.
    
    Takes a JSON hypergraph (serialized DataNode) and returns a formatted
    string showing the hierarchical tree structure with node information.
    
    Args:
        hypergraph_json (str or dict): Either a JSON string or a parsed dict
                                       representing the DataNode tree
        show_area (bool): Whether to display area values (default: True)
        show_angle (bool): Whether to display angle values in radians (default: True)
        show_mergeid (bool): Whether to display mergeid/room type (default: True)
        show_connected (bool): Whether to display connected nodes (default: False)
        show_final (bool): Whether to mark leaf nodes (default: True)
        compact (bool): Whether to use compact format (default: False)
    
    Returns:
        str: Formatted tree structure as a multi-line string
        
    Examples:
        >>> json_str = '{"name":"root","area":100,"children":[...]}'
        >>> print(visualize_tree_structure(json_str))
        root (100.00 m², hierarchical)
        ├── rootL (60.00 m², hierarchical)
        │   ├── rootLL (30.00 m², Bedroom) ✓ LEAF
        │   └── rootLR (30.00 m², Kitchen) ✓ LEAF
        └── rootR (40.00 m², Bathroom) ✓ LEAF
        
        >>> # Compact format
        >>> print(visualize_tree_structure(json_str, compact=True))
        root [100.00m²]
        ├─ rootL [60.00m²]
        │  ├─ rootLL [30.00m²] Bedroom ✓
        │  └─ rootLR [30.00m²] Kitchen ✓
        └─ rootR [40.00m²] Bathroom ✓
    """
    # Parse JSON if string
    if isinstance(hypergraph_json, str):
        try:
            tree_data = json.loads(hypergraph_json)
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON - {str(e)}"
    else:
        tree_data = hypergraph_json
    
    # Validate structure
    if not isinstance(tree_data, dict) or 'name' not in tree_data:
        return "Error: Invalid tree structure - must have 'name' field"
    
    lines = []
    
    def _traverse(node: Dict, prefix: str = "", is_last: bool = True, level: int = 0):
        """Recursively traverse and format tree nodes."""
        # Build node info line
        name = node.get('name', 'unnamed')
        area = node.get('area', 0)
        angle = node.get('angle', 0)
        mergeid = node.get('mergeid', '')
        final = node.get('final', False)
        connected = node.get('connected', [])
        children = node.get('children', [])
        
        # Determine node type
        is_leaf = final or len(children) == 0
        node_type = mergeid if (is_leaf and mergeid and mergeid != name) else ("hierarchical" if not is_leaf else "leaf")
        
        # Build the line
        if compact:
            # Compact format
            info_parts = []
            if show_area:
                info_parts.append(f"{area:.2f}m²")
            if show_angle and angle != 0:
                info_parts.append(f"∠{angle:.4f}")
            
            info_str = f" [{', '.join(info_parts)}]" if info_parts else ""
            type_str = f" {node_type}" if show_mergeid and node_type != "hierarchical" else ""
            leaf_marker = " ✓" if (show_final and is_leaf) else ""
            
            line = f"{prefix}{name}{info_str}{type_str}{leaf_marker}"
        else:
            # Detailed format
            info_parts = []
            if show_area:
                info_parts.append(f"{area:.2f} m²")
            if show_mergeid and node_type != "hierarchical":
                info_parts.append(node_type)
            elif not is_leaf:
                info_parts.append("hierarchical")
            if show_angle and angle != 0:
                info_parts.append(f"angle={angle:.4f}")
            
            info_str = f" ({', '.join(info_parts)})" if info_parts else ""
            leaf_marker = " ✓ LEAF" if (show_final and is_leaf) else ""
            
            line = f"{prefix}{name}{info_str}{leaf_marker}"
        
        # Add connected info if requested
        if show_connected and connected:
            conn_str = ", ".join(connected)
            line += f" → [{conn_str}]"
        
        lines.append(line)
        
        # Process children
        if children:
            for i, child in enumerate(children):
                is_last_child = (i == len(children) - 1)
                
                if compact:
                    # Compact tree symbols
                    if level == 0 and i == 0:
                        child_prefix = "├─ "
                    elif is_last_child:
                        child_prefix = "└─ "
                    else:
                        child_prefix = "├─ "
                    
                    extension = "   " if is_last_child else "│  "
                else:
                    # Detailed tree symbols
                    if is_last_child:
                        child_prefix = "└── "
                        extension = "    "
                    else:
                        child_prefix = "├── "
                        extension = "│   "
                
                new_prefix = prefix + extension if level > 0 else ""
                _traverse(child, new_prefix + child_prefix, is_last_child, level + 1)
    
    # Start traversal
    _traverse(tree_data, "", True, 0)
    
    return "\n".join(lines)


def get_tree_statistics(hypergraph_json: Union[str, Dict]) -> Dict:
    """
    Extract statistics from a hypergraph tree structure.
    
    Args:
        hypergraph_json (str or dict): JSON string or dict representing the DataNode tree
    
    Returns:
        dict: Statistics including total nodes, leaf nodes, depth, rooms by type, etc.
        
    Example:
        >>> json_str = '{"name":"root","area":100,"children":[...]}'
        >>> stats = get_tree_statistics(json_str)
        >>> print(f"Total rooms: {stats['leaf_count']}")
        >>> print(f"Tree depth: {stats['max_depth']}")
    """
    # Parse JSON if string
    if isinstance(hypergraph_json, str):
        tree_data = json.loads(hypergraph_json)
    else:
        tree_data = hypergraph_json
    
    stats = {
        'total_nodes': 0,
        'leaf_count': 0,
        'internal_count': 0,
        'max_depth': 0,
        'total_area': 0,
        'rooms_by_type': {},
        'node_names': [],
        'leaf_names': []
    }
    
    def _traverse(node: Dict, depth: int = 0):
        """Recursively collect statistics."""
        stats['total_nodes'] += 1
        stats['node_names'].append(node.get('name', 'unnamed'))
        stats['max_depth'] = max(stats['max_depth'], depth)
        
        children = node.get('children', [])
        is_leaf = len(children) == 0 or node.get('final', False)
        
        if is_leaf:
            stats['leaf_count'] += 1
            stats['leaf_names'].append(node.get('name', 'unnamed'))
            
            # Track room types
            mergeid = node.get('mergeid', 'unknown')
            if mergeid:
                stats['rooms_by_type'][mergeid] = stats['rooms_by_type'].get(mergeid, 0) + 1
            
            # Add leaf area
            stats['total_area'] += node.get('area', 0)
        else:
            stats['internal_count'] += 1
        
        # Recurse to children
        for child in children:
            _traverse(child, depth + 1)
    
    _traverse(tree_data, 0)
    
    return stats


def get_processing_order(hypergraph_json: Union[str, Dict]) -> List[Dict]:
    """
    Get the exact processing order of nodes as they would be processed
    during subdivision (depth-first, left-to-right).
    
    Args:
        hypergraph_json (str or dict): JSON string or dict representing the DataNode tree
    
    Returns:
        list: List of dicts containing node info in processing order
              Each dict has: name, level, index_in_parent, is_leaf, area, angle
        
    Example:
        >>> json_str = '{"name":"root","children":[...]}'
        >>> order = get_processing_order(json_str)
        >>> for i, node in enumerate(order):
        ...     print(f"{i+1}. {node['name']} (level {node['level']})")
        1. root (level 0)
        2. rootL (level 1)
        3. rootLL (level 2)
        4. rootLR (level 2)
        5. rootR (level 1)
        6. rootRL (level 2)
        7. rootRR (level 2)
    """
    # Parse JSON if string
    if isinstance(hypergraph_json, str):
        tree_data = json.loads(hypergraph_json)
    else:
        tree_data = hypergraph_json
    
    processing_order = []
    
    def _traverse(node: Dict, level: int = 0, parent_name: str = None, index_in_parent: int = 0):
        """Recursively collect nodes in processing order."""
        name = node.get('name', 'unnamed')
        children = node.get('children', [])
        is_leaf = len(children) == 0 or node.get('final', False)
        
        node_info = {
            'name': name,
            'level': level,
            'parent': parent_name,
            'index_in_parent': index_in_parent,
            'is_leaf': is_leaf,
            'area': node.get('area', 0),
            'angle': node.get('angle', 0),
            'mergeid': node.get('mergeid', ''),
            'num_children': len(children)
        }
        
        processing_order.append(node_info)
        
        # Recurse to children in order
        for i, child in enumerate(children):
            _traverse(child, level + 1, name, i)
    
    _traverse(tree_data, 0, None, 0)
    
    return processing_order




def load_and_compare_hypergraphs(
    file1: str,
    file2: str,
    verbose: bool = False
) -> tuple[bool, Dict]:
    """
    Load two hypergraph files and compare them for equality.
    
    Convenience function that loads hypergraph JSON files and compares them.
    The files should be in the format:
    {
        "accessgraph": {...},
        "hypergraph": "..."
    }
    
    Args:
        file1 (str): Path to first hypergraph JSON file
        file2 (str): Path to second hypergraph JSON file
        verbose (bool): If True, print detailed comparison information (default: False)
    
    Returns:
        tuple: (is_equal, report) - same as compare_hypergraphs()
    
    Example:
        >>> is_equal, report = load_and_compare_hypergraphs(
        ...     "training_data/extracted_hypergraph_msd/test_1001_subgraph_0.txt",
        ...     "training_data/extracted_hypergraph_msd/test_1002_subgraph_0.txt",
        ...     verbose=True
        ... )
    """
    try:
        with open(file1, 'r') as f:
            data1 = json.load(f)
            # Extract the hypergraph field from the JSON structure
            hg1 = data1.get('hypergraph', '')
            if not hg1:
                return False, {
                    'equal': False,
                    'error': f"No 'hypergraph' field found in {file1}",
                    'differences': [f"Missing 'hypergraph' field in {file1}"]
                }
    except FileNotFoundError:
        return False, {
            'equal': False,
            'error': f"File not found: {file1}",
            'differences': [f"Could not open file: {file1}"]
        }
    except json.JSONDecodeError as e:
        return False, {
            'equal': False,
            'error': f"Invalid JSON in {file1}: {str(e)}",
            'differences': [f"JSON parsing error in {file1}: {str(e)}"]
        }
    except Exception as e:
        return False, {
            'equal': False,
            'error': f"Error reading {file1}: {str(e)}",
            'differences': [f"Error reading {file1}: {str(e)}"]
        }
    
    try:
        with open(file2, 'r') as f:
            data2 = json.load(f)
            # Extract the hypergraph field from the JSON structure
            hg2 = data2.get('hypergraph', '')
            if not hg2:
                return False, {
                    'equal': False,
                    'error': f"No 'hypergraph' field found in {file2}",
                    'differences': [f"Missing 'hypergraph' field in {file2}"]
                }
    except FileNotFoundError:
        return False, {
            'equal': False,
            'error': f"File not found: {file2}",
            'differences': [f"Could not open file: {file2}"]
        }
    except json.JSONDecodeError as e:
        return False, {
            'equal': False,
            'error': f"Invalid JSON in {file2}: {str(e)}",
            'differences': [f"JSON parsing error in {file2}: {str(e)}"]
        }
    except Exception as e:
        return False, {
            'equal': False,
            'error': f"Error reading {file2}: {str(e)}",
            'differences': [f"Error reading {file2}: {str(e)}"]
        }
    
    from .hypergraph import Hypergraph as _HG
    h1 = _HG(hg1) if isinstance(hg1, dict) else _HG(json.loads(hg1))
    h2 = _HG(hg2) if isinstance(hg2, dict) else _HG(json.loads(hg2))
    return h1.compare(h2, verbose=verbose)






def filter_hypergraphs_by_connectivity(
    data: List[Dict],
    split_field: str = 'split',
    verbose: bool = False
) -> tuple[List[Dict], List[str], List[tuple]]:
    """
    Filter a dataset to keep only hypergraphs with connectivity information.
    
    Removes hypergraphs where no leaf nodes have any connections to other
    leaf nodes. This is useful for filtering out degenerate cases where
    the spatial subdivision produced isolated rooms with no adjacency.
    
    Args:
        data (list): List of hypergraph entries (each containing a 'split' field
                    with JSON string of the DataNode tree)
        split_field (str): The field name containing the hypergraph JSON string
                          (default: 'split')
        verbose (bool): If True, print progress and summary (default: False)
        
    Returns:
        tuple: (filtered_data, removed_ids, error_entries)
            - filtered_data: List of entries that have connectivity
            - removed_ids: List of IDs that were removed (no connectivity)
            - error_entries: List of (id, error_message) tuples for parsing errors
            
    Example:
        >>> data = [
        ...     {"id": "1", "split": '{"name":"root","children":[...]}'},
        ...     {"id": "2", "split": '{"name":"root","children":[...]}'}
        ... ]
        >>> filtered, removed, errors = filter_hypergraphs_by_connectivity(data)
        >>> print(f"Kept {len(filtered)}, removed {len(removed)}")
    """
    filtered = []
    removed_ids = []
    error_entries = []
    
    for entry in data:
        try:
            split_str = entry.get(split_field, '{}')
            split_data = json.loads(split_str)
            
            from .hypergraph import Hypergraph as _HG
            if _HG(split_data).has_connectivity():
                filtered.append(entry)
            else:
                removed_ids.append(entry.get('id', 'unknown'))
        except json.JSONDecodeError as e:
            error_entries.append((entry.get('id', 'unknown'), str(e)))
        except Exception as e:
            error_entries.append((entry.get('id', 'unknown'), str(e)))
    
    if verbose:
        original_count = len(data)
        filtered_count = len(filtered)
        removed_count = len(removed_ids)
        
        print("=" * 60)
        print("CONNECTIVITY FILTERING RESULTS")
        print("=" * 60)
        print(f"Original count:  {original_count}")
        print(f"With connectivity: {filtered_count} ({100*filtered_count/original_count:.1f}%)")
        print(f"No connectivity:   {removed_count} ({100*removed_count/original_count:.1f}%)")
        
        if error_entries:
            print(f"Parse errors:      {len(error_entries)}")
        print("=" * 60)
    
    return filtered, removed_ids, error_entries




def _convert_areas_to_ratios(node: dict) -> None:
    """Convert absolute areas to parent-relative ratios in-place (recursive)."""
    parent_area = node.get("area", 0)
    for child in node.get("children", []):
        _convert_areas_to_ratios(child)
        if parent_area > 0:
            child["area"] = child["area"] / parent_area
    if node.get("name") == "root":
        node["area"] = 1.0


def has_corner_only_connections(data: Dict) -> bool:
    """
    Check if a hypergraph has connections between rooms that only share a corner.
    
    This function detects cases where rooms are connected in the hypergraph
    but don't have a shared wall segment (only touch at a corner point).
    Such connections are problematic for door placement.
    
    Args:
        data (dict): Dataset entry with 'bounds' and 'split' fields
        
    Returns:
        bool: True if there are corner-only connections, False otherwise
        
    Example:
        >>> entry = {"id": "sample1", "bounds": "{...}", "split": "{...}"}
        >>> if has_corner_only_connections(entry):
        ...     print("This sample has corner-only connections")
    """
    try:
        try:
            from hypergraph.api.lib.tools import RGL
            import clr
            clr.AddReference("System.Collections")
            from System.Collections.Generic import List as DotNetList
        except ImportError:
            return False
        
        # Parse boundary and create mesh
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
        
        # Load and subdivide hypergraph (ratio-based for numerical stability)
        hypergraph_json_str = data['split']
        tree = json.loads(hypergraph_json_str)
        _convert_areas_to_ratios(tree)
        data_node = RGL.DataNode.deserializeDataNode(json.dumps(tree))
        generated_mesh = RGL.DataNode.subdivideWithMeshRatio(data_node, boundary_mesh)
        
        # Build room name to index mapping
        all_children = RGL.DataNode.GetAllChildren(data_node)
        name_to_idx = {}
        idx = 0
        
        for i in range(all_children.Count):
            child = all_children[i]
            if child.final:
                name_to_idx[child.name] = idx
                idx += 1
        
        # Extract hypergraph connections
        hypergraph_connections = set()
        for i in range(all_children.Count):
            child = all_children[i]
            if child.final and hasattr(child, 'connected') and child.connected:
                room_name = child.name
                if room_name in name_to_idx:
                    room_idx = name_to_idx[room_name]
                    connected_names = list(child.connected)
                    for neighbor_name in connected_names:
                        if neighbor_name in name_to_idx:
                            neighbor_idx = name_to_idx[neighbor_name]
                            edge = tuple(sorted([room_idx, neighbor_idx]))
                            hypergraph_connections.add(edge)
        
        # Find which room pairs share a wall segment
        shared_walls = set()
        tolerance = 0.05
        
        num_rooms = generated_mesh.faceList.Count
        for i in range(num_rooms):
            for j in range(i + 1, num_rooms):
                face_i = generated_mesh.faceList[i]
                face_j = generated_mesh.faceList[j]
                
                edges_i = RGL.RhConvert.NFaceToLineList(face_i)
                edges_j = RGL.RhConvert.NFaceToLineList(face_j)
                
                # Check for overlapping collinear edges
                has_shared_wall = False
                for edge_i in edges_i:
                    for edge_j in edges_j:
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
                        if abs(abs(dot_product) - 1.0) > 0.01:
                            continue
                        
                        # Check for overlap
                        if abs(dx_i) > abs(dy_i):  # Horizontal
                            if abs(edge_i.start.Y - edge_j.start.Y) > tolerance:
                                continue
                            
                            min_i = min(edge_i.start.X, edge_i.end.X)
                            max_i = max(edge_i.start.X, edge_i.end.X)
                            min_j = min(edge_j.start.X, edge_j.end.X)
                            max_j = max(edge_j.start.X, edge_j.end.X)
                            
                            overlap_min = max(min_i, min_j)
                            overlap_max = min(max_i, max_j)
                            
                            if overlap_max > overlap_min + 0.01:
                                has_shared_wall = True
                                break
                        else:  # Vertical
                            if abs(edge_i.start.X - edge_j.start.X) > tolerance:
                                continue
                            
                            min_i = min(edge_i.start.Y, edge_i.end.Y)
                            max_i = max(edge_i.start.Y, edge_i.end.Y)
                            min_j = min(edge_j.start.Y, edge_j.end.Y)
                            max_j = max(edge_j.start.Y, edge_j.end.Y)
                            
                            overlap_min = max(min_i, min_j)
                            overlap_max = min(max_i, max_j)
                            
                            if overlap_max > overlap_min + 0.01:
                                has_shared_wall = True
                                break
                    
                    if has_shared_wall:
                        break
                
                if has_shared_wall:
                    shared_walls.add((i, j))
        
        # Check if any hypergraph connection lacks a shared wall
        for idx1, idx2 in hypergraph_connections:
            room_pair = tuple(sorted([idx1, idx2]))
            if room_pair not in shared_walls:
                # Found a corner-only connection
                return True
        
        return False
        
    except Exception as e:
        # If any error occurs, assume no corner-only connections
        return False


def filter_hypergraphs_without_corner_connections(
    data: List[Dict],
    verbose: bool = False
) -> tuple[List[Dict], List[str], List[tuple]]:
    """
    Filter a dataset to remove hypergraphs with corner-only connections.
    
    Removes hypergraphs where rooms are connected in the hypergraph but only
    share a corner (no wall segment). This is useful for ensuring all
    connections can have doors placed between them.
    
    Args:
        data (list): List of hypergraph entries (each with 'bounds' and 'split' fields)
        verbose (bool): If True, print progress and summary (default: False)
        
    Returns:
        tuple: (filtered_data, removed_ids, error_entries)
            - filtered_data: List of entries WITHOUT corner-only connections
            - removed_ids: List of IDs that were removed (had corner-only connections)
            - error_entries: List of (id, error_message) tuples for processing errors
            
    Example:
        >>> data = [
        ...     {"id": "sample1", "bounds": "{...}", "split": "{...}"},
        ...     {"id": "sample2", "bounds": "{...}", "split": "{...}"}
        ... ]
        >>> filtered, removed, errors = filter_hypergraphs_without_corner_connections(data, verbose=True)
        >>> print(f"Kept {len(filtered)}, removed {len(removed)}")
    """
    filtered = []
    removed_ids = []
    error_entries = []
    
    for i, entry in enumerate(data):
        if verbose and (i + 1) % 10 == 0:
            print(f"Processing {i + 1}/{len(data)}...", end='\r')
        
        try:
            if has_corner_only_connections(entry):
                removed_ids.append(entry.get('id', 'unknown'))
            else:
                filtered.append(entry)
        except Exception as e:
            error_entries.append((entry.get('id', 'unknown'), str(e)))
    
    if verbose:
        print()  # Clear progress line
        original_count = len(data)
        filtered_count = len(filtered)
        removed_count = len(removed_ids)
        
        print("=" * 60)
        print("CORNER CONNECTION FILTERING RESULTS")
        print("=" * 60)
        print(f"Original count:        {original_count}")
        print(f"Without corner-only:   {filtered_count} ({100*filtered_count/original_count:.1f}%)")
        print(f"With corner-only:      {removed_count} ({100*removed_count/original_count:.1f}%)")
        
        if error_entries:
            print(f"Processing errors:     {len(error_entries)}")
        print("=" * 60)
    
    return filtered, removed_ids, error_entries


class TimeoutException(Exception):
    """Exception raised when generation times out."""
    pass



@contextmanager
def time_limit(seconds):
    """
    Context manager to limit execution time of a code block.

    Args:
        seconds: Maximum time allowed in seconds

    Raises:
        TimeoutException: If the code block exceeds the time limit

    Note: Only works on Unix-like systems (uses signal.SIGALRM)
    """
    def signal_handler(signum, frame):
        raise TimeoutException(f"Generation timed out after {seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

