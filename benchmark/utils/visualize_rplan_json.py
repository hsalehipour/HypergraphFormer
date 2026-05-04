#!/usr/bin/env python3
"""
Visualize graph and floor plan from HouseGAN++ JSON files.

This script reads RPLAN JSON format and generates:
1. Graph visualization showing room adjacency
2. Floor plan visualization showing room polygons
"""

import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import os

from houseganpp.scripts.viz import draw_graph, ID_COLOR


def make_sequence(edges):
    """
    Reconstruct polygon(s) from edge segments.
    
    Edges are in format: [x1, y1, x2, y2] representing line segments.
    This function connects edges to form closed polygons.
    
    Args:
        edges: numpy array of shape (n, 4) where each row is [x1, y1, x2, y2]
        
    Returns:
        List of polygons, where each polygon is a list of (x, y) tuples
    """
    polys = []
    v_curr = tuple(edges[0][:2])
    e_ind_curr = 0
    e_visited = [0]
    seq_tracker = [v_curr]
    find_next = False
    
    while len(e_visited) < len(edges):
        if find_next == False:
            if v_curr == tuple(edges[e_ind_curr][2:]):
                v_curr = tuple(edges[e_ind_curr][:2])
            else:
                v_curr = tuple(edges[e_ind_curr][2:])
            find_next = not find_next 
        else:
            # look for next edge
            for k, e in enumerate(edges):
                if k not in e_visited:
                    if (v_curr == tuple(e[:2])):
                        v_curr = tuple(e[2:])
                        e_ind_curr = k
                        e_visited.append(k)
                        break
                    elif (v_curr == tuple(e[2:])):
                        v_curr = tuple(e[:2])
                        e_ind_curr = k
                        e_visited.append(k)
                        break

        # extract next sequence
        if v_curr == seq_tracker[-1]:
            polys.append(seq_tracker)
            for k, e in enumerate(edges):
                if k not in e_visited:
                    v_curr = tuple(edges[0][:2])
                    seq_tracker = [v_curr]
                    find_next = False
                    e_ind_curr = k
                    e_visited.append(k)
                    break
        else:
            seq_tracker.append(v_curr)
    
    polys.append(seq_tracker)
    return polys


def json_to_graph_format(json_data):
    """
    Convert JSON data to the format expected by draw_graph from scripts/viz.py.
    
    draw_graph expects:
    - g_true[0]: array of room types (0-indexed, where 0 maps to RPLAN type 1)
    - g_true[1]: array of triples [k, m, l] where m is relationship (1=adjacent, -1=not adjacent)
    
    Args:
        json_data: Dictionary containing 'room_type', 'boxes', 'edges', 'ed_rm'
        
    Returns:
        [nodes, edges] in the format expected by draw_graph
    """
    room_types = json_data['room_type']
    eds_to_rms = json_data['ed_rm']
    
    # Convert RPLAN IDs (1,2,3,4,...) to 0-indexed (0,1,2,3,...)
    # draw_graph adds +1 back to get the actual RPLAN ID
    nodes = np.array([rt - 1 for rt in room_types])
    
    # Build triples: check all pairs of rooms for adjacency
    triples = []
    for k in range(len(room_types)):
        for l in range(k + 1, len(room_types)):
            # Check if k and l are adjacent (share an edge)
            is_adjacent = any((k in e_map and l in e_map) for e_map in eds_to_rms)
            relationship = 1 if is_adjacent else -1
            triples.append([k, relationship, l])
    
    edges = np.array(triples)
    
    return [nodes, edges]


def draw_floorplan_from_json(json_data, output_size=512):
    """
    Draw a floor plan visualization from JSON data.
    
    Edges are stored SEQUENTIALLY by room in the JSON (matching raster_to_json.py):
    - Edges for room 0 come first, then room 1, etc.
    - Each room's polygon is formed by the START POINTS (x1, y1) of its consecutive edges
    
    Args:
        json_data: Dictionary containing 'room_type', 'boxes', 'edges', 'ed_rm'
        output_size: Output image size in pixels (default: 512 for high quality)
        
    Returns:
        PIL Image object
    """
    # Extract data
    room_types = json_data['room_type']
    edges = json_data['edges']
    ed_rm = json_data['ed_rm']
    
    # Create high-resolution image
    img = Image.new('RGB', (output_size, output_size), 'white')
    draw_obj = ImageDraw.Draw(img)
    
    # Reconstruct poly array: find groups of consecutive edges belonging to same primary room
    # The primary room is the FIRST room in ed_rm (or only room if ed_rm has one element)
    room_edge_groups = []
    current_room = None
    current_group = []
    
    for i, (edge, rooms) in enumerate(zip(edges, ed_rm)):
        # Primary room is the first in ed_rm list
        primary_room = rooms[0] if isinstance(rooms, list) and len(rooms) > 0 else rooms
        
        if current_room is None:
            current_room = primary_room
            current_group = [i]
        elif primary_room == current_room:
            current_group.append(i)
        else:
            # New room started
            room_edge_groups.append((current_room, current_group))
            current_room = primary_room
            current_group = [i]
    
    # Don't forget the last group
    if current_group:
        room_edge_groups.append((current_room, current_group))
    
    # Draw each room
    for room_idx, edge_indices in room_edge_groups:
        if room_idx >= len(room_types):
            continue
            
        room_type = room_types[room_idx]
        
        # Build polygon from start points of consecutive edges
        polygon = []
        for edge_idx in edge_indices:
            edge = edges[edge_idx]
            # Start point of edge (x1, y1)
            polygon.append((edge[0], edge[1]))
        
        if len(polygon) >= 3:
            # Scale from [0, 256] to output size
            scaled_poly = [(x * output_size / 256.0, y * output_size / 256.0) for x, y in polygon]
            
            # Get color for this room type
            color = ID_COLOR.get(room_type, '#FFFFFF')
            
            # Draw filled polygon
            draw_obj.polygon(scaled_poly, fill=color, outline='black', width=3)
    
    return img


def main():
    parser = argparse.ArgumentParser(
        description='Visualize HouseGAN++ JSON graph and floor plan',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize a single JSON file
  python visualize_json_graph.py --json benchmark/houseganpp/rplan_json/1000.json --output_path ./output

  # Output will be:
  #   ./output/graph_1000.jpg
  #   ./output/floor_1000.png
        """
    )
    
    parser.add_argument('--json', type=str, required=True,
                        help='Path to JSON file')
    parser.add_argument('--output_path', type=str, default='.',
                        help='Directory to save output images (default: current directory)')
    
    args = parser.parse_args()
    
    # Create dump directory (needed by draw_graph)
    os.makedirs('./dump', exist_ok=True)
    
    # Load JSON
    print(f"Loading JSON from: {args.json}")
    with open(args.json, 'r') as f:
        json_data = json.load(f)
    
    print(f"Loaded {len(json_data['room_type'])} rooms")
    
    # Create output directory
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get JSON filename without extension
    json_name = Path(args.json).stem
    
    # Generate graph visualization using scripts/viz.py draw_graph
    print("Generating graph visualization...")
    graph_data = json_to_graph_format(json_data)
    G_true, graph_img = draw_graph(graph_data)
    graph_output = output_dir / f"graph_{json_name}.png"
    graph_img.save(graph_output)
    print(f"✓ Graph saved to: {graph_output}")
    
    # Generate floor plan visualization
    print("Generating floor plan visualization...")
    floor_img = draw_floorplan_from_json(json_data)
    floor_output = output_dir / f"floor_{json_name}.png"
    floor_img.save(floor_output)
    print(f"✓ Floor plan saved to: {floor_output}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
