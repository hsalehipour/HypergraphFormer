#!/usr/bin/env python3
"""
Visualize graph and floor plan from Hypergrph1K Dataset pickle files.

This script reads hypergraph-based pickle format and generates:
1. Adjacency graph visualization showing all room adjacency
2. Access graph visualization showing door connections only
3. Floor plan visualization showing room polygons
"""

import pickle
import argparse
import numpy as np
import os
import shutil
from pathlib import Path
from PIL import Image, ImageDraw
import networkx as nx
import matplotlib.pyplot as plt

# Import from packaged houseganpp
from houseganpp.scripts.viz import draw_graph
from houseganpp.misc.utils import ROOM_CLASS

# Create reverse mapping from RPLAN ID to room name
CLASS_ROOM = {v: k for k, v in ROOM_CLASS.items()}

# Color scheme for room types matching visualize_hypergraph.ipynb
ID_COLOR = {
    3: '#5B448C',       # Bedroom; Dark purple
    1: '#ED6D87',       # living_room; Pinkish
    2: '#9A5387',       # kitchen; Purple
    4: '#FFE953',       # bathroom; Yellow
    6: '#ED6D87',       # foyer; Pinkish (same as living)
    10: '#FF9FBB',      # extra; Light pink
    17: '#b8b5b4',      # door; light grey
}

def pickle_to_graph_format(sample):
    """
    Convert pickle sample data to the format expected by draw_graph from scripts/viz.py.
    
    draw_graph expects:
    - g_true[0]: array of room types (0-indexed, where 0 maps to RPLAN type 1)
    - g_true[1]: array of triples [k, m, l] where m is relationship (1=adjacent, -1=not adjacent)
    
    Args:
        sample: Dictionary containing 'room_types' and 'triples'
        
    Returns:
        [nodes, edges] in the format expected by draw_graph
    """
    room_types = sample['room_types']
    triples = sample['triples']
    
    # Convert RPLAN IDs (1,2,3,4,...) to 0-indexed (0,1,2,3,...)
    # draw_graph adds +1 back to get the actual RPLAN ID
    nodes = np.array([rt - 1 for rt in room_types])
    edges = np.array(triples)
    
    return [nodes, edges]

def build_access_graph(nodes, edges):
    """
    Build access graph where edges exist only between rooms connected through doors.
    
    Args:
        nodes: Array of room types (integer indices, 0-indexed from one-hot)
        edges: Edge triples [node1_idx, relationship, node2_idx]
               where relationship: 1 = adjacent, -1 = not adjacent
    
    Returns:
        G: NetworkX graph with only room nodes and door-based connections
        colors: List of node colors for visualization
        node_labels: Dict mapping node index to room type name
    """
    # Convert nodes to 1-indexed room types (RPLAN IDs)
    room_types = nodes + 1
    
    # Create graph and tracking structures
    G = nx.Graph()
    colors = []
    node_labels = {}
    door_connections = {}  # door_idx -> [room1, room2, ...]
    
    # Add only room nodes (exclude doors type 17)
    room_indices = []
    for k, room_type in enumerate(room_types):
        if room_type not in [15, 17]:  # Exclude front door and interior door
            G.add_node(k, label=int(room_type), type=CLASS_ROOM.get(int(room_type), 'unknown'))
            colors.append(ID_COLOR[int(room_type)])
            node_labels[k] = f"{CLASS_ROOM.get(int(room_type), 'unknown')}\n({k})"
            room_indices.append(k)
        elif room_type == 17:  # Track doors for later
            door_connections[k] = []
    
    # Find connections through doors
    # First pass: identify which rooms connect to which doors
    for edge in edges:
        node1, rel, node2 = int(edge[0]), int(edge[1]), int(edge[2])
        
        # Only consider adjacent edges (rel > 0)
        if rel > 0:
            type1 = int(room_types[node1])
            type2 = int(room_types[node2])
            
            # If one node is a door (17) and the other is a room
            if type1 == 17 and type2 not in [15, 17]:
                door_connections[node1].append(node2)
            elif type2 == 17 and type1 not in [15, 17]:
                door_connections[node2].append(node1)
    
    # Second pass: connect rooms that share a door
    for door_idx, connected_rooms in door_connections.items():
        # Connect all pairs of rooms that share this door
        for i, room1 in enumerate(connected_rooms):
            for room2 in connected_rooms[i+1:]:
                if room1 in room_indices and room2 in room_indices:
                    G.add_edge(room1, room2)
    
    return G, colors, node_labels



def draw_access_graph_from_pickle(sample, nodes, edges, output_path, title="Access Graph"):
    """
    Draw and save the access graph showing only door connections between rooms.
    
    Args:
        sample: Original pickle sample data (not used but kept for API consistency)
        nodes: Room type nodes
        edges: Edge triples
        output_path: Where to save the image
        title: Title for the plot (not displayed in the figure)
    
    Returns:
        G: NetworkX graph object
        fig: Matplotlib figure object
    """
    G, colors, node_labels = build_access_graph(nodes, edges)
    
    if len(G.nodes()) == 0:
        print(f"⚠️  No room nodes to draw for {title}")
        return None, None
    
    # Create single figure for access graph
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Use graphviz neato layout for cleaner visualization with minimal edge crossings
    # This matches the layout used in adjacency graph (houseganpp/scripts/viz.py)
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='neato')
    except:
        # Fallback to spring layout if graphviz is not available
        print("  ⚠️  Graphviz not available, using spring layout (may have overlapping edges)")
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
    
    # Draw graph with same styling as adjacency graph (nodes and edges only, no labels or title)
    # Styling parameters match houseganpp/scripts/viz.py draw_graph function
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=1000,
                          ax=ax, edgecolors='blue', linewidths=0.0)
    nx.draw_networkx_edges(G, pos, edge_color='#D3A2C7', width=4.0, ax=ax)
    
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return G, fig

def draw_floorplan_from_pickle(sample, output_size=512):
    """
    Draw a floor plan visualization from pickle sample data.
    
    Args:
        sample: Dictionary containing 'polygons' and 'room_types'
        output_size: Output image size in pixels (default: 512 for high quality)
        
    Returns:
        PIL Image object
    """
    # Extract data
    room_types = sample['room_types']
    polygons = sample['polygons']
    
    # Create high-resolution image
    img = Image.new('RGB', (output_size, output_size), 'white')
    draw_obj = ImageDraw.Draw(img)
    
    # Draw each room/door polygon
    for i, (room_type, polygon) in enumerate(zip(room_types, polygons)):
        if len(polygon) < 3:
            continue
        
        # Polygons are in normalized [0, 64] coordinates, scale to output size
        scaled_poly = [(x * output_size / 64.0, y * output_size / 64.0) for x, y in polygon]
        
        # Get color for this room type
        color = ID_COLOR.get(room_type, '#FFFFFF')
        
        # Draw filled polygon with outline
        draw_obj.polygon(scaled_poly, fill=color, outline='black', width=3)
    
    return img


def main():
    parser = argparse.ArgumentParser(
        description='Visualize hypergraph-based pickle data graph and floor plan',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize first 5 samples from training set
  python visualize_hypergraph_pickle.py --pickle processed_rplan_pickles/train_data.pkl --num_samples 5 --output_path ./hypergraph_viz

  # Visualize specific sample by ID
  python visualize_hypergraph_pickle.py --pickle processed_rplan_pickles/train_data.pkl --sample_id c-n-0100 --output_path ./hypergraph_viz
        """
    )
    
    parser.add_argument('--pickle', type=str, required=True,
                        help='Path to pickle file')
    parser.add_argument('--sample_id', type=str, default=None,
                        help='Specific sample ID to visualize')
    parser.add_argument('--sample_idx', type=int, default=None,
                        help='Specific sample index to visualize (0-based)')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of samples to visualize (default: 1, starting from index 0)')
    parser.add_argument('--output_path', type=str, default='.',
                        help='Directory to save output images (default: current directory)')
    
    args = parser.parse_args()
    
    # Create dump directory (needed by draw_graph)
    os.makedirs('./dump', exist_ok=True)
    
    # Load pickle
    print(f"Loading pickle from: {args.pickle}")
    with open(args.pickle, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    # Create output directory
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which samples to visualize
    samples_to_viz = []
    
    if args.sample_id:
        # Find sample by ID
        for sample in data:
            if sample['id'] == args.sample_id:
                samples_to_viz.append(sample)
                break
        if not samples_to_viz:
            print(f"❌ Error: Sample ID '{args.sample_id}' not found")
            return 1
    elif args.sample_idx is not None:
        # Use specific index
        if 0 <= args.sample_idx < len(data):
            samples_to_viz.append(data[args.sample_idx])
        else:
            print(f"❌ Error: Sample index {args.sample_idx} out of range [0, {len(data)-1}]")
            return 1
    else:
        # Use first N samples
        samples_to_viz = data[:min(args.num_samples, len(data))]
    
    # Visualize each sample
    for sample in samples_to_viz:
        sample_id = sample['id']
        print(f"\nProcessing sample: {sample_id}")
        print(f"  Rooms: {len([rt for rt in sample['room_types'] if rt != 17])}")
        print(f"  Doors: {len([rt for rt in sample['room_types'] if rt == 17])}")
        print(f"  Total nodes: {len(sample['room_types'])}")
        print(f"  Triples: {len(sample['triples'])}")
        
        # Generate adjacency graph visualization using scripts/viz.py draw_graph
        print("  Generating adjacency graph visualization...")
        graph_data = pickle_to_graph_format(sample)
        G_true, graph_img = draw_graph(graph_data)
        graph_output = output_dir / f"graph_adjacency_{sample_id}.png"
        graph_img.save(graph_output)
        print(f"  ✓ Adjacency graph saved to: {graph_output}")
        
        # Generate access graph visualization (door connections only)
        print("  Generating access graph visualization...")
        nodes, edges = graph_data
        access_output = output_dir / f"graph_access_{sample_id}.png"
        G_access, _ = draw_access_graph_from_pickle(sample, nodes, edges, 
                                                     access_output, 
                                                     title=f"Access Graph: {sample_id}")
        if G_access is not None:
            print(f"  ✓ Access graph saved to: {access_output}")
        
        # Generate floor plan visualization
        print("  Generating floor plan visualization...")
        floor_img = draw_floorplan_from_pickle(sample)
        floor_output = output_dir / f"floorplan_{sample_id}.png"
        floor_img.save(floor_output)
        print(f"  ✓ Floor plan saved to: {floor_output}")
    
    # Clean up temporary dump directory
    if os.path.exists('./dump'):
        shutil.rmtree('./dump')
    
    print(f"\n✓ Done! Visualized {len(samples_to_viz)} sample(s)")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
