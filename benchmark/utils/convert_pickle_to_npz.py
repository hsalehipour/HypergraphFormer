#!/usr/bin/env python3
"""
Convert hypergraph-based pickle files to house-diffusion's NPZ format.

This script converts the Hypergrph1K Dataset pickle files (used by HouseGAN++) into
the preprocessed NPZ format expected by house-diffusion's RPlanhgDataset.
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import cv2 as cv
from PIL import Image, ImageDraw

def load_pickle_data(pickle_path):
    """Load data from pickle file."""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data


def convert_sample_to_housediffusion_format(sample, max_num_points=100):
    """
    Convert a single hypergraph-based pickle sample to house-diffusion format.
    
    Applies room type remapping to match House-Diffusion's encoding:
        - Type 15 (front_door) -> 11
        - Type 17 (interior_door) -> 12
        - Type 16 (window) -> 13
    
    Applies layout centering to match House-Diffusion's preprocessing:
        - Computes global bounding box across all room contours
        - Centers the layout at (128, 128) in 256-space, which maps to (0, 0) in [-1, 1]
        - This matches House-Diffusion's centering logic (rplanhg_datasets.py lines 152-160)
    
    Returns:
        dict with keys: 'house', 'graph', 'door_mask', 'self_mask', 'gen_mask'
        OR tuple (None, reason) if sample should be skipped
    """
    polygons = sample['polygons']
    room_types = sample['room_types']
    triples = sample['triples']
    
    # Skip samples with too many rooms (max 31 due to one-hot encoding size)
    if len(polygons) > 31:
        return None, f"too many rooms ({len(polygons)} > 31)"
    
    # Build conditioning graph in House-Diffusion's expected format:
    # a complete set of pairwise triples (k < l), with relationship in {1, -1}.
    #
    # In House-Diffusion (see `rplanhg_datasets.py::build_graph`), the dataset stores
    # BOTH positive (adjacent) and negative (not-adjacent) edges.
    #
    # Our hypergraph-based pickle `triples` may already contain both; if not, we complete it by
    # assuming missing pairs are not-adjacent (-1).
    num_nodes = len(polygons)
    # House-Diffusion later pads graphs to exactly 200 triples in `__getitem__`:
    #   graph = np.concatenate((self.graphs[idx], np.zeros([200-len(self.graphs[idx]), 3])), 0)
    # so we must ensure len(graph) <= 200, otherwise it crashes with "negative dimensions".
    # A complete undirected graph has n*(n-1)/2 edges.
    max_graph_triples = 200
    if (num_nodes * (num_nodes - 1)) // 2 > max_graph_triples:
        return None, f"too many graph triples for padding (nodes={num_nodes} -> {num_nodes*(num_nodes-1)//2} > {max_graph_triples})"

    rel_by_pair = {(min(i, j), max(i, j)): -1 for i in range(num_nodes) for j in range(i + 1, num_nodes)}
    for triple in triples:
        if len(triple) != 3:
            continue
        i, rel, j = int(triple[0]), int(triple[1]), int(triple[2])
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in rel_by_pair:
            # Only accept the expected relationship values; default remains -1.
            if rel in (-1, 1):
                rel_by_pair[(a, b)] = rel
    graph = [[i, rel, j] for (i, j), rel in rel_by_pair.items()]
    if len(graph) > max_graph_triples:
        # Defensive guard (should be redundant with the n*(n-1)/2 check above).
        return None, f"too many graph triples ({len(graph)} > {max_graph_triples})"
    
    # PHASE 1: Extract contours for all rooms (before centering/normalization)
    # This two-pass approach allows us to apply global centering exactly like House-Diffusion
    extracted_contours = []
    for room_idx, (polygon, room_type) in enumerate(zip(polygons, room_types)):
        # House-Diffusion geometry pipeline (important for matching training distribution):
        # - Rasterize each room polygon into a 64×64 mask
        # - Resize mask to 256×256 (INTER_AREA)
        # - Extract contour vertices via OpenCV findContours(CHAIN_APPROX_SIMPLE)
        # - Apply global centering (shift bounding box center to 128)
        # - Normalize contour coords: (coords/256 - 0.5) * 2 -> [-1, 1]
        #
        # This addresses:
        #   (4) contour-extracted corners vs original polygon vertices
        #   (5) 64-grid vs 256-grid quantization mismatch
        #   (6) layout centering (matches House-Diffusion's bbox centering)
        if polygon is None or len(polygon) < 3:
            return None, f"invalid polygon (room_idx={room_idx}, corners={0 if polygon is None else len(polygon)})"
        poly_xy = [(int(round(x)), int(round(y))) for x, y in polygon]
        # Clamp into the 64×64 raster range. Some polygons can include 64 exactly.
        poly_xy = [(min(max(x, 0), 63), min(max(y, 0), 63)) for x, y in poly_xy]

        # Rasterize polygon in 64×64, then upsample to 256×256 before contouring,
        # matching `house_diffusion/house_diffusion/rplanhg_datasets.py`.
        rm_im = Image.new('L', (64, 64), 0)
        dr = ImageDraw.Draw(rm_im)
        dr.polygon(poly_xy, fill=1)
        room_mask64 = np.array(rm_im, dtype=np.uint8)
        room_mask256 = cv.resize(room_mask64, (256, 256), interpolation=cv.INTER_AREA)
        contours, _ = cv.findContours(room_mask256, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if contours is None or len(contours) == 0:
            return None, f"empty contour (room_idx={room_idx})"
        # House-Diffusion uses contours[0]; we defensively choose the largest contour.
        contour = max(contours, key=cv.contourArea)
        contour_xy = contour[:, 0, :].astype(np.float32)  # [N,2] in 0..255
        if contour_xy.shape[0] < 3:
            return None, f"degenerate contour (room_idx={room_idx}, corners={contour_xy.shape[0]})"
        
        extracted_contours.append((room_idx, contour_xy, room_type))
    
    # PHASE 2: Apply global centering exactly like House-Diffusion (lines 152-160)
    # Compute bounding box across ALL contours and center it at (128, 128) in 256-space
    all_coords = np.concatenate([c[1] for c in extracted_contours], axis=0)  # [total_corners, 2]
    tl = np.min(all_coords, axis=0)  # [2] - top-left
    br = np.max(all_coords, axis=0)  # [2] - bottom-right
    # In House-Diffusion's code (line 154): shift = (tl+br)/2.0 - 0.5
    # but that's in [0, 1] space. In 256-space, center should be at 128.
    shift = (tl + br) / 2.0 - 128.0
    
    # Apply centering shift to all contours
    centered_contours = []
    for room_idx, contour_xy, room_type in extracted_contours:
        contour_xy_centered = contour_xy - shift
        centered_contours.append((room_idx, contour_xy_centered, room_type))
    
    # PHASE 3: Build corner-based representation with centered+normalized coordinates
    house = []
    corner_bounds = []
    num_points = 0
    
    for room_idx, contour_xy_centered, room_type in centered_contours:
        # Normalize to [-1, 1] exactly like House-Diffusion preprocessing.
        room_coords = (contour_xy_centered / 256.0) - 0.5
        room_coords = room_coords * 2.0
        num_room_corners = int(room_coords.shape[0])
        if num_room_corners >= 32:
            # House-Diffusion uses a 32-dim one-hot for corner indices; its preprocessing
            # implicitly avoids rooms with >=32 corners after contour extraction.
            return None, f"too many corners in a room after contouring ({num_room_corners} >= 32)"
        
        # Remap room types to match House-Diffusion's encoding
        # Types > 10: 15 (front_door) -> 11, 17 (interior_door) -> 12, 16 (window) -> 13
        if room_type > 10:
            room_type = {15: 11, 17: 12, 16: 13}.get(room_type, room_type)
        
        # One-hot encode room type (25 types for RPLAN)
        rtype = np.repeat(np.array([get_one_hot(room_type, 25)]), num_room_corners, 0)
        
        # Room index (which room this corner belongs to)
        room_index = np.repeat(np.array([get_one_hot(room_idx + 1, 32)]), num_room_corners, 0)
        
        # Corner index (which corner within the room)
        corner_index = np.array([get_one_hot(i, 32) for i in range(num_room_corners)])
        
        # Padding mask (1 for real corners, 0 for padding)
        padding_mask = np.repeat(1, num_room_corners)
        padding_mask = np.expand_dims(padding_mask, 1)
        
        # Connections (each corner connects to next corner in cyclic order)
        connections = np.array([[i, (i+1) % num_room_corners] for i in range(num_room_corners)])
        connections += num_points
        
        corner_bounds.append([num_points, num_points + num_room_corners])
        num_points += num_room_corners
        
        # Concatenate all features
        room_data = np.concatenate((room_coords, rtype, corner_index, room_index, padding_mask, connections), 1)
        house.append(room_data)
    
    house_layouts = np.concatenate(house, 0)
    
    # Pad to max_num_points
    # Must be strictly less than max_num_points to ensure gen_mask has at least one 1
    # The pretrained house-diffusion model was trained with max_num_points=100, so:
    #   - All attention masks are 100×100
    #   - Positional encodings are for sequences of length 100
    #   - The transformer expects input shape [batch, channels, 100]
    if len(house_layouts) >= max_num_points:
        # Skip samples that are too large or exactly at max
        return None, f"too many corners ({len(house_layouts)} >= {max_num_points})"
    
    padding = np.zeros((max_num_points - len(house_layouts), 94))
    
    # Create gen_mask BEFORE padding (needs actual corner count)
    gen_mask = np.ones((max_num_points, max_num_points))
    gen_mask[:len(house_layouts), :len(house_layouts)] = 0
    
    # Now add padding to house_layouts
    house_layouts = np.concatenate((house_layouts, padding), 0)
    
    door_mask = np.ones((max_num_points, max_num_points))
    self_mask = np.ones((max_num_points, max_num_points))
    
    # Build masks based on corner bounds and graph
    adjacent_pairs = set()
    for g in graph:
        # Only positive edges open the door_mask.
        if len(g) == 3 and int(g[1]) == 1:
            a, b = int(g[0]), int(g[2])
            adjacent_pairs.add((a, b))
            adjacent_pairs.add((b, a))
    for i in range(len(corner_bounds)):
        for j in range(len(corner_bounds)):
            if i == j:
                # Same room - set self_mask
                self_mask[corner_bounds[i][0]:corner_bounds[i][1], 
                         corner_bounds[j][0]:corner_bounds[j][1]] = 0
            elif (i, j) in adjacent_pairs:
                # Connected via door - set door_mask
                door_mask[corner_bounds[i][0]:corner_bounds[i][1],
                         corner_bounds[j][0]:corner_bounds[j][1]] = 0
    
    # Convert graph to array format
    graph_array = np.array(graph) if len(graph) > 0 else np.zeros((0, 3))
    
    return {
        'house': house_layouts,
        'graph': graph_array,
        'door_mask': door_mask,
        'self_mask': self_mask,
        'gen_mask': gen_mask
    }


def get_one_hot(x, z):
    """Create one-hot encoding."""
    return np.eye(z)[x]


def convert_pickle_to_npz(pickle_path, output_dir, set_name, target_set):
    """
    Convert pickle file to NPZ format for house-diffusion.
    
    Args:
        pickle_path: Path to input pickle file
        output_dir: Output directory for NPZ files
        set_name: 'train', 'val', or 'eval'
        target_set: Target room count (e.g., 8) or None for no filtering
    """
    # Use "all" as identifier when target_set is None
    target_set_str = "all" if target_set is None else str(target_set)
    
    print(f"Loading pickle data from: {pickle_path}")
    data = load_pickle_data(pickle_path)
    
    if not isinstance(data, list):
        print(f"Error: Expected list of samples, got {type(data)}")
        return False
    
    print(f"Found {len(data)} samples")
    
    # Convert samples
    houses = []
    graphs = []
    door_masks = []
    self_masks = []
    gen_masks = []
    sample_ids = []  # Track sample IDs for successful conversions
    
    # Track corner number distribution (for training set)
    from collections import defaultdict
    cnumber_dist = defaultdict(list)
    
    # Track skipped samples
    skipped_samples = []
    
    for i, sample in enumerate(tqdm(data, desc=f"Converting {set_name}")):
        converted = convert_sample_to_housediffusion_format(sample)
        if isinstance(converted, tuple) and converted[0] is None:
            # Sample was skipped
            reason = converted[1]
            skipped_samples.append({
                'index': i,
                'id': sample.get('id', f'sample_{i}'),
                'reason': reason,
                'num_rooms': len([rt for rt in sample['room_types'] if rt != 17]),
                'num_doors': len([rt for rt in sample['room_types'] if rt == 17]),
                'total_corners': sum(len(poly) for poly in sample['polygons'])
            })
            continue
        
        # Store sample ID for successful conversion
        sample_id = sample.get('id', f'sample_{i}')
        sample_ids.append(sample_id)
        
        houses.append(converted['house'])
        graphs.append(converted['graph'])
        door_masks.append(converted['door_mask'])
        self_masks.append(converted['self_mask'])
        gen_masks.append(converted['gen_mask'])
        
        # Track corner counts AFTER conversion (using remapped room types and processed corners)
        # This matches House-Diffusion's logic at line 207
        if set_name == 'train':
            # Extract room type and corner count from the converted house layout
            # house layout structure: [coords(2), room_type(25), corner_idx(32), room_idx(32), padding(1), connections(2)]
            house_layout = converted['house']
            padding_mask = house_layout[:, 94-3]  # Column 91: padding mask (1=valid, 0=padding)
            
            # Process each corner
            current_room_idx = -1
            current_room_corners = []
            current_room_type = None
            
            for corner_idx in range(len(house_layout)):
                if padding_mask[corner_idx] == 0:
                    # Padding - save current room and stop
                    if current_room_type is not None and len(current_room_corners) > 0:
                        cnumber_dist[current_room_type].append(len(current_room_corners))
                    break
                
                # Get room index from one-hot encoding (columns 59:91)
                room_idx_onehot = house_layout[corner_idx, 59:91]
                room_idx = np.argmax(room_idx_onehot)
                
                # Get room type from one-hot encoding (columns 2:27)
                room_type_onehot = house_layout[corner_idx, 2:27]
                room_type = np.argmax(room_type_onehot)
                
                if room_idx != current_room_idx:
                    # New room - save previous room's corner count
                    if current_room_type is not None and len(current_room_corners) > 0:
                        cnumber_dist[current_room_type].append(len(current_room_corners))
                    
                    # Start tracking new room
                    current_room_idx = room_idx
                    current_room_type = room_type
                    current_room_corners = [corner_idx]
                else:
                    current_room_corners.append(corner_idx)
    
    if skipped_samples:
        print(f"\n⚠️  Skipped {len(skipped_samples)} samples:")
        for skip_info in skipped_samples[:10]:  # Show first 10
            print(f"  [{skip_info['index']}] {skip_info['id']}: {skip_info['reason']} "
                  f"(rooms={skip_info['num_rooms']}, doors={skip_info['num_doors']}, "
                  f"corners={skip_info['total_corners']})")
        if len(skipped_samples) > 10:
            print(f"  ... and {len(skipped_samples) - 10} more")
        
        # Save skipped sample info to a file
        skipped_log_path = output_dir / f'rplan_{set_name}_{target_set_str}_skipped.txt'
        with open(skipped_log_path, 'w') as f:
            f.write(f"Skipped {len(skipped_samples)} samples from {set_name} set:\n\n")
            f.write("To visualize a skipped sample, use:\n")
            f.write(f"  python benchmark/utils/visualize_hypergraph_pickle.py \\\n")
            f.write(f"    --pickle PICKLE_PATH \\\n")
            f.write(f"    --sample_idx INDEX \\\n")
            f.write(f"    --output_path ./skipped_viz\n\n")
            f.write("-" * 80 + "\n\n")
            for skip_info in skipped_samples:
                f.write(f"Index: {skip_info['index']}\n")
                f.write(f"ID: {skip_info['id']}\n")
                f.write(f"Reason: {skip_info['reason']}\n")
                f.write(f"Rooms: {skip_info['num_rooms']}, Doors: {skip_info['num_doors']}, Corners: {skip_info['total_corners']}\n")
                f.write("-" * 80 + "\n")
        print(f"\n📝 Full list saved to: {skipped_log_path}")
    
    # Save to NPZ
    output_path = output_dir / f'rplan_{set_name}_{target_set_str}.npz'
    print(f"Saving to: {output_path}")
    
    # Convert lists to numpy arrays
    # Houses should all be same shape (max_num_points, 94) since we padded them
    houses_array = np.array(houses)
    # Graphs have variable sizes, need object array
    graphs_array = np.empty(len(graphs), dtype=object)
    for i, g in enumerate(graphs):
        graphs_array[i] = g
    # Sample IDs as object array (to handle variable-length strings)
    ids_array = np.array(sample_ids, dtype=object)
    
    np.savez_compressed(
        output_path,
        houses=houses_array,
        graphs=graphs_array,
        door_masks=np.array(door_masks),
        self_masks=np.array(self_masks),
        gen_masks=np.array(gen_masks),
        ids=ids_array
    )
    
    print(f"✅ Saved {len(houses)} samples")
    
    # Save corner number distribution for training set
    if set_name == 'train':
        cndist_path = output_dir / f'rplan_train_{target_set_str}_cndist.npz'
        print(f"Saving corner number distribution to: {cndist_path}")
        np.savez_compressed(cndist_path, cnumber_dist=dict(cnumber_dist))
        print(f"✅ Saved corner number distribution")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert hypergraph-based pickle to house-diffusion NPZ format")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Input directory with pickle files (e.g., processed_rplan_pickles)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for NPZ files (e.g., processed_rplan)")
    parser.add_argument("--target-set", type=int, default=None,
                        help="Target room count for filtering (None = no filter, use all samples)")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("CONVERT hypergraph-based pickle TO HOUSE-DIFFUSION NPZ FORMAT")
    print("=" * 80)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    if args.target_set is None:
        print(f"Target set:       None (no filtering, use all samples)")
    else:
        print(f"Target set:       {args.target_set}")
    print()
    
    # Convert each split
    splits = {
        'train': 'train/train_data.pkl',
        'eval': 'test/test_data.pkl',  # house-diffusion uses 'eval' for test
    }
    
    for set_name, pickle_file in splits.items():
        pickle_path = input_dir / pickle_file
        
        if not pickle_path.exists():
            print(f"⚠️  Skipping {set_name}: {pickle_path} not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"Converting {set_name} set...")
        print(f"{'='*60}")
        
        success = convert_pickle_to_npz(pickle_path, output_dir, set_name, args.target_set)
        
        if not success:
            print(f"✗ Failed to convert {set_name}")
            sys.exit(1)
    
    # Use "all" as identifier when target_set is None
    target_set_str = "all" if args.target_set is None else str(args.target_set)
    
    print()
    print("=" * 80)
    print("✅ Conversion complete!")
    print("=" * 80)
    print(f"Output files saved in: {output_dir}")
    print(f"  - rplan_train_{target_set_str}.npz (includes 'ids' array)")
    print(f"  - rplan_eval_{target_set_str}.npz (includes 'ids' array)")
    print()
    print("NPZ arrays: houses, graphs, door_masks, self_masks, gen_masks, ids")
    print()


if __name__ == "__main__":
    main()
