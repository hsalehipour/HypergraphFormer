#!/usr/bin/env python3
"""
Evaluate HouseGAN++ floorplan generation.

Generates floorplans and computes four metrics against ground truth:
  1. GED  -- Graph Edit Distance between predicted and GT access graphs
  2. Room Accuracy -- Room count and type distribution matching
  3. Delta -- Room shape regularity deviation (geometric similarity)
  4. Area Proportion Error -- Mean absolute area-proportion difference between paired rooms

Expects an RPLAN pickle file via --data_path.  To convert from
RPLAN JSON files, use benchmark/utils/convert_json_to_pickle.py first
(the shell wrapper eval_houseganpp.sh does this automatically).

Supports caching generated outputs for reproducible re-evaluation.
"""

import argparse
import os
import pickle
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import multiprocessing
from collections import defaultdict
from datetime import datetime
import webcolors
import cv2
from PIL import Image, ImageDraw

from utils.metrics import ged_between_graphs, room_type_accuracy
from utils.metrics import compute_room_area_proportion_error as compute_area_prop_metric
from benchmark.src.dataset_loader import BenchmarkDataset, collate_fn
from utils.report_results import report_results
from houseganpp.misc.utils import _init_input
from houseganpp.models.models import Generator

# Room type constants
ROOM_CLASS = {
    "living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4,
    "balcony": 5, "entrance": 6, "dining room": 7, "study room": 8,
    "storage": 10, "front door": 15, "unknown": 16, "interior_door": 17
}
CLASS_ROOM = {v: k for k, v in ROOM_CLASS.items()}


def _room_geometries_to_area_tuples(room_geometries):
    """Convert room geometry dicts to ``(merge_id, area)`` tuples for metrics.py."""
    return [(CLASS_ROOM.get(r['room_type'], str(r['room_type'])), r['area'])
            for r in room_geometries]


# Color scheme for room types matching visualize_hypergraph.ipynb
ID_COLOR = {
    3: '#5B448C',       # Bedroom; Dark purple
    1: '#ED6D87',       # living_room; Pinkish
    2: '#9A5387',       # kitchen; Purple
    4: '#FFE953',       # bathroom; Yellow
    6: '#ED6D87',       # foyer; Pinkish (same as living)
    10: '#FF9FBB',      # extra; Light pink
    17: '#b8b5b4',      # door; light grey
    -1: 'white',        # outside node (matches original HouseGAN++ script)
}

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

def draw_masks(masks, room_types, image_size=256):
    """
    Draw floorplan image from masks (matches HouseGAN++ draw_masks function).
    
    Args:
        masks: Array of masks [num_rooms, H, W]
        room_types: Array of room type IDs
        image_size: Image size
    
    Returns:
        Image as numpy array [H, W, 3]
    """
    bg_img = Image.new("RGBA", (image_size, image_size), (255, 255, 255, 255))
    
    for mask, room_type in zip(masks, room_types):
        # Prepare mask
        m = mask.copy()
        m[m > 0] = 255
        m[m < 0] = 0
        m_lg = cv2.resize(m, (image_size, image_size), interpolation=cv2.INTER_AREA)
        
        # Pick color using webcolors (same as HouseGAN++)
        color = ID_COLOR.get(int(room_type), '#BEBEBE')
        r, g, b = webcolors.hex_to_rgb(color)
        
        # Set drawer
        dr_bkg = ImageDraw.Draw(bg_img)
        
        # Draw region
        m_pil = Image.fromarray(m_lg.astype(np.uint8))
        dr_bkg.bitmap((0, 0), m_pil.convert('L'), fill=(r, g, b, 256))
        
        # Draw contour (matches HouseGAN++ exactly)
        m_cv = m_lg[:, :, np.newaxis].astype('uint8')
        ret, thresh = cv2.threshold(m_cv, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if len(c) > 0]
        cnt = np.zeros((image_size, image_size, 3)).astype('uint8')
        cv2.drawContours(cnt, contours, -1, (255, 255, 255, 255), 1)
        cnt = Image.fromarray(cnt)
        dr_bkg.bitmap((0, 0), cnt.convert('L'), fill=(0, 0, 0, 255))
    
    return bg_img.resize((image_size, image_size))


def count_actual_rooms_from_masks(nodes, masks):
    """
    Count the actual number of room regions considering disconnected components.
    
    A single mask from HouseGAN++ might have multiple disconnected spatial regions,
    which should be counted as separate rooms.
    
    Args:
        nodes: Array of room types (integer indices, 0-indexed from one-hot)
        masks: Generated masks for each room/door [num_masks, H, W]
    
    Returns:
        Dictionary with:
            - 'total_rooms': Total count of actual room regions (including disconnected parts)
            - 'expected_rooms': Expected count (number of room nodes in input graph)
            - 'expected_room_types': List of expected room types (RPLAN IDs)
            - 'actual_room_types': List of actual room types (accounting for disconnected components)
            - 'per_mask_components': List of component counts per mask
            - 'disconnected_masks': List of mask indices that have multiple components
    """
    # Convert nodes to 1-indexed room types (RPLAN IDs)
    room_types = nodes + 1
    
    total_room_regions = 0
    expected_rooms = 0
    expected_room_types = []
    actual_room_types = []
    per_mask_components = []
    disconnected_masks = []
    
    for k, room_type in enumerate(room_types):
        # Only count actual rooms (exclude doors type 17 and front door type 15)
        if room_type not in [15, 17]:
            expected_rooms += 1
            expected_room_types.append(int(room_type))
            
            mask = (masks[k] > 0).astype(np.uint8)
            num_labels, _ = cv2.connectedComponents(mask)
            num_components = num_labels - 1  # subtract background label
            
            total_room_regions += num_components
            per_mask_components.append(num_components)
            
            # Add room type for each component (including disconnected parts)
            for _ in range(num_components):
                actual_room_types.append(int(room_type))
            
            # Track which masks have disconnected regions
            if num_components > 1:
                disconnected_masks.append({
                    'mask_idx': k,
                    'room_type': int(room_type),
                    'num_components': num_components
                })
    
    return {
        'total_rooms': total_room_regions,
        'expected_rooms': expected_rooms,
        'expected_room_types': expected_room_types,
        'actual_room_types': actual_room_types,
        'per_mask_components': per_mask_components,
        'disconnected_masks': disconnected_masks
    }



def polygon_sample_to_masks(sample, image_size=256):
    """
    Convert polygon data from pickle sample to masks format.
    
    Args:
        sample: Dictionary containing 'polygons' and 'room_types'
        image_size: Output mask size (default: 256 to match HouseGAN++ output)
    
    Returns:
        masks: numpy array [num_rooms, image_size, image_size]
    """
    polygons = sample['polygons']
    room_types = sample['room_types']
    num_rooms = len(room_types)
    
    # Initialize masks
    masks = np.zeros((num_rooms, image_size, image_size), dtype=np.float32)
    
    for i, poly in enumerate(polygons):
        # Create a blank image
        img = Image.new('L', (image_size, image_size), 0)
        draw = ImageDraw.Draw(img)
        
        # Scale polygon coordinates to image_size
        # Polygons are in [0, 64] coordinates (from Hypergrph1K Dataset)
        scaled_poly = [(int(x * image_size / 64.0), int(y * image_size / 64.0)) for x, y in poly]
        
        # Draw filled polygon
        if len(scaled_poly) >= 3:
            draw.polygon(scaled_poly, fill=255)
        
        # Convert to numpy array
        masks[i] = np.array(img) / 255.0
    
    return masks


def extract_room_geometries_from_masks(nodes, masks):
    """
    Extract geometric properties (area and perimeter) from room masks.
    
    Args:
        nodes: Array of room types (integer indices, 0-indexed from one-hot)
        masks: Generated masks for each room/door [num_masks, H, W]
    
    Returns:
        List of dicts with 'room_type', 'area', 'perimeter' for each room
    """
    # Convert nodes to 1-indexed room types (RPLAN IDs)
    room_types = nodes + 1
    
    room_geometries = []
    
    for k, room_type in enumerate(room_types):
        # Only process actual rooms (exclude doors type 17 and front door type 15)
        if room_type not in [15, 17]:
            mask = (masks[k] > 0).astype(np.uint8)
            
            # Find contours for this mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process each contour (handle disconnected regions)
            for contour in contours:
                if len(contour) >= 3:  # Need at least 3 points for valid contour
                    # Compute area and perimeter
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, closed=True)
                    
                    if area > 0 and perimeter > 0:  # Valid geometry
                        room_geometries.append({
                            'room_type': int(room_type),
                            'area': area,
                            'perimeter': perimeter,
                            'mask_idx': k
                        })
    
    return room_geometries


def compute_spatial_overlap_ratio_from_masks(nodes, masks):
    """Fraction of total room area that overlaps with other rooms (mask-based).

    Args:
        nodes: Array of room types (0-indexed from one-hot).
        masks: Generated masks ``[num_masks, H, W]``.

    Returns:
        float in ``[0, 1]``, or ``nan`` if no room pixels exist.
    """
    room_types = nodes + 1
    binary_masks = []
    for k, room_type in enumerate(room_types):
        if room_type not in [15, 17]:
            binary_masks.append(masks[k] > 0)
    if not binary_masks:
        return float('nan')
    sum_of_areas = sum(int(np.count_nonzero(m)) for m in binary_masks)
    if sum_of_areas == 0:
        return float('nan')
    union_mask = np.zeros_like(binary_masks[0], dtype=bool)
    for m in binary_masks:
        union_mask |= m
    union_area = int(np.count_nonzero(union_mask))
    return 1.0 - (union_area / sum_of_areas)


def compute_room_similarity_score(pred_nodes, pred_masks, gt_nodes, gt_masks):
    """
    Compute room similarity score based on geometric properties.
    
    The similarity score δ = abs(1 - (L_{SA} * L_B) / (L_A * L_{SB}))
    where:
        L_A = perimeter of predicted room A
        L_{SA} = perimeter of square with same area as A = 4*sqrt(area_A)
        L_B = perimeter of ground truth room B
        L_{SB} = perimeter of square with same area as B = 4*sqrt(area_B)
    
    Rooms are matched by type, pairing largest areas first.
    Returns NaN if room counts don't match (to not penalize count mismatch).
    
    Args:
        pred_nodes: Predicted room types (0-indexed)
        pred_masks: Predicted masks [num_masks, H, W]
        gt_nodes: Ground truth room types (0-indexed)
        gt_masks: Ground truth masks (converted from pickle data)
    
    Returns:
        Dictionary with:
            - 'similarity_score': Average similarity (lower is better, 0 = perfect match)
            - 'matched_pairs': Number of room pairs compared
            - 'unmatched_pred': Number of unmatched predicted rooms
            - 'unmatched_gt': Number of unmatched GT rooms
            - 'per_room_scores': List of scores for each matched pair
    """
    # Extract geometries from masks
    pred_rooms = extract_room_geometries_from_masks(pred_nodes, pred_masks)
    gt_rooms = extract_room_geometries_from_masks(gt_nodes, gt_masks)
    
    # Group rooms by type
    pred_by_type = {}
    gt_by_type = {}
    
    for room in pred_rooms:
        room_type = room['room_type']
        if room_type not in pred_by_type:
            pred_by_type[room_type] = []
        pred_by_type[room_type].append(room)
    
    for room in gt_rooms:
        room_type = room['room_type']
        if room_type not in gt_by_type:
            gt_by_type[room_type] = []
        gt_by_type[room_type].append(room)
    
    # Match rooms and compute similarity scores
    similarity_scores = []
    matched_pairs = 0
    unmatched_pred_count = 0
    unmatched_gt_count = 0
    
    # Get all room types present in either prediction or ground truth
    all_types = set(pred_by_type.keys()) | set(gt_by_type.keys())
    
    for room_type in all_types:
        pred_rooms_of_type = pred_by_type.get(room_type, [])
        gt_rooms_of_type = gt_by_type.get(room_type, [])
        
        # Sort by area (largest first)
        pred_rooms_of_type = sorted(pred_rooms_of_type, key=lambda r: r['area'], reverse=True)
        gt_rooms_of_type = sorted(gt_rooms_of_type, key=lambda r: r['area'], reverse=True)
        
        # Match as many rooms as possible (ignore extras)
        # If counts differ, match the common rooms and track unmatched
        num_to_match = min(len(pred_rooms_of_type), len(gt_rooms_of_type))
        
        # Track unmatched rooms for this type
        unmatched_pred_count += len(pred_rooms_of_type) - num_to_match
        unmatched_gt_count += len(gt_rooms_of_type) - num_to_match
        
        # Pair largest with largest for the common rooms
        for pred_room, gt_room in zip(pred_rooms_of_type[:num_to_match], 
                                       gt_rooms_of_type[:num_to_match]):
            # Extract properties
            area_A = pred_room['area']
            perimeter_A = pred_room['perimeter']
            area_B = gt_room['area']
            perimeter_B = gt_room['perimeter']
            
            # Compute perimeter of square with same area
            # For square: perimeter = 4 * side = 4 * sqrt(area)
            L_SA = 4 * np.sqrt(area_A)
            L_SB = 4 * np.sqrt(area_B)
            
            # Compute similarity score
            # δ = abs(1 - (L_{SA} * L_B) / (L_A * L_{SB}))
            denominator = perimeter_A * L_SB
            if denominator > 0:
                delta = abs(1 - (L_SA * perimeter_B) / denominator)
                similarity_scores.append(delta)
                matched_pairs += 1
    
    # Compute average similarity
    if len(similarity_scores) > 0:
        avg_similarity = np.mean(similarity_scores)
    else:
        # No rooms to compare at all
        avg_similarity = np.nan
    
    return {
        'similarity_score': avg_similarity,
        'matched_pairs': matched_pairs,
        'unmatched_pred': unmatched_pred_count,
        'unmatched_gt': unmatched_gt_count,
        'per_room_scores': similarity_scores
    }


def _infer(graph, model, prev_state=None):
    """Run inference on the model."""
    # Configure input to the network
    z, given_masks_in, given_nds, given_eds = _init_input(graph, prev_state)
    
    # Run inference model
    with torch.no_grad():
        device = next(model.parameters()).device
        masks = model(z.to(device), given_masks_in.to(device),
                     given_nds.to(device), given_eds.to(device))
        masks = masks.detach().cpu().numpy()
    
    return masks


def remove_multiple_components(masks, nodes):
    """
    For each mask, keep only the largest connected component.
    
    Matches HouseGAN++ misc/utils.py remove_multiple_components().
    
    Args:
        masks: List or array of masks
        nodes: Array of 0-indexed room type indices
    
    Returns:
        List of cleaned masks
    """
    new_masks = []
    for mk, nd in zip(masks, nodes):
        m_cv = np.array(mk)
        m_cv[m_cv > 0] = 255.0
        m_cv[m_cv < 0] = 0.0
        m_cv = m_cv.astype('uint8')
        ret, thresh = cv2.threshold(m_cv, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 1:
            cnt_m = np.zeros_like(m_cv)
            c = max(contours, key=cv2.contourArea)
            cv2.drawContours(cnt_m, [c], 0, (255, 255, 255), -1)
            cnt_m[cnt_m > 0] = 1.0
            cnt_m[cnt_m < 0] = -1.0
            new_masks.append(cnt_m)
        else:
            new_masks.append(mk)
    return new_masks


def build_access_graph_from_masks(nodes, masks):
    """
    Build access graph from generated masks using IoU-based door detection.
    
    Matches the algorithm from HouseGAN++ misc/utils.py estimate_graph():
    - Remove multiple components per mask (keep largest)
    - For each door, compute IoU with every room mask
    - Accept connections where 0 < IoU < 0.2
    - Interior doors (type 17): connect the top-2 rooms by IoU
    - Front doors (type 15): connect the top-1 room to an outside node (-1)
    
    Args:
        nodes: Array of room types (integer indices, 0-indexed from one-hot)
        masks: Generated masks for each room/door
    
    Returns:
        NetworkX graph with room nodes, an outside node (-1), and door-based connections
    """
    masks = remove_multiple_components(masks, nodes)

    G = nx.Graph()

    for k, label in enumerate(nodes):
        _type = label + 1
        if _type >= 0 and _type not in [15, 17]:
            G.add_node(k, label=int(_type))

    G.add_node(-1, label=-1)

    doors_inds = np.where((nodes == 14) | (nodes == 16))[0]
    rooms_inds = np.where((nodes != 14) & (nodes != 16))[0]
    doors_rooms_map = defaultdict(list)
    for k in doors_inds:
        for l in rooms_inds:
            if k > l:
                m1 = np.array(masks[k], dtype=float)
                m2 = np.array(masks[l], dtype=float)
                m1[m1 > 0] = 1.0
                m1[m1 <= 0] = 0.0
                m2[m2 > 0] = 1.0
                m2[m2 <= 0] = 0.0
                union = float(np.logical_or(m1, m2).sum())
                if union > 0:
                    iou = np.logical_and(m1, m2).sum() / union
                    if iou > 0 and iou < 0.2:
                        doors_rooms_map[k].append((l, iou))

    for k in doors_rooms_map.keys():
        _conn = doors_rooms_map[k]
        _conn = sorted(_conn, key=lambda tup: tup[1], reverse=True)
        _conn_top2 = _conn[:2]
        if nodes[k] + 1 != 15:  # interior door
            if len(_conn_top2) > 1:
                l1, l2 = _conn_top2[0][0], _conn_top2[1][0]
                if not G.has_edge(l1, l2):
                    G.add_edge(l1, l2)
        else:  # front door -> connect to outside node
            if len(_conn) > 0:
                l1 = _conn[0][0]
                if not G.has_edge(-1, l1):
                    G.add_edge(-1, l1)

    return G


def build_access_graph_from_structure(nodes, edges, masks=None):
    """
    Build GT access graph from graph structure (triples).
    
    Matches the algorithm from HouseGAN++ misc/utils.py draw_graph():
    - Room nodes for all non-door types
    - Outside node (-1) connected via front doors (type 15)
    - Room-room edges where both endpoints are rooms and rel > 0
    - Front door edges: if one endpoint is type 15 and rel > 0,
      connect the other (non-door) room to outside (-1)
    
    Args:
        nodes: Array of room types (integer indices, 0-indexed from one-hot)
        edges: Edge triples [node1_idx, relationship, node2_idx]
        masks: Unused (kept for API compatibility)
    
    Returns:
        NetworkX graph with room nodes, outside node (-1), and door-based connections
    """
    G = nx.Graph()

    for k, label in enumerate(nodes):
        _type = label + 1
        if _type >= 0 and _type not in [15, 17]:
            G.add_node(k, label=int(_type))

    G.add_node(-1, label=-1)

    for edge in edges:
        k = int(edge[0].item() if hasattr(edge[0], 'item') else edge[0])
        m = int(edge[1].item() if hasattr(edge[1], 'item') else edge[1])
        l = int(edge[2].item() if hasattr(edge[2], 'item') else edge[2])

        _type_k = int(nodes[k]) + 1
        _type_l = int(nodes[l]) + 1

        if m > 0 and (_type_k not in [15, 17] and _type_l not in [15, 17]):
            G.add_edge(k, l)
        elif m > 0 and (_type_k == 15 or _type_l == 15) and (_type_l != 17 and _type_k != 17):
            if _type_k == 15:
                G.add_edge(l, -1)
            elif _type_l == 15:
                G.add_edge(k, -1)

    return G


def generate_floorplan(graph, model, selected_types):
    """
    Generate floorplan using incremental generation strategy.
    
    Args:
        graph: [nodes, edges]
        model: Generator model
        selected_types: List of room type groups for incremental generation
    
    Returns:
        Generated masks
    """
    real_nodes = np.where(graph[0].detach().cpu() == 1)[-1]
    
    # Initialize layout
    state = {'masks': None, 'fixed_nodes': []}
    masks = _infer(graph, model, state)
    
    # Generate per room type (incremental)
    for _iter, _types in enumerate(selected_types):
        _fixed_nds = np.concatenate([np.where(real_nodes == _t)[0] for _t in _types]) \
            if len(_types) > 0 else np.array([])
        state = {'masks': masks, 'fixed_nodes': _fixed_nds}
        masks = _infer(graph, model, state)
    
    return masks



def save_generations_houseganpp(filepath, metadata, samples):
    """
    Save generated outputs for later analysis.
    
    Args:
        filepath: Path to save pickle file
        metadata: Dict with generation metadata
        samples: List of sample dicts with generated data
    """
    data = {
        'metadata': metadata,
        'samples': samples
    }
    
    print(f"\n💾 Saving generations to {filepath}...")
    
    # Create directory if needed
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"✅ Saved {len(samples)} samples ({file_size_mb:.1f} MB)")
    print(f"   Variations per sample: {metadata['num_variations']}")
    print(f"   Total variations: {len(samples) * metadata['num_variations']}")


def load_generations_houseganpp(filepath):
    """
    Load previously generated outputs.
    
    Args:
        filepath: Path to pickle file
    
    Returns:
        Dict with 'metadata' and 'samples'
    """
    print(f"\n📂 Loading generations from {filepath}...")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # Validate structure
    if 'metadata' not in data or 'samples' not in data:
        raise ValueError(f"Invalid generation cache format: {filepath}")
    
    metadata = data['metadata']
    samples = data['samples']
    
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"✅ Loaded {len(samples)} samples ({file_size_mb:.1f} MB)")
    print(f"   Model: {metadata.get('model', 'Unknown')}")
    print(f"   Checkpoint: {metadata.get('checkpoint', 'Unknown')}")
    print(f"   Variations: {metadata.get('num_variations', 'Unknown')}")
    print(f"   Generated: {metadata.get('timestamp', 'Unknown')}")
    
    return data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute GED for access graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Data parameters
    parser.add_argument("--data_path", type=str, required=True,
                       help="path to test pickle file")
    parser.add_argument("--target_set", type=int, default=None,
                       help="filter by number of rooms (None = no filter)")
    
    # Model parameters
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="path to trained generator checkpoint"
                            " (required unless --load_generations is used)")
    
    # Evaluation parameters
    parser.add_argument("--batch_size", type=int, default=1,
                       help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=4,
                       help="number of cpu threads")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="maximum samples to process (None = all)")
    parser.add_argument("--num_variations", type=int, default=1,
                       help="number of generation variations per sample")
    parser.add_argument("--upper_bound", type=int, default=40,
                       help="upper bound for GED computation")
    parser.add_argument("--save_graphs", action='store_true',
                       help="save visualizations of graphs")
    parser.add_argument("--out", type=str, default="./ged_results",
                       help="output folder for results")
    
    # Generation caching
    parser.add_argument("--save_generations", type=str, default=None,
                       help="path to save generated outputs for later analysis (enables caching)")
    parser.add_argument("--load_generations", type=str, default=None,
                       help="path to load previously generated outputs (skips model inference)")

    return parser.parse_args()

def visualize_comparison(
        raw_sample, 
        G_true, 
        G_pred, 
        gen_masks,
        real_nodes,
        sample_idx, 
        output_dir, 
        room_count_info, 
        similarity_result,
        sample_id=None
    ):
    """
    Visualize comparison of ground truth and predicted access graphs.

    Args:
        sample_id: Optional sample ID string to display in title
    """
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    
    # Top left: Ground truth floorplan
    ax_floor_gt = fig.add_subplot(gs[0, 0])
    if raw_sample is not None:
        floor_img_gt = draw_floorplan_from_pickle(raw_sample, output_size=512)
        ax_floor_gt.imshow(floor_img_gt)
        ax_floor_gt.set_title('Ground Truth Floor Plan', fontsize=12, fontweight='bold')
    ax_floor_gt.axis('off')
    
    # Bottom left: Ground truth access graph
    ax_graph_gt = fig.add_subplot(gs[1, 0])
    ax_graph_gt.set_title('GT: Door Connections', fontsize=10)
    if G_true.number_of_nodes() > 0:
        try:
            pos_true = nx.nx_agraph.graphviz_layout(G_true, prog='neato')
        except:
            pos_true = nx.spring_layout(G_true, seed=42, k=1.5)
        node_colors = [ID_COLOR.get(G_true.nodes[n]['label'], '#BEBEBE') for n in G_true.nodes()]
        node_sizes = [750 if n == -1 else 1000 for n in G_true.nodes()]
        node_edgecolors = ['black' if n == -1 else 'blue' for n in G_true.nodes()]
        node_linewidths = [3.0 if n == -1 else 0.0 for n in G_true.nodes()]
        edge_colors = ['#727171' if -1 in (u, v) else '#D3A2C7' for u, v in G_true.edges()]
        nx.draw(G_true, pos_true, node_color=node_colors, node_size=node_sizes,
                edgecolors=node_edgecolors, linewidths=node_linewidths, edge_color=edge_colors,
                style='dotted', width=4.0, with_labels=False, ax=ax_graph_gt)
    ax_graph_gt.axis('off')
    
    # Top right: Predicted floorplan
    room_types = real_nodes + 1
    ax_floor_pred = fig.add_subplot(gs[0, 1])
    floor_img_pred = draw_masks(gen_masks.copy(), room_types)
    ax_floor_pred.imshow(floor_img_pred)
    ax_floor_pred.set_title('Predicted Floor Plan (Generated)', fontsize=12, fontweight='bold')
    ax_floor_pred.axis('off')
    
    # Bottom right: Predicted access graph
    ax_graph_pred = fig.add_subplot(gs[1, 1])
    ax_graph_pred.set_title('Predicted: Door Connections', fontsize=10)
    if len(G_pred.nodes()) > 0:
        try:
            pos_pred = nx.nx_agraph.graphviz_layout(G_pred, prog='neato')
        except:
            pos_pred = nx.spring_layout(G_pred, seed=42, k=1.5)
        node_colors = [ID_COLOR.get(G_pred.nodes[n]['label'], '#BEBEBE') for n in G_pred.nodes()]
        node_sizes = [750 if n == -1 else 1000 for n in G_pred.nodes()]
        node_edgecolors = ['black' if n == -1 else 'blue' for n in G_pred.nodes()]
        node_linewidths = [3.0 if n == -1 else 0.0 for n in G_pred.nodes()]
        edge_colors = ['#727171' if -1 in (u, v) else '#b8b5b4' for u, v in G_pred.edges()]
        nx.draw(G_pred, pos_pred, node_color=node_colors, node_size=node_sizes,
                edgecolors=node_edgecolors, linewidths=node_linewidths, edge_color=edge_colors,
                style='dotted', width=4.0, with_labels=False, ax=ax_graph_pred)
    ax_graph_pred.axis('off')
    
    # Add room count and similarity info to title
    room_info_text = f"Expected: {room_count_info['expected_rooms']} rooms | Generated: {room_count_info['total_rooms']} rooms"
    if len(room_count_info['disconnected_masks']) > 0:
        room_info_text += f" (⚠ {len(room_count_info['disconnected_masks'])} disconnected masks)"
    
    # Add similarity score if available
    if not np.isnan(similarity_result['similarity_score']):
        room_info_text += f" | δ={similarity_result['similarity_score']:.3f}"
    
    # Create title with sample ID if available
    if sample_id:
        title = f'Sample {sample_idx} ({sample_id}) - Access Graph Comparison (Door Connections)\n{room_info_text}'
    else:
        title = f'Sample {sample_idx} - Access Graph Comparison (Door Connections)\n{room_info_text}'
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save PNG for quick viewing
    plt.savefig(os.path.join(output_dir, f'comparison_{sample_idx:04d}.png'),
                dpi=150, bbox_inches='tight')
    
    # Save PDF for publication quality
    plt.savefig(os.path.join(output_dir, f'comparison_{sample_idx:04d}.pdf'),
                format='pdf', bbox_inches='tight', dpi=300)
    
    plt.close()


def main():
    args = parse_args()
    print(args)
    
    # Convert paths to absolute paths (before any directory changes)
    args.data_path = os.path.abspath(args.data_path)
    if args.checkpoint:
        args.checkpoint = os.path.abspath(args.checkpoint)
    args.out = os.path.abspath(args.out)
    if args.save_generations:
        args.save_generations = os.path.abspath(args.save_generations)
    if args.load_generations:
        args.load_generations = os.path.abspath(args.load_generations)
    
    # Validate caching arguments
    if args.save_generations and args.load_generations:
        raise ValueError("Cannot use both --save_generations and --load_generations")
    
    if args.load_generations:
        if not os.path.exists(args.load_generations):
            raise FileNotFoundError(f"Generation cache not found: {args.load_generations}")
        if args.checkpoint:
            print("⚠️  Warning: --checkpoint is ignored when loading generations")
        # Checkpoint not required in load mode
        args.checkpoint = None
    else:
        # Generate mode requires checkpoint
        if not args.checkpoint:
            raise ValueError("--checkpoint is required when generating (not loading)")
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    
    # Determine mode
    load_mode = args.load_generations is not None
    save_mode = args.save_generations is not None
    
    if load_mode:
        print("\n🔄 MODE: Load cached generations")
    elif save_mode:
        print("\n💾 MODE: Generate and save")
    else:
        print("\n🚀 MODE: Generate only")
    
    # Initialize generator (only in generate mode)
    if not load_mode:
        print("\nInitializing generator...")
        model = Generator()
        
        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if isinstance(checkpoint, dict) and 'generator' in checkpoint:
            model.load_state_dict(checkpoint['generator'], strict=False)
            print(f"Loaded fine-tuned model from epoch {checkpoint.get('epoch', '?')}")
        else:
            model.load_state_dict(checkpoint, strict=False)
            print("Loaded pretrained model")
        
        model = model.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model.cuda()
            print("Using CUDA")
        else:
            print("Using CPU")
    else:
        model = None  # No model in load mode
    
    # Collect graph pairs
    graphs = []
    
    # ========================================================================
    # LOAD MODE: Process cached generations
    # ========================================================================
    if load_mode:
        cached_data = load_generations_houseganpp(args.load_generations)
        
        print(f"\nProcessing {len(cached_data['samples'])} cached samples...")
        
        for sample_data in tqdm(cached_data['samples'], desc="Processing cached"):
            sample_idx = sample_data['sample_idx']
            real_nodes = sample_data['real_nodes']
            raw_sample = sample_data.get('raw_sample', None)
            
            # Build GT access graph from polygon data
            # Use polygon_sample_to_masks to get GT masks, then build graph from them
            if raw_sample is not None:
                try:
                    gt_masks = polygon_sample_to_masks(raw_sample, image_size=256)
                    G_true = build_access_graph_from_masks(real_nodes, gt_masks)
                except Exception as e:
                    print(f"Warning: Could not build GT graph for sample {sample_idx}: {e}")
                    G_true = nx.Graph()
            else:
                G_true = nx.Graph()
            
            # Process each variation
            for var_data in sample_data['variations']:
                gen_masks = var_data['gen_masks']
                var = var_data['variation_idx']
                
                # Build predicted access graph from masks
                G_pred = build_access_graph_from_masks(real_nodes, gen_masks)
                
                # Count actual rooms and get room types
                room_count_info = count_actual_rooms_from_masks(real_nodes, gen_masks)
                                
                overlap_ratio = compute_spatial_overlap_ratio_from_masks(real_nodes, gen_masks)

                # Compute room similarity score and area proportion error
                similarity_result = {'similarity_score': np.nan}
                area_prop_result = {'area_proportion_error': np.nan}
                if raw_sample is not None:
                    try:
                        gt_masks = polygon_sample_to_masks(raw_sample, image_size=gen_masks.shape[1])
                        similarity_result = compute_room_similarity_score(
                            real_nodes, gen_masks,
                            real_nodes, gt_masks
                        )
                        pred_rooms = extract_room_geometries_from_masks(real_nodes, gen_masks)
                        gt_rooms = extract_room_geometries_from_masks(real_nodes, gt_masks)
                        area_prop_result = compute_area_prop_metric(
                            _room_geometries_to_area_tuples(pred_rooms),
                            _room_geometries_to_area_tuples(gt_rooms),
                        )
                    except Exception as e:
                        print(f"⚠️  Warning: Could not compute similarity/area scores for sample {sample_idx}: {e}")
                
                graphs.append((G_pred, G_true, sample_idx, var, room_count_info, similarity_result, area_prop_result, overlap_ratio))
                
                # Optionally save visualizations
                if args.save_graphs and var == 0:
                    # Extract sample ID from raw_sample
                    sample_id = raw_sample.get('id', None) if raw_sample else None
                    visualize_comparison(
                        raw_sample, 
                        G_true, 
                        G_pred, 
                        gen_masks,
                        real_nodes,
                        sample_idx, 
                        args.out, 
                        room_count_info, 
                        similarity_result,
                        sample_id=sample_id
                    )
    
    # ========================================================================
    # GENERATE MODE: Run model and optionally save
    # ========================================================================
    else:
        # Load raw pickle data for GT structure
        print("\nLoading raw pickle data...")
        with open(args.data_path, 'rb') as f:
            raw_data = pickle.load(f)
        
        # Load dataset
        print("\nLoading dataset...")
        fp_dataset_test = BenchmarkDataset(args.data_path, target_set=args.target_set, split='test')
        fp_loader = DataLoader(
            fp_dataset_test,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.n_cpu
        )
        
        print(f"\nEvaluating {len(fp_dataset_test)} samples...")
        print(f"Computing GED for access graphs...\n")
        
        # Storage for caching
        if save_mode:
            cache_samples = []
        
        for sample_idx, sample in enumerate(tqdm(fp_loader, desc="Generating")):
            if args.num_samples is not None and sample_idx >= args.num_samples:
                break
            
            # Unpack sample
            mks, nds, eds, nd_to_sample, ed_to_sample = sample
            real_nodes = np.where(nds.detach().cpu() == 1)[-1]
            graph = [nds, eds]
            
            # Get raw sample for floorplan
            raw_sample = raw_data[sample_idx] if sample_idx < len(raw_data) else None
            
            # Build ground truth access graph (rooms connected through doors)
            # Uses the ground truth edge structure to identify door connections
            G_true = build_access_graph_from_structure(
                real_nodes,
                eds.detach().cpu().numpy(),
                masks=None  # Use structure only for ground truth
            )
            
            # Get unique room types for incremental generation
            _types = sorted(list(set(real_nodes)))
            selected_types = [_types[:k+1] for k in range(len(_types))]
            
            # Storage for this sample's variations (if caching)
            if save_mode:
                sample_variations = []
            
            # Generate variations
            for var in range(args.num_variations):
                try:
                    # Generate floorplan
                    gen_masks = generate_floorplan(graph, model, selected_types)
                    
                    # Build predicted access graph from actual generated layout
                    # Extracts door connections from generated masks (which rooms connect through doors)
                    # Compares: GT door connections vs. predicted door connections
                    # Low GED = model learned correct door connectivity patterns
                    G_pred = build_access_graph_from_masks(real_nodes, gen_masks)
                    
                    # Count actual rooms in generated masks (accounting for disconnected regions)
                    room_count_info = count_actual_rooms_from_masks(real_nodes, gen_masks)
                    
                    overlap_ratio = compute_spatial_overlap_ratio_from_masks(real_nodes, gen_masks)

                    # Compute room similarity score and area proportion error
                    similarity_result = {'similarity_score': np.nan}
                    area_prop_result = {'area_proportion_error': np.nan}
                    if raw_sample is not None:
                        try:
                            gt_masks = polygon_sample_to_masks(raw_sample, image_size=gen_masks.shape[1])
                            similarity_result = compute_room_similarity_score(
                                real_nodes, gen_masks,
                                real_nodes, gt_masks
                            )
                            pred_rooms = extract_room_geometries_from_masks(real_nodes, gen_masks)
                            gt_rooms = extract_room_geometries_from_masks(real_nodes, gt_masks)
                            area_prop_result = compute_area_prop_metric(
                                _room_geometries_to_area_tuples(pred_rooms),
                                _room_geometries_to_area_tuples(gt_rooms),
                            )
                        except Exception as e:
                            print(f"⚠️  Warning: Could not compute similarity/area scores for sample {sample_idx}: {e}")
                    
                    graphs.append((G_pred, G_true, sample_idx, var, room_count_info, similarity_result, area_prop_result, overlap_ratio))
                    
                    # Store for caching if enabled
                    if save_mode:
                        sample_variations.append({
                            'variation_idx': var,
                            'gen_masks': gen_masks.copy()  # Copy to avoid reference issues
                        })
                    
                    # Optionally save graph visualizations with floorplan
                    if args.save_graphs and var == 0:
                        # Extract sample ID from raw_sample
                        sample_id = raw_sample.get('id', None) if raw_sample else None
                        visualize_comparison(
                            raw_sample, 
                            G_true, 
                            G_pred, 
                            gen_masks,
                            real_nodes,
                            sample_idx, 
                            args.out, 
                            room_count_info, 
                            similarity_result,
                            sample_id=sample_id
                        )
                    
                except Exception as e:
                    print(f"⚠️  Error processing sample {sample_idx}, variation {var}: {e}")
            
            # After all variations for this sample, store in cache if enabled
            if save_mode:
                cache_samples.append({
                    'sample_idx': sample_idx,
                    'real_nodes': real_nodes.copy(),
                    'raw_sample': raw_sample,
                    'variations': sample_variations
                })
        
        # After all samples, save cache if enabled
        if save_mode:
            metadata = {
                'model': 'HouseGAN++',
                'checkpoint': args.checkpoint,
                'data_path': args.data_path,
                'target_set': args.target_set,
                'num_variations': args.num_variations,
                'num_samples': len(cache_samples),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'version': '1.0'
            }
            save_generations_houseganpp(args.save_generations, metadata, cache_samples)
    
    print(f"\n{'='*60}")
    print(f"Generated {len(graphs)} graph pairs")

    if len(graphs) == 0:
        print("\n⚠️  No graph pairs to evaluate. Check that the input data is non-empty.")
        return

    print(f"Computing GED in parallel...\n")
    
    # Compute GED in parallel
    from joblib import Parallel, delayed
    num_cores = multiprocessing.cpu_count()
    
    results = Parallel(n_jobs=num_cores)(
        delayed(ged_between_graphs)(G_pred, G_true, _id, upper_bound=args.upper_bound)
        for G_pred, G_true, _id, _var, _room_info, _sim_result, _area_result, _overlap in tqdm(graphs, desc="Computing GED")
    )
    
    # Collect all GED results (with NaN for failed computations to preserve structure)
    all_dists = []
    room_count_accuracies = []
    room_count_accuracy_scores = []
    room_type_accuracy_scores = []
    similarity_scores = []
    area_proportion_errors = []
    spatial_overlap_ratios = []
    unmatched_pred_rooms = []
    unmatched_gt_rooms = []
    matched_room_pairs = []
    disconnected_count = 0
    
    for (min_dist, graph_len), (G_pred, G_true, _id, _var, room_count_info, similarity_result, area_prop_result, overlap_ratio) in zip(results, graphs):
        if min_dist is not None:
            all_dists.append(min_dist)
        else:
            # Add NaN to preserve num_variations structure
            all_dists.append(np.nan)
        
        # Compute room count and type accuracy
        room_count_metric = room_type_accuracy(
            room_count_info['expected_room_types'],
            room_count_info['actual_room_types']
        )
        room_count_accuracies.append(room_count_metric['accuracy'])
        room_count_accuracy_scores.append(room_count_metric['room_count_accuracy'])
        room_type_accuracy_scores.append(room_count_metric['room_type_accuracy'])
        
        # Collect similarity scores and matching stats
        similarity_scores.append(similarity_result['similarity_score'])
        unmatched_pred_rooms.append(similarity_result['unmatched_pred'])
        unmatched_gt_rooms.append(similarity_result['unmatched_gt'])
        matched_room_pairs.append(similarity_result['matched_pairs'])
        
        # Collect area proportion errors and spatial overlap
        area_proportion_errors.append(area_prop_result['area_proportion_error'])
        spatial_overlap_ratios.append(overlap_ratio)
        
        # Track disconnected masks
        if len(room_count_info['disconnected_masks']) > 0:
            disconnected_count += 1
    
    report_results(
        args.out,
        geds=np.array(all_dists),
        accuracies=np.array(room_count_accuracies),
        room_count_accuracies=np.array(room_count_accuracy_scores),
        room_type_accuracies=np.array(room_type_accuracy_scores),
        similarity_scores=np.array(similarity_scores),
        area_proportion_errors=np.array(area_proportion_errors),
        spatial_overlap_ratios=np.array(spatial_overlap_ratios),
        num_variations=args.num_variations,
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        extra={
            'target_set': args.target_set,
            'disconnected_count': disconnected_count,
            'total_samples': len(graphs),
            'matched_room_pairs': matched_room_pairs,
            'unmatched_pred_rooms': unmatched_pred_rooms,
            'unmatched_gt_rooms': unmatched_gt_rooms,
        },
    )


if __name__ == '__main__':
    main()
