#!/usr/bin/env python3
"""
Evaluate House-Diffusion floorplan generation.

Generates floorplans and computes four metrics against ground truth:
  1. GED  -- Graph Edit Distance between predicted and GT access graphs
  2. Room Accuracy -- Room count and type distribution matching
  3. Delta -- Room shape regularity deviation (geometric similarity)
  4. Area Difference -- Relative room-area error between paired rooms

Expects a pre-processed NPZ file via --data_path.  To convert from
RPLAN JSON files, use benchmark/utils/convert_json_to_npz.py first
(the shell wrapper eval_housediffusion.sh does this automatically).

Supports caching generated outputs for reproducible re-evaluation.
"""

import argparse
import os
import sys
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw
from shapely.geometry import Polygon as ShapelyPolygon
from shapely import make_valid
from datetime import datetime
import pickle

from utils.metrics import ged_between_graphs
from utils.metrics import room_type_accuracy
from benchmark.src.evaluate_houseganpp import draw_masks
from benchmark.utils.score_benchmark_result import score_benchmark_result
from house_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    update_arg_parser
)
from house_diffusion.rplanhg_datasets import load_rplanhg_data


# Room type constants (RPLAN format)
ROOM_CLASS = {
    "living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4,
    "balcony": 5, "entrance": 6, "dining room": 7, "study room": 8,
    "storage": 10, "front door": 15, "unknown": 16, "interior_door": 17
}
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
    -1: 'white',        # outside node (matches original House-Diffusion script)
}

# # Colors for visualization (RPLAN standard)
# ID_COLOR = {
#     1: '#EE4D4D',
#     2: '#C67C7B', 
#     3: '#FFD274', 
#     4: '#BEBEBE', 
#     5: '#BFE3E8', 
#     6: '#7BA779', 
#     7: '#E87A90', 
#     8: '#FF8C69', 
#     10: '#1F849B', 
#     15: '#727171', 
#     16: '#785A67', 
#     17: '#D3A2C7'
# }


def reverse_map_room_type(room_type):
    """
    Reverse map room types from House-Diffusion's internal encoding to RPLAN format.
    
    House-Diffusion remaps types > 10 during preprocessing:
        15 (front_door) -> 11
        16 (window) -> 13
        17 (interior_door) -> 12
    
    This function reverses that mapping for correct visualization and graph construction.
    
    Args:
        room_type: Room type ID from House-Diffusion model output
        
    Returns:
        Original RPLAN room type ID
    """
    if room_type in [11, 12, 13]:
        return {11: 15, 12: 17, 13: 16}[room_type]
    return room_type


def bin_to_int(binary_array):
    """Convert binary array to integer."""
    binary_array = binary_array.flatten()
    binary_array[binary_array < 0] = 0
    binary_array[binary_array > 0] = 1
    return int("".join([str(int(x)) for x in binary_array]), 2)


def corners_to_polygons_and_types(coords, room_types_onehot, room_indices_onehot, padding_mask, image_size=256):
    """
    Convert House-Diffusion corner representation to polygons and room types.
    
    Args:
        coords: Coordinates [2, num_points]
        room_types_onehot: Room types [num_points, 25]
        room_indices_onehot: Room indices [num_points, 32]
        padding_mask: Padding mask [num_points] (1 = valid, 0 = padding)
        image_size: Image size for coordinate scaling
    
    Returns:
        polygons: List of (polygon_coords, room_type, room_idx)
    """
    # Ensure coords is [2, num_points]
    if coords.shape[0] != 2:
        coords = coords.T
    
    num_points = coords.shape[1]
    
    # Group points by room
    room_polygons = {}
    current_room_idx = None
    current_poly = []
    current_room_type = None
    
    for j in range(num_points):
        # Skip padding
        if padding_mask[j] == 0:
            continue
        
        # Get room index and type
        room_idx = np.argmax(room_indices_onehot[j])
        room_type = np.argmax(room_types_onehot[j])
        room_type = reverse_map_room_type(room_type)
        
        # Convert coordinates from [-1, 1] to [0, image_size]
        x, y = coords[:, j]
        x_pixel = (x / 2 + 0.5) * image_size
        y_pixel = (y / 2 + 0.5) * image_size
        
        # Check if we're starting a new room
        if room_idx != current_room_idx:
            # Save previous polygon
            if current_poly and len(current_poly) >= 3:
                if current_room_idx not in room_polygons:
                    room_polygons[current_room_idx] = []
                room_polygons[current_room_idx].append({
                    'coords': current_poly,
                    'room_type': current_room_type,
                    'room_idx': current_room_idx
                })
            
            # Start new polygon
            current_room_idx = room_idx
            current_room_type = room_type
            current_poly = []
        
        current_poly.append((x_pixel, y_pixel))
    
    # Save last polygon
    if current_poly and len(current_poly) >= 3:
        if current_room_idx not in room_polygons:
            room_polygons[current_room_idx] = []
        room_polygons[current_room_idx].append({
            'coords': current_poly,
            'room_type': current_room_type,
            'room_idx': current_room_idx
        })
    
    # Flatten to list
    all_polygons = []
    for room_idx in sorted(room_polygons.keys()):
        all_polygons.extend(room_polygons[room_idx])
    
    return all_polygons


def count_actual_rooms_from_polygons(polygons):
    """
    Count the actual number of visible room regions from House-Diffusion polygons.
    
    Important: House-Diffusion can produce polygons that are completely overlapped
    by other (larger) polygons and thus not visible in the output. This function
    identifies and counts only the visible room regions.
    
    Args:
        polygons: List of dicts with 'coords', 'room_type', 'room_idx'
    
    Returns:
        Dictionary with:
            - 'total_rooms': Total count of visible room regions
            - 'expected_rooms': Expected count (number of room polygons excluding doors)
            - 'expected_room_types': List of expected room types (RPLAN IDs)
            - 'actual_room_types': List of actual visible room types
            - 'hidden_rooms': List of room polygons that are completely hidden
            - 'visible_rooms': List of visible room polygons
    """
    # Separate rooms from doors
    room_polygons = []
    expected_rooms = 0
    expected_room_types = []
    
    for poly_dict in polygons:
        room_type = poly_dict['room_type']
        
        # Only count actual rooms (exclude doors type 17 and front door type 15)
        if room_type not in [15, 17]:
            expected_rooms += 1
            expected_room_types.append(int(room_type))
            
            # Create Shapely polygon for area calculation
            try:
                shapely_poly = ShapelyPolygon(poly_dict['coords'])
                # Fix invalid geometries
                if not shapely_poly.is_valid:
                    shapely_poly = make_valid(shapely_poly)
                
                room_polygons.append({
                    'polygon': shapely_poly,
                    'room_type': room_type,
                    'room_idx': poly_dict.get('room_idx', -1),
                    'coords': poly_dict['coords']
                })
            except Exception as e:
                # Skip invalid polygons
                continue
    
    # Check which rooms are visible (not completely hidden by others)
    visible_rooms = []
    hidden_rooms = []
    
    for i, room_a in enumerate(room_polygons):
        poly_a = room_a['polygon']
        is_hidden = False
        
        # Check if this polygon is completely contained within any other polygon
        for j, room_b in enumerate(room_polygons):
            if i == j:
                continue
            
            poly_b = room_b['polygon']
            
            try:
                # Check if poly_a is almost completely covered by poly_b
                # A room is considered hidden if > 95% of its area is covered by another room
                intersection = poly_a.intersection(poly_b)
                if intersection.area / poly_a.area > 0.95:
                    is_hidden = True
                    hidden_rooms.append({
                        'room_type': room_a['room_type'],
                        'room_idx': room_a['room_idx'],
                        'covered_by': room_b['room_type'],
                        'coverage_ratio': intersection.area / poly_a.area
                    })
                    break
            except Exception:
                # If intersection fails, assume visible
                continue
        
        if not is_hidden:
            visible_rooms.append(room_a)
    
    # Extract actual room types from visible rooms
    actual_room_types = [int(room['room_type']) for room in visible_rooms]
    
    return {
        'total_rooms': len(visible_rooms),
        'expected_rooms': expected_rooms,
        'expected_room_types': expected_room_types,
        'actual_room_types': actual_room_types,
        'hidden_rooms': hidden_rooms,
        'visible_rooms': visible_rooms
    }


def extract_room_geometries_from_polygons(polygons):
    """
    Extract geometric properties (area, perimeter) from polygon data.
    
    Args:
        polygons: List of polygon dicts with 'room_type' and 'coords'
    
    Returns:
        List of dicts with room_type, area, perimeter
    """
    room_geometries = []
    
    for poly_data in polygons:
        room_type = poly_data['room_type']
        
        # Skip doors and front doors
        if room_type in [15, 17]:
            continue
        
        coords = poly_data['coords']
        
        try:
            # Create Shapely polygon
            shapely_poly = ShapelyPolygon(coords)
            
            # Fix invalid geometries
            if not shapely_poly.is_valid:
                shapely_poly = make_valid(shapely_poly)
            
            # Extract geometric properties
            area = shapely_poly.area
            perimeter = shapely_poly.length  # In Shapely, length gives perimeter for polygons
            
            if area > 0 and perimeter > 0:
                room_geometries.append({
                    'room_type': int(room_type),
                    'area': area,
                    'perimeter': perimeter
                })
        except Exception as e:
            # Skip invalid polygons
            continue
    
    return room_geometries


def compute_room_similarity_score_polygons(pred_polygons, gt_polygons):
    """
    Compute room similarity score using polygon-based representation.
    
    This is the polygon-based version of the metric - more accurate for
    polygon-based models like House-Diffusion.
    
    Args:
        pred_polygons: Predicted polygon list
        gt_polygons: Ground truth polygon list
    
    Returns:
        Dict with similarity_score, matched_pairs, unmatched counts, per_room_scores
    """
    # Extract geometries
    pred_rooms = extract_room_geometries_from_polygons(pred_polygons)
    gt_rooms = extract_room_geometries_from_polygons(gt_polygons)
    
    # Group by type
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


def compute_room_area_difference_score_polygons(pred_polygons, gt_polygons):
    """Compute mean relative room-area error using polygon-based representation.

    Pairing logic mirrors :func:`compute_room_similarity_score_polygons`:
    rooms grouped by type, sorted by area descending, paired largest-first.
    Per-pair error is ``abs(area_pred - area_gt) / area_gt``.
    """
    pred_rooms = extract_room_geometries_from_polygons(pred_polygons)
    gt_rooms = extract_room_geometries_from_polygons(gt_polygons)

    pred_by_type = {}
    gt_by_type = {}
    for room in pred_rooms:
        pred_by_type.setdefault(room['room_type'], []).append(room)
    for room in gt_rooms:
        gt_by_type.setdefault(room['room_type'], []).append(room)

    area_errors = []
    matched_pairs = 0
    unmatched_pred_count = 0
    unmatched_gt_count = 0

    for room_type in set(pred_by_type.keys()) | set(gt_by_type.keys()):
        pred_of_type = sorted(pred_by_type.get(room_type, []), key=lambda r: r['area'], reverse=True)
        gt_of_type = sorted(gt_by_type.get(room_type, []), key=lambda r: r['area'], reverse=True)
        num_to_match = min(len(pred_of_type), len(gt_of_type))
        unmatched_pred_count += len(pred_of_type) - num_to_match
        unmatched_gt_count += len(gt_of_type) - num_to_match

        for pred_room, gt_room in zip(pred_of_type[:num_to_match], gt_of_type[:num_to_match]):
            area_gt = gt_room['area']
            if area_gt > 0:
                area_errors.append(abs(pred_room['area'] - area_gt) / area_gt)
                matched_pairs += 1

    avg_error = np.mean(area_errors) if area_errors else np.nan

    return {
        'area_difference_score': avg_error,
        'matched_pairs': matched_pairs,
        'unmatched_pred': unmatched_pred_count,
        'unmatched_gt': unmatched_gt_count,
        'per_room_scores': area_errors,
    }


def build_access_graph_from_polygons(polygons):
    """
    Build access graph from polygon representation using IoU-based door detection.
    
    Matches the algorithm from House-Diffusion's original image_sample.py:
    - For each door, compute IoU with every room polygon
    - Accept connections where 0 < IoU < 0.2
    - Interior doors (type 17): connect the top-2 rooms by IoU
    - Front doors (type 15): connect the top-1 room to an outside node (-1)
    
    Args:
        polygons: List of dicts with 'coords', 'room_type', 'room_idx'
    
    Returns:
        NetworkX graph with room nodes, an outside node (-1), and door-based connections
    """
    from collections import defaultdict

    G = nx.Graph()
    
    # Separate rooms, interior doors, and front doors
    rooms = []
    interior_doors = []
    front_doors = []
    
    for i, poly_data in enumerate(polygons):
        room_type = poly_data['room_type']
        coords = poly_data['coords']
        
        try:
            poly = ShapelyPolygon(coords)
            if not poly.is_valid:
                poly = make_valid(poly)
        except Exception:
            continue
        
        if room_type == 17:  # interior door
            interior_doors.append({'idx': i, 'room_type': room_type, 'polygon': poly})
        elif room_type == 15:  # front door
            front_doors.append({'idx': i, 'room_type': room_type, 'polygon': poly})
        else:
            rooms.append({'idx': i, 'room_type': room_type, 'polygon': poly})
            G.add_node(i, label=int(room_type))
    
    # Add outside node (matches original script's node -1)
    G.add_node(-1, label=-1)
    
    all_doors = interior_doors + front_doors
    
    # For each door, compute IoU with every room and collect candidates
    doors_rooms_map = defaultdict(list)
    for door in all_doors:
        door_poly = door['polygon']
        for room in rooms:
            room_poly = room['polygon']
            try:
                intersection_area = door_poly.intersection(room_poly).area
                union_area = door_poly.union(room_poly).area
                if union_area > 0:
                    iou = intersection_area / union_area
                    if iou > 0 and iou < 0.2:
                        doors_rooms_map[door['idx']].append((room['idx'], iou))
            except Exception:
                continue
    
    # Connect rooms through doors (matching original top-K selection)
    for door in all_doors:
        k = door['idx']
        if k not in doors_rooms_map:
            continue
        conn = doors_rooms_map[k]
        conn = sorted(conn, key=lambda tup: tup[1], reverse=True)
        
        if door['room_type'] == 17:  # interior door -> connect top-2 rooms
            conn_top2 = conn[:2]
            if len(conn_top2) > 1:
                l1, l2 = conn_top2[0][0], conn_top2[1][0]
                if not G.has_edge(l1, l2):
                    G.add_edge(l1, l2)
        elif door['room_type'] == 15:  # front door -> connect top-1 room to outside
            if len(conn) > 0:
                l1 = conn[0][0]
                if not G.has_edge(-1, l1):
                    G.add_edge(-1, l1)
    
    return G


def corners_to_masks(coords, room_types_onehot, room_indices_onehot, padding_mask, image_size=256):
    """
    Convert House-Diffusion corner-based representation to masks.
    
    Args:
        coords: Coordinates [2, num_points] or [num_points, 2]
        room_types_onehot: Room types [num_points, 25]
        room_indices_onehot: Room indices [num_points, 32]
        padding_mask: Padding mask [num_points] (1 = valid, 0 = padding)
        image_size: Output mask size
        is_binary: Whether coordinates are binary (need conversion)
    
    Returns:
        masks: numpy array of shape [num_rooms, image_size, image_size]
        room_types: numpy array of room type IDs (RPLAN format, 1-indexed)
    """
    masks_list = []
    room_types_list = []
    
    # Ensure coords is [2, num_points]
    if coords.shape[0] != 2:
        coords = coords.T
    
    num_points = coords.shape[1]
    
    # Note: Even when model is trained with binary encoding (analog_bit=False),
    # the diffusion process outputs continuous values in [-1, 1] range.
    # No conversion needed - coords are already continuous.
    
    # Group points by room
    room_polygons = {}
    current_room_idx = None
    current_poly = []
    
    for j in range(num_points):
        # Skip padding
        if padding_mask[j] == 0:
            continue
        
        # Get room index and type
        room_idx = np.argmax(room_indices_onehot[j])
        room_type = np.argmax(room_types_onehot[j])
        room_type = reverse_map_room_type(room_type)
        
        # Convert coordinates from [-1, 1] to [0, image_size]
        x, y = coords[:, j]
        x_pixel = int((x / 2 + 0.5) * image_size)
        y_pixel = int((y / 2 + 0.5) * image_size)
        
        # Clamp to valid range
        x_pixel = max(0, min(image_size - 1, x_pixel))
        y_pixel = max(0, min(image_size - 1, y_pixel))
        
        # Check if we're starting a new room
        if room_idx != current_room_idx:
            # Save previous polygon
            if current_poly and len(current_poly) >= 3:
                if current_room_idx not in room_polygons:
                    room_polygons[current_room_idx] = {'polygons': [], 'room_type': None}
                room_polygons[current_room_idx]['polygons'].append(current_poly)
            
            # Start new polygon
            current_room_idx = room_idx
            current_poly = []
            if current_room_idx not in room_polygons:
                room_polygons[current_room_idx] = {'polygons': [], 'room_type': room_type}
        
        current_poly.append((x_pixel, y_pixel))
    
    # Save last polygon
    if current_poly and len(current_poly) >= 3:
        if current_room_idx not in room_polygons:
            room_polygons[current_room_idx] = {'polygons': [], 'room_type': None}
        room_polygons[current_room_idx]['polygons'].append(current_poly)
    
    # Create masks for each room
    for room_idx in sorted(room_polygons.keys()):
        room_data = room_polygons[room_idx]
        polygons = room_data['polygons']
        room_type = room_data['room_type']
        
        if not polygons or room_type is None:
            continue
        
        # Create mask by drawing all polygons for this room
        mask_img = Image.new('L', (image_size, image_size), 0)
        draw = ImageDraw.Draw(mask_img)
        for poly in polygons:
            if len(poly) >= 3:
                draw.polygon(poly, fill=255)
        
        mask = np.array(mask_img) / 255.0
        masks_list.append(mask)
        room_types_list.append(room_type)
    
    if len(masks_list) == 0:
        return np.zeros((0, image_size, image_size)), np.array([])
    
    return np.array(masks_list), np.array(room_types_list)



def build_access_graph_from_ground_truth(graph_data, room_types):
    """
    Build access graph from ground truth graph data.
    
    Matches the algorithm from House-Diffusion's original get_graph():
    - Room nodes for all non-door types
    - Outside node (-1) connected via front doors (type 15)
    - Interior doors (type 17) connect pairs of rooms
    - Direct room-room edges where both endpoints are rooms
    
    Args:
        graph_data: Ground truth graph triples [N, 3] with [node1, rel, node2]
                   where rel > 0 indicates adjacency
        room_types: Array of room type IDs (RPLAN format after reverse mapping)
    
    Returns:
        NetworkX graph with room nodes, outside node (-1), and door-based connections
    """
    G = nx.Graph()
    
    room_indices = []
    interior_door_indices = []
    front_door_indices = []
    for k, room_type in enumerate(room_types):
        if room_type not in [15, 17]:
            G.add_node(k, label=int(room_type))
            room_indices.append(k)
        elif room_type == 17:
            interior_door_indices.append(k)
        elif room_type == 15:
            front_door_indices.append(k)
    
    # Add outside node (matches original script's node -1)
    G.add_node(-1, label=-1)
    
    room_set = set(room_indices)
    interior_door_set = set(interior_door_indices)
    front_door_set = set(front_door_indices)
    
    # Build door connections maps
    interior_door_connections = {idx: [] for idx in interior_door_indices}
    front_door_connections = {idx: [] for idx in front_door_indices}
    
    for edge in graph_data:
        if len(edge) != 3:
            continue
        node1, rel, node2 = int(edge[0]), edge[1], int(edge[2])
        
        if rel <= 0:
            continue
        
        # Both endpoints are rooms -> direct edge (original script does this too)
        if node1 in room_set and node2 in room_set:
            if not G.has_edge(node1, node2):
                G.add_edge(node1, node2)
            continue
        
        # Interior door connects rooms
        if node1 in interior_door_set and node2 in room_set:
            interior_door_connections[node1].append(node2)
        elif node2 in interior_door_set and node1 in room_set:
            interior_door_connections[node2].append(node1)
        
        # Front door connects room to outside
        if node1 in front_door_set and node2 in room_set:
            front_door_connections[node1].append(node2)
        elif node2 in front_door_set and node1 in room_set:
            front_door_connections[node2].append(node1)
    
    # Connect rooms that share an interior door
    for door_idx, connected_rooms in interior_door_connections.items():
        for i, room1 in enumerate(connected_rooms):
            for room2 in connected_rooms[i+1:]:
                if not G.has_edge(room1, room2):
                    G.add_edge(room1, room2)
    
    # Connect rooms to outside via front doors
    for door_idx, connected_rooms in front_door_connections.items():
        for room_idx in connected_rooms:
            if not G.has_edge(-1, room_idx):
                G.add_edge(-1, room_idx)
    
    return G


def visualize_comparison(
        gt_masks, 
        gt_room_types, 
        pred_masks, 
        pred_room_types,
        G_true, 
        G_pred, 
        ged, 
        sample_idx, 
        output_dir,
        room_count_info=None, 
        similarity_result=None,
        sample_id=None
    ):
    """
    Create 4-panel visualization comparing ground truth and predicted access graphs.
    Matches styling from HouseGAN++ version.
    
    Args:
        room_count_info: Optional dict with 'expected_rooms', 'total_rooms', 'hidden_rooms'
        similarity_result: Optional dict with 'similarity_score', 'matched_pairs', etc.
        sample_id: Optional sample ID string to display in title
    """
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    
    # Top left: Ground truth floorplan
    ax_floor_gt = fig.add_subplot(gs[0, 0])
    if len(gt_masks) > 0:
        gt_img = draw_masks(gt_masks, gt_room_types)
        ax_floor_gt.imshow(gt_img)
        ax_floor_gt.set_title('Ground Truth Floor Plan', fontsize=12, fontweight='bold')
    ax_floor_gt.axis('off')
    
    # Bottom left: Ground truth access graph
    ax_graph_gt = fig.add_subplot(gs[1, 0])
    ax_graph_gt.set_title('GT: Door Connections', fontsize=10)
    if G_true.number_of_nodes() > 0:
        try:
            pos = nx.nx_agraph.graphviz_layout(G_true, prog='neato')
        except:
            pos = nx.spring_layout(G_true, seed=42, k=1.5)
        
        node_colors = [ID_COLOR.get(G_true.nodes[n]['label'], '#BEBEBE') for n in G_true.nodes()]
        node_sizes = [750 if n == -1 else 1000 for n in G_true.nodes()]
        node_edgecolors = ['black' if n == -1 else 'blue' for n in G_true.nodes()]
        node_linewidths = [3.0 if n == -1 else 0.0 for n in G_true.nodes()]
        edge_colors = ['#727171' if -1 in (u, v) else '#b8b5b4' for u, v in G_true.edges()]
        nx.draw(G_true, pos, node_color=node_colors, node_size=node_sizes,
               edgecolors=node_edgecolors, linewidths=node_linewidths, edge_color=edge_colors,
               style='dotted', width=4.0, with_labels=False, ax=ax_graph_gt)
    ax_graph_gt.axis('off')
    
    # Top right: Predicted floorplan
    ax_floor_pred = fig.add_subplot(gs[0, 1])
    if len(pred_masks) > 0:
        pred_img = draw_masks(pred_masks, pred_room_types)
        ax_floor_pred.imshow(pred_img)
        ax_floor_pred.set_title('Predicted Floor Plan (Generated)', fontsize=12, fontweight='bold')
    ax_floor_pred.axis('off')
    
    # Bottom right: Predicted access graph
    ax_graph_pred = fig.add_subplot(gs[1, 1])
    ax_graph_pred.set_title('Predicted: Door Connections', fontsize=10)
    if G_pred.number_of_nodes() > 0:
        try:
            pos = nx.nx_agraph.graphviz_layout(G_pred, prog='neato')
        except:
            pos = nx.spring_layout(G_pred, seed=42, k=1.5)
        
        node_colors = [ID_COLOR.get(G_pred.nodes[n]['label'], '#BEBEBE') for n in G_pred.nodes()]
        node_sizes = [750 if n == -1 else 1000 for n in G_pred.nodes()]
        node_edgecolors = ['black' if n == -1 else 'blue' for n in G_pred.nodes()]
        node_linewidths = [3.0 if n == -1 else 0.0 for n in G_pred.nodes()]
        edge_colors = ['#727171' if -1 in (u, v) else '#b8b5b4' for u, v in G_pred.edges()]
        nx.draw(G_pred, pos, node_color=node_colors, node_size=node_sizes,
               edgecolors=node_edgecolors, linewidths=node_linewidths, edge_color=edge_colors,
               style='dotted', width=4.0, with_labels=False, ax=ax_graph_pred)
    ax_graph_pred.axis('off')
    
    # Add room count info and similarity score to title
    room_info_text = ''
    if ged is not None:
        room_info_text += f'GED={ged:.1f}'
    
    if room_count_info is not None:
        room_info_text += f" | Expected: {room_count_info['expected_rooms']} rooms | Generated: {room_count_info['total_rooms']} rooms"
        if len(room_count_info['hidden_rooms']) > 0:
            room_info_text += f" (⚠ {len(room_count_info['hidden_rooms'])} hidden)"
    
    # Add similarity score if available
    if similarity_result is not None and not np.isnan(similarity_result['similarity_score']):
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


def save_generations_housediffusion(filepath, metadata, samples):
    """
    Save generated outputs for later analysis (House-Diffusion).
    
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


def load_generations_housediffusion(filepath):
    """
    Load previously generated outputs (House-Diffusion).
    
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
        description="Compute GED for access graphs (House-Diffusion)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Data parameters
    parser.add_argument("--data_path", type=str, required=True,
                       help="path to eval NPZ file (e.g., /path/to/processed_rplan/rplan_eval_all.npz)")
    
    # Model parameters
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="path to trained diffusion model checkpoint"
                            " (required unless --load_generations is used)")
    
    # Evaluation parameters
    parser.add_argument("--batch_size", type=int, default=64,
                       help="size of the batches (64 matches House-Diffusion's default)")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="number of samples to evaluate (None = auto-detect from NPZ file)")
    parser.add_argument("--num_variations", type=int, default=1,
                       help="number of generation variations per sample")
    parser.add_argument("--save_graphs", action='store_true',
                       help="save visualizations of graphs")
    parser.add_argument("--out", type=str, default="./ged_results_diffusion",
                       help="output folder for results")
    
    # Generation caching
    parser.add_argument("--save_generations", type=str, default=None,
                       help="path to save generated outputs for later analysis (enables caching)")
    parser.add_argument("--load_generations", type=str, default=None,
                       help="path to load previously generated outputs (skips model inference)")
    
    # Add House-Diffusion model defaults
    defaults = model_and_diffusion_defaults()
    defaults.update({
        'dataset': 'rplan',
        'clip_denoised': True,
        'use_ddim': False,
        'analog_bit': False,  # Pretrained model uses binary (False), not continuous (True)
        'set_name': 'eval',
        'target_set': None,
    })
    add_dict_to_argparser(parser, defaults)
    
    return parser.parse_args()


def postprocess_sample(sample_data, results, save_graphs, output_dir, sample_ids_array=None):
    """
    Post-process a sample from House-Diffusion.
    
    Args:
        sample_data: Dict with sample data
        sample_ids_array: Optional numpy array of sample IDs from NPZ file
    
    Returns:
        Dict with post-processed sample data
    """

    # Upper bound for GED computation
    ged_upper_bound = 40

    # Get sample index and ID
    sample_idx = sample_data['sample_idx']
    sample_id = None
    if sample_ids_array is not None and sample_idx < len(sample_ids_array):
        sample_id = sample_ids_array[sample_idx]
            
    # Convert to polygons (for access graph) and masks (for visualization)
    gt_polygons = corners_to_polygons_and_types(
        sample_data['gt_coords'],
        sample_data['gt_room_types_oh'],
        sample_data['gt_room_indices_oh'],
        sample_data['gt_padding_mask'],
        image_size=256
    )

    gt_masks, gt_room_types = corners_to_masks(
        sample_data['gt_coords'],
        sample_data['gt_room_types_oh'],
        sample_data['gt_room_indices_oh'],
        sample_data['gt_padding_mask'],
        image_size=256
    )

    # Build ground truth access graph from polygons
    G_true = build_access_graph_from_polygons(gt_polygons)

    # Clear variation_geds
    results['variation_geds'] = []

    # Use syn conditioning for predictions (the model generates using syn_ keys
    # which have different corner counts than GT). Fall back to GT if syn not available.
    syn_room_types_oh = sample_data.get('syn_room_types_oh', sample_data['gt_room_types_oh'])
    syn_room_indices_oh = sample_data.get('syn_room_indices_oh', sample_data['gt_room_indices_oh'])
    syn_padding_mask = sample_data.get('syn_padding_mask', sample_data['gt_padding_mask'])

    # Process each variation
    for var_data in sample_data['variations']:
        var_idx = var_data['variation_idx']
        
        # Convert to polygons (for access graph) and masks (for visualization)
        # Predictions must use syn_ conditioning (matches the structure the model generated)
        pred_polygons = corners_to_polygons_and_types(
            var_data['pred_coords'],
            syn_room_types_oh,
            syn_room_indices_oh,
            syn_padding_mask,
            image_size=256
        )
        
        pred_masks, pred_room_types = corners_to_masks(
            var_data['pred_coords'],
            syn_room_types_oh,
            syn_room_indices_oh,
            syn_padding_mask,
            image_size=256
        )

        # Build predicted access graph directly from polygons (more accurate!)
        G_pred = build_access_graph_from_polygons(pred_polygons)
        
        # Count actual rooms in generated polygons (accounting for hidden/overlapped regions)
        room_count_info = count_actual_rooms_from_polygons(pred_polygons)
        
        # Track hidden rooms
        if len(room_count_info['hidden_rooms']) > 0:
            results['hidden_room_count'] += 1
        
        # Compute room count and type accuracy
        room_count_metric = room_type_accuracy(
            room_count_info['expected_room_types'],
            room_count_info['actual_room_types']
        )
        results['room_count_accuracies'].append(room_count_metric['accuracy'])
        results['room_count_errors'].append(room_count_metric['count_error'])
        results['room_type_errors'].append(room_count_metric['type_error'])
        
        # Compute room similarity score
        similarity_result = compute_room_similarity_score_polygons(
            pred_polygons, gt_polygons
        )
        results['similarity_scores'].append(similarity_result['similarity_score'])
        results['unmatched_pred_rooms'].append(similarity_result['unmatched_pred'])
        results['unmatched_gt_rooms'].append(similarity_result['unmatched_gt'])
        results['matched_room_pairs'].append(similarity_result['matched_pairs'])
        
        # Compute room area difference
        area_diff_result = compute_room_area_difference_score_polygons(
            pred_polygons, gt_polygons
        )
        results['area_difference_scores'].append(area_diff_result['area_difference_score'])
        
        # Compute GED
        ged, graph_len = ged_between_graphs(
            G_pred, G_true, sample_idx,
            timeout=10000, upper_bound=ged_upper_bound
        )
        
        if ged is not None:
            results['variation_geds'].append(ged)
        else:
            results['variation_geds'].append(np.nan)

    
        # Save visualization for first variation
        if var_idx == 0 and save_graphs:
            visualize_comparison(
                gt_masks, gt_room_types,
                pred_masks, pred_room_types,
                G_true, G_pred, ged, sample_idx, output_dir,
                room_count_info=room_count_info,
                similarity_result=similarity_result,
                sample_id=sample_id
            )
        
    return results

def load_and_process_cached_samples(cached_data_path, num_variations, save_graphs, output_dir, sample_ids_array=None):
    """
    Load cached samples from House-Diffusion.
    
    Args:
        cached_data_path: Path to cached data
        num_variations: Number of variations per sample
        save_graphs: Whether to save graphs
        output_dir: Output directory
        sample_ids_array: Optional numpy array of sample IDs from NPZ file
    """

    cached_data = load_generations_housediffusion(cached_data_path)
    
    print(f"\nProcessing {len(cached_data['samples'])} cached samples...")
    print(f"Generating {num_variations} variation(s) per sample\n")
    
    pbar = tqdm(total=len(cached_data['samples']), desc="Processing cached")

    # Storage for results
    results = {
        'variation_geds': [],
        'all_geds': [],
        'room_count_accuracies': [],
        'room_count_errors': [],
        'room_type_errors': [],
        'similarity_scores': [],
        'area_difference_scores': [],
        'unmatched_pred_rooms': [],
        'unmatched_gt_rooms': [],
        'matched_room_pairs': [],
        'hidden_room_count': 0,
    }
    sample_count = 0

    try:
        for sample_data in cached_data['samples']:
            # Post-process each sample
            results = postprocess_sample(sample_data, results, save_graphs, output_dir, sample_ids_array)           
            sample_count += 1
                
            # Record results (always extend, even with NaNs)
            # This ensures we always have num_variations entries per sample
            results['all_geds'].extend(results['variation_geds'])
                
            pbar.update(1)
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    finally:
        pbar.close()
    return results

def _build_additional_info(results):
    """Package raw per-sample arrays for ``score_benchmark_result``."""
    return {
        'room_count_accuracy': {
            'accuracies': results['room_count_accuracies'],
            'count_errors': results['room_count_errors'],
            'type_errors': results['room_type_errors'],
            'hidden_count': results['hidden_room_count'],
            'total_samples': len(results['room_count_accuracies']),
        },
        'room_similarity': {
            'scores': results['similarity_scores'],
            'matched_room_pairs': results['matched_room_pairs'],
            'unmatched_pred_rooms': results['unmatched_pred_rooms'],
            'unmatched_gt_rooms': results['unmatched_gt_rooms'],
        },
        'area_difference': {
            'scores': results['area_difference_scores'],
        },
    }
    

def main():
    args = parse_args()
    
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
    
    # Determine mode
    load_mode = args.load_generations is not None
    save_mode = args.save_generations is not None
    
    if load_mode:
        print("\n🔄 MODE: Load cached generations")
    elif save_mode:
        print("\n💾 MODE: Generate and save")
    else:
        print("\n🚀 MODE: Generate only")
    
    # Update arguments for model configuration (sets num_channels=512 and other params)
    update_arg_parser(args)
    
    # Resolve paths to absolute paths
    args.data_path = os.path.abspath(args.data_path)
    if args.checkpoint:
        args.checkpoint = os.path.abspath(args.checkpoint)
    args.out = os.path.abspath(args.out)
    if args.save_generations:
        args.save_generations = os.path.abspath(args.save_generations)
    if args.load_generations:
        args.load_generations = os.path.abspath(args.load_generations)
    
    # Extract data directory for dataset loading
    # RPlanhgDataset expects to be run from house_diffusion root (parent of processed_rplan)
    data_dir = os.path.dirname(args.data_path)  # e.g., /path/to/processed_rplan
    house_diffusion_root = os.path.dirname(data_dir)  # e.g., /path/to/house_diffusion
    
    # Verify files exist
    if not os.path.exists(args.data_path):
        print(f"❌ Error: Data file not found: {args.data_path}")
        return 1
    
    if not load_mode and not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint not found: {args.checkpoint}")
        return 1
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    
    print("=" * 80)
    print("House-Diffusion Access Graph GED Evaluation")
    print("=" * 80)
    print(f"Data path: {args.data_path}")
    print(f"Data directory: {data_dir}")
    print(f"Working directory: {house_diffusion_root}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num variations: {args.num_variations}")
    print(f"Output: {args.out}")
    print("=" * 80)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and diffusion (only in generate mode)
    if not load_mode:
        print("\n📦 Loading model and diffusion...")
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        
        # Load checkpoint
        if os.path.exists(args.checkpoint):
            print(f"Loading checkpoint: {args.checkpoint}")
            state_dict = torch.load(args.checkpoint, map_location='cpu')
            model.load_state_dict(state_dict)
        else:
            print(f"⚠️  Warning: Checkpoint not found at {args.checkpoint}")
            print("Using randomly initialized model (for testing only)")
        
        model.to(device)
        model.eval()
    else:
        model = None
        diffusion = None
    
    # Load dataset
    print("\n📊 Loading dataset...")
    
    # Change to house_diffusion root and stay there for the entire data loading process
    # (RPlanhgDataset expects to be run from house_diffusion root, looks for processed_rplan/)
    original_cwd = os.getcwd()
    os.chdir(house_diffusion_root)
    
    data_loader = load_rplanhg_data(
        batch_size=args.batch_size,
        analog_bit=args.analog_bit,
        set_name=args.set_name,
        target_set=args.target_set,
    )
    
    # Load sample IDs from NPZ file (always attempt this)
    sample_ids_array = None  # Will store IDs if available in NPZ
    npz_path = Path(args.data_path)
    print(f"\n📋 Attempting to load sample IDs from: {npz_path}")
    if npz_path.exists():
        try:
            npz_data = np.load(npz_path, allow_pickle=True)
            print(f"   NPZ file loaded, keys: {list(npz_data.keys())}")
            # Check if IDs array exists
            if 'ids' in npz_data:
                sample_ids_array = npz_data['ids']
                print(f"✅ Loaded {len(sample_ids_array)} sample IDs from NPZ file")
                print(f"   First 3 IDs: {list(sample_ids_array[:min(3, len(sample_ids_array))])}")
            else:
                print(f"⚠️  Warning: 'ids' array not found in NPZ. Titles will show indices only.")
                print(f"   Available keys: {list(npz_data.keys())}")
                print(f"   To add IDs, regenerate NPZ with updated convert_pickle_to_npz.py")
        except Exception as e:
            print(f"⚠️  Warning: Could not load sample IDs from NPZ: {e}")
    else:
        print(f"❌ NPZ file not found at: {npz_path}")
    
    # Determine number of samples to process
    # If not specified, auto-detect from NPZ file to prevent infinite cycling
    if args.num_samples is not None:
        total_samples = args.num_samples
    else:
        # Auto-detect dataset size from NPZ file
        if npz_path.exists():
            try:
                npz_data = np.load(npz_path, allow_pickle=True)
                # NPZ files have 'houses' key with shape (num_samples, num_points, features)
                if 'houses' in npz_data:
                    total_samples = npz_data['houses'].shape[0]
                    print(f"📊 Auto-detected {total_samples} samples from NPZ file")
                else:
                    print(f"⚠️  Warning: 'houses' key not found in NPZ, using infinite loop")
                    print(f"   Available keys: {list(npz_data.keys())}")
                    total_samples = float('inf')
            except Exception as e:
                print(f"⚠️  Warning: Could not read NPZ file to detect size: {e}")
                print(f"   Using infinite loop (press Ctrl+C to stop)")
                total_samples = float('inf')
        else:
            print(f"⚠️  Warning: NPZ file not found at {npz_path}")
            print(f"   Using infinite loop (press Ctrl+C to stop)")
            total_samples = float('inf')
    

    # ========================================================================
    # LOAD MODE: Process cached generations
    # ========================================================================
    if load_mode:
        results = load_and_process_cached_samples(args.load_generations, args.num_variations, args.save_graphs, args.out, sample_ids_array)

    # ========================================================================
    # GENERATE MODE: Run model and optionally save
    # ========================================================================
    else:
        if total_samples == float('inf'):
            print(f"\n🚀 Starting evaluation (infinite loop mode - press Ctrl+C to stop)...")
        else:
            print(f"\n🚀 Starting evaluation (processing {total_samples} samples)...")
        print(f"Generating {args.num_variations} variation(s) per sample\n")
        
        # Sample generation function
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim 
            else diffusion.ddim_sample_loop
        )
        
        # Storage for caching
        if save_mode:
            cache_samples = []
        
        # Initialize results storage (same structure as load_and_process_cached_samples)
        results = {
            'variation_geds': [],
            'all_geds': [],
            'room_count_accuracies': [],
            'room_count_errors': [],
            'room_type_errors': [],
            'similarity_scores': [],
            'area_difference_scores': [],
            'unmatched_pred_rooms': [],
            'unmatched_gt_rooms': [],
            'matched_room_pairs': [],
            'hidden_room_count': 0,
        }
        
        # Process samples
        pbar = tqdm(total=total_samples if total_samples != float('inf') else None)
        sample_count = 0
        try:
            while sample_count < total_samples:
                # Get ground truth sample and metadata
                try:
                    data_sample, model_kwargs = next(data_loader)
                except StopIteration:
                    print("\n✅ Reached end of dataset")
                    break
                
                # Move to device
                data_sample = data_sample.to(device)
                for key in model_kwargs:
                    model_kwargs[key] = model_kwargs[key].to(device)
                
                batch_size_actual = data_sample.shape[0]

                # Generate all variations for the full batch at once
                batch_pred_coords = []  # [num_variations][batch_size_actual] of [2, num_points]
                for var_idx in range(args.num_variations):
                    with torch.no_grad():
                        pred_sample = sample_fn(
                            model,
                            data_sample.shape,
                            clip_denoised=args.clip_denoised,
                            model_kwargs=model_kwargs,
                            analog_bit=args.analog_bit,
                        )
                    # pred_sample shape: [timesteps, batch, 2, num_points]
                    batch_pred_coords.append(pred_sample[-1].cpu().numpy())  # [batch, 2, num_points]
                    del pred_sample
                    torch.cuda.empty_cache()

                # Post-process each sample in the batch
                for batch_idx in range(batch_size_actual):
                    if sample_count >= total_samples:
                        break

                    gt_coords = data_sample[batch_idx].cpu().numpy()  # [2, num_points]
                    gt_room_types_oh = model_kwargs['room_types'][batch_idx].cpu().numpy()
                    gt_room_indices_oh = model_kwargs['room_indices'][batch_idx].cpu().numpy()
                    gt_padding_mask = (1 - model_kwargs['src_key_padding_mask'][batch_idx].cpu().numpy()).astype(int)

                    syn_room_types_oh = model_kwargs['syn_room_types'][batch_idx].cpu().numpy()
                    syn_room_indices_oh = model_kwargs['syn_room_indices'][batch_idx].cpu().numpy()
                    syn_padding_mask = (1 - model_kwargs['syn_src_key_padding_mask'][batch_idx].cpu().numpy()).astype(int)

                    sample_variations = []
                    for var_idx in range(args.num_variations):
                        sample_variations.append({
                            'variation_idx': var_idx,
                            'pred_coords': batch_pred_coords[var_idx][batch_idx].copy(),
                        })

                    sample_data = {
                        'sample_idx': sample_count,
                        'variations': sample_variations,
                        'gt_coords': gt_coords,
                        'gt_room_types_oh': gt_room_types_oh,
                        'gt_room_indices_oh': gt_room_indices_oh,
                        'gt_padding_mask': gt_padding_mask,
                        'syn_room_types_oh': syn_room_types_oh,
                        'syn_room_indices_oh': syn_room_indices_oh,
                        'syn_padding_mask': syn_padding_mask,
                    }

                    results = postprocess_sample(sample_data, results, args.save_graphs, args.out, sample_ids_array)
                    results['all_geds'].extend(results['variation_geds'])

                    if save_mode:
                        cache_samples.append({
                            'sample_idx': sample_count,
                            'gt_coords': gt_coords.copy(),
                            'gt_room_types_oh': gt_room_types_oh.copy(),
                            'gt_room_indices_oh': gt_room_indices_oh.copy(),
                            'gt_padding_mask': gt_padding_mask.copy(),
                            'syn_room_types_oh': syn_room_types_oh.copy(),
                            'syn_room_indices_oh': syn_room_indices_oh.copy(),
                            'syn_padding_mask': syn_padding_mask.copy(),
                            'variations': sample_variations
                        })

                    sample_count += 1
                    pbar.update(1)
    
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        finally:
            pbar.close()
            # Restore original directory
            os.chdir(original_cwd)
        
        # After all samples, save cache if enabled
        if save_mode:
            metadata = {
                'model': 'House-Diffusion',
                'checkpoint': args.checkpoint,
                'data_path': args.data_path,
                'target_set': args.target_set,
                'num_variations': args.num_variations,
                'num_samples': len(cache_samples),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'version': '1.0'
            }
            save_generations_housediffusion(args.save_generations, metadata, cache_samples)
    
    score_benchmark_result(
        all_geds=np.array(results['all_geds']),
        num_variations=args.num_variations,
        output_dir=args.out,
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        additional_info=_build_additional_info(results),
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
