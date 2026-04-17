"""Benchmark Dataset loader for HouseGAN++."""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw

# Room type IDs are RPLAN IDs (mapped in converter)
# 1: living_room, 2: kitchen, 3: bedroom, 4: bathroom, 6: entrance, 10: storage, 17: interior_door
# Use 19 dimensions to match RPLAN (room types 1-17, then slice [:, 1:] → 18-dim)
NUM_ROOM_TYPES = 19


class BenchmarkDataset(Dataset):
    """Dataset loader for Benchmark Dataset in HouseGAN++ format."""
    
    def __init__(self, pickle_path, target_set=None, split='train'):
        """
        Args:
            pickle_path: Path to pickle file containing converted data
            target_set: Filter by number of rooms (None = no filter)
            split: 'train', 'val', or 'test'
        """
        super().__init__()
        self.split = split
        self.target_set = target_set
        
        # Load pickle data
        print(f"Loading data from: {pickle_path}")
        with open(pickle_path, 'rb') as f:
            self.data = pickle.load(f)
        
        print(f"Loaded {len(self.data)} samples")
        
        # Filter by target_set if specified
        if target_set is not None:
            filtered_data = []
            for sample in self.data:
                num_rooms = sum(1 for rt in sample['room_types'] if rt != 17)
                if split == 'train':
                    # For training: exclude target_set (train on all other sizes)
                    if num_rooms != target_set:
                        filtered_data.append(sample)
                else:
                    # For val/test: include only target_set
                    if num_rooms == target_set:
                        filtered_data.append(sample)
            self.data = filtered_data
            print(f"After filtering (target_set={target_set}): {len(self.data)} samples")
        
        self.augment = (split == 'train')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            room_masks: Tensor of room masks [num_rooms, 64, 64] (including doors)
            room_types_onehot: One-hot encoded room types [num_rooms, 18]
            edges: Adjacency triples [node1, relationship, node2]
        """
        sample = self.data[idx]
        
        # Extract data
        room_types = np.array(sample['room_types'])
        polygons = sample['polygons']
        triples = sample['triples']
        
        # Build room masks from polygons (including doors)
        room_masks = self._build_masks_from_polygons(polygons, room_types)
        
        # Build edges from triples
        edges = self._build_edges_from_triples(triples, room_types)
        
        # One-hot encode room types (standard RPLAN encoding)
        # Room types are already RPLAN IDs (1, 2, 3, 4, 6, 10, 17) from converter
        # Create 19-dim one-hot, then slice [:, 1:] to get 18-dim (matches RPLAN dataset)
        room_types_onehot = np.zeros((len(room_types), NUM_ROOM_TYPES), dtype=np.float32)
        for i, rt in enumerate(room_types):
            if 0 <= rt < NUM_ROOM_TYPES:
                room_types_onehot[i, rt] = 1.0
        
        # Match RPLAN's [:, 1:] slice to remove column 0
        room_types_onehot = room_types_onehot[:, 1:]  # 19-dim → 18-dim
        
        return torch.FloatTensor(room_masks), torch.FloatTensor(room_types_onehot), edges
    
    def _build_masks_from_polygons(self, polygons, room_types, out_size=64):
        """Convert polygons to binary masks."""
        num_rooms = len(polygons)
        masks = np.zeros((num_rooms, out_size, out_size), dtype=np.float32)
        
        for i, polygon in enumerate(polygons):
            # Create mask
            img = Image.new('L', (out_size, out_size), 0)
            draw = ImageDraw.Draw(img)
            
            # Convert polygon to tuple format
            poly_tuples = [(int(x), int(y)) for x, y in polygon]
            
            # Draw filled polygon
            if len(poly_tuples) >= 3:
                draw.polygon(poly_tuples, fill=255)
            
            # Convert to numpy and normalize to [-1, 1] (same as original HouseGAN++)
            mask = np.array(img, dtype=np.float32) / 255.0  # [0, 1]
            mask = (mask - 0.5) / 0.5  # Normalize to [-1, 1]
            masks[i] = mask
        
        return masks
    
    def _build_edges_from_triples(self, triples, room_types):
        """
        Build edge list from triples.
        
        Triples format: [room_idx_1, relationship, room_idx_2]
        where relationship: 1 = adjacent, -1 = not adjacent
        
        The original HouseGAN++ includes ALL nodes (rooms + doors) and ALL relationships.
        """
        edges = []
        for triple in triples:
            idx1, rel, idx2 = triple
            # Include ALL triples, both adjacent (1) and non-adjacent (-1)
            edges.append([idx1, rel, idx2])
        
        return torch.LongTensor(edges) if len(edges) > 0 else torch.LongTensor([[0, 0, 0]])


def collate_fn(batch):
    """
    Collate function for batching samples with variable numbers of rooms.
    
    Returns:
        all_masks: [total_rooms, 64, 64] - concatenated room masks (including doors)
        all_nodes: [total_rooms, 18] - concatenated one-hot room types (including doors)
                   Uses RPLAN IDs: 1,2,3,4,6,10,17 (positions 5,7-9,11-16 unused)
        all_edges: [total_edges, 3] - concatenated triples [idx1, relationship, idx2]
                   where relationship: 1 = adjacent, -1 = not adjacent
        nd_to_sample: [total_rooms] - which sample each node belongs to
        ed_to_sample: [total_edges] - which sample each edge belongs to
    """
    all_masks, all_nodes, all_edges = [], [], []
    nd_to_sample, ed_to_sample = [], []
    
    for sample_id, (masks, nodes, edges) in enumerate(batch):
        all_masks.append(masks)
        all_nodes.append(nodes)
        all_edges.append(edges)
        
        # Track which sample each node/edge belongs to
        nd_to_sample += [sample_id] * len(nodes)
        ed_to_sample += [sample_id] * len(edges)
    
    # Concatenate all
    all_masks = torch.cat(all_masks, dim=0)
    all_nodes = torch.cat(all_nodes, dim=0)
    
    # Handle edges carefully - adjust indices for concatenation
    # Edges are triples: [idx1, relationship, idx2]
    offset = 0
    shifted_edges = []
    for sample_id, edges in enumerate(all_edges):
        # Adjust edge indices to account for concatenation
        num_nodes = batch[sample_id][1].shape[0]
        if edges.shape[0] > 0 and edges.shape[1] == 3:
            # Only shift the first and third columns (node indices), not the relationship
            shifted = edges.clone()
            shifted[:, 0] += offset  # First node index
            shifted[:, 2] += offset  # Second node index
            shifted_edges.append(shifted)
        offset += num_nodes
    
    if len(shifted_edges) > 0:
        all_edges = torch.cat(shifted_edges, dim=0)
    else:
        all_edges = torch.LongTensor([[0, 1, 0]])
    
    nd_to_sample = torch.LongTensor(nd_to_sample)
    ed_to_sample = torch.LongTensor(ed_to_sample)
    
    return all_masks, all_nodes, all_edges, nd_to_sample, ed_to_sample
