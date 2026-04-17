#!/usr/bin/env python3
"""
Analyze GED and Accuracy metrics binned by room count.

This script processes results from evaluate_houseganpp.py or 
evaluate_housediffusion.py and groups metrics by the 
number of rooms in each apartment.

Room count bins: <=4, 5, 6, 7, 8, >8
"""

import argparse
import json
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def count_rooms_from_room_types(room_types: List[int]) -> int:
    """Count rooms excluding doors (types 11, 12, 13, 15, 17)"""
    door_types = {11, 12, 13, 15, 17}
    return sum(1 for rt in room_types if rt not in door_types)


def get_room_count_bin(num_rooms: int) -> str:
    """Assign room count to appropriate bin"""
    if num_rooms <= 4:
        return "<=4"
    elif num_rooms == 5:
        return "5"
    elif num_rooms == 6:
        return "6"
    elif num_rooms == 7:
        return "7"
    elif num_rooms == 8:
        return "8"
    else:
        return ">8"


def load_houseganpp_data(cache_pickle: Path, results_npz: Path) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Load HouseGAN++ cached generations and NPZ results"""
    print(f"Loading HouseGAN++ data...")
    
    # Load cached generations (contains ground truth)
    with open(cache_pickle, 'rb') as f:
        cache_data = pickle.load(f)
    
    # Handle both old and new cache formats
    if 'ground_truth' in cache_data:
        gt_data = cache_data['ground_truth']
    elif 'samples' in cache_data:
        # New format from save_generations_houseganpp
        gt_data = cache_data['samples']
    else:
        raise ValueError(f"Unknown cache format. Keys: {cache_data.keys()}")
    
    print(f"  ✓ Loaded {len(gt_data)} ground truth samples")
    
    # Load NPZ results
    npz_data = np.load(results_npz, allow_pickle=True)
    all_geds = npz_data['geds']
    
    # Extract room count accuracy data
    room_count_dict = npz_data['room_count_accuracy'].item()
    all_accuracies = np.array(room_count_dict['all_accuracies'])
    
    print(f"  ✓ Loaded {len(all_geds)} GED values")
    print(f"  ✓ Loaded {len(all_accuracies)} accuracy values")
    
    return all_geds, all_accuracies, gt_data


def load_housediffusion_data(cache_pickle: Path, results_npz: Path) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Load House-Diffusion cached generations and NPZ results"""
    print(f"Loading House-Diffusion data...")
    
    # Load cached generations (contains ground truth)
    with open(cache_pickle, 'rb') as f:
        cache_data = pickle.load(f)
    
    # Handle both old and new cache formats
    if 'ground_truth' in cache_data:
        gt_data = cache_data['ground_truth']
    elif 'samples' in cache_data:
        # New format from save_generations
        gt_data = cache_data['samples']
    else:
        raise ValueError(f"Unknown cache format. Keys: {cache_data.keys()}")
    
    print(f"  ✓ Loaded {len(gt_data)} ground truth samples")
    
    # Load NPZ results
    npz_data = np.load(results_npz, allow_pickle=True)
    all_geds = npz_data['geds']
    
    # Extract room count accuracy data
    room_count_dict = npz_data['room_count_accuracy'].item()
    all_accuracies = np.array(room_count_dict['all_accuracies'])
    
    print(f"  ✓ Loaded {len(all_geds)} GED values")
    print(f"  ✓ Loaded {len(all_accuracies)} accuracy values")
    
    return all_geds, all_accuracies, gt_data


def analyze_houseganpp_by_room_count(all_geds: np.ndarray, all_accuracies: np.ndarray, 
                                     gt_data: List[Dict], num_variations: int) -> Dict[str, Dict]:
    """Analyze HouseGAN++ metrics by room count bins"""
    
    # Group by room count
    bins = defaultdict(lambda: {'geds': [], 'accuracies': []})
    
    # Iterate through samples
    num_samples = len(gt_data)
    for sample_idx in range(num_samples):
        gt_sample = gt_data[sample_idx]
        
        # Count rooms in ground truth
        # Handle both old and new cache formats
        if 'raw_sample' in gt_sample:
            room_types = gt_sample['raw_sample'].get('room_types', [])
        else:
            room_types = gt_sample.get('room_types', [])
        
        num_rooms = count_rooms_from_room_types(room_types)
        
        # Get bin
        bin_label = get_room_count_bin(num_rooms)
        
        # Get all variations for this sample
        start_idx = sample_idx * num_variations
        end_idx = start_idx + num_variations
        
        # Add GEDs for all variations of this sample
        sample_geds = all_geds[start_idx:end_idx]
        bins[bin_label]['geds'].extend(sample_geds.tolist())
        
        # Add accuracies for all variations of this sample
        sample_accs = all_accuracies[start_idx:end_idx]
        bins[bin_label]['accuracies'].extend(sample_accs.tolist())
    
    # Compute statistics for each bin
    bin_stats = {}
    for bin_label in ["<=4", "5", "6", "7", "8", ">8"]:
        if bin_label in bins:
            geds = np.array(bins[bin_label]['geds'])
            accs = np.array(bins[bin_label]['accuracies'])
            
            # Filter out NaN values for GED statistics
            valid_geds = geds[~np.isnan(geds)]
            
            bin_stats[bin_label] = {
                'num_samples': len(geds),
                'num_valid_geds': len(valid_geds),
                'ged': {
                    'mean': float(np.mean(valid_geds)) if len(valid_geds) > 0 else None,
                    'std': float(np.std(valid_geds, ddof=1)) if len(valid_geds) > 1 else None,
                    'median': float(np.median(valid_geds)) if len(valid_geds) > 0 else None,
                    'min': float(np.min(valid_geds)) if len(valid_geds) > 0 else None,
                    'max': float(np.max(valid_geds)) if len(valid_geds) > 0 else None,
                },
                'accuracy': {
                    'mean': float(np.mean(accs)) if len(accs) > 0 else None,
                    'std': float(np.std(accs, ddof=1)) if len(accs) > 1 else None,
                    'median': float(np.median(accs)) if len(accs) > 0 else None,
                    'min': float(np.min(accs)) if len(accs) > 0 else None,
                    'max': float(np.max(accs)) if len(accs) > 0 else None,
                }
            }
        else:
            bin_stats[bin_label] = {
                'num_samples': 0,
                'num_valid_geds': 0,
                'ged': {'mean': None, 'std': None, 'median': None, 'min': None, 'max': None},
                'accuracy': {'mean': None, 'std': None, 'median': None, 'min': None, 'max': None}
            }
    
    return bin_stats


def analyze_housediffusion_by_room_count(all_geds: np.ndarray, all_accuracies: np.ndarray,
                                        gt_data: List[Dict], num_variations: int) -> Dict[str, Dict]:
    """Analyze House-Diffusion metrics by room count bins"""
    
    # Group by room count
    bins = defaultdict(lambda: {'geds': [], 'accuracies': []})
    
    # Iterate through samples
    num_samples = len(gt_data)
    for sample_idx in range(num_samples):
        gt_sample = gt_data[sample_idx]
        
        # Count rooms in ground truth (House-Diffusion stores one-hot encoded room types)
        # Decode from one-hot encoding
        if 'gt_room_types_oh' in gt_sample:
            room_types_oh = gt_sample['gt_room_types_oh']
            padding_mask = gt_sample['gt_padding_mask']
            
            # Decode room types from one-hot (argmax)
            room_types_full = np.argmax(room_types_oh, axis=1)
            
            # Filter to valid (non-padded) entries (padding_mask == 1 means valid)
            valid_mask = padding_mask == 1
            room_types_corners = room_types_full[valid_mask]
            
            # House-Diffusion stores CORNERS (4 per room with same type)
            # Count unique consecutive room sequences
            room_types = []
            if len(room_types_corners) > 0:
                prev_type = room_types_corners[0]
                room_types.append(prev_type)
                for rt in room_types_corners[1:]:
                    if rt != prev_type:
                        room_types.append(rt)
                        prev_type = rt
        elif 'raw_sample' in gt_sample:
            # Fallback for old format
            room_types = gt_sample['raw_sample'].get('room_types', [])
        else:
            # Another fallback
            room_types = gt_sample.get('room_types', [])
        
        num_rooms = count_rooms_from_room_types(room_types)
        
        # Get bin
        bin_label = get_room_count_bin(num_rooms)
        
        # Get all variations for this sample
        start_idx = sample_idx * num_variations
        end_idx = start_idx + num_variations
        
        # Add GEDs for all variations of this sample
        sample_geds = all_geds[start_idx:end_idx]
        bins[bin_label]['geds'].extend(sample_geds.tolist())
        
        # Add accuracies for all variations of this sample
        sample_accs = all_accuracies[start_idx:end_idx]
        bins[bin_label]['accuracies'].extend(sample_accs.tolist())
    
    # Compute statistics for each bin
    bin_stats = {}
    for bin_label in ["<=4", "5", "6", "7", "8", ">8"]:
        if bin_label in bins:
            geds = np.array(bins[bin_label]['geds'])
            accs = np.array(bins[bin_label]['accuracies'])
            
            # Filter out NaN values for GED statistics
            valid_geds = geds[~np.isnan(geds)]
            
            bin_stats[bin_label] = {
                'num_samples': len(geds),
                'num_valid_geds': len(valid_geds),
                'ged': {
                    'mean': float(np.mean(valid_geds)) if len(valid_geds) > 0 else None,
                    'std': float(np.std(valid_geds, ddof=1)) if len(valid_geds) > 1 else None,
                    'median': float(np.median(valid_geds)) if len(valid_geds) > 0 else None,
                    'min': float(np.min(valid_geds)) if len(valid_geds) > 0 else None,
                    'max': float(np.max(valid_geds)) if len(valid_geds) > 0 else None,
                },
                'accuracy': {
                    'mean': float(np.mean(accs)) if len(accs) > 0 else None,
                    'std': float(np.std(accs, ddof=1)) if len(accs) > 1 else None,
                    'median': float(np.median(accs)) if len(accs) > 0 else None,
                    'min': float(np.min(accs)) if len(accs) > 0 else None,
                    'max': float(np.max(accs)) if len(accs) > 0 else None,
                }
            }
        else:
            bin_stats[bin_label] = {
                'num_samples': 0,
                'num_valid_geds': 0,
                'ged': {'mean': None, 'std': None, 'median': None, 'min': None, 'max': None},
                'accuracy': {'mean': None, 'std': None, 'median': None, 'min': None, 'max': None}
            }
    
    return bin_stats


def print_statistics_table(bin_stats: Dict[str, Dict], model_name: str):
    """Print formatted statistics table"""
    print(f"\n{'='*100}")
    print(f"METRICS BY ROOM COUNT - {model_name}")
    print(f"{'='*100}\n")
    
    # Header
    print(f"{'Room Bin':<10} {'N':<8} {'GED Mean':<12} {'GED Std':<12} {'GED Median':<12} {'Acc Mean':<12} {'Acc Std':<12}")
    print(f"{'-'*100}")
    
    # Data rows
    for bin_label in ["<=4", "5", "6", "7", "8", ">8"]:
        stats = bin_stats[bin_label]
        n = stats['num_samples']
        n_valid = stats.get('num_valid_geds', n)
        
        ged_mean = f"{stats['ged']['mean']:.3f}" if stats['ged']['mean'] is not None else "N/A"
        ged_std = f"{stats['ged']['std']:.3f}" if stats['ged']['std'] is not None else "N/A"
        ged_median = f"{stats['ged']['median']:.3f}" if stats['ged']['median'] is not None else "N/A"
        
        acc_mean = f"{stats['accuracy']['mean']:.3f}" if stats['accuracy']['mean'] is not None else "N/A"
        acc_std = f"{stats['accuracy']['std']:.3f}" if stats['accuracy']['std'] is not None else "N/A"
        
        # Show valid count if there are NaN GEDs
        n_display = f"{n}" if n == n_valid else f"{n_valid}/{n}"
        
        print(f"{bin_label:<10} {n_display:<8} {ged_mean:<12} {ged_std:<12} {ged_median:<12} {acc_mean:<12} {acc_std:<12}")
    
    print(f"{'-'*100}\n")
    
    # Detailed statistics
    print(f"\nDETAILED STATISTICS BY BIN:\n")
    for bin_label in ["<=4", "5", "6", "7", "8", ">8"]:
        stats = bin_stats[bin_label]
        if stats['num_samples'] == 0:
            continue
        
        print(f"{'='*50}")
        n_total = stats['num_samples']
        n_valid = stats.get('num_valid_geds', n_total)
        if n_total == n_valid:
            print(f"Room Bin: {bin_label} (N={n_total})")
        else:
            print(f"Room Bin: {bin_label} (N={n_total}, {n_valid} valid GEDs)")
        print(f"{'='*50}")
        
        print(f"\n  GED Statistics:")
        print(f"    Mean:   {stats['ged']['mean']:.4f}" if stats['ged']['mean'] is not None else "    Mean:   N/A")
        print(f"    Std:    {stats['ged']['std']:.4f}" if stats['ged']['std'] is not None else "    Std:    N/A")
        print(f"    Median: {stats['ged']['median']:.4f}" if stats['ged']['median'] is not None else "    Median: N/A")
        print(f"    Min:    {stats['ged']['min']:.4f}" if stats['ged']['min'] is not None else "    Min:    N/A")
        print(f"    Max:    {stats['ged']['max']:.4f}" if stats['ged']['max'] is not None else "    Max:    N/A")
        
        print(f"\n  Accuracy Statistics:")
        print(f"    Mean:   {stats['accuracy']['mean']:.4f}" if stats['accuracy']['mean'] is not None else "    Mean:   N/A")
        print(f"    Std:    {stats['accuracy']['std']:.4f}" if stats['accuracy']['std'] is not None else "    Std:    N/A")
        print(f"    Median: {stats['accuracy']['median']:.4f}" if stats['accuracy']['median'] is not None else "    Median: N/A")
        print(f"    Min:    {stats['accuracy']['min']:.4f}" if stats['accuracy']['min'] is not None else "    Min:    N/A")
        print(f"    Max:    {stats['accuracy']['max']:.4f}" if stats['accuracy']['max'] is not None else "    Max:    N/A")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze GED and Accuracy metrics by room count bins",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # HouseGAN++
  python analyze_metrics_by_room_count.py \\
      --model houseganpp \\
      --cache_pickle cached_gen_5var/cache_houseganpp_pretrained.pkl \\
      --results_npz ged_results_houseganpp_pretrain2/benchmark_results.npz \\
      --num_variations 5
  
  # House-Diffusion
  python analyze_metrics_by_room_count.py \\
      --model housediffusion \\
      --cache_pickle cached_gen_5var/cache_housediffusion_pretrained.pkl \\
      --results_npz ged_results_housediffusion_pretrained2/benchmark_results.npz \\
      --num_variations 5
        """
    )
    
    parser.add_argument("--model", type=str, required=True, choices=["houseganpp", "housediffusion"],
                       help="Model type: houseganpp or housediffusion")
    parser.add_argument("--cache_pickle", type=str, required=True,
                       help="Path to cached generations pickle file")
    parser.add_argument("--results_npz", type=str, required=True,
                       help="Path to results NPZ file (benchmark_results.npz from evaluate_houseganpp.py or similar)")
    parser.add_argument("--num_variations", type=int, required=True,
                       help="Number of variations per sample used during evaluation")
    parser.add_argument("--output_json", type=str, default=None,
                       help="Path to save binned statistics as JSON (optional)")
    
    args = parser.parse_args()
    
    cache_pickle = Path(args.cache_pickle)
    results_npz = Path(args.results_npz)
    
    # Validate paths
    if not cache_pickle.exists():
        print(f"❌ Error: Cache file not found: {cache_pickle}")
        return 1
    
    if not results_npz.exists():
        print(f"❌ Error: Results NPZ not found: {results_npz}")
        return 1
    
    print(f"{'='*100}")
    print(f"ANALYZING METRICS BY ROOM COUNT")
    print(f"{'='*100}")
    print(f"Model: {args.model}")
    print(f"Cache: {cache_pickle}")
    print(f"Results: {results_npz}")
    print(f"Variations: {args.num_variations}")
    print()
    
    # Load data based on model type
    if args.model == "houseganpp":
        all_geds, all_accuracies, gt_data = load_houseganpp_data(cache_pickle, results_npz)
        bin_stats = analyze_houseganpp_by_room_count(all_geds, all_accuracies, gt_data, args.num_variations)
        model_name = "HouseGAN++"
    else:
        all_geds, all_accuracies, gt_data = load_housediffusion_data(cache_pickle, results_npz)
        bin_stats = analyze_housediffusion_by_room_count(all_geds, all_accuracies, gt_data, args.num_variations)
        model_name = "House-Diffusion"
    
    # Print statistics
    print_statistics_table(bin_stats, model_name)
    
    # Save to JSON if requested
    if args.output_json:
        output_path = Path(args.output_json)
        with open(output_path, 'w') as f:
            json.dump({
                'model': args.model,
                'model_name': model_name,
                'cache_pickle': str(cache_pickle),
                'results_npz': str(results_npz),
                'num_variations': args.num_variations,
                'bin_statistics': bin_stats
            }, f, indent=2)
        print(f"✓ Statistics saved to: {output_path}")
    
    print(f"\n{'='*100}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*100}\n")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
