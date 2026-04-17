#!/usr/bin/env python3
"""
Shared utilities for scoring and saving benchmark results.

This module provides consistent post-processing of evaluation results
(GED, room accuracy, similarity) across different model evaluation
scripts (HouseGAN++, House-Diffusion, etc.).
"""

import os
import numpy as np


def score_benchmark_result(all_geds, num_variations, output_dir,
                           checkpoint_path=None, data_path=None,
                           additional_info=None):
    """
    Compute statistics for benchmark results and save to NPZ file.

    This function provides consistent processing across all evaluation scripts:
    - Handles NaN values from timeouts/failures
    - Computes std of variation means (consistency metric)
    - Prints comprehensive statistics
    - Saves results in standardized NPZ format

    Args:
        all_geds: numpy array of all GED values (may contain np.nan)
        num_variations: number of generation variations per sample
        output_dir: directory to save results
        checkpoint_path: optional path to model checkpoint (for text file)
        data_path: optional path to data file (for text file)
        additional_info: optional dict with additional info to save

    Returns:
        dict: Statistics dictionary with keys:
            - geds: all GED values
            - mean: mean GED
            - std: std of variation means
            - median: median GED
            - num_valid: number of successful computations
            - num_failed: number of failed/timeout computations
    """
    print("\n" + "=" * 80)
    print("📈 GED RESULTS")
    print("=" * 80)
    
    if len(all_geds) == 0:
        print("No valid GED computations")
        return None
    
    all_geds = np.array(all_geds)
    
    # Count valid (non-NaN) GED computations
    num_valid = np.sum(~np.isnan(all_geds))
    num_failed = np.sum(np.isnan(all_geds))
    
    # Basic statistics
    print(f"Total samples processed: {len(all_geds) // num_variations}")
    print(f"Total GED computations: {len(all_geds)} ({len(all_geds) // num_variations} samples × {num_variations} variations)")
    print(f"Valid GED computations: {num_valid}")
    if num_failed > 0:
        print(f"Failed/timeout GED computations: {num_failed} ({num_failed/len(all_geds)*100:.1f}%)")
    
    # Use nanmean/nanstd to handle NaN values from timeouts
    mean_ged = np.nanmean(all_geds)
    median_ged = np.nanmedian(all_geds)
    min_ged = np.nanmin(all_geds)
    max_ged = np.nanmax(all_geds)
    
    print(f"\nMean GED: {mean_ged:.4f}")
    print(f"Median GED: {median_ged:.4f}")
    print(f"Min GED: {min_ged:.4f}")
    print(f"Max GED: {max_ged:.4f}")
    
    # Compute std of variation means (correct way to measure consistency across runs)
    # This measures if different generation runs produce systematically different results
    std_of_variation_means = 0.0
    if num_variations > 1:
        num_samples = len(all_geds) // num_variations
        geds_reshaped = all_geds.reshape(num_samples, num_variations)
        
        # Use nanmean to handle any NaN values from timeouts
        means_per_variation = np.nanmean(geds_reshaped, axis=0)
        std_of_variation_means = np.nanstd(means_per_variation, ddof=1)
        
        print(f"\nConsistency across {num_variations} variations:")
        print(f"  Std of variation means: {std_of_variation_means:.4f}")
        print(f"  Coefficient of variation: {(std_of_variation_means / mean_ged * 100):.2f}%")
        print(f"  Mean per variation: {means_per_variation}")
    
    # Save results to NPZ
    results_file = os.path.join(output_dir, 'benchmark_results.npz')
    save_dict = {
        'geds': all_geds,
        'mean': mean_ged,
        'std': std_of_variation_means,  # std of variation means (consistency metric)
        'median': median_ged,
        'num_valid': num_valid,
        'num_failed': num_failed
    }
    
    # Add any additional info
    if additional_info:
        save_dict.update(additional_info)
    
    np.savez(results_file, **save_dict)
    print(f"\n💾 Results saved to: {results_file}")
    
    # Optionally save text summary
    if checkpoint_path or data_path:
        text_file = os.path.join(output_dir, 'benchmark_results.txt')
        with open(text_file, 'w') as f:
            f.write("GED Results: Access Graph Evaluation\n")
            f.write("=" * 80 + "\n\n")
            
            if checkpoint_path:
                f.write(f"Checkpoint: {checkpoint_path}\n")
            if data_path:
                f.write(f"Data path: {data_path}\n")
            
            f.write(f"\nTotal samples: {len(all_geds) // num_variations}\n")
            f.write(f"Total GED computations: {len(all_geds)} ({len(all_geds) // num_variations} × {num_variations})\n")
            f.write(f"Valid results: {num_valid}\n")
            if num_failed > 0:
                f.write(f"Failed/timeout: {num_failed} ({num_failed/len(all_geds)*100:.1f}%)\n")
            
            f.write(f"\nMean GED: {mean_ged:.4f}\n")
            f.write(f"Median GED: {median_ged:.4f}\n")
            f.write(f"GED range: [{min_ged:.0f}, {max_ged:.0f}]\n")
            
            if num_variations > 1:
                f.write(f"\nConsistency across {num_variations} variations:\n")
                f.write(f"  Std of variation means: {std_of_variation_means:.4f}\n")
                f.write(f"  Coefficient of variation: {(std_of_variation_means / mean_ged * 100):.2f}%\n")
        
        print(f"💾 Summary saved to: {text_file}")
    
    print("=" * 80)
    print("✅ Evaluation complete!")
    
    return save_dict
