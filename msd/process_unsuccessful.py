#!/usr/bin/env python3
"""
Process Unsuccessful Apartments

This script processes N random pickle files that were not successfully converted to hypergraphs
by msd_to_hypergraph.py (batch mode). It reuses functionality from msd_to_hypergraph.py to:
- Load MSD graph data
- Extract subgraphs
- Rotate subgraphs to align with x-axis
- Save PNG plots and JSON files of the rotated apartments

Usage:
    python -m msd.process_unsuccessful [--count N] [--output-dir DIR] [--data-dir DIR]

Author: Generated from msd_to_hypergraph.py
"""

import argparse
import json
import random
from pathlib import Path

from .graph import MSDGraph


class UnsuccessfulApartmentProcessor:
    """
    A class for processing apartments that were not successfully converted to hypergraphs.
    
    This class identifies pickle files that weren't successfully processed and generates
    PNG plots and JSON files of their rotated subgraphs.
    """
    
    def __init__(self, data_dir: str, successful_dir: str):
        """
        Initialize the processor with data directories.
        
        Args:
            data_dir (str): Path to the MSD training data directory
            successful_dir (str): Path to the successfully converted hypergraphs directory
        """
        self.data_dir = Path(data_dir)
        self.successful_dir = Path(successful_dir)
        
        print("🔧 Unsuccessful Apartment Processor initialized")
        print(f"📁 Data directory: {self.data_dir}")
        print(f"📁 Successful hypergraphs directory: {self.successful_dir}")
    
    def get_successful_file_ids(self):
        """
        Extract file IDs from successfully converted hypergraph files.
        
        Returns:
            set: Set of file IDs that were successfully converted
        """
        successful_files = list(self.successful_dir.glob("*.txt"))
        successful_ids = set()
        
        for file_path in successful_files:
            # Extract file ID from filename like "train_10000_subgraph_0.txt"
            filename = file_path.stem
            parts = filename.split('_')
            if len(parts) >= 2:
                try:
                    file_id = int(parts[1])
                    successful_ids.add(file_id)
                except ValueError:
                    continue
        
        print(f"📊 Found {len(successful_ids)} successfully converted file IDs")
        return successful_ids
    
    def get_unsuccessful_pickle_files(self, successful_ids):
        """
        Get pickle files that were not successfully converted.
        
        Args:
            successful_ids (set): Set of successfully converted file IDs
            
        Returns:
            list: List of Path objects for unsuccessful pickle files
        """
        all_pickle_files = []
        
        # Check both train and test directories
        for split in ['train', 'test']:
            split_dir = self.data_dir / split / 'graph_out'
            if split_dir.exists():
                pickle_files = list(split_dir.glob("*.pickle"))
                all_pickle_files.extend(pickle_files)
        
        unsuccessful_files = []
        for file_path in all_pickle_files:
            try:
                file_id = int(file_path.stem)
                if file_id not in successful_ids:
                    unsuccessful_files.append(file_path)
            except ValueError:
                continue
        
        print(f"📊 Found {len(unsuccessful_files)} unsuccessful pickle files")
        return unsuccessful_files
    
    def process_apartment(self, file_path, output_dir):
        """
        Process a single apartment file: load, extract subgraphs, rotate, and plot.
        
        Args:
            file_path (Path): Path to the pickle file
            output_dir (Path): Output directory for plots
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Load graph data using MSDGraph static method
        graph_data = MSDGraph.load_graph_from_path(file_path)
        if graph_data is None:
            return False
        
        # Extract subgraphs using MSDGraph static method
        print("🔍 Extracting subgraphs...")
        subgraphs = MSDGraph.extract_subgraphs_without_entrance(graph_data, min_nodes=2)
        
        if not subgraphs:
            print("⚠️  No subgraphs found")
            return False
        
        print(f"✅ Found {len(subgraphs)} subgraphs")
        
        # Process each subgraph
        success_count = 0
        for i, subgraph in enumerate(subgraphs):
            print(f"   Processing subgraph {i}: {subgraph.number_of_nodes()} nodes")
            
            # Rotate the subgraph to align with x-axis using MSDGraph static method
            rotated_subgraph = MSDGraph.rotate_x_aligned(subgraph)
            
            # Create output filenames
            base_name = file_path.stem
            png_filename = f"{base_name}_subgraph_{i}_rotated.png"
            json_filename = f"{base_name}_subgraph_{i}_rotated.json"
            png_path = output_dir / png_filename
            json_path = output_dir / json_filename
            
            # Plot and save PNG using MSDGraph static method
            png_success = MSDGraph.plot_graph(rotated_subgraph, png_path, "Rotated Floor Plan")
            
            # Save JSON using MSDGraph static method
            json_success = MSDGraph.save_graph_as_json(rotated_subgraph, json_path)
            
            if png_success and json_success:
                success_count += 1
        
        print(f"✅ Successfully processed {success_count}/{len(subgraphs)} subgraphs from {file_path.name}")
        return success_count > 0
    
    def process_random_apartments(self, count=10, output_dir=None):
        """
        Process N random unsuccessful apartments.
        
        Args:
            count (int): Number of apartments to process
            output_dir (str): Output directory for plots
            
        Returns:
            dict: Processing results
        """
        print(f"🚀 Processing {count} random unsuccessful apartments...")
        print("=" * 60)
        
        # Get successful file IDs
        successful_ids = self.get_successful_file_ids()
        
        # Get unsuccessful pickle files
        unsuccessful_files = self.get_unsuccessful_pickle_files(successful_ids)
        
        if not unsuccessful_files:
            print("❌ No unsuccessful files found!")
            return {'processed': 0, 'failed': 0, 'total_files': 0}
        
        # Randomly select files
        selected_files = random.sample(unsuccessful_files, min(count, len(unsuccessful_files)))
        
        # Create output directory
        if output_dir is None:
            output_dir = Path("unsuccessful_apartments_plots")
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory: {output_dir}")
        
        # Process selected files
        processed_count = 0
        failed_count = 0
        
        for i, file_path in enumerate(selected_files, 1):
            print(f"\n📄 Processing file {i}/{len(selected_files)}: {file_path.name}")
            print("-" * 40)
            
            if self.process_apartment(file_path, output_dir):
                processed_count += 1
            else:
                failed_count += 1
        
        print("\n" + "=" * 60)
        print("🎉 Processing completed!")
        print(f"✅ Successfully processed: {processed_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"📁 PNG plots and JSON files saved to: {output_dir}")
        
        return {
            'processed': processed_count,
            'failed': failed_count,
            'total_files': len(selected_files),
            'output_dir': str(output_dir)
        }


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Process unsuccessful apartments and generate rotated plots and JSON files')
    parser.add_argument('--count', '-c', type=int, default=10,
                       help='Number of apartments to process (default: 10)')
    parser.add_argument('--output-dir', '-o', type=str, 
                       default='unsuccessful_apartments_plots',
                       help='Output directory for PNG plots and JSON files (default: unsuccessful_apartments_plots)')
    parser.add_argument('--data-dir', type=str, 
                       default='dataset/msd/modified-swiss-dwellings-v2',
                       help='Path to MSD data directory (default: dataset/msd/modified-swiss-dwellings-v2)')
    parser.add_argument('--successful-dir', type=str,
                       default='training_data/extracted_hypergraph_msd',
                       help='Path to successful hypergraphs directory (default: training_data/extracted_hypergraph_msd)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducible results')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"🎲 Using random seed: {args.seed}")
    
    # Create processor instance
    processor = UnsuccessfulApartmentProcessor(
        data_dir=args.data_dir,
        successful_dir=args.successful_dir
    )
    
    # Process random apartments
    results = processor.process_random_apartments(
        count=args.count,
        output_dir=args.output_dir
    )
    
    # Print final results
    print(f"\n📊 Final Results:")
    print(f"   Total files processed: {results['total_files']}")
    print(f"   Successfully processed: {results['processed']}")
    print(f"   Failed: {results['failed']}")
    print(f"   Output directory: {results['output_dir']}")


if __name__ == "__main__":
    main()
