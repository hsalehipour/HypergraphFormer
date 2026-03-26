#!/usr/bin/env python3
"""
Save successful hypergraph data_nodes in text format.

This script processes train and test data, validates hypergraphs (face_count == leaf_count),
and saves only successful hypergraphs with at least 4 leaves to text files using
RGL.DataNode.serializeDataNode().

Features:
- Professional logging with configurable levels (DEBUG, INFO, WARNING, ERROR)
- Progress bars with tqdm for real-time tracking
- Command line arguments for customization
- Automatic saving of serialized data_nodes to text files
- Filters for successful hypergraphs with at least 4 leaves/rooms

Usage:
    python save_successful_hypergraphs.py [options]

Examples:
    python save_successful_hypergraphs.py --verbose
    python save_successful_hypergraphs.py --min-leaves 5
    python save_successful_hypergraphs.py --output-dir output_hypergraphs/
"""

import pickle
import logging
import argparse
import json
from pathlib import Path
from tqdm import tqdm

from msd.graph import MSDGraph
from utils import remove_mergeid_suffixes

# Hypergraph library imports
from hypergraph.api.lib.tools import RGL
import clr
clr.AddReference("System.Collections")
from RGeoLib import DataNodeUtils

# Configure logging
def setup_logging(log_level=logging.INFO, log_file=None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create logger
    logger = logging.getLogger(__name__)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    return logger

# Initialize logger (will be properly configured in main())
logger = logging.getLogger(__name__)


def process_and_save_subgraph(subgraph, file_name, subgraph_idx, output_dir, data_type, min_leaves=4):
    """
    Process a single subgraph, validate it, and save if successful with sufficient leaves.
    
    Args:
        subgraph: The subgraph to process
        file_name: Original filename
        subgraph_idx: Index of the subgraph
        output_dir: Output directory path
        data_type: 'train' or 'test' to indicate the split
        min_leaves: Minimum number of leaves required (default: 4)
    
    Returns:
        tuple: (success, result_dict, saved_file_path or None)
    """
    try:
        # Rotate subgraph to align with x-axis (important for BSP algorithm)
        rotated_subgraph = MSDGraph.rotate_x_aligned(subgraph)
        
        # Generate access graph dictionary from rotated subgraph
        access_graph_dict = MSDGraph.generate_access_graph_json(rotated_subgraph)
        
        # Calculate convex hull using rotated subgraph
        hull_data = MSDGraph.get_geometry_based_boundaries(rotated_subgraph)
        if not hull_data or not hull_data.get('convex_hull_geometry'):
            return False, "No convex hull geometry", None
        
        face = MSDGraph.convex_hull_to_face_wrapper(hull_data['convex_hull_geometry'])
        if not face:
            return False, "Failed to create face wrapper", None
        
        # Create nmesh from rotated subgraph
        nmesh, room_faces = MSDGraph.create_nmesh_from_subgraph(rotated_subgraph)
        if not nmesh:
            return False, "Failed to create nmesh", None
        
        # Build connectivity dictionary (access graph) from rotated subgraph
        connectivity_dict, _ = MSDGraph.build_connectivity_dict(rotated_subgraph)
        
        # Create data_node with connectivity (includes access graph)
        data_node = DataNodeUtils.CreateDataNodeWithConnectivity(nmesh, face, connectivity_dict)
        if not data_node:
            return False, "Failed to create data_node", None
        
        # Count faces and leaves
        face_count = len(nmesh.faceList)
        
        try:
            all_children = RGL.DataNode.GetAllChildren(data_node)
            leaf_count = sum(1 for node in all_children if hasattr(node, 'final') and node.final)
        except Exception as e:
            return False, f"Failed to count leaves: {e}", None
        
        # Check if validation passes
        if face_count != leaf_count:
            return False, {
                'face_count': face_count,
                'leaf_count': leaf_count,
                'reason': 'Face count does not match leaf count'
            }, None
        
        # Check if leaf count meets minimum requirement
        if leaf_count < min_leaves:
            return False, {
                'face_count': face_count,
                'leaf_count': leaf_count,
                'reason': f'Leaf count ({leaf_count}) < minimum required ({min_leaves})'
            }, None
        
        # Serialize data_node to text
        try:
            serialized_text = RGL.DataNode.serializeDataNode(data_node)
            # Remove numeric suffixes from mergeid values (e.g., "Kitchen_1" -> "Kitchen")
            serialized_text = remove_mergeid_suffixes(serialized_text)
        except Exception as e:
            return False, f"Failed to serialize data_node: {e}", None
        
        # Create output filename
        # Format: {split}_{original_filename_without_ext}_subgraph_{subgraph_idx}.txt
        base_name = Path(file_name).stem
        output_filename = f"{data_type}_{base_name}_subgraph_{subgraph_idx}.txt"
        output_path = output_dir / output_filename
        
        # Create combined JSON structure
        combined_data = {
            "accessgraph": access_graph_dict,
            "hypergraph": serialized_text
        }
        
        # Save to JSON file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, indent=2)
        except Exception as e:
            return False, f"Failed to write to file: {e}", None
        
        # Success!
        return True, {
            'face_count': face_count,
            'leaf_count': leaf_count,
            'nodes_count': subgraph.number_of_nodes(),
            'edges_count': subgraph.number_of_edges(),
            'output_file': output_filename
        }, output_path
        
    except Exception as e:
        return False, f"Processing error: {str(e)}", None


def process_all_data(output_dir, min_leaves=4):
    """Process all train and test data, saving successful hypergraphs"""
    
    # Data directories
    train_dir = Path("dataset/msd/modified-swiss-dwellings-v2/train/graph_out")
    test_dir = Path("dataset/msd/modified-swiss-dwellings-v2/test/graph_out")
    
    # Create single output directory for all files
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Counters
    total_files = 0
    total_subgraphs = 0
    successful_subgraphs = 0
    saved_hypergraphs = 0
    failed_files = 0
    insufficient_leaves = 0
    validation_failures = 0
    
    # Results tracking
    results = {
        'successful_saved': [],
        'failed_validation': [],
        'insufficient_leaves': []
    }
    
    # Process both train and test data
    for data_type, data_dir in [("train", train_dir), ("test", test_dir)]:
        if not data_dir.exists():
            logger.warning(f"{data_type} directory not found: {data_dir}")
            continue
            
        logger.info(f"Processing {data_type} data from: {data_dir}")
        files = list(data_dir.glob("*.pickle"))
        logger.info(f"Found {len(files)} files in {data_type} dataset")
        
        # Progress bar for files
        file_pbar = tqdm(files, desc=f"📁 {data_type.capitalize()} files", 
                        unit="file", leave=True)
        
        for file_path in file_pbar:
            total_files += 1
            file_pbar.set_postfix_str(f"📄 {file_path.name}")
            
            try:
                # Load graph data
                with open(file_path, 'rb') as f:
                    graph_data = pickle.load(f)
                
                # Extract subgraphs
                subgraphs = MSDGraph.extract_subgraphs_without_entrance(graph_data)
                
                if not subgraphs:
                    continue
                
                # Process each subgraph
                for sub_idx, subgraph in enumerate(subgraphs):
                    total_subgraphs += 1
                    
                    # Skip subgraphs that are too small
                    if subgraph.number_of_nodes() < 2:
                        continue
                    
                    # Update file progress bar with current subgraph info
                    file_pbar.set_postfix_str(
                        f"🔍 {file_path.name} - Subgraph {sub_idx+1}/{len(subgraphs)} "
                        f"({subgraph.number_of_nodes()} nodes)"
                    )
                    
                    success, result, saved_path = process_and_save_subgraph(
                        subgraph, file_path.name, sub_idx, output_dir, data_type, min_leaves
                    )
                    
                    if success:
                        successful_subgraphs += 1
                        saved_hypergraphs += 1
                        results['successful_saved'].append({
                            'file': file_path.name,
                            'data_type': data_type,
                            'subgraph_idx': sub_idx,
                            'saved_to': str(saved_path),
                            **result
                        })
                        logger.debug(f"✅ Saved: {result['output_file']}")
                    else:
                        # Categorize failure
                        if isinstance(result, dict):
                            if result.get('reason', '').startswith('Leaf count'):
                                insufficient_leaves += 1
                                results['insufficient_leaves'].append({
                                    'file': file_path.name,
                                    'data_type': data_type,
                                    'subgraph_idx': sub_idx,
                                    **result
                                })
                            else:
                                validation_failures += 1
                                results['failed_validation'].append({
                                    'file': file_path.name,
                                    'data_type': data_type,
                                    'subgraph_idx': sub_idx,
                                    **result
                                })
                        else:
                            validation_failures += 1
                            results['failed_validation'].append({
                                'file': file_path.name,
                                'data_type': data_type,
                                'subgraph_idx': sub_idx,
                                'error': result
                            })
                
                # Update file progress bar with current stats
                file_pbar.set_postfix_str(
                    f"✅ Saved: {saved_hypergraphs}/{total_subgraphs} "
                    f"({saved_hypergraphs/total_subgraphs*100 if total_subgraphs > 0 else 0:.1f}%)"
                )
                            
            except Exception as e:
                failed_files += 1
                logger.error(f"Failed to process file {file_path.name}: {e}")
                file_pbar.set_postfix_str(f"❌ Error: {str(e)[:30]}...")
                continue
        
        file_pbar.close()
    
    return results, {
        'total_files': total_files,
        'total_subgraphs': total_subgraphs,
        'successful_subgraphs': successful_subgraphs,
        'saved_hypergraphs': saved_hypergraphs,
        'insufficient_leaves': insufficient_leaves,
        'validation_failures': validation_failures,
        'failed_files': failed_files
    }


def save_summary_report(results, stats, output_file):
    """Save a summary report of processed hypergraphs"""
    try:
        with open(output_file, 'w') as f:
            # Write header
            f.write("="*80 + "\n")
            f.write("SUCCESSFUL HYPERGRAPHS PROCESSING REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Write summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Total files processed: {stats['total_files']:,}\n")
            f.write(f"Total subgraphs analyzed: {stats['total_subgraphs']:,}\n")
            f.write(f"Successfully validated: {stats['successful_subgraphs']:,}\n")
            f.write(f"Successfully saved (>= min leaves): {stats['saved_hypergraphs']:,}\n")
            f.write(f"Insufficient leaves: {stats['insufficient_leaves']:,}\n")
            f.write(f"Validation failures: {stats['validation_failures']:,}\n")
            f.write(f"Failed files: {stats['failed_files']:,}\n")
            
            save_rate = (stats['saved_hypergraphs'] / stats['total_subgraphs'] * 100) if stats['total_subgraphs'] > 0 else 0
            f.write(f"Save rate: {save_rate:.2f}%\n")
            f.write("\n\n")
            
            # Write detailed list of saved hypergraphs
            f.write("SAVED HYPERGRAPHS\n")
            f.write("-"*80 + "\n")
            f.write(f"Total saved: {len(results['successful_saved'])}\n\n")
            
            # Group by data type
            train_saved = [r for r in results['successful_saved'] if r['data_type'] == 'train']
            test_saved = [r for r in results['successful_saved'] if r['data_type'] == 'test']
            
            # Write train data
            if train_saved:
                f.write("\n" + "="*80 + "\n")
                f.write(f"TRAIN DATA - {len(train_saved)} saved hypergraphs\n")
                f.write("="*80 + "\n\n")
                
                for i, result in enumerate(train_saved, 1):
                    f.write(f"{i}. File: {result['file']}\n")
                    f.write(f"   Subgraph Index: {result['subgraph_idx']}\n")
                    f.write(f"   Face Count: {result['face_count']}\n")
                    f.write(f"   Leaf Count: {result['leaf_count']}\n")
                    f.write(f"   Nodes: {result['nodes_count']}, Edges: {result['edges_count']}\n")
                    f.write(f"   Saved to: {result['output_file']}\n")
                    f.write("\n")
            
            # Write test data
            if test_saved:
                f.write("\n" + "="*80 + "\n")
                f.write(f"TEST DATA - {len(test_saved)} saved hypergraphs\n")
                f.write("="*80 + "\n\n")
                
                for i, result in enumerate(test_saved, 1):
                    f.write(f"{i}. File: {result['file']}\n")
                    f.write(f"   Subgraph Index: {result['subgraph_idx']}\n")
                    f.write(f"   Face Count: {result['face_count']}\n")
                    f.write(f"   Leaf Count: {result['leaf_count']}\n")
                    f.write(f"   Nodes: {result['nodes_count']}, Edges: {result['edges_count']}\n")
                    f.write(f"   Saved to: {result['output_file']}\n")
                    f.write("\n")
            
            # Footer
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        return True
    except Exception as e:
        logger.error(f"Failed to save summary report: {e}")
        return False


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Save successful hypergraph data_nodes in text format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python save_successful_hypergraphs.py                         # Run with defaults
  python save_successful_hypergraphs.py --verbose               # Enable verbose logging
  python save_successful_hypergraphs.py --min-leaves 5          # Require at least 5 leaves
  python save_successful_hypergraphs.py --output-dir my_output/ # Custom output directory
        """
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='training_data/extracted_hypergraph_msd',
        help='Output directory for saved hypergraphs (default: training_data/extracted_hypergraph_msd/)'
    )
    
    parser.add_argument(
        '--min-leaves', '-m',
        type=int,
        default=4,
        help='Minimum number of leaves/rooms required (default: 4)'
    )
    
    parser.add_argument(
        '--log-file', 
        type=str, 
        default='save_hypergraphs.log',
        help='Log file path (default: save_hypergraphs.log)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode - only warnings and errors'
    )
    
    parser.add_argument(
        '--summary-file', '-s',
        type=str,
        default='hypergraph_summary.txt',
        help='Summary report file (default: hypergraph_summary.txt)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Determine log level
    if args.verbose:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.WARNING
    else:
        log_level = logging.INFO
    
    # Setup logging
    log_file = Path(args.log_file) if args.log_file else None
    global logger
    logger = setup_logging(log_level=log_level, log_file=log_file)
    
    logger.debug(f"Command line arguments: {vars(args)}")
    logger.info("🚀 Hypergraph Saving Script Started")
    logger.info("📊 Processing all train and test data")
    logger.info(f"🎯 Saving successful hypergraphs with >= {args.min_leaves} leaves")
    logger.info(f"💾 Output directory: {args.output_dir}")
    logger.info("="*60)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the processing
    results, stats = process_all_data(output_dir, min_leaves=args.min_leaves)
    
    # Log completion message
    logger.info("🎉 Processing completed!")
    logger.info("="*60)
    
    # Log summary
    logger.info("📊 FINAL RESULTS:")
    logger.info(f"📁 Total files processed: {stats['total_files']:,}")
    logger.info(f"🔸 Total subgraphs analyzed: {stats['total_subgraphs']:,}")
    logger.info(f"✅ Validated successfully: {stats['successful_subgraphs']:,}")
    logger.info(f"💾 Saved to disk: {stats['saved_hypergraphs']:,}")
    logger.info(f"⚠️ Insufficient leaves (< {args.min_leaves}): {stats['insufficient_leaves']:,}")
    logger.info(f"❌ Validation failures: {stats['validation_failures']:,}")
    logger.info(f"❌ Failed files: {stats['failed_files']:,}")
    
    save_rate = (stats['saved_hypergraphs'] / stats['total_subgraphs'] * 100) if stats['total_subgraphs'] > 0 else 0
    logger.info(f"🎯 OVERALL SAVE RATE: {save_rate:.2f}%")
    logger.info("="*60)
    
    # Save summary report
    summary_file = Path(args.summary_file)
    if save_summary_report(results, stats, summary_file):
        logger.info(f"📄 Summary report saved to: {summary_file}")
    else:
        logger.warning(f"⚠️ Failed to save summary report to {summary_file}")
    
    return save_rate


if __name__ == "__main__":
    save_rate = main()
    logger.info(f"🏁 Script completed with {save_rate:.2f}% save rate")

