#!/usr/bin/env python3
"""
Comprehensive validation script: Check if len(nmesh.faceList) == leaf_count
for all train and test data.

This standalone script runs faster than the notebook version for large-scale processing.

Features:
- Professional logging with configurable levels (DEBUG, INFO, WARNING, ERROR)
- Progress bars with tqdm for real-time tracking
- Command line arguments for customization
- Automatic result saving to pickle file
- Comprehensive error reporting and statistics

Usage:
    python validate_faces_leaves.py [options]

Examples:
    python validate_faces_leaves.py --verbose
    python validate_faces_leaves.py --quiet --log-file results.log
"""

import pickle
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

from msd.graph import MSDGraph

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


def process_and_validate_subgraph(subgraph, file_name, subgraph_idx):
    """Process a single subgraph and validate face count vs leaf count"""
    try:
        # Calculate convex hull
        hull_data = MSDGraph.get_geometry_based_boundaries(subgraph)
        if not hull_data or not hull_data.get('convex_hull_geometry'):
            return False, "No convex hull geometry"
        
        face = MSDGraph.convex_hull_to_face_wrapper(hull_data['convex_hull_geometry'])
        if not face:
            return False, "Failed to create face wrapper"
        
        # Create nmesh
        nmesh, room_faces = MSDGraph.create_nmesh_from_subgraph(subgraph)
        if not nmesh:
            return False, "Failed to create nmesh"
        
        # Create data_node
        data_node = DataNodeUtils.CreateDataNodeWithGeometryUtils(nmesh, face)
        if not data_node:
            return False, "Failed to create data_node"
        
        # Count faces and leaves
        face_count = len(nmesh.faceList)
        
        try:
            all_children = RGL.DataNode.GetAllChildren(data_node)
            leaf_count = sum(1 for node in all_children if hasattr(node, 'final') and node.final)
        except Exception as e:
            return False, f"Failed to count leaves: {e}"
        
        # Check if they match
        success = (face_count == leaf_count)
        return success, {
            'face_count': face_count,
            'leaf_count': leaf_count,
            'nodes_count': subgraph.number_of_nodes(),
            'edges_count': subgraph.number_of_edges()
        }
        
    except Exception as e:
        return False, f"Processing error: {str(e)}"


def validate_all_data():
    """Validate face count vs leaf count across all train and test data"""
    
    # Data directories
    train_dir = Path("data/modified-swiss-dwellings-v2/train/graph_out")
    test_dir = Path("data/modified-swiss-dwellings-v2/test/graph_out")
    
    # Counters
    total_files = 0
    total_subgraphs = 0
    successful_subgraphs = 0
    failed_files = 0
    
    # Results tracking
    results = {
        'successful': [],
        'failed': [],
        'face_leaf_mismatches': []
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
                
                # Process each subgraph (no progress bar for subgraphs)
                for sub_idx, subgraph in enumerate(subgraphs):
                    total_subgraphs += 1
                    
                    # Skip subgraphs that are too small
                    if subgraph.number_of_nodes() < 2:
                        continue
                    
                    # Update file progress bar with current subgraph info
                    file_pbar.set_postfix_str(f"🔍 {file_path.name} - Subgraph {sub_idx+1}/{len(subgraphs)} ({subgraph.number_of_nodes()} nodes)")
                    
                    success, result = process_and_validate_subgraph(subgraph, file_path.name, sub_idx)
                    
                    if success:
                        successful_subgraphs += 1
                        results['successful'].append({
                            'file': file_path.name,
                            'data_type': data_type,
                            'subgraph_idx': sub_idx,
                            **result
                        })
                    else:
                        if isinstance(result, dict) and result.get('face_count') != result.get('leaf_count'):
                            results['face_leaf_mismatches'].append({
                                'file': file_path.name,
                                'data_type': data_type,
                                'subgraph_idx': sub_idx,
                                **result
                            })
                        else:
                            results['failed'].append({
                                'file': file_path.name,
                                'data_type': data_type,
                                'subgraph_idx': sub_idx,
                                'error': result
                            })
                
                # Update file progress bar with current stats
                success_rate = (successful_subgraphs / total_subgraphs * 100) if total_subgraphs > 0 else 0
                file_pbar.set_postfix_str(f"✅ {successful_subgraphs}/{total_subgraphs} ({success_rate:.1f}%)")
                            
            except Exception as e:
                failed_files += 1
                file_pbar.set_postfix_str(f"❌ Error: {str(e)[:30]}...")
                continue
        
        file_pbar.close()
    
    return results, {
        'total_files': total_files,
        'total_subgraphs': total_subgraphs,
        'successful_subgraphs': successful_subgraphs,
        'failed_files': failed_files
    }


def save_successful_subgraphs_to_file(results, stats, output_file):
    """Save successful subgraphs to a human-readable text file"""
    try:
        with open(output_file, 'w') as f:
            # Write header
            f.write("="*80 + "\n")
            f.write("SUCCESSFUL SUBGRAPHS VALIDATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Write summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Total files processed: {stats['total_files']:,}\n")
            f.write(f"Total subgraphs analyzed: {stats['total_subgraphs']:,}\n")
            f.write(f"Successful matches (face_count == leaf_count): {stats['successful_subgraphs']:,}\n")
            f.write(f"Failed files: {stats['failed_files']:,}\n")
            
            success_rate = (stats['successful_subgraphs'] / stats['total_subgraphs'] * 100) if stats['total_subgraphs'] > 0 else 0
            f.write(f"Success rate: {success_rate:.2f}%\n")
            f.write("\n\n")
            
            # Write detailed list of successful subgraphs
            f.write("DETAILED LIST OF SUCCESSFUL SUBGRAPHS\n")
            f.write("-"*80 + "\n")
            f.write(f"Total successful subgraphs: {len(results['successful'])}\n\n")
            
            # Group by data type for better organization
            train_successes = [r for r in results['successful'] if r['data_type'] == 'train']
            test_successes = [r for r in results['successful'] if r['data_type'] == 'test']
            
            # Write train data
            if train_successes:
                f.write("\n" + "="*80 + "\n")
                f.write(f"TRAIN DATA - {len(train_successes)} successful subgraphs\n")
                f.write("="*80 + "\n\n")
                
                for i, result in enumerate(train_successes, 1):
                    f.write(f"{i}. File: {result['file']}\n")
                    f.write(f"   Subgraph Index: {result['subgraph_idx']}\n")
                    f.write(f"   Face Count: {result['face_count']}\n")
                    f.write(f"   Leaf Count: {result['leaf_count']}\n")
                    f.write(f"   Nodes: {result['nodes_count']}, Edges: {result['edges_count']}\n")
                    f.write("\n")
            
            # Write test data
            if test_successes:
                f.write("\n" + "="*80 + "\n")
                f.write(f"TEST DATA - {len(test_successes)} successful subgraphs\n")
                f.write("="*80 + "\n\n")
                
                for i, result in enumerate(test_successes, 1):
                    f.write(f"{i}. File: {result['file']}\n")
                    f.write(f"   Subgraph Index: {result['subgraph_idx']}\n")
                    f.write(f"   Face Count: {result['face_count']}\n")
                    f.write(f"   Leaf Count: {result['leaf_count']}\n")
                    f.write(f"   Nodes: {result['nodes_count']}, Edges: {result['edges_count']}\n")
                    f.write("\n")
            
            # Footer
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        return True
    except Exception as e:
        logger.error(f"Failed to save successful subgraphs to file: {e}")
        return False


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Validate face count vs leaf count across all train and test data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_faces_leaves.py                    # Run with default settings
  python validate_faces_leaves.py --verbose          # Enable verbose logging
  python validate_faces_leaves.py --log-file val.log # Save logs to custom file
  python validate_faces_leaves.py --quiet            # Only show errors and warnings
  python validate_faces_leaves.py -o output.txt      # Save successful subgraphs to custom file
        """
    )
    
    parser.add_argument(
        '--log-file', 
        type=str, 
        default='validation.log',
        help='Log file path (default: validation.log)'
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
        '--output-file', '-o',
        type=str,
        default='successful_subgraphs.txt',
        help='Output text file for successful subgraphs (default: successful_subgraphs.txt)'
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
    logger.info("🚀 Face-Leaf Validation Script Started")
    logger.info("📊 Processing all train and test data")
    logger.info("⏱️ Running as standalone Python script for optimal performance")
    logger.info("🎯 Goal: Check if len(nmesh.faceList) == leaf_count")
    logger.info("="*60)
    
    # Run the validation
    results, stats = validate_all_data()
    
    # Log completion message
    logger.info("🎉 Validation completed!")
    logger.info("="*60)
    
    # Log summary
    logger.info("📊 FINAL VALIDATION RESULTS:")
    logger.info(f"📁 Total files processed: {stats['total_files']:,}")
    logger.info(f"🔸 Total subgraphs analyzed: {stats['total_subgraphs']:,}")
    logger.info(f"✅ Successful matches (face_count == leaf_count): {stats['successful_subgraphs']:,}")
    logger.info(f"❌ Failed files: {stats['failed_files']:,}")
    
    success_rate = (stats['successful_subgraphs'] / stats['total_subgraphs'] * 100) if stats['total_subgraphs'] > 0 else 0
    logger.info(f"🎯 OVERALL SUCCESS RATE: {success_rate:.2f}%")
    logger.info("="*60)
    
    if results['face_leaf_mismatches']:
        logger.warning(f"⚠️ Face/Leaf mismatches: {len(results['face_leaf_mismatches'])}")
        logger.warning("First few mismatches:")
        for mismatch in results['face_leaf_mismatches'][:5]:
            logger.warning(f"  {mismatch['file']}: faces={mismatch['face_count']}, leaves={mismatch['leaf_count']}")
    
    if results['failed']:
        logger.error(f"❌ Processing failures: {len(results['failed'])}")
        logger.error("Most common errors:")
        error_counts = {}
        for failure in results['failed']:
            error_type = str(failure['error'])[:50] + "..." if len(str(failure['error'])) > 50 else str(failure['error'])
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
            logger.error(f"  {error}: {count} occurrences")
    
    # Save successful subgraphs to text file
    output_file = Path(args.output_file)
    if save_successful_subgraphs_to_file(results, stats, output_file):
        logger.info(f"📄 Successful subgraphs saved to: {output_file}")
    else:
        logger.warning(f"⚠️ Failed to save successful subgraphs to {output_file}")
    
    return success_rate


if __name__ == "__main__":
    success_rate = main()
    logger.info(f"🏁 Script completed with {success_rate:.2f}% success rate")
