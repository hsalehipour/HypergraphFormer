#!/usr/bin/env python3
"""
MSD to Hypergraph and Reverse

This script demonstrates the complete workflow from MSD graph data to hypergraph generation and back to floor plan.

Usage:
    python msd_to_hypergraph.py [--file-id FILE_ID] [--subgraph-index INDEX] [--show-plots]

Author: Generated from MSD to Hypergraph and Reverse.ipynb
"""

import argparse
import matplotlib.pyplot as plt
from pathlib import Path

from msd.graph import MSDGraph
from utils import HypergraphPlotter
from msd.plotter import MSDPlotter

# Hypergraph library imports
from hypergraph.api.lib.tools import RGL
import clr
clr.AddReference("System.Collections")
from System import Collections
from RGeoLib import DataNodeUtils


class MSDHypergraph(MSDGraph):
    """
    A class for MSD to Hypergraph conversion workflow.
    
    This class encapsulates the complete workflow from loading MSD data,
    converting to hypergraph, and generating floor plans.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the MSD to Hypergraph conversion workflow with data directory.
        
        Args:
            data_dir (str): Path to the MSD training data directory
        """
        self.data_dir = Path(data_dir)
        
        # Initialize plotters
        self.msd_plotter = MSDPlotter()
        self.hypergraph_plotter = HypergraphPlotter()
        
        print("🔧 MSD to Hypergraph conversion workflow initialized")
        print(f"📁 Data directory: {self.data_dir}")
    
    def convert_to_hypergraph(self, graph_data):
        """
        Convert the selected subgraph to hypergraph representation.
        
        Returns:
            bool: True if successful, False otherwise
        """
        assert graph_data is not None, "No graph data to convert to hypergraph."
        
        # Step 1: Calculate geometry-based boundaries
        hull_data = self.get_geometry_based_boundaries(graph_data)
        
        # Step 2: Create face wrapper from convex hull
        print("   🔷 Creating face wrapper...")
        face = self.convex_hull_to_face_wrapper(hull_data['convex_hull_geometry'])
        
        # Step 3: Create NMesh from subgraph
        print("   🏗️  Creating NMesh...")
        nmesh, room_faces = self.create_nmesh_from_subgraph(graph_data)
        
        # Step 4: Build connectivity dictionary
        print("   🔗 Building connectivity...")
        connectivity_dict, _ = self.build_connectivity_dict(graph_data)
        
        # Step 5: Create DataNode with connectivity
        print("   🌳 Creating DataNode...")
        data_node = DataNodeUtils.CreateDataNodeWithConnectivity(
            nmesh, face, connectivity_dict
        )
        
        print("✅ Hypergraph conversion completed successfully!")
        return data_node
    
    def plot_hypergraph(self, hypergraph, prefix):
        """
        Visualize the generated hypergraph.
        Args:
            hypergraph (DataNode): The hypergraph data to plot
            prefix (str): The prefix of the file to save the plot
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        assert hypergraph is not None, "No hypergraph data to plot."
        # Serialize and plot the hypergraph
        serialized_data = RGL.DataNode.serializeDataNode(hypergraph)
        fig = self.hypergraph_plotter.plot_hypergraph(serialized_data)

        print(f"📊 Saving hypergraph plot to {prefix}_hypergraph.png")
        plt.savefig(f"{prefix}_hypergraph.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig
    
    def hypergraph_to_floor_plan(self, hypergraph, custom_boundary_points):
        """
        Generate a floor plan from the hypergraph using custom boundary points.
        Args:
            hypergraph (DataNode): The hypergraph data to generate a floor plan from
            custom_boundary_points (list): Custom boundary points [(x1,y1), (x2,y2), ...]
            
        Returns:
            bool: True if successful, False otherwise
        """
        assert hypergraph is not None, "No hypergraph data. Call convert_to_hypergraph() first."        
        assert custom_boundary_points is not None, "No custom boundary points provided."
        print(f"   📐 Using boundary: {custom_boundary_points}")
        
        # Create boundary face
        points_list = Collections.Generic.List[RGL.Vec3d]()
        for x, y in custom_boundary_points:
            p = RGL.Vec3d(x=x, y=y, z=0)
            points_list.Add(p)
        
        face_bounds = RGL.NFace(points_list)
        nface_list = Collections.Generic.List[RGL.NFace]()
        nface_list.Add(face_bounds)
        boundary_mesh = RGL.NMesh(nface_list)
        
        # Generate mesh using hypergraph
        print("   🔄 Applying hypergraph subdivision...")
        generated_mesh = RGL.DataNode.subdivideWithMeshRatio(hypergraph, boundary_mesh)
        print("✅ Floor plan generation completed successfully!")
        return generated_mesh
    
    def plot_hypergraph_floor_plan(self, hypergraph_floor_plan, prefix):
        """
        Visualize the generated floor plan.
        Args:
            hypergraph_floor_plan (DataNode): The floor plan data to plot
            prefix (str): The prefix of the file to save the plot
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        assert hypergraph_floor_plan is not None, "No hypergraph generated floor plan data to plot."
        fig = self.hypergraph_plotter.simple_plot_generated_mesh(hypergraph_floor_plan)
        
        print(f"📊 Saving generated floor plan plot to {prefix}_hypergraph_floor_plan.png")
        plt.savefig(f"{prefix}_hypergraph_floor_plan.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig
    
    def run_complete_workflow(self, file_id=7498, subgraph_index=1):
        """
        Run the complete MSD to Hypergraph and Reverse workflow.
        
        Args:
            file_id (int): ID of the MSD file to load
            subgraph_index (int): Index of subgraph to process
            
        Returns:
            bool: True if successful, False otherwise
        """
        print("🚀 Starting complete MSD to Hypergraph workflow...")
        print("=" * 60)
        
        # Step 1: Load data
        graph_data = self.load_graph_data(file_id, str(self.data_dir))
        
        # Step 2: Visualize original
        self.plot_graph(graph_data, f"{file_id}_msd_graph.png", "Original Floor Plan Graph")
        
        # Step 3: Extract subgraphs
        subgraphs = self.extract_subgraphs_without_entrance(graph_data, min_nodes=2)
        print(f"✅ Found {len(subgraphs)} subgraphs")
        
        for i, sg in enumerate(subgraphs):
            print(f"   Subgraph {i}: {sg.number_of_nodes()} nodes, {sg.number_of_edges()} edges")
        
        # Step 4: Select and visualize subgraph
        assert subgraph_index >= 0 and subgraph_index < len(subgraphs), f"Invalid subgraph index {subgraph_index}. Available: 0-{len(subgraphs)-1}"
        target_subgraph = subgraphs[subgraph_index]
        print(f"✅ Selected subgraph {subgraph_index}: {target_subgraph.number_of_nodes()} nodes")
        
        self.plot_graph(target_subgraph, f"{file_id}_{subgraph_index}_subgraph.png", "Selected Subgraph")
        
        # Step 5: Rotate the graph to align with the x axis
        rotated_subgraph = self.rotate_x_aligned(target_subgraph)
        self.plot_graph(rotated_subgraph, f"{file_id}_{subgraph_index}_rotated.png", "Rotated Floor Plan")

        # Step 6: Convert to hypergraph
        hypergraph = self.convert_to_hypergraph(rotated_subgraph)
        
        # Step 7: Visualize hypergraph
        self.plot_hypergraph(hypergraph, f"{file_id}_{subgraph_index}")
        
        # Step 8: Generate floor plan
        # Define boundary polygon
        custom_boundary_points = [
            (0, 0), (0, 8), (10, 8), (10, 3), (12, 3), (12, 0)
        ]
        floor_plan = self.hypergraph_to_floor_plan(hypergraph, custom_boundary_points)
        
        # Step 9: Visualize generated floor plan
        self.plot_hypergraph_floor_plan(floor_plan, f"{file_id}_{subgraph_index}")
        
        print("=" * 60)
        print("🎉 Complete workflow finished successfully!")
        return graph_data, subgraphs, rotated_subgraph, hypergraph, floor_plan

    def get_info(self, graph_data, subgraphs, target_subgraph, hypergraph, floor_plan):
        """
        Get information about the current state.
        
        Returns:
            dict: Information about the current state
        """
        info = {
            'graph_loaded': graph_data is not None,
            'subgraphs_extracted': subgraphs is not None,
            'subgraph_selected': target_subgraph is not None,
            'hypergraph_created': hypergraph is not None,
            'floor_plan_generated': floor_plan is not None
        }
        
        if graph_data:
            info['graph_nodes'] = graph_data.number_of_nodes()
            info['graph_edges'] = graph_data.number_of_edges()
        
        if subgraphs:
            info['num_subgraphs'] = len(subgraphs)
        
        if target_subgraph:
            info['subgraph_nodes'] = target_subgraph.number_of_nodes()
            info['subgraph_edges'] = target_subgraph.number_of_edges()
        
        return info


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='MSD to Hypergraph conversion workflow')
    parser.add_argument('--file-id', type=int, default=7498, 
                       help='ID of the MSD file to load (default: 7498)')
    parser.add_argument('--subgraph-index', type=int, default=1,
                       help='Index of subgraph to process (default: 1)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable plot display (useful for batch processing)')
    parser.add_argument('--data-dir', type=str, 
                       default="data/modified-swiss-dwellings-v2/train/graph_out",
                       help='Path to MSD training data directory')
    
    args = parser.parse_args()
    
    # Create MSD to Hypergraph conversion workflow instance
    msd_to_hypergraph = MSDHypergraph(data_dir=args.data_dir)
    
    # Run complete workflow
    graph_data, subgraphs, target_subgraph, hypergraph, floor_plan = msd_to_hypergraph.run_complete_workflow(
        file_id=args.file_id,
        subgraph_index=args.subgraph_index,
    )
    
    # Print information
    print("\n📊 Information:")
    info = msd_to_hypergraph.get_info(graph_data, subgraphs, target_subgraph, hypergraph, floor_plan)
    for key, value in info.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    main()