"""
Graph utilities for MSD floor plan data structures.

This module provides classes for:
- Graph processing and analysis
- Boundary calculations and geometry operations
- Mesh generation from graph structures
"""

import numpy as np
import networkx as nx
import pickle
import json
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import matplotlib.patches as mpatches
from shapely.geometry import Polygon
from shapely.ops import unary_union
from scipy.spatial import ConvexHull
from enum import Enum, auto


class RoomType(Enum):
    """Enumeration of room types with auto-incremented values."""
    BEDROOM = auto()
    LIVINGROOM = auto()
    KITCHEN = auto()
    DINING = auto()
    CORRIDOR = auto()
    STAIRS = auto()
    STOREROOM = auto()
    BATHROOM = auto()
    BALCONY = auto()
    STRUCTURE = auto()
    DOOR = auto()
    ENTRANCE_DOOR = auto()
    WINDOW = auto()
    
    @classmethod
    def from_index(cls, index: int):
        """Get room type from numeric index."""
        # Note: auto() starts from 1, so we need to subtract 1 and add 1 back
        # But actually, we'll use enum value 0-indexed matching
        values = list(cls)
        if 0 <= index < len(values):
            return values[index]
        raise ValueError(f"Invalid room type index: {index}")
    
    @classmethod
    def to_index(cls, room_type) -> int:
        """Convert room type to numeric index."""
        values = list(cls)
        return values.index(room_type)
    
    def to_name(self) -> str:
        """Get the display name for this room type."""
        # Map enum names to display names
        name_map = {
            self.BEDROOM: 'Bedroom',
            self.LIVINGROOM: 'Livingroom',
            self.KITCHEN: 'Kitchen',
            self.DINING: 'Dining',
            self.CORRIDOR: 'Corridor',
            self.STAIRS: 'Stairs',
            self.STOREROOM: 'Storeroom',
            self.BATHROOM: 'Bathroom',
            self.BALCONY: 'Balcony',
            self.STRUCTURE: 'Structure',
            self.DOOR: 'Door',
            self.ENTRANCE_DOOR: 'Entrance Door',
            self.WINDOW: 'Window',
        }
        return name_map.get(self, 'Unknown')


class MSDGraph:
    """
    Class for MSD graph operations and analysis.
    
    This class handles graph-related operations for floor plan data including:
    - Graph decomposition and subgraph extraction
    - Boundary calculations (union, convex hull, concave hull)
    - NMesh generation from graph structures
    - Geometry conversions and transformations
    """
    
    # ROOM_NAMES is kept for backward compatibility
    # Use RoomType enum and to_name() method instead
    ROOM_NAMES = [room_type.to_name() for room_type in RoomType]
    
    @staticmethod
    def get_room_name(room_type_index: int) -> str:
        """Get room name from index using the RoomType enum."""
        try:
            room_type = RoomType.from_index(room_type_index)
            return room_type.to_name()
        except (ValueError, IndexError):
            return f"Unknown_{room_type_index}"
    
    @staticmethod
    def extract_subgraphs_without_entrance(graph, min_nodes=2):
        """
        Extract connected subgraphs after removing entrance edges.
        
        This function removes all edges with connectivity='entrance' and then
        finds connected components. Useful for identifying separate apartment
        units or spatial zones connected by doors but not entrance connections.
        
        Args:
            graph: NetworkX graph with edges containing 'connectivity' attribute
            min_nodes: Minimum number of nodes for a subgraph to be included (default: 2)
            
        Returns:
            List of subgraphs (connected components with at least min_nodes nodes)
        """
        # Create a copy to avoid modifying the original graph
        filtered_graph = graph.copy()
        
        # Find and remove all edges with connectivity = "entrance"
        edges_to_remove = [
            (u, v) for u, v, data in filtered_graph.edges(data=True)
            if data.get('connectivity') == 'entrance'
        ]
        
        filtered_graph.remove_edges_from(edges_to_remove)
        
        # Find connected components after removing entrance edges
        connected_components = list(nx.connected_components(filtered_graph))
        all_subgraphs = [filtered_graph.subgraph(component).copy() 
                        for component in connected_components]
        
        # Filter subgraphs based on minimum node count
        subgraphs = [sg for sg in all_subgraphs if sg.number_of_nodes() >= min_nodes]
        
        return subgraphs
    
    @staticmethod
    def get_geometry_based_boundaries(subgraph):
        """
        Get different types of boundaries using the actual room geometries.
        
        This function calculates multiple boundary representations:
        - Union boundary: Exact boundary following room shapes
        - Convex hull: Tightest convex polygon containing all rooms
        - Concave hull: Alpha shape approximation (if alphashape available)
        
        Args:
            subgraph: NetworkX graph with nodes containing 'geometry' attribute
            
        Returns:
            Dictionary containing:
                - union_boundary: Boundary of union of all room polygons
                - union_geometry: Union geometry (Polygon or MultiPolygon)
                - convex_hull_boundary: Convex hull boundary
                - convex_hull_geometry: Convex hull polygon
                - concave_hull_boundary: Concave hull boundary (if available)
                - concave_hull_geometry: Concave hull polygon (if available)
                - all_vertices: List of all room vertices
                - room_polygons: List of room Polygon objects
        """
        # Extract all room geometries
        room_polygons = []
        for node in subgraph.nodes():
            geometry = subgraph.nodes[node]['geometry']
            room_polygons.append(Polygon(geometry))
        
        if not room_polygons:
            return None
        
        # Create union of all room polygons to get the overall shape
        if len(room_polygons) == 1:
            union_geometry = room_polygons[0]
        else:
            union_geometry = unary_union(room_polygons)
        
        # 1. Boundary of the union (exact boundary following room shapes)
        union_boundary = None
        if hasattr(union_geometry, 'exterior'):
            # Single polygon
            union_boundary = union_geometry.exterior
        elif hasattr(union_geometry, 'geoms'):
            # MultiPolygon - get all exterior boundaries
            union_boundary = [geom.exterior for geom in union_geometry.geoms 
                            if hasattr(geom, 'exterior')]
        
        # 2. Convex hull of all room polygon vertices
        all_vertices = []
        for polygon in room_polygons:
            if hasattr(polygon, 'exterior'):
                # Exclude last duplicate point
                all_vertices.extend(list(polygon.exterior.coords[:-1]))
        
        convex_hull_boundary = None
        convex_hull_polygon = None
        if len(all_vertices) >= 3:
            vertices_array = np.array(all_vertices)
            hull = ConvexHull(vertices_array)
            hull_points = vertices_array[hull.vertices]
            convex_hull_polygon = Polygon(hull_points)
            convex_hull_boundary = convex_hull_polygon.exterior
        
        # 3. Concave hull using alpha shape on room vertices
        concave_hull_boundary = None
        concave_hull_polygon = None
        try:
            from alphashape import alphashape
            concave_hull_polygon = alphashape(all_vertices, alpha=0.4)
            
            if concave_hull_polygon and hasattr(concave_hull_polygon, 'exterior'):
                concave_hull_boundary = concave_hull_polygon.exterior
            elif concave_hull_polygon and hasattr(concave_hull_polygon, 'geoms'):
                # MultiPolygon result
                concave_hull_boundary = [geom.exterior for geom in concave_hull_polygon.geoms 
                                        if hasattr(geom, 'exterior')]
        except ImportError:
            pass  # alphashape not available, skipping concave hull
        
        return {
            'union_boundary': union_boundary,
            'union_geometry': union_geometry,
            'convex_hull_boundary': convex_hull_boundary,
            'convex_hull_geometry': convex_hull_polygon,
            'concave_hull_boundary': concave_hull_boundary,
            'concave_hull_geometry': concave_hull_polygon,
            'all_vertices': all_vertices,
            'room_polygons': room_polygons
        }
    
    @staticmethod
    def create_nmesh_from_subgraph(target_subgraph):
        """
        Create NMesh from subgraph polygons following build_complex_apartment pattern.
        
        This function converts a NetworkX subgraph with room geometries into an
        RGL NMesh structure, which is used by the Research Geometry Library.
        
        Args:
            target_subgraph: NetworkX graph with nodes containing 'geometry' and 'room_type'
            
        Returns:
            Tuple of (room_mesh, all_rooms):
                - room_mesh: RGL.NMesh object
                - all_rooms: List of RGL.NFace objects
                
        Requires:
            - hypergraph.api.lib.tools (RGL library)
            - clr (Python.NET)
        """
        from hypergraph.api.lib.tools import RGL
        import clr
        clr.AddReference("System.Collections")
        from System import Collections
        
        Vec3d = RGL.Vec3d
        NFace = RGL.NFace
        NMesh = RGL.NMesh
        
        # Step 1: Create NFace objects for each room
        all_rooms = []
        
        for i, node in enumerate(target_subgraph.nodes()):
            # Get room geometry and properties
            node_geometry = target_subgraph.nodes[node]['geometry']
            room_type_index = target_subgraph.nodes[node]['room_type']
            
            # Create Vec3d list for room corners
            room_corners_list = Collections.Generic.List[Vec3d]()
            
            # Convert geometry coordinates to Vec3d objects
            for j, (x, y) in enumerate(node_geometry):
                vec3d_corner = Vec3d(x=x, y=y, z=0.0)
                room_corners_list.Add(vec3d_corner)
            
            # Create NFace from corners
            room_nface = NFace(room_corners_list)
            
            # Set merge_id using room name from enum with unique node ID
            room_name = MSDGraph.get_room_name(room_type_index)
            
            # Make merge_id unique by appending node ID
            room_nface.merge_id = f"{room_name}_{node}"
            
            all_rooms.append(room_nface)
        
        # Step 2: Update edge connectivity for all rooms
        for i, room in enumerate(all_rooms):
            room.updateEdgeConnectivity()
        
        # Step 3: Create room mesh
        room_list = Collections.Generic.List[NFace]()
        for room in all_rooms:
            room_list.Add(room)
        
        # Create NMesh
        room_mesh = NMesh(room_list)
        
        return room_mesh, all_rooms
    
    @staticmethod
    def build_connectivity_dict(target_subgraph):
        """
        Build connectivity dictionary from graph edges using room_type and ROOM_NAMES.
        
        This function creates a C# Dictionary mapping unique room identifiers (merge_ids) 
        to their connected neighbors. Each room gets a unique identifier by combining 
        the room type name with the node ID (e.g., "Bedroom_5", "Kitchen_12").
        
        Args:
            target_subgraph: NetworkX graph with nodes containing 'room_type' and edges
            
        Returns:
            Tuple of (connectivity_dict, node_to_merge_id):
                - connectivity_dict: C# Dictionary[str, List[str]] for DataNode
                - node_to_merge_id: Python dict mapping node IDs to merge_id strings
            
        Requires:
            - System.Collections.Generic (from Python.NET)
        """
        from System.Collections.Generic import Dictionary, List as CSharpList
        
        # Build mapping from node ID to unique merge_id
        node_to_merge_id = {}
        
        for node in target_subgraph.nodes():
            room_type_index = target_subgraph.nodes[node]['room_type']
            
            # Get room name using enum
            room_name = MSDGraph.get_room_name(room_type_index)
            
            # Create unique merge_id by appending node ID (matching create_nmesh_from_subgraph)
            merge_id = f"{room_name}_{node}"
            
            node_to_merge_id[node] = merge_id
        
        # Build connectivity dictionary
        connectivity_dict = Dictionary[str, CSharpList[str]]()
        
        for node in target_subgraph.nodes():
            merge_id = node_to_merge_id[node]
            
            # Get all neighbors from graph edges
            neighbors = list(target_subgraph.neighbors(node))
            
            # Create C# List for connected merge_ids (with unique identifiers)
            connected_ids = CSharpList[str]()
            for neighbor in neighbors:
                neighbor_merge_id = node_to_merge_id[neighbor]
                connected_ids.Add(neighbor_merge_id)
            
            # Add to dictionary (each merge_id is now unique, so no conflicts)
            connectivity_dict[merge_id] = connected_ids
        
        return connectivity_dict, node_to_merge_id
    
    @staticmethod
    def generate_access_graph_json(subgraph):
        """
        Generate access graph dictionary from a subgraph.
        
        Creates a simplified representation of the graph connectivity with:
        - Sequential node IDs starting from 1
        - Room type names instead of numeric indices
        - Edge list with string node IDs
        
        Args:
            subgraph: NetworkX graph with nodes containing 'room_type' attribute
            
        Returns:
            dict: Dictionary with 'nodes' and 'edges' keys:
                - nodes: dict mapping string IDs to room type names
                - edges: list of [u, v] pairs with string IDs
        """
        # Create a mapping from original node IDs to sequential IDs starting from 1
        node_id_mapping = {node: str(i + 1) for i, node in enumerate(subgraph.nodes())}
        
        # Create nodes dictionary with sequential IDs -> room type name mapping
        nodes_dict = {}
        for node in subgraph.nodes():
            room_type_index = subgraph.nodes[node]['room_type']
            room_name = MSDGraph.get_room_name(room_type_index)
            new_id = node_id_mapping[node]
            nodes_dict[new_id] = room_name
        
        # Create edges list with renumbered node IDs
        edges_list = [[node_id_mapping[u], node_id_mapping[v]] for u, v in subgraph.edges()]
        
        return {
            "nodes": nodes_dict,
            "edges": edges_list
        }
    
    @staticmethod
    def convex_hull_to_face_wrapper(convex_hull_polygon):
        """
        Convert convex hull polygon to Python Face wrapper.
        
        This function converts a Shapely Polygon (convex hull) to an RGL Face wrapper
        object that can be used with the hypergraph library.
        
        Args:
            convex_hull_polygon: Shapely Polygon object (convex hull)
            
        Returns:
            Face wrapper object with hull points, or None if polygon is invalid
            
        Requires:
            - hypergraph.api.lib.tools (Face, Pt classes)
        """
        from hypergraph.api.lib.tools import Face, Pt
        
        if not convex_hull_polygon:
            return None
        
        # Get hull coordinates
        hull_coords = list(convex_hull_polygon.exterior.coords[:-1])
        
        # Create list of Pt objects
        face_points = []
        
        for i, (x, y) in enumerate(hull_coords):
            pt = Pt(loc=(x, y, 0.0))
            face_points.append(pt)
        
        # Create Face wrapper
        hull_face = Face(corners=face_points)
        
        return hull_face.obj
    
    @staticmethod
    def plot_shapely_geometry(geom, filename):
        """
        Plot a Polygon or MultiPolygon geometry.
        Args:
            geom (shapely.geometry.Geometry): The geometry to plot
            filename (str): The filename to save the plot
        Returns:
            None
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        if geom.geom_type == 'Polygon':
            parts = [geom]
        elif geom.geom_type == 'MultiPolygon':
            parts = geom.geoms
        else:
            raise ValueError(f"Unsupported type: {geom.geom_type}")

        for poly in parts:
            # exterior
            x, y = poly.exterior.xy
            ax.plot(x, y)

            # holes
            for interior in poly.interiors:
                xh, yh = interior.xy
                ax.plot(xh, yh)

        ax.set_aspect('equal')
        plt.axis('off')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def _normalize_to_0_180(angle_deg):
        """
        Normalize an angle to the range [0, 180) degrees.
        
        This ensures that similar directional vectors are treated consistently.
        Negative angles are mapped to their positive complement.
        
        Args:
            angle_deg: Angle in degrees
            
        Returns:
            Normalized angle in degrees [0, 180)
        """
        # Normalize to [-180, 180) range first
        normalized = angle_deg % 360
        if normalized > 180:
            normalized -= 360
        
        # If negative, convert to positive equivalent
        # e.g., -55° → 125° (which is 180 + (-55))
        if normalized < 0:
            normalized = 180 + normalized
        
        return normalized
    
    @staticmethod
    def rotate_x_aligned(msd_graph_data):
        """
        Rotate the entire floor plan to align with x-axis by testing all extreme points.
        
        This function takes an MSD graph and rotates the entire floor plan as a single unit
        by testing all possible extreme points (min x, min y, max x, max y) as pivot points,
        finding adjacent vertices, calculating angles, and selecting the most common angle.
        
        Args:
            msd_graph_data (networkx.Graph): Input MSD graph with room geometries.
                Each node should have a 'geometry' attribute containing a list of
                (x, y) coordinate tuples defining the room's boundary.
            
        Returns:
            networkx.Graph: A deep copy of the input graph with all room geometries
                rotated by the most common angle determined from extreme points.
        """
        import copy
        from shapely.affinity import rotate
        
        print("🔄 Rotating entire floor plan to align with x-axis...")
        
        # Create a deep copy of the graph to avoid modifying the original
        rotated_graph = copy.deepcopy(msd_graph_data)
        
        # Collect all vertices from all room geometries (excluding Balcony)
        all_vertices = []
        vertex_to_polygon = {}  # Map vertex to its polygon and index
        excluded_count = 0
        for node in msd_graph_data.nodes():
            # Skip Balcony nodes to avoid affecting rotation angle determination
            if 'room_type' in msd_graph_data.nodes[node]:
                room_type_index = msd_graph_data.nodes[node]['room_type']
                # Skip Balcony using enum comparison
                try:
                    room_type = RoomType.from_index(room_type_index)
                    if room_type == RoomType.BALCONY:
                        excluded_count += 1
                        continue
                except (ValueError, IndexError):
                    pass  # Skip if room type index is invalid
            if 'geometry' in msd_graph_data.nodes[node]:
                geometry = msd_graph_data.nodes[node]['geometry']
                all_vertices.extend(geometry)
                for i, vertex in enumerate(geometry):
                    vertex_tuple = tuple(vertex)
                    if vertex_tuple not in vertex_to_polygon:
                        vertex_to_polygon[vertex_tuple] = []
                    vertex_to_polygon[vertex_tuple].append((geometry, i, node))
        
        if excluded_count > 0:
            print(f"   ⚠️  Excluded {excluded_count} Balcony node(s) from rotation angle calculation")
        
        if not all_vertices:
            print("⚠️  No room geometries found in the graph")
            return rotated_graph
        
        # Convert to numpy array
        all_vertices = np.array(all_vertices)
        
        # Find all extreme points
        min_x = np.min(all_vertices[:, 0])
        max_x = np.max(all_vertices[:, 0])
        min_y = np.min(all_vertices[:, 1])
        max_y = np.max(all_vertices[:, 1])
        
        min_x_vertices = all_vertices[all_vertices[:, 0] == min_x]
        max_x_vertices = all_vertices[all_vertices[:, 0] == max_x]
        min_y_vertices = all_vertices[all_vertices[:, 1] == min_y]
        max_y_vertices = all_vertices[all_vertices[:, 1] == max_y]
        
        # Pick one vertex from each extreme
        pivots = [
            ("min_x", min_x_vertices[np.argmin(min_x_vertices[:, 1])]),  # leftmost-bottommost
            ("min_y", min_y_vertices[np.argmin(min_y_vertices[:, 0])]),  # bottommost-leftmost
            ("max_x", max_x_vertices[np.argmax(max_x_vertices[:, 1])]),   # rightmost-topmost
            ("max_y", max_y_vertices[np.argmax(max_y_vertices[:, 0])])    # topmost-rightmost
        ]
        
        angles = []
        
        # Test each pivot point
        for pivot_name, pivot_vertex in pivots:
            # Find polygons containing this vertex
            pivot_tuple = tuple(pivot_vertex)
            if pivot_tuple not in vertex_to_polygon:
                continue
            
            for polygon_geometry, vertex_idx, node in vertex_to_polygon[pivot_tuple]:
                # Find adjacent vertex
                polygon_length = len(polygon_geometry)
                
                # Get adjacent vertices
                next_idx = (vertex_idx + 1) % polygon_length
                prev_idx = (vertex_idx - 1) % polygon_length
                next_vertex = polygon_geometry[next_idx]
                prev_vertex = polygon_geometry[prev_idx]
                
                # For min_x and min_y, look for higher coordinates
                # For max_x and max_y, look for lower coordinates (negative direction)
                if pivot_name in ["min_x", "min_y"]:
                    # Look for adjacent vertex with higher coordinate
                    candidates = []
                    if next_vertex[0] > pivot_vertex[0] if pivot_name == "min_x" else next_vertex[1] > pivot_vertex[1]:
                        candidates.append(next_vertex)
                    if prev_vertex[0] > pivot_vertex[0] if pivot_name == "min_x" else prev_vertex[1] > pivot_vertex[1]:
                        candidates.append(prev_vertex)
                    
                    if candidates:
                        # Choose the one with larger difference
                        adjacent_vertex = max(candidates, key=lambda v: 
                            (v[0] - pivot_vertex[0]) if pivot_name == "min_x" else (v[1] - pivot_vertex[1]))
                        
                        # Calculate angle and normalize to [0, 180)
                        vector = np.array(adjacent_vertex) - pivot_vertex
                        angle = np.arctan2(vector[1], vector[0])
                        angle_deg = np.degrees(angle)
                        normalized_angle = MSDGraph._normalize_to_0_180(angle_deg)
                        angles.append(("deg", normalized_angle))
                
                else:  # max_x or max_y
                    # Look for adjacent vertex with lower coordinate (negative direction)
                    candidates = []
                    if next_vertex[0] < pivot_vertex[0] if pivot_name == "max_x" else next_vertex[1] < pivot_vertex[1]:
                        candidates.append(next_vertex)
                    if prev_vertex[0] < pivot_vertex[0] if pivot_name == "max_x" else prev_vertex[1] < pivot_vertex[1]:
                        candidates.append(prev_vertex)
                    
                    if candidates:
                        # Choose the one with larger difference (most negative)
                        adjacent_vertex = min(candidates, key=lambda v: 
                            (v[0] - pivot_vertex[0]) if pivot_name == "max_x" else (v[1] - pivot_vertex[1]))
                        
                        # Calculate angle (pointing in negative direction) and normalize to [0, 180)
                        vector = np.array(pivot_vertex) - adjacent_vertex
                        angle = np.arctan2(vector[1], vector[0])
                        angle_deg = np.degrees(angle)
                        normalized_angle = MSDGraph._normalize_to_0_180(angle_deg)
                        angles.append(("deg", normalized_angle))
        
        if not angles:
            print("⚠️  Could not calculate any angles")
            return rotated_graph
        
        # Find the most common angle
        # Group similar angles together (within 1 degree)
        angle_bins = {}
        for angle_type, angle in angles:
            # angle is already in degrees
            angle_bin = round(angle)
            
            if angle_bin not in angle_bins:
                angle_bins[angle_bin] = []
            angle_bins[angle_bin].append(angle)
        
        # Find the most common angle bin
        most_common_bin = max(angle_bins.keys(), key=lambda k: len(angle_bins[k]))
        most_common_angles = angle_bins[most_common_bin]
        
        # Use the mean of angles in the most common bin
        # Convert back to radians for rotation
        rotation_degrees = -np.mean(most_common_angles)
        
        # Calculate centroid for rotation
        centroid = np.mean(all_vertices, axis=0)
        
        print(f"   📐 Calculated {len(angles)} angles")
        print(f"   📐 Most common angle (bin {most_common_bin}°): {len(most_common_angles)} occurrences")
        print(f"   📐 Floor plan rotation angle: {rotation_degrees:.2f}°")
        print(f"   📊 Centroid: ({centroid[0]:.2f}, {centroid[1]:.2f})")
        
        # Apply the same rotation to all room geometries
        for node in rotated_graph.nodes():
            if 'geometry' in rotated_graph.nodes[node]:
                # Get the original geometry coordinates
                original_geometry = rotated_graph.nodes[node]['geometry']
                
                # Create a Shapely polygon from the geometry
                original_polygon = Polygon(original_geometry)
                
                # Rotate the polygon around the floor plan centroid
                rotated_polygon = rotate(original_polygon, rotation_degrees, 
                                      origin=(centroid[0], centroid[1]))
                
                # Extract the rotated coordinates
                rotated_coords = list(rotated_polygon.exterior.coords[:-1])  # Exclude last duplicate point
                
                # Update the node's geometry with rotated coordinates
                rotated_graph.nodes[node]['geometry'] = rotated_coords
            
            # Rotate node centroid if it exists
            if 'centroid' in rotated_graph.nodes[node]:
                from shapely.geometry import Point
                original_centroid = rotated_graph.nodes[node]['centroid']
                
                # Create a point from the centroid and rotate it
                centroid_point = Point(original_centroid[0], original_centroid[1])
                rotated_centroid_point = rotate(centroid_point, rotation_degrees, 
                                             origin=(centroid[0], centroid[1]))
                
                # Update the node's centroid with rotated coordinates
                rotated_graph.nodes[node]['centroid'] = (rotated_centroid_point.x, rotated_centroid_point.y)
        
        # Rotate edge information as well
        for edge in rotated_graph.edges():
            if 'geometry' in rotated_graph.edges[edge]:
                edge_geometry = rotated_graph.edges[edge]['geometry']
                
                # Handle different edge geometry formats
                if isinstance(edge_geometry, list):
                    # List of coordinates
                    rotated_edge_geometry = []
                    for coord in edge_geometry:
                        if isinstance(coord, (list, tuple)) and len(coord) == 2:
                            point = Point(coord[0], coord[1])
                            rotated_point = rotate(point, rotation_degrees, origin=(centroid[0], centroid[1]))
                            rotated_edge_geometry.append((rotated_point.x, rotated_point.y))
                        else:
                            rotated_edge_geometry.append(coord)
                    rotated_graph.edges[edge]['geometry'] = rotated_edge_geometry
                else:
                    # Try to handle as Shapely object
                    try:
                        rotated_graph.edges[edge]['geometry'] = rotate(edge_geometry, rotation_degrees, 
                                                                    origin=(centroid[0], centroid[1]))
                    except:
                        pass  # Skip if it doesn't work
        
        print(f"✅ Successfully rotated entire floor plan with {rotated_graph.number_of_nodes()} rooms")
        return rotated_graph
    
    @staticmethod
    def load_graph_data(file_id=7498, data_dir="data/modified-swiss-dwellings-v2/train/graph_out"):
        """
        Load graph data from pickle files.
        
        Args:
            file_id (int): ID of the specific file to load
            data_dir (str): Path to the data directory
            
        Returns:
            NetworkX graph: Loaded graph data
        """
        data_path = Path(data_dir)
        
        # Find training files
        total_files = list(data_path.glob("*.pickle"))
        print(f"📊 Found {len(total_files)} training files")
        
        # Load specific file
        test_file = None
        for file_path in total_files:
            if file_path.stem == str(file_id):
                test_file = file_path
                break
        
        if not test_file:
            print(f"⚠️  File {file_id} not found, using first available file")
            test_file = total_files[0]
        
        print(f"📂 Loading file: {test_file.name}")
        with open(test_file, 'rb') as f:
            graph_data = pickle.load(f)
        
        print(f"✅ Graph loaded: {graph_data.number_of_nodes()} nodes, {graph_data.number_of_edges()} edges")
        return graph_data
    
    @staticmethod
    def load_graph_from_path(file_path):
        """
        Load graph data from a specific pickle file path.
        
        Args:
            file_path (Path): Path to the pickle file
            
        Returns:
            NetworkX graph or None if failed
        """
        try:
            print(f"📂 Loading file: {file_path.name}")
            with open(file_path, 'rb') as f:
                graph_data = pickle.load(f)
            
            print(f"✅ Graph loaded: {graph_data.number_of_nodes()} nodes, {graph_data.number_of_edges()} edges")
            return graph_data
        except Exception as e:
            print(f"❌ Failed to load {file_path.name}: {e}")
            return None
    
    @staticmethod
    def plot_graph(graph_data, output_path, title="Floor Plan Graph"):
        """
        Visualize graph data and save as PNG.
        
        Args:
            graph_data (NetworkX graph): The graph data to plot
            output_path (Path): Path to save the plot
            title (str): Title for the plot
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            from plotter import MSDPlotter
            plotter = MSDPlotter()
            
            # Create figure
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))
            
            # Plot the floor plan
            plotter.plot_floor(
                graph_data, ax1, 
                node_size=50,
                edge_size=2
            )
            
            # Create legend for room types
            room_types_in_graph = [graph_data.nodes[node]['room_type'] for node in graph_data.nodes()]
            room_type_counts = Counter(room_types_in_graph)
            
            legend_patches = []
            for room_type_index in sorted(room_type_counts.keys()):
                room_name = plotter.ROOM_NAMES[room_type_index]
                color = plotter.CMAP_ROOMTYPE(room_type_index)
                count = room_type_counts[room_type_index]
                
                patch = mpatches.Patch(color=color, label=f'{room_name} ({count})')
                legend_patches.append(patch)
            
            # Add room type legend
            legend1 = ax1.legend(
                handles=legend_patches, 
                title='Room Types (count)', 
                loc='center left', 
                bbox_to_anchor=(1.02, 0.5),
                fontsize=10,
                title_fontsize=12
            )
            
            # Add edge type legend
            edge_legend_patches = [
                mpatches.Patch(color='black', label='Door/Passage'),
                mpatches.Patch(color='red', label='Entrance')
            ]
            
            legend2 = ax1.legend(
                handles=edge_legend_patches, 
                title='Edge Types', 
                loc='center left', 
                bbox_to_anchor=(1.02, 0.15),
                fontsize=10,
                title_fontsize=12
            )
            
            ax1.add_artist(legend1)  # Keep both legends
            ax1.set_title(f"{title}\nNodes: {graph_data.number_of_nodes()}, Edges: {graph_data.number_of_edges()}")
            ax1.axis('equal')
            
            # Save plot
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Saved plot to {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to plot {output_path}: {e}")
            plt.close()
            return False
    
    @staticmethod
    def graph_to_json(graph):
        """
        Convert a NetworkX graph to JSON format.
        
        Args:
            graph (NetworkX graph): The graph to convert
            
        Returns:
            dict: JSON-serializable representation of the graph
        """
        # Convert nodes to JSON format
        nodes = {}
        for node_id, node_data in graph.nodes(data=True):
            # Handle centroid conversion from torch.Tensor to list for JSON serialization
            centroid = node_data.get('centroid', [0, 0])
            if hasattr(centroid, 'tolist'):  # torch.Tensor
                centroid = centroid.tolist()
            elif hasattr(centroid, '__iter__') and not isinstance(centroid, str):  # list/tuple
                centroid = list(centroid)
            else:  # scalar or other
                centroid = [0, 0]
            
            nodes[str(node_id)] = {
                'geometry': node_data.get('geometry', []),
                'room_type': node_data.get('room_type', -1),
                'centroid': centroid
            }
        
        # Convert edges to JSON format
        edges = []
        for source, target, edge_data in graph.edges(data=True):
            edges.append({
                'source': str(source),
                'target': str(target),
                'connectivity': edge_data.get('connectivity', 'unknown')
                # Note: No 'geometry' field as it's not in pickle files
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'num_nodes': graph.number_of_nodes(),
                'num_edges': graph.number_of_edges(),
                'room_types': list(set(node_data.get('room_type', -1) for _, node_data in graph.nodes(data=True)))
            }
        }
    
    @staticmethod
    def save_graph_as_json(graph, output_path):
        """
        Save a NetworkX graph as JSON file.
        
        Args:
            graph (NetworkX graph): The graph to save
            output_path (Path): Path to save the JSON file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"💾 Saving JSON to {output_path}")
            graph_data = MSDGraph.graph_to_json(graph)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ JSON saved successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save JSON {output_path}: {e}")
            return False
    
    @staticmethod
    def load_graph_from_json(file_path):
        """
        Load graph data from a JSON file.
        
        Args:
            file_path (Path): Path to the JSON file
            
        Returns:
            NetworkX graph or None if failed
        """
        try:
            print(f"📂 Loading JSON file: {file_path.name}")
            with open(file_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            # Reconstruct NetworkX graph
            import networkx as nx
            import torch
            
            graph = nx.Graph()
            
            # Add nodes
            for node_id, node_data in graph_data['nodes'].items():
                # Convert centroid back to torch.Tensor to match pickle structure
                centroid = torch.tensor(node_data['centroid'], dtype=torch.float64)
                
                # Convert geometry lists back to tuples to match pickle structure
                geometry = [tuple(coord) for coord in node_data['geometry']]
                
                graph.add_node(int(node_id), 
                             geometry=geometry,
                             room_type=node_data['room_type'],
                             centroid=centroid)
            
            # Add edges
            for edge_data in graph_data['edges']:
                source = int(edge_data['source'])
                target = int(edge_data['target'])
                graph.add_edge(source, target, **{k: v for k, v in edge_data.items() if k not in ['source', 'target']})
            
            print(f"✅ Graph loaded from JSON: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            return graph
            
        except Exception as e:
            print(f"❌ Failed to load JSON {file_path.name}: {e}")
            return None