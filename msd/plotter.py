"""
MSD (floor plan) visualization utilities.

Provides the MSDPlotter class for plotting floor plans with room shapes,
access graphs, adjacency/connectivity graphs, and zoning visualizations.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap


class MSDPlotter:
    """
    Class for plotting MSD (floor plan) data structures.
    
    This class handles visualization of:
    - Floor plans with room shapes and access graphs
    - Adjacency and connectivity graphs
    - Zoning and room type visualizations
    """
    
    # Subtype mappings
    ZONING_MAPPING = {
        'ROOM': 'Zone1',
        'BEDROOM': 'Zone1',
        'BALCONY': 'Zone4',
        'TERRACE': 'Zone4',
        'KITCHEN': 'Zone2',
        'DINING': 'Zone2',
        'KITCHEN_DINING': 'Zone2',
        'LIVING_ROOM': 'Zone2',
        'LIVING_DINING': 'Zone2',
        'BATHROOM': 'Zone3',
        'CORRIDOR': 'Zone2',
        'CORRIDORS_AND_HALLS': 'Zone2',
        'STAIRS': 'Zone3',
        'STAIRCASE': 'Zone3',
        'ELEVATOR': 'Zone3',
        'RAILING': 'Zone4',
        'VOID': 'Zone3',
        'SHAFT': 'Structure',
        'WALL': 'Structure',
        'COLUMN': 'Structure',
        'STOREROOM': 'Zone3',
        'ENTRANCE_DOOR': 'Entrance Door',
        'DOOR': 'Door',
        'WINDOW': 'Window'
    }
    
    ROOM_MAPPING = {
        'ROOM': 'Bedroom',
        'BEDROOM': 'Bedroom',
        'BALCONY': 'Balcony',
        'TERRACE': 'Balcony',
        'KITCHEN': 'Kitchen',
        'DINING': 'Dining',
        'KITCHEN_DINING': 'Kitchen',
        'LIVING_ROOM': 'Livingroom',
        'LIVING_DINING': 'Livingroom',
        'BATHROOM': 'Bathroom',
        'CORRIDOR': 'Corridor',
        'CORRIDORS_AND_HALLS': 'Corridor',
        'STAIRS': 'Stairs',
        'STAIRCASE': 'Stairs',
        'ELEVATOR': 'Stairs',
        'RAILING': 'Balcony',
        'VOID': 'Stairs',
        'SHAFT': 'Structure',
        'WALL': 'Structure',
        'COLUMN': 'Structure',
        'STOREROOM': 'Storeroom',
        'ENTRANCE_DOOR': 'Entrance Door',
        'DOOR': 'Door',
        'WINDOW': 'Window'
    }
    
    ZONING_NAMES = ['Zone1', 'Zone2', 'Zone3', 'Zone4', 'Structure', 
                    'Door', 'Entrance Door', 'Window']
    
    try:
        from graph import RoomType
        ROOM_NAMES = [room_type.to_name() for room_type in RoomType]
    except ImportError:
        ROOM_NAMES = ['Bedroom', 'Livingroom', 'Kitchen', 'Dining', 'Corridor', 
                      'Stairs', 'Storeroom', 'Bathroom', 'Balcony', 'Structure', 
                      'Door', 'Entrance Door', 'Window']
    
    ZONING_ROOMS = {
        'Zone1': ['Bedroom'],
        'Zone2': ['Livingroom', 'Kitchen', 'Dining', 'Corridor'],
        'Zone3': ['Stairs', 'Storeroom', 'Bathroom'],
        'Zone4': ['Balcony'],
        'Structure': ['Structure'],
        'Door': ['Door'],
        'Entrance Door': ['Entrance Door'],
        'Window': ['Window']
    }
    
    # Color schemes
    COLORS_ZONING = ['#1f77b4', '#ff7f0e', '#72246c', '#2ca02c', 
                     '#000000', '#ffc000', '#98df8a', '#d62728']
    
    COLORS_ROOMTYPE = ['#1f77b4', '#e6550d', '#fd8d3c', '#fdae6b', 
                       '#fdd0a2', '#72246c', '#5254a3', '#6b6ecf', 
                       '#2ca02c', '#000000', '#ffc000', '#98df8a', '#d62728']
    
    COLOR_MAP_ZONING = mcolors.ListedColormap(COLORS_ZONING)
    CMAP_ZONING = get_cmap(COLOR_MAP_ZONING)

    COLOR_MAP_ROOMTYPE = mcolors.ListedColormap(COLORS_ROOMTYPE)
    CMAP_ROOMTYPE = get_cmap(COLOR_MAP_ROOMTYPE)

    # Plotting constants
    FS = 10
    AWESOME_COLORS = ["#1932E1"]
    COLORSET = 'tab20'
    
    def __init__(self, cmap_zoning=None, cmap_roomtype=None):
        """
        Initialize the MSD plotter.
        
        Args:
            cmap_zoning: Colormap for zoning visualization (if None, creates from COLORS_ZONING)
            cmap_roomtype: Colormap for room type visualization (if None, creates from COLORS_ROOMTYPE)
        """
        if cmap_zoning is None:
            color_map_zoning = mcolors.ListedColormap(self.COLORS_ZONING)
            self.cmap_zoning = get_cmap(color_map_zoning)
        else:
            self.cmap_zoning = cmap_zoning
        
        if cmap_roomtype is None:
            color_map_roomtype = mcolors.ListedColormap(self.COLORS_ROOMTYPE)
            self.cmap_roomtype = get_cmap(color_map_roomtype)
        else:
            self.cmap_roomtype = cmap_roomtype
    
    @staticmethod
    def set_figure(nc, nr, fs=10, fs_title=7.5, fs_legend=10, 
                   fs_xtick=3, fs_ytick=3, fs_axes=4, ratio=1):
        """
        Custom figure setup function that generates a nicely looking figure outline.
        
        Includes "making-sense" fontsizes across all text locations (e.g. title, axes).
        You can always change things later yourself through the outputs or plt.rc(...).
        
        Args:
            nc: Number of columns
            nr: Number of rows
            fs: Base font size
            fs_title: Title font size multiplier
            fs_legend: Legend font size multiplier
            fs_xtick: X-tick font size multiplier
            fs_ytick: Y-tick font size multiplier
            fs_axes: Axes label font size multiplier
            ratio: Aspect ratio
            
        Returns:
            Tuple of (fig, axs)
        """
        fig, axs = plt.subplots(ncols=nc, nrows=nr, figsize=(fs*nc*ratio, fs*nr))
        
        try:
            axs = axs.flatten()
        except:
            pass
        
        plt.rc("figure", titlesize=fs*fs_title)
        plt.rc("legend", fontsize=fs*fs_legend)
        plt.rc("xtick", labelsize=fs*fs_xtick)
        plt.rc("ytick", labelsize=fs*fs_ytick)
        plt.rc("axes", labelsize=fs*fs_axes, titlesize=fs*fs_title)
        
        return fig, axs
    
    @staticmethod
    def plot_polygon(ax, poly, label=None, **kwargs):
        """
        Plot a single polygon on the given axis.
        
        Args:
            ax: Matplotlib axis
            poly: Shapely Polygon object
            label: Optional label for the polygon
            **kwargs: Additional matplotlib fill parameters
        """
        x, y = poly.exterior.xy
        ax.fill(x, y, label=label, **kwargs)
    
    @staticmethod
    def plot_shapes(ax, polygons, colors, **kwargs):
        """
        Plot multiple polygons with specified colors.
        
        Args:
            ax: Matplotlib axis
            polygons: List of polygon coordinates
            colors: List of colors for each polygon
            **kwargs: Additional matplotlib fill parameters
        """
        from shapely.geometry import Polygon
        
        for poly, color in zip(polygons, colors):
            MSDPlotter.plot_polygon(ax, Polygon(poly), color=color, **kwargs)
    
    @staticmethod
    def plot_graph(G, ax, c_node='black', c_edge=['white']*4, dw_edge=False, 
                   pos=None, node_size=10, edge_size=10):
        """
        Plot the adjacency or access graph of a floor plan's corresponding graph structure.
        
        Args:
            G: NetworkX graph
            ax: Matplotlib axis
            c_node: Node color
            c_edge: List of edge colors [default, passage, door, entrance]
            dw_edge: If True, draw edges with different styles based on connectivity type
            pos: Node positions (if None, uses spring layout)
            node_size: Size of nodes
            edge_size: Width of edges
        """
        if pos is None:
            pos = nx.spring_layout(G, seed=7)
        
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=c_node, ax=ax)
        
        if dw_edge:
            epass = [(u, v) for (u, v, d) in G.edges(data=True) if d["connectivity"] == 'passage']
            edoor = [(u, v) for (u, v, d) in G.edges(data=True) if d["connectivity"] == 'door']
            efront = [(u, v) for (u, v, d) in G.edges(data=True) if d["connectivity"] == 'entrance']
            
            nx.draw_networkx_edges(G, pos, edgelist=epass, edge_color=c_edge[1],
                                   width=edge_size, ax=ax)
            nx.draw_networkx_edges(G, pos, edgelist=edoor, edge_color=c_edge[2],
                                   width=edge_size, style="dashed", ax=ax)
            nx.draw_networkx_edges(G, pos, edgelist=efront, edge_color=c_edge[3],
                                   width=edge_size, style="-.", ax=ax)
        else:
            nx.draw_networkx_edges(G, pos, edge_color=c_edge[0],
                                   width=edge_size, ax=ax)
        
        ax.axis('off')
    
    def plot_floor(self, G, ax, node_size=50, edge_size=3):
        """
        Plot a floor plan's corresponding access graph including room shapes.
        
        Args:
            G: NetworkX graph with room geometry and type information
            ax: Matplotlib axis
            node_size: Size of nodes
            edge_size: Width of edges
        """
        if G.number_of_nodes() == 0:
            ax.text(0.5, 0.5, 'Empty Graph', ha='center', va='center', 
                   transform=ax.transAxes)
            return
        
        first_node = next(iter(G.nodes()))
        attribute_names = list(G.nodes[first_node].keys())
        column = 'room_type' if 'room_type' in attribute_names else 'zoning_type'
        cmap = self.cmap_roomtype if column == 'room_type' else self.cmap_zoning
        
        from shapely.geometry import Polygon
        
        shapes = [Polygon(n) for _, n in G.nodes('geometry')]
        colors = [np.array(cmap(n)).reshape(1, 4) for _, n in G.nodes(column)]
        pos = {n: np.array(G.nodes[n]['centroid']) for n in G.nodes}
        
        self.plot_shapes(ax, shapes, colors, ec="black", lw=0, alpha=1)
        
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='black', ax=ax)
        
        edges = [(u, v) for (u, v, d) in G.edges(data="connectivity") 
                if d in ["door", "passage"]]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='black',
                              width=edge_size, ax=ax)
        
        edges = [(u, v) for (u, v, d) in G.edges(data="connectivity") 
                if d == "entrance"]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='red',
                              width=edge_size*2, ax=ax)
