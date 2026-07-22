
import os
import numpy as np
import networkx as nx
import graphviz
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point
import geopandas
import matplotlib
from coral_network_analysis_setup import *
matplotlib.style.use("seaborn-v0_8-ticks")
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 7,              # Nature prefers 5-7 pt
    "axes.linewidth": 0.6,       # Thin but visible
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 0.6,
    "axes.labelsize": 5,
    "axes.titlesize": 6,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
})
matplotlib.rcParams['pdf.fonttype']=42

####################################
###### NETWORK VISUALISATION #######
####################################

# Function to draw the directed graph
def draw_graph(G, use_graphviz=False):

    fig, axs = plt.subplots(ncols=3, figsize=(7, 3))
    cmap = plt.cm.plasma  # Use a perceptually uniform colormap
    fixed_node_size = 8  # Set a fixed size for all nodes
    min_centrality = 0
    max_centrality = 10000000
    i = 0
    for g in G:

        if use_graphviz:
            try:
                pos = graphviz_layout(g, prog='sfdp')  # Use sfdp for sprawling layout
            except ImportError:
                print("Graphviz is not installed. Falling back to spring_layout.")
                pos = nx.spring_layout(g, seed=42, k=0.3, iterations=100)
        else:
            pos = nx.spring_layout(g, seed=42, k=0.3, iterations=100)  # Adjust k for spread
    
        # Compute degree centrality for coloring
        closeness_centrality = nx.closeness_centrality(g)
        centrality_values = list(closeness_centrality.values())
        max_centrality = min(max_centrality, max(centrality_values))
        min_centrality = max(min_centrality, min(centrality_values))
    
        # Node properties
        node_color = [closeness_centrality[node] for node in g.nodes()]
    
        # Edge properties
        edge_weights = nx.get_edge_attributes(g, 'weight')
        edge_width = [0.001 + 2 * edge_weights[edge] for edge in g.edges()]  # Scale edge width by weight
    
        # Draw graph
        nx.draw(
            g,
            pos,
            node_size=fixed_node_size,
            node_color=node_color,
            cmap=cmap,
            edge_color='black',
            width=0.08,#edge_width,
            alpha=1.0,
            arrows=False,
            arrowsize=2,
            ax = axs[i]
        )

        i = i + 1
    
    # Add colorbar for node centrality
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_centrality, vmax=max_centrality))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), orientation='horizontal',fraction=0.046, pad=0.04)
    cbar.set_label("Closeness Centrality", fontsize=6)
    plt.tight_layout()
    plt.savefig("centrality_network.pdf",dpi=300)

##############################################
####### EXECUTE SCRIPT FOR ALL REGIONS #######
##############################################

G = []
for region, filename in locations.items():
    print(f"Processing {region}...")
    
    # Read and process the adjacency matrix
    adjacency_matrix = read_adjacency_matrix(filename)

    G.append(create_adjacency_matrix_graph(adjacency_matrix.to_numpy()))
    
# Visualise the graph
draw_graph(G, use_graphviz=True)
    
    # Compute and save the network metrics
    #metrics_df = compute_network_metrics(G, region)
    # attach coords and names to metrics
    #metrics_df = metrics_df.set_index(adjacency_matrix.index)
    #coords = pd.read_csv(os.path.join("../data/",region+"_coords.csv"),index_col=0,header=0)
    #metrics_df = metrics_df.join(coords)
    #metrics_df['geometry'] = metrics_df.apply(lambda x: Point((float(x.Lon), float(x.Lat))), axis=1)
    #metrics_df = geopandas.GeoDataFrame(metrics_df, geometry='geometry')

    # dump to shapefile
    #metrics_df.to_file(os.path.join("../data/",region+'_geometries.shp'), driver='ESRI Shapefile')

