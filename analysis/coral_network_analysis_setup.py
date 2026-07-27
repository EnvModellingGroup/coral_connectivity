"""
This script sets up variables and methods used in most of the
other scripts.

The switch to compare the different GBR is all done by hand and 
carefully checking things

@author: jhill1; https://github.com/jhill1
@author: ia947; https://github.com/ia947
"""
# This work is licensed under a Creative Commons Attribution 4.0 International License.
#
# To view a copy of this license, visit creativecommons.org or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
# Copyright University of York, Isaac Abbott, 2026
import os
import networkx as nx
import pandas as pd
import numpy as np

#################################
###### DATA PRE-PROCESSING ######
#################################

# Define locations for corresponding adjacency matrix filenames
# NOTE GBR NEEDS altering for each resolution by hand. See below.
# Then run the network stats, vis and robustness scripts, etc
locations = {
    "GBR": os.path.join("../../modern_0.5km", "gbr_connectivity_decimal_G.retiformis.csv"),
    "IO": os.path.join("../data/", "IO_single_step_explicit_mean_connectivity_matrix.csv"),
    "Caribbean": os.path.join("../data", "Caribbean_matrix.csv"),
    }

# Comment this out if you want to compare the three GBR models instead.
#locations = {
#    "GBR_05": os.path.join("../../modern_0.5km", "gbr_connectivity_decimal_G.retiformis.csv"),
#    "GBR_2": os.path.join("../../modern_2km", "gbr_connectivity_decimal_G.retiformis.csv"),
#    "GBR_9": os.path.join("../../modern_9km", "gbr_connectivity_decimal_G.retiformis.csv"),
#    }

def read_adjacency_matrix(filename):
    """
    Reads in one of the standardised matricies
    """
    if filename.endswith(".csv"):
        return pd.read_csv(filename, index_col=0, header=0)
    raise ValueError(f"Unsupported file format: {filename}")

# Function to create directed graph from connectivity matrix
def create_adjacency_matrix_graph(adjacency_matrix):
    """
    Turns a matrix (numpy array) into a graph network
    """

    G = nx.DiGraph()
    num_nodes = len(adjacency_matrix)
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):  # Include all connections (i -> j)
            if adjacency_matrix[i][j] > 0:
                G.add_edge(i, j, weight=adjacency_matrix[i][j])
    return G

###############################
###### NETWORK MEASURES #######
###############################

# Function to compute all network measures
def compute_network_metrics(G):
    """
    Computes the network metrics and outputs a dataframe of the nodes. 
    For some metrics the same value will be at all nodes as they are
    network-wide metrics
    """

    # Centrality measures
    degree_centrality = nx.degree_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, normalized=True)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    harmonic_centrality = nx.harmonic_centrality(G)
    clustering_coefficient = nx.clustering(G.to_undirected())

    # Graph-level measures
    density = nx.density(G)

    G_no_selfloops = G.copy()  # Create copy of the graph
    G_no_selfloops.remove_edges_from(nx.selfloop_edges(G_no_selfloops))  # Remove self-loops
    if G_no_selfloops.number_of_edges() > 0:  # Ensure the graph has edges
        rich_club_coefficient = nx.rich_club_coefficient(G_no_selfloops.to_undirected(),
                                                         normalized=False)
        avg_rich_club = np.mean(list(rich_club_coefficient.values()))
    else:
        avg_rich_club = 0  # If no valid calculation, default to 0

    transitivity = nx.transitivity(G)
    local_efficiency = nx.local_efficiency(G.to_undirected())

    # Compute network centralisation
    degree_centrality = nx.degree_centrality(G)
    max_centrality = max(degree_centrality.values())
    N = len(G)
    network_centralisation = (sum(max_centrality - c for c in degree_centrality.values()) / (N - 2)) if N > 2 else 0

    # Create the DataFrame with all metrics
    metrics_df = pd.DataFrame({
        'Node': list(G.nodes),
        'Degree Centrality': [degree_centrality[node] for node in G.nodes()],
        'Network Centralisation': [network_centralisation] * len(G.nodes()),
        'Closeness Centrality': [closeness_centrality[node] for node in G.nodes()],
        'Betweenness Centrality': [betweenness_centrality[node] for node in G.nodes()],
        'Eigenvector Centrality': [eigenvector_centrality[node] for node in G.nodes()],
        'Harmonic Centrality': [harmonic_centrality[node] for node in G.nodes()],
        'Clustering Coefficient': [clustering_coefficient[node] for node in G.nodes()],
        'Graph Density': [density] * len(G.nodes()),
        'Rich Club Coefficient': [avg_rich_club] * len(G.nodes()),
        'Transitivity': [transitivity] * len(G.nodes()),
        'Local Efficiency': [local_efficiency] * len(G.nodes())
    })

    return metrics_df
