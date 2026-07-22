import os
import networkx as nx
import pandas as pd
import numpy as np

#################################
###### DATA PRE-PROCESSING ######
#################################

# Define locations for corresponding adjacency matrix filenames
# NOTE GBR NEEDS altering for each resolution. Then run the network stats, vis and robustness scripts
locations = {
    "GBR": os.path.join("../../modern_2km", "gbr_connectivity_decimal_G.retiformis.csv"),
    "IO": os.path.join("../data/", "IO_single_step_explicit_mean_connectivity_matrix.csv"),
    "Caribbean": os.path.join("../data", "Caribbean_matrix.csv"),
    }


def read_adjacency_matrix(filename):
    if filename.endswith(".csv"):
        return pd.read_csv(filename, index_col=0, header=0)
    else:
        raise ValueError(f"Unsupported file format: {filename}")

# Function to create directed graph from connectivity matrix
def create_adjacency_matrix_graph(adjacency_matrix):
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
def compute_network_metrics(G, region_name):
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
        rich_club_coefficient = nx.rich_club_coefficient(G_no_selfloops.to_undirected(), normalized=False)
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
    
    # Save results to csv
    #output_filename = f"{region_name}_network_metrics.csv"
    #metrics_df.to_csv(output_filename, index=False, mode="w")
    #print(f"Metrics for {region_name} saved to {output_filename}")
    
    return metrics_df


