# -*- coding: utf-8 -*-
"""
Within-Basin Modularity & Community Detection for Coral Reef Networks

Created on Sun 2026
@author: isaac / AI Collaborator
"""

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from coral_network_analysis_setup import *

# --- Matplotlib Styling Settings ---
matplotlib.style.use("seaborn-v0_8-ticks")
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 0.8,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
    }
)
matplotlib.rcParams["pdf.fonttype"] = 42


# -----------------------------------------------------------------------------
# Modularity & Community Detection Analysis
# -----------------------------------------------------------------------------

def analyze_basin_modularity(G, region_name, resolution=1.0, seed=42):
    """
    Performs Louvain community detection and computes modularity (Q) for a basin network.
    
    Parameters:
    -----------
    G : nx.DiGraph
        Directed connectivity graph for the basin.
    region_name : str
        Name of the region/basin.
    resolution : float
        Resolution parameter for Louvain community detection.
    seed : int
        Random seed for deterministic community detection.
        
    Returns:
    --------
    summary_dict : dict
        Regional modularity and community summary stats.
    node_community_df : pd.DataFrame
        DataFrame mapping each node ID to its detected module/community.
    community_sizes : list
        List of sizes (node counts) for each detected community.
    """
    N_0 = len(G)
    E_0 = G.number_of_edges()

    if N_0 == 0:
        return None, None, []

    # Convert to undirected representation with weighted edge aggregation for Louvain
    G_undirected = G.to_undirected(reciprocal=False)
    
    # 1. Detect Communities using Louvain Algorithm
    try:
        communities = nx.community.louvain_communities(
            G_undirected, weight="weight", resolution=resolution, seed=seed
        )
    except Exception:
        # Fallback to greedy modularity if Louvain fails
        communities = list(nx.community.greedy_modularity_communities(G_undirected, weight="weight"))

    # 2. Compute Modularity Score (Q) on Directed Network
    try:
        Q_score = nx.community.modularity(G, communities, weight="weight")
    except Exception:
        Q_score = nx.community.modularity(G_undirected, communities, weight="weight")

    # 3. Community Metrics Extraction
    num_communities = len(communities)
    community_sizes = sorted([len(c) for c in communities], reverse=True)
    largest_module_size = community_sizes[0] if community_sizes else 0
    largest_module_pct = (largest_module_size / N_0) * 100 if N_0 > 0 else 0.0

    # Create mapping of Node -> Community ID
    node_comm_map = {}
    for comm_id, comm_nodes in enumerate(communities, start=1):
        for node in comm_nodes:
            node_comm_map[node] = comm_id

    node_community_df = pd.DataFrame(
        list(node_comm_map.items()), columns=["Node", "Community_ID"]
    )
    node_community_df["region"] = region_name

    summary_dict = {
        "region": region_name,
        "num_nodes": N_0,
        "num_edges": E_0,
        "num_communities": num_communities,
        "modularity_Q": Q_score,
        "largest_module_size": largest_module_size,
        "largest_module_pct": largest_module_pct,
        "mean_module_size": np.mean(community_sizes) if community_sizes else 0.0,
        "std_module_size": np.std(community_sizes) if community_sizes else 0.0,
    }

    return summary_dict, node_community_df, community_sizes


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    region_palette = {
        "GBR": "#58a141",
        "IO": "#e68193",
        "Caribbean": "#529fd6",
    }

    basin_summaries = []
    all_node_communities = []
    basin_community_distributions = {}

    for region, filename in locations.items():
        print(f"\n{'='*50}\nAnalyzing Modularity & Community Structure: {region}\n{'='*50}")

        adjacency_matrix = read_adjacency_matrix(filename)
        G = create_adjacency_matrix_graph(adjacency_matrix.to_numpy())

        summary, node_comm_df, comm_sizes = analyze_basin_modularity(
            G, region_name=region, resolution=1.0, seed=42
        )

        if summary is not None:
            basin_summaries.append(summary)
            all_node_communities.append(node_comm_df)
            basin_community_distributions[region] = comm_sizes

            print(f"Nodes: {summary['num_nodes']} | Edges: {summary['num_edges']}")
            print(f"Detected Communities: {summary['num_communities']}")
            print(f"Modularity Score (Q): {summary['modularity_Q']:.4f}")
            print(f"Largest Module Size: {summary['largest_module_size']} nodes ({summary['largest_module_pct']:.1f}%)")

    # Combine summaries into Master DataFrames
    summary_df = pd.DataFrame(basin_summaries)
    master_node_comm_df = pd.concat(all_node_communities, ignore_index=True)

    # Save CSV Outputs for MPA Planning / Spatial GIS mapping
    summary_df.to_csv("basin_modularity_summary.csv", index=False)
    master_node_comm_df.to_csv("node_community_assignments.csv", index=False)
    print("\nSaved modularity summaries to 'basin_modularity_summary.csv'")
    print("Saved node assignments to 'node_community_assignments.csv'")

    # -------------------------------------------------------------------------
    # Visualization: Publication Multi-Panel Figure
    # -------------------------------------------------------------------------
    print("\nGenerating modularity and community distribution figure...")

    fig, axes = plt.subplots(1, 3, figsize=(7.08, 2.5))

    regions = summary_df["region"].tolist()
    colors = [region_palette.get(r, "#333333") for r in regions]

    # Panel A: Modularity Score (Q) Comparison
    ax1 = axes[0]
    bars1 = ax1.bar(regions, summary_df["modularity_Q"], color=colors, width=0.5, edgecolor="none", alpha=0.85)
    ax1.set_ylabel("Modularity Score ($Q$)")
    ax1.set_title("A. Basin Modularity ($Q$)", fontsize=8, fontweight="bold")
    ax1.set_ylim(0, max(summary_df["modularity_Q"]) * 1.25 if not summary_df.empty else 1.0)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.3f}", ha='center', va='bottom', fontsize=6)

    # Panel B: Number of Detected Sub-Communities
    ax2 = axes[1]
    bars2 = ax2.bar(regions, summary_df["num_communities"], color=colors, width=0.5, edgecolor="none", alpha=0.85)
    ax2.set_ylabel("Number of Modules ($k$)")
    ax2.set_title("B. Community Count", fontsize=8, fontweight="bold")
    ax2.set_ylim(0, max(summary_df["num_communities"]) * 1.25 if not summary_df.empty else 10)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{int(yval)}", ha='center', va='bottom', fontsize=6)

    # Panel C: Rank-Size Distribution of Communities
    ax3 = axes[2]
    for region in regions:
        sizes = basin_community_distributions.get(region, [])
        if sizes:
            ranks = np.arange(1, len(sizes) + 1)
            ax3.plot(
                ranks,
                sizes,
                marker="o",
                markersize=3,
                linewidth=0.8,
                color=region_palette.get(region, "black"),
                label=region,
            )

    ax3.set_xlabel("Module Rank")
    ax3.set_ylabel("Module Size (# Reef Nodes)")
    ax3.set_title("C. Module Size Rank-Distribution", fontsize=8, fontweight="bold")
    ax3.legend(loc="upper right", frameon=True, fontsize=6)

    plt.tight_layout()
    
    output_plot_path = "within_basin_modularity_analysis.pdf"
    plt.savefig(output_plot_path, dpi=300)
    plt.close()

    print(f"Successfully generated and saved plot: {output_plot_path}")
