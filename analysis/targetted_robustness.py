#!/usr/bin/env python3
#
# This work is licensed under a Creative Commons Attribution 4.0 International License.
#
# To view a copy of this license, visit creativecommons.org or send a letter to Creative 
# Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
# Copyright University of York, Isaac Abbott, 2026
import os
import random
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from coral_network_analysis_setup import *
from joblib import Parallel, delayed
from tqdm import tqdm

"""
Removal of nodes in graph based on the highest ranking node

We look at betweeness, eigenvector and degree centrality

Generates a single plot.

@author: jhill1; https://github.com/jhill1
@author: ia947; https://github.com/ia947
"""


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
# Core Attack Strategy Functions
# -----------------------------------------------------------------------------

def compute_centrality_ranking(G, strategy):
    """Computes node ranking based on the selected network strategy."""
    if strategy == "degree":
        scores = nx.degree_centrality(G)
    elif strategy == "betweenness":
        scores = nx.betweenness_centrality(G, normalized=True)
    elif strategy == "eigenvector":
        try:
            scores = nx.eigenvector_centrality(G, max_iter=2000)
        except nx.PowerIterationFailedConvergence:
            # Fallback to degree if eigenvector fails to converge
            scores = nx.degree_centrality(G)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Return list of nodes sorted by centrality score descending
    return [node for node, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def get_network_state(G, N_0, initial_efficiency):
    """Calculates S_LCC and Global Efficiency relative to original graph state."""
    if len(G) == 0:
        return {"S_LCC": 0.0, "rel_efficiency": 0.0}

    # Relative LCC size
    if G.is_directed():
        lcc_size = len(max(nx.weakly_connected_components(G), key=len))
    else:
        lcc_size = len(max(nx.connected_components(G), key=len))
    s_lcc = lcc_size / N_0

    # Global Efficiency relative to initial
    try:
        curr_eff = nx.global_efficiency(G.to_undirected())
        rel_eff = curr_eff / initial_efficiency if initial_efficiency > 0 else 0.0
    except Exception:
        rel_eff = 0.0

    return {"S_LCC": s_lcc, "rel_efficiency": rel_eff}


def run_targeted_attack(G_original, strategy, max_pct=60):
    """Simulates targeted removal using static ranking based on initial graph centrality."""
    N_0 = len(G_original)
    G_undirected = G_original.to_undirected()
    initial_eff = nx.global_efficiency(G_undirected)

    # Pre-rank nodes based on initial network topology
    ranked_nodes = compute_centrality_ranking(G_original, strategy)

    results = []
    for pct in range(1, max_pct + 1):
        num_remove = int(N_0 * (pct / 100.0))
        nodes_to_remove = ranked_nodes[:num_remove]

        G = G_original.copy()
        G.remove_nodes_from(nodes_to_remove)

        # Remove newly isolated nodes
        isolated = [node for node, deg in G.degree() if deg == 0]
        G.remove_nodes_from(isolated)

        state = get_network_state(G, N_0, initial_eff)
        results.append(
            {
                "removal_percent": pct,
                "strategy": strategy,
                "S_LCC": state["S_LCC"],
                "rel_efficiency": state["rel_efficiency"],
            }
        )

    return pd.DataFrame(results)


def _single_random_run(G_original, num_remove, N_0, initial_eff):
    """Worker function for single random removal iteration."""
    G = G_original.copy()
    nodes_to_remove = random.sample(list(G.nodes()), num_remove)
    G.remove_nodes_from(nodes_to_remove)

    isolated = [node for node, deg in G.degree() if deg == 0]
    G.remove_nodes_from(isolated)

    return get_network_state(G, N_0, initial_eff)


def run_random_attack(G_original, iterations=50, max_pct=60, n_jobs=-2):
    """Simulates random removal with Monte Carlo iterations."""
    N_0 = len(G_original)
    initial_eff = nx.global_efficiency(G_original.to_undirected())

    results = []
    for pct in range(1, max_pct + 1):
        num_remove = int(N_0 * (pct / 100.0))

        runs = Parallel(n_jobs=n_jobs)(
            delayed(_single_random_run)(G_original, num_remove, N_0, initial_eff)
            for _ in range(iterations)
        )

        s_lcc_vals = [r["S_LCC"] for r in runs]
        eff_vals = [r["rel_efficiency"] for r in runs]

        results.append(
            {
                "removal_percent": pct,
                "strategy": "Random",
                "S_LCC_mean": np.mean(s_lcc_vals),
                "S_LCC_min": np.min(s_lcc_vals),
                "S_LCC_max": np.max(s_lcc_vals),
                "rel_efficiency_mean": np.mean(eff_vals),
                "rel_efficiency_min": np.min(eff_vals),
                "rel_efficiency_max": np.max(eff_vals),
            }
        )

    return pd.DataFrame(results)


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    MAX_REMOVAL_PCT = 60
    RANDOM_ITERATIONS = 100
    JOBS = -2

    strategies = ["degree", "betweenness", "eigenvector"]
    
    # Palette mapping for strategies
    strategy_colors = {
        "Random": "#7f7f7f",       # Neutral Grey
        "degree": "#d62728",       # Red (Hubs)
        "betweenness": "#ff7f0e",  # Orange (Stepping Stones)
        "eigenvector": "#1f77b4",  # Blue (Clusters)
    }

    all_results = []

    for region, filename in locations.items():
        print(f"\n{'='*50}\nSimulating Targeted Attacks for Region: {region}\n{'='*50}")

        adjacency_matrix = read_adjacency_matrix(filename)
        G = create_adjacency_matrix_graph(adjacency_matrix.to_numpy())

        # 1. Run Random Removal Baseline
        print("Running Random removal baseline...")
        df_random = run_random_attack(
            G, iterations=RANDOM_ITERATIONS, max_pct=MAX_REMOVAL_PCT, n_jobs=JOBS
        )
        df_random["region"] = region
        
        # Format random results for uniform concatenation
        df_random_formatted = pd.DataFrame({
            "region": region,
            "removal_percent": df_random["removal_percent"],
            "strategy": "Random",
            "S_LCC": df_random["S_LCC_mean"],
            "S_LCC_min": df_random["S_LCC_min"],
            "S_LCC_max": df_random["S_LCC_max"],
            "rel_efficiency": df_random["rel_efficiency_mean"],
        })
        all_results.append(df_random_formatted)

        # 2. Run Targeted Attacks
        for strat in strategies:
            print(f"Running Targeted Attack strategy: '{strat}'...")
            df_strat = run_targeted_attack(G, strategy=strat, max_pct=MAX_REMOVAL_PCT)
            df_strat["region"] = region
            df_strat["S_LCC_min"] = df_strat["S_LCC"]
            df_strat["S_LCC_max"] = df_strat["S_LCC"]
            all_results.append(df_strat)

    master_df = pd.concat(all_results, ignore_index=True)

    # -------------------------------------------------------------------------
    # Plotting: Multi-Panel Region Comparison
    # -------------------------------------------------------------------------
    print("\nGenerating Targeted Attack comparative figure...")

    unique_regions = list(locations.keys())
    num_regions = len(unique_regions)

    # 2 rows (Top: S_LCC, Bottom: Global Efficiency) x N columns (Regions)
    fig, axes = plt.subplots(2, num_regions, figsize=(7.08, 3.8), sharex=True, sharey="row")

    for col_idx, region in enumerate(unique_regions):
        ax_lcc = axes[0, col_idx] if num_regions > 1 else axes[0]
        ax_eff = axes[1, col_idx] if num_regions > 1 else axes[1]

        # Filter region data
        reg_df = master_df[master_df["region"] == region]

        for strat in ["Random"] + strategies:
            strat_df = reg_df[reg_df["strategy"] == strat].sort_values("removal_percent")
            color = strategy_colors.get(strat, "black")
            label_name = strat.capitalize() if strat != "Random" else "Random Loss"

            # Plot S_LCC (Top row)
            ax_lcc.plot(
                strat_df["removal_percent"],
                strat_df["S_LCC"],
                color=color,
                linewidth=0.8,
                label=label_name,
            )
            # Add shaded band for Random variability
            if strat == "Random":
                ax_lcc.fill_between(
                    strat_df["removal_percent"],
                    strat_df["S_LCC_min"],
                    strat_df["S_LCC_max"],
                    color=color,
                    alpha=0.2,
                )

            # Plot Global Efficiency (Bottom row)
            ax_eff.plot(
                strat_df["removal_percent"],
                strat_df["rel_efficiency"],
                color=color,
                linewidth=0.8,
                label=label_name,
            )

        # Labels & Formatting
        ax_lcc.set_title(f"{region}", fontsize=8, fontweight="bold")
        ax_eff.set_xlabel("Node Loss (%)")

        if col_idx == 0:
            ax_lcc.set_ylabel(r"Relative $S_{\mathrm{LCC}}$")
            ax_eff.set_ylabel("Relative Global Efficiency")
            ax_lcc.legend(loc="lower left", frameon=True, fontsize=6)

    plt.suptitle("Network Collapse Trajectories: Random vs. Targeted Attacks", fontsize=9)
    plt.tight_layout()

    output_plot_path = "targeted_attack_resilience_comparison.pdf"
    plt.savefig(output_plot_path, dpi=300)
    plt.close()

    print(f"Successfully generated and saved plot: {output_plot_path}")
