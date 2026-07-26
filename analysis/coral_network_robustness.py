# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 17:39:13 2024

@author: isaac
"""

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

matplotlib.style.use("seaborn-v0_8-ticks")
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7,  #
        "axes.linewidth": 0.6,  # Thin but visible
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 0.6,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
    }
)
matplotlib.rcParams["pdf.fonttype"] = 42


#########################################################################
##### Simulate random node removal, including any unconnected nodes #####
#########################################################################


# --- Step 1: Extract the work of a SINGLE iteration into its own function ---
def _single_iteration(G_original, num_nodes_to_remove, iteration_index):
    """Worker function to process a single simulation run and measure LCC."""
    N_0 = len(G_original)
    # 1. Create a copy of the original graph so we don't destroy it
    G = G_original.copy()

    # 2. Select random nodes and remove them
    nodes = list(G.nodes())
    nodes_to_remove = random.sample(nodes, num_nodes_to_remove)
    G.remove_nodes_from(nodes_to_remove)

    # 3. Find and remove unconnected (isolated) nodes
    isolated_nodes = [node for node, degree in G.degree() if degree == 0]
    G.remove_nodes_from(isolated_nodes)

    # 4. Compute metrics using custom function
    region_label = f"Iteration_{iteration_index + 1}"
    metrics = compute_network_metrics(G, region_label)

    # --- ADDED: Compute Relative Largest Connected Component (S_LCC) ---
    if len(G) > 0:
        if G.is_directed():
            lcc_size = len(max(nx.weakly_connected_components(G), key=len))
        else:
            lcc_size = len(max(nx.connected_components(G), key=len))
    else:
        lcc_size = 0

    s_lcc = lcc_size / N_0 if N_0 > 0 else 0.0
    metrics["S_LCC"] = s_lcc

    return metrics


# --- Step 2: Create the wrapper function to handle parallelization ---
def simulate_node_removal_random(
    G_original, removal_percentage, iterations=100, n_jobs=-1
):
    if removal_percentage > 1:
        removal_percentage /= 100.0

    num_nodes_to_remove = int(len(G_original) * removal_percentage)

    print(f"Removing {num_nodes_to_remove} nodes...")

    # Run the parallel processes (this gives us a list of DataFrames)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_single_iteration)(G_original, num_nodes_to_remove, i)
        for i in tqdm(range(iterations), desc="Simulating iterations")
    )

    print("\nProcessing and merging results...")

    # 1. Add an 'iteration' column to each DataFrame
    for i, df in enumerate(results):
        df["iteration"] = i + 1

    # 2. Stack all DataFrames
    combined_df = pd.concat(results, ignore_index=True)

    return combined_df


def summarize_simulation_results(results_df):
    """Takes the massive stacked DataFrame, groups by iteration to find

    the network averages, and then summarizes the stats across all runs.
    """
    print("\nProcessing summary statistics...")

    df_no_node = results_df.drop(columns=["Node"], errors="ignore")
    network_level_runs = df_no_node.groupby("iteration").mean()

    summary = network_level_runs.describe().T
    summary = summary[["mean", "50%", "std", "min", "max"]].rename(
        columns={"50%": "median", "std": "sd"}
    )

    print("\n" + "=" * 50)
    print("      SIMULATION SUMMARY (NETWORK-WIDE MEANS)     ")
    print("=" * 50)
    print(summary.to_string(float_format="{:.5f}".format))

    return summary


############################################################
###### SIMULATE NODE REMOVAL based on a single metric ######
############################################################


def simulate_node_removal(G, region, metric="degree", removal_percent=10):
    """Simulate removal of top X% nodes by centrality metric and compute efficiency loss"""
    try:
        if metric == "degree":
            scores = nx.degree_centrality(G)
        elif metric == "eigenvector":
            scores = nx.eigenvector_centrality(G, max_iter=1000)
        elif metric == "betweenness":
            scores = nx.betweenness_centrality(G, normalized=True)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    except Exception as e:
        print(f"Error calculating {metric} centrality: {str(e)}")
        return None

    nodes = sorted(scores, key=scores.get, reverse=True)
    if not nodes:
        print("No nodes available for removal")
        return None

    n_remove = max(1, int(len(nodes) * removal_percent / 100))
    nodes_to_remove = nodes[:n_remove]

    try:
        G_removed = G.copy()
        G_removed.remove_nodes_from(nodes_to_remove)
        original_efficiency = nx.global_efficiency(G.to_undirected())
        original_edges = G.number_of_edges()
        new_efficiency = nx.global_efficiency(G_removed.to_undirected())
        edges_lost = original_edges - G_removed.number_of_edges()
    except Exception as e:
        print(f"Error during node removal: {str(e)}")
        return None

    return {
        "region": region,
        "metric": metric,
        "removal_percent": removal_percent,
        "edges_lost": edges_lost,
        "efficiency_loss_percent": (
            (1 - new_efficiency / original_efficiency) * 100
            if original_efficiency > 0
            else 0.0
        ),
        "nodes_removed": nodes_to_remove,
        "original_efficiency": original_efficiency,
        "new_efficiency": new_efficiency,
        "node_impacts": {node: scores[node] for node in nodes_to_remove},
    }


if __name__ == "__main__":

    # Configuration for different regions
    SIMULATION_PARAMS = {
        "Caribbean": {"metric": "eigenvector", "removal_percent": [5, 10, 15]},
        "IO": {"metric": "betweenness", "removal_percent": [5, 10, 15]},
        "GBR": {"metric": "degree", "removal_percent": [5, 10, 15]},
    }

    results = []

    # --------------------------------------------------------
    # Simulation and Combined Plotting for All Regions
    # --------------------------------------------------------

    total_iterations = 100  # Can reduce to 10 or 20 for quick testing
    jobs = -2  # Use all cores but leave one free

    combined_all_regions = []
    region_network_wide_metrics = {}

    for region, filename in locations.items():
        print(f"\n{'='*40}\nProcessing Robustness Data: {region}\n{'='*40}")

        adjacency_matrix = read_adjacency_matrix(filename)
        G = create_adjacency_matrix_graph(adjacency_matrix.to_numpy())

        # 1. Compute baseline (present day) metrics
        baseline_df = compute_network_metrics(G, region)

        # --- ADDED: Calculate Baseline S_LCC ---
        N_0 = len(G)
        if N_0 > 0:
            if G.is_directed():
                baseline_lcc = (
                    len(max(nx.weakly_connected_components(G), key=len)) / N_0
                )
            else:
                baseline_lcc = (
                    len(max(nx.connected_components(G), key=len)) / N_0
                )
        else:
            baseline_lcc = 0.0
        baseline_df["S_LCC"] = baseline_lcc

        baseline_means = baseline_df.drop(
            columns=["Node"], errors="ignore"
        ).mean(numeric_only=True)

        # Detect network-wide metrics automatically
        network_wide_metrics = []
        for col in baseline_df.columns:
            if col not in ["Node", "iteration", "region"]:
                if baseline_df[col].nunique() == 1:
                    network_wide_metrics.append(col)

        region_network_wide_metrics[region] = network_wide_metrics
        print(
            f"Detected network-wide metrics for {region}: {network_wide_metrics}"
        )

        # 2. Loop percentages from 1 to 60
        for pct in range(1, 61):
            num_nodes_to_remove = int(len(G) * (pct / 100.0))
            print(f"Running {pct}% removal for {region}...")

            results = Parallel(n_jobs=jobs)(
                delayed(_single_iteration)(G, num_nodes_to_remove, i)
                for i in tqdm(
                    range(total_iterations),
                    desc=f"Simulating {pct}%",
                    leave=False,
                )
            )

            for i, df in enumerate(results):
                df["iteration"] = i + 1
            sim_df = pd.concat(results, ignore_index=True)

            # Average across nodes for each iteration
            df_no_node = sim_df.drop(columns=["Node"], errors="ignore")
            network_level_runs = (
                df_no_node.groupby("iteration")
                .mean(numeric_only=True)
                .reset_index()
            )

            # Calculate mean and SD across iterations and normalize
            numeric_cols = network_level_runs.select_dtypes(
                include=[np.number]
            ).columns
            for col in numeric_cols:
                if col == "iteration":
                    continue

                mean_val = network_level_runs[col].mean()
                min_val = network_level_runs[col].min()
                max_val = network_level_runs[col].max()
                std_val = network_level_runs[col].std()
                baseline_val = baseline_means.get(col, np.nan)

                # Avoid dividing by zero if baseline value is 0
                if pd.isna(baseline_val) or baseline_val == 0:
                    relative_mean = np.nan
                    relative_std = np.nan
                    relative_min = np.nan
                    relative_max = np.nan
                else:
                    relative_mean = mean_val / baseline_val
                    relative_std = std_val / baseline_val
                    relative_min = min_val / baseline_val
                    relative_max = max_val / baseline_val

                combined_all_regions.append(
                    {
                        "region": region,
                        "removal_percent": pct,
                        "metric": col,
                        "mean": relative_mean,
                        "std": relative_std,
                        "min": relative_min,
                        "max": relative_max,
                    }
                )

    # Create master dataframe from the results
    master_df = pd.DataFrame(combined_all_regions)

    # --------------------------------------------------------
    # Generate the Unified Plot
    # --------------------------------------------------------
    print("\nGenerating combined multi-region plot...")

    metrics = [m for m in master_df["metric"].unique() if m != "iteration"]
    num_metrics = len(metrics)

    # Grid layout for subplots (3 columns)
    cols = 3
    rows = (num_metrics + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols, figsize=(7.08, 1.77 * rows), sharex=True
    )
    axes = axes.flatten()

    region_palette = {
        "GBR": "#58a141",
        "IO": "#e68193",
        "Caribbean": "#529fd6",
    }

    # Custom title lookup dictionary for panel clarity
    metric_titles = {"S_LCC": r"Largest Connected Component ($S_{\mathrm{LCC}}$)"}

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Present day value is always 1.0 because of relative scaling
        ax.axhline(
            1.0, color="red", linestyle="--", alpha=0.7, label="Baseline"
        )

        for region in master_df["region"].unique():
            region_data = master_df[
                (master_df["region"] == region) & (master_df["metric"] == metric)
            ].sort_values("removal_percent")

            if region_data.empty:
                continue

            pcts = region_data["removal_percent"]
            means = region_data["mean"]
            mins = region_data["min"]
            maxs = region_data["max"]

            colour = region_palette.get(region, "blue")

            # 1. Plot the Mean Line
            ax.plot(
                pcts, means, color=colour, linewidth=0.6, label=f"{region}"
            )

            # 2. Plot the Shaded Area (Min to Max)
            ax.fill_between(pcts, mins, maxs, color=colour, alpha=0.15)

        # Apply pretty title if mapped, otherwise use metric column name
        display_title = metric_titles.get(metric, metric)
        ax.set_title(display_title)
        ax.set_xlabel("Percentage Node Loss")
        ax.set_ylabel("Relative Value (to Baseline)")

        # Only add a legend to the very first subplot to avoid cluttering
        if i == 0:
            ax.legend(loc="lower left")

    # Hide any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        "Comparative Robustness Analysis (Normalized to Present Day)",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig("all_regions_combined_robustness_2km.pdf", dpi=72)
    plt.close()

    print(
        "Successfully saved unified plot: all_regions_combined_robustness_0.5km.pdf"
    )
