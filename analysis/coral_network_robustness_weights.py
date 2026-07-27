#!/usr/bin/env python3
"""
This script randomly removes nodes, weighted by the incoming edge weight,
from a graph and recalculates network metrics. 

Similar to coral_network_robustness.py

It then plots them in a nice graph

@author: jhill1; https://github.com/jhill1
@author: ia947; https://github.com/ia947
"""
#
# This work is licensed under a Creative Commons Attribution 4.0 International License.
#
# To view a copy of this license, visit creativecommons.org or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
# Copyright University of York, Isaac Abbott, 2026
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from coral_network_analysis_setup import *

# remember to change the files in coral_network_analysis_setup.py
alpha_val = 2.0
output_file = "all_regions_combined_robustness_weighted_alpha_"+str(alpha_val)+"_0.5km.pdf"


matplotlib.style.use("seaborn-v0_8-ticks")
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 7,              #
    "axes.linewidth": 0.6,       # Thin but visible
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 0.6,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
})
matplotlib.rcParams['pdf.fonttype']=42

#########################################################################
##### Simulate random node removal, including any unconnected nodes #####
#########################################################################

# Extract the work of a SINGLE iteration into its own function ---
def _single_iteration(G_original, num_nodes_to_remove, iteration_index, alpha=1.0):
    """Worker function to process a single simulation run using weighted probabilities."""
    G = G_original.copy()
    nodes = list(G.nodes())

    # 1. Calculate weights based on incoming larval flow
    # 'weight' corresponds to the edge attribute holding your larvae count
    node_weights = []
    for node in nodes:
        larval_intake = G.in_degree(node, weight="weight")

        # Apply the formula: 1 / (larval_intake + 1)^alpha
        prob_weight = 1.0 / ((larval_intake + 1) ** alpha)
        node_weights.append(prob_weight)

    # 2. Perform weighted random sampling without replacement
    nodes_to_remove = []
    while len(nodes_to_remove) < num_nodes_to_remove:
        # Over-sample with replacement to minimize loop cycles
        candidates = random.choices(
            nodes, weights=node_weights, k=num_nodes_to_remove * 2
        )
        for c in candidates:
            if c not in nodes_to_remove:
                nodes_to_remove.append(c)
            if len(nodes_to_remove) == num_nodes_to_remove:
                break

    # 3. Remove selected nodes
    G.remove_nodes_from(nodes_to_remove)

    # 4. Find and remove completely isolated nodes
    # For a directed graph, isolated means both in-degree AND out-degree are 0
    isolated_nodes = [
        node for node, deg in G.degree()
        if G.in_degree(node) == 0 and G.out_degree(node) == 0
    ]
    G.remove_nodes_from(isolated_nodes)

    # 5. Compute metrics using your custom function
    region_label = f"Iteration_{iteration_index + 1}"
    metrics = compute_network_metrics(G, region_label)

    return metrics


# Create the wrapper function to handle parallelization ---
def simulate_node_removal_random(
    G_original, removal_percentage, iterations=100, n_jobs=-1
):
    """
    Runs the random node removal in parallel. 100 iterations, 
    so running on 100 cores is fine (if you have them)
    """

    if removal_percentage > 1:
        removal_percentage /= 100.0

    num_nodes_to_remove = int(len(G_original) * removal_percentage)

    print(f"Removing {num_nodes_to_remove} nodes...")

    # Run the parallel processes (this gives us a list of 100 DataFrames)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_single_iteration)(G_original, num_nodes_to_remove, i)
        for i in tqdm(range(iterations), desc="Simulating iterations")
    )

    print("\nProcessing and merging results...")

    # 1. Add an 'iteration' column to each DataFrame so we know where the data came from
    for i, df in enumerate(results):
        df["iteration"] = i + 1

    # 2. Use pd.concat to stack all 100 DataFrames on top of each other
    combined_df = pd.concat(results, ignore_index=True)

    return combined_df


if __name__ == '__main__':

    results = []

    # --------------------------------------------------------
    # Simulation and Combined Plotting for All Regions
    # --------------------------------------------------------

    # Define simulation parameters
    total_iterations = 100 # Can reduce to 10 or 20 for quick testing
    jobs = -2 # Use all cores but leave one free

    # A container to hold normalized results across all regions
    combined_all_regions = []

    # Track which metrics are network-wide for each region
    region_network_wide_metrics = {}

    for region, filename in locations.items():
        print(f"\n{'='*40}\nProcessing Robustness Data: {region}\n{'='*40}")

        adjacency_matrix = read_adjacency_matrix(filename)
        G = create_adjacency_matrix_graph(adjacency_matrix.to_numpy())

        # Compute baseline (present day) metrics
        baseline_df = compute_network_metrics(G, region)
        baseline_means = baseline_df.drop(columns=['Node'], errors='ignore').mean(numeric_only=True)

        # Detect network-wide metrics automatically
        network_wide_metrics = []
        for col in baseline_df.columns:
            if col not in ['Node', 'iteration', 'region']:
                if baseline_df[col].nunique() == 1:
                    network_wide_metrics.append(col)

        region_network_wide_metrics[region] = network_wide_metrics
        print(f"Detected network-wide metrics for {region}: {network_wide_metrics}")

        # Loop percentages from 1 to 60
        for pct in range(1, 61):
            num_nodes_to_remove = int(len(G) * (pct / 100.0))
            print(f"Running {pct}% removal for {region}...")

            results = Parallel(n_jobs=jobs)(
                delayed(_single_iteration)(G, num_nodes_to_remove, i, alpha_val)
                for i in tqdm(range(total_iterations), desc=f"Simulating {pct}%", leave=False)
            )

            for i, df in enumerate(results):
                df["iteration"] = i + 1
            sim_df = pd.concat(results, ignore_index=True)

            # Average across nodes for each iteration
            df_no_node = sim_df.drop(columns=["Node"], errors="ignore")
            network_level_runs = df_no_node.groupby("iteration").mean(numeric_only=True).reset_index()

            # Calculate mean and SD across iterations and normalize
            numeric_cols = network_level_runs.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col == 'iteration':
                    continue

                mean_val = network_level_runs[col].mean()
                min_val = network_level_runs[col].min()  # Added
                max_val = network_level_runs[col].max()  # Added
                std_val = network_level_runs[col].std()
                baseline_val = baseline_means.get(col, np.nan)

                # Avoid dividing by zero if a baseline value is 0
                if pd.isna(baseline_val) or baseline_val == 0:
                    relative_mean = np.nan
                    relative_std = np.nan
                    relative_min = np.nan
                    relative_max = np.nan
                else:
                    relative_mean = mean_val / baseline_val
                    relative_std = std_val / baseline_val # Scale standard deviation as well
                    relative_min = min_val / baseline_val   # Normalize
                    relative_max = max_val / baseline_val   # Normalize

                combined_all_regions.append({
                    'region': region,
                    'removal_percent': pct,
                    'metric': col,
                    'mean': relative_mean,
                    'std': relative_std,
                    'min': relative_min,   # Store in Master DF
                    'max': relative_max    # Store in Master DF
                })

    # Create master dataframe from the results
    master_df = pd.DataFrame(combined_all_regions)

    # --------------------------------------------------------
    # Generate the Unified Plot
    # --------------------------------------------------------
    print("\nGenerating combined multi-region plot...")

    metrics = [m for m in master_df['metric'].unique() if m != 'iteration']
    num_metrics = len(metrics)

    # Grid layout for subplots (3 columns)
    cols = 3
    rows = (num_metrics + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(7.08, 1.77 * rows), sharex=True)
    axes = axes.flatten()

    # Color palette as requested in your script
    region_palette = {
        "GBR": "#58a141",
        "IO": "#e68193",
        "Caribbean": "#529fd6"
    }

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Present day value is always 1.0 because of relative scaling!
        ax.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='Baseline')

        for region in master_df['region'].unique():
            region_data = master_df[(master_df['region'] == region) & (master_df['metric'] == metric)].sort_values('removal_percent')

            if region_data.empty:
                continue

            pcts = region_data['removal_percent']
            means = region_data['mean']
            mins = region_data['min']  # Get min
            maxs = region_data['max']  # Get max

            colour = region_palette.get(region, "blue")

            # Mean line
            ax.plot(pcts, means, color=colour, linewidth=0.6, label=f'{region}')

            # 2. Plot the Shaded Area (Min to Max)
            # We use mins and maxs directly here
            ax.fill_between(pcts, mins, maxs, color=colour, alpha=0.15)

        ax.set_title(metric)
        ax.set_xlabel('Percentage Node Loss')
        ax.set_ylabel('Relative Value (to Baseline)')

        # Only add a legend to the very first subplot to avoid cluttering
        if i == 0:
            ax.legend(loc='lower left')

    # Hide any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Comparative Robustness Analysis (Normalized to Present Day)", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_file, dpi=72)
    plt.close()

    print("Successfully saved unified plot: "+output_file)
