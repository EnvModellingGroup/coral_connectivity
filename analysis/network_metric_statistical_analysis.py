# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:01:43 2025

@author: isaac
"""

import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import plotly.figure_factory as ff
import networkx as nx
from coral_network_analysis_setup import *
from shapely.geometry import Point
import geopandas
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
sns.set_style("ticks")
# --- Apply Nature-style defaults ---
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
############################################
###### LOAD CSV FILES FOR EACH REGION ######
############################################

# Initialise file paths dictionary
dataframes = {}

for region, filename in locations.items():
    print(f"Processing {region}...")
    
    # Read and process the adjacency matrix
    adjacency_matrix = read_adjacency_matrix(filename)
    G = create_adjacency_matrix_graph(adjacency_matrix.to_numpy())
    metrics_df = compute_network_metrics(G, region)
    metrics_df = metrics_df.set_index(adjacency_matrix.index)
    coords = pd.read_csv(os.path.join("../data/",region+"_coords.csv"),index_col=0,header=0)
    metrics_df = metrics_df.join(coords)
    metrics_df['geometry'] = metrics_df.apply(lambda x: Point((float(x.Lon), float(x.Lat))), axis=1)
    metrics_df = geopandas.GeoDataFrame(metrics_df, geometry='geometry')
    dataframes[region] = metrics_df

# Ensure DataFrames are not empty
for region, df in dataframes.items():
    if df.empty:
        raise ValueError(f"Dataset for {region} is empty. Check {network_metrics_files[region]}.")

# Add region identifiers
for region, df in dataframes.items():
    df["Region"] = region  # IO or Caribbean

# Combine into a single DataFrame
df_all = pd.concat(dataframes.values(), ignore_index=True)

# Ensure dataset isn't empty after merging
if df_all.empty:
    raise ValueError("Merged dataset is empty. Verify CSV files.")


###############################
###### NORMALITY TESTING ######
###############################

metrics = [
    "Degree Centrality", "Closeness Centrality", "Betweenness Centrality",
    "Eigenvector Centrality", "Harmonic Centrality", "Clustering Coefficient",
]

# Check all expected metric columns are present
missing_metrics = [m for m in metrics if m not in df_all.columns]
if missing_metrics:
    raise ValueError(f"Missing expected columns in dataset: {missing_metrics}")

df_all.dropna(subset=metrics, inplace=True)

# Standardize all metrics
df_box = df_all.copy()
for metric in metrics:
    df_box[metric] = pd.to_numeric(df_box[metric], errors='coerce')
    mean_metric = df_box[metric].mean()
    std_metric = df_box[metric].std()
    df_box[metric] = (df_box[metric] - mean_metric) / std_metric

df_long = df_box.melt(
    id_vars=["Region"],
    value_vars=metrics,
    var_name="Metric",
    value_name="Standardised Value"
)


df_all.to_csv("metrics.csv")
df_long.to_csv("matrics_stnds.csv")

normality_results = []
unique_regions = df_all["Region"].unique()

# Perform normality testing
for metric in metrics:
    print(f"\nNormality test for {metric}:\n" + "-" * 40)
    
    for region in unique_regions:
        data = df_all.loc[df_all["Region"] == region, metric].dropna()
        if len(data) == 0:
            print(f"Skipping {region} for {metric} (no data).")
            continue

        # Convert to numeric if needed
        data = pd.to_numeric(data, errors='coerce').dropna()
        if data.empty:
            print(f"Skipping {region} for {metric} (no valid data after conversion).")
            continue

        mean_val = data.mean()
        std_val = data.std()
        print(f"{region} - {metric}: Mean = {mean_val:.3f}, Std Dev = {std_val:.3f}")

        if std_val == 0 or np.isnan(std_val):
            print(f"Skipping {region} for {metric} (constant or invalid std).")
            continue
        
        # Standardize the data (z-score transformation)
        standardized_data = (data - mean_val) / std_val
        
        # Choose the normality test based on sample size:
        # - Use Kolmogorov-Smirnov for large samples (>500)
        # - Use Shapiro-Wilk for smaller samples
        if len(data) > 500:
            stat, p_val = stats.kstest(standardized_data, 'norm')
            test_used = "Kolmogorov-Smirnov"
        else:
            stat, p_val = stats.shapiro(data)
            test_used = "Shapiro-Wilk"

        normality_results.append({
            "Metric": metric,
            "Region": region,
            "Test": test_used,
            "Statistic": stat,
            "p-value": p_val,
            "Normally Distributed": "Yes" if p_val >= 0.05 else "No",
            "Mean": mean_val,
            "Std Dev": std_val
        })
        
        print(f"{region}: {test_used} test - Stat={stat:.3f}, p={p_val:.3f}")
        if p_val < 0.05:
            print(f"    **{metric} is NOT normally distributed** in {region}")
        else:
            print(f"    {metric} is normally distributed in {region}")

        # Plot histogram with normal curve
        plt.figure(figsize=(7, 5))
        sns.histplot(data, kde=True, bins=30, stat="density", color="skyblue", edgecolor="black")
        x_min, x_max = plt.xlim()
        x_vals = np.linspace(x_min, x_max, 200)
        pdf_vals = stats.norm.pdf(x_vals, mean_val, std_val)
        plt.plot(x_vals, pdf_vals, "r", label="Normal Dist")
        plt.title(f"Distribution of {metric} in {region}")
        plt.xlabel(metric)
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        sns.despine()
            
        plt.savefig(region+"_"+metric+"distribution.pdf",dpi=300)
        plt.close()
        
        #plt.show()

normality_df = pd.DataFrame(normality_results)
normality_df.to_csv("normality_results.csv", index=False)  # uncomment if you want to save

# Boxplot 1: GBR, IO, Caribbean
regions_of_interest = ["GBR", "IO", "Caribbean"]
df_long_interest = df_long[df_long["Region"].isin(regions_of_interest)]

# Create the figure
plt.figure(figsize=(3.5, 3.5))

palette = sns.color_palette("husl", 3)
palette[0], palette[1] = palette[1], palette[0]

# Assuming 'df_long_interest' is already in your environment
g = sns.catplot(
    data=df_long_interest, 
    x="Region", 
    y="Standardised Value", 
    col="Metric",
    order=["GBR", "IO", "Caribbean"],
    kind="violin", 
    height=1.6,
    col_wrap=2,          
    hue="Region",
    linewidth=0.1,
    palette=palette,
    inner="box",
    inner_kws=dict(box_width=2, whis_width=0.3),
    sharey=False          # Critical so each metric scales independently
)

# Clean up titles (removes the "Metric = " prefix from seaborn)
g.set_titles("{col_name}", size=7, weight='bold')

# -------------------------------------------------------------------
# 2. DEFINE THE MINI-NETWORK DESIGNS FOR EACH METRIC
# -------------------------------------------------------------------
base_col = '#aec7e8'     # Light blue for standard nodes
high_col = '#ff7f0e'     # Orange for the "highlighted" node
edge_col = 'gray'

# We iterate through the exact axes dictionary generated by seaborn
for metric, ax in g.axes_dict.items():
    
    # Create an inset axis in the top left [x0, y0, width, height]
    # Adjust the 'y0' (0.65) or 'width/height' (0.3) if it overlaps your violin tails
    inset_ax = ax.inset_axes([0.02, 0.65, 0.30, 0.30])
    
    # Default variables
    G = nx.Graph()
    colors = []
    
    if metric == "Degree Centrality":
        # Illustrate a hub (star graph)
        G = nx.star_graph(5)
        colors = [high_col if n == 0 else base_col for n in G.nodes()]
        
    elif metric == "Closeness Centrality":
        # Illustrate a line where the center is closest to all ends
        G = nx.path_graph(5)
        colors = [high_col if n == 2 else base_col for n in G.nodes()]
        
    elif metric == "Betweenness Centrality":
        # Illustrate a "bowtie" where one node bridges two clusters
        G.add_edges_from([(0,1), (1,2), (0,2), (2,3), (3,4), (2,4)])
        colors = [high_col if n == 2 else base_col for n in G.nodes()]
        
    elif metric == "Eigenvector Centrality":
        # Illustrate a node connected to a highly-connected dense cluster
        G.add_edges_from([(0,1), (1,2), (0,2), (2,3), (3,4)])
        colors = [high_col if n == 2 else base_col for n in G.nodes()]
        
    elif metric == "Harmonic Centrality":
        # Illustrate ability to handle disconnected networks
        # A central star, plus a completely separate edge
        G.add_edges_from([(0,1), (0,2), (0,3), (4,5)])
        colors = [high_col if n == 0 else base_col for n in G.nodes()]
        
    elif metric == "Clustering Coefficient":
        # Illustrate a closed triangle amongst neighbors
        G.add_edges_from([(0,1), (0,2), (0,3), (1,2)])
        # Highlight the nodes making up the closed cluster
        colors = [high_col if n in [0,1,2] else base_col for n in G.nodes()]
        
    else:
        # Fallback empty graph just in case
        G = nx.path_graph(3)
        colors = [base_col]*3

    # spring_layout with a fixed seed keeps the shape identical every run
    pos = nx.spring_layout(G, seed=42)
    
    nx.draw_networkx_nodes(G, pos, ax=inset_ax, node_size=18, node_color=colors, edgecolors='white', linewidths=0.1)
    nx.draw_networkx_edges(G, pos, ax=inset_ax, edge_color=edge_col, width=1.2, alpha=0.6)
    
    # Strip borders and background from the mini-plot so it looks like a floating icon
    inset_ax.axis('off')
    inset_ax.set_facecolor('none')

# -------------------------------------------------------------------
# 4. FINALIZE AND SAVE
# -------------------------------------------------------------------
plt.tight_layout()
sns.despine()
plt.savefig("violin_plot_with_network_icons.pdf", dpi=72, bbox_inches='tight')
plt.close()



############################################
###### ANOVA / KRUSKAL-WALLIS TESTING ######
############################################

# Define testable metrics for comparison
comparison_metrics = ["Degree Centrality", "Closeness Centrality", "Betweenness Centrality",
    "Eigenvector Centrality", "Harmonic Centrality", "Clustering Coefficient",
]
ANOVA_KruskallWallis_results = []

# Perform one-way ANOVA or Kruskal-Wallis depending on normality
for metric in comparison_metrics:
    print(f"\nComparing {metric} between regions:")

    # Group data by Region (GBR, IO, Caribbean) to match normality results
    data_by_region = [df_all[df_all["Region"] == region][metric].dropna() for region in unique_regions]

    # Check normality using Region_Combined labels — these match what normality_results stores
    normal_regions = [
        region for region in unique_regions
        if any(result['Metric'] == metric and result['Region'] == region and result['p-value'] >= 0.05 for result in normality_results)
    ]

    if all(region in normal_regions for region in unique_regions):
        # All regions are normal, apply ANOVA
        stat, p = stats.f_oneway(*data_by_region)
        test_used = "ANOVA"
    else:
        # At least one region is non-normal, apply Kruskal-Wallis
        stat, p = stats.kruskal(*data_by_region)
        test_used = "Kruskal-Wallis"

    print(f"{test_used} test: statistic={stat:.3f}, p={p:.3f}")
    if p < 0.05:
        print(f"    Significant difference found in {metric}")
    else:
        print(f"    No significant difference found in {metric}")

    # Store ANOVA/Kruskall-Wallis results
    ANOVA_KruskallWallis_results.append({
        "Metric": metric,
        "Test": test_used,
        "Statistic": stat,
        "p-value": p,
        "Significant difference": "Yes" if p < 0.05 else "No"
        })

    # If significant, perform post-hoc test
    if p < 0.05:
        # Prepare a combined dataframe for post-hocs just to be safe and clean
        all_data = pd.concat([df_all[df_all["Region"] == region][["Region", metric]] for region in unique_regions]).dropna()

        # ------------------- FOR ANOVA -------------------
        if test_used == "ANOVA":
            tukey_result = pairwise_tukeyhsd(all_data[metric], all_data["Region"])
            print("\nPost-hoc Tukey's HSD results:")
            print(tukey_result.summary())

            # Store Tukey's HSD results
            tukey_results = tukey_result.summary().data
            for row in tukey_results[1:]:
                ANOVA_KruskallWallis_results.append({
                    "Metric": metric,
                    "Test": "Tukey's HSD",
                    "Pairwise Comparison": f"{row[0]} vs {row[1]}",
                    "Statistic": row[2],  # Mean Difference
                    "p-value": row[3],    # Adjusted p-value
                    "Significant difference": "Yes" if row[3] < 0.05 else "No"
                    })

        # -------------- FOR KRUSKAL-WALLIS --------------
        elif test_used == "Kruskal-Wallis":
            # Perform Dunn's test with p-value correction (e.g., Bonferroni or Holm)
            dunn_p_values = sp.posthoc_dunn(all_data, val_col=metric, group_col='Region', p_adjust='holm')
            print("\nPost-hoc Dunn's test p-values:")
            print(dunn_p_values)
            
            # Unpack the symmetric square matrix from Dunn's test to store in results
            regions = list(dunn_p_values.columns)
            for i in range(len(regions)):
                for j in range(i + 1, len(regions)):  # Only take the upper triangle to avoid duplicates
                    region_a = regions[i]
                    region_b = regions[j]
                    p_val = dunn_p_values.loc[region_a, region_b]
                    
                    ANOVA_KruskallWallis_results.append({
                        "Metric": metric,
                        "Test": "Dunn's test",
                        "Pairwise Comparison": f"{region_a} vs {region_b}",
                        "Statistic": None,  # Dunn's test natively outputs the p-value matrix directly in scikit-posthocs
                        "p-value": p_val,
                        "Significant difference": "Yes" if p_val < 0.05 else "No"
                    })

# Convert results list into df and save to csv
ANOVA_KruskallWallis_df = pd.DataFrame(ANOVA_KruskallWallis_results)
ANOVA_KruskallWallis_df.to_csv(r"ANOVA_KruskallWallis_results.csv", index=False)

##########################################
###### PRINCIPAL COMPONENT ANALYSIS ######
##########################################

# Standardise results for PCA
metrics_data = df_all[metrics].dropna()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(metrics_data)

# Perform PCA
pca = PCA()
pca.fit(scaled_data)

# Visualise explained variance
plt.figure(figsize=(3.5, 3.5))
plt.plot(range(1, len(metrics) + 1), pca.explained_variance_ratio_, marker='o', linestyle='--')
plt.xlabel('Principal Component')
plt.ylabel('Variance Ratio')
plt.savefig("PCA_variance.pdf",dpi=300, bbox_inches='tight')
plt.close()

# Biplot
components = pca.components_
plt.figure(figsize=(10, 6))
sns.heatmap(
    components,
    cmap='coolwarm',
    center=0,               # tell seaborn to center the colormap at 0
    xticklabels=metrics,
    yticklabels=[f"PC{i+1}" for i in range(len(components))]
)

plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels to 45 degrees for readability
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("PCA_bivar.pdf",dpi=300, bbox_inches='tight')
plt.close()

# Extract principal components
pca_data = pca.transform(scaled_data)
df_pca = pd.DataFrame(pca_data, columns=[f"PC{i+1}" for i in range(len(pca_data[0]))])

# Add region info to PCA data
df_pca["Region"] = df_all["Region"].values

# Get the loadings for PC1 (first row of pca.components_)
pc1_loadings = pca.components_[0]

# Create a DataFrame to show the loadings of each metric for PC1 --> adjust for other PCs if necessary
pc1_loadings_df = pd.DataFrame(pc1_loadings, index=metrics, columns=['PC1'])
print(pc1_loadings_df)

# Get the loadings for PC1 (first row of pca.components_)
pc2_loadings = pca.components_[1]

# Create a DataFrame to show the loadings of each metric for PC1 --> adjust for other PCs if necessary
pc2_loadings_df = pd.DataFrame(pc2_loadings, index=metrics, columns=['PC2'])
print(pc2_loadings_df)



#######################################################################
###### HIERARCHICAL CLUSTERING (W. SILHOUETTE AND ELBOW METHODS) ######
#######################################################################

# Elbow Method
wcss = []  # List to store within-cluster sum of squares (WCSS) for each k
for k in range(2, 11):  # Check for k from 2 to 10 clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(1.0, 1.0))
plt.plot(range(2, 11), wcss, marker='o', linestyle='--', ms=1)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.savefig("Clusters_elbow.pdf",dpi=300, bbox_inches='tight')
plt.close()

# Silhouette Method
sil_scores = []  # List to store silhouette scores for each k
for k in range(2, 11):  # Check for k from 2 to 10 clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)
    sil_score = silhouette_score(scaled_data, cluster_labels)
    sil_scores.append(sil_score)

# Plot the silhouette scores
plt.figure(figsize=(1.0, 1.0))
plt.plot(range(2, 11), sil_scores, marker='o', linestyle='--', ms=1)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
sns.despine()
plt.savefig("Clusters_silhouette.pdf",dpi=300, bbox_inches='tight')
plt.close()

# Hierarchical Clustering in Principal Component space due to high dimensionality
# Use PC1 and PC2 --> 2 dimension
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

# Perform hierarchical clustering
linkage_matrix = linkage(reduced_data, method='ward')

# Assign clusters
num_clusters = 3  # Adjust as needed
cluster_labels = fcluster(linkage_matrix, t=num_clusters, criterion='maxclust')

# 2. Define distinct colors and SWAP Cluster 1 & Cluster 2
palette = sns.color_palette("husl", num_clusters)
palette[0], palette[1] = palette[1], palette[0]

# 3. Map regions to markers
region_markers = {
    'GBR': '.',
    'IO': '^',
    'Caribbean': 'X'
}

# 4. Add data to the dataframe
df_all['Cluster'] = cluster_labels
df_all['PC1'] = reduced_data[:, 0]
df_all['PC2'] = reduced_data[:, 1]

# --- THE CLEANUP MAGIC STARTS HERE ---

# Create a clean plotting dataframe and map visual properties directly to it
df_plot = df_all.copy()
df_plot['Color'] = df_plot['Cluster'].apply(lambda x: palette[x - 1])

# Shuffle the dataframe randomly so overlapping isn't biased to any specific cluster
# random_state ensures that your chart looks identical every time you run the code
df_plot = df_plot.sample(frac=1, random_state=42).reset_index(drop=True)

plt.figure(figsize=(3.5, 2.5))

# Matplotlib.scatter can't take a LIST of custom markers natively.
# So we still loop by region, but because the rows are shuffled, the colors/clusters 
# will be perfectly interleaved and randomized on the plot!
for region, marker in region_markers.items():
    region_subset = df_plot[df_plot["Region"] == region]
    
    plt.scatter(
        region_subset["PC1"], 
        region_subset["PC2"], 
        c=region_subset["Color"], 
        marker=marker,
        s=25,                  # Slightly smaller size to reduce overlap
        alpha=0.6,             # Transparency (makes dense areas look richer)
        edgecolors='k',    
        linewidths=0.5         # Thinner edge lines for a cleaner look
    )

# --- LEGEND FIXES ---
# We keep the custom legend handles so they still show as crisp, solid markers
cluster_legend = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=5, label=f"Cluster {i+1}") 
    for i in range(num_clusters)
]

region_legend = [
    plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor='dimgray', markersize=5, label=region) 
    for region, marker in region_markers.items()
]

plt.legend(handles=cluster_legend + region_legend, loc="best", fontsize=6)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
sns.despine()
plt.savefig("PCA.pdf",dpi=300, bbox_inches='tight')
plt.close()

# Add cluster labels to the dataframe
df_all['Cluster'] = cluster_labels
    
##########################################
###### PEARSON/SPEARMAN CORRELATION ######
##########################################

# Calculate correlation matrix for metrics
correlation_matrix = df_all[metrics].corr(method='pearson')
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, vmin=-1, vmax=1)
#plt.title('Pearson Correlation Matrix of Network Metrics')
plt.xticks(rotation=45, ha='right')
plt.savefig("Metric_Cor_Pearson.pdf",dpi=300)
#plt.show()
plt.close()

# For non-parametric correlation (Spearman)
correlation_matrix_spearman = df_all[metrics].corr(method='spearman')
sns.heatmap(correlation_matrix_spearman, annot=True, cmap='coolwarm', linewidths=0.5, vmin=-1, vmax=1)
#plt.title('Spearman Correlation Matrix of Network Metrics')
plt.xticks(rotation=45, ha='right')
plt.savefig("Metric_Cor_Spearman.pdf",dpi=300)
#plt.show()
plt.close()

