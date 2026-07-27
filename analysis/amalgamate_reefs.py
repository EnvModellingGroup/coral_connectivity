#!/usr/bin/env python3
"""
This script creates reef clusters from a shapefile (polygons of reefs)
by using a distance threshold to amalgamate these into a single "reef".

Saves a new shapefile, but that needs a bit more processing:

To add x and y locs, use QGIS to generate centroids, then you can use the append 
fields tool to copy them across. 

You also need to add a LOC_NAME_S field which is a string and can be the 
cluster_id (or other unique string).

@author: jhill1; https://github.com/jhill1
"""
#
# This work is licensed under a Creative Commons Attribution 4.0 International License.
#
# To view a copy of this license, visit creativecommons.org or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
# Copyright University of York 2026
import geopandas as gpd
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def resolve_sector(sectors):
    """
    Determines the SECTOR label for a cluster:
    1. Returns the most frequent SECTOR in the group.
    2. If there is a tie for the top frequency, concatenates all tied sectors alphabetically.
    """
    # Clean up any potential missing or blank sector values
    clean_sectors = [str(s).strip() for s in sectors if s is not None and str(s).strip() != '']
    if not clean_sectors:
        return "UNKNOWN"

    # Count occurrences of each sector in this cluster
    values, counts = np.unique(clean_sectors, return_counts=True)
    max_count = np.max(counts)

    # Get all sectors that share the maximum count (handles single modes and ties)
    top_sectors = sorted(values[counts == max_count])

    # Join them together (e.g., ['CAIRNS'] -> 'CAIRNS', ['CAIRNS', 'SWAIN'] -> 'CAIRNSSWAIN')
    return "".join(top_sectors)


def clump_reef_polygons(input_path, output_path, distance_threshold_meters=2000):
    """
    Main algorithm to clump the reefs
    """
    print("Loading original GBR shapefile...")
    gdf = gpd.read_file(input_path)
    original_count = len(gdf)
    print(f"Loaded {original_count} reef features. CRS: {gdf.crs}")

    if gdf.crs.is_geographic:
        raise ValueError("The shapefile is in a geographic CRS. Project to UTM 56S.")

    # Verify the 'SECTOR' field exists
    if 'SECTOR' not in gdf.columns:
        raise KeyError("The input shapefile does not contain a 'SECTOR' field.")

    print("Calculating polygon centroids...")
    centroids = gdf.geometry.centroid
    X = np.column_stack((centroids.x, centroids.y))

    print(f"Running Agglomerative Clustering (Threshold: {distance_threshold_meters}m)...")
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold_meters,
        linkage='complete',
        metric='euclidean'
    )

    # Fit and assign cluster labels
    cluster_labels = clustering.fit_predict(X)
    gdf['cluster_id'] = cluster_labels

    print("Resolving SECTOR labels and dissolving reef geometries...")
    # 1. Compute the resolved SECTOR label for each cluster_id
    sector_mapping = gdf.groupby('cluster_id')['SECTOR'].apply(resolve_sector).to_dict()

    # 2. Count original reefs per cluster
    reef_counts = gdf['cluster_id'].value_counts().to_dict()

    # 3. Dissolve geometries based on the cluster ID
    clumped_gdf = gdf.dissolve(by='cluster_id')
    clumped_gdf = clumped_gdf.reset_index()

    # 4. Assign the resolved attributes to the clumped dataframe
    clumped_gdf['SECTOR'] = clumped_gdf['cluster_id'].map(sector_mapping)
    clumped_gdf['reef_count'] = clumped_gdf['cluster_id'].map(reef_counts)

    print("Calculating area in square km...")
    clumped_gdf['area_km2'] = clumped_gdf.geometry.area / 1_000_000

    # Keep key structural columns for the output shapefile
    columns_to_keep = ['cluster_id', 'SECTOR', 'reef_count', 'area_km2', 'geometry']
    clumped_gdf = clumped_gdf[columns_to_keep]

    print(f"Saving clumped shapefile to: {output_path}")
    clumped_gdf.to_file(output_path)

    final_count = len(clumped_gdf)
    print(f"Success! Reduced {original_count} individual reefs to {final_count} aggregated reefs.")

if __name__ == "__main__":
    # Define your file paths here
    INPUT_SHAPEFILE = "GBR_names_reprojected.shp"
    OUTPUT_SHAPEFILE = "GBR_names_reprojected_2km.shp"

    clump_reef_polygons(INPUT_SHAPEFILE, OUTPUT_SHAPEFILE, distance_threshold_meters=2000)
    print("Add x and y locs")
