import geopandas as gpd
import numpy as np
from sklearn.cluster import AgglomerativeClustering

def clump_reef_polygons(input_path, output_path, distance_threshold_meters=2000):
    print("Loading original GBR shapefile...")
    # 1. Load the original GBR shapefile
    gdf = gpd.read_file(input_path)
    original_count = len(gdf)
    print(f"Loaded {original_count} reef features. CRS: {gdf.crs}")
    
    # Double-check that data is projected (e.g., UTM) so distances/areas are metric
    if gdf.crs.is_geographic:
        raise ValueError("The shapefile is in a geographic CRS (degrees). Please project it to UTM 56S first.")

    print("Calculating polygon centroids...")
    # 2. Extract coordinates from the centroids of the polygons
    centroids = gdf.geometry.centroid
    X = np.column_stack((centroids.x, centroids.y))

    print(f"Running Agglomerative Clustering (Threshold: {distance_threshold_meters}m)...")
    # 3. Set up the clustering algorithm
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold_meters,
        linkage='complete',
        metric='euclidean'
    )
    
    # Fit and predict cluster labels
    cluster_labels = clustering.fit_predict(X)
    gdf['cluster_id'] = cluster_labels
    
    print("Dissolving reef polygons into composite cluster boundaries...")
    # 4. Dissolve geometries based on the cluster ID
    clumped_gdf = gdf.dissolve(by='cluster_id')
    
    # Reset index so 'cluster_id' becomes a standard column again
    clumped_gdf = clumped_gdf.reset_index()
    
    # Clean up columns: keep a count of original reefs inside each cluster
    reef_counts = gdf['cluster_id'].value_counts().rename('reef_count')
    clumped_gdf = clumped_gdf.join(reef_counts, on='cluster_id')
    
    print("Calculating the area of each new clump in square km...")
    # 5. Calculate area
    # Because CRS is UTM (meters), .area returns square meters. 
    # Dividing by 1,000,000 converts it to square kilometers.
    clumped_gdf['area_km2'] = clumped_gdf.geometry.area / 1_000_000
    
    # Keep only relevant structural columns for the output shapefile
    columns_to_keep = ['cluster_id', 'reef_count', 'area_km2', 'geometry']
    clumped_gdf = clumped_gdf[columns_to_keep]

    print(f"Saving clumped shapefile to: {output_path}")
    # 6. Export to a new shapefile
    clumped_gdf.to_file(output_path)
    
    final_count = len(clumped_gdf)
    print(f"Success! Reduced {original_count} individual reefs into {final_count} aggregated reef zones.")

if __name__ == "__main__":
    # Define your file paths here
    INPUT_SHAPEFILE = "../GBR_names_reprojected.shp" 
    OUTPUT_SHAPEFILE = "GBR_names_reprojected_2km.shp"
    
    clump_reef_polygons(INPUT_SHAPEFILE, OUTPUT_SHAPEFILE, distance_threshold_meters=2000)
