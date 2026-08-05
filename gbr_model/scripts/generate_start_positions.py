import math
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, MultiPoint
import random
import time

shp_file = "../../data/GBR_names_reprojected_9km.shp"
particles_per_km2 = 4048

def generate_random_vectorized(number, polygon):
    """Generates random points within a polygon using vectorized NumPy operations."""
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    
    # To reduce iterations, we estimate how many raw points we need to generate 
    # to hit our target count after filtering out points outside the polygon.
    # We add a safety buffer (factor of 1.5) to account for irregular polygon boundaries.
    poly_box_area = (maxx - minx) * (maxy - miny)
    fill_ratio = max(0.01, polygon.area / poly_box_area) if poly_box_area > 0 else 1
    batch_size = int(math.ceil(number / fill_ratio) * 1.5)
    batch_size = max(batch_size, number, 100) # Ensure it's large enough

    while len(points) < number:
        # 1. Generate a large batch of random X and Y arrays in one go
        rand_x = np.random.uniform(minx, maxx, batch_size)
        rand_y = np.random.uniform(miny, maxy, batch_size)
        
        # 2. Bulk convert to Shapely points
        # Using MultiPoint or vectorized maps is much faster than standard loops
        candidate_points = [Point(x, y) for x, y in zip(rand_x, rand_y)]
        
        # 3. Filter points using a list comprehension or vector check
        # .contains() on a single geometry object is fast when passed candidates
        valid_points = [p for p in candidate_points if polygon.contains(p)]
        points.extend(valid_points)
        
        # Dynamic adjustment: if we still need more points, keep going
        if len(points) < number:
            needed = number - len(points)
            batch_size = int(math.ceil(needed / fill_ratio) * 1.5)
            batch_size = max(batch_size, 50)
            
    # Return exactly the number of points requested
    return points[:number]

def main():
    start_time = time.time()
    print("Reading shapefile with GeoPandas...")
    # Load with Geopandas (much faster indexing than standard shapefile library)
    gdf = gpd.read_file(shp_file)
    
    all_points_x = []
    all_points_y = []
    
    print("Processing features and seeding particles...")
    for idx, row in gdf.iterrows():
        # Try/catch safety block for column attributes as requested
        try:
            area_km2 = row['Area_HA'] * 0.01
        except (KeyError, AttributeError):
            area_km2 = row['area_km2']
            
        particles = round(area_km2 * particles_per_km2)
        if particles < particles_per_km2:
            particles = particles_per_km2
            
        # Extract the geometry object directly
        shp_geom = row.geometry
        if shp_geom is None or shp_geom.is_empty:
            continue
            
        # Generate points using our fast vectorized function
        points = generate_random_vectorized(particles, shp_geom)
        
        # Extract coordinates in bulk
        for p in points:
            all_points_x.append(p.x)
            all_points_y.append(p.y)
            
        if idx % 100 == 0 and idx > 0:
            print(f"Processed {idx}/{len(gdf)} features...")

    print("Saving text coordinates...")
    np.savetxt("x_points_high_9km.csv", all_points_x)
    np.savetxt("y_points_high_9km.csv", all_points_y)
    
    end_time = time.time()
    print(f"Finished! Generated {len(all_points_x)} points in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
