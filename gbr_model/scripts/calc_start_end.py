import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.strtree import STRtree
import dask.dataframe as dd
from dask import delayed
from dask.distributed import Client, LocalCluster
import dask.array as da
import geopandas as gpd

CHUNK = 5000
dataset_file = '~/work/projects/PalaeoTides/gbr_connectivity/scripts/test/test_dataset_100_trajs.nc'
output_file = 'intersection_events_test.parquet'

# --- 1. Setup the Cluster ---
if __name__ == "__main__":
    # This automatically detects your cores.
    # If you have 64GB+ RAM, this default is fine. 
    # If you have limited RAM per core, reduce n_workers and increase threads_per_worker.
    cluster = LocalCluster()
    #cluster.scale(1)
    client = Client(cluster)
    
    print(f"Dashboard link: {client.dashboard_link}")
    print(f"Running on {len(client.scheduler_info()['workers'])} cores.")

    # --- 2. Load Data ---
    
    # Load Polygons
    polygons_gdf = gpd.read_file("~/work/projects/PalaeoTides/gbr_connectivity/scripts/GBR_names_reprojected.shp")
    # Extract geometry array (Shapely 2.0 vectorized array)
    poly_geoms = polygons_gdf.geometry.values
    poly_ids = polygons_gdf.index.values

    # CRITICAL OPTIMIZATION: Scatter polygons to all workers
    # This creates a cached copy on every worker node instantly
    poly_geoms_future = client.scatter(poly_geoms, broadcast=True)
    poly_ids_future = client.scatter(poly_ids, broadcast=True)

    # Load Trajectories
    # Chunking: 'obs': -1 ensures time is contiguous. 
    # 'trajectory': 5000 is a tunable parameter. 
    # Smaller chunks = less RAM per worker, but more overhead.
    ds = xr.open_dataset(dataset_file, chunks={'trajectory': CHUNK, 'obs': -1})
    traj_dask = da.from_array(ds.trajectory.values, chunks=CHUNK)

    # --- 3. Define the Worker Function ---
    def process_batch(lat, lon, time, traj_ids, polygon_geoms, polygon_ids):
        """
        Runs on the worker. Receives the chunk of data + the scattered polygons.
        """
        # A. Prepare Points
        # Flatten to 1D arrays for vectorized check
        lat_flat = lat.ravel()
        lon_flat = lon.ravel()
        
        # Fast creation of Shapely point array (no Python loop)
        points = shapely.points(lon_flat, lat_flat)
        
        # B. Build Spatial Index (STRtree)
        # Building a tree for 1500 items takes < 1ms. 
        # It is faster to rebuild it locally than to serialize/transfer the tree object.
        tree = STRtree(polygon_geoms)
        
        # C. Query
        # Returns indices: [index_in_polygons, index_in_points]
        point_indices, poly_indices = tree.query(points, predicate='within')
        
        if len(point_indices) == 0:
            return gpd.GeoDataFrame({
                'trajectory': pd.Series(dtype='int64'),
                'polygon_id': pd.Series(dtype='int64'), # Must match your meta dtype
                'time': pd.Series(dtype='timedelta64[ns]')
            })        
        # D. Map 1D indices back to (Trajectory, Time)
        n_obs = time.shape[1]
        
        # Integer division to find which trajectory; Modulo to find which time step
        traj_idx_local = point_indices // n_obs 
        obs_idx = point_indices % n_obs

        # E. Extract Data
        # Use vectorized indexing
        intersect_times = time[traj_idx_local, obs_idx]
        intersect_traj_ids = traj_ids[traj_idx_local]
        intersect_poly_ids = polygon_ids[poly_indices]

        # F. Build Result DataFrame
        df = gpd.GeoDataFrame({
            'trajectory': intersect_traj_ids,
            'polygon_id': intersect_poly_ids,
            'time': intersect_times
        })
        
        df = df.sort_values(['time','trajectory'])
        
        return df


    # --- 4. Build the Task Graph ---

    results = []
    
    # We delay the data chunks, but we pass the SCATTERED futures for polygons
    # This ensures the polygons are not re-sent over the network/memory bus
    
    # Extract delayed objects for iteration
    lon_chunks = ds.lon.data.to_delayed().ravel()
    lat_chunks = ds.lat.data.to_delayed().ravel()
    time_chunks = ds.time.data.to_delayed().ravel()
    traj_chunks = traj_dask.to_delayed().ravel()

    # Note: Ensure all chunk lists are same length. 
    # If ds.trajectory is 1D and chunks match the 2D vars' axis 0, this works.
    
    for i in range(len(lon_chunks)):
        res = delayed(process_batch)(
            lat_chunks[i], 
            lon_chunks[i], 
            time_chunks[i], 
            traj_chunks[i],
            poly_geoms_future, # <--- Passing the Future
            poly_ids_future    # <--- Passing the Future
        )
        results.append(res)

    # --- 5. Compute and Save ---

    # Define schema for Dask DataFrame
    meta_df = gpd.GeoDataFrame({
        'trajectory': pd.Series(dtype='int64'),
        'polygon_id': pd.Series(dtype='int64'), # or int, depending on your ID
        'time': pd.Series(dtype='timedelta64[ns]')
    })

    final_ddf = dd.from_delayed(results, meta=meta_df)

    # Write to Parquet (Parallel write)
    # This will trigger the actual computation on the cluster
    final_ddf.to_parquet(output_file, engine='pyarrow')

    print("Done.")
