#!/usr/bin/env python3
"""
This script loads in the trajectories from an ocean parcels run
and works out *all* possible intersections with a reef (including itself).

Reefs are stored as polygons. An intersection is a time/location when the
particle is within a polygon. All are stored for later use.

@author: jhill1; https://github.com/jhill1
"""
#
# This work is licensed under a Creative Commons Attribution 4.0 International License.
#
# To view a copy of this license, visit creativecommons.org or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
# Copyright University of York 2026
import os
# CRITICAL: Must be set BEFORE importing anything else
os.environ["MALLOC_ARENA_MAX"] = "2"
from glob import glob
import gc
import zarr
# CRITICAL: Prevent Zarr/Blosc from spawning hidden C-threads that leak unmanaged memory
zarr.blosc.use_threads = False
import pandas as pd
import geopandas as gpd
import shapely
from shapely.strtree import STRtree
import dask.dataframe as dd
from dask import delayed
from dask.distributed import Client, LocalCluster
import numpy as np

dataset_dir = '../../../modern_0.5km_high_diff/Trajectory_tides_wholeGBR_1024.zarr'
# Changed to a directory format for safe distributed writing
output_dir = '../../../modern_0.5km_high_diff/intersection_events_0.5kmclumps_output'
os.makedirs(output_dir, exist_ok=True)

worker_cache = {}

def get_spatial_tree(poly_geoms):
    if 'tree' not in worker_cache:
        worker_cache['tree'] = STRtree(poly_geoms)
    return worker_cache['tree']

def process_time_step(proc_path, t_idx, poly_geoms, polygon_ids):
    root = zarr.open(proc_path, mode='r')
    lat = root['lat'][:, t_idx]
    lon = root['lon'][:, t_idx]
    time_val = root['time'][:, t_idx]

    traj_arr = root['trajectory']
    if len(traj_arr.shape) == 1:
        traj_ids = traj_arr[:]
    else:
        traj_ids = traj_arr[:, t_idx]

    valid_mask = ~(np.isnan(lat) | np.isnan(lon))
    if not np.any(valid_mask):
        return pd.DataFrame({
            'trajectory': pd.Series(dtype='int64'),
            'polygon_id': pd.Series(dtype='int64'), 
            'time': pd.Series(dtype='timedelta64[s]')
        })

    lat_valid = lat[valid_mask]
    lon_valid = lon[valid_mask]
    time_valid = time_val[valid_mask]
    traj_valid = traj_ids[valid_mask]

    points = shapely.points(lon_valid, lat_valid)
    tree = get_spatial_tree(poly_geoms)

    point_hits, poly_indices = tree.query(points, predicate='within')

    if len(point_hits) == 0:
        del points, valid_mask, lat_valid, lon_valid
        gc.collect()
        return pd.DataFrame({
            'trajectory': pd.Series(dtype='int64'),
            'polygon_id': pd.Series(dtype='int64'), 
            'time': pd.Series(dtype='timedelta64[s]')
        })

    intersect_times = time_valid[point_hits]
    intersect_traj_ids = traj_valid[point_hits]
    intersect_poly_ids = polygon_ids[poly_indices]

    df = pd.DataFrame({
        'trajectory': intersect_traj_ids,
        'polygon_id': intersect_poly_ids,
        'time': intersect_times
    })

    df['time'] = pd.to_timedelta(df['time'], unit='s')

    # Strip any implicit pandas indices to guarantee clean Parquet schemas
    df = df.reset_index(drop=True)

    del points, valid_mask, lat_valid, lon_valid, time_valid, traj_valid
    gc.collect()

    return df

if __name__ == "__main__":
    cluster = LocalCluster(
        n_workers=10,               
        threads_per_worker=1,
        memory_limit='11GB'
    )
    client = Client(cluster)
    print(f"Dashboard link: {client.dashboard_link}")

    polygons_gdf = gpd.read_file("~/work/projects/PalaeoTides/gbr_connectivity/scripts/GBR_names_reprojected.shp")
    poly_geoms_future = client.scatter(polygons_gdf.geometry.values, broadcast=True)
    poly_ids_future   = client.scatter(polygons_gdf.index.values, broadcast=True)

    proc_stores = sorted(glob(os.path.join(dataset_dir, 'proc*.zarr')))
    if not proc_stores:
        proc_stores = [dataset_dir]

    print(f"Processing {len(proc_stores)} MPI processor stores...")

    meta_df = pd.DataFrame({
        'trajectory': pd.Series(dtype='int64'),
        'polygon_id': pd.Series(dtype='int64'), 
        'time': pd.Series(dtype='timedelta64[s]')
    })

    for idx, proc_path in enumerate(proc_stores):
        print(f"[{idx+1}/{len(proc_stores)}] Submitting {os.path.basename(proc_path)}...")

        root = zarr.open(proc_path, mode='r')
        n_obs = root['lat'].shape[1]
        proc_name = os.path.basename(proc_path).replace('.zarr', '')

        BATCH_SIZE = 100
        for start_t in range(0, n_obs, BATCH_SIZE):
            end_t = min(start_t + BATCH_SIZE, n_obs)

            results = []
            for t_idx in range(start_t, end_t):
                res = delayed(process_time_step)(
                    proc_path,
                    t_idx,
                    poly_geoms_future,
                    poly_ids_future
                )
                results.append(res)

            file_ddf = dd.from_delayed(results, meta=meta_df)

            # Write to a completely isolated sub-directory per batch.
            # This perfectly avoids all Dask append/metadata mismatch bugs.
            batch_dir = os.path.join(output_dir, f"{proc_name}_batch_{start_t}")

            file_ddf.to_parquet(
                batch_dir,
                engine='pyarrow',
                write_index=False
            )

            del results, file_ddf
            gc.collect()

        print(f"Finished {os.path.basename(proc_path)}")

    print("Pipeline completed successfully!")
