#!/usr/bin/env python3
"""
This script loads in the intersections and uses data
on competency to work out which reef a particle ends up in

This is stochastic. You will end up with different
connections if you run it again (so use a lot of particles to
minimise the impact of this).

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
import glob
import gc
import uuid
import pyarrow.parquet as pq
import geopandas as gpd
import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster, as_completed

competency_curve = "../../data/G.retiformis_rubble.csv"
intersection_dir = '../../../modern_0.5km_high_diff/intersection_events_0.5kmclumps_output'

# --- Create Temporary Directories for Reduced Data ---
TEMP_STARTS_DIR = '../../../modern_0.5km_high_diff/temp_starts'
TEMP_HITS_DIR = '../../../modern_0.5km_high_diff/temp_hits'
os.makedirs(TEMP_STARTS_DIR, exist_ok=True)
os.makedirs(TEMP_HITS_DIR, exist_ok=True)

# --- 1. Disk-Backed Worker Function ---
def process_parquet_file(file_path, comp_times, comp_probs):
    """Streams a Parquet file, reduces it, and writes directly to disk to spare the client's RAM."""
    try:
        parquet_file = pq.ParquetFile(file_path)
    except Exception:
        return False

    start_chunks = []
    hit_chunks = []

    for batch in parquet_file.iter_batches(batch_size=100000):
        df = batch.to_pandas()
        if df.empty:
            continue

        starts = df[df['time'] == pd.Timedelta(0)]
        if not starts.empty:
            start_chunks.append(starts)

        times_days = df["time"].dt.days.to_numpy()
        comp = np.interp(times_days, comp_times, comp_probs)
        survival = comp * np.exp(-0.05 * times_days)

        probs = np.random.rand(len(times_days))
        is_hit = probs < survival

        hits = df[is_hit]
        if not hits.empty:
            hits = hits.sort_values('time').drop_duplicates(subset='trajectory', keep='first')
            hit_chunks.append(hits)

        del df, times_days, comp, survival, probs, is_hit, hits

    final_starts = pd.concat(start_chunks, ignore_index=True) if start_chunks else pd.DataFrame()
    final_hits = pd.concat(hit_chunks, ignore_index=True) if hit_chunks else pd.DataFrame()

    if not final_hits.empty:
        final_hits = final_hits.sort_values('time').drop_duplicates(subset='trajectory',
                                                                    keep='first')

    # Generate a unique ID for these specific output files
    file_id = f"{uuid.uuid4().hex[:8]}"

    # Write to disk. DO NOT return the data to the client.
    if not final_starts.empty:
        final_starts.to_parquet(os.path.join(TEMP_STARTS_DIR, f"starts_{file_id}.parquet"))
    if not final_hits.empty:
        final_hits.to_parquet(os.path.join(TEMP_HITS_DIR, f"hits_{file_id}.parquet"))

    del final_starts, final_hits
    return True

if __name__ == "__main__":

    comp_curve = pd.read_csv(competency_curve)
    comp_times = comp_curve["Time"].values
    comp_probs = comp_curve["Probability"].values

    cluster = LocalCluster(n_workers=10, threads_per_worker=1, memory_limit='10GB')
    client = Client(cluster)
    print(f"Dashboard link: {client.dashboard_link}")

    polys = gpd.read_file("../../data/GBR_names_reprojected.shp")

    print("Mapping files to workers...")
    parquet_files = glob.glob(os.path.join(intersection_dir, '**/*.parquet'), recursive=True)

    futures = client.map(
        process_parquet_file,
        parquet_files,
        comp_times=comp_times,
        comp_probs=comp_probs
    )

    # --- 2. Asynchronous Wait (No data gathering) ---
    print(f"Processing {len(futures)} files directly to disk...")
    for i, future in enumerate(as_completed(futures)):
        future.result()
        future.release()

        if (i + 1) % 100 == 0:
            print(f"Completed {i + 1}/{len(futures)} files...")

    # Close the client and workers to free up all their RAM
    client.close()
    cluster.close()
    gc.collect()

    print("Complete. Now run part 2")
