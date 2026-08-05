import os
import duckdb
import geopandas as gpd 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

raw_matrix_csv = '../../../modern_0.5km_high_diff/gbr_connectivity_G.retiformis.csv'
output_graphic = "../../../modern_0.5km_high_diff/GBR_G.retiformis.pdf"
decimal_matrix = "../../../modern_0.5km_high_diff/gbr_connectivity_decimal_G.retiformis.csv"

TEMP_STARTS_DIR = '../../../modern_0.5km_high_diff/temp_starts'
TEMP_HITS_DIR = '../../../modern_0.5km_high_diff/temp_hits'

if __name__ == "__main__":
    
    print("Initializing DuckDB Out-of-Core Engine...")
    
    # Create a database file on disk so DuckDB can safely spill over RAM limits without crashing
    con = duckdb.connect('gbr_processing.db')
    
    # Tell DuckDB it has a hard limit, keeping your system safe
    con.execute("PRAGMA memory_limit='100GB'")
    # Utilize multiple CPU cores for reading the 34k files
    con.execute("PRAGMA threads=8")

    # --- 1. Process Starts ---
    print("Extracting unique starts directly from disk...")
    # This query reads all files, partitions them by trajectory, and keeps only the first one
    starts_query = f"""
        SELECT trajectory, polygon_id FROM (
            SELECT trajectory, polygon_id, 
                   ROW_NUMBER() OVER (PARTITION BY trajectory) as rn 
            FROM read_parquet('{TEMP_STARTS_DIR}/*.parquet')
        ) WHERE rn = 1
    """
    result_df = con.execute(starts_query).df()
    print(f"Starts extracted: {len(result_df)} unique trajectories.")

    # --- 2. Process Hits ---
    print(f"Extracting first hits from 34,000+ files (This may take a few minutes, but it won't crash)...")
    # This query sorts by time on the fly and extracts the earliest hit per trajectory
    hits_query = f"""
        SELECT trajectory, polygon_id FROM (
            SELECT trajectory, polygon_id, 
                   ROW_NUMBER() OVER (PARTITION BY trajectory ORDER BY time ASC) as rn 
            FROM read_parquet('{TEMP_HITS_DIR}/*.parquet')
        ) WHERE rn = 1
    """
    local_df = con.execute(hits_query).df()
    print(f"Hits extracted: {len(local_df)} valid connections.")
    
    # Close connection and clean up the database file
    con.close()
    if os.path.exists('gbr_processing.db'):
        os.remove('gbr_processing.db')

    # --- 3. Matrix Generation ---
    print("Mapping to polygons and building matrix...")
    polys = gpd.read_file("../../data/GBR_names_reprojected.shp")

    merged_df = result_df.merge(
        polys, left_on='polygon_id', right_index=True, how='left'
    )
    joined_start = merged_df[['trajectory','LOC_NAME_S','SECTOR','Y_COORD']]
    start = joined_start.groupby('LOC_NAME_S', sort=False)['trajectory'].agg(list).to_dict() 
    del merged_df, result_df

    joined_end = local_df.merge(
        polys, left_on='polygon_id', right_index=True, how='left'
    )
    del local_df
    joined_end = joined_end[['LOC_NAME_S','SECTOR','Y_COORD','trajectory']]

    sectors = ["Cape_Grenville", "Princess_Charlotte_B", "Cook_Town",
               "Cairns", "Innisfail", "Townsville", "Cape_Upstart", 
               "Whitsunday", "Pompey", "Swain", "Capricorn_Bunker"]
    polys["SECTOR"] = pd.Categorical(polys["SECTOR"], categories = sectors)
    polys.sort_values(by=["SECTOR","Y_COORD"], inplace=True, ascending=[True,False])
    poly_to_idx = {name: idx for idx, name in enumerate(polys.LOC_NAME_S)}

    start_end_values  = []
    count = 0
    for sector in sectors:
        start_end_values.append([count, count+len(polys[polys["SECTOR"] == sector])-1])
        count = count + len(polys[polys["SECTOR"] == sector])

    matrix = np.zeros((len(polys), len(polys)))
    matrix_dec = np.zeros((len(polys), len(polys)))
    
    for reef_name in start.keys():
        start_reef_idx = poly_to_idx[reef_name]  
        start_ids = joined_start[joined_start["LOC_NAME_S"] == reef_name]["trajectory"].to_numpy()
        end_reefs = joined_end[joined_end['trajectory'].isin(start_ids)]
        end_count = end_reefs.groupby("LOC_NAME_S").size()
        
        for end_key in end_count.keys():
            matrix[start_reef_idx, poly_to_idx[end_key]] = end_count[end_key]
            matrix_dec[start_reef_idx, poly_to_idx[end_key]] = end_count[end_key] / len(start_ids)

    column_names = polys["LOC_NAME_S"]
    row_names = polys["LOC_NAME_S"]
    
    pd.DataFrame(matrix, index=row_names, columns=column_names).to_csv(raw_matrix_csv)
    
    matrix_dec = matrix_dec.round(6)
    pd.DataFrame(matrix_dec, index=row_names, columns=column_names).to_csv(decimal_matrix)

    # --- 4. Plotting ---
    print("Generating plot...")
    matrix_dec[matrix_dec == 0] = 1e-9    
    connectivity_log = np.log10(matrix_dec)

    sectors_pretty = [x.replace("_", " ") for x in sectors]
    f = plt.figure(figsize=(19, 15))
    plt.matshow(connectivity_log, fignum=f.number)
    a = f.get_axes()[0]
    plt.grid()
    cb = plt.colorbar()
    cb.ax.set_ylabel('Connectivity', size=14)
    
    midpoints = []
    major = [0]
    for values in start_end_values:
        midpoints.append(values[0] + (values[1] - values[0]) / 2)
        major.append(values[1])
        
    a.xaxis.set_major_formatter(ticker.NullFormatter())
    a.xaxis.set_major_locator(ticker.FixedLocator(major))
    a.xaxis.set_minor_locator(ticker.FixedLocator(midpoints))
    a.xaxis.set_minor_formatter(ticker.FixedFormatter(sectors_pretty))
    a.tick_params('both', length=10, width=2, which='major')
    a.xaxis.set_label_position('top') 
    a.tick_params("x", rotation=90, which="minor")

    a.yaxis.set_major_locator(ticker.FixedLocator(major))
    a.yaxis.set_major_formatter(ticker.NullFormatter())
    a.yaxis.set_minor_locator(ticker.FixedLocator(midpoints))
    a.yaxis.set_minor_formatter(ticker.FixedFormatter(sectors_pretty))
    plt.ylabel('Source Reefs', size=14)
    plt.xlabel('Sink Reefs', size=14)
    plt.savefig(output_graphic, bbox_inches='tight')
    
    print("Pipeline complete. You conquered the memory limit.")
