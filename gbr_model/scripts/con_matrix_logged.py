import geopandas as gpd 
import numpy as np
from datetime import timedelta as delta
import xarray as xr
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib import ticker
from matplotlib.ticker import PercentFormatter
import pandas as pd
import time
from datetime import datetime
from dask.distributed import Client
import dask.dataframe as dd
from numpy import interp

competency_curve = "../../data/G.retiformis_rubble.csv"
intersection_file = '../../../modern_0.5km_high_diff/intersection_events_0.5kmclumps_output' # from generate_source_sinks.py 
raw_matrix_csv = '../../../modern_0.5km_high_diff/gbr_connectivity_G.retiformas.csv'
output_graphic = "../../../modern_0.5km_high_diff/GBR_G.retiformas.pdf"
decimal_matrix = "../../../modern_0.5km_high_diff/gbr_connectivity_decimal_G.retiformas.csv"

if __name__ == "__main__":
    ########################################
    ## RUN generate_source_sinks.py first ##
    ########################################

    comp_curve = pd.read_csv(competency_curve)

    # time in days
    def survival_func(times):
        comp = np.interp(times, comp_curve["Time"], comp_curve["Probability"])
        return comp*np.exp(-0.05*times)

    client = Client()

    #import reef polygon shp file
    polys = gpd.read_file("../../data/GBR_names_reprojected.shp")

    #load in intersection data which is a dask dataframe
    ddf = dd.read_parquet(intersection_file, engine='pyarrow')

    #which particles start in which reef
    filtered_ddf = ddf[ddf['time'] == pd.Timedelta(0)]
    result_df = filtered_ddf.compute() # takes a while...
    merged_df = result_df.merge(
        polys, 
        left_on='polygon_id', 
        right_index=True, 
        how='left'
    )
    joined_start = merged_df[['trajectory','LOC_NAME_S','SECTOR','Y_COORD']]
    start = joined_start.groupby('LOC_NAME_S', sort=False)['trajectory'].agg(list).to_dict() 
    #dictionary with reef names as key and the trajectories starting that reef as values
    del merged_df

    # which particles finish in which reef

    # Define a function to process each partition independently
    def find_local_first_true(df):
        if df.empty:
            return df
        
        times = df["time"].dt.days.to_numpy()
        # 1. Calculate boolean logic (Vectorized is fastest)
        # Note: Using numpy arrays is much faster than df.apply()
        probs = np.random.rand(len(times))
        is_hit = probs < survival_func(times)  # Replace with your specific probability logic
        
        # 2. Filter immediately to drastically reduce data volume
        hits = df[is_hit]
        
        if hits.empty:
            return hits
            
        # 3. Local Reduction: Find the first hit per trajectory in THIS partition
        # We sort by time to ensure we keep the earliest event in this chunk
        hits = hits.sort_values('time')
        return hits.drop_duplicates(subset='trajectory', keep='first')

    # Apply the function to all 5130 partitions
    # meta=ddf._meta tells Dask the output structure matches the input structure
    reduced_ddf = ddf.map_partitions(find_local_first_true, meta=ddf._meta)

    # 4. Compute: Bring the reduced results into local memory
    # Even with 25M trajectories, the filtered result (1 row per trajectory) 
    # should fit in RAM (approx. 500MB - 1GB).
    local_df = reduced_ddf.compute()

    # 5. Final Global Reduction (Pandas)
    # Since a trajectory might appear in multiple partitions (e.g., partition 1 has time 10:00, 
    # partition 2 has time 09:00), we must sort and dedup one last time.
    final_result = (
        local_df
        .sort_values('time')
        .drop_duplicates(subset='trajectory', keep='first')
    )

    joined_end = final_result.merge(
        polys, 
        left_on='polygon_id', 
        right_index=True, 
        how='left'
    )
    del final_result
    del local_df
    joined_end = joined_end[['LOC_NAME_S','SECTOR','Y_COORD','trajectory']] # dataframe

    # sort by the desired orer of north to south, within sectors
    sectors = ["Cape_Grenville", "Princess_Charlotte_B", "Cook_Town",
               "Cairns", "Innisfail", "Townsville", "Cape_Upstart", 
               "Whitsunday", "Pompey", "Swain", "Capricorn_Bunker"]
    polys["SECTOR"] = pd.Categorical(polys["SECTOR"], categories = sectors)
    polys.sort_values(by=["SECTOR","Y_COORD"],inplace=True, ascending=[True,False])
    poly_to_idx = {name: idx for idx, name in enumerate(polys.LOC_NAME_S)}

    start_end_values  = []
    count = 0
    for sector in sectors:
        start_end_values.append([count,count+len(polys[polys["SECTOR"] == sector])-1])
        count = count + len(polys[polys["SECTOR"] == sector])

    # connectivity matrix
    # axis 0 is source (rows), axis 1 is sink (columns)
    matrix = np.zeros((len(polys),len(polys)))
    matrix_dec = np.zeros((len(polys),len(polys)))
    for reef_name in start.keys():
        start_reef_idx = poly_to_idx[reef_name]  
        start_ids = joined_start[joined_start["LOC_NAME_S"] == reef_name]["trajectory"].to_numpy()
        end_reefs = joined_end[joined_end['trajectory'].isin(start_ids)]
        end_count = end_reefs.groupby("LOC_NAME_S").size()
        for end_key in end_count.keys():
            matrix[start_reef_idx,poly_to_idx[end_key]] = end_count[end_key]
            matrix_dec[start_reef_idx,poly_to_idx[end_key]] = end_count[end_key]/len(start_ids)


    # turn matrix into dataframe
    column_names = polys["LOC_NAME_S"]
    row_names = polys["LOC_NAME_S"]
    con_matrix = pd.DataFrame(matrix, index=row_names, columns=column_names) #convert to pandas
    con_matrix.to_csv(raw_matrix_csv)

    # calculate connectivity in decimals
    matrix_dec = matrix_dec.round(6)
    con_matrix = pd.DataFrame(matrix_dec, index=row_names, columns=column_names) #convert to pandas    
    con_matrix.to_csv(decimal_matrix)

    # remove zeros and log scale for vis
    matrix_dec[matrix_dec == 0] = 1e-9    
    connectivity_log = np.log10(matrix_dec)

    # plot connectivity matrix. Source is the rows. Sink is columns
    sectors_pretty = [x.replace("_", " ") for x in sectors]
    f = plt.figure(figsize=(19, 15))
    plt.matshow(connectivity_log, fignum=f.number)
    a = f.get_axes()[0]
    plt.grid()
    cb = plt.colorbar()
    cb.ax.set_ylabel('Connectivity', size=14)
    # plots the sector labels
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
#plt.show()
