
import geopandas as gpd 
import pandas as pd

sector_to_extract = "Swain"

decimal_matrix = "gbr_connectivity_decimal_G.retiformas.csv"
polygons_gdf = gpd.read_file("~/work/projects/PalaeoTides/gbr_connectivity/scripts/GBR_names_reprojected.shp")
sectors = ["Cape_Grenville", "Princess_Charlotte_B", "Cook_Town",
           "Cairns", "Innisfail", "Townsville", "Cape_Upstart", 
           "Whitsunday", "Pompey", "Swain", "Capricorn_Bunker"]

reefs_in_sector_gdf = polygons_gdf[polygons_gdf['SECTOR'] == sector_to_extract]

connectivity_matrix = pd.read_csv(decimal_matrix)

reef_names = reefs_in_sector_gdf["LOC_NAME_S"].to_list()
valid_names = [name for name in reef_names if name in connectivity_matrix.index and name in connectivity_matrix.columns]
subset_matrix = connectivity_matrix.loc[valid_names, valid_names]

subset_matrix.to_csv('filtered_connectivity_matrix_'+sector_to_extract+'.csv', index=True, header=True)
