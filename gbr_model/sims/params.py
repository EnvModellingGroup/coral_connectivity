import datetime
import utm
from utm import*

# path relative to the root dir of this template. Leave as mesh/blah.msh in most cases
mesh_file = 'mesh/improved_modern.msh'
forcing_boundary = 666
utm_zone = 56
utm_band="K"
cent_lat = -16.5
cent_lon = 148.5
spin_up = 432000 # 5 days
end_time = 3024000 # 35 days 604800 # 345600 40 days / 604800 # 7 days
output_dir = "output"
output_time = 900
constituents = ['M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'Q1', 'M4']
# year, month, day, hour, min, sec
start_datetime = datetime.datetime(2000,11,12,0,0,0)
