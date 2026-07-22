import shutil
from thetis import *
import os.path
import sys
import math
import numpy as np
import argparse
import uptide
import re

# is there are any other possible h5 output files, you'll need to add to this mapping
# Mapping is filename (without h5 extension) to the function name
# chk = CheckpointFile('tidal_range', 'w')
# chk.save_function(tr, name='TidalRange')
# would map to "tidal_range" : "TidalRange"
mapping = {
        'tidal_range': "TidalRange",
        'min_fs' : 'MinFS',
        'max_fs' : 'MaxFS',
        'average_speed' : "AveSpeed",
        'max_speed' : "MaxSpeed",
        'average_bss' : "AveBSS",
        'max_bss' : "MaxBSS",
        'average_vel' : "AveVel",
        'max_vel' : "MaxVel",
        'Elevation2d' : "elev_2d",
        'Velocity2d' : "uv_2d",
        }

# add all possible tidal components to mapping
for c in uptide.ALL_FES2014_TIDAL_CONSTITUENTS:
    mapping[c+"_amp"] = c+"_amp"
    mapping[c+"_phase"] = c+"_phase"
    mapping[c+"_phase_mod_pi"] = c+"_phase"

#create function
def main():

    parser = argparse.ArgumentParser(
         prog="create_raster_from_h5",
         description="""Will create a raster files for a h5 file"""
    )
    parser.add_argument(
            '--resolution',
            help="Set the resolution of the XY file. Default is 5000m",
            type=float,
            default=5000
            )
    parser.add_argument(
            'input_file',
            metavar='input_file',
            help='The h5 file to rasterise. Without the .h5'
            )
    parser.add_argument(
            '--mesh',
            metavar='mesh',
            help='The mesh file.'
            )
    parser.add_argument(
            '--func',
            help="Harcoded function name. Only used when using temporal_stat* or tidal_stat* files"
            )
    parser.add_argument(
            '--wd_mask',
            help="Use the elevation (supply h5 filename) and the elevation to mask dry areas in a velocity"
            )
    parser.add_argument(
            '--min_max',
            type=float,
            nargs=4,
            help='The min and max of x and y in order: minx maxx miny maxy. Default is whole domain'
            )
    parser.add_argument(
            '--velocity',
            action="store_true",
            help='Your h5 file is a velocity, so a U and V raster will be produced',
            )
    parser.add_argument(
            '-v',
            '--verbose',
            action='store_true',
            help="Verbose output: mainly progress reports.",
            default=False
            )
    parser.add_argument(
            'output_file',
            metavar='output_file',
            help='The output xyz file. We add the xyz extension and the thetis timestep if included in the input file.'
            )
    args = parser.parse_args()
    verbose = args.verbose
    resolution = args.resolution
    input_file = args.input_file
    meshfile = args.mesh
    output_file = args.output_file
    min_max = args.min_max
    velocity = args.velocity
    func = args.func
    wd_mask = args.wd_mask
    if wd_mask == None:
        masking = False
    else:
        masking = True
    if verbose:
        PETSc.Sys.Print("loading mesh")   

    # need to check that min_max or mesh have been supplied
    
    # check filename doesn't end in h5

    #load Mesh
    if min_max == None:
        mesh2d = Mesh(meshfile)

        # produces coords per core. Damn.
        min_coords = np.amin(mesh2d.coordinates.dat.data_ro, axis=0)
        max_coords = np.amax(mesh2d.coordinates.dat.data_ro, axis=0)
        x_min = min_coords[0]
        y_min = min_coords[1]
        x_max = max_coords[0]
        y_max = max_coords[1]
        comm = COMM_WORLD
        # so we need to do an all reduce
        import mpi4py.MPI
        global_x_max = np.zeros(1)
        global_x_min = np.zeros(1)
        global_y_max = np.zeros(1)
        global_y_min = np.zeros(1)
        comm.Barrier()
        comm.Allreduce(x_max, global_x_max, op=MPI.MAX)
        comm.Allreduce(x_min, global_x_min, op=MPI.MIN)
        comm.Allreduce(y_max, global_y_max, op=MPI.MAX)
        comm.Allreduce(y_min, global_y_min, op=MPI.MIN)
        comm.Barrier()
        x_min = global_x_min[0]
        x_max = global_x_max[0]
        y_min = global_y_min[0]
        y_max = global_y_max[0]
    else:
        x_min = min_max[0]
        x_max = min_max[1]
        y_min = min_max[2]
        y_max = min_max[3]

    if verbose:
        PETSc.Sys.Print("Coords: ", x_min, y_min, x_max, y_max)

    # we create a list of 2D points. These are the centre points of the raster we want
    # hence the resolution/2
    # arange won't go past the max, so we don't need to account for resolution at this end
    x_coords = np.arange(x_min+(resolution/2.), x_max, resolution)
    y_coords = np.arange(y_min+(resolution/2), y_max, resolution)
    raster_coords = []
    # loop y, then x so we get the xyz format right
    for y in y_coords:
        for x in x_coords:
            raster_coords.append([x,y])

    # create empty 2D arrays - same size as the raster.
    # One for u, one for v
    # these arrays should match the size of the *output* raster file
    if (velocity):
        u_data_set = np.empty(len(raster_coords))
        v_data_set = np.empty(len(raster_coords))
        if masking:
            elev_data = np.empty(len(raster_coords))
            bathy_data = np.empty(len(raster_coords))
    else:
        data_set = np.empty(len(raster_coords))

    # loop over the requested h5 files, pulling out the velocity at
    # our raster points, then saving to xyz files for each output
    head, tail = os.path.split(input_file)
    if (tail.startswith("temporal_stats") or tail.startswith("tidal_stats")) :
        # special case where the functions are in the same file
        if (func == None):
            PETSc.Sys.Print("Need to specify a function name via --func if using the temporal or tidal stats file")
            sys.exit(-1)
        func_name = func
        timestep = None
    else:
        # loop over the requested h5 files, pulling out the velocity at
        # our raster points, then saving to xyz files for each output
        # this re searchs for the timestep in the h5 file if it exists
        # _ followed by 5 digits.
        p = re.compile(r'_\d{5}')
        # need to post_process tail (the filename) to check if it's not part of a timestep
        m = p.search(tail)
        if m:
            tail = tail[0:m.start()]
            timestep = m.group()
        else:
            timestep = None
        func_name = mapping[tail]
    if verbose:
        PETSc.Sys.Print('Reading h5 file: ', input_file)
        PETSc.Sys.Print('Looking for: ', func_name)

    with CheckpointFile(input_file, "r") as f:
        mesh2d = f.load_mesh()
        if verbose:
            PETSc.Sys.Print('Looking for: ', func_name)
        function = f.load_function(mesh2d,func_name)
        point_evaluator = PointEvaluator(mesh2d, raster_coords, missing_points_behaviour="ignore")
        
        data_raster = point_evaluator.evaluate(function)
        #        function.at(raster_coords,dont_raise=True)

        if velocity:
            # Replace None with np.nan to keep data_raster as a fast, clean float array
            data_raster = np.array([ np.array([np.nan, np.nan]) if d is None else d for d in data_raster ], dtype=float)
            
            u_data_set = np.stack(data_raster, 0)[:, 0]
            v_data_set = np.stack(data_raster, 0)[:, 1]
            
            if masking:
                elev_file = input_file.replace("Velocity", "Elevation")
                with CheckpointFile(elev_file, "r") as ef:
                    emsh = ef.load_mesh()
                    elev = ef.load_function(emsh, "elev_2d")
                    e_point_evaluator = PointEvaluator(emsh, raster_coords, missing_points_behaviour="ignore")

                    if verbose:
                        PETSc.Sys.Print('Read elevation: ' + elev_file)
                    elev_data = e_point_evaluator.evaluate(elev)
                    
                    with CheckpointFile(wd_mask, "r") as bf:
                        bmsh = bf.load_mesh()
                        bathy = bf.load_function(bmsh, "bathymetry")
                        b_point_evaluator = PointEvaluator(bmsh, raster_coords, missing_points_behaviour="ignore")
                        bathy_data = b_point_evaluator.evaluate(bathy)
                    
                    # Convert None to np.nan for points outside the mesh
                    elev_data_clean = np.array([np.nan if e is None else e for e in elev_data], dtype=float)
                    bathy_data_clean = np.array([np.nan if b is None else b for b in bathy_data], dtype=float)
                    
                    # Compute physical total depth: H = elev + bathy
                    total_depth = elev_data_clean + bathy_data_clean
                    
                    # Target dry mask
                    wd_threshold = 0.02 
                    dry = total_depth <= wd_threshold
                    u_data_set[dry] = 0.0
                    v_data_set[dry] = 0.0

            # write u to file
            if timestep:
                u_filename = output_file + "_u" + timestep + ".xyz"
            else:
                u_filename = output_file + "_u" ".xyz"
            with open(u_filename,"w") as f:
                for p,u in zip(raster_coords,u_data_set):
                    if u == None or math.isnan(u):
                        u = -9999
                    f.write(str(p[0]) + "\t" + str(p[1]) + "\t" + str(u) + "\n")
            # write v file
            if timestep:
                v_filename = output_file + "_v" + timestep + ".xyz"
            else:
                v_filename = output_file + "_v" ".xyz"
            with open(v_filename,"w") as f:
                for p,v in zip(raster_coords,v_data_set):
                    if v == None or math.isnan(v):
                        v = -9999
                    f.write(str(p[0]) + "\t" + str(p[1]) + "\t" + str(v) + "\n")
        else:
            data_set = np.stack(data_raster,0)
            if timestep:
                filename = output_file + timestep + ".xyz"
            else:
                filename = output_file + ".xyz"
            with open(filename,"w") as f:
                for p,u in zip(raster_coords,data_set):
                    if u == None or math.isnan(u):
                        u = -9999
                    f.write(str(p[0]) + "\t" + str(p[1]) + "\t" + str(u) + "\n")


if __name__ == "__main__":
    main()
