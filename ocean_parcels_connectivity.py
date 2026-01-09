import math
import os
import numpy as np

from parcels import (
    AdvectionRK4,
    FieldSet,
    JITParticle,
    ParticleSet,
    Field
)

filenames = {'U': "Velocity2d_U.nc", #change this name
             'V': "Velocity2d_V.nc"} #change this name too

# for real: load in the points
all_points_x = np.loadtxt("x_points2.csv")
all_points_y = np.loadtxt("y_points2.csv")

directory = "/backup_fs/Archive/kl909/chapter_1/tides/hdf5/"
filenames = {'U': os.path.join(directory,"Velocity2d_U.nc"), #change this name
             'V': os.path.join(directory,"Velocity2d_V.nc")} #change this name too
variables = {'U': 'Band1',
             'V': 'Band1'}
dimensions = {'lat': 'y',
              'lon': 'x',
              'time': 'time'}
model_output_step = 900
U = Field.from_netcdf(filenames["U"], {"U":"Band1"}, dimensions, mesh="flat")
V = Field.from_netcdf(filenames["V"], {"V":"Band1"}, dimensions, mesh="flat")

fieldset = FieldSet(U, V)

def DeleteErrorParticle(particle, fieldset, time):
    if particle.state >= 40:  # deletes every particle that throws an error
        particle.delete()

pset = ParticleSet.from_list(fieldset=fieldset, pclass=JITParticle,
                             lon=all_points_x,   # releasing on a line: the start longitude and latitude
                             lat=all_points_y)  # releasing on a line: the end longitude and latitude


output_file = pset.ParticleFile(name="Trajectory_tides_wholeGBR_1024.zarr", outputdt=900)
pset.execute([AdvectionRK4,DeleteErrorParticle],
             runtime=1209600,# 14 days, 16000, 171900, 1255885
             dt=120,
             output_file=output_file)

