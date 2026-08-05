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

def smagdiff(particle, fieldset, time):
    dx = 500
    # gradients are computed by using a local central difference.
    updx, vpdx = fieldset.UV[time, particle.depth, particle.lat, particle.lon + dx]
    umdx, vmdx = fieldset.UV[time, particle.depth, particle.lat, particle.lon - dx]
    updy, vpdy = fieldset.UV[time, particle.depth, particle.lat + dx, particle.lon]
    umdy, vmdy = fieldset.UV[time, particle.depth, particle.lat - dx, particle.lon]

    dudx = (updx - umdx) / (2 * dx)
    dudy = (updy - umdy) / (2 * dx)

    dvdx = (vpdx - vmdx) / (2 * dx)
    dvdy = (vpdy - vmdy) / (2 * dx)

    A = fieldset.cell_areas[time, 0, particle.lat, particle.lon]
    Kh = fieldset.Cs * A * math.sqrt(dudx**2 + 0.5 * (dudy + dvdx) ** 2 + dvdy**2)

    dlat = parcels.ParcelsRandom.normalvariate(0.0, 1.0) * math.sqrt(
        2 * math.fabs(particle.dt) * Kh
    )
    dlon = parcels.ParcelsRandom.normalvariate(0.0, 1.0) * math.sqrt(
        2 * math.fabs(particle.dt) * Kh
    )

    particle_dlat += dlat
    particle_dlon += dlon

# for real: load in the points
all_points_x = np.loadtxt("x_points_high_9km.csv")
all_points_y = np.loadtxt("y_points_high_9km.csv")

directory = "../sims/wind_and_tides/output/hdf5/"
filenames = {'U': os.path.join(directory,"Velocity2d_9000_U.nc"), #change this name
             'V': os.path.join(directory,"Velocity2d_9000_V.nc")} #change this name too
variables = {'U': 'Band1',
             'V': 'Band1'}
dimensions = {'lat': 'y',
              'lon': 'x',
              'time': 'time'}
model_output_step = 900
U = Field.from_netcdf(filenames["U"], {"U":"Band1"}, dimensions, mesh="flat")
V = Field.from_netcdf(filenames["V"], {"V":"Band1"}, dimensions, mesh="flat")

fieldset = FieldSet(U, V)
x = fieldset.U.grid.lon
y = fieldset.U.grid.lat
cell_areas = Field(
    name="cell_areas", data=fieldset.U.cell_areas(), lon=x, lat=y
)
fieldset.add_field(cell_areas)
fieldset.add_constant("Cs", 0.1)

def DeleteErrorParticle(particle, fieldset, time):
    if particle.state >= 40:  # deletes every particle that throws an error
        particle.delete()

pset = ParticleSet.from_list(fieldset=fieldset, pclass=JITParticle,
                             lon=all_points_x,   # releasing on a line: the start longitude and latitude
                             lat=all_points_y)  # releasing on a line: the end longitude and latitude


output_file = pset.ParticleFile(name="/home/jh1889/work/projects/PalaeoTides/gbr_connectivity/modern_9km_high_diff/Trajectory_tides_wholeGBR_1024.zarr", outputdt=900)
pset.execute([AdvectionRK4, smagdiff, DeleteErrorParticle],
             runtime=1209600,# 14 days, 16000, 171900, 1255885
             dt=120,
             output_file=output_file)

