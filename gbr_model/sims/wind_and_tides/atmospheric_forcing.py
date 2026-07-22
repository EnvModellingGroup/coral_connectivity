"""
ERA5 forcing
"""
from thetis import *
from thetis import forcing

class ERA5Interpolator(object):
    """
    Interpolates ERA5 atmospheric model data on 2D fields.
    """
    @PETSc.Log.EventDecorator("thetis.ATMInterpolator.__init__")
    def __init__(self, function_space, wind_stress_field,
                 atm_pressure_field, coord_system,
                 ncfile, init_date,
                 vect_rotator=None,
                 east_wind_var_name='u10', north_wind_var_name='v10',
                 pressure_var_name='msl', fill_mode=None,
                 fill_value=numpy.nan,
                 verbose=False):
        """
        :arg function_space: Target (scalar) :class:`FunctionSpace` object onto
            which data will be interpolated.
        :arg wind_stress_field: A 2D vector :class:`Function` where the output
            wind stress will be stored.
        :arg atm_pressure_field: A 2D scalar :class:`Function` where the output
            atmospheric pressure will be stored.
        :arg coord_system: :class:`CoordinateSystem` object
        :arg ncfile: A file name for reading the atmospheric data
        :arg init_date: A :class:`datetime` object that indicates the start
            date/time of the Thetis simulation. Must contain time zone. E.g.
            'datetime(2006, 5, 1, tzinfo=pytz.utc)'
        :kwarg vect_rotator: function that rotates vectors from ENU coordinates
            to target function space (optional).
        :kwarg east_wind_var_name, north_wind_var_name, pressure_var_name:
            wind component and pressure field names in netCDF file.
        :kwarg fill_mode: Determines how points outside the source grid will be
            treated. If 'nearest', value of the nearest source point will be
            used. Otherwise a constant fill value will be used (default).
        :kwarg float fill_value: Set the fill value (default: NaN)
        :kwarg bool verbose: Se True to print debug information.
        """
        self.function_space = function_space
        self.wind_stress_field = wind_stress_field
        self.atm_pressure_field = atm_pressure_field

        # construct interpolators
        self.grid_interpolator = interpolation.NetCDFLatLonInterpolator2d(
            self.function_space, coord_system, fill_mode=fill_mode,
            fill_value=fill_value)
        var_list = [east_wind_var_name, north_wind_var_name, pressure_var_name]
        self.reader = interpolation.NetCDFSpatialInterpolator(
            self.grid_interpolator, var_list)
        self.timesearch_obj = interpolation.NetCDFTimeSearch(ncfile, init_date, interpolation.NetCDFTimeParser, time_variable_name="valid_time", verbose=False)
        self.time_interpolator = interpolation.LinearTimeInterpolator(self.timesearch_obj, self.reader)
        lon = self.grid_interpolator.mesh_lonlat[:, 0]
        lat = self.grid_interpolator.mesh_lonlat[:, 1]
        if vect_rotator is None:
            self.vect_rotator = coord_system.get_vector_rotator(lon, lat)
        else:
            self.vect_rotator = vect_rotator

    @PETSc.Log.EventDecorator("thetis.ERA5Interpolator.set_fields")
    def set_fields(self, time):
        """
        Evaluates forcing fields at the given time.
        Performs interpolation and updates the output wind stress and
        atmospheric pressure fields in place.
        :arg float time: Thetis simulation time in seconds.
        """
        east_wind, north_wind, prmsl = self.time_interpolator(time)
        east_strs, north_strs = forcing.compute_wind_stress(east_wind, north_wind)
        u_strs, v_strs = self.vect_rotator(east_strs, north_strs)
        self.wind_stress_field.dat.data_with_halos[:, 0] = u_strs
        self.wind_stress_field.dat.data_with_halos[:, 1] = v_strs
        self.atm_pressure_field.dat.data_with_halos[:] = prmsl

