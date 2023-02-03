import numpy as np
import pymap3d.enu
import os.path
import json


class CoordinateSystem:
    """
    Class to convert between different coordinate systems. Not that the standard coordinate system
    used by RNO-G is centered around the DISC hole.
    """
    def __init__(
            self,
            origin='DISC',
            year=2022
    ):
        """
        Initialize the class

        :param origin: string
            Name of the reference point of the coordinate system. Should be DISC by default
        :param year: integer
            Year in which the reference was mapped. As the glacier and the buildings at
            Summit move, this may change from year to year.
        """
        filename = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'coordinate_origins.json'
        )
        with open(filename, 'r') as json_file:
            self.__origin = np.array(json.load(json_file)[origin][str(year)])

    def geodetic_to_enu(
            self,
            latitude,
            longitude,
            height=3260.,
            deg=True
    ):
        """
        Convert from geodetic coordinates (latitude, longitude, elevation)
        to ENU coordinates (east, north, up)

        :param latitude: float
            Latitude of the point
        :param longitude: float
            Longitude of the point
        :param height: float
            Height above the reference ellipsoid of the point
        :param deg: boolean
            If True, parameters are assumed to be in degrees, otherwise they are
            assumed to be in radians
        :return: array of float
            East, north and up coordinates relative to the coordinate system's reference point
        """
        origin = np.copy(self.__origin)
        if not deg:
            origin[:2] *= np.pi / 180.
        return pymap3d.enu.geodetic2enu(
            latitude,
            longitude,
            height,
            origin[0],
            origin[1],
            origin[2],
            None,
            deg
        )

    def enu_to_geodetic(
            self,
            easting,
            northing,
            height=0.,
            deg=True
    ):
        """
        Convert from ENU coordinates (east, north, up) to
        geodetic coordinates (latitude, longitude, elevation)

        :param easting: float
            East, relative to the coordinate system's reference point
        :param northing: float
            North, relative to the coordinate system's reference point
        :param height: float
            Up, relative to the coordinate system's reference point
        :param deg: boolean
            If True, geodetic coordinates will be returned in degrees, otherwise in radians
        :return: Array of floats with shape (3,)
            Latitude, longitude and elevation coordinate
        """
        origin = np.copy(self.__origin)
        if not deg:
            origin[:2] *= np.pi / 180.
        return pymap3d.enu.enu2geodetic(
            easting,
            northing,
            height,
            origin[0],
            origin[1],
            origin[2],
            None,
            deg
        )

    def enu_to_enu(
            self,
            easting,
            northing,
            height,
            origin,
            deg=True
    ):
        """
        Transform ENU (east, north, up) coordinates from a different coordinate system
        into this one.

        :param easting: float
            East, relative to the other coordinate system's origin
        :param northing: float
            North, relative to the other coordinate system's origin
        :param height: float
            Up, relative to the other coordinate system's origin
        :param origin: Array of floats with shape (3,)
            latitude, longitue and elevation of the other coordinate system's origin
        :param deg: boolean
            Specifies if the other coordinate system's origin is given in degrees or radians
        :return: Array of floats with shape (3,)
            East, north and up coordinates in this coordinate system
        """
        lon, lat, h = pymap3d.enu.enu2geodetic(
            easting,
            northing,
            height,
            origin[0],
            origin[1],
            origin[2],
            None,
            deg
        )
        new_origin = np.copy(self.__origin)
        if not deg:
            new_origin[:2] *= np.pi / 180.
        return pymap3d.enu.geodetic2enu(
            lon,
            lat,
            h,
            new_origin[0],
            new_origin[1],
            new_origin[2],
            None,
            deg
        )

    def get_origin(
            self,
            deg=True
    ):
        """
        Get the origin of this coordinate system

        :param deg: boolean
            If True, latitude and longitude are given in degrees, otherwise in radians
        :return: Array of floats with shape (3,)
            Latitude, Longitude and elevation of this coordinate system's origin.
        """
        if deg:
            return self.__origin
        else:
            origin = np.copy(self.__origin)
            origin[:2] *= np.pi / 180.
            return origin
