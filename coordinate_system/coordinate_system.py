import numpy as np
import pymap3d.enu
import os.path
import json


class CoordinateSystem:
    def __init__(
            self,
            origin='DISC',
            year=2022
    ):
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
        return pymap3d.enu.enu2geodetic(
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
        if deg:
            return self.__origin
        else:
            origin = np.copy(self.__origin)
            origin[:2] *= np.pi / 180.
            return origin
