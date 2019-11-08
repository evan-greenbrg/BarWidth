import math
import cv2
import gdal
import numpy as np
import pandas
from pyproj import Proj
from rivamap import preprocess, singularity_index, delineate, georef, visualization
import rasterio
from rasterio.plot import show
from scipy.ndimage import label as ndlabel
from scipy.ndimage import sum as ndsum

from RasterHandler import RasterHandler


class RiverHandler():

    def __init__(self):
        pass

    def get_centerline(self, B3, B6):
        """
        Using RivaMap library finds the largest centerline from 
        Bands 3 and 6 lansat 8 imagery
        """
        I1 = preprocess.mndwi(B3, B6)
        filters = singularity_index.SingularityIndexFilters()
        psi, widthMap, orient = singularity_index.applyMMSI(I1, filters)
        nms = delineate.extractCenterlines(orient, psi)
        
        return delineate.getBiggestCenterline(nms)

    def get_river_coordinates(self, centerline, gm):
        """
        Gets lat, lon coordinates of the river centerline
        """
        river_coordinates = pandas.DataFrame(columns=['lat', 'lon'])
        for idx, row in enumerate(centerline):
            for jdx, value in enumerate(row):
                if value:
                    lat, lon = georef.pix2lonlat(gm, jdx, idx)
                    rowdf = pandas.DataFrame(
		        data=[[lat, lon]],
			columns=['lat', 'lon']
		    )
                    river_coordinates = river_coordinates.append(rowdf)

        return river_coordinates

    def get_direction(self, coordinates):
        """
        Calculates UNIT directions for each river coordinate
        This creates two columns:
            - one for the vector direction in LON
            - one for vector direction in LAT
        This is simple case that uses a forward difference model
        """

        dlon = []
        dlat = []
        for idx, row in coordinates.iterrows():
            if idx == coordinates.index.max():
                distance = [
                    row['lon'] - coordinates.iloc[idx-1]['lon'],
                    row['lat'] - coordinates.iloc[idx-1]['lat']
                ]
                norm = math.sqrt(distance[0] ** 2 + distance[1] ** 2)
                dlon_t, dlat_t = distance[0] / norm, distance[1] / norm
            else:
                distance = [
                    coordinates.iloc[idx+1]['lon'] - row['lon'],
                    coordinates.iloc[idx+1]['lat'] - row['lat']
                ]
                norm = math.sqrt(distance[0] ** 2 + distance[1] ** 2)
                dlon_t, dlat_t = distance[0] / norm, distance[1] / norm                   
            dlon.append(dlon_t)
            dlat.append(dlat_t)

        coordinates['dlon'] = dlon
        coordinates['dlat'] = dlat

        return coordinates 

    def get_inverse_direction(self, coordinates):
        """
        Calculates the direction that is inverse of the river at that point
        The convention is to swap the vector directions and make LON negative
        """
        coordinates['dlon_inv'] = -1 * coordinates['dlat']
        coordinates['dlat_inv'] = coordinates['dlon']

        return coordinates

