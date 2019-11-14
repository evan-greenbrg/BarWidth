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
from scipy import spatial 

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
        # Convert pixel value to coordinates
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

    def knn_smoothing(self, df, n=3):
        """
        Uses a KNN smoothing to smooth the river centerline
        This does not downsample
        """
        xs = []
        ys = []
        tree = spatial.KDTree(df)
        for idx, row in df.iterrows():
            if not row[1]:
                xs.append(None)
                ys.append(None)
            else:
                distance, neighbors = tree.query(
                    [(row[0], row[1])],
                    n-1
                )
                neighbor_df = df.iloc[neighbors[0]]
                x = list(neighbor_df.iloc[:,1])
                x.append(row[1])
                y = list(neighbor_df.iloc[:,0])
                y.append(row[0])

                xs.append(sum(x) / n)
                ys.append(sum(y) / n)

        return xs, ys

    def xsection_smoothing(self, idx, section, smoothing):
        """
        Smooths out the channel cross section
        The logic is a little difficult because numpy
        makes you create a new dataframe to add a column to a struct
        Also includes logic to remove the large negative values at non-existent
        from the raw DEM values
        """
        d = {'position': section['position'], 'demvalue': section['demvalue']}
        df = pandas.DataFrame(data=d)
        df = df.fillna(False)
        demvalues, position = self.knn_smoothing(df, n=smoothing)
        # For values off the DEM, the value will be less than 0 -> set to False
        demvalues = [
            False if not x else False if x < 0 else x 
            for x in demvalues
        ]

        b_dt = np.dtype(
            section.dtype.descr + [('demvalue_sm', 'f4')]
        )
        b = np.zeros(section.shape, dtype=b_dt)
        b['demvalue_sm'] = demvalues

        for col in section.dtype.names:
            b[col] = section[col]

        return b

    def get_direction(self, coordinates, smooth=5):
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
        coordinates['dlon_inv'] = coordinates['dlat']
        coordinates['dlat_inv'] = -1 * coordinates['dlon']

        return coordinates

