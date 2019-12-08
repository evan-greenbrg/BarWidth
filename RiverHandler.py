import math
import cv2
import gdal
import numpy as np
import pandas
from pyproj import Proj
from rivamap import preprocess, singularity_index, delineate, georef, visualization
import rasterio
from rasterio.plot import show
from scipy.signal import argrelextrema
from scipy.ndimage import label as ndlabel
from scipy.ndimage import sum as ndsum
from scipy import spatial 

from RasterHandler import RasterHandler


class RiverHandler():

    def __init__(self):
        pass

    def get_centerline(self, B3, B6, size):
        """
        Using RivaMap library finds the largest centerline from 
        Bands 3 and 6 lansat 8 imagery
        """
        I1 = preprocess.mndwi(B3, B6)
        filters = singularity_index.SingularityIndexFilters()
        psi, widthMap, orient = singularity_index.applyMMSI(I1, filters)
        nms = delineate.extractCenterlines(orient, psi)
        
        return delineate.getBiggestCenterline(nms, size)

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
        d = {'distance': section['distance'], 'demvalue': section['demvalue']}
        df = pandas.DataFrame(data=d)
        df = df.fillna(False)
        demvalues, distance = self.knn_smoothing(df, n=smoothing)
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
        coordinates['dlon_inv'] = coordinates['dlat']
        coordinates['dlat_inv'] = -1 * coordinates['dlon']

        return coordinates

    def find_channel_width(self, p, t, order):
        """
        Finds the endpoints of the channel from which the width will be 
        calculated. Uses the biggest difference between adjeacent maxima and
        minima. It will then take the lowest of the two extremes (big->small
        and small-> big) and project across the stream. Project works because
        the reference frame of the model has the centerline as distance = 0
        """
        data = {'distance': p, 'elevation': t}
        cross_section = pandas.DataFrame(data=data, columns=['distance', 'elevation'])

        # Find Maxima and Minima
        maxs = argrelextrema(t, np.greater, order=order)
        maxima = np.column_stack((p[maxs], t[maxs], np.full(len(t[maxs]), 0)))
        mins = argrelextrema(t, np.less, order=order)
        minima = np.column_stack((p[mins], t[mins], np.full(len(t[mins]), 1)))

        extremes = np.concatenate([maxima, minima])
        extremes = extremes[extremes[:,0].argsort()]

        # Get biggest difference between ADJACENT maxima and minma
        d = []
        for i in range(0, len(extremes)):
            minmax = extremes[i][2]
            if i == len(extremes) - 1:
                d.append(0) 
            elif extremes[i+1][2] == minmax:
                d.append(0)
                continue
            else:
                diff = extremes[i+1][1] - extremes[i][1]
                d.append(diff)
        maxi = np.where(d == np.amax(d))[0][0]
        mini = np.where(d == np.amin(d))[0][0]

        # Error Handling
        if (len(extremes) - 1 < maxi + 1) or (len(extremes) - 1 < mini + 1):
            print('No Channel Found')

            return False,None

        # Save the banks for later
        banks = [
            extremes[maxi + 1][0], 
            extremes[maxi][0],
            extremes[mini][0], 
            extremes[mini + 1][0]
        ]

        try:
            max_val = extremes[maxi + 1]
        except IndexError:
            max_val = extremes[maxi]
        min_val = extremes[mini]

        # Take lowest of the two and project across the stream
        if min_val[1] >= max_val[1]:
            opposite_channel_section = cross_section[
                (cross_section['distance'] > min_val[0]) 
                & (cross_section['distance'] < max_val[0])
                & (cross_section['distance'] < 0)
            ]
            width_val = max_val
            banks[2] = None
        else:
            opposite_channel_section = cross_section[
                (cross_section['distance'] > min_val[0]) 
                & (cross_section['distance'] < max_val[0])
                & (cross_section['distance'] > 0)
            ]
            width_val = min_val
            banks[0] = None

        # Find opposite bank projection
        try:
            opposite_val = opposite_channel_section.iloc[
                (
                    opposite_channel_section['elevation']
                    - width_val[1]
                ).abs().argsort()[:1]
            ].to_numpy()[0]
            banks = [opposite_val[0] if not x else x for x in banks]

            width = abs(width_val[0]) + abs(opposite_val[0])

            return [tuple(banks[0:2]), tuple(banks[2:4])], width
        except IndexError:
            print('No Channel Found')

            return False,None
