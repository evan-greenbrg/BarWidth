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
        demvalue = np.where(section['demvalue'] < 0, None, section['demvalue'])
        d = {'distance': section['distance'], 'demvalue': demvalue} 
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
        tree = spatial.KDTree(coordinates[['lon', 'lat']])
        for idx, row in coordinates.iterrows():
            distance, neighbors = tree.query(
                [(row['lon'], row['lat'])],
                2
            )
            # Find the max and min distance values from the nearest neighbors
            max_distance = np.argmax(distance[0])
            max_neighbor = neighbors[0][max_distance]
            min_distance = np.argmin(distance[0])
            min_neighbor = neighbors[0][min_distance]

            # Calculate lat and lon distances between coordinates
            distance = [
                (
                    coordinates.iloc[max_neighbor]['lon']
                    - coordinates.iloc[min_neighbor]['lon']
                ),
                (
                    coordinates.iloc[max_neighbor]['lat']
                    - coordinates.iloc[min_neighbor]['lat']
                )
            ]

            # Converts distance to unit distance
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

    def find_channel_width(self, section, order):
        """
        Finds the endpoints of the channel from which the width will be 
        calculated. Uses the biggest difference between adjeacent maxima and
        minima. It will then take the lowest of the two extremes (big->small
        and small-> big) and project across the stream. Project works because
        the reference frame of the model has the centerline as distance = 0

        Outputs - 
        banks: list of tuple of two channel banks - used to find the bar
        width: numeric channel width from bank top to bank top 
        points: distance indexes of the channel bank tops
        """
        # Gets the channel geometry
        p = section['distance']
        t = section['demvalue_sm']

        # Add Step where I Optimize the channel Width
        data = {'distance': p, 'elevation': t}
        cross_section = pandas.DataFrame(
            data=data, 
            columns=['distance', 'elevation']
        )

        # Find Maxima and Minima
        maxs = argrelextrema(t, np.greater, order=order)
        maxima = np.column_stack((p[maxs], t[maxs], np.full(len(t[maxs]), 0)))
        mins = argrelextrema(t, np.less, order=order)
        minima = np.column_stack((p[mins], t[mins], np.full(len(t[mins]), 1)))

        extremes = np.concatenate([maxima, minima])
        extremes = extremes[extremes[:,0].argsort()]

        if len(extremes) == 0:
            print('No Channel Found')
            return False, None, None

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
            return False, None, None

        # Save the banks for later
        banks = [
            extremes[maxi + 1][0], 
            extremes[maxi][0],
            extremes[mini][0], 
            extremes[mini + 1][0]
        ]

        # Gets the maximum value in the cross section
        try:
            max_val = extremes[maxi + 1]
        except IndexError:
            max_val = extremes[maxi]

        # Get the minimum value in the cross section
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

            if width_val[0] < 0 or opposite_val[0] < 0:
                width = abs(width_val[0]) + abs(opposite_val[0])
            else:
                if width_val[0] > opposite_val[0]:
                    width = width_val[0] - opposite_val[0]
                else:
                    width = opposite_val[0] - width_val[0]

            points = (width_val[0], opposite_val[0])

            return [tuple(banks[0:2]), tuple(banks[2:4])], width, points
        except IndexError:
            print('No Channel Found')
            return False, None, None

    def get_bank_positions(self, xsection, points):
        """
        Takes the points from the channel_widths and turns them into a 
        dataframe to save all of the channel banks for output

        Inputs - 
        xsection: Numpy structure of the cross-section
        points: Distance indexes for the channel margins
        """
        # Get the data for one side
        bank0 = xsection[xsection['distance'] == points[0]]
        data0 = np.append(bank0['easting'], bank0['northing'])

        # Get the data for the other side
        bank1 = xsection[xsection['distance'] == points[1]]
        data1 = np.append(bank1['easting'], bank1['northing'])

        data = np.vstack((data0, data1)).astype('float')

        return pandas.DataFrame(data, columns=['easting', 'northing'])
    
    def save_channel_widths(self, xsections):
        """
        Takes the major cross-section structure and parses out the
        channel widths.

        Inputs -
        xsections: Numpy structure of all channel cross-sections and their
            properties

        Outputs -
        width_df: pandas dataframe of the channel widths by position
        """
        # Find the channel bar widths
        widths = []
        eastings = []

        northings = []
        # Create a width DF
        for section in xsections:
            eastings.append(section[0][0])
            northings.append(section[0][1])
            widths.append(section[1])

        # Save as pandas dataframe
        data = {'easting': eastings, 'northing': northings, 'width': widths}
        width_df = pandas.DataFrame(
            data=data,
            columns=['easting', 'northing', 'width']
        )
        
        return width_df.reset_index()
