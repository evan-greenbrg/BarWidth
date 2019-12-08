import math
import statistics
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

import matplotlib.pyplot as plt
import matplotlib.image as mpimgkjk
import matplotlib.colors as colors



class BarHandler():

    def __init__(self):
        self.slope_threshold = 1
        self.slope_smooth = 10

    def get_bar_xsections(self, coordinates, xsections, bar):
        tree = spatial.KDTree(coordinates[['easting', 'northing']])
        distance, upstream_n = tree.query(
            [(bar['upstream_easting'], bar['upstream_northing'])],
            1
        )
        distance, downstream_n = tree.query(
            [(bar['downstream_easting'], bar['downstream_northing'])],
            1
        )
        
        return xsections[upstream_n[0]:downstream_n[0]]

    def find_bar_width(self, banks):
        """
        Simplest approach I can implement. Use the extrema found from the
        Width finder. Pick the side that has the greatest difference between
        the bank maxima and minima
	Returns a sign:
	    - Positive means the bar is on the positive side 
	    - Negative means the bar is on the negative side of the channel
        """
        if not banks:
            return False

        distance0 = abs(banks[0][0] - banks[0][1])
        distance1 = abs(banks[1][0] - banks[1][1])

        if distance0 > distance1:
            return distance0
        elif distance1 > distance0:
            return distance1


    def find_bar_side1(self, bar_section, dem_col):
        """
        Could implement automated method to find the side of the bar
        Going to hold off on this for now, adding it in the input DF
        """

        # Initialize samples across bar
        bar_len = len(bar_sections)
        quarter_len = int(bar_len/4)
        side_test = [quarter_len, quarter_len * 2, quarter_len * 3]

        zero_bank = 0
        one_bank = 0
        for idx in side_test:
            # Get the distances and dem value
            t = test_bar[idx]['xsection'][dem_col]
            p = test_bar[idx]['xsection']['distance']
            endpoints = test_bar[idx]['ch_endpoints']

            # Find the indexes of the endpoints in the p, t vector
            i0 = [i for i, x in enumerate(p) if x == endpoints[0]]
            i1 = [i for i, x in enumerate(p) if x == endpoints[1]]

            # Define the endpoints
            endpoint0 = (endpoint[0], t[i0][0])
            endpoint1 = (endpoint[1], t[i1][0])

            # Define the index vector for the two xsection directions
            i_ = [i0[0], i1[0]]
            direction0 = [*range(min(i_), max(i_), 1)]
            direction1 = np.flip(direction0, 0)

            # Find the mean slope of each channel bank
            mean0, side0 = self.descend_bank(p, t, direction0)
            mean1, side1 = self.descend_bank(p, t, direction1)

            # Make the vote
            if mean0 > mean1:
                zero_bank += 1
            elif mean1 > mean0:
                one_bank +=1

        # Return the results
        if zero_bank > one_bank:
            print('zero')
            return side0
        elif one_bank > zero_bank:
            print('one')
            return side1
        else:
            return None

    def descend_bank(self, p, t, direction):
        thresh = max(t) / (max(p) * self.slope_threshold)
        slopes = []
        for i, idx in enumerate(direction):
            if i < self.slope_smooth:
                slope = (
                    (t[idx + self.slope_smooth] - t[idx]) 
                    / (p[idx + self.slope_smooth] - p[idx])
                )
                slopes.append(slope)
            elif i > len(direction0) - self.slope_smooth:
                slope = (
                    (t[idx] - t[idx - self.slope_smooth]) 
                    / (p[idx] - p[idx - self.slope_smooth])
                )
                slopes.append(slope)
            else:
                slope = (
                    (t[idx + self.slope_smooth] - t[idx - self.slope_smooth]) 
                    / (p[idx + self.slope_smooth] - p[idx - self.slope_smooth])
                )
                slopes.append(slope)

            if i == 0:
                continue

            elif slopes[i] < 0 and (abs(slopes[i-1]) - abs(slopes[i]) < thresh):
                slope_mean = statistics.mean(slopes)
                bar_points = (p[direction[0]], p[idx])

                return (
                    statistics.mean([abs(i) for i in slopes]), 
                    (int(p[0]), int(p[idx]))
                )

            else: 
                diff = abs(slopes[i-1] - slopes[i])
                continue
