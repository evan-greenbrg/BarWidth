import argparse
import errno
import random
import os
import sys
import math
from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import gdal
import osr
import numpy as np
import pandas
from pyproj import Proj
from matplotlib import pyplot as plt
from scipy import spatial 

from BarHandler import BarHandler
from RasterHandler import RasterHandler
from RiverHandler import RiverHandler
from PointPicker import WidthPicker

STEP = None
MIN_RSQUARE = 0.05
random.seed(60)


def get_direction(coordinates, idx):
    """
    Takes in the full bar coordinates as well as the index for the iteration
    """
    tree = spatial.KDTree(coordinates[['lon', 'lat']])
    distance, neighbors = tree.query(
        [coordinates.iloc[idx]['lon'], coordinates.iloc[idx]['lat']],
        2
    )
    max_distance = np.argmax(distance)
    max_neighbor = neighbors[max_distance]
    min_distance = np.argmin(distance)
    min_neighbor = neighbors[min_distance]

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

    return distance[0] / norm, distance[1] / norm


def angle_between(v1, v2):
    """ 
    Returns the angle in radians between vectors 'v1' and 'v2'
    """
#    v1_u = unit_vector(v1)
#    v2_u = unit_vector(v2)

    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

param = {
    'DEMpath': '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/barCut/trinityDEM_clip_26915.tif',
    'CenterlinePath': '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/barCut/trinityCenterline_4326.csv',
    'esaPath': '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/barCut/trinityOccurence_clip_26915.tif',
    'barPath': '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/barCut/trinity_bar_coords.csv',
    'CenterlineSmoothing': 10,
    'SectionLength': 350,
    'SectionSmoothing': 15,
    'WidthSens': 32,
    'mannual': True,
    'step': 1,
    'OutputRoot': '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/barCut/Output',
    'depth': 6,
    'interpolate': True
}

# Initialize classes, objects, get ProjStr
riv = RiverHandler()
rh = RasterHandler()
ds = gdal.Open(param['DEMpath'], 0)
water_ds = gdal.Open(param['esaPath'], 0)
ProjStr = "epsg:{0}".format(
    osr.SpatialReference(
        wkt=ds.GetProjection()
    ).GetAttrValue('AUTHORITY', 1)
)

# Load the Centerline Coordinates File
print('Loading the Centerline File')
coordinates = pandas.read_csv(
    param['CenterlinePath'],
    names=['Longitude', 'Latitude'],
    header=1,
)

# Smooth the river centerline
print('Smoothing the river centerline')
coordinates['Longitude'], coordinates['Latitude'] = riv.knn_smoothing(
    coordinates,
    n=param['CenterlineSmoothing']
)

# Convert centerline in Lat-Lon to UTM
print('Converting coordinates to UTM')
myProj = Proj(ProjStr)
coord_transform = pandas.DataFrame(
    columns=['lon', 'lat', 'easting', 'northing']
)
for idx, row in coordinates.iterrows():
    # lat, lon -> utm
    east, north = myProj(row['Latitude'], row['Longitude'])
    df = pandas.DataFrame(
        data=[[row['Longitude'], row['Latitude'], east, north]],
        columns=['lat', 'lon', 'easting', 'northing']
    )
    coord_transform = coord_transform.append(df)

coordinates = coord_transform.reset_index(drop=True)

# Loading DEM data and metadata
print('Loading DEM Data and MetaData')
dem = ds.ReadAsArray()
transform = ds.GetGeoTransform()
dem_transform = {
    'xOrigin': transform[0],
    'yOrigin': transform[3],
    'pixelWidth': transform[1],
    'pixelHeight': -transform[5]
}
dem_transform['xstep'], dem_transform['ystep'] = rh.get_pixel_size(
    param['DEMpath']
)

# Loading Water Surface data and metadata
water = water_ds.ReadAsArray()
transform = water_ds.GetGeoTransform()
water_transform = {
    'xOrigin': transform[0],
    'yOrigin': transform[3],
    'pixelWidth': transform[1],
    'pixelHeight': -transform[5]
}
water_transform['xstep'], water_transform['ystep'] = rh.get_pixel_size(
    param['DEMpath']
)

# Downsample if you want
if STEP:
    coordinates = coordinates.iloc[::STEP, :]

if len(coordinates) == 0:
    sys.exit("No coordinates")

######################################################
########   This is the primary sampling
######################################################
bar_data_df = pandas.DataFrame()
finn = 25
n = 10
# 1 - 1:10
start = 73
stop = 91
for theta in range(start, stop):
    scoordinates = pandas.DataFrame(columns=[
        'lon', 
        'lat', 
        'easting', 
        'northing', 
        'dlon', 
        'dlat', 
        'dlon_inv',
        'dlat_inv',
        'dlon_cut',
        'dlat_cut',
    ])
    for idx in coordinates.sample(n, replace=False).index:
        # Find the centerline direction at that bar point
        dlon, dlat = get_direction(coordinates, idx)

        # Gets the inverse direction
        dlon_inv = dlat
        dlat_inv = -dlon

        # Find vector at angle
        phi = math.degrees(math.atan(dlat_inv/dlon_inv))
        dlon_cut = math.cos(math.radians(phi - theta))
        dlat_cut = math.sin(math.radians(phi - theta))
#        dlon_cut = dlon_inv    # If I'm just down the normal cuts
#        dlat_cut = dlat_inv

#       # Plot the angle
#        origin = [0, 0], [0, 0] # origin point
#        vinv = [dlon_inv, dlat_inv]
#        vcut = [dlon_cut, dlat_cut]
#        V = np.array([vinv, vcut])
#        plt.quiver(*origin, V[:,0], V[:,1], color=['r','b'], scale=5)
#        plt.title(theta)
#        plt.show()

        # Put it all together
        sample = coordinates.iloc[idx]
        sample['dlon'] = dlon
        sample['dlat'] = dlat
        sample['dlon_inv'] = dlon_inv
        sample['dlat_inv'] = dlat_inv
        sample['dlon_cut'] = dlon_cut
        sample['dlat_cut'] = dlat_cut
        sample['theta'] = theta
#        sample['angle'] = 0    # If I'm just doing the normal cuts

        scoordinates = scoordinates.append(sample)

    # Split into two dataframes. 
    # This makes creating the cross sections easier
    # I want to track botht the true width of the river 
    # as well as the apparent bar width
    true_coords = scoordinates.drop(columns=['dlon_cut', 'dlat_cut'])
    scoordinates = scoordinates.drop(columns=['dlon_inv', 'dlat_inv'])
    scoordinates = scoordinates.rename(columns={
        'dlon_cut': 'dlon_inv', 
        'dlat_cut': 'dlat_inv'
    })
    print(scoordinates)

    # Build the cross-section structure
    print('Building channel cross sections')
    types = [
        ('coords', 'object'),
        ('dem_width', 'f8'),
        ('water_width', 'f8'),
        ('bank', 'object'),
        ('elev_section', 'object'),
        ('water_section', 'object'),
    ]
    xsections = np.array([], dtype=types)
    # Iterate through each coordinate of the channel centerline
    for idx, row in scoordinates.iterrows():
        # Get the cross-section and set-up numpy stucture
        section = np.array(
            tuple(
                [
                    (row['easting'], row['northing']),
                    None,
                    None,
                    None,
                    # Perpendicular Cross-Section
                    rh.get_xsection(
                        row,
                        dem,
                        dem_transform['xOrigin'],
                        dem_transform['yOrigin'],
                        dem_transform['pixelWidth'],
                        dem_transform['pixelHeight'],
                        int(
                            param['SectionLength'] 
                            * 2 * (1 + abs(row['theta']/180))
                        ),
                        dem_transform['xstep'],
                        dem_transform['ystep']
                    ),
                    # Cross section of ESA water occurence
                    rh.get_xsection(
                        row,
                        water,
                        water_transform['xOrigin'],
                        water_transform['yOrigin'],
                        water_transform['pixelWidth'],
                        water_transform['pixelHeight'],
                        param['SectionLength'],
                        water_transform['xstep'],
                        water_transform['ystep']
                    )
                ]
            ),
            dtype=xsections.dtype
        )
        xsections = np.append(xsections, section)

    # Smooth Cross Sections
    print('Smoothing Cross-Sections')
    for idx, section in np.ndenumerate(xsections):
        print(idx)
        b = riv.xsection_smoothing(
            idx,
            section['elev_section'],
            param['SectionSmoothing'],
        )
        xsections[idx[0]]['elev_section'] = b

    print('Finding Channel Widths')
    # Set up channel banks dataframe
    bank_df = pandas.DataFrame(
        columns=[
            'dem_easting',
            'dem_northing',
            'water_easting',
            'water_northing'
        ]
    )

    # Iterate through exsections to find widths
    for idx in range(0, len(xsections), param['step']):
        # Finds the channel width and associated points
        if param['mannual']:
            banks, dem_width, dem_points = riv.mannual_find_channel_width(
                idx,
                xsections[idx]['elev_section']
            )
            plt.close('all')
        else:
            banks, dem_width, dem_points = riv.find_channel_width(
                xsections[idx]['elev_section'],
                xsections[idx]['elev_section'],
                order=param['WidthSens']
            )
        if len(
            xsections[idx]['water_section'][
                xsections[idx]['water_section']['value'] > 0
            ]
        ) > 0:
            water_width, water_points = riv.find_channel_width_surface_water(
                xsections[idx]
            )
        else:
            water_width = None
            water_points = None

        # If the program found channel banks will construct banks dataframe
        if banks:
            bank_df = bank_df.append(riv.get_bank_positions(
                xsections[idx]['elev_section'],
                dem_points,
                water_points
            ))

        # Save width values to the major cross-section structure
        xsections[idx]['bank'] = banks
        xsections[idx]['dem_width'] = dem_width
        xsections[idx]['water_width'] = water_width

        # If there is a step, fill the rest of the values
        if param['step'] > 1:
            if idx != len(xsections) - 1:
                for j in range(param['step'] -1):
                    if idx + j > len(xsections) - 1:
                        break
                    xsections[idx + j]['bank'] = banks
                    xsections[idx + j]['dem_width'] = dem_width
                    xsections[idx + j]['water_width'] = water_width

    # Get that bar data
    # Get Proj String
    ds = gdal.Open(param['DEMpath'], 0)
    ProjStr = "epsg:{0}".format(
        osr.SpatialReference(
            wkt=ds.GetProjection()
        ).GetAttrValue('AUTHORITY', 1)
    )
    myProj = Proj(ProjStr)

    # Initialize BarHandler
    rh = RasterHandler()
    bh = BarHandler(
        xsections[0]['coords'][0],
        xsections[0]['coords'][1]
    )

    # Read in the bar file to find the channel bars
    print('Loading Bar .csv file')
    bar_df = pandas.read_csv(
        param['barPath'],
        names=['Latitude_us', 'Longitude_us', 'Latitude_ds', 'Longitude_ds'],
        header=0
    )
    # Convert the Bar Lat Long to UTM Easting Northing
    print('Converting Bar Coordinates to Easting Northing')
    bar_df = bh.convert_bar_to_utm(myProj, bar_df)
    ds = None

    # Save the parameters
    parameters = {
        'idx': [],
        'L': [],
        'X0': [],
        'k': [],
    }

    print('Fitting sigmoid to channel bars')
    types = [
        ('location', 'object'),
        ('dem_width', 'f8'),
        ('water_width', 'f8'),
        ('sigmoid', 'object'),
        ('easting', 'object'),
        ('northing', 'object'),
        ('distance', 'object'),
        ('elevation', 'object'),
    ]

    # Run through all of the bars and sections
    widths = np.array([], dtype=types)
    for idx, section in np.ndenumerate(xsections):
        if (
            section['dem_width'] == 'nan'
        ) or (
            not section['bank']
        ) or (
            section['water_width'] == 'nan'
        ):
            width = np.array(
                tuple(
                    [
                        section[0],
                        section['dem_width'],
                        section['water_width'],
                        None,
                        section['elev_section']['easting'],
                        section['elev_section']['northing'],
                        section['elev_section']['distance'],
                        section['elev_section']['value_smooth']
                    ]
                ),
                dtype=widths.dtype
            )
        else:
            # interpolate profile down
            if param['interpolate'] == True:
                section = bh.interpolate_down(
                    param.get('depth'),
                    section
                )

            # Find the minimum and shift the cross-section
            section = bh.shift_cross_section_down(
                section, 
            )

            popt, rsquared = bh.mannual_fit_bar(section)

            # Keep track of the parameters
            parameters['idx'].append(idx[0])
            parameters['L'].append(popt[0])
            parameters['X0'].append(popt[1])
            parameters['k'].append(popt[2])

            # store the sigmoid parameters and the cross section
            width = np.array(
                tuple(
                    [
                        section[0],
                        section['dem_width'],
                        section['water_width'],
                        popt,
                        section['elev_section']['easting'],
                        section['elev_section']['northing'],
                        section['elev_section']['distance'],
                        section['elev_section']['value_smooth']
                    ]
                ),
                dtype=widths.dtype
            )

        widths = np.append(widths, width)

    # Find the bar Geometry
    # Find the width and height of the channel bars
    print('Finding clinoform width and height')
    bar_data = {
        'theta': [],
        'easting': [],
        'northing': [],
        'channel_width_dem': [],
        'channel_width_water': [],
        'channel_width_mean': [],
        'bar_width': [],
        'bar_height': []
    }

    for idx, section in np.ndenumerate(widths):
        # Don't track if there is no channel width
        if str(section['dem_width']) == 'nan':
            continue

        # Filter out the ill-fit sigmoid parameters
        elif not section['sigmoid']:
            continue

        else:
            # Get the bar width from the sigmoid
            bar_width, bar_height = bh.get_bar_geometry(
                section['distance'],
                section['sigmoid']
            )

            try:
               water_width = int(section['water_width'])
            except:
                water_width = int(section['dem_width'])

            # Store data
            bar_data['theta'].append(theta)
            bar_data['easting'].append(section['location'][0])
            bar_data['northing'].append(section['location'][1])
            bar_data['channel_width_dem'].append(int(section['dem_width']))
            bar_data['channel_width_water'].append(water_width)
            bar_data['channel_width_mean'].append(
                (int(section['dem_width']) + water_width) / 2
            )
            bar_data['bar_width'].append(bar_width)
            bar_data['bar_height'].append(bar_height)

    # Create dataframes from data dicts
    parameters_df = pandas.DataFrame(parameters)
    data = pandas.DataFrame(bar_data)
    bar_data_df = bar_data_df.append(data)
    bar_data_df['sample'] = str(n)
    bar_data_df['ratio'] = bar_data_df['channel_width_mean'] / bar_data_df['bar_width']

bar_data_df.to_csv('barCuts/angle_data/angle_{}_{}.csv'.format(start, stop-1))
