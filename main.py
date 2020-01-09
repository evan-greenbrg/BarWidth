import argparse
import sys
import json

import gdal
import numpy as np
import pandas
from pyproj import Proj

from BarHandler import BarHandler
from RasterHandler import RasterHandler
from RiverHandler import RiverHandler
from TestHandler import TestHandler
from Visualizer import Visualizer


# Next Two Additions
# DONE # 1. Only use part of centerline dataframe that overlaps with DEM
# DONE # 2. Produce csv with coordinates of each river bank (where there is width)
# DONE # 3. Produce csv with coordinates of bar width point

def main(DEMpath, CenterlinePath, BarPath, ProjStr, CenterlineSmoothing,
         SectionLength, SectionSmoothing, WidthSens, OutputRoot):

    # Initialize classes, objects
    riv = RiverHandler()
    rh = RasterHandler()
    test = TestHandler()
    ds = gdal.Open(DEMpath, 0)

    # Load the Centerline Coordinates File
    print('Loading the Centerline File')
    coordinates = pandas.read_csv(
        CenterlinePath,
        names=['Longitude', 'Latitude'],
        header=1,
        index_col=[0]
    )

    # Smooth the river centerline
    print('Smoothing the river centerline')
    coordinates['Longitude'], coordinates['Latitude'] = riv.knn_smoothing(
        coordinates, n=CenterlineSmoothing
    )

    # MATCH PROJECTIONS
    # Convert between landsat and dem projections and lat,WGS_1984 lon to utm
    print('Converting coordinates to projection')
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

    print('Loading DEM Data and MetaData')
    # Loading DEM data and metadata
    dem = ds.ReadAsArray()

    transform = ds.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = -transform[5]
    xstep, ystep = rh.get_pixel_size(DEMpath)

    # Find what portion of centerline is within the DEM
    coordinates = rh.coordinates_in_dem(
        coordinates, 
        ds,
        ('easting', 'northing')
    )

    if len(coordinates) == 0:
        sys.exit("No coordinates")
        

    # Get values at each coordinate location
    values = rh.values_from_coordinates(ds, dem, coordinates)
    coordinates['elev_0'] = values

    print('Finding channel and cross-section directions')
    # Find the channel direction and inverse channel direction
    coordinates = riv.get_direction(coordinates)
    coordinates = riv.get_inverse_direction(coordinates)
    coordinates = coordinates.dropna(axis=0, how='any')

    # Save the Coordinates file
    coordinates.to_csv(OutputRoot + 'coordinates.csv')

    print('Building channel cross sections')
    # BUILD CROSS-SECTION STRUCTURE
    types = [
        ('coords', 'object'),
        ('width', 'f8'),
        ('bank', 'object'),
        ('xsection', 'object'),
    ]
    xsections = np.array([], dtype=types)
    # Iterate through each coordinate of the channel centerline
    for idx, row in coordinates.iterrows():
        xsection = rh.get_xsection(
            row,
            dem,
            xOrigin,
            yOrigin,
            pixelWidth,
            pixelHeight,
            SectionLength,
            xstep,
            ystep
        )
        section = np.array(
            tuple(
                [
                    (row['easting'], row['northing']),
                    None,
                    None,
                    xsection
                ]
            ),
            dtype=xsections.dtype
        )
        xsections = np.append(xsections, section)

    print('Smoothing Cross-Sections')
    # Smooth Cross Sections
    for idx, section in np.ndenumerate(xsections):
        print(idx)
        b = riv.xsection_smoothing(
            idx,
            section['xsection'],
            SectionSmoothing,
        )
        xsections[idx[0]]['xsection'] = b

    print('Generating Test Xsections')
    test.save_example_sections(
        xsections,
        20,
        OutputRoot + 'Test/'
    )

    print('Finding Channel Widths')
    # Set up channel bank dataframe
    bank_df = pandas.DataFrame(columns=['easting', 'northing'])

    # Iterate through exsections to find widths
    for idx, section in np.ndenumerate(xsections):
        xsection = xsections[idx[0]]['xsection']
        p = xsection['distance']
        t = xsection['demvalue_sm']
        banks, width, points = riv.find_channel_width(p, t, order=WidthSens)
        # banks will be used to find the bar
        # width will be used for the channel width
        # points will be used for the banks output product

        if banks:
            bank0 = xsection[xsection['distance'] == points[0]]
            data0 = np.append(bank0['easting'], bank0['northing'])

            bank1 = xsection[xsection['distance'] == points[1]]
            data1 = np.append(bank1['easting'], bank1['northing'])

            data = np.vstack((data0, data1)).astype('float')
            df = pandas.DataFrame(data, columns=['easting', 'northing'])
            bank_df = bank_df.append(df)

        xsections[idx[0]]['bank'] = banks
        xsections[idx[0]]['width'] = width

    print('Saving Cross-Section Structure')
    # Save the Channel Cross Sections Structure
    np.save(OutputRoot + 'xsections.npy', xsections)

    if len(bank_df) > 0:
        print('Saving Channel Banks')
        bank_df.to_csv(OutputRoot + 'channel_banks.csv')

    print('Finding Channel Bar Widths')
    # Find the channel bar widths
    widths = []
    eastings = []
    northings = []
    # Create a width DF
    for section in xsections:
        eastings.append(section[0][0])
        northings.append(section[0][1])
        widths.append(section[1])
    data = {'easting': eastings, 'northing': northings, 'width': widths}
    width_df = pandas.DataFrame(
        data=data,
        columns=['easting', 'northing', 'width']
    )
    width_df = width_df.reset_index()

    print('Saving Width DataFrame')
    # Save the width dataframe
    width_df.to_csv(OutputRoot + 'width_dataframe')

    # Load in the Bar coordinate data
    bh = BarHandler(
        xsections[0]['coords'][0],
        xsections[0]['coords'][1]
    )
    bar_df = pandas.read_csv(
        BarPath,
        names=['Latitude_us', 'Longitude_us', 'Latitude_ds', 'Longitude_ds'],
        header=1
    )

    # Convert the Bar Lat Long to UTM Easting Northing
    myProj = Proj(ProjStr)
    coord_transform = pandas.DataFrame(
        columns=[
            'upstream_lat',
            'upstream_lon',
            'upstream_easting',
            'upstream_northing',
            'downstream_lat',
            'downstream_lon',
            'downstream_easting',
            'downstream_northing'
        ]
    )
    print('Converting Bar Coordinates to Easting Northing')
    # Find the channel bar widths
    for idx, row in bar_df.iterrows():
        # lat, lon -> utm
        us_east, us_north = myProj(
            row['Longitude_us'],
            row['Latitude_us']
        )
        ds_east, ds_north = myProj(
            row['Longitude_ds'],
            row['Latitude_ds']
        )
        df = pandas.DataFrame(
            data=[[
                row['Latitude_us'],
                row['Longitude_us'],
                us_east,
                us_north,
                row['Latitude_ds'],
                row['Longitude_ds'],
                ds_east,
                ds_north
            ]],
            columns=[
                'upstream_lat',
                'upstream_lon',
                'upstream_easting',
                'upstream_northing',
                'downstream_lat',
                'downstream_lon',
                'downstream_easting',
                'downstream_northing'
            ]
        )
        coord_transform = coord_transform.append(df)

    bar_df = coord_transform.reset_index(drop=True)

    # Find the bar coords within the DEM
    bar_df = rh.coordinates_in_dem(
        bar_df, 
        ds, 
        ('upstream_easting', 'upstream_northing')
    )
    bar_df = rh.coordinates_in_dem(
        bar_df, 
        ds, 
        ('downstream_easting', 'downstream_northing')
    )

    # NEED TO FIX ERROR WHEN IT HITS THIS STEP
    print('Generating Test Bar Xsections')
    test.save_example_bar_sections(
        coordinates,
        xsections,
        bar_df,
        OutputRoot + 'Test/'
    )

    print('Generating Bar-Channel Width Data Structure')
    # Create Dict with all of the bar and channel widths and ratio
    n = 1
    bars_ = {}
    # Initialize bar_points df
    bar_points_df = pandas.DataFrame(columns=['easting', 'northing', 'bar'])
    for idx, bar in bar_df.iterrows():
        i = 0 # Index within a bar
        ratio = [] # Ratio of channel width and bar width
        idxs = [] # Indexes list for each bar
        name = 'bar_{n}'.format(n=n) # Name of the bar 
        widths = [] # Width of the channel
        bar_widths = [] # Width of the bar
        coords0 = [] # Coords pairs 
        coords1 = [] # All coord bars in the bar
        # Get the portions of xsections that are the bars
        bar_sections = bh.get_bar_xsections(
            coordinates,
            xsections,
            bar_df.iloc[idx]
        )

        # For each section in bars, find the width
        for section in bar_sections:
            if section['width'] == 'nan':
                widths.append(False)
            else:
                widths.append(section['width'])
                coords0.append(section[0])
                bar_width, bar_points = bh.find_bar_width(section['bank'])
                bar_widths.append(bar_width)

                if bar_points:
                    xsection = section['xsection']
                    bank0 = xsection[xsection['distance'] == bar_points[0]]
                    data0 = np.append(
                        [bank0['easting'], bank0['northing']], 
                        idx
                    )

                    bank1 = xsection[xsection['distance'] == bar_points[1]]
                    data1 = np.append(
                        [bank1['easting'], bank1['northing']], 
                        idx
                    )

                    data = np.vstack((data0, data1)).astype('float')
                    print(data)
                    df = pandas.DataFrame(
                        data, 
                        columns=['easting', 'northing', 'bar']
                    )
                    bar_points_df = bar_points_df.append(df)

        if len(bar_points_df) > 0:
            print('Saving Channel Bars')
            bar_points_df.to_csv(OutputRoot + 'channel_bars.csv')

        for idx, width in enumerate(widths):
            i += 1
            if width and bar_widths[idx]:
                ratio.append(width/bar_widths[idx])
                idxs.append(i)
                coords1.append(coords0[idx])
        bars_[name] = {
            'idx': idxs,
            'coords': coords1,
            'channel_width': widths,
            'bar_width': bar_widths,
            'ratio': ratio
        }
        n += 1

    bars_ = bh.get_downstream_distance(bars_)
    print('Saving Channel Bar - Width JSON')
    # Turn Bars dictionary into a json and save it
    with open((OutputRoot + 'bars.json'), 'w') as f:
        json.dump(bars_, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Return Channel Width and Bar Width Measurements'
    )
    parser.add_argument('DEMpath', metavar='dem', type=str,
                        help='Path to the DEM file')
    parser.add_argument('CenterlinePath', metavar='c', type=str,
                        help='Path to the centerline coordinates file')
    parser.add_argument('BarPath', metavar='b', type=str,
                        help='Path to the bar coordinates file')
    parser.add_argument('ProjStr', metavar='p', type=str,
                        help='Projection String')
    parser.add_argument('CenterlineSmoothing', metavar='cs', type=int,
                        help='Smoothing factor for the channel coordinates')
    parser.add_argument('SectionLength', metavar='sl', type=int,
                        help='Length of the cross section to take')
    parser.add_argument('SectionSmoothing', metavar='ss', type=int,
                        help='Smoothing factor for the cross-sections')
    parser.add_argument('WidthSens', metavar='ws', type=int,
                        help='Sensitivity of the channel width measurement')
    parser.add_argument('OutputRoot', metavar='out', type=str,
                        help='Root for the file outputs')

    args = parser.parse_args()

    main(args.DEMpath, args.CenterlinePath, args.BarPath,
         args.ProjStr, args.CenterlineSmoothing, args.SectionLength,
         args.SectionSmoothing, args.WidthSens, args.OutputRoot)
