import argparse
import sys
import json

import gdal
import osr
import numpy as np
import pandas
from pyproj import Proj

from BarHandler import BarHandler
from RasterHandler import RasterHandler
from RiverHandler import RiverHandler
from TestHandler import TestHandler
from Visualizer import Visualizer


def main(DEMpath, CenterlinePath, BarPath, CenterlineSmoothing,
         SectionLength, SectionSmoothing, WidthSens, OutputRoot):

    # Initialize classes, objects, get ProjStr
    riv = RiverHandler()
    rh = RasterHandler()
    test = TestHandler()
    ds = gdal.Open(DEMpath, 0)
    ProjStr = "epsg:{0}".format(
        osr.SpatialReference(
            wkt=ds.GetProjection()
        ).GetAttrValue('AUTHORITY', 1)
    )

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

    # Find the channel direction and inverse channel direction
    print('Finding channel and cross-section directions')
    coordinates = riv.get_direction(coordinates)
    coordinates = riv.get_inverse_direction(coordinates)
    coordinates = coordinates.dropna(axis=0, how='any')

    # Save the Coordinates file
    coordinates.to_csv(OutputRoot + 'coordinates.csv')

    # Build the cross-section structure
    print('Building channel cross sections')
    types = [
        ('coords', 'object'),
        ('width', 'f8'),
        ('bank', 'object'),
        ('xsection', 'object'),
    ]
    xsections = np.array([], dtype=types)
    # Iterate through each coordinate of the channel centerline
    for idx, row in coordinates.iterrows():
        # Get the cross-section and set-up numpy stucture
        section = np.array(
            tuple(
                [
                    (row['easting'], row['northing']),
                    None,
                    None,
                    rh.get_xsection(
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
                ]
            ),
            dtype=xsections.dtype
        )
        xsections = np.append(xsections, section)

    # Load in the Bar coordinate data
    bh = BarHandler(
        xsections[0]['coords'][0],
        xsections[0]['coords'][1]
    )

    # Smooth Cross Sections
    print('Smoothing Cross-Sections')
    for idx, section in np.ndenumerate(xsections):
        print(idx)
        b = riv.xsection_smoothing(
            idx,
            section['xsection'],
            SectionSmoothing,
        )
        xsections[idx[0]]['xsection'] = b

    # Make some test cross-section
    print('Generating Test Xsections')
    test.save_example_sections(
        xsections,
        20,
        OutputRoot + 'Test/'
    )

    print('Finding Channel Widths')
    # Set up channel banks dataframe
    bank_df = pandas.DataFrame(columns=['easting', 'northing'])

    # Iterate through exsections to find widths
    for idx, section in np.ndenumerate(xsections):
        # Finds the channel width and associated points
        banks, width, points = riv.find_channel_width(
            xsections[idx[0]]['xsection'], 
            order=WidthSens
        )

        # If the program found channel banks will construct banks dataframe
        if banks:
            bank_df = bank_df.append(riv.get_bank_positions(
                xsections[idx[0]]['xsection'], 
                points
            ))

        # Save width values to the major cross-section structure
        xsections[idx[0]]['bank'] = banks
        xsections[idx[0]]['width'] = width

    # Save the Channel Cross Sections Structure
    print('Saving Cross-Section Structure')
    np.save(OutputRoot + 'xsections.npy', xsections)

    if len(bank_df) > 0:
        print('Saving Channel Banks')
        bank_df.to_csv(OutputRoot + 'channel_banks.csv')

    # Save the width dataframe
    print('Saving Width DataFrame')
    riv.save_channel_widths(xsections).to_csv(OutputRoot + 'width_dataframe')

    # Read in the bar file to find the channel bars
    print('Loading Bar .csv file')
    bar_df = pandas.read_csv(
        BarPath,
        names=['Latitude_us', 'Longitude_us', 'Latitude_ds', 'Longitude_ds'],
        header=1
    )

    # Convert the Bar Lat Long to UTM Easting Northing
    print('Converting Bar Coordinates to Easting Northing')
    bar_df = bh.convert_bar_to_utm(myProj, bar_df)

    # Find the bar coords within the DEM
    print('Find Bars within the DEM')
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

    # Generate test cross sections
    print('Generating Test Bar Xsections')
    test.save_example_bar_sections(
        coordinates,
        xsections,
        bar_df,
        OutputRoot + 'Test/'
    )

    # Make structure that contains the sections for each bar
    print('Making Bar section structure')
    bar_sections = {} 
    for idx, bar in bar_df.iterrows():
        sections = bh.get_bar_xsections(
            coordinates,
            xsections,
            bar_df.iloc[idx]
        )
        bar_sections[str(idx)] = sections

    # Make structure that contains the sigmoids and only the bar side section
    print('Fitting sigmoid to channel bars')
    bar_widths = {}
    types = [
        ('location', 'object'),
        ('width', 'f8'),
        ('sigmoid', 'object'),
        ('easting', 'object'),
        ('northing', 'object'),
        ('distance', 'object'),
        ('elevation', 'object'),
    ]
    for bar, sections in bar_sections.items():
        widths = np.array([], dtype=types)
        for idx, section in np.ndenumerate(sections):
            if section['width'] == 'nan' or not section['bank']:
                width = np.array(
                    tuple(
                        [
                            section[0],
                            section['width'],
                            None,
                            section['xsection']['easting'],
                            section['xsection']['northing'],
                            section['xsection']['distance'],
                            section['xsection']['demvalue_sm']
                        ]
                    ),
                    dtype=widths.dtype
                )
            else:
                # Find the side of the channel with the bar
                banks = bh.find_bar_side(section['bank'])

                # Flip the cross-sections so that they are all facing the same way
                section, banks = bh.flip_bars(section, banks)

                # Find the distance for maximum slope and the maximum slope
                x0 , dydx = bh.find_maximum_slope(
                    section['xsection'], 
                    banks
                )

                # Find the minimum and shift the cross-section
                section = bh.shift_cross_section_down(section, banks)

                # Fit sigmoid parameters
                popt = bh.fit_sigmoid_parameters(section, banks, x0, dydx)

                # store the sigmoid parameters and the cross section
                width = np.array(
                    tuple(
                        [
                            section[0],
                            section['width'],
                            popt,
                            section['xsection']['easting'],
                            section['xsection']['northing'],
                            section['xsection']['distance'],
                            section['xsection']['demvalue_sm']
                        ]
                    ),
                    dtype=widths.dtype
                )

            widths = np.append(widths, width)

        bar_widths[bar] = widths

    # Find the width and height of the channel bars
    print('Finding clinoform width and height')
    columns = [
        'bar', 
        'idx', 
        'easting', 
        'northing', 
        'channel_width', 
        'bar_width',
        'bar_height'
    ]
    bar_data_df = pandas.DataFrame(columns=columns)
    n = 0
    for bar, sections in bar_widths.items():
        print(bar)
        L_mean = np.median(
            [i['sigmoid'][0] for i in sections if i['sigmoid']]
        )
        for idx, section in np.ndenumerate(sections):
            # Don't track if there is no channel width
            if str(section['width']) == 'nan':
                continue

            # Filter out the ill-fit sigmoid parameters
            elif (section['sigmoid'][0] / L_mean) < 0.01:
                continue

            else:
                # Get the bar width from the sigmoid
                bar_width, bar_height = bh.get_bar_geometry(
                    section['distance'],
                    section['sigmoid']
                )

                # Store data
                data = {
                    'bar': bar,
                    'idx': '{0}_{1}'.format(bar, idx[0]),
                    'easting': section['location'][0],
                    'northing': section['location'][1],
                    'channel_width': int(section['width']),
                    'bar_width': bar_width,
                    'bar_height': bar_height
                }

                # Append to dataframe
                bar_data_df = bar_data_df.append(
                    pandas.DataFrame(data=data, index=[n])
                )
                n += 1

    # Save the bar data
    print('Saving Bar Data')
    bar_data_df.to_csv(OutputRoot + 'bar_data.csv')
                

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
         args.CenterlineSmoothing, args.SectionLength,
         args.SectionSmoothing, args.WidthSens, args.OutputRoot)
