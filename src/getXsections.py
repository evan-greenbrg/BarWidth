import argparse
import errno
import json
import os
import sys

import gdal
import osr
import numpy as np
import pandas
from pyproj import Proj

from BarHandler import BarHandler
from RasterHandler import RasterHandler
from RiverHandler import RiverHandler


def main(DEMpath, CenterlinePath, BarPath, esaPath, 
         CenterlineSmoothing, SectionLength, SectionSmoothing, 
         WidthSens, OutputRoot):

    # Raise Errors if files don't exists
    if not os.path.exists(DEMpath):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), DEMpath)
    if not os.path.exists(CenterlinePath):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), CenterlinePath)
    if not os.path.exists(BarPath):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), BarPath)
    if not os.path.exists(esaPath):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), esaPath)

    # Initialize classes, objects, get ProjStr
    riv = RiverHandler()
    rh = RasterHandler()
    ds = gdal.Open(DEMpath, 0)
    water_ds = gdal.Open(esaPath, 0)
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
#        index_col=[0]
    )

    # Smooth the river centerline
    print('Smoothing the river centerline')
    coordinates['Longitude'], coordinates['Latitude'] = riv.knn_smoothing(
        coordinates, n=CenterlineSmoothing
    )
#    coordinates['Latitude'], coordinates['Longitude'] = riv.knn_smoothing(
#        coordinates, n=CenterlineSmoothing
#    )

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
        DEMpath
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
        DEMpath
    )

    # Find what portion of centerline is within the DEM
    coordinates = rh.coordinates_in_dem(
        coordinates, 
        ds,
        ('easting', 'northing')
    )

    # Downsample if you want
    step = 3
    coordinates = coordinates.iloc[::step, :]

    if len(coordinates) == 0:
        sys.exit("No coordinates")
        
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
        ('dem_width', 'f8'),
        ('water_width', 'f8'),
        ('bank', 'object'),
        ('elev_section', 'object'),
        ('water_section', 'object'),
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
                    None,
                    rh.get_xsection(
                        row,
                        dem,
                        dem_transform['xOrigin'],
                        dem_transform['yOrigin'],
                        dem_transform['pixelWidth'],
                        dem_transform['pixelHeight'],
                        SectionLength,
                        dem_transform['xstep'],
                        dem_transform['ystep']
                    ),
                    rh.get_xsection(
                        row,
                        water,
                        water_transform['xOrigin'],
                        water_transform['yOrigin'],
                        water_transform['pixelWidth'],
                        water_transform['pixelHeight'],
                        SectionLength,
                        water_transform['xstep'],
                        water_transform['ystep']
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
            section['elev_section'],
            SectionSmoothing,
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
    for idx, section in np.ndenumerate(xsections):
        # Finds the channel width and associated points
        banks, dem_width, dem_points = riv.find_channel_width(
            xsections[idx[0]]['elev_section'], 
            order=WidthSens
        )
        if len(
            xsections[idx[0]]['water_section'][
                xsections[idx[0]]['water_section']['value'] > 0
            ]
        ) > 0:
            water_width, water_points = riv.find_channel_width_surface_water(
                xsections[idx[0]]
            )
        else:
            water_width = None
            water_points = None

        # If the program found channel banks will construct banks dataframe
        if banks:
            bank_df = bank_df.append(riv.get_bank_positions(
                xsections[idx[0]]['elev_section'], 
                dem_points,
		water_points
            ))

        # Save width values to the major cross-section structure
        xsections[idx[0]]['bank'] = banks
        xsections[idx[0]]['dem_width'] = dem_width
        xsections[idx[0]]['water_width'] = water_width

    # Save the Channel Cross Sections Structure
    print('Saving Cross-Section Structure')
    np.save(OutputRoot + 'xsections.npy', xsections)

    if len(bank_df) > 0:
        print('Saving Channel Banks')
        bank_df.to_csv(OutputRoot + 'channel_banks.csv')

    # Save the width dataframe
    print('Saving Width DataFrame')
    riv.save_channel_widths(xsections).to_csv(OutputRoot + 'width_dataframe.csv')

    return True


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
    parser.add_argument('esaPath', metavar='esa', type=str,
                        help='Path to the esa file')
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

    main(args.DEMpath, args.CenterlinePath, args.BarPath, args.esaPath,
         args.CenterlineSmoothing, args.SectionLength,
         args.SectionSmoothing, args.WidthSens, args.OutputRoot)
