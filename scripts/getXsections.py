import argparse
import errno
import os
import sys
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

from BarWidth import RasterHandler
from BarWidth import RiverHandler


# If you don't want to sample all of the centerline points
STEP = 5


def main():

    parser = argparse.ArgumentParser(description='Input Prams for Xsection')
    parser.add_argument('param', type=argparse.FileType('r'))
    args = parser.parse_args()
    param = load(args.param, Loader=Loader)

    # Raise Errors if files don't exists
    if not os.path.exists(param['DEMpath']):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT),
            param['DEMpath']
        )
    if not os.path.exists(param['CenterlinePath']):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT),
            param['CenterlinePath']
        )
    if not os.path.exists(param['esaPath']):
        raise FileNotFoundError(
            errno.ENOENT,
            os.strerror(errno.ENOENT),
            param['esaPath']
        )
    if not param.get('CenterlineSmoothing'):
        raise NameError('No Centerline Smoothing Parameter given')
    if not param.get('OutputRoot'):
        raise NameError('No Output root given')
    if not param.get('SectionLength'):
        raise NameError('No section length given')
    if not param.get('manual'):
        param['manual'] = True
#        raise NameError('No given manual or automatic direction')

    # Initialize classes, objects, get ProjStr
    riv = RiverHandler.RiverHandler()
    rh = RasterHandler.RasterHandler()
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

    # Save the full coordinate dataframe
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

    # Find the channel direction and inverse channel direction
    print('Finding channel and cross-section directions')
    coordinates = riv.get_direction(coordinates)
    coordinates = riv.get_inverse_direction(coordinates)
    coordinates = coordinates.dropna(axis=0, how='any')

    # Save the Coordinates file
    coordinates.to_csv(param['OutputRoot'] + 'coordinates.csv')

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
                        param['SectionLength'],
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
                        param['SectionLength'],
                        water_transform['xstep'],
                        water_transform['ystep']
                    )
                ]
            ),
            dtype=xsections.dtype
        )
        xsections = np.append(xsections, section)

    print(len(xsections))
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
    for idx in range(0, len(xsections), 1):
        # Finds the channel width and associated points
        if param['manual']:
            dem_width, dem_points = riv.mannual_find_channel_width(
                idx,
                xsections[idx]['elev_section']
            )
            plt.close('all')
        else:
            banks, dem_width, dem_points = riv.find_channel_width(
                xsections[idx]['elev_section'],
                xsections[idx]['elev_section'],
                order=param.get('WidthSens')
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
        if dem_points:
            bank_df = bank_df.append(riv.get_bank_positions(
                xsections[idx]['elev_section'],
                dem_points,
                water_points
            ))

        # Save width values to the major cross-section structure
        xsections[idx]['bank'] = dem_points
        xsections[idx]['dem_width'] = dem_width
        xsections[idx]['water_width'] = water_width

    # Save the Channel Cross Sections Structure
    print('Saving Cross-Section Structure')
    np.save(param['OutputRoot'] + 'xsections.npy', xsections)

    if len(bank_df) > 0:
        print('Saving Channel Banks')
        bank_df.to_csv(param['OutputRoot'] + 'channel_banks.csv')

    # Save the width dataframe
    print('Saving Width DataFrame')
    riv.save_channel_widths(xsections).to_csv(
        param['OutputRoot'] + 'width_dataframe.csv'
    )

    return True


if __name__ == "__main__":
    main()
