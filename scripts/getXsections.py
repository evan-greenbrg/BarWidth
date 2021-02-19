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
from BarWidth import BarHandler


# If you don't want to sample all of the centerline points
STEP = 1


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
    if not os.path.exists(param['barPath']):
        raise FileNotFoundError(
            errno.ENOENT,
            os.strerror(errno.ENOENT),
            param['barPath']
        )
    if not param.get('CenterlineSmoothing'):
        raise NameError('No Centerline Smoothing Parameter given')
    if not param.get('OutputRoot'):
        raise NameError('No Output root given')
    if not param.get('SectionLength'):
        raise NameError('No section length given')
    if not param.get('manual'):
        param['manual'] = True

    # Initialize classes
    riv = RiverHandler.RiverHandler()
    rh = RasterHandler.RasterHandler()
    bh = BarHandler.BarHandler()

    # Load DEM data set
    ds = gdal.Open(param['DEMpath'], 0)

    # Get Proj string from the DEM
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

    if len(coordinates) == 0:
        sys.exit("No coordinates")

    # Find the channel direction and inverse channel direction
    print('Finding channel and cross-section directions')
    coordinates = riv.get_direction(coordinates)
    coordinates = riv.get_inverse_direction(coordinates)
    coordinates = coordinates.dropna(axis=0, how='any')

    # Downsample if you want
    if STEP:
        coordinates = coordinates.iloc[::STEP, :].reset_index(drop=True)

    # Get Bar Coordinates
    # Read in the bar file to find the channel bars
    print('Loading Bar .csv file')
    bar_df = pandas.read_csv(
        param['barPath'],
        names=[
            'Latitude_us',
            'Longitude_us',
            'Latitude_ds',
            'Longitude_ds'
        ],
        header=1
    )

    # Convert the Bar Lat Long to UTM Easting Northing
    print('Converting Bar Coordinates to Easting Northing')
    bar_df = bh.convert_bar_to_utm(myProj, bar_df)

    # Filter river coordinates by bar
    print('Making Bar section structure')
    bar_sections = pandas.DataFrame()
    for idx, bar in bar_df.iterrows():
        sections = bh.get_bar_coordinates(
            coordinates,
            bar
        )
        print(len(sections))
        sections['bar'] = [idx for pdx in range(len(sections))]
        bar_sections = pandas.concat([bar_sections, sections])
        bar_sections = bar_sections.reset_index(drop=True)
#        bar_sections[str(idx)] = sections

    # Save the Coordinates file
    bar_sections.to_csv(param['OutputRoot'] + 'coordinates.csv')

    # Build the cross-section structure
    print('Building channel cross sections')
    types = [
        ('coords', 'object'),
        ('bar', 'object'),
        ('dem_width', 'f8'),
        ('water_width', 'f8'),
        ('bank', 'object'),
        ('elev_section', 'object'),
    ]
    xsections = np.array([], dtype=types)
    # Iterate through each coordinate of the channel centerline
    for idx, row in bar_sections.iterrows():
        # Get the cross-section and set-up numpy stucture
        section = np.array(
            tuple(
                [
                    (row['easting'], row['northing']),
                    str(int(row['bar'])),
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
                    )
                ]
            ),
            dtype=xsections.dtype
        )
        xsections = np.append(xsections, section)

    bar_xsections = {str(i): [] for i in bar_df.index}
    # Separate sections to each bar
    for idx, section in enumerate(xsections):
        bar_xsections[section['bar']].append(section)

    # Sample 6 sections per bar
    for key, value in bar_xsections.items():
        step = len(value) // 5
        if step == 0:
            bar_xsections[key] = value
        else:
            bar_xsections[key] = value[step::step]

    new_xsections = []
    # Combine back into xsections
    for key, value in bar_xsections.items():
        new_xsections += value

    xsections = np.array(new_xsections)

    ds = None
    coordinates = None
    bar_xsections = None
    new_xsections = None

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
        ]
    )

    # Iterate through exsections to find widths
    for idx in range(0, len(xsections), 1):

        # Finds the channel width and associated points
        dem_width, dem_points = riv.mannual_find_channel_width(
            idx,
            xsections[idx]['elev_section'],
            xsections[idx]['bar'],
            thresh=0
        )
        plt.close('all')

        if dem_points:
            bank_df = bank_df.append(riv.get_bank_positions(
                xsections[idx]['elev_section'],
                dem_points,
            ))

        # Save width values to the major cross-section structure
        xsections[idx]['bank'] = dem_points
        xsections[idx]['dem_width'] = dem_width

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

param = {
    'DEMpath': '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Powder_River/output_be_26913.tif',
    'CenterlinePath': '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Powder_River/Powder_centerline_manual.csv',
    'barPath': '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Powder_River/powder_river_ar_bar_coords.csv',
    'CenterlineSmoothing': 3,
    'SectionLength': 200,
    'SectionSmoothing': 5,
    'WidthSens': 12,
    'OutputRoot': '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Powder/',
}
