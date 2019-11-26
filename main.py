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

from RasterHandler import RasterHandler
from RiverHandler import RiverHandler


CENTERLINE_SMOOTHING = 10
# LANSATEPSG = 4326 # Trinity
LANSATEPSG = 32604
# DEMEPSG = 4269 # Trinity
DEMEPSG = 3413# Koyukuk 
 
# PRJ_STR = '+proj=utm +zone=15U, +north +ellps=GRS80 +datum=NAD83 +units=m +no_defs' # Trinity
PRJ_STR = '+proj=utm +zone=4U, +north +ellps=GRS80 +datum=NAD83 +units=m +no_defs' # KOYKUK
WIDTH_ORDER = 35
section_len = 300
section_smoothing = 35
# section_smoothing = False
build_sections = True
find_width = True

b3path = '/Users/evangreenberg/PhD Documents/Projects/river-profiles/Landsat/B3_clip.tif'
b6path = '/Users/evangreenberg/PhD Documents/Projects/river-profiles/Landsat/B6_clip.tif'
DEMpath = '/Users/evangreenberg/PhD Documents/Projects/river-profiles/DEM/output_be.tif'


def main(b3path, b6path, DEMpath, build_sections, 
         section_len, section_smoothing, find_width, save_vars): 

    # Load the File
    print('Loading the files')
    B3 = cv2.imread(b3path, cv2.IMREAD_UNCHANGED)
    B6 = cv2.imread(b6path, cv2.IMREAD_UNCHANGED)

    riv = RiverHandler()
    # Find Biggest Centerline in Image
    print('Finding the Centerline')
    centerline = riv.get_centerline(B3, B6)

    gmLandsat = georef.loadGeoMetadata(b3path)
    # Get Lat, Lon Coordinates from Image
    print('Finding coordiantes along centerline')
    coordinates = riv.get_river_coordinates(centerline, gmLandsat)
    coordinates = coordinates.reset_index(drop=True)

    # Smooth the river centerline
    coordinates['lon'], coordinates['lat'] = riv.knn_smoothing(
        coordinates, n=CENTERLINE_SMOOTHING
    )

    rh = RasterHandler()
    # MATCH PROJECTIONS
    # Convert between landsat and dem projections and lat, lon to utm
    lansatEPSG = LANSATEPSG
    demEPSG = DEMEPSG
    prj_str = PRJ_STR

    print('Converting coordinates to projection')
    myProj = Proj(prj_str)       
    coord_transform = pandas.DataFrame(
        columns=['lat', 'lon', 'easting', 'northing']
    )
    for idx, row in coordinates.iterrows():
        # landsat -> dem projection
        lon, lat = rh.transform_coordinates(
            row['lon'], 
            row['lat'], 
            lansatEPSG, 
            demEPSG
        )
        # lat, lon -> utm
        lon_, lat_ = myProj(lon, lat) 
        df = pandas.DataFrame(
            data=[[lon, lat, lon_, lat_]],
            columns=['lat', 'lon', 'easting', 'northing']
        )
        coord_transform = coord_transform.append(df)

    coordinates = coord_transform.reset_index(drop=True)

    print('Loading DEM Data')
    # LOAD IN DEM DATA AND META DATA 
    ds = gdal.Open(DEMpath, 0)
    dem = ds.ReadAsArray()
    dem_clean = np.where(dem < 0, False, dem)
    transform = ds.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = -transform[5]
    gmDEM = georef.loadGeoMetadata(DEMpath)
    xstep, ystep = rh.get_pixel_size(DEMpath)

    if build_sections:
        # Get values at each coordinate location
        values = rh.values_from_coordinates(ds, dem, coordinates)
        coordinates['elev_0'] = values

        print('Finding channel and cross-section directions')
        # Find the channel direction and inverse channel direction
        coordinates = riv.get_direction(coordinates)
        coordinates = riv.get_inverse_direction(coordinates)

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
                section_len,
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
   
    # Smooth Cross sections if Smoothing is set
    if section_smoothing:
        print('Smoothing Cross-Sections')
        for idx, section in np.ndenumerate(xsections):
            print(idx)
            b = riv.xsection_smoothing(
                idx, 
                section['xsection'], 
                section_smoothing
            )
            xsections[idx[0]]['xsection'] = b

    if find_width:
        for idx, section in np.ndenumerate(xsections):
            p = xsections[idx[0]]['xsection']['distance']
            if section_smoothing:
                t = xsections[idx[0]]['xsection']['demvalue_sm']
            else:
                t = xsections[idx[0]]['xsection']['demvalue']
            banks , width = riv.find_channel_width(p, t, order=WIDTH_ORDER)
            xsections[idx[0]]['bank'] = banks
            xsections[idx[0]]['width'] = width

        widths = []
        eastings = []
        northings = []
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

    if find_bar_width:
        bh = BarHandler()
        bar_df_path = '/Users/evangreenberg/PhD Documents/Projects/river-profiles/test_bars.csv'
        bar_df = pandas.read_csv(bar_df_path)
        n = 1
        bars_ = {}
        for idx, bar in bar_df.iterrows():
            i = 0
            ratio = []
            idxs = []
            name = 'bar_{n}'.format(n=n)
            widths = []
            bar_widths = []
            coords0 = []
            coords1 = []
            bar_sections = bh.get_bar_xsections(
                coordinates, 
                xsections, 
                bar_df.iloc[idx]
            )
            for section in bar_sections:
                if section['width'] == 'nan':
                    widths.append(False)
                else:
                    widths.append(section['width'])
                    coords0.append(section[0])
                    bar_widths.append(bh.find_bar_width(section['bank']))

            for idx, width in enumerate(widths):
                i += 1
                if width and bar_widths[idx]:
                    ratio.append(width/bar_widths[idx])
                    idxs.append(i)
                    coords1.append(coords0[idx])
            bars_[name] = {'idx': idxs, 'coords': coords1, 'ratio': ratio}
            n += 1

    if visualize:
        vh = Visualizer(
            xsections[0]['coords'][0], 
            xsections[0]['coords'][1]
        )
        root = '/Users/evangreenberg/PhD Documents/Projects/river-profiles/plots/'
        name = 'Trinity_bars.png'
        vh.plot_downstream_bars(bars_, root + name)
