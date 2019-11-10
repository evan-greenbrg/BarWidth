import math
import cv2
import gdal
import numpy as np
import pandas
from pyproj import Proj
from rivamap import preprocess, singularity_index, delineate, georef, visualization
import rasterio
from rasterio.plot import show
from scipy.ndimage import label as ndlabel
from scipy.ndimage import sum as ndsum

from RasterHandler import RasterHandler
from RiverHandler import RiverHandler


b3path = '/Users/evangreenberg/PhD Documents/Projects/river-profiles/Landsat/B3_clip.tif'
b6path = '/Users/evangreenberg/PhD Documents/Projects/river-profiles/Landsat/B6_clip.tif'
DEMpath = '/Users/evangreenberg/PhD Documents/Projects/river-profiles/DEM/output_be.tif'


def main(b3path, b6path, DEMpath): 

    # Load the File
    B3 = cv2.imread(b3path, cv2.IMREAD_UNCHANGED)
    B6 = cv2.imread(b6path, cv2.IMREAD_UNCHANGED)

    riv = RiverHandler()
    # Find Biggest Centerline in Image
    centerline = riv.get_centerline(B3, B6)

    
    gmLandsat = georef.loadGeoMetadata(b3path)
    # Get Lat, Lon Coordinates from Image
    coordinates = riv.get_river_coordinates(centerline, gmLandsat)

    rh = RasterHandler()
    # Convert between landsat and dem projections and lat, lon to utm
    lansatEPSG = 4326
    demEPSG = 4269
    prj_str = '+proj=utm +zone=15U, +north +ellps=GRS80 +datum=NAD83 +units=m +no_defs'
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

    # Load in DEM and find meta data
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

    # Get values at each coordinate location
    values = rh.values_from_coordinates(ds, dem, coordinates)
    coordinates['elev_0'] = values

    # Find the channel direction and inverse channel direction
    coordinates = riv.get_direction(coordinates)
    coordinates = riv.get_inverse_direction(coordinates)

    # Start building river dem
    river_dem = np.full(dem.shape, 0.)
    for idx, row in coordinates.iterrows():
        x, y = georef.lonlat2pix(gmDEM, row['lon'], row['lat'])
        river_dem[y, x] = row['elev_0']

    # Build Cross-Section Structure
    types = [
        ('coords', 'object'), 
        ('pixel', 'object'), 
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
            200,
            xstep,
            ystep
        )
        xpix, ypix = rh.get_dem_pixels(
            row['easting'], 
            row['northing'], 
            xOrigin,
            yOrigin,
            pixelWidth,
            pixelHeight
        )
        section = np.array(
            tuple(
                [(row['easting'], row['northing']), (xpix, ypix), xsection]
            ),
            dtype=xsections.dtype
        )
        xsections = np.append(xsections, section)


#################################################
####         NEXT STEPS                      ####
#################################################
# 1. Find the northing easting magnitude for each pixel on DEM
# 2. Find coordinates at each pixel step moving in inv direction from channel
#    going in both directions at some fixed diistance -> cross section
# 3. Automatic calculation of channel widths (find distance between maxima?)
# 4. Automatic detection of point bars. Area of maximum curvature? (will be hard)
# 4. Automatic calculation of clinoform surfaces (This will be the hard part)
#    Need to make sure that you are not hard coding the answer
#    Will it work if clinoform width is distance between inner channel bank 
#    and bottom of channel?


# # plotting
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimgkjk
# import matplotlib.colors as colors
# from copy import copy
# # 
# # imgplot = plt.imshow(centerline)
# # plt.show()
# 
# 
# #################################################
# ####         PLOTTING                        ####
# #################################################
# river_mask_d = np.ma.masked_where(
#     river_dem <= 0.,
#     river_dem 
# )
# river_mask = np.ma.masked_where(
#     centerline == False,
#     centerline 
# )
# 
# x, y = dem_clean.shape
# extent = 0, y, 0, x 
# fig = plt.figure(frameon=False)
# 
# im1 = plt.imshow(
#     dem_clean, 
#     cmap=plt.cm.gray, 
#     norm=colors.Normalize(),
#     interpolation='nearest', 
#     extent=extent
# )
# # Set up a colormap:
# # use copy so that we do not mutate the global colormap instance
# im2 = plt.imshow(
#     river_mask_d,
#     cmap=plt.cm.plasma, 
#     norm=colors.Normalize(),
#     interpolation='bilinear',
#     extent=extent
# )
# 
# plt.show()
