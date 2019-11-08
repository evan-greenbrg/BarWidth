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

    
    gm = georef.loadGeoMetadata(b3path)
    # Get Lat, Lon Coordinates from Image
    coordinates = riv.get_river_coordinates(centerline, gm)

    rh = RasterHandler()
    # Convert between landsat and dem projections and lat, lon to utm
    lansatEPSG = 4326
    demEPSG = 4269
    prj_str = '+proj=utm +zone=15U, +north +ellps=GRS80 +datum=NAD83 +units=m +no_defs'
    myProj = Proj(prj_str)       
    coord_transform = pandas.DataFrame(columns=['lat', 'lon'])
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
	    data=[[lon_, lat_]],
	    columns=['lat', 'lon']
	)
	coord_transform = coord_transform.append(df)

    coordinates = coord_transform.reset_index(drop=True)

    # Find dem values at coordinate array
    ds = gdal.Open(DEMpath, 0)
    dem = ds.ReadAsArray()
    values = rh.values_from_coordinates(ds, dem, coordinates)
    del ds

    coordinates = riv.get_direction(coordinates)
    coordinates = riv.get_inverse_direction(coordinates)

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


# plotting
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimgkjk
# 
# imgplot = plt.imshow(centerline)
# plt.show()


