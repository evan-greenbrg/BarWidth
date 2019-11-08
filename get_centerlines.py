import cv2
import gdal
import numpy as np
from rivamap import preprocess, singularity_index, delineate, georef, visualization
import rasterio
from rasterio.plot import show
from scipy.ndimage import label as ndlabel
from scipy.ndimage import sum as ndsum

from DownloadHandler import DownloadHandler
from RasterHandler import RasterHandler

output_name = '/Users/evangreenberg/PhD Documents/Projects/river-profiles/Landsat/B6_clip.tif'
B3_name = '/Users/evangreenberg/PhD Documents/Projects/river-profiles/Landsat/LC08_L1TP_025039_20190514_20190521_01_T1_B3.TIF'
B6_name = '/Users/evangreenberg/PhD Documents/Projects/river-profiles/Landsat/LC08_L1TP_025039_20190514_20190521_01_T1_B6.TIF'
DEM_name = '/Users/evangreenberg/PhD Documents/Projects/river-profiles/DEM/output_be.tif'

rh = RasterHandler()
ds = gdal.Open(DEM_name)

demEPSG = 4269
landsatEPSG = 4326
pointX = 320254.5 
pointY = 3319675.5 

minx, miny, maxx, maxy = rh.bounding_coordinates(ds)
minx, miny = rh.transform_coordinates(
    minx,
    miny,
    demEPSG,
    landsatEPSG
)
maxx, maxy = rh.transform_coordinates(
    maxx,
    maxy,
    demEPSG,
    landsatEPSG
)

rh.clip_raster(B6_name, output_name, minx, maxy, maxx, miny)
DOWNLOAD = False 
paths = [
    'http://landsat-pds.s3.amazonaws.com/c1/L8/025/039/LC08_L1TP_025039_20190514_20190521_01_T1/LC08_L1TP_025039_20190514_20190521_01_T1_B3.TIF',
    'http://landsat-pds.s3.amazonaws.com/c1/L8/025/039/LC08_L1TP_025039_20190514_20190521_01_T1/LC08_L1TP_025039_20190514_20190521_01_T1_B6.TIF'
]

######################################
###        RivaMap Stuff           ###
######################################

dl = DownloadHandler('max')
images = {}
if DOWNLOAD:
    for path in paths:
        dl.put(path)

    root = '~/Phd Documents/Projects/river-profiles/Landsat/'
    for path in paths:
        dl.save(path, root)

# I wanted to work from memory only, and get files from GD as needed
# I couldn't quite get the download to work properly (was loosing information)
# The below code block is where I started on this. I still like this avenue

# for path in paths:
#     filename = path.split('/')[-1]
#     image = download.get(filename)
#     images[filename] = np.fromstring(
#         image,
#         np.uint8
#     )
#     images[filename] = np.asarray(
#         bytearray(image.read()),
#         dtype=np.uint8
#     )

# B3 = cv2.imdecode(images[B3_name], cv2.IMREAD_COLOR)
# B6 = cv2.imdecode(images[B6_name], cv2.IMREAD_COLOR)

# File names to load in:
b3name = '/Users/evangreenberg/PhD Documents/Projects/river-profiles/Landsat/B3_clip.tif'
b6name = '/Users/evangreenberg/PhD Documents/Projects/river-profiles/Landsat/B6_clip.tif'
# Load the File
B3 = cv2.imread(b3name, cv2.IMREAD_UNCHANGED)
B6 = cv2.imread(b6name, cv2.IMREAD_UNCHANGED)

# Clean the DEM
DEM_name = '/Users/evangreenberg/PhD Documents/Projects/river-profiles/DEM/output_be.tif'
DEM = cv2.imread(DEM_name, -1)
DEM_FIX = np.where(DEM<0, False, DEM)
DEM_max = DEM_FIX.max()
DEM_min = DEM_FIX.min()
DEM_NORM = (DEM_FIX - DEM_min) / (DEM_max - DEM_min)

# RivaMap Processing
I1 = preprocess.mndwi(B3, B6)
filters = singularity_index.SingularityIndexFilters()
psi, widthMap, orient = singularity_index.applyMMSI(I1, filters)
nms = delineate.extractCenterlines(orient, psi)
centerline = delineate.getBiggestCenterline(nms)

gm = georef.loadGeoMetadata(B3_name)
psi = preprocess.contrastStretch(raster)
psi = preprocess.double2im(raster, 'uint16')
georef.saveAsGeoTiff(gm, raster, "raster_geotagged.TIF")

# plotting
import matplotlib.pyplot as plt
import matplotlib.image as mpimgkjk

imgplot = plt.imshow(centerline)
plt.show()


######################################
###         Rasterio Stuff         ###
######################################


rows, columns = centerlines.shape
empty_row = np.zeros((3, columns))
A = np.vstack([centerlines, empty_row])
A = np.vstack([empty_row, A])

strel = np.ones((3, 3), dtype=bool)
cclabels, numcc = ndlabel(centerlines, strel)
sumstrong = ndsum(centerlines, cclabels, list(range(1, numcc+1)))
