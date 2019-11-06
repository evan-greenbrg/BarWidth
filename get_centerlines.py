import cv2
import numpy
from rivamap import preprocess, singularity_index, delineate, georef, visualization
import rasterio

from download_image import Downloader

DOWNLOAD = False 
paths = [
    'http://landsat-pds.s3.amazonaws.com/c1/L8/025/039/LC08_L1TP_025039_20190514_20190521_01_T1/LC08_L1TP_025039_20190514_20190521_01_T1_B3.TIF',
    'http://landsat-pds.s3.amazonaws.com/c1/L8/025/039/LC08_L1TP_025039_20190514_20190521_01_T1/LC08_L1TP_025039_20190514_20190521_01_T1_B6.TIF'
]

######################################
###        RivaMap Stuff           ###
######################################

download = Downloader('windows')
images = {}
if DOWNLOAD:
    for path in paths:
        download.put(path)

root = 'downloads/'
for path in paths:
    download.save(path, root)
#for path in paths:
#    filename = path.split('/')[-1]
#    image = download.get(filename)
#    images[filename] = numpy.fromstring(
#        image,
#        numpy.uint8
#    )
#    images[filename] = numpy.asarray(
#        bytearray(image.read()),
#        dtype=numpy.uint8
#    )

B3_name = 'LC08_L1TP_025039_20190514_20190521_01_T1_B3.TIF'
B6_name = 'LC08_L1TP_025039_20190514_20190521_01_T1_B6.TIF'
B3 = cv2.imread(B3_name, cv2.IMREAD_UNCHANGED)
B6 = cv2.imread(B6_name, cv2.IMREAD_UNCHANGED)

DEM_name = 'DEMs/output_be.tif'
DEM = cv2.imread(DEM_name, -1)
DEM_FIX = numpy.where(DEM<0, False, DEM)
DEM_max = DEM_FIX.max()
DEM_min = DEM_FIX.min()
DEM_NORM = (DEM_FIX - DEM_min) / (DEM_max - DEM_min)


# B3 = cv2.imdecode(images[B3_name], cv2.IMREAD_COLOR)
# B6 = cv2.imdecode(images[B6_name], cv2.IMREAD_COLOR)
 #B3 = cv2.imdecode(images[B3_name], cv2.IMREAD_COLOR)
 #B3 = B3.astype('float32')
 #B6 = cv2.imdecode(images[B6_name], cv2.IMREAD_COLOR)
 #B6 = B6.astype('float32')

I1 = preprocess.mndwi(B3, B6)
cv2.imwrite('mndwi.TIF', cv2.normalize(I1, None, 0, 255, cv2.NORM_MINMAX))

filters = singularity_index.SingularityIndexFilters()
psi, widthMap, orient = singularity_index.applyMMSI(DEM_NORM, filters)

nms = delineate.extractCenterlines(orient, psi)
centerlines = delineate.thresholdCenterlines(nms)

raster = visualization.generateRasterMap(centerlines, orient, widthMap)
visualization.generateVectorMap(centerlines, orient, widthMap, saveDest = "vector.pdf")
#visualization.quiverPlot(psi, orient, saveDest = "quiver.pdf")

gm = georef.loadGeoMetadata(B3_name)
psi = preprocess.contrastStretch(raster)
psi = preprocess.double2im(raster, 'uint16')
georef.saveAsGeoTiff(gm, raster, "raster_geotagged.TIF")

# plotting
import matplotlib.pyplot as plt
import matplotlib.image as mpimgkjk

imgplot = plt.imshow(DEM_NORM)
plt.show()


######################################
###         Rasterio Stuff         ###
######################################

dataset = rasterio.open('DEMs/output_be.tif')
band1 = dataset.read(1)
