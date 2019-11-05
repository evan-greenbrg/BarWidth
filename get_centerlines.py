import cv2
import numpy
from rivamap import preprocess, singularity_index, delineate, georef, visualization

from download_image import Downloader

DOWNLOAD = False 
paths = [
    'http://landsat-pds.s3.amazonaws.com/c1/L8/025/039/LC08_L1TP_025039_20190514_20190521_01_T1/LC08_L1TP_025039_20190514_20190521_01_T1_B3.TIF',
    'http://landsat-pds.s3.amazonaws.com/c1/L8/025/039/LC08_L1TP_025039_20190514_20190521_01_T1/LC08_L1TP_025039_20190514_20190521_01_T1_B6.TIF'
]

download = Downloader('windows')
images = {}
if DOWNLOAD:
    for path in paths:
        download.put(path)

for path in paths:
    filename = path.split('/')[-1]
    image = download.get(filename)
    images[filename] = numpy.fromstring(
        image,
        numpy.uint8
    )
#    images[filename] = numpy.asarray(
#        bytearray(image.read()),
#        dtype=numpy.uint8
#    )

B3_name = 'LC08_L1TP_025039_20190514_20190521_01_T1_B3.TIF'
B6_name = 'LC08_L1TP_025039_20190514_20190521_01_T1_B6.TIF'

# B3 = cv2.imdecode(images[B3_name], cv2.IMREAD_COLOR)
# B6 = cv2.imdecode(images[B6_name], cv2.IMREAD_COLOR)
B3 = cv2.imdecode(images[B3_name], cv2.IMREAD_COLOR)
B3 = B3.astype('float32')
B6 = cv2.imdecode(images[B6_name], cv2.IMREAD_COLOR)
B6 = B6.astype('float32')

I1 = preprocess.mndwi(B3, B6)
I1 - I1.astype('float32')
cv2.imwrite('test/mndwi.TIF', cv2.normalize(I1, None, 0, 255, cv2.NORM_MINMAX))

filters = singularity_index.SingularityIndexFilters()
psi, widthMap, orient = singularity_index.applyMMSI(I1, filters)
