from matplotlib import pyplot as plt
import rasterio
import pandas
import gdal
import osr
from pyproj import Proj
from skimage import graph
import numpy as np


def convertCoordinates(centerline, cols):
    easts = []
    norths = []
    for idx, row in centerline.iterrows():
        east, north = myProj(row[cols[0]], row[cols[1]])
        easts.append(east)
        norths.append(north)

    return easts, norths

def getProj(img):
    # Get ESPG
    ds = gdal.Open(img, 0)
    ProjStr = "epsg:{0}".format(
        osr.SpatialReference(
            wkt=ds.GetProjection()
        ).GetAttrValue('AUTHORITY', 1)
    )
    ds = None
    return Proj(ProjStr)


# Get all the sources
sources = {
    'Brazos River': (
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Brazos_Near_Calvert/brazos_centerline.csv',
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Brazos_Near_Calvert/BrazosCalvert_26914.tif',
        ['lon', 'lat'],
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Brazos_Near_Calvert/brazos_bar_coords.csv'
    ),
    'Koyukuk River': (
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Koyukuk/koyukuk_centerlines.txt',
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Koyukuk/Koyukuk_dem.tif_3995.tif',
        ['POINT_X', 'POINT_Y'],
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Koyukuk/koyukuk_bar_coords.csv'
    ),
    'Mississippi River': (
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Mississippi/mississippi_centerline_lats.csv',
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Mississippi/Mississippi_1_26915_meter.tif',
        ['y_lon', 'x_lat'],
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Mississippi/mississippi_bar_lats.csv'
    ),
    'Mississippi River - Leclair': (
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Mississippi_Leclair/miss_leclair_centerline_4326.csv',
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Mississippi_Leclair/MississippiDEM_meter.tif',
        ['lon_', 'lat_'],
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Mississippi_Leclair/miss_leclair_bar_coords.csv'
    ),
    'Nestucca River': (
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Beaver_OR/beaver_centerline_4326.csv',
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Beaver_OR/beaver_26910_meter.tif',
        ['POINT_X', 'POINT_Y'],
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Beaver_OR/beaver_bar_coords.csv'
    ),
    'Powder River': (
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Powder_River/powder_centerline_dense_4326.csv',
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Powder_River/output_be_26913.tif',
        ['lon_', 'lat_'],
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Powder_River/powder_river_ar_bar_coords.csv'
    ),
    'Red River': (
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Red_River/red_river_ar_centerline.csv',
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Red_River/red_river_dem_meter.tif',
        ['POINT_X', 'POINT_Y'],
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Red_River/red_river_ar_bar_coords.csv'
    ),
    'Rio Grande River': (
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Rio_Grande_TX/RioGrandeCenterlineLats.csv',
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Rio_Grande_TX/RioGrandeTxDEM.tif',
        ['lon_', 'lat_'],
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Rio_Grande_TX/riogrande_TX_bar_coords.csv'
    ),
    'Sacramento River': (
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Sacramento/savramento_centerline_4326.csv',
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Sacramento/sacramento_merged_26910.tif',
        ['lon_', 'lat_'],
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Sacramento/sacramento_bar_coords.csv'
    ),
    'Tombigbee River': (
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Tombigbee/tombigbee_centerline_4326.csv',
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Tombigbee/tombigbee_26916_10m_clip.tif',
        ['lon_', 'lat_'],
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Tombigbee/tobigbee_bar_coords.csv'
    ),
    'Trinity River': (
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Trinity/Trinity_Centerline.txt',
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Trinity/Trinity_1m_be_clip.tif',
        ['POINT_X', 'POINT_Y'],
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Trinity/trinity_bar_coords.csv'
    ),
    'White River': (
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/White_river/white_river_centerline_4326.csv',
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/White_river/WhiteRiverDEM_32616_meter.tif',
        ['lon_', 'lat_'],
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/White_river/white_river_in_bar_coords.csv'
    )
}


test = sources['Brazos River']
ds = rasterio.open(test[1])
image = ds.read(1)
image[image < 0] = None
transform = ds.transform

myProj = getProj(test[1])

centerline = pandas.read_csv(test[0])
centerline['x'], centerline['y'] = convertCoordinates(centerline, test[2])

centerline['row'], centerline['col'] = rasterio.transform.rowcol(
    transform, 
    centerline['x'], 
    centerline['y']
)

centerline_image = np.empty_like(image).astype(int)
centerline_image[list(centerline['row']), list(centerline['col'])] = 1

# plt.imshow(centerline_image)
# plt.scatter(centerline['col'], centerline['row'])
# plt.show()

# Crop image
rmin = centerline['row'].min()
rmax = centerline['row'].max()
cmin = centerline['col'].min()
cmax = centerline['col'].max()

centerline_image = centerline_image[rmin:rmax, cmin:cmax]
centerline_ind = np.array(np.where(centerline_image == 1)).T

# plt.imshow(centerline_image)
# plt.scatter(i[1], i[0])
# plt.show()


image = None
ds = None

# Get Endpoints
endpoint1 = centerline_ind[0]
endpoint2 = centerline_ind[-1]

costs = np.where(centerline_image, 1, 1000)
route, cost = graph.shortest_path(
    centerline_image, 
    reach=2,
    output_indexlist=True
)

test = pandas.DataFrame(route, columns=['row', 'col'])

plt.imshow(centerline_image)
plt.scatter(test['col'], test['row'])
plt.show()

