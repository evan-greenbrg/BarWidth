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


b3path = '/Users/evangreenberg/PhD Documents/Projects/river-profiles/Landsat/B3_clip.tif'
b6path = '/Users/evangreenberg/PhD Documents/Projects/river-profiles/Landsat/B6_clip.tif'
DEMpath = '/Users/evangreenberg/PhD Documents/Projects/river-profiles/DEM/output_be.tif'


def main(b3path, b6path, DEMpath, smoothing): 

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
        coordinates, n=10
    )

    rh = RasterHandler()
    # Convert between landsat and dem projections and lat, lon to utm
    lansatEPSG = 4326
    demEPSG = 4269
    prj_str = '+proj=utm +zone=15U, +north +ellps=GRS80 +datum=NAD83 +units=m +no_defs'
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

    print('Finding channel and cross-section directions')
    # Find the channel direction and inverse channel direction
    coordinates = riv.get_direction(coordinates, smooth=5)
    coordinates = riv.get_inverse_direction(coordinates)

    # Start building river dem
    river_dem = np.full(dem.shape, 0.)
    for idx, row in coordinates.iterrows():
        x, y = georef.lonlat2pix(gmDEM, row['lon'], row['lat'])
        river_dem[y, x] = row['elev_0']

    print('Building channel cross sections')
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
            300,
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
   
    # Smooth Cross sections if Smoothing is set
    if smoothing:
        print('Smoothing Cross-Sections')
        for idx, section in np.ndenumerate(xsections):
            b = riv.xsection_smoothing(idx, section['xsection'], smoothing)
            xsections[idx[0]]['xsection'] = b


#################################################
####         FINDING CHANNEL WIDTH FOR ONE   ####
#################################################
# Method 1.: Use local minima and maxima, find the biggest difference between
#            neighbors. Use the maximas as the channel endpoints
# Method 2.: Use curvature, find places of maximum curvature (2nd deivative)
def get_d1(test_section):
    d1 = []
    for idx in range(0, len(test_section)):
        if idx == 0:
            d = (
                (test_section[idx+1][1] - test_section[idx][1])
                / (test_section[idx+1][0] - test_section[idx][0])
            )
            d1.append(d)
        elif idx == len(test_section) - 1:
            d = (
                (test_section[idx][1] - test_section[idx-1][1])
                / (test_section[idx][0] - test_section[idx-1][0])
            )
            d1.append(d)
        else:
            d = (
                (test_section[idx+1][1] - test_section[idx-1][1])
                / (test_section[idx+1][0] - test_section[idx-1][0])
            )
            d1.append(d)

    return d1

def get_d2(test_section):
    d2 = []
    for idx in range(0, len(test_section)):
        if idx == 0:
            d = (
                (test_section[idx+1][2] - test_section[idx][2])
                / (test_section[idx+1][0] - test_section[idx][0])
            )
            d2.append(d)
        elif idx == len(test_section) - 1:
            d = (
                (test_section[idx][2] - test_section[idx-1][2])
                / (test_section[idx][0] - test_section[idx-1][0])
            )
            d2.append(d)
        else:
            d = (
                (test_section[idx+1][2] - test_section[idx-1][2])
                / (test_section[idx+1][0] - test_section[idx-1][0])
            )
            d2.append(d)

    return d2


# Get X Section
idx = 1000 
t = xsections[idx]['xsection']['demvalue_sm']
p = xsections[idx]['xsection']['position']
data = {'position': p, 'elevation': t}
cross_section = pandas.DataFrame(data=data, columns=['position', 'elevation'])

# Find Maxima and Minima
order = 5 
maxs = argrelextrema(t, np.greater, order=order)
maxima = np.column_stack((p[maxs], t[maxs]))
mins = argrelextrema(t, np.less, order=order)
minima = np.column_stack((p[mins], t[mins]))

extremes = np.concatenate([maxima, minima])
extremes = extremes[extremes[:,0].argsort()]
# Get biggest difference between ADJACENT maxima and minma
d = []
for i in range(0, len(extremes)):
    if i == len(extremes) - 1:
        d.append(0) 
    else:
        diff = extremes[i+1][1] - extremes[i][1]
        d.append(diff)
maxi = np.where(d == np.amax(d))[0][0]
mini = np.where(d == np.amin(d))[0][0]
max_val = extremes[maxi + 1]
min_val = extremes[mini]
# Take lowest of the two and project across the stream
# This works because the opposite side of the channel HAS to have a different
# Position sine. If that changes, this logic will have to change
if min_val[1] >= max_val[1]:
    opposite_channel_section = cross_section[
        (cross_section['position'] > min_val[0]) 
        & (cross_section['position'] < max_val[0])
        & (cross_section['position'] < 0)
    ]
    width_val = max_val
else:
    opposite_channel_section = cross_section[
        (cross_section['position'] > min_val[0]) 
        & (cross_section['position'] < max_val[0])
        & (cross_section['position'] > 0)
    ]
    width_val = min_val

opposite_val = opposite_channel_section.iloc[
    (
        opposite_channel_section['elevation']
        - width_val[1]
    ).abs().argsort()[:1]
].to_numpy()[0]

# Width Val is the minimum of the adjacent max-mins
# Opposite Val is the closes value on the opposite side of the channel

# Plot
plt.scatter(width_val[0], width_val[1], color='red')
plt.scatter(opposite_val[0], opposite_val[1], color='red')
plt.plot(p, t)
plt.show()

# Find first derivative of curve
test_section = np.column_stack((p, t))
test_section = np.column_stack((
    test_section,
    get_d1(test_section)
))
test_section = np.column_stack((
    test_section,
    get_d2(test_section)
))

# Plot
fig, ax1 = plt.subplots()
color = 'tab:red'
plt.scatter(p[maxs], t[maxs], color='red')
plt.scatter(p[mins], t[mins], color='green')
ax1.plot(p, t, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.plot(p, test_section[:,2], color=color)
ax2.plot(p, test_section[:,3], color='tab:green')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
#################################################
####         MISC                            ####
#################################################
# To CSV
index = 868 
root = '~/PhD Documents/Projects/river-profiles/'
fn = 'test_xsection_sm8.csv'
test_array = xsections[index]['xsection']
df = pandas.DataFrame(test_array)
df.to_csv((root + fn))

# From .npy
root = '~/PhD Documents/Projects/river-profiles/'
fn = 'xsections_test.npy'
path = root + fn
xsections = np.load(fn, allow_pickle=True)

fig = plt.figure()
ax1 = plt.plot(df_sm['position'], df_sm['demvalue'])
plt.plot(x, y, linestyle='dashed')
plt.show()

np.save(root + 'xsections_test', xsections)

#################################################
####         NEXT STEPS                      ####
#################################################
# 1. Convert the "position" field to a true distance
# 2. Automatic calculation of channel widths (find distance between maxima?)
# 3. Automatic detection of point bars. Area of maximum curvature? (will be hard)
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
