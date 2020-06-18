import os
import json
from itertools import product

import pandas
import cv2
import numpy
from rivamap import preprocess, singularity_index, delineate, georef, visualization
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.mask import mask
from osgeo import gdal
from fiona.crs import from_epsg
from matplotlib import pyplot as plt


def bounding_coordinates(ds):
    """
    Finds bounding coordinates from a geoTif file
    """
    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()
    minx = gt[0]
    miny = gt[3] + width*gt[4] + height*gt[5]
    maxx = gt[0] + width*gt[1] + height*gt[2]
    maxy = gt[3]

    return minx, miny, maxx, maxy


def clip_raster(ipath, mask_path, epsg, opath):
    """
    Clips raster file based on bounding box coordinates
    """
    data = rasterio.open(ipath)
    dsmask = gdal.Open(mask_path)

    minx0, miny0, maxx0, maxy0 = bounding_coordinates(dsmask)
    dsmask = None

    bbox = box(minx0, miny0, maxx0, maxy0)
    geo = gpd.GeoDataFrame(
        {'geometry': bbox},
        index=[0],
        crs=from_epsg(epsg)
    )
    coords = [json.loads(geo.to_json())['features'][0]['geometry']]

    out_img, out_transform = mask(
        dataset=data,
        shapes=coords,
        crop=True
    )

    out_meta = data.meta.copy()
    data = None

    out_meta.update(
        {
            "driver": "GTiff",
            "height": out_img.shape[1],
            "width": out_img.shape[2],
            "transform": out_transform,
            "crs": rasterio.crs.CRS.from_epsg(epsg)
        }
    )

    with rasterio.open(opath, "w", **out_meta) as dest:
        dest.write(out_img)

    return opath


def get_centerline(B3, B6, size):
    """
    Using RivaMap library finds the largest centerline from 
    Bands 3 and 6 lansat 8 imagery
    """
    I1 = preprocess.mndwi(B3, B6)
    filters = singularity_index.SingularityIndexFilters()
    psi, widthMap, orient = singularity_index.applyMMSI(I1, filters)
    nms = delineate.extractCenterlines(orient, psi)
    
    return delineate.getBiggestCenterline(nms, size)


def get_tiles(path, n):
    with rasterio.open(path) as inds:
        # Get shape of image array 
        ncols = inds.meta['width']
        nrows = inds.meta['height']

        # Get width and height
        width = ncols//n
        height = nrows//n

        offsets = product(
            range(0, ncols, width), 
            range(0, nrows, height)
        )

        big_window = rasterio.windows.Window(
            col_off=0, 
            row_off=0, 
            width=ncols, 
            height=nrows
        )

        for col_off, row_off in  offsets:
            window = rasterio.windows.Window(
                col_off=col_off, 
                row_off=row_off, 
                width=width, 
                height=height
            ).intersection(big_window)

            transform = rasterio.windows.transform(window, inds.transform)

            yield window, transform

DemPath = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Rio_Grande_TX'

B3path = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Rio_Grande_TX/Landsat/RioGrande_B3_26914_clip.tif'
B6path = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Rio_Grande_TX/Landsat/RioGrande_B6_26914_clip.tif'

B3out = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Rio_Grande_TX/Landsat/RioGrande_B3_26914_clip.tif'
B6out = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Rio_Grande_TX/Landsat/RioGrande_B6_26914_clip.tif'
epsg = 29614

# Clip the Raster
# clip_raster(B3path, DemPath, epsg, B3out)
# clip_raster(B6path, DemPath, epsg, B6out)

# Find the centerline
srcB3 = rasterio.open(B3out)
srcB6 = rasterio.open(B6out)

B3 = srcB3.read(1)
B6 = srcB6.read(1)
centerline = get_centerline(B3, B6, 200)

print('Finding Centerline')
coordinates = []
for window, transform in get_tiles(B3out, 1):

    print(window)
    print(transform)

    winB3 = srcB3.read(1)
    winB6 = srcB6.read(1)

    if (winB3.shape[0] < 10) | (winB3.shape[1] < 10):
        continue

    else:
        centerline = get_centerline(winB3, winB6, 200)

        for pair in numpy.transpose(numpy.nonzero(centerline)):
            coordinates.append(
                rasterio.transform.xy(transform, pair[0], pair[1])
            )

oroot = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Rio_Grande_TX'
f = 'riogrande_water_points.csv'
outpath = os.path.join(oroot, f)

coordinate_df = pandas.DataFrame(coordinates, columns=['lon', 'lat'])
coordinate_df.to_csv(outpath)


# Reproject to NAD83
from RasterHandler import RasterHandler


rh = RasterHandler()
centerline_path = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Rio_Grande_TX/riograndeTX_centerline.csv'

iepsg = 26914
oepsg = 4326 
df = pandas.read_csv(centerline_path)
df = df.iloc[::4, :]
df = df.astype('float')
lon = []
lat = []
for i, row in df.iterrows():
    x, y = rh.transform_coordinates(row['lon'], row['lat'], iepsg, oepsg)
    lat.append(x)
    lon.append(y)

df['lon_'] = lon
df['lat_'] = lat

out_path = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Rio_Grande_TX/riograndeTX_centerline_4326.csv'
df.to_csv(out_path)
