import os
import json

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


DemPath = '/Volumes/EGG-HD/PhD Documents/Projects/Bar_Width/White_River_IN/WhiteRiverDEM_32616.tif'

B3path = '/Volumes/EGG-HD/PhD Documents/Projects/Bar_Width/White_River_IN/LC08_L1TP_021033_20191110_20191115_01_T1_B3.tiff'
B6path = '/Volumes/EGG-HD/PhD Documents/Projects/Bar_Width/White_River_IN/LC08_L1TP_021033_20191110_20191115_01_T1_B6.tiff'

B3out = '/Volumes/EGG-HD/PhD Documents/Projects/Bar_Width/White_River_IN/LC08_L1TP_021033_20191110_20191115_01_T1_B3_clip.tif'
B6out = '/Volumes/EGG-HD/PhD Documents/Projects/Bar_Width/White_River_IN/LC08_L1TP_021033_20191110_20191115_01_T1_B6_clip.tif'
epsg = 32616

# Clip the Raster
clip_raster(B3path, DemPath, epsg, B3out)
clip_raster(B6path, DemPath, epsg, B6out)


# Find the centerline
ds = gdal.Open(B3out)
B3 = numpy.array(ds.GetRasterBand(1).ReadAsArray())
ds = gdal.Open(B6out)
B6 = numpy.array(ds.GetRasterBand(1).ReadAsArray())

print('Finding Centerline')
centerline = get_centerline(B3, B6, 200)

print('Finding Coordinates')
raster_ds = rasterio.open(B3out)
coordinates = []
for pair in numpy.transpose(numpy.nonzero(centerline)):
    coordinates.append(
        rasterio.transform.xy(raster_ds.transform, pair[0], pair[1])
    )

oroot = '/Users/evangreenberg/PhD Documents/Projects/river-profiles/Input_Data/White_river'
f = 'white_river_water_points.csv'
outpath = os.path.join(oroot, f)

coordinate_df = pandas.DataFrame(coordinates, columns=['lon', 'lat'])
coordinate_df.to_csv(outpath)


# Reproject to NAD83
from RasterHandler import RasterHandler


rh = RasterHandler()
centerline_path = '/Users/evangreenberg/PhD Documents/Projects/river-profiles/Input_Data/White_river/white_river_centerline_32616.csv'

iepsg = 32616
oepsg = 4326
df = pandas.read_csv(centerline_path)
lon = []
lat = []
for i, row in df.iterrows():
    x, y = rh.transform_coordinates(row['lon'], row['lat'], iepsg, oepsg)
    lat.append(x)
    lon.append(y)

df['lon_'] = lon
df['lat_'] = lat

out_path = '/Users/evangreenberg/PhD Documents/Projects/river-profiles/Input_Data/White_river/white_river_centerline_4326.csv'
df.to_csv(out_path)
