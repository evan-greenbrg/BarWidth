import glob
import os
import json

from osgeo import gdal
import ogr
import osr
import rasterio
import pandas
import numpy as np
from rasterio.mask import mask
from rasterio.merge import merge
from shapely import geometry
from shapely.geometry import box
import geopandas as gpd
from fiona.crs import from_epsg
import pycrs


class RasterHandler():

    def __init__(self):
        pass

    def get_indices(self, x, y, ox, oy, pw, ph):
        """
        Gets the row (i) and column (j) indices in an array
        for a given set of coordinates.
        :param x:   array of x coordinates (longitude)
        :param y:   array of y coordinates (latitude)
        :param ox:  raster x origin
        :param oy:  raster y origin
        :param pw:  raster pixel width
        :param ph:  raster pixel height
        :return:    row (i) and column (j) indices
        """

        i = np.floor((oy-y) / ph).astype('int')
        j = np.floor((x-ox) / pw).astype('int')

        return i, j

    def values_from_coordinates(self, ds, dem, coordinates):
        """
        Finds DEM values from set of coordinates
        """
        xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()
        # calculate indices and index array
        indices = self.get_indices(
            coordinates['easting'].to_numpy(),
            coordinates['northing'].to_numpy(),
            xmin,
            ymax,
            xres,
            -yres
        )
        return dem[indices]

    def bounding_coordinates(self, ds):
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

    def transform_coordinates(self, pointX, pointY, iEPSG, oEPSG):
        """
        Transforms set of coordinates from one coordinate system to another
        """
        # create a geometry from coordinates
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(pointX, pointY)
        # create coordinate transformation
        inSpatialRef = osr.SpatialReference()
        srs = inSpatialRef.ImportFromEPSG(iEPSG)

        outSpatialRef = osr.SpatialReference()
        srs = outSpatialRef.ImportFromEPSG(oEPSG)

        coordTransform = osr.CoordinateTransformation(
            inSpatialRef,
            outSpatialRef
        )

        # transform point
        point.Transform(coordTransform)

        return point.GetX(), point.GetY()

    def clip_raster(self, ipath, dem_path):
        """
        Clips raster file based on bounding box coordinates
        """
        opath = self.rename_path(ipath)
        data = rasterio.open(ipath)

        dsdem = gdal.Open(dem_path)
        dslandsat = gdal.Open(B3input)

        dem_srs = osr.SpatialReference(wkt=dsdem.GetProjection())
        demEPSG = int(dem_srs.GetAttrValue('AUTHORITY', 1))

        data_srs = osr.SpatialReference(wkt=dslandsat.GetProjection())
        dataEPSG = int(data_srs.GetAttrValue('AUTHORITY', 1))

        minx0, miny0, maxx0, maxy0 = self.bounding_coordinates(dsdem)
        bbox = box(minx0, miny0, maxx0, maxy0)
        geo = gpd.GeoDataFrame(
            {'geometry': bbox},
            index=[0],
            crs=from_epsg(demEPSG)
        )
        geo = geo.to_crs(crs=dataEPSG)
        coords = [json.loads(geo.to_json())['features'][0]['geometry']]

        out_img, out_transform = mask(
            dataset=data,
            shapes=coords,
            crop=True
        )

        out_meta = data.meta.copy()
        epsg_code = int(data.crs.data['init'][5:])

        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_img.shape[1],
                "width": out_img.shape[2],
                "transform": out_transform,
                "crs": pycrs.parse.from_epsg_code(epsg_code).to_proj4()
            }
        )

        with rasterio.open(opath, "w", **out_meta) as dest:
            dest.write(out_img)

    def rename_path(self, input_path):
        path_sp = input_path.split('/')
        name_li = path_sp[-1].split('.')
        name_temp = name_li[0] + '_clip'
        name_li[0] = name_temp
        name = '.'.join(name_li)
        path_sp[-1] = name

        return '/'.join(path_sp)

    def get_pixel_size(self, raster_path):
        raster = rasterio.open(raster_path)
        gt = raster.transform
        pixelSizeX = gt[0]
        pixelSizeY = -gt[4]

        return pixelSizeX, pixelSizeY

    def get_dem_pixels(self, east, north, xOrigin, yOrigin,
                       pixelWidth, pixelHeight):

        dem_col = int((east - xOrigin) / pixelWidth)
        dem_row = int((yOrigin - north) / pixelHeight)

        return dem_col, dem_row

    def get_coords_by_step(self, easting, northing, dlon_inv, dlat_inv,
                           xstep, ystep, i, sign=1):
        """
        For a given centerline point, takes the channel direction at that
        point and multiples it by the step to find the next cross-sectional
        point
        """

        # Find the next Easting
        dd_east = dlon_inv * xstep * i
        east = easting + (dd_east * sign)

        # Find the next Northing
        dd_north = dlat_inv * ystep * i
        north = northing + (dd_north * sign)

        # Find the distance between the origin and the new point
        distance = ((dd_east**2) + (dd_north**2))**(1/2) * sign

        return east, north, distance

    def files_to_mosaic(self, dirpath, outpath,
                        search_regex=None, dem_fps=None, write=True):
        """
        Takes a directory path and either a search regex or file list
        to create a mosaic of a bunch of raster tif files.
        Writes to a specified output path
        """
        if search_regex:
            q = os.path.join(dirpath, search_regex)
            dem_fps = glob.glob(q)

        src_files_to_mosaic = []
        for fp in dem_fps:
            src = rasterio.open(fp)
            src_files_to_mosaic.append(src)

        mosaic, out_trans = merge(src_files_to_mosaic)

        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans
            }
        )

        if write:
            with rasterio.open(outpath, "w", **out_meta) as dest:
                dest.write(mosaic)

        return mosaic

    def get_xsection(self, row, dem, xOrigin, yOrigin, pixelWidth,
                     pixelHeight, xlength, xstep, ystep):

        demcol, demrow = self.get_dem_pixels(
            row['easting'],
            row['northing'],
            xOrigin,
            yOrigin,
            pixelWidth,
            pixelHeight
        )
        types = [
            ('distance', 'f4'),
            ('easting', 'U10'),
            ('northing', 'U10'),
            ('demcol', 'i4'),
            ('demrow', 'i4'),
            ('demvalue', 'f4'),
        ]
        xsection = np.array(
            tuple([
                0,
                row['easting'],
                row['northing'],
                demcol,
                demrow,
                row['elev_0'],
            ]),
            dtype=types
        )
        for i in range(1, xlength + 1):
            eastd, northd, distanced = self.get_coords_by_step(
                row['easting'],
                row['northing'],
                row['dlon_inv'],
                row['dlat_inv'],
                xstep,
                ystep,
                i,
                sign=1
            )
            eastu, northu, distanceu = self.get_coords_by_step(
                row['easting'],
                row['northing'],
                row['dlon_inv'],
                row['dlat_inv'],
                xstep,
                ystep,
                i,
                sign=-1
            )

            demcol_d, demrow_d = self.get_dem_pixels(
                eastd,
                northd,
                xOrigin,
                yOrigin,
                pixelWidth,
                pixelHeight
            )
            demcol_u, demrow_u = self.get_dem_pixels(
                eastu,
                northu,
                xOrigin,
                yOrigin,
                pixelWidth,
                pixelHeight
            )
            try:
                value_d = dem[demrow_d][demcol_d]
            except IndexError:
                print('Index out of bounds for axis')
                value_d = None
            try:
                value_u = dem[demrow_u][demcol_u]
            except IndexError:
                print('Index out of bounds for axis')
                value_u = None

            d_pos = distanced
            u_pos = distanceu
            dlist = np.array(
                tuple([d_pos, eastd, northd, demcol_d, demrow_d, value_d]),
                dtype=xsection.dtype
            )
            ulist = np.array(
                tuple([u_pos, eastu, northu, demcol_u, demrow_u, value_u]),
                dtype=xsection.dtype
            )

            xsection = np.insert(xsection, 0, dlist)
            xsection = np.append(xsection, ulist)

        return xsection

    def coordinates_in_dem(self, coordinates, ds, xy):
        """
        Finds the coordinates that are within the dem
        """
        # Find the EPSG
        dem_srs = osr.SpatialReference(wkt=ds.GetProjection())
        EPSG = int(dem_srs.GetAttrValue('AUTHORITY', 1))

        # Find the bounding coordinates of DEM
        minx, miny, maxx, maxy = self.bounding_coordinates(ds)

        # Create Geopandas bounding geometry from bounding coordinates
        bbox = box(minx, miny, maxx, maxy)
        crs = {'init': 'epsg:{}'.format(EPSG)}
        spoly = gpd.GeoSeries([bbox],crs=crs)

        # Convert coordinate points into spatial points
        xcol, ycol = xy
        dem_coordinates = pandas.DataFrame(columns=coordinates.columns)
        for idx, row in coordinates.iterrows():
            point = geometry.Point((row[xcol], row[ycol]))
            if point.within(spoly.geometry.iloc[0]):
                dem_coordinates = dem_coordinates.append(row)

        return dem_coordinates


def main(B3input, B6input, DEM_name):

    rh = RasterHandler()
    rh.clip_raster(B3input, DEM_name)
    rh.clip_raster(B6input, DEM_name)

    DEMdata = rasterio.open(DEM_name)
    landsatB3Data = rasterio.open(B3input)
    landsatB6Data = rasterio.open(B6input)


if __name__ == "__main__":
    B3input = '/Users/evangreenberg/PhD Documents/Projects/river-profiles/Landsat/LC08_L1TP_076014_20190620_20190704_01_T1_B3_clip.tiff'
    B6input = '/Users/evangreenberg/PhD Documents/Projects/river-profiles/Landsat/LC08_L1TP_076014_20190620_20190704_01_T1_B6_clip.tiff'
    DEM_name = '/Users/evangreenberg/PhD Documents/Projects/river-profiles/Landsat/koyukuk_dem_5_clip_2.tif'

    main(B3input, B6input, DEM_name)
