from osgeo import gdal
import ogr
import osr
import numpy as np


class RasterHandler():

    def __init__(self):
        pass

    def get_indices(self, x, y, ox, oy, pw, ph):
        """
        Gets the row (i) and column (j) indices in an array for a given set of coordinates.
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
        xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()
        # calculate indices and index array
        indices = self.get_indices(
            coordinates['lat'].to_numpy(),
            coordinates['lon'].to_numpy(),
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
        inSpatialRef.ImportFromEPSG(iEPSG)

        outSpatialRef = osr.SpatialReference()
        outSpatialRef.ImportFromEPSG(oEPSG)

        coordTransform = osr.CoordinateTransformation(
            inSpatialRef, 
            outSpatialRef
        )

        # transform point
        point.Transform(coordTransform)

        return point.GetX(), point.GetY()

    def clip_raster(self, ipath, opath, minx, maxy, maxx, miny):
        """
        Clips raster file based on bounding box coordinates
        """
        ds = gdal.Open(ipath)
        ds = gdal.Translate(
            opath, 
            ds, 
            projWin = [minx, maxy, maxx, miny]
        )
        ds = None

    def rename_path(self, input_path):
        path_sp = input_path.split('/')
        name_li = path_sp[-1].split('.')
        name_temp = name_li[0] + '_clip'
        name_li[0] = name_temp
        name = '.'.join(name_li)
        path_sp[-1] = name

        return '/'.join(path_sp)


def main(B3input, B6input, DEM_name, demEPSG, landsatEPSG):
    rh = RasterHandler()
    ds = gdal.Open(DEM_name)

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

    B3out_path = rh.rename_path(B3input)
    rh.clip_raster(B3input, B3out_path, minx, maxy, maxx, miny)
    B6out_path = rh.rename_path(B6input)
    rh.clip_raster(B6input, B6out_path, minx, maxy, maxx, miny)


if __name__ == "__main__":
    B3input = '/Users/evangreenberg/PhD Documents/Projects/river-profiles/Landsat/LC08_L1TP_025039_20190514_20190521_01_T1_B3.TIF'
    B6input = '/Users/evangreenberg/PhD Documents/Projects/river-profiles/Landsat/LC08_L1TP_025039_20190514_20190521_01_T1_B6.TIF'
    DEM_name = '/Users/evangreenberg/PhD Documents/Projects/river-profiles/DEM/output_be.tif'

    inEPSG = 4269
    outEPSG = 4326

    main(B3input, B6input, DEM_name, inEPSG, outEPSG)
