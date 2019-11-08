from osgeo import gdal
import ogr
import osr


class RasterHandler():

    def __init__(self):
        pass

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

