from osgeo import gdal
import ogr
import osr
import rasterio
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
    
    def get_pixel_size(self, raster_path):
        raster =  rasterio.open(raster_path)
        gt = raster.transform
        pixelSizeX = gt[0]
        pixelSizeY =-gt[4]

        return pixelSizeX, pixelSizeY

    def get_dem_pixels(self, east, north, xOrigin, yOrigin, 
                    pixelWidth, pixelHeight):

        dem_col = int((east - xOrigin) / pixelWidth)
        dem_row = int((yOrigin - north ) / pixelHeight)

        return dem_col, dem_row

    def get_coords_by_step(self, easting, northing, dlon_inv, dlat_inv,
                           xstep, ystep, i, sign=1):

        dd = dlat_inv * xstep * i
        east = easting + (dd * sign)

        dd = dlon_inv * ystep * i
        north = northing + (dd * sign)

        return east, north

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
            ('position', 'i4'),
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
            eastd, northd = self.get_coords_by_step(
                row['easting'],
                row['northing'],
                row['dlon_inv'],
                row['dlat_inv'],
                xstep,
                ystep,
                i,
                sign=1
            )
            eastu, northu = self.get_coords_by_step(
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

            d_pos = i
            u_pos = i * -1
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
