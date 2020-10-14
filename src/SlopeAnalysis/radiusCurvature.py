import itertools

import gdal
import osr
import pandas
from scipy import optimize
from scipy import stats
from scipy import spatial
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Proj


class ComputeCurvature:
    def __init__(self):
        """ Initialize some variables """
        self.xc = 0  # X-coordinate of circle center
        self.yc = 0  # Y-coordinate of circle center
        self.r = 0   # Radius of the circle
        self.xx = np.array([])  # Data points
        self.yy = np.array([])  # Data points

    def calc_r(self, xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((self.xx-xc)**2 + (self.yy-yc)**2)

    def f(self, c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        ri = self.calc_r(*c)
        return ri - ri.mean()

    def df(self, c):
        """ Jacobian of f_2b
        The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
        xc, yc = c
        df_dc = np.empty((len(c), self.xx.size))

        ri = self.calc_r(xc, yc)
        df_dc[0] = (xc - self.xx)/ri                   # dR/dxc
        df_dc[1] = (yc - self.yy)/ri                   # dR/dyc
        df_dc = df_dc - df_dc.mean(axis=1)[:, np.newaxis]
        return df_dc

    def fit(self, xx, yy):
        self.xx = xx
        self.yy = yy
        center_estimate = np.r_[np.mean(xx), np.mean(yy)]
        center = optimize.leastsq(self.f, center_estimate, Dfun=self.df, col_deriv=True)[0]

        self.xc, self.yc = center
        ri = self.calc_r(*center)
        self.r = ri.mean()

        return self.r  # Return the curvature


def window(seq, n=5):
    """
    Returns sliding window of n
    """
    it = iter(seq)
    result = tuple(itertools.islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def closest(lst, K):
    """
    Finds the closest value in list to value, K
    """
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i]-K))]


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

def convertCoordinates(centerline, cols):
    easts = []
    norths = []
    for idx, row in centerline.iterrows():
        east, north = myProj(row[cols[0]], row[cols[1]])
        easts.append(east)
        norths.append(north)

    return easts, norths


def findCurvature(centerline, B):
    # Initialize the curve_fit class
    comp_curv = ComputeCurvature()

    # Initialize the search tree to find nearest points
    tree = spatial.KDTree(centerline[['x', 'y']])

    data = {
        'Northing': [],
        'Easting': [],
        'Curvature': []
    }
    for idx, row in centerline.iterrows():
        # Find the points within 1 channel width
        distance, neighbors = tree.query(
            [(row['x'], row['y'])],
            50
        )
        n_i = neighbors[distance < B]

        if len(n_i) < 2:
            continue

        # Get x and y arrays
        x = centerline.iloc[n_i]['x']
        y = centerline.iloc[n_i]['y']

        # Get curvature
        curvature = comp_curv.fit(x, y)
        
        # Append to data store
        data['Northing'].append(row['y'])
        data['Easting'].append(row['x'])
        data['Curvature'].append(curvature)

    return pandas.DataFrame(data)
        

    i = round(n / 2) - 1

    xs = list(window(centerline['x'], n))
    ys = list(window(centerline['y'], n))
    pairs = [z for z in zip(xs, ys)]
    data = {
        'Northing': [],
        'Easting': [],
        'Curvature': []
    }
    for pair in pairs:
        x = np.array(pair[0])
        y = np.array(pair[1])
        curvature = comp_curv.fit(x, y)

        data['Northing'].append(pair[1][i])
        data['Easting'].append(pair[0][i])
        data['Curvature'].append(curvature)

        # Plot the result
#        theta_fit = np.linspace(-np.pi, np.pi, 180)
#        x_fit = comp_curv.xc + comp_curv.r*np.cos(theta_fit)
#        y_fit = comp_curv.yc + comp_curv.r*np.sin(theta_fit)
#        plt.plot(x_fit, y_fit, 'k--', label='fit', lw=2)
#        plt.plot(x, y, 'ro', label='data', ms=8, mec='b', mew=1)
#        plt.xlabel('x')
#        plt.ylabel('y')
#        plt.title('curvature = {:.3e}'.format(curvature))
#        plt.show()

    return pandas.DataFrame(data)


def matchBar(river, df, bar_data_filt):
    # Set up search tree
    tree = spatial.KDTree(df[['Easting', 'Northing']])

    # Find curvature for each bar point
    curvatures = []
    for idx, row in bar_data_filt.iterrows():
        distance, neighbors = tree.query(
            [(row['easting'], row['northing'])],
            1
        )
        curvatures.append(df.iloc[neighbors[0]]['Curvature'])

    # Set curvature in filtered datafrae
    bar_data_filt['curvature'] = curvatures

    return bar_data_filt


def removeOutliers(bar_data_filt, thresh=3):
    zs = stats.zscore(bar_data_filt['curvature'])
    abs_zs = np.abs(zs)
    filt = (abs_zs < thresh)

    return bar_data_filt[filt]

# Total bar data
total_data = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/sampled_total_data.csv'
bar_data = pandas.read_csv(total_data)
columns = [
    'bar',
    'idx',
    'easting',
    'northing',
    'channel_width_mean',
    'bar_width',
    'river'
]
bar_data = bar_data[columns]
rivers = list(bar_data['river'].unique())

# Get all the sources
sources = {
    'Brazos River': (
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Brazos/brazos_centerline_4326.csv',
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Brazos/brazos_26914_clip.tif',
        ['lon_', 'lat_'],
    ),
    'Koyukuk River': (
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Koyukuk/koyukuk_centerlines.txt',
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Koyukuk/Koyukuk_dem.tif_3995.tif',
        ['POINT_X', 'POINT_Y'],
    ),
    'Mississippi River': (
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Mississippi/mississippi_centerline_lats.csv',
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Mississippi/Mississippi_1_26915_meter.tif',
        ['y_lon', 'x_lat'],
    ),
    'Mississippi River - Leclair': (
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Mississippi_Leclair/miss_leclair_centerline_4326.csv',
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Mississippi_Leclair/MississippiDEM_meter.tif',
        ['lon_', 'lat_'],
    ),
    'Nestucca River': (
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Beaver_OR/beaver_centerline_4326.csv',
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Beaver_OR/beaver_26910_meter.tif',
        ['POINT_X', 'POINT_Y'],
    ),
    'Powder River': (
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Powder_River/powder_centerline_dense_4326.csv',
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Powder_River/output_be_26913.tif',
        ['lon_', 'lat_'],
    ),
    'Red River': (
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Red_River/red_river_ar_centerline.csv',
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Red_River/red_river_dem_meter.tif',
        ['POINT_X', 'POINT_Y'],
    ),
    'Rio Grande River': (
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Rio_Grande_TX/RioGrandeCenterlineLats.csv',
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Rio_Grande_TX/RioGrandeTxDEM.tif',
        ['lon_', 'lat_'],
    ),
    'Sacramento River': (
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Sacramento/savramento_centerline_4326.csv',
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Sacramento/sacramento_merged_26910.tif',
        ['lon_', 'lat_'],
    ),
    'Tombigbee River': (
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Tombigbee/tombigbee_centerline_4326.csv',
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Tombigbee/tombigbee_26916_10m_clip.tif',
        ['lon_', 'lat_'],
    ),
    'Trinity River': (
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Trinity/Trinity_Centerline.txt',
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Trinity/Trinity_1m_be_clip.tif',
        ['POINT_X', 'POINT_Y'],
    ),
    'White River': (
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/White_river/white_river_centerline_4326.csv',
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/White_river/WhiteRiverDEM_32616_meter.tif',
        ['lon_', 'lat_'],
    )
}

new_bar_data = pandas.DataFrame()
for river, source in sources.items():

    # Get stats for river
    bar_data_filt = bar_data[bar_data['river'] == river]
    B = bar_data_filt['channel_width_mean'].mean()

    # Get ESPG
    myProj = getProj(source[1])

    # Load in the centerline
    centerline = pandas.read_csv(source[0])

    # Convert to UTM
    centerline['x'], centerline['y'] = convertCoordinates(
        centerline, 
        source[2]
    )

    curvature_df = findCurvature(centerline, B) 

    # Match each river bar to a curvature
    bar_data_filt = matchBar(river, curvature_df, bar_data_filt)

    # Remove Outliers
    bar_data_filt = removeOutliers(bar_data_filt)

    # Stack the data
    new_bar_data = pandas.concat(
        [new_bar_data, bar_data_filt], 
        ignore_index=True
    )

#plt.scatter(new_bar_data['channel_width_mean'], new_bar_data['curvature'])
#plt.yscale('log')
#plt.show()

bar_data_group = new_bar_data.groupby(['river', 'bar']).median()
bar_data_group = bar_data_group.reset_index(drop=False)

bar_data_group.to_csv('curvature.csv')
print(bar_data_group)

# EXTRA
