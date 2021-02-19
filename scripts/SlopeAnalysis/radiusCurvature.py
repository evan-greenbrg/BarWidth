import itertools
import math

import gdal
import osr
import pandas
from scipy import optimize
from scipy import stats
from scipy import spatial
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Proj
from skimage import measure, draw, morphology, feature, graph


def sortCenterline(centerline):
    costs = np.where(centerline, 1, 1000)
    graph.route_through_array(
        centerline, 
        [0, 0], 
        [1, 1], 
        fully_connected=False
    )
    path, cost = graph.route_through_array(
        costs, 
        start=(int(riv_endpoints[0][1]), int(riv_endpoints[0][0])),
        end=(int(riv_endpoints[1][1]), int(riv_endpoints[1][0])),
        fully_connected=True
    )
    path = np.array(path)

def convert_bar_to_utm(myProj, bar_df):
    """
    Converts the coordinates in the bar .csv file from lat-lon to UTM

    Inputs -
    myProj: Proj string used to convert between projects
    bar_df: pandas dataframe of the bar upstream and downstream
        coordinates

    Outputs -
    bar_df: pandas dataframe of bar coords in lat-lon AND UTM
    """
    # Set up the column names and the dataframe
    columns = [
        'upstream_lat',
        'upstream_lon',
        'upstream_easting',
        'upstream_northing',
        'downstream_lat',
        'downstream_lon',
        'downstream_easting',
        'downstream_northing'
    ]
    coord_transform = pandas.DataFrame(
        columns=columns
    )

    # Iterate through Incoming bar df and convert
    for idx, row in bar_df.iterrows():
        us_east, us_north = myProj(
            row['Longitude_us'],
            row['Latitude_us']
        )
        ds_east, ds_north = myProj(
            row['Longitude_ds'],
            row['Latitude_ds']
        )

        # Set up append dataframe
        df = pandas.DataFrame(
            data=[[
                row['Latitude_us'],
                row['Longitude_us'],
                us_east,
                us_north,
                row['Latitude_ds'],
                row['Longitude_ds'],
                ds_east,
                ds_north
            ]],
            columns=columns
        )
        coord_transform = coord_transform.append(df)

    return coord_transform.reset_index(drop=True)


def get_bar_coordinates(centerline, bar):
    # Set up nearest neighbor search
    tree = spatial.KDTree(centerline[['x', 'y']])

    # Find the upstream index
    distance, upstream_n = tree.query(
        [(bar['upstream_easting'], bar['upstream_northing'])],
        1
    )
    # Find the downstream index
    distance, downstream_n = tree.query(
        [(bar['downstream_easting'], bar['downstream_northing'])],
        1
    )

    ns = [upstream_n[0], downstream_n[0]]
    print(ns)
    
    # Return the coordinates between
    return centerline[min(ns):max(ns)]


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



# Calculate distance
# Get Curvature
def addDistance(centerline):
    # Add distance
    distances = []
    distance = 0
    for idx, (x, y) in enumerate(zip(centerline['x'], centerline['y'])):
        if idx == 0:
            distances.append(0)
            continue
        distance += math.sqrt(
            (x - centerline['x'].iloc[idx-1])**2 
            + (y - centerline['y'].iloc[idx-1])**2
        )
        distances.append(distance)

    return distances

def getCurvature(centerline):

    Rs = []
    xtree = spatial.KDTree(centerline[['s', 'x']])
    ytree = spatial.KDTree(centerline[['s', 'y']])

    for idx, (s, x, y) in enumerate(zip(centerline['s'], centerline['x'], centerline['y'])):
        d, n = xtree.query([s, x], 5)

        ss = np.array(centerline['s'].iloc[n])
        xs = np.array(centerline['x'].iloc[n])
        ys = np.array(centerline['y'].iloc[n])

        # Fit Poly x
        np.warnings.filterwarnings('ignore')
        xz = np.polyfit(ss, xs, deg=3)
        px = np.poly1d(xz)
        xz1 = np.polyder(px, m=1)
        xz2 = np.polyder(px, m=2)

        # Fit Poly y
        yz = np.polyfit(ss, ys, deg=3)
        py = np.poly1d(yz)
        yz1 = np.polyder(py, m=1)
        yz2 = np.polyder(py, m=2)

        # Get derivatives
        dxds = xz1(s)
        d2xds2 = xz2(s)

        dyds = yz1(s)
        d2yds2 = yz2(s)

        # Get Curvature
        C = (
            ((dxds * d2yds2) - (dyds * d2xds2))
            / (((dxds**2) + (dyds**2))**(3/2))
        )

        Rs.append(np.abs(math.pi / C * 2))

    return Rs


def findCurvature(sections):
    # Initialize the curve_fit class
    comp_curv = ComputeCurvature()

    # Get x and y arrays
    xs = np.array(sections['x'])
    ys = np.array(sections['y'])

    if len(xs) == 0:
        return None

    # Get curvature
    curvature = comp_curv.fit(xs, ys)
        
#     # Plot the result
#     plt.plot(centerline['x'], centerline['y'])
#     theta_fit = np.linspace(-np.pi, np.pi, 180)
#     x_fit = comp_curv.xc + comp_curv.r*np.cos(theta_fit)
#     y_fit = comp_curv.yc + comp_curv.r*np.sin(theta_fit)
#     plt.plot(x_fit, y_fit, 'k--', label='fit', lw=2)
#     plt.plot(xs, ys, 'ro', label='data', ms=8, mec='b', mew=1)
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.title('curvature = {:.3e}'.format(curvature))
#     plt.show()

    return curvature 


def matchBar(river, df, bar_data_filt):
    # Set up search tree
    tree = spatial.KDTree(df[['Easting', 'Northing']])

    # Find curvature for each bar point
    curvatures = []
    Rs = []
    for idx, row in bar_data_filt.iterrows():
        distance, neighbors = tree.query(
            [(row['easting'], row['northing'])],
            1
        )
        curvatures.append(df.iloc[neighbors[0]]['Curvature'])
        Rs.append(df.iloc[neighbors[0]]['R'])

    # Set curvature in filtered datafrae
    bar_data_filt['curvature'] = curvatures
    bar_data_filt['R'] = Rs 

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
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Brazos_Near_Calvert/brazos_centerline.csv',
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Brazos_Near_Calvert/BrazosCalvert_26914.tif',
        ['lon', 'lat'],
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Brazos_Near_Calvert/brazos_bar_coords.csv'
    ),
    'Koyukuk River': (
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Koyukuk/koyukuk_manual_centerline_4326.csv',
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Koyukuk/Koyukuk_dem.tif_3995.tif',
        ['POINT_X', 'POINT_Y'],
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Koyukuk/koyukuk_bar_coords.csv'
    ),
    'Mississippi River': (
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Mississippi/mississippi_centerline_manual.csv',
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Mississippi/Mississippi_1_26915_meter.tif',
        ['y_lon', 'x_lat'],
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Mississippi/mississippi_bar_lats.csv'
    ),
    'Mississippi River - Leclair': (
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Mississippi_Leclair/miss_leclair_manual_centerline_4326.csv',
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Mississippi_Leclair/MississippiDEM_meter.tif',
        ['lon_', 'lat_'],
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Mississippi_Leclair/miss_leclair_bar_coords.csv'
    ),
    'Nestucca River': (
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Beaver_OR/nestucca_centerline_manual_4326.csv',
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Beaver_OR/beaver_26910_meter.tif',
        ['POINT_X', 'POINT_Y'],
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Beaver_OR/beaver_bar_coords.csv'
    ),
    'Powder River': (
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Powder_River/Powder_centerline_manual.csv',
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
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Sacramento/savramento_manual_centerline.csv',
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Sacramento/sacramento_merged_26910.tif',
        ['lon_', 'lat_'],
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Sacramento/sacramento_bar_coords.csv'
    ),
    'Tombigbee River': (
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Tombigbee/tombigbee_manual_centerline.csv',
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
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/White_river/white_river_centerline_manual.csv',
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/White_river/WhiteRiverDEM_32616_meter.tif',
        ['lon_', 'lat_'],
        '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/White_river/white_river_in_bar_coords.csv'
    )
}

data = {
    'river': [],
    'bar': [],
    'channel_width_mean': [],
    'curvature_circle': [],
    'curvature_fagherazzi': [],
}

for river, source in sources.items():
    print(source)

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

    centerline['s'] = addDistance(centerline)
    centerline['R1'] = getCurvature(centerline)

    # Get Bar df
    bar_df = pandas.read_csv(
        source[3],
        names=[
            'Latitude_us',
            'Longitude_us',
            'Latitude_ds',
            'Longitude_ds'
        ],
        header=1
    )
    bar_df = convert_bar_to_utm(myProj, bar_df)

    for idx, bar in bar_df.iterrows():
        sections = get_bar_coordinates(
            centerline,
            bar
        ).reset_index(drop=True)

        data['river'].append(river)
        data['bar'].append(idx)
        data['channel_width_mean'].append(B)
        data['curvature_circle'].append(findCurvature(sections))
        data['curvature_fagherazzi'].append(sections['R1'].median())

df = pandas.DataFrame(data)
df.to_csv('curvature.csv')

