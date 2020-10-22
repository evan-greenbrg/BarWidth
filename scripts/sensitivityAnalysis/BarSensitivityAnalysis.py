import argparse
import errno
import json
import os
import sys

import gdal
import osr
from scipy.optimize import curve_fit
import scipy.stats as stats
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas
from pyproj import Proj

from BarWidth import BarHandler
from BarWidth import RasterHandler


# Paths
xpaths = [
    '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Trinity/xsections.npy',
    '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Koyukuk/xsections.npy',
    '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Red_River/xsections.npy',
    '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/White_River/xsections.npy',
]

coordpath = [
    '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Trinity/coordinates.csv',
    '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Koyukuk/coordinates.csv',
    '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Red_River/coordinates.csv',
    '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/White_River/coordinates.csv',
]

BarPath = [
    '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Trinity/trinity_bar_coords.csv',
    '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Koyukuk/koyukuk_bar_coords.csv',
    '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Red_River/red_river_ar_bar_coords.csv',
    '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/White_river/white_river_in_bar_coords.csv',
]

DEMpath = [
    '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Trinity/Trinity_1m_be_clip.tif',
    '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Koyukuk/Koyukuk_dem.tif_3995.tif',
    '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Red_River/red_river_arkansas_mosaic-002-epsg26915.tif',
    '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/White_river/WhiteRiverDEM_32616.tif',
]

river = [
    'Trinity',
    'Koyukuk',
    'Red',
    'White'
]

sens_cdf = pandas.DataFrame(columns=['river', 'idx', '0.01', '0.10', 'B'])
parameters_df = pandas.DataFrame()
for path in range(0, len(xpaths)):
    # Load xsections and coordinates
    xsections = np.load(xpaths[path], allow_pickle=True)
    coordinates = pandas.read_csv(coordpath[path])

    # Get Proj String
    ds = gdal.Open(DEMpath[path], 0)
    ProjStr = "epsg:{0}".format(
        osr.SpatialReference(
            wkt=ds.GetProjection()
        ).GetAttrValue('AUTHORITY', 1)
    )
    myProj = Proj(ProjStr)

    # Initialize BarHandler
    rh = RasterHandler.RasterHandler()
    bh = BarHandler.BarHandler(
        xsections[0]['coords'][0],
        xsections[0]['coords'][1]
    )

    # Read in the bar file to find the channel bars
    print('Loading Bar .csv file')
    bar_df = pandas.read_csv(
        BarPath[path],
        names=['Latitude_us', 'Longitude_us', 'Latitude_ds', 'Longitude_ds'],
        header=1
    )

    # Convert the Bar Lat Long to UTM Easting Northing
    print('Converting Bar Coordinates to Easting Northing')
    bar_df = bh.convert_bar_to_utm(myProj, bar_df)

    # Find the bar coords within the DEM
    print('Find Bars within the DEM')
    bar_df = rh.coordinates_in_dem(
        bar_df, 
        ds, 
        ('upstream_easting', 'upstream_northing')
    )
    bar_df = rh.coordinates_in_dem(
        bar_df, 
        ds, 
        ('downstream_easting', 'downstream_northing')
    )
    ds = None

    # Make structure that contains the sections for each bar
    print('Making Bar section structure')
    bar_sections = {} 
    for idx, bar in bar_df.iterrows():
        sections = bh.get_bar_xsections(
            coordinates,
            xsections,
            bar_df.iloc[idx - 1]
        )
        bar_sections[str(idx)] = sections

    # Save the parameters
    riversl = []
    barsl = []
    idsl = []
    Ls = []
    X0s = []
    ks = []

    # Make structure that contains the sigmoids and only the bar side section
    print('Fitting sigmoid to channel bars')
    bar_widths = {}
    types = [
        ('location', 'object'),
        ('dem_width', 'f8'),
        ('water_width', 'f8'),
        ('sigmoid', 'object'),
        ('easting', 'object'),
        ('northing', 'object'),
        ('distance', 'object'),
        ('elevation', 'object'),
    ]
    for bar, sections in bar_sections.items():
        widths = np.array([], dtype=types)
        for idx, section in np.ndenumerate(sections):
            if (
                section['dem_width'] == 'nan' 
            ) or (
                not section['bank']
            ) or (
                section['water_width'] == 'nan'
            ):
                width = np.array(
                    tuple(
                        [
                            section[0],
                            section['dem_width'],
                            section['water_width'],
                            None,
                            section['elev_section']['easting'],
                            section['elev_section']['northing'],
                            section['elev_section']['distance'],
                            section['elev_section']['value_smooth']
                        ]
                    ),
                    dtype=widths.dtype
                )
            else:
                # Find the side of the channel with the bar
                banks = bh.find_bar_side(section['bank'])

                # Flip the cross-sections so that they are all facing the same way
                section, banks = bh.flip_bars(section, banks)

                # Find the distance for maximum slope and the maximum slope
                x0 , dydx = bh.find_maximum_slope(
                    section['elev_section'], 
                    banks
                )

                # Find the minimum and shift the cross-section
                section = bh.shift_cross_section_down(section, banks)

                # Fit sigmoid parameters
                popt = bh.fit_sigmoid_parameters(section, banks, x0, dydx)

                # Keep track of the parameters
                riversl.append(river[path])
                barsl.append(bar)
                idsl.append('{0}_{1}'.format(bar, idx[0]))
                Ls.append(popt[0])
                X0s.append(popt[1])
                ks.append(popt[2])

                # store the sigmoid parameters and the cross section
                width = np.array(
                    tuple(
                        [
                            section[0],
                            section['dem_width'],
                            section['water_width'],
                            popt,
                            section['elev_section']['easting'],
                            section['elev_section']['northing'],
                            section['elev_section']['distance'],
                            section['elev_section']['value_smooth']
                        ]
                    ),
                    dtype=widths.dtype
                )

            widths = np.append(widths, width)

        bar_widths[bar] = widths

    parameters_df = parameters_df.append(pandas.DataFrame(data={
        'river': riversl,
        'bar': barsl,
        'idx': idsl,
        'L': Ls,
        'X0': X0s,
        'k': ks
    }))

    # Find the width and height of the channel bars
    print('Finding clinoform width and height')
    columns = [
        'bar', 
        'idx', 
        'easting', 
        'northing', 
        'channel_width_dem', 
        'channel_width_water', 
        'bar_width',
        'bar_height'
    ]
    bar_data_df = pandas.DataFrame(columns=columns)
    n = 0

    # Sensitivity array
    sens_array = np.linspace(0.01, 0.10, 10) 

    widths = []
    senses = []
    bars = []
    idxs = []
    for bar, sections in bar_widths.items():
        print(bar)
        L_mean = np.median(
            [i['sigmoid'][0] for i in sections if i['sigmoid']]
        )
        # Initialize the sensitivity width structure
        for idx, section in np.ndenumerate(sections):
            bar_idx = '{0}_{1}'.format(bar, idx[0])

            # Don't track if there is no channel width
            if str(section['dem_width']) == 'nan':
                continue

            # Filter out the ill-fit sigmoid parameters
            elif (section['sigmoid'][0] / L_mean) < 0.01:
                continue

            else:
                # Get the bar width from the sigmoid
                for sens in sens_array:
                    bar_width, bar_height = bh.get_bar_geometry(
                        section['distance'],
                        section['sigmoid'],
                        sens=sens
                    )
                    # Build list for each sensitivity width
                    senses.append(sens)
                    bars.append(bar)
                    idxs.append(bar_idx)
                    if bar_width:
                        widths.append(bar_width)
                    else:
                        widths.append(False)

                    # I want to keep track of what I would use in the program
                    bar_width, bar_height = bh.get_bar_geometry(
                        section['distance'],
                        section['sigmoid']
                    )

                try: 
                    water_width = int(section['water_width'])
                except:
                    water_width = int(section['dem_width'])
                # Store data
                data = {
                    'bar': bar,
                    'idx': '{0}_{1}'.format(bar, idx[0]),
                    'easting': section['location'][0],
                    'northing': section['location'][1],
                    'channel_width_dem': int(section['dem_width']),
                    'channel_width_water': water_width,
                    'mean_width': (int(section['dem_width']) + water_width) / 2,
                    'bar_width_norm': bar_width
                }

                # Append to dataframe
                bar_data_df = bar_data_df.append(
                    pandas.DataFrame(data=data, index=[n])
                )
                n += 1

    # Set up DataFrame
    columns = ['bar', 'idx', 'sens', 'bar_width']
    sens_df = pandas.DataFrame(data={
        'bar': bars,
        'idx': idxs,
        'sens': senses,
        'bar_width': widths
    })

    i = 0
    for ids in bar_data_df.idx:
        y1 = sens_df[
            (sens_df.idx==ids)
            & (sens_df.sens==0.01)
        ]['bar_width']
        y10 = sens_df[
            (sens_df.idx==ids)
            & (sens_df.sens==0.10)
        ]['bar_width']
        sens_cdf = sens_cdf.append(pandas.DataFrame(data={
            'river': river[path],
            'idx': ids,
            '0.01': float(y1),
            '0.10': float(y10),
            'B': bar_data_df[bar_data_df.idx==ids]['mean_width'].iloc[0]
        }, index=[i]))
        i += 1

    sens_cdf['0.01_norm'] = sens_cdf['0.01'] / sens_cdf['B']
    sens_cdf['0.10_norm'] = sens_cdf['0.10'] / sens_cdf['B']
    sens_cdf['diff'] = (sens_cdf['0.01'] - sens_cdf['0.10']) / sens_cdf['B']


clean = sens_cdf[(sens_cdf['diff'] < 1) & (sens_cdf['diff'] > 0)]
# Histogram with just the differences
fig = plt.figure()
plt.boxplot(clean['diff'])
plt.xlabel('1% - 10% estimate / B')

op = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/figures/sensitivity.png'
fig.savefig(op, format='png')
plt.show()

# Parameters
river = [
    'Trinity',
    'Koyukuk',
    'Red',
    'White'
]

param_group = parameters_df.groupby('river')
param_group.boxplot()
plt.show()

param = parameters_df[(parameters_df['k'] < 1) & ( parameters_df['k'] > -1)]
trinity_k = param[param['river']=='Trinity']['k']
koyukuk_k = param[param['river']=='Koyukuk']['k']
red_k = param[param['river']=='Red']['k']
white_k = param[param['river']=='White']['k']
k_df = pandas.DataFrame(data={
    'Trinity': trinity_k,
    'Koyukuk': koyukuk_k,
    'Red': red_k,
    'white': white_k
})
k_df.boxplot()
plt.show()


sns.boxplot(
    data=param_group, 
)

# # Two Histograms comparing the two bar estimates
# # 0.01
# density_01 = stats.gaussian_kde(sens_cdf['0.01_norm'])
# n_01, x_01, _01 = plt.hist(sens_cdf['0.01_norm'], density=True, bins=20)
# # 0.10
# density_10 = stats.gaussian_kde(sens_cdf['0.10_norm'])
# n_10, x_10, _10 = plt.hist(sens_cdf['0.10_norm'], density=True, bins=20)
# plt.close()
# 
# plt.plot(x_01, density_01(x_01))
# plt.plot(x_10, density_10(x_10))
# plt.show()

# Potting
# for ids in bar_data_df.idx:
#     x = sens_df[
#         (sens_df.idx==ids) 
#         & (sens_df.sens<=0.1)
#         & (sens_df.sens>=0.01)
#     ]['sens']
#     y = sens_df[
#         (sens_df.idx==ids) 
#         & (sens_df.sens<=0.1)
#         & (sens_df.sens>=0.01)
#     ]['bar_width'] / bar_data_df[bar_data_df.idx==ids]['mean_width'].iloc[0]
#     xu = .057
#     yu = (
#         bar_data_df[bar_data_df.idx==ids]['bar_width_norm'].iloc[0] 
#         / bar_data_df[bar_data_df.idx==ids]['mean_width'].iloc[0] 
#     )
#     plt.plot(x, y)
#     plt.scatter(xu, yu)
# plt.show()
