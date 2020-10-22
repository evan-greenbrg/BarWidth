import argparse
import errno
import os
from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import gdal
import osr
import numpy as np
import pandas
from pyproj import Proj
from matplotlib import pyplot as plt
from matplotlib.widgets import Button

from BarHandler import BarHandler
from PointPicker import BarPicker


def closest(lst, K):
    """
    Finds the closest value in list to value, K
    """
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i]-K))]


def sigmoid_fun(x, L, x0, k):
    y = L / (1 + np.exp(-k*(x-x0)))
    return (y)


def mannual_fit_bar(x, y):
    """
    Mannually picks the bar points
    """

    fig, ax = plt.subplots(1, 1)
    line, = ax.plot(x, y, linewidth=3)
    BC = BarPicker(ax, x, y)

    fig.canvas.mpl_connect('button_press_event', BC)
    line.set_picker(1)

    axclear = plt.axes([0.81, 0.17, 0.1, 0.055])
    bclear = Button(axclear, 'Clear')
    bclear.on_clicked(BC.clear)

    axnext = plt.axes([0.81, 0.1, 0.1, 0.055])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(BC.next)

    axskip = plt.axes([0.81, 0.03, 0.1, 0.055])
    bskip = Button(axskip, 'Skip')
    bskip.on_clicked(BC.skip)

    plt.show()

    return BC.popt


def get_bar_geometry(p, sigmoid, sens=.057):
    """
    From the given sigmoid parameters finds the bar width.
    Uses X% of the asymptote cutoff value to find the width end points
    Will also return the height

    Inputs -
    p: Array of channel distances to fit the sigmoid to
    sigmoid: sigmoir parameters

    Outputs -
    bar_width: width of the fit clinoform
    bar_height: height of the fit clinoform
    """
    # Set up L
    L = sigmoid[0]
    # Set up x array
    pos_x = np.linspace(
        min(p),
        max(p),
        len(p)
    )

    # Generte dataframe and create the sigmoid values
    df = pandas.DataFrame(data={
        'distance': np.round(pos_x, 0),
        'elevation': sigmoid_fun(pos_x, *sigmoid)
    })

    # Find the middle point (x0) will use this to iterate both directions
    close_mid = closest(df['distance'], sigmoid[1])
    middle_point = np.where(df['distance'] == close_mid)[0]

    # Get the top of the clinoform
    top_df = df.iloc[middle_point[0]:].reset_index(drop=True)
    top_df['diff'] = abs((L - top_df['elevation']) / L)
    if max(top_df['diff']) > 0.98:
        try:
            top = top_df[top_df['diff'] >= (1 - sens)].iloc[0]
        except IndexError:
            return None, None
    else:
        try:
            top = top_df[top_df['diff'] <= sens].iloc[0]
        except IndexError:
            return None, None

    # Get the bottom of the clinoform
    bot_df = df.iloc[:middle_point[0]].reset_index(drop=True).iloc[::-1]
    bot_df['diff'] = abs(L - bot_df['elevation']) / L
    if max(bot_df['diff']) > 0.98:
        bot = bot_df[bot_df['diff'] >= (1 - sens)].iloc[0]
    else:
        bot = bot_df[bot_df['diff'] <= sens].iloc[0]

    # Calculate geometry
    if len(top) > 0 and len(bot) > 0:
        width = float(top['distance']) - float(bot['distance'])
        height = float(top['elevation']) - float(bot['elevation'])
    else:
        width = False
        height = False

    return width, height, top, bot


csv_root = '/home/greenberg/Code/Github/river-profiles/src/ancientBars'

beaufort = os.path.join(csv_root, 'Beaufort.csv')
joggins = os.path.join(csv_root, 'joggins.csv')
scalby = os.path.join(csv_root, 'scalby.csv')


dfs = {
    'beafort': pandas.read_csv(beaufort),
    'joggins': pandas.read_csv(joggins),
    'scalby':  pandas.read_csv(scalby),
}

for key, value in dfs.items():
    print(key)
    popt = mannual_fit_bar(value['X_m'], value['Y_m'])
    width, height, top, bot = get_bar_geometry(value['X_m'], popt)

    sigmoid_y = sigmoid_fun(value['X_m'], *popt)

    plt.plot(value['X_m'], value['Y_m'])
    plt.plot(value['X_m'], sigmoid_y)
    plt.scatter(top['distance'], top['elevation'])
    plt.scatter(bot['distance'], bot['elevation'])
    plt.show()
    
