import statistics
import math

import numpy as np
import pandas
from pyproj import Proj
from scipy import spatial 
from scipy.optimize import curve_fit
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.widgets import Button

from PointPicker import BarPicker


def closest(lst, K): 

    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))] 


class BarHandler():

    def __init__(self, x0, y0):
        self.slope_threshold = 1
        self.slope_smooth = 10
        self.x0 = x0
        self.y0 = y0

    def get_bar_xsections(self, coordinates, xsections, bar):
        tree = spatial.KDTree(coordinates[['easting', 'northing']])
        distance, upstream_n = tree.query(
            [(bar['upstream_easting'], bar['upstream_northing'])],
            1
        )
        distance, downstream_n = tree.query(
            [(bar['downstream_easting'], bar['downstream_northing'])],
            1
        )
        
        return xsections[upstream_n[0]:downstream_n[0]]

    def convert_bar_to_utm(self, myProj, bar_df):
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
        columns=[
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

    def find_bar_side(self, banks):
        """
        Similar implementation as the bar width, 
        but I'm just using it to find the bar side

        Inputs -
        banks: list of tuples of the channel bank pairs (max / min position)
            on the channel bank.
        xsection:  structure that includes distance and dem value
        """
        # Return if there are no banks
        if not banks:
            return False

        # Find the distance between channel bank points. Bar should have min
        distance0 = abs(banks[0][0] - banks[0][1])
        distance1 = abs(banks[1][0] - banks[1][1])

        # creates tuple from the banks list for deciding the bank side
        if distance1 > distance0:
            return (banks[1][1], banks[1][0])
        else: 
            return (banks[0][0], banks[0][1])

    def flip_bars(self, section, banks):
        """
        Flips the bar cross-sections so that they bar-side is always oriented in the same way.
        The bar side will be such that the heighest bar point will be at
        a greater distance than the lowest bar point (I.E. sloped upwards)

        Inputs - 
        section: the bar section structure that is being worked
        banks: the tuple of the banks positions
        """
        # Find closes values
        e0_close = closest(section['elev_section']['distance'], banks[0])
        e1_close = closest(section['elev_section']['distance'], banks[1])

        # Find the banks elevations for the direction of the section
        e0 = section['elev_section'][
                section['elev_section']['distance'] == e0_close 
        ]['value_smooth'][0]
        
        e1 = section['elev_section'][
                section['elev_section']['distance'] == e1_close 
        ]['value_smooth'][0]

        # e0 -earlier of the two bar points
        # e1 is the latter of the two bar points
        if e1 > e0:
            section['elev_section']['distance'] = np.flip(
                section['elev_section']['distance'], 
                0
            )
            banks = (banks[0] * -1, banks[1] * -1)

        return section, banks

    def find_maximum_slope(self, section, banks, value='value_smooth', step=5):
        """
        Finds the index of the maximum slope on the bar side
        Finds the local slopes over some step length
        
        Inputs - 
        section: The data structure that has distances and dem values (Table)
        banks: The tuple with the distances of the banks
        value (optional): the name of the value column in the section struct
        ste (optional): How many points to smooth over to find the slope
        """

        # Find clsoes banks
        bank0_closest = closest(section['distance'], banks[0])
        bank1_closest = closest(section['distance'], banks[1])
        # Find the section indexes of the bar bank
        banks_idx = [
            i 
            for i, val 
            in enumerate(section['distance'])
            if (val == bank0_closest) or (val == bank1_closest)
        ]

        # Filter the section into just the bar to find the slope
        banks_section = section[min(banks_idx):max(banks_idx)][::step]

        # Find all of the slopes on the bar bank
        ydiffs = np.diff(banks_section[value])
        xdiffs = np.diff(banks_section['distance'])

        # Find the index of the maximum bar slope
        idx = [
            i 
            for i, val 
            in enumerate(abs(ydiffs))
            if (val == max(abs(ydiffs)))
        ]

        # Find the slope
        dydx = (ydiffs[idx] / xdiffs[idx])[0]

        # Find the corresponding index in the entire section structure
        max_idx = [
            i 
            for i, val 
            in enumerate(section['distance']) 
            if (round(val, 0) == round(banks_section[idx[0] - 1]['distance'], 0))
        ]

        return section[max_idx[0]]['distance'], dydx 

    def shift_cross_section_down(self, section):
        """
        Shifts the cross section down to the minimum channel position
        This makes it easy to fit the sigmoid
        
        Inputs = 
        section: Numpy structure containing the cross section data
        banks: tuple with the distance positions of the two banks points
        """
        # Get minimnum elevation on the banks
        minimum = min(section['elev_section']['value_smooth'])

        # Shift the cross section
        section['elev_section']['value_smooth'] = (
            section['elev_section']['value_smooth'] - minimum
        )

        return section

    def fit_sigmoid_parameters(self, section, banks, x0, dydx):
        """
        Fits the paramters values for the sigmoid function. 
        This includes:
            L - the sigmoid top asymptote
            x0 - the x value where maximum slope is
            k - the growth rate

        Inputs -
        section: Numpy structure contianing the cross-section data
        x0: the distance position where the maximum slope is
        dydx: the bar's maximum slope
        """

        # Find closest banks
        min_closest = closest(section['elev_section']['distance'], min(banks))
        max_closest = closest(section['elev_section']['distance'], max(banks))
        # Find the banks positions in the structure
        bar_section = section['elev_section'][
            (section['elev_section']['distance'] == min_closest)
            | (section['elev_section']['distance'] == max_closest)
        ]

        # Get maximum elevation on the banks
        L = max(section['elev_section']['value_smooth'])

        # Solve for growth rate, k
        k = (4 * dydx) / L

        return [L, x0, k] 
    
    def find_bar_width(self, banks):
        """
        Simplest approach I can implement. Use the extrema found from the
        Width finder. Pick the side that has the greatest difference between
        the bank maxima and minima
        """
        if not banks:
            return False, False

        distance0 = abs(banks[0][0] - banks[0][1])
        distance1 = abs(banks[1][0] - banks[1][1])

        if distance0 > distance1:
            return distance0, (banks[0][0], banks[0][1])
        elif distance1 > distance0:
            return distance1, (banks[1][0], banks[1][1])


    def find_bar_side1(self, bar_section, dem_col):
        """
        Could implement automated method to find the side of the bar
        Going to hold off on this for now, adding it in the input DF
        """

        # Initialize samples across bar
        bar_len = len(bar_sections)
        quarter_len = int(bar_len/4)
        side_test = [quarter_len, quarter_len * 2, quarter_len * 3]

        zero_bank = 0
        one_bank = 0
        for idx in side_test:
            # Get the distances and dem value
            t = test_bar[idx]['elev_section'][dem_col]
            p = test_bar[idx]['elev_section']['distance']
            endpoints = test_bar[idx]['ch_endpoints']

            # Find the indexes of the endpoints in the p, t vector
            i0 = [i for i, x in enumerate(p) if x == endpoints[0]]
            i1 = [i for i, x in enumerate(p) if x == endpoints[1]]

            # Define the endpoints
            endpoint0 = (endpoint[0], t[i0][0])
            endpoint1 = (endpoint[1], t[i1][0])

            # Define the index vector for the two xsection directions
            i_ = [i0[0], i1[0]]
            direction0 = [*range(min(i_), max(i_), 1)]
            direction1 = np.flip(direction0, 0)

            # Find the mean slope of each channel bank
            mean0, side0 = self.descend_bank(p, t, direction0)
            mean1, side1 = self.descend_bank(p, t, direction1)

            # Make the vote
            if mean0 > mean1:
                zero_bank += 1
            elif mean1 > mean0:
                one_bank +=1

        # Return the results
        if zero_bank > one_bank:
            print('zero')
            return side0
        elif one_bank > zero_bank:
            print('one')
            return side1
        else:
            return None

    def descend_bank(self, p, t, direction):
        """
        Descends the bank to find the bar length
        """
        thresh = max(t) / (max(p) * self.slope_threshold)
        slopes = []
        for i, idx in enumerate(direction):
            if i < self.slope_smooth:
                slope = (
                    (t[idx + self.slope_smooth] - t[idx]) 
                    / (p[idx + self.slope_smooth] - p[idx])
                )
                slopes.append(slope)
            elif i > len(direction0) - self.slope_smooth:
                slope = (
                    (t[idx] - t[idx - self.slope_smooth]) 
                    / (p[idx] - p[idx - self.slope_smooth])
                )
                slopes.append(slope)
            else:
                slope = (
                    (t[idx + self.slope_smooth] - t[idx - self.slope_smooth]) 
                    / (p[idx + self.slope_smooth] - p[idx - self.slope_smooth])
                )
                slopes.append(slope)

            if i == 0:
                continue

            elif slopes[i] < 0 and (abs(slopes[i-1]) - abs(slopes[i]) < thresh):
                slope_mean = statistics.mean(slopes)
                bar_points = (p[direction[0]], p[idx])

                return (
                    statistics.mean([abs(i) for i in slopes]), 
                    (int(p[0]), int(p[idx]))
                )

            else: 
                diff = abs(slopes[i-1] - slopes[i])
                continue

    def get_downstream_distance(self, bars):
        """
        Take UTM coordinates from bars dictionary and 
        converts to downstream distance
        """
        for key in bars.keys():
            distance = []
            for idx, coor in enumerate(bars[key]['coords']):
                length = (
                    ((coor[0] - self.x0)**2)
                    + ((coor[1] - self.y0)**2)
                )**(1/2)
                distance.append(length)
            bars[key]['distance'] = distance

        return bars

    def get_bar_geometry(self, p, sigmoid, sens=.157):
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
        # Set up sigmoid function
        def sigmoid_fun(x, L ,x0, k):
            y = L / (1 + np.exp(-k*(x-x0)))
            return (y)

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
        top_df['diff'] = abs(L - top_df['elevation']) / L
        try:
            top = top_df[top_df['diff'] <= sens].iloc[0]
        except:
            top = []

        # Get the bottom of the clinoform 
        bot_df = df.iloc[:middle_point[0]].reset_index(drop=True).iloc[::-1]
        bot_df['diff'] = abs(0 - bot_df['elevation']) / L
        try:
            bot = bot_df[bot_df['diff'] <= sens].iloc[0]
        except:
            bot = []

        # Calculate geometry
        if len(top) > 0 and len(bot) > 0:
            width = float(top['distance']) - float(bot['distance'])
            height = float(top['elevation']) - float(bot['elevation'])
        else:
            width = False
            height = False

        return width, height

    def fit_sigmoid(self, section, banks):
        """
        Fits the sigmoid to the bar
        section: numpy structure of the channel cross-section
        distance: name of the distance column to use in the section struct
        value: name of the value column to use in the section struct

        This is an old method that I'm not using
        """

        def sigmoid(x, L ,x0, k, s):
            y = L / (1 + np.exp(-k*(x-x0))) + s
            return (y)
        
        bar_section = section['elev_section'][
            (
                section['elev_section']['distance'] 
                >= section['elev_section']['distance'][
                    section['elev_section']['distance'] == min(banks)
                ]
            )
            & (
                section['elev_section']['distance'] 
                <= section['elev_section']['distance'][
                    section['elev_section']['distance'] == max(banks)
                ]
            )
        ]
        
        p0 = [
            max(bar_section['value_smooth']), 
            np.median(bar_section['distance']),
            -.09,
            1
        ] 
        popt, pcov = curve_fit(
        	sigmoid, 
        	bar_section['distance'], 
        	bar_section['value_smooth'], 
        	p0, 
        	method='dogbox', 
        	maxfev=100000,
        )
        
        return popt

    def get_r_squared(self, section, banks, popt):
        """
        Calculate the R-Squared from the estimated bar sigmoid and the
        bar section
        """

        def sigmoid(x, L ,x0, k):
            y = L / (1 + np.exp(-k*(x-x0)))
            return (y)

        # Get just the bar section
        bar_section = section['elev_section'][
            (section['elev_section']['distance'] >= min(banks))
            & (section['elev_section']['distance'] <= max(banks))
        ]

        # If there is no bar section -> exit
        if len(bar_section) == 0:
            return 0

        # Get the x, y, f vectors
        x = bar_section['distance']
        y = bar_section['value_smooth']
        f = sigmoid(x, *popt)
        # mean
        ymean = np.mean(y)
        # SSres
        ss_res = sum([(y[i] - f[i])**2 for i in range(0, len(x))])
        # SStot
        ss_tot = sum([(y[i] - ymean)**2 for i in range(0, len(x))])

        # Return R-squared
        return 1 - (ss_res / ss_tot)

    def interpolate_down(self, depth, section):
        """
        Draw the shifted channel depth from a lognormal distribution.
        The inputs are the depth of the channel from gauge and a 
        coefficient of variation.
        """
        if not depth:
            return 0

        # Simplify the section
        elev = np.copy(section['elev_section'])

        # Get banks
        banks = np.copy(section['bank'])

        # Get banks-closest
        bank0_closest = closest(elev['distance'], banks[0][0])
        bank1_closest = closest(elev['distance'], banks[1][0])

        # Get index of banks points
        banks_idx = [
            np.where(elev['distance'] == bank0_closest)[0][0],
            np.where(elev['distance'] == bank1_closest)[0][0]
        ]

        # Get elevation of channel top
        channel_top = float(max(elev['value_smooth'][tuple([banks_idx])]))

        # Find all of the slopes within the channel
        channel = np.copy(elev[min(banks_idx):max(banks_idx)])
        ydiffs = np.diff(channel['value_smooth'])

        # Find max and minimum slope - with centered difference
        maxi = np.where(ydiffs == max(ydiffs))[0][0]
        if maxi < 10:
            maxslope = ydiffs[maxi]
        else:
            maxslope = np.mean(ydiffs[maxi-10:maxi+10])

        # Find the minimum slope - with centered difference
        mini = np.where(ydiffs == min(ydiffs))[0][0]
        if mini < 10:
            minslope = ydiffs[mini]
        else:
            minslope = np.mean(ydiffs[mini-10:mini+10])

        # Create a new interpreted elevations array
        interp_channel = np.copy(channel)

        # Decend each bank - Max
        interp_depth = 0
        channel_bot = interp_channel['value_smooth'][maxi]
        i = maxi
        m = abs(interp_channel['distance'][1] - interp_channel['distance'][0])
        interpolate = True
        # Need to handle if the depth is greater than the expected
        if channel_top - channel_bot > depth:
            interpolate = False
            pass
        else:
            while (
                interp_depth < depth 
                and abs(i) < len(channel['value_smooth']) - 1
            ):
                interp_depth = channel_top - channel_bot
                i -= 1
                channel_bot = channel_bot - (m * maxslope)
                if channel_top - channel_bot > depth:
                    channel_bot = channel_top - depth
                interp_channel['value_smooth'][i] = channel_bot

            interp_maxi = i
            channel_bot_max = channel_bot

        # Decend each bank - Min
        interp_depth = 0
        channel_bot = interp_channel['value_smooth'][mini]
        i = mini

        # Handle if depth is greater than the expected
        if channel_top - channel_bot > depth:
            interpolate = False
            pass
        else:
            while (
                interp_depth < depth
                and abs(i) < len(interp_channel['value_smooth']) - 1
            ):
                interp_depth = channel_top - channel_bot
                i += 1
                channel_bot = channel_bot + (m * minslope)
                if channel_top - channel_bot > depth:
                    channel_bot = channel_top - depth
                interp_channel['value_smooth'][i] = channel_bot

            interp_mini = i
            channel_bot_min = channel_bot

        # Set the value between the two edges
        if interpolate:
            interpi = [interp_mini, interp_maxi]
            interp_channel['value_smooth'][min(interpi):max(interpi)] = min(
                [channel_bot_max, channel_bot_min]
            )

        # Insert channel values into the whole section
        for idx, point in enumerate(interp_channel['distance']):
            close = closest(section['elev_section']['distance'], point) 
            i = np.where(section['elev_section']['distance'] == close)[0][0]
            section['elev_section'][i]['value_smooth'] = interp_channel[
                'value_smooth'
            ][idx]

        return section

    def mannual_fit_bar(self, section):
        """
        Mannually picks the bar points
        """
        x = section['elev_section']['distance']
        y = section['elev_section']['value_smooth']
        
        fig, ax = plt.subplots(1, 1)
        line, = ax.plot(x, y, linewidth=3)
        BC = BarPicker(ax, x, y)

        fig.canvas.mpl_connect('pick_event', BC)
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

        print(BC.popt)
        print(BC.rsquared)
        return BC.popt, BC.rsquared
