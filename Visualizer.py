import matplotlib.pyplot as plt
import numpy as np
import pandas

class Visualizer():

    def __init__(self, x0, y0):
        self.x0 = x0
        self.y0 = y0

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



    def plot_downstream_bars(self, bars, path):
        """
        Generates plot of all bar widths as ratio with channel width
        going downstream
        """
        bars = self.get_downstream_distance(bars)

        max0 = 0
        fig = plt.figure(figsize = (11, 7))
        for key in bars.keys():
            plt.scatter(bars[key]['distance'], bars[key]['ratio'])
            maxd = max(bars[key]['distance'])
            if maxd > max0:
                max0 = maxd

        line_pl = [i for i in range(0, int(max0), 100)]
        line_val = [1.5 for i in line_pl]
        plt.plot(line_pl, line_val)
        plt.xlabel('downstream distance')
        plt.ylabel('channel width/bar width')
        plt.savefig(path)

    def plot_widths(self, bars, path):
        """
        Generates plot of all bar widths with respect to channel width
        """

        width_df = pandas.DataFrame(columns=['channel_width', 'bar_width'])
        for key in bars.keys():
            print(key)
            data = {
                'channel_width': bars[key]['channel_width'],
                'bar_width': bars[key]['bar_width']
            }
            df = pandas.DataFrame(data=data)
            width_df = width_df.append(df)

        width_df = width_df.reset_index(drop=True)
        width_df = width_df.dropna(axis=0, how='any')

        x = np.linspace(0, max(width_df['bar_width']) ,100)
        y = 1.5*x

        plt.scatter(width_df['bar_width'], width_df['channel_width'])
        plt.xlabel('Bar Width (m)')
        plt.ylabel('Channel Width (m)')
        plt.savefig(path)


