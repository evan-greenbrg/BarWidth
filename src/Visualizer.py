import matplotlib.pyplot as plt
import numpy as np
import pandas
from scipy.stats import gaussian_kde
import seaborn as sns

class Visualizer():

    def __init__(self):
        pass

    def get_downstream_distance(self, bargroup):
        """
        Take UTM coordinates from bars dictionary and 
        converts to downstream distance
        """
        df = pandas.DataFrame()
        for name, group in bargroup:
            distance = []
            i = 0
            for idx, row in group.iterrows():
                if i == 0:
                    x0 = row['easting']
                    y0 = row['northing']
                length = (
                    ((row['easting'] - x0)**2)
                    + ((row['northing'] - y0)**2)
                )**(1/2)
                distance.append(length)
                i += 1
            group['distance'] = distance
            df = df.append(group)

        return df

    def plot_downstream_bars(self, bargroup):
        """
        Generates plot of all bar widths as ratio with channel width
        going downstream
        """
        plt.close()
        fig = plt.figure(figsize = (11, 7))
        widthcol = 'channel_width_water'
        for name, group in bargroup:
            group = group[group[widthcol] > 0]
            plt.scatter(
                group['distance'] / max_distance, 
                group[widthcol] / median_width, 
                s=10
            )
        plt.show()

    def plot_widths(self, bars, path):
        """
        Generates plot of all bar widths with respect to channel width
        """
        plt.close()
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

        plt.close()
        plt.scatter(width_df['bar_width'], width_df['channel_width'])
        plt.xlabel('Bar Width (m)')
        plt.ylabel('Channel Width (m)')
        plt.savefig(path)

    def setAlpha(self, ax,a):
        for art in ax.get_children():
            if isinstance(art, PolyCollection):
                art.set_alpha(a)

    def data_figure(self, out, group_river, group_bar, bar_coefs, reach_coefs,
                       lit_df, median_size=10, alpha=0.2, density_size=5, 
                       bar_intercept=None, reach_intercept=None, fmt='png'):
        """ 
        Data Figure that shows bar width and channel width. 
        Density cloud for the individual bar points
        """
        # Get the lines for the estimated parameters
        # Xs
        xs = np.linspace(0, 1000, 100)
        # Mean Width
        if bar_intercept:
            Ymean5_bar = bar_intercept['mean']['5'] + bar_coefs['mean']['5'] * xs
            Ymean50_bar = (
                bar_intercept['mean']['50'] + bar_coefs['mean']['50'] * xs
            )
            Ymean95_bar = (
                bar_intercept['mean']['95'] + bar_coefs['mean']['95'] * xs
            )
        else:
            Ymean5_bar = bar_coefs['mean']['5'] * xs
            Ymean50_bar = (
                bar_coefs['mean']['50'] * xs
            )
            Ymean95_bar = (
                bar_coefs['mean']['95'] * xs
            )
#            Ymean5_bar = xs**bar_coefs['mean']['5']
#            Ymean50_bar = xs**bar_coefs['mean']['50']
#            Ymean95_bar = xs**bar_coefs['mean']['95']

        if reach_intercept:
            Ymean5_reach = reach_intercept['mean']['5'] + reach_coefs['mean']['5'] * xs
            Ymean50_reach = (
                reach_intercept['mean']['50'] + reach_coefs['mean']['50'] * xs
            )
            Ymean95_reach = (
                reach_intercept['mean']['95'] + reach_coefs['mean']['95'] * xs
            )
        else:
            Ymean5_reach = reach_coefs['mean']['5'] * xs
            Ymean50_reach = (
                reach_coefs['mean']['50'] * xs
            )
            Ymean95_reach = (
                reach_coefs['mean']['95'] * xs
            )
#            Ymean5_reach = xs**reach_coefs['mean']['5']
#            Ymean50_reach = xs**reach_coefs['mean']['50']
#            Ymean95_reach = xs**reach_coefs['mean']['95']

        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        fig.set_figheight(15)
        fig.set_figwidth(15)

        # Plot the parameter fill
        ax[0].fill_between(
            xs, 
            Ymean5_bar, 
            Ymean95_bar, 
            color='lightgray', 
            edgecolor='lightgray',
            linestyle='--'
        )
        ax[1].fill_between(
            xs, 
            Ymean5_reach, 
            Ymean95_reach, 
            color='lightgray', 
            edgecolor='lightgray',
            linestyle='--'
        )
        # Plot the parameter estimates
        # Water Width
        # Median Bars
        color = 'lightgray'
        line_style = '--'
        line50 = '-'
        width = 0
        width50 = 0
        # Mean Width
        # Median Bars
        ax[0].plot(
            xs,
            Ymean5_bar,
            linewidth=width,
            color=color,
            linestyle=line_style,
            zorder=2
        )
        ax[0].plot(
            xs,
            Ymean50_bar,
            linewidth=width50,
            color=color,
            linestyle=line50,
            zorder=2
        )
        ax[0].plot(
            xs,
            Ymean95_bar,
            linewidth=width,
            color=color,
            linestyle=line_style,
            zorder=2
        )
        # Reach 
        ax[1].plot(
            xs,
            Ymean5_reach,
            linewidth=width,
            color=color,
            linestyle=line_style,
            zorder=2
        )
        ax[1].plot(
            xs,
            Ymean50_reach,
            linewidth=width50,
            color=color,
            linestyle=line50,
            zorder=2
        )
        ax[1].plot(
            xs,
            Ymean95_reach,
            linewidth=width,
            color=color,
            linestyle=line_style,
            zorder=2
        )

        # Plot the data
        bar_colors = [
            'b', 
            'g', 
            'r', 
            'orange', 
            'm', 
            'yellow', 
            'brown', 
            'White', 
            'pink',
            'lime'
        ]
        ms_cmaps = [
            'Blues', 
            'Greens', 
            'Reds', 
            'Oranges', 
            'Purples',
            'Blues',
            'Yellows',
            'Grays',
            'PuBu',
            'YlOrRd'
        ]
        i = 0
        for name, group in group_bar:
            ax[0].plot(
                group['bar_width'], 
                group['mean_width'], 
                marker='o', 
                markeredgewidth=1.5,
                markeredgecolor='black',
                linestyle='', 
                ms=median_size, 
                label=name,
                color=bar_colors[i]
            )
            ax[1].plot(
                group['bar_width'].median(),
                group['mean_width'].median(),
                marker='o', 
                markeredgewidth=1.5,
                markeredgecolor='black',
                linestyle='', 
                ms=median_size, 
                label=name,
                color=bar_colors[i]
            )
            i += 1

        # Literature Values
        colors = [
            'bisque',
            'yellowgreen',
            'paleturquoise',
            'plum',
            'red',
            'coral',
            'coral',
            'violet',
            'violet',
            'orange',
            'lime',
            'gold'
        ]
        j = 0
        for idx, row in lit_df.iterrows():
            ax[1].plot(
                row['Bar Width'],
                row['Channel Width'], 
                marker='^',
                c=colors[j],
                label=row['River']
            ) 
            j += 1

#        # Validation Data
#        my_val = (43, 93)
#        swartz_val = (42.5, 127)
#        ax[1].plot(
#            my_val[0], 
#            my_val[1],
#            marker='s',
#            markerfacecolor='gold',
#            markeredgecolor='black',
#            markeredgewidth=1.5
#        )
#        ax[1].plot(
#            swartz_val[0], 
#            swartz_val[1],
#            marker='s',
#            markerfacecolor='red',
#            markeredgecolor='black',
#            markeredgewidth=1.5
#        )

        # Add the 1:1 line and 1:10
        y11 = xs
        y110 = xs * 10
        col = 'black'
        lin = '--'
        ax[0].plot(xs, y11, c=col, linestyle=lin)
        ax[0].plot(xs, y110, c=col, linestyle=lin)
        ax[1].plot(xs, y11, c=col, linestyle=lin)
        ax[1].plot(xs, y110, c=col, linestyle=lin)

        # Log axis 
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
        ax[0].set_xscale('log')
        ax[1].set_xscale('log')

        #  Subplot titles
        ax[0].title.set_text('Bar Averages')
        ax[1].title.set_text('Reach Averages')

        #  Legend
        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=7)
        
        # X 
        ax[0].set_xlabel('')
        ax[1].set_xlabel('')
        # Y
        ax[0].set_ylabel('')
        ax[1].set_ylabel('')

        # Sizeing
        ax[0].set(aspect=1)
        ax[1].set(aspect=1)

        # Set axis range
        ax[0].set_xlim(1, 1500)
        ax[0].set_ylim(1, 1500)
        ax[1].set_xlim(1, 1500)
        ax[1].set_ylim(1, 1500)

        # Axis text
        fig.text(0.5, 0.1, 'Bar Width (m)', ha='center')
        fig.text(
            0.04, 
            0.5, 
            'Channel Width (m)', 
            va='center', 
            rotation='vertical'
        )
        # Save
        fig.savefig(out, format=fmt)
        plt.show()


if __name__ == '__main__':
    xs = np.linspace(0, 1000, 10000)
    y1 = xs**2
    y2 = xs**4
    plt.loglog(xs, y1)
    plt.loglog(xs, y2)
    
#    plt.yscale('log')
#    plt.xscale('log')

#    plt.xlim(.001, 400)
#    plt.ylim(.001, 1200)

    plt.show()
