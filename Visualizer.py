import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns

class Visualizer():

    def __init__(self):
        pass

    def get_downstream_distance(self, bars, x0, y0):
        """
        Take UTM coordinates from bars dictionary and 
        converts to downstream distance
        """
        for key in bars.keys():
            distance = []
            for idx, coor in enumerate(bars[key]['coords']):
                length = (
                    ((coor[0] - x0)**2)
                    + ((coor[1] - y0)**2)
                )**(1/2)
                distance.append(length)
            bars[key]['distance'] = distance

        return bars



    def plot_downstream_bars(self, bars, path):
        """
        Generates plot of all bar widths as ratio with channel width
        going downstream
        """
        plt.close()
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

    def data_figure(self, out, ms_df, group_bar,
                    bar_intercept, bar_coefs,
                    ms_intercept, ms_coefs, 
                    bw=20, median_size=10):
        """ 
        Data Figure that shows bar width and channel width. 
        Density cloud for the individual bar points
        """
        fig, ax = plt.subplots(3, 2, sharex=True, sharey=True)
        # Plot the data
        i = 0
        bar_colors = ['b', 'g', 'r', 'orange', 'm']
        ms_cmaps = [
            'Reds', 
            'Greens', 
            'Purples',
            'Oranges', 
            'Blues', 
        ]
        rivers = [
            'Red River', 
            'Platte River', 
            'White River', 
            'Trinity River', 
            'Koyukuk River'
        ]
        bw = 2
        median_size = 5
        i = 0
        #for river in rivers:
        sns.kdeplot(
            ms_df['bar_width'], 
            ms_df['channel_width_water'], 
            cmap=ms_cmaps[i], 
            shade=True,
            shade_lowest=False,
            bw=bw,
            ax=ax[0, 0]
        )
        sns.kdeplot(
            ms_df['bar_width'], 
            ms_df['channel_width_water'], 
            cmap=ms_cmaps[i], 
            shade=True,
            shade_lowest=False,
            bw=bw,
            ax=ax[0, 1]
        )
        sns.kdeplot(
            ms_df['bar_width'], 
            ms_df['channel_width_dem'], 
            cmap=ms_cmaps[i], 
            shade=True,
            shade_lowest=False,
            bw=bw,
            ax=ax[1, 0]
        )
        sns.kdeplot(
            ms_df['bar_width'], 
            ms_df['channel_width_dem'], 
            cmap=ms_cmaps[i], 
            shade=True,
            shade_lowest=False,
            bw=bw,
            ax=ax[1, 1]
        )
        sns.kdeplot(
            ms_df['bar_width'], 
            ms_df['mean_width'], 
            cmap=ms_cmaps[i], 
            shade=True,
            shade_lowest=False,
            bw=bw,
            ax=ax[2, 0]
        )
        sns.kdeplot(
            ms_df['bar_width'], 
            ms_df['mean_width'], 
            cmap=ms_cmaps[i], 
            shade=True,
            shade_lowest=False,
            bw=bw,
            ax=ax[2, 1]
        )
        i += 1
        i = 0
        for name, group in group_bar:
            ax[0, 0].plot(
                group['bar_width'], 
                group['channel_width_water'], 
                marker='o', 
                markeredgewidth=1.5,
                markeredgecolor='black',
                linestyle='', 
                ms=median_size, 
                label=name,
                color=bar_colors[i]
            )
            ax[0, 1].plot(
                group['bar_width'], 
                group['channel_width_water'], 
                markeredgewidth=1.5,
                markeredgecolor='black',
                marker='o', 
                linestyle='', 
                ms=median_size, 
                label=name,
                color=bar_colors[i]
            )

            ax[1, 0].plot(
                group['bar_width'], 
                group['channel_width_dem'], 
                marker='o', 
                markeredgewidth=1.5,
                markeredgecolor='black',
                linestyle='', 
                ms=median_size, 
                label=name,
                color=bar_colors[i]
            )
            ax[1, 1].plot(
                group['bar_width'], 
                group['channel_width_dem'], 
                markeredgewidth=1.5,
                markeredgecolor='black',
                marker='o', 
                linestyle='', 
                ms=median_size, 
                label=name,
                color=bar_colors[i]
            )

            ax[2, 0].plot(
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
            ax[2, 1].plot(
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
            i += 1

        # Plot the parameter estimates
        xs = np.linspace(0, ms_df['bar_width'].max(), 100)
        # Water Width
        # Median Bars
        ax[0, 0].plot(
            xs,
            bar_intercept['water']['5'] + bar_coefs['water']['5'] * xs,
            linewidth=2
        )
        ax[0, 0].plot(
            xs,
            bar_intercept['water']['50'] + bar_coefs['water']['50'] * xs,
            linewidth=2
        )
        ax[0, 0].plot(
            xs,
            bar_intercept['water']['95'] + bar_coefs['water']['95'] * xs,
            linewidth=2
        )
        # All Bars
        ax[0, 1].plot(
            xs,
            ms_intercept['water']['5'] + ms_coefs['water']['5'] * xs,
            linewidth=2
        )
        ax[0, 1].plot(
            xs,
            ms_intercept['water']['50'] + ms_coefs['water']['50'] * xs,
            linewidth=2
        )
        ax[0, 1].plot(
            xs,
            ms_intercept['water']['95'] + ms_coefs['water']['95'] * xs,
            linewidth=2
        )
        # DEM Width
        # Median Bars
        ax[1, 0].plot(
            xs,
            bar_intercept['dem']['5'] + bar_coefs['dem']['5'] * xs,
            linewidth=2
        )
        ax[1, 0].plot(
            xs,
            bar_intercept['dem']['50'] + bar_coefs['dem']['50'] * xs,
            linewidth=2
        )
        ax[1, 0].plot(
            xs,
            bar_intercept['dem']['95'] + bar_coefs['dem']['95'] * xs,
            linewidth=2
        )
        # All Bars
        ax[1, 1].plot(
            xs,
            ms_intercept['dem']['5'] + ms_coefs['dem']['5'] * xs,
            linewidth=2
        )
        ax[1, 1].plot(
            xs,
            ms_intercept['dem']['50'] + ms_coefs['dem']['50'] * xs,
            linewidth=2
        )
        ax[1, 1].plot(
            xs,
            ms_intercept['dem']['95'] + ms_coefs['dem']['95'] * xs,
            linewidth=2
        )
        # Mean Width
        # Median Bars
        ax[2, 0].plot(
            xs,
            bar_intercept['mean']['5'] + bar_coefs['mean']['5'] * xs,
            linewidth=2
        )
        ax[2, 0].plot(
            xs,
            bar_intercept['mean']['50'] + bar_coefs['mean']['50'] * xs,
            linewidth=2
        )
        ax[2, 0].plot(
            xs,
            bar_intercept['mean']['95'] + bar_coefs['mean']['95'] * xs,
            linewidth=2
        )
        # All Bars
        ax[2, 1].plot(
            xs,
            ms_intercept['mean']['5'] + ms_coefs['mean']['5'] * xs,
            linewidth=2
        )
        ax[2, 1].plot(
            xs,
            ms_intercept['mean']['50'] + ms_coefs['mean']['50'] * xs,
            linewidth=2
        )
        ax[2, 1].plot(
            xs,
            ms_intercept['mean']['95'] + ms_coefs['mean']['95'] * xs,
            linewidth=2
        )

        # Subplot titles
        ax[0, 0].title.set_text('Water Occurence Width - Median Bars')
        ax[0, 1].title.set_text('Water Occurence Width - All Bars')
        ax[1, 0].title.set_text('DEM Width - Median Bars')
        ax[1, 1].title.set_text('DEM Width - All Bars')
        ax[2, 0].title.set_text('Mean Width - Median Bars')
        ax[2, 1].title.set_text('Mean Width - All Bars')
        # Legend
        handles, labels = ax[2, 1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right', ncol=2)
        # Get rid of seaborn labels
        # X 
        ax[0, 0].set_xlabel('')
        ax[0, 1].set_xlabel('')
        ax[1, 0].set_xlabel('')
        ax[1, 1].set_xlabel('')
        ax[2, 0].set_xlabel('')
        ax[2, 1].set_xlabel('')
        # Y
        ax[0, 0].set_ylabel('')
        ax[0, 1].set_ylabel('')
        ax[1, 0].set_ylabel('')
        ax[1, 1].set_ylabel('')
        ax[2, 0].set_ylabel('')
        ax[2, 1].set_ylabel('')
        # Axis text
        fig.text(0.5, 0.04, 'Bar Width (m)', ha='center')
        fig.text(
            0.04, 
            0.5, 
            'Channel Width (m)', 
            va='center', 
            rotation='vertical'
        )
        # Save
        fig.savefig(out, format='svg')
        plt.show()


