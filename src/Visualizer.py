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

    def data_figure_v1(self, out, group_river, group_bar,
                    bar_intercept, bar_coefs,
                    ms_intercept, ms_coefs, 
                    median_size=10, alpha=0.2, density_size=5):
        """ 
        Data Figure that shows bar width and channel width. 
        Density cloud for the individual bar points
        """
        # Get the lines for the estimated parameters
        # Xs
        xs = np.linspace(0, 500, 100)
        # Water Ys
        Ywater5_bar = (
            bar_intercept['water']['5'] + bar_coefs['water']['5'] * xs
        )
        Ywater50_bar = (
            bar_intercept['water']['50'] + bar_coefs['water']['50'] * xs
        )
        Ywater95_bar = (
            bar_intercept['water']['95'] + bar_coefs['water']['95'] * xs
        )
        Ywater5_ms = (
            ms_intercept['water']['5'] + ms_coefs['water']['5'] * xs
        )
        Ywater50_ms = (
            ms_intercept['water']['50'] + ms_coefs['water']['50'] * xs
        )
        Ywater95_ms = (
            ms_intercept['water']['95'] + ms_coefs['water']['95'] * xs
        )
        # DEM Ys
        Ydem5_bar = bar_intercept['dem']['5'] + bar_coefs['dem']['5'] * xs
        Ydem50_bar = bar_intercept['dem']['50'] + bar_coefs['dem']['50'] * xs
        Ydem95_bar = bar_intercept['dem']['95'] + bar_coefs['dem']['95'] * xs
        Ydem5_ms = ms_intercept['dem']['5'] + ms_coefs['dem']['5'] * xs
        Ydem50_ms = ms_intercept['dem']['50'] + ms_coefs['dem']['50'] * xs
        Ydem95_ms = ms_intercept['dem']['95'] + ms_coefs['dem']['95'] * xs
        # Mean Width
        Ymean5_bar = bar_intercept['mean']['5'] + bar_coefs['mean']['5'] * xs
        Ymean50_bar = (
            bar_intercept['mean']['50'] + bar_coefs['mean']['50'] * xs
        )
        Ymean95_bar = (
            bar_intercept['mean']['95'] + bar_coefs['mean']['95'] * xs
        )
        Ymean5_ms = ms_intercept['mean']['5'] + ms_coefs['mean']['5'] * xs
        Ymean50_ms = ms_intercept['mean']['50'] + ms_coefs['mean']['50'] * xs
        Ymean95_ms = ms_intercept['mean']['95'] + ms_coefs['mean']['95'] * xs

        fig, ax = plt.subplots(3, 2, sharex=True, sharey=True)
        # Plot the parameter fill
        ax[0, 0].fill_between(
            xs, 
            Ywater5_bar, 
            Ywater95_bar, 
            color='lightgray', 
            edgecolor='lightgray',
            linestyle='--'
        )
        ax[0, 1].fill_between(
            xs, 
            Ywater5_ms, 
            Ywater95_ms, 
            color='lightgray',
            edgecolor='lightgray',
            linestyle='--'
        )
        ax[1, 0].fill_between(
            xs, 
            Ydem5_bar, 
            Ydem95_bar, 
            color='lightgray',
            edgecolor='lightgray',
            linestyle='--'
        )
        ax[1, 1].fill_between(
            xs, 
            Ydem5_ms, 
            Ydem95_ms, 
            color='lightgray',
            edgecolor='lightgray',
            linestyle='--'
        )
        ax[2, 0].fill_between(
            xs,
            Ymean5_bar, 
            Ymean95_bar, 
            color='lightgray',
            edgecolor='lightgray',
            linestyle='--'
        )
        ax[2, 1].fill_between(
            xs, 
            Ymean5_ms, 
            Ymean95_ms, 
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
        ax[0, 0].plot(
            xs,
            Ywater5_bar,
            linewidth=width,
            color=color,
            linestyle=line_style,
            zorder=2
        )
        ax[0, 0].plot(
            xs,
            Ywater50_bar,
            linewidth=width50,
            color=color,
            linestyle=line50,
            zorder=2
        )
        ax[0, 0].plot(
            xs,
            Ywater95_bar,
            linewidth=width,
            color=color,
            linestyle=line_style,
            zorder=2
        )
        # All Bars
        ax[0, 1].plot(
            xs,
            Ywater5_ms,
            linewidth=width,
            color=color,
            linestyle=line_style,
            zorder=2
        )
        ax[0, 1].plot(
            xs,
            Ywater50_ms,
            linewidth=width50,
            color=color,
            linestyle=line50,
            zorder=2
        )
        ax[0, 1].plot(
            xs,
            Ywater95_ms,
            linewidth=width,
            color=color,
            linestyle=line_style,
            zorder=2
        )
        # DEM Width
        # Median Bars
        ax[1, 0].plot(
            xs,
            Ydem5_bar,
            linewidth=width,
            color=color,
            linestyle=line_style,
            zorder=2
        )
        ax[1, 0].plot(
            xs,
            Ydem50_bar,
            linewidth=width50,
            color=color,
            linestyle=line50,
            zorder=2
        )
        ax[1, 0].plot(
            xs,
            Ydem95_bar,
            linewidth=width,
            color=color,
            linestyle=line_style,
            zorder=2
        )
        # All Bars
        ax[1, 1].plot(
            xs,
            Ydem5_ms,
            linewidth=width,
            color=color,
            linestyle=line_style,
            zorder=2
        )
        ax[1, 1].plot(
            xs,
            Ydem50_ms,
            linewidth=width50,
            color=color,
            linestyle=line50,
            zorder=2
        )
        ax[1, 1].plot(
            xs,
            Ydem95_ms,
            linewidth=width,
            color=color,
            linestyle=line_style,
            zorder=2
        )
        # Mean Width
        # Median Bars
        ax[2, 0].plot(
            xs,
            Ymean5_bar,
            linewidth=width,
            color=color,
            linestyle=line_style,
            zorder=2
        )
        ax[2, 0].plot(
            xs,
            Ymean50_bar,
            linewidth=width50,
            color=color,
            linestyle=line50,
            zorder=2
        )
        ax[2, 0].plot(
            xs,
            Ymean95_bar,
            linewidth=width,
            color=color,
            linestyle=line_style,
            zorder=2
        )
        # All Bars
        ax[2, 1].plot(
            xs,
            Ymean5_ms,
            linewidth=width,
            color=color,
            linestyle=line_style,
            zorder=2
        )
        ax[2, 1].plot(
            xs,
            Ymean50_ms,
            linewidth=width50,
            color=color,
            linestyle=line50,
            zorder=2
        )
        ax[2, 1].plot(
            xs,
            Ymean95_ms,
            linewidth=width,
            color=color,
            linestyle=line_style,
            zorder=2
        )
        # Plot the data
        i = 0
        bar_colors = ['b', 'g', 'r', 'orange', 'm']
        ms_cmaps = [
            'Blues', 
            'Greens', 
            'Reds', 
            'Oranges', 
            'Purples',
        ]
        i = 0
        for name, group in group_river:
            print(name)
            xy_water = np.vstack([
                group['bar_width'], 
                group['channel_width_water'], 
            ])
            z_water = gaussian_kde(xy_water)(xy_water)
         
            xy_dem= np.vstack([
                group['bar_width'], 
                group['channel_width_dem'], 
            ])
            z_dem= gaussian_kde(xy_dem)(xy_dem)
         
            xy_mean = np.vstack([
                group['bar_width'], 
                group['mean_width'], 
            ])
            z_mean = gaussian_kde(xy_mean)(xy_mean)

            ax[0, 0].scatter(
                group['bar_width'], 
                group['channel_width_water'], 
                c=z_water,
                linewidth=0,
                alpha=alpha,
                marker='o', 
                s=density_size, 
                label=name,
                cmap=ms_cmaps[i]
            )
            ax[0, 1].scatter(
                group['bar_width'], 
                group['channel_width_water'], 
                c=z_water,
                linewidth=0,
                alpha=alpha,
                marker='o', 
                s=density_size, 
                label=name,
                cmap=ms_cmaps[i]
            )
            ax[1, 0].scatter(
                group['bar_width'], 
                group['channel_width_dem'], 
                c=z_water,
                linewidth=0,
                alpha=alpha,
                marker='o', 
                s=density_size, 
                label=name,
                cmap=ms_cmaps[i]
            )
            ax[1, 1].scatter(
                group['bar_width'], 
                group['channel_width_dem'], 
                c=z_water,
                linewidth=0,
                alpha=alpha,
                marker='o', 
                s=density_size, 
                label=name,
                cmap=ms_cmaps[i]
            )
            ax[2, 0].scatter(
                group['bar_width'], 
                group['mean_width'], 
                c=z_water,
                linewidth=0,
                alpha=alpha,
                marker='o', 
                s=density_size, 
                label=name,
                cmap=ms_cmaps[i]
            )
            ax[2, 1].scatter(
                group['bar_width'], 
                group['mean_width'], 
                c=z_water,
                linewidth=0,
                alpha=alpha,
                marker='o', 
                s=density_size, 
                label=name,
                cmap=ms_cmaps[i]
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
        # Opacity
#        self.setAlpha(ax[0, 1], 0.3):
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


    def data_figure_v2(self, out, group_river, group_bar, bar_coefs, 
                           median_size=10, alpha=0.2, density_size=5, 
                           bar_intercept=None, fmt='png'):
        """ 
        Data Figure that shows bar width and channel width. 
        Density cloud for the individual bar points
        """
        # Get the lines for the estimated parameters
        # Xs
        xs = np.linspace(0, 500, 100)
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

        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
        # Plot the parameter fill
        ax.fill_between(
            xs, 
            Ymean5_bar, 
            Ymean95_bar, 
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
        ax.plot(
            xs,
            Ymean5_bar,
            linewidth=width,
            color=color,
            linestyle=line_style,
            zorder=2
        )
        ax.plot(
            xs,
            Ymean50_bar,
            linewidth=width50,
            color=color,
            linestyle=line50,
            zorder=2
        )
        ax.plot(
            xs,
            Ymean95_bar,
            linewidth=width,
            color=color,
            linestyle=line_style,
            zorder=2
        )
        # Plot the data
        i = 0
        bar_colors = ['b', 'g', 'r', 'orange', 'm']
        ms_cmaps = [
            'Blues', 
            'Greens', 
            'Reds', 
            'Oranges', 
            'Purples',
        ]
        i = 0
        for name, group in group_river:
            xy_mean = np.vstack([
                group['bar_width'], 
                group['mean_width'], 
            ])
            z_mean = gaussian_kde(xy_mean)(xy_mean)

            ax.scatter(
                group['bar_width'], 
                group['mean_width'], 
                c=z_mean,
                linewidth=0,
                alpha=alpha,
                marker='o', 
                s=density_size, 
                label=name,
                cmap=ms_cmaps[i]
            )
            i += 1
        i = 0
        for name, group in group_bar:
            ax.plot(
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

        # Subplot titles
        ax.title.set_text('Mean Width - Median Bars')
        # Legend
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right', ncol=2)
        # Get rid of seaborn labels
        # X 
        ax.set_xlabel('')
        # Y
        ax.set_ylabel('')
        # Opacity
#        self.setAlpha(ax[0, 1], 0.3):
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
        fig.savefig(out, format=fmt)
        plt.show()

    def data_figure_v3(self, out, group_river, group_bar, bar_coefs, lit_df,
                           median_size=10, alpha=0.2, density_size=5, 
                           bar_intercept=None, fmt='png'):
        """ 
        Data Figure that shows bar width and channel width. 
        Density cloud for the individual bar points
        """
        # Get the lines for the estimated parameters
        # Xs
        xs = np.linspace(0, 500, 100)
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

        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
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
            Ymean5_bar, 
            Ymean95_bar, 
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
            Ymean5_bar,
            linewidth=width,
            color=color,
            linestyle=line_style,
            zorder=2
        )
        ax[1].plot(
            xs,
            Ymean50_bar,
            linewidth=width50,
            color=color,
            linestyle=line50,
            zorder=2
        )
        ax[1].plot(
            xs,
            Ymean95_bar,
            linewidth=width,
            color=color,
            linestyle=line_style,
            zorder=2
        )

        # Plot the data
        bar_colors = ['b', 'g', 'r', 'orange', 'm']
        ms_cmaps = [
            'Blues', 
            'Greens', 
            'Reds', 
            'Oranges', 
            'Purples',
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
            'orange'
        ]
        j = 0
        for idx, row in lit_df.iterrows():
            ax[1].scatter(
                row['Bar Width'],
                row['Channel Width'], 
                marker='^',
                c=colors[j],
                label=row['River']
            ) 
            j += 1
          
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
        ax[0].set(aspect=0.4)
        ax[1].set(aspect=0.4)

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
