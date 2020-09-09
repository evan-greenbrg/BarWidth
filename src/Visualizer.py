import matplotlib.pyplot as plt
import numpy as np
import pandas


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

    def plot_downstream_bars(self, bargroup, max_distance, median_width):
        """
        Generates plot of all bar widths as ratio with channel width
        going downstream
        """
        plt.close()
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

        plt.close()
        plt.scatter(width_df['bar_width'], width_df['channel_width'])
        plt.xlabel('Bar Width (m)')
        plt.ylabel('Channel Width (m)')
        plt.savefig(path)

    def data_figure(self, out, ms_df, group_river, group_bar,
                    lit_df, median_size=10, alpha=0.2, density_size=5,
                    bar_coefs=None, reach_coefs=None,
                    bar_intercept=None, reach_intercept=None,
                    fmt='png', log=True):
        """
        Data Figure that shows bar width and channel width.
        Density cloud for the individual bar points
        """
        # Get the lines for the estimated parameters
        # Xs
        xs = np.linspace(0, 1000, 10000)
        # Mean Width
        if bar_intercept and bar_coefs:
            if not log:
                Ymean5_bar = (
                    bar_intercept['mean']['5'] + bar_coefs['mean']['5'] * xs
                )
                Ymean50_bar = (
                    bar_intercept['mean']['50'] + bar_coefs['mean']['50'] * xs
                )
                Ymean95_bar = (
                    bar_intercept['mean']['95'] + bar_coefs['mean']['95'] * xs
                )
            else:
                Ymean5_bar = (
                    bar_intercept['mean']['5'] * (xs**bar_coefs['mean']['5'])
                )
                Ymean50_bar = (
                    bar_intercept['mean']['50']
                    * (xs**bar_coefs['mean']['50'])
                )
                Ymean95_bar = (
                    bar_intercept['mean']['95']
                    * (xs**bar_coefs['mean']['95'])
                )
        elif not bar_intercept and bar_coefs:
            if not log:
                Ymean5_bar = bar_coefs['mean']['5'] * xs
                Ymean50_bar = (
                    bar_coefs['mean']['50'] * xs
                )
                Ymean95_bar = (
                    bar_coefs['mean']['95'] * xs
                )
            else:
                Ymean5_bar = xs**bar_coefs['mean']['5']
                Ymean50_bar = xs**bar_coefs['mean']['50']
                Ymean95_bar = xs**bar_coefs['mean']['95']

        elif bar_intercept and not bar_coefs:
            if log:
                Ymean5_bar = bar_intercept['mean']['5'] * xs
                Ymean50_bar = bar_intercept['mean']['50'] * xs
                Ymean95_bar = bar_intercept['mean']['95'] * xs

        if reach_intercept and reach_coefs:
            if not log:
                Ymean5_reach = (
                    reach_intercept['mean']['5']
                    + (reach_coefs['mean']['5'] * xs)
                )
                Ymean50_reach = (
                    reach_intercept['mean']['50']
                    + (reach_coefs['mean']['50'] * xs)
                )
                Ymean95_reach = (
                    reach_intercept['mean']['95']
                    + (reach_coefs['mean']['95'] * xs)
                )
            else:
                Ymean5_reach = (
                    reach_intercept['mean']['5']
                    * (xs**reach_coefs['mean']['5'])
                )
                Ymean50_reach = (
                    reach_intercept['mean']['50']
                    * (xs**reach_coefs['mean']['50'])
                )
                Ymean95_reach = (
                    reach_intercept['mean']['95']
                    * (xs**reach_coefs['mean']['95'])
                )
        elif not reach_intercept and reach_coefs:
            if not log:
                Ymean5_reach = reach_coefs['mean']['5'] * xs
                Ymean50_reach = (
                    reach_coefs['mean']['50'] * xs
                )
                Ymean95_reach = (
                    reach_coefs['mean']['95'] * xs
                )
            else:
                Ymean5_reach = xs**reach_coefs['mean']['5']
                Ymean50_reach = xs**reach_coefs['mean']['50']
                Ymean95_reach = xs**reach_coefs['mean']['95']
        elif reach_intercept and not reach_coefs:
            Ymean5_reach = reach_intercept['mean']['5'] * xs
            Ymean50_reach = reach_intercept['mean']['50'] * xs
            Ymean95_reach = reach_intercept['mean']['95'] * xs

        fig, ax = plt.subplots(1, 2)
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
            'lime',
            '#c0b4ff',
            '#ffc0b4',
            'c'
        ]
        i = 0
        for name, group in group_bar:
            print(i)
            # Bend
            ax[0].plot(
                ms_df[ms_df['river'] == name]['bar_width'],
                ms_df[ms_df['river'] == name]['mean_width'],
                marker='o',
                linestyle='',
                ms=6,
                alpha=0.3,
                label=name,
                color=bar_colors[i]
            )
            ax[0].plot(
                group['bar_width'],
                group['mean_width'],
                marker='o',
                markeredgewidth=1.5,
                markeredgecolor='black',
                linestyle='',
                zorder=98,
                ms=median_size,
                label=name,
                color=bar_colors[i]
            )
            ax[0].errorbar(
                group['bar_width'],
                group['mean_width'],
                yerr=(group['channel_width_mean_std']),
                xerr=(group['bar_width_std']),
                ecolor='gray',
                linestyle='',
                capthick=5
            )

#            # Reach
#            ax[1].plot(
#                group['bar_width'].median(),
#                group['mean_width'].median(),
#                marker='o',
#                markeredgewidth=1.5,
#                markeredgecolor='black',
#                linestyle='',
#                ms=median_size,
#                label=name,
#                color=bar_colors[i]
#            )
#            ax[1].errorbar(
#                group['bar_width'].median(),
#                group['mean_width'].median(),
#                yerr=(group['mean_width'].std()),
#                xerr=(group['bar_width'].std()),
#                ecolor=bar_colors[i],
#                capthick=5
#            )
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
            'gold',
            '#151B8D'
        ]
        j = 0
        for idx, row in lit_df.iterrows():
            ax[1].plot(
                row['Bar Width'],
                row['Channel Width'],
                marker='^',
                c=colors[j],
                ms=15,
                label=row['River']
            )
            j += 1
        # fig.legend()
        # Ancient Values
        data = {
            'bar_width': [124, 11.143, 40],
            'channel_width': [301, 23.21, 63]
        }
        ancient_df = pandas.DataFrame(data)
        colors = [
            'green',
            'red',
            'blue'
        ]
        j = 0
        for idx, row in ancient_df.iterrows():
            ax[1].plot(
                row['bar_width'],
                row['channel_width'],
                marker='s',
                c=colors[j],
                ms=15
            )
            j += 1

        # 1:1
        col = 'gray'
        lin = '--'
        ax[0].plot(xs, xs**1, c=col, linestyle=lin)
        ax[1].plot(xs, xs**1, c=col, linestyle=lin)

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
        # fig.legend(handles, labels, loc='lower center', ncol=7)

        # X
        ax[0].set_xlabel('')
        ax[1].set_xlabel('')
        # Y
        ax[0].set_ylabel('')
        ax[1].set_ylabel('')

        # Sizeing
        ax[0].set(aspect=.9)
        ax[1].set(aspect=.9)

        # Set axis range
        ax[0].set_xlim(10, 1000)
        ax[0].set_ylim(10, 3000)
        ax[1].set_xlim(5, 1000)
        ax[1].set_ylim(5, 3000)

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

    def predicted_vs_actual(self, ms_df, bar_df, lit_df, ancient_df):
        fig, axs = plt.subplots(1, 2, sharey=True, sharex=True)
        xs = np.linspace(0, 10000, 10000)
        axs[0].scatter(ms_df['predicted'], ms_df['mean_width'], c='gray')
        axs[0].scatter(bar_df['predicted'], bar_df['mean_width'], c='black')
        axs[0].plot(xs, xs, linestyle='--', c='black')
        axs[0].set_yscale('log')
        axs[0].set_xscale('log')

        axs[1].scatter(
            lit_df['predicted'],
            lit_df['Channel Width'],
            marker='^'
        )
        axs[1].scatter(
            ancient_df['predicted'],
            ancient_df['channel_width'],
            marker='s'
        )

        axs[1].plot(xs, xs, linestyle='--', c='black')
        axs[1].set_yscale('log')
        axs[1].set_xscale('log')

        axs[0].set_xlim(10, 4000)
        axs[0].set_ylim(10, 3000)

        fig.text(0.5, 0.01, 'Predicted Channel Width (m)', ha='center')
        fig.text(
            0.04,
            0.5,
            'Channel Width (m)',
            va='center',
            rotation='vertical'
        )

        plt.show()
