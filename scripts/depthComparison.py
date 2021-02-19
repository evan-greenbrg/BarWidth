import math
import pandas
import numpy
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


data_path = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/total_data.csv'
trampush_path = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Global Datasets Dunne.csv'
leeder_path = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Leeder1973Data.csv'
lit_path = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Lit_values.csv'


df = pandas.read_csv(data_path)
trampush_df = pandas.read_csv(trampush_path)
trampush_df['depth'] = trampush_df['Depth (m)']
trampush_df['width'] = trampush_df['Width (m)']
trampush_df = trampush_df[['width', 'depth']]
trampush_df['width_depth'] = trampush_df['width'] / trampush_df['depth']
trampush_df = trampush_df.dropna(how='any')

leeder_df = pandas.read_csv(leeder_path)

lit_df = pandas.read_csv(lit_path)

# Make sure the heights are absolute values
df['bar_height'] = df['bar_height'].abs()
df = df[df['river'] != 'Koyukuk']
df = df[df['river'] != 'Platte River']
df_river = df.groupby('river')

for name, group in df_river:
    print(name)
    print(group['bar_height'].mean())
#     plt.hist(group['bar_height'])
#     plt.show()


def boot(wd, n, l):
    medians = []
    for n in range(1, n+1):
        sample = wd.sample(l, replace=True)
        medians.append(sample.median())
    
    return medians

river_df = df[
    ['river', 'channel_width_dem', 'bar_width', 'bar_height']
].groupby('river').median()

# Hayden
hayden_predicted = river_df['bar_height'] * numpy.quantile(trampush_df['width_depth'], .5)

x = numpy.linspace(.1, 1000)
hayden_median = x * numpy.quantile(trampush_df['width_depth'], .5)
hayden_lower = x * numpy.quantile(trampush_df['width_depth'], .05)
hayden_upper = x * numpy.quantile(trampush_df['width_depth'], .95)

# Leeder
leeder_predicted = 6.8 * (river_df['bar_height']**1.54)

x = numpy.linspace(.1, 1000)
leeder_median = 6.8 * (x**1.54)
leeder_lower = 1.34896 * (x**1.54)
leeder_upper = 33.8844 * (x**1.54)

# Mine
my_predicted = river_df['bar_width'] * 2.42

x = numpy.linspace(.1, 1000)
my_median = 2.42 * x
my_upper = 5.5 * x
my_lower = 1.1 * x

# Get R2
haydenr2 = r2_score(river_df['channel_width_dem'], hayden_predicted)
leederr2 = r2_score(river_df['channel_width_dem'], leeder_predicted)
myr2 = r2_score(river_df['channel_width_dem'], my_predicted)

# Plotting 
fig, axs = plt.subplots(1, 3, figsize=(20, 7), sharey=True)

# Hayden
axs[0].fill_between(x, hayden_lower, hayden_upper, color='green', alpha=.5)
axs[0].plot(x, hayden_median, color='black', linestyle='dashed')
axs[0].plot(x, hayden_lower, color='black', linestyle='dashed')
axs[0].plot(x, hayden_upper, color='black', linestyle='dashed')

# Leeder
axs[1].fill_between(x, leeder_lower, leeder_upper, color='blue', alpha=.5)
axs[1].plot(x, leeder_median, color='black', linestyle='dashed')
axs[1].plot(x, leeder_lower, color='black', linestyle='dashed')
axs[1].plot(x, leeder_upper, color='black', linestyle='dashed')

# Me
axs[2].fill_between(x, my_lower, my_upper, color='gray', alpha=.5)
axs[2].plot(x, my_median, color='black', linestyle='dashed')
axs[2].plot(x, my_lower, color='black', linestyle='dashed')
axs[2].plot(x, my_upper, color='black', linestyle='dashed')

s = 30
s1 = 40
pcolor = 'black'
scolor = 'gray'
fcolor = 'white'
fscolor = 'black'
axs[0].scatter(
    river_df['bar_height'], 
    river_df['channel_width_dem'], 
    s=100, 
    edgecolors='black',
    facecolors='white',
    marker='s'
)

axs[1].scatter(
    river_df['bar_height'], 
    river_df['channel_width_dem'], 
    s=100, 
    edgecolors='black',
    facecolors='white',
    marker='s'
)

axs[2].scatter(
    river_df['bar_width'], 
    river_df['channel_width_dem'], 
    s=100, 
    edgecolors='black',
    facecolors='white',
    marker='s'
)

axs[0].set_xscale('log')
axs[0].set_yscale('log')

axs[1].set_xscale('log')
axs[1].set_yscale('log')

axs[2].set_xscale('log')
axs[2].set_yscale('log')

axs[0].set_xlabel('Depth (m)')
axs[1].set_xlabel('Depth (m)')
axs[2].set_xlabel('Bar Width (m)')

axs[0].set_ylabel('Channel Width (m)')

axs[0].set_ylim([1, 10000])
axs[1].set_ylim([1, 10000])
axs[2].set_ylim([1, 10000])

axs[0].set_xlim([.1, 100])
axs[1].set_xlim([.1, 100])
axs[2].set_xlim([1, 1000])

# Legends
hayden_legend = [
    Line2D(
        [0], 
        [0], 
        marker='s',
        markerfacecolor='white', 
        markeredgecolor='black',
        lw=0
    ),
    Line2D([0], [0], color='green', alpha=.5, lw=4),
]
axs[0].legend(
    hayden_legend, 
    [
        'Data from this study',
        'Hayden (2020) uncertainty'
    ],
    frameon=False
)

# Leeder
leeder_legend = [
    Line2D(
        [0], 
        [0], 
        marker='s',
        markerfacecolor='white', 
        markeredgecolor='black',
        lw=0
    ),
    Line2D([0], [0], color='b', alpha=.5, lw=4),
]
axs[1].legend(
    leeder_legend, 
    [
        'Data from this study',
        'Leeder (1973) uncertainty'
    ],
    frameon=False
)

# Me 
my_legend = [
    Line2D(
        [0], 
        [0], 
        marker='s',
        markerfacecolor='white', 
        markeredgecolor='black',
        lw=0
    ),
    Line2D([0], [0], color='gray', alpha=.5, lw=4),
]
axs[2].legend(
    my_legend, 
    [
        'Data from this study',
        'This study uncertainty'
    ],
    frameon=False
)

# Titles
axs[0].set_title('Hayden (2020)')
axs[1].set_title('Leeder (1973)')
axs[2].set_title('This study (using bar widths)')

plt.savefig('data_comparison_new.pdf', format='pdf')
plt.show()



###### FIGURE 1 ################
### Uncertainty Comparrison ####
################################

x = numpy.linspace(.1, 1000)
# Hayden
hayden_depth = x / numpy.quantile(trampush_df['width_depth'], .5)
hayden_lower = numpy.quantile(trampush_df['width_depth'], .05)*hayden_depth
hayden_upper = numpy.quantile(trampush_df['width_depth'], .95)*hayden_depth

# Leeder
leeder_depth = (x / 6.8)**(1/1.54)
leeder_lower = 1.34896 * (leeder_depth**1.54)
leeder_upper = 33.8844 * (leeder_depth**1.54)

# Me
my_bar_width = x / 2.42
my_upper = 5.5 * my_bar_width
my_lower = 1.1 * my_bar_width

fig, axs = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
axs[0].plot(x, x, color='black')
axs[1].plot(x, x, color='black')

# Fills 
axs[0].fill_between(x, hayden_lower, hayden_upper, alpha=.5, color='green')
axs[0].fill_between(x, my_lower, my_upper, alpha=1, color='lightgray')

axs[1].fill_between(x, leeder_lower, leeder_upper, alpha=.5, color='blue')
axs[1].fill_between(x, my_lower, my_upper, alpha=1, color='lightgray')

#Hayden
axs[0].plot(x, hayden_lower, label='Hayden Lower', linestyle='dashed', color='green')
axs[0].plot(x, hayden_upper, label='Hayden Upper', linestyle='dashed', color='green')

# Leeder
axs[1].plot(x, leeder_lower, label='Leeder Lower', linestyle='dashed', color='blue')
axs[1].plot(x, leeder_upper, label='Leeder Upper', linestyle='dashed', color='blue')
# Me
axs[0].plot(x, my_lower, label='Bar Lower', color='black', linestyle='dashed')
axs[0].plot(x, my_upper, label='Bar Upper', color='black', linestyle='dashed')
axs[1].plot(x, my_lower, label='Bar Lower', color='black', linestyle='dashed')
axs[1].plot(x, my_upper, label='Bar Upper', color='black', linestyle='dashed')

# Axis scales
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[1].set_xscale('log')
axs[1].set_yscale('log')

# Labels
axs[0].set_xlabel('Actual Width')
axs[0].set_ylabel('Prediced Width')
axs[1].set_xlabel('Actual Width')
axs[1].set_ylabel('Prediced Width')

# Legend
hayden_lines = [
    Line2D([0], [0], color='gray', lw=4),
    Line2D([0], [0], color='green', lw=4),
]
leeder_lines = [
    Line2D([0], [0], color='gray', lw=4),
    Line2D([0], [0], color='blue', lw=4),
]

leg1 = axs[0].legend(
    hayden_lines, 
    ['Bar method', 'Hayden (2020)'], 
    title='Range of 95% confidence estimates', 
    frameon=False
)
leg2 = axs[1].legend(
    leeder_lines, 
    ['Bar method', 'Leeder (1973)'], 
    title='Range of 95% confidence estimates', 
    frameon=False
)
leg1._legend_box.align = "left"
leg2._legend_box.align = "left"

# Anotation
axs[0].annotate(
    '1:1', 
    (.2, .2), 
    rotation=38, 
    bbox=dict(boxstyle="square,pad=0.3", fc="lightgray", ec="b", lw=0)
)
axs[1].annotate(
    '1:1', 
    (.2, .2), 
    rotation=38, 
    bbox=dict(boxstyle="square,pad=0.3", fc="lightgray", ec="b", lw=0)
)

axs[0].set_title('Comparison between bar method and Hayden (2020)')
axs[1].set_title('Comparison between bar method and Leeder (1973)')

plt.savefig('Uncertainty_comparrison.pdf', format='pdf')
plt.show()


###### FIGURE 2 ################
##### Data Comparison ##########
################################

# plotting
x = numpy.linspace(.1, 5000)
# Hayden
y = numpy.quantile(trampush_df['width_depth'], .5)*x
yb = numpy.quantile(trampush_df['width_depth'], .05)*x
yt = numpy.quantile(trampush_df['width_depth'], .95)*x

# Leeder
lm = 6.8 * (x**1.54)
lb = 1.34896 * (x**1.54)
lt = 33.8844 * (x**1.54)

# Me
m = 2.42 * x
mb = 2.32 * x
mt = 2.50 * x
b = 1.10 * x
t = 5.50 * x

# Hayden
fig, axs = plt.subplots(1, 3, figsize=(20, 7), sharey=True)

# Hayden
axs[0].fill_between(x, yb, yt, color='green', alpha=.5)
axs[0].plot(x, y, color='black', linestyle='dashed')
axs[0].plot(x, yb, color='black', linestyle='dashed')
axs[0].plot(x, yt, color='black', linestyle='dashed')

# Leeder
axs[1].fill_between(x, lb, lt, color='blue', alpha=.5)
axs[1].plot(x, lm, color='black', linestyle='dashed')
axs[1].plot(x, lb, color='black', linestyle='dashed')
axs[1].plot(x, lt, color='black', linestyle='dashed')

# Me
axs[2].fill_between(x, b, t, color='gray', alpha=.5)
axs[2].plot(x, m, color='black', linestyle='dashed')
axs[2].plot(x, mb, color='black', linestyle='dashed')
axs[2].plot(x, mt, color='black', linestyle='dashed')
axs[2].plot(x, b, color='black', linestyle='dashed')
axs[2].plot(x, t, color='black', linestyle='dashed')


s = 30
s1 = 40
pcolor = 'black'
scolor = 'gray'
fcolor = 'white'
fscolor = 'black'
axs[0].scatter(
    leeder_df['Depth'], 
    leeder_df['Width'], 
    s=30, 
    edgecolors='black',
    facecolors='black',
    marker='^'
)
axs[0].scatter(
    df['bar_height'], 
    df['channel_width_dem'], 
    s=30, 
    edgecolors='black',
    facecolors='black',
    marker='s'
)
axs[0].scatter(
    trampush_df['depth'], 
    trampush_df['width'], 
    s=45, 
    edgecolors='black',
    facecolors='white',
    marker='o'
)

axs[1].scatter(
    df['bar_height'], 
    df['channel_width_dem'], 
    s=30, 
    edgecolors='black',
    facecolors='black',
    marker='s'
)
axs[1].scatter(
    trampush_df['depth'], 
    trampush_df['width'], 
    s=30, 
    edgecolors='black',
    facecolors='black',
    marker='o'
)
axs[1].scatter(
    leeder_df['Depth'], 
    leeder_df['Width'], 
    s=65, 
    edgecolors='black',
    facecolors='white',
    marker='^'
)

axs[2].scatter(
    df['bar_width'], 
    df['channel_width_dem'], 
    s=30, 
    edgecolors='black',
    facecolors='white',
    marker='s'
)

axs[0].set_xscale('log')
axs[0].set_yscale('log')

axs[1].set_xscale('log')
axs[1].set_yscale('log')

axs[2].set_xscale('log')
axs[2].set_yscale('log')

axs[0].set_xlabel('Depth (m)')
axs[1].set_xlabel('Depth (m)')
axs[2].set_xlabel('Bar Width (m)')

axs[0].set_ylabel('Channel Width (m)')

axs[0].set_ylim([1, 10000])
axs[1].set_ylim([1, 10000])
axs[2].set_ylim([1, 10000])

axs[0].set_xlim([.1, 100])
axs[1].set_xlim([.1, 100])
axs[2].set_xlim([1, 5000])

# Legends
hayden_legend = [
    Line2D(
        [0], 
        [0], 
        marker='o',
        markerfacecolor='white', 
        markeredgecolor='black',
        lw=0
    ),
    Line2D(
        [0], 
        [0], 
        marker='^',
        markerfacecolor='black', 
        markeredgecolor='black',
        lw=0
    ),
    Line2D(
        [0], 
        [0], 
        marker='s',
        markerfacecolor='black', 
        markeredgecolor='black',
        lw=0
    ),
    Line2D([0], [0], color='green', alpha=.5, lw=4),
]
axs[0].legend(
    hayden_legend, 
    [
        'Hayden (2020) data', 
        'Leeder (1973) data', 
        'Data from this study',
        'Hayden (2020) uncertainty'
    ],
    frameon=False
)

# Leeder
leeder_legend = [
    Line2D(
        [0], 
        [0], 
        marker='^',
        markerfacecolor='white', 
        markeredgecolor='black',
        lw=0
    ),
    Line2D(
        [0], 
        [0], 
        marker='s',
        markerfacecolor='black', 
        markeredgecolor='black',
        lw=0
    ),
    Line2D(
        [0], 
        [0], 
        marker='o',
        markerfacecolor='black', 
        markeredgecolor='black',
        lw=0
    ),
    Line2D([0], [0], color='b', alpha=.5, lw=4),
]
axs[1].legend(
    leeder_legend, 
    [
        'Leeder (1973) data', 
        'Data from this study',
        'Hayden (2020) data', 
        'Leeder (1973) uncertainty'
    ],
    frameon=False
)

# Me 
my_legend = [
    Line2D(
        [0], 
        [0], 
        marker='s',
        markerfacecolor='white', 
        markeredgecolor='black',
        lw=0
    ),
    Line2D(
        [0], 
        [0], 
        marker='o',
        markerfacecolor='black', 
        markeredgecolor='black',
        lw=0
    ),
    Line2D(
        [0], 
        [0], 
        marker='^',
        markerfacecolor='black', 
        markeredgecolor='black',
        lw=0
    ),
    Line2D([0], [0], color='gray', alpha=.5, lw=4),
]
axs[2].legend(
    my_legend, 
    [
        'Data from this study',
        'Hayden (2020) data', 
        'Leeder (1973) data', 
        'Uncertainty from this study'
    ],
    frameon=False
)

# Titles
axs[0].set_title('Hayden (2020)')
axs[1].set_title('Leeder (1973)')
axs[2].set_title('This study (using bar widths)')

plt.savefig('data_comparison.pdf', format='pdf')
plt.show()

# Equation
# logw = 1.54 * log h + .83
# sd on B = .35
# 2sd = .7
# 1.53 -> 33.8844
# .13 -> 1.3489

# w = 6.8 * h**1.54

