import os
import math

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas
from matplotlib import pyplot as plt
import numpy


def closest(lst, K):
    """
    Finds the closest value in list to value, K
    """
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i]-K))]


def sigmoid(x, L ,x0, k):
    y = L / (1 + numpy.exp(-k*(x-x0)))
    return (y)


param_fn = 'bar_parameters.csv'
data_fn = 'bar_data.csv'

koyukuk_root = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Koyukuk'
trinity_root = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Trinity'
red_root = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Red_River'
white_root = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/White_River'
powder_root = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Powder'
rio_root = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Rio_Grande_TX'
tombigbee_root = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Tombigbee'
brazos_root = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Brazos'
miss_root = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Mississippi_1'
misslc_root = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Mississippi'
sac_root = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Sacramento'
bev_root = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Beaver'

# Set-up main DataFrame

# Koyukuk
path = os.path.join(koyukuk_root, param_fn)
ky_paramdf = pandas.read_csv(path)
ky_paramdf['river'] = 'Koyukuk'

# Trinity
path = os.path.join(trinity_root, param_fn)
tr_paramdf = pandas.read_csv(path)
tr_paramdf['river'] = 'Trinity'

# Red
path = os.path.join(red_root, param_fn)
red_paramdf = pandas.read_csv(path)
red_paramdf['river'] = 'Red'

# White
path = os.path.join(white_root, param_fn)
wh_paramdf = pandas.read_csv(path)
wh_paramdf['river'] = 'White'

# Powder
path = os.path.join(powder_root, param_fn)
pr_paramdf = pandas.read_csv(path)
pr_paramdf['river'] = 'Powder'

# Rio
path = os.path.join(rio_root, param_fn)
rg_paramdf = pandas.read_csv(path)
rg_paramdf['river'] = 'Rio Grande'

# Tombigbee
path = os.path.join(tombigbee_root, param_fn)
to_paramdf = pandas.read_csv(path)
to_paramdf['river'] = 'Tombigbee'

# Brazos 
path = os.path.join(brazos_root, param_fn)
br_paramdf = pandas.read_csv(path)
br_paramdf['river'] = 'Brazos'

# Mississippi 
path = os.path.join(miss_root, param_fn)
ms_paramdf = pandas.read_csv(path)
ms_paramdf['river'] = 'Mississippi'

path = os.path.join(misslc_root, param_fn)
mslc_paramdf = pandas.read_csv(path)
mslc_paramdf['river'] = 'Mississippi - Leclair'

# Sacramento 
path = os.path.join(sac_root, param_fn)
sac_paramdf = pandas.read_csv(path)
sac_paramdf['river'] = 'Sacramento'

# Beaver 
path = os.path.join(bev_root, param_fn)
bev_paramdf = pandas.read_csv(path)
bev_paramdf['river'] = 'Nestucca'

paramdf = ky_paramdf.append(tr_paramdf)
paramdf = paramdf.append(red_paramdf)
paramdf = paramdf.append(wh_paramdf)
paramdf = paramdf.append(pr_paramdf)
paramdf = paramdf.append(rg_paramdf)
paramdf = paramdf.append(to_paramdf)
paramdf = paramdf.append(br_paramdf)
paramdf = paramdf.append(ms_paramdf)
paramdf = paramdf.append(mslc_paramdf)
paramdf = paramdf.append(sac_paramdf)
paramdf = paramdf.append(bev_paramdf)

# data
# Koyukuk
path = os.path.join(koyukuk_root, data_fn)
ky_datadf = pandas.read_csv(path)
ky_datadf['river'] = 'Koyukuk'

# Trinity
path = os.path.join(trinity_root, data_fn)
tr_datadf = pandas.read_csv(path)
tr_datadf['river'] = 'Trinity'

# Red
path = os.path.join(red_root, data_fn)
red_datadf = pandas.read_csv(path)
red_datadf['river'] = 'Red'

# White
path = os.path.join(white_root, data_fn)
wh_datadf = pandas.read_csv(path)
wh_datadf['river'] = 'White'

# Powder
path = os.path.join(powder_root, data_fn)
pr_datadf = pandas.read_csv(path)
pr_datadf['river'] = 'Powder'

#  Rio Grande
path = os.path.join(rio_root, data_fn)
rg_datadf = pandas.read_csv(path)
rg_datadf['river'] = 'Rio Grande'

# Tombigbee 
path = os.path.join(tombigbee_root, data_fn)
to_datadf = pandas.read_csv(path)
to_datadf['river'] = 'Tombigbee'

# Brazos 
path = os.path.join(brazos_root, data_fn)
br_datadf = pandas.read_csv(path)
br_datadf['river'] = 'Brazos'

# Mississippi 
path = os.path.join(miss_root, data_fn)
ms_datadf = pandas.read_csv(path)
ms_datadf['river'] = 'Mississippi'

path = os.path.join(misslc_root, data_fn)
mslc_datadf = pandas.read_csv(path)
mslc_datadf['river'] = 'Mississippi - Leclair'

# Sacramento 
path = os.path.join(sac_root, data_fn)
sac_datadf = pandas.read_csv(path)
sac_datadf['river'] = 'Sacramento'

# Nestucca 
path = os.path.join(bev_root, data_fn)
bev_datadf = pandas.read_csv(path)
bev_datadf['river'] = 'Nestucca'

datadf = ky_datadf.append(tr_datadf)
datadf = datadf.append(red_datadf)
datadf = datadf.append(wh_datadf)
datadf = datadf.append(pr_datadf)
datadf = datadf.append(rg_datadf)
datadf = datadf.append(to_datadf)
datadf = datadf.append(br_datadf)
datadf = datadf.append(ms_datadf)
datadf = datadf.append(mslc_datadf)
datadf = datadf.append(sac_datadf)
datadf = datadf.append(bev_datadf)

widthdf = datadf[['river', 'bar', 'idx', 'channel_width_dem']]
paramdf = paramdf[['river', 'bar', 'idx', 'k', 'X0', 'L']]

paramdf = paramdf.replace('False', numpy.nan)
paramdf['k'] = paramdf['k'].astype('float')
paramdf['X0'] = paramdf['X0'].astype('float')
paramdf['L'] = paramdf['L'].astype('float')

# Get all of the bar slopes
rivers = []
idxs = []
bars = []
thetas = []
for idx, row in paramdf.iterrows():

    row = row.dropna(how='any')

    # If there is no recorded bar slope skip
    if not row.get('k'):
        continue

    # there is no matching bar in the two dataframes skip
    cond = (
        (widthdf['river'] == row['river']) 
        & (widthdf['idx'] == row['idx'])
    )
    if len(widthdf[cond]) == 0:
        continue

    # Generate fit sigmoid from the actual bar
    ch_width = float(widthdf[cond]['channel_width_dem'])
    x = numpy.linspace((-1 * ch_width * 0.5), (ch_width * 0.5), 300)
    y = (sigmoid(x, row['L'], 0, abs(row['k'])))

    # Find the minimum value and where the objective channel bot is
    ydiff = numpy.where(numpy.diff(y) >= .01)
    ch_bot_i = ydiff[0][0] - 1
    ch_bot = x[ch_bot_i]

    # Find the x-position where I 
    slope_point = ch_bot + (.1 * ch_width)
    slope_x = closest(x, slope_point)
    slope_x_i = numpy.where(x == slope_x)[0][0]

    # If the position is at the end of the sigmoid
    if (slope_x_i == len(x)-1) or (slope_x_i == 0):
        continue
    
    # Find the slope
    dy = y[slope_x_i + 1] - y[slope_x_i -1]
    dx = x[slope_x_i + 1] - x[slope_x_i -1]
    slope = math.degrees(math.atan(dy/dx))

    rivers.append(row['river'])
    bars.append(row['bar'])
    idxs.append(row['idx'])
    thetas.append(slope)

# Set up data frame
thetasdf = pandas.DataFrame(data={
    'river': rivers,
    'bar': bars,
    'theta': thetas
})

# Set up statistics dataframe
theta_df = pandas.DataFrame(data={
    'minSlope': thetasdf.groupby(['river', 'bar']).min()['theta'],
    'medianSlope': thetasdf.groupby(['river', 'bar']).mean()['theta'],
    'maxSlope': thetasdf.groupby(['river', 'bar']).max()['theta'],
#    'StdSlope': thetasdf.groupby(['river', 'bar']).std()['theta'],
}).reset_index(drop=False)
theta_df.to_csv('thetadf.csv')

# Regression to find relationship between two
paramdf['k'] = abs(paramdf['k'])
param_bar = paramdf.dropna(
    how='any'
).groupby(
    ['river', 'bar']
).median().reset_index(drop=False)[['river', 'bar', 'k', 'L']]
theta_bar = theta_df[['river', 'bar', 'medianSlope']]

df = param_bar.merge(theta_bar, on=['river', 'bar'])
df['kL'] = df['k'] * df['L']
df['tan'] = numpy.tan(numpy.radians(df['medianSlope']))

X = numpy.array(df['kL']).reshape(-1, 1)
y = numpy.array(df['tan'])
reg = LinearRegression(fit_intercept=False).fit(X, y)
reg.score(X, y)

# Combine two mississippis
df.loc[df['river'] == 'Mississippi - Leclair', 'river'] = 'Mississippi'

river_df = df.groupby('river')
river_df = df.groupby('river')

# Plot figure 4c
x = numpy.linspace(.001, 5)
y = (x * reg.coef_[0])
plt.plot(x, x, color='black', linestyle='--')
plt.plot(x, y, color='black')
for name, group in river_df:
    print(name)
    ysem = group['tan'].std() / numpy.sqrt(len(group))
    xsem = group['kL'].std() / numpy.sqrt(len(group))
    plt.scatter(
        group['kL'].mean(), 
        group['tan'].mean(), 
        zorder=99, 
        facecolor='black', 
        edgecolor='black',
        s=60
    )
    plt.errorbar(
        group['kL'].mean(), 
        group['tan'].mean(), 
        ysem,
        xsem,
        ecolor='black'
    )
plt.xscale('log')
plt.yscale('log')

plt.ylim([.01, 1])
plt.xlim([.1, 5])

plt.xlabel('k * L [-]')
plt.ylabel('dz/dn')
plt.show()
