import os
import math

import pandas
from matplotlib import pyplot as plt
import numpy


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
sac_root = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Sacramento'
bev_root = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Beaver'

# Parameters
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
datadf = datadf.append(sac_datadf)
datadf = datadf.append(bev_datadf)

widthdf = datadf[['river', 'bar', 'idx', 'channel_width_mean']]
paramdf = paramdf[['river', 'bar', 'idx', 'k', 'X0', 'L']]

paramdf = paramdf.replace('False', numpy.nan)
paramdf['k'] = paramdf['k'].astype('float')
paramdf['X0'] = paramdf['X0'].astype('float')
paramdf['L'] = paramdf['L'].astype('float')

# Get all of the slopes
rivers = []
idxs = []
bars = []
thetas = []
for idx, row in paramdf.iterrows():
    cond = (
        (widthdf['river'] == row['river']) 
        & (widthdf['idx'] == row['idx'])
    )
    if len(widthdf[cond]) == 0:
        continue

    ch_width = float(widthdf[cond]['channel_width_mean'])

    x = numpy.linspace((-1 * ch_width * 0.1), (ch_width * 0.1), 3)
    y = (sigmoid(x, row['L'], 0, abs(row['k'])))

    rivers.append(row['river'])
    bars.append(row['bar'])
    idxs.append(row['idx'])
    thetas.append(math.degrees(math.atan(
        (y[2] - y[0])
        / (x[2] - x[0])
    )))
thetasdf = pandas.DataFrame(data={
    'river': rivers,
    'bar': bars,
    'theta': thetas
})

theta_df = pandas.DataFrame(data={
    'minSlope': thetasdf.groupby(['river', 'bar']).min()['theta'],
    'medianSlope': thetasdf.groupby(['river', 'bar']).median()['theta'],
    'maxSlope': thetasdf.groupby(['river', 'bar']).max()['theta'],
    'StdSlope': thetasdf.groupby(['river', 'bar']).std()['theta'],
}).reset_index(drop=False)
theta_df.to_csv('thetadf.csv')

# Get slopes
rivers = []
thetas = []
thetas_dml = []
for idx, row in paramdf.iterrows():
    cond = (
        (widthdf['river'] == row['river']) 
        & (widthdf['idx'] == row['idx'])
    )
    ch_width = widthdf.loc[cond]['channel_width_mean']
    if len(ch_width) == 0:
        continue
    ch_width = float(ch_width)
    x = numpy.linspace((-1 * ch_width * 0.1), (ch_width * 0.1), 3)
    y = (sigmoid(x, row['L'], 0, abs(row['k'])))
    x_dml = x / ch_width
    y_dml = y / row['L']
    thetas.append(math.degrees(math.atan(
        (y[2] - y[0])
        / (x[2] - x[0])
    )))
    thetas_dml.append(math.degrees(math.atan(
        (y_dml[2] - y_dml[0])
        / (x_dml[2] - x_dml[0])
    )))
    rivers.append(row['river'])
theta_df = pandas.DataFrame(data={'river': rivers, 'theta': thetas, 'theta_dml': thetas_dml})
thetasdf.to_csv('thetadf.csv')
