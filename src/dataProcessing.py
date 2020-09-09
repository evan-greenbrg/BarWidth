import math
import pandas
from matplotlib import pyplot as plt
import numpy as np


def sample_bars(df, n):
    """
    Sample the bars dataframe by rows
    """
    return df.sample(n)


def sample_sections(df, n):
    """
    Samples so there is same number of cross-sections in each bar
    """
    bar = df.groupby('bar')
    final_df = pandas.DataFrame()
    for name, group in bar:
        lens = len(group)
        step = int(round(lens / n, 0))
        if step < 1:
            step = 1
        group = group.iloc[::step]
        final_df = final_df.append(group)

    return final_df.reset_index(drop=True)


def get_downstream_distance(bargroup):
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


def get_normalized(bargroup, widthcol):
    df = pandas.DataFrame()
    for name, group in bargroup:
        median_width = group[widthcol].median()
        max_distance = group['distance'].max()
        widths = []
        distances = []
        for idx, row in group.iterrows():
            if row['distance'] == max_distance:
                distances.append(1)
            else:
                distances.append(row['distance'] / max_distance)
            widths.append(row[widthcol] / median_width)
        group['normalized_distance'] = distances
        group['normalized_width'] = widths
        df = df.append(group)

    return df


col_list = [
    'channel_width_dem', 
    'channel_width_water', 
    'channel_width_mean', 
    'bar_width'
]
rename_list = {
    'channel_width_dem': 'channel_width_dem_std',
    'channel_width_water': 'channel_width_water_std',
    'channel_width_mean': 'channel_width_mean_std',
    'bar_width': 'bar_width_std'
}

n = 15
m = 10
# Red River
red = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Red_River/bar_data.csv'
red_df = pandas.read_csv(red)
red_df = red_df[red_df['bar_width'] != 'False']
red_df['bar_width'] = red_df['bar_width'].astype(float)
red_df = red_df[red_df['bar_width'] > 0]
red_df = get_downstream_distance(red_df.groupby('bar'))

# Bar
red_bar_mean = red_df.groupby('bar').mean()
red_bar_mean = red_bar_mean[col_list]
ns = [math.sqrt(x) for x in red_df.groupby('bar').count()['distance']]
red_bar_std = red_df.groupby('bar').sem()
red_bar_std = red_bar_std[col_list].rename(columns=rename_list)
red_bars = red_bar_mean.merge(red_bar_std, on='bar')

red_df_sample = sample_sections(red_df, n)
red_df_bsample = sample_bars(red_bars, m)

# Trinity River
trinity = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Trinity/bar_data.csv'
trinity_df = pandas.read_csv(trinity)
trinity_df = trinity_df[trinity_df['bar_width'] != 'False']
trinity_df['bar_width'] = trinity_df['bar_width'].astype(float)
trinity_df = trinity_df[trinity_df['bar_width'] > 0]
trinity_df = get_downstream_distance(trinity_df.groupby('bar'))

# Bars
trinity_bar_mean = trinity_df.groupby('bar').mean()
trinity_bar_mean = trinity_bar_mean[col_list]
trinity_bar_std = trinity_df.groupby('bar').sem()
trinity_bar_std = trinity_bar_std[col_list].rename(columns=rename_list)
trinity_bars = trinity_bar_mean.merge(trinity_bar_std, on='bar')

trinity_df_sample = sample_sections(trinity_df, n)
trinity_df_bsample = sample_bars(trinity_bars, m)

# Koyukuk River
koyukuk = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Koyukuk/bar_data.csv'
koyukuk_df = pandas.read_csv(koyukuk)
koyukuk_df = koyukuk_df[koyukuk_df['bar_width'] != 'False']
koyukuk_df['bar_width'] = koyukuk_df['bar_width'].astype(float)
koyukuk_df = koyukuk_df[koyukuk_df['bar_width'] > 0]
koyukuk_df = get_downstream_distance(koyukuk_df.groupby('bar'))

# Bars
koyukuk_bar_mean = koyukuk_df.groupby('bar').mean()
koyukuk_bar_mean = koyukuk_bar_mean[col_list]
koyukuk_bar_std = koyukuk_df.groupby('bar').sem()
koyukuk_bar_std = koyukuk_bar_std[col_list].rename(columns=rename_list)
koyukuk_bars = koyukuk_bar_mean.merge(koyukuk_bar_std, on='bar')

koyukuk_df_sample = sample_sections(koyukuk_df, n)
koyukuk_df_bsample = sample_bars(koyukuk_bars, m)

# Platte River
platte = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Platte/bar_data.csv'
platte_df = pandas.read_csv(platte)
platte_df = platte_df[platte_df['bar_width'] > 0]
platte_df_df = get_downstream_distance(platte_df.groupby('bar'))

# Bars
platte_bar_mean = platte_df.groupby('bar').mean()
platte_bar_mean = platte_bar_mean[col_list]
platte_bar_std = platte_df.groupby('bar').sem()
platte_bar_std = platte_bar_std[col_list].rename(columns=rename_list)
platte_bars = platte_bar_mean.merge(platte_bar_std, on='bar')

platte_bars = platte_df.groupby('bar').mean()
platte_df_sample = sample_sections(platte_df, n)
platte_df_bsample = platte_bars

platte_df['channel_width_water'] = platte_df['channel_width_dem']
platte_bars['channel_width_water'] = platte_bars['channel_width_dem']
platte_df_sample['channel_width_water'] = platte_df_sample['channel_width_dem']

# White River
white = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/White_River/bar_data.csv'
white_df = pandas.read_csv(white)
white_df = white_df[white_df['bar_width'] != 'False']
white_df['bar_width'] = white_df['bar_width'].astype(float)
white_df['bar_height'] = white_df['bar_height'].astype(float)
white_df = white_df[white_df['bar_width'] > 0]
white_df = get_downstream_distance(white_df.groupby('bar'))

# Convert white river to m
to_change = [
    'channel_width_dem', 
    'channel_width_water', 
    'bar_width', 
    'bar_height', 
]
for change in to_change:
    white_df[change] = white_df[change] * 0.3048

# Bars
white_bar_mean = white_df.groupby('bar').mean()
white_bar_mean = white_bar_mean[col_list]
white_bar_std = white_df.groupby('bar').sem()
white_bar_std = white_bar_std[col_list].rename(columns=rename_list)
white_bars = white_bar_mean.merge(white_bar_std, on='bar')

white_df_sample = sample_sections(white_df, n)
white_df_bsample = white_bars

# Powder River
powder = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Powder/bar_data.csv'
powder_df = pandas.read_csv(powder)
powder_df['bar_width'] = powder_df['bar_width'].astype(float)
powder_df = powder_df[powder_df['bar_width'] > 0]
powder_df = get_downstream_distance(powder_df.groupby('bar'))

# Bars
powder_bar_mean = powder_df.groupby('bar').mean()
powder_bar_mean = powder_bar_mean[col_list]
powder_bar_std = powder_df.groupby('bar').sem()
powder_bar_std = powder_bar_std[col_list].rename(columns=rename_list)
powder_bars = powder_bar_mean.merge(powder_bar_std, on='bar')

powder_df_sample = sample_sections(powder_df, n)
powder_df_bsample = powder_bars

# Mississippi River
mississippi = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Mississippi/bar_data.csv'
miss_df = pandas.read_csv(mississippi)
miss_df['bar_width'] = miss_df['bar_width'].astype(float)
miss_df = miss_df[miss_df['bar_width'] > 0]
miss_df = get_downstream_distance(miss_df.groupby('bar'))

# Bars
miss_bar_mean = miss_df.groupby('bar').mean()
miss_bar_mean = miss_bar_mean[col_list]
miss_bar_std = miss_df.groupby('bar').sem()
miss_bar_std = miss_bar_std[col_list].rename(columns=rename_list)
miss_bars = miss_bar_mean.merge(miss_bar_std, on='bar')

miss_df_sample = miss_df
miss_df_bsample = miss_bars

# Mississippi River
mississippi_1 = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Mississippi_1/bar_data.csv'
miss1_df = pandas.read_csv(mississippi_1)
miss1_df = miss1_df[miss1_df['bar_width'] != 'False']
miss1_df['bar_width'] = miss1_df['bar_width'].astype(float)
miss1_df = miss1_df[miss1_df['bar_width'] > 0]
miss1_df = get_downstream_distance(miss1_df.groupby('bar'))

# Bars
miss1_bar_mean = miss1_df.groupby('bar').mean()
miss1_bar_mean = miss1_bar_mean[col_list]
miss1_bar_std = miss1_df.groupby('bar').sem()
miss1_bar_std = miss1_bar_std[col_list].rename(columns=rename_list)
miss1_bars = miss1_bar_mean.merge(miss1_bar_std, on='bar')

miss1_df_sample = sample_sections(miss1_df, n)
miss1_df_bsample = miss1_bars

# Brazos
brazos = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Brazos/bar_data.csv'
brazos_df = pandas.read_csv(brazos)
brazos_df = brazos_df[brazos_df['bar_width'] != 'False']
brazos_df['bar_width'] = brazos_df['bar_width'].astype(float)
brazos_df = brazos_df[brazos_df['bar_width'] > 0]
brazos_df = get_downstream_distance(brazos_df.groupby('bar'))

# Bars
brazos_bar_mean = brazos_df.groupby('bar').mean()
brazos_bar_mean = brazos_bar_mean[col_list]
brazos_bar_std = brazos_df.groupby('bar').sem()
brazos_bar_std = brazos_bar_std[col_list].rename(columns=rename_list)
brazos_bars = brazos_bar_mean.merge(brazos_bar_std, on='bar')

brazos_df_sample = sample_sections(brazos_df, n)
brazos_df_bsample = brazos_bars

# Tombigbee 
tombigbee = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Tombigbee/bar_data.csv'
tom_df = pandas.read_csv(tombigbee)
tom_df = tom_df[tom_df['bar_width'] != 'False']
tom_df['bar_width'] = tom_df['bar_width'].astype(float)
tom_df = tom_df[tom_df['bar_width'] > 0]
tom_df = get_downstream_distance(tom_df.groupby('bar'))

# Bars
tom_bar_mean = tom_df.groupby('bar').mean()
tom_bar_mean = tom_bar_mean[col_list]
tom_bar_std = tom_df.groupby('bar').sem()
tom_bar_std = tom_bar_std[col_list].rename(columns=rename_list)
tom_bars = tom_bar_mean.merge(tom_bar_std, on='bar')

tom_df_sample = sample_sections(tom_df, n)
tom_df_bsample = tom_bars

# Rio Grande TX 
rio = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Rio_Grande_TX/bar_data.csv'
rio_df = pandas.read_csv(rio)
rio_df = rio_df[rio_df['bar_width'] != 'False']
rio_df['bar_width'] = rio_df['bar_width'].astype(float)
rio_df = rio_df[rio_df['bar_width'] > 0]
rio_df = get_downstream_distance(rio_df.groupby('bar'))

# Bars
rio_bar_mean = rio_df.groupby('bar').mean()
rio_bar_mean = rio_bar_mean[col_list]
rio_bar_std = rio_df.groupby('bar').sem()
rio_bar_std = rio_bar_std[col_list].rename(columns=rename_list)
rio_bars = rio_bar_mean.merge(rio_bar_std, on='bar')

rio_df_sample = sample_sections(rio_df, n)
rio_df_bsample = sample_bars(rio_bars, m)

# Sacramento 
sac = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Sacramento/bar_data.csv'
sac_df = pandas.read_csv(sac)
sac_df = sac_df[sac_df['bar_width'] != 'False']
sac_df['bar_width'] = sac_df['bar_width'].astype(float)
sac_df = sac_df[sac_df['bar_width'] > 0]
sac_df = get_downstream_distance(sac_df.groupby('bar'))

# Bars
sac_bar_mean = sac_df.groupby('bar').mean()
sac_bar_mean = sac_bar_mean[col_list]
sac_bar_std = sac_df.groupby('bar').sem()
sac_bar_std = sac_bar_std[col_list].rename(columns=rename_list)
sac_bars = sac_bar_mean.merge(sac_bar_std, on='bar')

sac_df_sample = sample_sections(sac_df, n)
sac_df_bsample = sac_bars

# Nestucca
bev = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Beaver/bar_data.csv'
bev_df = pandas.read_csv(bev)
bev_df = bev_df[bev_df['bar_width'] != 'False']
bev_df['bar_width'] = bev_df['bar_width'].astype(float)
bev_df = bev_df[bev_df['bar_width'] > 0]
bev_df = get_downstream_distance(bev_df.groupby('bar'))

# Bars
bev_bar_mean = bev_df.groupby('bar').mean()
bev_bar_mean = bev_bar_mean[col_list]
bev_bar_std = bev_df.groupby('bar').sem()
bev_bar_std = bev_bar_std[col_list].rename(columns=rename_list)
bev_bars = bev_bar_mean.merge(bev_bar_std, on='bar')

bev_df_sample = sample_sections(bev_df, n)
bev_df_bsample = sample_bars(bev_bars, m)

# Bars DF
red_bars['river'] = 'Red River'
trinity_bars['river'] = 'Trinity River'
koyukuk_bars['river'] = 'Koyukuk River'
platte_bars['river'] = 'Platte River'
white_bars['river'] = 'White River'
powder_bars['river'] = 'Powder River'
miss_bars['river'] = 'Mississippi River - Lecliar'
miss1_bars['river'] = 'Mississippi River'
brazos_bars['river'] = 'Brazos River'
tom_bars['river'] = 'Tombigbee River'
rio_bars['river'] = 'Rio Grande River'
sac_bars['river'] = 'Sacramento River'
bev_bars['river'] = 'Nestucca River'

bars_df = red_bars.append(trinity_bars)
bars_df = bars_df.append(koyukuk_bars)
bars_df = bars_df.append(platte_bars)
bars_df = bars_df.append(white_bars)
bars_df = bars_df.append(powder_bars)
bars_df = bars_df.append(miss_bars)
bars_df = bars_df.append(miss1_bars)
bars_df = bars_df.append(brazos_bars)
bars_df = bars_df.append(tom_bars)
bars_df = bars_df.append(rio_bars)
bars_df = bars_df.append(sac_bars)
bars_df = bars_df.append(bev_bars)

# Measurement DF
red_df['river'] = 'Red River'
trinity_df['river'] = 'Trinity River'
koyukuk_df['river'] = 'Koyukuk River'
platte_df['river'] = 'Platte River'
white_df['river'] = 'White River'
powder_df['river'] = 'Powder River'
miss_df['river'] = 'Mississippi River - Leclair'
miss1_df['river'] = 'Mississippi River'
brazos_df['river'] = 'Brazos River'
tom_df['river'] = 'Tombigbee River'
rio_df['river'] = 'Rio Grande River'
sac_df['river'] = 'Sacramento River'
bev_df['river'] = 'Nestucca River'

ms_df = red_df.append(trinity_df)
ms_df = ms_df.append(koyukuk_df)
ms_df = ms_df.append(platte_df)
ms_df = ms_df.append(white_df)
ms_df = ms_df.append(powder_df)
ms_df = ms_df.append(miss_df)
ms_df = ms_df.append(miss1_df)
ms_df = ms_df.append(brazos_df)
ms_df = ms_df.append(tom_df)
ms_df = ms_df.append(rio_df)
ms_df = ms_df.append(sac_df)
ms_df = ms_df.append(bev_df)

# Sample DF
red_df_sample['river'] = 'Red River'
trinity_df_sample['river'] = 'Trinity River'
koyukuk_df_sample['river'] = 'Koyukuk River'
platte_df_sample['river'] = 'Platte River'
white_df_sample['river'] = 'White River'
powder_df_sample['river'] = 'Powder River'
miss_df_sample['river'] = 'Mississippi River - Leclair'
miss1_df_sample['river'] = 'Mississippi River'
brazos_df_sample['river'] = 'Brazos River'
tom_df_sample['river'] = 'Tombigbee River'
rio_df_sample['river'] = 'Rio Grande River'
sac_df_sample['river'] = 'Sacramento River'
bev_df_sample['river'] = 'Nestucca River'

red_df_bsample['river'] = 'Red River'
trinity_df_bsample['river'] = 'Trinity River'
koyukuk_df_bsample['river'] = 'Koyukuk River'
platte_df_bsample['river'] = 'Platte River'
white_df_bsample['river'] = 'White River'
powder_df_bsample['river'] = 'Powder River'
miss_df_bsample['river'] = 'Mississippi River - Leclair'
miss1_df_bsample['river'] = 'Mississippi River'
brazos_df_bsample['river'] = 'Brazos River'
tom_df_bsample['river'] = 'Tombigbee River'
rio_df_bsample['river'] = 'Rio Grande River'
sac_df_bsample['river'] = 'Sacramento River'
bev_df_bsample['river'] = 'Nestucca River'

sample_df = red_df_sample.append(trinity_df_sample)
sample_df = sample_df.append(koyukuk_df_sample)
sample_df = sample_df.append(platte_df_sample)
sample_df = sample_df.append(white_df_sample)
sample_df = sample_df.append(powder_df_sample)
sample_df = sample_df.append(miss_df_sample)
sample_df = sample_df.append(miss1_df_sample)
sample_df = sample_df.append(brazos_df_sample)
sample_df = sample_df.append(tom_df_sample)
sample_df = sample_df.append(rio_df_sample)
sample_df = sample_df.append(sac_df_sample)
sample_df = sample_df.append(bev_df_sample)

sample_bar_df = red_df_bsample.append(trinity_df_bsample)
sample_bar_df = sample_bar_df.append(koyukuk_df_bsample)
sample_bar_df = sample_bar_df.append(platte_df_bsample)
sample_bar_df = sample_bar_df.append(white_df_bsample)
sample_bar_df = sample_bar_df.append(white_df_bsample)
sample_bar_df = sample_bar_df.append(powder_df_bsample)
sample_bar_df = sample_bar_df.append(miss_df_bsample)
sample_bar_df = sample_bar_df.append(miss1_df_bsample)
sample_bar_df = sample_bar_df.append(brazos_df_bsample)
sample_bar_df = sample_bar_df.append(tom_df_bsample)
sample_bar_df = sample_bar_df.append(rio_df_bsample)
sample_bar_df = sample_bar_df.append(sac_df_bsample)
sample_bar_df = sample_bar_df.append(bev_df_bsample)

# normalized
ms_df = get_normalized(ms_df.groupby(['river', 'bar']), 'channel_width_water')
sample_df = get_normalized(sample_df.groupby(['river', 'bar']), 'channel_width_water')

# Export the data
ms_df.to_csv('/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/total_data.csv')
bars_df.to_csv('/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/bar_average_data.csv')
sample_df.to_csv('/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/sampled_total_data.csv')
sample_bar_df.to_csv('/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/sampled_bar_average_data.csv')
