import math 

import itertools
import pandas
import numpy
from matplotlib import pyplot as plt
from scipy import stats

import scipy
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde
import pymc3 as pm
from sklearn.metrics import r2_score


def peakedness(df):
    wettest_month = max(df.groupby('month').mean()['discharge'])
    mean = df['discharge'].mean()

    return wettest_month / mean


def DVIa(df):
    wettest_month = max(df.groupby('month').mean()['discharge'])
    driest_month = min(df.groupby('month').mean()['discharge'])
    mean = df['discharge'].mean()

    return (wettest_month - driest_month) / mean


def DVIc(df):
    wettest_month = max(df.groupby('year-month').mean()['discharge'])
    driest_month = min(df.groupby('year-month').mean()['discharge'])
    mean = df['discharge'].mean()

    return (wettest_month - driest_month) / mean


def DVIy(df):
    sums = 0
    n = 0
    for name, group in df.groupby('year'):
        wet = max(group['discharge'])
        dry = min(group['discharge'])
        mean = group['discharge'].mean()
        calc = (wet - dry) / mean

        sums += calc 
        n += 1

    return sums / n


def getRsquared(slope, bar_df, intercept=None):

    return r2_score(bar_df['mean_width'], bar_df['pred'])


# Load the discharge data
paths = {
    'brazos': '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Discharge/GRDC/Brazos_Daily.txt',
    'miss': '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Discharge/GRDC/Mississippi_Daily.txt',
    'red': '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Discharge/GRDC/Red_Daily.txt',
    'sacramento': '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Discharge/GRDC/Sacramento_Daily.txt',
    'tombigbee': '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Discharge/GRDC/Tombigbee_Daily.txt',
    'trinity': '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Discharge/GRDC/Trinity_Daily.txt',
    'white': '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Discharge/GRDC/White_Daily.txt',
    'koyukuk': '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Discharge/GRDC/Koyukuk_Daily.txt'
}
# Load the data
dfs = {}
for river, path in paths.items():
    df = pandas.read_csv(
        path,
        sep=';',
        header=36
    )

    df['dt'] = pandas.to_datetime(df['YYYY-MM-DD'])
    df['discharge'] = df[' Value']
    df = df.replace(-999., numpy.nan)
    df = df.dropna(how='any')
    dfs[river] = df[['dt', 'discharge']]

paths_2 = {
    'nestucca': '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Discharge/Nestucca_beaver_1990_2020_.csv',
    'powder': '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Discharge/Powder_moorhead_Daily_discharge_1990',
}
# Load the data
dfs2 = {}
for river, path in paths_2.items():
    print(river)
    with open(path) as f:
        if len(f.readline().split('\t')) == 5:
            names = ['agency', 'site', 'dt', 'discharge', 'code']
        elif len(f.readline().split('\t')) == 6:
            names = ['agency', 'site', 'dt', 'tz', 'discharge', 'code']
        else:
            raise Exception

    dfs2[river] =  pandas.read_csv(
        path, 
        names=names, 
        sep='\t', 
        header=1
    )

# Clean up the dataframes so it's only what I care about
for river, df in dfs2.items():
    df = df[
        pandas.to_numeric(df['discharge'], errors='coerce').notnull()
    ]
    df['discharge'] = df['discharge'].astype(float)
    df['discharge'] = df['discharge'] * 0.0283168
    df['dt'] = pandas.to_datetime(df['dt'])
#    df = df[['dt', 'discharge']]
    dfs2[river] = df

# Make sure each dataframe is daily average
for river, df in dfs2.items():
    df = df.resample(
        'd', 
        on='dt'
    ).median().dropna(
        how='all'
    ).reset_index(
        drop=False
    )
    dfs2[river] = df

# Combine
for river, df in dfs2.items():
    dfs[river] = df

# Add month and day
for river, df in dfs.items():
    df['doy'] = [i.dayofyear for i in df['dt']]
    df['month'] = [i.month for i in df['dt']]
    df['year'] = [i.year for i in df['dt']]
    df['year-month'] = pandas.to_datetime(df['dt']).dt.to_period('M')
    dfs[river] = df

# Get stats
summary = {
    'river': [],
    'peakedness': [],
    'DVIa': [],
    'DVIc': [],
    'DVIy': [],
    'median': [],
    'std': [],
    'q85': [],
    'q15': [],
}
for river, df in dfs.items():
    summary['river'].append(river)
    summary['peakedness'].append(peakedness(df))
    summary['DVIa'].append(DVIa(df))
    summary['DVIc'].append(DVIc(df))
    summary['DVIy'].append(DVIy(df))
    summary['median'].append(df['discharge'].mean())
    summary['std'].append(df['discharge'].std())
    summary['q85'].append(numpy.quantile(df['discharge'], .85))
    summary['q15'].append(numpy.quantile(df['discharge'], .15))

sum_df = pandas.DataFrame(summary)
