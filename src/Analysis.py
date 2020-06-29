import math

import pandas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde
import pymc3 as pm
from sklearn.metrics import r2_score

from Visualizer import Visualizer

def posterior_distribution(X, y, N, fit_intercept=True):
    reg = LinearRegression(fit_intercept=fit_intercept).fit(X.values.reshape(-1, 1), y.values)
    # Set up
    with pm.Model() as model:
        # Intercept
        intercept = pm.Normal('Intercept', mu=float(reg.intercept_), sd=25)
        # sd = 25

        # Slope
        slope = pm.Normal('slope', mu=float(reg.coef_), sd=1)
        # sd = 1

        # Standard Deviation
        sigma = pm.HalfNormal('sigma', sd=25)
        # sd = 25

        # Estimate of Mean
        if not fit_intercept:
            mean = slope * X.values
        else:
            mean = intercept + slope * X.values

        # Observed Values
        Y_obs = pm.Normal('Y_obs', mu=mean, sd=sigma, observed=y.values)

        # Sampler
        step = pm.NUTS()

        # Posterior distribution
        return pm.sample(N, step)


def getRsquared(slope, bar_df, intercept=None):
    if intercept:
        bar_df['pred'] = (slope * bar_df['bar_width']) + intercept
    else:
        bar_df['pred'] = slope * bar_df['bar_width']

    return r2_score(bar_df['mean_width'], bar_df['pred'])


# bars_f = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/all_bar_data.csv'
bars_f = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/sampled_bar_average_data.csv'
all_f = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/total_data.csv'
lit_path = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Lit_values.csv'
bar_df = pandas.read_csv(bars_f)
ms_df = pandas.read_csv(all_f)

# Get Rid of Platte
bar_df = bar_df[bar_df.river!='Platte River']
ms_df = ms_df[ms_df.river!='Platte River']

# Get rid of tombigbee
# bar_df = bar_df[bar_df.river!='Tombigbee River']
# ms_df = ms_df[ms_df.river!='Tombigbee River']

# Get rid of Mississippi Leclair
bar_df = bar_df[bar_df.river!='Mississippi River - Leclair']
ms_df = ms_df[ms_df.river!='Mississippi River - Leclair']

bar_df = bar_df[bar_df.river!='Mississippi River']
ms_df = ms_df[ms_df.river!='Mississippi River']

# Clean
bar_df = bar_df[[
    'river',
    'bar', 
    'channel_width_dem', 
    'channel_width_water', 
    'bar_width'
]]
ms_df = ms_df[[
    'river',
    'bar', 
    'channel_width_dem', 
    'channel_width_water', 
    'bar_width'
]]

# Filter out Null
bar_df = bar_df.dropna(how='any')
ms_df = ms_df.dropna(how='any')
# Calculate Mean water width from both methods
ms_df['mean_width'] = (
    ms_df['channel_width_water'] + ms_df['channel_width_dem']
) / 2
bar_df['mean_width'] = (
    bar_df['channel_width_water'] + bar_df['channel_width_dem']
) / 2

# Generate the reach average dataframe
reach_df = bar_df.groupby('river').median().drop('bar', axis=1).reset_index()

# Do the log transforms
ms_df['log_bar_width'] = np.log(ms_df['bar_width'])
ms_df['log_mean_width'] = np.log(ms_df['mean_width'])
ms_df['log_dem_width'] = np.log(ms_df['channel_width_dem'])
ms_df['log_water_width'] = np.log(ms_df['channel_width_water'])

bar_df['log_bar_width'] = np.log(bar_df['bar_width'])
bar_df['log_mean_width'] = np.log(bar_df['mean_width'])
bar_df['log_dem_width'] = np.log(bar_df['channel_width_dem'])
bar_df['log_water_width'] = np.log(bar_df['channel_width_water'])

reach_df['log_bar_width'] = np.log(reach_df['bar_width'])
reach_df['log_mean_width'] = np.log(reach_df['mean_width'])
reach_df['log_dem_width'] = np.log(reach_df['channel_width_dem'])
reach_df['log_water_width'] = np.log(reach_df['channel_width_water'])

# Ms_df
Xms = ms_df.loc[:, 'bar_width']
yms = {
    'dem': ms_df.loc[:, 'channel_width_dem'],
    'water': ms_df.loc[:, 'channel_width_water'],
    'mean': ms_df.loc[:, 'mean_width'],
}

# Bar_df
Xbar = bar_df.loc[:, 'bar_width']
ybar = {
    'dem': bar_df.loc[:, 'channel_width_dem'],
    'water': bar_df.loc[:, 'channel_width_water'],
    'mean': bar_df.loc[:, 'mean_width'],
}

# Reach_df
Xreach = reach_df.loc[:, 'bar_width']
yreach = {
    'dem': reach_df.loc[:, 'channel_width_dem'],
    'water': reach_df.loc[:, 'channel_width_water'],
    'mean': reach_df.loc[:, 'mean_width'],
}

fit_intercept = False
# ms df trace
trace_ms = {}
for key, value in yms.items():
    trace_ms[key] = posterior_distribution(Xms, value, 3000, fit_intercept)

# bar df trace
trace_bar = {}
for key, value in ybar.items():
    trace_bar[key] = posterior_distribution(Xbar, value, 3000, fit_intercept)

# reach df trace
trace_reach = {}
for key, value in yreach.items():
    trace_reach[key] = posterior_distribution(Xreach, value, 3000, fit_intercept)

# Coef estimation
ms_coefs = {}
for key, value in trace_ms.items():
    ms_coefs[key] = {
        '5': np.quantile(value['slope'], 0.05),
        '50': np.quantile(value['slope'], 0.5),
        '95': np.quantile(value['slope'], 0.95)
    }
bar_coefs = {}
for key, value in trace_bar.items():
    bar_coefs[key] = {
        '5': np.quantile(value['slope'], 0.05),
        '50': np.quantile(value['slope'], 0.5),
        '95': np.quantile(value['slope'], 0.95)
    }
reach_coefs = {}
for key, value in trace_reach.items():
    reach_coefs[key] = {
        '5': np.quantile(value['slope'], 0.05),
        '50': np.quantile(value['slope'], 0.5),
        '95': np.quantile(value['slope'], 0.95)
    }

# Intercept Estimation
ms_intercept = {}
for key, value in trace_ms.items():
    ms_intercept[key] = {
        '5': np.quantile(value['Intercept'], 0.05),
        '50': np.quantile(value['Intercept'], 0.5),
        '95': np.quantile(value['Intercept'], 0.95)
    }
bar_intercept = {}
for key, value in trace_bar.items():
    bar_intercept[key] = {
        '5': np.quantile(value['Intercept'], 0.05),
        '50': np.quantile(value['Intercept'], 0.5),
        '95': np.quantile(value['Intercept'], 0.95)
    }
reach_intercept = {}
for key, value in trace_reach.items():
    reach_intercept[key] = {
        '5': np.quantile(value['Intercept'], 0.05),
        '50': np.quantile(value['Intercept'], 0.5),
        '95': np.quantile(value['Intercept'], 0.95)
    }

# Group by river
group_river = ms_df.groupby('river')
group_bar = bar_df.groupby('river')

vh = Visualizer()
outpath = (
    '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/figures/62420_data-v3.svg'
)

# Literature
lit_df = pandas.read_csv(lit_path)
# platte_data = {
#     'River': 'Platte River',
#     'Channel Width': platte_df['channel_width_mean'].mean(),
#     'Bar Width': platte_df['bar_width'].mean()
# }
# lit_df = lit_df.append(pandas.DataFrame(platte_data, index=[9]))

vh.data_figure(
    outpath,
    group_river,
    group_bar,
    bar_coefs,
    reach_coefs,
    lit_df,
    median_size=5,
    alpha=0.25,
    density_size=35,
    fmt='svg',
#    bar_intercept=bar_intercept
)

median_df = group_bar.median() 
allr2 = getRsquared(bar_coefs['mean']['50'], bar_df)
medr2 = getRsquared(bar_coefs['mean']['50'], median_df)
