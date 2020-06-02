import pandas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde
import pymc3 as pm

from Visualizer import Visualizer

def posterior_distribution(X, y, N, fit_intercept=True):
    reg = LinearRegression().fit(X.values.reshape(-1, 1), y.values)
    # Set up
    with pm.Model() as model:
        # Intercept
        intercept = pm.Normal('Intercept', mu=float(reg.intercept_), sd=25)

        # Slope
        slope = pm.Normal('slope', mu=float(reg.coef_), sd=1)

        # Standard Deviation
        sigma = pm.HalfNormal('sigma', sd=25)

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

# bars_f = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/all_bar_data.csv'
bars_f = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/sampled_bar_average_data.csv'
all_f = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/total_data.csv'
lit_path = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Lit_values.csv'
bar_df = pandas.read_csv(bars_f)
ms_df = pandas.read_csv(all_f)

# platte_df = bar_df[bar_df.river=='Platte River']
# Get Rid of Platte
bar_df = bar_df[bar_df.river!='Platte River']
ms_df = ms_df[ms_df.river!='Platte River']

# bar_df = bar_df[bar_df.river!='Red River']
# ms_df = ms_df[ms_df.river!='Red River']

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

# Bar_df
Xms = ms_df.loc[:, 'bar_width']
yms = {
    'dem': ms_df.loc[:, 'channel_width_dem'],
    'water': ms_df.loc[:, 'channel_width_water'],
    'mean': ms_df.loc[:, 'mean_width'],
}

Xbar = bar_df.loc[:, 'bar_width']
ybar = {
    'dem': bar_df.loc[:, 'channel_width_dem'],
    'water': bar_df.loc[:, 'channel_width_water'],
    'mean': bar_df.loc[:, 'mean_width'],
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

# Group by river
group_river = ms_df.groupby('river')
group_bar = bar_df.groupby('river')

vh = Visualizer()
outpath = (
    '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/figures/data-v3.svg'
)

# Literature
lit_df = pandas.read_csv(lit_path)
# platte_data = {
#     'River': 'Platte River',
#     'Channel Width': platte_df['channel_width_mean'].mean(),
#     'Bar Width': platte_df['bar_width'].mean()
# }
# lit_df = lit_df.append(pandas.DataFrame(platte_data, index=[9]))
#vh.data_figure(
#    outpath,
#    group_river,
#    group_bar,
#    bar_intercept,
#    bar_coefs,
#    ms_intercept,
#    ms_coefs,
#    median_size=5,
#    alpha=0.25,
#    density_size=35
#)

# vh.data_figure_v2(
#     outpath,
#     group_river,
#     group_bar,
#     bar_coefs,
#     median_size=5,
#     alpha=0.25,
#     density_size=35,
#     fmt='png'
# )

vh.data_figure_v3(
    outpath,
    group_river,
    group_bar,
    bar_coefs,
    lit_df,
    median_size=5,
    alpha=0.25,
    density_size=35,
    fmt='svg',
#    bar_intercept=bar_intercept
)
