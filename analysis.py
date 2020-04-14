import pandas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde
import pymc3 as pm

from Visualizer import Visualizer

def posterior_distribution(X, y, N):
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
        mean = intercept + slope * X.values

        # Observed Values
        Y_obs = pm.Normal('Y_obs', mu=mean, sd=sigma, observed=y.values)

        # Sampler
        step = pm.NUTS()

        # Posterior distribution
        return pm.sample(N, step)


bars_f = '/Users/evangreenberg/PhD Documents/Projects/river-profiles/Output_Data/bar_average_data.csv'
all_f = '/Users/evangreenberg/PhD Documents/Projects/river-profiles/Output_Data/sampled_total_data.csv'
# all_f = '/Users/evangreenberg/PhD Documents/Projects/river-profiles/Output_Data/total_data.csv'
bar_df = pandas.read_csv(bars_f)
ms_df = pandas.read_csv(all_f)

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

# ms df trace
trace_ms = {}
for key, value in yms.items():
    trace_ms[key] = posterior_distribution(Xms, value, 1000)

# bar df trace
trace_bar = {}
for key, value in ybar.items():
    trace_bar[key] = posterior_distribution(Xbar, value, 1000)

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
    '/Users/evangreenberg/PhD Documents/Projects/river-profiles/figures/data.svg'
)
vh.data_figure(
    outpath,
    group_river,
    group_bar,
    bar_intercept,
    bar_coefs,
    ms_intercept,
    ms_coefs,
    median_size=5,
    alpha=0.25,
    density_size=35
)
