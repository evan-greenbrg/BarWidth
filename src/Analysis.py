import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pymc3 as pm
from sklearn.metrics import r2_score

from Visualizer import Visualizer


def posterior_distribution(X, y, N, fit_intercept=False, fit_slope=True):
    reg = LinearRegression(
        fit_intercept=fit_intercept
    ).fit(
        X.values.reshape(-1, 1),
        y.values
    )

    # Set up
    with pm.Model() as model:
        # Intercept
        intercept = pm.Normal('Intercept', mu=1, sd=1000)
        # sd = 25

        # Slope
        slope = pm.Normal('slope', mu=float(reg.coef_), sd=1000)
        # sd = 1

        # Standard Deviation
        sigma = pm.HalfNormal('sigma', sd=1000)
        # sd = 25

        # Estimate of Mean
        if fit_slope and not fit_intercept:
            mean = slope * X.values
        if fit_intercept and not fit_slope:
            mean = intercept + X.values
        else:
            mean = intercept + (slope * X.values)

        # Observed Values
        Y_obs = pm.Normal('Y_obs', mu=mean, sd=sigma, observed=y.values)

        # Sampler
        step = pm.NUTS()

        # Posterior distribution
        return pm.sample(N, step)


def getRsquared(slope, bar_df, log=False, intercept=None):
    if intercept:
        if log:
            bar_df['pred'] = (slope * bar_df['bar_width']) + intercept
        else:
            bar_df['pred'] = intercept * (bar_df['log_bar_width']**slope)
    else:
        if log:
            bar_df['pred'] = slope * bar_df['bar_width']
        else:
            bar_df['pred'] = (bar_df['log_bar_width']**slope)

    return r2_score(bar_df['mean_width'], bar_df['pred'])


bars_f = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/sampled_bar_average_data.csv'
# use pm.df_summary(normal_trace) for unsampled
all_f = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/total_data.csv'
lit_path = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Lit_values.csv'
bar_df = pandas.read_csv(bars_f)
ms_df = pandas.read_csv(all_f)

# Get Rid of Platte
bar_df = bar_df[bar_df.river != 'Platte River']
ms_df = ms_df[ms_df.river != 'Platte River']

# Clean df
bar_df = bar_df[[
    'river',
    'bar',
    'channel_width_dem',
    'channel_width_water',
    'bar_width',
    'bar_width_std',
    'channel_width_mean_std'
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
ms_df['log_bar_width'] = np.log10(ms_df['bar_width'])
ms_df['log_mean_width'] = np.log10(ms_df['mean_width'])
ms_df['log_dem_width'] = np.log10(ms_df['channel_width_dem'])
ms_df['log_water_width'] = np.log10(ms_df['channel_width_water'])

bar_df['log_bar_width'] = np.log10(bar_df['bar_width'])
bar_df['log_mean_width'] = np.log10(bar_df['mean_width'])
bar_df['log_dem_width'] = np.log10(bar_df['channel_width_dem'])
bar_df['log_water_width'] = np.log10(bar_df['channel_width_water'])

reach_df['log_bar_width'] = np.log10(reach_df['bar_width'])
reach_df['log_mean_width'] = np.log10(reach_df['mean_width'])
reach_df['log_dem_width'] = np.log10(reach_df['channel_width_dem'])
reach_df['log_water_width'] = np.log10(reach_df['channel_width_water'])

# Construct X and Y arrays
log = False
if log:
    Xms = ms_df.loc[:, 'log_bar_width']
    yms = {
        'dem': ms_df.loc[:, 'log_dem_width'],
        'water': ms_df.loc[:, 'log_water_width'],
        'mean': ms_df.loc[:, 'log_mean_width'],
    }

    # Bar_df
    Xbar = bar_df.loc[:, 'log_bar_width']
    ybar = {
        'dem': bar_df.loc[:, 'log_dem_width'],
        'water': bar_df.loc[:, 'log_water_width'],
        'mean': bar_df.loc[:, 'log_mean_width'],
    }

    # Reach_df
    Xreach = reach_df.loc[:, 'log_bar_width']
    yreach = {
        'dem': reach_df.loc[:, 'log_dem_width'],
        'water': reach_df.loc[:, 'log_water_width'],
        'mean': reach_df.loc[:, 'log_mean_width'],
    }
else:
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

# Fit the Bayesian Model
fit_intercept = False
fit_slope = True

# All data points
trace_ms = {}
for key, value in yms.items():
    trace_ms[key] = posterior_distribution(
        Xms,
        value,
        1000,
        fit_intercept,
        fit_slope
    )

# Bar-averaged data points
trace_bar = {}
for key, value in ybar.items():
    trace_bar[key] = posterior_distribution(
        Xbar,
        value,
        1000,
        fit_intercept,
        fit_slope
    )

# Reach-averaged data points
trace_reach = {}
for key, value in yreach.items():
    trace_reach[key] = posterior_distribution(
        Xreach,
        value,
        1000,
        fit_intercept,
        fit_slope
    )

# Estimated uncertainties on the coefficients
if not log:
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

elif log:
    # Estimate uncertainties on the intercept (won't be relevent)
    ms_intercept = {}
    for key, value in trace_ms.items():
        ms_intercept[key] = {
            '5': 10**np.quantile(value['Intercept'], 0.05),
            '50': 10**np.quantile(value['Intercept'], 0.5),
            '95': 10**np.quantile(value['Intercept'], 0.95)
        }
    bar_intercept = {}
    for key, value in trace_bar.items():
        bar_intercept[key] = {
            '5': 10**np.quantile(value['Intercept'], 0.05),
            '50': 10**np.quantile(value['Intercept'], 0.5),
            '95': 10**np.quantile(value['Intercept'], 0.95)
        }
    reach_intercept = {}
    for key, value in trace_reach.items():
        reach_intercept[key] = {
            '5': 10**np.quantile(value['Intercept'], 0.05),
            '50': 10**np.quantile(value['Intercept'], 0.5),
            '95': 10**np.quantile(value['Intercept'], 0.95)
        }

# Group data by river
ms_group = ms_df.groupby(['river', 'bar'])
group_river = ms_df.groupby('river')
group_bar = bar_df.groupby('river')

# Initialize the visulizer class
vh = Visualizer()
outpath = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/figures/62420_data-v3.svg'

# Read in literature data
lit_df = pandas.read_csv(lit_path)

###############################
###         FIGURE 3ac      ###
###############################
vh.data_figure(
    outpath,
    ms_df,
    group_river,
    group_bar,
    lit_df,
    median_size=15,
    alpha=0.25,
    density_size=35,
    bar_coefs=bar_coefs,
    reach_coefs=reach_coefs,
    #    bar_intercept=bar_intercept,
    #    reach_intercept=reach_intercept,
    fmt='svg',
    log=False
)


# Load ancient Values
data = {
    'bar_width': [124, 11.143, 40],
    'channel_width': [301, 23.21, 63]
}
ancient_df = pandas.DataFrame(data)

# Set up predicted series for Figure 3bd
i = '50'
s = '50'
if not log:
    #    Normal
    ms_df['predicted'] = (
        ms_df['bar_width']*ms_coefs['mean'][s]
    )
    ms_df['predicted_er'] = (
        ms_df['bar_width']*(ms_coefs['mean']['50'] - ms_coefs['mean']['5'])
    )
    reach_df['predicted'] = (
        reach_df['bar_width']*ms_coefs['mean'][s]
    )
    bar_df['predicted'] = (
        bar_df['bar_width']*ms_coefs['mean'][s]
    )
    lit_df['predicted'] = (
        lit_df['Bar Width']*ms_coefs['mean'][s]
    )
    ancient_df['predicted'] = ancient_df['bar_width'] * ms_coefs['mean'][s]
elif log:
    # Log
    ms_df['predicted'] = (
        ms_intercept['mean'][i]
        * (10**(ms_df['log_bar_width'])**ms_coefs['mean'][s])
    )
    reach_df['predicted'] = (
        ms_intercept['mean'][i]
        * (10**(reach_df['log_bar_width'])**ms_coefs['mean'][s])
    )
    bar_df['predicted'] = (
        ms_intercept['mean'][i]
        * (10**(bar_df['log_bar_width'])**ms_coefs['mean'][s])
    )
    lit_df['predicted'] = (
        ms_intercept['mean'][i]
        * ((lit_df['Bar Width'])**ms_coefs['mean'][s])
    )

# Combine literature and collected data
lit_df = lit_df.dropna(how='any')
lit_df['mean_width'] = lit_df['Channel Width']
lit_df['river'] = lit_df['River']
lit_stack = lit_df[['river', 'mean_width', 'predicted']]
reach_stack = reach_df[['river', 'mean_width', 'predicted']]
combo_df = reach_stack.append(lit_stack)

# Predict with other scaling
bar_df['SchummPred'] = bar_df['bar_width'] * 1.5
reach_df['SchummPred'] = reach_df['bar_width'] * 1.5

# Get r-squared for all combinations
r2Ms = r2_score(ms_df['mean_width'], ms_df['predicted'])
r2Bar = r2_score(bar_df['mean_width'], bar_df['predicted'])
r2Reach = r2_score(reach_df['mean_width'], reach_df['predicted'])
r2Lit = r2_score(lit_df['Channel Width'], lit_df['predicted'])
r2Combo = r2_score(combo_df['mean_width'], combo_df['predicted'])
r2SchummBar = r2_score(bar_df['mean_width'], bar_df['SchummPred'])
r2SchummReach = r2_score(reach_df['mean_width'], reach_df['SchummPred'])

###############################
##         FIGURE 3bd        ##
###############################
vh.predicted_vs_actual(ms_df, bar_df, lit_df, ancient_df)
