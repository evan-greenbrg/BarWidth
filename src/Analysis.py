import math

import arviz as az
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pymc3 as pm
from sklearn.metrics import r2_score

from Visualizer import Visualizer


LOG = True 
FIT_INTERCEPT = True 
FIT_SLOPE = False

def getModel(X, y, fit_intercept=False, fit_slope=True):
    # Set up
    with pm.Model() as model:
        # Intercept
        intercept = pm.Normal('Intercept', mu=0, sd=1000)
        # sd = 25

        # Slope
#        slope = pm.Normal('slope', mu=float(reg.coef_), sd=1000)
        slope = pm.Normal('slope', mu=0, sd=1000)
#        slope = pm.Uniform('slope', 0, 1000)
        # sd = 1

        # Standard Deviation
#        sigma = pm.HalfNormal('sigma', sd=1000)
        sigma = pm.Uniform('sigma', lower=0, upper=1000)
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

        # Posterior distribution
        return model 


def samplePosterior(model, N, fit_intercept=False, fit_slope=True):
    """
    Monte Carlo for the posterior. Sample posterior predictive
    """
    RANDOM_SEED = 58
    with model:
        step = pm.NUTS()
        trace = pm.sample(N, step)
        
        if fit_intercept and not fit_slope:
            var_names = ["Intercept", "Y_obs"]
            summary_names = ["Intercept"]
        elif not fit_intercept and fit_slope:
            var_names = ["slope", "Y_obs"]
            summary_names = ["slope"]
        else:
            var_names = ["Intercept", "slope", "Y_obs"]
            summary_names = ["Intercept", "slope"]

        ppc = pm.sample_posterior_predictive(
            trace, var_names=var_names, random_seed=RANDOM_SEED
        )


    summary = az.summary(trace, var_names=summary_names, round_to=3)

    params = {}
    for name in summary_names:
        params[name] = {}
        params[name]['hpd_3%'] = summary['hpd_3%'][name]
        params[name]['hpd_mean'] = summary['mean'][name]
        params[name]['hpd_97%'] = summary['hpd_97%'][name]

    return params, ppc['Y_obs']


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


def predictionInterval(X, Y, params, fit_intercept, fit_slope):
    """
    Finds the prediction interval (frequentist)
    """
    t = 1.96
    
    # Get predicted values
    if fit_intercept and not fit_slope:
        predicted = params['Intercept']['hpd_mean'] + X
    if not fit_intercept and fit_slope:
        predicted = params['slope']['hpd_mean'] * X
    if fit_intercept and fit_slope:
        predicted = (
            params['Intercept']['hpd_mean']
            + (params['slope']['hpd_mean'] * X)
        )

    # Find MSE
    n = len(Y)
    mse = (1 / n) * np.sum((Y - predicted)**2)

    # Find SE
    se = np.std(predicted) / math.sqrt(n)

    bound = t * math.sqrt(mse + (se**2))

    return bound


bars_f = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/sampled_bar_average_data.csv'
# use pm.df_summary(normal_trace) for unsampled
all_f = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/total_data.csv'
lit_path = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Lit_values.csv'
bar_df = pandas.read_csv(bars_f)
ms_df = pandas.read_csv(all_f)

# Get Rid of Platte
bar_df = bar_df[bar_df.river != 'Platte River']
ms_df = ms_df[ms_df.river != 'Platte River']

# Get Rid of Koyukuk
# bar_df = bar_df[bar_df.river != 'Koyukuk River']
# ms_df = ms_df[ms_df.river != 'Koyukuk River']


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

# Replace 0 width with NAN
bar_df = bar_df.replace(0., None)
ms_df = ms_df.replace(0., None)

# Filter out Null
bar_df = bar_df.dropna(how='any')
ms_df = ms_df.dropna(how='any')

# ms_df[ms_df.river == 'Koyukuk River']

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
log = LOG
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
fit_intercept = FIT_INTERCEPT
fit_slope = FIT_SLOPE
n = 5000

# All data points
params_ms = {}
ppc_ms = {}
for key, value in yms.items():
    model = getModel(Xms, value, fit_intercept, fit_slope)
    params_ms[key], ppc_ms[key] = samplePosterior(model, n, fit_intercept, fit_slope)

# Bar-averaged data points
params_bar = {}
ppc_bar = {}
for key, value in ybar.items():
    model = getModel(Xbar, value, fit_intercept, fit_slope)
    params_bar[key], ppc_bar[key] = samplePosterior(model, n, fit_intercept, fit_slope)

# Reach-averaged data points
params_reach = {}
ppc_reach = {}
for key, value in yreach.items():
    model = getModel(Xreach, value, fit_intercept, fit_slope)
    params_reach[key], ppc_reach[key] = samplePosterior(model, n, fit_intercept, fit_slope)

# Unpack parameters
if fit_intercept and not fit_slope:
    ms_coefs = {}
    bar_coefs = {}
    reach_coefs = {}
    for key, value in params_ms.items():
        ms_coefs[key] = {
            '3': 10**params_ms[key]['Intercept']['hpd_3%'],
            '50': 10**params_ms[key]['Intercept']['hpd_mean'],
            '97': 10**params_ms[key]['Intercept']['hpd_97%'],
        }
        bar_coefs[key] = {
            '3': 10**params_bar[key]['Intercept']['hpd_3%'],
            '50': 10**params_bar[key]['Intercept']['hpd_mean'],
            '97': 10**params_bar[key]['Intercept']['hpd_97%'],
        }
        reach_coefs[key] = {
            '3': 10**params_reach[key]['Intercept']['hpd_3%'],
            '50': 10**params_reach[key]['Intercept']['hpd_mean'],
            '97': 10**params_reach[key]['Intercept']['hpd_97%'],
        }
elif not fit_intercept and fit_slope:
    ms_coefs = {}
    bar_coefs = {}
    reach_coefs = {}
    for key, value in params_ms.items():
        ms_coefs[key] = {
            '3': params_ms[key]['slope']['hpd_3%'],
            '50': params_ms[key]['slope']['hpd_mean'],
            '97': params_ms[key]['slope']['hpd_97%'],
        }
        bar_coefs[key] = {
            '3': params_bar[key]['slope']['hpd_3%'],
            '50': params_bar[key]['slope']['hpd_mean'],
            '97': params_bar[key]['slope']['hpd_97%'],
        }
        reach_coefs[key] = {
            '3': params_reach[key]['slope']['hpd_3%'],
            '50': params_reach[key]['slope']['hpd_mean'],
            '97': params_reach[key]['slope']['hpd_97%'],
        }

elif fit_intercept and fit_slope:
    ms_coefs = {}
    bar_coefs = {}
    reach_coefs = {}
    for key, value in params_ms.items():
        ms_coefs[key] = {
            'slope': {
                '3': params_ms[key]['slope']['hpd_3%'],
                '50': params_ms[key]['slope']['hpd_mean'],
                '97': params_ms[key]['slope']['hpd_97%'],
            },
            'Intercept': {
                '3': params_ms[key]['Intercept']['hpd_3%'],
                '50': params_ms[key]['Intercept']['hpd_mean'],
                '97': params_ms[key]['Intercept']['hpd_97%'],
            },
        }
        bar_coefs[key] = {
            'slope': {
                '3': params_bar[key]['slope']['hpd_3%'],
                '50': params_bar[key]['slope']['hpd_mean'],
                '97': params_bar[key]['slope']['hpd_97%'],
            },
            'Intercept': {
                '3': params_bar[key]['Intercept']['hpd_3%'],
                '50': params_bar[key]['Intercept']['hpd_mean'],
                '97': params_bar[key]['Intercept']['hpd_97%'],
            },
        }
        reach_coefs[key] = {
            'slope': {
                '3': params_reach[key]['slope']['hpd_3%'],
                '50': params_reach[key]['slope']['hpd_mean'],
                '97': params_reach[key]['slope']['hpd_97%'],
            },
            'Intercept': {
                '3': params_reach[key]['Intercept']['hpd_3%'],
                '50': params_reach[key]['Intercept']['hpd_mean'],
                '97': params_reach[key]['Intercept']['hpd_97%'],
            },
        }


# Find equations for limits of HPD
hpd = pandas.DataFrame(az.hpd(ppc_ms['mean']), columns=['lower', 'upper'])
hpd['lower'] = 10**hpd['lower']
hpd['upper'] = 10**hpd['upper']
ppc_coefs = {}
for name, col in hpd.iteritems():
     ppc_coefs[name] = LinearRegression(
        fit_intercept=False
    ).fit(
       ms_df['bar_width'].values.reshape(-1, 1),
       hpd[name].values 
    ).coef_[0]

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
    ms_df,
    ppc_coefs,
    group_river,
    group_bar,
    lit_df,
    ms_coefs['mean'],
    median_size=15,
    alpha=0.25,
    density_size=35,
    log=log,
    fit_intercept=fit_intercept,
    fit_slope=fit_slope
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
if fit_intercept and fit_slope:
    # Log
    ms_df['predicted'] = (
        (10**ms_coefs['mean']['Intercept'][i])
        * (10**(ms_df['log_bar_width'])**ms_coefs['mean']['slope'][s])
    )
    reach_df['predicted'] = (
        (10**ms_coefs['mean']['Intercept'][i])
        * (10**(reach_df['log_bar_width'])**ms_coefs['mean']['slope'][s])
    )
    bar_df['predicted'] = (
        (10**ms_coefs['mean']['Intercept'][i])
        * (10**(bar_df['log_bar_width'])**ms_coefs['mean']['slope'][s])
    )
    lit_df['predicted'] = (
        (10**ms_coefs['mean']['Intercept'][i])
        * ((lit_df['Bar Width'])**ms_coefs['mean']['slope'][s])
    )
    ancient_df['predicted'] = (
        (10**ms_coefs['mean']['Intercept'][i])
        * ((ancient_df['bar_width'])**ms_coefs['mean']['slope'][s])
    )
else:
    ms_df['predicted'] = (
        ms_df['bar_width']*ms_coefs['mean'][s]
    )
    ms_df['predicted_er'] = (
        ms_df['bar_width']*(ms_coefs['mean']['97'] - ms_coefs['mean']['3'])
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
r2ancient = r2_score(ancient_df['Channel Width'], ancient_df['predicted'])
r2Combo = r2_score(combo_df['mean_width'], combo_df['predicted'])
r2SchummBar = r2_score(bar_df['mean_width'], bar_df['SchummPred'])
r2SchummReach = r2_score(reach_df['mean_width'], reach_df['SchummPred'])

###############################
##         FIGURE 3bd        ##
###############################
vh.predicted_vs_actual(ms_df, bar_df, lit_df, ancient_df)
