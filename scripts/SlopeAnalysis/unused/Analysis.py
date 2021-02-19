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


def posterior_distribution(X, y, N):
    reg = LinearRegression(fit_intercept=True).fit(X.values.reshape(-1, 1), y.values)
    # Set up
    with pm.Model() as model:
        # Intercept
        intercept = pm.Normal('Intercept', mu=.5, sd=1)
        # sd = 25

        # Slope
        slope = pm.Normal('slope', mu=float(2*reg.coef_), sd=1)
        # sd = 1

        # Standard Deviation
        sigma = pm.HalfNormal('sigma', sd=1)
        # sd = 25

        # Estimate of Mean
        mean = intercept + (slope * X.values)

        # Observed Values
        Y_obs = pm.Normal('Y_obs', mu=mean, sd=sigma, observed=y.values)

        # Sampler
        step = pm.NUTS()

        # Posterior distribution
        return pm.sample(N, step)


# Load the data
ktau_path = 'k_taoRivers.csv'
theta_path = 'thetadf.csv'
curvature_path = 'curvature.csv'
grain_path = 'grain_sizes.csv'

thetadf = pandas.read_csv(theta_path)
ktaudf = pandas.read_csv(ktau_path)
curdf = pandas.read_csv(curvature_path)
graindf = pandas.read_csv(grain_path)

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
    df = df[['dt', 'discharge']]
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

# Get variability
data = {
    'river': [],
    'peakedness': [],
    'DVIa': [],
    'DVIc': [],
    'DVIy': []
}
for river, df in dfs.items():
    data['river'].append(river)
    data['peakedness'].append(peakedness(df))
    data['DVIa'].append(DVIa(df))
    data['DVIc'].append(DVIc(df))
    data['DVIy'].append(DVIy(df))

discharge_df = pandas.DataFrame(data=data)

# Make the column names the same
names = [
    'white',
    'koyukuk',
    'trinity',
    'red',
    'powder',
    'brazos',
    'rio',
    'tombigbee',
    'miss',
    'miss',
    'sacramento',
    'nestucca'
]
ktaudf['river'] = names

# change theta names
names = {
    'Brazos': 'brazos',
    'Koyukuk': 'koyukuk',
    'Mississippi': 'miss',
    'Nestucca': 'nestucca',
    'Powder': 'powder',
    'Red': 'red',
    'Rio Grande': 'rio',
    'Sacramento': 'sacramento',
    'Tombigbee': 'tombigbee',
    'Trinity': 'trinity',
    'White': 'white'
}
for old, name in names.items():
    thetadf = thetadf.replace(old, name)

# Merge
df = ktaudf.merge(discharge_df, how='left', on='river')
df['w_d'] = df['width'] / df['Depth']
df['tbank_tbed'] = df['Bank Shear Stress'] / df['Bed Shear Stress']

# Change names 
lookup = {
    'Brazos River': 'brazos',
    'Koyukuk River': 'koyukuk',
    'Mississippi River': 'miss',
    'Mississippi River - Leclair': 'miss',
    'Nestucca River': 'nestucca',
    'Powder River': 'powder',
    'Red River': 'red',
    'Rio Grande River': 'rio',
    'Sacramento River': 'sacramento',
    'Tombigbee River': 'tombigbee',
    'Trinity River': 'trinity',
    'White River': 'white'
}

cols_new_cur = []
cols_new_grain = []
for idx, row in curdf.iterrows():
    cols_new_cur.append(lookup[row['river']])
for idx, row in graindf.iterrows():
    cols_new_grain.append(lookup[row['river']])

curdf['river'] = cols_new_cur
graindf['river'] = cols_new_grain

curdf = curdf.groupby('river').median().reset_index(drop=False)

# merrge
df = df.merge(curdf, how='left', on='river')
df = df.merge(graindf, how='left', on='river')

thetadf['theta'] = thetadf['medianSlope']
thetadf['tan'] = numpy.tan(numpy.radians(thetadf['theta']))

# Regression
col = 'DVIc'
thetar = thetadf[['river', 'theta']].set_index('river')

maxd = thetar.groupby('river').max()
medd = thetar.groupby('river').median()
mind = thetar.groupby('river').min()
stdd = thetar.groupby('river').std()
thetar = pandas.DataFrame(data={
    'max': maxd['theta'],
    'median': medd['theta'],
    'min': mind['theta'],
    'std': stdd['theta']
})

dfr = df[['river', 'peakedness', 'DVIa', 'DVIc', 'DVIy']].set_index(
    'river'
).dropna(how='any')
rdf = thetar.join(dfr, on='river').dropna(how='any')

# Filter out mississippi
rdf = rdf[rdf.index != 'miss']

ys = {
#    'bot': rdf['median'] - (2 * rdf['std']),
    'med': rdf['median'],
    'top': rdf['median'] + (2 * rdf['std']),
}
# ys = {
#     'bot': rdf['min'],
#     'med': rdf['median'],
#     'top': rdf['max'],
# }
X = rdf[col]
coefs = {}
for name, y in ys.items():
    print(name)
    (intercept, slope), s = scipy.optimize.curve_fit(
        lambda t,a,b: a*numpy.exp(b*t),  
        X,  
        y,
        p0=(.001, .5),
        maxfev=6000
    )
    coefs[name] = {
        'intercept': intercept,
        'slope': slope
    }

#### HAVE PARAMETERS NEED TO MAKE CURVES TO PLOT

lins = numpy.linspace(0, 24, 30)
# linB = coefs['bot']['intercept'] * numpy.exp(lins * abs(coefs['bot']['slope']))
linM = coefs['med']['intercept'] * numpy.exp(lins * coefs['med']['slope'])
linT = coefs['top']['intercept'] * numpy.exp(lins * coefs['top']['slope'])

# Plot
thetadf = thetadf[thetadf['river'] != 'miss']
theta_group = thetadf.groupby('river')

colors = {
    'brazos': 'gray',
    'koyukuk': '#E69F00',
#    'miss': '#56B4E9',
    'nestucca': '#092C48',
    'powder': '#F0E442',
    'red': '#0072B2',
    'rio': '#D55E00',
    'sacramento': '#CC79A7',
    'tombigbee': '#DC267F',
    'trinity': '#009e73',
    'white': 'white'
}
i = 0
for name, group in theta_group:
    print(name)
    bplot = plt.boxplot(
        group['theta'], 
        positions=[float(df[df['river'] == name][col])],
        manage_ticks=False,
        patch_artist=True,
        widths=0.3
    )
    bplot['boxes'][0].set_facecolor(colors[name])
    i += 1

#    plt.fill_between(
#        lins, 
#        linB, 
#        linT, 
#        color='lightgray', 
#        edgecolor='lightgray',
#        linestyle='--'
#    )
plt.plot(lins, linM, linestyle='--', color='gray', linewidth=0.9)

plt.xlim((0, 24))
plt.xlabel('DVIc')
plt.ylabel('Tan(theta)')

custom_lines = [
    plt.Line2D([0], [0], color=color, lw=4) for color in colors.values()
]
plt.legend(
    custom_lines,
    colors.keys(), 
    loc='upper left',
    frameon=False
)
plt.show()


# R-squared
r2s = {}
for name, values in ys.items():
    pred = (
        coefs[name]['intercept'] 
        * numpy.exp(X * abs(coefs[name]['slope']))
    )
    r2s[name] = r2_score(values, pred)

linB = coefs['bot']['intercept'] * numpy.exp(lins * abs(coefs['bot']['slope']))


# Correlation
thetamed = thetadf.groupby('river').median().reset_index(drop=False)
join = df.merge(thetamed, on='river')
join['Rn'] = join['curvature'] / join['Depth']
columns = [
    'Slope',
    'w_d',
    'Bank Shear Stress',
    'Bed Shear Stress',
    'k',
    'Sinuosity',
    'Slope',
    'w_d',
    'tbank_tbed',
    'peakedness',
    'DVIa',
    'DVIc',
    'DVIy',
    'theta',
    'tan',
    'curvature',
    'Rn',
#    'Bed material -Lower (m)',
#    'Bed material -Upper (m)'
]
combos = list(itertools.combinations(columns, 2))
data = {
    'col1': [],
    'col2': [],
    'cor': [],
    'p': []
}
for combo in combos:
    cor, p = stats.spearmanr(join[combo[0]], join[combo[1]], nan_policy='omit')
    data['col1'].append(combo[0])
    data['col2'].append(combo[1])
    data['cor'].append(cor)
    data['p'].append(round(p, 4))
spearmdf = pandas.DataFrame(data=data)
cond = spearmdf[
    (spearmdf['col1'] == 'theta')
    | (spearmdf['col1'] == 'tan')
    | (spearmdf['col2'] == 'theta')
    | (spearmdf['col2'] == 'tan')
]

spearmdf['sig'] = (spearmdf['p'] > 0.95) | (spearmdf['p'] < 0.05)
sigdf = spearmdf[spearmdf['sig']]
cond.to_csv('correlations.csv')


plt.scatter(join['Rn'], join['tan'])
plt.xscale('log')
plt.yscale('log')
plt.xlim([10,1000])
plt.ylim([.0010,1])
plt.show()

