import math
import numpy as np
import pandas
from matplotlib import pyplot as plt


def changeNames(df, names):
    new_names = []
    for idx, row in df.iterrows():
        new_names.append(names[row['river']])
    
    df['river'] = new_names

    return df


def chezyC(h, D90):
    return 18 * (np.log10(12*h/D90))

def u_star(h, S):
    g = 9.81        # m3/s
    rho = 1000      # kg/m3
    return (g * h * S)**.5

def shieldsus(u_star, d):
    g = 9.81        # m3/s
    rho = 1000      # kg/m3
    rho_s = 1650    # kg/ms
    s = (rho_s - rho) / rho

    return    (u_star**2) / (s * g * d)

def shields(h, S, D):
    g = 9.81        # m3/s
    rho = 1000      # kg/m3
    rho_s = 1650    # kg/ms

    return (h * S) / (1.65 * D)
    

def talmon(D, h, S, D90, R):
    d_h = D / h
    us = u_star(h, S)
    shield = shields(h, S, D)
#    shield = shieldsus(us, D)
    C = chezyC(h, D * 10)
    h_R = h / R

    k = 0.4
    g = 9.81        # m3/s

    return np.arctan(
        9
        * (d_h**.3)
        * (shield**.5)
        * (2 / (k**2))
        * (1 - ((g**.5) / (k * C)))
        * h_R
    )


thetadf = pandas.read_csv('thetadf.csv')
ktaudf = pandas.read_csv('k_taoRivers.csv')
curvedf = pandas.read_csv('curvature.csv')

# Clean theta df
river_names = {
    'Brazos': 'Brazos River',
    'Koyukuk': 'Koyukuk River',
    'Mississippi': 'Mississippi River',
    'Mississippi - Leclair': 'Mississippi River',
    'Nestucca': 'Nestucca River',
    'Powder': 'Powder River',
    'Red': 'Red River',
    'Rio Grande': 'Rio Grande River',
    'Sacramento': 'Sacramento River',
    'Tombigbee': 'Tombigbee River',
    'Trinity': 'Trinity River',
    'White': 'White River'
}

thetadf = thetadf[['river', 'bar', 'medianSlope']]
thetadf = changeNames(thetadf, river_names)
thetadf['tan'] = np.tan(np.radians(thetadf['medianSlope']))

# Clean curve df
curvedf = curvedf[['river', 'bar', 'channel_width_mean', 'curvature_circle', 'curvature_fagherazzi']]

# Clean ktaudf
river_names = {
    'Brazos': 'Brazos River',
    'Koyukuk River, AK': 'Koyukuk River',
    'Mississippi': 'Mississippi River',
    'Mississippi - Leclair': 'Mississippi River',
    'Nestucca': 'Nestucca River',
    'Powder River': 'Powder River',
    'Red River, TX': 'Red River',
    'Rio Grande - Texas': 'Rio Grande River',
    'Sacramento': 'Sacramento River',
    'Tombigbee River': 'Tombigbee River',
    'Trinity River, TX': 'Trinity River',
    'White River, Indiana': 'White River'
}
ktaudf['river'] = ktaudf['Channel Geometry']
ktaudf = ktaudf[['river', 'Slope', 'Depth', 'width']]
ktaudf = changeNames(ktaudf, river_names)

# Merge ktau and graindf
df = ktaudf.merge(curvedf, how='left', on='river')
df = df.merge(thetadf, how='left', on=['river', 'bar'])

#df['Rn'] = df['curvature_fagherazzi'] / df['Depth']
df['Rn'] = df['curvature_circle'] / df['Depth']

df = df.reset_index(drop=False)

use_rivers = [
    'Brazos River',
    'Red River',
    'Sacramento River',
    'Tombigbee River',
    'Mississippi River',
    'White River',
    'Trinity River',
    'Powder River',
    'Brazos River',
    'Rio Grande River',
    'Koyukuk River',
    'Nestucca River',
]
df = df[df['river'].isin(use_rivers)]

# Drop NA
df = df[df['Rn'] != df['Rn'].max()]

# Find C via mannings
n = 0.025
df['A'] = df['Depth'] * df['channel_width_mean']
df['Rh'] = (2 * df['Depth']) + df['channel_width_mean']
df['C'] = (1 / n) * df['Rh']**(1/6)

# Test with experimental parameters from Van de Lageweg (2016)
# Archimetrics paper
d50 = 0.00058
d90 = .00075
h = .015
w = .3

A = h * w
P = (2 * h) + w
Rh = A / P
n = 0.025
#C = (1 / n) * Rh**(1/6)

Rns = np.linspace(10, 1000, 100)
S = 0.0055
curvs = Rns * h
van = []
for curv in curvs:
    van.append(talmon(
        d50,
        h,
        S,
        d90,
        curv
    ))

# Plot experimental parameters
plt.plot(Rns, van)
plt.xlim([10, 1000])
plt.ylim([.01, 1])
plt.xscale('log')
plt.yscale('log')
plt.show()

# Find predicted slope at varying relative roughness
river_means = df.groupby('river').mean()

# Set up sample space
Rns = np.linspace(10, 1000, 100)

# Set up stepped relative roughness
fixd50 = .0001
hs = np.linspace(1, 50, 10)
hs = [100, 10, 1]
predicted_slopes = {}
slope = .0001
for h in hs:
    curvs = Rns * h 
    ys = []
    for curv in curvs:
        ys.append(talmon(
            fixd50,
            h,
            slope,
            fixd50 * 10,
            curv
        ))
    predicted_slopes[str(fixd50/h)] = ys

# Final Plotting
fig, ax1 = plt.subplots()
for name, value in predicted_slopes.items():
    ax1.plot(Rns, value, label=name)

river_df = df.groupby('river')
for name, group in river_df:
    tan_sem = group['tan'].std() / np.sqrt(len(group))
    Rn_sem = group['Rn'].std() / np.sqrt(len(group))
    ax1.scatter(group['Rn'].mean(), group['tan'].mean(), label=name)
    ax1.errorbar(
        group['Rn'].mean(), 
        group['tan'].mean(), 
        tan_sem,
        Rn_sem
    )
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_xlim([10, 1000])
ax1.set_ylim([0.01, 1])

plt.legend()
fig.tight_layout()
plt.show()

