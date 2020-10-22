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


def chezyC(h, D):
    return 18* (math.log(12*h/D))


def talmonz(h, D, S, R):
    ps = 1650
    p = 1000
    g = 9.81
    k = 0.4
    return (
        9
        * (D**.3)
        * (h**0.3)
        * ((p/(ps-p))**.5)
        * (h**.5)
        * (S**.5)
        * (D**-.5)
        * (2/k**2)
        * (1 - ((g**.5)/(k * chezyC(h, D))))
        * (h)
        * (R**-1)
    )

thetadf = pandas.read_csv('thetadf.csv')
ktaudf = pandas.read_csv('k_taoRivers.csv')
curvedf = pandas.read_csv('curvature.csv')
graindf = pandas.read_csv('grain_sizes.csv')

# Clean theta df
river_names = {
    'Brazos': 'Brazos River',
    'Koyukuk': 'Koyukuk River',
    'Mississippi': 'Mississippi River',
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
curvedf = curvedf[['river', 'bar', 'channel_width_mean', 'curvature']]

# Clean ktaudf
river_names = {
    'Brazos': 'Brazos River',
    'Koyukuk River, AK': 'Koyukuk River',
    'Mississippi': 'Mississippi River',
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
param = ktaudf.merge(graindf, how='left', on='river')
df = curvedf.merge(param, how='left', on='river')
df = df.merge(thetadf, how='left', on=['river', 'bar'])
#df = df.groupby('river').median()
df['Rn'] = df['curvature']/df['Depth']
df = df.reset_index(drop=False)
print(df[['river', 'tan', 'Rn']])

test = df[df['river'] == 'Trinity River']
lim = .5
preds = []
for idx, row in test.iterrows():
    preds.append(talmonz(
        row['Depth'],
        row['Bed material -Lower (m)'],
        row['Slope'],
        row['curvature']
    ))
test['pred'] = preds

xs = [i for i in range(5)]
ys = xs
plt.scatter(test['pred'], test['tan'])
plt.plot(xs, ys)
# plt.xlim([0, lim])
# plt.ylim([0, lim])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# Visualize it
df_group = df.groupby('river')
for name, group in df_group:
    r = random.random()
    b = random.random()
    g = random.random()
    color = (r, g, b)
    print(name)
    plt.scatter(group['Rn'], group['tan'], c=color, label=name)

plt.legend(framealpha=1, );
plt.xscale('log')
plt.yscale('log')
xs = df['Rn']
yslow = [0.0349 for i in xs]
yshigh = [0.176 for i in xs]
plt.plot(xs, yslow, c='gray')
plt.plot(xs, yshigh, c='gray')
plt.xlim([10, 1000])
plt.ylim([.01, 1])
plt.xlabel('R/h')
plt.ylabel('Tan(dz/dy)')
plt.show()
