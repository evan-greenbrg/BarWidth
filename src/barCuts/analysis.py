import pandas
import numpy as np
from matplotlib import pyplot as plt
import glob


cutf1 = '/home/greenberg/Code/Github/river-profiles/src/barCuts/cut_0_9.csv'
cutf2 = '/home/greenberg/Code/Github/river-profiles/src/barCuts/cut_10_16.csv'
normf1 = '/home/greenberg/Code/Github/river-profiles/src/barCuts/norm_0_15.csv'


cut1 = pandas.read_csv(cutf1)
cut2 = pandas.read_csv(cutf2)
norm = pandas.read_csv(normf1)

cut = cut1.append(cut2)

cut_g = cut.groupby('idx').mean()
norm_g = norm.groupby('idx').mean()

plt.scatter(cut_g.index, cut_g['ratio'], label='Oblique Cuts')
plt.scatter(norm_g.index, norm_g['ratio'], label='Perpendicular Cuts')
plt.legend()
plt.show()

n = 25
times = 101
idx = [i for i in range(1, n+1)]
cuts = {str(i): [] for i in range(1, n+1)}
norms = {str(i): [] for i in range(1, n+1)}
for j in range(1, times):
    for i in range(1, n+1):
        cuts[str(i)].append(cut.sample(i)['ratio'].median())
        norms[str(i)].append(norm.sample(i)['ratio'].median())

df = pandas.DataFrame()
for i in range(1, n+1):
    plt.scatter(
        [i for j in cuts[str(i)]],
        cuts[str(i)],
        c='#b2b2b2'
    )
    plt.scatter(
        [i for j in norms[str(i)]],
        norms[str(i)],
        c='#b3c5f8'
    )
#    plt.scatter(i, np.mean(cuts[str(i)]), c='#333333')
#    plt.scatter(i, np.mean(norms[str(i)]), c='#4c4cff')

plt.xlabel('Samples')
plt.ylabel('Channel Width / Bar Width')
plt.yticks(np.arange(0, 30, 2.0))
plt.show()


    df = df.append(pandas.DataFrame({
        'i': idx[i-1],
        'cut_min': min(cuts[str(i)]),
        'cut_med': np.median(cuts[str(i)]),
        'cut_max': max(cuts[str(i)]),
        'norm_min': min(norms[str(i)]),
        'norm_med': np.median(norms[str(i)]),
        'norm_max': max(norms[str(i)])
    }, index=[i]))


plt.scatter(df['i'], df['cut_min'])
plt.scatter(df['i'], df['cut_max'])
plt.scatter(df['i'], df['cut_med'], label='Oblique Cuts')
plt.scatter(df['i'], df['norm_min'])
plt.scatter(df['i'], df['norm_max'])
plt.scatter(df['i'], df['norm_med'], label='Perpendicular Cuts')
plt.legend()
plt.show()



f = 'angle_data/*.csv'
fps = glob.glob(f)

df = pandas.DataFrame()
for fi in fps:
    df = df.append(pandas.read_csv(fi))
df = df.sort_values('theta')
theta_df = df.groupby('theta').mean()

fig = plt.figure()
plt.scatter(theta_df.index, theta_df['ratio'])
plt.yticks(np.arange(0, 26, 2.0))
plt.xlabel('Angle from Channel Normal')
plt.ylabel('Channel Width / Bar Surface Width')
fig.savefig('/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/figures/62420bar_cut_angles.svg', format='svg')
plt.show()

normf = '/home/greenberg/Code/Github/river-profiles/src/barCuts/norm_0_15.csv'
norm = pandas.read_csv(normf)

n = 25
times = 101
idx = [i for i in range(1, n+1)]
cuts = {str(i): [] for i in range(1, n+1)}
norms = {str(i): [] for i in range(1, n+1)}
fig2 = plt.figure()
for j in range(1, times):
    for i in range(1, n+1):
        cuts[str(i)].append(df.sample(i)['ratio'].median())
        norms[str(i)].append(norm.sample(i)['ratio'].median())

for i in range(1, n+1):
    plt.scatter(
        [i for j in cuts[str(i)]],
        cuts[str(i)],
        c='#333333'
    )
    plt.scatter(
        [i for j in norms[str(i)]],
        norms[str(i)],
        c='#b3c5f8'
    )

plt.xlabel('Samples')
plt.ylabel('Channel Width / Bar Surface Width')
plt.yticks(np.arange(0, 44, 4.0))
plt.legend(['Oblique Cross-Sections', 'Normal Cross-Sections'])

fig2.savefig('/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/figures/62420bar_cut_samples.svg', format='svg')
plt.show()
