import pandas
import numpy as np
from matplotlib import pyplot as plt
import glob


f = 'angle_data/*.csv'
fps = glob.glob(f)

df = pandas.DataFrame()
for fi in fps:
    df = df.append(pandas.read_csv(fi))
df = df.sort_values('theta')

theta_df = pandas.DataFrame()
theta_group = df.groupby('theta')
for name, group in theta_group:
    sampdf = group.sample(3)
    sampdf = sampdf.mean()
    theta_df = theta_df.append(sampdf, ignore_index=True)

# theta_df = df.groupby('theta').mean()
theta_df['ratio'] = (
    theta_df[theta_df['theta']==0]['channel_width_mean'].mean()
    / theta_df['bar_width']
)

fig = plt.figure()
plt.scatter(theta_df['theta'], theta_df['ratio'])
plt.yticks(np.arange(0, 10, 2.0))
plt.xlabel('Angle from Channel Normal')
plt.ylabel('Channel Width / Bar Surface Width')
fig.savefig('/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/figures/62420bar_cut_angles.svg', format='svg')
plt.show()

normf = '/home/greenberg/Code/Github/river-profiles/src/barCuts/angle_data/norm.csv'
norm = pandas.read_csv(normf)

df = df[df['theta'] < 70]
n = 15
times = 101
idx = [i for i in range(1, n+1)]
cuts = {str(i): [] for i in range(1, n+1)}
norms = {str(i): [] for i in range(1, n+1)}
fig, ax = plt.subplots()
for j in range(1, times):
    for i in range(1, n+1):
        cuts[str(i)].append(df.sample(i)['ratio'].median())
        norms[str(i)].append(norm.sample(i)['ratio'].median())

c = '#50C878'
c2 = '#ffd381'
for i in range(1, n+1):
    ax.boxplot(
        norms[str(i)], positions=[i], widths=0.7, showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor=c, color=c),
        capprops=dict(color=c),
        whiskerprops=dict(color=c),
        flierprops=dict(color=c, markeredgecolor=c),
        medianprops=dict(color=c),
        zorder=20
    )
    ax.boxplot(
        cuts[str(i)], positions=[i], widths=0.7, showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor=c2, color=c2),
        capprops=dict(color=c2),
        whiskerprops=dict(color=c2),
        flierprops=dict(color=c2, markeredgecolor=c2),
        medianprops=dict(color=c2),
    )

ax.set_ylim([0, 10])

plt.xlabel('Samples')
plt.ylabel('Channel Width / Bar Surface Width')
plt.yticks(np.arange(0, 10, 2.0))
plt.legend(['Oblique Cross-Sections', 'Normal Cross-Sections'])

plt.show()
