#################################################
####         FINDING CHANNEL WIDTH FOR ONE   ####
#################################################
# Method 1.: Use local minima and maxima, find the biggest difference between
#            neighbors. Use the maximas as the channel endpoints
# Method 2.: Use curvature, find places of maximum curvature (2nd deivative)


# Get X Section
idx = 390
t = xsections[idx]['xsection']['demvalue_sm']
p = xsections[idx]['xsection']['distance']
bank1 = xsections[idx][2][0][0]
bank2 = xsections[idx][2][1][0]
b1 = np.where(p == bank1)
b2 = np.where(p == bank2)
plt.scatter(bank1, t[b1])
plt.scatter(bank2, t[b2])
plt.plot(p, t)
plt.show()


data = {'distance': p, 'elevation': t}
cross_section = pandas.DataFrame(data=data, columns=['distance', 'elevation'])

# Find Maxima and Minima
order = 20 
maxs = argrelextrema(t, np.greater, order=order)
maxima = np.column_stack((p[maxs], t[maxs]))
mins = argrelextrema(t, np.less, order=order)
minima = np.column_stack((p[mins], t[mins]))

extremes = np.concatenate([maxima, minima])
extremes = extremes[extremes[:,0].argsort()]
# Get biggest difference between ADJACENT maxima and minma
d = []
for i in range(0, len(extremes)):
    if i == len(extremes) - 1:
        d.append(0) 
    else:
        diff = extremes[i+1][1] - extremes[i][1]
        d.append(diff)
maxi = np.where(d == np.amax(d))[0][0]
mini = np.where(d == np.amin(d))[0][0]
max_val = extremes[maxi + 1]
min_val = extremes[mini]
# Take lowest of the two and project across the stream
# This works because the opposite side of the channel HAS to have a different
# Position sine. If that changes, this logic will have to change
if min_val[1] >= max_val[1]:
    opposite_channel_section = cross_section[
        (cross_section['distance'] > min_val[0]) 
        & (cross_section['distance'] < max_val[0])
        & (cross_section['distance'] < 0)
    ]
    width_val = max_val
else:
    opposite_channel_section = cross_section[
        (cross_section['distance'] > min_val[0]) 
        & (cross_section['distance'] < max_val[0])
        & (cross_section['distance'] > 0)
    ]
    width_val = min_val

opposite_val = opposite_channel_section.iloc[
    (
        opposite_channel_section['elevation']
        - width_val[1]
    ).abs().argsort()[:1]
].to_numpy()[0]

# Width Val is the minimum of the adjacent max-mins
# Opposite Val is the closes value on the opposite side of the channel

# Plot
plt.scatter(width_val[0], width_val[1], color='red')
plt.scatter(opposite_val[0], opposite_val[1], color='red')
plt.scatter(extremes[:, 0], extremes[:, 1])
plt.plot(p, t)
plt.show()

# Find first derivative of curve
test_section = np.column_stack((p, t))
test_section = np.column_stack((
    test_section,
    get_d1(test_section)
))
test_section = np.column_stack((
    test_section,
    get_d2(test_section)
))

# Plot
fig, ax1 = plt.subplots()
color = 'tab:red'
plt.scatter(p[maxs], t[maxs], color='red')
plt.scatter(p[mins], t[mins], color='green')
ax1.plot(p, t, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.plot(p, test_section[:,2], color=color)
ax2.plot(p, test_section[:,3], color='tab:green')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
#################################################
####         MISC                            ####
#################################################
# To CSV
indezes = [i for i in range(1, len(xsections), 100)]
for index in indezes:
    root = '~/PhD Documents/Projects/river-profiles/test/'
    fn = 'test_xsection_{}.csv'.format(index)
    test_array = xsections[index]['xsection']
    df = pandas.DataFrame(test_array)
    df.to_csv((root + fn))

# From .npy
root = '~/PhD Documents/Projects/river-profiles/'
fn = 'xsections_test_1120.npy'
path = root + fn
xsections = np.load(fn, allow_pickle=True)

fig = plt.figure()
ax1 = plt.plot(df_sm['distance'], df_sm['demvalue'])
plt.plot(x, y, linestyle='dashed')
plt.show()

fn = 'xsections_koyukuk_1204.npy'
np.save(fn, xsections)
xsections = np.load(fn, allow_pickle=True)

#################################################
####         NEXT STEPS                      ####
#################################################
# 1. Convert the "position" field to a true distance
# 2. Automatic calculation of channel widths (find distance between maxima?)
# 3. Automatic detection of point bars. Area of maximum curvature? (will be hard)
# 4. Automatic calculation of clinoform surfaces (This will be the hard part)
#    Need to make sure that you are not hard coding the answer
#    Will it work if clinoform width is distance between inner channel bank 
#    and bottom of channel?


# # plotting
import matplotlib.pyplot as plt
import matplotlib.image as mpimgkjk
import matplotlib.colors as colors
from copy import copy

dem_vis = np.where(dem<0, False, dem) 
x, y = dem_vis.shape
extent = 0, y, 0, x 

center_mask = np.ma.masked_where(centerline == False, centerline)

n_levels = 9
dem_max = dem_vis.max()
step = dem_max / n_levels

fig = plt.figure(figsize = (11, 7))
imgplot = plt.imshow(
    dem_vis, 
    cmap=plt.get_cmap('gray'),
    interpolation='nearest',
    extent=extent
)
ilot = plt.imshow(
    center_mask,
    extent=extent
)
plt.show()

# Start building river dem
river_dem = np.full(dem.shape, False)
for idx, row in coordinates.iterrows():
    x, y = georef.lonlat2pix(gmDEM, row['lat'], row['lon'])
    for i in range(1,100):
        river_dem[y, x+i] = True
        river_dem[y, x-i] = True
        river_dem[y+i, x] = True
        river_dem[y-i, x] = True

river_mask_d = np.ma.masked_where(
    river_dem == False,
    river_dem 
)
river_mask = np.ma.masked_where(
    centerline == False,
    centerline 
)

x, y = dem_clean.shape
extent = 0, y, 0, x 
fig = plt.figure(frameon=False)

im1 = plt.imshow(
    dem_vis, 
    cmap=plt.cm.gray, 
    norm=colors.Normalize(),
    interpolation='nearest', 
    extent=extent
)
# Set up a colormap:
# use copy so that we do not mutate the global colormap instance
im2 = plt.imshow(
    river_mask_d,
    interpolation='none',
    cmap=plt.cm.plasma, 
    extent=extent
)

plt.show()


################################
# Load in test_bars
################################
from matplotlib import pyplot as plt 

path = '/Users/evangreenberg/PhD Documents/Projects/river-profiles/test_bars.csv'
bar_df = pandas.read_csv(path)

upstream = bar_df[['upstream_easting', 'upstream_northing']].iloc[0]
downstream = bar_df[['downstream_easting', 'downstream_northing']].iloc[0]
bar = bar_df.iloc[0]

bar_sections = xsections[upstream[0]:downstream[0]]
bar_len = len(test_bar)
q_len = int(bar_len / 4)
half_len = q_len * 2
tq_len = q_len * 3
t = test_bar[half_len]['xsection']['demvalue']
p = test_bar[half_len]['xsection']['distance']
endpoints = test_bar[half_len]['ch_endpoints']
width = test_bar[half_len]['width']

bar = BarHandler()
bar_sections = bar.get_bar_xsections(coordinates, xsections, bar_df)

i0 = [i for i, x in enumerate(p) if x == banks[0][0]]
i1 = [i for i, x in enumerate(p) if x == banks[1][0]]
i2 = [i for i, x in enumerate(p) if x == banks[0][1]]
i3 = [i for i, x in enumerate(p) if x == banks[1][1]]

plt.plot(p, t)
plt.scatter(banks[0][0], t[i0])
plt.scatter(banks[1][0], t[i1])
plt.scatter(banks[0][1], t[i2])
plt.scatter(banks[1][1], t[i3])
plt.show()

b0 = [i for i, x in enumerate(p) if x == bar_points[0]]
b1 = [i for i, x in enumerate(p) if x == bar_points[1]]

plt.plot(p, t)
plt.scatter(bar_points[0], t[b0])
plt.scatter(bar_points[1], t[b1])
plt.show()

bh = BarHandler()
path = '/Users/evangreenberg/PhD Documents/Projects/river-profiles/test_bars.csv'
bar_df = pandas.read_csv(path)
bar_df = bar_df[[
	'upstream_easting',
	'upstream_northing', 
	'downstream_easting', 
	'downstream_northing'
]]
bar_sections = bh.get_bar_xsections(coordinates, xsections, bar_df.iloc[1])
widths = []
bar_widths = []
for section in bar_sections:
    if section['width'] == 'nan':
        widths.append(False)
    else:
        widths.append(section['width'])
    bar_widths.append(bh.find_bar_width(section['bank']))

ratio = []
idxs = []
for idx, width in enumerate(widths):
    if width and bar_widths[idx]:
        ratio.append(width/bar_widths[idx])
        idxs.append(idx)

ratio0 = ratio
ratio1 = ratio


bar_side = bh.find_bar_side(bar_sections, 'demvalue_sm')
print(bar_side)

# [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
test_i = 36
test1 = bar_sections[test_i]['xsection']['distance']
test2 = bar_sections[test_i]['xsection']['demvalue_sm']
banks = bar_sections[test_i]['bank']

i0 = [i for i, x in enumerate(p) if x == banks[0][0]]
i1 = [i for i, x in enumerate(p) if x == banks[1][0]]
i2 = [i for i, x in enumerate(p) if x == banks[0][1]]
i3 = [i for i, x in enumerate(p) if x == banks[1][1]]

plt.plot(test1, test2)
plt.scatter(banks[0][0], test2[i0[0]])
plt.scatter(banks[1][0], test2[i1[0]])
plt.scatter(banks[0][1], test2[i2[0]])
plt.scatter(banks[1][1], test2[i3[0]])
plt.show()

b0 = [i for i, x in enumerate(p) if x == bar_[0]]
b1 = [i for i, x in enumerate(p) if x == bar_[1]]

plt.plot(p, t)
plt.scatter(bar_[0], t[b0])
plt.scatter(bar_[1], t[b1])
plt.show()



################################ 
### BAR PLOTTING
################################
line = len(bars_['bar_1']['ratio']) + 20
line_pl = [i for i in range(0, line, 1)]
line_val = [1.5 for i in line_pl]
for key in bars_.keys():
    plt.scatter(bars_[key]['idx'], bars_[key]['ratio'])
plt.plot(line_pl, line_val)
plt.show()

#####
mp = 0
for row in centerline:
    m = max(row)
    if m > mp:
        mp = m

ml = 0
mi = 0
for i in range(1,mp):
    l = len(centerline[centerline==i])
    if l > ml:
        ml = l 
        mi = i

centerline2 = np.where(centerline not in idxs, 0, centerline)
plt.imshow(centerline)
plt.show()

#XXX
# Add in the dilineate algo to sort the centerlines by length
centerlines3 = np.hstack((0, sumstrong > 0)).astype('bool')
centerlines3 = centerlines3[cclabels]

np.sort(data,axis=0)
np.sort(lengths, axis=0)
data[np.argsort(data.A[:, 1])])

ind=np.argsort(data[:,1],axis=0)
data[ind.ravel(),:] 

ind = np.argsort(-lengths[:, 1], axis=0)
sort_len = lengths[ind.ravel(), :]
