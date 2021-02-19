import numpy as np
import pandas


xPaths = {
    'Koyukuk': '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Koyukuk/xsections.npy',
    'Nestucca': '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Beaver/xsections.npy',
    'Brazos': '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Brazos_Near_Calvert/xsections.npy',
    'Mississippi': '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Mississippi_1/xsections.npy',
    'Powder': '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Powder/xsections.npy',
    'Red': '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Red_River/xsections.npy',
    'RioGrande': '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Rio_Grande_TX/xsections.npy',
    'Sacramento': '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Sacramento/xsections.npy',
    'Tombigbee': '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Tombigbee/xsections.npy',
    'Trinity': '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Trinity/xsections.npy',
    'White': '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/White_River/xsections.npy',
}

depths = {}
for key, path in xPaths.items():
    xsections = np.load(path, allow_pickle=True)
    widths = np.empty((len(xsections),))
    for idx, xsection in enumerate(xsections):
        widths[idx] = xsection['dem_width']

    df = pandas.DataFrame()
    df['width'] = widths
    df = df.dropna(how='any')
    mean_width = df['width'].mean()
    print(key)
    print(mean_width)

    # (0.17 +- 0.05)w**(.65+-.06) 
    lower = .12 * mean_width**.59
    mean = .17 * mean_width**.65
    upper = .22 * mean_width**.71

    depths[key] = {
        'lower': lower,
        'mean': mean,
        'upper': upper
    }
