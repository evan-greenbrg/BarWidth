import pandas
import seaborn as sns
from matplotlib import pyplot as plt


trinity_sc = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Trinity/bar_parameters.csv'
koyukuk_sc = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Koyukuk/bar_parameters.csv'
red_sc = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Red_River/bar_parameters.csv'
white_sc = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/White_River/bar_parameters.csv' 
platte_sc = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Platte/bar_parameters.csv'

trinity_df = pandas.read_csv(trinity_sc)
trinity_df['river'] = 'Trinity'

koyukuk_df = pandas.read_csv(koyukuk_sc)
koyukuk_df['river'] = 'Koyukuk'

red_df = pandas.read_csv(red_sc)
red_df['river'] = 'Red River'

white_df = pandas.read_csv(white_sc)
white_df['river'] = 'White River'

platte_df = pandas.read_csv(platte_sc)
platte_df['river'] = 'Platte River'

parameters_df = trinity_df.append(koyukuk_df)
parameters_df = parameters_df.append(red_df)
parameters_df = parameters_df.append(white_df)

param_group = parameters_df.groupby('river')

k_df = pandas.DataFrame(data={
    'Trinity': trinity_df['k'],
    'Koyukuk': koyukuk_df['k'],
    'Red': red_df['k'],
    'White': white_df['k'],
    'Platte': platte_df['k'],
})
ax = k_df.boxplot()
ax.grid(b=None)
ax.set_ylabel('Growth Rate, k')
plt.show()
