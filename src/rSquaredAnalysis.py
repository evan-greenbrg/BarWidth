import pandas
import seaborn as sns
from matplotlib import pyplot as plt


trinity_sc = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Trinity/rsquared_dataframe.csv'
koyukuk_sc = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Koyukuk/rsquared_dataframe.csv'
red_sc = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Red_River/rsquared_dataframe.csv'
white_sc = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/White_River/rsquared_dataframe.csv' 
platte_sc = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Output_Data/Platte/rsquared_dataframe.csv'

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

rsquare_df = trinity_df.append(koyukuk_df)
rsquare_df = rsquare_df.append(red_df)
rsquare_df = rsquare_df.append(white_df)

rsquare_group = rsquare_df.groupby('river')

rsquare_df = pandas.DataFrame(data={
    'Trinity': trinity_df['rsquared'],
    'Koyukuk': koyukuk_df['rsquared'],
    'Red': red_df['rsquared'],
    'White': white_df['rsquared'],
    'Platte': platte_df['rsquared'],
})
ax = rsquare_df.boxplot()
ax.grid(b=None)
ax.set_ylabel('R-Squared')
plt.show()
