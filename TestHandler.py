from datetime import datetime
import random
import pandas

from BarHandler import BarHandler

class TestHandler():

    def __init__(self):
        pass

    def save_example_sections(self, xsections, n, outpath):
        fn = '{0}_test_section_{1}.csv'
        outpath += fn
        idxs = [random.randint(0, len(xsections)) for i in range(n)]
        for idx in idxs:
            easting = xsections[idx]['xsection']['easting']
            northing = xsections[idx]['xsection']['northing']
            elevation = xsections[idx]['xsection']['demvalue_sm']
            data = {
                'easting': easting,
                'northing': northing,
                'elevation': elevation
            }
            df = pandas.DataFrame(data=data)
            print(idx)
            print(df.head())
            df.to_csv(outpath.format(
                datetime.now().strftime('%Y%m%d'),
                idx
            ))

    def save_example_bar_sections(self, coordinates, xsections, 
                                  bar_df, outpath):
        bh = BarHandler(
            xsections[0]['coords'][0],
            xsections[0]['coords'][1]
        )
        fn = '{0}_test_bar_section_{1}.csv'
        outpath += fn

        for idx, bar in bar_df.iterrows():
            bar_sections = bh.get_bar_xsections(
                coordinates,
                xsections,
                bar_df.iloc[idx]
            )
            print(len(bar_sections))
            i = random.randint(0, len(bar_sections))
            xsection = xsections[i]
            easting = xsection[3]['easting']
            northing = xsection[3]['northing']
            elevation = xsection[3]['demvalue_sm']
            data = {
                'easting': easting,
                'northing': northing,
                'elevation': elevation
            }
            df = pandas.DataFrame(data=data)
            df.to_csv(outpath.format(
                datetime.now().strftime('%Y%m%d'),
                idx
            ))

    def save_channel_points(self):
        """
        Save the channel width boundaries and the channel floor points
        """
        pass
