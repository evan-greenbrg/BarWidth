import matplotlib.pyplot as plt

class Visualizer():

    def __init__(self, x0, y0):
        self.x0 = x0
        self.y0 = y0

    def get_downstream_distance(self, bars):
        """
        Take UTM coordinates from bars dictionary and 
        converts to downstream distance
        """
        for key in bars.keys():
            distance = []
            for idx, coor in enumerate(bars[key]['coords']):
                length = (
                    ((coor[0] - self.x0)**2)
                    + ((coor[1] - self.y0)**2)
                )**(1/2)
                distance.append(length)
            bars[key]['distance'] = distance

        return bars



    def plot_downstream_bars(self, bars, path):
        """
        Generates plot of all bar widths as ratio with channel depth
        going downstream
        """
        bars = self.get_downstream_distance(bars)


        max0 = 0
        fig = plt.figure(figsize = (11, 7))
        for key in bars_.keys():
            plt.scatter(bars[key]['distance'], bars[key]['ratio'])
            maxd = max(bars[key]['distance'])
            if maxd > max0:
                max0 = maxd

        line_pl = [i for i in range(0, int(max0), 100)]
        line_val = [1.5 for i in line_pl]
        plt.plot(line_pl, line_val)
        plt.xlabel('Downstream Distance')
        plt.ylabel('Channel Width/Bar Width')
        plt.savefig(path)
