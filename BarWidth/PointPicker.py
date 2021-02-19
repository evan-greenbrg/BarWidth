from matplotlib import pyplot as plt
from matplotlib.widgets import Button
import numpy as np


X = []
Y = []
TESTDEM = '/home/greenberg/ExtraSpace/PhD/Projects/Bar-Width/Input_Data/Powder_River/output_be_26913.tif'


def closest(lst, K):
    """
    Finds the closest value in list
    """
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - K))]


class WidthPicker(object):
    text_template = 'x: %0.2f\ny: %0.2f'
    x, y = 0.0, 0.0
    xoffset, yoffset = -20, 20
    text_template = 'x: %0.2f\ny: %0.2f'

    def __init__(self, ax):
        self.ax = ax
        self.annotation = ax.annotate(
                self.text_template,
                xy=(self.x, self.y), xytext=(self.xoffset, self.yoffset),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                )
        self.annotation.set_visible(False)
        self.mouseX = []
        self.mouseY = []

    def clear(self, event):
        self.annotation.set_visible(False)
        self.mouseX = []
        self.mouseY = []

    def skip(self, event):
        # All done picking points
        plt.close('all')
        print('Skipping')

    def __call__(self, event):
        self.event = event
        self.x, self.y = event.mouseevent.xdata, event.mouseevent.ydata
        if self.x is not None:
            self.annotation.xy = self.x, self.y
            self.annotation.set_text(self.text_template % (self.x, self.y))
            self.annotation.set_visible(True)
            event.canvas.draw()
        self.mouseX.append(self.x)
        self.mouseY.append(self.y)

        if len(self.mouseX) >= 2:
            plt.close('all')


class InterpolatePicker(object):
    text_template = 'x: %0.2f\ny: %0.2f'
    x, y = 0.0, 0.0
    xoffset, yoffset = -20, 20
    text_template = 'x: %0.2f\ny: %0.2f'

    def __init__(self, ax, xaxis, yaxis):
        self.ax = ax
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.annotation = ax.annotate(
            self.text_template,
            xy=(self.x, self.y),
            xytext=(self.xoffset, self.yoffset),
            textcoords='offset points',
            ha='right',
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
        )
        self.annotation.set_visible(False)
        self.events = []
        self.Xs0 = []
        self.Ys0 = []
        self.Xs1 = []
        self.Ys1 = []
        self.ps = []

    def clear(self, event):
        self.annotation.set_visible(False)
        self.events = []
        self.Xs = []
        self.Ys = []
        
        # Remove picked points
        for p in self.ps:
            p.remove()

        print('Cleared')

    def done(self, event):
        # All done picking points
        plt.close('all')
        print('Done')

    def __call__(self, event):

        if not event.dblclick:
            return 0

        self.event = event
        self.x, self.y = event.xdata, event.ydata
        self.events.append(self.x)
        self.events = list(set(self.events))

        if self.x is not None:
            self.annotation.xy = self.x, self.y
            self.annotation.set_text(self.text_template % (self.x, self.y))
            self.annotation.set_visible(True)
            event.canvas.draw()

        # Handle which event it was
        if len(self.events) == 1:
            self.Xs0.append(self.x)
            self.Ys0.append(self.y)

            # Plot the picked points
            self.ps.append(self.ax.scatter(self.x, self.y))
            plt.draw()

        elif len(self.events) == 2:
            self.Xs0.append(self.x)
            self.Ys0.append(self.y)
            self.dydx0, (self.x0, self.y0) = self.getSlopePosition(
                self.Xs0, 
                self.Ys0
            )

            # Find the indices of the middle points
            self.mini = self.getIndexes(self.x0, self.y0)

            # Plot the picked points
            self.ps.append(self.ax.scatter(self.x, self.y))
            plt.draw()

        elif len(self.events) == 3:
            self.Xs1.append(self.x)
            self.Ys1.append(self.y)

            # Plot the picked points
            self.ps.append(self.ax.scatter(self.x, self.y))
            plt.draw()

        elif len(self.events) == 4:
            self.Xs1.append(self.x)
            self.Ys1.append(self.y)
            self.dydx1, (self.x1, self.y1) = self.getSlopePosition(
                self.Xs1, 
                self.Ys1
            )

            # Find the indices of the middle points
            self.maxi = self.getIndexes(self.x1, self.y1)

            # Plot the picked points
            self.ps.append(self.ax.scatter(self.x, self.y))
            plt.draw()

    def getIndexes(self, x, y):
        # Get the indiceses for chosen points from the plot
        close_x = closest(self.xaxis, x)

        # Get min and max index for handling
        i = np.where(self.xaxis == close_x)[0]

        return i 

    def getSlopePosition(self, Xs, Ys):
        # Get slope
        run = Xs[1] - Xs[0]
        rise = Ys[1] - Ys[0]
        dydx = rise / run
        x = np.mean(Xs)
        y = np.mean(Ys)

        return dydx, (x, y)


class BarPicker(object):
    text_template = 'x: %0.2f\ny: %0.2f'
    x, y = 0.0, 0.0
    xoffset, yoffset = -20, 20
    text_template = 'x: %0.2f\ny: %0.2f'

    def __init__(self, ax, xaxis, yaxis):
        self.ax = ax
        self.annotation = ax.annotate(
            self.text_template,
            xy=(self.x, self.y),
            xytext=(self.xoffset, self.yoffset),
            textcoords='offset points',
            ha='right',
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
        )
        self.annotation.set_visible(False)
        self.events = []
        self.LsX = []
        self.LsY = []
        self.xaxis = xaxis
        self.yaxis = yaxis

    def clear(self, event):
        self.annotation.set_visible(False)
        self.events = []
        self.LsX = []
        self.LsY = []
        self.X0 = None
        self.barln.remove()
        print('Cleared')

    def next(self, event):
        plt.close('all')

    def skip(self, event):
        self.popt = [False, False, False]
        self.rsquared = -99
        plt.close('all')

    def sigmoid(self, x, L, x0, k):
        y = L / (1 + np.exp(-k*(x-x0)))
        return (y)

    def draw_bar(self, event):
        # Get positions from plot call
        L = max(self.LsY) - min(self.LsY)
        x0 = self.x0X

        # Find the slope at x0
        close_x0 = closest(self.xaxis, x0)
        i = np.where(self.xaxis == close_x0)[0][0]

        # Find the end points of what was picked - X
        close_xs = [
            closest(self.xaxis, self.LsX[0]),
            closest(self.xaxis, self.LsX[1])
        ]

        # Get min and max index for handling
        mini = np.where(close_xs == min(close_xs))[0][0]
        maxi = np.where(close_xs == max(close_xs))[0][0]

        # Find the end points of what was picked - Y
        close_ys = [
            closest(self.yaxis, self.LsY[0]),
            closest(self.yaxis, self.LsY[1])
        ]

        # Get slope and fit k
        dydx = (
            (close_ys[maxi] - close_ys[mini])
            / (close_xs[maxi] - close_xs[mini])
        )
        k = (4 * dydx) / L

        # Save popt
        self.popt = [L, x0, k]

        self.barln, = self.ax.plot(
            self.xaxis,
            self.sigmoid(self.xaxis, *self.popt)
        )
        plt.draw()

    def __call__(self, event):
        self.event = event
        self.x, self.y = event.xdata, event.ydata
        self.events.append(self.x)
        self.events = list(set(self.events))

        if self.x is not None:
            self.annotation.xy = self.x, self.y
            self.annotation.set_text(self.text_template % (self.x, self.y))
            self.annotation.set_visible(True)
            event.canvas.draw()

        # Handle which event it was
        if len(self.events) == 1:
            self.LsX.append(self.x)
            self.LsY.append(self.y)
        if len(self.events) == 2:
            self.LsX.append(self.x)
            self.LsY.append(self.y)
        if len(self.events) == 3:
            self.x0X = self.x
            self.x0Y = self.y
            self.draw_bar(event)


if __name__ == "__main__":
    fig = plt.figure()
    line, = plt.plot(range(10), 'ro-')
    DC = PointPicker(plt.gca())
    fig.canvas.mpl_connect('pick_event', DC)
    line.set_picker(5) # Tolerance in points

    plt.show()

    print(DC.mouseX)
    print(DC.mouseY)


    bar = '2'
    sections = bar_sections[bar]
    idx = 0
    section = sections[idx]
    
    x = section['elev_section']['distance']
    y = section['elev_section']['value_smooth']
    
    fig, ax = plt.subplots(1, 1)
    line, = plt.plot(x, y)
    BC = BarPicker(plt.gca(), x, y)
    fig.canvas.mpl_connect('pick_event', BC)
    line.set_picker(4)
    axclear = plt.axes([0.7, 0.05, 0.1, 0.075])
    bclear = Button(axclear, 'Clear')
    bclear.on_clicked(BC.clear)
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(BC.next)
    plt.show()

    print(BC.popt)

    print(BC.X0)
    close_x = closest(x, BC.X0[0])
    i = np.where(x == close_x)[0][0]

    dydx = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
    L = max(BC.LsY) - min(BC.LsY)
    k = (4 * dydx) / L

    popt = [L, BC.X0[0], k]

    def sigmoid(x, L ,x0, k):
        y = L / (1 + np.exp(-k*(x-x0)))

        return (y)
    plt.plot(x, y)
    plt.plot(x, sigmoid(x, *popt))
    plt.show()
