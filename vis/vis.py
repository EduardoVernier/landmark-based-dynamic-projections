import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation
import sys
import re


def get_projection_as_array(path):
    df = pd.read_csv(path, index_col=0)
    vs = df.values.reshape(len(df), -1, 2)
    return vs, df.index, vs.shape[1]


class Projection:
    def __init__(self, path):
        self.name = re.match(r'.*/(.*).csv', path).group(1)
        pos, self.indices, self.n_timesteps = get_projection_as_array(path)
        colormap = matplotlib.cm.Set3
        self.colors = [colormap(cl) for cl in pd.factorize(self.indices.str.split('-').str[0])[0]]

        # # Shuffle arrays
        # perm = np.random.permutation(len(ind))
        # pos = pos[perm]
        # indexes[p] = ind[perm]

        # Set x and y axis limits
        x_max = max(pos[:, :, 0].flatten())
        x_min = min(pos[:, :, 0].flatten())
        y_max = max(pos[:, :, 1].flatten())
        y_min = min(pos[:, :, 1].flatten())
        x_max = x_max + (x_max - x_min) * .03
        x_min = x_min - (x_max - x_min) * .03
        y_max = y_max + (y_max - y_min) * .03
        y_min = y_min - (y_max - y_min) * .03
        self.limits = (x_min, x_max, y_min, y_max)

        # Add more points with akima interpolation
        n_nans = 9
        pos_int = []
        for points in pos:
            extended = []
            for po in points:
                extended.append(list(po))
                for i in range(n_nans):
                    extended.append([np.nan, np.nan])
            df = pd.DataFrame(extended)
            df = df.interpolate(method='akima')
            df = df.dropna()
            pos_int.append(df.values)
        self.coords = np.swapaxes(np.array(pos_int), 0, 1)


if __name__ == '__main__':

    projections = []
    for i in range(1, len(sys.argv)):
        projections.append(Projection(sys.argv[i]))

    print(len(projections[0].coords))

    fig, ax = plt.subplots()
    xdata, ydata = [], []
    # ln, = plt.plot([], [], 'ro')


    scatter = ax.scatter(projections[0].coords[0][:, 0], projections[0].coords[0][:, 1], s=10, edgecolors='#000000', animated=True)

    # if n_proj < 4:
    #     cls.fig, cls.axes = plt.subplots(ncols=n_proj, nrows=1, figsize=(n_proj * 5, 5))
    # else:
    #     ncols = 4
    #     nrows = (n_proj - 1) // 4 + 1
    #     cls.fig, cls.axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * 5, nrows * 5))
    #     cls.axes = cls.axes.flatten()


    def init():
        ax.set_xlim(projections[0].limits[:2])
        ax.set_ylim(projections[0].limits[2:])
        return scatter,

    def update(frame):
        # scatter.set_data(projections[0].coords[frame][:, 0], projections[0].coords[frame][:, 1])
        scatter.set_offsets(np.vstack((projections[0].coords[frame][:, 0], projections[0].coords[frame][:, 1])).T)
        scatter.set_color(projections[0].colors)
        scatter.set_edgecolor('k')
        scatter.set_linewidth(.3)
        return scatter,

    ani = FuncAnimation(fig, update, frames=range(len(projections[0].coords)),
                        init_func=init, interval=1, blit=True, repeat=True)
    plt.show()