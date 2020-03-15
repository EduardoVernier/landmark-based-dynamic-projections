import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
import sys
import re

anim_running = True
anim = None

def get_projection_as_array(path):
    df = pd.read_csv(path, index_col=0)
    vs = df.values.reshape(len(df), -1, 2)
    return vs, df.index, vs.shape[1]


class Projection:
    def __init__(self, path):
        self.name = re.match(r'.*/(.*).csv', path).group(1)
        pos, self.indexes, self.n_timesteps = get_projection_as_array(path)
        colormap = matplotlib.cm.Set3
        self.colors = [colormap(cl) for cl in pd.factorize(self.indexes.str.split('-').str[0])[0]]

        self.landmarks = []
        if 'ldtsne' in self.name:
            landmark_file = 'generate-landmarks/output/' + self.name.split('-')[0] + '-' + '-'.join(self.name.split('-')[5:]) + '.csv'
            self.landmarks = pd.read_csv(landmark_file, index_col=0).values[:, -2:]

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

        if self.n_timesteps > 1:
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
        else:
            self.coords = np.swapaxes(np.array(pos), 0, 1)


def key_press(event):
    global pause, autopause
    print('press', event.key)
    sys.stdout.flush()
    if event.key == ' ':  # Continue animation with spacebar
        global anim_running, anim
        if anim_running:
            anim.event_source.stop()
            anim_running = False
        else:
            anim.event_source.start()
            anim_running = True
    if event.key == '+':
        if anim.event_source.interval > 0:
            anim.event_source.interval *= 2
        else:
            anim.event_source.interval = 1
        print(anim.event_source.interval)
    if event.key == '-':
        anim.event_source.interval *= .5
        print(anim.event_source.interval)

if __name__ == '__main__':

    projections = []
    for i in range(1, len(sys.argv)):
        projections.append(Projection(sys.argv[i]))

    n_timesteps = projections[0].n_timesteps
    n_proj = len(projections)
    if n_proj == 1:
        fig, axes = plt.subplots(ncols=n_proj, nrows=1, figsize=(n_proj * 5, 5))
        axes = [axes]
    elif n_proj < 4:
        fig, axes = plt.subplots(ncols=n_proj, nrows=1, figsize=(n_proj * 5, 5))
    else:
        ncols = 4
        nrows = (n_proj - 1) // 4 + 1
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * 5, nrows * 5))
        axes = axes.flatten()

    # Initial draw of projections
    scatters = []
    for i, p in enumerate(projections):
        x = p.coords[0][:, 0]
        y = p.coords[0][:, 1]
        c = p.colors
        s = np.full(len(c), 10)
        if len(p.landmarks):
            x = np.append(p.landmarks[:, 0], x)
            y = np.append(p.landmarks[:, 1], y)
            s = np.append(np.full(len(p.landmarks), 2), s)
            for _ in range(len(p.landmarks)):
                c.insert(0, (1., 1., 1., .3))

        scatter = axes[i].scatter(x, y, s=s, animated=True)
        scatter.set_color(c)
        scatter.set_edgecolor((.6, .6, .6, 1.))
        scatter.set_linewidth(.3)
        scatters.append(scatter)
        axes[i].set_title(p.name, fontsize=8)

    time_text = plt.text(0, 0, '0', fontsize=12, transform=axes[0].transAxes, animated=True)

    def init():
        for i, p in enumerate(projections):
            axes[i].set_xlim(p.limits[:2])
            axes[i].set_ylim(p.limits[2:])

        return tuple(scatters + [time_text])

    def update(frame):
        for i, p in enumerate(projections):
            x = p.coords[frame][:, 0]
            y = p.coords[frame][:, 1]
            if len(p.landmarks):
                x = np.append(p.landmarks[:, 0], x)
                y = np.append(p.landmarks[:, 1], y)
            scatters[i].set_offsets(np.vstack((x, y)).T)

        time_text.set_text(str(math.floor(frame/10))[:4])
        return tuple(scatters + [time_text])

    anim = FuncAnimation(fig, update, frames=range(len(projections[0].coords)),
                         init_func=init, interval=1., blit=True, repeat=True)

    fig.canvas.mpl_connect('key_press_event', key_press)

    plt.show()
