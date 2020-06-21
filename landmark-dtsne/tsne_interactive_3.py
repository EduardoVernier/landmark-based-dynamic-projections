#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python
# 2.7.10, and it requires a working installation of NumPy. The implementation
# comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

import itertools
import sys
from numba import jit

import numpy as np
import pandas as pd
from natsort import natsorted
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import multiprocessing as mp
import matplotlib.cm as cm
import time
import shared
from sklearn.decomposition import PCA

# Shared variables
lmbda = mp.Value('d', .0)
landmark_scale = mp.Value('d', 1.)
vector_size = mp.Value('d', .002)
global_exaggeration = mp.Value('d', 1)
local_exaggeration = mp.Value('d', 1)
t = mp.Value('i', 0)
save_state = mp.Value('i', False)
# https://docs.python.org/3/library/multiprocessing.shared_memory.html#module-multiprocessing.shared_memory
lock = mp.Lock()


def euclidian_distance(X, Y=None):
    if type(Y) is not np.ndarray:
        Y = X
    return (X ** 2).sum(1).reshape((X.shape[0], 1)) + (Y ** 2).sum(1).reshape((1, Y.shape[0])) - 2 * X.dot(Y.T)


def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
        Note that
        Perp(Pi) = 2^H(Pi)
        log Perp(Pi) = H(Pi)
        and
        Pi = −sum(pj|i log2 pj|i)
    """
    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = np.maximum(sum(P), 1e-12)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def p_given_L_betas(lX, X, beta):
    """
    Given the precomputed landmark betas, compute p for each pair landmark-projected point
    """
    # print("Computing p given X and lX betas...")
    n, _ = lX.shape
    m, _ = X.shape

    assert len(beta) == len(lX), print('len(beta) != len(lX)')

    D = euclidian_distance(lX, X)
    P = np.zeros((n, m))

    # Loop over all datapoints
    for i in range(n):
        # Compute the Gaussian kernel and entropy for the current precision
        Di = D[i, :]
        (H, thisP) = Hbeta(Di, beta[i])

        # Set the final row of P
        P[i, :] = thisP

    # Return final P-matrix
    return P


def regular_p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
        Instead of testing different sigma values, we test different beta values where
        Beta = np.sqrt(1 / beta)
    """
    # print("Computing pairwise distances...")
    (n, d) = X.shape
    D = euclidian_distance(X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):
        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    # print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P, beta


def pca(X=np.array([]), no_dims=2):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y

def ldtsne(Xs=np.array([]), Y_init=None, lX=np.array([]), lY=np.array([]),
           perplexity=30.0, max_iter=1000, title='', plotter=None, index=np.array([])):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """
    global global_exaggeration, local_exaggeration, lmbda, landmark_scale
    # Initialize variables
    X = Xs[0]
    (n, d) = X.shape
    momentum = 0.5
    eta = 1  # original value was 500
    min_gain = 0.01

    no_dims = 2
    dY = np.zeros((n, no_dims))
    dY_local = np.zeros((n, no_dims))
    dY_global = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))
    df_out = pd.DataFrame(index=index)
    if Y_init is None:
        Y = np.random.randn(n, no_dims)  # Random initialization
    else:
        Y = Y_init


    # Precompute P and lP for all timesteps
    # _, L_betas = regular_p(lX, 1e-5, perplexity)  # Compute betas for L
    Ps = []
    # lPs = []
    for X in Xs:
        # Compute P-values and betas for X
        P, _ = regular_p(X, 1e-5, perplexity)
        P = P + np.transpose(P)
        P = P / np.sum(P)
        # P = P * 4.  # early exaggeration
        P = np.maximum(P, 1e-12)
        Ps.append(P)


    # Run iterations
    for iter in range(max_iter):        # Compute pairwise affinities
        num = 1 / (1 + euclidian_distance(Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        scaled_lY = (lY[t.value] - np.mean(lY[t.value], axis=0)) * landmark_scale.value + np.mean(lY[t.value], axis=0)
        # Compute pairwise affinities between Y and lY (projected landmarks)

        l = lmbda.value
        # Compute gradient. The second term is what computes "landmark attraction"
        if t.value < len(Xs):
            PQ = local_exaggeration.value * Ps[t.value] - Q
            # lPQ = global_exaggeration.value * lPs[t.value] - lQ
            for i in range(n):
                dY_local[i, :] = (1 - l) * np.sum(np.tile( PQ[:, i] *   num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)
                # dY_global[i, :] = l  * np.sum(np.tile(lPQ[:, i] * l_num[:, i], (no_dims, 1)).T * (Y[i, :] - scaled_lY), 0)
                dY_global[i, :] = l  * (Y[i, :] - scaled_lY[i, :])
                dY[i, :] = dY_local[i, :] + dY_global[i, :]

            # Perform the update
            gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                    (gains * 0.8) * ((dY > 0.) == (iY > 0.))
            gains[gains < min_gain] = min_gain
            iY = momentum * iY - eta * (gains * dY)
            Y = Y + iY
            Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        plotter.plot((Y, dY_local, dY_global, iter))
        time.sleep(.005)

        if save_state.value:
            # Save results
            df_out['t{}d0'.format(t.value)] = Y.T[0]  # Only doing 2D for now
            df_out['t{}d1'.format(t.value)] = Y.T[1]
            df_out = df_out.reindex(natsorted(df_out.columns), axis=1)
            df_out.to_csv('./tests/dynamic-tests/{}.csv'.format(title), index_label='id')
            lock.acquire()
            save_state.value = False
            lock.release()


class ProcessPlotter(object):
    def __init__(self, n_points, colors, landmarks):
        self.landmarks = landmarks
        self.n_points = n_points
        colormap = matplotlib.cm.Set1
        self.colors = [colormap(cl) for cl in colors]
        for _ in range(len(landmarks)):
            self.colors.insert(0, (1., 1., 1., .3))

    def terminate(self):
        plt.close('all')

    def call_back(self):
        while self.pipe.poll():
            command = self.pipe.recv()
            if command is None:
                self.terminate()
                return False
            else:
                pts, dY_local, dY_global, iter = command
                title = 'iter={}  lambda={:.5f}  ge={:.2f}  le={:.2f}  ls={:.2f}  t={}'.format(iter, lmbda.value, global_exaggeration.value, local_exaggeration.value, landmark_scale.value, t.value)
                self.ax.set_title(title)
                lY = self.landmarks
                scaled_lY = (lY - np.mean(lY, axis=0)) * landmark_scale.value + np.mean(lY, axis=0)
                x = np.append(scaled_lY[:, 0], pts[:, 0])
                y = np.append(scaled_lY[:, 1], pts[:, 1])
                self.scatter.set_offsets(np.vstack((x, y)).T)

                self.quiver_local.set_offsets(np.vstack((pts[:, 0], pts[:, 1])).T)
                self.quiver_local.U = -dY_local[:, 0]
                self.quiver_local.V = -dY_local[:, 1]
                self.quiver_local.scale = 10e10 #vector_size.value


                self.quiver_global.set_offsets(np.vstack((pts[:, 0], pts[:, 1])).T)
                self.quiver_global.U = -dY_global[:, 0]
                self.quiver_global.V = -dY_global[:, 1]
                self.quiver_global.scale = 10e10 # vector_size.value

                x_min = x.min()
                x_max = x.max()
                y_min = y.min()
                y_max = y.max()
                x_max = x_max + (x_max - x_min) * .03
                x_min = x_min - (x_max - x_min) * .03
                y_max = y_max + (y_max - y_min) * .03
                y_min = y_min - (y_max - y_min) * .03
                self.ax.set_xlim(x_min, x_max)
                self.ax.set_ylim(y_min, y_max)
        self.fig.canvas.draw()
        return True

    def __call__(self, pipe):
        print('starting plotter...')
        self.pipe = pipe
        self.fig, self.ax = plt.subplots()
        timer = self.fig.canvas.new_timer(interval=0)

        R = np.random.randn(self.n_points, 2)  # Random initialization
        x = np.append(self.landmarks[:, 0], R[:, 0])
        y = np.append(self.landmarks[:, 1], R[:, 1])
        self.scatter = self.ax.scatter(x, y, s=25, c=self.colors, cmap=cm.Set3, zorder=100)
        self.scatter.set_edgecolor((.6, .6, .6, 1.))
        self.scatter.set_linewidth(.3)

        X, Y = R[:, 0], R[:, 1]
        U = np.ones_like(X)
        V = np.ones_like(X)
        # self.quiver_local = self.ax.quiver(X, Y, U, V, units='xy', scale=.001, color='r')
        # self.quiver_global = self.ax.quiver(X, Y, U, V, units='xy', scale=.001, color='b')
        self.quiver_local = self.ax.quiver(X, Y, U, V, units='width', scale=vector_size.value, color='r')
        self.quiver_global = self.ax.quiver(X, Y, U, V, units='width', scale=vector_size.value, color='b')


        self.fig.canvas.mpl_connect('key_press_event', self.key_press)
        timer.add_callback(self.call_back)
        timer.start()
        print('...done')
        plt.show()

    def key_press(self, event):
        print('press', event.key)
        sys.stdout.flush()
        global global_exaggeration, local_exaggeration, lmbda

        if event.key == '1':
            lock.acquire()
            lmbda.value = max(0, lmbda.value - .0001)
            print('Lambda: ', lmbda.value)
            lock.release()
        if event.key == '2':
            lock.acquire()
            lmbda.value = min(1, lmbda.value + .0001)
            print('Lambda: ', lmbda.value)
            lock.release()

        if event.key == '3':
            lock.acquire()
            global_exaggeration.value *= 0.9 # max(0, global_exaggeration.value - 1)
            print('Global Exaggeration: ', global_exaggeration.value)
            lock.release()
        if event.key == '4':
            lock.acquire()
            global_exaggeration.value *= 1.1  # min(16, global_exaggeration.value + 1)
            print('Global Exaggeration: ', global_exaggeration.value)
            lock.release()

        if event.key == '5':
            lock.acquire()
            local_exaggeration.value *= 0.9  # max(0, local_exaggeration.value - 1)
            print('Local Exaggeration: ', local_exaggeration.value)
            lock.release()
        if event.key == '6':
            lock.acquire()
            local_exaggeration.value *= 1.1  # min(16, local_exaggeration.value + 1)
            print('Local Exaggeration: ', local_exaggeration.value)
            lock.release()

        if event.key == '7':
            lock.acquire()
            landmark_scale.value *= 0.9
            print('Landmarks scaled by: ', landmark_scale.value)
            lock.release()
        if event.key == '8':
            lock.acquire()
            landmark_scale.value *= 1.1
            print('Landmarks scaled by: ', landmark_scale.value)
            lock.release()

        if event.key == '9':
            lock.acquire()
            t.value = max(0, t.value - 1)
            print('t: ', t.value)
            lock.release()
        if event.key == '0':
            lock.acquire()
            t.value += 1
            print('t: ', t.value)
            lock.release()

        if event.key == 'w':
            lock.acquire()
            save_state.value = True
            print('Save current state - t: ', t.value)
            lock.release()

        if event.key == 'n':
            lock.acquire()
            vector_size.value *= 1.1
            print('vector_size scaled by: ', vector_size.value)
            lock.release()
        if event.key == 'm':
            lock.acquire()
            vector_size.value *= .9
            print('vector_size scaled by: ', vector_size.value)
            lock.release()


class NBPlot(object):
    def __init__(self, n_points, colors, landmarks):
        self.plot_pipe, plotter_pipe = mp.Pipe()
        self.plotter = ProcessPlotter(n_points, colors, landmarks)
        self.plot_process = mp.Process(target=self.plotter, args=(plotter_pipe,), daemon=True)
        self.plot_process.start()

    def plot(self, data):
        send = self.plot_pipe.send
        send(data)


if __name__ == "__main__":
    np.random.seed(0)
    dataset_id = 'minigaussians'
    dataset_dir = './datasets/{}/'.format(dataset_id)
    print(dataset_id)

    # Read dataset
    Xs, labels, categories = shared.read_dataset(dataset_dir)

    # Params
    p = 30
    landmark_scaling = False
    max_iter = 100000  # 1000 is default

    (n, d) = Xs[0].shape
    lY = PCA(n_components=2).fit_transform(np.array(Xs).reshape((-1, d))).reshape((-1, n, 2))

    pl = NBPlot(len(Xs[0]), categories, lY.reshape(-1, 2))

    Y = None
    Ys = []
    title = '{}-pcdtsne-p{}--interactive-{}'.format(dataset_id, p, '-')
    # timestamp = str(int(time.time()))
    # print(title)

    ldtsne(Xs, Y, None, lY, perplexity=p, max_iter=max_iter, title=title, plotter=pl, index=labels)
