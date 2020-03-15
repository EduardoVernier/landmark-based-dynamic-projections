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
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import multiprocessing as mp
import matplotlib.cm as cm
from matplotlib import lines
import time

from matplotlib.animation import FuncAnimation

import shared


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


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def ldtsne(X=np.array([]), Y_init=None, lX=np.array([]), lY=np.array([]), lmbda=.01, global_exaggeration=4, no_dims=2,
           perplexity=30.0, max_iter=1000, timestep=0, plotter=None):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """
    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    # X = pca(X, 2).real
    (n, d) = X.shape
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500  # original value was 500
    min_gain = 0.01
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))
    if Y_init is None:
        Y = np.random.randn(n, no_dims)  # Random initialization
        # Y = pca(X, no_dims)  # PCA initialization
    else:
        Y = Y_init

    # Compute P-values and betas for X
    P, X_betas = regular_p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    # P = P * 4.  # early exaggeration
    P = np.maximum(P, 1e-12)

    # Compute betas for L
    _, L_betas = regular_p(lX, 1e-5, perplexity)

    # Compute P-matrix between X and L with previously computed beta values
    lP = p_given_L_betas(lX, X, L_betas)
    lP = lP / np.sum(lP)
    lP = lP * global_exaggeration  # early exaggeration (forever?) -- if we remove it the points won't fall inside the convex hull
    lP = np.maximum(lP, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        num = 1 / (1 + euclidian_distance(Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute pairwise affinities between Y and lY (projected landmarks)
        l_num = 1 / (1 + euclidian_distance(lY, Y))
        lQ = l_num / np.sum(l_num)
        lQ = np.maximum(lQ, 1e-12)

        # Force global influence on the first iterations of the first timestep
        if timestep == 0 and iter < 100:
            l = 1
        else:
            l = lmbda

        # Compute gradient. The second term is what computes "landmark attraction"
        PQ = P - Q
        lPQ = lP - lQ
        for i in range(n):
            dY[i, :] = (1 - l) * np.sum(np.tile( PQ[:, i] *   num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0) \
                          + l  * np.sum(np.tile(lPQ[:, i] * l_num[:, i], (no_dims, 1)).T * (Y[i, :] - lY), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 100 == 0:
            C = (1 - l) * np.sum(P * np.log(P / Q))
            lC = l * np.sum(lP * np.log(lP / lQ))
            print("Iteration {}: error local={}  global={}".format(iter + 1, C, lC))

        pl.plot((Y, iter))
        time.sleep(.1)


def scale_lY(lY, Y):
    """
        Make std dev of landmarks match the std dev of the first iteration of dtsne in the largest dimension.
    """
    if np.std(Y[:, 0]) > np.std(Y[:, 1]):
        scaling_factor = np.std(Y[:, 0]) / np.std(lY[:, 1])
    else:
        scaling_factor = np.std(Y[:, 1]) / np.std(lY[:, 0])
    return (lY - np.mean(lY, axis=0)) * scaling_factor + np.mean(Y, axis=0)


def save_scaled_landmarks(lY, landmarks_file, timestamp):
    df_landmarks = pd.read_csv(landmarks_file, index_col=0)
    df_landmarks['y0'] = lY[:, 0]
    df_landmarks['y1'] = lY[:, 1]
    landmarks_file = landmarks_file.replace('.csv', '-' + timestamp + '.csv')
    df_landmarks.to_csv(landmarks_file)


class ProcessPlotter(object):
    def __init__(self, n_points, colors):
        self.n_points = n_points
        self.colors = colors

    def terminate(self):
        plt.close('all')

    def call_back(self):
        while self.pipe.poll():
            command = self.pipe.recv()
            if command is None:
                self.terminate()
                return False
            else:
                pts, iter = command
                xmin, ymin = pts.min(axis=0)
                xmax, ymax = pts.max(axis=0)
                self.ax.set_xlim(xmin, xmax)
                self.ax.set_ylim(ymin, ymax)
                self.ax.set_title(iter)
                self.scatter.set_offsets(np.vstack((pts[:, 0], pts[:, 1])).T)
        self.fig.canvas.draw()
        return True

    def __call__(self, pipe):
        print('starting plotter...')
        self.pipe = pipe
        self.fig, self.ax = plt.subplots()
        timer = self.fig.canvas.new_timer(interval=0)

        Y = np.random.randn(self.n_points, 2)  # Random initialization
        self.scatter = self.ax.scatter(Y[:, 0], Y[:, 1], s=10, c=self.colors, cmap=cm.Set3)

        timer.add_callback(self.call_back)
        timer.start()
        print('...done')
        plt.show()


class NBPlot(object):
    def __init__(self, n_points, colors):
        self.plot_pipe, plotter_pipe = mp.Pipe()
        self.plotter = ProcessPlotter(n_points, colors)
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
    perplexity_list = [30]
    lambda_list = [.0]  # [.25, .5, .75]
    global_exaggeration_list = [4]  # [2, 4, 8]
    landmark_scaling = [False]
    max_iter = 1000  # 1000 is default
    param_grid = itertools.product(perplexity_list, lambda_list, global_exaggeration_list, landmark_scaling)

    # Read landmarks
    landmarks_file = './generate-landmarks/output/{}-krandom-n-PCA.csv'.format(dataset_id)
    landmarks_info = landmarks_file.split('/')[-1].split('-', 1)[1][:-4]
    # landmarks_info = landmarks_info + '-ls' + str(int(landmark_scaling))
    df_landmarks = pd.read_csv(landmarks_file, index_col=0)
    lX = df_landmarks[[c for c in df_landmarks.columns if c.startswith('x')]].values
    lY = df_landmarks[[c for c in df_landmarks.columns if c.startswith('y')]].values

    pl = NBPlot(len(Xs[0]), categories)

    for p, l, ge, ls in param_grid:
        Y = None
        Ys = []
        l_str = '{:1.4f}'.format(l).replace('.', '_')
        title = '{}-ldtsne-p{}-l{}-ge{}-{}'.format(dataset_id, p, l_str, ge, landmarks_info)
        timestamp = str(int(time.time()))
        print(title)
        for t in [0]:
            X = Xs[t]
            print('Timestep: ' + str(t))
            ldtsne(X, Y, lX, lY, lmbda=l, perplexity=p, global_exaggeration=ge, no_dims=2, max_iter=max_iter, timestep=t, plotter=pl)
