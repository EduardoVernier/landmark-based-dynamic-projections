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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import lines

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
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P

def p_given_L_betas(lX, X, beta):

    print("Computing p given X and lX betas...")
    n, _ = lX.shape
    m, _ = X.shape

    assert len(beta) == len(lX), print('len(beta) != len(lX)')

    D = euclidian_distance(lX, X)
    P = np.zeros((n, m))
    # beta = np.ones((n, 1))
    # logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        # Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:m]))]
        Di = D[i, :]
        (H, thisP) = Hbeta(Di, beta[i])

        # Set the final row of P
        # P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:m]))] = thisP
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
    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    # sum_X = np.sum(np.square(X), 1)
    # D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    D = euclidian_distance(X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

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
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P, beta


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """
    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def ldtsne(X=np.array([]), Y_init=None, lX=np.array([]), lY=np.array([]), lmbda=.01, no_dims=2, initial_dims=50, perplexity=30.0, max_iter=1000):
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
    # X = pca(X, initial_dims).real
    (n, d) = X.shape
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 1  # original value was 500
    min_gain = 0.01
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))
    if Y_init is None:
        Y = np.random.randn(n, no_dims)
        first_revision = True
    else:
        Y = Y_init
        first_revision = False


    # Compute P-values and betas for X
    P, X_betas = regular_p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.  # early exaggeration
    P = np.maximum(P, 1e-12)

    # Compute betas for L
    _, L_betas = regular_p(lX, 1e-5, perplexity)

    # Compute P-matrix between X and L with previously computed beta values
    lP = p_given_L_betas(lX, X, L_betas)
    lP = lP / np.sum(lP)
    lP = lP * 4.  # early exaggeration (forever?) -- if we remove it the points won't fall inside the convex hull
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

        # Compute gradient. The second term is what computes "landmark attraction"
        PQ = P - Q
        lPQ = lP - lQ
        for i in range(n):
            dY[i, :] = (1 - lmbda) * np.sum(np.tile( PQ[:, i] *   num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0) \
                          + lmbda  * np.sum(np.tile(lPQ[:, i] * l_num[:, i], (no_dims, 1)).T * (Y[i, :] - lY), 0)


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
        if (iter + 1) % 10 == 0:
            C = (1 - lmbda) * np.sum(P * np.log(P / Q))
            lC = lmbda * np.sum(lP * np.log(lP / lQ))
            print("Iteration {}: error is local={}  global={}".format(iter + 1, C, lC))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.
            # lP = lP / 4.  # Points fall outside the convex hull if normal P-values for lP are reestablished

    return Y, lY, lP


if __name__ == "__main__":
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    # print("Running example on 2,500 MNIST digits...")

    seed = 0

    dataset_id = 'walk'
    # revision = 50
    dataset_dir = './datasets/{}/'.format(dataset_id)
    print(dataset_id)

    # Read dataset
    Xs, labels = shared.read_dataset(dataset_dir)

    # Read landmarks
    df = pd.read_csv('./landmarking/output/{}_krandom_1000_PCA.csv'.format(dataset_id), index_col=0)
    lX = df[[c for c in df.columns if c.startswith('x')]].values
    lY = df[[c for c in df.columns if c.startswith('y')]].values

    p = 30
    l = 0.0
    Y = None
    xlims = ylims = None
    for revision in range(len(Xs)):
        X = Xs[revision]
        l_str = '{:.4f}'.format(l)
        title = '{}-p{}-l{}-rev{}.png'.format(dataset_id, p, l_str.replace('.','_'), revision)
        print(title)

        Y, lY, lP = ldtsne(X, Y, lX, lY, lmbda=l, perplexity=p, no_dims=2, max_iter=1000)

        if xlims is None:
            xlims = (min(Y[:,0]), max(Y[:,0]))
            ylims = (min(Y[:,1]), max(Y[:,1]))

        fig, ax = plt.subplots()
        ax.scatter(lY[:, 0], lY[:, 1], 3, marker='x', c='k', zorder=1)
        ax.set_title(title)
        max_dist = max(Y[:,0]) - min(Y[:,0])  # bad
        for a, b in zip(lY[np.arange(len(lY))], Y[np.argmax(lP, axis=1)]):
            alpha = np.sqrt(np.linalg.norm([a,b])) / max_dist
            line = lines.Line2D([a[0], b[0]], [a[1], b[1]], lw=1, color='silver', alpha=alpha, axes=ax, zorder=0)
            ax.add_line(line)
        ax.scatter(Y[:, 0], Y[:, 1], 20, labels, cmap=cm.Set3, zorder=2)
        ax.set_aspect('equal')
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        plt.show()
        fig.savefig('./landmark-dtsne/{}'.format(title))
