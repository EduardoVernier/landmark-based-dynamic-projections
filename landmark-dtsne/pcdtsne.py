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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import lines
import time
from sklearn.decomposition import PCA

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


def ldtsne(X=np.array([]), Y_init=None, lY=np.array([]), initialization_setup=None,
           lmbda=.01, local_exaggeration=4, perplexity=30.0, max_iter=1000, timestep=0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Initialize variables
    # X = pca(X, 2).real
    (n, d) = X.shape
    no_dims = 2
    momentum = 0.5
    eta = 1  # original value was 500
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
    P = np.maximum(P, 1e-12)

    # First timestep we will perform a smarter initialization
    if t == 0:
        max_iter *= 2

    # Run iterations
    for iter in range(max_iter):
        # Compute pairwise affinities
        num = 1 / (1 + euclidian_distance(Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Force global influence on the first iterations of the first timestep
        l = lmbda
        le = local_exaggeration

        # Compute gradient. The second term is what computes "landmark attraction"
        PQ  = le * P  - Q
        for i in range(n):
            dY[i, :] = (1 - l) * np.sum(np.tile( PQ[:, i] *   num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0) \
                          + l  * (Y[i, :] - lY[i, :])

        # Perform the update
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 100 == 0:
            # C = (1 - l) * np.sum(P * np.log(P / Q))
            # lC = l * np.sum(lP * np.log(lP / lQ))
            print("Iteration {}".format(iter + 1))

        # xlims = ylims = None
        # if iter % 50 == 0:
        #     # Generate image for inspection
        #     if xlims is None:
        #         xlims = (min(Y[:,0]), max(Y[:,0]))
        #         ylims = (min(Y[:,1]), max(Y[:,1]))
        #
        #     fig, ax = plt.subplots()
        #     ax.scatter(lY[:, 0], lY[:, 1], 3, marker='x', c='k', zorder=1)
        #     ax.set_title(title + '   iter={}'.format(iter))
        #     max_dist = max(Y[:,0]) - min(Y[:,0])  # bad
        #     for a, b in zip(lY[np.arange(len(lY))], Y[np.argmax(lP, axis=1)]):
        #         alpha = np.sqrt(np.linalg.norm([a,b])) / max_dist
        #         line = lines.Line2D([a[0], b[0]], [a[1], b[1]], lw=1, color='silver', alpha=alpha, axes=ax, zorder=0)
        #         ax.add_line(line)
        #     ax.scatter(Y[:, 0], Y[:, 1], 20, categories, cmap=cm.Set3, zorder=2)
        #     ax.set_aspect('equal')
        #     ax.set_xlim(xlims)
        #     ax.set_ylim(ylims)
        #     plt.show()

    return Y


def scale_lY(lY, scaling_factor):
    """
        Make std dev of landmarks match the std dev of the first iteration of dtsne in the largest dimension.
    """
    # if np.std(Y[:, 0]) > np.std(Y[:, 1]):
    #     scaling_factor = np.std(Y[:, 0]) / np.std(lY[:, 1])
    # else:
    #     scaling_factor = np.std(Y[:, 1]) / np.std(lY[:, 0])
    if scaling_factor == 1:
        return lY
    else:
        return (lY - np.mean(lY.reshape(-1, 2), axis=0)) * scaling_factor + np.mean(lY.reshape(-1, 2), axis=0)


# def save_scaled_landmarks(lY, landmarks_file, timestamp):
#     df_landmarks = pd.read_csv(landmarks_file, index_col=0)
#     df_landmarks['y0'] = lY[:, 0]
#     df_landmarks['y1'] = lY[:, 1]
#     landmarks_file = landmarks_file.replace('.csv', '-' + timestamp + '.csv')
#     df_landmarks.to_csv(landmarks_file)


if __name__ == "__main__":
    np.random.seed(0)
    dataset_id = 'walk'
    dataset_dir = './datasets/{}/'.format(dataset_id)
    print(dataset_id)

    # Read dataset
    Xs, labels, categories = shared.read_dataset(dataset_dir)

    # Params
    initialization = None # {'l': .6, 'ge': 8, 'le': 8}
    perplexity_list = [30]
    lambda_list = [.0001, .001, .01]  # [.25, .5, .75]
    local_exageration_list = [1]
    landmark_scaling = [.1, 1., 10.]
    max_iter = 200  # 1000 is default
    param_grid = itertools.product(perplexity_list, lambda_list, local_exageration_list, landmark_scaling)

    (n, d) = Xs[0].shape
    lY = PCA(n_components=2).fit_transform(np.array(Xs).reshape((-1, d))).reshape((-1, n, 2))

    for p, l, le, ls in param_grid:
        Y = None
        Ys = []
        slY = scale_lY(lY, ls)  # Scale landmarks
        l_str = '{:1.6f}'.format(l).replace('.', '_')
        ls_str = '{:.2f}'.format(ls).replace('.', '_')
        title = '{}-pcadtsne-p{}-l{}-le{}-ls{}'.format(dataset_id, p, l_str, le, ls_str)
        timestamp = str(int(time.time()))
        print(title)
        for t in range(len(Xs)):
            print('Timestep: ' + str(t))

            Y = ldtsne(Xs[t], Y, slY[t], lmbda=l, perplexity=p, local_exaggeration=le, max_iter=max_iter, timestep=t)
            Ys.append(Y)

            # Show results
            if t == len(Xs) - 1:
                fig, ax = plt.subplots()
                ax.scatter(slY.reshape(-1, 2)[:, 0], slY.reshape(-1, 2)[:, 1], 3, marker='x', c='k', zorder=1)
                ax.set_title(title)
                max_dist = max(Y[:,0]) - min(Y[:,0])  # bad
                ax.scatter(Y[:, 0], Y[:, 1], 20, categories, cmap=cm.Set3, zorder=2)
                ax.set_aspect('equal')
                plt.show()

        # Save results
        df_out = pd.DataFrame(index=labels)
        for t in range(len(Ys)):
            df_out['t{}d0'.format(t)] = Ys[t].T[0]  # Only doing 2D for now
            df_out['t{}d1'.format(t)] = Ys[t].T[1]

        df_out.to_csv('./tests/dynamic-tests/{}.csv'.format(title), index_label='id')
