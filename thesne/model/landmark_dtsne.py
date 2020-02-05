import numpy as np
import theano
import theano.tensor as T

from sklearn.utils import check_random_state

from core import floath
from core import cost_var, landmark_cost_var
from core import find_sigma, find_sigma_old

theano.config.exception_verbosity = 'high'



def find_Ys_with_landmarks(Xs_shared, Ys_shared, sigmas_shared, N, steps, output_dims,
                           lX, lY, lsigmas, N_landmarks,
                           n_epochs, initial_lr, final_lr, lr_switch, init_stdev,
                           initial_momentum, final_momentum, momentum_switch, lmbda, metric, verbose=0):
    """Optimize cost wrt Ys[t], simultaneously for all t"""
    # Optimization hyperparameters
    initial_lr = np.array(initial_lr, dtype=floath)
    final_lr = np.array(final_lr, dtype=floath)
    initial_momentum = np.array(initial_momentum, dtype=floath)
    final_momentum = np.array(final_momentum, dtype=floath)

    lr = T.fscalar('lr')
    lr_shared = theano.shared(initial_lr)

    momentum = T.fscalar('momentum')
    momentum_shared = theano.shared(initial_momentum)

    # Penalty hyperparameter
    lmbda_var = T.fscalar('lmbda')
    lmbda_shared = theano.shared(np.array(lmbda, dtype=floath))

    # Yv velocities
    Yvs_shared = []
    zero_velocities = np.zeros((N, output_dims), dtype=floath)
    for t in range(steps):
        Yvs_shared.append(theano.shared(np.array(zero_velocities)))

    # Cost
    Xvars = T.fmatrices(steps)
    Yvars = T.fmatrices(steps)
    Yv_vars = T.fmatrices(steps)
    sigmas_vars = T.fvectors(steps)
    lX_vars = T.fmatrices(steps)
    lY_vars = T.fmatrices(steps)
    lsigmas_vars = T.fvectors(steps)

    c_vars = []
    for t in range(steps):
        c_vars.append(cost_var(Xvars[t], Yvars[t], sigmas_vars[t], metric) +
                      lmbda_var * landmark_cost_var(Xvars[t], Yvars[t], sigmas_vars[t], lX_vars[t], lY_vars[t], lsigmas_vars[t], metric))


    cost = T.sum(c_vars) # + lmbda_var * movement_penalty(Yvars, N)

    # Setting update for Ys velocities
    grad_Y = T.grad(cost, Yvars)
    # clip = 1
    # grad_Y = T.grad(theano.gradient.grad_clip(cost, -1 * clip, clip), Yvars)

    givens = {lr: lr_shared, momentum: momentum_shared,
              lmbda_var: lmbda_shared}
    updates = []
    for t in range(steps):
        updates.append((Yvs_shared[t], momentum * Yv_vars[t] - lr * grad_Y[t]))

        givens[Xvars[t]] = Xs_shared[t]
        givens[Yvars[t]] = Ys_shared[t]
        givens[Yv_vars[t]] = Yvs_shared[t]
        givens[sigmas_vars[t]] = sigmas_shared[t]
        givens[lX_vars[t]] = lX
        givens[lY_vars[t]] = lY
        givens[lsigmas_vars[t]] = lsigmas[t]

    update_Yvs = theano.function([], cost, givens=givens, updates=updates, on_unused_input='warn')

    # Setting update for Ys positions
    updates = []
    givens = dict()
    for t in range(steps):
        updates.append((Ys_shared[t], Yvars[t] + Yv_vars[t]))
        givens[Yvars[t]] = Ys_shared[t]
        givens[Yv_vars[t]] = Yvs_shared[t]

    update_Ys = theano.function([], [], givens=givens, updates=updates)

    # Momentum-based gradient descent
    for epoch in range(n_epochs):
        if epoch == lr_switch:
            lr_shared.set_value(final_lr)
        if epoch == momentum_switch:
            momentum_shared.set_value(final_momentum)

        c = update_Yvs()
        update_Ys()
        if verbose:
            print('Epoch: {0}. Cost: {1:.6f}.'.format(epoch + 1, float(c)))

    Ys = []
    for t in range(steps):
        Ys.append(np.array(Ys_shared[t].get_value(), dtype=floath))

    return Ys


def landmark_dtsne(Xs, lX, lY, perplexity=30, Ys=None, output_dims=2, n_epochs=1000,
                   initial_lr=2400, final_lr=200, lr_switch=80, init_stdev=1e-4,
                   initial_momentum=0.5, final_momentum=0.8,
                   momentum_switch=80, lmbda=0.0, metric='euclidean',
                   random_state=None, verbose=1):
    random_state = check_random_state(random_state)

    steps = len(Xs)
    N = Xs[0].shape[0]
    N_landmarks = len(lX)
    lX = np.array(lX, dtype=floath)
    lY = np.array(lY, dtype=floath)

    # Initialize Ys
    if Ys is None:
        Y = random_state.normal(0, init_stdev, size=(N, output_dims))
        Ys = [Y] * steps

    for t in range(steps):
        if Xs[t].shape[0] != N or Ys[t].shape[0] != N:
            raise Exception('Invalid datasets.')
        Xs[t] = np.array(Xs[t], dtype=floath)

    # Compute sigmas
    # sigmas_lX = find_sigma(lX, perplexity=perplexity)

    # lX_shared = theano.shared(lX)
    # lX_sigma_shared = theano.shared(np.ones(N_landmarks, dtype=floath))
    # find_sigma_old(lX_shared, lX_sigma_shared, N_landmarks, perplexity, sigma_iters=50, metric=metric, verbose=verbose)

    Xs_shared, Ys_shared, sigmas_shared, landmark_sigmas = [], [], [], []
    for t in range(steps):
        if verbose:
            print(t, '/', steps)

        # X_with_landmarks = np.append(Xs[t], lX, axis=0)
        # X_shared = theano.shared(X_with_landmarks)
        # sigma_shared = theano.shared(np.ones(N + N_landmarks, dtype=floath))

        X_shared = theano.shared(Xs[t])
        sigma_shared = theano.shared(np.ones(N, dtype=floath))
        find_sigma_old(X_shared, sigma_shared, N, perplexity, sigma_iters=50, metric=metric, verbose=verbose)
        np.save('old.npy', np.array(sigma_shared.container.data))

        # return 0

    #     Xs_shared.append(theano.shared(Xs[t]))
    #     Ys_shared.append(theano.shared(np.array(Ys[t], dtype=floath)))
    #     sigmas_shared.append(sigma_shared)
    #     landmark_sigmas.append(lX_sigma_shared)
    #
    # Ys = find_Ys_with_landmarks(Xs_shared, Ys_shared, sigmas_shared, N, steps, output_dims,
    #                             lX, lY, landmark_sigmas, N_landmarks,
    #                             n_epochs, initial_lr, final_lr, lr_switch, init_stdev,
    #                             initial_momentum, final_momentum, momentum_switch, lmbda,
    #                             metric, verbose)
    # return Ys

# Y = random_state.normal(loc=[np.average(lY[:,i]) for i in range(output_dims)],
#                         scale=[np.std(lY[:,i]) for i in range(output_dims)],
#                         size=(N, output_dims))

#
# import matplotlib.pyplot as plt;
# plt.plot(Y[:, 0], Y[:, 1], 'ro');
# plt.plot(lY[:, 0], lY[:, 1], 'bo');
# plt.show()
