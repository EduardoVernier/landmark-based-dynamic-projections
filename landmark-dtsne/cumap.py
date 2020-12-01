from Models import Shared
import sys
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import umap
import natsort
import glob
import matplotlib.cm as cm

if __name__ == '__main__':
    dataset_id = sys.argv[1]
    # n_neigh = int(sys.argv[2])
    # print(dataset_dir, perplexity)

    # dataset_id = 'gaussians'
    dataset_dir = './Datasets/{}/'.format(dataset_id)

    Xs = []
    labels = []
    n_revisions = 0
    if 'quickdraw' in dataset_dir or 'fashion' in dataset_dir or 'faces' in dataset_dir:
        X, info_df, n_revisions, CATEGORIES = Shared.load_drawings(dataset_dir + '/')
        N = len(X)
        X_flat = np.reshape(np.ravel(X), (N, -1))
        for t, df in info_df.groupby('t'):
            df = df.sort_values(['drawing_cat_id', 'drawing_id'])
            if len(labels) == 0:
                labels = df['drawing_cat_str'].str.cat(df['drawing_id'].astype(str), sep='-')
            Xs.append(X_flat[df.X_index])
        categories = pd.factorize(df['drawing_cat_str'])[0]

    else:
        csvs = natsort.natsorted(glob.glob(dataset_dir + '/*'))
        n_revisions = len(csvs)
        for csv in csvs:
            df = pd.read_csv(csv, index_col=0)
            if len(labels) == 0:
                labels = df.index
                categories = df.index.str.split('-').str[0]
                categories = pd.factorize(categories)[0]
            Xs.append(df.values)


    (n, d) = Xs[0].shape
    Y = None
    Ys = []
    title = '{}-cumap'.format(dataset_id)
    timestamp = str(int(time.time()))
    print(title)
    for t in range(len(Xs)):
        print('Timestep: ' + str(t))

        if (t == 0):
            Y = umap.UMAP(n_epochs=200).fit_transform(Xs[t])
        else:
            Y = umap.UMAP(n_epochs=200, init=Ys[-1]).fit_transform(Xs[t])
        Ys.append(Y)
        # Y = ctsne(Xs[t], Y, perplexity=perplexity, max_iter=max_iter, timestep=t)

        # Show results
        # if t == len(Xs) - 1:
        # fig, ax = plt.subplots()
        # ax.set_title(title)
        # max_dist = max(Y[:,0]) - min(Y[:,0])  # bad
        # ax.scatter(Y[:, 0], Y[:, 1], 20, categories, cmap=cm.Set3, zorder=2)
        # ax.set_aspect('equal')
        # plt.show()

        # Save results
        df_out = pd.DataFrame(index=labels)
        for t in range(len(Ys)):
            df_out['t{}d0'.format(t)] = Ys[t].T[0]  # Only doing 2D for now
            df_out['t{}d1'.format(t)] = Ys[t].T[1]

        df_out.to_csv('./Output/{}.csv'.format(title), index_label='id')
