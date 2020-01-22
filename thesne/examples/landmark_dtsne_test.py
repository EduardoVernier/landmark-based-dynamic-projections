import glob
import os

import natsort
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from thesne.model.dynamic_tsne import dynamic_tsne, landmark_dtsne
from thesne.examples import plot

from helper import shared


def to_csv(Ys, labels, dataset_id, p, l):
    output_path = '../../output/{}-ldtsne_{}p_{}l.csv'.format(dataset_id, int(p), str(l).replace('.', '-'))
    try:
        df_out = pd.DataFrame(index=labels)

        for t in range(len(Ys)):
            df_out['t{}d0'.format(t)] = Ys[t].T[0]
            df_out['t{}d1'.format(t)] = Ys[t].T[1]

        if (len(df_out) - df_out.count()).sum() == 0:
            df_out.to_csv(output_path, index_label='id')
            print(p, 'p', l, 'l', 'OK')
        else:
            pd.DataFrame().to_csv(output_path, index_label='id')
            print(p, 'p', l, 'l', 'crashed')
    except Exception as e:
        pd.DataFrame().to_csv(output_path, index_label='id')
        print(e)
        print(p, 'p', l, 'l', 'crashed')


def main():
    seed = 0

    p = 70
    l = 0.1

    dataset_dir = '../../datasets/fashion/'
    dataset_id = os.path.basename(os.path.dirname(dataset_dir))

    # Read dataset
    Xs = []
    labels = []
    if 'quickdraw' in dataset_dir or 'fashion' in dataset_dir or 'faces' in dataset_dir:
        X, info_df, n_revisions, CATEGORIES = shared.load_drawings(dataset_dir + '/')
        N = len(X)
        X_flat = np.reshape(np.ravel(X), (N, -1))
        for t, df in info_df.groupby('t'):
            df = df.sort_values(['drawing_cat_id', 'drawing_id'])
            if len(labels) == 0:
                labels = df['drawing_cat_str'].str.cat(df['drawing_id'].astype(str), sep='-')
            Xs.append(X_flat[df.X_index])
    else:
        csvs = natsort.natsorted(glob.glob(dataset_dir + '/*'))
        for csv in csvs:
            df = pd.read_csv(csv, index_col=0)
            if len(labels) == 0:
                labels = df.index
            Xs.append(df.values)


    # Read landmarks
    df = pd.read_csv('../../landmarking/output/fashion_krandom_1000_PCA.csv', index_col=0)
    lX = df[[c for c in df.columns if c.startswith('x')]].values
    lY = df[[c for c in df.columns if c.startswith('y')]].values


    # Ys = dynamic_tsne(Xs, perplexity=p, lmbda=l, verbose=1, n_epochs=10, sigma_iters=10, random_state=seed)
    Ys = landmark_dtsne(Xs, lX, lY, perplexity=70, lmbda=0.1, verbose=1, sigma_iters=10, random_state=seed)

    # for Y in Ys:
    #     plot.plot(Y, labels)

    to_csv(Ys, labels, dataset_id, p, l)


if __name__ == "__main__":
    main()
