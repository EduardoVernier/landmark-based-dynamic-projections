import numpy as np
import pandas as pd
import cv2
import re
import glob
import natsort
from sklearn.utils import shuffle

# Load drawings from the quickdraw/fashion dataset
def load_drawings(base_path):
    categories = list(set([img.split('/')[-1].split('-')[0] for img in glob.glob(base_path + '*')]))
    X = []
    CATEGORIES = {}
    drawing_cat_str = []
    drawing_cat_id = []
    drawing_id = []
    drawing_t = []
    X_index = []

    # print(sorted(categories))
    for cat_i, cat in enumerate(sorted(categories)):
        CATEGORIES[cat_i] = cat  # Maps a int to a string
        paths = glob.glob(base_path + cat + '-*')
        for p in paths:
            # Generate array from img
            im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            im = im / 255.
            X.extend(np.array([im]))

            # Drawing data
            drawing_cat_id.append(cat_i)
            drawing_cat_str.append(CATEGORIES[cat_i])
            match = re.match(r'.*/{}-(\d*)-(\d*).png'.format(CATEGORIES[cat_i]), p)
            drawing_id.append(int(match.group(1)))
            drawing_t.append(int(match.group(2)))
            X_index.append(len(X)-1)


    info_df = pd.DataFrame({'drawing_cat_id': drawing_cat_id,
                            'drawing_cat_str': drawing_cat_str,
                            'drawing_id': drawing_id,
                            't': drawing_t,
                            'X_index': X_index})



    # Replicate last image if drawing sequence doesn't have the maximum number of timesteps
    MAX_T = info_df['t'].max()
    g = info_df.groupby(by=['drawing_cat_id', 'drawing_id'])
    for key, df in g:
        max_row = (df[df['t'] == df['t'].max()])

        drawing_cat_str = []
        drawing_cat_id = []
        drawing_id = []
        drawing_t = []
        X_index = []

        for i in range(df['t'].max() + 1, MAX_T + 1):
            drawing_cat_id.append(int(max_row['drawing_cat_id'].values[0]))
            drawing_cat_str.append(max_row['drawing_cat_str'].values[0])
            drawing_id.append(int(max_row['drawing_id'].values[0]))
            X.extend([X[int(max_row['X_index'].values[0])]])
            X_index.append(len(X)-1)
            drawing_t.append(i)

        appendix = pd.DataFrame({'drawing_cat_id': drawing_cat_id,
                                 'drawing_cat_str': drawing_cat_str,
                                 'drawing_id': drawing_id,
                                 't': drawing_t,
                                 'X_index': X_index})
        info_df = info_df.append(appendix)

    X, info_df = shuffle(X, info_df, random_state=0)
    info_df['X_index'] = info_df.index
    info_df.index = range(len(info_df))
    info_df['X_index'] = range(len(info_df))
    info_df = info_df.astype({'drawing_cat_id':'int', 'drawing_id':'int', 't':'int'})
    return X, info_df, MAX_T+1, CATEGORIES


def read_dataset(dataset_dir):
    # Read dataset
    Xs = []
    labels = []
    if 'quickdraw' in dataset_dir or 'fashion' in dataset_dir or 'faces' in dataset_dir:
        X, info_df, n_revisions, CATEGORIES = load_drawings(dataset_dir + '/')
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
        for csv in csvs:
            df = pd.read_csv(csv, index_col=0)
            if len(labels) == 0:
                labels = df.index
                categories = df.index.str.split('-').str[0]
                categories = pd.factorize(categories)[0]

            Xs.append(df.values)

    return Xs, labels, categories