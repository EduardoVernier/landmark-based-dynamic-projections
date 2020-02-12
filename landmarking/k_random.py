import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys
import os

import shared

## USAGE: python k_random.py <dataset_path> <k_samples> <method>
## ../datasets/qtables/ 1000 PCA

dataset_path = sys.argv[1]
k = sys.argv[2]
method = sys.argv[3]
k = int(k)

if 'quickdraw' in dataset_path or 'fashion' in dataset_path:
    X, info_df, T, CATEGORIES = shared.load_drawings(dataset_path)
    X = np.array(X).reshape(len(X), -1)  # Flatten images
else:
    X, info_df, T, CATEGORIES = shared.load_tabular(dataset_path)

assert k < len(X), 'k larger than N*T.'
indexes = np.random.choice(len(X), k)
X = X[indexes]

assert method in ['TSNE', 'PCA'], 'Invalid method.'
if method == 'TSNE':
    Y = TSNE(n_components=2).fit_transform(X)
elif method == 'PCA':
    Y = PCA(n_components=2).fit_transform(X)


# Create single df with both X and Y to save in disk and open on the next script
X_df = pd.DataFrame(X, columns=['x'+str(i) for i in range(X.shape[1])])
Y_df = pd.DataFrame(Y, columns=['y'+str(i) for i in range(Y.shape[1])])
df = pd.concat([X_df, Y_df], axis=1)

# plt.scatter(Y[:,0], Y[:,1],c=[int(x) for x in info_df.drawing_cat_id.iloc[indexes]],
#             s=[10*(x/T) for x in info_df.t.iloc[indexes]], cmap=plt.cm.get_cmap('Set1'));plt.show()
# plt.scatter(Y[:,0], Y[:,1],c=[ord(x[0]) - 64 - 32 for x in info_df.cat.iloc[indexes]], s=[10*(x/T) for x in info_df.t.iloc[indexes]], cmap=plt.cm.get_cmap('Set1'));plt.show()
# print(df)

dataset_id = os.path.basename(os.path.dirname(dataset_path))
out = 'output/{}_{}_{}_{}.csv'.format(dataset_id, 'krandom', str(k), method)
df.to_csv(out)