# Landmark based dynamic projections

## Examples

### Generate landmarks
```
export PYTHONPATH=${PYTHONPATH}:${PWD}/landmark-dtsne/
python generate-landmarks/k_random.py datasets/fashion/ n PCA
python generate-landmarks/k_random.py datasets/qtables/ 1000 TSNE
```



