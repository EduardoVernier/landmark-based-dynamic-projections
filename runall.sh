#!/bin/bash

# export PYTHONPATH=${PYTHONPATH}:${PWD}/landmark-dtsne/

datasets=$(cat datasets/datasets.txt)
for d in $datasets; do
  echo $d;
#  python landmark-dtsne/ctsne.py $d
  python generate-landmarks/k_random.py ./datasets/$d/ nt PCA;
done;


