#!/bin/bash

datasets=$(cat datasets/datasets.txt)
for d in $datasets; do
  echo $d;
  python generate-landmarks/k_random.py datasets/$d/ n TSNE
done;


