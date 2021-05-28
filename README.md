# Landmark based dynamic projections

Projections attempt to convey the relationships and similarity of data points from a high dimensional dataset into a lower- dimensional representation. Most projections techniques are designed for static data. When used for time-dependent data, they usually fail to create a stable and suitable low dimensional representation. We propose two new dynamic projection methods (PCD-tSNE and LD-tSNE) based on the idea of using global guides to steer projection points. This avoids unstable movement that hinders the ability to reason about high dimensional dynamics while keeping t-SNE’s neighborhood preservation ability. PCD-tSNE scores a good balance between stability, neighborhood preservation, and distance preservation, while LD-tSNE allows us to create stable and customizable projections. We demonstrate our methods by comparing them to 11 other techniques using quality metrics and datasets provided by a recent benchmark for dynamic projections.

Below are the instructions for running each of them on your own data.

## Data
The data must follow the format as seen in the [datasets](https://github.com/EduardoVernier/landmark-based-dynamic-projections/tree/master/datasets) folder. That is, each timestep/revision of the data is an individual csv file, containing all observation in the data (rows) and dimensions (columns). Note that these must be consistent on all csv files.  


## PCD-tSNE
The implementation in python of the technique can be found [here](https://github.com/EduardoVernier/landmark-based-dynamic-projections/blob/master/methods/pcdtsne.py).

All hyperparemeters are set in the python file itself, and the program was written to receive lists of hyperparameters and generate projections for all combination (to ease testing).

To run it, it is first necessary to install the pip environment via:
```
pip install pipenv
pipenv run pip install pip==18.0
pipenv install
sudo apt-get install python3-tk
```
To run the technique with the parameters set in the main function, just run:

```
export PYTHONPATH=${PYTHONPATH}:${PWD}/methods/
python methods/pcdtsne.py
```


## LD-tSNE

The algorithm consists of two steps. First we need to generate the landmarks and then we run the LD-tSNE algorithm that uses these landmarks to steer points in the projection.

#### Generate landmarks
```
export PYTHONPATH=${PYTHONPATH}:${PWD}/landmark-dtsne/
python generate-landmarks/k_random.py datasets/fashion/ n PCA
python generate-landmarks/k_random.py datasets/qtables/ 1000 TSNE
```

#### Run algorithm
All parameters are set on the main function.
```
export PYTHONPATH=${PYTHONPATH}:${PWD}/methods/
python methods/pcdtsne.py
```
