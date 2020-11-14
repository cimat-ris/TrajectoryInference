"""
@author: jbhayet
"""
from test_common import *
from utils.loaders.loader_ind import load_ind
from utils.loaders.loader_gcs import load_gcs
from utils.loaders.loader_edinburgh import load_edinburgh
import matplotlib.pyplot as plt


# InD dataset
dataset_dir = "/home/jbhayet/opt/datasets/inD-dataset-v1.0/data/"
for i in range(32):
    dataset_file= "{:02d}_tracks.csv".format(i)
    traj_dataset= load_ind(dataset_dir+dataset_file)
    traj_set    = traj_dataset.get_trajectories()
    print("[INF] Loaded InD sequence {:02d}, length: {:03d} ".format(i,len(traj_set)))

# GCS (Grand Central) dataset
dataset_dir = "./datasets/GC/Annotation/"
traj_dataset= load_gcs(dataset_dir)
traj_set    = traj_dataset.get_trajectories()
print("[INF] Loaded GCS sequence, length: {:03d} ".format(len(traj_set)))

# Edinburgh dataset
dataset_dir = "./datasets/Edinburgh/annotations"
traj_dataset= load_edinburgh(dataset_dir)
traj_set    = traj_dataset.get_trajectories()
print("[INF] Loaded Edinburgh sequence, length: {:03d} ".format(len(traj_set)))
