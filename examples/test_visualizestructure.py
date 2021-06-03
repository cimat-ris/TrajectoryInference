"""
@author: jbhayet
"""
from test_common import *
from utils.loaders.loader_ind import load_ind
from utils.loaders.loader_gcs import load_gcs
from utils.loaders.loader_edinburgh import load_edinburgh
from utils.io_trajectories import read_and_filter
import matplotlib.pyplot as plt


# InD dataset
test_ind = False
if test_ind:
    dataset_dir = "/home/jbhayet/opt/datasets/inD-dataset-v1.0/data/"
    for i in range(32):
        dataset_file= "{:02d}_tracks.csv".format(i)
        traj_dataset= load_ind(dataset_dir+dataset_file)
        traj_set    = traj_dataset.get_trajectories()
        print("[INF] Loaded InD set {:02d}, length: {:03d} ".format(i,len(traj_set)))

# GCS (Grand Central) dataset
test_gcs = False
if test_gcs:
    dataset_dir = "./datasets/GC/Annotation/"
    traj_dataset= load_gcs(dataset_dir)
    traj_set    = traj_dataset.get_trajectories()
    print("[INF] Loaded gcs set, length: {:03d} ".format(len(traj_set)))

# Edinburgh dataset
test_edi = False
if test_edi:
    dataset_dir = "./datasets/Edinburgh/annotations"
    traj_dataset= load_edinburgh(dataset_dir)
    traj_set    = traj_dataset.get_trajectories()
    print("[INF] Loaded Edinburgh set, length: {:03d} ".format(len(traj_set)))

img_bckgd        = './imgs/train_station.jpg'
img_bckgd        = './datasets/Edinburgh/edinburgh.jpg'
coordinates      ='img'
traj_dataset, goalsData, trajMat, __ = read_and_filter('EIF',coordinate_system=coordinates,use_pickled_data=True)
print("[INF] Number of trajectories: ",len(traj_dataset))
# Plot trajectories and structure
showDataset = True
p = plotter()
if coordinates=='img':
    p.set_background(img_bckgd)
p.plot_scene_structure(goalsData,draw_ids=True)
p.plot_paths_samples_gt(trajMat,n_samples=1)

p.save("structure.pdf")
p.show()
