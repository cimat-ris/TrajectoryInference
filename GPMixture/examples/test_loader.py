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

goalsDescriptions= './parameters/CentralStation_GoalsDescriptions.csv'
#goalsDescriptions= './parameters/Edinburgh_GoalsDescriptions.csv'
trajFile         = './datasets/GC/'
#trajFile         = "./datasets/Edinburgh/annotations"
img_bckgd        = './imgs/train_station.jpg'
#img_bckgd        = './datasets/Edinburgh/edinburgh.jpg'
coordinates      ='img'
traj_dataset, goalsData, trajMat, __ = read_and_filter('GCS',trajFile,coordinate_system=coordinates,use_pickled_data=False)

# Plot trajectories and structure
showDataset = True
p = plotter()
p.set_background(img_bckgd)
p.plot_scene_structure(goalsData)
p.plot_trajectories(traj_dataset[:200])
p.show()

if showDataset:
    n = goalsData.goals_n
    for i in range(n):
        for j in range(n):
            print(len(trajMat[i][j]))
            if len(trajMat[i][j])>0:
                #p.set_background(img_bckgd)
                p.plot_scene_structure(goalsData)
                p.plot_trajectories(trajMat[i][j])
                p.show()
