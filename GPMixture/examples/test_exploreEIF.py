"""
@author: jbhayet
"""
from test_common import *
from utils.loaders.loader_edinburgh import load_edinburgh
from utils.io_trajectories import read_and_filter,partition_train_test
import matplotlib.pyplot as plt


# Edinburgh dataset
dataset_dir = "./datasets/Edinburgh/Annotations"
coordinates      = 'img'
traj_dataset= load_edinburgh(dataset_dir)
traj_dataset= traj_dataset.get_trajectories()
trajectories= []
for tr in traj_dataset:
    x, y, t = np.array(tr[:,0]), np.array(tr[:,1]), np.array(tr[:,4])
    trajectories.append(np.array([x,y,t]))
print("[INF] Loaded Edinburgh set, length: {:03d} ".format(len(trajectories)))

img_bckgd        = './datasets/Edinburgh/edinburgh.jpg'
# Plot trajectories and structure
showDataset = True
p = plotter()
if coordinates=='img':
    p.set_background(img_bckgd)
print(trajectories[0].shape)
p.plot_paths(trajectories)
p.save("structure.pdf")
p.show()
