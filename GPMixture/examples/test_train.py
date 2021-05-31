"""
@author: karenlc
"""
from test_common import *
from utils.io_parameters import *
from utils.io_trajectories import read_and_filter,partition_train_test
# Read the areas file, dataset, and form the goalsLearnedStructure object
img_bckgd        = './imgs/train_station.jpg'
img_bckgd        = './datasets/Edinburgh/edinburgh.jpg'
dataset_name     = 'EIF'
coordinates      = 'img'

traj_dataset, goalsData, trajMat, __ = read_and_filter(dataset_name,coordinate_system=coordinates,use_pickled_data=True)

# Partition test/train set
train_trajectories_matrix, test_trajectories_matrix = partition_train_test(trajMat)

# Plot trajectories and structure
showDataset = False
if showDataset:
    for i in range(goalsData.goals_n):
        for j in range(goalsData.goals_n):
            if len(train_trajectories_matrix[i][j])>0:
                p = plotter(title=f"Training trajectories from {i} to {j}")
                p.set_background(img_bckgd)
                p.plot_scene_structure(goalsData)
                p.plot_paths(test_trajectories_matrix[i][j])
                p.show()

# Selection of the kernel type
kernelType  = "linePriorCombined"

"""**************    Learning GP parameters     **************************"""
print("[INF] Starting the learning phase")
goalsData.optimize_kernel_parameters(kernelType,train_trajectories_matrix)
write_parameters(goalsData.kernelsX,"parameters/linearpriorcombined20x20_"+dataset_name+"_"+coordinates+"_x.txt")
write_parameters(goalsData.kernelsY,"parameters/linearpriorcombined20x20_"+dataset_name+"_"+coordinates+"_y.txt")
print("[INF] End of the learning phase")
