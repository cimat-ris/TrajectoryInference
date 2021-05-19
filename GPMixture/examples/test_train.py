"""
@author: karenlc
"""
from test_common import *
from utils.io_parameters import *
# Read the areas file, dataset, and form the goalsLearnedStructure object
coordinates      = "img"
trajFile         = './datasets/GC/'
traj_dataset, goalsData, trajMat, __ = read_and_filter('GCS',trajFile,coordinate_system=coordinates,use_pickled_data=True)
imgGCS           = 'imgs/train_station.jpg'

# Partition test/train set
train_trajectories_matrix, test_trajectories_matrix = partition_train_test(trajMat)

# Plot trajectories and structure
showDataset = False
if showDataset:
    for i in range(goalsData.goals_n):
        for j in range(goalsData.goals_n):
            if len(train_trajectories_matrix[i][j])>0:
                p = plotter(title=f"Training trajectories from {i} to {j}")
                p.set_background(imgGCS)
                p.plot_scene_structure(goalsData)
                p.plot_paths(test_trajectories_matrix[i][j])
                p.show()

# Selection of the kernel type
kernelType  = "linePriorCombined"

"""**************    Learning GP parameters     **************************"""
print("[INF] Starting the learning phase")
goalsData.optimize_kernel_parameters(kernelType,train_trajectories_matrix)
write_parameters(goalsData.kernelsX,"parameters/linearpriorcombined20x20_"+coordinates+"_x.txt")
write_parameters(goalsData.kernelsY,"parameters/linearpriorcombined20x20_"+coordinates+"_y.txt")
print("[INF] End of the learning phase")
