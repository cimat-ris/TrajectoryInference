"""
@author: karenlc
"""
from test_common import *
from utils.io_parameters import *

# Read the areas file, dataset, and form the goalsLearnedStructure object
traj_dataset, goalsData, trajMat, __ = read_and_filter('GCS','datasets/GC/',use_pickled_data=True)
imgGCS           = 'imgs/train_station.jpg'

# Plot trajectories and structure
showDataset = False
if showDataset:
    s = goalsData.areas_coordinates.shape
    for i in range(s[0]):
        for j in range(s[0]):
            if len(trajMat[i][j])>0:
                p = plotter(title=f"Trajectories from {i} to {j}")
                p.set_background(imgGCS)
                p.plot_scene_structure(goalsData)
                p.plot_trajectories(trajMat[i][j])
                p.show()

print("[INF] Number of filtered paths: ",len(traj_dataset))

# Selection of the kernel type
kernelType  = "linePriorCombined"

"""**************    Learning GP parameters     **************************"""
print("[INF] Starting the learning phase")
goalsData.optimize_kernel_parameters(kernelType,trajMat)
write_parameters(goalsData.kernelsX,"parameters/linearpriorcombined20x20_x.txt")
write_parameters(goalsData.kernelsY,"parameters/linearpriorcombined20x20_y.txt")
print("[INF] End of the learning phase")
