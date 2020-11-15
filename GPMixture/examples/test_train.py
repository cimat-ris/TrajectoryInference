"""
@author: karenlc
"""
from test_common import *
from utils.io_parameters import *
from utils.manip_trajectories import get_paths_in_areas,time_compare,filter_path_matrix,define_trajectories_start_and_end_areas
from utils.io_trajectories import get_paths_from_file

# Read the areas file, dataset, and form the goalsLearnedStructure object
goalsData, pathMat, learnSet = read_and_filter('parameters/CentralStation_GoalsDescriptions.csv','datasets/GCS/CentralStation_trainingSet.txt')

# Read the areas data from a csv file
#data     = pd.read_csv('parameters/CentralStation_GoalsDescriptions.csv')
#areas    = data.values[:,2:].astype(float)
#areasAxis= data.values[:,1].astype(int)

# The areas array will contain the zones of interest
#nGoals    = len(areas)

# This function segments the trajectories in function of the goal areas
#dataPaths,__ = get_paths_from_file('datasets/GCS/CentralStation_paths_10000.txt',areas)
# Filter paths
#usefulPaths  = get_paths_in_areas(dataPaths,areas)

# Split the trajectories into pairs of goals
#startToGoalPath, arclenMat = define_trajectories_start_and_end_areas(areas,areas,usefulPaths)
# Remove the trajectories that are either too short or too long (within a few stddev of the median)
#pathMat, learnSet          = filter_path_matrix(startToGoalPath, nGoals, nGoals)
#sortedPaths                = sorted(learnSet, key=time_compare)

# Form the object goal_pairs
#goalsData = goal_pairs(areas,areasAxis,pathMat)

# Plot trajectories and structure
showDataset = False
if showDataset:
    s = goalsData.areas_coordinates.shape
    for i in range(s[0]):
        for j in range(s[0]):
            if len(pathMat[i][j])>0:
                p = plotter("imgs/train_station.jpg",title=f"Trajectories from {i} to {j}")
                p.plot_scene_structure(goalsData)
                p.plot_paths(pathMat[i][j])
                p.show()

print("[INF] Number of filtered paths: ",len(learnSet))

# Selection of the kernel type
kernelType  = "linePriorCombined"

"""**************    Learning GP parameters     **************************"""
print("[INF] Starting the learning phase")
goalsData.optimize_kernel_parameters(kernelType, pathMat)
write_parameters(goalsData.kernelsX,"parameters/linearpriorcombined20x20_x.txt")
write_parameters(goalsData.kernelsY,"parameters/linearpriorcombined20x20_y.txt")
print("[INF] End of the learning phase")
