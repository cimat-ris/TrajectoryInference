"""
@author: karenlc
"""
from gp_code.goal_pairs import *
from utils.plotting import plotter
from utils.io_parameters import *
from utils.manip_trajectories import getUsefulPaths,time_compare,filter_path_matrix,define_trajectories_start_and_end_areas
from utils.io_trajectories import get_paths_from_file
from utils.stats_trajectories import get_data_from_paths
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

# Read the areas data from a file
data     = pd.read_csv('parameters/CentralStation_GoalsDescriptions.csv')
areas    = data.values[:,2:].astype(float)
areasAxis= data.values[:,1].astype(int)

# This array will contain the zones of interest
nGoals    = len(areas)
img       = mpimg.imread('imgs/goals.jpg')

# This function segments the trajectories in function of the goal areas
dataPaths, multigoal = get_paths_from_file('datasets/CentralStation_paths_10000.txt',areas)
usefulPaths          = getUsefulPaths(dataPaths,areas)
print("[INF] Number of useful paths: ",len(usefulPaths))

# Split the trajectories into pairs of goals
startToGoalPath, arclenMat = define_trajectories_start_and_end_areas(areas,areas,usefulPaths)
# Remove the trajectories that are either too short or too long
pathMat, learnSet = filter_path_matrix(startToGoalPath, nGoals, nGoals)
sortedPaths = sorted(learnSet, key=time_compare)

# Form the object goal_pairs
goalsData = goal_pairs(areas,areasAxis,pathMat)

# Plot trajectories and structure
showDataset = False
if showDataset:
    s = areas.shape
    for i in range(s[0]):
        for j in range(s[0]):
            p = plotter(img,title=f"Trajectories from {i} to {j}")
            p.plot_scene_structure(goalsData)
            p.plot_paths(pathMat[i][j])
            p.show()

print("[INF] Number of filtered paths: ",len(learnSet))

# For each pair of goals, determine the line priors
goalsData.compute_linear_priors(pathMat)

# Selection of the kernel type
kernelType = "linePriorCombined"#"combined"
nParameters = 4

"""**************    Learning parameters     **************************"""
print("[INF] Starting the learning phase")
goalsData.optimize_kernel_parameters(kernelType, pathMat)
write_parameters(goalsData.kernelsX,nGoals,nGoals,"parameters/linearpriorcombined6x6_x.txt")
write_parameters(goalsData.kernelsY,nGoals,nGoals,"parameters/linearpriorcombined6x6_y.txt")
print("[INF] End of the learning phase")
