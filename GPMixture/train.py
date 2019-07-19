"""
@author: karenlc
"""
from gp_code.io_parameters import *
from gp_code.goalsLearnedStructure import *
from testing import *
from multipleAgents import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse
from copy import copy
import pandas as pd
# Read the areas data from a file and take only the first 6 goals
data     = pd.read_csv('parameters/CentralStation_areasDescriptions.csv')
areas    = data.values[:6,2:]
areasAxis= data.values[:6,1]

# This array will contain the zones of interest
nGoals    = len(areas)
img       = mpimg.imread('imgs/goals.jpg')

# Al leer cortamos las trayectorias multiobjetivos por pares consecutivos
# y las aimportgregamos como trayectorias independientes
dataPaths, multigoal = get_paths_from_file('datasets/CentralStation_paths_10000.txt',areas)
usefulPaths = getUsefulPaths(dataPaths,areas)

print("[INF] Number of useful paths: ",len(usefulPaths))

# Split the trajectories into pairs of goals
startToGoalPath, arclenMat = define_trajectories_start_and_end_areas(areas,areas,usefulPaths)

# Remove the trajectories that are either too short or too long
pathMat, learnSet = filter_path_matrix(startToGoalPath, nGoals, nGoals)
sortedPaths = sorted(learnSet, key=time_compare)
showDataset = False
if showDataset:
    plotPaths(pathMat, img)
print("[INF] Number of filtered paths: ",len(learnSet))

# Form the object goalsLearnedStructure
goalsData = goalsLearnedStructure(areas,areasAxis,pathMat)

stepUnit = 0.0438780780171   #get_number_of_steps_unit(pathMat, nGoals)
speed    = 1.65033755511     #get_pedestrian_average_speed(dataPaths)

# For each pair of goals, determine the line priors
useLinearPriors = True
if useLinearPriors:
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
