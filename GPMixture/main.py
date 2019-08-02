"""
@author: karenlc
"""
from gp_code.io_parameters import *
from gp_code.goalsLearnedStructure import *
from gp_code.mixtureOfGPs import *
from gp_code.singleGP import *
from utils.plotting import *
from utils.dataManagement import *
from testing import *
from multipleAgents import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse
from copy import copy
from copy import deepcopy
import pandas as pd

# Read the areas data from a file and take only the first 6 goals
data     = pd.read_csv('parameters/CentralStation_areasDescriptions.csv')
areas    = data.values[:6,2:]
areasAxis= data.values[:6,1]
nGoals   = len(areas)
img      = mpimg.imread('imgs/goals.jpg')

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

# Read the kernel parameters from file
goalsData.kernelsX = read_and_set_parameters("parameters/linearpriorcombined6x6_x.txt",nParameters)
goalsData.kernelsY = read_and_set_parameters("parameters/linearpriorcombined6x6_y.txt",nParameters)

"""******************************************************************************"""
"""**************    Testing                           **************************"""
# We give the start and ending goals
startG = 0
nextG = 2

# Kernels for this pair of goals
kernelX = goalsData.kernelsX[startG][nextG]
kernelY = goalsData.kernelsY[startG][nextG]

# Index of the trajectory to predict
pathId = 3
# Get the ground truth path
_path = pathMat[startG][nextG][pathId]
# Get the path data
pathX, pathY, pathL, pathT = _path.x, _path.y, _path.l, _path.t
# Total path length
pathSize = len(pathX)

# Test function: simple path sampling
samplingViz = False
if samplingViz==True:
    path_sampling_test(img,stepUnit,goalsData)

# Test function: prediction of single trajectories with single goals
singleTest = False
if singleTest==True:
    gp = singleGP(startG,nextG,stepUnit,goalsData)
    part_num = 10
    steps    = 10
    for i in range(1,part_num-1):
        # Data we will suppose known
        knownN = int((i+1)*(pathSize/part_num))
        trueX,trueY,trueL = get_known_set(pathX,pathY,pathL,knownN)
        """Single goal prediction test"""
        # Update the GP
        likelihood        = gp.update(trueX,trueY,trueL)
        # Perform prediction
        predictedXY,varXY = gp.predict()
        plot_prediction(img,pathX,pathY,knownN,predictedXY,varXY)
        print('[INF] Plotting')
        print("[RES] [Likelihood]: ",likelihood)
        # Generate samples
        vecX,vecY,__         = gp.generate_samples(100)
        plot_path_samples_with_observations(img,trueX,trueY,vecX,vecY)

# Test function: prediction of single trajectories with multiple goals
mixtureTest = True
if mixtureTest==True:
    mgps = mixtureOfGPs(startG,stepUnit,goalsData)
    part_num = 10
    steps = 10
    for i in range(1,part_num-1):
        knownN = int((i+1)*(pathSize/part_num)) #numero de datos conocidos
        trueX,trueY,trueL = get_known_set(pathX,pathY,pathL,knownN)
        """Multigoal prediction test"""
        likelihoods = mgps.update(trueX,trueY,trueL)
        predictedXYVec,varXYVec = mgps.predict()
        print('[INF] Plotting')
        plot_multiple_predictions_and_goal_likelihood(img,pathX,pathY,knownN,goalsData.nGoals,likelihoods,predictedXYVec,varXYVec)
        print("[RES] Goals likelihood\n",mgps.goalsLikelihood)
        print("[RES] [ean likelihood:", mgps.meanLikelihood)
        vecX,vecY,__ = mgps.generate_samples(100)
        plot_path_samples_with_observations(img,trueX,trueY,vecX,vecY)

# Test function: evaluation of interaction potentials on complete trajectories from the dataset
interactionTest = False
if interactionTest == True:
    sortedSet     = get_path_set_given_time_interval(sortedPaths,200,700)
    print("[INF] Number of trajectories in the selected time interval:",len(sortedSet))
    pot = interaction_potential_for_a_set_of_pedestrians(sortedSet)
    print("[RES] Potential value: ",'{:1.4e}'.format(pot))
    plotPathSet(sortedSet,img)

# Test function: evaluation of interaction potentials on sampled trajectories
interactionWithSamplingTest = True
if interactionWithSamplingTest == True:
    # Get all the trajectories that exist in the dataset within some time interval
    sortedSet = get_path_set_given_time_interval(sortedPaths,350,750)

    samplesJointTrajectories, potentialVec = [], []
    observedPaths, samplesVec, potentialVec = [], [], []# Guardan en i: [obsX, obsY] y [sampleX, sampleY]

    numTests = 6
    part_num = 3
    currentTime = 800
    allSampleTrajectories = []
    # Perform numTests sampling for all trajectories
    for j in range(len(sortedSet)):
        # Get the partial path up to now
        observedPaths.append(get_observed_path_given_current_time(sortedSet[j], currentTime))
        trueX,trueY,trueL = observedPaths[j].x.copy(), observedPaths[j].y.copy(), observedPaths[j].l.copy()
        # Determine the starting goal
        startG      = get_path_start_goal(observedPaths[j],areas)
        # Create the mixture, and update it with the observed data
        mgps        = mixtureOfGPs(startG,stepUnit,goalsData)
        likelihoods = mgps.update(trueX,trueY,trueL)
        mgps.predict()
        # Generate samples
        vecX,vecY,vecL  = mgps.generate_samples(numTests)
        allSampleTrajectories.append([vecX,vecY,vecL])

    # Use the previously sampled trajectories to form joint samples
    for i in range(numTests):
        jointTrajectories    = []
        # For all the paths in the set
        for j in range(len(sortedSet)):
            # Form trajectory from path
            sampleX,sampleY,sampleL = allSampleTrajectories[j]
            predictedTrajectory     = deepcopy(observedPaths[j])
            # TODO: make it a method instead of function
            predictedTrajectory     = get_trajectory_from_path(predictedTrajectory,sampleX[i],sampleY[i],sampleL[i],speed)
            # Keep trajectory as an element of the joint sample
            jointTrajectories.append(predictedTrajectory)
            # When we have all the predictions to get one joint prediction
            if(len(jointTrajectories) == len(sortedSet)):
                samplesJointTrajectories.append(jointTrajectories)
                # Evaluation of the potential
                interactionPotential = interaction_potential_for_a_set_of_pedestrians(jointTrajectories)
                potentialVec.append(interactionPotential)
    plot_interaction_with_sampling_test(img,observedPaths,samplesJointTrajectories,potentialVec)

    maxPotential = 0
    maxId = -1
    for i in range(len(potentialVec)):
        if(potentialVec[i] > maxPotential):
            maxPotential = potentialVec[i]
            maxId = i
    #print("Best configuration: figure ", maxId+2) #Figure 1 son las trayectorias reales


#Prueba el error de la prediccion variando:
# - el numero de muestras del punto final
# - numero de pasos a comparar dado un objetivo final
#number_of_samples_and_points_to_compare_to_destination(areas,pathMat,nGoals,nGoals,unitMat,meanLenMat,goalSamplingAxis,kernelMat_x,kernelMat_y)

#Compara el error de la prediccion hacia el centro del goal contra la prediccion hacia los subgoales
#compare_error_goal_to_subgoal_test(img,pathX,pathY,pathL,startG,nextG,areas,unitMat,stepUnit,kernelMat_x,kernelMat_y,goalSamplingAxis)

#path_sampling_test(img,areas,nGoals,goalSamplingAxis,unitMat,stepUnit,kernelMat_x,kernelMat_y,linearPriorMatX,linearPriorMatY)
