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

samplingViz = False
if samplingViz==True:
    path_sampling_test(img,stepUnit,goalsData)

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
        vecX,vecY         = gp.generate_samples(100)
        plot_path_samples_with_observations(img,trueX,trueY,vecX,vecY)

mixtureTest = False
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
        print("[RES] [Goals likelihood]\n",mgps.goalsLikelihood)
        print("[RES] [Mean likelihood]:", mgps.meanLikelihood)
        vecX,vecY = mgps.generate_samples(100)
        plot_path_samples_with_observations(img,trueX,trueY,vecX,vecY)

#quit()

interactionTest = False
if interactionTest == True:
    sortedSet     = get_path_set_given_time_interval(sortedPaths,200,700)
    print("Numero de trayectorias en el conjunto:",len(sortedSet))
    plotPathSet(sortedSet,img)

    interaction_potential_for_a_set_of_pedestrians(sortedSet)

interactionWithSamplingTest = True
if interactionWithSamplingTest == True:
    sortedSet = get_path_set_given_time_interval(sortedPaths,350,750)
    #plotPathSet(sortedSet,img)

    sampleSetVec, potentialVec = [], []
    observationsVec, samplesVec, potentialVec = [], [], []# Guardan en i: [obsX, obsY] y [sampleX, sampleY]
        
    numTests = 4
    part_num = 3
    currentTime = 800
    for i in range(numTests):
        samplePathSet = []
        observedX, observedY = [], []
        sampleXVec, sampleYVec = [], []
        for j in range(len(sortedSet)):
            currentPath = sortedSet[j]
            pathSize = len(currentPath.x)
            knownN = int(pathSize/part_num)#int((i+1)*(pathSize/part_num))
            #observedPath = get_partial_path(currentPath,knownN)
            #trueX,trueY,trueL = get_known_set(currentPath.x,currentPath.y,currentPath.l,knownN)
            observedPath = get_observed_path_given_current_time(currentPath, currentTime)   
            trueX,trueY,trueL = observedPath.x.copy(), observedPath.y.copy(), observedPath.l.copy() 
            observedX.append(trueX)
            observedY.append(trueY)

            startG = get_path_start_goal(observedPath,areas)
            #print("Path #",j)
            #print("[INF] Start goal", startG)
            mgps = mixtureOfGPs(startG,stepUnit,goalsData)
            likelihoods = mgps.update(trueX,trueY,trueL)

            nSamples = 1
            vecX,vecY = mgps.generate_samples(nSamples)
            for k in range(nSamples): #num de samples
                x, y = vecX[k], vecY[k]
                sampleX, sampleY = np.reshape(x,(x.shape[0])), np.reshape(y,(y.shape[0]))
                sampleXVec.append(sampleX)
                sampleYVec.append(sampleY)
                newL = arclength(sampleX,sampleY)
                samplePath = get_path_from_data(observedPath,sampleX,sampleY,newL,speed)
                samplePathSet.append(samplePath)
        if(len(samplePathSet) == len(sortedSet)):
            sampleSetVec.append(samplePathSet)
            interactionPotential = interaction_potential_for_a_set_of_pedestrians(samplePathSet)
            potentialVec.append(interactionPotential)
            observationsVec.append([observedX, observedY])
            samplesVec.append([sampleXVec,sampleYVec])

        #plot_path_set_samples_with_observations(img,observedX,observedY,sampleXVec,sampleYVec)#plotPathSet(samplePathSet,img)
    plot_interaction_with_sampling_test(img,observationsVec,samplesVec,potentialVec)
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
