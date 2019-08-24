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

# Test function: prediction of single paths with single goals
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
        # TODO: too slow
        likelihood        = gp.update(trueX,trueY,trueL)
        # Perform prediction
        predictedXY,varXY = gp.predict()
        plot_prediction(img,pathX,pathY,knownN,predictedXY,varXY)
        print('[INF] Plotting')
        print("[RES] [Likelihood]: ",likelihood)
        # Generate samples
        vecX,vecY,__         = gp.generate_samples(100)
        plot_path_samples_with_observations(img,trueX,trueY,vecX,vecY)

# Test function: prediction of single paths with multiple goals
mixtureTest = False
if mixtureTest==True:
    mgps = mixtureOfGPs(startG,stepUnit,goalsData)
    part_num = 10
    # For different sub-parts of the trajectory
    for i in range(1,part_num-1):
        knownN = int((i+1)*(pathSize/part_num)) #numero de datos conocidos
        trueX,trueY,trueL = get_known_set(pathX,pathY,pathL,knownN)
        """Multigoal prediction test"""
        print('[INF] Updating likelihoods')
        likelihoods = mgps.update(trueX,trueY,trueL)
        print('[INF] Performing prediction')
        predictedXYVec,varXYVec = mgps.predict()
        print('[INF] Plotting')
        plot_multiple_predictions_and_goal_likelihood(img,pathX,pathY,knownN,goalsData.nGoals,likelihoods,predictedXYVec,varXYVec)
        print("[RES] Goals likelihood\n",mgps.goalsLikelihood)
        print("[RES] Mean likelihood:", mgps.meanLikelihood)
        print('[INF] Generating samples')
        vecX,vecY,__ = mgps.generate_samples(100)
        plot_path_samples_with_observations(img,trueX,trueY,vecX,vecY)

# Test function: prediction of single paths with multiple goals
animateMixtureTest = True
if animateMixtureTest==True:
    mgps = mixtureOfGPs(startG,stepUnit,goalsData)
    part_num = 10
    # For different sub-parts of the trajectory
    for i in range(1,part_num-1):
        knownN = int((i+1)*(pathSize/part_num)) #numero de datos conocidos
        trueX,trueY,trueL = get_known_set(pathX,pathY,pathL,knownN)
        """Multigoal prediction test"""
        print('[INF] Updating likelihoods')
        likelihoods = mgps.update(trueX,trueY,trueL)
        print('[INF] Performing prediction')
        predictedXYVec,varXYVec = mgps.predict()
        print('[INF] Plotting')
        animate_multiple_predictions_and_goal_likelihood(img,pathX,pathY,knownN,goalsData.nGoals,likelihoods,predictedXYVec,varXYVec)


# Test function: evaluation of interaction potentials on complete trajectories from the dataset
interactionTest = True
if interactionTest == True:
    # Get trajectories within some time interval
    sortedSet     = get_path_set_given_time_interval(sortedPaths,200,700)
    print("[INF] Number of trajectories in the selected time interval:",len(sortedSet))
    # Potential is evaluated based on the timestamps (array t)
    pot = interaction_potential_for_a_set_of_trajectories(sortedSet)
    print("[RES] Potential value: ",'{:1.4e}'.format(pot))
    plotPathSet(sortedSet,img)

# Test function: evaluation of interaction potentials on sampled trajectories
interactionWithSamplingTest = True
if interactionWithSamplingTest == True:
    # Get all the trajectories that exist in the dataset within some time interval
    sortedSet = get_path_set_given_time_interval(sortedPaths,350,750)
    plotPathSet(sortedSet,img)

    samplesJointTrajectories, potentialVec = [], []
    observedPaths, samplesVec, potentialVec = [], [], []# Guardan en i: [obsX, obsY] y [sampleX, sampleY]

    numTests = 4
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
        print("_____Test #",i+1,"_____")
        meanError = 0.
        for j in range(len(sortedSet)):
            currentPath = sortedSet[j]
            knownN = len(observedPaths[j].x)
            sampleX = allSampleTrajectories[j][0][i][:,0]#aqui hay un indice mal, i deberia ser j
            sampleY = allSampleTrajectories[j][1][i][:,0]
            currentSample = [sampleX, sampleY]
            steps = 8
            error = ADE_given_future_steps(currentPath, currentSample, knownN, steps)
            print("ADE",steps,"future steps | Path",j,"Sample",i,":",error)
            meanError += error

            # Form trajectory from path
            sampleX,sampleY,sampleL = allSampleTrajectories[j]
            predictedTrajectory     = deepcopy(observedPaths[j])
            predictedTrajectory.join_path_with_sample(sampleX[i],sampleY[i],sampleL[i],speed)#     = get_trajectory_from_path(predictedTrajectory,sampleX[i],sampleY[i],sampleL[i],speed)
            # Keep trajectory as an element of the joint sample
            jointTrajectories.append(predictedTrajectory)
            # When we have all the predictions to get one joint prediction
            if(len(jointTrajectories) == len(sortedSet)):
                samplesJointTrajectories.append(jointTrajectories)
                # Evaluation of the potential
                interactionPotential = interaction_potential_for_a_set_of_trajectories(jointTrajectories)
                potentialVec.append(interactionPotential)
        meanError = meanError/len(sortedSet)
        print("Mean error:",meanError)
    plot_interaction_with_sampling_test(img,observedPaths,samplesJointTrajectories,potentialVec)

    maxPotential = 0
    maxId = -1
    for i in range(len(potentialVec)):
        if(potentialVec[i] > maxPotential):
            maxPotential = potentialVec[i]
            maxId = i
    #print("Best configuration: figure ", maxId+2) #Figure 1 son las trayectorias reales

testingData = get_uncut_paths_from_file('datasets/CentralStation_paths_10000-12500.txt')
testingPaths = getUsefulPaths(testingData,areas)
#problematic paths:
testingPaths.pop(106)
testingPaths.pop(219)
testingPaths.pop(244)
testingPaths.pop(321)
testingPaths.pop(386)
#plotPathSet(testingPaths,img)
print("testingPaths size:",len(testingPaths))

errorTablesTest = False
if errorTablesTest == True:
    predictionTable, samplingTable = [], []
    rows, columns = [], []

    futureSteps = [8,10,12]
    partNum = 5
    nSamples = 50
    nPaths = len(testingPaths)
    for steps in futureSteps:
        rows.append( str(steps) )
        print("__Comparing",steps,"steps__")
        meanPredError = []
        meanSampleError = []
        for i in range(partNum-1):
            predictionFile = 'results/Prediction_error_'+'%d'%(steps)+'_steps_%d'%(i+1)+'_of_%d'%(partNum)+'_data.txt'
            samplingFile   = 'results/Sampling_error_'+'%d'%(steps)+'_steps_%d'%(i+1)+'_of_%d'%(partNum)+'_data.txt'

            predError, samplingError = [], []
            meanP, meanS = 0., 0.
            for j in range(nPaths):
                print("\nPath #",j)
                currentPath = testingPaths[j]
                startG = get_path_start_goal(currentPath,areas)
                mgps = mixtureOfGPs(startG,stepUnit,goalsData)

                print("Observed data:",i+1,"/",partNum)
                pathSize = len(currentPath.x)
                knownN = int((i+1)*(pathSize/partNum)) #numero de datos conocidos
                trueX,trueY,trueL = get_known_set(currentPath.x,currentPath.y,currentPath.l,knownN)
                """Multigoal prediction test"""
                likelihoods = mgps.update(trueX,trueY,trueL)
                predictedMeans,varXYVec = mgps.predict()
                predictedXYVec = get_prediction_arrays(predictedMeans)

                mostLikelyG = mgps.mostLikelyGoal
                if mgps.gpPathRegressor[mostLikelyG + nGoals] != None:
                    sgError = []
                    sgError.append(ADE_given_future_steps(currentPath, predictedXYVec[mostLikelyG], knownN, steps))
                    for it in range(mgps.nSubgoals):
                        k = mostLikelyG + (it+1)*nGoals
                        sgError.append(ADE_given_future_steps(currentPath, predictedXYVec[k], knownN, steps))
                    minId = 0
                    for ind in range(len(sgError)):
                        if sgError[minId] == 0 and sgError[ind] > 0:
                            minId = ind
                        elif sgError[minId] > sgError[ind] and sgError[ind] > 0:
                            minId = ind
                    error = sgError[minId]
                else:
                    error = ADE_given_future_steps(currentPath, predictedXYVec[mostLikelyG], knownN, steps)
                predError.append(error)
                meanP += error
                """Sampling"""
                # Generate samples
                vecX,vecY,vecL  = mgps.generate_samples(nSamples)
                samplesError = []
                for k in range(nSamples):
                    sampleXY = [vecX[k][:,0], vecY[k][:,0]]
                    error = ADE_given_future_steps(currentPath,sampleXY, knownN, steps)
                    samplesError.append(error)
                samplingError.append(min(samplesError))
                meanS += min(samplesError)

                write_data(predError,predictionFile)
                write_data(samplingError,samplingFile)
            meanP /= nPaths
            meanS /= nPaths
            meanPredError.append(meanP)
            meanSampleError.append(meanS)
        predictionTable.append(meanPredError)
        samplingTable.append(meanSampleError)
    print("Prediction error:\n",predictionTable)
    print("Sampling error:\n",samplingTable)

    for i in range(partNum-1):
        s = str(i+1) + '/' + str(partNum)
        columns.append(s)
    #Plot tables
    plot_table(predictionTable,rows,columns,'Prediction Error')
    plot_table(samplingTable,rows,columns,'Sampling Error')

boxPlots = True
if boxPlots == True:
    futureSteps = [8,10]
    partNum = 5
    for steps in futureSteps:
        for j in range(1,partNum-2):
            plotName = 'Predictive mean\n'+'%d'%(steps)+' steps | %d'%(j+1)+'/%d'%(partNum)+' data'
            predData = read_data('results/Prediction_error_'+'%d'%(steps)+'_steps_%d'%(j+1)+'_of_%d'%(partNum)+'_data.txt')
            boxplot(predData, plotName)

            plotName = 'Best of samples\n'+'%d'%(steps)+' steps | %d'%(j+1)+'/%d'%(partNum)+' data'
            samplingData = read_data('results/Sampling_error_'+'%d'%(steps)+'_steps_%d'%(j+1)+'_of_%d'%(partNum)+'_data.txt')
            boxplot(samplingData, plotName)


#Prueba el error de la prediccion variando:
# - el numero de muestras del punto final
# - numero de pasos a comparar dado un objetivo final
#number_of_samples_and_points_to_compare_to_destination(areas,pathMat,nGoals,nGoals,unitMat,meanLenMat,goalSamplingAxis,kernelMat_x,kernelMat_y)

#Compara el error de la prediccion hacia el centro del goal contra la prediccion hacia los subgoales
#compare_error_goal_to_subgoal_test(img,pathX,pathY,pathL,startG,nextG,areas,unitMat,stepUnit,kernelMat_x,kernelMat_y,goalSamplingAxis)

#path_sampling_test(img,areas,nGoals,goalSamplingAxis,unitMat,stepUnit,kernelMat_x,kernelMat_y,linearPriorMatX,linearPriorMatY)
