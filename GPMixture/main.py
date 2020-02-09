"""
@author: karenlc
"""
import gp_code
from gp_code.goals_structure import goalsLearnedStructure
from gp_code.interactions import interaction_potential_for_a_set_of_trajectories
from utils.manip_trajectories import get_known_set, getUsefulPaths
from utils.manip_trajectories import get_path_set_given_time_interval, time_compare
from utils.io_parameters import read_and_set_parameters
from utils.io_trajectories import read_and_filter, get_uncut_paths_from_file
from utils.plotting import plot_prediction, plot_pathset
from utils.plotting import plot_path_samples_with_observations
from utils.plotting import plot_multiple_predictions_and_goal_likelihood
import pandas as pd
import numpy as np
import time
import matplotlib.image as mpimg

img         = mpimg.imread('imgs/goals.jpg')
station_img = mpimg.imread('imgs/train_station.jpg')
# Read the areas file, dataset, and form the goalsLearnedStructure object
goalsData, pathMat, __ = read_and_filter('parameters/CentralStation_areasDescriptions.csv','datasets/CentralStation_trainingSet.txt')
stepUnit  = 0.0438780780171   #get_number_of_steps_unit(pathMat, nGoals)

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

# Test function: evaluation of interaction potentials on sampled trajectories
interactionWithSamplingTest = False
if interactionWithSamplingTest == True:
    # Get all the trajectories that exist in the dataset within some time interval
    sortedSet = get_path_set_given_time_interval(sortedPaths,350,750)
    #plotPathSet(sortedSet,img)

    samplesJointTrajectories, potentialVec, errorVec = [], [], []
    observedPaths, samplesVec = [], []# Guardan en i: [obsX, obsY] y [sampleX, sampleY]

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
        startG      = get_path_start_goal(observedPaths[j],goalsData.areas)
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
            sampleX = allSampleTrajectories[j][0][i][:,0]
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
        errorVec.append(meanError)
    #plot_interaction_with_sampling_test(img,observedPaths,samplesJointTrajectories,potentialVec)
    plot_interaction_test_weight_and_error(img,observedPaths, samplesJointTrajectories, potentialVec, errorVec)
    maxPotential = 0
    maxId = -1
    for i in range(len(potentialVec)):
        plot_interaction_with_sampling(img, observedPaths, samplesJointTrajectories[i], potentialVec[i], errorVec[i])
        if(potentialVec[i] > maxPotential):
            maxPotential = potentialVec[i]
            maxId = i

boxPlots = False
if boxPlots == True:
    futureSteps = [8,10,12]
    partNum = 5

    for steps in futureSteps:
        predMeanBoxes, samplesBoxes = [], []
        for j in range(partNum-1):
            plotName = 'Predictive mean\n'+'%d'%(steps)+' steps | %d'%(j+1)+'/%d'%(partNum)+' data'
            predData = read_data('results/Prediction_error_'+'%d'%(steps)+'_steps_%d'%(j+1)+'_of_%d'%(partNum)+'_data.txt')
            predMeanBoxes.append(predData)

            plotName = 'Best of samples\n'+'%d'%(steps)+' steps | %d'%(j+1)+'/%d'%(partNum)+' data'
            samplingData = read_data('results/Sampling_error_'+'%d'%(steps)+'_steps_%d'%(j+1)+'_of_%d'%(partNum)+'_data.txt')
            samplesBoxes.append(samplingData)
        #plot multiple boxplots
        title = 'Error comparing %d'%(steps) + ' steps'
        joint_multiple_boxplots(predMeanBoxes, samplesBoxes, title)


#Plots a partial path, the predictive mean to the most likely goal and the best among 50 samples
plotsPredMeanSamples = False
if plotsPredMeanSamples:
    realPath = testingPaths[0]
    real_path_predicted_mean_and_sample(img,realPath,goalsData.areas,goalsData,stepUnit)


#Prueba el error de la prediccion variando:
# - el numero de muestras del punto final
# - numero de pasos a comparar dado un objetivo final
#number_of_samples_and_points_to_compare_to_destination(areas,pathMat,nGoals,nGoals,unitMat,meanLenMat,goalSamplingAxis,kernelMat_x,kernelMat_y)

#Compara el error de la prediccion hacia el centro del goal contra la prediccion hacia los subgoales
#compare_error_goal_to_subgoal_test(img,pathX,pathY,pathL,startG,nextG,areas,unitMat,stepUnit,kernelMat_x,kernelMat_y,goalSamplingAxis)

#path_sampling_test(img,areas,nGoals,goalSamplingAxis,unitMat,stepUnit,kernelMat_x,kernelMat_y,linearPriorMatX,linearPriorMatY)
