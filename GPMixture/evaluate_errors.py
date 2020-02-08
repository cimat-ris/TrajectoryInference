"""
@author: karenlc
"""
import gp_code
from gp_code.goals_structure import goalsLearnedStructure
from gp_code.likelihood import ADE_given_future_steps, nearestPD
from gp_code.mixture_gp import mixtureOfGPs
from gp_code.interactions import interaction_potential_for_a_set_of_trajectories
from utils.manip_trajectories import get_known_set, getUsefulPaths, get_path_start_goal, get_prediction_arrays
from utils.manip_trajectories import get_path_set_given_time_interval, time_compare
from utils.io_parameters import read_and_set_parameters
from utils.io_trajectories import read_and_filter, get_uncut_paths_from_file, write_data
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

## Evaluation of the prediction
testingData = get_uncut_paths_from_file('datasets/CentralStation_paths_10000-12500.txt')
testingPaths = getUsefulPaths(testingData,goalsData.areas)
#problematic paths:
testingPaths.pop(106)
testingPaths.pop(219)
testingPaths.pop(244)
testingPaths.pop(321)
testingPaths.pop(386)
plot_pathset(img,testingPaths)
print("Testing dataset size:",len(testingPaths))

errorTablesTest = True
if errorTablesTest == True:
    predictionTable, samplingTable = [], []
    rows, columns = [], []
    # Evaluation at 8,10,12 steps
    # futureSteps = [8,10,12,14]
    futureSteps = [8]
    partNum = 5
    nSamples = 50
    nPaths = len(testingPaths)

    # Iterate over possible horizon windows
    for steps in futureSteps:
        rows.append( str(steps) )
        print("[EVL] Comparing",steps,"steps")
        # Comparison: errors from the most probable trajectory vs. best-of-many
        meanPredError = []
        meanSampleError = []
        # Iterate
        for i in range(partNum-1):
            # Files are written for each horizon, for each portion of observed trajectory
            predictionFile = 'results/Prediction_error_'+'%d'%(steps)+'_steps_%d'%(i+1)+'_of_%d'%(partNum)+'_data.txt'
            samplingFile   = 'results/Sampling_error_'+'%d'%(steps)+'_steps_%d'%(i+1)+'_of_%d'%(partNum)+'_data.txt'

            predError, samplingError = [], []
            meanP, meanS = 0., 0.
            # Iterate over all the paths
            for j in range(nPaths):
                print("\n[EVL] Path #",j)
                # Get the path
                currentPath = testingPaths[j]
                # Goal id
                startG = get_path_start_goal(currentPath,goalsData.areas)
                # Define a mixture of GPs for this starting goal
                mgps   = mixtureOfGPs(startG,stepUnit,goalsData)

                print("[EVL] Observed data:",i+1,"/",partNum)
                pathSize = len(currentPath.x)
                # Partnum is the number of parts we consider within the trajectory for testing in partnum points
                knownN = int((i+1)*(pathSize/partNum))
                # Get the observed data as i+1 first parts
                trueX,trueY,trueL = get_known_set(currentPath.x,currentPath.y,currentPath.l,knownN)
                """Prediction from the most likely trajectory"""
                # Predict based on the observations and the goals. Determine the likeliest goal
                likelihoods             = mgps.update(trueX,trueY,trueL)
                predictedMeans,varXYVec = mgps.predict()
                predictedXYVec          = get_prediction_arrays(predictedMeans)
                mostLikelyG             = mgps.mostLikelyGoal
                # When there are subgoals available
                if mgps.gpPathRegressor[mostLikelyG + goalsData.nGoals] != None:
                    sgError = []
                    # Keep the error associate to the most likely goal
                    sgError.append(ADE_given_future_steps(currentPath, predictedXYVec[mostLikelyG], knownN, steps))
                    # Iterate over the subgoals
                    for it in range(mgps.nSubgoals):
                        k = mostLikelyG + (it+1)*goalsData.nGoals
                        # Keep the error associate to the most likely sub-goal
                        sgError.append(ADE_given_future_steps(currentPath, predictedXYVec[k], knownN, steps))
                    minId = 0
                    for ind in range(len(sgError)):
                        if sgError[minId] == 0 and sgError[ind] > 0:
                            minId = ind
                        elif sgError[minId] > sgError[ind] and sgError[ind] > 0:
                            minId = ind
                    error = sgError[minId]
                else:
                    # Compute the ADE
                    error = ADE_given_future_steps(currentPath, predictedXYVec[mostLikelyG], knownN, steps)
                # Add the ADE
                predError.append(error)
                meanP += error

                """Prediction from best of many samples"""
                # Generate samples over goals and trajectories
                vecX,vecY,vecL  = mgps.generate_samples(nSamples)
                samplesError = []
                # Iterate over the generated samples
                for k in range(nSamples):
                    sampleXY = [vecX[k][:,0], vecY[k][:,0]]
                    # ADE for a specific sample
                    error = ADE_given_future_steps(currentPath,sampleXY, knownN, steps)
                    samplesError.append(error)
                # Take the minimum error value
                samplingError.append(min(samplesError))
                # Add to the mean
                meanS += min(samplesError)

                # TODO: indentation error?
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
