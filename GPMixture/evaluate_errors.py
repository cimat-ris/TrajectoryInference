"""
@author: karenlc
"""
import gp_code
from gp_code.goal_pairs import goal_pairs
from gp_code.likelihood import ADE_FDE
from gp_code.mixture_gp import mixtureOfGPs
from gp_code.interactions import interaction_potential_for_a_set_of_trajectories
from utils.manip_trajectories import get_known_set, getUsefulPaths, get_path_start_goal, get_prediction_arrays
from utils.manip_trajectories import get_path_set_given_time_interval, time_compare
from utils.io_parameters import read_and_set_parameters
from utils.io_trajectories import read_and_filter, get_uncut_paths_from_file, write_data
from utils.plotting import plot_prediction, plot_pathset, plot_table
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
testing_data = get_uncut_paths_from_file('datasets/CentralStation_paths_10000-12500.txt')
testing_paths = getUsefulPaths(testing_data,goalsData.areas)
#problematic paths:
testing_paths.pop(106)
testing_paths.pop(219)
testing_paths.pop(244)
testing_paths.pop(321)
testing_paths.pop(386)
plot_pathset(img,testing_paths)
print("Testing dataset size:",len(testing_paths))

table_ade_sp, table_ade_mp = [], []
table_fde_sp, table_fde_mp = [], []

# For the table
rows, columns = [], []
# Evaluation at 8,10,12 steps
future_steps = [6,8,10,12,14]
# future_steps = [8]
part_num     = 6
#part_num     = 2
n_samples    = 50
n_paths      = len(testing_paths)

# Iterate over possible horizon windows
for steps in future_steps:
    rows.append( str(steps)+" steps " )
    print("[EVL] Comparing",steps,"steps")
    # Comparison: errors from the most probable trajectory vs. best-of-many
    ade_mp_avgs = []
    ade_sp_avgs = []
    fde_mp_avgs = []
    fde_sp_avgs = []
    # Iterate over parts of the trajectories
    for i in range(part_num-1):
        # Files are written for each horizon, for each portion of observed trajectory
        file_ade_mp = 'results/ade_mp_'+'%d'%(steps)+'_steps_%d'%(i+1)+'_of_%d'%(part_num)+'_data.txt'
        file_ade_sp = 'results/ade_sp_'+'%d'%(steps)+'_steps_%d'%(i+1)+'_of_%d'%(part_num)+'_data.txt'

        ade_mp,     ade_sp     = [], []
        fde_mp,     fde_sp     = [], []
        ade_mp_avg, ade_sp_avg = 0., 0.
        fde_mp_avg, fde_sp_avg = 0., 0.

        # Iterate over all the paths
        for j in range(n_paths):
            print("\n[EVL] Path #",j,"/",n_paths)
            # Get the path
            path = testing_paths[j]
            # Goal id
            g_id = get_path_start_goal(path,goalsData.areas)
            # Define a mixture of GPs for this starting goal
            mgps = mixtureOfGPs(g_id,stepUnit,goalsData)

            print("[EVL] Observed data:",i+1,"/",part_num)
            path_size = len(path.x)
            # part_num is the number of parts we consider within the trajectory for testing in part_num points
            observed = int((i+1)*(path_size/part_num))
            # Get the observed data as i+1 first parts
            observed_x,observed_y,observed_l = get_known_set(path.x,path.y,path.l,observed)
            """Prediction from the most likely trajectory"""
            # Predict based on the observations and the goals. Determine the likeliest goal
            likelihoods             = mgps.update(observed_x,observed_y,observed_l)
            predictedMeans,varXYVec = mgps.predict()
            predictedXYVec          = get_prediction_arrays(predictedMeans)
            most_likely_g           = mgps.mostLikelyGoal

            # Keep the error associate to the most likely goal
            ade,fde = ADE_FDE(path, predictedXYVec[most_likely_g], observed, steps)
            kmin    = most_likely_g
            # Iterate over the subgoals
            for it in range(mgps.nSubgoals):
                k = most_likely_g + (it+1)*goalsData.nGoals
                # Keep the error associate to the most likely sub-goal
                ade_sg,fde_sg = ADE_FDE(path, predictedXYVec[k], observed, steps)
                if ade_sg > 0:
                    if ade<=0 or ade>ade_sg:
                        ade          = ade_sg
                        kmin         = k

            #realX = path.x[observed : observed+steps]
            #realY = path.y[knownN : knownN+futureSteps]
            #predX = predictedXYVec[kmin][0][:steps]
            #predY = predictedXYVec[kmin][1][:futureSteps]
            #print(realX)
            #print(predX)
            #print(error)
            #plot_prediction(img,path.x,path.y,observed,predictedMeans[kmin],varXYVec[kmin])
            # Add the ADE and the FDE
            ade_mp.append(ade)
            ade_mp_avg += ade
            fde_mp.append(fde)
            fde_mp_avg += fde
            """Prediction from best of many samples"""
            # Generate samples over goals and trajectories
            sp_x,sp_y,sp_l  = mgps.generate_samples(n_samples)
            ade_sps       = []
            fde_sps       = []
            # Iterate over the generated samples
            for k in range(n_samples):
                if sp_x[k] is not None:
                    sp    = [sp_x[k][:,0], sp_y[k][:,0]]
                    # ADE for a specific sample
                    ade,fde = ADE_FDE(path, sp, observed, steps)
                    ade_sps.append(ade)
                    fde_sps.append(fde)

            # Take the minimum error value
            ade_sp.append(min(ade_sps))
            fde_sp.append(min(fde_sps))
            # Add to the mean
            ade_sp_avg += min(ade_sps)
            fde_sp_avg += min(ade_sps)

        # Write the results
        write_data(ade_mp,file_ade_mp)
        write_data(ade_sp,file_ade_sp)
        ade_mp_avg /= n_paths
        ade_sp_avg /= n_paths
        fde_mp_avg /= n_paths
        fde_sp_avg /= n_paths

        ade_mp_avgs.append(ade_mp_avg)
        ade_sp_avgs.append(ade_sp_avg)
        fde_mp_avgs.append(fde_mp_avg)
        fde_sp_avgs.append(fde_sp_avg)

    table_ade_mp.append(ade_mp_avgs)
    table_ade_sp.append(ade_sp_avgs)
    table_fde_mp.append(fde_mp_avgs)
    table_fde_sp.append(fde_sp_avgs)

    print("Error with most likely trajectory:\n",table_ade_mp)
    print("Error with best-of-many samples:\n",table_ade_sp)

for i in range(part_num-1):
    s = str(i+1) + '/' + str(part_num)
    columns.append(s)
# Plot tables
plot_table(table_ade_mp,rows,columns,'ADE with most likely trajectory')
plot_table(table_ade_sp,rows,columns,'ADE with best-of-many samples')
plot_table(table_fde_mp,rows,columns,'FDE with most likely trajectory')
plot_table(table_fde_sp,rows,columns,'FDE with best-of-many samples')

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
