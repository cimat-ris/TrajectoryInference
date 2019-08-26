# -*- coding: utf-8 -*-
"""
Testing functions

@author: karenlc
"""

from gp_code.gpRegressor import *
from gp_code.mixtureOfGPs import *
from gp_code.regression import *
from gp_code.evaluation import *
from gp_code.kernels import *
from gp_code.sampling import*
from multipleAgents import*
import matplotlib.pyplot as plt
import numpy as np
import math
from copy import copy

def trajectory_prediction_test(img,x,y,l,knownN,startG,finishG,goals,unitMat,stepUnit,kernelMatX,kernelMatY,linearPriorMatX,linearPriorMatY):
    kernelX = kernelMatX[startG][finishG]
    kernelY = kernelMatY[startG][finishG]

    trueX, trueY, trueL = get_known_set(x,y,l,knownN)
    lastKnownPoint = [x[knownN-1], y[knownN-1], l[knownN-1] ]
    unit = unitMat[startG][finishG]

    #print("last known point:",lastKnownPoint)
    finalPoints = get_goal_center_and_boundaries(goals[finishG])

    #newL, finalL = get_prediction_set_given_size(lastKnownPoint,finalPoints[0],unit,20)
    newL, finalL, __ = get_prediction_set(lastKnownPoint,finalPoints[0],unit,stepUnit)
    finalArcLen = []
    for i in range(1):#len(finalPoints)):
        finalArcLen.append(get_arclen_to_finish_point(lastKnownPoint,finalPoints[i],unit))
        trueX.append(finalPoints[i][0])
        trueY.append(finalPoints[i][1])
        trueL.append(finalArcLen[i])

    #regular prediction
    newX,newY,varX,varY = prediction_XY(trueX,trueY,trueL,newL,kernelX,kernelY)
    #line prior prediction
    priorMeanX = linearPriorMatX[startG][finishG]
    priorMeanY = linearPriorMatY[startG][finishG]
    #newX,newY,varX,varY = prediction_XY_lp(trueX,trueY,trueL,newL,kernelX,kernelY,priorMeanX,priorMeanY)


    plot_prediction(img,x,y,knownN,newX,newY,varX,varY)
    return newX, newY, varX, varY, newL


def subgoal_prediction(x,y,l,knownN,subgoal,unit,stepUnit,kernelX,kernelY):
    trueX, trueY, trueL = get_known_set(x,y,l,knownN)
    lastKnownPoint = [x[knownN-1], y[knownN-1], l[knownN-1] ]

    finalPoints = get_goal_center_and_boundaries(subgoal)

    newL, finalL, __ = get_prediction_set(lastKnownPoint,finalPoints[0],unit,stepUnit)
    finalArcLen = []

    for i in range(1):#len(finalPoints)):
        finalArcLen.append(get_arclen_to_finish_point(lastKnownPoint,finalPoints[i],unit))
        trueX.append(finalPoints[i][0])
        trueY.append(finalPoints[i][1])
        trueL.append(finalArcLen[i])

    newX,newY,varX,varY = prediction_XY(trueX,trueY,trueL,newL,kernelX,kernelY)
    return newX, newY, varX, varY


def trajectory_subgoal_prediction_test(img,x,y,l,knownN,startG,finishG,goals,unitMat,stepUnit,kernelMatX,kernelMatY,samplingAxis):
    kernelX = kernelMatX[startG][finishG]
    kernelY = kernelMatY[startG][finishG]

    trueX, trueY, trueL = get_known_set(x,y,l,knownN)
    lastKnownPoint = [x[knownN-1], y[knownN-1], l[knownN-1] ]
    unit = unitMat[startG][finishG]

    nSubgoals = 2
    subgoalsXY, size = get_subgoals_center_and_size(nSubgoals,goals[finishG],samplingAxis[finishG])
    predictedXYVec, varXYVec = [], []
    for i in range(nSubgoals):
        _x, _y = subgoalsXY[i][0], subgoalsXY[i][1]
        lx, ly = size[0]/2, size[1]/2
        subgoal = [_x -lx, _y -ly, _x +lx, _y -ly,_x -lx, _y +ly, _x +lx, _y +ly]
        predX, predY, varX, varY = subgoal_prediction(x,y,l,knownN,subgoal,unit,stepUnit,kernelX,kernelY)
        predictedXYVec.append([predX,predY])
        varXYVec.append([varX,varY])

    plot_subgoal_prediction(img,x,y,knownN,nSubgoals,predictedXYVec,varXYVec)

def subgoal_prediction_test(nSubgoals,x,y,l,knownN,startG,finishG,goals,unitMat,stepUnit,kernelMatX,kernelMatY,samplingAxis):
    kernelX = kernelMatX[startG][finishG]
    kernelY = kernelMatY[startG][finishG]

    trueX, trueY, trueL = get_known_set(x,y,l,knownN)
    lastKnownPoint = [x[knownN-1], y[knownN-1], l[knownN-1] ]
    unit = unitMat[startG][finishG]

    subgoalsXY, size = get_subgoals_center_and_size(nSubgoals,goals[finishG],samplingAxis[finishG])
    predictedXYVec, varXYVec = [], []
    for i in range(nSubgoals):
        _x, _y = subgoalsXY[i][0], subgoalsXY[i][1]
        lx, ly = size[0]/2, size[1]/2
        subgoal = [_x -lx, _y -ly, _x +lx, _y -ly,_x -lx, _y +ly, _x +lx, _y +ly]
        predX, predY, varX, varY = subgoal_prediction(x,y,l,knownN,subgoal,unit,stepUnit,kernelX,kernelY)
        predictedXYVec.append([predX,predY])
        varXYVec.append([varX,varY])

    return predictedXYVec, varXYVec

def compare_error_goal_to_subgoal_test(img,x,y,l,startG,finishG,goals,unitMat,stepUnit,kernelMatX,kernelMatY,samplingAxis):
    nSubgoals = 2
    goalError = []
    subgoalError = []
    index = []
    part_num = 5
    traj_len = len(x)
    for i in range(1,part_num-1):
        index.append(i)
        knownN = int((i+1)*(traj_len/part_num))

        trueX, trueY = [], []
        for j in range(knownN,traj_len):
            trueX.append(x[j])
            trueY.append(y[j])

        goalPredX, goalPredY, goalVarX, goalVarY = trajectory_prediction_test(img,x,y,l,knownN,startG,finishG,goals,unitMat,stepUnit,kernelMatX,kernelMatY)
        subPredXYVec, subVarXYVec = subgoal_prediction_test(nSubgoals,x,y,l,knownN,startG,finishG,goals,unitMat,stepUnit,kernelMatX,kernelMatY,samplingAxis)
        plot_prediction(img,x,y,knownN,goalPredX, goalPredY, goalVarX, goalVarY,[0,0])
        plot_subgoal_prediction(img,x,y,knownN,nSubgoals,subPredXYVec,subVarXYVec,[0,0])

        gError = average_displacement_error([trueX,trueY],[goalPredX,goalPredY])
        sgErrorVec = []
        for k in range(nSubgoals):
            sgErrorVec.append(average_displacement_error([trueX,trueY], subPredXYVec[k]))
        goalError.append(gError)
        subgoalError.append( min(sgErrorVec) )

    print("[goal Error]",goalError)
    print("[subgoal Error]",subgoalError)
    plt.plot(index,goalError,'m')
    plt.plot(index,subgoalError,'b')

#recibe: datos conocidos, valores por predecir, areas de inicio y final
def prediction_test(img,x,y,z,knownN,startG,finishG,goals,unitMat,meanLenMat,steps,kernelMat_x,kernelMat_y):
    kernelX = kernelMat_x[startG][finishG]
    kernelY = kernelMat_y[startG][finishG]

    trueX, trueY, trueZ = get_known_set(x,y,z,knownN)
    lastKnownPoint = [x[knownN-1], y[knownN-1], z[knownN-1] ]
    unit = unitMat[startG][finishG]
    meanLen = meanLenMat[startG][finishG]

    #final_xy = get_finish_point(trueX,trueY,trueZ,finishG,goals,kernelX,kernelY,unit,goalSamplingAxis)
    #final_xy = get_finish_point_singleGP(trueX,trueY,trueZ,finishG,goals,kernelX,kernelY,unit,img)
    final_xy = middle_of_area(goals[finishG])
    newZ, final_z = get_prediction_set_given_size(lastKnownPoint,final_xy,unit,steps)
    trueX.append(final_xy[0])
    trueY.append(final_xy[1])
    trueZ.append(final_z)

    newX,newY,varX,varY = prediction_XY(trueX,trueY,trueZ,newZ,kernelX,kernelY)
    plot_prediction(img,x,y,knownN,newX,newY,varX,varY,[0,0])

# Sampling 3 trajectories between all the pairs of goals
def path_sampling_test(img,stepUnit,goalsData):
    vecX, vecY = [], []
    for i in range(goalsData.nGoals):
        for j in range(i,goalsData.nGoals):
            if(i != j):
                iCenter = middle_of_area(goalsData.areas[i])
                jCenter = middle_of_area(goalsData.areas[j])
                # The basic element here is this object, that will do the regression work
                gp = gpRegressor(goalsData.kernelsX[i][j], goalsData.kernelsY[i][j],goalsData.units[i][j],stepUnit,goalsData.areas[j],goalsData.areasAxis[j],goalsData.linearPriorsX[i][j], goalsData.linearPriorsY[i][j])
                gp.updateObservations([iCenter[0]],[iCenter[1]],[0.0])
                gp.prediction_to_finish_point()
                for k in range(3): #num of samples
                    #x, y,__,__,__  = sample_path_between_goals(i,j,stepUnit,goalsData)
                    # Sample end point around the sampled goal
                    finishX, finishY, axis = uniform_sampling_1D(1, goalsData.areas[j], goalsData.areasAxis[i])
                    # Use a pertubation approach to get the sample
                    deltaX = finishX[0]-jCenter[0]
                    deltaY = finishY[0]-jCenter[1]
                    x,y = gp.sample_with_perturbed_finish_point()
                    vecX.append(x)
                    vecY.append(y)
    plot_path_samples(img, vecX,vecY)

#Devuelve el punto con menor error de n muestras comparando k pasos
def sample_finish_point(nSamples,kSteps,knownX, knownY, knownL, finishGoal, goals, kernelX, kernelY, unit, samplingAxis):
    n = len(knownX)
    _x, _y, flag = uniform_sampling_1D(nSamples, goals[finishGoal], samplingAxis[finishGoal])
    if(n < 2*kSteps):
        return middle_of_area(goals[finishGoal])

    _knownX = knownX[0:n-kSteps]
    _knownY = knownY[0:n-kSteps]
    _knownL = knownL[0:n-kSteps]

    predSet = knownL[n-kSteps:kSteps]
    trueX = knownX[n-kSteps:kSteps]
    trueY = knownY[n-kSteps:kSteps]

    error = []
    for i in range(nSamples):
        auxX = _knownX.copy()
        auxY = _knownY.copy()
        auxL = _knownL.copy()
        auxX.append(_x[i])
        auxY.append(_y[i])
        dist = math.sqrt( (knownX[n-1] - _x[i])**2 + (knownY[n-1] - _y[i])**2 )
        lastL = knownL[n-1] + dist*unit
        auxL.append(lastL)
        predX, predY, vx, vy = prediction_XY(auxX, auxY, auxL, predSet, kernelX, kernelY)
        #error.append(geometricError(trueX,trueY,predX,predY))
        error.append(average_displacement_error([trueX,trueY],[predX,predY]))
    #encuentra el punto que genera el error minimo
    min_id, min_error = 0, error[0]
    for i in range(nSamples):
        if(error[i] < min_error):
            min_error = error[i]
            min_id = i
    return [_x[min_id], _y[min_id]]

def get_error_of_final_point_comparing_k_steps(samples,steps,trajectorySet,startG,finishG,goals,unitMat,meanLenMat,samplingAxis,kernelSetX,kernelSetY):
    meanError = 0.0
    for i in range( len(trajectorySet) ):
        x = trajectorySet[i].x
        y = trajectorySet[i].y
        l = trajectorySet[i].l
        final = [x[len(x)-1], y[len(y)-1]]
        knownN = int(len(x)* (0.9))
        trueX, trueY, trueZ = get_known_set(x,y,l,knownN)
        #lastKnownPoint = [x[knownN-1], y[knownN-1], l[knownN-1] ]
        unit = unitMat[startG][finishG]

        predicted_final = sample_finish_point(samples,steps,trueX,trueY,trueZ,finishG,goals,kernelSetX[i],kernelSetY[i],unit,samplingAxis)
        error = final_displacement_error(final,predicted_final)
        #print("FDE: ",error)
        meanError += error
    meanError /= len(trajectorySet)
    return meanError

def choose_number_of_steps_to_compare(trajectorySet,startG,finishG,goals,unitMat,meanLenMat,samplingAxis,kernelSpredictions_aetX,kernelSetY):
    errorVec = []
    stepsVec = []
    samples = 9
    steps = 1
    for i in range(20):
        stepsVec.append(steps)
        error = get_error_of_final_point_comparing_k_steps(samples,steps,trajectorySet,startG,finishG,goals,unitMat,meanLenMat,samplingAxis,kernelSetX,kernelSetY)
        errorVec.append(error)
        steps += 1

    #print("error: ", errorVec)
    plt.plot(stepsVec, errorVec, 'g')
    plt.xlabel('Number of steps to compare')
    plt.ylabel('Mean FDE')
    plt.show()
    #para ambos el error sera FDE
    #en 2 te detienes cuando el error deje de disminuir significativamente

def choose_number_of_destination_samples(trajectorySet,startG,finishG,goals,unitMat,meanLenMat,samplingAxis,kernelSetX,kernelSetY):
    errorVec = []
    samplesVec = []
    samples = 1
    steps = 4
    for i in range(20):
        samplesVec.append(samples)
        error = get_error_of_final_point_comparing_k_steps(samples,steps,trajectorySet,startG,finishG,goals,unitMat,meanLenMat,samplingAxis,kernelSetX,kernelSetY)
        errorVec.append(error)
        samples += 1

    #print("error: ", errorVec)
    plt.plot(samplesVec, errorVec,'b')
    plt.xlabel('Size of sample')
    plt.ylabel('Mean FDE')
    plt.show()
    #para ambos el error sera FDE
    #en 2 te detienes cuando el error deje de disminuir significativamente

def number_of_samples_and_points_to_compare_to_destination(goals,pathMat,rows,columns,unitMat,meanLenMat,samplingAxis,kernelMatX,kernelMatY):
    #test para encontrar los valores de k y m al elegir el punto final
    trajectorySet, kernelSetX, kernelSetY = [], [], []
    for i in range(rows):
        for j in range(columns):
            startGoal, finishGoal = i,j
            numOfPaths = len(pathMat[startGoal][finishGoal])
            lenTrainingSet = min(20,int(numOfPaths/2))
            #print( len(pathMat[startGoal][finalGoal]) )
            for k in range(lenTrainingSet):
                trajectorySet.append(pathMat[startGoal][finishGoal][k])
                kernelSetX.append(kernelMatX[startGoal][finishGoal])
                kernelSetY.append(kernelMatY[startGoal][finishGoal])
    #choose_number_of_steps_and_samples(trajectorySet,startGoal,finishGoal,areas,unitMat,meanLenMat)
    print("Test: number of steps to compare")
    choose_number_of_steps_to_compare(trajectorySet,startGoal,finishGoal,goals,unitMat,meanLenMat,samplingAxis,kernelSetX,kernelSetY)
    print("Test: number of destination samples")
    choose_number_of_destination_samples(trajectorySet,startGoal,finishGoal,goals,unitMat,meanLenMat,samplingAxis,kernelSetX,kernelSetY)

def interaction_using_sampling_test(img,pathSet,startGoals,finishGoals,stepUnit,speed,goalsData):
    #Refinar: unsar n/m datos observados
    observedXVec, observedYVec = [], []
    sampleXVec, sampleYVec = [], []
    samplePathSet = []
    nPaths = len(pathSet)
    for i in range(nPaths):
        #Obtener observaciones
        currentPath = pathSet[i]
        knownN = int(len(currentPath.x)/2)
        partialPath = get_partial_path(currentPath,knownN)
        observedX,observedY,observedL = get_known_set(currentPath.x,currentPath.y,currentPath.l,knownN)
        #Samplear el resto de la trayectoria
        x, y, l, mx, my = sample_path_to_goal(observedX,observedY,observedL,knownN,stepUnit,startGoals[i],finishGoals[i],goalsData)
        observedXVec.append(observedX)
        observedYVec.append(observedY)
        sampleX, sampleY = np.reshape(x,(x.shape[0])), np.reshape(y,(y.shape[0]))
        sampleXVec.append(sampleX)
        sampleYVec.append(sampleY)
        samplePath = get_trajectory_from_path(partialPath,sampleX,sampleY,l,speed)
        samplePathSet.append(samplePath)
    interaction_potential_for_a_set_of_pedestrians(samplePathSet)
    #plot_multiple_path_samples_with_observations(img,observedXVec,observedYVec,sampleXVec,sampleYVec)
    plotPathSet(samplePathSet,img)
    #Mas adelante: guardar potencial + config

def real_path_predicted_mean_and_sample(img,realPath,areas,goalsData,stepUnit):
    futureSteps = [8]
    partNum = 7
    nSamples = 50
    nGoals = goalsData.nGoals
    PM, samples, knownData = [], [], []
    for steps in futureSteps:
        print("__Comparing",steps,"steps__")
        
        for i in range(partNum-1):
            currentPath = realPath
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
            PredMean = predictedXYVec[mostLikelyG]
            minPMError = ADE_given_future_steps(currentPath, predictedXYVec[mostLikelyG], knownN, steps)
            if mgps.gpPathRegressor[mostLikelyG + nGoals] != None:
                for it in range(mgps.nSubgoals):
                    k = mostLikelyG + (it+1)*nGoals
                    error = ADE_given_future_steps(currentPath, predictedXYVec[k], knownN, steps)
                    if minPMError == 0:
                        minPMError = error
                        PredMean = predictedXYVec[k]
                    if error > 0 and error < minPMError:
                        minPMError = error
                        PredMean = predictedXYVec[k]
            """Sampling"""
            vecX,vecY,vecL  = mgps.generate_samples(nSamples) #Generate samples
            #Save the sample that gives the minimum error
            sample = [vecX[0][:,0], vecY[0][:,0]]
            minSampleError = ADE_given_future_steps(currentPath, sample, knownN, steps) 
            for k in range(nSamples):
                sampleXY = [vecX[k][:,0], vecY[k][:,0]]
                error = ADE_given_future_steps(currentPath, sampleXY, knownN, steps)
                if error < minSampleError:
                    minSampleError = error
                    sample = sampleXY
            PM.append(PredMean)
            samples.append(sample)
            knownData.append(knownN)
            #plot_observations_predictive_mean_and_sample(img,realPath,knownN,PredMean,sample)
    sequence_of_observations_predmean_samples(img,realPath,knownData,PM,samples)




