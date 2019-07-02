# -*- coding: utf-8 -*-
"""
Testing functions

@author: karenlc
"""

from GPRlib import *
from path import *
from plotting import *
from kernels import *
from statistics import*
from sampling import*
import matplotlib.pyplot as plt
import numpy as np
import math
from copy import copy


def trajectory_prediction_test_using_sampling(img,x,y,l,knownN,startG,finishG,goals,unitMat,stepUnit,kernelMatX,kernelMatY,samplingAxis):
    kernelX = kernelMatX[startG][finishG]
    kernelY = kernelMatY[startG][finishG]

    trueX, trueY, trueL = get_known_set(x,y,l,knownN)
    lastKnownPoint = [x[knownN-1], y[knownN-1], l[knownN-1] ]
    unit = unitMat[startG][finishG]

    final_xy = get_finish_point(trueX,trueY,trueL,finishG,goals,kernelX,kernelY,unit,samplingAxis)
    #final_xy = get_finish_point_singleGP(trueX,trueY,trueZ,finishG,goals,kernelX,kernelY,unit,img)
    newL, final_l = get_prediction_set(lastKnownPoint,final_xy,unit,stepUnit)
    trueX.append(final_xy[0])
    trueY.append(final_xy[1])
    trueL.append(final_l)

    newX,newY,varX,varY = prediction_XY(trueX,trueY,trueL,newL,kernelX,kernelY)
    plot_prediction(img,x,y,knownN,newX,newY,varX,varY)

def trajectory_prediction_test(img,x,y,l,knownN,startG,finishG,goals,unitMat,stepUnit,kernelMatX,kernelMatY,linearPriorMatX,linearPriorMatY):
    kernelX = kernelMatX[startG][finishG]
    kernelY = kernelMatY[startG][finishG]

    trueX, trueY, trueL = get_known_set(x,y,l,knownN)
    lastKnownPoint = [x[knownN-1], y[knownN-1], l[knownN-1] ]
    unit = unitMat[startG][finishG]

    #print("last known point:",lastKnownPoint)
    finalPoints = get_goal_center_and_boundaries(goals[finishG])

    #newL, finalL = get_prediction_set_given_size(lastKnownPoint,finalPoints[0],unit,20)
    newL, finalL = get_prediction_set(lastKnownPoint,finalPoints[0],unit,stepUnit)
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

    newL, finalL = get_prediction_set(lastKnownPoint,finalPoints[0],unit,stepUnit)
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
def prediction_test_over_time(x,y,z,knownN,start,end,goals):
    kernelX = kernelMat_x[start][end]
    kernelY = kernelMat_y[start][end]

    trueX, trueY, trueZ = get_known_set(x,y,z,knownN)
    final_xy = middle_of_area(goals[end])
    N = len(x)
    trueX.append(final_xy[0])
    trueY.append(final_xy[1])
    trueZ.append(z[N-1])
    newZ = get_prediction_set_from_data(z,knownN)
    newX, newY,varX,varY = prediction_XY(trueX,trueY,trueZ,newZ,kernelX,kernelY)
    plot_prediction(img,x,y,knownN,newX,newY,varX,varY)

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

#recibe: datos reales, num de datos conocidos, area de inicio, vec de areas objetivo, vec de areas
def _multigoal_prediction_test(x,y,z,knownN,startG,finishG,goals,unitMat,steps,kernelMat_x,kernelMat_y):
    trueX, trueY, trueZ = get_known_set(x,y,z,knownN)
    lastKnownPoint = [x[knownN-1], y[knownN-1], z[knownN-1] ]

    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(img) # Show the image

    for i in range(len(finishG)):
        nextGoal = finishG[i]
        kernelX = kernelMat_x[startG][nextGoal]
        kernelY = kernelMat_y[startG][nextGoal]
        auxX = copy(trueX)
        auxY = copy(trueY)
        auxZ = copy(trueZ)
        final_point = middle_of_area(goals[nextGoal])
        auxX.append(final_point[0])
        auxY.append(final_point[1])
        #steps = 20
        end_, newZ, l_ = getPredictionSet(trueX[knownN-1],trueY[knownN-1],trueZ[knownN-1],start,nextGoal,goals)
        auxZ.append(l_)
        newX, newY, varX, varY = prediction_XY(auxX,auxY,auxZ,newZ,kernelX,kernelY)

        plt.plot(trueX,trueY,'r')
        plt.plot(newX,newY,'b')
        #elipses
        for j in range(len(newX)):
            xy = [newX[j],newY[j]]
            ell = Ellipse(xy,2.*np.sqrt(varX[j]),2.*np.sqrt(varY[j]))
            ell.set_alpha(.4)
            ell.set_lw(0)
            ell.set_facecolor('g')
            ax.add_patch(ell)

    v = [0,1920,1080,0]
    plt.axis(v)
    plt.show()

def multigoal_prediction_test(img,x,y,l,knownN,startG,goals,unitMat,stepUnit,kernelMatX,kernelMatY,priorLikelihood,subgoalAxis):
    trueX, trueY, trueL = get_known_set(x,y,l,knownN)
    likelyGoals, goalsLikelihood = [],[]
    errorG = []
    nPoints = 5
    for i in range(len(goals)):
        unit = unitMat[startG][i]
        kernelX = kernelMatX[startG][i]
        kernelY = kernelMatY[startG][i]

        error = prediction_error_of_points_along_the_path(nPoints,trueX,trueY,trueL,goals[i],unit,kernelX,kernelY)
        #error = prediction_error_of_last_known_points(nPoints,trueX,trueY,trueL,goals[i],unit,stepUnit,kernelX,kernelY)
        #error = get_goal_likelihood(trueX,trueY,trueL,startG,i,goals,unitMat,kernelMatX,kernelMatY)
        errorG.append(error)

    print("[Prediction Error]\n",errorG)
    D = 150.
    for i in range(len(goals)):
        val = priorLikelihood[startG][i]*(math.exp(-1.*( errorG[i]**2)/D**2 ))#   *(1.-errorG[i])
        goalsLikelihood.append(val)

    meanLikelihood = 0.85*mean(goalsLikelihood)
    for i in range(len(goals)):
        if(goalsLikelihood[i] > meanLikelihood):
            likelyGoals.append(i)

    #print("\n[Prior likelihood]\n",priorLikelihood[startG])
    print("[Goals likelihood]\n",goalsLikelihood)
    print("[Mean likelihood]:", meanLikelihood)
    nSubgoals = 2
    goalCount = 0
    plotLikelihood = []
    predictedXYVec, varXYVec = [], []
    for i in range(len(likelyGoals)):
        nextG = likelyGoals[i]
        unit = unitMat[startG][nextG]
        kernelX = kernelMatX[startG][nextG]
        kernelY = kernelMatY[startG][nextG]

        goalCenter = middle_of_area(goals[nextG])
        distToGoal = euclidean_distance([trueX[knownN-1],trueY[knownN-1]], goalCenter)
        dist = euclidean_distance([trueX[0],trueY[0]], goalCenter)
        if(distToGoal < 0.4*dist):
            subgoalsCenter, size = get_subgoals_center_and_size(nSubgoals, goals[nextG], subgoalAxis[nextG])
            for j in range(nSubgoals):
                predictedX, predictedY, varX, varY = prediction_to_finish_point(trueX,trueY,trueL,knownN,subgoalsCenter[j],unit,stepUnit,kernelX,kernelY)#trajectory_prediction_test(x,y,l,knownN,startG,nextGoal,goals,unitMat,stepUnit,kernelMatX,kernelMatY)
                predictedXYVec.append([predictedX, predictedY])
                varXYVec.append([varX, varY])
                plotLikelihood.append(goalsLikelihood[nextG])
            goalCount += nSubgoals
        else:
            predictedX, predictedY, varX, varY = prediction_to_finish_point(trueX,trueY,trueL,knownN,goalCenter,unit,stepUnit,kernelX,kernelY)#trajectory_prediction_test(x,y,l,knownN,startG,nextGoal,goals,unitMat,stepUnit,kernelMatX,kernelMatY)
            predictedXYVec.append([predictedX, predictedY])
            varXYVec.append([varX, varY])
            plotLikelihood.append(goalsLikelihood[nextG])
            goalCount += 1
    plot_multiple_predictions_and_goal_likelihood(img,x,y,knownN,goalCount,plotLikelihood,predictedXYVec,varXYVec)
    #plot_multiple_predictions(img,x,y,knownN,goalCount,predictedXYVec,varXYVec)

# Prediction with multiple goals and line priors
def multigoal_prediction_test_lp(img,x,y,l,knownN,startG,goals,unitMat,stepUnit,kernelMatX,kernelMatY,priorLikelihood,linearPriorMatX,linearPriorMatY,subgoalAxis):
    # Ground truth
    trueX, trueY, trueL = get_known_set(x,y,l,knownN)
    # Goals
    likelyGoals, goalsLikelihood = [],[]
    errorG  = []
    nPoints = 5
    for i in range(len(goals)):
        # Length/euclidean distance ratio
        unit = unitMat[startG][i]
        # Kernels
        kernelX = kernelMatX[startG][i]
        kernelY = kernelMatY[startG][i]
        # Linear priors
        priorMeanX = linearPriorMatX[startG][i]
        priorMeanY = linearPriorMatY[startG][i]
        # Evaluate the error
        error = prediction_error_of_points_along_the_path_lp(nPoints,trueX,trueY,trueL,goals[i],unit,kernelX,kernelY,priorMeanX,priorMeanY)
        errorG.append(error)

    print("[RES] [Prediction Error]\n",errorG)
    norma = np.linalg.norm(errorG)
    #errorG = errorG/norma
    D = 150.
    # Compute the likelihoods
    for i in range(len(goals)):
        val = priorLikelihood[startG][i]*(math.exp(-1.*( errorG[i]**2)/D**2 ))#   *(1.-errorG[i])
        goalsLikelihood.append(val)
    # Consider the mean likelihood
    meanLikelihood = 0.85*mean(goalsLikelihood)
    for i in range(len(goals)):
        if(goalsLikelihood[i] > meanLikelihood):
            likelyGoals.append(i)

    print("[RES] [Goals likelihood]\n",goalsLikelihood)
    print("[RES] [Mean likelihood]:", meanLikelihood)
    nSubgoals = 2
    goalCount = 0
    plotLikelihood = []
    predictedXYVec, varXYVec = [], []

    # For all likely goals
    for i in range(len(likelyGoals)):
        nextG = likelyGoals[i]
        unit = unitMat[startG][nextG]
        kernelX = kernelMatX[startG][nextG]
        kernelY = kernelMatY[startG][nextG]
        priorMeanX = linearPriorMatX[startG][nextG] #line prior prediction
        priorMeanY = linearPriorMatY[startG][nextG]

        goalCenter = middle_of_area(goals[nextG])
        distToGoal = euclidean_distance([trueX[knownN-1],trueY[knownN-1]], goalCenter)
        dist = euclidean_distance([trueX[0],trueY[0]], goalCenter)
        # When close to the goal, define sub-goals
        if(distToGoal < 0.4*dist):
            subgoalsCenter, size = get_subgoals_center_and_size(nSubgoals, goals[nextG], subgoalAxis[nextG])
            for j in range(nSubgoals):
                # Generate prediction to this goal
                predictedX, predictedY, varX, varY = prediction_to_finish_point_lp(trueX,trueY,trueL,knownN,subgoalsCenter[j],unit,stepUnit,kernelX,kernelY,priorMeanX,priorMeanY)#trajectory_prediction_test(x,y,l,knownN,startG,nextGoal,goals,unitMat,stepUnit,kernelMatX,kernelMatY)
                # Lists of predictive means
                predictedXYVec.append([predictedX, predictedY])
                # Lost of variances
                varXYVec.append([varX, varY])
                plotLikelihood.append(goalsLikelihood[nextG])
            goalCount += nSubgoals
        else:
            # Evaluate the prediction and the corresponding covariance matrices
            predictedX, predictedY, varX, varY = prediction_to_finish_point_lp(trueX,trueY,trueL,knownN,goalCenter,unit,stepUnit,kernelX,kernelY,priorMeanX,priorMeanY)#trajectory_prediction_test(x,y,l,knownN,startG,nextGoal,goals,unitMat,stepUnit,kernelMatX,kernelMatY)
            # Lists of predictive means
            predictedXYVec.append([predictedX, predictedY])
            # Covariance matrices
            varXYVec.append([varX, varY])
            plotLikelihood.append(goalsLikelihood[nextG])
            goalCount += 1
    # Plot everything
    print('[INF] Plotting')
    plot_multiple_predictions_and_goal_likelihood(img,x,y,knownN,goalCount,plotLikelihood,predictedXYVec,varXYVec)

# Sampling 3 trajectories between all the pairs of goals
def path_sampling_test(img,goals,nGoals,samplingAxis,unitMat,stepUnit,kernelMatX,kernelMatY,priorMeanMatX,priorMeanMatY):
    vecX, vecY = [], []
    for i in range(nGoals):
        for j in range(i,nGoals):
            if(i != j):
                startG, finishG = i, j
                for k in range(3): #num of samples
                    x, y = sample_path(goals,startG,finishG,samplingAxis,unitMat[startG][finishG],stepUnit,kernelMatX[startG][finishG],kernelMatY[startG][finishG],priorMeanMatX[startG][finishG],priorMeanMatY[startG][finishG])
                    vecX.append(x)
                    vecY.append(y)
    plot_path_samples(img, vecX,vecY)

# Sampling trajectories between two goals
def path_sampling_between_goals_test(img,nSamples,goals,startG,finishG,samplingAxis,unitMat,stepUnit,kernelMatX,kernelMatY,priorMeanMatX,priorMeanMatY):
    vecX, vecY = [], []
    for k in range(nSamples):
        x, y, l, mx, my = sample_path(goals,startG,finishG,samplingAxis,unitMat[startG][finishG],stepUnit,kernelMatX[startG][finishG],kernelMatY[startG][finishG],priorMeanMatX[startG][finishG],priorMeanMatY[startG][finishG])
        vecX.append(x)
        vecY.append(y)
        # For debugging
        plt.plot(l,x)
        plt.plot(l,y)
        plt.plot(l,mx)
        plt.plot(l,mx)
    plot_path_samples(img, vecX,vecY)

# Sampling trajectories to a given goal
def path_sampling_to_goal_test(img,observedX,observedY,observedL,knownN,nSamples,goals,startG,finishG,samplingAxis,unitMat,stepUnit,kernelMatX,kernelMatY,linearPriorMatX,linearPriorMatY):
    # Ground truth
    trueX, trueY, trueL = get_known_set(observedX,observedY,observedL,knownN)

    # Length/euclidean distance ratio
    unit = unitMat[startG][finishG]
    # Kernels
    kernelX = kernelMatX[startG][finishG]
    kernelY = kernelMatY[startG][finishG]
    # Linear priors
    priorMeanX = linearPriorMatX[startG][finishG]
    priorMeanY = linearPriorMatY[startG][finishG]

    vecX, vecY = [], []
    for k in range(nSamples):
        x, y, l, mx, my = sample_path_to_goal(observedX,observedY,observedL,knownN,goals,finishG,samplingAxis,unitMat[startG][finishG],stepUnit,kernelMatX[startG][finishG],kernelMatY[startG][finishG],linearPriorMatX[startG][finishG],linearPriorMatY[startG][finishG])
        vecX.append(x)
        vecY.append(y)
    plot_path_samples_with_observations(img,observedX,observedY,vecX,vecY)



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

def choose_number_of_steps_to_compare(trajectorySet,startG,finishG,goals,unitMat,meanLenMat,samplingAxis,kernelSetX,kernelSetY):
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
