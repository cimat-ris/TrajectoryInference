"""
Regression with Gaussian Processes
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.optimize import minimize
from scipy.linalg import *
import kernels
import path
from dataManagement import *
import dataManagement
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse
from copy import copy
import random
import timeit
from termcolor import colored

# The main regression function: perform regression for a vector of values lnew
def joint_regression(l,x_meanl,lnew,kernel,linearPriorMean=None):
    # Number of observed data
    n    = len(l)
    # Number of predicted data
    nnew = len(lnew)
    # Compute K (nxn), k (nxnnew), C (nnewxnnew)
    K  = np.zeros((n,n))
    k  = np.zeros((n,nnew))
    C  = np.zeros((nnew,nnew))
    # Fill in K
    for i in range(n):
        for j in range(n):
            K[i][j] = kernel(l[i],l[j])
    K_1 = inv(K)
    # Fill in k
    for i in range(n):
        for j in range(nnew):
            k[i][j] = kernel(l[i],lnew[j],False)
    # Fill in C
    for i in range(nnew):
        for j in range(nnew):
            C[i][j] = kernel(lnew[i],lnew[j],False)
    # Predictive mean
    xnew = k.transpose().dot(K_1.dot(x_meanl))
    if linearPriorMean!=None:
        for j in range(nnew):
            xnew[j] += linear_mean(lnew[j],linearPriorMean[0])
    # Estimate the variance
    K_1kt = K_1.dot(k)
    kK_1kt = k.transpose().dot(K_1kt)
    # Variance
    var = C - kK_1kt
    return xnew, var

# Individual regression for a single scalar lnew
def single_regression(l,x_meanl,lnew,kernel,priorMean=None):
    # Number of observed data
    n    = len(l)
    # Compute K, k and c
    K  = np.zeros((n,n))
    k  = np.zeros(n)
    # Fill in K
    for i in range(n):
        for j in range(n):
            K[i][j] = kernel(l[i],l[j])
    # Fill in k
    for i in range(n):
        k[i] = kernel(lnew,l[i],False)
    K_1 = inv(K)
    # Predictive mean
    xnew = k.dot(K_1.dot(x_meanl))
    if linearPriorMean!=None:
        xnew += linear_mean(lnew,linearPriorMean[0])
    # Estimate the variance
    K_1kt = K_1.dot(k.transpose())
    kK_1kt = k.dot(K_1kt)
    # Variance
    var = kernel(lnew,lnew,False) - kK_1kt
    # Clip variance
    if var<0.1:
        var = 0.1
    return xnew, var

# Function to get the ground truth data: knownN data
# TODO: not sure if it is the good place
def get_known_set(x,y,l,knownN):
    trueX = x[0:knownN]
    trueY = y[0:knownN]
    trueL = l[0:knownN]
    return trueX, trueY, trueL

# TODO: not sure if it is the good place
def get_goal_likelihood(observedX,observedY,observedL,startG,finishG,stepsToCompare,goalsData):

    error = prediction_error_of_points_along_the_path(nPoints,observedX,observedY,observedL,finishG,goalsDat)


    # All the observed data
    _observedX = observedX.copy()
    _observedY = observedY.copy()
    _observedL = observedL.copy()

    # Takes the last stepsToCompare data from the observed data
    stepsToCompare = 5
    trueX, trueY, predSet = [], [], []
    for i in range(stepsToCompare):
        valX = _knownX.pop()
        trueX.append(valX)
        valY = _knownY.pop()
        trueY.append(valY)
        step = _knownL.pop()
        predSet.append(step)

    # New set of observed data (amputed from stepsToCompare)
    n = len(_observedX)
    # Takes the center of the
    finish_xy = middle_of_area(goals[i])
    _knownX.append(finish_xy[0])
    _knownY.append(finish_xy[1])
    dist = math.sqrt( (_knownX[n-1] - finish_xy[0])**2 + (_knownY[n-1] - finish_xy[1])**2 )
    unit = unitMat[startG][i]
    lastL = knownL[n-1] + dist*unit
    _knownL.append(lastL)
    kernelX = kernelMatX[startG][i]
    kernelY = kernelMatY[startG][i]
    predX, predY, vx, vy = prediction_xy(_knownX, _knownY, _knownL, predSet, kernelX, kernelY)
    error = average_displacement_error([trueX,trueY],[predX,predY])
    return error

# TODO: not sure if it is the good place
def get_finish_point(knownX, knownY, knownL, finishGoal, goals, kernelX, kernelY, unit, samplingAxis):
    n = len(knownX)
    numSamples = 9 #numero de muestras
    _x, _y, flag = uniform_sampling_1D(numSamples, goals[finishGoal], samplingAxis[finishGoal])
    k = 3          #num de puntos por comparar
    if(n < 2*k):
        return middle_of_area(goals[finishGoal])

    _knownX = knownX[0:n-k]
    _knownY = knownY[0:n-k]
    _knownL = knownL[0:n-k]

    predSet = knownL[n-k:k]
    trueX = knownX[n-k:k]
    trueY = knownY[n-k:k]

    error = []
    for i in range(numSamples):
        auxX = _knownX.copy()
        auxY = _knownY.copy()
        auxL = _knownL.copy()
        auxX.append(_x[i])
        auxY.append(_y[i])
        dist = math.sqrt( (knownX[n-1] - _x[i])**2 + (knownY[n-1] - _y[i])**2 )
        lastL = knownL[n-1] + dist*unit
        auxL.append(lastL)
        predX, predY, vx, vy = prediction_xy(auxX, auxY, auxL, predSet, kernelX, kernelY)
        #error.append(geometricError(trueX,trueY,predX,predY))
        error.append(average_displacement_error([trueX,trueY],[predX,predY]))
    #encuentra el punto que genera el error minimo
    min_id, min_error = 0, error[0]
    for i in range(numSamples):
        if(error[i] < min_error):
            min_error = error[i]
            min_id = i
    return [_x[min_id], _y[min_id]]

# Evaluate the prediction error
def compute_prediction_error_1D(trueX, trueY, prediction, flag):
    error = 0.0
    for i in range(len(prediction) ):
        if flag == 'x':
            error += abs(trueX[i] - prediction[i])
        if flag == 'y':
            error += abs(trueY[i] - prediction[i])
    return error

def get_finish_point_singleGP(knownX, knownY, knownL, finishGoal, goals, kernelX, kernelY, unit, img, samplingAxis):
    n = len(knownX)
    m = 10 #numero de muestras
    _x, _y, flag  = uniform_sampling_1D(m, goals[finishGoal], samplingAxis[finishGoal])
    k = 5 #numero de puntos por comparar

    _knownX = knownX[0:n-k]
    _knownY = knownY[0:n-k]

    trueX = knownX[n-k:k]
    trueY = knownY[n-k:k]

    error = []
    for i in range(m):
        auxX = _knownX.copy()
        auxY = _knownY.copy()
        auxX.append(_x[i])
        auxY.append(_y[i])
        if flag == 'y': #x(y)
            predSet = trueY.copy()
            prediction, var = estimate_new_set_of_values(auxY,auxX,predSet,kernelX)
            plot_sampling_prediction(img,knownX,knownY,n-k,prediction,trueY,var,var,[_x[i],_y[i]])
            error.append(prediction_error_1D(trueX, trueY, prediction, 'x'))
        if flag == 'x': #y(x)
            predSet = trueX.copy()
            prediction, var = estimate_new_set_of_values(auxX,auxY,predSet,kernelY)
            plot_sampling_prediction(img,knownX,knownY,n-k,trueX,prediction,var,var,[_x[i],_y[i]])
            error.append(prediction_error_1D(trueX, trueY, prediction, 'y')) #y(x)

    #encuentra el punto que genera el error minimo
    min_id, min_error = 0, error[0]
    for i in range(m):
        if(error[i] < min_error):
            min_error = error[i]
            min_id = i
    return [_x[min_id], _y[min_id]]

# Determines the set of l values to predict from a given set of data,
# where knownN are specified as known
def get_prediction_set_from_data(l,knownN):
    N    = len(l)
    newL = l[knownN-1:N]
    return newL

#usa una unidad de distancia segun el promedio de la arc-len de las trayectorias
#start, last know (x,y,l), indices del los goals de inicio y fin, unitMat, numero de pasos
def get_prediction_set(lastKnownPoint, finishPoint, distUnit, stepUnit):
    x, y, l = lastKnownPoint[0], lastKnownPoint[1], lastKnownPoint[2]
    _x, _y  = finishPoint[0], finishPoint[1]

    euclideanDist = euclidean_distance([x,y], [_x,_y])
    dist          = euclideanDist*distUnit
    numSteps      = int(dist*stepUnit)
    newset = []
    if(numSteps > 0):
        step = dist/float(numSteps)
        for i in range(numSteps+1):
            newset.append( l + i*step )
    return newset, l + dist

def get_prediction_set_given_size(lastKnownPoint, finishPoint, unit, steps):
    x, y, l = lastKnownPoint[0], lastKnownPoint[1], lastKnownPoint[2]
    _x, _y = finishPoint[0], finishPoint[1]
    dist = math.sqrt( (_x-x)**2 + (_y-y)**2 )
    newset = []
    if(steps > 0):
        step = dist/float(steps)
        for i in range(steps+1):
            newset.append( l + i*step*unit )

    return newset, l + dist*unit

# Compute the arc-length from one point to the final points
# given the unit
def get_arclen_to_finish_point(point, finishPoint, unit):
    x, y, l = point[0], point[1], point[2]
    _x, _y = finishPoint[0], finishPoint[1]
    dist = math.sqrt( (_x-x)**2 + (_y-y)**2 )
    return l + dist*unit

def get_subgoals_center_and_size(nSubgoals, goal, axis):
    goalX = goal[len(goal) -2] - goal[0]
    goalY = goal[len(goal) -1] - goal[1]
    goalCenterX = goal[0]+ goalX/2
    goalCenterY = goal[1]+ goalY/2

    subgoalsCenter = []
    subgoalX, subgoalY = 0,0
    if axis == 'x':
        subgoalX = goalX/nSubgoals
        subgoalY = goalY
        _x = goal[0]
        _y = goalCenterY
        for i in range(nSubgoals):
            subgoalsCenter.append( [_x+subgoalX/2, _y] )
            _x += subgoalX

    if axis == 'y':
        subgoalX = goalX
        subgoalY = goalY/nSubgoals
        _x = goalCenterX
        _y = goal[1]
        for i in range(nSubgoals):
            subgoalsCenter.append( [_x, _y+subgoalY/2] )
            _y += subgoalY

    return subgoalsCenter, [subgoalX, subgoalY]

# Applies joint regression for a whole set newL of values L, given knownL, knownX
def joint_estimate_new_set_of_values(observedL,observedX,newL,kernel,linearPriorMean=None):
    centeredX = np.zeros((len(observedL),1),dtype=float)
    if linearPriorMean==None:
        for i in range(len(observedL)):
            centeredX[i][0] = observedX[i]
    else:
        for i in range(len(observedL)):
            centeredX[i][0] = observedX[i] - linear_mean(observedL[i], linearPriorMean[0])
    # Applies regression for the joint values predictedX
    predictedX, covarianceX = joint_regression(observedL,centeredX,newL,kernel,linearPriorMean)
    return predictedX, covarianceX

# Prediction of future positions towards a given finish point, given observations
def prediction_to_finish_point(observedX,observedY,observedL,nObservations,finishPoint,unit,stepUnit,kernelX,kernelY,priorMeanX=None,priorMeanY=None):
    # Last observed point
    lastObservedPoint = [observedX[nObservations-1], observedY[nObservations-1], observedL[nObservations-1] ]
    # Generate the set of l values at which to predict x,y
    newL, finalL = get_prediction_set(lastObservedPoint,finishPoint,unit,stepUnit)
    # One point at the final of the path
    observedX.append(finishPoint[0])
    observedY.append(finishPoint[1])
    observedL.append(finalL)

    # Performs regression for newL
    newX,newY,varX,varY = prediction_xy(observedX,observedY,observedL,newL,kernelX,kernelY,priorMeanX,priorMeanY)

    # Removes the last observed point (which was artificially added)
    observedX.pop()
    observedY.pop()
    observedL.pop()
    return newX, newY, newL, varX, varY

#Toma N-nPoints como datos conocidos y predice los ultimos nPoints, regresa el error de la prediccion
def prediction_error_of_last_known_points(nPoints,knownX,knownY,knownL,goal,unit,stepUnit,kernelX,kernelY):
    knownN = len(knownX)
    trueX = knownX[0:knownN -nPoints]
    trueY = knownY[0:knownN -nPoints]
    trueL = knownL[0:knownN -nPoints]

    finishXY = middle_of_area(goal)
    finishD = euclidean_distance([trueX[len(trueX)-1],trueY[len(trueY)-1]],finishXY)
    trueX.append(finishXY[0])
    trueY.append(finishXY[1])
    trueL.append(finishD*unit)

    lastX = knownX[knownN -nPoints: nPoints]
    lastY = knownY[knownN -nPoints: nPoints]
    predictionSet = knownL[knownN -nPoints: nPoints]

    predX, predY, varX,varY = prediction_xy(trueX,trueY,trueL, predictionSet, kernelX, kernelY)
    #print("[Prediccion]\n",predX)
    #print(predY)
    error = average_displacement_error([lastX,lastY],[predX,predY])
    #print("[Error]:",error)
    return error

"""ARC LENGHT TO TIME"""
def arclen_to_time(initTime,l,speed):
    t = [initTime]
    for i in range(1,len(l)):
        time_i = int(t[i-1] +(l[i]-l[i-1])/speed)
        t.append(time_i)
    return t

# Mean of the Gaussian process with a linear prior
def linear_mean(l, priorMean):
    m = priorMean[0]*l + priorMean[1]
    return m

# Performs prediction in X and Y
# Takes as input observed values (x,y,l) and the points at which we want to perform regression (newL)
def prediction_xy(observedX, observedY, observedL, newL, kernelX, kernelY, priorMeanX=None, priorMeanY=None):
    # Regression for X
    newX, varX = joint_estimate_new_set_of_values(observedL,observedX,newL,kernelX,priorMeanX)
    # Regression for Y
    newY, varY = joint_estimate_new_set_of_values(observedL,observedY,newL,kernelY,priorMeanY)
    return newX, newY, varX, varY

# For a given dataset (knownX,knownY,knownL), takes half of the data as known
# and predicts the remaining half. Then, evaluate the prediction error.
def compute_prediction_error_of_points_along_the_path(nPoints,observedX,observedY,observedL,startG,finishG,goalsData):
    # Known data
    observedN = len(observedX)
    halfN     = int(observedN/2)

    # First half of the known data
    trueX = observedX[0:halfN]
    trueY = observedY[0:halfN]
    trueL = observedL[0:halfN]

    # Get the last point and add it to the observed data
    finishXY = middle_of_area(goalsData.areas[finishG])
    finishD  = euclidean_distance([trueX[len(trueX)-1],trueY[len(trueY)-1]],finishXY)
    trueX.append(finishXY[0])
    trueY.append(finishXY[1])
    trueL.append(finishD*goalsData.units[startG][finishG])

    d = int(halfN/nPoints)
    realX, realY, predictionSet = [],[],[]
    # Prepare the ground truths and the list of l to evaluate
    for i in range(nPoints):
        realX.append(observedX[halfN + i*d])
        realY.append(observedY[halfN + i*d])
        predictionSet.append(observedL[halfN + i*d])
    # Get the prediction based on the
    predX, predY, varX,varY = prediction_xy(trueX,trueY,trueL, predictionSet, goalsData.kernelsX[startG][finishG],goalsData.kernelsY[startG][finishG],
    goalsData.linearPriorsX[startG][finishG],goalsData.linearPriorsY[startG][finishG])

    # Evaluate the error
    print('[INF] Evaluate the error')
    error = average_displacement_error([realX,realY],[predX,predY])

    return error
