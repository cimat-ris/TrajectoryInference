"""
Regression with Gaussian Processes
"""
import numpy as np
import math
from numpy.linalg import inv
from scipy.linalg import *
from matplotlib.patches import Ellipse
from copy import copy
from utils.stats_trajectories import euclidean_distance
import random

# The main regression function: perform regression for a vector of values lnew
def joint_regression(l,x_meanl,lnew,kernel,linearPriorMean=None):
    # Number of observed data
    n    = len(l)
    # Number of predicted data
    nnew = len(lnew)
    # Compute K (nxn), k (nxnnew), C (nnewxnnew)kernel
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
            k[i][j] = kernel(l[i],lnew[j])
    # Fill in C
    for i in range(nnew):
        for j in range(nnew):
            C[i][j] = kernel(lnew[i],lnew[j])
    # Predictive mean
    xnew = k.transpose().dot(K_1.dot(x_meanl))
    if linearPriorMean is not None:
        for j in range(nnew):
            xnew[j] += linear_mean(lnew[j],linearPriorMean[0])
    # Estimate the variance
    K_1kt = K_1.dot(k)
    kK_1kt = k.transpose().dot(K_1kt)
    # Variance
    var = C - kK_1kt
    return xnew, var

# Determines the set of l values to predict from a given set of data,
# where knownN are specified as known
def get_prediction_set_from_data(l,knownN):
    N    = len(l)
    newL = l[knownN-1:N]
    return newL

#distUnit - unidad de distancia segun el promedio de la arc-len de las trayectorias
#last know (x,y,l), finish point, distUnit, stepUnit - pasos por unidad de dist
def get_prediction_set_arclengths(lastKnownPoint, finishPoint, distUnit, stepUnit):
    # Coordinates of the last observed point
    x, y, l = lastKnownPoint[0], lastKnownPoint[1], lastKnownPoint[2]
    # Coordinates if the assumed finish point
    _x, _y  = finishPoint[0], finishPoint[1]
    # Euclidean distance between the last observed point and the finish point
    euclideanDist = euclidean_distance([x,y], [_x,_y])
    # Rough estimate of the remaining arc length
    dist_to_goal  = euclideanDist
    numSteps      = int(dist_to_goal*stepUnit)
    newset = np.zeros((numSteps,1))
    if(numSteps > 0):
        step = dist_to_goal/float(numSteps)
        for i in range(1,numSteps+1):
            newset[i-1,0] = l + i*step
    return newset, l + dist_to_goal, dist_to_goal

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

#****************** Functions for Trautmans code ******************
#last know (x,y,l), finish point, distUnit, stepUnit - pasos por unidad de dist, speed
def get_prediction_set_time(lastKnownPoint, elapsedTime, timeTransitionData, timeStep):
    # Coordinates of the last observed point
    x, y, t = lastKnownPoint[0], lastKnownPoint[1], lastKnownPoint[2]
    # TODO: I think we should first do here a fully deterministic model (conditioned on the mean transition time)
    # Sample a duration
    transitionTime = int(np.random.normal(timeTransitionData[0], timeTransitionData[1]) )
    # Remaining time
    remainingTime = transitionTime - elapsedTime
    numSteps      = int(remainingTime/timeStep)
    newset        = []
    if(numSteps > 0):
        for i in range(1,numSteps+1):
            newset.append( t + i*timeStep )
        if newset[numSteps-1] < t+ remainingTime:
            newset.append(t+remainingTime)
    else:
        newset.append(t+remainingTime)
    return newset, t + remainingTime, remainingTime

#******************
# Compute the arc-length from one point to the final points
# given the unit
def get_arclen_to_finish_point(point, finishPoint, unit):
    x, y, l = point[0], point[1], point[2]
    _x, _y = finishPoint[0], finishPoint[1]
    dist = math.sqrt( (_x-x)**2 + (_y-y)**2 )
    return l + dist*unit

# TODO: REDO IN A BETTER WAY
def get_subgoals_areas(nSubgoals, goal, axis):
    goalDX = goal[len(goal) -2] - goal[0]
    goalDY = goal[len(goal) -1] - goal[1]
    goalCenterX = goal[0]+ goalDX/2.0
    goalCenterY = goal[1]+ goalDY/2.0
    goalMinX    = goal[0]
    goalMinY    = goal[1]
    goalMaxX    = goal[-2]
    goalMaxY    = goal[-1]
    subGoalsAreas = []
    if axis == 0:
        subgoalDX = goalDX/nSubgoals
        subgoalDY = goalDY
        for i in range(nSubgoals):
            subGoalsAreas.append( [goalMinX+i*subgoalDX,goalMinY,goalMinX+(i+1)*subgoalDX,goalMinY,goalMinX+i*subgoalDX,goalMaxY,goalMinX+(i+1)*subgoalDX,goalMaxY] )
    else:
        subgoalDX = goalDX
        subgoalDY = goalDY/nSubgoals
        _x = goalCenterX
        _y = goal[1]
        for i in range(nSubgoals):
            subGoalsAreas.append([goalMinX,goalMinY+i*subgoalDY,goalMaxX,goalMinY+i*subgoalDY,goalMinX,goalMinY+(i+1)*subgoalDY,goalMaxX,goalMinY+(i+1)*subgoalDY])

    return subGoalsAreas


def get_subgoals_center_and_size(nSubgoals, goal, axis):
    goalX = goal[len(goal) -2] - goal[0]
    goalY = goal[len(goal) -1] - goal[1]
    goalCenterX = goal[0]+ goalX/2
    goalCenterY = goal[1]+ goalY/2

    subgoalsCenter = []
    subgoalX, subgoalY = 0,0
    if axis == 0:
        subgoalX = goalX/nSubgoals
        subgoalY = goalY
        _x = goal[0]
        _y = goalCenterY
        for i in range(nSubgoals):
            subgoalsCenter.append( [_x+subgoalX/2.0, _y] )
            _x += subgoalX
    else:
        subgoalX = goalX
        subgoalY = goalY/nSubgoals
        _x = goalCenterX
        _y = goal[1]
        for i in range(nSubgoals):
            subgoalsCenter.append( [_x, _y+subgoalY/2.0] )
            _y += subgoalY

    return subgoalsCenter, [subgoalX, subgoalY]

# Applies joint regression for a whole set newL of values L, given knownL, knownX
def joint_estimate_new_set_of_values(observedL,observedX,newL,kernel,linearPriorMean=None):
    centeredX = np.zeros((len(observedL),1),dtype=float)
    if linearPriorMean is None:
        for i in range(len(observedL)):
            centeredX[i][0] = observedX[i]
    else:
        for i in range(len(observedL)):
            centeredX[i][0] = observedX[i] - linear_mean(observedL[i], linearPriorMean[0])
    # Applies regression for the joint values predictedX
    predictedX, covarianceX = joint_regression(observedL,centeredX,newL,kernel,linearPriorMean)
    return predictedX, covarianceX

# Prediction of future positions towards a given finish point, given observations
def prediction_to_finish_point(observedX,observedY,observedL,nObservations,finishPoint,stepUnit,start,end,goalsData):
    # Last observed point
    lastObservedPoint = [observedX[nObservations-1], observedY[nObservations-1], observedL[nObservations-1] ]
    # Generate the set of l values at which to predict x,y
    newL, finalL = get_prediction_set(lastObservedPoint,finishPoint,goalsData.units[start][end],stepUnit)
    # One point at the final of the path
    observedX.append(finishPoint[0])
    observedY.append(finishPoint[1])
    observedL.append(finalL)

    # Performs regression for newL
    newX,newY,varX,varY = prediction_xy(observedX,observedY,observedL,newL,goalsData.kernelsX[start][end],goalsData.kernelsY[start][end],goalsData.linearPriorsX[start][end],goalsData.linearPriorsX[start][end])

    # Removes the last observed point (which was artificially added)
    observedX.pop()
    observedY.pop()
    observedL.pop()
    return newX, newY, newL, varX, varY

# Mean of the Gaussian process with a linear prior (slope and constant)
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
