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
