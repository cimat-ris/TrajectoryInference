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
