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
