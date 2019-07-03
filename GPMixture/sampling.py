# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 17:56:43 2018
@author: karenlc
"""

from GPRlib import *
from kernels import *
from path import *
from regression import *
from statistics import *
import matplotlib.pyplot as plt
import numpy as np
import math
from copy import copy


def goal_sequence(L, n):
    s = []
    for i in range(n):
        s.append[i]
    return s

# Sample m points (x,y) in an area, with uniform sampling.
def uniform_sampling_2D(m, goal):
    _x, _y = [], []

    # Determines the bounding box
    xmin, xmax = goal[0], goal[2]
    ymin, ymax = goal[1], goal[len(goal)-1]

    # Performs the sampling
    for i  in range(m):
        t = random.uniform(0,1.)
        val = (1.-t)*xmin + t*xmax
        _x.append(val)
        r = random.uniform(0,1.)
        val = (1.-r)*ymin + r*ymax
        _y.append(val)

    return _x, _y

# Sample m points (x,y) along a line segment, with uniform sampling
def uniform_sampling_1D(m, goal, axis):
    _x, _y = [], []
    xmin, xmax = goal[0], goal[2]
    ymin, ymax = goal[1], goal[len(goal)-1]

    for i  in range(m):
        t = random.uniform(0,1.)
        if(axis == 'x'):
            val = (1.-t)*xmin + t*xmax
            _x.append(val)
            _y.append( ymin + (ymax-ymin)/2 )
        if(axis == 'y'):
            val = (1.-t)*ymin + t*ymax
            _y.append(val)
            _x.append( xmin + (xmax-xmin)/2 )
    # Returns the axis of sampling too
    return _x, _y, axis

# Sample a full path from startG to finishG
def sample_path_between_goals(startG,finishG,stepUnit,goalsData):
    # Sample start point
    startX, startY, axis   = uniform_sampling_1D(1, goalsData.areas[startG],  goalsData.areasAxis[startG])
    # Sample end point
    finishX, finishY, axis = uniform_sampling_1D(1, goalsData.areas[finishG], goalsData.areasAxis[finishG])
    startL = [0]
    # Number of known points
    knownN = 1
    # Prediction of the whole trajectory given the # start and finish points
    newX, newY, newL, varX, varY = prediction_to_finish_point(startX,startY,startL,knownN,[finishX[0],finishY[0]],stepUnit,startG,finishG,goalsData)
    # Number of predicted points
    nPredictions = newX.shape[0]
    # Regularization to avoid singular matrices
    varX = varX + 0.1*np.eye(newX.shape[0])
    varY = varY + 0.1*np.eye(newX.shape[0])
    # Cholesky on varX
    LX = cholesky(varX,lower=True)
    LY = cholesky(varY,lower=True)
    # Noise from a normal distribution
    sX = np.random.normal(size=(nPredictions,1))
    sY = np.random.normal(size=(nPredictions,1))
    return newX+LX.dot(sX), newY+LY.dot(sY), newL, newX, newY

# Sample a full path to finishG
def sample_path_to_goal(observedX,observedY,observedL,knownN,stepUnit,start,end,goalsData):
    # Sample end point
    finishX, finishY, axis = uniform_sampling_1D(1, goalsData.areas[end], goalsData.areasAxis[end])
    # Prediction of the whole trajectory given the # start and finish points
    newX, newY, newL, varX, varY = prediction_to_finish_point(observedX,observedY,observedL,knownN,[finishX[0], finishY[0]],stepUnit,start,end,goalsData)

    # Number of predicted points
    nPredictions = newX.shape[0]

    # Regularization to avoid singular matrices
    varX = varX + 0.1*np.eye(newX.shape[0])
    varY = varY + 0.1*np.eye(newX.shape[0])
    # Cholesky on varX
    LX = cholesky(varX,lower=True)
    LY = cholesky(varY,lower=True)
    # Noise from a normal distribution
    sX = np.random.normal(size=(nPredictions,1))
    sY = np.random.normal(size=(nPredictions,1))
    return newX+LX.dot(sX), newY+LY.dot(sY), newL, newX, newY
