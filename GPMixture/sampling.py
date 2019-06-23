# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 17:56:43 2018

@author: karenlc
"""

from GPRlib import *
from path import *
from plotting import *
from kernels import *
from testing import*
from statistics import*
import matplotlib.pyplot as plt
import numpy as np
import math
from copy import copy


def goal_sequence(L, n):
    s = []
    for i in range(n):
        s.append[i]
    return s

def sample_path(goals,startG,finishG,samplingAxis,distUnit,stepUnit,kernelX,kernelY,priorMeanX,priorMeanY):
    #sample start and finish point
    startX, startY, axis   = uniform_sampling_1D(1, goals[startG],  samplingAxis[startG])
    finishX, finishY, axis = uniform_sampling_1D(1, goals[finishG], samplingAxis[finishG])
    startL = [0]
    knownN = 1
    newX, newY, newL, varX, varY = prediction_to_finish_point_lp(startX,startY,startL,knownN,[finishX[0], finishY[0]],distUnit,stepUnit,kernelX,kernelY,priorMeanX,priorMeanY)
    
    nPredictions = newX.shape[0]
    # Regularization to avoid singular matrices
    varX = varX + 0.000001*np.eye(newX.shape[0])
    varY = varY + 0.000001*np.eye(newX.shape[0])
    # Cholesky on varX
    LX = cholesky(varX,lower=True)
    LY = cholesky(varY,lower=True)
    sX = np.random.normal(size=(nPredictions,1))
    sY = np.random.normal(size=(nPredictions,1))
    return newX+LX.dot(sX), newY+LY.dot(sY), newL, newX, newY
