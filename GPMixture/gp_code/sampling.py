# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 17:56:43 2018
@author: karenlc
"""
import numpy as np
import math

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
        t = np.random.uniform(0,1.)
        val = (1.-t)*xmin + t*xmax
        _x.append(val)
        r = np.random.uniform(0,1.)
        val = (1.-r)*ymin + r*ymax
        _y.append(val)

    return _x, _y

# Sample m points (x,y) along a line segment, with uniform sampling
def uniform_sampling_1D(m, goal, axis):
    _x, _y = [], []
    xmin, xmax = goal[0], goal[2]
    ymin, ymax = goal[1], goal[len(goal)-1]
    for i  in range(m):
        t = np.random.uniform(0,1.)
        if(axis == 0):
            val = (1.-t)*xmin + t*xmax
            _x.append(val)
            _y.append( (ymax+ymin)/2.0 )
        if(axis == 1):
            val = (1.-t)*ymin + t*ymax
            _y.append(val)
            _x.append((xmax+xmin)/2.0 )
    # Returns the axis of sampling too
    return _x, _y, axis

# Sample m points (x,y) along a line segment centered on point, with uniform sampling
def uniform_sampling_1D_around_point(m, point, size, axis):
    _x, _y = [], []
    for i  in range(m):
        t = np.random.uniform(0,size)
        if(axis == 0):
            _x.append(point[0]-size/2.0+t)
            _y.append(point[1])
        else:
            _y.append(point[1]-size/2.0+t)
            _x.append(point[0])
    # Returns the axis of sampling too
    return _x, _y, axis
