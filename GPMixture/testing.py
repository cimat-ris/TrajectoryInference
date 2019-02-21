# -*- coding: utf-8 -*-
"""
Testing functions

@author: karenlc
"""

from GPRlib import *
from path import *
import numpy as np
import math
import GPRlib
from copy import copy

    
def trajectory_prediction_test_using_sampling(x,y,l,knownN,startG,finishG,goals,unitMat,stepUnit,kernelMatX,kernelMatY,samplingAxis):
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
    return newX, newY, varX, varY

def trajectory_prediction_test(x,y,l,knownN,startG,finishG,goals,unitMat,stepUnit,kernelMatX,kernelMatY):
    kernelX = kernelMatX[startG][finishG]
    kernelY = kernelMatY[startG][finishG]
    
    trueX, trueY, trueL = get_known_set(x,y,l,knownN)
    lastKnownPoint = [x[knownN-1], y[knownN-1], l[knownN-1] ]
    unit = unitMat[startG][finishG]
    
    finalPoints = get_goal_center_and_boundaries(goals[finishG])
    
    newL, finalL = get_prediction_set(lastKnownPoint,finalPoints[0],unit,stepUnit)
    finalArcLen = []
    
    for i in range(1):#len(finalPoints)):
        finalArcLen.append(get_arclen_to_finish_point(lastKnownPoint,finalPoints[i],unit))
        trueX.append(finalPoints[i][0])    
        trueY.append(finalPoints[i][1])
        trueL.append(finalArcLen[i])
        
    #print("[final points]:",finalPoints)
    #print("[final arclen]:",finalArcLen)
    newX,newY,varX,varY = prediction_XY(trueX,trueY,trueL,newL,kernelX,kernelY) 
    return newX, newY, varX, varY
    

def subgoal_prediction(x,y,l,knownN,subgoal,unit,stepUnit,kernelX,kernelY):
    trueX, trueY, trueL = get_known_set(x,y,l,knownN)
    lastKnownPoint = [x[knownN-1], y[knownN-1], l[knownN-1] ]
    
    finalPoints = get_goal_center_and_boundaries(subgoal)
    
    newL, finalL = get_prediction_set(lastKnownPoint,finalPoints[0],unit,stepUnit)
    finalArcLen = []
    
    for i in range(len(finalPoints)):
        finalArcLen.append(get_arclen_to_finish_point(lastKnownPoint,finalPoints[i],unit))
        trueX.append(finalPoints[i][0])    
        trueY.append(finalPoints[i][1])
        trueL.append(finalArcLen[i])
        
    #print("[final points]:",finalPoints)
    #print("[final arclen]:",finalArcLen)
    newX,newY,varX,varY = prediction_XY(trueX,trueY,trueL,newL,kernelX,kernelY) 
    return newX, newY, varX, varY


def trajectory_subgoal_prediction_test(img,x,y,l,knownN,startG,finishG,goals,unitMat,stepUnit,kernelMatX,kernelMatY,samplingAxis):
    kernelX = kernelMatX[startG][finishG]
    kernelY = kernelMatY[startG][finishG]
    
    trueX, trueY, trueL = get_known_set(x,y,l,knownN)
    lastKnownPoint = [x[knownN-1], y[knownN-1], l[knownN-1] ]
    unit = unitMat[startG][finishG]
    #goalCenter = middle_of_area(goals[finishG])
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
    
    """
    sampleX, sampleY, axis = uniform_sampling_1D(nSubgoals,goals[finishG],samplingAxis[finishG])
    subgoalsXY = []
    for i in range(nSubgoals):
        subgoalsXY.append([sampleX[i], sampleY[i]])
    
    predictedXYVec, varXYVec = [],[]
    for i in range(nSubgoals):
        newL, finalL = get_prediction_set(lastKnownPoint,subgoalsXY[i],unit,stepUnit)
        trueX.append(subgoalsXY[i][0])    
        trueY.append(subgoalsXY[i][1])
        trueL.append(finalL)
        newX,newY,varX,varY = prediction_XY(trueX,trueY,trueL,newL,kernelX,kernelY) 
        predictedXYVec.append([newX,newY])
        varXYVec.append([varX,varY])
        trueX.pop()
        trueY.pop()
        trueL.pop()
        
    subgoalElipseX, subgoalElipseY = 0,0
    if samplingAxis[finishG] == 'x':
        subgoalElipseX = (goals[finishG][len(goals[finishG]) -2] - goals[finishG][0])/nSubgoals 
        subgoalElipseY = (goals[finishG][len(goals[finishG]) -1] - goals[finishG][1])/2
    
    if samplingAxis[finishG] == 'y':
        subgoalElipseX = (goals[finishG][len(goals[finishG]) -2] - goals[finishG][0])/2 
        subgoalElipseY = (goals[finishG][len(goals[finishG]) -1] - goals[finishG][1])/nSubgoals
    """
    
    elipse = size#[subgoalElipseX,subgoalElipseY]
    print("[Elipse]:", elipse)
    plot_subgoal_prediction(img,trueX,trueY,knownN,nSubgoals,predictedXYVec,varXYVec,elipse)

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