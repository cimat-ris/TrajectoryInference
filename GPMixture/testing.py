# -*- coding: utf-8 -*-
"""
Testing functions

@author: karenlc
"""

from GPRlib import *
from path import *
from plotting import *
from kernels import *
import matplotlib.pyplot as plt
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
        
    newX,newY,varX,varY = prediction_XY(trueX,trueY,trueL,newL,kernelX,kernelY) 
    return newX, newY, varX, varY
    

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
    
    elipse = size
    plot_subgoal_prediction(img,trueX,trueY,knownN,nSubgoals,predictedXYVec,varXYVec,elipse)

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
    
        goalPredX, goalPredY, goalVarX, goalVarY = trajectory_prediction_test(x,y,l,knownN,startG,finishG,goals,unitMat,stepUnit,kernelMatX,kernelMatY)
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
    
def multigoal_prediction_test(x,y,l,knownN,startG,goals,unitMat,stepUnit,kernelMatX,kernelMatY,priorLikelihood):
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(img)
    
    trueX, trueY, trueL = get_known_set(x,y,l,knownN) 
    likelyGoals = []    
    goalsLikelihood = [] 
    errorG = []
    for i in range(len(goals)):
        error = get_goal_likelihood(trueX,trueY,trueL,startG,i,goals,unitMat,kernelMatX,kernelMatY)
        errorG.append(error)
        val = priorLikelihood[startG][i]*(1./error)
        if(val > 0):
            likelyGoals.append(i)        
        goalsLikelihood.append(val)
    
    for i in range(len(likelyGoals)):
        nextGoal = likelyGoals[i]
        predictedX, predictedY, varX, varY = trajectory_prediction_test(x,y,l,knownN,startG,nextGoal,goals,unitMat,stepUnit,kernelMatX,kernelMatY)
        trueX, trueY, trueL = get_known_set(x,y,l,knownN)
        
        linewidth = 1500*goalsLikelihood[nextGoal]
        plt.plot(trueX,trueY,'c',predictedX,predictedY,'b', lw= linewidth)
        #elipses
        for j in range(len(predictedX)):
            xy = [predictedX[j],predictedY[j]]
            ell = Ellipse(xy,2.*np.sqrt(varX[j]),2.*np.sqrt(varY[j]))
            ell.set_alpha(.4)
            ell.set_lw(0)
            ell.set_facecolor('g')
            ax.add_patch(ell)
        
    v = [0,1920,1080,0]
    plt.axis(v)
    plt.show() 
    
def goal_weight_test(knownX,knownY,knownL,knownN,startG,goals,unitMat,stepUnit,kernelMatX,kernelMatY):
    nPoints = 4
    maxError = 0.
    errorVec = []
    for i in range(len(goals)):
        #print("[",startG,",",i,"]")
        unit = unitMat[startG][i]
        kernelX = kernelMatX[startG][i]
        kernelY = kernelMatY[startG][i]
        error = prediction_error_of_last_known_points(nPoints,knownX,knownY,knownL,goals[i],unit,stepUnit,kernelX,kernelY)
        errorVec.append(error)
        if(error > maxError):
            maxError = error
        
    for i in range(len(goals)):
        errorVec[i] = errorVec[i]/maxError
    
    print("[Error vec]:\n",errorVec)
    
    
    
    
    
    
    
    