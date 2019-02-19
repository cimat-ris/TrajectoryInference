# coding: utf-8
"""
Created on Mon Oct 24 00:50:28 2016

@author: karenlc
"""
from GPRlib import *
from path import *
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import inv
import string
import path
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse
from copy import copy
import sampling
    
#******************************************************************************#

# Lectura de los nombres de los archivos de datos
def readDataset(name):   
    file = open(name,'r')
    lines = file.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].strip("\n")
    return lines

#Regresa una matriz con la probabilidad de ir de g_i a g_j en cada entrada
def next_goal_probability_matrix(M, nGoals):
    probMat = []
    for i in range(nGoals):
        p = []
        n = 0.
        for j in range(nGoals):
            n += len(M[i][j])
        
        for j in range(nGoals):
            if n == 0:
                p.append(0.)
            else:
                p.append(float(len(M[i][j])/n))
        probMat.append(p)
        
    return probMat
    
def getKnownData(x,y,z,percent):
    trueX, trueY, trueZ = [],[],[]
    for j in range(len(x)):
        M = int(len(x[j])*percent)  
        if M == 0:
            return [],[],[]
        auxX, auxY, auxZ = np.zeros(M), np.zeros(M), np.zeros(M)
        for i in range(M):
            auxX[i] = x[j][i]
            auxY[i] = y[j][i]
            auxZ[i] = z[j][i]
        if len(x[j]) > 0:
            auxX[M-1] = x[j][len(x[j])-1]
            auxY[M-1] = y[j][len(y[j])-1]
            auxZ[M-1] = z[j][len(z[j])-1]

        trueX.append(auxX)
        trueY.append(auxY)
        trueZ.append(auxZ)
    
    return trueX, trueY, trueZ
         

def most_likely_goals(likelihood, nGoals):
    next_goals = []
    likely = copy(likelihood)
    for i in range(2):
        maxVal = 0
        maxInd = 0
        for j in range(nGoals):
            if likely[j] > maxVal:
                maxVal = likely[j]
                maxInd = j
        next_goals.append(maxInd)
        likely[maxInd] = 0
    return next_goals
   
#start, goals, last know (x,y,l), nextGoal
def getPredictionSet(x,y,l,start,nextGoal,goals):
    #calcula en centro del area siguiente
    dx, dy = goals[nextGoal][6]-goals[nextGoal][0], goals[nextGoal][7]-goals[nextGoal][1]
    end = [goals[nextGoal][0] + dx/2., goals[nextGoal][1] + dy/2.]
    dist = math.sqrt( (end[0]-x)**2 + (end[1]-y)**2 )
    
    steps =30#num de pasos
    step = dist/float(steps)
    newset = []
    for i in range(steps+1):
        newset.append( l + i*step )
        
    return end, newset, l + dist
    
def get_prediction_set_from_data(z,knownN):
    N = len(z)
    newZ = []
    knownN = int(knownN)
    for j in range(knownN-1, N): #numero de datos conocidos
        newZ.append(z[j])
    return newZ

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

def goal_to_subgoal_prediction_test(x,y,l,knownN,startG,finishG,goals,subgoals,unitMat,stepUnit,kernelMatX,kernelMatY,subgoalsUnitMat,subgoalsKernelMatX,subgoalsKernelMatY):
    trueX, trueY, trueL = get_known_set(x,y,l,knownN)
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(img)
    
    predictedX, predictedY, varX, varY = trajectory_prediction_test(x,y,l,knownN,startG,finishG,goals,unitMat,stepUnit,kernelMatX,kernelMatY,goalSamplingAxis) 
    #plot_prediction(img,x,y,knownN,predictedX, predictedY,varX,varY)

    subG = 2*finishG    
    predictedX_0, predictedY_0, varX_0, varY_0 = trajectory_prediction_test(x,y,l,knownN,startG,subG,subgoals,subgoalsUnitMat,stepUnit,subgoalsKernelMatX,subgoalsKernelMatY,subgoalSamplingAxis)

    subG = 2*finishG +1    
    predictedX_1, predictedY_1, varX_1, varY_1 = trajectory_prediction_test(x,y,l,knownN,startG,subG,subgoals,subgoalsUnitMat,stepUnit,subgoalsKernelMatX,subgoalsKernelMatY,subgoalSamplingAxis)    #plot_prediction(img,x,y,knownN,predictedX, predictedY,varX,varY)
    plt.plot(trueX,trueY,'c')
    
    plt.plot(predictedX,predictedY,'b')
    for j in range(len(predictedX)):
        xy = [predictedX[j],predictedY[j]]
        ell = Ellipse(xy,2.*np.sqrt(varX[j]),2.*np.sqrt(varY[j]))
        ell.set_alpha(.4)
        ell.set_lw(0)
        ell.set_facecolor('g')
        ax.add_patch(ell)    
    
    plt.plot(predictedX_0,predictedY_0,'b-.')
    for j in range(len(predictedX_0)):
        xy = [predictedX_0[j],predictedY_0[j]]
        ell = Ellipse(xy,2.*np.sqrt(varX_0[j]),2.*np.sqrt(varY_0[j]))
        ell.set_alpha(.4)
        ell.set_lw(0)
        ell.set_facecolor('y')
        ax.add_patch(ell)    
        
    plt.plot(predictedX_1,predictedY_1,'b--')
    for j in range(len(predictedX_1)):
        xy = [predictedX_1[j],predictedY_1[j]]
        ell = Ellipse(xy,2.*np.sqrt(varX_1[j]),2.*np.sqrt(varY_1[j]))
        ell.set_alpha(.4)
        ell.set_lw(0)
        ell.set_facecolor('y')
        ax.add_patch(ell) 
    
    v = [0,1920,1080,0]
    plt.axis(v)
    plt.show() 

def goal_to_4subgoal_prediction_test(x,y,l,knownN,startG,finishG,goals,subgoals,unitMat,stepUnit,kernelMatX,kernelMatY,subgoalsUnitMat,subgoalsKernelMatX,subgoalsKernelMatY):
    trueX, trueY, trueL = get_known_set(x,y,l,knownN)
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(img)
    
    predictedX, predictedY, varX, varY = trajectory_prediction_test(x,y,l,knownN,startG,finishG,goals,unitMat,stepUnit,kernelMatX,kernelMatY,goalSamplingAxis) 
    #plot_prediction(img,x,y,knownN,predictedX, predictedY,varX,varY)
    plt.plot(trueX,trueY,'c')
    subG = 4*finishG    
    predictedX_0, predictedY_0, varX_0, varY_0 = trajectory_prediction_test(x,y,l,knownN,startG,subG,subgoals,subgoalsUnitMat,stepUnit,subgoalsKernelMatX,subgoalsKernelMatY,subgoalSamplingAxis)

    subG = 4*finishG +1    
    predictedX_1, predictedY_1, varX_1, varY_1 = trajectory_prediction_test(x,y,l,knownN,startG,subG,subgoals,subgoalsUnitMat,stepUnit,subgoalsKernelMatX,subgoalsKernelMatY,subgoalSamplingAxis)    #plot_prediction(img,x,y,knownN,predictedX, predictedY,varX,varY)
    
    subG = 4*finishG +2  
    predictedX_2, predictedY_2, varX_2, varY_2 = trajectory_prediction_test(x,y,l,knownN,startG,subG,subgoals,subgoalsUnitMat,stepUnit,subgoalsKernelMatX,subgoalsKernelMatY,subgoalSamplingAxis)

    subG = 4*finishG +3    
    predictedX_3, predictedY_3, varX_3, varY_3 = trajectory_prediction_test(x,y,l,knownN,startG,subG,subgoals,subgoalsUnitMat,stepUnit,subgoalsKernelMatX,subgoalsKernelMatY,subgoalSamplingAxis)    #plot_prediction(img,x,y,knownN,predictedX, predictedY,varX,varY)
    
    plt.plot(predictedX,predictedY,'b')
    for j in range(len(predictedX)):
        xy = [predictedX[j],predictedY[j]]
        ell = Ellipse(xy,2.*np.sqrt(varX[j]),2.*np.sqrt(varY[j]))
        ell.set_alpha(.4)
        ell.set_lw(0)
        ell.set_facecolor('g')
        ax.add_patch(ell)    
    
    plt.plot(predictedX_0,predictedY_0,'r--')
    for j in range(len(predictedX_0)):
        xy = [predictedX_0[j],predictedY_0[j]]
        ell = Ellipse(xy,2.*np.sqrt(varX_0[j]),2.*np.sqrt(varY_0[j]))
        ell.set_alpha(.4)
        ell.set_lw(0)
        ell.set_facecolor('m')
        ax.add_patch(ell)    
        
    plt.plot(predictedX_1,predictedY_1,'r--')
    for j in range(len(predictedX_1)):
        xy = [predictedX_1[j],predictedY_1[j]]
        ell = Ellipse(xy,2.*np.sqrt(varX_1[j]),2.*np.sqrt(varY_1[j]))
        ell.set_alpha(.4)
        ell.set_lw(0)
        ell.set_facecolor('m')
        ax.add_patch(ell) 
        
    plt.plot(predictedX_2,predictedY_2,'r--')
    for j in range(len(predictedX_2)):
        xy = [predictedX_2[j],predictedY_2[j]]
        ell = Ellipse(xy,2.*np.sqrt(varX_2[j]),2.*np.sqrt(varY_2[j]))
        ell.set_alpha(.4)
        ell.set_lw(0)
        ell.set_facecolor('m')
        ax.add_patch(ell)    
        
    plt.plot(predictedX_3,predictedY_3,'r--')
    for j in range(len(predictedX_3)):
        xy = [predictedX_3[j],predictedY_3[j]]
        ell = Ellipse(xy,2.*np.sqrt(varX_3[j]),2.*np.sqrt(varY_3[j]))
        ell.set_alpha(.4)
        ell.set_lw(0)
        ell.set_facecolor('m')
        ax.add_patch(ell)
    
    v = [0,1920,1080,0]
    plt.axis(v)
    plt.show() 

def single_goal_prediction_test(x,y,l,knownN,startG,finishG,goals,unitMat,stepUnit,kernelMatX,kernelMatY,goalSamplingAxis): 
    predictedX, predictedY, varX, varY = trajectory_prediction_test(x,y,l,knownN,startG,finishG,goals,unitMat,stepUnit,kernelMatX,kernelMatY,goalSamplingAxis) 
    lenX = goals[finishG][len(goals[finishG]) -2] - goals[finishG][0]
    lenY = goals[finishG][len(goals[finishG]) -1] - goals[finishG][1]
    elipse = [lenX, lenY]
    
    plot_prediction(img,x,y,knownN,predictedX, predictedY,varX,varY,elipse)
    
#recibe: datos conocidos, valores por predecir, areas de inicio y final
def prediction_test(x,y,z,knownN,startG,finishG,goals,unitMat,meanLenMat,steps):
    kernelX = kernelMat_x[startG][finishG]
    kernelY = kernelMat_y[startG][finishG]
    
    trueX, trueY, trueZ = get_known_set(x,y,z,knownN)
    lastKnownPoint = [x[knownN-1], y[knownN-1], z[knownN-1] ]
    unit = unitMat[startG][finishG]
    meanLen = meanLenMat[startG][finishG]
    
    final_xy = get_finish_point(trueX,trueY,trueZ,finishG,goals,kernelX,kernelY,unit,goalSamplingAxis)
    #final_xy = get_finish_point_singleGP(trueX,trueY,trueZ,finishG,goals,kernelX,kernelY,unit,img)
    newZ, final_z = get_prediction_set(lastKnownPoint,final_xy,unit,steps)  
    trueX.append(final_xy[0])    
    trueY.append(final_xy[1])
    trueZ.append(final_z)
    
    newX,newY,varX,varY = prediction_XY(trueX,trueY,trueZ,newZ,kernelX,kernelY) 
    plot_prediction(img,x,y,knownN,newX,newY,varX,varY)
    
#recibe: datos reales, num de datos conocidos, area de inicio, vec de areas objetivo, vec de areas 
def _multigoal_prediction_test(x,y,z,knownN,startG,finishG,goals,unitMat,steps):
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
    
def get_error_of_final_point_comparing_k_steps(k,trajectorySet,startG,finishG,goals,unitMat,meanLenMat):
    kernelX = kernelMat_x[startG][finishG]
    kernelY = kernelMat_y[startG][finishG]

    meanError = 0.0    
    for i in range( len(trajectorySet) ):
        x = trajectorySet[i].x
        y = trajectorySet[i].y
        l = trajectorySet[i].l
        final = [x[len(x)-1], y[len(y)-1]]
        knownN = len(x)* (0.9)
        trueX, trueY, trueZ = get_known_set(x,y,l,knownN)
        #lastKnownPoint = [x[knownN-1], y[knownN-1], l[knownN-1] ]
        unit = unitMat[startG][finishG]
        
        predicted_final = get_finish_point(k,trueX,trueY,trueZ,finishG,goals,kernelX,kernelY,unit,img,goalSamplingAxis)
        error = final_displacement_error(final,predicted_final)
        #print("FDE: ",error)
        meanError += error
    meanError /= len(trajectorySet)
    return meanError
    
def choose_number_of_steps_and_samples(trajectorySet,startG,finishG,goals,unitMat,meanLenMat):
    errorVec = []    
    stepsVec = []
    samplesVec = []
    steps = 1
    samples = 1
    for i in range(20):
        samplesVec.append(samples)
        error = get_error_of_final_point_comparing_k_steps(samples,trajectorySet,startG,finishG,goals,unitMat,meanLenMat)
        errorVec.append(error)
        samples += 1
        
    print("error: ", errorVec)
    plt.plot(samplesVec, errorVec)
    plt.xlabel('size of sample')
    plt.ylabel('mean FDE')
    #para ambos el error sera FDE
    #en 2 te detienes cuando el error deje de disminuir significativamente
    
def get_number_of_steps_unit(Mat, nGoals):
    unit = 0.0
    numUnits = 0
    for i in range(nGoals):
        for j in range(nGoals):
            numPaths = len(Mat[i][j])
            meanU = 0.0
            for k in range(numPaths):
                path = Mat[i][j][k]
                l = path.l[len(path.l)-1]
                if(l == 0):
                    numPaths -= 1
                else: 
                    stps = len(path.l)
                    u = stps/l
                    meanU += u
            if(numPaths > 0):
                meanU = meanU/numPaths
            if(meanU >0):
                unit += meanU
                numUnits += 1
    unit = unit/numUnits
    #print("mean mean unit:", unit)
    return unit
    
def copy_unitMat(unitMat, nGoals, nSubgoals):
    mat = []
    m = int(nSubgoals/nGoals)
    for i in range(nGoals):
        r = []
        for j in range(nSubgoals):
            k = int(j/m)
            r.append(unitMat[i][k])
        mat.append(r)
    return mat
        
def goal_to_subgoal_prediction_error(x,y,l,knownN,startG,finishG,goals,subgoals,unitMat,stepUnit,kernelMatX,kernelMatY,subgoalsUnitMat,subgoalsKernelMatX,subgoalsKernelMatY):
    trueX, trueY, trueL = get_known_set(x,y,l,knownN)
    predictedX, predictedY, varX, varY = trajectory_prediction_test(x,y,l,knownN,startG,finishG,goals,unitMat,stepUnit,kernelMatX,kernelMatY) 

    subG = 2*finishG    
    predictedX_0, predictedY_0, varX_0, varY_0 = trajectory_prediction_test(x,y,l,knownN,startG,subG,subgoals,subgoalsUnitMat,stepUnit,subgoalsKernelMatX,subgoalsKernelMatY)

    subG = 2*finishG +1    
    predictedX_1, predictedY_1, varX_1, varY_1 = trajectory_prediction_test(x,y,l,knownN,startG,subG,subgoals,subgoalsUnitMat,stepUnit,subgoalsKernelMatX,subgoalsKernelMatY)    #plot_prediction(img,x,y,knownN,predictedX, predictedY,varX,varY)
    #errores
    N = len(x)
    realX, realY = [], []
    for i in range(knownN,N):
        realX.append(x[i])
        realY.append(y[i])
    #print("longitudes de vec:",len(realX), len(predictedX))
    error = average_displacement_error([realX,realY],[predictedX,predictedY])    
    error0 = average_displacement_error([realX,realY],[predictedX_0,predictedY_0])
    error1 = average_displacement_error([realX,realY],[predictedX_1,predictedY_1])
    return error, error0, error1
    
def goal_to_4subgoal_prediction_error(x,y,l,knownN,startG,finishG,goals,subgoals,unitMat,stepUnit,kernelMatX,kernelMatY,subgoalsUnitMat,subgoalsKernelMatX,subgoalsKernelMatY):
    trueX, trueY, trueL = get_known_set(x,y,l,knownN)
    predictedX, predictedY, varX, varY = trajectory_prediction_test(x,y,l,knownN,startG,finishG,goals,unitMat,stepUnit,kernelMatX,kernelMatY,goalSamplingAxis) 

    subG = 4*finishG    
    predictedX_0, predictedY_0, varX_0, varY_0 = trajectory_prediction_test(x,y,l,knownN,startG,subG,subgoals,subgoalsUnitMat,stepUnit,subgoalsKernelMatX,subgoalsKernelMatY,subgoalSamplingAxis)

    subG = 4*finishG +1    
    predictedX_1, predictedY_1, varX_1, varY_1 = trajectory_prediction_test(x,y,l,knownN,startG,subG,subgoals,subgoalsUnitMat,stepUnit,subgoalsKernelMatX,subgoalsKernelMatY,subgoalSamplingAxis)    #plot_prediction(img,x,y,knownN,predictedX, predictedY,varX,varY)
    
    subG = 4*finishG +2  
    predictedX_2, predictedY_2, varX_2, varY_2 = trajectory_prediction_test(x,y,l,knownN,startG,subG,subgoals,subgoalsUnitMat,stepUnit,subgoalsKernelMatX,subgoalsKernelMatY,subgoalSamplingAxis)

    subG = 4*finishG +3    
    predictedX_3, predictedY_3, varX_3, varY_3 = trajectory_prediction_test(x,y,l,knownN,startG,subG,subgoals,subgoalsUnitMat,stepUnit,subgoalsKernelMatX,subgoalsKernelMatY,subgoalSamplingAxis)    #plot_prediction(img,x,y,knownN,predictedX, predictedY,varX,varY)
    
    #errores
    N = len(x)
    realX, realY = [], []
    for i in range(knownN,N):
        realX.append(x[i])
        realY.append(y[i])
    #print("longitudes de vec:",len(realX), len(predictedX))
    error = average_displacement_error([realX,realY],[predictedX,predictedY])    
    error0 = average_displacement_error([realX,realY],[predictedX_0,predictedY_0])
    error1 = average_displacement_error([realX,realY],[predictedX_1,predictedY_1])
    error2 = average_displacement_error([realX,realY],[predictedX_2,predictedY_2])
    error3 = average_displacement_error([realX,realY],[predictedX_3,predictedY_3])
    return error, error0, error1, error2, error3
    
def test_prediction_goal_to_subgoal(trajectorySet,startG,finishG,goals,subgoals,unitMat,stepUnit,kernelMatX,kernelMatY,subgoalsUnitMat,subgoalsKernelMatX,subgoalsKernelMatY):
    vectorErrorG, vectorErrorSG0, vectorErrorSG1, vectorErrorSG2, vectorErrorSG3 = [],[],[],[],[]
    num = len(trajectorySet)
    partNum = 10
    part = []
    for i in range(partNum-1):
        part.append(i+1)
        errorG, errorSG0, errorSG1, errorSG2, errorSG3 = 0., 0., 0., 0., 0.
        for j in range(num): 
            x, y, l = trajectorySet[j].x, trajectorySet[j].y, trajectorySet[j].l
            knownN = int((i+1)*(len(x)/partNum))
            e,e0,e1,e2,e3 = goal_to_4subgoal_prediction_error(x,y,l,knownN,startG,finishG,goals,subgoals,unitMat,stepUnit,kernelMatX,kernelMatY,subgoalsUnitMat,subgoalsKernelMatX,subgoalsKernelMatY)    
            errorG += e
            errorSG0 += e0
            errorSG1 += e1
            errorSG2 += e2
            errorSG3 += e3
        vectorErrorG.append(errorG/num)
        vectorErrorSG0.append(errorSG0/num)
        vectorErrorSG1.append(errorSG1/num)
        vectorErrorSG2.append(errorSG2/num)
        vectorErrorSG3.append(errorSG3/num)
    print("error_Goal = ",vectorErrorG)
    print("error_subGoal0 = ",vectorErrorSG0)
    print("error_subGoal1 = ",vectorErrorSG1)
    print("error_subGoal2 = ",vectorErrorSG2)
    print("error_subGoal3 = ",vectorErrorSG3)
    """    
    plt.plot(part,vectorErrorG,'r')
    plt.plot(part,vectorErrorSG0,'m')
    plt.plot(part,vectorErrorSG1,'m')
    plt.plot(part,vectorErrorSG2,'m')
    plt.plot(part,vectorErrorSG3,'m')
    """

"""******************************************************************************"""
# Areas de interes [x1,y1,x2,y2,...]
#R0 = [400,40,680,40,400,230,680,230] #azul
#R1 = [1110,40,1400,40,1110,230,1400,230] #cian
R0 = [400,10,680,10,400,150,680,150] #azul
R1 = [1100,10,1400,10,1100,150,1400,150] #cian
R2 = [1650,490,1810,490,1650,740,1810,740] #verde
R3 = [1450,950,1800,950,1450,1080,1800,1080]#amarillo
R4 = [100,950,500,950,100,1080,500,1080] #naranja
R5 = [300,210,450,210,300,400,450,400] #rosa
goalSamplingAxis = ['x','x','y','x','x','y']
#R6 = [750,460,1050,460,750,730,1050,730] #rojo
"""***Subgoals***"""
"""
r00 = [400,10,540,10,400,130,540,130]
r01 = [540,10,680,10,540,130,680,130]
r10 = [1100,10,1250,10,1100,130,1250,130]
r11 = [1250,10,1400,10,1250,130,1400,130]
r20 = [1680,490,1800,490,1680,615,1800,615]
r21 = [1680,615,1800,615,1680,740,1800,740]
r30 = [1450,950,1625,950,1450,1100,1625,1100]
r31 = [1625,950,1800,950,1625,1100,1800,1100]
r40 = [100,950,300,950,100,1100,300,1100]
r41 = [300,950,500,950,300,1100,500,1100]
r50 = [320,190,430,190,320,305,430,305]
r51 = [320,305,430,305,320,420,430,420]
"""
r00 = [400,10,470,10,400,150,470,150]
r01 = [470,10,540,10,470,150,540,150]
r02 = [540,10,610,10,540,150,610,150]
r03 = [610,10,680,10,610,150,680,150]

r10 = [1100,10,1175,10,1100,150,1175,150]
r11 = [1175,10,1250,10,1175,150,1250,150]
r12 = [1250,10,1325,10,1250,150,1325,150]
r13 = [1325,10,1400,10,1325,150,1400,150]

r20 = [1650,490,1810,490,1650,527,1810,527]
r21 = [1650,527,1810,527,1650,564,1810,564]
r22 = [1650,564,1810,564,1650,601,1810,601]
r23 = [1650,601,1810,601,1650,740,1810,740]

r30 = [1450,950,1537,950,1450,1100,1537,1100]
r31 = [1537,950,1624,950,1537,1100,1624,1100]
r32 = [1624,950,1711,950,1624,1100,1711,1100]
r33 = [1711,950,1800,950,1711,1100,1800,1100]

r40 = [100,950,200,950,100,1100,200,1100]
r41 = [200,950,300,950,200,1100,300,1100]
r42 = [300,950,400,950,300,1100,400,1100]
r43 = [400,950,500,950,400,1100,500,1100]

r50 = [300,210,450,210,300,257,450,257]
r51 = [300,257,450,257,300,304,450,304]
r52 = [300,304,450,304,300,351,450,351]
r53 = [300,351,450,351,300,400,450,400]
#Arreglo que contiene las areas de interes
areas = [R0,R1,R2,R3,R4,R5]
nGoals = len(areas)
subgoals =[r00,r01,r02,r03, r10,r11,r12,r13, r20,r21,r22,r23, r30,r31,r32,r33, r40,r41,r42,r43, r50,r51,r52,r53] #[r00,r01,r10,r11,r20,r21,r30,r31,r40,r41,r50,r51]
nSubgoals = len(subgoals)
subgoalSamplingAxis = []
for i in range(nGoals):
    for j in range(4):
        subgoalSamplingAxis.append(goalSamplingAxis[i])
#print("subG sampling axis:", subgoalSamplingAxis)
img = mpimg.imread('imgs/goals.jpg')  

#Al leer cortamos las trayectorias multiobjetivos por pares consecutivos 
#y las agregamos como trayectorias independientes 
true_paths, multigoal = get_paths_from_file('datasets/CentralStation_paths_10000.txt',areas)
usefulPaths = path.getUsefulPaths(true_paths,areas)
#path.plotPaths(usefulPaths, img)
#print("useful paths: ",len(usefulPaths))
#Histograma    
#histogram(true_paths,"duration")
"""
Matrices útiles:
- pathMat: Quita las trayectorias que se alejan de la media del conjunto que va de g_i a g_j
- meanLenMat: Guarda en ij el arc-len promedio de los caminos de g_i a g_j
- euclideanDistMat: Guarda en ij la distancia euclidiana del goal g_i al g_j
- unitMat: Guarda en ij la unidad de distancia para los caminos de g_i a g_j
"""
#Separamos las trayectorias en pares de goals
startToGoalPath, arclenMat = define_trajectories_start_and_end_areas(areas,areas,usefulPaths)#true_paths)#learnSet)#
#Quitamos las trayectorias demasiado largas o demasiado cortas
pathMat, learnSet = filter_path_matrix(startToGoalPath, nGoals, nGoals)
meanLenMat = get_mean_length(pathMat, nGoals)
euclideanDistMat = get_euclidean_goal_distance(areas, nGoals)
unitMat = get_unit(meanLenMat, euclideanDistMat, nGoals)
stepUnit = 0.0438780780171 #get_number_of_steps_unit(pathMat, nGoals)
#print("***arc-len promedio***\n", meanLenMat)
#print("***distancia euclidiana entre goals***\n", euclideanDistMat)
#print("***unidades de distancia***\n", unitMat)
priorLikelihoodMat = next_goal_probability_matrix(pathMat, nGoals)
#print("likelihood mat:", priorLikelihoodMat)

"""
#***********APRENDIZAJE***********
print("***********INICIO APRENDIZAJE*********")
kernelMat, parametersMat = create_kernel_matrix("combined", nGoals, nGoals)
kernelMat_x, kernelMat_y = optimize_parameters_between_goals(pathMat, parametersMat, nGoals, nGoals)
write_parameters(kernelMat_x,nGoals,nGoals,"parameters_x.txt")
write_parameters(kernelMat_y,nGoals,nGoals,"parameters_y.txt")
print("***********FIN DEL APRENDIZAJE*********")
"""
#fijamos los parámetros para cada matriz de kernel
kernelMat_x = read_and_set_parameters("parameters_x.txt",nGoals,nGoals,2)
kernelMat_y = read_and_set_parameters("parameters_y.txt",nGoals,nGoals,2)

"""***********TRAYECTORIAS SUBGOALS***********"""
subgoalStartToGoalPath, subgoalsArclenMat = define_trajectories_start_and_end_areas(areas,subgoals,usefulPaths)
subgoalsPathMat, sublearnSet = filter_path_matrix(subgoalStartToGoalPath, nGoals, nSubgoals)
"""
#***********APRENDIZAJE SUBGOALS***********
print("***********INICIO APRENDIZAJE*********")
subgoalsKernelMat, subgoalsParametersMat = create_kernel_matrix("combined", nGoals, nSubgoals)
subgoalsKernelMatX, subgoalsKernelMatY = optimize_parameters_between_goals(subgoalsPathMat, subgoalsParametersMat, nGoals, nSubgoals)
write_parameters(subgoalsKernelMatX,nGoals,nSubgoals,"subgoalsParameters_x.txt")
write_parameters(subgoalsKernelMatY,nGoals,nSubgoals,"subgoalsParameters_y.txt")
print("***********FIN DEL APRENDIZAJE*********")
"""
#fijamos los parámetros para cada matriz de kernel
subgoalsKernelMat_x = read_and_set_parameters("subgoalsParameters_x.txt",nGoals,nSubgoals,2)
subgoalsKernelMat_y = read_and_set_parameters("subgoalsParameters_y.txt",nGoals,nSubgoals,2)
subgoalsUnitMat = copy_unitMat(unitMat, nGoals, nSubgoals); 

startG = 0
nextG = 2#getNextKGoals(start[0], 2, priorLikelihoodMat, nGoals)
kernelid = nextG
kernelX = kernelMat_x[startG][kernelid]
kernelY = kernelMat_y[startG][kernelid]

traj_id = 0
traj = pathMat[startG][nextG][traj_id]
traj_len = len(pathMat[startG][nextG][traj_id].x)
traj_arclen = pathMat[startG][nextG][traj_id].length
likelihoodVector, error_vector = [], []
arclen_vec = []

part_num = 4
steps = 15

for i in range(1,part_num-1):
    arclen_vec.append( (i+1)*(traj_arclen/float(part_num))  )
    knownN = int((i+1)*(traj_len/part_num)) 
    
    trueX,trueY,trueL = get_known_set(traj.x,traj.y,traj.l,knownN)
    likelihood, error = get_goals_likelihood(trueX,trueY,trueL,startG,kernelMat_x,kernelMat_x,areas,nGoals)
    likelihoodVector.append(likelihood)  
    error_vector.append(error)
    likely_goals = most_likely_goals(likelihood, nGoals)
    #print("likely goals:", likely_goals)
    """Simple prediction test"""
    single_goal_prediction_test(traj.x,traj.y,traj.l,knownN,startG,nextG,areas,unitMat,stepUnit,kernelMat_x,kernelMat_y,goalSamplingAxis)
    #goal_to_4subgoal_prediction_test(traj.x,traj.y,traj.l,knownN,startG,nextG,areas,subgoals,unitMat,stepUnit,kernelMat_x,kernelMat_y,subgoalsUnitMat,subgoalsKernelMat_x,subgoalsKernelMat_y)    
    #trajectory_subgoal_prediction_test(img,traj.x,traj.y,traj.l,knownN,startG,nextG,areas,unitMat,stepUnit,kernelMat_x,kernelMat_y,goalSamplingAxis)
    #prediction_test(traj.x,traj.y,traj.l,knownN,startG,nextG,areas,unitMat,meanLenMat,steps)
    """Multigoal prediction test"""    
    #multigoal_prediction_test(traj.x,traj.y,traj.l,knownN,startG,areas,unitMat,stepUnit,kernelMat_x,kernelMat_y,priorLikelihoodMat)
    #prediction_test_over_time(traj.x,traj.y,traj.t,knownN,start[0],nextG[0],areas)
#path.plotPaths([pathMat[startG][nextG][traj_id]], img) 

for i in range(0):#1,nGoals):
    for j in range(nGoals):
        startG, finishG = i,j
        _trajectorySet = pathMat[startG][finishG]
        if(len(_trajectorySet) > 0):  
            trajectorySet = []
            _min = min(100, len(_trajectorySet))
            for k in range(_min):
                trajectorySet.append(_trajectorySet[k])
            print("[",i,",",j,"]")
            test_prediction_goal_to_subgoal(trajectorySet,startG,finishG,areas,subgoals,unitMat,stepUnit,kernelMat_x,kernelMat_y,subgoalsUnitMat,subgoalsKernelMat_x,subgoalsKernelMat_y)


"""
error_Goal =  [233.92540827022103, 236.12565406391835, 234.02639307026723, 215.47835069302988, 204.97275030071614, 181.1237882895093, 140.4263525726929, 102.34102601529557, 61.69507321046843]
error_subGoal0 =  [289.7192562613634, 279.71591288230525, 276.79269370108597, 269.95134946061063, 250.56488010858695, 242.6371546582687, 245.3735067348365, 237.75133551771555, 196.51826753419869]
error_subGoal1 =  [296.07629831653713, 288.63991651898624, 289.82654030081176, 285.80657209114804, 272.16341883168536, 267.230442785437, 268.8398198407697, 254.36016606677762, 204.9571224243962]


v = []
for i in range(9):
    v.append(i+1)
plt.plot(v,error_Goal,'r')
plt.plot(v,error_subGoal0,'c')
plt.plot(v,error_subGoal1,'b')
plt.xlabel('porcentaje de datos observados')
plt.ylabel('error')
#test para encontrar los valores de k y m al elegir el punto final
"""

error_Goal =  [106.1826703077137, 89.9380132628228, 77.56520966201825, 80.84045365187663, 90.25903660689445, 100.19431477465076, 106.70648552550497, 106.02383981973837, 87.6704447818017]
error_subGoal0 =  [114.48804568018654, 88.92841794481858, 80.35720220183902, 83.40378458085559, 91.39313861250328, 102.43655653295193, 111.73429047914964, 112.36662946714574, 95.71579728359804]
error_subGoal1 =  [106.28548018132851, 83.89076845911113, 76.4673625404256, 79.8881631853389, 89.16757754188646, 99.34554299393301, 106.9944763444792, 105.80520437192527, 88.47254542985269]
error_subGoal2 =  [108.7542886447821, 88.39929695896268, 80.19083956986691, 83.50001575830358, 92.36492304978556, 102.41186886551584, 108.84666945325397, 108.25085328253934, 90.88727071842061]
error_subGoal3 =  [134.49510793648676, 109.11341996118945, 96.47123811690778, 92.84995554203122, 99.21081611450764, 107.36004308001398, 114.20003288721459, 111.87407677177262, 94.4693826622927]

v = []
for i in range(9):
    v.append(i+1)
"""
plt.plot(v,error_Goal,'r')
plt.plot(v,error_subGoal0,'c')
plt.plot(v,error_subGoal1,'c')
plt.plot(v,error_subGoal2,'c')
plt.plot(v,error_subGoal3,'c')
plt.xlabel('porcentaje de datos observados')
plt.ylabel('error')
"""
"""
trajectorySet = []
startGoal, finishGoal = 4,0
#x_, y_, flag = uniform_sampling_1D(15, areas[finishGoal])
#print("_x=",x_)
#print("_y=",y_)
#path.plotPaths(pathMat[startGoal][finishGoal], img) 
numOfPaths = len(pathMat[startGoal][finishGoal])
lenTrainingSet = int(numOfPaths/2)
if(lenTrainingSet > 50):
    lenTrainingSet = 50
#print( len(pathMat[startGoal][finalGoal]) )
for i in range(lenTrainingSet):
    trajectorySet.append(pathMat[startGoal][finishGoal][i])
#choose_number_of_steps_and_samples(trajectorySet,startGoal,finishGoal,areas,unitMat,meanLenMat)

kVec = []
mVec = []
for i in range(20):
    mVec.append(i+1)
    kVec.append(i+1)
"""
 
 
 
 
 
 
 
 
 