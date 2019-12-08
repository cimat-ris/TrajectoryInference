# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 00:50:28 2016

@author: karenlc
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import inv
import string
import path
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse
from copy import copy

#******************************************************************************#


# Lectura de los nombres de los archivos de datos
def readDataset(name):# in dataManagement
    file = open(name,'r')
    lines = file.readlines()
    for i in range(len(lines)):
        lines[i]=lines[i].strip("\n")
    return lines

def histogram(paths,flag):# in dataManagement
    n = len(paths)
    if flag == "duration":
        vec, vmin, vmax = getDuration(paths)
    if flag == "length":
        vec, vmin, vmax = getLength(paths)
    # Taking bins of size 10
    numBins = int( (vmax-vmin)/10)+1
    h = np.histogram(vec, bins = numBins)
    x = []
    ymin = []
    ymax = []
    for i in range(len(h[0])):
        x.append(h[1][i])
        ymin.append(0)
        ymax.append(h[0][i])
    plt.vlines(x,ymin,ymax,colors='m',linestyles='solid')

# Regresa una matriz de trayectorias:
# en la entrada (i,j) estan los caminos que comienzan en g_i y terminan en g_j
def define_trajectories_start_and_end_areas(goals,paths):# in dataManagement
    goalNum = len(goals)
    mat = np.empty((goalNum,goalNum),dtype=object)
    arclenMat = np.empty((goalNum,goalNum),dtype=object)
    #usefulPaths = []
    for i in range(goalNum):
        for j in range(goalNum):
            mat[i][j]=[]
            arclenMat[i][j] = []
    for i in range(len(paths)):
        lenData = len(paths[i].x) # Number of data for each trajectory
        # Start and finish points
        startX, startY = paths[i].x[0], paths[i].y[0]
        endX, endY = paths[i].x[lenData-1], paths[i].y[lenData-1]
        startIndex, endIndex = -1, -1

        for j in range(goalNum):
            if(isInArea(startX,startY,goals[j])):
                startIndex = j
            if(isInArea(endX,endY,goals[j])):
                endIndex = j
        if(startIndex > -1 and endIndex > -1):
            #usefulPaths.append(i+1)
            mat[startIndex][endIndex].append(paths[i])
            arclenMat[startIndex][endIndex].append(paths[i].length)
    return mat, arclenMat#, usefulPaths

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


def geomError(mError): #not needed
    Ex, Ey = 0, 0
    for i in range(len(mError)):
        Ex += mError[i][0]
        Ey += mError[i][1]
    Ex = Ex/len(mError)
    Ey = Ey/len(mError)
    return math.sqrt(Ex**2+Ey**2)

def getKnownData(x,y,z,percent):#not needed
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

#not needed
#regresa un vector para cada goal g con la cantidad de trayectorias que van de g a gi
def getNextGoal(M, numGoals): #matriz con tray. de gi a gj
    numVec,aux,indVec = [],[],[]
    for i in range(numGoals):
        vec = []
        m, im = 0, -1
        for j in range(numGoals):
            if len(M[i][j]) > m:
                m = len(M[i][j])
                im = j
            vec.append(len(M[i][j]))
        indVec.append(im) #guarda el indice j con mas trayectorias de gi a gj
        numVec.append(vec) #vec de vec con el numero de trayectorias que van de gi a gj
        aux.append(i)
    return indVec

#recibe la matriz de probabilidad, el área inicial y el numero de k goals que se piden
def getNextKGoals(start, k, pMat, nGoals):
    nextGoals = []
    aux = pMat[start]
    for i in range(k):
        maxi = 0
        maxp = 0
        for j in range(nGoals):
            if aux[j] > maxp:
                maxi = j
                maxp = aux[j]
        aux[maxi] = 0.
        nextGoals.append( maxi )
    return nextGoals

def most_likely_goals(likelihood, nGoals):
    next_goals = []
    likely = copy(likelihood)
    for i in range(3):
        maxVal = 0
        maxInd = 0
        for j in range(nGoals):
            if likely[j] > maxVal:
                maxVal = likely[j]
                maxInd = j
        next_goals.append(maxInd)
        likely[maxInd] = 0
    return next_goals


#path, goals, vecNextGoal, steps
def getPathPredictionSet(p,goals,nextGoal,steps):#no se usa
    startG = startGoal(p,goals)
    i = nextGoal[startG]
    l = len(p.x)
    start = [p.x[l-1],p.y[l-1]]
    dx, dy = goals[i][6]-goals[i][0], goals[i][7]-goals[i][1]
    end = [goals[i][0] + dx/2., goals[i][1] + dy/2.]
    dist = math.sqrt( (end[0]-start[0])**2 + (end[1]-start[1])**2 )
    step = dist/steps
    newset = []
    for i in range(steps+1):
        newset.append( p.l[l-1] + i*step )
    return end, newset, p.l[l-1] +dist
      
        
        
        
        