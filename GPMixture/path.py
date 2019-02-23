# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 15:13:31 2016

@author: karenlc
"""
import numpy as np
import math
import matplotlib.pyplot as plt

class path:
    def __init__(self,vecT,vecX,vecY):
        self.t = vecT
        self.x = vecX
        self.y = vecY
        self.l = pathArcLength(self.x,self.y)
        self.duration, self.length = statistics(self.t,self.x,self.y)
        if self.duration > 0:            
            self.v = self.length/self.duration
        else:
            self.v = -1.

#************************************************************#

def write_useful_paths_file(paths): #paths es un vector de indices
    N = len(paths)
    #f = open("usefulPaths_%d.txt"%N,"w")
    f = open("usefulPaths.txt","w")
    for j in range(N):
        i = paths[j]
        if i < 10:
            s = "Data/00000%d.txt\n"%(i)
            f.write(s)   
        if i >= 10 and i < 100:
            s = "Data/0000%d.txt\n"%(i)
            f.write(s)   
        if i >= 100 and i < 1000:
            s = "Data/000%d.txt\n"%(i)
            f.write(s)   
        if i >= 1000 and i <= 2000:
            s = "Data/00%d.txt\n"%(i)
            f.write(s)  
    f.close()
    
#Devuelve un conjunto con las trayectorias que van entre los goals
def getUsefulPaths(paths, goals):
    useful = []
    for i in range(len(paths)):
        pathLen = len(paths[i].x) 
        first = [paths[i].x[0],paths[i].y[0]]
        last = [paths[i].x[pathLen-1],paths[i].y[pathLen-1]]
        isFirst, isLast = -1, -1
        for j in range(len(goals)):
            if(isInArea(first,goals[j])):
                isFirst = j
            if(isInArea(last,goals[j])):
                isLast = j
        if(isFirst > -1 and isLast > -1):
            useful.append(paths[i])
            
    return useful

# Recibe un punto (x,y) y un area de interes R
def isInArea(p,R):
    x = p[0]
    y = p[1]
    if(x >= R[0] and x <= R[len(R)-2]):
        if(y >= R[1] and y <= R[len(R)-1]):
            return 1
        else:
            return 0
    else: 
        return 0
        
""" Regresa una matriz de trayectorias:
en la entrada (i,j) estan los caminos que comienzan en g_i y terminan en g_j"""
def define_trajectories_start_and_end_areas(startGoals, finishGoals, paths):
    nRows = len(startGoals)
    nColumns = len(finishGoals)
    #goalNum = len(goals)
    mat = np.empty((nRows,nColumns),dtype=object) 
    arclenMat = np.empty((nRows,nColumns),dtype=object) 
    #usefulPaths = []
    for i in range(nRows):
        for j in range(nColumns):
            mat[i][j]=[]
            arclenMat[i][j] = []
            
    for i in range(len(paths)):
        lenData = len(paths[i].x) # Number of data for each trajectory
        # Start and finish points
        startX, startY = paths[i].x[0], paths[i].y[0]
        endX, endY = paths[i].x[lenData-1], paths[i].y[lenData-1]
        startIndex, endIndex = -1, -1
                
        for j in range(nRows):
            if(isInArea([startX,startY], startGoals[j])):
                startIndex = j
        for k in range(nColumns):
            if(isInArea([endX,endY], finishGoals[k])):
                endIndex = k
        if(startIndex > -1 and endIndex > -1):
            #usefulPaths.append(i+1)
            mat[startIndex][endIndex].append(paths[i])
            arclenMat[startIndex][endIndex].append(paths[i].length)
    return mat, arclenMat#, usefulPaths


def pathArcLength(x,y):
    n = len(x)
    l = [0]
    for i in range(n):
        if i > 0:
            l.append(np.sqrt( (x[i]-x[i-1])**2 + (y[i]-y[i-1])**2 ) )
    for i in range(n):
        if(i>0):
            l[i] = l[i] +l[i-1]
    return l

def statistics(t,x,y):
    n = len(t)
    duration = t[n-1] - t[0]        
    l = pathArcLength(x,y)
    length = l[len(l)-1]
    return duration, length


def histogram(paths,flag):
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
    