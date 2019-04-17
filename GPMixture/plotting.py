# -*- coding: utf-8 -*-
"""
Plotting functions

@author: karenlc
"""
        
from GPRlib import *
from path import *
import numpy as np
import math
import GPRlib
from copy import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

color = ['m','m','r','b','c','y','k','w']

#******************************************************************************#
""" PLOT FUNCTIONS """
#recibe: un conjunto de trayectorias
def plotPaths(vec, img):
    n = len(vec)
    if(n == 0):
        return
    #print("num de trayectorias:", n)
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    # Show the image
    ax.imshow(img)
    for i in range(n):
        plt.plot(vec[i].x,vec[i].y)
    
    v = [0,1920,1080,0]
    plt.axis(v)
    plt.show() 
    
#Grafica los datos reales, los datos conocidos y los calculados
def plot_prediction(img,trueX,trueY,knownN,predictedX,predictedY,varX,varY): 
    realX, realY = [],[]
    N = int(len(trueX))
    
    knownX = trueX[0:knownN]
    knownY = trueY[0:knownN]
    
    realX = trueX[knownN:N]
    realY = trueY[knownN:N]
    
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(img) # Show the image 
    
    plt.plot(knownX,knownY,'c',predictedX,predictedY,'bo')
    
    predictedN = len(predictedX)
    for i in range(predictedN):
        xy = [predictedX[i],predictedY[i]]
        ell = Ellipse(xy,varX[i], varY[i])
        ell.set_lw(1.)
        ell.set_fill(0)
        ell.set_edgecolor('m')
        ax.add_patch(ell)
        
    plt.plot(realX,realY,'c--')
        
    v = [0,1920,1080,0]
    plt.axis(v)
    plt.show()     

#Grafica los datos reales, los datos conocidos, los calculados y el punto muestreado
def plot_sampling_prediction(img,trueX,trueY,knownN,predictedX,predictedY,varX,varY,finish_xy): 
    realX, realY = [],[]
    partialX, partialY = [], []
    N = int(len(trueX))
    
    partialX = trueX[0:knownN]
    partialY = trueY[0:knownN]
    
    realX = trueX[knownN:N]
    realY = trueY[knownN:N]
    
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(img) # Show the image 
    
    plt.plot(partialX,partialY,'c')
    plt.plot(predictedX,predictedY,'bo')
    plt.plot(realX,realY,'ro')
    for i in range(len(predictedX)):
        xy = [predictedX[i],predictedY[i]]
        ell = Ellipse(xy,varX[i], varY[i])
        ell.set_lw(1.)
        ell.set_fill(0)
        ell.set_edgecolor('g')
        ax.add_patch(ell)
    plt.plot([finish_xy[0]], [finish_xy[1]], 'yo')
        
    v = [0,1920,1080,0]
    plt.axis(v)
    plt.show() 

#Pinta las predicciones de los subgoals
def plot_subgoal_prediction(img,trueX,trueY,knownN,nSubgoals,predictedXYVec,varXYVec): 
    N = int(len(trueX))
    
    partialX = trueX[0:knownN]
    partialY = trueY[0:knownN]
    
    realX = trueX[knownN:N]
    realY = trueY[knownN:N]
    
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(img) # Show the image 
    
    plt.plot(partialX,partialY,'c')
    
    for i in range(nSubgoals): #pinta la prediccion para cada subgoal
        plt.plot(predictedXYVec[i][0],predictedXYVec[i][1],'b--')    
        predictedN = len(predictedXYVec[i][0])
        for j in range(predictedN):
            xy = [predictedXYVec[i][0][j],predictedXYVec[i][1][j]]
            ell = Ellipse(xy,varXYVec[i][0][j], varXYVec[i][1][j])
            ell.set_lw(1.)
            ell.set_fill(0)
            ell.set_edgecolor(color[i])
            ax.add_patch(ell)        
        
    plt.plot(realX,realY,'c--')
    
    v = [0,1920,1080,0]
    plt.axis(v)
    plt.show() 
    
#Imagen en seccion 2: partial path + euclidian distance
def plot_euclidean_distance_to_finish_point(img,trueX,trueY,knownN,finalXY): 
    partialX, partialY = [], []
    for i in range(int(knownN)):
        partialX.append(trueX[i])
        partialY.append(trueY[i])
    
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(img) # Show the image 
      
    plt.plot(partialX,partialY,'c', label='Partial path')    
    lineX, lineY =[],[]
    lineX.append(trueX[knownN-1])
    lineX.append(finalXY[0])
    lineY.append(trueY[knownN-1])
    lineY.append(finalXY[1])
    plt.plot(lineX,lineY,'b', label='Euclidean distance')
    ax.legend()
    
    v = [0,1920,1080,0]
    plt.axis(v)
    plt.show()     

def plot_multiple_predictions(img,x,y,knownN,nGoals,predictedXYVec,varXYVec): 
    N = int(len(x))

    partialX = trueX[0:knownN]
    partialY = trueY[0:knownN]
    
    realX = trueX[knownN:N]
    realY = trueY[knownN:N]
    
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(img) # Show the image 
    
    plt.plot(partialX,partialY,'c')
    
    for i in range(nGoals): #pinta la prediccion para cada subgoal
        plt.plot(predictedXYVec[i][0],predictedXYVec[i][1],'b--')    
        predictedN = len(predictedXYVec[i][0])
        for j in range(predictedN):
            xy = [predictedXYVec[i][0][j],predictedXYVec[i][1][j]]
            ell = Ellipse(xy,varXYVec[i][0][j], varXYVec[i][1][j])
            ell.set_lw(1.)
            ell.set_fill(0)
            ell.set_edgecolor(color[i])
            ax.add_patch(ell)      
            
    plt.plot(realX,realY,'c--')
    
    v = [0,1920,1080,0]
    plt.axis(v)
    plt.show() 
    
def plot_multiple_predictions_and_goal_likelihood(img,x,y,nUsedData,nGoals,goalsLikelihood,predictedXYVec,varXYVec): 
    realX, realY = [],[]
    partialX, partialY = [], []
    N = int(len(x))
    
    for i in range(int(nUsedData)):
        partialX.append(x[i])
        partialY.append(y[i])
    
    for i in range(int(nUsedData-1),N):
        realX.append(x[i])
        realY.append(y[i])
    
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(img) # Show the image 
    
    plt.plot(partialX,partialY,'c')
    
    for i in range(nGoals): #pinta la prediccion para cada subgoal
        plt.plot(predictedXYVec[i][0],predictedXYVec[i][1],'b--')    
        predictedN = len(predictedXYVec[i][0])
        for j in range(predictedN):
            xy = [predictedXYVec[i][0][j],predictedXYVec[i][1][j]]
            ell = Ellipse(xy,varXYVec[i][0][j], varXYVec[i][1][j])
            lw = goalsLikelihood[i]*7.
            ell.set_lw(lw)
            ell.set_fill(0)
            ell.set_edgecolor(color[i])
            ax.add_patch(ell)      
            
    plt.plot(realX,realY,'c--')
    
    v = [0,1920,1080,0]
    plt.axis(v)
    plt.show() 

def plot_subgoals(img, goal, numSubgoals, axis):
    subgoalsCenter, size = get_subgoals_center_and_size(numSubgoals, goal, axis)
    
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(img)
    
    for i in range(numSubgoals): 
        xy = [subgoalsCenter[i][0],subgoalsCenter[i][1]]
        ell = Ellipse(xy,size[0]*0.75, size[1])
        ell.set_lw(2.5)
        ell.set_fill(0)
        ell.set_edgecolor(color[i])
        ax.add_patch(ell)      
    
    v = [0,1920,1080,0]
    plt.axis(v)
    plt.show() 