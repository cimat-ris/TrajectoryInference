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

#color = ['b','g','r','c','m','y','k','w']
color = ['g','m','r','b','c','y','k','w']

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
def plot_prediction(img,trueX,trueY,nUsedData,predictedX,predictedY,varX,varY,finalPointElipse): 
    realX, realY = [],[]
    knownX, knownY = [], []
    N = int(len(trueX))
    
    for i in range(int(nUsedData)):
        knownX.append(trueX[i])
        knownY.append(trueY[i])
    
    for i in range(int(nUsedData-1),N):
        realX.append(trueX[i])
        realY.append(trueY[i])
    
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(img) # Show the image 
    
    plt.plot(knownX,knownY,'c',predictedX,predictedY,'b')
    
    predictedN = len(predictedX)
    for i in range(predictedN):
        xy = [predictedX[i],predictedY[i]]
        ell = Ellipse(xy,varX[i], varY[i])
        ell.set_lw(1.)
        ell.set_fill(0)
        ell.set_edgecolor('g')
        ax.add_patch(ell)
        
    plt.plot(realX,realY,'k')
        
    v = [0,1920,1080,0]
    plt.axis(v)
    plt.show()     

#Grafica los datos reales, los datos conocidos, los calculados y el punto muestreado
def plot_sampling_prediction(img,trueX,trueY,nUsedData,predictedX,predictedY,varX,varY,finish_xy): 
    realX, realY = [],[]
    partialX, partialY = [], []
    N = int(len(trueX))
    
    #seccion 2, partial path + euclidian distance
    for i in range(int(nUsedData)):
        partialX.append(trueX[i])
        partialY.append(trueY[i])
    
    for i in range(int(nUsedData),N):
        realX.append(trueX[i])
        realY.append(trueY[i])
    
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(img) # Show the image 
    
    plt.plot(partialX,partialY,'c')
    plt.plot(predictedX,predictedY,'bo')
    plt.plot(realX,realY,'ro')
    for i in range(len(predictedX)):
        xy = [predictedX[i],predictedY[i]]
        ell = Ellipse(xy,2.*np.sqrt(varX[i]), 2.*np.sqrt(varY[i]))
        ell.set_lw(1.)
        ell.set_fill(0)
        ell.set_edgecolor('g')
        ax.add_patch(ell)
    plt.plot([finish_xy[0]], [finish_xy[1]], 'yo')
        
    v = [0,1920,1080,0]
    plt.axis(v)
    plt.show() 

#Pinta las predicciones de los subgoals
def plot_subgoal_prediction(img,trueX,trueY,nUsedData,nSubgoals,predictedXYVec,varXYVec,finalPointElipse): 
    realX, realY = [],[]
    partialX, partialY = [], []
    N = int(len(trueX))
    
    for i in range(int(nUsedData)):
        partialX.append(trueX[i])
        partialY.append(trueY[i])
    
    for i in range(int(nUsedData-1),N):
        realX.append(trueX[i])
        realY.append(trueY[i])
    
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(img) # Show the image 
    
    plt.plot(partialX,partialY,'c')
    
    for i in range(nSubgoals): #pinta la prediccion para cada subgoal
        plt.plot(predictedXYVec[i][0],predictedXYVec[i][1],'b')    
        predictedN = len(predictedXYVec[i][0])
        for j in range(predictedN):
            xy = [predictedXYVec[i][0][j],predictedXYVec[i][1][j]]
            ell = Ellipse(xy,varXYVec[i][0][j], varXYVec[i][1][j])
            ell.set_lw(1.)
            ell.set_fill(0)
            ell.set_edgecolor(color[i])
            ax.add_patch(ell)        
        #final point
        xy = [predictedXYVec[i][0][predictedN-1],predictedXYVec[i][1][predictedN-1]]
        ell = Ellipse(xy,finalPointElipse[0], finalPointElipse[1])
        ell.set_lw(1.)
        ell.set_fill(0)
        ell.set_edgecolor(color[i])
        ax.add_patch(ell)
        
    plt.plot(realX,realY,'k')
    
    v = [0,1920,1080,0]
    plt.axis(v)
    plt.show() 
    
def plot_straight_line_to_finish_point(img,trueX,trueY,nUsedData,predictedX,predictedY,varX,varY,finalPointElipse): 
    partialX, partialY = [], []
    
    #seccion 2, partial path + euclidian distance
    for i in range(int(nUsedData)):
        partialX.append(trueX[i])
        partialY.append(trueY[i])
    
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(img) # Show the image 
    
    plt.plot(partialX,partialY,'c',predictedX,predictedY,'bo')
    
    #seccion 2, partial path + euclidian distance    
    plt.plot(partialX,partialY,'c', label='Partial path')    
    lineX, lineY =[],[]
    lineX.append(realX[0])
    lineX.append(predictedX[len(predictedX)-1])
    lineY.append(realY[0])
    lineY.append(predictedY[len(predictedY)-1])
    plt.plot(lineX,lineY,'b', label='Euclidean distance')
    ax.legend()
    
    v = [0,1920,1080,0]
    plt.axis(v)
    plt.show()     

def plot_multiple_predictions(img,x,y,nUsedData,nGoals,predictedXYVec,varXYVec): 
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
            ell.set_lw(1.)
            ell.set_fill(0)
            ell.set_edgecolor(color[i])
            ax.add_patch(ell)      
            
    plt.plot(realX,realY,'c--')
    
    v = [0,1920,1080,0]
    plt.axis(v)
    plt.show() 
    