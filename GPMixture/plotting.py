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

color = ['g','m','r','b','c','y','w','k']

#******************************************************************************#
""" PLOT FUNCTIONS """

# Takes as an input a set of paths and plot them all on img
def plotPathSet(vec, img):
    n = len(vec)
    if(n == 0):
        return
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    # Show the image
    ax.imshow(img)
    # Plot each trajectory
    for i in range(n):
        plt.plot(vec[i].x,vec[i].y)
    s = img.shape
    v = [0,s[1],s[0],0]
    plt.axis(v)
    plt.show()

# Takes as an input a matrix of sets of paths and plot them all on img
def plotPaths(pathSetMat, img):
    print(pathSetMat)
    s         = pathSetMat.shape
    fig, axes = plt.subplots(s[0], s[1])
    for i in range(s[0]):
        for j in range(s[1]):
            n = len(pathSetMat[i][j])
            axes[i,j].set_aspect('equal')
            # Show the image
            axes[i,j].imshow(img)
            # Plot each trajectory
            for k in range(n):
                axes[i,j].plot(pathSetMat[i][j][k].x,pathSetMat[i][j][k].y)
            axes[i,j].axis('off')
    plt.show()

# Grafica los datos reales, los datos conocidos y los calculados
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

    plt.plot(knownX,knownY,'c',predictedX,predictedY,'b')

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

    maxLikelihood = max(goalsLikelihood)
    maxLW = 2
    for i in range(nGoals): #pinta la prediccion para cada subgoal
        plt.plot(predictedXYVec[i][0],predictedXYVec[i][1],'b--')
        predictedN = len(predictedXYVec[i][0])
        # For the jth predicted element
        for j in range(predictedN):
            xy = [predictedXYVec[i][0][j],predictedXYVec[i][1][j]]
            print(varXYVec[i][0][j][j],varXYVec[i][1][j][j])
            # 6.0 = 2.0 x 3.0
            # It is: to have 3.0 sigmas. Then, the Ellipse constructor asks for the diameter, hence the 2.0
            ell = Ellipse(xy,6.0*math.sqrt(math.fabs(varXYVec[i][0][j][j])),6.0*math.sqrt(math.fabs(varXYVec[i][1][j][j])))
            lw = (goalsLikelihood[i]/maxLikelihood)*maxLW
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

    v = [0,img.columns,img.rows,0]
    plt.axis(v)
    plt.show()

# Plot a set of sample trajectories
def plot_path_samples(img,x,y):
    n = len(x)
    if(n == 0):
        return
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    # Show the image
    ax.imshow(img)
    for i in range(n):
        plt.plot(x[i],y[i])
    s = img.shape
    v = [0,s[1],s[0],0]
    plt.axis(v)
    plt.show()

# Plot a set of sample trajectories and an observed partial trajectories
def plot_path_samples_with_observations(img,ox,oy,x,y):
    n = len(x)
    if(n == 0):
        return
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    # Show the image
    ax.imshow(img)
    plt.plot(ox,oy,'c')
    for i in range(n):
        plt.plot(x[i],y[i])
    s = img.shape
    v = [0,s[1],s[0],0]
    plt.axis(v)
    plt.show()
