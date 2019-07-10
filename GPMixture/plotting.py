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

# Plot the true data, the predicted ones and their variance
def plot_prediction(img,trueX,trueY,knownN,predictedXY,varXY):
    realX, realY = [],[]
    N = int(len(trueX))

    knownX = trueX[0:knownN]
    knownY = trueY[0:knownN]
    realX = trueX[knownN:N]
    realY = trueY[knownN:N]

    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(img) # Show the image

    plt.plot(knownX,knownY,'c',predictedXY[:,0],predictedXY[:,1],'b')
    predictedN = predictedXY.shape[0]
    for i in range(predictedN):
        xy = [predictedXY[i,0],predictedXY[i,1]]
        ell = Ellipse(xy,6.0*math.sqrt(math.fabs(varXY[0][i,i])),6.0*math.sqrt(math.fabs(varXY[1][i,i])))
        ell.set_lw(1.)
        ell.set_fill(0)
        ell.set_edgecolor('m')
        ax.add_patch(ell)
    plt.plot(realX,realY,'c--')

    v = [0,img.shape[1],img.shape[0],0]
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

def plot_multiple_predictions_and_goal_likelihood(img,x,y,nUsedData,nGoals,goalsLikelihood,predictedXYVec,varXYVec):
    realX, realY = [],[]
    partialX, partialY = [], []
    N = int(len(x))
    # Observed data
    for i in range(int(nUsedData)):
        partialX.append(x[i])
        partialY.append(y[i])
    # Data to predict (ground truth)
    for i in range(int(nUsedData-1),N):
        realX.append(x[i])
        realY.append(y[i])

    fig,ax = plt.subplots(1)
    plt.margins(0, 0)
    plt.gca().set_axis_off()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    ax.set_aspect('equal')
    # Show the image
    ax.imshow(img)
    # Plot the observed data
    plt.plot(partialX,partialY,'c')

    maxLikelihood = max(goalsLikelihood)
    maxLW = 2
    for i in range(len(predictedXYVec)):
        if predictedXYVec[i]!=None:
            print('[RES] Plotting GP ',i)
            # For each goal/subgoal, draws the prediction
            plt.plot(predictedXYVec[i][0],predictedXYVec[i][1],'b--')
            predictedN = len(predictedXYVec[i][0])
            # For the jth predicted element
            for j in range(predictedN):
                xy = [predictedXYVec[i][0][j],predictedXYVec[i][1][j]]
                # 6.0 = 2.0 x 3.0
                # It is: to have 3.0 sigmas. Then, the Ellipse constructor asks for the diameter, hence the 2.0
                ell = Ellipse(xy,6.0*math.sqrt(math.fabs(varXYVec[i][0][j][j])),6.0*math.sqrt(math.fabs(varXYVec[i][1][j][j])))
                lw = (goalsLikelihood[i%nGoals]/maxLikelihood)*maxLW
                ell.set_lw(lw)
                ell.set_fill(0)
                ell.set_edgecolor(color[i%nGoals])
                ax.add_patch(ell)

    plt.plot(realX,realY,'c--')
    v = [0,img.shape[1],img.shape[0],0]
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
