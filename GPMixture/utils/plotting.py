# -*- coding: utf-8 -*-
"""
Plotting functions

@author: karenlc
"""

import numpy as np
import math
from copy import copy
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('TkAgg')
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
import pandas as pd
import time

color = ['g','m','r','b','c','y','w','k']

#******************************************************************************#
""" PLOT FUNCTIONS """
def plotPaths(vec):
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    # Show the image
    ax.imshow(img)
    n = len(vec)
    for i in range(n):
        plt.plot(vec[i].x,vec[i].y)

    v = [0,1920,1080,0]
    plt.axis(v)
    plt.show()

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
    realX = trueX[knownN-1:N]
    realY = trueY[knownN-1:N]

    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(img) # Show the image

    plt.plot(knownX,knownY,'c',predictedXY[:,0],predictedXY[:,1],'b')
    plt.plot([knownX[-1],predictedXY[0,0]],[knownY[-1],predictedXY[0,1]],'b')
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
    v = [0,img.shape[1],img.shape[0],0]
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

    v = [0,img.shape[1],img.shape[0],0]
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
        if (predictedXYVec[i].shape[0]==0):
            continue
        print('[RES] Plotting GP ',i)
        # For each goal/subgoal, draws the prediction
        #plt.plot(knownX,knownY,'c',predictedXY[:,0],predictedXY[:,1],'b')
        #plt.plot([knownX[-1],predictedXY[0,0]],[knownY[-1],predictedXY[0,1]],'b')

        plt.plot(predictedXYVec[i][:,0],predictedXYVec[i][:,1],'b--')
        plt.plot([partialX[-1],predictedXYVec[i][0,0]],[partialY[-1],predictedXYVec[i][0,1]],'b--')
        predictedN = predictedXYVec[i].shape[0]
        # For the jth predicted element
        for j in range(predictedN):
            xy = [predictedXYVec[i][j,0],predictedXYVec[i][j,1]]
            # 6.0 = 2.0 x 3.0
            # It is: to have 3.0 sigmas. Then, the Ellipse constructor asks for the diameter, hence the 2.0
            vx  = varXYVec[i][0,j,j]
            vy  = varXYVec[i][1,j,j]
            ell = Ellipse(xy,6.0*math.sqrt(math.fabs(vx)),6.0*math.sqrt(math.fabs(vy)))
            lw = (goalsLikelihood[i%nGoals]/maxLikelihood)*maxLW
            ell.set_lw(lw)
            ell.set_fill(0)
            ell.set_edgecolor(color[i%nGoals])
            ax.add_patch(ell)

    plt.plot(realX,realY,'c--')
    v = [0,img.shape[1],img.shape[0],0]
    plt.axis(v)
    plt.show()

def animate_multiple_predictions_and_goal_likelihood(img,x,y,nUsedData,nGoals,goalsLikelihood,predictedXYVec,varXYVec):
    realX, realY = [],[]
    partialX, partialY = [], []
    N = int(len(x))
    # Observed data
    print(nUsedData,N)
    for i in range(int(nUsedData)):
        partialX.append(x[i])
        partialY.append(y[i])
    # Data to predict (ground truth)
    for i in range(int(nUsedData-1),N):
        realX.append(x[i])
        realY.append(y[i])

    fig,ax = plt.subplots()
    plt.gca().set_axis_off()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # Plot the observed data
    lineObs, = ax.plot(partialX,partialY, lw=3,color='c')

    # Show the image
    ax.imshow(img)
    plt.tight_layout()
    # Likelihoods
    maxLikelihood = max(goalsLikelihood)
    maxLW = 3.0

    # For all potential goals
    ells      = []
    pls       = []
    #ax.set_aspect('equal')
    for i in range(nGoals):
            lw = max((goalsLikelihood[i]/maxLikelihood)*maxLW,1)
            e = Ellipse([0,0],0,0)
            e.set_fill(0)
            if (predictedXYVec[i].shape[0]==0):
                l, = ax.plot([0],[0], lw=0,color='b')
                e.set_lw(0)
            else:
                l, = ax.plot(predictedXYVec[i][0,0],predictedXYVec[i][0,1], lw=lw,color='b')
                ## Ellipse center
                xy  = [predictedXYVec[i][0,0],predictedXYVec[i][0,1]]
                vx  = varXYVec[i][0,0,0]
                vy  = varXYVec[i][1,0,0]
                e.center = xy
                e.width  = 6.0*math.sqrt(math.fabs(vx))
                e.height = 6.0*math.sqrt(math.fabs(vx))
                e.set_edgecolor('b')
                e.set_lw(lw)
            ax.add_patch(e)
            ells.append(e)
            pls.append(l)
    v = [0,img.shape[1],img.shape[0],0]
    plt.axis(v)


    def animate(i):
        for j in range(nGoals):
            if (predictedXYVec[j].shape[0]==0):
                continue
            predictedN = predictedXYVec[j].shape[0]
            if (i<predictedXYVec[j].shape[0]):
                p   = [predictedXYVec[j][i,0],predictedXYVec[j][i,1]]
                vx  = varXYVec[j][0,i,i]
                vy  = varXYVec[j][1,i,i]
                pls[j].set_data(predictedXYVec[j][0:i,0],predictedXYVec[j][0:i,1])
                ells[j].center = p
                ells[j].width  = 6.0*math.sqrt(math.fabs(vx))
                ells[j].height = 6.0*math.sqrt(math.fabs(vx))


    anim = FuncAnimation(fig, animate,frames=100,interval=100)
    plt.show()
    return


def plot_subgoals(img, goal, numSubgoals, axis):
    subgoalsCenter, size = get_subgoals_center_and_size(numSubgoals, goal, axis)

    for i in range(numSubgoals):
        xy  = [subgoalsCenter[i][0],subgoalsCenter[i][1]]
        if axis==0:
            plt.plot([subgoalsCenter[i][0]-size[0]/2.0,subgoalsCenter[i][0]+size[0]/2.0],[subgoalsCenter[i][1],subgoalsCenter[i][1]],color[i])
        else:
            plt.plot([subgoalsCenter[i][0],subgoalsCenter[i][0]],[subgoalsCenter[i][1]-size[1]/2.0,subgoalsCenter[i][1]+size[1]/2.0],color[i])


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

# Plot the scene structure: goals and sub-goals
def plot_scene_structure(img,goalsData):
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(img)

    for i in range(goalsData.nGoals):
        plot_subgoals(img, goalsData.areas[i], 2, goalsData.areasAxis[i])
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
        plt.plot([ox[-1],x[i][0]],[oy[-1],y[i][0]])
    s = img.shape
    v = [0,s[1],s[0],0]
    plt.axis(v)
    plt.show()

def plot_path_set_samples_with_observations(img,ox,oy,x,y):
    n = len(x)
    if(n == 0):
        return
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    # Show the image
    ax.imshow(img)
    for i in range(n):
        colorId = i%len(color)
        plt.plot(ox[i],oy[i],color[colorId],lw=2.0)
        plt.plot(x[i],y[i],color[colorId]+'--',lw=2.0)
    s = img.shape
    v = [0,s[1],s[0],0]
    plt.axis(v)
    plt.show()

#Grafica con subplots los tests del sampling dado un conjunto de observaciones
def plot_interaction_with_sampling_test(img,observedPaths, samplesVec, potentialVec):
    N        = len(samplesVec) # Number of joint samples
    numPaths = len(observedPaths)
    n, m = 1, 1
    if(N%3 == 0):
        n = 3
        m = int(N/3)
    elif(N%2 == 0):
        n = 2
        m = int(N/2)
    else:
        m = N
    fig, axes = plt.subplots(n,m)
    for i in range(n):
        for j in range(m):
            axes[i,j].set_aspect('equal')
            # Show the image
            axes[i,j].imshow(img)
            # Plot each test
            t = (i*m)+j
            for k in range(numPaths):
                colorId = (k)%len(color)
                ox,oy = observedPaths[k].x,observedPaths[k].y
                axes[i,j].plot(ox,oy,color[colorId],lw=2.0)
                sx,sy = samplesVec[t][k].x,samplesVec[t][k].y
                axes[i,j].plot(sx,sy,color[colorId]+'--',lw=2.0)
                string = "{0:1.3e}".format(potentialVec[t])#str(potentialVec[t])
                axes[i,j].set_title('IP='+string)
            axes[i,j].axis('off')
    plt.show()

def plot_table__():
    data = [[ 66386, 174296,  75131, 577908,  32015],
        [ 58230, 381139,  78045,  99308, 160454],
        [ 89135,  80552, 152558, 497981, 603535],
        [ 78415,  81858, 150656, 193263,  69638],
        [139361, 331509, 343164, 781380,  52269]]
    stringData = []
    n, m = len(data), len(data[0])
    for i in range(n):
        row = []
        for j in range(m):
            string = "{0:1.3e}".format(data[i][j])
            row.append(string)
        stringData.append(row)

    fig, axes = plt.subplots(1)

    axes.xaxis.set_visible(False)
    axes.yaxis.set_visible(False)
    # Show the image
    plt.table(cellText=stringData,loc='center')
    plt.show()

def plot_table(data, rowLabels, colLabels, tittle):
    stringData = []
    n, m = len(data), len(data[0])
    for i in range(n):
        row = []
        for j in range(m):
            string = "{0:1.3e}".format(data[i][j])
            row.append(string)
        stringData.append(row)
    fig, ax = plt.subplots()
    plt.title(tittle)
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    ax.table(cellText=stringData,cellLoc='center',rowLabels = rowLabels, colLabels = colLabels, loc='center')

    fig.tight_layout()

    plt.show()

def boxplot(data, title):
    boxprops = dict(linewidth=1.2, color='black')
    flierprops = dict(marker='o', markerfacecolor='blue', markersize=8)
    meanlineprops = dict(linestyle='--', linewidth=2.3, color='green')

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.boxplot(data, boxprops=boxprops, whiskerprops=dict(linestyle='-', linewidth=1.2, color='black'), flierprops=flierprops, meanprops=meanlineprops, showmeans=True, meanline=True)
