# -*- coding: utf-8 -*-
"""
Plotting functions

@author: karenlc
"""

from path import *
import numpy as np
import math
from copy import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

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
        #plt.plot([ox[-1],x[i][0]],[oy[-1],y[i][0]])
    s = img.shape
    v = [0,s[1],s[0],0]
    plt.axis(v)
    plt.show()
    
#Grafica con subplots los tests del sampling dado un conjunto de observaciones
def plot_interaction_with_sampling_test(img,obsVec, samplesVec, potentialVec):
    N = len(obsVec) #num de tests
    print("Number of tests:",N)
    n, m = 1, 1    
    if(N%3 == 0):
        n = 3
        m = int(N/3)
    elif(N%2 == 0):
        n = 2
        m = int(N/2)
    else:
        m = N
    print("Dim:",n,",",m)
    fig, axes = plt.subplots(n,m)
    for i in range(n):
        for j in range(m):
            axes[i,j].set_aspect('equal')
            # Show the image
            axes[i,j].imshow(img)
            # Plot each test
            t = (i*m)+j
            print("Test num:",t)
            numPaths = len(obsVec[t][0])
            for k in range(numPaths):
                colorId = (k)%len(color)
                axes[i,j].plot(obsVec[t][0][k], obsVec[t][1][k],color[colorId],lw=2.0)
                axes[i,j].plot(samplesVec[t][0][k],samplesVec[t][1][k],color[colorId]+'--',lw=2.0)
            axes[i,j].axis('off')
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    