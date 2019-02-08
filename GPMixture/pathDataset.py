# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 00:50:28 2016

@author: karenlc
"""
from GPRlib import *
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import inv
import string
import path
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse
from copy import copy

#**************************PLOT*******************************#

#recibe: un conjunto de trayectorias
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

    
#******************************************************************************#

# Lectura de los nombres de los archivos de datos
def readDataset(name):   
    file = open(name,'r')
    lines = file.readlines()
    for i in range(len(lines)):
        lines[i]=lines[i].strip("\n")
    return lines
   
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
    
# Regresa una matriz de trayectorias:
# en la entrada (i,j) estan los caminos que comienzan en g_i y terminan en g_j
def define_trajectories_start_and_end_areas(goals,paths):
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

    
def geomError(mError):
    Ex, Ey = 0, 0
    for i in range(len(mError)):
        Ex += mError[i][0]
        Ey += mError[i][1]
    Ex = Ex/len(mError) 
    Ey = Ey/len(mError)
    return math.sqrt(Ex**2+Ey**2)
    
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

#start, goals, last know (x,y,l), nextGoal
def getPredictionSet(x,y,l,start,nextGoal,goals):
    #calcula en centro del area siguiente
    dx, dy = goals[nextGoal][6]-goals[nextGoal][0], goals[nextGoal][7]-goals[nextGoal][1]
    end = [goals[nextGoal][0] + dx/2., goals[nextGoal][1] + dy/2.]
    dist = math.sqrt( (end[0]-x)**2 + (end[1]-y)**2 )
    
    steps = 20#num de pasos
    step = dist/float(steps)
    newset = []
    for i in range(steps+1):
        newset.append( l + i*step )
        
    return end, newset, l + dist
    
    
def get_known_set(x,y,z,knownN):
    trueX,trueY, trueZ = [],[],[]
    knownN = int(knownN)
    for j in range(knownN): #numero de datos conocidos
        trueX.append(x[j])
        trueY.append(y[j])
        trueZ.append(z[j])
        
    return trueX, trueY, trueZ
    
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

#recibe: datos conocidos, valores por predecir, areas de inicio y final
def prediction_test(x,y,z,knownN,start,end,goals):
    kernelX = kernelMat_x[start][end]
    kernelY = kernelMat_y[start][end]
    
    trueX, trueY, trueZ = get_known_set(x,y,z,knownN)
    last_x = x[knownN-1] 
    last_y = y[knownN-1] 
    last_z = z[knownN-1]   
    
    final_xy, newZ, final_z = getPredictionSet(last_x,last_y,last_z,start,end,goals)  
    trueX.append(final_xy[0])    
    trueY.append(final_xy[1])
    trueZ.append(final_z)
    
    newX,newY,varX,varY = prediction_XY(trueX,trueY,trueZ,newZ,kernelX,kernelY) 
    plot_prediction(img,x,y,knownN,newX,newY,varX,varY)
    
#recibe: datos conocidos, valores por predecir, trayectoria real
def multigoal_prediction_test(x,y,z,knownN,start,end,goals):
    trueX, trueY, trueZ = get_known_set(x,y,z,knownN)
    
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(img) # Show the image
    
    for i in range(len(end)):
        nextGoal = end[i]
        kernelX = kernelMat_x[start][nextGoal]
        kernelY = kernelMat_y[start][nextGoal]
        auxX = copy(trueX) 
        auxY = copy(trueY) 
        auxZ = copy(trueZ)
        final_point = middle_of_area(goals[nextGoal])      
        auxX.append(final_point[0])
        auxY.append(final_point[1])
        steps = 20
        end_, newZ, l_ = getPredictionSet(trueX[knownN-1],trueY[knownN-1],trueZ[knownN-1],start,nextGoal,goals)     
        auxZ.append(l_)          
        newX, newY, varX, varY = prediction_XY(auxX,auxY,auxZ,newZ,kernelX,kernelY) 
         
        plt.plot(trueX,trueY,'r')
        plt.plot(newX,newY,'b')

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
    

#******************************************************************************#
# Areas de interes [x1,y1...x2,y2]
R0 = [400,40,760,40,400,230,760,230] #rojo
R1 = [300,260,500,260,300,400,500,400] #amarillo
R2 = [0,400,180,400,0,710,180,710] #verde
R3 = [760,460,1050,460,760,730,1050,730] #cian
R4 = [1650,450,1800,450,1650,750,1800,750] #azul
R5 = [1030,40,1400,40,1030,230,1400,230] #naranja

#Arreglo que contiene las areas de interes
#areas = np.array([R0,R1,R2,R3,R4,R5])
areas = [R0,R1,R2,R3,R4,R5]
nGoals = len(areas)
img=mpimg.imread('areasDeInteres.png')  

#pathFiles = readDataset('pathSet2000.txt')
pathFiles = readDataset('usefulPaths_198.txt')

#Al leer cortamos las trayectorias multiobjetivos por pares consecutivos 
#y las agregamos como trayectorias independientes 
true_paths, multigoal = get_paths_from_files(pathFiles,areas)

#Histograma    
#histogram(true_paths,"duration")
#Quitamos las trayectorias demasiado largas o demasiado cortas
learnSet = filter_paths(true_paths)
startToGoalPath, arclenMat = define_trajectories_start_and_end_areas(areas,learnSet)#true_paths)  
#print("Trajectories between goals\n",startToGoalPath)  
probabilityMat = next_goal_probability_matrix(startToGoalPath, nGoals)
""" 
histogram(startToGoalPath[0][4],"duration")
histogram(startToGoalPath[0][4],"length")
arclenVec, minl, maxl = getLength(startToGoalPath[0][4])
med = np.median(arclenVec)
var = np.var(arclenVec)
print("media: ",med,"varianza: ",var)
"""

kernelMat, parametersMat = create_kernel_matrix("combined", nGoals)
probGoal = getNextGoal(startToGoalPath,6)
"""
#Aprendizaje:
kernelMat_x, kernelMat_y = optimize_parameters_between_goals(startToGoalPath, kernelMat, parametersMat, nGoals)
write_parameters(kernelMat_x,nGoals,"x")
write_parameters(kernelMat_y,nGoals,"y")
print("***********FIN DEL APRENDIZAJE*********")
"""
kernelMat_x = copy(kernelMat)
kernelMat_y = copy(kernelMat)
#fijamos los parámetros para cada matriz de kernel
kernelMat_x = read_and_set_parameters(kernelMat_x,"parameters_x.txt",nGoals)
kernelMat_y = read_and_set_parameters(kernelMat_y,"parameters_y.txt",nGoals)

s1 = 0 #area inicial
print("start =",s1)
start = [s1]
#siguiente area mas probable
nextG = getNextKGoals(start[0], 1, probabilityMat, nGoals)
print("next goal", nextG[0])

kernelX = kernelMat_x[start[0]][nextG[0]]
kernelY = kernelMat_y[start[0]][nextG[0]]

traj_n = 0
traj = startToGoalPath[start[0]][nextG[0]][traj_n] #trayectoria para predecir
traj_len = len(startToGoalPath[start[0]][nextG[0]][traj_n].x)
traj_arclen = startToGoalPath[start[0]][nextG[0]][traj_n].length
likelihood_vector, error_vector = [], []
arclen_vec = []

part_num = 6
for i in range(part_num-1):
    arclen_vec.append( (i+1)*(traj_arclen/float(part_num))  )
    knownN = int((i+1)*(traj_len/part_num)) 
    
    trueX,trueY,trueL = get_known_set(traj.x,traj.y,traj.l,knownN)
    likelihood, error = get_goals_likelihood(trueX,trueY,trueL,start[0],kernelMat_x,kernelMat_x,areas,nGoals)
    likelihood_vector.append(likelihood)  
    error_vector.append(error)
    likely_goals = most_likely_goals(likelihood, nGoals)
    print("likely goals:", likely_goals)
    
    multigoal_prediction_test(traj.x,traj.y,traj.l,knownN,start[0],likely_goals,areas)
    #prediction_test(traj.x,traj.y,traj.l,knownN,start[0],nextG[0],areas)
    #prediction_test_over_time(traj.x,traj.y,traj.t,knownN,start[0],nextG[0],areas)
        
        
color = ['r','y','g','c','b','m']

for i in range(nGoals):
    likelihood_gi = []
    error_gi = []
    for j in range(part_num-1):
        likelihood_gi.append(likelihood_vector[j][i])
        error_gi.append(error_vector[j][i])
    plt.plot(arclen_vec,likelihood_gi,color[i])
    #plt.plot(arclen_vec,error_gi,color[i])
    plt.xlabel('arc length')
    #plt.ylabel('error')
    plt.ylabel('likelihood')
     
