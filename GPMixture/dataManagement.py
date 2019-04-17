"""
@author: karenlc
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.optimize import minimize
import kernels
import GPRlib
import path
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse
from copy import copy
import random

#manejo de data
""" READ DATA """
# Lectura de los nombres de los archivos de datos
def readDataset(fileName):   
    file = open(fileName,'r')
    lines = file.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].strip("\n")
    return lines
    

def get_paths_from_file(path_file,areas):
    paths, multigoal_paths = [],[] 
    # Open file
    with open(path_file) as f:
        # Each line should contain a path
        for line in f:
            auxX, auxY, auxT = [],[],[]
            # Split the line into sub-strings
            data = line.split()
            for i in range(0, len(data), 3):
                x_ = int(data[i])
                y_ = int(data[i+1])
                t_ = int(data[i+2])
                if equal(auxX,auxY,x_,y_) == 0:
                    auxX.append(x_)
                    auxY.append(y_)
                    auxT.append(t_)
            auxPath = path.path(auxT,auxX,auxY)
            gi = get_goal_sequence(auxPath,areas)
            if len(gi) > 2:
                multigoal_paths.append(auxPath)
                new_paths = break_multigoal_path(auxPath,gi,areas)   
                for j in range(len(new_paths)):
                    paths.append(new_paths[j])
            else:
                paths.append(auxPath) 
    return paths, multigoal_paths
    
"""Recibe un conjunto de paths y obtiene los puntos (x,y,z)
con z = {tiempo, long de arco} si flag = {"time", "length"}"""
def get_data_from_paths(paths, flag):
    for i in range(len(paths)):
        auxX, auxY, auxT = paths[i].x, paths[i].y, paths[i].t
        auxL = arclength(auxX, auxY)
        if(i==0):
            x, y, t = [auxX], [auxY], [auxT]
            l = [auxL]
        else:
            x.append(auxX)
            y.append(auxY)
            t.append(auxT) 
            l.append(auxL)
           
    if(flag == "time"):
        z = t
    if(flag == "length"):
        z = l
    return x, y, z

"""
Lee la lista de archivos y obtiene los puntos (x,y,z),
con z = {tiempo, long de arco} si flag = {"time", "length"}
"""
def get_data_from_files(files, flag):
    for i in range(len(files)):
        auxX, auxY, auxT = read_file(files[i])
        auxL = arclength(auxX, auxY)
        if(i==0):
            x, y, t = [auxX], [auxY], [auxT]
            l = [auxL]
        else:
            x.append(auxX)
            y.append(auxY)
            t.append(auxT) 
            l.append(auxL)
           
    if(flag == "time"):
        z = t
    if(flag == "length"):
        z = l
    return x, y, z
    
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
    
""" FILTER PATHS """
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
    
#Calcula la mediana m, y la desviacion estandar s de un vector de paths
#si alguna trayectoria difiere de la mediana en 4s o mas, se descarta
def filter_path_vec(vec):
    arclen = get_paths_arcLength(vec)    
    m = np.median(arclen)
    var = np.sqrt(np.var(arclen))
    #print("mean len:", m)
    learnSet = []
    for i in range(len(arclen)):
        if abs(arclen[i] - m) < 1.0*var:
            learnSet.append(vec[i])
            
    return learnSet

#Arreglo de NxNxM_ij
#N = num de goals, M_ij = num de paths de g_i a g_j
def filter_path_matrix(M, nRows, nColumns):#nGoals):
    learnSet = []
    mat = np.empty((nRows, nColumns),dtype=object) 
    for i in range(nRows):
        for j in range(nColumns):
            mat[i][j]=[]    
    
    for i in range(nRows):
        for j in range(nColumns):
            if(len(M[i][j]) > 0):
                aux = filter_path_vec(M[i][j])
            for m in range(len(aux)):
                mat[i][j].append(aux[m])
            for k in range(len(aux)):
                learnSet.append(aux[k])
    return mat, learnSet
    
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
    
    
""" GOAL RELATED FUNCTIONS """

def startGoal(p,goals):
    x, y = p.x[0], p.y[0]
    for i in range(len(goals)):
        if isInArea(x,y,goals[i]):
            return i    
    return -1
    
def get_euclidean_goal_distance(goals, nGoals):
    mat = []
    for i in range(nGoals):
        row = []
        p = middle_of_area(goals[i])
        for j in range(nGoals):            
            q = middle_of_area(goals[j])
            d = np.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)
            row.append(d)
        mat.append(row)
    return mat
    
def get_goal_sequence(p, goals):
    g = []
    for i in range(len(p.x)):
        for j in range(len(goals)):
            xy = [p.x[i], p.y[i]]
            if isInArea(xy, goals[j])==1:
                if len(g) == 0:
                    g.append(j)
                else:
                    if j != g[len(g)-1]:
                        g.append(j)
    return g
    
def getGoalSeqSet(vec,goals):
    x,y,z = get_data_from_paths(vec,"length")
    g = []
    for i in range(len(vec)):
        gi = get_goal_sequence(vec[i],goals)
        g.append(gi)
        
    return g

def getMultigoalPaths(paths,goals):
    N = len(paths)
    p = []
    g = []
    for i in range(N):
        gi = get_goal_sequence(paths[i],goals)
        if len(gi) > 2:
            p.append(i)
            g.append(gi)
    return p, g
    
def break_multigoal_path(multigoalPath, goalVec, goals):
    newPath = []
    p = multigoalPath
    g = goalVec
    newX, newY, newT = [], [], []
    goalInd = 0
    for j in range(len(p.x)):
        newX.append(p.x[j])
        newY.append(p.y[j])
        newT.append(p.t[j])
        
        if goalInd < len(g)-1:
            nextgoal = g[goalInd+1]
            xy = [p.x[j], p.y[j]]
            if isInArea(xy,goals[nextgoal]):
                new = path.path(newT,newX,newY)
                newPath.append(new)
                newX, newY, newT = [p.x[j]], [p.y[j]], [p.t[j]]
                goalInd += 1
    return newPath 
      

def get_goal_center_and_boundaries(goal):
    points = []
    p = middle_of_area(goal)
    points.append(p)
    lenX = goal[len(goal) -2] - goal[0]
    lenY = goal[len(goal) -1] - goal[1]
    q1 = [p[0]-lenX/2, p[1]]
    q2 = [p[0], p[1]+lenY/2]
    q3 = [p[0]+lenX/2, p[1]]
    q4 = [p[0], p[1]-lenY/2]
    points.append(q1)
    points.append(q2)
    points.append(q3)
    points.append(q4)
    return points
    
"""DATA RELATED FUNCTIONS"""

def histogram(paths,flag):
    if flag == "duration":
        vec = get_paths_duration(paths)
    if flag == "length":
        vec = get_paths_arcLength(paths)
    _max = max(vec)
    # Taking bins of size 10
    numBins = int(_max/10)+1
    h = np.histogram(vec, bins = numBins)
    x = []
    ymin = []
    ymax = []
    for i in range(len(h[0])):
        x.append(h[1][i])
        ymin.append(0)
        ymax.append(h[0][i])
    plt.vlines(x,ymin,ymax,colors='b',linestyles='solid')
    
def get_paths_duration(paths):
    x,y,z = get_data_from_paths(paths,"time")
    t = []
    for i in range(len(paths)):
        N = len(z[i])
        t.append(z[i][N-1])
    
    return t

#Recibe un vector de trayectorias y regresa un vector con las longitudes de arco correspondientes
def get_paths_arcLength(paths):
    x,y,z = get_data_from_paths(paths,"length")
    l = []
    for i in range(len(paths)):
        N = len(z[i])
        l.append(z[i][N-1])
    
    return l
    
#calcula la longitud de arco de un conjunto de puntos (x,y)
def arclength(x,y):
    l = [0]
    for i in range(len(x)):
        if i > 0:
            l.append(np.sqrt( (x[i]-x[i-1])**2 + (y[i]-y[i-1])**2 ) )
    for i in range(len(x)):
        if(i>0):
            l[i] = l[i] +l[i-1]
    return l

#matriz de goalsxgoalsxN_ij
#Regresa una matriz con la arc-len primedio de las trayectorias de g_i a g_j
def get_mean_length(M, nGoals):
    mat = []
    for i in range(nGoals):
        row = []
        for j in range(nGoals):
            if(len(M[i][j]) > 0):        
                arclen = get_paths_arcLength(M[i][j])    
                m = np.median(arclen)
            else:
                m = 0
            row.append(m)
        mat.append(row)
    return mat
    
#Unidad de distancia que camina una persona por unidad de distancia euclidiana
def get_distance_unit(mean, euclidean, nGoals):
    mat = []    
    for i in range(nGoals):
        row = []
        for j in range(nGoals):
            if(euclidean[i][j] == 0 or mean[i][j] == 0):
                u = 1
            else:
                u = mean[i][j]/euclidean[i][j]
            row.append(u)
        mat.append(row)
        
    sumUnit = 0.
    for i in range(nGoals):
        for j in range(nGoals):
            sumUnit += mat[i][j]
    unit = sumUnit / nGoals**2
    return mat, unit

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
    return unit
    
#Regresa una matriz con la probabilidad de ir de g_i a g_j en cada entrada
def prior_probability_matrix(pathMat, nGoals):
    priorMat = []
    for i in range(nGoals):
        p = []
        paths_i = 0.
        for j in range(nGoals):
            paths_i += len(pathMat[i][j])
        
        for j in range(nGoals):
            if paths_i == 0:
                p.append(0.)
            else:
                p.append(float(len(pathMat[i][j])/paths_i))
        priorMat.append(p)
        
    return priorMat

#regresa la duracion minima y maxima de un conjunto de trayectorias
def get_min_and_max_Duration(paths):
    n = len(paths)
    duration = np.zeros(n)
    maxDuration = 0
    minDuration = 10000
    
    for i in range(n):
        duration[i] = paths[i].duration
        if(duration[i] > maxDuration):
            # Determine max. duration
            maxDuration = duration[i]
        if(duration[i] < minDuration):
            # Determine min. duration
            minDuration = duration[i]
    return duration, minDuration, maxDuration
    
def get_min_and_max_arcLength(paths):
    n = len(paths)
    arcLen = []
    maxl = 0
    minl = 10000
    
    for i in range(n):
        arcLen.append(paths[i].length)
        if(arcLen[i] > maxl):
            maxl = arcLen[i]
        if(arcLen[i] < minl):
            minl = arcLen[i]   
    return arcLen, minl, maxl

def get_pedestrian_average_speed(paths):
    speed, validPaths = 0., 0
    for i in range(len(paths)):
        if paths[i].duration > 0:
            speed += paths[i].length / paths[i].duration
            validPaths += 1
    avSpeed = speed/ validPaths
    return avSpeed

""" HELPFUL FUNCTIONS"""    
def equal(vx,vy,x,y):
    N = len(vx)
    if N == 0:
        return 0
    
    if vx[N-1] == x and vy[N-1] == y:
        return 1
    else:
        return 0
        
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
        
def euclidean_distance(pointA, pointB):
    dist = math.sqrt( (pointA[0]-pointB[0])**2 + (pointA[1]-pointB[1])**2 )
    return dist

def middle_of_area(rectangle):
    dx, dy = rectangle[6]-rectangle[0], rectangle[7]-rectangle[1]
    middle = [rectangle[0] + dx/2., rectangle[1] + dy/2.]
    return middle
    
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
        

"""ERROR FUNCTIONS"""
def mean_error(u,v):
    error = 0.
    for i in range(len(u)):   
        error += math.sqrt((u[i]- v[i])**2) 
    return error/len(u)

#Mean error (mx,my) de los valores reales con los predichos
def meanError(trueX, trueY, predX, predY):
    e = [0,0]
    lp, l = len(predX), len(trueX)
    for i in range(lp):
        e[0] += abs(trueX[l-1-i]-predX[lp-1-i])
        e[1] += abs(trueY[l-1-i]-predY[lp-1-i]) 
    e[0] = e[0]/len(predX)
    e[1] = e[1]/len(predY)   
    return e
    
def geometricError(trueX, trueY, predX, predY):
    e = 0
    lp, l = len(predX), len(trueX)
    for i in range(lp):
        e += math.sqrt((trueX[l-1-i]-predX[lp-1-i])**2 + (trueY[l-1-i]-predY[lp-1-i])**2)
    return e
    
def geomError(meanError):
    Ex, Ey = 0, 0
    for i in range(len(meanError)):
        Ex += meanError[i][0]
        Ey += meanError[i][1]
    Ex = Ex/len(meanError) 
    Ey = Ey/len(meanError)
    return math.sqrt(Ex**2+Ey**2)

def getError(trueX, trueY, predX,predY):
    for i in range(len(predX)):
        if(i == 0):
            error = [meanError(trueX[i],trueY[i],predX[i],predY[i])]
        else:
            error.append(meanError(trueX[i],trueY[i],predX[i],predY[i]))    
    return error
   
#Average L2 distance between ground truth and our prediction
def average_displacement_error(true_XY, prediction_XY):
    error = 0.
    trueX, trueY = true_XY[0], true_XY[1]
    predictionX, predictionY = prediction_XY[0], prediction_XY[1]
    l = min(len(trueX),len(predictionX))
    for i in range(l):
        error += math.sqrt((trueX[i]-predictionX[i])**2 + (trueY[i]-predictionY[i])**2)
    if(l>0):
        error = error/l
    return error
    
#The distance between the predicted final destination and the true final destination
def final_displacement_error(final, predicted_final):
    error = math.sqrt((final[0]-predicted_final[0])**2 + (final[1]-predicted_final[1])**2)
    return error
    
