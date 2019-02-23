"""
Libreria de funciones para Gaussian Processes for Regression 
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.optimize import minimize
import kernels
import path
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse
from copy import copy
import random

""" READ DATA """
def get_paths_from_file(path_file,areas):
    
    paths, multigoal_paths = [],[] 
    # Open file
    with open(path_file) as f:
        # Each line should contain a path
        for line in f:
            auxX, auxY, auxT = [],[],[]
            # Split the line into sub-strings
            data    = line.split()
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
    
""" FILTER PATHS """    
#Calcula la mediana m, y la desviacion estandar s de un vector de paths
#si alguna trayectoria difiere de la mediana en 4s o mas, se descarta
def filter_path_vec(vec):
    arclen = getArcLength(vec)    
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

#matriz de goalsxgoalsxN_ij
"""Regresa una matriz con la arc-len primedio de las trayectorias de g_i a g_j"""
def get_mean_length(M, nGoals):
    mat = []
    for i in range(nGoals):
        row = []
        for j in range(nGoals):
            if(len(M[i][j]) > 0):        
                arclen = getArcLength(M[i][j])    
                m = np.median(arclen)
            else:
                m = 0
            row.append(m)
        mat.append(row)
    return mat
    
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
    
def get_unit(mean, euclidean, nGoals):
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
    return mat

def get_goal_sequence(p, goals):
    g = []
    for i in range(len(p.x)):
        for j in range(len(goals)):
            if isInArea(p.x[i], p.y[i], goals[j])==1:
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
        gi = getGoalSeq(paths[i],goals)
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
            if isInArea(p.x[j],p.y[j],goals[nextgoal]):
                new = path.path(newT,newX,newY)
                newPath.append(new)
                newX, newY, newT = [p.x[j]], [p.y[j]], [p.t[j]]
                goalInd += 1
    return newPath  
    
#******************************************************************************#
""" REGRESSION FUNCTIONS """
# The main regression function 
def regression(x,y,xnew,kernel):  #regresion sin recibir los parametros, el kernel ya los trae fijos  
    n = len(x) # Number of observed data
    # Compute K, k and c
    K  = np.zeros((n,n))
    k = np.zeros(n)
    c = 0
    # Fill in K
    for i in range(n):
        for j in range(n):
            K[i][j] = kernel.k(x[i],x[j])
    # Fill in k
    for i in range(n):
        k[i] = kernel.k(xnew,x[i])
    # compute c       
    c = kernel.k(xnew,xnew)
    # Estimate the mean
    K_1 = inv(K)
    ynew = k.dot(K_1.dot(y)) 
    # Estimate the variance
    K_1kt = K_1.dot(k.transpose())
    kK_1kt = k.dot(K_1kt)
    var = c - kK_1kt
    return ynew, var 
    
def equal(vx,vy,x,y):
    N = len(vx)
    if N == 0:
        return 0
    
    if vx[N-1] == x and vy[N-1] == y:
        return 1
    else:
        return 0

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
        
#******************************************************************************#    
""" LEARNING """
nsigma = 8.0 #error de las observaciones 
#Parametros: theta, vector de vectores: x,y
    
def setKernel(name):
    if(name == "squaredExponential"):
        parameters = [80., 80.] #{sigma_f, l}
        kernel = kernels.squaredExponentialKernel(parameters[0],parameters[1],nsigma)
    elif(name == "_combined"): #kernel de Trautman
        parameters = [60., 80., 80.]#{gamma, sMatern, lMatern}, sigma
        kernel = kernels._combinedKernel(parameters[0],parameters[1],parameters[2],nsigma)
    elif(name == "combined"): #kernel simplificado para la optimizacion
        parameters = [50., 50.]#{sMatern, lMatern}, sigma
        kernel = kernels.combinedKernel(parameters[0],parameters[1],nsigma)
    elif(name == "exponential"):
        parameters = [80., 80.] #{s,l}
        kernel = kernels.exponentialKernel(parameters[0],parameters[1])
    elif(name == "gammaExponential"):
        parameters = [80., 8.] #{l, gamma}
        kernel = kernels.gammaExponentialKernel(parameters[0],parameters[1])
    elif(name == "rationalQuadratic"):
        parameters = [10., 5.] #{l, alpha}
        kernel = kernels.rationalQuadraticKernel(parameters[0],parameters[1],nsigma)
    elif(name == "SQ"):
        parameters = [80.0, nsigma] #{l, sigma}
        kernel = kernels.SQKernel(parameters[0],parameters[1])
    elif(name == "expCombined"):
        parameters = [80., 80.] #{s,l}
        kernel = kernels.expKernel(parameters[0],parameters[1],nsigma)
    
    return kernel, parameters

#ngoals = numero de areas de interes
def create_kernel_matrix_(kerType, ngoals):
    kerMatrix = []  
    parameters = []
    for i in range(ngoals):
        aux = []
        auxP = []
        for j in range(ngoals):
            kernel, theta = setKernel(kerType) 
            aux.append(kernel)
            auxP.append(theta)
        kerMatrix.append(aux)
        parameters.append(auxP)
    return kerMatrix, parameters 
    
    
def create_kernel_matrix(kerType, rows, columns):
    kerMatrix = []  
    parameters = []
    for i in range(rows):
        aux = []
        auxP = []
        for j in range(columns):
            kernel, theta = setKernel(kerType) 
            aux.append(kernel)
            auxP.append(theta)
        kerMatrix.append(aux)
        parameters.append(auxP)
    return kerMatrix, parameters 
    
def log_p(theta,x,y,kernel):
    kernel.setParameters(theta)
    n = len(x)
    K = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            K[i][j] = kernel.k(x[i],x[j])
    invK = inv(K)    
    invKy = invK.dot(y)  
    yKy = np.inner(y,invKy)
    detK = np.linalg.det(K)
    
    val = (-1/2.)*yKy-(1/2.)*np.log(detK)-(n/2.)*np.log(2*math.pi)
    return val    

def sum_log_p(theta,x,y,kernel):
    size = len(x)
    s = 0
    for i in range(size):
        s += log_p(theta,x[i],y[i],kernel)
    return s
  
def neg_sum_log_p(theta,x,y,kernel):
    s = sum_log_p(theta,x,y,kernel)
    return (-1.)*s
    
#elegir parametros que funcionen para un conjunto de trayectorias
def optimize_kernel_parameters_XY(t,x,y,theta,kernel):#creo que eran 14 it pobar
    parametersX = minimize(neg_sum_log_p,theta,(t,x,kernel),method='Nelder-Mead', options={'maxiter':25,'disp': False})
    parametersY = minimize(neg_sum_log_p,theta,(t,y,kernel),method='Nelder-Mead', options={'maxiter':25,'disp': False})
        
    return parametersX.x, parametersY.x
  
def learning(l,x,y,kernel,parameters):
    thetaX, thetaY = optimize_kernel_parameters_XY(l,x,y,parameters,kernel)
    
    return thetaX, thetaY

def optimize_parameters_between_2goals(learnSet, kernelMat, parametersMat, startGoal, finishGoal):
    kernelX = kernelMat[startGoal][finishGoal]
    kernelY = kernelMat[startGoal][finishGoal]
    
    x,y,z = get_data_from_paths(learnSet,"length") 
    k = kernelMat[startGoal][finishGoal]    
    parameters = parametersMat[startGoal][finishGoal]
    paramX, paramY = learning(z,x,y,k,parameters)
    kernelX.setParameters(paramX)
    kernelY.setParameters(paramY)
    print("x:",paramX)
    print("y:",paramY)
    
#Aprendizaje para cada par de trayectorias 
#Regresa una matriz de kernels con los parametros optimizados  
#learnSet = [[(start_i,goal_j)]], kernelMat = [[k_ij]], thetaMat = [[parametros del kernel_ij]]
def optimize_parameters_between_goals_(learnSet, parametersMat, nGoals):
    #parameters = []
    kernelMatX, parametersX = create_kernel_matrix("combined", nGoals)
    kernelMatY, parametersY = create_kernel_matrix("combined", nGoals)#,  kernelMat, kernelMat 
    for i in range(nGoals):
        r = []
        for j in range(nGoals):
            #print("[",i,"][",j,"]")
            paths = learnSet[i][j]
            if len(paths) > 0:
                x,y,z = get_data_from_paths(paths,"length") 
                ker, theta = setKernel("combined")       #checar si funciona         
                #ker = #kernelMat[i][j]
                #theta = ker.get_parameters()#parametersMat[i][j]
                
                thetaX, thetaY = learning(z,x,y,ker,theta)
                print("[",i,"][",j,"]")
                print("x: ",thetaX)
                print("y: ",thetaY)
                kernelMatX[i][j].setParameters(thetaX)                
                kernelMatY[i][j].setParameters(thetaY)
                #print("after setting parameters")
                #print("[",kernelMatX[i][j].s,",",kernelMatX[i][j].l,"]")
                #print("[",kernelMatY[i][j].s,",",kernelMatY[i][j].l,"]")
                r.append([thetaX, thetaY])                
        #parameters.append(r)
    return kernelMatX, kernelMatY
    
def optimize_parameters_between_goals(learnSet, parametersMat, rows, columns):
    #parameters = []
    kernelMatX, parametersX = create_kernel_matrix("combined", rows, columns)
    kernelMatY, parametersY = create_kernel_matrix("combined", rows, columns)
    for i in range(rows):
        r = []
        for j in range(columns):
            #print("[",i,"][",j,"]")
            paths = learnSet[i][j]
            if len(paths) > 0:
                x,y,z = get_data_from_paths(paths,"length") 
                ker, theta = setKernel("combined")       #checar si funciona         
                #ker = #kernelMat[i][j]
                #theta = ker.get_parameters()#parametersMat[i][j]
                
                thetaX, thetaY = learning(z,x,y,ker,theta)
                print("[",i,"][",j,"]")
                print("x: ",thetaX)
                print("y: ",thetaY)
                kernelMatX[i][j].setParameters(thetaX)                
                kernelMatY[i][j].setParameters(thetaY)
                print("after setting parameters")
                print("[",kernelMatX[i][j].s,",",kernelMatX[i][j].l,"]")
                print("[",kernelMatY[i][j].s,",",kernelMatY[i][j].l,"]")
                r.append([thetaX, thetaY])                
        #parameters.append(r)
    return kernelMatX, kernelMatY
    
#recibe una matriz de kernel [[kij]], con parametros [gamma,s,l]
def _write_parameters(matrix,nGoals,flag):
    if flag == "x": 
        f = open("parameters_x.txt","w")
    if flag == "y": 
        f = open("parameters_y.txt","w")

    for i in range(nGoals):
        for j in range(nGoals):
            ker = matrix[i][j]
            gamma = "%d "%(ker.gamma)
            f.write(gamma)    
            s = "%d "%(ker.s)
            f.write(s)    
            l = "%d "%(ker.l)
            f.write(l)    
            skip = "\n"
            f.write(skip) 
    f.close()  
    
#recibe una matriz de kernel [[kij]], con parametros [s,l]
def write_parameters(matrix,rows,columns,fileName):
    #if flag == "x": 
    #    f = open("parameters_x.txt","w")
    #if flag == "y": 
    #    f = open("parameters_y.txt","w")
    f = open(fileName,"w")
    for i in range(rows):
        for j in range(columns):
            ker = matrix[i][j]
            s = "%d "%(ker.s)
            f.write(s)    
            l = "%d "%(ker.l)
            f.write(l)    
            skip = "\n"
            f.write(skip) 
    f.close()  
    
def read_and_set_parameters(file_name, rows, columns, nParameters):
    matrix, parametersMat = create_kernel_matrix("combined", rows, columns)
    f = open(file_name,'r')
       
    for i in range(rows):
        for j in range(columns):
            parameters = []#el kernel combinado usa 2 parametros: [s,l] 
            line = f.readline()
            parameters_str = line.split()
            for k in range(nParameters):
                parameters.append(float(parameters_str[k]))
            matrix[i][j].setParameters(parameters)
        
    f.close()
    return matrix
    
    
""" PREDICTION """
def get_known_set(x,y,l,knownN):
    trueX, trueY, trueL = [],[],[]
    knownN = int(knownN)
    for j in range(knownN): #numero de datos conocidos
        trueX.append(x[j])
        trueY.append(y[j])
        trueL.append(l[j])
        
    return trueX, trueY, trueL
  
def get_goal_likelihood(knownX,knownY,knownL,startG,finishG,goals,unitMat,kernelMatX,kernelMatY):  
    _knownX = knownX.copy()        
    _knownY = knownY.copy()
    _knownL = knownL.copy()
    
    stepsToCompare = 5
    trueX, trueY, predSet = [], [], []
    for i in range(stepsToCompare):
        valX = _knownX.pop()
        trueX.append(valX)
        valY = _knownY.pop()
        trueY.append(valY)
        step = _knownL.pop()
        predSet.append(step)
    
    n = len(_knownX)
    finish_xy = middle_of_area(goals[i])
    _knownX.append(finish_xy[0])
    _knownY.append(finish_xy[1])
    dist = math.sqrt( (_knownX[n-1] - finish_xy[0])**2 + (_knownY[n-1] - finish_xy[1])**2 )
    unit = unitMat[startG][i]        
    lastL = knownL[n-1] + dist*unit
    _knownL.append(lastL)
    kernelX = kernelMatX[startG][i]
    kernelY = kernelMatY[startG][i]
    predX, predY, vx, vy = prediction_XY(_knownX, _knownY, _knownL, predSet, kernelX, kernelY)
    error = average_displacement_error([trueX,trueY],[predX,predY])
    
    return error
        
#Elige m puntos (x,y) de un area usando muestreo uniforme
def uniform_sampling_2D(m, goal):
    _x, _y = [], []
    xmin, xmax = goal[0], goal[2]
    ymin, ymax = goal[1], goal[len(goal)-1] 
    
    for i  in range(m):
        t = random.uniform(0,1.)
        val = (1.-t)*xmin + t*xmax
        _x.append(val)
        r = random.uniform(0,1.)
        val = (1.-r)*ymin + r*ymax
        _y.append(val)
        
    return _x, _y

def uniform_sampling_1D(m, goal, axis):
    _x, _y = [], []
    xmin, xmax = goal[0], goal[2]
    ymin, ymax = goal[1], goal[len(goal)-1] 
    
    for i  in range(m):
        t = random.uniform(0,1.)
        if(axis == 'x'):
            val = (1.-t)*xmin + t*xmax
            _x.append(val)
            _y.append( ymin + (ymax-ymin)/2 )
        if(axis == 'y'):
            val = (1.-t)*ymin + t*ymax
            _y.append(val)
            _x.append( xmin + (xmax-xmin)/2 )
        
    return _x, _y, axis #devuelve axis para las pruebas de single GP

def get_finish_point(knownX, knownY, knownL, finishGoal, goals, kernelX, kernelY, unit, samplingAxis):
    n = len(knownX)
    m = 9 #numero de muestras
    _x, _y, flag = uniform_sampling_1D(m, goals[finishGoal], samplingAxis[finishGoal])
    k = 3
    
    if(n < 2*k):
        return middle_of_area(goals[finishGoal])
        
    _knownX, _knownY, _knownL = [], [], []
    for i in range(n-k):
        _knownX.append(knownX[i])
        _knownY.append(knownY[i])
        _knownL.append(knownL[i])

    predSet, trueX, trueY = [], [], []
    for i in range(k):
        predSet.append(knownL[n-k+i])
        trueX.append(knownX[n-k+i])
        trueY.append(knownY[n-k+i])
        
    error = []
    for i in range(m):
        auxX = _knownX.copy()
        auxY = _knownY.copy()
        auxL = _knownL.copy()
        auxX.append(_x[i])
        auxY.append(_y[i])
        dist = math.sqrt( (knownX[n-1] - _x[i])**2 + (knownY[n-1] - _y[i])**2 )
        lastL = knownL[n-1] + dist*unit
        auxL.append(lastL)
        predX, predY, vx, vy = prediction_XY(auxX, auxY, auxL, predSet, kernelX, kernelY)
        #error.append(geometricError(trueX,trueY,predX,predY))
        error.append(average_displacement_error([trueX,trueY],[predX,predY]))
    #encuentra el punto que genera el error minimo
    min_id, min_error = 0, error[0]
    for i in range(m):
        if(error[i] < min_error):
            min_error = error[i]
            min_id = i        
    return [_x[min_id], _y[min_id]]

def prediction_error_1D(trueX, trueY, prediction, flag):
    error = 0.0
    for i in range(len(prediction) ):
        if flag == 'x':
            error += abs(trueX[i] - prediction[i])
        if flag == 'y':
            error += abs(trueY[i] - prediction[i])
    return error

def get_finish_point_singleGP(knownX, knownY, knownL, finishGoal, goals, kernelX, kernelY, unit, img, samplingAxis):
    n = len(knownX)
    m = 10 #numero de muestras
    _x, _y, flag  = uniform_sampling_1D(m, goals[finishGoal], samplingAxis[finishGoal])
    k = 5 #numero de puntos por comparar
    _knownX, _knownY = [], []
    for i in range(n-k):
        _knownX.append(knownX[i])
        _knownY.append(knownY[i])
        
    trueX, trueY = [], []
    for i in range(k):
        trueX.append(knownX[n-k+i])
        trueY.append(knownY[n-k+i])
        
    error = []
    for i in range(m):
        auxX = _knownX.copy()
        auxY = _knownY.copy()
        auxX.append(_x[i])
        auxY.append(_y[i])
        if flag == 'y': #x(y)
            predSet = trueY.copy()
            prediction, var = estimate_new_set_of_values(auxY,auxX,predSet,kernelX)
            plot_sampling_prediction(img,knownX,knownY,n-k,prediction,trueY,var,var,[_x[i],_y[i]])
            error.append(prediction_error_1D(trueX, trueY, prediction, 'x'))
        if flag == 'x': #y(x)
            predSet = trueX.copy()
            prediction, var = estimate_new_set_of_values(auxX,auxY,predSet,kernelY)
            plot_sampling_prediction(img,knownX,knownY,n-k,trueX,prediction,var,var,[_x[i],_y[i]])
            error.append(prediction_error_1D(trueX, trueY, prediction, 'y')) #y(x)

    #encuentra el punto que genera el error minimo
    min_id, min_error = 0, error[0]
    for i in range(m):
        if(error[i] < min_error):
            min_error = error[i]
            min_id = i
    return [_x[min_id], _y[min_id]]
    

def get_prediction_set_from_data(z,knownN):
    N = len(z)
    newZ = []
    knownN = int(knownN)
    for j in range(knownN-1, N): #numero de datos conocidos
        newZ.append(z[j])
    return newZ

#usa una unidad de distancia segun el promedio de la arc-len de las trayectorias
#start, last know (x,y,l), indices del los goals de inicio y fin, unitMat, numero de pasos
def get_prediction_set(lastKnownPoint, finishPoint, unit, stepUnit):
    x, y, l = lastKnownPoint[0], lastKnownPoint[1], lastKnownPoint[2] 
    _x, _y = finishPoint[0], finishPoint[1]
    dist = math.sqrt( (_x-x)**2 + (_y-y)**2 )
    steps = int(dist*stepUnit)
    newset = []
    if(steps > 0):    
        step = dist/float(steps)
        for i in range(steps+1):
            newset.append( l + i*step*unit )
        
    return newset, l + dist*unit     
    
def get_arclen_to_finish_point(lastKnownPoint, finishPoint, unit):
    x, y, l = lastKnownPoint[0], lastKnownPoint[1], lastKnownPoint[2] 
    _x, _y = finishPoint[0], finishPoint[1]
    dist = math.sqrt( (_x-x)**2 + (_y-y)**2 )
    
    return l + dist*unit

def get_subgoals_center_and_size(nSubgoals, goal, axis):
    goalX = goal[len(goal) -2] - goal[0]
    goalY = goal[len(goal) -1] - goal[1]
    goalCenterX = goal[0]+ goalX/2    
    goalCenterY = goal[1]+ goalY/2
    
    subgoalsCenter = []
    subgoalX, subgoalY = 0,0
    if axis == 'x':
        subgoalX = goalX/nSubgoals
        subgoalY = goalY
        _x = goal[0]
        _y = goalCenterY
        for i in range(nSubgoals):
            subgoalsCenter.append( [_x+subgoalX/2, _y] )
            _x += subgoalX
            
    if axis == 'y':
        subgoalX = goalX
        subgoalY = goalY/nSubgoals
        _x = goalCenterX
        _y = goal[1]
        for i in range(nSubgoals):
            subgoalsCenter.append( [_x, _y+subgoalY/2] )
            _y += subgoalY
            
    return subgoalsCenter, [subgoalX, subgoalY]

#Recibe los puntos conocidos (x,y) el conjunto de puntos por predecir newX.
def estimate_new_set_of_values(known_x,known_y,newX,kernel):
    lenNew = len(newX)
    predicted_y, variance_y = [], []
    
    for i in range(lenNew):
        new_y, var_y = regression(known_x,known_y,newX[i],kernel)
        predicted_y.append(new_y)
        variance_y.append(var_y)
    
    return predicted_y, variance_y

#prediccion en X y en Y
#Recibe los datos conocidos (x,y,z) y los puntos para la regresion.
def prediction_XY(x, y, z, newZ, kernel_x, kernel_y):#prediction 
    new_x, var_x = estimate_new_set_of_values(z,x,newZ,kernel_x)#prediccion para x
    new_y, var_y = estimate_new_set_of_values(z,y,newZ,kernel_y)#prediccion para y

    return new_x, new_y, var_x, var_y
    
#necesita recibir el kernel para X y el kernel para Y
#Recibe los datos conocidos [(x,y,z)] y los puntos para la regresion. x, y, z,... son vectores de vectores
def prediction_XY_of_set(x, y, z, newZ, kernel_x, kernel_y):#prediction 
    numPred = len(newZ)   
    predicted_x, predicted_y = [], []
    variance_x, variance_y = [], []
    for i in range(numPred):
        new_x, var_x = estimate_new_set_of_values(z[i],x[i],newZ[i],kernel_x)#prediccion para x
        predicted_x.append(new_x)
        variance_x.append(var_x) 
        
        new_y, var_y = estimate_new_set_of_values(z[i],y[i],newZ[i],kernel_y)#prediccion para y
        predicted_y.append(new_y)
        variance_y.append(var_y)
    return predicted_x, predicted_y,variance_x, variance_y


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

#******************************************************************************#

def middle_of_area(rectangle):
    dx, dy = rectangle[6]-rectangle[0], rectangle[7]-rectangle[1]
    middle = [rectangle[0] + dx/2., rectangle[1] + dy/2.]
    return middle
    
def get_goals_likelihood(x,y,l,start,kernelMat_x,kernelMat_y, goals, nGoals):
    known_data = 1
    likelihood = []
    var, error =[], []

    known_x, known_y, known_l = [], [],[]
    for i in range(known_data):
        known_x.append(x[i])
        known_y.append(y[i])
        known_l.append(l[i])
        
    N = len(x)
    l_ = l[N-1]
    sum_likelihood = 0.
    for i in range(nGoals):
        kernel_x = kernelMat_x[start][i]
        kernel_y = kernelMat_y[start][i]
        
        end = middle_of_area(goals[i])
        dist = math.sqrt( (end[0]-x[N-1])**2 + (end[1]-y[N-1])**2 )
        aux_known_l = copy(known_l)
        aux_known_x = copy(known_x) 
        aux_known_y = copy(known_y)
        
        aux_known_l.append(l[N-1]+dist)
        aux_known_x.append(end[0])
        aux_known_y.append(end[1])
        
        x_, var_x = regression(aux_known_l,aux_known_x,l_,kernel_x)
        y_, var_y = regression(aux_known_l,aux_known_y,l_,kernel_y)
        var.append([var_x,var_y])
        likelihood_gi = math.exp( (-1./2.)*( math.fabs(x_ - x[N-1])/var_x + math.fabs(y_ - y[N-1])/var_y )  )
        likelihood.append(likelihood_gi)
        sum_likelihood += likelihood_gi
        error_x = math.fabs(x_ - x[N-1])
        error_y = math.fabs(y_ - y[N-1])
        error.append(math.sqrt(error_x*2 + error_y**2))
       
    for i in range(nGoals): 
        likelihood[i] = likelihood[i]/sum_likelihood
    
    return likelihood, error

# Recibe un punto (x,y) y un area de interes R
def isInArea(x,y,R):
    if(x >= R[0] and x <= R[2]):
        if(y >= R[1] and y <= R[len(R)-1]):
            return 1
        else:
            return 0
    else: 
        return 0

def startGoal(p,goals):
    x, y = p.x[0], p.y[0]
    for i in range(len(goals)):
        if isInArea(x,y,goals[i]):
            return i

#Recibe un vector de trayectorias y regresa un vector con las longitudes de arco correspondientes
def getArcLength(paths):
    x,y,z = get_data_from_paths(paths,"length")
    l = []
    for i in range(len(paths)):
        N = len(z[i])
        l.append(z[i][N-1])
    
    return l

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
