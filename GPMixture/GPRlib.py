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
from dataManagement import *
import dataManagement
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse
from copy import copy
import random

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
            K[i][j] = kernel(x[i],x[j])
    # Fill in k
    for i in range(n):
        k[i] = kernel(xnew,x[i])
    # compute c
    c = kernel(xnew,xnew)
    # Estimate the mean
    #print("[kernel Matrix]\n",K)
    #print("[kernel vector]\n",k)

    K_1 = inv(K)
    ynew = k.dot(K_1.dot(y))
    # Estimate the variance
    K_1kt = K_1.dot(k.transpose())
    kK_1kt = k.dot(K_1kt)
    var = c - kK_1kt
    return ynew, var



#******************************************************************************#
""" LEARNING """
nsigma = 7.50 #error de las observaciones
#Parametros: theta, vector de vectores: x,y

def setKernel(name):
    if(name == "squaredExponential"):
        parameters = [80., 80.] #{sigma_f, l}
        kernel = kernels.squaredExponentialKernel(parameters[0],parameters[1],nsigma)
    elif(name == "combined"): #kernel de Trautman
        parameters = [60., 80., 80.]#{gamma, sMatern, lMatern}, sigma
        kernel = kernels.combinedKernel(parameters[0],parameters[1],parameters[2],nsigma)
    elif(name == "combinedModified"): #kernel simplificado para la optimizacion
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
    elif(name == "linePriorCombined"):
        parameters = [60., 60., 80., 80.]#{gamma, sMatern, lMatern}, sigma
        kernel = kernels.linePriorCombinedKernel(parameters[0],parameters[1],parameters[2],parameters[3],nsigma)


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
            K[i][j] = kernel(x[i],x[j])
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
    parametersX = minimize(neg_sum_log_p,theta,(t,x,kernel),method='Nelder-Mead', options={'maxiter':16,'disp': False})
    parametersY = minimize(neg_sum_log_p,theta,(t,y,kernel),method='Nelder-Mead', options={'maxiter':16,'disp': False})

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

#Aprendizaje para cada par de trayectorias
#Regresa una matriz de kernels con los parametros optimizados
#learnSet = [[(start_i,goal_j)]], kernelMat = [[k_ij]], thetaMat = [[parametros del kernel_ij]]
def optimize_parameters_between_goals_(kernelType, learnSet, nGoals):
    #parameters = []
    kernelMatX, parametersX = create_kernel_matrix(kernelType, nGoals)
    kernelMatY, parametersY = create_kernel_matrix(kernelType, nGoals)#,  kernelMat, kernelMat
    for i in range(nGoals):
        r = []
        for j in range(nGoals):
            paths = learnSet[i][j]
            if len(paths) > 0:
                x,y,z = get_data_from_paths(paths,"length")
                ker, theta = setKernel(kernelType)
                thetaX, thetaY = learning(z,x,y,ker,theta)
                print("[",i,"][",j,"]")
                print("x: ",thetaX)
                print("y: ",thetaY)
                kernelMatX[i][j].setParameters(thetaX)
                kernelMatY[i][j].setParameters(thetaY)
                r.append([thetaX, thetaY])
        #parameters.append(r)
    return kernelMatX, kernelMatY

def optimize_parameters_between_goals(kernelType, learnSet, rows, columns):
    kernelMatX, parametersX = create_kernel_matrix(kernelType, rows, columns)
    kernelMatY, parametersY = create_kernel_matrix(kernelType, rows, columns)
    for i in range(rows):
        r = []
        for j in range(columns):
            paths = learnSet[i][j]
            if len(paths) > 0:
                x,y,z = get_data_from_paths(paths,"length")
                ker, theta = setKernel(kernelType)
                thetaX, thetaY = learning(z,x,y,ker,theta)
                print("[",i,"][",j,"]")
                print("x: ",thetaX)
                print("y: ",thetaY)
                kernelMatX[i][j].setParameters(thetaX)
                kernelMatY[i][j].setParameters(thetaY)
                r.append([thetaX, thetaY])
    return kernelMatX, kernelMatY

#recibe una matriz de kernel [[kij]], con parametros [gamma,s,l]
def write_squared_matrix_parameters(matrix,nGoals,flag):
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
    f = open(fileName,"w")
    for i in range(rows):
        for j in range(columns):
            ker = matrix[i][j]
            parameters = ker.get_parameters()
            for k in range(len(parameters)):
                val = "%d "%(parameters[k])
                f.write(val)
            skip = "\n"
            f.write(skip)
    f.close()

#recibe una matriz de kernel [[kij]], con parametros [s,l]
def _write_parameters(matrix,rows,columns,fileName):
    f = open(fileName,"w")
    for i in range(rows):
        for j in range(columns):
            ker = matrix[i][j]
            parameters = ker.get_parameters()
            gamma = "%d "%(ker.gamma)
            f.write(gamma)
            s = "%d "%(ker.s)
            f.write(s)
            l = "%d "%(ker.l)
            f.write(l)
            skip = "\n"
            f.write(skip)
    f.close()

def read_and_set_parameters(file_name, rows, columns, kernelType, nParameters):
    matrix, parametersMat = create_kernel_matrix(kernelType, rows, columns)
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
    trueX = x[0:knownN]
    trueY = y[0:knownN]
    trueL = l[0:knownN]
    return trueX, trueY, trueL

def get_known_data(x,y,z,knownN):
    trueX = x[0:knownN]
    trueY = y[0:knownN]
    trueZ = z[0:knownN]
    return trueX, trueY, trueZ

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
    numSamples = 9 #numero de muestras
    _x, _y, flag = uniform_sampling_1D(numSamples, goals[finishGoal], samplingAxis[finishGoal])
    k = 3          #num de puntos por comparar
    if(n < 2*k):
        return middle_of_area(goals[finishGoal])

    _knownX = knownX[0:n-k]
    _knownY = knownY[0:n-k]
    _knownL = knownL[0:n-k]

    predSet = knownL[n-k:k]
    trueX = knownX[n-k:k]
    trueY = knownY[n-k:k]

    error = []
    for i in range(numSamples):
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
    for i in range(numSamples):
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

    _knownX = knownX[0:n-k]
    _knownY = knownY[0:n-k]

    trueX = knownX[n-k:k]
    trueY = knownY[n-k:k]

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
    newZ = z[knownN-1:N]
    return newZ

#usa una unidad de distancia segun el promedio de la arc-len de las trayectorias
#start, last know (x,y,l), indices del los goals de inicio y fin, unitMat, numero de pasos
def get_prediction_set(lastKnownPoint, finishPoint, distUnit, stepUnit):
    x, y, l = lastKnownPoint[0], lastKnownPoint[1], lastKnownPoint[2]
    _x, _y = finishPoint[0], finishPoint[1]

    euclideanDist = euclidean_distance([x,y], [_x,_y])
    dist = euclideanDist*distUnit

    numSteps = int(dist*stepUnit)
    newset = []
    if(numSteps > 0):
        step = dist/float(numSteps)
        for i in range(numSteps+1):
            newset.append( l + i*step )

    return newset, l + dist

def get_prediction_set_given_size(lastKnownPoint, finishPoint, unit, steps):
    x, y, l = lastKnownPoint[0], lastKnownPoint[1], lastKnownPoint[2]
    _x, _y = finishPoint[0], finishPoint[1]
    dist = math.sqrt( (_x-x)**2 + (_y-y)**2 )
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
        #new_y, var_y = line_prior_regression(known_x,known_y,newX[i],kernel) #line prior regression
        new_y, var_y = regression(known_x,known_y,newX[i],kernel)
        predicted_y.append(new_y)
        variance_y.append(var_y)

    return predicted_y, variance_y

#prediccion en X y en Y
#Recibe los datos conocidos (x,y,z) y los puntos para la regresion.
def prediction_XY(x, y, z, newZ, kernelX, kernelY):#prediction
    newX, varX = estimate_new_set_of_values(z,x,newZ,kernelX)#prediccion para x
    newY, varY = estimate_new_set_of_values(z,y,newZ,kernelY)#prediccion para y

    return newX, newY, varX, varY

#necesita recibir el kernel para X y el kernel para Y
#Recibe los datos conocidos [(x,y,z)] y los puntos para la regresion. x, y, z,... son vectores de vectores
def prediction_XY_of_set_of_trajectories(x, y, z, newZ, kernel_x, kernel_y):#prediction
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


#Toma N-nPoints como datos conocidos y predice los ultimos nPoints, regresa el error de la prediccion
def prediction_error_of_last_known_points(nPoints,knownX,knownY,knownL,goal,unit,stepUnit,kernelX,kernelY):
    knownN = len(knownX)
    trueX = knownX[0:knownN -nPoints]
    trueY = knownY[0:knownN -nPoints]
    trueL = knownL[0:knownN -nPoints]

    finishXY = middle_of_area(goal)
    finishD = euclidean_distance([trueX[len(trueX)-1],trueY[len(trueY)-1]],finishXY)
    trueX.append(finishXY[0])
    trueY.append(finishXY[1])
    trueL.append(finishD*unit)

    lastX = knownX[knownN -nPoints: nPoints]
    lastY = knownY[knownN -nPoints: nPoints]
    predictionSet = knownL[knownN -nPoints: nPoints]

    predX, predY, varX,varY = prediction_XY(trueX,trueY,trueL, predictionSet, kernelX, kernelY)
    #print("[Prediccion]\n",predX)
    #print(predY)
    error = average_displacement_error([lastX,lastY],[predX,predY])
    #print("[Error]:",error)
    return error

#Toma la mitad de los datos observados como conocidos y predice nPoints en la mitad restante, regresa el error de la prediccion
def prediction_error_of_points_along_the_path(nPoints,knownX,knownY,knownL,goal,unit,kernelX,kernelY):
    knownN = len(knownX)
    halfN = int(knownN/2)

    trueX = knownX[0:halfN]
    trueY = knownY[0:halfN]
    trueL = knownL[0:halfN]

    finishXY = middle_of_area(goal)
    finishD = euclidean_distance([trueX[len(trueX)-1],trueY[len(trueY)-1]],finishXY)
    trueX.append(finishXY[0])
    trueY.append(finishXY[1])
    trueL.append(finishD*unit)

    d = int(halfN/nPoints)
    realX, realY, predictionSet = [],[],[]
    for i in range(nPoints):
        realX.append(knownX[halfN + i*d])
        realY.append(knownY[halfN + i*d])
        predictionSet.append(knownL[halfN + i*d])

    #predX, predY, varX,varY = prediction_XY(trueX,trueY,trueL, predictionSet, kernelX, kernelY)
    predX, predY, varX,varY = prediction_XY(trueX,trueY,trueL, predictionSet, kernelX, kernelY)

    error = average_displacement_error([realX,realY],[predX,predY])

    return error

"""ARC LENGHT TO TIME"""
def arclen_to_time(initTime,l,speed):
    t = [initTime]
    for i in range(1,len(l)):
        time_i = int(t[i-1] +(l[i]-l[i-1])/speed)
        t.append(time_i)
    return t

#******************************************************************************#
"""***REGRESSION USING LINE PRIOR ***"""
#TODO
def linear_mean(l, priorMean):
    m = priorMean[0]*l + priorMean[1]
    return m

def line_prior_regression(l,x_meanl,lnew,kernel,priorMean):  #regresion sin recibir los parametros, el kernel ya los trae fijos
    n = len(l) # Number of observed data
    # Compute K, k and c
    K  = np.zeros((n,n))
    k = np.zeros(n)
    # Fill in K
    for i in range(n):
        for j in range(n):
            K[i][j] = kernel(l[i],l[j])
    # Fill in k
    for i in range(n):
        k[i] = kernel(lnew,l[i])

    K_1 = inv(K)
    xnew = linear_mean(lnew,priorMean) + k.dot(K_1.dot(x_meanl))
    # Estimate the variance
    K_1kt = K_1.dot(k.transpose())
    kK_1kt = k.dot(K_1kt)
    var = kernel(lnew,lnew) - kK_1kt

    return xnew, var

#lp = line prior
def estimate_new_set_of_values_lp(knownL,knownX,newL,kernel,priorMean):
    lenNew = len(newL)
    predictedX, varianceX = [], []
    X_meanL = []
    for i in range(len(knownL)):
        X_meanL.append(knownX[i] - linear_mean(knownL[i], priorMean))

    for i in range(lenNew):
        val, var = line_prior_regression(knownL,X_meanL,newL[i],kernel,priorMean) #line prior regression
        #new_y, var_y = regression(known_x,known_y,newX[i],kernel)
        predictedX.append(val)
        varianceX.append(var)

    return predictedX, varianceX

#lp = line prior
#prediccion en X y en Y
#Recibe los datos conocidos (x,y,z) y los puntos para la regresion.
def prediction_XY_lp(x, y, l, newL, kernelX, kernelY, priorMeanX, priorMeanY):#prediction
    newX, varX = estimate_new_set_of_values_lp(l,x,newL,kernelX,priorMeanX)#prediccion para x
    newY, varY = estimate_new_set_of_values_lp(l,y,newL,kernelY,priorMeanY)#prediccion para y

    return newX, newY, varX, varY

#Toma la mitad de los datos observados como conocidos y predice nPoints en la mitad restante, regresa el error de la prediccion
def prediction_error_of_points_along_the_path_lp(nPoints,knownX,knownY,knownL,goal,unit,kernelX,kernelY,priorMeanX,priorMeanY):
    knownN = len(knownX)
    halfN = int(knownN/2)

    trueX = knownX[0:halfN]
    trueY = knownY[0:halfN]
    trueL = knownL[0:halfN]

    finishXY = middle_of_area(goal)
    finishD = euclidean_distance([trueX[len(trueX)-1],trueY[len(trueY)-1]],finishXY)
    trueX.append(finishXY[0])
    trueY.append(finishXY[1])
    trueL.append(finishD*unit)

    d = int(halfN/nPoints)
    realX, realY, predictionSet = [],[],[]
    for i in range(nPoints):
        realX.append(knownX[halfN + i*d])
        realY.append(knownY[halfN + i*d])
        predictionSet.append(knownL[halfN + i*d])

    #predX, predY, varX,varY = prediction_XY(trueX,trueY,trueL, predictionSet, kernelX, kernelY)
    predX, predY, varX,varY = prediction_XY_lp(trueX,trueY,trueL, predictionSet, kernelX, kernelY,priorMeanX,priorMeanY)

    error = average_displacement_error([realX,realY],[predX,predY])

    return error
