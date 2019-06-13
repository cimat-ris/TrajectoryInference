"""
Libreria de funciones para Gaussian Processes for Regression
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.optimize import minimize
from scipy.linalg import *
import kernels
import path
from dataManagement import *
import dataManagement
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse
from copy import copy
import random
import timeit
from termcolor import colored

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

# Set kernel: a function that creates a kernel with default parameters, given its type name
def setKernel(name):
    if(name == "squaredExponential"):
        parameters = [80., 80.]  #{Covariance magnitude factor, Characteristic length}
        kernel = kernels.squaredExponentialKernel(parameters[0],parameters[1],nsigma)
    elif(name == "combinedTrautman"):
        parameters = [60., 80., 80.]  #{Precision of the line constant, Covariance magnitude factor, Characteristic length}
        kernel = kernels.combinedKernel(parameters[0],parameters[1],parameters[2],nsigma)
    elif(name == "exponential"):
        parameters = [80., 80.]  #{Covariance magnitude factor, Characteristic length}
        kernel = kernels.exponentialKernel(parameters[0],parameters[1])
    elif(name == "gammaExponential"):
        parameters = [80., 80., 8.]  #{Covariance magnitude factor, Characteristic length, Gamma exponent}
        kernel = kernels.gammaExponentialKernel(parameters[0],parameters[1])
    elif(name == "rationalQuadratic"):
        parameters = [80.0,10., 5.] #{Covariance magnitude factor, Characteristic length, Alpha parameter}
        kernel = kernels.rationalQuadraticKernel(parameters[0],parameters[1],nsigma)
    elif(name == "squaredExponentialAndNoise"):
        parameters = [80., 80.] #{Covariance magnitude factor, Characteristic length}
        kernel = kernels.expKernel(parameters[0],parameters[1],nsigma)
    elif(name == "linePriorCombined"):
        parameters = [0.01,1.0, 80., 80.]  #{Standard deviation slope, Standard deviation constant, Covariance magnitude factor, Characteristic length}
        kernel = kernels.linePriorCombinedKernel(parameters[0],parameters[1],parameters[2],parameters[3],nsigma)

    return kernel

# Returns two nGoalsxnGoals matrices:
# - the matrix of kernels with the default parameters
# - the matrix of kernel parameters (with default values)
def create_kernel_matrix_(kerType, ngoals):
    kerMatrix = []
    parameters = []
    # For goal i
    for i in range(ngoals):
        aux = []
        auxP = []
        # For goal j
        for j in range(ngoals):
            kernel = setKernel(kerType)
            theta  = kernel.get_parameters()
            aux.append(kernel)
            auxP.append(theta)
        kerMatrix.append(aux)
        parameters.append(auxP)
    return kerMatrix, parameters

# Returns two rowsxcolumns matrices:
# - the matrix of kernels with the default parameters
# - the matrix of kernel parameters (with default values)
def create_kernel_matrix(kerType, rows, columns):
    kerMatrix = []
    parameters = []
    # For goal i
    for i in range(rows):
        aux = []
        auxP = []
        # For goal j
        for j in range(columns):
            kernel = setKernel(kerType)
            theta  = kernel.get_parameters()
            aux.append(kernel)
            auxP.append(theta)
        kerMatrix.append(aux)
        parameters.append(auxP)
    return kerMatrix, parameters

# Evaluate the minus log-likelihood
def mlog_p(theta,x,y,kernel):
    kernel.set_optimizable_parameters(theta)
    n = len(x)
    K = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):
            K[i][j] = kernel(x[i],x[j])
            K[j][i] = K[i][j]
    # Use Cholesky to solve x = K^{-1} y
    c_and_lower = cho_factor(K, overwrite_a=True)
    invKy       = cho_solve(c_and_lower, y)
    yKy = np.inner(y,invKy)
    # Get the log-determinant as the sum of the log of the diagonal elements in C
    logDetK = 0.0
    for i in range(n):
        logDetK += np.log(abs(c_and_lower[0].diagonal()[i]))
    # I removed the constant terms (they do not depend on theta)
    val = max(0,0.5*yKy+logDetK)
    return val

# Evaluate minus sum of the log-likelihoods
def neg_sum_log_p(theta,t,x,kernel):
    s = 0.0
    for i in range(len(t)):
        s += mlog_p(theta,t[i],x[i],kernel)
    return s

# Opimization of the parameters, in x then in y
def optimize_kernel_parameters(t,x,theta,kernel):
    # TODO: set these bounds elswhere
    bnds = ((100.0, 5000.0), (10.0, 200.0))
    try:
        #parametersX = minimize(neg_sum_log_p, theta,(t,x,kernel), method='SLSQP', bounds=bnds,options={'maxiter':40,'disp': False})
        parametersX = minimize(neg_sum_log_p,theta,(t,x,kernel),method='Nelder-Mead', options={'maxiter':16,'disp': False})
        px          = parametersX.x
    except Exception as e:
        print(colored("[ERR] {:s} ".format(e),'red'))
        px = theta
    kernel.set_optimizable_parameters(px)
    return px

# Learn parameters of the kernel, given l,x,y as data (will maximize likelihood)
def learn_parameters(l,x,kernel,parameters):
    return optimize_kernel_parameters(l,x,parameters,kernel)

# For each pair of goals, realize the optimization of the kernel parameters
def optimize_parameters_between_goals(kernelType, learnSet, rows, columns, linearPriorMatX, linearPriorMatY):
    # Build the kernel matrices with the default values
    kernelMatX, parametersX = create_kernel_matrix(kernelType, rows, columns)
    kernelMatY, parametersY = create_kernel_matrix(kernelType, rows, columns)
    # For goal i
    for i in range(rows):
        # For goal j
        for j in range(columns):
            # Get the paths that go from i to j
            paths = learnSet[i][j]
            if len(paths) > 0:
                start = timeit.default_timer()
                # Get the path data as x,y,z (z is arclength)
                x,y,z = get_data_from_paths(paths,"length")
                # Build a kernel with the specified type and initial parameters theta
                ker   = setKernel(kernelType)
                params= ker.get_parameters()
                theta = ker.get_optimizable_parameters()
                print("[OPT] Init parameters ",theta)
                print("[OPT] [",i,"][",j,"]")
                print("[OPT] #trajectories: ",len(z))
                # Learn parameters in X
                params[0] = linearPriorMatX[i][j][1][0]
                params[1] = linearPriorMatX[i][j][1][1]
                ker.set_parameters(params)
                thetaX  = learn_parameters(z,x,ker,theta)
                print("[OPT] x: ",thetaX)
                kernelMatX[i][j].set_parameters(ker.get_parameters())
                # Learn parameters in Y
                params[0] = linearPriorMatY[i][j][1][0]
                params[1] = linearPriorMatY[i][j][1][1]
                ker.set_parameters(params)
                thetaY  = learn_parameters(z,y,ker,theta)
                print("[OPT] y: ",thetaY)
                kernelMatY[i][j].set_parameters(ker.get_parameters())
                stop = timeit.default_timer()
                execution_time = stop - start
                print("[OPT] Parameter optimization done in %.2f seconds"%execution_time)
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

# Takes as an input a matrix of kernels. Exports the parameters, line by line
def write_parameters(matrix,rows,columns,fileName):
    f = open(fileName,"w")
    f.write('%d %d %s\n' % (rows,columns,matrix[0][0].type))
    for i in range(rows):
        for j in range(columns):
            ker = matrix[i][j]
            f.write('{:d} {:d} '.format(i,j))
            parameters = ker.get_parameters()
            for k in range(len(parameters)):
                f.write('{:07.4f} '.format(parameters[k]))
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

# Read a parameter file and return the matrix of kernels corresponding to this file
def read_and_set_parameters(file_name, nParameters):
    file = open(file_name,'r')
    firstline = file.readline()
    header    = firstline.split()
    # Get rows, columns, kernelType from the header
    rows      = int(header[0])
    columns   = int(header[1])
    kernelType= header[2]
    print("[INF] Opening ",file_name," to read parameters of ",rows,"x",columns," kernels of type: ",kernelType)
    matrix, parametersMat = create_kernel_matrix(kernelType, rows, columns)

    for line in file:
        parameters = []
        parameters_str = line.split()
        i = int(parameters_str[0])
        j = int(parameters_str[1])
        for k in range(2,len(parameters_str)):
            parameters.append(float(parameters_str[k]))
        print("[INF] From goal ",i," to ", j, " parameters: ",parameters)
        matrix[i][j].set_parameters(parameters)
    file.close()
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
# Mean of the Gaussian process with a linear prior
def linear_mean(l, priorMean):
    m = priorMean[0]*l + priorMean[1]
    return m

# Regression with line prior
def line_prior_regression(l,x_meanl,lnew,kernel,priorMean):
    # Number of observed data
    n = len(l)
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
    # Predictive mean
    xnew = linear_mean(lnew,priorMean[0]) + k.dot(K_1.dot(x_meanl))
    # Estimate the variance
    K_1kt = K_1.dot(k.transpose())
    kK_1kt = k.dot(K_1kt)
    # Variance
    var = kernel(lnew,lnew) - kK_1kt
    if var<0.1: 
        var = 0.1
    return xnew, var

# Applies regression for a whole set newL of values L, given knownL, knownX
def estimate_new_set_of_values_lp(knownL,knownX,newL,kernel,priorMean):
    lenNew = len(newL)
    predictedX, varianceX = [], []
    X_meanL = []
    for i in range(len(knownL)):
        X_meanL.append(knownX[i] - linear_mean(knownL[i], priorMean[0]))
    for i in range(lenNew):
        # For each i, applies regression for newL[i]
        val, var = line_prior_regression(knownL,X_meanL,newL[i],kernel,priorMean)
        # Predictive mean
        predictedX.append(val)
        # Variance
        varianceX.append(var)
    return predictedX, varianceX

# Performs prediction in X and Y with a line prior
# Takes as input known values (x,y,l) and the points at which we want to perform regression (newL)
def prediction_XY_lp(x, y, l, newL, kernelX, kernelY, priorMeanX, priorMeanY):
    # Regression for X
    newX, varX = estimate_new_set_of_values_lp(l,x,newL,kernelX,priorMeanX)
    # Regression for Y
    newY, varY = estimate_new_set_of_values_lp(l,y,newL,kernelY,priorMeanY)
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
