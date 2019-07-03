import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.optimize import minimize
from scipy.linalg import *
from kernels import *
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
""" LEARNING """


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
    yKy         = np.inner(y,invKy)
    # Get the log-determinant as the sum of the log of the diagonal elements in C
    logDetK = 0.0
    for i in range(n):
        logDetK += np.log(abs(c_and_lower[0].diagonal()[i]))
    # I removed the constant terms (they do not depend on theta)
    return max(0,0.5*yKy+logDetK)

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
        parametersX = minimize(neg_sum_log_p,theta,(t,x,kernel),method='Nelder-Mead', options={'maxiter':18,'disp': False})
        px          = parametersX.x
    except Exception as e:
        print(colored("[ERR] {:s} ".format(e),'red'))
        px = theta
    kernel.set_optimizable_parameters(px)
    return px

# Learn parameters of the kernel, given l,x,y as data (will maximize likelihood)
def learn_parameters(l,x,kernel,parameters):
    return optimize_kernel_parameters(l,x,parameters,kernel)

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
