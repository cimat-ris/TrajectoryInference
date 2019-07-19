import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.optimize import minimize
from scipy.linalg import *
from gp_code.kernels import *
from copy import copy
import random
import timeit
from termcolor import colored

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
