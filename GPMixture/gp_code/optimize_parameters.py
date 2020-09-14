import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import inv,eig
from scipy.optimize import minimize
from utils.manip_trajectories import euclidean_distance
from utils.linalg import positive_definite
from scipy.linalg import *
from gp_code.kernels import *
from copy import copy
import random
import timeit
from termcolor import colored

# Evaluate the minus log-likelihood, for a given value of the hyper-parameters and for a given trajectory
def mlog_p(x,y,kernel):
    n = len(x)
    # Evaluate the Gram matrix
    K = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):
            K[i][j] = kernel(x[i],x[j])
            K[j][i] = K[i][j]
    # Use Cholesky to solve x = K^{-1} y
    if positive_definite(K):
        c_and_lower = cho_factor(K, overwrite_a=True)
    else:
        # In case, add a regularization term.
        c_and_lower = cho_factor(K+100.0*np.identity(K.shape[0]), overwrite_a=True)

    invKy       = cho_solve(c_and_lower, y)
    yKy         = np.inner(y,invKy)
    # Get the log-determinant as the sum of the log of the diagonal elements in C
    logDetK = 0.0
    for i in range(n):
        logDetK += np.log(abs(c_and_lower[0].diagonal()[i]))
    # I removed the constant terms (they do not depend on theta)
    return max(0,0.5*yKy+logDetK)

# Evaluate minus sum of the log-likelihoods for all the data
def neg_sum_log_p(theta,t,x,kernel):
    kernel.set_optimizable_parameters(theta)
    s = 0.0
    for i in range(len(t)):
        s += mlog_p(t[i],x[i],kernel)
    return s

# Opimization of the parameters, in x then in y
def optimize_kernel_parameters(t,x,theta,kernel):
    # TODO: set these bounds elsewhere
    bnds = ((100.0, 5000.0), (10.0, 200.0))
    try:
        parametersX = minimize(neg_sum_log_p,theta,(t,x,kernel),method='Nelder-Mead',options={'maxiter':25,'disp': False})
        px          = parametersX.x
    except Exception as e:
        print(colored("[ERR] {:s} ".format(e),'red'))
        px = theta
    kernel.set_optimizable_parameters(px)
    return px

# Optimize parameters of the kernel, given l,x as data (will maximize likelihood), kernel as the kernel
# and parameters as the current values of the parameters
def fit_parameters(l,x,kernel,parameters):
    return optimize_kernel_parameters(l,x,parameters,kernel)
