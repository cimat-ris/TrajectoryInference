import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import inv,eig
from scipy.optimize import minimize
from utils.stats_trajectories import euclidean_distance
from utils.linalg import positive_definite
from scipy.linalg import *
from gp_code.kernels import *
from copy import copy
import random, logging
import timeit
from termcolor import colored

# Evaluate the minus log-likelihood,
# for a given value of the hyper-parameters and for a given trajectory
# We follow here Algorithm 2.1 from Rasmussen
def mlog_p(t,x,kernel,sigmaNoise):
    # Evaluate the Gram matrix
    K = kernel(t,t)
    K+=sigmaNoise*sigmaNoise*np.identity(K.shape[0])
    # Use Cholesky to solve x = K^{-1} y
    if not positive_definite(K):
        return 0.0
    # In case, add a regularization term.
    c_and_lower= cho_factor(K,overwrite_a=True)
    invKx      = cho_solve(c_and_lower,x)
    xKx        = np.inner(x,invKx)
    # Get the log-determinant as the sum of the log of the diagonal elements in C
    logDetK    = 0.0
    n          = len(x)
    for i in range(n):
        logDetK += np.log(c_and_lower[0][i][i])
    # I removed the constant terms (they do not depend on theta)
    return 0.5*xKx+logDetK

# Evaluate minus sum of the log-likelihoods for all the data
def neg_sum_log_p(theta,all_t,all_x,kernel,sigmaNoise,traj_min_length=10):
    kernel.set_optimizable_parameters(theta)
    mll = 0.0
    for t,x in zip(all_t,all_x):
        if (t.shape[0]>=traj_min_length):
            if kernel.linearPrior:
                mll += mlog_p(t,x-(kernel.meanSlope*t+kernel.meanConstant),kernel,sigmaNoise)
            else:
                mll += mlog_p(t,x,kernel,sigmaNoise)
    return mll

# Opimization of the parameters, in x then in y
def fit_parameters(t,x,kernel,theta,sigmaNoise):
    try:
        parametersX = minimize(neg_sum_log_p,theta,(t,x,kernel,sigmaNoise),method='L-BFGS-B',options={'maxiter':100,'disp': True},bounds=((1.0,10000.0),(1.0,500.0)))
        px          = parametersX.x
    except Exception as e:
        logging.error("[ERR] {:s} ".format(e))
        px = theta
    kernel.set_optimizable_parameters(px)
    return px
