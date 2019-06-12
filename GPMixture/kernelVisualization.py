from GPRlib import *
from path import *
from plotting import *
from kernels import *
from statistics import*
import matplotlib.pyplot as plt
import numpy as np
import math
from copy import copy


# Evaluate covariance matrices on the interval [0,length]
def evaluateCovarianceMatrix(kernel,length):
    l = np.arange(0,length)
    C = np.zeros((l.size,l.size),dtype=float)
    for i in range(0,length):
        for j in range(0,length):
            C[i][j]=kernel(i,j)
    return C

s = 1000


parameters = [0.01,-0.5, 280., 280., 1.0]
kernel = kernels.squaredExponentialKernel(parameters[2],parameters[3])
CSqe = evaluateCovarianceMatrix(kernel,s)

kernel = kernels.maternKernel(parameters[2],parameters[3])
CM   = evaluateCovarianceMatrix(kernel,s)

kernel = kernels.linePriorCombinedKernel(parameters[0],parameters[1],parameters[2],parameters[3],parameters[4])
CCk  = evaluateCovarianceMatrix(kernel,s)

kernel = kernels.gammaExponentialKernel(parameters[2],parameters[3],0.8)
CG   =  evaluateCovarianceMatrix(kernel,s)

kernel = kernels.squaredExponentialAndNoiseKernel(parameters[2],parameters[3],parameters[4])
CSqeN   =  evaluateCovarianceMatrix(kernel,s)

kernel = kernels.exponentialAndNoiseKernel(parameters[2],parameters[3],parameters[4])
Cexp   =  evaluateCovarianceMatrix(kernel,s)

fg, axes = plt.subplots(2, 3, sharey=True)
axes[0][0].matshow(CSqe)
axes[0][1].matshow(CM)
axes[0][2].matshow(CG)

axes[1][0].matshow(CCk)
axes[1][1].matshow(CSqeN)
axes[1][2].matshow(Cexp)

plt.show()
