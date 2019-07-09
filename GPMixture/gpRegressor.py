"""
A class for GP regression
"""
import numpy as np
import math
from regression import *

class gpRegressor:
    # Constructor
    def __init__(self, kernelX, kernelY, linearPriorMeanX=None, linearPriorMeanY=None):
        self.observedX       = None
        self.observedY       = None
        self.observedL       = None
        self.Kx              = None
        self.Ky              = None
        self.kx              = None
        self.ky              = None
        self.Cx              = None
        self.Cy              = None
        self.Kx_1            = None
        self.Ky_1            = None
        self.kernelX         = kernelX
        self.kernelY         = kernelY

    def updateObservations(observedX,observedY,observedL):
        n                    = len(observedX)
        self.observedX       = np.zeros((n,1))
        self.observedY       = np.zeros((n,1))
        self.observedL       = np.zeros((n,1))
        self.Kx              = np.zeros((n,n))
        self.Ky              = np.zeros((n,n))
        # Fill in K
        for i in range(n):
            for j in range(n):
                self.Kx[i][j] = self.kernelX(observedL[i],observedL[j])
                self.Ky[i][j] = self.kernelY(observedL[i],observedL[j])
        self.Kx_1 = inv(self.Kx)
        self.Ky_1 = inv(self.Ky)
        # Center the data in case we use the linear prior
        if self.linearPriorMeanX ==None:
            for i in range(n):
                self.observedX[i][0] = observedX[i]
                self.observedY[i][0] = observedX[i]
                self.observedL[i][0] = observedL[i]
            else:
                for i in range(n):
                    self.observedX[i][0] = observedX[i] - linear_mean(observedL[i], self.linearPriorMeanX[0])
                    self.observedY[i][0] = observedY[i] - linear_mean(observedL[i], self.linearPriorMeanY[0])
                    self.observedL[i][0] = observedL[i]


    # The main regression function: perform regression for a vector of values lnew
    def regression_xy(self,lnew):
        # Number of observed data
        n    = self.observedX.shape[0]
        # Number of predicted data
        nnew = len(lnew)
        # Compute k (nxnnew), C (nnewxnnew)
        kx  = np.zeros((n,nnew))
        ky  = np.zeros((n,nnew))
        Cx  = np.zeros((nnew,nnew))
        Cx  = np.zeros((nnew,nnew))

        # Fill in k
        for i in range(n):
            for j in range(nnew):
                kx[i][j] = kernelX(l[i],lnew[j],False)
                ky[i][j] = kernelY(l[i],lnew[j],False)
        # Fill in C
        for i in range(nnew):
            for j in range(nnew):
                Cx[i][j] = kernel(lnew[i],lnew[j],False)
                Cy[i][j] = kernel(lnew[i],lnew[j],False)

        # Predictive mean
        xnew = k.transpose().dot(K_1.dot(x_meanl))
        if linearPriorMeanX!=None:
            for j in range(nnew):
                xnew[j] += linear_mean(lnew[j],linearPriorMean[0])
        # Estimate the variance
        K_1kt = K_1.dot(k)
        kK_1kt = k.transpose().dot(K_1kt)
        # Variance
        var = C - kK_1kt
        return xnew, varx, ynew, vary
