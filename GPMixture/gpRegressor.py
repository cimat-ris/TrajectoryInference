"""
A class for GP regression
"""
import numpy as np
import math
from regression import *

class gpRegressor:
    # Constructor
    def __init__(self, kernelX, kernelY, unit, stepUnit, linearPriorX=None, linearPriorY=None):
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
        self.sqRootVarX      = None
        self.sqRootVarY      = None
        self.newL            = None
        self.epsilon         = 0.1
        self.kernelX         = kernelX
        self.kernelY         = kernelY
        self.linearPriorX    = linearPriorX
        self.linearPriorY    = linearPriorY
        self.unit            = unit
        self.stepUnit        = stepUnit

    def updateObservations(self,observedX,observedY,observedL,finishPoint):
        n                    = len(observedX)
        self.observedX       = np.zeros((n+1,1))
        self.observedY       = np.zeros((n+1,1))
        self.observedL       = np.zeros((n+1,1))
        self.Kx              = np.zeros((n+1,n+1))
        self.Ky              = np.zeros((n+1,n+1))
        # Last really observed point
        lastObservedPoint = [observedX[-1], observedY[-1], observedL[-1]]
        # Generate the set of l values at which to predict x,y
        self.newL, finalL = get_prediction_set(lastObservedPoint,finishPoint,self.unit,self.stepUnit)
        # Fill in K (n+1 x n+1)
        for i in range(n):
            self.Kx[i][i] = self.kernelX(observedL[i],observedL[i])
            self.Ky[i][i] = self.kernelY(observedL[i],observedL[i])
            for j in range(i):
                self.Kx[i][j] = self.kernelX(observedL[i],observedL[j])
                self.Kx[j][i] = self.Kx[i][j]
                self.Ky[i][j] = self.kernelY(observedL[i],observedL[j])
                self.Ky[j][i] = self.Ky[i][j]
        # Last row/column
        for i in range(n):
            self.Kx[i][n] = self.kernelX(observedL[i],finalL)
            self.Kx[n][i] = self.Kx[i][n]
            self.Ky[i][n] = self.kernelY(observedL[i],finalL)
            self.Ky[n][i] = self.Ky[i][n]
        self.Kx[n][n] = self.kernelX(finalL,finalL)
        self.Ky[n][n] = self.kernelY(finalL,finalL)
        self.Kx_1 = inv(self.Kx)
        self.Ky_1 = inv(self.Ky)
        for i in range(n):
            self.updateObserved(i,observedX[i],observedY[i],observedL[i])
        self.updateObserved(n,finishPoint[0],finishPoint[1],finalL)

    def updateObserved(self,i,x,y,l):
        # Center the data in case we use the linear prior
        if self.linearPriorX==None:
            self.observedX[i][0] = x
            self.observedY[i][0] = y
            self.observedL[i][0] = l
        else:
            self.observedX[i][0] = x - linear_mean(l, self.linearPriorX[0])
            self.observedY[i][0] = y - linear_mean(l, self.linearPriorY[0])
            self.observedL[i][0] = l

    #
    def prediction_to_perturbed_finish_point(self,deltax,deltay):
        n            = len(self.observedX)
        nnew         = len(self.newL)
        deltaX       = np.zeros((n,1))
        deltaX[n-1,0]= deltax
        deltaY       = np.zeros((n,1))
        deltaY[n-1,0]= deltay

        # In this approximation, only the predictive mean is adapted
        xnew = self.kx.transpose().dot(self.Kx_1.dot(self.observedX+deltaX))
        ynew = self.ky.transpose().dot(self.Ky_1.dot(self.observedY+deltaY))
        if self.linearPriorX!=None:
            for j in range(nnew):
                xnew[j] += linear_mean(self.newL[j],self.linearPriorX[0])
                ynew[j] += linear_mean(self.newL[j],self.linearPriorY[0])
        return xnew, ynew, self.newL, self.varx, self.vary

    def sample_with_perturbed_finish_point(self,deltaX,deltaY):
        predictedX, predictedY, predictedL, varX, varY = self.prediction_to_perturbed_finish_point(deltaX,deltaY)
        # Number of predicted points
        nPredictions = len(predictedX)

        # Noise from a normal distribution
        sX = np.random.normal(size=(nPredictions,1))
        sY = np.random.normal(size=(nPredictions,1))
        return predictedX+self.sqRootVarX.dot(sX), predictedY+self.sqRootVarY.dot(sY)

    # The main regression function: perform regression for a vector of values
    # lnew, that has been computed in update
    def prediction_to_finish_point(self):
        if self.newL==None:
            return None
        # Number of observed data
        n    = self.observedX.shape[0]
        # Number of predicted data
        nnew = len(self.newL)
        # Compute k (nxnnew), C (nnewxnnew)
        self.kx  = np.zeros((n,nnew))
        self.ky  = np.zeros((n,nnew))
        self.Cx  = np.zeros((nnew,nnew))
        self.Cy  = np.zeros((nnew,nnew))

        # Fill in k
        for i in range(n):
            for j in range(nnew):
                self.kx[i][j] = self.kernelX(self.observedL[i],self.newL[j],False)
                self.ky[i][j] = self.kernelY(self.observedL[i],self.newL[j],False)
        # Fill in C
        for i in range(nnew):
            for j in range(nnew):
                self.Cx[i][j] = self.kernelX(self.newL[i],self.newL[j],False)
                self.Cy[i][j] = self.kernelY(self.newL[i],self.newL[j],False)

        # Predictive mean
        self.xnew = self.kx.transpose().dot(self.Kx_1.dot(self.observedX))
        self.ynew = self.ky.transpose().dot(self.Ky_1.dot(self.observedY))
        if self.linearPriorX!=None:
            for j in range(nnew):
                self.xnew[j] += linear_mean(self.newL[j],self.linearPriorX[0])
                self.ynew[j] += linear_mean(self.newL[j],self.linearPriorY[0])
        # Estimate the variance in x
        K_1kt  = self.Kx_1.dot(self.kx)
        kK_1kt = self.kx.transpose().dot(K_1kt)
        self.varx   = self.Cx - kK_1kt
        # Estimate the variance in y
        K_1kt  = self.Ky_1.dot(self.ky)
        kK_1kt = self.ky.transpose().dot(K_1kt)
        self.vary   = self.Cy - kK_1kt
        # Regularization to avoid singular matrices
        self.varx += self.epsilon*np.eye(self.varx.shape[0])
        self.vary += self.epsilon*np.eye(self.vary.shape[0])
        # Cholesky on varX
        self.sqRootVarX     = cholesky(self.varx,lower=True)
        self.sqRootVarY     = cholesky(self.vary,lower=True)
        return self.xnew, self.ynew, self.newL, self.varx, self.vary
